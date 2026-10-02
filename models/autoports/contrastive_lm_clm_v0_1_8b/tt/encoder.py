# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

import numpy as np
import torch
from loguru import logger

HF_MODEL = "Qwen/Qwen3-8B"
HIDDEN = 4096
BLOCK_SIZE = 32


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_final_norm() -> tuple[torch.Tensor, float]:
    import glob
    import json

    from safetensors import safe_open

    snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/")))
    if not snap:
        from huggingface_hub import snapshot_download

        snap = [
            snapshot_download(
                HF_MODEL,
                allow_patterns=["config.json", "model.safetensors.index.json", "model-00004-of-00005.safetensors"],
            )
        ]
    root = snap[-1]
    index = json.load(open(os.path.join(root, "model.safetensors.index.json")))
    cfg = json.load(open(os.path.join(root, "config.json")))
    shard = os.path.join(root, index["weight_map"]["model.norm.weight"])
    with safe_open(shard, "pt") as f:
        weight = f.get_tensor("model.norm.weight").float()
    return weight, float(cfg.get("rms_norm_eps", 1e-6))


def parse_mesh_shape(spec: str | None) -> tuple[int, int]:
    if not spec:
        return (1, 1)
    rows, cols = spec.lower().split("x")
    return (int(rows), int(cols))


CUSTOM_POLICIES = {
    "bf16_all": {
        "TensorPrecision": {"WQKV": "BF16", "KV_CACHE": "BF16", "WO": "BF16", "FF1_FF3": "BF16", "FF2": "BF16"},
        "OpFidelity": {
            "LI_FF1_FF3": "HIFI4",
            "LI_FF2": "HIFI4",
            "LI_QKV_PREFILL": "HIFI4",
            "LI_O_PREFILL": "HIFI4",
            "SDPA_PREFILL": "HIFI4",
            "LI_QKV_DECODE": "HIFI4",
            "LI_O_DECODE": "HIFI4",
            "SDPA_DECODE": "HIFI4",
        },
    },
    "bfp8_lofi_mlp": {
        "TensorPrecision": {"WQKV": "BFP8", "KV_CACHE": "BFP8", "WO": "BFP8"},
        "OpFidelity": {"LI_FF1_FF3": "LOFI", "LI_FF2": "LOFI"},
    },
    "accuracy_lofi_mlp": {
        "TensorPrecision": {"WQKV": "BF16", "KV_CACHE": "BF16", "WO": "BF16"},
        "OpFidelity": {
            "LI_QKV_PREFILL": "HIFI4",
            "LI_O_PREFILL": "HIFI4",
            "SDPA_PREFILL": "HIFI4",
            "LI_QKV_DECODE": "HIFI4",
            "LI_O_DECODE": "HIFI4",
            "SDPA_DECODE": "HIFI4",
            "LI_FF1_FF3": "LOFI",
            "LI_FF2": "LOFI",
        },
    },
    "bfp8_attn": {
        "TensorPrecision": {"WQKV": "BFP8", "KV_CACHE": "BFP8", "WO": "BFP8"},
        "OpFidelity": {"LI_FF1_FF3": "HIFI2_FP16", "LI_FF2": "HIFI2_FP16"},
    },
    "bfp8_attn_hifi2": {
        "TensorPrecision": {"WQKV": "BFP8", "KV_CACHE": "BFP8", "WO": "BFP8"},
        "OpFidelity": {
            "LI_FF1_FF3": "HIFI2_FP16",
            "LI_FF2": "HIFI2_FP16",
            "LI_QKV_PREFILL": "HIFI2",
            "LI_O_PREFILL": "HIFI2",
            "SDPA_PREFILL": "HIFI2",
        },
    },
}


def precision_policy(name: str):
    from models.tt_transformers.tt.model_config import (
        DecodersPrecision,
        MathFidelitySetting,
        ModelOptimizations,
        OpGroup,
        PrecisionSetting,
        TensorGroup,
    )

    if name in ("accuracy", "performance"):
        return DecodersPrecision.from_string(name)
    spec = CUSTOM_POLICIES[name]

    def make(num_decoders, model_name):
        conf = ModelOptimizations(
            {
                "TensorPrecision": {TensorGroup[k]: PrecisionSetting[v] for k, v in spec["TensorPrecision"].items()},
                "OpFidelity": {OpGroup[k]: MathFidelitySetting[v] for k, v in spec["OpFidelity"].items()},
            }
        )
        conf.__name__ = name
        inst = DecodersPrecision(num_decoders, model_name, decoder_conf=conf)
        inst.__name__ = name
        return inst

    return make


def mark_trace_output_corruptible(tensor) -> None:
    try:
        from ttnn.unsafe_allocation_tracker import UnsafeAllocationTracker
    except ImportError:
        return
    try:
        UnsafeAllocationTracker.mark_corruptible(tensor)
    except Exception as exc:
        logger.warning(f"mark_corruptible failed: {exc}")


def open_mesh(shape: tuple[int, int], trace_region_size: int, l1_small_size: int):
    os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
    import ttnn

    kwargs = dict(l1_small_size=l1_small_size, trace_region_size=trace_region_size, num_command_queues=1)
    n = shape[0] * shape[1]
    if n > 1:
        fabric = os.environ.get("CLM_FABRIC_CONFIG", "FABRIC_1D")
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric))
    return ttnn.open_mesh_device(ttnn.MeshShape(*shape), **kwargs)


class TtQwen3Encoder:
    def __init__(
        self,
        mesh_device,
        max_batch_size: int = 8,
        max_seq_len: int = 2048,
        precision: str = "accuracy",
        weight_dtype=None,
        max_tokens: int | None = None,
        warmup: bool = True,
    ):
        from transformers import AutoTokenizer

        import ttnn
        from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
        from models.tt_transformers.tt.generator import Generator

        os.environ.setdefault("HF_MODEL", HF_MODEL)
        self.mesh_device = mesh_device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_tokens = max_tokens or max_seq_len
        self.precision = precision
        self.weight_dtype = weight_dtype or ttnn.bfloat8_b
        self._lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        self._fallback_ids = self.tokenizer(" ", add_special_tokens=False)["input_ids"]
        blocks_per_seq = (max_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        self.paged_attention_config = PagedAttentionConfig(
            block_size=BLOCK_SIZE, max_num_blocks=max(1024, blocks_per_seq * max_batch_size)
        )
        opt = precision_policy(precision)
        t0 = time.perf_counter()
        self.model_args, self.model, self.kv_cache, _ = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=max_batch_size,
            optimizations=lambda ma: opt(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            paged_attention_config=self.paged_attention_config,
            dtype=self.weight_dtype,
        )
        self.load_seconds = time.perf_counter() - t0
        self.program_config_overrides = self._install_program_configs()
        self.sharded_norm_rows = self._install_sharded_norms()
        self.generator = Generator([self.model], [self.model_args], mesh_device, tokenizer=self.model_args.tokenizer)
        self.page_table = torch.arange(self.paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            max_batch_size, -1
        )
        self.trace_lens = self._select_trace_lens(max_seq_len)
        self.model_args.trace_prefill_supported_seq_lens = list(self.trace_lens)
        self.batch_sizes = sorted({b for b in (1, 4, max_batch_size) if b <= max_batch_size})
        self.norm_weight, self.norm_eps = load_final_norm()
        self.calls = 0
        self.tokens_spent = 0
        self.device_seconds = 0.0
        self.ready = False
        logger.info(
            f"TtQwen3Encoder loaded {self.model_args.model_name} on {self.model_args.device_name} in {self.load_seconds:.1f}s; "
            f"max_batch_size={max_batch_size} max_seq_len={max_seq_len} precision={precision} trace_lens={self.trace_lens} "
            f"program_config_overrides={self.program_config_overrides} sharded_norm_rows={self.sharded_norm_rows}"
        )
        if warmup:
            self.warmup()
        self.ready = True

    @classmethod
    def from_env(cls) -> "TtQwen3Encoder":
        shape_env = os.environ.get("CLM_MESH_SHAPE_ENV", "CLM_MESH_SHAPE")
        shape = parse_mesh_shape(os.environ.get(shape_env) or os.environ.get("CLM_MESH_SHAPE"))
        mesh = open_mesh(
            shape,
            trace_region_size=int(os.environ.get("CLM_TRACE_REGION_SIZE", 200_000_000)),
            l1_small_size=int(os.environ.get("CLM_L1_SMALL_SIZE", 32768)),
        )
        max_tokens = int(os.environ.get("CLM_MAX_TOKENS", 2048))
        max_seq_len_env = int(os.environ.get("CLM_MAX_SEQ_LEN", max_tokens))
        if max_tokens > max_seq_len_env:
            raise ValueError(f"CLM_MAX_TOKENS ({max_tokens}) must not exceed CLM_MAX_SEQ_LEN ({max_seq_len_env})")
        enc = cls(
            mesh,
            max_batch_size=int(os.environ.get("CLM_MAX_BATCH", 8)),
            max_seq_len=int(os.environ.get("CLM_MAX_SEQ_LEN", max_tokens)),
            precision=os.environ.get("CLM_PRECISION", "accuracy"),
            max_tokens=max_tokens,
            warmup=os.environ.get("CLM_WARMUP", "1") != "0",
        )
        enc.owns_mesh = True
        return enc

    DEFAULT_TRACE_LENS = (128, 256, 512, 1024, 2048)
    SHARDED_NORM_ROWS = (128, 256, 512)

    def _install_program_configs(self) -> list[str]:
        if os.environ.get("CLM_PROGRAM_CONFIGS", "1") == "0" or self.model_args.num_devices != 1:
            return []
        import math

        import ttnn
        from models.tt_transformers.tt.common import Mode

        ma = self.model_args
        orig_qkv = ma.get_attn_qkv_program_config
        orig_ff2 = ma.get_mlp_ff2_prg_config
        orig_minimal = ma.use_minimal_qkv_prefill_matmul
        qkv_128 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 10),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=math.ceil(ma.qkv_size / 32 / 8),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        minimal_11x10 = ttnn.MinimalMatmulConfig(
            M_block_size=8, K_block_size=8, N_block_size=8, compute_with_storage_grid_size=ttnn.CoreCoord(11, 10)
        )

        def get_qkv(mode, seq_len=1, prefetcher=None):
            if mode == Mode.PREFILL and prefetcher is None:
                return qkv_128 if seq_len <= 128 else minimal_11x10
            return orig_qkv(mode, seq_len, prefetcher)

        def get_ff2(mode, seq_len=1, prefetcher=None):
            if mode == Mode.PREFILL and prefetcher is None and seq_len > 128:
                return minimal_11x10
            return orig_ff2(mode, seq_len, prefetcher)

        def use_minimal_qkv(seq_len):
            return seq_len > 128 or orig_minimal(seq_len)

        ma.get_attn_qkv_program_config = get_qkv
        ma.get_mlp_ff2_prg_config = get_ff2
        ma.use_minimal_qkv_prefill_matmul = use_minimal_qkv
        return [
            "qkv_prefill_128: in0_block_w 4, out_subblock_w 4",
            "qkv_prefill_gt128: MinimalMatmul 11x10",
            "ff2_prefill_gt128: MinimalMatmul 11x10",
        ]

    def _install_sharded_norms(self) -> tuple:
        if os.environ.get("CLM_SHARDED_NORM", "1") == "0" or self.model_args.num_devices != 1:
            return ()
        import ttnn
        from models.tt_transformers.tt.common import Mode

        dim = self.model_args.dim
        cols, rows_grid = 8, 4
        shard_w = dim // cols
        block_w = shard_w // 32
        sub_w = next(w for w in (4, 2, 1) if block_w % w == 0)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows_grid - 1))})
        plans = {}
        for rows in self.SHARDED_NORM_ROWS:
            memcfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (rows // rows_grid, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
            )
            prg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[cols, rows_grid],
                subblock_w=sub_w,
                block_h=rows // rows_grid // 32,
                block_w=block_w,
                inplace=False,
            )
            plans[rows] = (memcfg, prg)

        def wrap(dn):
            orig = dn.forward

            def forward(x, mode, norm_config=None):
                if mode != Mode.PREFILL:
                    return orig(x, mode, norm_config)
                shape = tuple(x.shape)
                rows = 1
                for v in shape[:-1]:
                    rows *= int(v)
                plan = plans.get(rows)
                if plan is None:
                    return orig(x, mode, norm_config)
                memcfg, prg = plan
                xs = ttnn.to_memory_config(x, memcfg)
                y = ttnn.rms_norm(
                    xs,
                    epsilon=dn.norm.eps,
                    weight=dn.norm.weight,
                    program_config=prg,
                    compute_kernel_config=dn.norm.compute_kernel_config_hifi2,
                )
                ttnn.deallocate(xs)
                out = ttnn.sharded_to_interleaved(y, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(y)
                return out

            dn.forward = forward

        for layer in self.model.layers:
            wrap(layer.attention_norm)
            wrap(layer.ff_norm)
        return tuple(self.SHARDED_NORM_ROWS)

    def _select_trace_lens(self, max_seq_len: int) -> list[int]:
        spec = os.environ.get("CLM_TRACE_LENS", "").strip()
        lens = [int(x) for x in spec.split(",") if x.strip()] if spec else list(self.DEFAULT_TRACE_LENS)
        lens = sorted({n for n in lens if 0 < n <= max_seq_len})
        if not lens or lens[-1] < max_seq_len:
            lens.append(max_seq_len)
        if any(n % 128 for n in lens):
            raise ValueError(
                f"prefill bucket lengths must be multiples of 128 (CLM_TRACE_LENS and max_seq_len), got {lens}"
            )
        return lens

    def padded_len(self, n: int) -> int:
        for cand in self.trace_lens:
            if cand >= n:
                return cand
        return self.trace_lens[-1]

    def tokenize(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = list(self._fallback_ids)
        if len(ids) > self.max_tokens:
            ids = ids[-self.max_tokens :]
        return ids

    def _batch_bucket(self, b: int) -> int:
        for cand in self.batch_sizes:
            if cand >= b:
                return cand
        return self.batch_sizes[-1]

    def _page_table_for(self, b: int, seq_len: int):
        if b == 1:
            return self.page_table
        return self.generator._get_prefill_user_page_table(
            self.page_table,
            self.kv_cache,
            seq_len,
            trace_enabled=True,
            prefill_seq_len=seq_len,
            use_batched_prefill=True,
            user_id=list(range(b)),
            padded_batch_size=b,
        )

    def _traced_prefill(self, prefill_ids: torch.Tensor, seq_len: int, b: int):
        import ttnn

        page_table = self._page_table_for(b, seq_len)
        tt_out = self.generator._easy_trace_prefill(
            prefill_ids,
            page_table=page_table,
            user_id=list(range(b)) if b > 1 else 0,
            last_token_idx=[seq_len - 1] * b if b > 1 else seq_len - 1,
            kv_cache=self.kv_cache,
            model_id=0,
            prefill_seq_len=seq_len,
            batch_size=b,
            num_cached_tokens=0,
        )
        shards = ttnn.get_device_tensors(tt_out)
        if len(shards) == 1:
            return ttnn.to_torch(shards[0])
        host = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        return host

    def _pool_and_norm(self, host: torch.Tensor, lens: list[int], seq_len: int) -> np.ndarray:
        dim = self.model_args.dim
        rows = host.reshape(-1, host.shape[-1])
        if rows.shape[-1] > dim and rows.shape[-1] % dim == 0 and rows.shape[0] % (rows.shape[-1] // dim) != 0:
            rows = rows[:, :dim]
        if rows.shape[-1] > dim:
            rows = rows[:, :dim]
        picks = torch.stack([rows[i * seq_len + (n - 1), :dim] for i, n in enumerate(lens)]).float()
        var = picks.pow(2).mean(-1, keepdim=True)
        normed = picks * torch.rsqrt(var + self.norm_eps) * self.norm_weight
        return normed.numpy()

    def _forward_group(self, id_lists: list[list[int]]) -> np.ndarray:
        n_real = len(id_lists)
        lens = [len(x) for x in id_lists]
        seq_len = self.padded_len(max(lens))
        b = self._batch_bucket(n_real)
        prefill_ids = torch.zeros(b, seq_len, dtype=torch.long)
        for i, ids in enumerate(id_lists):
            prefill_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        t0 = time.perf_counter()
        host = self._traced_prefill(prefill_ids, seq_len, b)
        vecs = self._pool_and_norm(host, lens, seq_len)
        self.device_seconds += time.perf_counter() - t0
        self.calls += 1
        self.tokens_spent += sum(lens)
        return vecs

    def embed_ids(self, id_lists: list[list[int]]) -> np.ndarray:
        out = np.zeros((len(id_lists), HIDDEN), dtype=np.float32)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, ids in enumerate(id_lists):
            groups[self.padded_len(len(ids))].append(i)
        with self._lock:
            for _, idxs in sorted(groups.items()):
                for s in range(0, len(idxs), self.max_batch_size):
                    chunk = idxs[s : s + self.max_batch_size]
                    vecs = self._forward_group([id_lists[i] for i in chunk])
                    for row, i in enumerate(chunk):
                        out[i] = vecs[row]
        return l2(out)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        id_lists = [self.tokenize(t) for t in texts]
        vecs = self.embed_ids(id_lists)
        return vecs, sum(len(x) for x in id_lists)

    def warmup(self) -> None:
        from models.tt_transformers.tt.generator import _get_max_blocks_prefill, _pad_or_create_page_table

        t0 = time.perf_counter()
        gen = self.generator
        lens = [n for n in self.trace_lens if n <= self.max_seq_len] or [min(128, self.max_seq_len)]
        max_blocks = _get_max_blocks_prefill(self.kv_cache)
        prepared = {}
        with self._lock:
            for n in lens:
                for b in self.batch_sizes:
                    key = f"{n}_0_{b}_sp0"
                    if gen.trace_id_prefill.get(key) is not None:
                        continue
                    prefill_ids = torch.full((b, n), int(self.tokenizer.eos_token_id), dtype=torch.long)
                    source = self.page_table[0:1] if b == 1 else self._page_table_for(b, n)
                    page_table = _pad_or_create_page_table(source, max_blocks)
                    prepared[key] = gen._prepare_trace_prefill(
                        prefill_ids,
                        page_table=page_table,
                        chunk_page_table=None,
                        kv_cache=self.kv_cache,
                        model_id=0,
                        batch_size=b,
                        user_id=list(range(b)) if b > 1 else 0,
                        start_pos=0,
                    )
            t1 = time.perf_counter()
            for key, prep in prepared.items():
                trace_id, tt_out_trace, *device_inputs = gen._record_trace_prefill(prep)
                gen.trace_id_prefill[key] = trace_id
                gen.trace_inputs_prefill[key] = device_inputs
                gen.trace_output_prefill[key] = tt_out_trace
                mark_trace_output_corruptible(tt_out_trace)
        logger.info(
            f"TtQwen3Encoder warmup done in {time.perf_counter() - t0:.1f}s (prepare {t1 - t0:.1f}s, capture {time.perf_counter() - t1:.1f}s); traces: {list(prepared)}"
        )

    def healthy(self) -> bool:
        return bool(self.ready)

    def stats(self) -> dict:
        return {
            "calls": self.calls,
            "tokens": self.tokens_spent,
            "device_seconds": round(self.device_seconds, 3),
            "device_name": self.model_args.device_name,
            "precision": self.precision,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "trace_lens": self.trace_lens,
            "program_config_overrides": self.program_config_overrides,
            "sharded_norm_rows": list(self.sharded_norm_rows),
        }

    def release(self) -> None:
        import gc

        import ttnn

        with self._lock:
            gen = getattr(self, "generator", None)
            if gen is not None:
                for attr in ("trace_id_prefill", "trace_inputs_prefill", "trace_output_prefill"):
                    store = getattr(gen, attr, None)
                    if isinstance(store, dict):
                        if attr == "trace_id_prefill":
                            for trace_id in list(store.values()):
                                if trace_id is not None:
                                    try:
                                        ttnn.release_trace(self.mesh_device, trace_id)
                                    except Exception as exc:
                                        logger.warning(f"release_trace failed: {exc}")
                        store.clear()
            self.generator = None
            self.kv_cache = None
            self.model = None
            gc.collect()
            ttnn.synchronize_device(self.mesh_device)

    def close(self) -> None:
        import ttnn

        self.release()
        if getattr(self, "owns_mesh", False):
            ttnn.close_mesh_device(self.mesh_device)
