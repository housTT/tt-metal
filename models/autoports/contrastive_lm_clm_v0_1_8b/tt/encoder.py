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


def parse_mesh_shape(spec: str | None) -> tuple[int, int]:
    if not spec:
        return (1, 1)
    rows, cols = spec.lower().split("x")
    return (int(rows), int(cols))


def open_mesh(shape: tuple[int, int], trace_region_size: int, l1_small_size: int):
    os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
    import ttnn

    kwargs = dict(l1_small_size=l1_small_size, trace_region_size=trace_region_size, num_command_queues=1)
    n = shape[0] * shape[1]
    if n > 1:
        fabric = os.environ.get("CLM_FABRIC_CONFIG", "FABRIC_1D_RING")
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
        from models.tt_transformers.tt.model_config import DecodersPrecision

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
        opt = DecodersPrecision.from_string(precision)
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
        self.generator = Generator([self.model], [self.model_args], mesh_device, tokenizer=self.model_args.tokenizer)
        self.page_table = torch.arange(self.paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            max_batch_size, -1
        )
        self.trace_lens = sorted(int(x) for x in self.model_args.trace_prefill_supported_seq_lens)
        self.calls = 0
        self.tokens_spent = 0
        self.device_seconds = 0.0
        self.ready = False
        logger.info(
            f"TtQwen3Encoder loaded {self.model_args.model_name} on {self.model_args.device_name} in {self.load_seconds:.1f}s; "
            f"max_batch_size={max_batch_size} max_seq_len={max_seq_len} precision={precision} trace_lens={self.trace_lens}"
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

    def padded_len(self, n: int) -> int:
        from models.tt_transformers.tt.common import get_padded_prefill_len

        return get_padded_prefill_len(n)

    def tokenize(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = list(self._fallback_ids)
        if len(ids) > self.max_tokens:
            ids = ids[-self.max_tokens :]
        return ids

    def _forward_group(self, id_lists: list[list[int]]) -> np.ndarray:
        b = len(id_lists)
        lens = [len(x) for x in id_lists]
        tokens = torch.zeros(b, max(lens), dtype=torch.long)
        for i, ids in enumerate(id_lists):
            tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        t0 = time.perf_counter()
        out = self.generator.prefill_forward_text(
            tokens,
            page_table=self.page_table[:b],
            kv_cache=[self.kv_cache],
            prompt_lens=lens,
            enable_trace=True,
            return_hidden_states=True,
        )
        self.device_seconds += time.perf_counter() - t0
        self.calls += 1
        self.tokens_spent += sum(lens)
        return out.float().numpy().reshape(b, HIDDEN)

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
        t0 = time.perf_counter()
        lens = [n for n in self.trace_lens if n <= self.max_seq_len] or [min(128, self.max_seq_len)]
        for n in lens:
            for b in sorted({1, self.max_batch_size}):
                ids = [self._fallback_ids * 1 + [self.tokenizer.eos_token_id] * (min(n, self.max_tokens) - 1)] * b
                self.embed_ids(ids)
        logger.info(f"TtQwen3Encoder warmup done in {time.perf_counter() - t0:.1f}s for lens={lens}")

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
