# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full autoregressive Qwen3.8-Flash-Next model for the fixed P300 TP2 mesh.

The decoder stack is the optimized :class:`MultichipDecoder` graph.  The
stack-internal residual stays fractured as ``[1, 1, 4*M, 1280]`` for every
layer.  Token embeddings are hidden-sharded at ingress and the only residual
all-gather is immediately before the checkpoint-exact final hyperconnection
mixer.  The LM head is vocabulary-sharded and feeds :class:`Sampling1D`
without a full-vocabulary gather.

Routed experts and PLE rows use the exact, declared host boundary implemented
by ``host_weight_cache.py``.  Activations, recurrence/KV state, final mixing,
logits, sampling, token feedback, and position advancement remain on TT.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import functional_decoder as _functional_decoder
from models.autoports.qwen_qwen3_8_flash_next.tt.host_weight_cache import Qwen38PLEHostStore, SafetensorCheckpoint
from models.autoports.qwen_qwen3_8_flash_next.tt.model_config import (
    HF_ADVERTISED_CONTEXT,
    LINEAR_ATTENTION,
    PREFILL_CHUNK,
    decoder_shapes,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    COLLECTIVE_NUM_LINKS,
    RESIDUAL_SHARD_WIDTH,
    TARGET_MESH,
    HostBackedSegmentedDecodeTrace,
    MultichipDecoder,
    MultichipDecodeStateWorkspace,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import _hifi2, _lofi
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _create_dram_sharded_mem_config
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
DEFAULT_SNAPSHOT = Path(
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/" f"snapshots/{MODEL_REVISION}"
)
VOCAB_SIZE = 248_320
HIDDEN_SIZE = 2_560
HC_COUNT = 4
HC_WIDTH = HIDDEN_SIZE * HC_COUNT
HC_LOWRANK = 320
BLOCK_SIZE = 64
LM_HEAD_COLUMNS_PER_RANK = 32_768
PAD_TOKEN_ID = 248_044
EOS_TOKEN_IDS = (248_046, 248_044)
REQUIRED_L1_SMALL_SIZE = 24_576


@dataclasses.dataclass(frozen=True)
class LMHeadPolicySpec:
    """Static LM-head geometry; dtype objects are resolved only at model load."""

    weight_dtype: str
    fidelity: str
    logits_dtype: str | None = None
    dram_splits: int | None = None
    worker_grid: tuple[int, int] | None = None
    in0_block_w: int | None = None

    @property
    def dram_sharded(self) -> bool:
        return self.dram_splits is not None

    @property
    def worker_cores(self) -> int | None:
        return None if self.worker_grid is None else math.prod(self.worker_grid)

    @property
    def sampler_dtype(self) -> str:
        return self.logits_dtype or self.weight_dtype

    def split_sizes(self, columns_per_rank: int) -> tuple[int, ...]:
        per_rank = VOCAB_SIZE // 2
        if self.dram_splits is not None:
            if per_rank % self.dram_splits:
                raise ValueError(f"local vocabulary {per_rank} is not divisible by {self.dram_splits} splits")
            return (per_rank // self.dram_splits,) * self.dram_splits
        return tuple(min(columns_per_rank, per_rank - offset) for offset in range(0, per_rank, columns_per_rank))


# The DRAM frontier keeps exact logical local-vocabulary slices while allowing
# the kernel's ordinary physical N padding. The one-split/per_core_N=97 point
# is retained as an explicit L1-capacity probe; capacity-directed points stay
# at <=25 N tiles per worker and include the common ~668-columns/core bound.
LM_HEAD_POLICIES = {
    "bf16_hifi2": LMHeadPolicySpec("bf16", "hifi2"),
    "bf16_lofi": LMHeadPolicySpec("bf16", "lofi"),
    "bfp8_hifi2": LMHeadPolicySpec("bfp8", "hifi2"),
    "bfp8_lofi": LMHeadPolicySpec("bfp8", "lofi"),
    "bfp4_lofi": LMHeadPolicySpec("bfp4", "lofi", logits_dtype="bf16"),
    "bfp8_hifi2_dram_s1_c40": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=1, worker_grid=(10, 4), in0_block_w=2),
    "bfp8_hifi2_dram_s4_c40": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=4, worker_grid=(10, 4), in0_block_w=2),
    "bfp8_hifi2_dram_s5_c40": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=5, worker_grid=(10, 4), in0_block_w=2),
    "bfp8_hifi2_dram_s5_c40_b1": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=5, worker_grid=(10, 4), in0_block_w=1),
    "bfp8_hifi2_dram_s8_c40": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=8, worker_grid=(10, 4), in0_block_w=2),
    "bfp8_hifi2_dram_s10_c20": LMHeadPolicySpec("bfp8", "hifi2", dram_splits=10, worker_grid=(10, 2), in0_block_w=4),
}


def _lm_head_rank_slices(split_sizes: Sequence[int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the exact checkpoint slices placed on each TP rank, split-major."""

    per_rank = VOCAB_SIZE // 2
    result = []
    offset = 0
    for split_size in split_sizes:
        result.append(
            tuple((rank * per_rank + offset, rank * per_rank + offset + int(split_size)) for rank in range(2))
        )
        offset += int(split_size)
    if offset != per_rank:
        raise ValueError(f"LM-head splits cover {offset} local columns, expected {per_rank}")
    return tuple(result)


def _config_namespace(value):
    """Load this unreleased HF config without depending on one Transformers build."""

    if isinstance(value, dict):
        return SimpleNamespace(**{key: _config_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_config_namespace(item) for item in value]
    return value


def _shape(value) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _require_target_mesh(mesh_device) -> None:
    actual = tuple(int(value) for value in mesh_device.shape)
    if actual != TARGET_MESH:
        raise ValueError(f"Qwen3.8 full model requires the P300 mesh {TARGET_MESH}, got {actual}")


def _replicated_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _hidden_shard_mapper(mesh_device):
    return ttnn.ShardTensorToMesh(mesh_device, dim=-1)


def _upload_replicated(
    value: torch.Tensor,
    mesh_device,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        mesh_mapper=_replicated_mapper(mesh_device),
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
    )


def _copy_host_to_device(value: torch.Tensor, target, *, dtype, layout) -> None:
    host = ttnn.from_torch(value, dtype=dtype, layout=layout)
    ttnn.copy_host_to_device_tensor(host, target)


def _deallocate(value) -> None:
    if isinstance(value, ttnn.Tensor) and value.is_allocated():
        ttnn.deallocate(value)


def _rope_tables(max_seq_len: int, rotary_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    frequencies = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inverse)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().bfloat16(), embedding.sin().bfloat16()


@dataclasses.dataclass
class Qwen38BatchState:
    """Explicit fixed-slot state shared by standalone and serving callers.

    A state owns no weights or KV tensors.  Decoder KV/recurrence remains
    model-owned; this object names the corresponding page-table, token,
    position, request, prompt-length, and active-slot state.  Buffers are
    stable for trace lifetime and refreshed in place between requests.
    """

    token_input: object
    current_pos: object
    page_table: object
    page_table_host: torch.Tensor
    prompt_lens: torch.Tensor
    active_mask: torch.Tensor
    request_ids: tuple[object, ...]
    generation: int = 0
    token_host_copies: int = 0
    position_host_copies: int = 0
    page_table_host_copies: int = 0
    page_table_unchanged_skips: int = 0
    compact_token_readbacks: int = 0

    @property
    def batch_size(self) -> int:
        return int(self.prompt_lens.numel())

    @property
    def active_slots(self) -> tuple[int, ...]:
        return tuple(torch.nonzero(self.active_mask, as_tuple=False).flatten().tolist())


class Qwen38FullModel:
    """Checkpoint-exact text model around the optimized P300 decoder stack."""

    model_id = MODEL_ID
    revision = MODEL_REVISION
    max_context_len = HF_ADVERTISED_CONTEXT
    block_size = BLOCK_SIZE
    prefill_chunk_size = PREFILL_CHUNK
    vocab_size = VOCAB_SIZE
    valid_vocab_size = VOCAB_SIZE
    eos_token_ids = EOS_TOKEN_IDS
    pad_token_id = PAD_TOKEN_ID
    _tt_allow_decode_trace_buffer_reuse = True

    def __init__(
        self,
        *,
        snapshot: str | Path,
        hf_config,
        mesh_device,
        max_batch: int = 1,
        max_seq_len: int = HF_ADVERTISED_CONTEXT,
        layer_indices: Sequence[int] | None = None,
        expert_cache_slots: int = 10,
        packed_host_experts: int = 512,
        prepack_host_experts: bool | None = None,
        lm_head_columns_per_rank: int = LM_HEAD_COLUMNS_PER_RANK,
        lm_head_policy: str | None = None,
    ):
        _require_target_mesh(mesh_device)
        self.snapshot = Path(snapshot).resolve()
        self.hf_config = hf_config
        self.text_config = getattr(hf_config, "text_config", hf_config)
        self.mesh_device = mesh_device
        self.max_batch = int(max_batch)
        self.max_seq_len = int(max_seq_len)
        if not 1 <= self.max_batch <= 32:
            raise ValueError("max_batch must be in [1, 32]")
        if not 1 <= self.max_seq_len <= HF_ADVERTISED_CONTEXT:
            raise ValueError(f"max_seq_len must be in [1, {HF_ADVERTISED_CONTEXT}]")
        self._validate_config()

        all_layers = tuple(range(int(self.text_config.num_hidden_layers)))
        self.layer_indices = tuple(all_layers if layer_indices is None else (int(index) for index in layer_indices))
        if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must be a non-empty sequence without duplicates")
        if any(index not in all_layers for index in self.layer_indices):
            raise ValueError(f"layer index outside [0, {len(all_layers)})")
        qsa_indices = [index for index in self.layer_indices if self.text_config.layer_types[index] != LINEAR_ATTENTION]
        if qsa_indices:
            qsa_shape = decoder_shapes(self.hf_config, qsa_indices[0])
            qsa_min_context = _functional_decoder.QSA_BLOCK_TOPK * qsa_shape.indexer_compress_ratio
            if self.max_seq_len < qsa_min_context:
                raise ValueError(
                    f"QSA cache capacity must be at least {qsa_min_context} tokens for its fixed traced top-"
                    f"{_functional_decoder.QSA_BLOCK_TOPK} selector; got {self.max_seq_len}"
                )
        self.is_full_stack = self.layer_indices == all_layers
        self.prepack_host_experts = self.is_full_stack if prepack_host_experts is None else bool(prepack_host_experts)

        self.checkpoint = SafetensorCheckpoint(self.snapshot)
        self.ple_store = Qwen38PLEHostStore(self.checkpoint)
        self.decode_state_workspace = MultichipDecodeStateWorkspace(mesh_device) if self.max_batch == 1 else None
        self.layers: list[MultichipDecoder] = []
        self._closed = False
        self._trace_ready = False
        self._trace_execution_mode: str | None = None
        self._trace_state: Qwen38BatchState | None = None
        self.ingress_trace_id = None
        self.ingress_trace_output = None
        self.layer_traces: list[HostBackedSegmentedDecodeTrace] = []
        self.terminal_trace_id = None
        self.trace_logits = None
        self.sampling_trace_id = None
        self.position_trace_id = None
        self.trace_capture_seconds = 0.0
        self.trace_replays = 0
        self.trace_page_table_changes = 0
        self.last_decode_timing: dict[str, float | int] | None = None
        self.host_preload_report: dict[str, object] | None = None
        self._sampling_force_argmax = True
        self._sampling_seed_rngs: tuple[random.Random, ...] | None = None
        self.sampling_seed_host_copies = 0
        self.lm_head_policy = str(lm_head_policy or os.getenv("QWEN38_LM_HEAD_POLICY", "bfp8_hifi2"))

        try:
            self._load_endpoints(lm_head_columns_per_rank, self.lm_head_policy)
            self._load_layers(expert_cache_slots, packed_host_experts)
            if self.prepack_host_experts:
                self.preload_packed_host_experts()
            self._load_rope()
            self._build_sampler()
            self._allocate_persistent_decode_inputs()
        except BaseException:
            self.close(best_effort=True)
            raise

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        mesh_device,
        **kwargs,
    ) -> "Qwen38FullModel":
        snapshot = Path(model_dir).resolve()
        # The checkpoint identifies the not-yet-released ``qwen4_exp``
        # architecture. Runtime construction needs configuration data, not
        # the HF module implementation, so parse config.json directly.
        config = _config_namespace(json.loads((snapshot / "config.json").read_text()))
        return cls(snapshot=snapshot, hf_config=config, mesh_device=mesh_device, **kwargs)

    def _validate_config(self) -> None:
        cfg = self.text_config
        expected = {
            "hidden_size": HIDDEN_SIZE,
            "hc_count": HC_COUNT,
            "hc_lowrank": HC_LOWRANK,
            "vocab_size": VOCAB_SIZE,
            "max_position_embeddings": HF_ADVERTISED_CONTEXT,
            "num_hidden_layers": 48,
        }
        for name, value in expected.items():
            if int(getattr(cfg, name)) != value:
                raise ValueError(f"{name}={getattr(cfg, name)!r}; expected {value}")
        if bool(getattr(self.hf_config, "tie_word_embeddings", False)):
            raise ValueError("Qwen3.8-Flash-Next requires an untied LM head")
        for index in range(int(cfg.num_hidden_layers)):
            decoder_shapes(self.hf_config, index)

    def _load_endpoints(self, lm_head_columns_per_rank: int, lm_head_policy: str) -> None:
        """Load one endpoint tensor at a time; no full model host residency."""

        prefix = "model.language_model"
        embedding = self.checkpoint.tensor(f"{prefix}.embed_tokens.weight")
        if tuple(embedding.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise ValueError(f"unexpected embedding shape {tuple(embedding.shape)}")
        self.embedding_weight = ttnn.from_torch(
            embedding.reshape(1, 1, VOCAB_SIZE, HIDDEN_SIZE),
            device=self.mesh_device,
            mesh_mapper=_hidden_shard_mapper(self.mesh_device),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        del embedding
        gc.collect()

        final_prefix = f"{prefix}.hyper_connection_mixer"
        norm = self.checkpoint.tensor(f"{final_prefix}.hc_norm.weight")
        self.final_norm_weight = _upload_replicated(
            (norm.float() + 1.0).bfloat16().reshape(1, 1, 1, HC_WIDTH), self.mesh_device
        )
        del norm
        down = self.checkpoint.tensor(f"{final_prefix}.input_mix_weight_down.weight")
        self.final_down_weight = _upload_replicated(
            down.transpose(0, 1).reshape(1, 1, HC_WIDTH, HC_LOWRANK),
            self.mesh_device,
            dtype=ttnn.bfloat8_b,
        )
        del down
        up = self.checkpoint.tensor(f"{final_prefix}.input_mix_weight_up.weight")
        self.final_up_weight = _upload_replicated(
            up.transpose(0, 1).reshape(1, 1, HC_LOWRANK, HC_WIDTH),
            self.mesh_device,
            dtype=ttnn.bfloat8_b,
        )
        del up
        gc.collect()

        if lm_head_columns_per_rank <= 0 or lm_head_columns_per_rank % 32:
            raise ValueError("lm_head_columns_per_rank must be a positive multiple of 32")
        if lm_head_policy not in LM_HEAD_POLICIES:
            raise ValueError(f"unknown LM-head policy {lm_head_policy!r}; expected one of {tuple(LM_HEAD_POLICIES)}")
        policy = LM_HEAD_POLICIES[lm_head_policy]
        lm_head_dtype = {
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }[policy.weight_dtype]
        lm_head_compute = {"hifi2": _hifi2, "lofi": _lofi}[policy.fidelity]()
        split_sizes = policy.split_sizes(int(lm_head_columns_per_rank))
        rank_slices = _lm_head_rank_slices(split_sizes)

        program_configs = None
        weights_memcfgs = None
        input_memcfg = ttnn.DRAM_MEMORY_CONFIG
        dram_cores = None
        if policy.dram_sharded:
            assert policy.worker_grid is not None
            assert policy.worker_cores is not None
            assert policy.in0_block_w is not None
            grid_x, grid_y = policy.worker_grid
            core_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
            k_tiles_per_worker = (HIDDEN_SIZE // 32) // policy.worker_cores
            if (HIDDEN_SIZE // 32) % policy.worker_cores or k_tiles_per_worker % policy.in0_block_w:
                raise ValueError(f"invalid exact DRAM K geometry for {lm_head_policy}")
            input_memcfg = ttnn.create_sharded_memory_config(
                (32, HIDDEN_SIZE // policy.worker_cores),
                core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            program_configs = []
            for split_size in split_sizes:
                n_tiles = split_size // 32
                program_configs.append(
                    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=policy.in0_block_w,
                        per_core_M=1,
                        per_core_N=math.ceil(n_tiles / policy.worker_cores),
                        fused_activation=None,
                    )
                )

            dram_size = self.mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(int(dram_size.x) - 1, int(dram_size.y) - 1),
                    )
                }
            )
            dram_cores = int(dram_size.x) * int(dram_size.y)
            weights_memcfgs = [
                _create_dram_sharded_mem_config(
                    k=HIDDEN_SIZE,
                    n=split_size,
                    dram_grid=dram_grid,
                    dram_cores=dram_cores,
                )
                for split_size in split_sizes
            ]

        lm_weight = self.checkpoint.tensor("lm_head.weight")
        if tuple(lm_weight.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise ValueError(f"unexpected LM-head shape {tuple(lm_weight.shape)}")
        lazy_weights = []
        for split_index, (split_size, split_rank_slices) in enumerate(zip(split_sizes, rank_slices)):
            rank_parts = [lm_weight[start:end].transpose(0, 1) for start, end in split_rank_slices]
            combined = torch.cat(rank_parts, dim=-1).contiguous().reshape(1, 1, HIDDEN_SIZE, 2 * split_size)
            weight_memcfg = ttnn.DRAM_MEMORY_CONFIG if weights_memcfgs is None else weights_memcfgs[split_index]
            lazy_weights.append(
                LazyWeight(
                    source=combined,
                    dtype=lm_head_dtype,
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=weight_memcfg,
                )
            )
        del lm_weight
        gc.collect()
        self.lm_head = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=lazy_weights,
                mesh_device=self.mesh_device,
                dim=HIDDEN_SIZE,
                max_batch_size=self.max_batch,
                program_configs=program_configs,
                compute_kernel_config=lm_head_compute,
                lm_head_dtype=lm_head_dtype,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                input_memcfg=input_memcfg,
                weights_memcfgs=weights_memcfgs,
            )
        )
        self.lm_head_policy_spec = policy
        self.lm_head_split_sizes = split_sizes
        self.lm_head_rank_slices = rank_slices
        self.lm_head_dram_cores = dram_cores
        self.lm_head.load_device_weights()
        # Device values are now materialized.  Drop the sizeable source tensors
        # retained by LazyWeight so endpoint loading cannot become transient
        # full residency during layer construction.
        for weight in self.lm_head.config.output_weights:
            weight.source = torch.empty(tuple(weight.source.shape), device="meta", dtype=torch.bfloat16)
        gc.collect()

    def _load_layers(self, expert_cache_slots: int, packed_host_experts: int) -> None:
        for layer_index in self.layer_indices:
            kwargs = {}
            if (
                self.decode_state_workspace is not None
                and self.text_config.layer_types[layer_index] == LINEAR_ATTENTION
            ):
                kwargs["decode_state_workspace"] = self.decode_state_workspace
            layer = MultichipDecoder.from_checkpoint_host_backed(
                self.checkpoint,
                hf_config=self.hf_config,
                layer_idx=layer_index,
                mesh_device=self.mesh_device,
                max_batch=self.max_batch,
                max_seq_len=self.max_seq_len,
                block_size=BLOCK_SIZE,
                expert_cache_slots=expert_cache_slots,
                packed_host_experts=packed_host_experts,
                ple_store=self.ple_store if layer_index == 1 else None,
                **kwargs,
            )
            self.layers.append(layer)
        self._kv_cache = tuple(
            (layer.shapes.layer_idx, layer.kv_cache, layer.indexer_cache)
            for layer in self.layers
            if layer.shapes.layer_type != LINEAR_ATTENTION
        )

    def preload_packed_host_experts(self) -> dict[str, object]:
        """Prepack every layer's exact experts before request service."""

        started = time.perf_counter()
        layers = {
            str(layer.shapes.layer_idx): layer.host_expert_cache.preload_packed_host()
            for layer in self.layers
            if layer.host_expert_cache is not None
        }
        self.host_preload_report = {
            "seconds": time.perf_counter() - started,
            "layers": layers,
            "loaded_entries": sum(int(value["loaded_entries"]) for value in layers.values()),
            "packed_host_bytes": sum(
                int(layer.host_expert_cache.packed_host_bytes)
                for layer in self.layers
                if layer.host_expert_cache is not None
            ),
        }
        return self.host_preload_report

    def _load_rope(self) -> None:
        rotary_dim = int(self.text_config.head_dim * float(self.text_config.partial_rotary_factor))
        rope_parameters = getattr(self.text_config, "rope_parameters", {})
        theta = float(rope_parameters.get("rope_theta", getattr(self.text_config, "rope_theta", 10_000_000.0)))
        cos, sin = _rope_tables(self.max_seq_len, rotary_dim, theta)
        self.rot_mats = (
            _upload_replicated(cos.reshape(1, 1, self.max_seq_len, rotary_dim), self.mesh_device),
            _upload_replicated(sin.reshape(1, 1, self.max_seq_len, rotary_dim), self.mesh_device),
        )
        del cos, sin

    def _build_sampler(self) -> None:
        self.sampling = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=VOCAB_SIZE,
                valid_vocab_size=VOCAB_SIZE,
                mesh_device=self.mesh_device,
                max_batch_size=self.max_batch,
                max_top_k=32,
                num_gather_links=COLLECTIVE_NUM_LINKS,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                pad_to_power_of_2=True,
            )
        )
        self.sampling.load_device_buffers()

    def _allocate_persistent_decode_inputs(self) -> None:
        token_host = torch.full((1, 1, 1, self.max_batch), PAD_TOKEN_ID, dtype=torch.int32)
        pos_host = torch.zeros(self.max_batch, dtype=torch.int32)
        blocks = math.ceil(self.max_seq_len / BLOCK_SIZE)
        page_host = torch.arange(self.max_batch * blocks, dtype=torch.int32).reshape(self.max_batch, blocks)
        self.decode_token_input = _upload_replicated(
            token_host,
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.decode_current_pos = _upload_replicated(
            pos_host,
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.decode_page_table = _upload_replicated(
            page_host,
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        mapper = _replicated_mapper(self.mesh_device)
        self.sampling_k = _upload_replicated(
            torch.ones(self.max_batch, dtype=torch.int32),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.sampling_p = ttnn.from_torch(
            torch.zeros(self.max_batch, dtype=torch.float32),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sampling_temp = ttnn.from_torch(
            torch.ones(self.max_batch, dtype=torch.float32),
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._default_page_table_host = page_host

    # ------------------------------------------------------------------ state

    def new_batch_state(
        self,
        prompt_lens: Sequence[int] | torch.Tensor,
        *,
        request_ids: Sequence[object] | None = None,
        page_table: torch.Tensor | None = None,
        active_mask: Sequence[bool] | torch.Tensor | None = None,
    ) -> Qwen38BatchState:
        lengths = torch.as_tensor(prompt_lens, dtype=torch.int32, device="cpu").reshape(-1)
        if int(lengths.numel()) != self.max_batch:
            raise ValueError(f"prompt_lens must contain exactly {self.max_batch} fixed slots")
        active = lengths > 0 if active_mask is None else torch.as_tensor(active_mask, dtype=torch.bool).reshape(-1)
        if int(active.numel()) != self.max_batch:
            raise ValueError(f"active_mask must contain exactly {self.max_batch} slots")
        if bool(torch.any(active & ((lengths < 1) | (lengths > self.max_seq_len)))):
            raise ValueError(f"active prompt lengths must be in [1, {self.max_seq_len}]")
        if bool(torch.any((~active) & (lengths != 0))):
            raise ValueError("inactive slots must have prompt length zero")
        if not bool(torch.any(active)):
            raise ValueError("at least one slot must be active")
        requests = tuple(range(self.max_batch)) if request_ids is None else tuple(request_ids)
        if len(requests) != self.max_batch or len(set(requests)) != len(requests):
            raise ValueError("request_ids must uniquely name every fixed slot")
        pages = (
            self._default_page_table_host.clone() if page_table is None else page_table.to(torch.int32).cpu().clone()
        )
        expected = tuple(self._default_page_table_host.shape)
        if tuple(pages.shape) != expected:
            raise ValueError(f"page_table must have shape {expected}, got {tuple(pages.shape)}")
        if bool(torch.any(pages < 0)) or bool(torch.any(pages >= self.max_batch * expected[1])):
            raise ValueError("page table contains a physical block outside the model-owned KV caches")
        state = Qwen38BatchState(
            token_input=self.decode_token_input,
            current_pos=self.decode_current_pos,
            page_table=self.decode_page_table,
            page_table_host=pages,
            prompt_lens=lengths,
            active_mask=active,
            request_ids=requests,
        )
        self.reset_batch_state(state)
        return state

    def reset_batch_state(self, state: Qwen38BatchState) -> None:
        self._require_state_buffers(state)
        if self._trace_ready and state is not self._trace_state:
            raise RuntimeError("live traces are bound to a different batch-state object")
        tokens = torch.full((1, 1, 1, self.max_batch), PAD_TOKEN_ID, dtype=torch.int32)
        positions = torch.where(state.active_mask, state.prompt_lens, torch.full_like(state.prompt_lens, -1))
        _copy_host_to_device(tokens, state.token_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_host_to_device(positions, state.current_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_host_to_device(
            state.page_table_host,
            state.page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        state.token_host_copies += 1
        state.position_host_copies += 1
        state.page_table_host_copies += 1
        state.generation += 1
        for layer in self.layers:
            if layer.host_expert_cache is not None:
                layer.host_expert_cache.reset()
        for request_id in state.request_ids:
            self.ple_store.reset_request(request_id)

    def update_page_table(self, state: Qwen38BatchState, page_table: torch.Tensor) -> bool:
        self._require_state_buffers(state)
        pages = torch.as_tensor(page_table, dtype=torch.int32, device="cpu")
        if tuple(pages.shape) != tuple(state.page_table_host.shape):
            raise ValueError("page-table shape cannot change while fixed-slot traces are live")
        if torch.equal(pages, state.page_table_host):
            state.page_table_unchanged_skips += 1
            return False
        _copy_host_to_device(pages, state.page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        state.page_table_host = pages.clone()
        state.page_table_host_copies += 1
        self.trace_page_table_changes += 1
        return True

    def set_sampling_params(
        self,
        *,
        top_k: int | Sequence[int] = 1,
        top_p: float | Sequence[float] = 0.0,
        temperature: float | Sequence[float] = 1.0,
        seeds: int | Sequence[int] | None = None,
    ) -> None:
        def expand(value, dtype):
            tensor = torch.as_tensor(value, dtype=dtype).reshape(-1)
            if tensor.numel() == 1:
                tensor = tensor.repeat(self.max_batch)
            if tensor.numel() != self.max_batch:
                raise ValueError(f"sampling parameter must be scalar or have {self.max_batch} values")
            return tensor

        k = expand(top_k, torch.int32)
        p = expand(top_p, torch.float32)
        temp = expand(temperature, torch.float32)
        if bool(torch.any((k < 1) | (k > 32))):
            raise ValueError("top_k must be in [1, 32]")
        if bool(torch.any((p < 0) | (p > 1))):
            raise ValueError("top_p must be in [0, 1]")
        if bool(torch.any(temp <= 0)):
            raise ValueError("temperature must be positive")
        force_argmax = bool(torch.all(k == 1).item() and torch.all(p == 0).item() and torch.all(temp == 1).item())
        if force_argmax != self._sampling_force_argmax and self._trace_ready:
            self.release_decode_traces()
        self._sampling_force_argmax = force_argmax
        if seeds is None:
            self._sampling_seed_rngs = None
        else:
            values = (int(seeds),) * self.max_batch if isinstance(seeds, int) else tuple(int(seed) for seed in seeds)
            if len(values) != self.max_batch:
                raise ValueError(f"seeds must contain exactly {self.max_batch} values")
            self._sampling_seed_rngs = tuple(random.Random(seed) for seed in values)
        _copy_host_to_device(k, self.sampling_k, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_host_to_device(p, self.sampling_p, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_host_to_device(temp, self.sampling_temp, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _advance_sampling_seeds(self) -> None:
        """Refresh explicit non-greedy request seeds through the persistent buffer."""

        if self._sampling_force_argmax or self._sampling_seed_rngs is None:
            return
        values = torch.tensor(
            [rng.randint(1, 0x7FFFFFFE) for rng in self._sampling_seed_rngs],
            dtype=torch.int32,
        )
        host = ttnn.from_torch(values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.sampling._seeds)
        self.sampling_seed_host_copies += 1

    def _require_state_buffers(self, state: Qwen38BatchState) -> None:
        if not isinstance(state, Qwen38BatchState):
            raise TypeError("state must be Qwen38BatchState")
        expected = (self.decode_token_input, self.decode_current_pos, self.decode_page_table)
        actual = (state.token_input, state.current_pos, state.page_table)
        if any(left is not right for left, right in zip(expected, actual)):
            raise ValueError("state does not reference this model's persistent decode buffers")

    # --------------------------------------------------------------- endpoints

    def embed_tokens(self, token_ids) -> object:
        embedded = ttnn.embedding(
            token_ids,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        rows = int(embedded.shape[-2])
        expanded = ttnn.reshape(embedded, (1, rows, 1, RESIDUAL_SHARD_WIDTH))
        expanded = ttnn.repeat(expanded, (1, 1, HC_COUNT, 1))
        residual = ttnn.reshape(expanded, (1, 1, rows * HC_COUNT, RESIDUAL_SHARD_WIDTH))
        _functional_decoder._free(embedded, residual)
        _functional_decoder._free(expanded, residual)
        return residual

    def final_hidden(self, residual) -> object:
        gathered = self.layers[-1].gather_residual(residual)
        normed = _functional_decoder.FunctionalDecoder._rms_norm(
            gathered,
            self.final_norm_weight,
            float(self.text_config.rms_norm_eps),
            group_count=HC_COUNT,
        )
        _functional_decoder._free(gathered, residual, normed)
        low = ttnn.linear(normed, self.final_down_weight, dtype=ttnn.bfloat16, compute_kernel_config=_hifi2())
        scaled = ttnn.multiply(low, 1.0 / HC_COUNT)
        ttnn.deallocate(low)
        low = ttnn.silu(scaled)
        ttnn.deallocate(scaled)
        mix = ttnn.linear(low, self.final_up_weight, dtype=ttnn.bfloat16, compute_kernel_config=_hifi2())
        ttnn.deallocate(low)
        rows = math.prod(_shape(normed)[:-1])
        norm_groups = ttnn.reshape(normed, (rows, HC_COUNT, HIDDEN_SIZE))
        mix_groups = ttnn.reshape(mix, (rows, HC_COUNT, HIDDEN_SIZE))
        mixed = ttnn.multiply(
            norm_groups,
            mix_groups,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        _functional_decoder._free(norm_groups, normed, mixed)
        _functional_decoder._free(mix_groups, mix, mixed)
        ttnn.deallocate(normed)
        ttnn.deallocate(mix)
        hidden = ttnn.mean(mixed, dim=1, keepdim=True)
        ttnn.deallocate(mixed)
        return ttnn.reshape(hidden, (1, 1, rows, HIDDEN_SIZE))

    def lm_head_configuration(self) -> dict[str, object]:
        """Serializable geometry used by focused A/B evidence."""

        policy = self.lm_head_policy_spec
        return {
            "policy": self.lm_head_policy,
            "weight_dtype": policy.weight_dtype,
            "logits_dtype": policy.sampler_dtype,
            "fidelity": policy.fidelity,
            "dram_sharded": policy.dram_sharded,
            "split_sizes": self.lm_head_split_sizes,
            "rank_slices": self.lm_head_rank_slices,
            "worker_grid": policy.worker_grid,
            "worker_cores": policy.worker_cores,
            "in0_block_w": policy.in0_block_w,
            "dram_cores": self.lm_head_dram_cores,
            "per_dram_reader_n": (
                None
                if self.lm_head_dram_cores is None
                else tuple(math.ceil((size // 32) / self.lm_head_dram_cores) for size in self.lm_head_split_sizes)
            ),
            "per_core_n": (
                None
                if not policy.dram_sharded
                else tuple(int(config.per_core_N) for config in self.lm_head.config.program_configs)
            ),
            "physical_output_tiles": (
                None
                if not policy.dram_sharded
                else tuple(
                    int(config.per_core_N) * int(policy.worker_cores) for config in self.lm_head.config.program_configs
                )
            ),
            "program_configs": tuple(repr(config) for config in self.lm_head.config.program_configs),
            "input_memory_config": repr(self.lm_head.config.input_memcfg),
            "weight_memory_configs": tuple(repr(config) for config in self.lm_head.config.weights_memcfgs),
        }

    def _project_hidden_logits_one_tile(self, hidden) -> object:
        """Project at most one physical tile, sharding input only for the DRAM frontier."""

        policy = self.lm_head_policy_spec
        if policy.dram_sharded:
            sharded = hidden
            owns_sharded = not hidden.memory_config().is_sharded()
            if owns_sharded:
                sharded = ttnn.interleaved_to_sharded(hidden, self.lm_head.config.input_memcfg)
            logits = self.lm_head(sharded)
            if owns_sharded:
                ttnn.deallocate(sharded)
        else:
            logits = self.lm_head(hidden)
        if policy.sampler_dtype == "bf16" and policy.weight_dtype != "bf16":
            sampler_logits = ttnn.typecast(logits, dtype=ttnn.bfloat16)
            ttnn.deallocate(logits)
            logits = sampler_logits
        return logits

    def project_hidden_logits(self, hidden) -> object:
        """Project BF16 hidden rows while retaining the exact local-vocabulary order.

        The specialized DRAM program accepts one 32-row physical tile. Decode,
        last-token prefill, and batch 1--32 use that direct path. The diagnostic
        ``return_all_logits`` path tiles larger M and concatenates logical rows,
        avoiding an incompatible program-config or a second weight copy.
        """

        rows = math.prod(_shape(hidden)[:-1])
        if not self.lm_head_policy_spec.dram_sharded or rows <= 32:
            return self._project_hidden_logits_one_tile(hidden)

        outputs = []
        for start in range(0, rows, 32):
            end = min(start + 32, rows)
            chunk = ttnn.slice(hidden, [0, 0, start, 0], [1, 1, end, HIDDEN_SIZE])
            outputs.append(self._project_hidden_logits_one_tile(chunk))
            _functional_decoder._free(chunk, hidden)
        logits = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for output in outputs:
            _functional_decoder._free(output, logits)
        return logits

    def project_logits(self, residual) -> object:
        hidden = self.final_hidden(residual)
        logits = self.project_hidden_logits(hidden)
        ttnn.deallocate(hidden)
        return logits

    def logits_to_torch(self, logits) -> torch.Tensor:
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        return ttnn.to_torch(logits, mesh_composer=composer).float()[..., :VOCAB_SIZE]

    def sampled_tokens_to_torch(self, tokens, state: Qwen38BatchState | None = None) -> torch.Tensor:
        shard = ttnn.get_device_tensors(tokens)[0]
        result = ttnn.to_torch(shard).reshape(-1)[: self.max_batch].to(torch.int64)
        if state is not None:
            state.compact_token_readbacks += 1
        return result

    # ---------------------------------------------------------------- prefill

    def _prefill_page_inputs(self, layer, state: Qwen38BatchState, slot: int, seq_len: int):
        row_host = state.page_table_host[slot : slot + 1]
        row_device = _upload_replicated(
            row_host,
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        chunks = []
        for start, _, padded in layer.prefill_chunk_plan(seq_len):
            first = start // BLOCK_SIZE
            last = (start + padded) // BLOCK_SIZE
            chunks.append(
                _upload_replicated(
                    row_host[:, first:last],
                    self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )
        return row_device, tuple(chunks)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        state: Qwen38BatchState,
        prompt_lens: Sequence[int] | torch.Tensor | None = None,
        page_table: torch.Tensor | None = None,
        kv_cache=None,
        return_all_logits: bool = False,
    ):
        """Fill cache/state for mixed logical prompts and return per-slot logits.

        Physical 128-token chunk/page/tile padding is entirely internal.  Each
        active fixed slot may have a different logical length.  ``kv_cache``
        is accepted to make ownership explicit; when supplied it must be the
        model-owned cache identity returned by :attr:`kv_cache`.
        """

        self._require_state_buffers(state)
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("this optimized stack supports only its stable model-owned KV cache")
        if page_table is not None:
            self.update_page_table(state, page_table)
        lengths = state.prompt_lens if prompt_lens is None else torch.as_tensor(prompt_lens, dtype=torch.int32)
        if not torch.equal(lengths.reshape(-1), state.prompt_lens):
            raise ValueError("prompt_lens must match the batch state used to allocate cache slots")
        ids = torch.as_tensor(tokens, dtype=torch.int64, device="cpu")
        if ids.ndim != 2 or ids.shape[0] != self.max_batch:
            raise ValueError(f"prefill tokens must be [{self.max_batch}, padded_prompt_width]")
        if int(ids.shape[1]) < int(state.prompt_lens.max()):
            raise ValueError("prefill token width is shorter than a logical prompt")
        if return_all_logits and self.max_batch != 1:
            raise ValueError("all-token logits are a batch-one diagnostic path")

        per_slot_logits: list[object | None] = [None] * self.max_batch
        first_active_logits = None
        for slot in state.active_slots:
            logical = int(state.prompt_lens[slot])
            prompt = ids[slot : slot + 1, :logical].contiguous()
            token_host = ttnn.from_torch(
                prompt.to(torch.int32).reshape(1, 1, 1, logical),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=_replicated_mapper(self.mesh_device),
            )
            token_device = ttnn.to_device(token_host, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            residual = self.embed_tokens(token_device)
            ttnn.deallocate(token_device)
            page_device = chunk_pages = None
            qsa_layer = next(
                (layer for layer in self.layers if layer.shapes.layer_type != LINEAR_ATTENTION),
                None,
            )
            if qsa_layer is not None:
                page_device, chunk_pages = self._prefill_page_inputs(qsa_layer, state, slot, logical)
            try:
                for layer in self.layers:
                    if layer.shapes.has_ple:
                        output = layer.prefill_forward_host_backed_fractured(
                            residual,
                            input_ids=prompt,
                            request_id=state.request_ids[slot],
                            user_id=slot,
                        )
                    else:
                        kwargs = {"user_id": slot}
                        if layer.shapes.layer_type != LINEAR_ATTENTION:
                            kwargs.update(
                                page_table=page_device,
                                page_tables_per_chunk=chunk_pages,
                                rot_mats=self.rot_mats,
                            )
                        output = layer.prefill_forward_fractured(residual, **kwargs)
                    _functional_decoder._free(residual, output)
                    residual = output
                if not return_all_logits:
                    start = HC_COUNT * (logical - 1)
                    last = ttnn.slice(
                        residual,
                        [0, 0, start, 0],
                        [1, 1, start + HC_COUNT, RESIDUAL_SHARD_WIDTH],
                    )
                    _functional_decoder._free(residual, last)
                    residual = last
                slot_logits = self.project_logits(residual)
                ttnn.deallocate(residual)
                per_slot_logits[slot] = slot_logits
                if first_active_logits is None:
                    first_active_logits = slot_logits
            finally:
                _deallocate(page_device)
                if chunk_pages is not None:
                    for chunk in chunk_pages:
                        _deallocate(chunk)

        for layer in self.layers:
            layer.prepare_decode_state()
        if return_all_logits:
            return per_slot_logits[state.active_slots[0]]
        assert first_active_logits is not None
        for slot, value in enumerate(per_slot_logits):
            if value is None:
                per_slot_logits[slot] = ttnn.multiply(first_active_logits, 0.0)
        output = per_slot_logits[0] if self.max_batch == 1 else ttnn.concat(per_slot_logits, dim=2)
        if self.max_batch > 1:
            for value in per_slot_logits:
                ttnn.deallocate(value)
        return output

    # ----------------------------------------------------------------- decode

    @property
    def kv_cache(self):
        return self._kv_cache

    def _stage_active_ple_decode(self, layer, state: Qwen38BatchState, ple_input_ids: torch.Tensor):
        """Lookup only live requests while preserving the fixed device batch."""

        active_slots = state.active_slots
        active_ids = ple_input_ids[list(active_slots)]
        active_requests = tuple(state.request_ids[slot] for slot in active_slots)
        selected = self.ple_store.prepare(active_requests, active_ids)
        full = torch.zeros((self.max_batch, 1, HIDDEN_SIZE), dtype=torch.bfloat16)
        full[list(active_slots)] = selected
        return layer.ple_staging.upload_decode(full)

    def _decode_stack_eager(self, state: Qwen38BatchState, ple_input_ids: torch.Tensor):
        residual = self.embed_tokens(state.token_input)
        for layer in self.layers:
            if layer.shapes.has_ple:
                ple_embeddings = self._stage_active_ple_decode(layer, state, ple_input_ids)
                output = layer.decode_forward_fractured(
                    residual,
                    current_pos=state.current_pos,
                    ple_embeddings=ple_embeddings,
                )
            else:
                kwargs = {"current_pos": state.current_pos}
                if layer.shapes.layer_type != LINEAR_ATTENTION:
                    kwargs.update(page_table=state.page_table, rot_mats=self.rot_mats)
                output = layer.decode_forward_fractured(residual, **kwargs)
            _functional_decoder._free(residual, output)
            residual = output
        logits = self.project_logits(residual)
        ttnn.deallocate(residual)
        return logits

    def decode_forward(
        self,
        tokens,
        start_pos=None,
        *,
        state: Qwen38BatchState,
        page_table: torch.Tensor | None = None,
        kv_cache=None,
        prompt_lens=None,
        active_mask=None,
        request_ids=None,
        enable_trace: bool = True,
        on_device_sampling: bool = False,
    ):
        """Serving-ready low-level one-token decode.

        ``tokens`` is the compact host shadow used only by exact PLE lookup;
        the optimized trace consumes ``state.token_input`` directly.  A caller
        may pass a torch token vector to explicitly refresh that persistent
        buffer (teacher forcing/host compatibility).  Free-running token-out
        decode passes the already returned compact token shadow while leaving
        the device feedback buffer untouched.
        """

        self._require_state_buffers(state)
        if kv_cache is not None and kv_cache is not self.kv_cache:
            raise ValueError("kv_cache must be the model-owned stable cache")
        if page_table is not None:
            self.update_page_table(state, page_table)
        if prompt_lens is not None and not torch.equal(torch.as_tensor(prompt_lens).reshape(-1), state.prompt_lens):
            raise ValueError("prompt_lens changed after fixed-slot prefill")
        if active_mask is not None and not torch.equal(
            torch.as_tensor(active_mask).bool().reshape(-1), state.active_mask
        ):
            raise ValueError("active slots cannot change inside a fixed decode cohort")
        if request_ids is not None and tuple(request_ids) != state.request_ids:
            raise ValueError("request IDs cannot change inside a fixed decode cohort")
        if start_pos is not None and enable_trace:
            raise ValueError(
                "traced decode owns device positions; start_pos is accepted only for explicit eager refresh"
            )

        ple_ids = torch.as_tensor(tokens, dtype=torch.int64, device="cpu").reshape(self.max_batch, 1)
        if start_pos is not None:
            positions = torch.as_tensor(start_pos, dtype=torch.int32, device="cpu").reshape(self.max_batch)
            _copy_host_to_device(positions, state.current_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            state.position_host_copies += 1
        if enable_trace:
            if self.max_batch != 1:
                raise RuntimeError("host-backed segmented token-out trace is currently a batch-one optimized path")
            logits, sampled = self.decode_token_out_traced(state, ple_ids)
            return sampled if on_device_sampling else logits
        logits = self._decode_stack_eager(state, ple_ids)
        if on_device_sampling:
            sampled = self.sample_logits(logits, state)
            ttnn.plus_one(state.current_pos, skip_negative_entries=True)
            return sampled
        return logits

    def copy_tokens(self, state: Qwen38BatchState, tokens: Sequence[int] | torch.Tensor) -> None:
        self._require_state_buffers(state)
        values = torch.as_tensor(tokens, dtype=torch.int32).reshape(self.max_batch)
        host = values.reshape(1, 1, 1, self.max_batch)
        _copy_host_to_device(host, state.token_input, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        state.token_host_copies += 1

    def sample_logits(self, logits, state: Qwen38BatchState, *, advance_seed: bool = True):
        if self._sampling_force_argmax:
            sampled, _ = self.sampling.decode_forward(
                logits,
                tt_out_tok=state.token_input,
            )
            return sampled
        if advance_seed:
            self._advance_sampling_seeds()
        sampled, _ = self.sampling.decode_forward(
            logits,
            k=self.sampling_k,
            p=self.sampling_p,
            temp=self.sampling_temp,
            tt_out_tok=state.token_input,
        )
        return sampled

    def _sample_logits_for_decode_trace(self, logits, state: Qwen38BatchState):
        return self.sample_logits(logits, state, advance_seed=False)

    # --------------------------------------------------------------- tracing

    def _trace_layer_kwargs(self, layer, state, ple_input_ids):
        kwargs = {"current_pos": state.current_pos}
        if layer.shapes.layer_type != LINEAR_ATTENTION:
            kwargs.update(page_table=state.page_table, rot_mats=self.rot_mats)
        if layer.shapes.has_ple:
            kwargs.update(ple_input_ids=ple_input_ids, request_ids=state.request_ids)
        return kwargs

    def _warm_trace_programs(self, state: Qwen38BatchState, ple_input_ids: torch.Tensor) -> None:
        warm_residual = self.embed_tokens(state.token_input)
        warm_logits = None
        try:
            for layer in self.layers:
                HostBackedSegmentedDecodeTrace.warm_programs(
                    layer,
                    warm_residual,
                    **self._trace_layer_kwargs(layer, state, ple_input_ids),
                )
            warm_logits = self.project_logits(warm_residual)
            self._sample_logits_for_decode_trace(warm_logits, state)
            ttnn.plus_one(state.current_pos, skip_negative_entries=True)
            ttnn.synchronize_device(self.mesh_device)
        finally:
            _deallocate(warm_logits)
            _deallocate(warm_residual)

    def capture_decode_traces(
        self,
        state: Qwen38BatchState,
        ple_input_ids: torch.Tensor,
        *,
        execution_mode: str = "token_out",
    ) -> None:
        """Capture ingress, every host-split layer, terminal, sampling, and position traces.

        Capture deploys the first decode token exactly once, matching the
        existing segmented-layer contract.  The caller consumes
        ``trace_logits``/``decode_token_input`` as that first step's outputs;
        later calls replay the retained traces.
        """

        self._require_state_buffers(state)
        if execution_mode not in {"token_out", "model_only"}:
            raise ValueError("execution_mode must be 'token_out' or 'model_only'")
        if self.max_batch != 1:
            raise RuntimeError("segmented full-stack capture currently requires max_batch=1")
        if self._trace_ready:
            if state is not self._trace_state:
                raise RuntimeError("decode traces are already bound to another state")
            return
        started = time.perf_counter()
        original_tokens = self.sampled_tokens_to_torch(state.token_input)
        original_positions = ttnn.to_torch(ttnn.get_device_tensors(state.current_pos)[0]).reshape(-1).to(torch.int32)
        self._warm_trace_programs(state, ple_input_ids)
        self.copy_tokens(state, original_tokens)
        _copy_host_to_device(
            original_positions,
            state.current_pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        state.position_host_copies += 1

        try:
            self.mesh_device.set_program_cache_misses_allowed(False)
            self.ingress_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.ingress_trace_output = self.embed_tokens(state.token_input)
            ttnn.end_trace_capture(self.mesh_device, self.ingress_trace_id, cq_id=0)
            ttnn.mark_corruptible(self.ingress_trace_output)
            # Trace capture registers commands but does not deploy them.  The
            # segmented layer captures below consume this fixed output now,
            # so make the capture token's ingress an ordinary first replay.
            ttnn.execute_trace(self.mesh_device, self.ingress_trace_id, cq_id=0, blocking=True)
        finally:
            self.mesh_device.set_program_cache_misses_allowed(True)

        residual = self.ingress_trace_output
        try:
            for layer in self.layers:
                trace = HostBackedSegmentedDecodeTrace.capture(
                    layer,
                    residual,
                    programs_prepared=True,
                    **self._trace_layer_kwargs(layer, state, ple_input_ids),
                )
                self.layer_traces.append(trace)
                residual = trace.output

            self.mesh_device.set_program_cache_misses_allowed(False)
            self.terminal_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.trace_logits = self.project_logits(residual)
            ttnn.end_trace_capture(self.mesh_device, self.terminal_trace_id, cq_id=0)
            ttnn.mark_corruptible(self.trace_logits)
            ttnn.execute_trace(self.mesh_device, self.terminal_trace_id, cq_id=0, blocking=True)

            self._advance_sampling_seeds()
            self.sampling_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self._sample_logits_for_decode_trace(self.trace_logits, state)
            ttnn.end_trace_capture(self.mesh_device, self.sampling_trace_id, cq_id=0)

            self.position_trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            ttnn.plus_one(state.current_pos, skip_negative_entries=True)
            ttnn.end_trace_capture(self.mesh_device, self.position_trace_id, cq_id=0)
            if execution_mode == "token_out":
                ttnn.execute_trace(self.mesh_device, self.sampling_trace_id, cq_id=0, blocking=True)
                ttnn.execute_trace(self.mesh_device, self.position_trace_id, cq_id=0, blocking=True)
        except BaseException:
            self.release_decode_traces()
            raise
        finally:
            self.mesh_device.set_program_cache_misses_allowed(True)

        self._trace_state = state
        self._trace_ready = True
        self._trace_execution_mode = execution_mode
        self.trace_capture_seconds = time.perf_counter() - started

    def _replay_decode_traces(self, state: Qwen38BatchState, ple_input_ids: torch.Tensor) -> None:
        if not self._trace_ready or state is not self._trace_state:
            raise RuntimeError("decode traces are not captured for this state")
        started = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, self.ingress_trace_id, cq_id=0, blocking=False)
        layer_seconds = 0.0
        expert_seconds = 0.0
        route_stall_seconds = 0.0
        cache_submit_seconds = 0.0
        trace_submit_seconds = 0.0
        ple_seconds = 0.0
        per_layer = []
        for trace in self.layer_traces:
            kwargs = {}
            if trace.layer.shapes.has_ple:
                kwargs = {"ple_input_ids": ple_input_ids, "request_ids": state.request_ids}
            layer_started = time.perf_counter()
            trace.replay(**kwargs)
            layer_seconds += time.perf_counter() - layer_started
            expert_seconds += float(trace.last_timing["expert_service_seconds"])
            route_stall_seconds += float(trace.last_timing["route_read_and_tt_stall_seconds"])
            cache_submit_seconds += float(trace.last_timing["cache_control_dma_submit_seconds"])
            trace_submit_seconds += float(trace.last_timing["front_trace_seconds"])
            trace_submit_seconds += float(trace.last_timing["back_trace_seconds"])
            ple_seconds += float(trace.last_timing["ple_seconds"])
            per_layer.append(
                {
                    "layer": int(trace.layer.shapes.layer_idx),
                    "type": str(trace.layer.shapes.layer_type),
                    **{key: value for key, value in trace.last_timing.items() if isinstance(value, (int, float))},
                }
            )
        ttnn.execute_trace(self.mesh_device, self.terminal_trace_id, cq_id=0, blocking=False)
        self._advance_sampling_seeds()
        ttnn.execute_trace(self.mesh_device, self.sampling_trace_id, cq_id=0, blocking=False)
        ttnn.execute_trace(self.mesh_device, self.position_trace_id, cq_id=0, blocking=False)
        self.trace_replays += 1
        self.last_decode_timing = {
            "total_submit_seconds": time.perf_counter() - started,
            "layer_boundary_seconds": layer_seconds,
            "expert_service_seconds": expert_seconds,
            "route_read_and_tt_stall_seconds": route_stall_seconds,
            "expert_cache_control_dma_submit_seconds": cache_submit_seconds,
            "layer_trace_submit_seconds": trace_submit_seconds,
            "ple_service_seconds": ple_seconds,
            "trace_replay_index": self.trace_replays,
            "layers": per_layer,
        }

    def decode_token_out_traced(self, state: Qwen38BatchState, ple_input_ids: torch.Tensor):
        if self._trace_ready and self._trace_execution_mode != "token_out":
            self.release_decode_traces()
        if not self._trace_ready:
            self.capture_decode_traces(state, ple_input_ids, execution_mode="token_out")
        else:
            self._replay_decode_traces(state, ple_input_ids)
        return self.trace_logits, state.token_input

    def replay_model_only_traced(self, state: Qwen38BatchState, ple_input_ids: torch.Tensor):
        """Explicit host-sampling compatibility replay.

        All model compute remains traced.  The caller may read full logits and
        choose/copy a token, then calls :meth:`advance_positions_traced`.
        This mode is excluded from token-out measurements.
        """

        if not self._trace_ready:
            self.capture_decode_traces(state, ple_input_ids, execution_mode="model_only")
            return self.trace_logits
        ttnn.execute_trace(self.mesh_device, self.ingress_trace_id, cq_id=0, blocking=False)
        for trace in self.layer_traces:
            kwargs = {}
            if trace.layer.shapes.has_ple:
                kwargs = {"ple_input_ids": ple_input_ids, "request_ids": state.request_ids}
            trace.replay(**kwargs)
        ttnn.execute_trace(self.mesh_device, self.terminal_trace_id, cq_id=0, blocking=False)
        return self.trace_logits

    def advance_positions_traced(self) -> None:
        if not self._trace_ready:
            raise RuntimeError("position trace is not captured")
        ttnn.execute_trace(self.mesh_device, self.position_trace_id, cq_id=0, blocking=False)

    def release_decode_traces(self) -> None:
        terminal_output = self.trace_logits
        ingress_output = self.ingress_trace_output
        for trace_id_name in ("position_trace_id", "sampling_trace_id", "terminal_trace_id"):
            trace_id = getattr(self, trace_id_name, None)
            if trace_id is not None:
                ttnn.release_trace(self.mesh_device, trace_id)
                setattr(self, trace_id_name, None)
        for trace in reversed(self.layer_traces):
            trace.release()
        self.layer_traces.clear()
        if self.ingress_trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.ingress_trace_id)
            self.ingress_trace_id = None
        _deallocate(terminal_output)
        _deallocate(ingress_output)
        self.ingress_trace_output = None
        self.trace_logits = None
        self._trace_ready = False
        self._trace_execution_mode = None
        self._trace_state = None

    # ------------------------------------------------------------------ audit

    def host_service_totals(self) -> dict[str, float]:
        """Snapshot numeric host-boundary counters for decode-window deltas."""

        expert_metrics = [
            layer.host_expert_cache.metrics() for layer in self.layers if layer.host_expert_cache is not None
        ]
        ple = self.ple_store.metrics()
        ple_device = next(
            (layer.ple_staging.metrics() for layer in self.layers if layer.ple_staging is not None),
            {},
        )
        return {
            f"expert_{name}": sum(float(metrics.get(name, 0)) for metrics in expert_metrics)
            for name in (
                "hits",
                "misses",
                "packed_host_hits",
                "packed_host_misses",
                "h2d_bytes",
                "zero_d2d_bytes",
                "source_pack_seconds",
                "h2d_seconds",
                "index_h2d_bytes",
                "index_upload_seconds",
                "deferred_dma_misses",
                "dma_completion_syncs",
            )
        } | {
            "ple_lookup_calls": float(ple["lookup_calls"]),
            "ple_table_rows_read": float(ple["table_rows_read"]),
            "ple_table_bytes_read": float(ple["table_bytes_read"]),
            "ple_lookup_seconds": float(ple["lookup_seconds"]),
            "ple_device_h2d_bytes": float(ple_device.get("h2d_bytes", 0)),
            "ple_device_h2d_seconds": float(ple_device.get("h2d_seconds", 0)),
            "ple_device_deferred_uploads": float(ple_device.get("deferred_uploads", 0)),
            "ple_device_completion_syncs": float(ple_device.get("completion_syncs", 0)),
        }

    def runtime_fallback_audit(self, state: Qwen38BatchState) -> dict[str, object]:
        return {
            "declared_host_work": {
                "model_load_exact_expert_prepack": self.prepack_host_experts,
                "expert_route_id_read_and_exact_weight_dma": True,
                "ple_ngram_hash_row_lookup_and_dma": True,
                "caller_visible_compact_token_readback": True,
                "explicit_non_greedy_seed_control_h2d": self._sampling_seed_rngs is not None,
            },
            "prohibited_host_work": {
                "expert_projection": False,
                "ple_projection": False,
                "activation_roundtrip": False,
                "kv_or_recurrence": False,
                "optimized_sampling_or_argmax": False,
                "token_feedback_reconstruction": False,
                "per_token_position_refresh": False,
                "unchanged_page_table_refresh": False,
            },
            "ownership": {
                "kv_cache": "model",
                "recurrence": "model",
                "page_table": "state with stable model buffer",
                "tokens_and_positions": "device feedback after request reset",
                "expert_store": (
                    "per-layer exact mmap source, model-load packed-host preload, and fixed TT slots"
                    if self.prepack_host_experts
                    else "per-layer exact mmap source, lazy packed-host cache, and fixed TT slots"
                ),
                "ple_store": "shared exact mmap table with request-isolated two-token history",
            },
            "counters": {
                "trace_replays": self.trace_replays,
                "token_host_copies": state.token_host_copies,
                "position_host_copies": state.position_host_copies,
                "page_table_host_copies": state.page_table_host_copies,
                "page_table_unchanged_skips": state.page_table_unchanged_skips,
                "compact_token_readbacks": state.compact_token_readbacks,
                "sampling_seed_host_copies": self.sampling_seed_host_copies,
            },
            "ple": self.ple_store.metrics(),
            "host_preload": self.host_preload_report,
            "ple_device_staging": next(
                (layer.ple_staging.metrics() for layer in self.layers if layer.ple_staging is not None),
                None,
            ),
            "experts": {
                str(layer.shapes.layer_idx): layer.host_expert_cache.metrics()
                for layer in self.layers
                if layer.host_expert_cache is not None
            },
        }

    # ---------------------------------------------------------------- release

    def close(self, *, best_effort: bool = False) -> None:
        if self._closed:
            return
        failures = []
        try:
            self.release_decode_traces()
        except BaseException as error:
            failures.append(error)
        for layer in reversed(getattr(self, "layers", ())):
            try:
                layer.close_host_backing()
            except BaseException as error:
                failures.append(error)
        workspace = getattr(self, "decode_state_workspace", None)
        if workspace is not None:
            try:
                workspace.close()
            except BaseException as error:
                failures.append(error)
        ple_store = getattr(self, "ple_store", None)
        if ple_store is not None:
            try:
                ple_store.close()
            except BaseException as error:
                failures.append(error)
        sampling = getattr(self, "sampling", None)
        if sampling is not None:
            try:
                sampling.release()
            except BaseException as error:
                failures.append(error)
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None:
            for tensor in getattr(lm_head, "output_weights", ()):
                try:
                    _deallocate(tensor)
                except BaseException as error:
                    failures.append(error)
        for name in (
            "embedding_weight",
            "final_norm_weight",
            "final_down_weight",
            "final_up_weight",
            "decode_token_input",
            "decode_current_pos",
            "decode_page_table",
            "sampling_k",
            "sampling_p",
            "sampling_temp",
        ):
            try:
                _deallocate(getattr(self, name, None))
            except BaseException as error:
                failures.append(error)
        for tensor in getattr(self, "rot_mats", ()):
            try:
                _deallocate(tensor)
            except BaseException as error:
                failures.append(error)
        self._closed = True
        if failures and not best_effort:
            raise failures[0]


__all__ = [
    "BLOCK_SIZE",
    "DEFAULT_SNAPSHOT",
    "EOS_TOKEN_IDS",
    "MODEL_ID",
    "MODEL_REVISION",
    "PAD_TOKEN_ID",
    "Qwen38BatchState",
    "Qwen38FullModel",
    "REQUIRED_L1_SMALL_SIZE",
]
