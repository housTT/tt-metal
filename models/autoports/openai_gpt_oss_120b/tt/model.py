# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resident full-model assembly for the optimized GPT-OSS 120B autoport.

The terminal embedding, normalization, LM-head, RoPE, and TT-Transformer
interfaces intentionally reuse the maintained GPT-OSS implementation.  Every
decoder block, however, is the autoport's optimized :class:`MultichipDecoder`;
there is no single-chip, replicated-weight, or host-executed decoder fallback.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

import ttnn
from models.autoports.openai_gpt_oss_120b.tt.multichip_decoder import (
    DECODE_K_CHUNK_SIZE,
    DEFAULT_MULTICHIP_POLICY,
    MultichipDecoder,
    MultichipDecoderPolicy,
    tensor_plan,
)
from models.autoports.openai_gpt_oss_120b.tt.optimized_decoder import _DecodeShardedRMSNorm
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.model import Model as _GPTOSSModel
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_LAYERS = 36
HF_CONTEXT_LENGTH = 131072
PAGE_SIZE = 64
DEVICE_DRAM_BYTES = 32 * 1024**3
TRACE_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
INTERLEAVED_LM_HEAD = "interleaved"
DRAM_SHARDED_LM_HEAD = "dram_sharded"
_BFP8_TILE_BYTES = 1088
_TILE_ELEMENTS = 32 * 32

# Measured physical TT tensor storage from the completed optimized multichip
# stage.  TP2 includes its duplicated DRAM-sharded O-projection tensor.
DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS = {
    1: 65_619_952_128,
    2: 32_964_795_648,
    4: 16_719_757_056,
}


class FullModelCapacityError(RuntimeError):
    """Raised instead of silently selecting a less capable runtime path."""


@dataclass(frozen=True)
class CapacityEvidence:
    tp: int
    num_layers: int
    max_batch_size: int
    max_context_length: int
    decoder_weight_bytes: int
    embedding_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int
    rope_bytes: int
    page_table_bytes: int
    kv_cache_bytes: int
    trace_activation_reserve_bytes: int
    total_bytes_per_device: int
    device_dram_bytes: int
    largest_context_for_batch: int
    fits: bool

    def to_dict(self):
        result = asdict(self)
        result["total_gib_per_device"] = self.total_bytes_per_device / 1024**3
        result["device_dram_gib"] = self.device_dram_bytes / 1024**3
        return result


def _bfp8_tensor_bytes(height: int, width: int) -> int:
    if height % 32 or width % 32:
        raise ValueError(f"BFP8 tensor dimensions must be tile aligned, got {(height, width)}")
    return height * width // _TILE_ELEMENTS * _BFP8_TILE_BYTES


def _kv_cache_bytes(*, tp: int, num_layers: int, batch_size: int, context_length: int) -> int:
    physical_context = math.ceil(context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
    blocks = math.ceil(physical_context / PAGE_SIZE)
    local_kv_heads = 8 // tp
    elements = 2 * num_layers * batch_size * blocks * local_kv_heads * PAGE_SIZE * 64
    return math.ceil(elements / _TILE_ELEMENTS) * _BFP8_TILE_BYTES


def capacity_evidence(
    *,
    tp: int,
    max_batch_size: int = 1,
    max_context_length: int = HF_CONTEXT_LENGTH,
    num_layers: int = MODEL_LAYERS,
    reserve_bytes: int = TRACE_ACTIVATION_RESERVE_BYTES,
) -> CapacityEvidence:
    """Return conservative physical TT storage for one resident TP rank."""

    if tp not in DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS:
        raise ValueError(f"capacity accounting supports TP=1/2/4, got TP={tp}")
    if not 1 <= num_layers <= MODEL_LAYERS:
        raise ValueError(f"num_layers must be within [1, {MODEL_LAYERS}], got {num_layers}")
    if not 1 <= max_batch_size <= 32:
        raise ValueError(f"max_batch_size must be within [1, 32], got {max_batch_size}")
    if not 1 <= max_context_length <= HF_CONTEXT_LENGTH:
        raise ValueError(f"max_context_length must be within [1, {HF_CONTEXT_LENGTH}], got {max_context_length}")

    decoder_weights = math.ceil(DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS[tp] * num_layers / MODEL_LAYERS)
    embedding = 201088 * 2880 * 2  # replicated BF16
    final_norm = 2880 * 2
    per_device_vocab = 262144 // tp
    lm_head = _bfp8_tensor_bytes(2880, per_device_vocab)
    rope = 2 * HF_CONTEXT_LENGTH * 64 * 2  # replicated BF16 cosine + sine
    fixed = decoder_weights + embedding + final_norm + lm_head + rope + reserve_bytes
    physical_context = math.ceil(max_context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
    page_table = max_batch_size * math.ceil(physical_context / PAGE_SIZE) * 4

    def resident_bytes(context_length: int) -> int:
        physical_context = math.ceil(context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
        return (
            fixed
            + max_batch_size * math.ceil(physical_context / PAGE_SIZE) * 4
            + _kv_cache_bytes(
                tp=tp,
                num_layers=num_layers,
                batch_size=max_batch_size,
                context_length=context_length,
            )
        )

    low, high = 0, HF_CONTEXT_LENGTH
    while low < high:
        midpoint = (low + high + 1) // 2
        if resident_bytes(midpoint) <= DEVICE_DRAM_BYTES:
            low = midpoint
        else:
            high = midpoint - 1
    largest_context = low
    kv_cache = _kv_cache_bytes(
        tp=tp,
        num_layers=num_layers,
        batch_size=max_batch_size,
        context_length=max_context_length,
    )
    total = fixed + page_table + kv_cache
    return CapacityEvidence(
        tp=tp,
        num_layers=num_layers,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        decoder_weight_bytes=decoder_weights,
        embedding_bytes=embedding,
        final_norm_bytes=final_norm,
        lm_head_bytes=lm_head,
        rope_bytes=rope,
        page_table_bytes=page_table,
        kv_cache_bytes=kv_cache,
        trace_activation_reserve_bytes=reserve_bytes,
        total_bytes_per_device=total,
        device_dram_bytes=DEVICE_DRAM_BYTES,
        largest_context_for_batch=largest_context,
        fits=total <= DEVICE_DRAM_BYTES,
    )


def require_resident_capacity(
    *,
    tp: int,
    max_batch_size: int,
    max_context_length: int,
    num_layers: int,
    allow_reduced_model: bool = False,
) -> CapacityEvidence:
    evidence = capacity_evidence(
        tp=tp,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        num_layers=num_layers,
    )
    if num_layers != MODEL_LAYERS and not allow_reduced_model:
        raise FullModelCapacityError(
            f"Production full-model construction requires {MODEL_LAYERS} layers; got {num_layers}. "
            "Reduced stacks are accepted only by explicit hardware probes."
        )
    if not evidence.fits:
        raise FullModelCapacityError(
            "Resident optimized GPT-OSS 120B does not fit this target without a forbidden fallback: "
            f"TP={tp}, batch={max_batch_size}, context={max_context_length} requires "
            f"{evidence.total_bytes_per_device / 1024**3:.3f} GiB/device including the measured "
            f"decoder policy and {TRACE_ACTIVATION_RESERVE_BYTES / 1024**3:.1f} GiB trace/activation "
            f"reserve, but P150 provides {DEVICE_DRAM_BYTES / 1024**3:.0f} GiB/device. "
            f"Largest accounted context for this batch is {evidence.largest_context_for_batch}."
        )
    return evidence


class StreamingCheckpoint:
    """Load one dense decoder layer at a time from the public MXFP4 checkpoint."""

    _INDEX = "model.safetensors.index.json"

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()
        index_path = self.snapshot_path / self._INDEX
        if not index_path.is_file():
            raise FileNotFoundError(f"GPT-OSS checkpoint index is missing: {index_path}")
        with index_path.open(encoding="utf-8") as index_file:
            self.weight_map = json.load(index_file)["weight_map"]

    def _read(self, checkpoint_key: str) -> torch.Tensor:
        try:
            shard = self.weight_map[checkpoint_key]
        except KeyError as error:
            raise KeyError(f"GPT-OSS checkpoint tensor is missing: {checkpoint_key}") from error
        shard_path = self.snapshot_path / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"GPT-OSS checkpoint shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(checkpoint_key)

    def terminal_state_dict(self) -> dict[str, torch.Tensor]:
        return {key: self._read(key) for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")}

    def layer_state_dict(self, layer_idx: int, *, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        raw = {key[len(prefix) :]: self._read(key) for key in self.weight_map if key.startswith(prefix)}
        for projection in ("gate_up_proj", "down_proj"):
            packed = f"mlp.experts.{projection}"
            blocks = raw.pop(f"{packed}_blocks")
            scales = raw.pop(f"{packed}_scales")
            raw[packed] = convert_moe_packed_tensors(blocks, scales, dtype=dtype)
        return {
            key: tensor.to(dtype=dtype) if tensor.is_floating_point() and tensor.dtype != dtype else tensor
            for key, tensor in raw.items()
        }


class FullModelArgs:
    """Small tt-transformers argument surface with no alternate model policy."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        generation_config,
        tokenizer,
        snapshot_path: Path,
        tensor_cache_path: Path,
        max_batch_size: int,
        max_context_length: int,
        num_layers: int,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.processor = None
        self.model_path = str(snapshot_path)
        self.weights_path = str(snapshot_path)
        self.tensor_cache_path = Path(tensor_cache_path)
        self.model_name = "gpt-oss-120b"
        self.vocab_size = int(hf_config.vocab_size)
        self.n_layers = int(num_layers)
        self.dim = int(hf_config.hidden_size)
        self.head_dim = int(hf_config.head_dim)
        self.max_batch_size = int(max_batch_size)
        self.max_local_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_context_length)
        self.max_context_len = int(max_context_length)
        self.decode_k_chunk_size = DECODE_K_CHUNK_SIZE
        self.physical_kv_context_len = (
            math.ceil(max_context_length / self.decode_k_chunk_size) * self.decode_k_chunk_size
        )
        self.max_prefill_chunk_size = self.physical_kv_context_len
        self.disable_batched_prefill = True
        self.capped_warmup_seq_len = min(128, max_context_length)
        self.trace_prefill_supported_seq_lens = [128] if max_context_length >= 128 else []
        self.cluster_shape = tuple(int(v) for v in mesh_device.shape)
        self.num_devices = mesh_device.get_num_devices()
        self.sampling_all_gather_axis = 1
        self.sampling_dp = 1
        self.use_topk_logprobs = True
        self.is_galaxy = False
        # Empty config selects SamplingGenerator/TTSampling's regular sharded
        # top-k path.  It deliberately disables force-argmax/full-vocab gather.
        self.model_config = {}

    @property
    def base_model_name(self):
        return self.model_name

    def is_llama_vision(self):
        return False

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        return prefill_seq_len in self.trace_prefill_supported_seq_lens and num_cached_tokens == 0

    def get_warmup_prefill_supported_seq_lens(self):
        return list(self.trace_prefill_supported_seq_lens)

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        if instruct:
            raise ValueError("GPT-OSS uses its tokenizer chat template; instruct=True is not a separate mode")
        if isinstance(prompt_text, str):
            messages = []
            if system_prompt_text:
                messages.append({"role": "system", "content": system_prompt_text})
            messages.append({"role": "user", "content": prompt_text})
        else:
            messages = prompt_text
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.reshape(-1).tolist()
        while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token) for token in encoded]


class _LayerAdapter:
    """Match the shared GPT-OSS stack call surface without changing the block."""

    def __init__(self, decoder: MultichipDecoder):
        self.decoder = decoder
        self.self_attn = decoder.self_attn

    @property
    def kv_cache(self):
        return self.decoder.kv_cache

    def __call__(
        self,
        hidden_states,
        *,
        position_embeddings,
        position_idx,
        page_table,
        kv_cache,
        is_decode,
        user_id,
        batch_size,
    ):
        if is_decode:
            decode_batch_size = int(hidden_states.shape[-2])
            if position_idx.shape[-1] < decode_batch_size:
                raise ValueError(
                    "decode current_position must cover every fixed slot, including inactive rows; "
                    f"got {position_idx.shape[-1]} positions for batch {decode_batch_size}"
                )
            return self.decoder.decode_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                current_position=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                batch_size=decode_batch_size,
            )
        return self.decoder.prefill_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            page_table=page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
        )


class _DramShardedLMHead:
    """Opt-in BFP8/HiFi2 terminal candidate split over physical DRAM banks."""

    def __init__(
        self,
        *,
        mesh_device,
        mesh_config,
        torch_weight: torch.Tensor,
        vocab_size: int,
        hidden_size: int,
        input_memory_config,
        tensor_cache_path: Path,
        split_size: int = 8192,
    ):
        tp = int(mesh_device.shape[1])
        local_vocab_size = 1 << math.ceil(math.log2(math.ceil(vocab_size / tp)))
        if local_vocab_size % split_size:
            raise ValueError(f"local padded vocabulary {local_vocab_size} must divide split size {split_size}")
        dram_grid_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        dram_banks = dram_grid.num_cores()
        if split_size % (dram_banks * ttnn.TILE_SIZE):
            raise ValueError(
                "LM-head split must divide evenly over tile-aligned DRAM banks: "
                f"split={split_size}, banks={dram_banks}"
            )
        input_shard = input_memory_config.shard_spec.shape
        if hidden_size % input_shard[1]:
            raise ValueError(f"hidden size {hidden_size} is incompatible with input shard {input_shard}")

        self.input_memory_config = input_memory_config
        self.output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=input_shard[1] // ttnn.TILE_SIZE,
            per_core_M=1,
            per_core_N=split_size // dram_banks // ttnn.TILE_SIZE,
            fused_activation=None,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.weights = []
        cache_root = Path(tensor_cache_path)
        for split_index, offset in enumerate(range(0, local_vocab_size, split_size)):
            rank_splits = []
            for rank in range(tp):
                global_start = rank * local_vocab_size + offset
                global_end = global_start + split_size
                rank_weight = torch.zeros(hidden_size, split_size, dtype=torch_weight.dtype)
                valid_end = min(global_end, vocab_size)
                if global_start < valid_end:
                    valid_width = valid_end - global_start
                    rank_weight[:, :valid_width] = torch_weight[global_start:valid_end].transpose(0, 1)
                rank_splits.append(rank_weight)
            combined_weight = torch.cat(rank_splits, dim=-1)
            weight_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (hidden_size, split_size // dram_banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self.weights.append(
                ttnn.as_tensor(
                    combined_weight,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=get_cache_file_name(
                        cache_root,
                        f"split_{split_index}_k{hidden_size}_n{split_size}_banks{dram_banks}",
                    ),
                    memory_config=weight_memory_config,
                    mesh_mapper=mesh_config.column_parallel(mesh_device),
                )
            )
            del combined_weight, rank_splits
            gc.collect()

        self.manifest = {
            "dtype": "BFP8_B",
            "compute_fidelity": "HiFi2",
            "local_padded_vocab": local_vocab_size,
            "split_size": split_size,
            "num_splits": len(self.weights),
            "dram_banks": dram_banks,
            "input_shard_shape": list(input_shard),
            "output_memory": "DRAM interleaved",
        }

    def __call__(self, hidden_states):
        owns_input = hidden_states.memory_config() != self.input_memory_config
        sharded_input = ttnn.to_memory_config(hidden_states, self.input_memory_config) if owns_input else hidden_states
        outputs = []
        for weight in self.weights:
            sharded_output = ttnn.linear(
                sharded_input,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self.program_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            outputs.append(ttnn.sharded_to_interleaved(sharded_output, memory_config=self.output_memory_config))
            sharded_output.deallocate(True)
        logits = ttnn.concat(outputs, dim=-1, memory_config=self.output_memory_config)
        for output in outputs:
            output.deallocate(True)
        if owns_input:
            sharded_input.deallocate(True)
        return logits


class Model(_GPTOSSModel):
    """Full autoregressive model with 36 optimized TP decoder layers."""

    optimization_manifest = MultichipDecoder.optimization_manifest + (
        "resident_stream_loaded_36_layer_stack",
        "replicated_bf16_embedding",
        "decode_sharded_final_rmsnorm",
        "tp_column_sharded_bfp8_pow2_lm_head",
        "canonical_sampling_generator_split_sampling",
    )

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        terminal_state_dict,
        layer_loader,
        args: FullModelArgs,
        tensor_cache_path: str | Path,
        max_batch_size: int,
        max_context_length: int,
        num_layers: int,
        policy: MultichipDecoderPolicy = DEFAULT_MULTICHIP_POLICY,
        lm_head_policy: str = INTERLEAVED_LM_HEAD,
    ):
        tp = int(mesh_device.shape[1])
        tensor_plan(mesh_device.shape, hf_config)
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=tp, ep=1, sp=1),
            prefill=ModeConfig(tp=tp, ep=1, sp=1),
        )
        ccl_manager = CCLManager(
            mesh_device,
            num_links=get_default_num_links(mesh_device),
            topology=policy.topology,
        )
        cache_root = Path(tensor_cache_path)
        cache_root.mkdir(parents=True, exist_ok=True)

        # Reuse only the maintained terminal/runtime portion of the demo model.
        # A zero-layer config prevents construction of its decoder family.
        terminal_config = copy.deepcopy(hf_config)
        terminal_config.num_hidden_layers = 0
        terminal_config.layer_types = []
        super().__init__(
            mesh_device=mesh_device,
            hf_config=terminal_config,
            state_dict=terminal_state_dict,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat8_b,
            tensor_cache_path=str(cache_root / "terminal"),
            paged_attention_config=None,
            mesh_config=mesh_config,
            create_kv_cache=False,
            max_local_batch_size=max_batch_size,
            users_row_sharded=False,
            use_throughput_experts=False,
        )
        self.hf_config = hf_config
        self.args = args
        self.vocab_size = int(hf_config.vocab_size)
        self.n_layers = int(num_layers)
        self.dtype = ttnn.bfloat8_b
        self.max_context_length = int(max_context_length)
        self.page_size = PAGE_SIZE
        self.policy = policy
        self.capacity = capacity_evidence(
            tp=tp,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
        )
        if lm_head_policy not in {INTERLEAVED_LM_HEAD, DRAM_SHARDED_LM_HEAD}:
            raise ValueError(f"unknown LM-head policy {lm_head_policy!r}")
        self.lm_head_policy = lm_head_policy
        self._terminal_uses_single_tile = False
        self.dram_sharded_lm_head = None

        # The final norm consumes the last decoder's L1 replicated residual.
        if getattr(self.norm, "tt_weight", None) is not None:
            self.norm.tt_weight.deallocate(True)
        self.norm = _DecodeShardedRMSNorm(
            mesh_device,
            hf_config,
            {"weight": terminal_state_dict["model.norm.weight"]},
            tensor_cache_path=str(cache_root / "terminal" / "final_norm_decode_sharded"),
            mesh_config=mesh_config,
            enable_decode_sharding=max_batch_size < ttnn.TILE_SIZE,
        )
        if lm_head_policy == DRAM_SHARDED_LM_HEAD:
            self.dram_sharded_lm_head = _DramShardedLMHead(
                mesh_device=mesh_device,
                mesh_config=mesh_config,
                torch_weight=terminal_state_dict["lm_head.weight"],
                vocab_size=self.vocab_size,
                hidden_size=int(hf_config.hidden_size),
                input_memory_config=self.norm.decode_memory_config,
                tensor_cache_path=cache_root / "terminal" / "lm_head_dram_sharded",
            )

        self.layers = []
        for layer_idx in range(num_layers):
            logger.info(f"Loading optimized GPT-OSS 120B layer {layer_idx + 1}/{num_layers}")
            layer_state = layer_loader(layer_idx)
            decoder = MultichipDecoder.from_state_dict(
                layer_state,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_context_length=max_context_length,
                page_size=PAGE_SIZE,
                tensor_cache_path=str(cache_root / f"layer_{layer_idx:02d}"),
                calibrated_checkpoint_revision=MODEL_REVISION,
                policy=policy,
            )
            self.layers.append(_LayerAdapter(decoder))
            del layer_state
            gc.collect()
        self.kv_cache = [layer.kv_cache for layer in self.layers]

    def _forward_layers_and_head(self, *args, is_decode=True, **kwargs):
        self.norm.decode_mode = is_decode
        self._terminal_uses_single_tile = is_decode or int(kwargs.get("get_last_token", -1)) != -1
        return super()._forward_layers_and_head(*args, is_decode=is_decode, **kwargs)

    def _apply_lm_head(self, hidden_states):
        if self.dram_sharded_lm_head is not None and self._terminal_uses_single_tile:
            return self.dram_sharded_lm_head(hidden_states)
        return super()._apply_lm_head(hidden_states)

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device,
        *,
        snapshot_path: str | Path | None = None,
        tensor_cache_path: str | Path | None = None,
        max_batch_size: int = 1,
        max_context_length: int = HF_CONTEXT_LENGTH,
        num_layers: int = MODEL_LAYERS,
        allow_reduced_model: bool = False,
        policy: MultichipDecoderPolicy = DEFAULT_MULTICHIP_POLICY,
        lm_head_policy: str = INTERLEAVED_LM_HEAD,
    ):
        snapshot_path = Path(
            snapshot_path or os.environ.get("GPT_OSS_120B_SNAPSHOT", "") or os.environ.get("HF_MODEL", "")
        ).expanduser()
        if not str(snapshot_path) or str(snapshot_path) == ".":
            raise FileNotFoundError("Set GPT_OSS_120B_SNAPSHOT or HF_MODEL to the pinned openai/gpt-oss-120b snapshot")
        snapshot_path = snapshot_path.resolve()
        hf_config = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True, local_files_only=True)
        if int(hf_config.max_position_embeddings) != HF_CONTEXT_LENGTH:
            raise ValueError(
                f"Pinned GPT-OSS config must advertise {HF_CONTEXT_LENGTH}, got {hf_config.max_position_embeddings}"
            )
        tp = int(mesh_device.shape[1])
        evidence = require_resident_capacity(
            tp=tp,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
            allow_reduced_model=allow_reduced_model,
        )
        logger.info(
            f"Resident full-model capacity: TP={tp}, {evidence.total_bytes_per_device / 1024**3:.3f} GiB/device"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        cache_path = Path(
            tensor_cache_path
            or os.environ.get("GPT_OSS_120B_FULL_MODEL_TENSOR_CACHE", "/tmp/gpt_oss_120b_full_model_tensor_cache")
        ).expanduser()
        checkpoint = StreamingCheckpoint(snapshot_path)
        args = FullModelArgs(
            mesh_device=mesh_device,
            hf_config=hf_config,
            generation_config=generation_config,
            tokenizer=tokenizer,
            snapshot_path=snapshot_path,
            tensor_cache_path=cache_path,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
        )
        terminal = checkpoint.terminal_state_dict()
        model = cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            terminal_state_dict=terminal,
            layer_loader=checkpoint.layer_state_dict,
            args=args,
            tensor_cache_path=cache_path,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
            policy=policy,
            lm_head_policy=lm_head_policy,
        )
        del terminal
        gc.collect()
        return model, args


def build_model(mesh_device, **kwargs):
    """Standard autoport builder returning model, args, and owned paged cache."""

    model, args = Model.from_checkpoint(mesh_device, **kwargs)
    return model, args, model.kv_cache


__all__ = [
    "CapacityEvidence",
    "FullModelArgs",
    "FullModelCapacityError",
    "DRAM_SHARDED_LM_HEAD",
    "HF_CONTEXT_LENGTH",
    "INTERLEAVED_LM_HEAD",
    "MODEL_ID",
    "MODEL_REVISION",
    "Model",
    "StreamingCheckpoint",
    "build_model",
    "capacity_evidence",
    "require_resident_capacity",
]
