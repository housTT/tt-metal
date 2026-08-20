# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TP4 autoregressive model for Qwen/Qwen3.6-27B.

This module deliberately builds on :class:`MultichipDecoder`; it does not
contain a second, less optimized decoder path.  The embedding and final norm
are replicated and the LM head is vocabulary-column sharded, so the residual
contract between all 64 decoder layers remains replicated BF16 TILE/DRAM.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import (
    MESH_SHAPE,
    TP_SIZE,
    MultichipDecoder,
    _mesh_sharded_weight,
    _replicated_weight,
)
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    _decode_l1_memory,
    _norm_l1_memory,
)
from models.common.sampling.generator import SamplingGenerator
from models.common.modules.tt_ccl import get_tt_ccl
from models.tt_transformers.tt.rope import HfRotarySetup


MODEL_ID = "Qwen/Qwen3.6-27B"
MAX_BATCH_SIZE = 32
PAGE_BLOCK_SIZE = 64
PADDED_VOCAB_SIZE = 262_144
LOCAL_VOCAB_SIZE = PADDED_VOCAB_SIZE // TP_SIZE
# Blackhole's established full-vocabulary LM-head ceiling is roughly four
# thousand local columns per DRAM matmul; larger splits exceed L1 circular
# buffer capacity even though the output itself fits.
LM_HEAD_SPLIT_SIZE = 4096
LM_HEAD_CORES = 16
if LOCAL_VOCAB_SIZE % LM_HEAD_SPLIT_SIZE or LM_HEAD_SPLIT_SIZE % (32 * LM_HEAD_CORES):
    raise ValueError("LM-head split must divide the local vocabulary and tile evenly over cores")


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _resolve_checkpoint(checkpoint: str | Path | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    hub = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots"
    snapshots = sorted(path for path in hub.glob("*") if (path / "config.json").exists())
    if not snapshots:
        raise FileNotFoundError(
            f"No local {MODEL_ID} snapshot found under {hub}; pass checkpoint_path explicitly"
        )
    return snapshots[-1]


class LazySafetensorState:
    """Read only the checkpoint tensors needed by the layer currently built."""

    def __init__(self, checkpoint: Path):
        index_path = checkpoint / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(index_path)
        self.checkpoint = checkpoint
        self.weight_map = json.loads(index_path.read_text())["weight_map"]

    def tensor(self, name: str) -> torch.Tensor:
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"Checkpoint tensor not found: {name}")
        with safe_open(self.checkpoint / filename, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        prefix = f"model.language_model.layers.{layer_idx}."
        return {name: self.tensor(name) for name in self.weight_map if name.startswith(prefix)}


@dataclass
class FullModelState:
    """Explicit mutable state shared by low-level prefill and decode calls."""

    page_table_host: torch.Tensor
    page_table: ttnn.Tensor
    kv_cache: list[Any]
    linear_state: list[Any]
    num_blocks: int
    owns_page_table: bool
    current_positions: ttnn.Tensor | None = None
    rotary_positions: ttnn.Tensor | None = None
    token_buffer: ttnn.Tensor | None = None
    prompt_lens: tuple[int, ...] = ()
    active_slots: tuple[int, ...] = ()


class VocabParallelLMHead:
    """BFP8 TP4 LM head with a power-of-two local sampling width.

    Each device owns 65,536 vocabulary columns.  Sixteen 4,096-column matmuls
    stay below the Blackhole decode L1 circular-buffer ceiling while retaining
    one physical copy of every weight.  Columns beyond the real 248,320-token
    vocabulary are masked on device before sampling.
    """

    def __init__(self, weight: torch.Tensor, mesh_device, *, hidden_size: int, vocab_size: int):
        if tuple(weight.shape) != (vocab_size, hidden_size):
            raise ValueError(f"Unexpected LM-head shape {tuple(weight.shape)}")
        self.mesh_device = mesh_device
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.padded_vocab_size = PADDED_VOCAB_SIZE
        padded = torch.nn.functional.pad(weight, (0, 0, 0, PADDED_VOCAB_SIZE - vocab_size))
        self.weights = []
        for split_start in range(0, LOCAL_VOCAB_SIZE, LM_HEAD_SPLIT_SIZE):
            device_parts = []
            for device_idx in range(TP_SIZE):
                start = device_idx * LOCAL_VOCAB_SIZE + split_start
                device_parts.append(padded[start : start + LM_HEAD_SPLIT_SIZE])
            combined = torch.cat(device_parts, dim=0).transpose(-2, -1).contiguous()
            self.weights.append(
                _mesh_sharded_weight(
                    combined,
                    mesh_device,
                    shard_dim=-1,
                    dtype=ttnn.bfloat8_b,
                    dram_sharded=True,
                    local_k=hidden_size,
                    local_n=LM_HEAD_SPLIT_SIZE,
                )
            )
        del padded

        mask = torch.zeros((1, 1, MAX_BATCH_SIZE, PADDED_VOCAB_SIZE), dtype=torch.bfloat16)
        mask[..., vocab_size:] = torch.finfo(torch.bfloat16).min
        self.invalid_token_mask = ttnn.from_torch(
            mask,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _linear(self, hidden_states: ttnn.Tensor, weight: ttnn.Tensor, *, decode: bool):
        del decode
        cores = LM_HEAD_CORES
        hidden_states = ttnn.to_memory_config(
            hidden_states, _norm_l1_memory(MAX_BATCH_SIZE, self.hidden_size, cores)
        )
        blocks = self.hidden_size // 32 // cores
        in0_block_w = next(divisor for divisor in (10, 5, 2, 1) if blocks % divisor == 0)
        return ttnn.linear(
            hidden_states,
            weight,
            dtype=ttnn.bfloat16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=LM_HEAD_SPLIT_SIZE // 32 // cores,
                fused_activation=None,
            ),
            memory_config=_decode_l1_memory(MAX_BATCH_SIZE, LM_HEAD_SPLIT_SIZE, cores),
            compute_kernel_config=self.compute_kernel_config,
        )

    def __call__(self, hidden_states: ttnn.Tensor, *, decode: bool, mask_invalid: bool = True):
        if decode:
            chunks = [hidden_states]
            chunk_lengths = [hidden_states.shape[-2]]
        else:
            chunks = []
            chunk_lengths = []
            for start in range(0, hidden_states.shape[-2], MAX_BATCH_SIZE):
                end = min(start + MAX_BATCH_SIZE, hidden_states.shape[-2])
                chunk = ttnn.slice(
                    hidden_states,
                    [0, 0, start, 0],
                    [hidden_states.shape[0], 1, end, self.hidden_size],
                )
                if end - start < MAX_BATCH_SIZE:
                    chunk = ttnn.pad(
                        chunk,
                        padding=[(0, 0), (0, 0), (0, MAX_BATCH_SIZE - (end - start)), (0, 0)],
                        value=0.0,
                    )
                chunks.append(chunk)
                chunk_lengths.append(end - start)
        logit_chunks = []
        for chunk, logical_rows in zip(chunks, chunk_lengths):
            outputs = [self._linear(chunk, weight, decode=True) for weight in self.weights]
            outputs = [
                ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
                if output.is_sharded()
                else output
                for output in outputs
            ]
            output = ttnn.concat(outputs, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if logical_rows != output.shape[-2]:
                output = ttnn.slice(
                    output,
                    [0, 0, 0, 0],
                    [output.shape[0], output.shape[1], logical_rows, output.shape[-1]],
                )
            logit_chunks.append(output)
        logits = logit_chunks[0] if len(logit_chunks) == 1 else ttnn.concat(logit_chunks, dim=-2)
        if mask_invalid:
            mask = self.invalid_token_mask
            if logits.shape[-2] != MAX_BATCH_SIZE:
                mask = ttnn.slice(mask, [0, 0, 0, 0], [1, 1, logits.shape[-2], PADDED_VOCAB_SIZE])
            logits = ttnn.add(logits, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits


class QwenFullModel:
    """Qwen3.6 text-only autoregressive stack on the selected 1x4 TP mesh."""

    def __init__(
        self,
        *,
        mesh_device,
        checkpoint_path: str | Path | None = None,
        max_seq_len: int | None = None,
        override_num_layers: int | None = None,
        override_layer_indices: list[int] | tuple[int, ...] | None = None,
    ):
        if mesh_device.get_num_devices() != TP_SIZE or tuple(mesh_device.shape) != MESH_SHAPE:
            raise ValueError("QwenFullModel requires a 1x4 mesh; fallback meshes are not supported")
        self.mesh_device = mesh_device
        self.checkpoint_path = _resolve_checkpoint(checkpoint_path)
        root_config = AutoConfig.from_pretrained(self.checkpoint_path, local_files_only=True)
        self.config = root_config.text_config
        self.max_seq_len = max_seq_len or self.config.max_position_embeddings
        if not 1 <= self.max_seq_len <= self.config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the checkpoint context contract")
        if override_layer_indices is not None:
            if override_num_layers is not None:
                raise ValueError("override_num_layers and override_layer_indices are mutually exclusive")
            self.layer_indices = tuple(int(index) for index in override_layer_indices)
            if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError("override_layer_indices must be non-empty and unique")
            if min(self.layer_indices) < 0 or max(self.layer_indices) >= self.config.num_hidden_layers:
                raise ValueError("override_layer_indices contains an out-of-range checkpoint layer")
        else:
            num_layers = override_num_layers or self.config.num_hidden_layers
            if not 1 <= num_layers <= self.config.num_hidden_layers:
                raise ValueError("override_num_layers is outside the checkpoint layer range")
            self.layer_indices = tuple(range(num_layers))
        self.num_layers = len(self.layer_indices)
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.page_block_size = PAGE_BLOCK_SIZE
        self.max_batch_size = MAX_BATCH_SIZE

        checkpoint = LazySafetensorState(self.checkpoint_path)
        embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.embedding_weight = ttnn.as_tensor(
            embedding,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        del embedding

        self.layers = []
        for layer_idx in self.layer_indices:
            layer_state = checkpoint.layer(layer_idx)
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    layer_state,
                    hf_config=self.config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    page_block_size=PAGE_BLOCK_SIZE,
                )
            )
            del layer_state

        norm = checkpoint.tensor("model.language_model.norm.weight").float() + 1.0
        self.final_norm = _replicated_weight(norm.reshape(1, 1, 1, -1), mesh_device)
        del norm
        lm_head = checkpoint.tensor("lm_head.weight")
        self.lm_head = VocabParallelLMHead(
            lm_head, mesh_device, hidden_size=self.hidden_size, vocab_size=self.vocab_size
        )
        del lm_head

        rotary_dim = int(self.config.head_dim * self.config.partial_rotary_factor)
        self.rope = HfRotarySetup(
            device=mesh_device,
            batch_size=MAX_BATCH_SIZE,
            head_dim=rotary_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=float(self.config.rope_parameters["rope_theta"]),
            rope_scaling=None,
            datatype=ttnn.bfloat16,
        )
        self.hf_rope = Qwen3_5TextRotaryEmbedding(self.config)
        self.sampling = SamplingGenerator(
            args=self._sampling_args(), mesh_device=mesh_device, tt_ccl=get_tt_ccl(mesh_device)
        )

    def _sampling_args(self):
        return SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=PADDED_VOCAB_SIZE,
            cluster_shape=tuple(self.mesh_device.shape),
            sampling_all_gather_axis=1,
            sampling_dp=1,
            num_devices=TP_SIZE,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": True,
                    # This is a physical four-device ring, not a smaller
                    # logical slice of an eight-device topology.
                    "allow_small_ring": True,
                    "allow_small_ring_sampling": True,
                    "num_links": 1,
                    "chunks_per_sync": 10,
                    "topology": ttnn.Topology.Ring,
                }
            },
            is_galaxy=False,
            max_batch_size=MAX_BATCH_SIZE,
            max_top_k=32,
            use_topk_logprobs=False,
            allow_force_argmax=True,
            trace_seeded_sampling=True,
        )

    def _to_replicated(
        self,
        value: torch.Tensor,
        *,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            value,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def allocate_state(self, *, batch_size: int, page_table: torch.Tensor | None = None) -> FullModelState:
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be in [1, {MAX_BATCH_SIZE}]")
        total_blocks = math.ceil(self.max_seq_len / PAGE_BLOCK_SIZE)
        owns_page_table = page_table is None
        if owns_page_table:
            # The context contract is a total physical KV-token budget, not
            # max_context multiplied by 32 slots.  Mixed prompts are packed
            # into this pool immediately before prefill.
            page_table_host = torch.zeros((MAX_BATCH_SIZE, total_blocks), dtype=torch.int32)
            page_table_host[0] = torch.arange(total_blocks, dtype=torch.int32)
            num_blocks = total_blocks
        else:
            if (
                page_table.ndim != 2
                or not batch_size <= page_table.shape[0] <= MAX_BATCH_SIZE
            ):
                raise ValueError("page_table must have shape [active_batch, blocks]")
            if page_table.shape[1] <= 0 or torch.any(page_table < 0):
                raise ValueError("page_table must contain non-negative physical block IDs")
            page_table_host = torch.zeros((MAX_BATCH_SIZE, page_table.shape[1]), dtype=torch.int32)
            page_table_host[:batch_size] = page_table[:batch_size].to(torch.int32)
            num_blocks = int(page_table_host[:batch_size].max().item()) + 1
            if num_blocks > total_blocks:
                raise ValueError("external page table exceeds the total physical KV-token budget")
        tt_page_table = self._to_replicated(
            page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        kv_cache = []
        for layer in self.layers:
            kv_cache.append(
                layer.allocate_paged_kv_cache(num_blocks=num_blocks)
                if layer.layer_kind == "full_attention"
                else None
            )
        return FullModelState(
            page_table_host=page_table_host,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            linear_state=[None] * self.num_layers,
            num_blocks=num_blocks,
            owns_page_table=owns_page_table,
        )

    def _embed_host_tokens(self, tokens: torch.Tensor) -> ttnn.Tensor:
        tt_tokens = self._to_replicated(
            tokens.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        hidden = ttnn.embedding(
            tt_tokens, self.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        return ttnn.unsqueeze_to_4D(hidden) if len(hidden.shape) == 3 else hidden

    def _prefill_rope(self, logical_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cos, sin = self.hf_rope(
            torch.zeros((1, 1, 1), dtype=torch.bfloat16),
            torch.arange(logical_len, dtype=torch.long).unsqueeze(0),
        )
        return (
            self._to_replicated(cos.unsqueeze(1)),
            self._to_replicated(sin.unsqueeze(1)),
        )

    def prefill(
        self,
        tokens: torch.Tensor,
        *,
        prompt_lens: Iterable[int],
        state: FullModelState,
        return_all_logits: bool,
    ) -> torch.Tensor:
        prompt_lens = tuple(int(length) for length in prompt_lens)
        batch = len(prompt_lens)
        if tokens.ndim != 2 or tokens.shape[0] != batch:
            raise ValueError("tokens must have shape [batch, padded_prompt_len]")
        if any(length <= 0 or length > self.max_seq_len or length > tokens.shape[1] for length in prompt_lens):
            raise ValueError("prompt lengths exceed token storage or supported context")

        if state.owns_page_table:
            required_blocks = [math.ceil(length / PAGE_BLOCK_SIZE) for length in prompt_lens]
            if sum(required_blocks) > state.num_blocks:
                raise ValueError(
                    "mixed prompts exceed the advertised total physical KV-cache token budget"
                )
            state.page_table_host.zero_()
            next_block = 0
            for user, count in enumerate(required_blocks):
                state.page_table_host[user, :count] = torch.arange(
                    next_block, next_block + count, dtype=torch.int32
                )
                next_block += count
            page_source = self._to_replicated(
                state.page_table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy(page_source, state.page_table)

        physical_lens = tuple(_round_up(length, 32) for length in prompt_lens)
        hidden_by_user = []
        for user, (logical_len, physical_len) in enumerate(zip(prompt_lens, physical_lens)):
            token_row = torch.zeros((physical_len,), dtype=tokens.dtype)
            token_row[:logical_len] = tokens[user, :logical_len]
            hidden_by_user.append(self._embed_host_tokens(token_row))
        rope_by_len: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        page_rows = [
            self._to_replicated(
                state.page_table_host[user : user + 1],
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for user in range(batch)
        ]

        for layer_idx, layer in enumerate(self.layers):
            outputs = []
            user_linear_states = []
            for user, logical_len in enumerate(prompt_lens):
                hidden = hidden_by_user[user]
                physical_len = physical_lens[user]
                if hidden.shape[2] != physical_len:
                    hidden = ttnn.pad(
                        hidden,
                        padding=[(0, 0), (0, 0), (0, physical_len - hidden.shape[2]), (0, 0)],
                        value=0.0,
                    )
                if layer.layer_kind == "full_attention":
                    cos, sin = rope_by_len.setdefault(physical_len, self._prefill_rope(physical_len))
                    output = layer.prefill_forward(
                        hidden,
                        logical_seq_len=logical_len,
                        cos=cos,
                        sin=sin,
                        page_table=page_rows[user],
                        kv_cache=state.kv_cache[layer_idx],
                    )
                else:
                    user_state = layer.allocate_linear_state(batch_size=1)
                    output = layer.prefill_forward(
                        hidden, logical_seq_len=logical_len, linear_state=user_state
                    )
                    user_linear_states.append(user_state)
                outputs.append(output)
            hidden_by_user = outputs
            if layer.layer_kind == "linear_attention":
                packed_state = (
                    ttnn.concat([item[0] for item in user_linear_states], dim=0),
                    ttnn.concat([item[1] for item in user_linear_states], dim=0),
                )
                if state.linear_state[layer_idx] is None:
                    state.linear_state[layer_idx] = packed_state
                else:
                    ttnn.copy(packed_state[0], state.linear_state[layer_idx][0])
                    ttnn.copy(packed_state[1], state.linear_state[layer_idx][1])

        host_logits = []
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        for user, hidden in enumerate(hidden_by_user):
            normalized = ttnn.rms_norm(
                hidden, epsilon=self.config.rms_norm_eps, weight=self.final_norm
            )
            if not return_all_logits:
                length = prompt_lens[user]
                normalized = ttnn.slice(
                    normalized, [0, 0, length - 1, 0], [1, 1, length, self.hidden_size]
                )
            logits = self.lm_head(normalized, decode=False, mask_invalid=False)
            host = ttnn.to_torch(logits, mesh_composer=composer)[0, 0, :, : self.vocab_size]
            if return_all_logits:
                # Internal tile padding is never part of the public contract.
                # Preserve the caller's common prompt-storage width so mixed
                # logical lengths remain stackable, and leave inactive suffix
                # rows as zeros.
                public_rows = tokens.shape[1]
                if host.shape[0] < public_rows:
                    host = torch.nn.functional.pad(host, (0, 0, 0, public_rows - host.shape[0]))
                else:
                    host = host[:public_rows]
            host_logits.append(host)
        state.prompt_lens = prompt_lens
        state.active_slots = tuple(range(batch))
        return torch.stack(host_logits, dim=0)

    def prepare_decode_state(
        self,
        state: FullModelState,
        first_tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> None:
        """Initialize or refresh persistent decode inputs without changing tensor identity."""

        first_tokens = first_tokens.reshape(-1)
        batch = first_tokens.numel()
        if positions is None:
            if len(state.prompt_lens) != batch:
                raise ValueError("prompt_lens must match first_tokens when positions are omitted")
            positions = torch.tensor(state.prompt_lens, dtype=torch.int32)
        else:
            positions = positions.reshape(-1).to(torch.int32)
            if positions.numel() != batch:
                raise ValueError("positions must match first_tokens")
            state.prompt_lens = tuple(int(value) for value in positions.tolist())
        if torch.any(positions < 0) or torch.any(positions >= self.max_seq_len):
            raise ValueError("decode positions exceed the supported context")
        state.active_slots = tuple(range(batch))

        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_kind == "linear_attention" and state.linear_state[layer_idx] is None:
                state.linear_state[layer_idx] = layer.allocate_linear_state(batch_size=batch)

        tokens = torch.zeros((1, 1, 1, MAX_BATCH_SIZE), dtype=torch.int32)
        tokens[0, 0, 0, :batch] = first_tokens.to(torch.int32)
        current = torch.full((MAX_BATCH_SIZE,), -1, dtype=torch.int32)
        current[:batch] = positions
        rotary = torch.zeros((1, MAX_BATCH_SIZE), dtype=torch.int32)
        rotary[0, :batch] = current[:batch]
        values = (
            ("token_buffer", tokens, ttnn.uint32),
            ("current_positions", current, ttnn.int32),
            ("rotary_positions", rotary, ttnn.uint32),
        )
        for name, host, dtype in values:
            target = getattr(state, name)
            source = self._to_replicated(host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
            if target is None:
                setattr(state, name, source)
            else:
                ttnn.copy(source, target)

    def decode_device(self, state: FullModelState) -> ttnn.Tensor:
        if any(value is None for value in (state.token_buffer, state.current_positions, state.rotary_positions)):
            raise RuntimeError("prepare_decode_state must be called before decode")
        hidden = ttnn.embedding(
            state.token_buffer,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        if len(hidden.shape) == 3:
            hidden = ttnn.unsqueeze_to_4D(hidden)
        cos, sin = self.rope.get_rot_mats(state.rotary_positions)
        active_batch = len(state.active_slots)
        if not active_batch or state.active_slots != tuple(range(active_batch)):
            raise ValueError("decode currently requires stable contiguous slots [0, active_batch)")
        for layer_idx, layer in enumerate(self.layers):
            if layer.layer_kind == "full_attention":
                if hidden.shape[2] != MAX_BATCH_SIZE:
                    hidden = ttnn.pad(
                        hidden,
                        padding=[(0, 0), (0, 0), (0, MAX_BATCH_SIZE - hidden.shape[2]), (0, 0)],
                        value=0.0,
                    )
                hidden = layer.decode_forward(
                    hidden,
                    current_positions=state.current_positions,
                    cos=cos,
                    sin=sin,
                    page_table=state.page_table,
                    kv_cache=state.kv_cache[layer_idx],
                )
            else:
                if hidden.shape[2] != active_batch:
                    hidden = ttnn.slice(
                        hidden,
                        [0, 0, 0, 0],
                        [1, 1, active_batch, self.hidden_size],
                    )
                hidden = layer.decode_forward(
                    hidden,
                    current_positions=state.current_positions,
                    linear_state=state.linear_state[layer_idx],
                )
        if hidden.shape[2] != MAX_BATCH_SIZE:
            hidden = ttnn.pad(
                hidden,
                padding=[(0, 0), (0, 0), (0, MAX_BATCH_SIZE - hidden.shape[2]), (0, 0)],
                value=0.0,
            )
        hidden = ttnn.rms_norm(hidden, epsilon=self.config.rms_norm_eps, weight=self.final_norm)
        logits = self.lm_head(hidden, decode=True, mask_invalid=True)
        ttnn.plus_one(state.current_positions, skip_negative_entries=True)
        ttnn.plus_one(state.rotary_positions)
        return logits


__all__ = [
    "FullModelState",
    "MAX_BATCH_SIZE",
    "MODEL_ID",
    "PADDED_VOCAB_SIZE",
    "QwenFullModel",
    "VocabParallelLMHead",
]
