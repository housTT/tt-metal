# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for ``Qwen/Qwen3.8-Flash-Next``.

The checkpoint resolves to Transformers' ``Qwen4ExpTextDecoderLayer``.  Its
four residual streams are carried as one 10240-wide tensor and every layer is
one of the following meaningful kinds:

``linear_attention``
    Gated DeltaNet followed by the shared 512-expert MoE.

``linear_attention`` with PLE (zero-based layer 1)
    The same layer with the projected lexical embedding injection.  The full
    hashed n-gram table is 320001536 x 160 BF16 (102400491520 bytes) and cannot
    reside on one P300.  The public boundary therefore takes the input-specific
    2560-wide ``ple_embeddings`` device tensor.  Hashing and sparse row lookup
    are setup/preprocessing; all PLE projection, gating, state and convolution
    remain in the measured device pass.

``qwen_sparse_attention``
    Qwen Sparse Attention (QSA): its indexer raw-key cache and main K/V cache
    are both paged with the caller's virtual-to-physical page table.

Public contract
---------------

``prefill_forward(hidden_states, user_id, page_table, page_tables_per_chunk,
rot_mats, ple_embeddings=None)``
    ``hidden_states`` is a TTNN tensor ``[1, 1, seq_len, 10240]``.  The call is
    single-user, accepts every logical ``1 <= seq_len <= max_seq_len`` including
    tile/page/chunk-unaligned lengths, owns padding/chunking, resets that user's
    recurrent state, and returns exactly ``[1, 1, seq_len, 10240]``.  QSA needs
    an int32 page table and partial-RoPE ``(cos, sin)`` tensors.  PLE layer 1
    needs the matching ``[1, 1, seq_len, 2560]`` device embedding tensor.

``prepare_decode_state()``
    Folds per-user post-prefill DeltaNet/PLE state into stable batch buffers.
    Call it after all users are prefilled and before trace capture.

``decode_forward(hidden_states, current_pos, page_table, rot_mats,
ple_embeddings=None)``
    ``hidden_states`` is ``[1, 1, max_batch, 10240]`` and ``current_pos`` is a
    device int32 tensor ``[max_batch]``.  State updates use in-place copies, so
    the entire forward can be captured and replayed with ``ttnn`` tracing.

Only :meth:`from_state_dict` imports torch.  No measured prefill/decode path
calls torch, ``ttnn.from_torch``, ``ttnn.as_tensor`` or ``ttnn.to_torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
    build_fused_const_tiles,
    chunk_gated_delta_rule_fused_adapter,
)

from .model_config import (
    HF_ADVERTISED_CONTEXT,
    LINEAR_ATTENTION,
    PAGE_BLOCK_SIZE,
    PREFILL_CHUNK,
    QWEN_SPARSE_ATTENTION,
    DecoderShapes,
    decoder_shapes,
)

PADDED_DECODE_HEADS = 32
QSA_TOKEN_BUDGET = 2048
QSA_BLOCK_TOPK = 512


def _shape(tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _free(tensor, *live) -> None:
    """Deallocate unless ``tensor`` aliases a live TTNN view."""

    if tensor is None or not tensor.is_allocated():
        return
    address = tensor.buffer_address()
    if any(other is not None and other.is_allocated() and other.buffer_address() == address for other in live):
        return
    ttnn.deallocate(tensor)


def _hifi4(*, fp32: bool = True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def _pad_seq(tensor, padded: int, *keep):
    length = int(tensor.shape[-2])
    if length == padded:
        return tensor
    out = ttnn.pad(tensor, [(0, 0), (0, 0), (0, padded - length), (0, 0)], 0.0)
    _free(tensor, out, *keep)
    return out


def _embedding_tiled_output(indices, weight):
    """Run embedding with TILE output while avoiding unsafe tiled-index input."""

    indices_rm = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.embedding(indices_rm, weight, layout=ttnn.TILE_LAYOUT)
    _free(indices_rm, indices, out)
    return out


def _slice_seq(tensor, start: int, logical: int, padded: int):
    piece = ttnn.slice(
        tensor,
        [0, 0, start, 0],
        [int(tensor.shape[0]), int(tensor.shape[1]), start + logical, int(tensor.shape[-1])],
    )
    return _pad_seq(piece, padded, tensor)


@dataclass(frozen=True)
class ExpertWeights:
    gate: object
    up: object
    down: object


class FunctionalDecoder(LightweightModule):
    """One exact-shape Qwen4Exp text decoder layer on a TTNN mesh device."""

    def __init__(
        self,
        *,
        shapes: DecoderShapes,
        mesh_device,
        weights: dict,
        experts: ExpertWeights,
        max_batch: int,
        max_seq_len: int,
        block_size: int,
        max_num_blocks: int,
        kv_cache=None,
        indexer_cache=None,
        recurrent_state=None,
        conv_state=None,
        ple_conv_state=None,
        user_recurrent_state=None,
        user_conv_state=None,
        user_ple_conv_state=None,
        constants=None,
    ):
        self.shapes = shapes
        self.mesh_device = mesh_device
        self.w = weights
        self.experts = experts
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.kv_cache = kv_cache
        self.indexer_cache = indexer_cache
        self.recurrent_state = recurrent_state
        self.conv_state = conv_state
        self.ple_conv_state = ple_conv_state
        self.user_recurrent_state = list(user_recurrent_state or [])
        self.user_conv_state = list(user_conv_state or [])
        self.user_ple_conv_state = list(user_ple_conv_state or [])
        # Request resets must be exact and allocation-free.  Multiplying a
        # live state by zero both leaks the replaced tensor and preserves any
        # NaN/Inf values.  Stable zero sources let every request copy clean
        # state in place before prefill, including after a traced decode.
        self.user_recurrent_zero = ttnn.zeros_like(self.user_recurrent_state[0]) if self.user_recurrent_state else None
        self.user_conv_zero = ttnn.zeros_like(self.user_conv_state[0]) if self.user_conv_state else None
        self.user_ple_conv_zero = ttnn.zeros_like(self.user_ple_conv_state[0]) if self.user_ple_conv_state else None
        self.const = constants or {}
        self.compute_cfg = _hifi4(fp32=True)
        self.sdpa_compute_cfg = _hifi4(fp32=True)
        self.matmul_output_dtype = ttnn.bfloat16
        self.cache_update_dtype = ttnn.bfloat16
        self.ple_activation_dtype = ttnn.bfloat16
        self.norm_weight_dtype = ttnn.bfloat16
        self.norm_compute_fidelity = "hifi4"
        self.router_output_dtype = ttnn.bfloat16
        self.decode_head_mem_cfg = None
        self.decode_index_mem_cfg = None
        if shapes.layer_type == QWEN_SPARSE_ATTENTION:
            self.decode_head_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(PADDED_DECODE_HEADS, shapes.head_dim),
                core_grid=ttnn.num_cores_to_corerangeset(max_batch, ttnn.CoreCoord(8, 8), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.decode_index_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(PADDED_DECODE_HEADS, shapes.indexer_head_dim),
                core_grid=ttnn.num_cores_to_corerangeset(max_batch, ttnn.CoreCoord(8, 8), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Optional[Mapping],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_batch: int = 1,
        max_seq_len: Optional[int] = None,
        block_size: int = PAGE_BLOCK_SIZE,
        max_num_blocks: Optional[int] = None,
        weight_dtype=ttnn.bfloat16,
        expert_weight_dtype=ttnn.bfloat16,
        cache_dtype=ttnn.bfloat16,
    ) -> "FunctionalDecoder":
        """Upload an HF submodule-relative state dict and allocate persistent state.

        Passing ``None`` creates exact-shaped zero tensors for structural/operator
        tests.  Real correctness tests pass checkpoint tensors.  Transposition,
        zero-centred norm adjustment and PLE sparse embedding lookup are setup work.
        """

        import torch  # setup-only by contract

        s = decoder_shapes(hf_config, layer_idx)
        max_seq_len = int(max_seq_len or s.max_position_embeddings)
        if not 1 <= max_seq_len <= HF_ADVERTISED_CONTEXT:
            raise ValueError(f"max_seq_len {max_seq_len} outside [1, {HF_ADVERTISED_CONTEXT}]")
        if block_size != PAGE_BLOCK_SIZE:
            raise ValueError(f"this stage validates the target page size {PAGE_BLOCK_SIZE}, got {block_size}")
        blocks_per_user = math.ceil(max_seq_len / block_size)
        minimum_blocks = max_batch * blocks_per_user
        max_num_blocks = int(max_num_blocks or minimum_blocks)
        if max_num_blocks < minimum_blocks:
            raise ValueError(f"max_num_blocks {max_num_blocks} < required {minimum_blocks}")

        is_mesh = hasattr(mesh_device, "shape")
        mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        def raw(name: str, shape: tuple[int, ...]):
            if state_dict is None:
                return None
            value = state_dict.get(name)
            if value is None:
                return None
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != target {shape}")
            return value

        def upload(value, *, shape, dtype=weight_dtype, transpose=False, add_one=False):
            if value is None:
                return ttnn.zeros(
                    shape,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            if transpose:
                value = value.transpose(-2, -1).contiguous()
            if add_one:
                value = value.float().add(1.0)
            return ttnn.as_tensor(
                value,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        w: dict[str, object] = {}
        for prefix in ("attn_hyper_connection", "mlp_hyper_connection"):
            tag = "attn_hc" if prefix.startswith("attn") else "mlp_hc"
            w[f"{tag}_norm"] = upload(
                raw(f"{prefix}.hc_norm.weight", (s.hc_hidden_size,)),
                shape=(1, 1, 1, s.hc_hidden_size),
                add_one=True,
            )
            w[f"{tag}_down"] = upload(
                raw(f"{prefix}.input_mix_weight_down.weight", (s.hc_lowrank, s.hc_hidden_size)),
                shape=(1, 1, s.hc_hidden_size, s.hc_lowrank),
                transpose=True,
            )
            w[f"{tag}_up"] = upload(
                raw(f"{prefix}.input_mix_weight_up.weight", (s.hc_hidden_size, s.hc_lowrank)),
                shape=(1, 1, s.hc_lowrank, s.hc_hidden_size),
                transpose=True,
            )
            w[f"{tag}_inject"] = upload(
                raw(f"{prefix}.block_inject_weight.weight", (s.hc_count, s.hc_hidden_size)),
                shape=(1, 1, s.hc_hidden_size, s.hc_count),
                transpose=True,
            )

        # Shared expert and router.
        w["router"] = upload(
            raw("mlp.gate.weight", (s.num_experts, s.hidden_size)),
            shape=(1, 1, s.hidden_size, s.num_experts),
            transpose=True,
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            out_size = s.shared_expert_intermediate_size if name != "down_proj" else s.hidden_size
            in_size = s.hidden_size if name != "down_proj" else s.shared_expert_intermediate_size
            w[f"shared_{name}"] = upload(
                raw(f"mlp.shared_expert.{name}.weight", (out_size, in_size)),
                shape=(1, 1, in_size, out_size),
                transpose=True,
            )
        w["shared_gate"] = upload(
            raw("mlp.shared_expert_gate.weight", (1, s.hidden_size)),
            shape=(1, 1, s.hidden_size, 1),
            transpose=True,
        )

        fused = raw(
            "mlp.experts.gate_up_proj",
            (s.num_experts, 2 * s.moe_intermediate_size, s.hidden_size),
        )
        down = raw(
            "mlp.experts.down_proj",
            (s.num_experts, s.hidden_size, s.moe_intermediate_size),
        )
        if fused is None:
            expert_gate = expert_up = expert_down = None
        else:
            expert_gate = fused[:, : s.moe_intermediate_size, :].transpose(-2, -1).unsqueeze(0).contiguous()
            expert_up = fused[:, s.moe_intermediate_size :, :].transpose(-2, -1).unsqueeze(0).contiguous()
            expert_down = down.transpose(-2, -1).unsqueeze(0).contiguous()
        experts = ExpertWeights(
            gate=upload(
                expert_gate,
                shape=(1, s.num_experts, s.hidden_size, s.moe_intermediate_size),
                dtype=expert_weight_dtype,
            ),
            up=upload(
                expert_up,
                shape=(1, s.num_experts, s.hidden_size, s.moe_intermediate_size),
                dtype=expert_weight_dtype,
            ),
            down=upload(
                expert_down,
                shape=(1, s.num_experts, s.moe_intermediate_size, s.hidden_size),
                dtype=expert_weight_dtype,
            ),
        )

        constants: dict[str, object] = {}
        kv_cache = indexer_cache = None
        recurrent_state = conv_state = ple_conv_state = None
        user_recurrent_state: list = []
        user_conv_state: list = []
        user_ple_conv_state: list = []

        if s.layer_type == LINEAR_ATTENTION:
            p = "linear_attn"
            for name, out_size in (
                ("in_proj_qkv", s.linear_qkv_width),
                ("in_proj_z", s.linear_value_width),
                ("in_proj_b", s.linear_num_value_heads),
                ("in_proj_a", s.linear_num_value_heads),
            ):
                w[name] = upload(
                    raw(f"{p}.{name}.weight", (out_size, s.hidden_size)),
                    shape=(1, 1, s.hidden_size, out_size),
                    transpose=True,
                )
            conv = raw(f"{p}.conv1d.weight", (s.linear_qkv_width, 1, s.linear_conv_kernel_dim))
            for tap in range(s.linear_conv_kernel_dim):
                value = None if conv is None else conv[:, 0, tap].reshape(1, 1, 1, -1)
                w[f"conv_tap_{tap}"] = upload(value, shape=(1, 1, 1, s.linear_qkv_width))
            w["dt_bias"] = upload(
                raw(f"{p}.dt_bias", (s.linear_num_value_heads,)),
                shape=(1, 1, 1, s.linear_num_value_heads),
                dtype=ttnn.float32,
            )
            a_log = raw(f"{p}.A_log", (s.linear_num_value_heads,))
            neg_exp_a = None if a_log is None else -torch.exp(a_log.float()).reshape(1, 1, 1, -1)
            w["neg_exp_A"] = upload(
                neg_exp_a,
                shape=(1, 1, 1, s.linear_num_value_heads),
                dtype=ttnn.float32,
            )
            w["gdn_norm"] = upload(
                raw(f"{p}.norm.weight", (s.linear_value_head_dim,)),
                shape=(1, 1, 1, s.linear_value_head_dim),
            )
            w["gdn_out"] = upload(
                raw(f"{p}.out_proj.weight", (s.hidden_size, s.linear_value_width)),
                shape=(1, 1, s.linear_value_width, s.hidden_size),
                transpose=True,
            )
            recurrent_state = ttnn.zeros(
                (max_batch, s.linear_num_value_heads, s.linear_key_head_dim, s.linear_value_head_dim),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            conv_state = ttnn.zeros(
                (max_batch, 1, s.linear_conv_kernel_dim - 1, s.linear_qkv_width),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            user_recurrent_state = [
                ttnn.zeros(
                    (1, s.linear_num_value_heads, s.linear_key_head_dim, s.linear_value_head_dim),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                )
                for _ in range(max_batch)
            ]
            user_conv_state = [
                ttnn.zeros(
                    (1, 1, s.linear_conv_kernel_dim - 1, s.linear_qkv_width),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                )
                for _ in range(max_batch)
            ]
            constants["gdn_tiles"] = build_fused_const_tiles(mesh_device)

        if s.layer_type == QWEN_SPARSE_ATTENTION:
            p = "self_attn"
            q_weight = raw(f"{p}.q_proj.weight", (2 * s.q_width, s.hidden_size))
            if q_weight is None:
                query_weight = gate_weight = None
            else:
                split = q_weight.reshape(s.num_attention_heads, 2, s.head_dim, s.hidden_size)
                query_weight = split[:, 0].reshape(s.q_width, s.hidden_size)
                gate_weight = split[:, 1].reshape(s.q_width, s.hidden_size)
            w["q_proj"] = upload(query_weight, shape=(1, 1, s.hidden_size, s.q_width), transpose=True)
            w["attn_gate"] = upload(gate_weight, shape=(1, 1, s.hidden_size, s.q_width), transpose=True)
            for name in ("k_proj", "v_proj"):
                w[name] = upload(
                    raw(f"{p}.{name}.weight", (s.kv_width, s.hidden_size)),
                    shape=(1, 1, s.hidden_size, s.kv_width),
                    transpose=True,
                )
            w["attn_out"] = upload(
                raw(f"{p}.o_proj.weight", (s.hidden_size, s.q_width)),
                shape=(1, 1, s.q_width, s.hidden_size),
                transpose=True,
            )
            for name in ("q_norm", "k_norm"):
                w[name] = upload(
                    raw(f"{p}.{name}.weight", (s.head_dim,)),
                    shape=(1, 1, 1, s.head_dim),
                    add_one=True,
                )
            iqk = raw(f"{p}.indexer.index_qk_proj.weight", (640, s.hidden_size))
            iq = None if iqk is None else iqk[: s.indexer_n_heads * s.indexer_head_dim]
            ik = None if iqk is None else iqk[s.indexer_n_heads * s.indexer_head_dim :]
            w["index_q"] = upload(
                iq,
                shape=(1, 1, s.hidden_size, s.indexer_n_heads * s.indexer_head_dim),
                transpose=True,
            )
            w["index_k"] = upload(ik, shape=(1, 1, s.hidden_size, s.indexer_head_dim), transpose=True)
            for name in ("q_layernorm", "k_layernorm"):
                short = "index_q_norm" if name.startswith("q") else "index_k_norm"
                w[short] = upload(
                    raw(f"{p}.indexer.{name}.weight", (s.indexer_head_dim,)),
                    shape=(1, 1, 1, s.indexer_head_dim),
                    add_one=True,
                )
            kv_cache = tuple(
                ttnn.zeros(
                    (max_num_blocks, s.num_key_value_heads, block_size, s.head_dim),
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            )
            indexer_cache = ttnn.zeros(
                (max_num_blocks, s.indexer_kv_heads, block_size, s.indexer_head_dim),
                dtype=cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            compressed_blocks = math.ceil(max_seq_len / s.indexer_compress_ratio)
            block_ids = torch.arange(compressed_blocks, dtype=torch.int32).reshape(1, 1, 1, -1)
            token_offsets = torch.arange(s.indexer_compress_ratio, dtype=torch.int32).reshape(1, 1, 1, 1, -1)
            selected_width = s.indexer_budget + s.indexer_compress_ratio
            gathered_width = _round_up(selected_width, 32)
            gathered_valid = torch.zeros(1, 1, 1, gathered_width, dtype=torch.bfloat16)
            gathered_valid[..., :selected_width] = 1
            for name, value, dtype, layout in (
                ("index_block_ids", block_ids, ttnn.int32, ttnn.TILE_LAYOUT),
                ("index_token_offsets", token_offsets, ttnn.int32, ttnn.TILE_LAYOUT),
                ("gathered_valid", gathered_valid, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ):
                constants[name] = ttnn.as_tensor(
                    value,
                    dtype=dtype,
                    layout=layout,
                    device=mesh_device,
                    mesh_mapper=mapper,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            constants["compressed_blocks"] = compressed_blocks
            constants["gathered_width"] = gathered_width

        if s.has_ple:
            p = "ple"
            w["ple_key"] = upload(
                raw(f"{p}.key_proj.weight", (s.hc_hidden_size, s.ple_embed_dim)),
                shape=(1, 1, s.ple_embed_dim, s.hc_hidden_size),
                transpose=True,
            )
            w["ple_value"] = upload(
                raw(f"{p}.value_proj.weight", (s.hidden_size, s.ple_embed_dim)),
                shape=(1, 1, s.ple_embed_dim, s.hidden_size),
                transpose=True,
            )
            for name in ("norm_key", "norm_query", "norm_conv"):
                w[f"ple_{name}"] = upload(
                    raw(f"{p}.{name}.weight", (s.hc_hidden_size,)),
                    shape=(1, 1, 1, s.hc_hidden_size),
                    add_one=True,
                )
            conv = raw(f"{p}.conv1d.weight", (s.hc_hidden_size, 1, s.ple_conv_kernel_size))
            for tap in range(s.ple_conv_kernel_size):
                value = None if conv is None else conv[:, 0, tap].reshape(1, 1, 1, -1)
                w[f"ple_conv_tap_{tap}"] = upload(value, shape=(1, 1, 1, s.hc_hidden_size))
            ple_conv_state = ttnn.zeros(
                (max_batch, 1, s.ple_conv_state_len, s.hc_hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            user_ple_conv_state = [
                ttnn.zeros(
                    (1, 1, s.ple_conv_state_len, s.hc_hidden_size),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                )
                for _ in range(max_batch)
            ]

        return cls(
            shapes=s,
            mesh_device=mesh_device,
            weights=w,
            experts=experts,
            max_batch=max_batch,
            max_seq_len=max_seq_len,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            kv_cache=kv_cache,
            indexer_cache=indexer_cache,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            ple_conv_state=ple_conv_state,
            user_recurrent_state=user_recurrent_state,
            user_conv_state=user_conv_state,
            user_ple_conv_state=user_ple_conv_state,
            constants=constants,
        )

    # ------------------------------------------------------------- primitives

    @staticmethod
    def _rms_norm(x, weight, epsilon: float, *, group_count: int = 1):
        """Zero-centred HF RMSNorm; ``weight`` already stores ``1 + parameter``."""

        if group_count == 1:
            return ttnn.rms_norm(x, epsilon=epsilon, weight=weight, compute_kernel_config=_hifi4(fp32=True))

        width = int(x.shape[-1])
        group_width = width // group_count
        rows = math.prod(_shape(x)[:-1])
        grouped = ttnn.reshape(x, (1, 1, rows * group_count, group_width))
        normed = ttnn.rms_norm(grouped, epsilon=epsilon, compute_kernel_config=_hifi4(fp32=True))
        group_weight = ttnn.reshape(weight, (1, 1, group_count, group_width))
        repeated_weight = ttnn.repeat(group_weight, (1, 1, rows, 1))
        out = ttnn.multiply(normed, repeated_weight)
        _free(grouped, x, normed)
        ttnn.deallocate(normed)
        _free(group_weight, weight, repeated_weight)
        ttnn.deallocate(repeated_weight)
        return ttnn.reshape(out, _shape(x))

    @staticmethod
    def _l2norm(x, epsilon: float = 1e-6):
        squared = ttnn.multiply(x, x)
        total = ttnn.sum(squared, dim=-1, keepdim=True)
        ttnn.deallocate(squared)
        inv = ttnn.rsqrt(ttnn.add(total, epsilon))
        ttnn.deallocate(total)
        out = ttnn.multiply(x, inv)
        ttnn.deallocate(inv)
        return out

    def _linear(self, x, weight, *, dtype=None):
        return ttnn.linear(
            x,
            weight,
            dtype=self.matmul_output_dtype if dtype is None else dtype,
            compute_kernel_config=self.compute_cfg,
        )

    def _hyper_mix(self, hyper_input, prefix: str):
        """HF ``Qwen4ExpTextGatedResidual`` without its final injection."""

        s = self.shapes
        normed = self._rms_norm(
            hyper_input,
            self.w[f"{prefix}_norm"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        low = self._linear(normed, self.w[f"{prefix}_down"])
        low = ttnn.silu(ttnn.multiply(low, 1.0 / s.hc_count))
        mix = ttnn.sigmoid(self._linear(low, self.w[f"{prefix}_up"]))
        ttnn.deallocate(low)

        rows = math.prod(_shape(normed)[:-1])
        norm_groups = ttnn.reshape(normed, (rows, s.hc_count, s.hidden_size))
        mix_groups = ttnn.reshape(mix, (rows, s.hc_count, s.hidden_size))
        mixed = ttnn.multiply(norm_groups, mix_groups)
        _free(norm_groups, normed, mixed)
        _free(mix_groups, mix, mixed)
        ttnn.deallocate(mix)
        mixed = ttnn.sum(mixed, dim=1, keepdim=True)
        mixed = ttnn.multiply(mixed, 1.0 / s.hc_count)
        mixed = ttnn.reshape(mixed, (*_shape(hyper_input)[:-1], s.hidden_size))

        injection = self._linear(normed, self.w[f"{prefix}_inject"])
        ttnn.deallocate(normed)
        injection = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(injection, 1.0 / s.hc_count)), 2.0)
        return mixed, hyper_input, injection

    def _hyper_inject(self, hyper_input, block_output, injection):
        s = self.shapes
        rows = math.prod(_shape(block_output)[:-1])
        value = ttnn.reshape(block_output, (rows, 1, s.hidden_size))
        gate = ttnn.reshape(injection, (rows, s.hc_count, 1))
        projected = ttnn.multiply(value, gate)
        _free(value, block_output, projected)
        _free(gate, injection, projected)
        ttnn.deallocate(block_output)
        ttnn.deallocate(injection)
        projected = ttnn.reshape(projected, _shape(hyper_input))
        out = ttnn.add(hyper_input, projected)
        ttnn.deallocate(projected)
        return out

    # --------------------------------------------------------- mixture of experts

    @staticmethod
    def _sparse_matmul_config(m: int, n: int, k: int):
        n_tiles = math.ceil(n / 32)
        best_cores, best_x, best_y = 1, 1, 1
        for cores in range(1, min(65, n_tiles + 1)):
            if n_tiles % cores:
                continue
            for y in range(1, 9):
                if cores % y == 0 and cores // y <= 8 and cores > best_cores:
                    best_cores, best_x, best_y = cores, cores // y, y
                    break
        k_tiles = math.ceil(k / 32)
        in0_block_w = next((candidate for candidate in (8, 5, 4, 2, 1) if k_tiles % candidate == 0), 1)
        per_core_n = n_tiles // best_cores
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(best_x, best_y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _routing(self, x):
        """Full softmax, top-10, sum normalisation and dense scatter on device."""

        s = self.shapes
        logits = self._linear(x, self.w["router"])
        probabilities = ttnn.softmax(logits, dim=-1)
        ttnn.deallocate(logits)
        values, indices = ttnn.topk(probabilities, k=s.num_experts_per_tok, dim=-1, sorted=True)
        denominator = ttnn.sum(values, dim=-1, keepdim=True)
        values = ttnn.div(values, denominator)
        ttnn.deallocate(denominator)
        routing = ttnn.scatter(ttnn.zeros_like(probabilities), dim=-1, index=indices, src=values)
        ttnn.deallocate(probabilities)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        return routing

    def _routed_experts(self, x, routing):
        """Tile-union sparse expert evaluation for prefill and single-token decode.

        Every token keeps its exact top-10 routing weights.  Sparse matmul operates
        at tile granularity, so each 32-token tile evaluates the union of experts
        selected by its rows; inactive token/expert pairs are removed after down.
        """

        s = self.shapes
        tokens = int(x.shape[-2])
        if tokens % 32:
            raise ValueError(f"expert input must be tile padded, got {tokens}")
        groups = tokens // 32
        grouped_x = ttnn.reshape(x, (1, groups, 32, s.hidden_size))
        routing_groups = ttnn.reshape(routing, (1, groups, 32, s.num_experts))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, groups, s.num_experts))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])
        gate_cfg = self._sparse_matmul_config(32, s.moe_intermediate_size, s.hidden_size)
        down_cfg = self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size)

        def project(weight):
            value = ttnn.sparse_matmul(
                grouped_x,
                weight,
                sparsity=sparsity,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=gate_cfg,
                compute_kernel_config=self.compute_cfg,
                dtype=ttnn.bfloat16,
            )
            value = ttnn.transpose(value, 1, 3)
            return ttnn.reshape(value, (1, s.num_experts, tokens, s.moe_intermediate_size))

        gate = project(self.experts.gate)
        up = project(self.experts.up)
        _free(grouped_x, x, gate, up)
        activated = ttnn.silu(gate)
        ttnn.deallocate(gate)
        hidden = ttnn.multiply(activated, up)
        ttnn.deallocate(activated)
        ttnn.deallocate(up)
        hidden = ttnn.reshape(hidden, (1, s.num_experts, tokens, s.moe_intermediate_size))
        down_pieces = []
        for group in range(groups):
            hidden_piece = ttnn.slice(
                hidden,
                [0, 0, group * 32, 0],
                [1, s.num_experts, (group + 1) * 32, s.moe_intermediate_size],
            )
            sparsity_piece = ttnn.slice(
                sparsity,
                [0, 0, group, 0],
                [1, 1, group + 1, s.num_experts],
            )
            down_piece = ttnn.sparse_matmul(
                hidden_piece,
                self.experts.down,
                sparsity=sparsity_piece,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_a_sparse=True,
                program_config=down_cfg,
                compute_kernel_config=self.compute_cfg,
                dtype=ttnn.bfloat16,
            )
            _free(hidden_piece, hidden, down_piece)
            _free(sparsity_piece, sparsity, down_piece)
            down_pieces.append(ttnn.reshape(down_piece, (1, s.num_experts, 32, s.hidden_size)))
        ttnn.deallocate(hidden)
        ttnn.deallocate(sparsity)
        if len(down_pieces) == 1:
            down = down_pieces[0]
        else:
            down = ttnn.concat(down_pieces, dim=2)
            for piece in down_pieces:
                ttnn.deallocate(piece)
        token_weights = ttnn.permute(routing, (0, 3, 2, 1))
        down = ttnn.multiply(down, token_weights)
        _free(token_weights, routing, down)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))

    def _shared_expert(self, x):
        gate = ttnn.silu(self._linear(x, self.w["shared_gate_proj"]))
        up = self._linear(x, self.w["shared_up_proj"])
        hidden = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self._linear(hidden, self.w["shared_down_proj"])
        ttnn.deallocate(hidden)
        scalar = ttnn.sigmoid(self._linear(x, self.w["shared_gate"]))
        result = ttnn.multiply(out, scalar)
        ttnn.deallocate(out)
        ttnn.deallocate(scalar)
        return result

    def _moe(self, x):
        logical = int(x.shape[-2])
        padded = _round_up(logical, 32)
        work = _pad_seq(x, padded, x) if padded != logical else x
        routing = self._routing(work)
        routed = self._routed_experts(work, routing)
        shared = self._shared_expert(work)
        ttnn.deallocate(routing)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        if padded != logical:
            trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, 1, logical, self.shapes.hidden_size])
            _free(out, trimmed)
            out = trimmed
        return out

    # --------------------------------------------------- projected lexical embed

    def _ple_gate(self, hyper_input, embeddings):
        s = self.shapes
        key = self._linear(embeddings, self.w["ple_key"])
        key = self._rms_norm(key, self.w["ple_norm_key"], s.rms_norm_eps, group_count=s.hc_count)
        query = self._rms_norm(
            hyper_input,
            self.w["ple_norm_query"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        rows = math.prod(_shape(hyper_input)[:-1])
        key = ttnn.reshape(key, (rows, s.hc_count, s.hidden_size))
        query = ttnn.reshape(query, (rows, s.hc_count, s.hidden_size))
        product = ttnn.multiply(key, query)
        ttnn.deallocate(key)
        ttnn.deallocate(query)
        gate = ttnn.sum(product, dim=-1, keepdim=True)
        ttnn.deallocate(product)
        gate = ttnn.multiply(gate, 1.0 / math.sqrt(s.hidden_size))
        sign = ttnn.sign(gate)
        magnitude = ttnn.sqrt(ttnn.clamp(ttnn.abs(gate), min=1e-6, max=None))
        ttnn.deallocate(gate)
        gate = ttnn.sigmoid(ttnn.multiply(sign, magnitude))
        ttnn.deallocate(sign)
        ttnn.deallocate(magnitude)
        value = self._linear(embeddings, self.w["ple_value"])
        value = ttnn.reshape(value, (rows, 1, s.hidden_size))
        gated = ttnn.multiply(gate, value)
        ttnn.deallocate(gate)
        ttnn.deallocate(value)
        return ttnn.reshape(gated, _shape(hyper_input))

    def _ple_prefill(self, hyper_input, embeddings, *, user_id: int, logical: int):
        s = self.shapes
        gated = self._ple_gate(hyper_input, embeddings)
        normed = self._rms_norm(
            gated,
            self.w["ple_norm_conv"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        state = self.user_ple_conv_state[user_id]
        window = ttnn.concat([state, normed], dim=-2)
        length = int(normed.shape[-2])
        acc = None
        for tap in range(s.ple_conv_kernel_size):
            start = tap * s.ple_conv_dilation
            piece = ttnn.slice(window, [0, 0, start, 0], [1, 1, start + length, s.hc_hidden_size])
            term = ttnn.multiply(piece, self.w[f"ple_conv_tap_{tap}"])
            _free(piece, window, term)
            if acc is None:
                acc = term
            else:
                updated = ttnn.add(acc, term)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = updated
        new_state = ttnn.slice(
            window,
            [0, 0, logical, 0],
            [1, 1, logical + s.ple_conv_state_len, s.hc_hidden_size],
        )
        _free(window, normed, new_state)
        ttnn.deallocate(normed)
        ttnn.deallocate(self.user_ple_conv_state[user_id])
        self.user_ple_conv_state[user_id] = ttnn.clone(new_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(new_state, self.user_ple_conv_state[user_id])
        convolved = ttnn.silu(acc)
        ttnn.deallocate(acc)
        out = ttnn.add(gated, convolved)
        ttnn.deallocate(gated)
        ttnn.deallocate(convolved)
        return out

    def _ple_decode(self, hyper_input, embeddings):
        s = self.shapes
        gated_public = self._ple_gate(hyper_input, embeddings)
        normed_public = self._rms_norm(
            gated_public,
            self.w["ple_norm_conv"],
            s.rms_norm_eps,
            group_count=s.hc_count,
        )
        gated = ttnn.reshape(gated_public, (self.max_batch, 1, 1, s.hc_hidden_size))
        normed = ttnn.reshape(normed_public, (self.max_batch, 1, 1, s.hc_hidden_size))
        window = ttnn.concat([self.ple_conv_state, normed], dim=2)
        acc = None
        for tap in range(s.ple_conv_kernel_size):
            index = tap * s.ple_conv_dilation
            piece = ttnn.slice(
                window,
                [0, 0, index, 0],
                [self.max_batch, 1, index + 1, s.hc_hidden_size],
            )
            term = ttnn.multiply(piece, self.w[f"ple_conv_tap_{tap}"])
            _free(piece, window, term)
            if acc is None:
                acc = term
            else:
                updated = ttnn.add(acc, term)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = updated
        new_state = ttnn.slice(
            window,
            [0, 0, 1, 0],
            [self.max_batch, 1, s.ple_conv_state_len + 1, s.hc_hidden_size],
        )
        ttnn.copy(new_state, self.ple_conv_state)
        _free(new_state, window, self.ple_conv_state)
        _free(window, normed, self.ple_conv_state)
        _free(gated, gated_public)
        _free(normed, normed_public)
        ttnn.deallocate(normed_public)
        convolved = ttnn.silu(acc)
        ttnn.deallocate(acc)
        out = ttnn.add(gated_public, ttnn.reshape(convolved, _shape(gated_public)))
        ttnn.deallocate(gated_public)
        ttnn.deallocate(convolved)
        return out

    # ----------------------------------------------------- gated DeltaNet mixer

    def _gdn_inputs(self, x):
        mixed = self._linear(x, self.w["in_proj_qkv"], dtype=ttnn.float32)
        z = self._linear(x, self.w["in_proj_z"])
        b = self._linear(x, self.w["in_proj_b"], dtype=ttnn.float32)
        a = self._linear(x, self.w["in_proj_a"], dtype=ttnn.float32)
        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        biased = ttnn.add(a, self.w["dt_bias"])
        ttnn.deallocate(a)
        soft = ttnn.softplus(biased, beta=1.0, threshold=20.0)
        ttnn.deallocate(biased)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed, z, beta, g

    def _causal_conv_prefill(self, mixed, *, user_id: int, logical: int):
        s = self.shapes
        state = self.user_conv_state[user_id]
        window = ttnn.concat([state, mixed], dim=-2)
        length = int(mixed.shape[-2])
        acc = None
        for tap in range(s.linear_conv_kernel_dim):
            piece = ttnn.slice(window, [0, 0, tap, 0], [1, 1, tap + length, s.linear_qkv_width])
            term = ttnn.multiply(piece, self.w[f"conv_tap_{tap}"])
            _free(piece, window, term)
            if acc is None:
                acc = term
            else:
                updated = ttnn.add(acc, term)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = updated
        new_state = ttnn.slice(
            window,
            [0, 0, logical, 0],
            [1, 1, logical + s.linear_conv_kernel_dim - 1, s.linear_qkv_width],
        )
        _free(window, mixed, new_state)
        ttnn.deallocate(self.user_conv_state[user_id])
        self.user_conv_state[user_id] = ttnn.clone(new_state, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _free(new_state, self.user_conv_state[user_id])
        out = ttnn.silu(acc)
        ttnn.deallocate(acc)
        return out

    def _split_gdn(self, conv_out):
        s = self.shapes
        lead = _shape(conv_out)[:-1]
        starts = [0] * len(lead)
        q = ttnn.slice(conv_out, [*starts, 0], [*lead, s.linear_qk_width])
        k = ttnn.slice(conv_out, [*starts, s.linear_qk_width], [*lead, 2 * s.linear_qk_width])
        v = ttnn.slice(conv_out, [*starts, 2 * s.linear_qk_width], [*lead, s.linear_qkv_width])
        return q, k, v

    def _gdn_epilogue(self, core, z, *, public_shape):
        s = self.shapes
        core = self._rms_norm(core, self.w["gdn_norm"], s.rms_norm_eps)
        z = ttnn.reshape(z, _shape(core))
        gated = ttnn.multiply(core, ttnn.sigmoid(z))
        ttnn.deallocate(core)
        ttnn.deallocate(z)
        flat = ttnn.reshape(gated, (*public_shape[:-1], s.linear_value_width))
        _free(gated, flat)
        out = self._linear(flat, self.w["gdn_out"])
        ttnn.deallocate(flat)
        return out

    def _gdn_prefill(self, x, *, user_id: int, logical: int):
        s = self.shapes
        padded = int(x.shape[-2])
        mixed, z, beta, g = self._gdn_inputs(x)
        conv = self._causal_conv_prefill(mixed, user_id=user_id, logical=logical)
        ttnn.deallocate(mixed)
        q, k, v = self._split_gdn(conv)
        ttnn.deallocate(conv)

        # Padded positions must be identity recurrence updates (beta=g=0).
        if logical != padded:
            positions = ttnn.arange(0, padded, 1, device=self.mesh_device, dtype=ttnn.int32)
            valid = ttnn.lt(positions, logical)
            valid = ttnn.reshape(valid, (1, 1, padded, 1))
            beta = ttnn.multiply(beta, valid)
            g = ttnn.multiply(g, valid)
            ttnn.deallocate(positions)
            ttnn.deallocate(valid)

        q = ttnn.reshape(q, (1, padded, s.linear_qk_width))
        k = ttnn.reshape(k, (1, padded, s.linear_qk_width))
        v = ttnn.reshape(v, (1, padded, s.linear_value_width))
        beta = ttnn.reshape(beta, (1, padded, s.linear_num_value_heads))
        g = ttnn.reshape(g, (1, padded, s.linear_num_value_heads))
        core, state = chunk_gated_delta_rule_fused_adapter(
            q,
            k,
            v,
            beta,
            g,
            initial_state=self.user_recurrent_state[user_id],
            device=self.mesh_device,
            qkv_head_dims=(
                s.linear_num_key_heads,
                s.linear_key_head_dim,
                s.linear_num_value_heads,
                s.linear_value_head_dim,
            ),
            const_tiles=self.const["gdn_tiles"],
        )
        ttnn.deallocate(self.user_recurrent_state[user_id])
        self.user_recurrent_state[user_id] = state
        core = ttnn.reshape(
            core,
            (1, padded, s.linear_num_value_heads, s.linear_value_head_dim),
        )
        return self._gdn_epilogue(core, z, public_shape=_shape(x))

    def _gdn_decode(self, x):
        s = self.shapes
        batch = self.max_batch
        mixed_public, z, beta, g = self._gdn_inputs(x)
        mixed = ttnn.reshape(mixed_public, (batch, 1, 1, s.linear_qkv_width))
        window = ttnn.concat([self.conv_state, mixed], dim=2)
        _free(mixed, mixed_public, window)
        ttnn.deallocate(mixed_public)
        acc = None
        for tap in range(s.linear_conv_kernel_dim):
            piece = ttnn.slice(window, [0, 0, tap, 0], [batch, 1, tap + 1, s.linear_qkv_width])
            term = ttnn.multiply(piece, self.w[f"conv_tap_{tap}"])
            _free(piece, window, term)
            if acc is None:
                acc = term
            else:
                updated = ttnn.add(acc, term)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = updated
        new_conv = ttnn.slice(
            window,
            [0, 0, 1, 0],
            [batch, 1, s.linear_conv_kernel_dim, s.linear_qkv_width],
        )
        ttnn.copy(new_conv, self.conv_state)
        _free(new_conv, window, self.conv_state)
        _free(window, self.conv_state)
        conv = ttnn.silu(acc)
        ttnn.deallocate(acc)
        q, k, v = self._split_gdn(conv)
        ttnn.deallocate(conv)
        q = ttnn.reshape(q, (batch, 1, s.linear_num_key_heads, s.linear_key_head_dim))
        k = ttnn.reshape(k, (batch, 1, s.linear_num_key_heads, s.linear_key_head_dim))
        v = ttnn.reshape(v, (batch, 1, s.linear_num_value_heads, s.linear_value_head_dim))
        repeat = s.linear_num_value_heads // s.linear_num_key_heads
        q = ttnn.repeat_interleave(q, repeat, dim=2)
        k = ttnn.repeat_interleave(k, repeat, dim=2)
        q = self._l2norm(q)
        k = self._l2norm(k)
        q = ttnn.multiply(q, 1.0 / math.sqrt(s.linear_key_head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        beta = ttnn.reshape(beta, (batch, s.linear_num_value_heads, 1, 1))
        g = ttnn.reshape(g, (batch, s.linear_num_value_heads, 1, 1))
        decay = ttnn.exp(g)
        ttnn.deallocate(g)
        decayed = ttnn.multiply(self.recurrent_state, decay)
        ttnn.deallocate(decay)
        memory = ttnn.matmul(k, decayed, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        delta = ttnn.multiply(ttnn.subtract(v, memory), beta)
        ttnn.deallocate(memory)
        ttnn.deallocate(v)
        ttnn.deallocate(beta)
        update = ttnn.matmul(
            ttnn.transpose(k, -2, -1), delta, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg
        )
        ttnn.deallocate(k)
        ttnn.deallocate(delta)
        state = ttnn.add(decayed, update)
        ttnn.deallocate(decayed)
        ttnn.deallocate(update)
        core = ttnn.matmul(q, state, dtype=ttnn.float32, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(q)
        ttnn.copy(state, self.recurrent_state)
        ttnn.deallocate(state)
        core = ttnn.permute(core, (0, 2, 1, 3))
        return self._gdn_epilogue(core, z, public_shape=_shape(x))

    # ---------------------------------------------------- Qwen sparse attention

    @staticmethod
    def _apply_rope(x, cos, sin, rotary_dim: int):
        head_dim = int(x.shape[-1])
        lead = _shape(x)[:-1]
        starts = [0] * len(lead)
        rotary = ttnn.slice(x, [*starts, 0], [*lead, rotary_dim])
        first = ttnn.slice(rotary, [*starts, 0], [*lead, rotary_dim // 2])
        second = ttnn.slice(rotary, [*starts, rotary_dim // 2], [*lead, rotary_dim])
        rotated = ttnn.concat([ttnn.neg(second), first], dim=-1)
        ttnn.deallocate(first)
        ttnn.deallocate(second)
        embedded = ttnn.add(ttnn.multiply(rotary, cos), ttnn.multiply(rotated, sin))
        ttnn.deallocate(rotary)
        ttnn.deallocate(rotated)
        if rotary_dim == head_dim:
            return embedded
        rest = ttnn.slice(x, [*starts, rotary_dim], [*lead, head_dim])
        out = ttnn.concat([embedded, rest], dim=-1)
        ttnn.deallocate(embedded)
        ttnn.deallocate(rest)
        return out

    def _rotation_rows(self, rot_mats, positions, *, batch: int, tokens: int):
        """Gather partial-RoPE rows from caller-owned full context tables."""

        s = self.shapes
        cos_table, sin_table = rot_mats
        cos_weight = ttnn.reshape(cos_table, (self.max_seq_len, s.rotary_dim))
        sin_weight = ttnn.reshape(sin_table, (self.max_seq_len, s.rotary_dim))
        indices = ttnn.typecast(positions, ttnn.uint32)
        indices = ttnn.reshape(indices, (1, batch * tokens))
        cos = _embedding_tiled_output(indices, cos_weight)
        sin = _embedding_tiled_output(indices, sin_weight)
        _free(cos_weight, cos_table, cos)
        _free(sin_weight, sin_table, sin)
        _free(indices, positions, cos, sin)
        return (
            ttnn.reshape(cos, (batch, 1, tokens, s.rotary_dim)),
            ttnn.reshape(sin, (batch, 1, tokens, s.rotary_dim)),
        )

    def _qsa_projections(self, x, positions, rot_mats, *, decode: bool):
        s = self.shapes
        batch = self.max_batch if decode else 1
        tokens = 1 if decode else int(x.shape[-2])
        q = self._linear(x, self.w["q_proj"])
        gate = self._linear(x, self.w["attn_gate"])
        k = self._linear(x, self.w["k_proj"])
        v = self._linear(x, self.w["v_proj"])
        index_q = self._linear(x, self.w["index_q"])
        raw_index_k = self._linear(x, self.w["index_k"])

        if decode:
            q = ttnn.reshape(q, (batch, 1, s.num_attention_heads, s.head_dim))
            q = ttnn.permute(q, (0, 2, 1, 3))
            k = ttnn.reshape(k, (batch, 1, s.num_key_value_heads, s.head_dim))
            k = ttnn.permute(k, (0, 2, 1, 3))
            v = ttnn.reshape(v, (batch, 1, s.num_key_value_heads, s.head_dim))
            v = ttnn.permute(v, (0, 2, 1, 3))
            index_q = ttnn.reshape(index_q, (batch, 1, s.indexer_n_heads, s.indexer_head_dim))
            index_q = ttnn.permute(index_q, (0, 2, 1, 3))
            raw_index_k = ttnn.reshape(raw_index_k, (batch, 1, 1, s.indexer_head_dim))
        else:
            q = ttnn.reshape(q, (1, tokens, s.num_attention_heads, s.head_dim))
            q = ttnn.permute(q, (0, 2, 1, 3))
            k = ttnn.reshape(k, (1, tokens, s.num_key_value_heads, s.head_dim))
            k = ttnn.permute(k, (0, 2, 1, 3))
            v = ttnn.reshape(v, (1, tokens, s.num_key_value_heads, s.head_dim))
            v = ttnn.permute(v, (0, 2, 1, 3))
            index_q = ttnn.reshape(index_q, (1, tokens, s.indexer_n_heads, s.indexer_head_dim))
            index_q = ttnn.permute(index_q, (0, 2, 1, 3))
            raw_index_k = ttnn.reshape(raw_index_k, (1, tokens, 1, s.indexer_head_dim))

        q = self._rms_norm(q, self.w["q_norm"], s.rms_norm_eps)
        k = self._rms_norm(k, self.w["k_norm"], s.rms_norm_eps)
        index_q = self._rms_norm(index_q, self.w["index_q_norm"], s.rms_norm_eps)
        cos, sin = self._rotation_rows(rot_mats, positions, batch=batch, tokens=tokens)
        q = self._apply_rope(q, cos, sin, s.rotary_dim)
        k = self._apply_rope(k, cos, sin, s.rotary_dim)
        index_q = self._apply_rope(index_q, cos, sin, s.rotary_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return q, k, v, gate, index_q, raw_index_k

    def _physical_compressed_ids(self, page_table, *, batch: int):
        """Map every virtual four-token index block to the paged raw-key cache."""

        s = self.shapes
        count = self.const["compressed_blocks"]
        per_page = self.block_size // s.indexer_compress_ratio
        ids = self.const["index_block_ids"]
        # These are non-negative int32 indices and every divisor in the QSA
        # paging contract is a power of two.  TTNN ``floor_div`` currently
        # lowers integer/scalar division through a floating reciprocal that is
        # zero for these operands; shifts/masks preserve the exact index math.
        virtual_pages = ttnn.bitwise_right_shift(ids, int(math.log2(per_page)))
        in_page = ttnn.bitwise_and(ids, per_page - 1)
        pages = ttnn.reshape(page_table, (batch, 1, 1, int(page_table.shape[-1])))
        # TILE indices pad their query axis to 32; gather validates padded
        # non-gather dimensions, so expose a matching broadcast source view.
        pages = ttnn.repeat(pages, (1, 1, 32, 1))
        pages = ttnn.to_layout(pages, ttnn.TILE_LAYOUT)
        if batch > 1:
            virtual_pages = ttnn.repeat(virtual_pages, (batch, 1, 1, 1))
            in_page = ttnn.repeat(in_page, (batch, 1, 1, 1))
        index = ttnn.typecast(virtual_pages, ttnn.uint32)
        physical_pages = ttnn.gather(pages, dim=-1, index=index)
        _free(pages, page_table, physical_pages)
        ttnn.deallocate(virtual_pages)
        ttnn.deallocate(index)
        physical = ttnn.add(ttnn.multiply(physical_pages, per_page), in_page)
        ttnn.deallocate(physical_pages)
        ttnn.deallocate(in_page)
        return ttnn.reshape(physical, (batch, count))

    def _selected_virtual_tokens(self, index_q, page_table, current_pos, rot_mats):
        """Run the QSA compressed-block indexer entirely on device."""

        s = self.shapes
        batch, _, tokens, _ = _shape(index_q)
        count = self.const["compressed_blocks"]
        per_page = self.block_size // s.indexer_compress_ratio

        raw = ttnn.reshape(
            self.indexer_cache,
            (self.max_num_blocks, per_page, s.indexer_compress_ratio, s.indexer_head_dim),
        )
        pooled = ttnn.mean(raw, dim=2, keepdim=False)
        _free(raw, self.indexer_cache, pooled)
        pooled_weight = ttnn.reshape(pooled, (self.max_num_blocks * per_page, s.indexer_head_dim))
        physical_ids = self._physical_compressed_ids(page_table, batch=batch)
        physical_ids = ttnn.typecast(physical_ids, ttnn.uint32)
        physical_ids = ttnn.reshape(physical_ids, (1, batch * count))
        keys = _embedding_tiled_output(physical_ids, pooled_weight)
        _free(pooled_weight, pooled, keys)
        ttnn.deallocate(pooled)
        ttnn.deallocate(physical_ids)
        keys = ttnn.reshape(keys, (batch, 1, count, s.indexer_head_dim))
        keys = self._rms_norm(keys, self.w["index_k_norm"], s.rms_norm_eps)

        block_positions = ttnn.multiply(self.const["index_block_ids"], s.indexer_compress_ratio)
        block_positions = ttnn.reshape(block_positions, (1, count))
        if batch > 1:
            block_positions = ttnn.repeat(block_positions, (batch, 1))
        block_cos, block_sin = self._rotation_rows(
            rot_mats,
            block_positions,
            batch=batch,
            tokens=count,
        )
        keys = self._apply_rope(keys, block_cos, block_sin, s.rotary_dim)
        ttnn.deallocate(block_cos)
        ttnn.deallocate(block_sin)
        repeated_keys = ttnn.repeat(keys, (1, s.indexer_n_heads, 1, 1))
        ttnn.deallocate(keys)
        keys = repeated_keys
        scores = ttnn.matmul(index_q, ttnn.transpose(keys, -2, -1), compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(keys)
        scores = ttnn.relu(scores)
        scores = ttnn.sum(scores, dim=1, keepdim=True)
        scores = ttnn.multiply(scores, 1.0 / math.sqrt(s.indexer_head_dim))

        pos = ttnn.reshape(current_pos, (batch, 1, tokens, 1))
        complete = ttnn.bitwise_right_shift(
            ttnn.add(pos, 1),
            int(math.log2(s.indexer_compress_ratio)),
        )
        block_ids = self.const["index_block_ids"]
        if batch > 1 or tokens > 1:
            block_ids = ttnn.repeat(block_ids, (batch, 1, tokens, 1))
        visible = ttnn.lt(block_ids, complete)
        scores = ttnn.where(visible, scores, -1.0e4)
        ttnn.deallocate(visible)
        selected_scores, selected_blocks = ttnn.topk(scores, k=QSA_BLOCK_TOPK, dim=-1, sorted=True)
        ttnn.deallocate(selected_scores)
        ttnn.deallocate(scores)

        # Blackhole top-k returns uint16 indices for this width; reshape-view
        # supports uint32/int32, so widen before constructing token ids.
        narrow_selected_blocks = selected_blocks
        selected_blocks = ttnn.typecast(narrow_selected_blocks, ttnn.int32)
        ttnn.deallocate(narrow_selected_blocks)
        # HF requests min(512, num_complete_blocks).  TTNN top-k has a fixed
        # traced shape, so masked filler blocks still occupy unused lanes.  Keep
        # those lanes physically present but invalidate their expanded tokens;
        # otherwise the filler equal to ``complete`` can duplicate the
        # separately appended incomplete tail.
        selected_blocks = ttnn.reshape(selected_blocks, (batch, 1, tokens, QSA_BLOCK_TOPK, 1))
        offsets = self.const["index_token_offsets"]
        selected = ttnn.add(ttnn.multiply(selected_blocks, s.indexer_compress_ratio), offsets)
        ttnn.deallocate(selected_blocks)
        selected = ttnn.reshape(selected, (batch, 1, tokens, s.indexer_budget))
        complete_tokens = ttnn.multiply(complete, s.indexer_compress_ratio)
        selected_valid = ttnn.lt(selected, complete_tokens, dtype=ttnn.bfloat16)
        tail_base = ttnn.reshape(complete_tokens, (batch, 1, tokens, 1, 1))
        tail = ttnn.add(tail_base, offsets)
        _free(tail_base, complete_tokens, tail)
        ttnn.deallocate(complete_tokens)
        ttnn.deallocate(complete)
        tail = ttnn.reshape(tail, (batch, 1, tokens, s.indexer_compress_ratio))
        tail_valid = ttnn.le(tail, pos, dtype=ttnn.bfloat16)
        complete_selected = selected
        selected = ttnn.concat([selected, tail], dim=-1)
        _free(complete_selected, selected)
        valid = ttnn.concat([selected_valid, tail_valid], dim=-1)
        ttnn.deallocate(selected_valid)
        ttnn.deallocate(tail_valid)
        ttnn.deallocate(tail)
        if int(selected.shape[-1]) != self.const["gathered_width"]:
            unpadded_selected = selected
            selected = ttnn.pad(
                unpadded_selected,
                [(0, 0), (0, 0), (0, 0), (0, self.const["gathered_width"] - int(selected.shape[-1]))],
                0,
            )
            _free(unpadded_selected, selected)
            unpadded_valid = valid
            valid = ttnn.pad(
                unpadded_valid,
                [(0, 0), (0, 0), (0, 0), (0, self.const["gathered_width"] - int(valid.shape[-1]))],
                0,
            )
            _free(unpadded_valid, valid)
        # At the final advertised position the four-lane tail begins one past
        # the cache.  Those lanes are invalid, but their gather addresses must
        # still be in range before the additive attention mask is applied.
        unclamped_selected = selected
        selected = ttnn.minimum(unclamped_selected, self.max_seq_len - 1)
        _free(unclamped_selected, selected)
        unscaled_valid = valid
        valid = ttnn.multiply(unscaled_valid, self.const["gathered_valid"])
        _free(unscaled_valid, valid)
        return selected, valid

    def _virtual_to_physical_tokens(self, virtual_tokens, page_table):
        batch, _, tokens, width = _shape(virtual_tokens)
        virtual_page = ttnn.bitwise_right_shift(virtual_tokens, int(math.log2(self.block_size)))
        in_page = ttnn.bitwise_and(virtual_tokens, self.block_size - 1)
        pages = ttnn.reshape(page_table, (batch, 1, 1, int(page_table.shape[-1])))
        pages = ttnn.repeat(pages, (1, 1, _round_up(tokens, 32), 1))
        pages = ttnn.to_layout(pages, ttnn.TILE_LAYOUT)
        index = ttnn.typecast(virtual_page, ttnn.uint32)
        physical_page = ttnn.gather(pages, dim=-1, index=index)
        _free(pages, page_table, physical_page)
        ttnn.deallocate(virtual_page)
        ttnn.deallocate(index)
        physical = ttnn.add(ttnn.multiply(physical_page, self.block_size), in_page)
        ttnn.deallocate(physical_page)
        ttnn.deallocate(in_page)
        return ttnn.typecast(ttnn.reshape(physical, (batch, tokens, width)), ttnn.uint32)

    def _gathered_qsa_attention(self, q, selected, valid, page_table):
        s = self.shapes
        batch, _, tokens, _ = _shape(q)
        width = self.const["gathered_width"]
        physical = self._virtual_to_physical_tokens(selected, page_table)
        ttnn.deallocate(selected)
        # Embedding's tiled 2-D index path reorders rows when the leading
        # dimension is greater than one.  A single flattened row is exact and
        # reshapes back without a host-side gather or conversion.
        physical = ttnn.reshape(physical, (1, batch * tokens * width))

        gathered = []
        for cache in self.kv_cache:
            flattened = ttnn.permute(cache, (0, 2, 1, 3))
            flattened = ttnn.reshape(flattened, (self.max_num_blocks * self.block_size, s.kv_width))
            value = _embedding_tiled_output(physical, flattened)
            _free(flattened, cache, value)
            expected_volume = batch * tokens * width * s.kv_width
            if math.prod(_shape(value)) != expected_volume:
                raise RuntimeError(
                    f"paged cache embedding returned {_shape(value)} (volume {math.prod(_shape(value))}); "
                    f"expected logical volume {expected_volume} for [{batch}, {tokens}, {width}, {s.kv_width}]"
                )
            value = ttnn.reshape(value, (batch, tokens, width, s.num_key_value_heads, s.head_dim))
            value = ttnn.permute(value, (0, 3, 1, 2, 4))
            value = ttnn.repeat_interleave(
                value,
                s.num_attention_heads // s.num_key_value_heads,
                dim=1,
            )
            gathered.append(value)
        ttnn.deallocate(physical)
        k, v = gathered
        q_rows = ttnn.reshape(q, (batch * s.num_attention_heads * tokens, 1, s.head_dim))
        k_rows = ttnn.reshape(k, (batch * s.num_attention_heads * tokens, width, s.head_dim))
        v_rows = ttnn.reshape(v, (batch * s.num_attention_heads * tokens, width, s.head_dim))
        scores = ttnn.matmul(q_rows, ttnn.transpose(k_rows, -2, -1), compute_kernel_config=self.compute_cfg)
        _free(q_rows, q, scores)
        ttnn.deallocate(k_rows)
        scores = ttnn.multiply(scores, 1.0 / math.sqrt(s.head_dim))
        additive = ttnn.where(valid, 0.0, -1.0e4)
        ttnn.deallocate(valid)
        additive = ttnn.repeat(additive, (1, s.num_attention_heads, 1, 1))
        additive = ttnn.reshape(additive, (batch * s.num_attention_heads * tokens, 1, width))
        scores = ttnn.add(scores, additive)
        ttnn.deallocate(additive)
        probabilities = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(scores)
        out = ttnn.matmul(probabilities, v_rows, compute_kernel_config=self.compute_cfg)
        ttnn.deallocate(probabilities)
        ttnn.deallocate(v_rows)
        out = ttnn.reshape(out, (batch, s.num_attention_heads, tokens, s.head_dim))
        return out

    def _qsa_epilogue(self, attention, gate, *, decode: bool):
        s = self.shapes
        if decode:
            attention = ttnn.permute(attention, (0, 2, 1, 3))
            attention = ttnn.reshape(attention, (1, 1, self.max_batch, s.q_width))
        else:
            attention = ttnn.permute(attention, (0, 2, 1, 3))
            attention = ttnn.reshape(attention, (1, 1, int(attention.shape[1]), s.q_width))
        gated = ttnn.multiply(attention, ttnn.sigmoid(gate))
        ttnn.deallocate(attention)
        ttnn.deallocate(gate)
        out = self._linear(gated, self.w["attn_out"])
        ttnn.deallocate(gated)
        return out

    def _qsa_prefill(
        self,
        x,
        *,
        page_table,
        chunk_page_table,
        chunk_start: int,
        rot_mats,
    ):
        length = int(x.shape[-2])
        positions = ttnn.arange(chunk_start, chunk_start + length, 1, device=self.mesh_device, dtype=ttnn.int32)
        positions = ttnn.minimum(positions, self.max_seq_len - 1)
        positions = ttnn.reshape(positions, (1, length))
        q, k, v, gate, index_q, raw_index = self._qsa_projections(x, positions, rot_mats, decode=False)
        raw_heads = ttnn.permute(raw_index, (0, 2, 1, 3))
        cache_tokens = int(chunk_page_table.shape[-1]) * self.block_size
        if not 0 < cache_tokens <= length:
            raise ValueError(f"QSA cache fill length {cache_tokens} is outside padded chunk [1, {length}]")
        for cache, value in (
            (self.kv_cache[0], k),
            (self.kv_cache[1], v),
            (self.indexer_cache, raw_heads),
        ):
            cache_value = value
            if cache_tokens != length:
                cache_value = ttnn.slice(
                    value,
                    [0, 0, 0, 0],
                    [int(value.shape[0]), int(value.shape[1]), cache_tokens, int(value.shape[-1])],
                )
            fill = ttnn.typecast(cache_value, cache.dtype)
            ttnn.experimental.paged_fill_cache(cache, fill, chunk_page_table, batch_idx=0)
            _free(fill, cache_value)
            _free(cache_value, value)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(raw_index)
        _free(raw_heads, raw_index)
        selected, valid = self._selected_virtual_tokens(index_q, page_table, positions, rot_mats)
        ttnn.deallocate(index_q)
        attention = self._gathered_qsa_attention(q, selected, valid, page_table)
        ttnn.deallocate(q)
        ttnn.deallocate(positions)
        return self._qsa_epilogue(attention, gate, decode=False)

    def _qsa_decode(self, x, *, current_pos, page_table, rot_mats):
        s = self.shapes
        positions = ttnn.reshape(current_pos, (self.max_batch, 1))
        q, k, v, gate, index_q, raw_index = self._qsa_projections(x, positions, rot_mats, decode=True)

        def padded_heads(value, dim, mem_cfg):
            heads = int(value.shape[2])
            if heads < PADDED_DECODE_HEADS:
                value = ttnn.pad(value, [(0, 0), (0, 0), (0, PADDED_DECODE_HEADS - heads), (0, 0)], 0.0)
            return ttnn.to_memory_config(value, mem_cfg)

        # paged_update_cache consumes [1, batch, padded_heads, dim].
        k_update = ttnn.permute(k, (2, 0, 1, 3))
        v_update = ttnn.permute(v, (2, 0, 1, 3))
        raw_update = ttnn.permute(raw_index, (2, 0, 1, 3))
        if k_update.dtype != self.cache_update_dtype:
            k_update = ttnn.typecast(k_update, self.cache_update_dtype)
        if v_update.dtype != self.cache_update_dtype:
            v_update = ttnn.typecast(v_update, self.cache_update_dtype)
        if raw_update.dtype != self.cache_update_dtype:
            raw_update = ttnn.typecast(raw_update, self.cache_update_dtype)
        k_update = padded_heads(k_update, s.head_dim, self.decode_head_mem_cfg)
        v_update = padded_heads(v_update, s.head_dim, self.decode_head_mem_cfg)
        raw_update = padded_heads(raw_update, s.indexer_head_dim, self.decode_index_mem_cfg)
        ttnn.experimental.paged_update_cache(
            self.kv_cache[0],
            k_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.kv_cache[1],
            v_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.indexer_cache,
            raw_update,
            update_idxs_tensor=current_pos,
            page_table=page_table,
        )
        ttnn.deallocate(k_update)
        ttnn.deallocate(v_update)
        ttnn.deallocate(raw_update)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        ttnn.deallocate(raw_index)

        selected, valid = self._selected_virtual_tokens(index_q, page_table, positions, rot_mats)
        ttnn.deallocate(index_q)
        attention = self._gathered_qsa_attention(q, selected, valid, page_table)
        ttnn.deallocate(q)
        return self._qsa_epilogue(attention, gate, decode=True)

    # ------------------------------------------------------------- public API

    def _reset_user_state(self, user_id: int) -> None:
        s = self.shapes
        if s.layer_type == LINEAR_ATTENTION:
            ttnn.copy(self.user_recurrent_zero, self.user_recurrent_state[user_id])
            ttnn.copy(self.user_conv_zero, self.user_conv_state[user_id])
        if s.has_ple:
            ttnn.copy(self.user_ple_conv_zero, self.user_ple_conv_state[user_id])

    def prefill_forward(
        self,
        hidden_states,
        *,
        user_id: int = 0,
        page_table=None,
        page_tables_per_chunk: Optional[Sequence] = None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Run one user's complete logical prompt under the module contract."""

        s = self.shapes
        if len(hidden_states.shape) != 4 or _shape(hidden_states)[:2] != [1, 1]:
            raise ValueError(f"prefill expects [1, 1, seq_len, {s.hc_hidden_size}], got {hidden_states.shape}")
        seq_len = int(hidden_states.shape[-2])
        if int(hidden_states.shape[-1]) != s.hc_hidden_size:
            raise ValueError(f"prefill hidden width {int(hidden_states.shape[-1])} != {s.hc_hidden_size}")
        if not 1 <= seq_len <= self.max_seq_len:
            raise ValueError(f"prefill seq_len {seq_len} outside [1, {self.max_seq_len}]")
        if not 0 <= user_id < self.max_batch:
            raise ValueError(f"user_id {user_id} outside [0, {self.max_batch})")
        if s.has_ple:
            if ple_embeddings is None or _shape(ple_embeddings) != [1, 1, seq_len, s.ple_embed_dim]:
                raise ValueError(f"PLE layer needs embeddings [1, 1, {seq_len}, {s.ple_embed_dim}]")
        elif ple_embeddings is not None:
            raise ValueError("ple_embeddings were passed to a layer without PLE")
        if s.layer_type == QWEN_SPARSE_ATTENTION:
            if page_table is None or page_tables_per_chunk is None or rot_mats is None:
                raise ValueError("QSA prefill requires page_table, page_tables_per_chunk and full RoPE tables")
            if len(page_tables_per_chunk) != len(self.prefill_chunk_plan(seq_len)):
                raise ValueError("page_tables_per_chunk does not match prefill_chunk_plan")

        self._reset_user_state(user_id)
        pieces = []
        for chunk_index, (start, logical, padded) in enumerate(self.prefill_chunk_plan(seq_len)):
            x = _slice_seq(hidden_states, start, logical, padded)
            if s.has_ple:
                embedding_chunk = _slice_seq(ple_embeddings, start, logical, padded)
                ple = self._ple_prefill(x, embedding_chunk, user_id=user_id, logical=logical)
                _free(embedding_chunk, ple_embeddings)
                updated = ttnn.add(x, ple)
                _free(x, hidden_states, updated)
                ttnn.deallocate(ple)
                x = updated

            mixed, hyper, injection = self._hyper_mix(x, "attn_hc")
            if s.layer_type == LINEAR_ATTENTION:
                block = self._gdn_prefill(mixed, user_id=user_id, logical=logical)
            else:
                block = self._qsa_prefill(
                    mixed,
                    page_table=page_table,
                    chunk_page_table=page_tables_per_chunk[chunk_index],
                    chunk_start=start,
                    rot_mats=rot_mats,
                )
            ttnn.deallocate(mixed)
            hidden = self._hyper_inject(hyper, block, injection)
            mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
            block = self._moe(mixed)
            ttnn.deallocate(mixed)
            out = self._hyper_inject(hyper, block, injection)
            if logical != padded:
                trimmed = ttnn.slice(out, [0, 0, 0, 0], [1, 1, logical, s.hc_hidden_size])
                _free(out, trimmed)
                out = trimmed
            pieces.append(out)

        if len(pieces) == 1:
            return pieces[0]
        output = ttnn.concat(pieces, dim=-2)
        for piece in pieces:
            ttnn.deallocate(piece)
        return output

    def decode_forward(
        self,
        hidden_states,
        *,
        current_pos,
        page_table=None,
        rot_mats=None,
        ple_embeddings=None,
    ):
        """Run one fixed-batch token; all state writes preserve trace addresses."""

        s = self.shapes
        expected = [1, 1, self.max_batch, s.hc_hidden_size]
        if _shape(hidden_states) != expected:
            raise ValueError(f"decode expects {expected}, got {_shape(hidden_states)}")
        if current_pos is None or _shape(current_pos) != [self.max_batch]:
            raise ValueError(f"decode current_pos must be device int32 [{self.max_batch}]")
        if s.has_ple:
            expected_ple = [1, 1, self.max_batch, s.ple_embed_dim]
            if ple_embeddings is None or _shape(ple_embeddings) != expected_ple:
                raise ValueError(f"PLE decode embeddings must be {expected_ple}")
            ple = self._ple_decode(hidden_states, ple_embeddings)
            hidden_states = ttnn.add(hidden_states, ple)
            ttnn.deallocate(ple)
        elif ple_embeddings is not None:
            raise ValueError("ple_embeddings were passed to a layer without PLE")

        mixed, hyper, injection = self._hyper_mix(hidden_states, "attn_hc")
        if s.layer_type == LINEAR_ATTENTION:
            block = self._gdn_decode(mixed)
        else:
            if page_table is None or rot_mats is None:
                raise ValueError("QSA decode requires page_table and full RoPE tables")
            block = self._qsa_decode(
                mixed,
                current_pos=current_pos,
                page_table=page_table,
                rot_mats=rot_mats,
            )
        ttnn.deallocate(mixed)
        hidden = self._hyper_inject(hyper, block, injection)
        mixed, hyper, injection = self._hyper_mix(hidden, "mlp_hc")
        block = self._moe(mixed)
        ttnn.deallocate(mixed)
        return self._hyper_inject(hyper, block, injection)

    def prepare_decode_state(self) -> None:
        """Copy per-user prefill recurrence into stable traced-decode buffers."""

        s = self.shapes
        if s.layer_type == LINEAR_ATTENTION:
            recurrent = ttnn.concat(self.user_recurrent_state, dim=0)
            ttnn.copy(recurrent, self.recurrent_state)
            _free(recurrent, *self.user_recurrent_state)
            conv = ttnn.concat(self.user_conv_state, dim=0)
            ttnn.copy(conv, self.conv_state)
            _free(conv, *self.user_conv_state)
        if s.has_ple:
            ple = ttnn.concat(self.user_ple_conv_state, dim=0)
            ttnn.copy(ple, self.ple_conv_state)
            _free(ple, *self.user_ple_conv_state)

    def prefill_chunk_plan(self, seq_len: int) -> list[tuple[int, int, int]]:
        """Return ``(start, logical, padded)`` chunks used by paged prefill."""

        if not 1 <= seq_len <= self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} outside [1, {self.max_seq_len}]")
        plan = []
        for start in range(0, seq_len, PREFILL_CHUNK):
            logical = min(PREFILL_CHUNK, seq_len - start)
            # 128 is simultaneously a DeltaNet chunk, two 64-token pages and
            # four TT tiles; every short final chunk is masked and trimmed.
            plan.append((start, logical, PREFILL_CHUNK))
        return plan
