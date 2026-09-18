# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Graph-fused TTNN decoder layer for ``google/gemma-4-26B-A4B-it``.

This stage preserves :class:`FunctionalDecoder`'s public tensor, paged-cache,
logical-length, and trace contracts.  It changes only numerically equivalent
subgraphs inside a decoder pass:

* dense gate/up projections share one packed matmul and fuse fast GELU into
  the following binary multiply;
* expert gate/up projections share one packed sparse matmul (the accurate
  Gemma GeGLU is fused into the following binary multiply);
* the router's feature scale and constant scale are folded into its FP32
  projection weights at setup time;
* the expert routing scale is folded into the down weights, and the three
  equal pre-FFN normalizations share one unweighted RMSNorm; and
* zero-padding the packed expert output to 1,536 columns exposes a 48-core
  sparse-matmul geometry while retaining the public 704-wide expert result;
* the otherwise-dead MoE branch sum/RMSNorm fusion remains available for A/B
  evidence, but is disabled because it regresses traced decode latency.

All host-side construction remains in ``from_state_dict`` through the
functional parent.  The forward methods below operate only on TTNN tensors.
Environment switches are retained solely for focused on-device A/B evidence;
the defaults are the selected fused graph.
"""

from __future__ import annotations

import os
from typing import Any

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import (
    HIDDEN_SIZE,
    MLP_INTERMEDIATE_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TILE_SIZE,
    TOP_K_EXPERTS,
    FunctionalDecoder,
    _build_sparse_matmul_config,
)
from models.demos.gemma4.tt.experts.operations import apply_geglu


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def resolved_fusion_policy() -> dict[str, bool | int]:
    """Return the effective graph policy used by a newly constructed layer."""

    return {
        "packed_dense_gate_up": _env_flag("GEMMA4_FUSED_PACKED_DENSE_GATE_UP", True),
        "dense_geglu": _env_flag("GEMMA4_FUSED_DENSE_GEGLU", False),
        "binary_dense_geglu": _env_flag("GEMMA4_FUSED_BINARY_DENSE_GEGLU", True),
        "folded_router_projection": _env_flag("GEMMA4_FUSED_FOLDED_ROUTER", True),
        "fused_branch_norm": _env_flag("GEMMA4_FUSED_BRANCH_NORM", False),
        "packed_expert_gate_up": _env_flag("GEMMA4_FUSED_PACKED_EXPERT_GATE_UP", True),
        "expert_geglu": _env_flag("GEMMA4_FUSED_EXPERT_GEGLU", False),
        "binary_expert_geglu": _env_flag("GEMMA4_FUSED_BINARY_EXPERT_GEGLU", True),
        "folded_expert_scale": _env_flag("GEMMA4_FUSED_FOLDED_EXPERT_SCALE", True),
        "shared_ffn_norm": _env_flag("GEMMA4_FUSED_SHARED_FFN_NORM", True),
        "fused_final_scalar": _env_flag("GEMMA4_FUSED_FINAL_SCALAR", True),
        "routed_reduction_matmul": _env_flag("GEMMA4_FUSED_ROUTED_REDUCTION", False),
        "expert_padded_width": int(os.getenv("GEMMA4_FUSED_EXPERT_PADDED_WIDTH", "1536")),
        "expert_in0_block_w": int(os.getenv("GEMMA4_FUSED_EXPERT_IN0_BLOCK_W", "4")),
    }


class FusedDecoder(FunctionalDecoder):
    """Functional Gemma-4 decoder with correctness-proven graph fusions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        policy = resolved_fusion_policy()
        self.use_packed_dense_gate_up = policy["packed_dense_gate_up"]
        self.use_dense_geglu = policy["dense_geglu"]
        self.use_binary_dense_geglu = policy["binary_dense_geglu"]
        self.use_folded_router_projection = policy["folded_router_projection"]
        self.use_fused_branch_norm = policy["fused_branch_norm"]
        self.use_packed_expert_gate_up = policy["packed_expert_gate_up"]
        self.use_expert_geglu = policy["expert_geglu"]
        self.use_binary_expert_geglu = policy["binary_expert_geglu"]
        self.use_folded_expert_scale = policy["folded_expert_scale"]
        self.use_shared_ffn_norm = policy["shared_ffn_norm"]
        self.use_fused_final_scalar = policy["fused_final_scalar"]
        self.use_routed_reduction_matmul = policy["routed_reduction_matmul"]
        self.packed_expert_width = policy["expert_padded_width"]
        if self.packed_expert_width not in {2 * MOE_INTERMEDIATE_SIZE, 1536, 1792, 2048}:
            raise ValueError("GEMMA4_FUSED_EXPERT_PADDED_WIDTH must be one of 1408, 1536, 1792, or 2048")
        self.expert_packed_half_width = self.packed_expert_width // 2
        self.expert_in0_block_w = policy["expert_in0_block_w"]
        if self.expert_in0_block_w not in {1, 2, 4, 8, 11}:
            raise ValueError("GEMMA4_FUSED_EXPERT_IN0_BLOCK_W must divide 88 and be one of 1, 2, 4, 8, or 11")
        if self.use_shared_ffn_norm and not (
            self.use_packed_dense_gate_up and self.use_packed_expert_gate_up and self.use_folded_router_projection
        ):
            raise ValueError("shared FFN normalization requires all three packed/folded consumers")
        if self.use_folded_expert_scale and not self.use_folded_router_projection:
            raise ValueError("folding the expert scale requires the folded router path")

        self.fusion_path_counts = {
            "packed_dense_gate_up": 0,
            "dense_geglu": 0,
            "binary_dense_geglu": 0,
            "folded_router_projection": 0,
            "fused_branch_norm": 0,
            "packed_expert_gate_up_prefill": 0,
            "packed_expert_gate_up_decode": 0,
            "expert_geglu": 0,
            "binary_expert_geglu": 0,
            "folded_expert_scale": 0,
            "shared_ffn_norm": 0,
            "fused_final_scalar": 0,
            "routed_reduction_matmul": 0,
        }

        # ttnn.geglu computes first_half * gelu(second_half), hence [up, gate].
        self.packed_dense_gate_up = None
        if self.use_packed_dense_gate_up:
            dense_up = self.weights.mlp_up
            dense_gate = self.weights.mlp_gate
            scaled_dense_inputs = []
            if self.use_shared_ffn_norm:
                norm_weight = ttnn.reshape(self.weights.pre_ff_ln, [1, 1, HIDDEN_SIZE, 1])
                dense_up = ttnn.mul(dense_up, norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                dense_gate = ttnn.mul(dense_gate, norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                scaled_dense_inputs = [dense_up, dense_gate]
            self.packed_dense_gate_up = ttnn.concat(
                [dense_up, dense_gate],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for scaled_input in scaled_dense_inputs:
                scaled_input.deallocate(True)
            self.weights.mlp_up.deallocate(True)
            self.weights.mlp_gate.deallocate(True)
            if self.use_shared_ffn_norm:
                self.weights.pre_ff_ln.deallocate(True)

        # Expert packing uses the same [up, gate] order.  The selected path
        # splits it before accurate GeGLU; the dedicated fast-GeGLU candidate
        # can consume the same packed output during an A/B run.
        self.packed_expert_gate_up = None
        if self.use_packed_expert_gate_up:
            expert_up = self.weights.expert_up
            expert_gate = self.weights.expert_gate
            temporary_expert_inputs = []
            if self.use_shared_ffn_norm:
                norm_weight = ttnn.reshape(self.weights.pre_ff_ln_2, [1, 1, HIDDEN_SIZE, 1])
                expert_up = ttnn.mul(expert_up, norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                expert_gate = ttnn.mul(expert_gate, norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                temporary_expert_inputs.extend([expert_up, expert_gate])
            if self.expert_packed_half_width != MOE_INTERMEDIATE_SIZE:
                padding = self.expert_packed_half_width - MOE_INTERMEDIATE_SIZE
                pad_spec = [(0, 0), (0, 0), (0, 0), (0, padding)]
                padded_up = ttnn.pad(expert_up, pad_spec, 0.0)
                padded_gate = ttnn.pad(expert_gate, pad_spec, 0.0)
                temporary_expert_inputs.extend([padded_up, padded_gate])
                expert_up = padded_up
                expert_gate = padded_gate
            self.packed_expert_gate_up = ttnn.concat(
                [expert_up, expert_gate],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for temporary_input in temporary_expert_inputs:
                temporary_input.deallocate(True)
            self.weights.expert_up.deallocate(True)
            self.weights.expert_gate.deallocate(True)
            if self.use_shared_ffn_norm:
                self.weights.pre_ff_ln_2.deallocate(True)

        self.folded_expert_down = None
        if self.use_folded_expert_scale:
            expert_scale = ttnn.reshape(self.weights.router_per_expert_scale, [1, NUM_EXPERTS, 1, 1])
            self.folded_expert_down = ttnn.mul(
                self.weights.expert_down,
                expert_scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weights.expert_down.deallocate(True)
            self.weights.router_per_expert_scale.deallocate(True)

        # (x * feature_scale * hidden_scale) @ W
        #   == x @ ((feature_scale * hidden_scale)[:, None] * W)
        self.folded_router_projection = None
        if self.use_folded_router_projection:
            feature_scale = ttnn.reshape(self.weights.router_scale, [1, 1, HIDDEN_SIZE, 1])
            feature_scaled_projection = ttnn.mul(
                self.weights.router_proj,
                feature_scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.folded_router_projection = ttnn.mul(
                feature_scaled_projection,
                self.router_hidden_scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            feature_scaled_projection.deallocate(True)
            self.weights.router_proj.deallocate(True)
            self.weights.router_scale.deallocate(True)

        arch = self.mesh_device.arch() if hasattr(self.mesh_device, "arch") else ttnn.device.GetDefaultDevice().arch()
        self.prefill_expert_compute_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Any], *, layer_idx: int, **kwargs: Any) -> "FusedDecoder":
        decoder = super().from_state_dict(state_dict, layer_idx=layer_idx, **kwargs)
        if decoder.use_fused_final_scalar:
            suffix = f"layers.{layer_idx}.layer_scalar"
            scalar = next(value for name, value in state_dict.items() if name.endswith(suffix))
            decoder.layer_scalar_value = float(scalar.item())
            decoder.weights.layer_scalar.deallocate(True)
        return decoder

    def _final_residual(self, residual: ttnn.Tensor, branch: ttnn.Tensor) -> ttnn.Tensor:
        if not self.use_fused_final_scalar:
            return self._apply_layer_scalar(ttnn.add(residual, branch, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        self.fusion_path_counts["fused_final_scalar"] += 1
        return ttnn.add(
            residual,
            branch,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar_value)],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _branch_sum_rms_norm(
        self,
        hidden_1: ttnn.Tensor,
        hidden_2: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if not self.use_fused_branch_norm:
            merged = ttnn.add(hidden_1, hidden_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return self._rms_norm(merged, self.weights.post_ff_ln)
        self.fusion_path_counts["fused_branch_norm"] += 1
        return ttnn.rms_norm(
            hidden_1,
            epsilon=self.eps,
            weight=self.weights.post_ff_ln,
            residual_input_tensor=hidden_2,
            compute_kernel_config=self.correctness_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_forward_single_user(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int,
        chunk_page_table: ttnn.Tensor | None,
        cache_position_modulo: int | None,
    ) -> ttnn.Tensor:
        logical_seq_len = hidden_states.shape[-2]
        if logical_seq_len < 1:
            raise ValueError("prefill requires at least one logical token")
        padded_seq_len = ((logical_seq_len + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        if padded_seq_len != logical_seq_len:
            pad = [(0, 0), (0, 0), (0, padded_seq_len - logical_seq_len), (0, 0)]
            hidden_states = ttnn.pad(hidden_states, pad, 0.0)
            position_cos = ttnn.pad(position_cos, pad, 0.0)
            position_sin = ttnn.pad(position_sin, pad, 0.0)

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_prefill(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            cache_position_modulo=cache_position_modulo,
            logical_seq_len=logical_seq_len,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        shared_ffn_in = self._shared_ffn_input(residual)
        mlp_in = shared_ffn_in if self.use_shared_ffn_norm else self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        hidden_1 = self._rms_norm(self._dense_mlp(mlp_in), self.weights.post_ff_ln_1)
        router_weights = self._router_weights(shared_ffn_in if self.use_shared_ffn_norm else residual)
        moe_in = shared_ffn_in if self.use_shared_ffn_norm else self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._rms_norm(self._moe_prefill(moe_in, router_weights), self.weights.post_ff_ln_2)

        hidden_states = self._final_residual(residual, self._branch_sum_rms_norm(hidden_1, hidden_2))
        if padded_seq_len != logical_seq_len:
            hidden_states = ttnn.slice(
                hidden_states,
                starts=[0, 0, 0, 0],
                ends=[1, 1, logical_seq_len, HIDDEN_SIZE],
                steps=[1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return hidden_states

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: tuple[ttnn.Tensor, ttnn.Tensor],
        cache_position_modulo: int | None = None,
    ) -> ttnn.Tensor:
        if hidden_states.shape[-2] < 1:
            raise ValueError("decode requires at least one batch row")

        residual = hidden_states
        attn_in = self._rms_norm(hidden_states, self.weights.input_ln)
        attn_out = self._attention_decode(
            attn_in,
            position_cos=position_cos,
            position_sin=position_sin,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            cache_position_modulo=cache_position_modulo,
        )
        attn_out = self._rms_norm(attn_out, self.weights.post_attn_ln)
        hidden_states = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        residual = hidden_states
        shared_ffn_in = self._shared_ffn_input(residual)
        mlp_in = shared_ffn_in if self.use_shared_ffn_norm else self._rms_norm(hidden_states, self.weights.pre_ff_ln)
        hidden_1 = self._rms_norm(self._dense_mlp(mlp_in), self.weights.post_ff_ln_1)
        router_weights = self._router_weights(shared_ffn_in if self.use_shared_ffn_norm else residual)
        moe_in = shared_ffn_in if self.use_shared_ffn_norm else self._rms_norm(residual, self.weights.pre_ff_ln_2)
        hidden_2 = self._rms_norm(self._moe_decode(moe_in, router_weights), self.weights.post_ff_ln_2)

        return self._final_residual(residual, self._branch_sum_rms_norm(hidden_1, hidden_2))

    def _dense_mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self.use_packed_dense_gate_up:
            return super()._dense_mlp(x)
        self.fusion_path_counts["packed_dense_gate_up"] += 1
        gate_up = ttnn.linear(
            x,
            self.packed_dense_gate_up,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.use_binary_dense_geglu:
            self.fusion_path_counts["binary_dense_geglu"] += 1
            width = MLP_INTERMEDIATE_SIZE
            gate_up_shape = list(gate_up.shape)
            up = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], width])
            gate = ttnn.slice(gate_up, [0, 0, 0, width], gate_up_shape)
            hidden = ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0)],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif self.use_dense_geglu:
            self.fusion_path_counts["dense_geglu"] += 1
            hidden = ttnn.geglu(gate_up, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # The composite GEGLU currently exposes its tile-padded M as the
            # logical M. Restore the caller's logical token/batch dimension by
            # metadata-only reshape; the physical tile remains unchanged.
            logical_shape = [*list(x.shape)[:-1], MLP_INTERMEDIATE_SIZE]
            padded_shape = [*logical_shape]
            padded_shape[-2] = max(TILE_SIZE, logical_shape[-2])
            hidden = ttnn.reshape(hidden, logical_shape, padded_shape)
        else:
            width = MLP_INTERMEDIATE_SIZE
            gate_up_shape = list(gate_up.shape)
            up = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], width])
            gate = ttnn.slice(gate_up, [0, 0, 0, width], list(gate_up.shape))
            gate = ttnn.gelu(gate, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hidden = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.linear(
            hidden,
            self.weights.mlp_down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _shared_ffn_input(self, residual: ttnn.Tensor) -> ttnn.Tensor | None:
        if not self.use_shared_ffn_norm:
            return None
        self.fusion_path_counts["shared_ffn_norm"] += 1
        return self._rms_norm(residual, None)

    def _router_weights(self, residual: ttnn.Tensor) -> ttnn.Tensor:
        if not self.use_folded_router_projection:
            return super()._router_weights(residual)
        self.fusion_path_counts["folded_router_projection"] += 1
        tokens = residual.shape[-2]
        router_in = residual if self.use_shared_ffn_norm else self._rms_norm(residual, None)
        router_in = ttnn.reshape(router_in, [tokens, HIDDEN_SIZE])
        router_in = ttnn.typecast(router_in, ttnn.float32)
        logits = ttnn.linear(
            router_in,
            self.folded_router_projection,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits = ttnn.typecast(logits, ttnn.bfloat16)
        top_values, top_indices = ttnn.topk(logits, k=TOP_K_EXPERTS, dim=-1, sorted=True)
        top_values = ttnn.softmax(
            top_values,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=-1, index=top_indices, src=top_values)
        if not self.use_folded_expert_scale:
            routing = ttnn.mul(routing, self.weights.router_per_expert_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            routing = ttnn.typecast(routing, ttnn.bfloat16)
        else:
            self.fusion_path_counts["folded_expert_scale"] += 1
        return ttnn.reshape(routing, [1, 1, tokens, NUM_EXPERTS])

    def _packed_expert_activation(self, gate_up: ttnn.Tensor) -> ttnn.Tensor:
        if self.use_expert_geglu:
            self.fusion_path_counts["expert_geglu"] += 1
            original_shape = list(gate_up.shape)
            if len(original_shape) == 3:
                gate_up = ttnn.reshape(gate_up, [1, *original_shape])
            result = ttnn.geglu(gate_up, dim=-1, memory_config=gate_up.memory_config())
            if self.expert_packed_half_width != MOE_INTERMEDIATE_SIZE:
                result = ttnn.slice(
                    result,
                    [0] * len(result.shape),
                    [*list(result.shape)[:-1], MOE_INTERMEDIATE_SIZE],
                    memory_config=result.memory_config(),
                )
            if len(original_shape) == 3:
                result = ttnn.reshape(result, [*original_shape[:-1], MOE_INTERMEDIATE_SIZE])
            return result
        up = ttnn.slice(
            gate_up,
            [0] * len(gate_up.shape),
            [*list(gate_up.shape)[:-1], MOE_INTERMEDIATE_SIZE],
            memory_config=gate_up.memory_config(),
        )
        gate = ttnn.slice(
            gate_up,
            [*([0] * (len(gate_up.shape) - 1)), self.expert_packed_half_width],
            [*list(gate_up.shape)[:-1], self.expert_packed_half_width + MOE_INTERMEDIATE_SIZE],
            memory_config=gate_up.memory_config(),
        )
        if self.use_binary_expert_geglu:
            self.fusion_path_counts["binary_expert_geglu"] += 1
            return ttnn.mul(
                gate,
                up,
                input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)],
                memory_config=gate_up.memory_config(),
            )
        return apply_geglu(gate, up)

    def _moe_decode_single_user(
        self,
        hidden_states: ttnn.Tensor,
        routing_weights: ttnn.Tensor,
        *,
        sparsity: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        if not self.use_packed_expert_gate_up:
            return super()._moe_decode_single_user(hidden_states, routing_weights, sparsity=sparsity)
        self.fusion_path_counts["packed_expert_gate_up_decode"] += 1
        batch = hidden_states.shape[2]
        if sparsity is None:
            sparsity = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(
            batch,
            self.packed_expert_width,
            in0_block_w=self.expert_in0_block_w,
        )
        down_config = _build_sparse_matmul_config(batch, HIDDEN_SIZE)

        gate_up = ttnn.sparse_matmul(
            hidden_states,
            self.packed_expert_gate_up,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            dtype=self.activation_dtype,
            compute_kernel_config=self.correctness_compute_config,
        )
        # sparse_matmul emits [1, 1, 1, E, batch, N] (expert-major); bring the
        # rows in front so every row owns a contiguous [E, N] block.
        gate_up = ttnn.reshape(gate_up, (1, NUM_EXPERTS, batch, self.packed_expert_width))
        gate_up = ttnn.transpose(gate_up, 1, 2)
        gate_up = ttnn.reshape(gate_up, (batch, NUM_EXPERTS, self.packed_expert_width))
        down_input = self._packed_expert_activation(gate_up)
        down_input = ttnn.transpose(down_input, 1, 0)
        down_input = ttnn.reshape(down_input, (1, NUM_EXPERTS, batch, MOE_INTERMEDIATE_SIZE))
        down = ttnn.sparse_matmul(
            down_input,
            self.folded_expert_down if self.use_folded_expert_scale else self.weights.expert_down,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.permute(down, (0, 2, 1, 3))
        next_states = ttnn.reshape(next_states, (batch, NUM_EXPERTS, HIDDEN_SIZE))
        if self.use_routed_reduction_matmul:
            self.fusion_path_counts["routed_reduction_matmul"] += 1
            routing_row = ttnn.reshape(routing_weights, (batch, 1, NUM_EXPERTS))
            next_states = ttnn.matmul(
                routing_row,
                next_states,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            next_states = ttnn.reshape(
                next_states,
                (1, 1, batch, HIDDEN_SIZE),
                (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
            )
            return next_states
        routing_3d = ttnn.reshape(routing_weights, (batch, NUM_EXPERTS, 1))
        next_states = ttnn.mul(next_states, routing_3d)
        next_states = ttnn.sum(next_states, dim=1)
        next_states = ttnn.unsqueeze_to_4D(next_states)
        return ttnn.reshape(
            next_states,
            (1, 1, batch, HIDDEN_SIZE),
            (1, 1, max(TILE_SIZE, batch), HIDDEN_SIZE),
        )

    def _moe_prefill_chunk(self, hidden_states: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        if not self.use_packed_expert_gate_up:
            return super()._moe_prefill_chunk(hidden_states, routing_weights)
        self.fusion_path_counts["packed_expert_gate_up_prefill"] += 1
        chunk_len = hidden_states.shape[2]
        # sparse_matmul's group dimension contributes to its M blocks. Match
        # the canonical expert path's 32-token bound before applying the packed
        # projection, otherwise a 1024-token prefill requests 256 down-matmul
        # blocks from an 8-core program.
        if chunk_len > TILE_SIZE:
            hidden_chunks = ttnn.split(hidden_states, TILE_SIZE, dim=2)
            routing_chunks = ttnn.split(routing_weights, TILE_SIZE, dim=2)
            results = []
            for hidden_chunk, routing_chunk in zip(hidden_chunks, routing_chunks):
                results.append(self._moe_prefill_chunk(hidden_chunk, routing_chunk))
            result = ttnn.concat(results, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for chunk_result in results:
                chunk_result.deallocate(True)
            return result
        groups = chunk_len // TILE_SIZE
        hidden_grouped = ttnn.reshape(hidden_states, (1, groups, TILE_SIZE, HIDDEN_SIZE))
        sparsity = ttnn.repeat(self.expert_prefill_sparsity, (1, 1, groups, 1))
        nnz = NUM_EXPERTS * groups
        output_tile = ttnn.Tile([TILE_SIZE, TILE_SIZE])
        gate_up_config = _build_sparse_matmul_config(
            TILE_SIZE,
            self.packed_expert_width,
            in0_block_w=self.expert_in0_block_w,
        )
        down_config = _build_sparse_matmul_config(TILE_SIZE, HIDDEN_SIZE)

        gate_up = ttnn.sparse_matmul(
            hidden_grouped,
            self.packed_expert_gate_up,
            sparsity=sparsity,
            nnz=nnz,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=gate_up_config,
            compute_kernel_config=self.prefill_expert_compute_config,
            dtype=self.activation_dtype,
        )
        gate_up = ttnn.transpose(gate_up, 1, 3)
        gate_up = ttnn.reshape(gate_up, (1, NUM_EXPERTS, chunk_len, self.packed_expert_width))
        down_input = self._packed_expert_activation(gate_up)
        down = ttnn.sparse_matmul(
            down_input,
            self.folded_expert_down if self.use_folded_expert_scale else self.weights.expert_down,
            sparsity=self.expert_prefill_sparsity,
            nnz=NUM_EXPERTS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            program_config=down_config,
            is_input_a_sparse=True,
            compute_kernel_config=self.prefill_expert_compute_config,
            dtype=self.activation_dtype,
        )
        next_states = ttnn.reshape(down, (1, NUM_EXPERTS, chunk_len, HIDDEN_SIZE))
        if self.use_routed_reduction_matmul:
            self.fusion_path_counts["routed_reduction_matmul"] += 1
            token_major = ttnn.permute(next_states, (0, 2, 1, 3))
            routing_row = ttnn.reshape(routing_weights, (1, chunk_len, 1, NUM_EXPERTS))
            result = ttnn.matmul(
                routing_row,
                token_major,
                dtype=self.activation_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return ttnn.reshape(result, (1, 1, chunk_len, HIDDEN_SIZE))
        routing_permuted = ttnn.permute(routing_weights, (0, 3, 2, 1))
        next_states = ttnn.mul(next_states, routing_permuted)
        next_states = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(next_states, dims=[1]))
        return ttnn.reshape(next_states, (1, 1, chunk_len, HIDDEN_SIZE))


__all__ = ["FusedDecoder", "resolved_fusion_policy"]
