# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-device optimized decoder layer for Qwen/Qwen3.6-27B.

The optimized stage inherits the proven fused graph and changes only measured
runtime contracts that have independent correctness and latency evidence.
"""

from __future__ import annotations

import math
import os

import torch
import ttnn

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    _as_weight,
    _rotate_half_partial,
    _slice_last,
    _state_key,
)
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder


def _dram_sharded_weight(source, mesh_device, *, dtype):
    """Materialize a transposed linear weight across the device DRAM banks."""
    source = source.transpose(-2, -1).contiguous()
    k, n = source.shape[-2:]
    dram_grid_size = mesh_device.dram_grid_size()
    dram_cores = dram_grid_size.x * dram_grid_size.y
    grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )
    padded_n = math.ceil(n / (32 * dram_cores)) * 32 * dram_cores
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            grid,
            (k, padded_n // dram_cores),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.as_tensor(
        source,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _decode_l1_memory(m, width, cores):
    """Match Blackhole's native row-major worker mapping for DRAM matmul."""
    full_rows, tail = divmod(cores, 11)
    ranges = set()
    if full_rows:
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, full_rows - 1)))
    if tail:
        ranges.add(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, full_rows),
                ttnn.CoreCoord(tail - 1, full_rows),
            )
        )
    grid = ttnn.CoreRangeSet(ranges)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid,
            (m, math.ceil(width / 32 / cores) * 32),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _norm_l1_memory(m, width, cores):
    """Use a rectangular grid required by sharded RMSNorm."""
    grid_x = min(8, cores)
    grid_y = cores // grid_x
    return ttnn.create_sharded_memory_config(
        (m, math.ceil(width / 32 / cores) * 32),
        ttnn.CoreGrid(x=grid_x, y=grid_y),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class OptimizedDecoder(FusedDecoder):
    """Fused decoder with measured Blackhole precision and fidelity policies."""

    OPTIMIZATION_MANIFEST = (
        "low_precision_mlp",
        "low_precision_projections",
        "gdn_read_geometry",
        "dram_sharded_mlp",
        "dram_sharded_projections",
        "gdn_update_geometry",
        "gdn_prefill_geometry",
        "large_prefill_geometry",
        "sharded_input_norm",
        "sharded_qk_norm_rope",
    )
    TOTAL_OPTIMIZATION_COUNTS = {name: 0 for name in OPTIMIZATION_MANIFEST}

    @classmethod
    def reset_optimization_counts(cls):
        cls.TOTAL_OPTIMIZATION_COUNTS = {name: 0 for name in cls.OPTIMIZATION_MANIFEST}

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.__class__ = cls
        decoder.optimization_counters = {name: 0 for name in cls.OPTIMIZATION_MANIFEST}
        decoder._runtime_phase = None
        global_mlp_policy = os.environ.get("QWEN36_OPT_MLP_DTYPE")
        dtype_by_name = {
            "bf16": ttnn.bfloat16,
            "bfp8": ttnn.bfloat8_b,
            "bfp4": ttnn.bfloat4_b,
        }
        decoder.mlp_policies = {
            role: os.environ.get(
                f"QWEN36_OPT_MLP_{role.upper()}_DTYPE",
                global_mlp_policy
                or (
                    "bfp4"
                    if decoder.layer_kind == "linear_attention" or role in ("gate", "up")
                    else "bfp8"
                ),
            )
            for role in ("gate", "up", "down")
        }
        decoder.mlp_weight_dtypes = {
            role: dtype_by_name[policy] for role, policy in decoder.mlp_policies.items()
        }
        decoder.mlp_compute_configs = {
            role: ttnn.WormholeComputeKernelConfig(
                math_fidelity=(
                    ttnn.MathFidelity.HiFi2 if policy == "bf16" else ttnn.MathFidelity.LoFi
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            for role, policy in decoder.mlp_policies.items()
        }
        decoder.mlp_weight_dtype = decoder.mlp_weight_dtypes["gate"]
        decoder.matmul_compute_config = decoder.mlp_compute_configs["gate"]
        decoder.dram_sharded_mlp = os.environ.get("QWEN36_OPT_DRAM_SHARDED_MLP", "1") == "1"
        global_mlp_cores = int(os.environ.get("QWEN36_OPT_DRAM_MLP_CORES", "32"))
        decoder.dram_sharded_mlp_cores = {
            role: int(os.environ.get(f"QWEN36_OPT_DRAM_MLP_{role.upper()}_CORES", global_mlp_cores))
            for role in ("gate", "up", "down")
        }
        if any(cores not in (8, 16, 32, 64) for cores in decoder.dram_sharded_mlp_cores.values()):
            raise ValueError("DRAM MLP role core counts must be 8, 16, 32, or 64")
        decoder.dram_sharded_mlp_blocks = {
            role: int(os.environ.get(f"QWEN36_OPT_DRAM_MLP_{role.upper()}_IN0_BLOCK_W", "0"))
            for role in ("gate", "up", "down")
        }
        decoder.mlp_padded_64 = any(cores == 64 for cores in decoder.dram_sharded_mlp_cores.values())
        if decoder.mlp_padded_64 and any(
            cores != 64 for cores in decoder.dram_sharded_mlp_cores.values()
        ):
            raise ValueError("the adapted 64-core MLP requires gate/up/down all use 64 cores")
        if decoder.dram_sharded_mlp_cores["gate"] != decoder.dram_sharded_mlp_cores["up"]:
            raise ValueError("gate and up must use the same core count")
        decoder.full_decode_mixed_mlp = os.environ.get("QWEN36_OPT_FULL_DECODE_MIXED_MLP", "0") == "1"
        decoder.dram_sharded_projections = os.environ.get(
            "QWEN36_OPT_DRAM_PROJECTIONS",
            "both" if decoder.layer_kind == "full_attention" else "output",
        )
        if decoder.dram_sharded_projections not in ("none", "input", "output", "both"):
            raise ValueError("QWEN36_OPT_DRAM_PROJECTIONS must be none, input, output, or both")
        decoder.projection_fidelity = os.environ.get("QWEN36_OPT_PROJECTION_FIDELITY", "auto")
        if decoder.projection_fidelity not in ("lofi", "hifi2", "auto"):
            raise ValueError("QWEN36_OPT_PROJECTION_FIDELITY must be lofi, hifi2, or auto")
        decoder.gdn_decode_update_geometry = os.environ.get(
            "QWEN36_OPT_GDN_DECODE_UPDATE_GEOMETRY", "reuse96m"
        )
        if decoder.gdn_decode_update_geometry not in ("auto", "reuse48", "reuse96m", "reuse96n"):
            raise ValueError("invalid QWEN36_OPT_GDN_DECODE_UPDATE_GEOMETRY")
        decoder.gdn_prefill_geometry = os.environ.get(
            "QWEN36_OPT_GDN_PREFILL_GEOMETRY", "reuse"
        )
        if decoder.gdn_prefill_geometry not in ("auto", "reuse"):
            raise ValueError("QWEN36_OPT_GDN_PREFILL_GEOMETRY must be auto or reuse")
        decoder.prefill_matmul_geometry = os.environ.get(
            "QWEN36_OPT_PREFILL_MATMUL_GEOMETRY", "auto"
        )
        if decoder.prefill_matmul_geometry not in ("auto", "2d"):
            raise ValueError("QWEN36_OPT_PREFILL_MATMUL_GEOMETRY must be auto or 2d")
        decoder.prefill_in0_block_w = int(
            os.environ.get("QWEN36_OPT_PREFILL_IN0_BLOCK_W", "2")
        )
        decoder.sharded_input_norm = os.environ.get(
            "QWEN36_OPT_SHARDED_INPUT_NORM", "1"
        ) == "1"
        decoder.sharded_qk_norm_rope = os.environ.get(
            "QWEN36_OPT_SHARDED_QK_NORM_ROPE", "0"
        ) == "1"
        layer_idx = decoder.layer_idx
        gate_source = state_dict[_state_key(state_dict, layer_idx, "mlp.gate_proj.weight")]
        up_source = state_dict[_state_key(state_dict, layer_idx, "mlp.up_proj.weight")]
        down_source = state_dict[_state_key(state_dict, layer_idx, "mlp.down_proj.weight")]
        if decoder.mlp_padded_64:
            gate_source = torch.nn.functional.pad(gate_source, [0, 1024, 0, 1024])
            up_source = torch.nn.functional.pad(up_source, [0, 1024, 0, 1024])
            down_source = torch.nn.functional.pad(down_source, [0, 1024, 0, 1024])
        decoder.mlp_runtime_hidden_size = gate_source.shape[1]
        decoder.mlp_runtime_intermediate_size = gate_source.shape[0]
        decoder.mlp_runtime_output_size = down_source.shape[0]
        for role, source in (("gate", gate_source), ("up", up_source)):
            name = f"mlp_{role}"
            ttnn.deallocate(getattr(decoder, name), force=True)
            setattr(
                decoder,
                name,
                _as_weight(
                    source,
                    decoder.mesh_device,
                    transpose=True,
                    dtype=decoder.mlp_weight_dtypes[role],
                ),
            )
            setattr(
                decoder,
                f"{name}_dram_prefill",
                (
                    _dram_sharded_weight(
                        source,
                        decoder.mesh_device,
                        dtype=decoder.mlp_weight_dtypes[role],
                    )
                    if decoder.dram_sharded_mlp
                    else None
                ),
            )
            setattr(
                decoder,
                f"{name}_decode",
                (
                    _dram_sharded_weight(
                        source,
                        decoder.mesh_device,
                        dtype=(
                            decoder.mlp_weight_dtypes[role]
                            if decoder.layer_kind == "linear_attention"
                            or decoder.full_decode_mixed_mlp
                            else ttnn.bfloat8_b
                        ),
                    )
                    if decoder.dram_sharded_mlp
                    else None
                ),
            )
        ttnn.deallocate(decoder.mlp_down, force=True)
        decoder.mlp_down = _as_weight(
            down_source,
            decoder.mesh_device,
            transpose=True,
            dtype=decoder.mlp_weight_dtypes["down"],
        )
        decoder.mlp_down_decode = (
            _dram_sharded_weight(
                down_source,
                decoder.mesh_device,
                dtype=decoder.mlp_weight_dtypes["down"],
            )
            if decoder.dram_sharded_mlp
            else None
        )
        decoder.mlp_down_dram_prefill = (
            _dram_sharded_weight(
                down_source,
                decoder.mesh_device,
                dtype=decoder.mlp_weight_dtypes["down"],
            )
            if decoder.dram_sharded_mlp
            else None
        )

        input_projection_policy = os.environ.get("QWEN36_OPT_INPUT_PROJ_DTYPE", "bfp8")
        output_projection_policy = os.environ.get(
            "QWEN36_OPT_OUTPUT_PROJ_DTYPE",
            "bfp8",
        )
        input_projection_dtype = dtype_by_name[input_projection_policy]
        output_projection_dtype = dtype_by_name[output_projection_policy]
        decoder.input_projection_weight_dtype = input_projection_dtype
        decoder.output_projection_weight_dtype = output_projection_dtype
        if decoder.layer_kind == "full_attention":
            q_and_gate = state_dict[_state_key(state_dict, layer_idx, "self_attn.q_proj.weight")]
            q_and_gate = q_and_gate.reshape(decoder.num_heads, decoder.head_dim * 2, decoder.hidden_size)
            query = q_and_gate[:, : decoder.head_dim].reshape(-1, decoder.hidden_size)
            gate = q_and_gate[:, decoder.head_dim :].reshape(-1, decoder.hidden_size)
            source = torch.cat(
                [
                    query,
                    state_dict[_state_key(state_dict, layer_idx, "self_attn.k_proj.weight")],
                    state_dict[_state_key(state_dict, layer_idx, "self_attn.v_proj.weight")],
                    gate,
                ],
                dim=0,
            )
            replacements = (
                ("full_qkv", source, input_projection_dtype),
                (
                    "o_proj",
                    state_dict[_state_key(state_dict, layer_idx, "self_attn.o_proj.weight")],
                    output_projection_dtype,
                ),
            )
        else:
            beta = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_b.weight")]
            a = state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_a.weight")]
            beta = torch.nn.functional.pad(beta, [0, 0, 0, (-beta.shape[0]) % 32])
            a = torch.nn.functional.pad(a, [0, 0, 0, (-a.shape[0]) % 32])
            source = torch.cat(
                [
                    state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_qkv.weight")],
                    state_dict[_state_key(state_dict, layer_idx, "linear_attn.in_proj_z.weight")],
                    beta,
                    a,
                ],
                dim=0,
            )
            replacements = (
                ("linear_projections", source, input_projection_dtype),
                (
                    "linear_out_proj",
                    state_dict[_state_key(state_dict, layer_idx, "linear_attn.out_proj.weight")],
                    output_projection_dtype,
                ),
            )
        for name, source, dtype in replacements:
            ttnn.deallocate(getattr(decoder, name), force=True)
            setattr(
                decoder,
                name,
                _as_weight(source, decoder.mesh_device, transpose=True, dtype=dtype),
            )
            is_input = name in ("full_qkv", "linear_projections")
            use_dram = decoder.dram_sharded_projections in (
                "both",
                "input" if is_input else "output",
            )
            role = (
                "full_input"
                if name == "full_qkv"
                else "full_output"
                if name == "o_proj"
                else "linear_input"
                if name == "linear_projections"
                else "linear_output"
            )
            logical_n = source.shape[0]
            automatic_cores = min(32, math.gcd(source.shape[1] // 32, logical_n // 32))
            cores = int(
                os.environ.get(
                    f"QWEN36_OPT_DRAM_PROJ_{role.upper()}_CORES",
                    os.environ.get("QWEN36_OPT_DRAM_PROJ_CORES", str(automatic_cores)),
                )
            )
            if cores <= 0 or cores > 110 or source.shape[1] // 32 % cores:
                raise ValueError(f"illegal {role} projection core count {cores}")
            padded_n = math.ceil(logical_n / (32 * cores)) * 32 * cores
            dram_source = (
                torch.nn.functional.pad(source, [0, 0, 0, padded_n - logical_n])
                if padded_n != logical_n
                else source
            )
            setattr(decoder, f"{role}_dram_cores", cores)
            setattr(decoder, f"{role}_logical_n", logical_n)
            setattr(decoder, f"{role}_dram_block", int(os.environ.get(
                f"QWEN36_OPT_DRAM_PROJ_{role.upper()}_IN0_BLOCK_W", "0"
            )))
            setattr(
                decoder,
                f"{name}_dram_decode",
                _dram_sharded_weight(dram_source, decoder.mesh_device, dtype=dtype) if use_dram else None,
            )
        return decoder

    def prefill_forward(self, *args, **kwargs):
        previous_phase = self._runtime_phase
        self._runtime_phase = "prefill"
        try:
            return super().prefill_forward(*args, **kwargs)
        finally:
            self._runtime_phase = previous_phase

    def decode_forward(
        self,
        hidden_states,
        *,
        current_positions,
        cos=None,
        sin=None,
        page_table=None,
        kv_cache=None,
        linear_state=None,
    ):
        previous_phase = self._runtime_phase
        self._runtime_phase = "decode"
        try:
            if not self.sharded_input_norm:
                return super().decode_forward(
                    hidden_states,
                    current_positions=current_positions,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    linear_state=linear_state,
                )
            if hidden_states.shape[2] > 1 and self.layer_kind == "linear_attention":
                return self._linear_decode_users_independently(
                    hidden_states,
                    current_positions=current_positions,
                    linear_state=linear_state,
                )
            self._record_optimization("sharded_input_norm")
            residual_memory = _norm_l1_memory(32, self.hidden_size, 32)
            residual = ttnn.to_memory_config(hidden_states, residual_memory)
            normalized = ttnn.rms_norm(
                residual,
                epsilon=self.rms_norm_eps,
                weight=self.input_norm,
                memory_config=residual_memory,
                program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[8, 4],
                    subblock_w=1,
                    block_h=1,
                    block_w=math.ceil(self.hidden_size / 32 / 32),
                    inplace=False,
                ),
            )
            if self.layer_kind == "full_attention":
                if any(
                    value is None
                    for value in (cos, sin, page_table, kv_cache, current_positions)
                ):
                    raise ValueError(
                        "full_attention decode requires cos, sin, page_table, "
                        "current_positions, and kv_cache"
                    )
                mixed = self._full_decode(
                    normalized,
                    cos=cos,
                    sin=sin,
                    page_table=page_table,
                    current_positions=current_positions,
                    kv_cache=kv_cache,
                )
            else:
                if linear_state is None or current_positions is None:
                    raise ValueError(
                        "linear_attention decode requires current_positions and linear_state"
                    )
                mixed = self._linear_decode(normalized, linear_state=linear_state)
            return self._finish_layer(residual, mixed)
        finally:
            self._runtime_phase = previous_phase

    def _record_optimization(self, name: str):
        self.optimization_counters[name] += 1
        type(self).TOTAL_OPTIMIZATION_COUNTS[name] += 1

    def _dram_projection(self, hidden_states, interleaved_weight, dram_weight, *, role):
        """Run a decode projection with the largest legal common K/N core factor."""
        if dram_weight is None:
            return ttnn.linear(hidden_states, interleaved_weight, dtype=ttnn.bfloat16)
        self._record_optimization("dram_sharded_projections")
        k = hidden_states.shape[-1]
        n = dram_weight.shape[-1]
        cores = getattr(self, f"{role}_dram_cores")
        input_memory = _norm_l1_memory(32, k, cores)
        output_memory = _decode_l1_memory(32, n, cores)
        if hidden_states.memory_config() != input_memory:
            hidden_states = ttnn.to_memory_config(hidden_states, input_memory)
        blocks = k // 32 // cores
        requested_block = getattr(self, f"{role}_dram_block")
        in0_block_w = requested_block or next(
            divisor
            for divisor in (17, 10, 8, 6, 5, 4, 3, 2, 1)
            if blocks % divisor == 0
        )
        if blocks % in0_block_w:
            raise ValueError(f"{role} block {in0_block_w} does not divide {blocks}")
        compute_kernel_config = None
        if self.projection_fidelity != "auto":
            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=(
                    ttnn.MathFidelity.LoFi
                    if self.projection_fidelity == "lofi"
                    else ttnn.MathFidelity.HiFi2
                ),
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        output = ttnn.linear(
            hidden_states,
            dram_weight,
            dtype=ttnn.bfloat16,
            program_config=ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=n // 32 // cores,
                fused_activation=None,
            ),
            memory_config=output_memory,
            compute_kernel_config=compute_kernel_config,
        )
        logical_n = getattr(self, f"{role}_logical_n")
        if logical_n != n:
            end = list(output.shape)
            end[-1] = logical_n
            output = ttnn.slice(output, [0, 0, 0, 0], end)
            # The padded linear-input candidate feeds interleaved convolution state
            # and elementwise consumers; its sliced shard cannot be concatenated
            # with those tensors directly.
            if role == "linear_input":
                output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
        return output

    def _prefill_program_config(self, m, k, n):
        """Create a phase-appropriate 2D interleaved matmul config for M >= 64."""
        if (
            self.prefill_matmul_geometry != "2d"
            or self._runtime_phase != "prefill"
            or m < 64
        ):
            return None
        self._record_optimization("large_prefill_geometry")
        m_tiles, k_tiles, n_tiles = (math.ceil(size / 32) for size in (m, k, n))
        grid_x = next(divisor for divisor in (8, 7, 6, 5, 4, 3, 2, 1) if n_tiles % divisor == 0)
        grid_y = next(divisor for divisor in (8, 7, 6, 5, 4, 3, 2, 1) if m_tiles % divisor == 0)
        per_core_m = m_tiles // grid_y
        per_core_n = n_tiles // grid_x
        block_w = self.prefill_in0_block_w
        if k_tiles % block_w:
            raise ValueError(f"prefill block {block_w} does not divide K tiles {k_tiles}")
        subblock_w = next(divisor for divisor in (4, 3, 2, 1) if per_core_n % divisor == 0)
        subblock_h = next(
            divisor
            for divisor in (4, 3, 2, 1)
            if per_core_m % divisor == 0 and divisor * subblock_w <= 4
        )
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=block_w,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )

    def allocate_paged_kv_cache(self, *, num_blocks: int, dtype=None):
        """Allocate the selected BFP8 cache unless a caller requests another dtype."""
        if dtype is None:
            dtype = (
                ttnn.bfloat8_b
                if os.environ.get("QWEN36_OPT_KV_CACHE_DTYPE", "bfp8") == "bfp8"
                else ttnn.bfloat16
            )
        return super().allocate_paged_kv_cache(num_blocks=num_blocks, dtype=dtype)

    def _full_prefill(self, hidden_states, **kwargs):
        """Preserve the fused path and cast fills to the selected cache dtype."""
        query, key, value, gate = self._full_qkv_prefill(hidden_states, kwargs["cos"], kwargs["sin"])
        key_cache, value_cache = kwargs["kv_cache"]
        batch = hidden_states.shape[0]
        cache_batch_idx = kwargs.get("cache_batch_idx", 0)
        for batch_idx in range(batch):
            key_user = ttnn.slice(
                key,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, key.shape[1], key.shape[2], key.shape[3]],
            )
            value_user = ttnn.slice(
                value,
                [batch_idx, 0, 0, 0],
                [batch_idx + 1, value.shape[1], value.shape[2], value.shape[3]],
            )
            key_fill = ttnn.typecast(key_user, key_cache.dtype) if key_user.dtype != key_cache.dtype else key_user
            value_fill = (
                ttnn.typecast(value_user, value_cache.dtype)
                if value_user.dtype != value_cache.dtype
                else value_user
            )
            ttnn.experimental.paged_fill_cache(
                key_cache,
                key_fill,
                kwargs["page_table"],
                batch_idx=cache_batch_idx + batch_idx,
            )
            ttnn.experimental.paged_fill_cache(
                value_cache,
                value_fill,
                kwargs["page_table"],
                batch_idx=cache_batch_idx + batch_idx,
            )
            if key_fill is not key_user:
                ttnn.deallocate(key_fill)
            if value_fill is not value_user:
                ttnn.deallocate(value_fill)

        attention = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.attention_scale,
        )
        attention = ttnn.permute(attention, [0, 2, 1, 3])
        attention = ttnn.reshape(
            attention,
            [batch, 1, hidden_states.shape[2], self.num_heads * self.head_dim],
        )
        attention = self._gated_attention(attention, gate)
        output = ttnn.linear(
            attention,
            self.o_proj,
            dtype=ttnn.bfloat16,
            program_config=self._prefill_program_config(
                attention.shape[0] * attention.shape[1] * attention.shape[2],
                attention.shape[-1],
                self.hidden_size,
            ),
        )
        logical_seq_len = kwargs["logical_seq_len"]
        if logical_seq_len != hidden_states.shape[2]:
            output = ttnn.slice(output, [0, 0, 0, 0], [batch, 1, logical_seq_len, self.hidden_size])
        return output

    def _full_prefill_chunked_layer(
        self,
        hidden_states,
        *,
        cos,
        sin,
        page_table,
        kv_cache,
        logical_seq_len: int,
    ):
        """Long-context fused path with cache-dtype-correct device fills."""
        batch = hidden_states.shape[0]
        physical_seq_len = hidden_states.shape[2]
        key_cache, value_cache = kv_cache
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=self.full_prefill_q_chunk_size,
            k_chunk_size=self.full_prefill_q_chunk_size,
            exp_approx_mode=False,
        )
        outputs = []
        for chunk_start in range(0, physical_seq_len, self.full_prefill_chunk_size):
            chunk_end = min(chunk_start + self.full_prefill_chunk_size, physical_seq_len)
            chunk_len = chunk_end - chunk_start
            logical_chunk_len = min(chunk_len, max(0, logical_seq_len - chunk_start))
            residual = ttnn.slice(
                hidden_states,
                [0, 0, chunk_start, 0],
                [batch, 1, chunk_end, self.hidden_size],
            )
            normalized = ttnn.rms_norm(residual, epsilon=self.rms_norm_eps, weight=self.input_norm)
            cos_chunk = ttnn.slice(
                cos,
                [0, 0, chunk_start, 0],
                [cos.shape[0], cos.shape[1], chunk_end, cos.shape[3]],
            )
            sin_chunk = ttnn.slice(
                sin,
                [0, 0, chunk_start, 0],
                [sin.shape[0], sin.shape[1], chunk_end, sin.shape[3]],
            )
            query, key, value, gate = self._full_qkv_prefill(normalized, cos_chunk, sin_chunk)

            first_page = chunk_start // self.page_block_size
            chunk_pages = (chunk_len + self.page_block_size - 1) // self.page_block_size
            fill_page_table = ttnn.slice(
                page_table,
                [0, first_page],
                [page_table.shape[0], first_page + chunk_pages],
            )
            for batch_idx in range(batch):
                key_user = ttnn.slice(
                    key,
                    [batch_idx, 0, 0, 0],
                    [batch_idx + 1, key.shape[1], key.shape[2], key.shape[3]],
                )
                value_user = ttnn.slice(
                    value,
                    [batch_idx, 0, 0, 0],
                    [batch_idx + 1, value.shape[1], value.shape[2], value.shape[3]],
                )
                key_fill = (
                    ttnn.typecast(key_user, key_cache.dtype)
                    if key_user.dtype != key_cache.dtype
                    else key_user
                )
                value_fill = (
                    ttnn.typecast(value_user, value_cache.dtype)
                    if value_user.dtype != value_cache.dtype
                    else value_user
                )
                ttnn.experimental.paged_fill_cache(key_cache, key_fill, fill_page_table, batch_idx=batch_idx)
                ttnn.experimental.paged_fill_cache(value_cache, value_fill, fill_page_table, batch_idx=batch_idx)
                if key_fill is not key_user:
                    ttnn.deallocate(key_fill)
                if value_fill is not value_user:
                    ttnn.deallocate(value_fill)
                ttnn.deallocate(key_user)
                ttnn.deallocate(value_user)
            for tensor in (fill_page_table, key, value, normalized, cos_chunk, sin_chunk):
                ttnn.deallocate(tensor)

            query_pad = (-chunk_len) % self.full_prefill_q_chunk_size
            if query_pad:
                query_unpadded = query
                query = ttnn.pad(
                    query_unpadded,
                    padding=[(0, 0), (0, 0), (0, query_pad), (0, 0)],
                    value=0.0,
                )
                ttnn.deallocate(query_unpadded)
            attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=query,
                input_tensor_k=key_cache,
                input_tensor_v=value_cache,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start,
                scale=self.attention_scale,
                program_config=program_config,
            )
            ttnn.deallocate(query)
            if query_pad:
                padded_attention = attention
                attention = ttnn.slice(
                    padded_attention,
                    [0, 0, 0, 0],
                    [batch, self.num_heads, chunk_len, self.head_dim],
                )
                ttnn.deallocate(padded_attention)

            attention = ttnn.permute(attention, [0, 2, 1, 3])
            attention = ttnn.reshape(
                attention,
                [batch, 1, chunk_len, self.num_heads * self.head_dim],
            )
            gated_attention = self._gated_attention(attention, gate)
            mixed = ttnn.linear(
                gated_attention,
                self.o_proj,
                dtype=ttnn.bfloat16,
                program_config=self._prefill_program_config(
                    gated_attention.shape[0]
                    * gated_attention.shape[1]
                    * gated_attention.shape[2],
                    gated_attention.shape[-1],
                    self.hidden_size,
                ),
            )
            for tensor in (attention, gated_attention, gate):
                ttnn.deallocate(tensor)

            if logical_chunk_len:
                if logical_chunk_len != chunk_len:
                    logical_residual = ttnn.slice(
                        residual,
                        [0, 0, 0, 0],
                        [batch, 1, logical_chunk_len, self.hidden_size],
                    )
                    logical_mixed = ttnn.slice(
                        mixed,
                        [0, 0, 0, 0],
                        [batch, 1, logical_chunk_len, self.hidden_size],
                    )
                    ttnn.deallocate(residual)
                    ttnn.deallocate(mixed)
                    residual, mixed = logical_residual, logical_mixed
                output = self._finish_layer(residual, mixed)
                ttnn.deallocate(residual)
                ttnn.deallocate(mixed)
                outputs.append(output)
            else:
                ttnn.deallocate(residual)
                ttnn.deallocate(mixed)

        if len(outputs) == 1:
            return outputs[0]
        result = ttnn.concat(outputs, dim=2)
        for output in outputs:
            ttnn.deallocate(output)
        return result

    def _full_decode(self, hidden_states, **kwargs):
        """Paged decode with an optional DRAM-sharded output projection candidate."""
        padded_batch = hidden_states.shape[2]
        query, key, value, gate = self._full_qkv_decode(hidden_states, kwargs["cos"], kwargs["sin"])
        batch_grid = ttnn.num_cores_to_corerangeset(
            padded_batch,
            ttnn.CoreCoord(8, 8),
            row_wise=True,
        )
        decode_head_memcfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        query = ttnn.to_memory_config(query, decode_head_memcfg)
        key = ttnn.to_memory_config(key, decode_head_memcfg)
        value = ttnn.to_memory_config(value, decode_head_memcfg)
        key_cache, value_cache = kwargs["kv_cache"]
        ttnn.experimental.paged_update_cache(
            key_cache,
            key,
            update_idxs_tensor=kwargs["current_positions"],
            page_table=kwargs["page_table"],
        )
        ttnn.experimental.paged_update_cache(
            value_cache,
            value,
            update_idxs_tensor=kwargs["current_positions"],
            page_table=kwargs["page_table"],
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=kwargs["current_positions"],
            page_table_tensor=kwargs["page_table"],
            scale=self.attention_scale,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.reshape(
            attention,
            [1, 1, padded_batch, self.num_heads * self.head_dim],
        )
        attention = self._gated_attention(attention, gate)
        return self._dram_projection(
            attention,
            self.o_proj,
            self.o_proj_dram_decode,
            role="full_output",
        )

    def _mlp(self, hidden_states):
        """Run the dominant expert MLP with the selected real-weight policy."""
        self._record_optimization("low_precision_mlp")
        m = math.prod(hidden_states.shape[index] for index in range(len(hidden_states.shape) - 1))

        use_sharded = self.dram_sharded_mlp and m <= 32 and self._runtime_phase in (
            "prefill",
            "decode",
        )
        if self.mlp_padded_64 and use_sharded:
            hidden_states = ttnn.pad(
                hidden_states,
                padding=[(0, 0), (0, 0), (0, 0), (0, 1024)],
                value=0.0,
            )

        def width_sharded_memory(role, width):
            return _decode_l1_memory(32, width, self.dram_sharded_mlp_cores[role])

        def dram_program(role, k, n):
            if not use_sharded:
                return None
            compute_cores = self.dram_sharded_mlp_cores[role]
            if k // 32 % compute_cores or n // 32 % compute_cores:
                raise ValueError(
                    f"{role} shape {k}x{n} is not tile-divisible across {compute_cores} cores"
                )
            blocks = k // 32 // compute_cores
            requested_block = self.dram_sharded_mlp_blocks[role]
            in0_block_w = requested_block or next(
                divisor
                for divisor in (68, 34, 20, 17, 10, 9, 8, 5, 4, 3, 2, 1)
                if blocks % divisor == 0
            )
            if blocks % in0_block_w:
                raise ValueError(f"{role} block {in0_block_w} does not divide {blocks}")
            return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=in0_block_w,
                per_core_M=1,
                per_core_N=n // 32 // compute_cores,
                fused_activation=None,
            )

        def sharded_weight(role):
            if not use_sharded:
                return getattr(self, f"mlp_{role}")
            suffix = "decode" if self._runtime_phase == "decode" else "dram_prefill"
            return getattr(self, f"mlp_{role}_{suffix}")

        gate = ttnn.linear(
            hidden_states,
            sharded_weight("gate"),
            dtype=ttnn.bfloat16,
            program_config=(
                dram_program("gate", self.mlp_runtime_hidden_size, self.mlp_runtime_intermediate_size)
                or self._prefill_program_config(
                    m, self.mlp_runtime_hidden_size, self.mlp_runtime_intermediate_size
                )
            ),
            memory_config=(
                width_sharded_memory("gate", self.mlp_runtime_intermediate_size)
                if use_sharded
                else None
            ),
            compute_kernel_config=self.mlp_compute_configs["gate"],
        )
        up = ttnn.linear(
            hidden_states,
            sharded_weight("up"),
            dtype=ttnn.bfloat16,
            program_config=(
                dram_program("up", self.mlp_runtime_hidden_size, self.mlp_runtime_intermediate_size)
                or self._prefill_program_config(
                    m, self.mlp_runtime_hidden_size, self.mlp_runtime_intermediate_size
                )
            ),
            memory_config=(
                width_sharded_memory("up", self.mlp_runtime_intermediate_size)
                if use_sharded
                else None
            ),
            compute_kernel_config=self.mlp_compute_configs["up"],
        )
        self._record_fusion("mlp_silu_multiply")
        activated = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        if use_sharded and self.dram_sharded_mlp_cores["down"] != self.dram_sharded_mlp_cores["gate"]:
            activated = ttnn.to_memory_config(
                activated,
                width_sharded_memory("down", self.mlp_runtime_intermediate_size),
            )
        output = ttnn.linear(
            activated,
            sharded_weight("down"),
            dtype=ttnn.bfloat16,
            program_config=(
                dram_program("down", self.mlp_runtime_intermediate_size, self.mlp_runtime_output_size)
                or self._prefill_program_config(
                    m, self.mlp_runtime_intermediate_size, self.mlp_runtime_output_size
                )
            ),
            memory_config=(
                width_sharded_memory("down", self.mlp_runtime_output_size)
                if use_sharded
                else None
            ),
            compute_kernel_config=self.mlp_compute_configs["down"],
        )
        if self.mlp_runtime_output_size != self.hidden_size:
            output = ttnn.slice(
                output,
                [0, 0, 0, 0],
                [output.shape[0], output.shape[1], output.shape[2], self.hidden_size],
            )
        return output

    def _finish_layer(self, residual, mixed):
        """Keep the post-attention residual, RMSNorm, and MLP coherently sharded."""
        m = math.prod(residual.shape[index] for index in range(len(residual.shape) - 1))
        use_sharded = not (
            not self.dram_sharded_mlp
            or m > 32
            or self._runtime_phase not in ("prefill", "decode")
            or self.mlp_padded_64
        )
        if not use_sharded:
            hidden_states = ttnn.add(residual, mixed)
            mlp_input = ttnn.rms_norm(
                hidden_states,
                epsilon=self.rms_norm_eps,
                weight=self.post_attention_norm,
            )
            return ttnn.add(hidden_states, self._mlp(mlp_input))

        self._record_optimization("dram_sharded_mlp")
        residual_memory = _norm_l1_memory(
            32, self.hidden_size, self.dram_sharded_mlp_cores["gate"]
        )
        hidden_states = ttnn.add(residual, mixed, memory_config=residual_memory)
        if hidden_states.memory_config() != residual_memory:
            hidden_states = ttnn.to_memory_config(hidden_states, residual_memory)
        mlp_input = ttnn.rms_norm(
            hidden_states,
            epsilon=self.rms_norm_eps,
            weight=self.post_attention_norm,
            memory_config=residual_memory,
            program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, self.dram_sharded_mlp_cores["gate"] // 8],
                subblock_w=1,
                block_h=1,
                block_w=math.ceil(
                    self.hidden_size / 32 / self.dram_sharded_mlp_cores["gate"]
                ),
                inplace=False,
            ),
        )
        mlp_output = self._mlp(mlp_input)
        if self.dram_sharded_mlp_cores["down"] != self.dram_sharded_mlp_cores["gate"]:
            mlp_output = ttnn.to_memory_config(mlp_output, residual_memory)
        return ttnn.add(hidden_states, mlp_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear_input_projections(self, hidden_states, *, activate_beta: bool = True):
        """Run the packed GDN input projection with the canonical LoFi policy."""
        self._record_optimization("low_precision_projections")
        self._record_fusion("shared_lhs_linear_projections")
        self._record_fusion("tile_aligned_linear_projection_tails")
        m = math.prod(hidden_states.shape[index] for index in range(len(hidden_states.shape) - 1))
        packed = (
            self._dram_projection(
                hidden_states,
                self.linear_projections,
                self.linear_projections_dram_decode,
                role="linear_input",
            )
            if m <= 32
            else ttnn.linear(
                hidden_states,
                self.linear_projections,
                dtype=ttnn.bfloat16,
                program_config=self._prefill_program_config(
                    m, self.hidden_size, self.linear_projections.shape[-1]
                ),
            )
        )
        qkv_end = self.conv_dim
        z_end = qkv_end + self.linear_num_value_heads * self.linear_value_head_dim
        beta_end = z_end + self.linear_num_value_heads
        beta_padded_end = z_end + math.ceil(self.linear_num_value_heads / 32) * 32
        mixed_qkv = _slice_last(packed, 0, qkv_end)
        z = _slice_last(packed, qkv_end, z_end)
        beta = _slice_last(packed, z_end, beta_end)
        if activate_beta:
            beta = ttnn.sigmoid(beta)
        a = _slice_last(packed, beta_padded_end, beta_padded_end + self.linear_num_value_heads)
        return mixed_qkv, z, beta, a

    def _gdn_read_program_config(self):
        """Use the Qwen36 recurrent GDN geometry for the two row/state reads."""
        self._record_optimization("gdn_read_geometry")
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            in0_block_w=math.ceil(self.linear_key_head_dim / 32),
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=math.ceil(self.linear_value_head_dim / 32),
        )

    def _gdn_prefill_matmul(self, a, b, *, transpose_a=False, transpose_b=False):
        """Apply a legal full-output reuse geometry to a chunked GDN matmul."""
        program_config = None
        if self.gdn_prefill_geometry == "reuse":
            self._record_optimization("gdn_prefill_geometry")
            m = a.shape[-1] if transpose_a else a.shape[-2]
            k = a.shape[-2] if transpose_a else a.shape[-1]
            n = b.shape[-2] if transpose_b else b.shape[-1]
            m_tiles = math.ceil(m / 32)
            k_tiles = math.ceil(k / 32)
            n_tiles = math.ceil(n / 32)
            block_w = next(divisor for divisor in (4, 3, 2, 1) if k_tiles % divisor == 0)
            subblock_w = next(divisor for divisor in (4, 2, 1) if n_tiles % divisor == 0)
            subblock_h = next(
                divisor
                for divisor in (4, 2, 1)
                if m_tiles % divisor == 0 and divisor * subblock_w <= 4
            )
            program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                in0_block_w=block_w,
                out_subblock_h=subblock_h,
                out_subblock_w=subblock_w,
                per_core_M=m_tiles,
                per_core_N=n_tiles,
            )
        return ttnn.matmul(
            a,
            b,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            dtype=ttnn.float32,
            program_config=program_config,
        )

    def _linear_chunk(self, hidden_states, *, conv_state, recurrent_state, valid_tokens: int):
        """Fused linear chunk with configured large-prefill projections."""
        batch = hidden_states.shape[0]
        mixed_qkv, z, beta, a = self._linear_input_projections(hidden_states)
        a = ttnn.typecast(a, ttnn.float32)
        convolved = self._linear_causal_conv_chunk(
            mixed_qkv, conv_state=conv_state, valid_tokens=valid_tokens
        )
        key_width = self.linear_num_key_heads * self.linear_key_head_dim
        query = _slice_last(convolved, 0, key_width)
        key = _slice_last(convolved, key_width, key_width * 2)
        value = _slice_last(convolved, key_width * 2, self.conv_dim)
        query = ttnn.reshape(
            query,
            [batch, valid_tokens, self.linear_num_key_heads, self.linear_key_head_dim],
        )
        key = ttnn.reshape(
            key,
            [batch, valid_tokens, self.linear_num_key_heads, self.linear_key_head_dim],
        )
        query = self._repeat_linear_qk_chunk(query, sequence_length=valid_tokens)
        key = self._repeat_linear_qk_chunk(key, sequence_length=valid_tokens)
        value = ttnn.permute(
            ttnn.reshape(
                value,
                [batch, valid_tokens, self.linear_num_value_heads, self.linear_value_head_dim],
            ),
            [0, 2, 1, 3],
        )
        query = self._pad_linear_chunk(
            self._linear_l2_norm(query), valid_tokens=valid_tokens, sequence_dim=2
        )
        key = self._pad_linear_chunk(
            self._linear_l2_norm(key), valid_tokens=valid_tokens, sequence_dim=2
        )
        value = self._pad_linear_chunk(value, valid_tokens=valid_tokens, sequence_dim=2)
        beta = ttnn.permute(
            ttnn.reshape(beta, [batch, valid_tokens, self.linear_num_value_heads, 1]),
            [0, 2, 1, 3],
        )
        a = ttnn.permute(
            ttnn.reshape(a, [batch, valid_tokens, self.linear_num_value_heads, 1]),
            [0, 2, 1, 3],
        )
        g = ttnn.multiply(self.a_decay, ttnn.softplus(ttnn.add(a, self.dt_bias)))
        beta = self._pad_linear_chunk(beta, valid_tokens=valid_tokens, sequence_dim=2)
        g = self._pad_linear_chunk(g, valid_tokens=valid_tokens, sequence_dim=2)
        core = self._linear_gated_delta_chunk(
            query, key, value, beta, g, recurrent_state=recurrent_state
        )
        core = ttnn.permute(core, [0, 2, 1, 3])
        core = ttnn.reshape(
            core,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads, self.linear_value_head_dim],
        )
        z = self._pad_linear_chunk(z, valid_tokens=valid_tokens, sequence_dim=2)
        z = ttnn.reshape(
            z,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads, self.linear_value_head_dim],
        )
        norm = ttnn.rsqrt(
            ttnn.add(
                ttnn.mean(ttnn.multiply(core, core), dim=-1, keepdim=True),
                self.rms_norm_eps,
            )
        )
        core = ttnn.multiply(ttnn.multiply(core, norm), self.linear_norm)
        self._record_fusion("linear_silu_multiply")
        core = ttnn.multiply(core, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        core = ttnn.reshape(
            core,
            [batch, 1, self.linear_chunk_size, self.linear_num_value_heads * self.linear_value_head_dim],
        )
        output = ttnn.linear(
            core,
            self.linear_out_proj,
            dtype=ttnn.bfloat16,
            program_config=self._prefill_program_config(
                self.linear_chunk_size,
                self.linear_num_value_heads * self.linear_value_head_dim,
                self.hidden_size,
            ),
        )
        if valid_tokens != self.linear_chunk_size:
            output = ttnn.slice(
                output, [0, 0, 0, 0], [batch, 1, valid_tokens, self.hidden_size]
            )
        return output

    def _linear_gated_delta_chunk(self, query, key, value, beta, g, *, recurrent_state):
        """Chunked GDN delta rule with explicit reusable matmul geometries."""
        query = ttnn.typecast(query, ttnn.float32)
        key = ttnn.typecast(key, ttnn.float32)
        value = ttnn.typecast(value, ttnn.float32)
        beta = ttnn.typecast(beta, ttnn.float32)
        g = ttnn.typecast(g, ttnn.float32)

        query = ttnn.multiply(query, 1.0 / math.sqrt(self.linear_key_head_dim))
        cumulative_g = ttnn.cumsum(g, dim=2, dtype=ttnn.float32)
        decay_exponent = ttnn.subtract(cumulative_g, ttnn.transpose(cumulative_g, -2, -1))
        decay_exponent = ttnn.multiply(decay_exponent, self.linear_chunk_lower)
        decay = ttnn.multiply(ttnn.exp(decay_exponent), self.linear_chunk_lower)

        value_beta = ttnn.multiply(value, beta)
        key_beta = ttnn.multiply(key, beta)
        self._record_fusion("matmul_transpose_flags")
        base_attention = self._gdn_prefill_matmul(key_beta, key, transpose_b=True)
        base_attention = ttnn.multiply(ttnn.neg(base_attention), decay)
        base_attention = ttnn.multiply(base_attention, self.linear_chunk_strict_lower)
        inverse_attention = self._linear_chunk_inverse(base_attention)

        transformed_value = self._gdn_prefill_matmul(inverse_attention, value_beta)
        cumulative_exp = ttnn.exp(cumulative_g)
        cumulative_key = self._gdn_prefill_matmul(
            inverse_attention, ttnn.multiply(key_beta, cumulative_exp)
        )
        state_projection = self._gdn_prefill_matmul(cumulative_key, recurrent_state)
        adjusted_value = ttnn.subtract(transformed_value, state_projection)

        within_chunk = self._gdn_prefill_matmul(query, key, transpose_b=True)
        within_chunk = ttnn.multiply(ttnn.multiply(within_chunk, decay), self.linear_chunk_lower)
        from_initial_state = self._gdn_prefill_matmul(
            ttnn.multiply(query, cumulative_exp), recurrent_state
        )
        core = ttnn.add(
            from_initial_state,
            self._gdn_prefill_matmul(within_chunk, adjusted_value),
        )

        final_g = ttnn.slice(
            cumulative_g,
            [0, 0, self.linear_chunk_size - 1, 0],
            [cumulative_g.shape[0], cumulative_g.shape[1], self.linear_chunk_size, 1],
        )
        state_key = ttnn.multiply(key, ttnn.exp(ttnn.subtract(final_g, cumulative_g)))
        state_update = self._gdn_prefill_matmul(
            state_key, adjusted_value, transpose_a=True
        )
        new_recurrent_state = ttnn.add(
            ttnn.multiply(recurrent_state, ttnn.exp(final_g)), state_update
        )
        ttnn.copy(new_recurrent_state, recurrent_state)
        return ttnn.typecast(core, ttnn.bfloat16)

    def _linear_chunk_inverse(self, base_attention):
        """Evaluate the exact triangular inverse with configured FP32 matmuls."""
        power = base_attention
        inverse = ttnn.add(self.linear_chunk_identity, power)
        for _ in range(1, int(math.log2(self.linear_chunk_size))):
            power = self._gdn_prefill_matmul(power, power)
            inverse = self._gdn_prefill_matmul(
                inverse,
                ttnn.add(self.linear_chunk_identity, power),
            )
        return inverse

    def _gdn_update_program_config(self):
        geometry = self.gdn_decode_update_geometry
        if geometry == "auto":
            return None
        self._record_optimization("gdn_update_geometry")
        if geometry == "reuse48":
            per_core_m, per_core_n, subblock_h, subblock_w = 4, 4, 2, 2
        elif geometry == "reuse96m":
            per_core_m, per_core_n, subblock_h, subblock_w = 2, 4, 1, 4
        elif geometry == "reuse96n":
            per_core_m, per_core_n, subblock_h, subblock_w = 4, 2, 2, 2
        else:
            raise ValueError(
                "QWEN36_OPT_GDN_UPDATE_GEOMETRY must be auto, reuse48, reuse96m, or reuse96n"
            )
        return ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            in0_block_w=1,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
        )

    def _linear_token(self, hidden_states, *, conv_state, recurrent_state):
        """Fused recurrent GDN decode with tuned row/state matmul geometry."""
        batch = hidden_states.shape[0]
        mixed_qkv, z, beta, a = self._linear_input_projections(hidden_states, activate_beta=False)

        old_tail = ttnn.slice(
            conv_state,
            [0, 0, 1, 0],
            [batch, 1, self.conv_kernel_size, self.conv_dim],
        )
        mixed_row = ttnn.reshape(mixed_qkv, [batch, 1, 1, self.conv_dim])
        new_conv_state = ttnn.concat([old_tail, mixed_row], dim=2)
        ttnn.copy(new_conv_state, conv_state)
        convolved = ttnn.sum(ttnn.multiply(new_conv_state, self.conv_weight), dim=2, keepdim=True)
        convolved = ttnn.silu(convolved)

        key_start = self.linear_num_key_heads * self.linear_key_head_dim
        query = _slice_last(convolved, 0, key_start)
        key = _slice_last(convolved, key_start, key_start * 2)
        value = _slice_last(convolved, key_start * 2, self.conv_dim)
        query = self._repeat_linear_qk(
            ttnn.reshape(query, [batch, self.linear_num_key_heads, 1, self.linear_key_head_dim])
        )
        key = self._repeat_linear_qk(
            ttnn.reshape(key, [batch, self.linear_num_key_heads, 1, self.linear_key_head_dim])
        )
        value = ttnn.reshape(value, [batch, self.linear_num_value_heads, 1, self.linear_value_head_dim])

        query = ttnn.multiply(self._linear_l2_norm(query), 1.0 / math.sqrt(self.linear_key_head_dim))
        key = self._linear_l2_norm(key)
        beta = ttnn.reshape(beta, [batch, self.linear_num_value_heads, 1, 1])
        a = ttnn.reshape(a, [batch, self.linear_num_value_heads, 1, 1])
        a = ttnn.typecast(a, ttnn.float32)
        decay = ttnn.exp(ttnn.multiply(self.a_decay, ttnn.softplus(ttnn.add(a, self.dt_bias))))
        decayed_state = ttnn.multiply(recurrent_state, decay)
        read_config = self._gdn_read_program_config()
        recalled = ttnn.matmul(
            key,
            decayed_state,
            dtype=ttnn.float32,
            program_config=read_config,
        )
        self._record_fusion("decode_beta_sigmoid_multiply")
        delta = ttnn.multiply(
            ttnn.subtract(value, recalled),
            beta,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
        )
        self._record_fusion("matmul_transpose_flags")
        update = ttnn.matmul(
            key,
            delta,
            transpose_a=True,
            dtype=ttnn.float32,
            program_config=self._gdn_update_program_config(),
        )
        new_recurrent_state = ttnn.add(decayed_state, update)
        ttnn.copy(new_recurrent_state, recurrent_state)
        core = ttnn.matmul(
            query,
            new_recurrent_state,
            dtype=ttnn.bfloat16,
            program_config=read_config,
        )

        core = ttnn.reshape(core, [batch, 1, 1, self.linear_num_value_heads, self.linear_value_head_dim])
        z = ttnn.reshape(z, [batch, 1, 1, self.linear_num_value_heads, self.linear_value_head_dim])
        norm = ttnn.rsqrt(
            ttnn.add(
                ttnn.mean(ttnn.multiply(core, core), dim=-1, keepdim=True),
                self.rms_norm_eps,
            )
        )
        core = ttnn.multiply(ttnn.multiply(core, norm), self.linear_norm)
        self._record_fusion("linear_silu_multiply")
        core = ttnn.multiply(core, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        core = ttnn.reshape(core, [batch, 1, 1, self.linear_num_value_heads * self.linear_value_head_dim])
        return self._dram_projection(
            core,
            self.linear_out_proj,
            self.linear_out_proj_dram_decode,
            role="linear_output",
        )

    def _full_qkv_prefill(self, hidden_states, cos, sin):
        batch, _, seq_len, _ = hidden_states.shape
        self._record_optimization("low_precision_projections")
        self._record_fusion("shared_lhs_full_qkv")
        packed = ttnn.linear(
            hidden_states,
            self.full_qkv,
            dtype=ttnn.bfloat16,
            program_config=self._prefill_program_config(
                hidden_states.shape[0]
                * hidden_states.shape[1]
                * hidden_states.shape[2],
                self.hidden_size,
                self.full_qkv.shape[-1],
            ),
        )
        qkv_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        qkv = _slice_last(packed, 0, qkv_width)
        gate = _slice_last(packed, qkv_width, packed.shape[-1])
        qkv = ttnn.reshape(qkv, [batch, seq_len, qkv_width])
        self._record_fusion("dedicated_full_qkv_heads")
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
        )
        query = ttnn.rms_norm(query, epsilon=self.rms_norm_eps, weight=self.q_norm)
        key = ttnn.rms_norm(key, epsilon=self.rms_norm_eps, weight=self.k_norm)
        return self._partial_rope(query, cos, sin), self._partial_rope(key, cos, sin), value, gate

    def _full_qkv_decode(self, hidden_states, cos, sin):
        _, _, padded_batch, _ = hidden_states.shape
        self._record_optimization("low_precision_projections")
        self._record_fusion("shared_lhs_full_qkv")
        packed = self._dram_projection(
            hidden_states,
            self.full_qkv,
            self.full_qkv_dram_decode,
            role="full_input",
        )
        qkv_width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        qkv = _slice_last(packed, 0, qkv_width)
        gate = _slice_last(packed, qkv_width, packed.shape[-1])
        qkv = ttnn.to_memory_config(qkv, ttnn.L1_MEMORY_CONFIG)
        batch_grid = ttnn.num_cores_to_corerangeset(padded_batch, ttnn.CoreCoord(8, 8), row_wise=True)
        decode_head_memcfg = ttnn.create_sharded_memory_config(
            shape=(32, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._record_fusion("dedicated_full_qkv_heads")
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            overlap_qk_coregrid=True,
            memory_config=decode_head_memcfg,
        )
        if self.sharded_qk_norm_rope:
            self._record_optimization("sharded_qk_norm_rope")
            norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[8, padded_batch // 8],
                subblock_w=1,
                block_h=1,
                block_w=self.head_dim // 32,
                inplace=False,
            )
            query = ttnn.rms_norm(
                query,
                epsilon=self.rms_norm_eps,
                weight=self.q_norm,
                memory_config=decode_head_memcfg,
                program_config=norm_program,
            )
            key = ttnn.rms_norm(
                key,
                epsilon=self.rms_norm_eps,
                weight=self.k_norm,
                memory_config=decode_head_memcfg,
                program_config=norm_program,
            )
            query = _rotate_half_partial(query, cos, sin, self.rotary_dim)
            key = _rotate_half_partial(key, cos, sin, self.rotary_dim)
            return query, key, value, gate
        query = ttnn.to_memory_config(query, ttnn.L1_MEMORY_CONFIG)
        key = ttnn.to_memory_config(key, ttnn.L1_MEMORY_CONFIG)
        query = ttnn.rms_norm(query, epsilon=self.rms_norm_eps, weight=self.q_norm)
        key = ttnn.rms_norm(key, epsilon=self.rms_norm_eps, weight=self.k_norm)
        query = _rotate_half_partial(query, cos, sin, self.rotary_dim)
        key = _rotate_half_partial(key, cos, sin, self.rotary_dim)
        return query, key, value, gate
