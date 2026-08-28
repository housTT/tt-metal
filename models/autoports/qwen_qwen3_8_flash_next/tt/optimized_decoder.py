# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-device optimized decoder for ``Qwen/Qwen3.8-Flash-Next``.

The class deliberately inherits the completed fused graph: optimization must
preserve that graph's public prefill/decode, paging, state, and non-aligned
length contracts.  Setup selects a named precision/fidelity and sparse-matmul
geometry policy; measured entry points always dispatch through this class.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import ttnn

from .functional_decoder import _embedding_tiled_output, _free, _hifi4, _shape
from .fused_decoder import FusedDecoder


def _lofi():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _hifi2():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@dataclass(frozen=True)
class OptimizationPolicy:
    name: str
    expert_weight_dtype: object
    expert_fidelity: str
    gate_up_cores: int
    gate_up_in0_block_w: int
    down_cores: int
    down_in0_block_w: int


POLICIES = {
    # Same precision and material sparse geometry as FusedDecoder.  This is a
    # control inside the optimized class, not a functional/fused fallback.
    "fused_precision": OptimizationPolicy("fused_precision", ttnn.bfloat16, "hifi4", 40, 8, 40, 5),
    "expert_bfp8_hifi2_g40d40": OptimizationPolicy("expert_bfp8_hifi2_g40d40", ttnn.bfloat8_b, "hifi2", 40, 8, 40, 5),
    "expert_bfp8_lofi_g40d40": OptimizationPolicy("expert_bfp8_lofi_g40d40", ttnn.bfloat8_b, "lofi", 40, 8, 40, 5),
    "expert_bfp4_lofi_g40d40": OptimizationPolicy("expert_bfp4_lofi_g40d40", ttnn.bfloat4_b, "lofi", 40, 8, 40, 5),
    "expert_bfp4_lofi_legacy": OptimizationPolicy("expert_bfp4_lofi_legacy", ttnn.bfloat4_b, "lofi", 40, 8, 40, 5),
    "expert_bfp4_lofi_g20d40": OptimizationPolicy("expert_bfp4_lofi_g20d40", ttnn.bfloat4_b, "lofi", 20, 16, 40, 10),
    "expert_bfp4_lofi_g20b8_d40b5": OptimizationPolicy(
        "expert_bfp4_lofi_g20b8_d40b5", ttnn.bfloat4_b, "lofi", 20, 8, 40, 5
    ),
    "expert_bfp4_lofi_g20b16_d40b5": OptimizationPolicy(
        "expert_bfp4_lofi_g20b16_d40b5", ttnn.bfloat4_b, "lofi", 20, 16, 40, 5
    ),
    "expert_bfp4_lofi_g40b16_d40b5": OptimizationPolicy(
        "expert_bfp4_lofi_g40b16_d40b5", ttnn.bfloat4_b, "lofi", 40, 16, 40, 5
    ),
    "expert_bfp4_lofi_g40b8_d40b10": OptimizationPolicy(
        "expert_bfp4_lofi_g40b8_d40b10", ttnn.bfloat4_b, "lofi", 40, 8, 40, 10
    ),
    "expert_bfp4_lofi_g40d80": OptimizationPolicy("expert_bfp4_lofi_g40d80", ttnn.bfloat4_b, "lofi", 40, 16, 80, 10),
    "expert_bfp4_lofi_g20d80": OptimizationPolicy("expert_bfp4_lofi_g20d80", ttnn.bfloat4_b, "lofi", 20, 16, 80, 10),
}

PROJECTION_POLICIES = {
    "bf16_hifi4": (ttnn.bfloat16, "hifi4"),
    "bf16_hifi2": (ttnn.bfloat16, "hifi2"),
    "bfp8_hifi2": (ttnn.bfloat8_b, "hifi2"),
    "bfp8_lofi": (ttnn.bfloat8_b, "lofi"),
    "bfp4_lofi": (ttnn.bfloat4_b, "lofi"),
}


class OptimizedDecoder(FusedDecoder):
    """Fused decoder with a measured single-device optimization policy."""

    DEFAULT_POLICY = "expert_bfp4_lofi_g40b16_d40b5"
    DEFAULT_SHARED_POLICY = "bfp8_lofi"
    DEFAULT_GDN_POLICY = "bfp8_hifi2"
    DEFAULT_ATTENTION_POLICY = "bf16_hifi2"
    DEFAULT_CACHE_POLICY = "bf16"
    DEFAULT_EXPERT_TOPOLOGY = "packed"
    DEFAULT_DECODE_EXPERT_MODE = "indexed"
    DEFAULT_SEPARATE_EXPERT_CORES = 20
    DEFAULT_SEPARATE_EXPERT_IN0_BLOCK_W = 16
    DEFAULT_DECODE_1D_CONFIG = (
        "in_proj_z:48,gdn_out:20,"
        "attn_hc_down_inject:10,mlp_hc_down_inject:10,"
        "attn_hc_up:80,mlp_hc_up:80,moe_input:40,shared_down_proj:40,"
        "qsa_input:110,attn_out:20,ple_key_value:100"
    )
    DEFAULT_DECODE_OUTPUT = "l1"
    DEFAULT_PREFILL_CONFIG = (
        "attn_hc_down_inject:11x4x8,mlp_hc_down_inject:11x4x8,"
        "gdn_out:10x4x8,qsa_input:11x4x2,attn_out:10x4x8,"
        "ple_key_value:10x4x4"
    )
    DEFAULT_PREFILL_OUTPUT = "l1"
    DEFAULT_PREFILL_SDPA_CONFIG = "11x10x32x64"
    DEFAULT_DECODE_SDPA_CONFIG = "11x10x0x32"
    DEFAULT_DRAM_SHARDED_ROLES = ("qsa_input", "attn_out")
    OPTIMIZATION_MANIFEST = (
        "named_precision_fidelity_policy",
        "bfp4_lofi_routed_experts",
        "precision_locked_sparse_geometry",
        "explicit_sparse_program_config",
        "explicit_prefill_2d_program_configs",
        "explicit_decode_1d_program_configs",
        "prefill_l1_intermediate_outputs",
        "decode_l1_intermediate_outputs",
        "packed_same_input_projections",
        "tuned_native_sdpa_prefill_and_decode",
        "dram_sharded_decode_attention_projections",
        "indexed_batch_one_active_expert_execution",
        "dynamic_union_prefill_and_batched_expert_execution",
        "fused_graph_contract_preserved",
        "traced_decode_only_performance_evidence",
    )

    @classmethod
    def from_state_dict(cls, *args, **kwargs) -> "OptimizedDecoder":
        policy_name = kwargs.pop(
            "optimization_policy",
            os.environ.get("QWEN38_OPT_POLICY", cls.DEFAULT_POLICY),
        )
        try:
            policy = POLICIES[policy_name]
        except KeyError as exc:
            raise ValueError(f"unknown Qwen3.8 optimized policy {policy_name!r}") from exc

        attention_policy_name = kwargs.pop(
            "attention_projection_policy",
            os.environ.get("QWEN38_OPT_ATTENTION_POLICY", cls.DEFAULT_ATTENTION_POLICY),
        )
        group_policy_names = {
            "shared": kwargs.pop(
                "shared_projection_policy",
                os.environ.get("QWEN38_OPT_SHARED_POLICY", cls.DEFAULT_SHARED_POLICY),
            ),
            "gdn": kwargs.pop(
                "gdn_projection_policy",
                os.environ.get("QWEN38_OPT_GDN_POLICY", cls.DEFAULT_GDN_POLICY),
            ),
            "qsa_input": kwargs.pop(
                "qsa_input_policy",
                os.environ.get("QWEN38_OPT_QSA_INPUT_POLICY", attention_policy_name),
            ),
            "attention_output": kwargs.pop(
                "attention_output_policy",
                os.environ.get("QWEN38_OPT_ATTENTION_OUTPUT_POLICY", attention_policy_name),
            ),
        }
        unknown = set(group_policy_names.values()) - set(PROJECTION_POLICIES)
        if unknown:
            raise ValueError(f"unknown projection policy value(s): {sorted(unknown)}")
        cache_policy = kwargs.pop(
            "cache_policy",
            os.environ.get("QWEN38_OPT_CACHE_POLICY", cls.DEFAULT_CACHE_POLICY),
        )
        if cache_policy not in {"bf16", "bfp8"}:
            raise ValueError(f"unknown cache policy {cache_policy!r}")
        decode_1d_spec = kwargs.pop(
            "decode_1d_config",
            os.environ.get("QWEN38_OPT_1D_CONFIG", cls.DEFAULT_DECODE_1D_CONFIG),
        )
        prefill_spec = kwargs.pop(
            "prefill_config",
            os.environ.get("QWEN38_OPT_PREFILL_CONFIG", cls.DEFAULT_PREFILL_CONFIG),
        )
        default_dram_sharded_role = ",".join(cls.DEFAULT_DRAM_SHARDED_ROLES) if kwargs.get("layer_idx") == 3 else ""
        dram_sharded_role = kwargs.pop(
            "dram_sharded_role",
            os.environ.get("QWEN38_OPT_DRAM_SHARDED_ROLE", default_dram_sharded_role),
        )
        dram_sharded_roles = tuple(role.strip() for role in dram_sharded_role.split(",") if role.strip())
        if len(dram_sharded_roles) != len(set(dram_sharded_roles)):
            raise ValueError(f"duplicate DRAM-sharded role in {dram_sharded_role!r}")
        expert_topology = kwargs.pop(
            "expert_topology",
            os.environ.get("QWEN38_OPT_EXPERT_TOPOLOGY", cls.DEFAULT_EXPERT_TOPOLOGY),
        )
        if expert_topology not in {"packed", "separate"}:
            raise ValueError(f"unknown expert topology {expert_topology!r}")

        # Expert dtype is materialized directly from real checkpoint tensors;
        # no runtime cast or stale BF16 tensor can bypass the policy.
        kwargs["expert_weight_dtype"] = policy.expert_weight_dtype
        kwargs["cache_dtype"] = ttnn.bfloat8_b if cache_policy == "bfp8" else ttnn.bfloat16
        layer = super().from_state_dict(*args, **kwargs)
        layer.optimization_policy = policy
        layer.expert_topology = expert_topology
        layer.decode_expert_mode = os.environ.get("QWEN38_OPT_DECODE_EXPERT_MODE", cls.DEFAULT_DECODE_EXPERT_MODE)
        if layer.decode_expert_mode not in {"scan", "indexed"}:
            raise ValueError(f"unknown decode expert mode {layer.decode_expert_mode!r}")
        layer._decode_expert_indices = None
        layer._decode_expert_weights = None
        layer.projection_policy_names = group_policy_names
        layer.cache_policy = cache_policy
        layer.decode_1d_cores = cls._parse_decode_1d_config(decode_1d_spec, layer_idx=kwargs.get("layer_idx"))
        layer.prefill_configs = cls._parse_prefill_config(prefill_spec)
        layer.prefill_sdpa_config = cls._parse_sdpa_config(
            os.environ.get("QWEN38_OPT_PREFILL_SDPA_CONFIG", cls.DEFAULT_PREFILL_SDPA_CONFIG),
            decode=False,
        )
        layer.decode_sdpa_config = cls._parse_sdpa_config(
            os.environ.get("QWEN38_OPT_DECODE_SDPA_CONFIG", cls.DEFAULT_DECODE_SDPA_CONFIG),
            decode=True,
        )
        layer.prefill_output = os.environ.get("QWEN38_OPT_PREFILL_OUTPUT", cls.DEFAULT_PREFILL_OUTPUT)
        if layer.prefill_output not in {"dram", "l1"}:
            raise ValueError(f"unknown prefill output policy {layer.prefill_output!r}")
        layer.dram_sharded_role = ",".join(dram_sharded_roles)
        layer.dram_sharded_roles = frozenset(dram_sharded_roles)
        layer.dram_activation_config_by_role = {}
        layer.dram_sharded_cores_by_role = {}
        layer.dram_sharded_weight_by_role = {}
        layer.decode_1d_output = os.environ.get("QWEN38_OPT_1D_OUTPUT", cls.DEFAULT_DECODE_OUTPUT)
        if layer.decode_1d_output not in {"dram", "l1"}:
            raise ValueError(f"unknown decode 1D output policy {layer.decode_1d_output!r}")
        layer._decode_active = False

        # The packed GDN qkv/beta/decay width is 323 tiles.  A legal wide
        # rectangular 1D grid cannot divide that prime-factor tile count.  If
        # this role is explicitly configured, pad the setup-time weight/bias
        # output width to a whole number of cores and slice the logical 10336
        # values in _gdn_inputs.  This avoids the per-core tail reordering that
        # made the unpadded material candidate numerically invalid.
        gdn_qkv_cores = layer.decode_1d_cores.get("gdn_qkv_b_a")
        if gdn_qkv_cores is not None and "gdn_qkv_b_a" in layer.w:
            weight = layer.w["gdn_qkv_b_a"]
            output_width = int(weight.shape[-1])
            padded_width = math.ceil(output_width / (32 * gdn_qkv_cores)) * 32 * gdn_qkv_cores
            if padded_width != output_width:
                output_padding = [(0, 0)] * (len(weight.shape) - 1) + [(0, padded_width - output_width)]
                padded_weight = ttnn.pad(weight, output_padding, 0.0)
                _free(weight, padded_weight)
                layer.w["gdn_qkv_b_a"] = padded_weight
                bias = layer.w["gdn_qkv_b_a_bias"]
                output_padding = [(0, 0)] * (len(bias.shape) - 1) + [(0, padded_width - output_width)]
                padded_bias = ttnn.pad(bias, output_padding, 0.0)
                _free(bias, padded_bias)
                layer.w["gdn_qkv_b_a_bias"] = padded_bias
        layer.expert_compute_cfg = {
            "hifi4": _hifi4(fp32=True),
            "hifi2": _hifi2(),
            "lofi": _lofi(),
        }[policy.expert_fidelity]
        layer.projection_compute_cfgs = {
            group: {
                "hifi4": _hifi4(fp32=True),
                "hifi2": _hifi2(),
                "lofi": _lofi(),
            }[PROJECTION_POLICIES[name][1]]
            for group, name in group_policy_names.items()
        }

        role_names = {
            "shared": (
                "attn_hc_down_inject",
                "attn_hc_up",
                "mlp_hc_down_inject",
                "mlp_hc_up",
                "moe_input",
                "shared_down_proj",
                "ple_key_value",
            ),
            "gdn": ("gdn_qkv_b_a", "in_proj_z", "gdn_out"),
            "qsa_input": ("qsa_input",),
            "attention_output": ("attn_out",),
        }
        layer.weight_group_by_id = {}
        layer.weight_role_by_id = {}
        for group, names in role_names.items():
            target_dtype = PROJECTION_POLICIES[group_policy_names[group]][0]
            for name in names:
                if name not in layer.w:
                    continue
                value = layer.w[name]
                if value.dtype != target_dtype:
                    converted = ttnn.typecast(value, target_dtype)
                    _free(value, converted)
                    layer.w[name] = converted
                    value = converted
                if name in layer.dram_sharded_roles:
                    dram_value = ttnn.to_memory_config(value, cls._dram_weight_memory_config(value))
                    layer.dram_sharded_weight_by_role[name] = dram_value
                    activation_config, sharded_cores = cls._dram_activation_memory_config(int(value.shape[-2]))
                    layer.dram_activation_config_by_role[name] = activation_config
                    layer.dram_sharded_cores_by_role[name] = sharded_cores
                layer.weight_group_by_id[id(value)] = group
                layer.weight_role_by_id[id(value)] = name
        missing_dram_roles = layer.dram_sharded_roles - layer.dram_sharded_weight_by_role.keys()
        if missing_dram_roles:
            raise ValueError(f"unknown or unavailable DRAM-sharded role(s) {sorted(missing_dram_roles)!r}")
        if expert_topology == "separate":
            intermediate = layer.shapes.moe_intermediate_size
            layer.optimized_expert_gate = layer._slice_last(layer.expert_gate_up, 0, intermediate)
            layer.optimized_expert_up = layer._slice_last(layer.expert_gate_up, intermediate, 2 * intermediate)
            ttnn.deallocate(layer.expert_gate_up)
            layer.expert_gate_up = None
        return layer

    @staticmethod
    def _parse_decode_1d_config(spec: str, *, layer_idx: int | None = None) -> dict[str, int]:
        parsed = {}
        if not spec:
            return parsed
        for item in spec.split(","):
            try:
                role, cores = item.split(":", 1)
                cores = int(cores)
            except ValueError as exc:
                raise ValueError(f"invalid QWEN38_OPT_1D_CONFIG item {item!r}") from exc
            if cores <= 0:
                raise ValueError(f"decode 1D core count must be positive, got {cores}")
            if "@" in role:
                role, layer_scope = role.split("@", 1)
                try:
                    selected_layers = {int(value) for value in layer_scope.split("+")}
                except ValueError as exc:
                    raise ValueError(f"invalid layer scope in QWEN38_OPT_1D_CONFIG item {item!r}") from exc
                if layer_idx is not None and layer_idx not in selected_layers:
                    continue
            parsed[role] = cores
        return parsed

    @staticmethod
    def _parse_prefill_config(spec: str) -> dict[str, tuple[int, int, int]]:
        parsed = {}
        if not spec:
            return parsed
        for item in spec.split(","):
            try:
                role, dimensions = item.split(":", 1)
                grid_x, grid_y, in0_block_w = (int(value) for value in dimensions.split("x"))
            except ValueError as exc:
                raise ValueError(f"invalid QWEN38_OPT_PREFILL_CONFIG item {item!r}") from exc
            if not 1 <= grid_x <= 11 or not 1 <= grid_y <= 10 or in0_block_w <= 0:
                raise ValueError(f"illegal prefill program dimensions in {item!r}")
            parsed[role] = (grid_x, grid_y, in0_block_w)
        return parsed

    @staticmethod
    def _parse_sdpa_config(spec: str, *, decode: bool) -> tuple[int, int, int, int]:
        try:
            grid_x, grid_y, q_chunk_size, k_chunk_size = (int(value) for value in spec.split("x"))
        except ValueError as exc:
            raise ValueError(f"invalid Qwen3.8 SDPA config {spec!r}") from exc
        if not 1 <= grid_x <= 11 or not 1 <= grid_y <= 10:
            raise ValueError(f"illegal Qwen3.8 SDPA grid {grid_x}x{grid_y}")
        if (decode and q_chunk_size != 0) or (not decode and q_chunk_size <= 0) or k_chunk_size <= 0:
            raise ValueError(f"illegal Qwen3.8 {'decode' if decode else 'prefill'} SDPA chunks in {spec!r}")
        if (not decode and q_chunk_size % 32) or k_chunk_size % 32:
            raise ValueError(f"Qwen3.8 SDPA chunks must be tile aligned in {spec!r}")
        return grid_x, grid_y, q_chunk_size, k_chunk_size

    @staticmethod
    def _largest_divisor(value: int, limit: int = 8) -> int:
        return next(divisor for divisor in range(limit, 0, -1) if value % divisor == 0)

    @staticmethod
    def _dram_weight_memory_config(weight):
        dram_cores = 8
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        padded_n = math.ceil(n / (32 * dram_cores)) * 32 * dram_cores
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR),
        )

    @staticmethod
    def _dram_activation_memory_config(k: int):
        k_tiles = k // 32
        legal = []
        for cores in range(1, 65):
            if k_tiles % cores:
                continue
            for y in range(1, 9):
                if cores % y == 0 and cores // y <= 8:
                    legal.append((abs(cores - 32), cores, cores // y, y))
                    break
        _, cores, grid_x, grid_y = min(legal)
        return (
            ttnn.create_sharded_memory_config(
                shape=(32, k // cores),
                core_grid=ttnn.CoreGrid(x=grid_x, y=grid_y),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            cores,
        )

    def _dram_sharded_program_config(self, weight, role):
        k_tiles = math.ceil(int(weight.shape[-2]) / 32)
        n = int(weight.shape[-1])
        n_tiles = math.ceil(n / (32 * 8)) * 8
        sharded_cores = self.dram_sharded_cores_by_role[role]
        k_tiles_per_core = k_tiles // sharded_cores
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self._largest_divisor(k_tiles_per_core),
            per_core_M=1,
            per_core_N=math.ceil(n_tiles / sharded_cores),
            fused_activation=None,
        )

    def _decode_1d_program_config(self, weight, cores: int):
        k_tiles = math.ceil(int(weight.shape[-2]) / 32)
        n_tiles = math.ceil(int(weight.shape[-1]) / 32)
        grid = self._rectangular_grid(cores)
        per_core_n = math.ceil(n_tiles / cores)
        out_subblock_w = next(width for width in range(min(4, per_core_n), 0, -1) if per_core_n % width == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=self._largest_divisor(k_tiles),
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _prefill_program_config(self, x, weight, config):
        grid_x, grid_y, in0_block_w = config
        m_tiles = math.ceil(int(x.shape[-2]) / 32)
        k_tiles = math.ceil(int(weight.shape[-2]) / 32)
        n_tiles = math.ceil(int(weight.shape[-1]) / 32)
        if k_tiles % in0_block_w:
            raise ValueError(f"prefill in0_block_w={in0_block_w} does not divide Kt={k_tiles}")
        per_core_m = max(1, math.ceil(m_tiles / grid_y))
        per_core_n = max(1, math.ceil(n_tiles / grid_x))
        out_subblock_w = next(width for width in range(min(4, per_core_n), 0, -1) if per_core_n % width == 0)
        out_subblock_h = next(
            height for height in range(min(4 // out_subblock_w, per_core_m), 0, -1) if per_core_m % height == 0
        )
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def _linear_impl(self, x, weight, *, dtype=ttnn.bfloat16, bias=None):
        group = self.weight_group_by_id.get(id(weight))
        compute_cfg = self.projection_compute_cfgs[group] if group is not None else self.compute_cfg
        role = self.weight_role_by_id.get(id(weight))
        prefill_config = self.prefill_configs.get(role) if not self._decode_active else None
        if prefill_config is not None:
            return ttnn.linear(
                x,
                weight,
                bias=bias,
                dtype=dtype,
                compute_kernel_config=compute_cfg,
                program_config=self._prefill_program_config(x, weight, prefill_config),
                memory_config=(ttnn.L1_MEMORY_CONFIG if self.prefill_output == "l1" else ttnn.DRAM_MEMORY_CONFIG),
            )
        if self._decode_active and role in self.dram_sharded_roles:
            dram_weight = self.dram_sharded_weight_by_role[role]
            activation_config = self.dram_activation_config_by_role[role]
            already_sharded = x.memory_config() == activation_config
            x_sharded = x if already_sharded else ttnn.to_memory_config(x, activation_config)
            out_sharded = ttnn.linear(
                x_sharded,
                dram_weight,
                bias=bias,
                dtype=dtype,
                compute_kernel_config=compute_cfg,
                program_config=self._dram_sharded_program_config(dram_weight, role),
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            if not already_sharded:
                ttnn.deallocate(x_sharded)
            target_memory = ttnn.L1_MEMORY_CONFIG if self.decode_1d_output == "l1" else ttnn.DRAM_MEMORY_CONFIG
            out = ttnn.to_memory_config(out_sharded, target_memory)
            _free(out_sharded, out)
            return out
        cores = self.decode_1d_cores.get(role) if self._decode_active else None
        if cores is None:
            return ttnn.linear(x, weight, bias=bias, dtype=dtype, compute_kernel_config=compute_cfg)
        converted_input = x.memory_config() != ttnn.L1_MEMORY_CONFIG
        x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG) if converted_input else x
        out = ttnn.linear(
            x_l1,
            weight,
            bias=bias,
            dtype=dtype,
            compute_kernel_config=compute_cfg,
            program_config=self._decode_1d_program_config(weight, cores),
            memory_config=(ttnn.L1_MEMORY_CONFIG if self.decode_1d_output == "l1" else ttnn.DRAM_MEMORY_CONFIG),
        )
        if converted_input:
            ttnn.deallocate(x_l1)
        return out

    def _linear(self, x, weight, *, dtype=ttnn.bfloat16):
        return self._linear_impl(x, weight, dtype=dtype)

    def _gdn_inputs(self, x):
        # The fused implementation uses a direct biased linear for this packed
        # FP32 projection.  Route it through the same measured program-policy
        # hook without changing the projection's slicing or nonlinear math.
        s = self.shapes
        packed = self._linear_impl(
            x,
            self.w["gdn_qkv_b_a"],
            bias=self.w["gdn_qkv_b_a_bias"],
            dtype=ttnn.float32,
        )
        mixed = self._slice_last(packed, 0, s.linear_qkv_width)
        b_start = s.linear_qkv_width
        b = self._slice_last(packed, b_start, b_start + s.linear_num_value_heads)
        a = self._slice_last(
            packed,
            b_start + s.linear_num_value_heads,
            b_start + 2 * s.linear_num_value_heads,
        )
        ttnn.deallocate(packed)
        z = self._linear(x, self.w["in_proj_z"])
        beta = ttnn.sigmoid(b)
        ttnn.deallocate(b)
        soft = ttnn.softplus(a, beta=1.0, threshold=20.0)
        ttnn.deallocate(a)
        g = ttnn.multiply(soft, self.w["neg_exp_A"])
        ttnn.deallocate(soft)
        return mixed, z, beta, g

    def _gathered_qsa_attention(self, q, selected, valid, page_table):
        """Preserve fused paging/gather semantics with a measured SDPA policy."""

        s = self.shapes
        batch, _, tokens, _ = _shape(q)
        width = self.const["gathered_width"]
        physical = self._virtual_to_physical_tokens(selected, page_table)
        ttnn.deallocate(selected)
        physical = ttnn.reshape(physical, (1, batch * tokens * width))

        gathered = []
        for cache in self.kv_cache:
            gathered_heads = []
            cache_shape = _shape(cache)
            for head in range(s.num_key_value_heads):
                cache_head = ttnn.slice(
                    cache,
                    [0, head, 0, 0],
                    [cache_shape[0], head + 1, cache_shape[2], cache_shape[3]],
                )
                flattened = ttnn.reshape(cache_head, (self.max_num_blocks * self.block_size, s.head_dim))
                _free(cache_head, cache, flattened)
                value = _embedding_tiled_output(physical, flattened)
                _free(flattened, cache, value)
                expected_volume = batch * tokens * width * s.head_dim
                if math.prod(_shape(value)) != expected_volume:
                    raise RuntimeError(
                        f"paged cache embedding returned {_shape(value)} (volume {math.prod(_shape(value))}); "
                        f"expected logical volume {expected_volume} for "
                        f"[{batch}, {tokens}, 1, {width}, {s.head_dim}]"
                    )
                head_rows = ttnn.reshape(value, (batch, tokens, 1, width, s.head_dim))
                _free(value, head_rows)
                gathered_heads.append(head_rows)
            gathered_value = ttnn.concat(gathered_heads, dim=2)
            for head_rows in gathered_heads:
                _free(head_rows, gathered_value)
            gathered.append(gathered_value)
        ttnn.deallocate(physical)
        k, v = gathered

        query_batches = batch * tokens
        if tokens == 1:
            q_rows = ttnn.reshape(q, (query_batches, s.num_attention_heads, 1, s.head_dim))
        else:
            query_tokens = ttnn.permute(q, (0, 2, 1, 3))
            q_rows = ttnn.reshape(query_tokens, (query_batches, s.num_attention_heads, 1, s.head_dim))
            _free(query_tokens, q, q_rows)
        k_rows = ttnn.reshape(k, (query_batches, s.num_key_value_heads, width, s.head_dim))
        v_rows = ttnn.reshape(v, (query_batches, s.num_key_value_heads, width, s.head_dim))
        additive = ttnn.where(valid, 0.0, -1.0e4)
        ttnn.deallocate(valid)
        if tokens > 1:
            token_mask = ttnn.permute(additive, (0, 2, 1, 3))
            ttnn.deallocate(additive)
            additive = ttnn.reshape(token_mask, (query_batches, 1, 1, width))
            _free(token_mask, additive)

        grid_x, grid_y, q_chunk_size, k_chunk_size = (
            self.decode_sdpa_config if tokens == 1 else self.prefill_sdpa_config
        )
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )
        if tokens == 1:
            decode_q = ttnn.permute(q_rows, (2, 0, 1, 3))
            decode_mask = ttnn.repeat(additive, (1, 1, s.num_attention_heads, 1))
            out = ttnn.transformer.scaled_dot_product_attention_decode(
                decode_q,
                k_rows,
                v_rows,
                is_causal=False,
                attn_mask=decode_mask,
                scale=1.0 / math.sqrt(s.head_dim),
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_cfg,
            )
            ttnn.deallocate(decode_q)
            ttnn.deallocate(decode_mask)
        else:
            out = ttnn.transformer.scaled_dot_product_attention(
                q_rows,
                k_rows,
                v_rows,
                is_causal=False,
                attn_mask=additive,
                scale=1.0 / math.sqrt(s.head_dim),
                program_config=sdpa_cfg,
                compute_kernel_config=self.compute_cfg,
            )
        _free(q_rows, q, out)
        ttnn.deallocate(k_rows)
        ttnn.deallocate(v_rows)
        ttnn.deallocate(additive)
        token_out = ttnn.reshape(out, (batch, tokens, s.num_attention_heads, s.head_dim))
        _free(out, token_out)
        return token_out

    def prefill_forward(self, *args, **kwargs):
        self._decode_active = False
        return super().prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        self._decode_active = True
        try:
            return super().decode_forward(*args, **kwargs)
        finally:
            self._decode_active = False

    @staticmethod
    def _rectangular_grid(cores: int) -> ttnn.CoreCoord:
        # P300c exposes an 11x10 worker grid.  Prefer a wide rectangle to
        # shorten multicast distance while retaining exactly ``cores`` cores.
        for x in range(11, 0, -1):
            if cores % x == 0 and cores // x <= 10:
                return ttnn.CoreCoord(x, cores // x)
        raise ValueError(f"{cores} cores do not form a legal P300c rectangle")

    def _sparse_matmul_config(self, m: int, n: int, k: int):
        policy = self.optimization_policy
        if policy.name in {"fused_precision", "expert_bfp4_lofi_legacy"}:
            return FusedDecoder._sparse_matmul_config(m, n, k)
        is_gate_up = n == 2 * self.shapes.moe_intermediate_size
        cores = policy.gate_up_cores if is_gate_up else policy.down_cores
        in0_block_w = policy.gate_up_in0_block_w if is_gate_up else policy.down_in0_block_w
        k_tiles = (k + 31) // 32
        n_tiles = (n + 31) // 32
        if k_tiles % in0_block_w:
            raise ValueError(f"in0_block_w={in0_block_w} does not divide Kt={k_tiles}")
        if n_tiles % cores:
            raise ValueError(f"cores={cores} does not divide Nt={n_tiles}")
        per_core_n = n_tiles // cores
        out_subblock_w = next(width for width in range(min(4, per_core_n), 0, -1) if per_core_n % width == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self._rectangular_grid(cores),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _routed_experts(self, x, routing):
        # FusedDecoder uses ``self.compute_cfg`` for both sparse projections.
        # Scope the selected expert kernel config to this subgraph so sensitive
        # FP32 GDN recurrence and norms retain their proven HiFi contracts.
        if self._decode_active and self.max_batch == 1 and self.decode_expert_mode == "indexed":
            return self._routed_experts_indexed(x, routing)
        if self.expert_topology == "separate":
            return self._routed_experts_separate(x, routing)
        saved = self.compute_cfg
        self.compute_cfg = self.expert_compute_cfg
        try:
            return super()._routed_experts(x, routing)
        finally:
            self.compute_cfg = saved

    def _routing_from_logits(self, logits):
        """Retain compact device top-k metadata only for indexed batch-one decode."""

        s = self.shapes
        zeros = ttnn.zeros_like(logits)
        values, indices = ttnn.topk(logits, k=s.num_experts_per_tok, dim=-1, sorted=True)
        ttnn.deallocate(logits)
        selected_values = values
        values = ttnn.softmax(selected_values, dim=-1)
        _free(selected_values, values)
        indexed = self._decode_active and self.max_batch == 1 and self.decode_expert_mode == "indexed"
        if indexed:
            index_row = ttnn.slice(indices, [0, 0, 0, 0], [1, 1, 1, s.num_experts_per_tok])
            self._decode_expert_indices = ttnn.to_layout(index_row, ttnn.ROW_MAJOR_LAYOUT)
            _free(index_row, indices, self._decode_expert_indices)
            self._decode_expert_weights = ttnn.slice(values, [0, 0, 0, 0], [1, 1, 1, s.num_experts_per_tok])
        routing = ttnn.scatter(zeros, dim=-1, index=indices, src=values)
        ttnn.deallocate(zeros)
        ttnn.deallocate(values)
        ttnn.deallocate(indices)
        return routing

    def _routed_experts_indexed(self, x, routing):
        """Evaluate only the ten selected experts for batch-one decode."""

        s = self.shapes
        tokens = int(x.shape[-2])
        if tokens != 32:
            raise ValueError(f"indexed expert decode expects one tile, got {tokens} rows")
        if self._decode_expert_indices is None or self._decode_expert_weights is None:
            raise RuntimeError("indexed expert decode is missing device top-k metadata")
        grouped_x = ttnn.reshape(x, (1, 1, 32, s.hidden_size))
        routing_groups = ttnn.reshape(routing, (1, 1, 32, s.num_experts))
        sparsity = ttnn.max(routing_groups, dim=2, keepdim=True)
        sparsity = ttnn.reshape(sparsity, (1, 1, 1, s.num_experts))
        sparsity = ttnn.to_layout(sparsity, ttnn.ROW_MAJOR_LAYOUT)
        output_tile = ttnn.Tile([32, 32])

        if self.expert_topology == "packed":
            gate_up_sparse = ttnn.sparse_matmul(
                grouped_x,
                self.expert_gate_up,
                sparsity=sparsity,
                indices=self._decode_expert_indices,
                nnz=None,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                output_tile=output_tile,
                is_input_b_sparse=True,
                program_config=self._sparse_matmul_config(32, 2 * s.moe_intermediate_size, s.hidden_size),
                compute_kernel_config=self.expert_compute_cfg,
                dtype=ttnn.bfloat16,
            )
            _free(grouped_x, x, gate_up_sparse)
            gate_up = ttnn.reshape(
                gate_up_sparse,
                (1, s.num_experts_per_tok, 32, 2 * s.moe_intermediate_size),
            )
            _free(gate_up_sparse, gate_up)
            gate = self._slice_last(gate_up, 0, s.moe_intermediate_size)
            up = self._slice_last(gate_up, s.moe_intermediate_size, 2 * s.moe_intermediate_size)
            ttnn.deallocate(gate_up)
        else:
            projections = []
            for weight in (self.optimized_expert_gate, self.optimized_expert_up):
                sparse = ttnn.sparse_matmul(
                    grouped_x,
                    weight,
                    sparsity=sparsity,
                    indices=self._decode_expert_indices,
                    nnz=None,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    output_tile=output_tile,
                    is_input_b_sparse=True,
                    program_config=self._separate_expert_program_config(32, s.moe_intermediate_size, s.hidden_size),
                    compute_kernel_config=self.expert_compute_cfg,
                    dtype=ttnn.bfloat16,
                )
                projection = ttnn.reshape(sparse, (1, s.num_experts_per_tok, 32, s.moe_intermediate_size))
                _free(sparse, projection)
                projections.append(projection)
            _free(grouped_x, x, *projections)
            gate, up = projections
        hidden = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        selected_weights = ttnn.permute(self._decode_expert_weights, (0, 3, 2, 1))
        weighted_hidden = ttnn.multiply(hidden, selected_weights)
        ttnn.deallocate(hidden)
        _free(selected_weights, self._decode_expert_weights, weighted_hidden)

        down = ttnn.sparse_matmul(
            weighted_hidden,
            self.experts.down,
            sparsity=sparsity,
            indices=self._decode_expert_indices,
            nnz=None,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _free(weighted_hidden, down)
        ttnn.deallocate(sparsity)
        ttnn.deallocate(self._decode_expert_indices)
        ttnn.deallocate(self._decode_expert_weights)
        self._decode_expert_indices = None
        self._decode_expert_weights = None
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))

    def _separate_expert_program_config(self, m: int, n: int, k: int):
        cores = int(os.environ.get("QWEN38_OPT_SEPARATE_EXPERT_CORES", str(self.DEFAULT_SEPARATE_EXPERT_CORES)))
        in0_block_w = int(
            os.environ.get("QWEN38_OPT_SEPARATE_EXPERT_IN0_BLOCK_W", str(self.DEFAULT_SEPARATE_EXPERT_IN0_BLOCK_W))
        )
        k_tiles = math.ceil(k / 32)
        n_tiles = math.ceil(n / 32)
        if k_tiles % in0_block_w or n_tiles % cores:
            raise ValueError(
                f"illegal separate expert geometry cores={cores}, in0_block_w={in0_block_w}, "
                f"Kt={k_tiles}, Nt={n_tiles}"
            )
        per_core_n = n_tiles // cores
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self._rectangular_grid(cores),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=per_core_n,
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=max(32, m) // 32,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    def _routed_experts_separate(self, x, routing):
        """Measured legal control: two BFP4/LoFi sparse gate/up matmuls."""

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
        gate_up_cfg = self._separate_expert_program_config(32, s.moe_intermediate_size, s.hidden_size)
        projections = []
        for weight in (self.optimized_expert_gate, self.optimized_expert_up):
            sparse = ttnn.sparse_matmul(
                grouped_x,
                weight,
                sparsity=sparsity,
                nnz=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
                program_config=gate_up_cfg,
                compute_kernel_config=self.expert_compute_cfg,
                dtype=ttnn.bfloat16,
            )
            projections.append(ttnn.reshape(sparse, (groups, s.num_experts, 32, s.moe_intermediate_size)))
            _free(sparse, projections[-1])
        _free(grouped_x, x)
        gate, up = projections
        hidden = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        token_weights = ttnn.permute(routing_groups, (1, 3, 2, 0))
        weighted_hidden = ttnn.multiply(hidden, token_weights)
        ttnn.deallocate(hidden)
        _free(token_weights, routing, routing_groups, weighted_hidden)

        down = ttnn.sparse_matmul(
            weighted_hidden,
            self.experts.down,
            sparsity=sparsity,
            nnz=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tile=output_tile,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            program_config=self._sparse_matmul_config(32, s.hidden_size, s.moe_intermediate_size),
            compute_kernel_config=self.expert_compute_cfg,
            dtype=ttnn.bfloat16,
        )
        _free(weighted_hidden, down)
        ttnn.deallocate(sparsity)
        out = ttnn.experimental.fast_reduce_nc(down, dims=[1])
        ttnn.deallocate(down)
        return ttnn.reshape(ttnn.unsqueeze_to_4D(out), (1, 1, tokens, s.hidden_size))


__all__ = ["OptimizedDecoder", "OptimizationPolicy", "POLICIES", "PROJECTION_POLICIES"]
