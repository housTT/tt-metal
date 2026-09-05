# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer experiments for the optimized TP4 decoder stage."""

import math
from dataclasses import replace

import ttnn

from ..tt.fused_decoder import _field
from ..tt.multichip_decoder import TP, MeshConfig, MultichipDecoder, _projection_weights, partition_state_dict
from ..tt.optimized_decoder import OptimizedDecoder


class PackedGDN(MultichipDecoder):
    """Pack QKV/A/B/Z from raw per-rank HF weights; preserve the Z SiLU."""

    fused_z = False
    z_operand = "rhs"
    packed_prefill = False
    compact_z = False

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        plan = kwargs.pop("mesh_config", None) or MeshConfig()
        # This historical experiment owns packing and needs the separate source
        # projections even after production packing becomes the default.
        kwargs["mesh_config"] = replace(plan, pack_gdn=False, pack_mlp_decode=False, decode_dram_roles=())
        decoder = super().from_state_dict(state_dict, **kwargs)
        if decoder.is_full_attention:
            return decoder
        parts = []
        for rank in range(TP):
            state = partition_state_dict(state_dict, decoder.global_hf_config, rank)
            weights = _projection_weights(state, decoder.cfg)
            parts.append(torch.cat([weights["gdn_packed"], weights["gdn_z_epilogue"]], dim=1))
        decoder.w["gdn_all"] = ttnn.from_torch(
            torch.cat(parts, dim=1).contiguous(),
            device=decoder.device,
            dtype=decoder.w["gdn_packed"].dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=1),
        )
        decoder.projection_compute["gdn_all"] = decoder.projection_compute["gdn_packed"]
        roles = dict(decoder.optimization.role_configs)
        roles.setdefault("gdn_all", dict(roles["gdn_packed"]))
        decoder.optimization = replace(decoder.optimization, role_configs=roles)
        return decoder

    def _gdn_project(self, x):
        if x.shape[1] != 1 and not self.packed_prefill:
            return super()._gdn_project(x)
        cfg = self.cfg
        packed = self._linear(x, "gdn_all")
        qkv = _field(packed, 0, cfg.conv_dim)
        a = _field(packed, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(packed, cfg.conv_dim + 32, cfg.conv_dim + 32 + cfg.linear_num_value_heads)
        z_raw = _field(packed, cfg.conv_dim + 64, packed.shape[-1])
        if self.fused_z or x.shape[1] != 1:
            z = z_raw
        else:
            z = ttnn.silu(z_raw, memory_config=z_raw.memory_config())
            ttnn.deallocate(z_raw)
        ttnn.deallocate(packed)
        return qkv, z, a, b

    def _gdn_out_head_major(self, core, z, batch, seq):
        if seq > 1 or not self.fused_z:
            return super()._gdn_out_head_major(core, z, batch, seq)
        normed = ttnn.rms_norm(core, weight=self.w["gdn_norm"], epsilon=self.cfg.norm_eps)
        ttnn.deallocate(core)
        heads = ttnn.reshape(normed, [batch, self.cfg.linear_num_value_heads, 1, self.cfg.linear_value_head_dim])
        combined = ttnn.permute(heads, (0, 2, 1, 3))
        ttnn.deallocate(normed)
        merged = ttnn.reshape(combined, [batch, 1, self.cfg.linear_v_dim])
        if self.z_operand == "lhs":
            gate_input = z
            if self.compact_z:
                gate_input = ttnn.reshape(z, [1, batch, self.cfg.linear_v_dim])
                merged = ttnn.reshape(merged, [1, batch, self.cfg.linear_v_dim])
            gated = ttnn.multiply(
                gate_input, merged, dtype=ttnn.float32, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]
            )
            if self.compact_z:
                gated = ttnn.reshape(gated, [batch, 1, self.cfg.linear_v_dim])
        else:
            gated = ttnn.multiply(merged, z, input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(combined)
        result = self._linear(gated, "gdn_out")
        ttnn.deallocate(gated)
        return result


class SharedMLPInput(MultichipDecoder):
    """Use one interleaved working input for the selected interleaved MLP."""

    def _activate_mlp(self, x, mode):
        if mode != "decode" or self.mesh_config.decode_grid is None:
            return super()._activate_mlp(x, mode)
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        gate = self._linear(local, "gate_proj")
        up = self._linear(local, "up_proj")
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        if local.buffer_address() != x.buffer_address():
            ttnn.deallocate(local)
        return result


class PackedGDNSharedMLP(PackedGDN, SharedMLPInput):
    pass


class PackedGDNFusedZ(PackedGDNSharedMLP):
    fused_z = True


class PackedGDNFusedZLhs(PackedGDNFusedZ):
    z_operand = "lhs"


class PackedGDNAllModes(PackedGDNFusedZLhs):
    packed_prefill = True
    compact_z = True


class PackedGDNGrid(PackedGDNAllModes):
    grid = (8, 4)

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role != "gdn_all":
            return super()._linear(x, role, **kwargs)
        plan = self.mesh_config
        self.mesh_config = replace(plan, decode_grid=self.grid)
        try:
            return super()._linear(x, role, **kwargs)
        finally:
            self.mesh_config = plan


class DramMLP(PackedGDNAllModes):
    """Use DRAM readers with a shared width-sharded MLP activation."""

    dram_roles = ("gate_proj", "up_proj", "down_proj")

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        plan = kwargs["mesh_config"]
        kwargs["mesh_config"] = replace(plan, decode_grid=None)
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.mesh_config = plan
        for role in list(decoder.decode_weights):
            if role not in decoder.dram_roles and role != "qkvg":
                ttnn.deallocate(decoder.decode_weights.pop(role))
        return decoder

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role not in self.dram_roles:
            return super()._linear(x, role, **kwargs)
        plan = self.mesh_config
        self.mesh_config = replace(plan, decode_grid=None)
        try:
            return super()._linear(x, role, **kwargs)
        finally:
            self.mesh_config = plan

    def _activate_mlp(self, x, mode):
        return OptimizedDecoder._activate_mlp(self, x, mode)


class Cache4(PackedGDNAllModes):
    def allocate_kv_cache(self, num_blocks, dtype=ttnn.bfloat4_b):
        return super().allocate_kv_cache(num_blocks, dtype=dtype)


class ProductionCandidate(MultichipDecoder):
    """Exercise the production implementation with an explicit trial policy."""

    qkvg_dtype = "bfloat8_b"
    qkvg_cores = 32
    qkvg_block = 4
    qkvg_readers = 2

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        plan = kwargs.pop("mesh_config", None) or MeshConfig()
        roles = dict(plan.local.role_configs)
        for role in ("gate_proj", "up_proj", "down_proj"):
            roles[role] = dict(roles[role], readers=2)
        roles["gdn_all"] = dict(cores=4, block_w=8, readers=1)
        roles["qkvg"] = dict(cores=cls.qkvg_cores, block_w=cls.qkvg_block, readers=cls.qkvg_readers)
        plan = replace(
            plan,
            pack_gdn=True,
            pack_mlp_decode=False,
            decode_dram_roles=("gate_proj", "up_proj", "down_proj"),
            decode_qkvg_dtype=cls.qkvg_dtype,
            local=replace(plan.local, role_configs=roles),
        )
        return super().from_state_dict(state_dict, mesh_config=plan, **kwargs)


class ProductionCache4(ProductionCandidate):
    def allocate_kv_cache(self, num_blocks, dtype=ttnn.bfloat4_b):
        return super().allocate_kv_cache(num_blocks, dtype=dtype)


class PrefillWideK(ProductionCandidate):
    """Trade output blocking for wider K buffers without changing logical work."""

    k_block = 64

    def _prefill_linear(self, x, role, **kwargs):
        if role != "gdn_all" or x.shape[1] < 2048:
            return super()._prefill_linear(x, role, **kwargs)
        grid = self.optimization.large_prefill_grid
        pm = math.ceil(math.prod(list(x.padded_shape)[:-1]) / (32 * grid[1]))
        pn = math.ceil(self.w[role].shape[-1] / (32 * grid[0]))
        out_n = max(v for v in range(1, min(pn, 3) + 1) if pn % v == 0)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=self.k_block,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=1,
            out_block_w=out_n,
            out_subblock_h=1,
            out_subblock_w=out_n,
            transpose_mcast=False,
            fuse_batch=True,
        )
        return ttnn.linear(
            x,
            self.w[role],
            program_config=program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )


class ShardedMLPInput(PackedGDNSharedMLP):
    """Keep a wider working shard as direct input to both multicast matmuls."""

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role not in ("gate_proj", "up_proj"):
            return super()._linear(x, role, **kwargs)
        batch, _, k = x.shape
        n = self.w[role].shape[-1]
        grid = self.mesh_config.decode_grid
        cores, block, _ = self._role_config(role)
        local = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, batch, k])
        local = ttnn.to_memory_config(local, self._width_memory(k, cores))
        weight = ttnn.reshape(self.w[role], [1, 1, k, n])
        per_n = math.ceil(n / (32 * math.prod(grid)))
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=block,
            out_subblock_h=1,
            out_subblock_w=max(v for v in range(1, 9) if per_n % v == 0),
            per_core_M=1,
            per_core_N=per_n,
            fuse_batch=True,
            mcast_in0=True,
        )
        out = ttnn.linear(
            local,
            weight,
            program_config=program,
            compute_kernel_config=self.projection_compute[role],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        return ttnn.reshape(out, [batch, 1, n])


CANDIDATES = {
    "packed_gdn": PackedGDN,
    "shared_mlp_input": SharedMLPInput,
    "packed_gdn_shared_mlp": PackedGDNSharedMLP,
    "packed_gdn_fused_z": PackedGDNFusedZ,
    "packed_gdn_fused_z_lhs": PackedGDNFusedZLhs,
    "packed_gdn_all_modes": PackedGDNAllModes,
    "packed_gdn_grid": PackedGDNGrid,
    "sharded_mlp_input": ShardedMLPInput,
    "dram_mlp": DramMLP,
    "cache4": Cache4,
    "production_candidate": ProductionCandidate,
    "production_cache4": ProductionCache4,
    "prefill_wide_k64": PrefillWideK,
    "prefill_wide_k128": type("PrefillWideK128", (PrefillWideK,), dict(k_block=128)),
}

for cores, block in ((32, 4), (16, 8), (8, 16)):
    CANDIDATES[f"production_qkv4_c{cores}"] = type(
        f"ProductionQkv4C{cores}",
        (ProductionCandidate,),
        dict(qkvg_dtype="bfloat4_b", qkvg_cores=cores, qkvg_block=block),
    )
CANDIDATES["production_qkv4_cache4"] = type(
    "ProductionQkv4Cache4",
    (ProductionCache4,),
    dict(qkvg_dtype="bfloat4_b", qkvg_cores=8, qkvg_block=16),
)

# Cross the cumulative projection path with the existing whole-layer CCL
# families. Each family retains its own residual/collective consumer methods.
from .multichip_topology_candidates import CANDIDATES as topology_candidates

for family in ("ag_mm", "fused_ag_mm", "fused_mm_rs", "packed_mlp", "persistent", "ccl_bfp8", "fused_norm_ag_mm"):
    CANDIDATES["cumulative_" + family] = type(
        "Cumulative" + topology_candidates[family].__name__,
        (topology_candidates[family], PackedGDNSharedMLP),
        {},
    )
    CANDIDATES["allmodes_" + family] = type(
        "AllModes" + topology_candidates[family].__name__,
        (topology_candidates[family], PackedGDNAllModes),
        {},
    )
    if family not in ("packed_mlp", "fused_norm_ag_mm"):
        CANDIDATES["production_" + family] = type(
            "Production" + topology_candidates[family].__name__,
            (topology_candidates[family], ProductionCandidate),
            dict(qkvg_dtype="bfloat4_b", qkvg_cores=8, qkvg_block=16),
        )


class ProductionPackedMLP(topology_candidates["packed_mlp"], ProductionCandidate):
    qkvg_dtype = "bfloat4_b"
    qkvg_cores = 8
    qkvg_block = 16

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        plan = kwargs.get("mesh_config") or MeshConfig()
        packed_config = plan.local.role_configs.get("gate_up")
        decoder = super().from_state_dict(state_dict, **kwargs)
        if packed_config is not None:
            roles = {**decoder.optimization.role_configs, "gate_up": dict(packed_config)}
            decoder.optimization = replace(decoder.optimization, role_configs=roles)
        dram = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        k, n = decoder.w["gate_up"].shape
        readers = decoder._role_config("gate_up")[2]
        width = math.ceil(n / (32 * dram.x * dram.y * readers)) * 32 * readers
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        decoder.decode_weights["gate_up"] = ttnn.to_memory_config(decoder.w["gate_up"], memory)
        decoder.mesh_config = replace(
            decoder.mesh_config, decode_dram_roles=(*decoder.mesh_config.decode_dram_roles, "gate_up")
        )
        return decoder


class ProductionFusedNorm(topology_candidates["fused_norm_ag_mm"], ProductionCandidate):
    qkvg_dtype = "bfloat4_b"
    qkvg_cores = 8
    qkvg_block = 16

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        for role in ("gate_proj", "up_proj"):
            old = decoder.decode_weights[role]
            decoder.decode_weights[role] = ttnn.to_memory_config(old, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(old)
        decoder.mesh_config = replace(decoder.mesh_config, decode_dram_roles=("down_proj",))
        return decoder


CANDIDATES.update(production_packed_mlp=ProductionPackedMLP, production_fused_norm_ag_mm=ProductionFusedNorm)


from .multichip_sdpa_candidate import SdpaLowLiveBuffers

CANDIDATES["production_qkv4_c8_sdpa_dram"] = type(
    "ProductionQkv4C8SdpaDram", (SdpaLowLiveBuffers, CANDIDATES["production_qkv4_c8"]), {}
)
CANDIDATES["production_qkv4_c8_sdpa_dram8"] = type(
    "ProductionQkv4C8SdpaDram8",
    (SdpaLowLiveBuffers, CANDIDATES["production_qkv4_c8"]),
    dict(sdpa_max_cores_per_head_batch=8),
)


CANDIDATES["production_qkv8_c8"] = type(
    "ProductionQkv8C8",
    (ProductionCandidate,),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=8, qkvg_block=16),
)


from .multichip_qkv_mixed_candidate import MixedQKVG

CANDIDATES["production_qkv4_gate8_split"] = type("ProductionQkv4Gate8Split", (MixedQKVG, ProductionCandidate), {})


CANDIDATES["qkv_gate_split_c8_b16_q1_g1_gb16"] = type(
    "qkv_gate_split_c8_b16_q1_g1_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 1, "gate_readers": 1, "gate_block": 16},
)
CANDIDATES["qkv_gate_split_c8_b16_q1_g2_gb16"] = type(
    "qkv_gate_split_c8_b16_q1_g2_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 1, "gate_readers": 2, "gate_block": 16},
)
CANDIDATES["qkv_gate_split_c8_b16_q1_g3_gb16"] = type(
    "qkv_gate_split_c8_b16_q1_g3_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 1, "gate_readers": 3, "gate_block": 16, "gate_per_core_n": 6},
)
CANDIDATES["qkv_gate_split_c8_b16_q2_g2_gb16"] = type(
    "qkv_gate_split_c8_b16_q2_g2_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 2, "gate_readers": 2, "gate_block": 16},
)
CANDIDATES["qkv_gate_split_c8_b16_q2_g3_gb16"] = type(
    "qkv_gate_split_c8_b16_q2_g3_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 2, "gate_readers": 3, "gate_block": 16, "gate_per_core_n": 6},
)
CANDIDATES["qkv_gate_split_c8_b16_q3_g1_gb16"] = type(
    "qkv_gate_split_c8_b16_q3_g1_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 3, "gate_readers": 1, "gate_block": 16},
)
CANDIDATES["qkv_gate_split_c8_b16_q3_g2_gb16"] = type(
    "qkv_gate_split_c8_b16_q3_g2_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 3, "gate_readers": 2, "gate_block": 16},
)
CANDIDATES["qkv_gate_split_c8_b16_q3_g3_gb16"] = type(
    "qkv_gate_split_c8_b16_q3_g3_gb16",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 3, "gate_readers": 3, "gate_block": 16, "gate_per_core_n": 6},
)
CANDIDATES["qkv_gate_split_c4_b32_q2_g1_gb32"] = type(
    "qkv_gate_split_c4_b32_q2_g1_gb32",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 4, "qkvg_block": 32, "qkvg_readers": 2, "gate_readers": 1, "gate_block": 32},
)
CANDIDATES["qkv_gate_split_c16_b8_q2_g1_gb8"] = type(
    "qkv_gate_split_c16_b8_q2_g1_gb8",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 16, "qkvg_block": 8, "qkvg_readers": 2, "gate_readers": 1, "gate_block": 8},
)
CANDIDATES["qkv_gate_split_c32_b4_q2_g1_gb4"] = type(
    "qkv_gate_split_c32_b4_q2_g1_gb4",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 32, "qkvg_block": 4, "qkvg_readers": 2, "gate_readers": 1, "gate_block": 4},
)
CANDIDATES["qkv_gate_split_c8_b16_q2_g1_gb8"] = type(
    "qkv_gate_split_c8_b16_q2_g1_gb8",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 2, "gate_readers": 1, "gate_block": 8},
)
CANDIDATES["qkv_gate_split_c8_b16_q2_g2_gb8"] = type(
    "qkv_gate_split_c8_b16_q2_g2_gb8",
    (MixedQKVG, ProductionCandidate),
    {"qkvg_cores": 8, "qkvg_block": 16, "qkvg_readers": 2, "gate_readers": 2, "gate_block": 8},
)


# Matched topology controls for the QKV8 policy that passes changed-input traces.
CANDIDATES["validated_default"] = type(
    "Validateddefault",
    (ProductionCandidate,),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_ag_mm"] = type(
    "Validatedag_mm",
    (CANDIDATES["production_ag_mm"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_fused_ag_mm"] = type(
    "Validatedfused_ag_mm",
    (CANDIDATES["production_fused_ag_mm"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_fused_mm_rs"] = type(
    "Validatedfused_mm_rs",
    (CANDIDATES["production_fused_mm_rs"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_packed_mlp"] = type(
    "Validatedpacked_mlp",
    (CANDIDATES["production_packed_mlp"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_persistent"] = type(
    "Validatedpersistent",
    (CANDIDATES["production_persistent"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_ccl_bfp8"] = type(
    "Validatedccl_bfp8",
    (CANDIDATES["production_ccl_bfp8"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)
CANDIDATES["validated_fused_norm_ag_mm"] = type(
    "Validatedfused_norm_ag_mm",
    (CANDIDATES["production_fused_norm_ag_mm"],),
    dict(qkvg_dtype="bfloat8_b", qkvg_cores=32, qkvg_block=4, qkvg_readers=1),
)


class PackedFusedNorm(topology_candidates["fused_norm_ag_mm"]):
    """Keep packed MLP while fusing each distributed norm's hidden gather."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        old = decoder.w["gate_up"]
        weight = ttnn.to_memory_config(old, ttnn.DRAM_MEMORY_CONFIG)
        decoder.w["gate_up"] = decoder.decode_weights["gate_up"] = weight
        ttnn.deallocate(old)
        return decoder

    def _activate_mlp(self, x, mode):
        return MultichipDecoder._activate_mlp(self, x, mode)


CANDIDATES["packed_fused_norm"] = PackedFusedNorm


class PrefillL1Inputs(MultichipDecoder):
    """Try advice-backed L1 input0, sharing one converted MLP input."""

    def _prefill_linear(self, x, role, **kwargs):
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        result = super()._prefill_linear(local, role, **kwargs)
        if local.buffer_address() != x.buffer_address():
            ttnn.deallocate(local)
        return result

    def _activate_mlp(self, x, mode):
        if mode == "decode":
            return super()._activate_mlp(x, mode)
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        result = super()._activate_mlp(local, mode)
        if local.buffer_address() != x.buffer_address():
            ttnn.deallocate(local)
        return result


CANDIDATES["prefill_l1_inputs"] = PrefillL1Inputs
