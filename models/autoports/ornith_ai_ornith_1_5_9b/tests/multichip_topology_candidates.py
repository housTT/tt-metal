# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Whole-layer candidate families; not production defaults."""

import copy
import math
from dataclasses import replace

import ttnn

from ..tt.multichip_decoder import TP, MultichipDecoder, _projection_weights, partition_state_dict
from ..tt.optimized_decoder import OptimizedDecoder


def local_projection(decoder, x, role, **kwargs):
    """Same selected local projection, without its row-output collective."""
    if (
        x.shape[1] != 1
        or decoder.mesh_config.decode_grid is None
        or (role == "qkvg" and decoder.mesh_config.decode_qkvg_dram)
    ):
        return OptimizedDecoder._linear(decoder, x, role, **kwargs)
    batch, _, k = x.shape
    n = decoder.w[role].shape[-1]
    grid = decoder.mesh_config.decode_grid
    local = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, batch, k])
    weight = ttnn.reshape(decoder.decode_weights.get(role, decoder.w[role]), [1, 1, k, n])
    per_n = math.ceil(n / (32 * math.prod(grid)))
    program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=decoder._role_config(role)[1],
        out_subblock_h=1,
        out_subblock_w=max(v for v in range(1, 9) if per_n % v == 0),
        per_core_M=1,
        per_core_N=per_n,
        fuse_batch=True,
        mcast_in0=True,
    )
    kwargs.setdefault("dtype", ttnn.bfloat16)
    out = ttnn.linear(
        local,
        weight,
        program_config=program,
        compute_kernel_config=decoder.projection_compute[role],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **kwargs,
    )
    return ttnn.reshape(out, [batch, 1, n])


class GatherOutputProjection(MultichipDecoder):
    """Gather local mixer/MLP output, then use column-sharded WO/down weights."""

    fused = False
    roles = ("gdn_out", "o_proj", "down_proj")

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.column_weights = {}
        names = {
            "gdn_out": "linear_attn.out_proj.weight",
            "o_proj": "self_attn.o_proj.weight",
            "down_proj": "mlp.down_proj.weight",
        }
        for role, source in names.items():
            if source not in state_dict:
                continue
            host = state_dict[source].T.contiguous().reshape(1, 1, state_dict[source].shape[1], -1)
            decoder.column_weights[role] = ttnn.from_torch(
                host,
                device=decoder.device,
                dtype=decoder.w[role].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=-1),
            )
        return decoder

    def _linear(self, x, role, **kwargs):
        if role not in self.roles or x.shape[1] != 1:
            return super()._linear(x, role, **kwargs)
        batch, _, k = x.shape
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        local = ttnn.reshape(local, [1, 1, batch, k])
        weight = self.column_weights[role]
        width = self.cfg.dim // TP
        kwargs.setdefault("dtype", ttnn.bfloat16)
        if self.fused:
            local = ttnn.to_memory_config(local, self._width_memory(k, 1))
            gathered_mem = self._width_memory(k * TP, TP)
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=width // 32 // 8,
                fuse_batch=True,
                mcast_in0=True,
            )
            gathered, projected = ttnn.experimental.all_gather_matmul_async(
                local,
                weight,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                all_gather_core_grid_offset=(0, 4),
                num_links=1,
                memory_config_ag=gathered_mem,
                memory_config_mm=self._width_memory(width, 8),
                program_config=program,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
            ttnn.deallocate(gathered)
        else:
            gathered = self._gather(local)
            program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=width // 32 // 8,
                fuse_batch=True,
                mcast_in0=True,
            )
            projected = ttnn.linear(
                gathered,
                weight,
                program_config=program,
                compute_kernel_config=self.projection_compute[role],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                **kwargs,
            )
            ttnn.deallocate(gathered)
        if self.mesh_config.residual == "replicated":
            projected = self._gather(ttnn.to_memory_config(projected, ttnn.L1_MEMORY_CONFIG))
        return ttnn.reshape(projected, [batch, 1, self.residual_width])


class FusedGatherOutputProjection(GatherOutputProjection):
    fused = True


class FusedReduceProjection(MultichipDecoder):
    """Fused row-parallel matmul/RS, consumed by sharded norm in the full layer."""

    roles = ("gdn_out", "o_proj", "down_proj")

    def _linear(self, x, role, **kwargs):
        if role not in self.roles or x.shape[1] != 1:
            return super()._linear(x, role, **kwargs)
        batch, _, k = x.shape
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        local = ttnn.reshape(local, [1, 1, batch, k])
        weight = ttnn.reshape(self.w[role], [1, 1, k, self.cfg.dim])
        # These buffers are local outputs, owned by this invocation. They are
        # recorded allocations during capture, then stable on replay.
        intermediate = ttnn.empty(
            [1, 1, batch, self.cfg.dim],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output = ttnn.empty(
            [1, 1, batch, self.cfg.dim // TP],
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            per_core_M=1,
            per_core_N=16,
            transpose_mcast=False,
            fuse_batch=True,
        )
        mm, reduced = ttnn.experimental.matmul_reduce_scatter_async(
            local,
            weight,
            persistent_intermediate_buffer=intermediate,
            persistent_output_buffer=output,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            reduce_scatter_core_grid_offset=(0, 6),
            num_links=1,
            memory_config_rs=ttnn.L1_MEMORY_CONFIG,
            intermediate_memory_config_rs=ttnn.L1_MEMORY_CONFIG,
            memory_config_mm=ttnn.L1_MEMORY_CONFIG,
            program_config=program,
            compute_kernel_config=self.projection_compute[role],
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(mm)
        if self.mesh_config.residual == "replicated":
            reduced = self._gather(reduced)
        return ttnn.reshape(reduced, [batch, 1, self.residual_width])


class PackedMLP(MultichipDecoder):
    """Local gate/up packing preserves the selected residual/CCL contract."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        gate, up = (state_dict[f"mlp.{role}_proj.weight"].chunk(TP, dim=0) for role in ("gate", "up"))
        host = torch.cat([torch.cat([g, u]).T for g, u in zip(gate, up)], dim=1).contiguous()
        k, n = decoder.cfg.dim, decoder.cfg.intermediate_size * 2
        dg = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [k, math.ceil(n / (32 * dg.x * dg.y)) * 32], ttnn.ShardOrientation.ROW_MAJOR),
        )
        for target, memory in ((decoder.w, ttnn.DRAM_MEMORY_CONFIG), (decoder.decode_weights, mem)):
            if target is decoder.decode_weights and decoder.mesh_config.decode_grid is not None:
                continue
            target["gate_up"] = ttnn.from_torch(
                host,
                device=decoder.device,
                dtype=decoder.w["gate_proj"].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=1),
            )
        decoder.projection_compute["gate_up"] = decoder.projection_compute["gate_proj"]
        decoder.optimization = replace(
            decoder.optimization,
            role_configs={
                **decoder.optimization.role_configs,
                "gate_up": dict(decoder.optimization.role_configs["gate_proj"]),
            },
        )
        return decoder

    def _activate_mlp(self, x, mode):
        if mode != "decode":
            return super()._activate_mlp(x, mode)
        packed = self._linear(x, "gate_up")
        batch, _, width = packed.shape
        memory = self._width_memory(width // 2, self._role_config("down_proj")[0])
        gate = ttnn.slice(packed, [0, 0, 0], [batch, 1, width // 2], memory_config=memory)
        up = ttnn.slice(packed, [0, 0, width // 2], [batch, 1, width], memory_config=memory)
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=memory)
        for value in (packed, gate, up):
            ttnn.deallocate(value)
        return result


class CclBfp8(MultichipDecoder):
    """Quantize only row-projection collective boundaries; norms remain BF16."""

    def _linear(self, x, role, **kwargs):
        if role not in ("gdn_out", "o_proj", "down_proj"):
            return super()._linear(x, role, **kwargs)
        out = local_projection(self, x, role, **kwargs)
        shape = list(out.shape)
        memory = ttnn.L1_MEMORY_CONFIG if shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG
        local = ttnn.typecast(ttnn.to_memory_config(out, memory), ttnn.bfloat8_b)
        folded = ttnn.reshape(local, [1, 1, shape[0] * shape[1], shape[2]])
        if self.mesh_config.residual == "replicated" and self.mesh_config.collective == "native":
            reduced = ttnn.all_reduce(folded, num_links=self.mesh_config.links, topology=ttnn.Topology.Ring)
        else:
            reduced = ttnn.experimental.reduce_scatter_minimal_async(
                folded,
                dim=3,
                persistent_output_buffers=None,
                multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                topology=ttnn.Topology.Ring,
                memory_config=memory,
                intermediate_memory_config=memory,
            )
            if self.mesh_config.residual == "replicated":
                reduced = self._gather(reduced)
        shape[-1] = self.residual_width
        return ttnn.reshape(ttnn.typecast(reduced, ttnn.bfloat16), shape)


class InterleavedQkvg(MultichipDecoder):
    """Interleaved QKV weight contract for standard/fused multicast matmul."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        from ..tt.multichip_decoder import MeshConfig

        plan = kwargs.pop("mesh_config", None) or MeshConfig()
        return super().from_state_dict(state_dict, mesh_config=replace(plan, decode_qkvg_dram=False), **kwargs)


class QkvgGrid(InterleavedQkvg):
    """Change only QKV compute grid, preserving the other selected projections."""

    grid = (8, 8)

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role != "qkvg":
            return super()._linear(x, role, **kwargs)
        plan = self.mesh_config
        self.mesh_config = replace(plan, decode_grid=self.grid)
        try:
            return super()._linear(x, role, **kwargs)
        finally:
            self.mesh_config = plan


class WideInterleaved(InterleavedQkvg):
    """Spread interleaved-weight decode projections across up to 64 workers."""

    grid = (8, 8)

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1:
            return super()._linear(x, role, **kwargs)
        batch, _, k = x.shape
        n = self.w[role].shape[-1]
        local = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        local = ttnn.reshape(local, [1, 1, batch, k])
        weight = ttnn.reshape(self.decode_weights.get(role, self.w[role]), [1, 1, k, n])
        per_n = math.ceil(n / (32 * math.prod(self.grid)))
        subblock = max(v for v in range(1, 9) if per_n % v == 0)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.grid,
            in0_block_w=self._role_config(role)[1],
            out_subblock_h=1,
            out_subblock_w=subblock,
            per_core_M=1,
            per_core_N=per_n,
            fuse_batch=True,
            mcast_in0=True,
        )
        kwargs.setdefault("dtype", ttnn.bfloat16)
        out = ttnn.linear(
            local,
            weight,
            program_config=program,
            compute_kernel_config=self.projection_compute[role],
            memory_config=ttnn.L1_MEMORY_CONFIG,
            **kwargs,
        )
        if role in ("gdn_out", "o_proj", "down_proj"):
            if self.mesh_config.residual == "replicated" and self.mesh_config.collective == "native":
                out = ttnn.all_reduce(out, num_links=2, topology=ttnn.Topology.Ring)
            else:
                out = ttnn.experimental.reduce_scatter_minimal_async(
                    out,
                    dim=3,
                    persistent_output_buffers=None,
                    multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
                    barrier_semaphore=self.ccl.get_barrier_semaphore(),
                    num_links=1,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                if self.mesh_config.residual == "replicated":
                    out = self._gather(out)
            n = self.residual_width
        return ttnn.reshape(out, [batch, 1, n])


class PersistentCollectives(MultichipDecoder):
    """Decode RS/AG buffers allocated once; residual consumers borrow them."""

    def allocate_state(self, batch_size):
        super().allocate_state(batch_size)
        self.collective_buffers = {}
        for role in ("o_proj" if self.is_full_attention else "gdn_out", "down_proj"):
            self.collective_buffers[role] = [
                ttnn.empty(
                    [1, 1, batch_size, width],
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                for width in (4096, 1024, 4096)
            ]

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role not in self.collective_buffers:
            return super()._linear(x, role, **kwargs)
        out = local_projection(self, x, role, **kwargs)
        shape = list(out.shape)
        local = ttnn.reshape(ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG), [1, 1, shape[0], 4096])
        intermediate, reduced_buffer, gathered_buffer = self.collective_buffers[role]
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            local,
            dim=3,
            persistent_output_buffers=[intermediate, reduced_buffer],
            multi_device_global_semaphore=self.ccl.get_rs_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if self.mesh_config.residual == "replicated":
            reduced = ttnn.experimental.all_gather_async(
                reduced,
                dim=3,
                persistent_output_buffer=gathered_buffer,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                num_links=1,
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        return ttnn.reshape(reduced, [shape[0], 1, self.residual_width])

    def _block(self, x, *, mode, current_pos=None, rot_idxs=None, page_table=None, **kwargs):
        if mode != "decode":
            return super()._block(
                x, mode=mode, current_pos=current_pos, rot_idxs=rot_idxs, page_table=page_table, **kwargs
            )
        memory = self._width_memory(self.residual_width, 32)

        def norm(value, name):
            return (
                self._norm(value, self.w[name])
                if self.mesh_config.residual == "replicated"
                else self._residual_norm(value, name)
            )

        normalized = norm(x, "attn_norm")
        mixed = (
            self._attention_decode(normalized, current_pos, rot_idxs, page_table)
            if self.is_full_attention
            else self._gdn_decode(normalized)
        )
        ttnn.deallocate(normalized)
        h = self._residual_add(x, mixed, memory)
        # mixed and ff_out borrow role-specific persistent collective buffers.
        normalized = norm(h, "ff_norm")
        activated = self._activate_mlp(normalized, mode)
        ttnn.deallocate(normalized)
        ff_out = self._linear(activated, "down_proj")
        ttnn.deallocate(activated)
        result = self._residual_add(h, ff_out, memory)
        ttnn.deallocate(h)
        return result


class FusedNormGatherProjection(InterleavedQkvg):
    """Fuse hidden gather into QKV/gate, sharing gathered input with Z/up."""

    def _residual_norm(self, x, name):
        if x.shape[1] != 1:
            return super()._residual_norm(x, name)
        previous = getattr(self, "_gathered_norm", None)
        if previous is not None:
            ttnn.deallocate(previous)
        self._gathered_norm = None
        batch, _, width = x.shape
        local = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, batch, width])
        stats = ttnn.rms_norm_pre_all_gather(
            local, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config
        )
        gathered_stats = self._gather(stats)
        normalized = ttnn.rms_norm_post_all_gather(
            local,
            gathered_stats,
            epsilon=self.cfg.norm_eps,
            weight=self.distributed_norms[name],
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(stats)
        ttnn.deallocate(gathered_stats)
        return ttnn.reshape(normalized, [batch, 1, width])

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role in ("gdn_out", "o_proj", "down_proj"):
            return super()._linear(x, role, **kwargs)
        if x.shape[-1] == self.cfg.dim:
            return super()._linear(x, role, **kwargs)
        batch = x.shape[0]
        n = self.w[role].shape[-1]
        local = ttnn.reshape(ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG), [1, 1, batch, 1024])
        per_n = math.ceil(n / (32 * 64))
        activation = kwargs.pop("activation", None)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=self._role_config(role)[1],
            out_subblock_h=1,
            out_subblock_w=max(v for v in range(1, 9) if per_n % v == 0),
            per_core_M=1,
            per_core_N=per_n,
            fuse_batch=True,
            mcast_in0=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation == "silu" else None,
        )
        weight = ttnn.reshape(self.decode_weights.get(role, self.w[role]), [1, 1, 4096, n])
        kwargs.setdefault("dtype", ttnn.bfloat16)
        if self._gathered_norm is None:
            local = ttnn.to_memory_config(local, self._width_memory(1024, 1))
            gathered, projected = ttnn.experimental.all_gather_matmul_async(
                local,
                weight,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=self.ccl.get_barrier_semaphore(),
                all_gather_core_grid_offset=(0, 8),
                num_links=1,
                memory_config_ag=self._width_memory(4096, 4),
                memory_config_mm=ttnn.L1_MEMORY_CONFIG,
                program_config=program,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
            self._gathered_norm = gathered
        else:
            projected = ttnn.linear(
                self._gathered_norm,
                weight,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=program,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
        return ttnn.reshape(projected, [batch, 1, n])

    def _activate_mlp(self, x, mode):
        if mode != "decode":
            return super()._activate_mlp(x, mode)
        gate, up = self._linear(x, "gate_proj"), self._linear(x, "up_proj")
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result


CANDIDATES = {
    "ag_mm": GatherOutputProjection,
    "fused_ag_mm": FusedGatherOutputProjection,
    "fused_mm_rs": FusedReduceProjection,
    "packed_mlp": PackedMLP,
    "ccl_bfp8": CclBfp8,
    "wide_interleaved": WideInterleaved,
    "qkvg_grid": QkvgGrid,
    "persistent": PersistentCollectives,
    "fused_norm_ag_mm": FusedNormGatherProjection,
}


def fp32_attention_candidate(base):
    """Decode-only accumulator control; preserve the baseline prefill policy."""

    class Fp32Attention(base):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            decoder = super().from_state_dict(*args, **kwargs)
            decoder.fp32_attention_compute = {
                role: ttnn.init_device_compute_kernel_config(
                    decoder.device.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, decoder.policy.attention_fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                for role in decoder.projection_compute
                if role not in ("gate_proj", "up_proj", "down_proj", "gate_up")
            }
            return decoder

        def _linear(self, x, role, **kwargs):
            if x.shape[1] != 1 or role not in self.fp32_attention_compute:
                return super()._linear(x, role, **kwargs)
            old = self.projection_compute[role]
            self.projection_compute[role] = self.fp32_attention_compute[role]
            try:
                return super()._linear(x, role, **kwargs)
            finally:
                self.projection_compute[role] = old

    return Fp32Attention


class DecodeAttention8(MultichipDecoder):
    """BFP8 decode weights with unchanged BFP4 prefill and local head ownership."""

    roles = ("qkvg", "o_proj")

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            raise ValueError("DecodeAttention8 is a full-attention candidate")
        local = copy.deepcopy(decoder.global_hf_config)
        local.num_attention_heads //= TP
        parts = [
            _projection_weights(partition_state_dict(state_dict, decoder.global_hf_config, rank), local)
            for rank in range(TP)
        ]
        decoder.decode_attention_weights = {}
        for role in cls.roles:
            axis = 0 if role == "o_proj" else 1
            decoder.decode_attention_weights[role] = ttnn.from_torch(
                torch.cat([part[role] for part in parts], dim=axis).contiguous(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=decoder.decode_weights.get(role, decoder.w[role]).memory_config(),
                mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=axis),
            )
        return decoder

    def _linear(self, x, role, **kwargs):
        if x.shape[1] != 1 or role not in self.decode_attention_weights:
            return super()._linear(x, role, **kwargs)
        old = self.decode_weights.get(role)
        self.decode_weights[role] = self.decode_attention_weights[role]
        try:
            return super()._linear(x, role, **kwargs)
        finally:
            if old is None:
                self.decode_weights.pop(role)
            else:
                self.decode_weights[role] = old


class DecodeQkvg8(DecodeAttention8):
    roles = ("qkvg",)


CANDIDATES.update(decode_attention8=DecodeAttention8, decode_qkvg8=DecodeQkvg8)


class DramQkvg(MultichipDecoder):
    """Use the dedicated DRAM matmul only for decode QKVG; retain other32-core ops."""

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        from ..tt.multichip_decoder import MeshConfig

        plan = kwargs.pop("mesh_config", None) or MeshConfig()
        decoder = super().from_state_dict(state_dict, mesh_config=replace(plan, decode_grid=None), **kwargs)
        if not decoder.is_full_attention:
            raise ValueError("DramQkvg is a full-attention candidate")
        for role in list(decoder.decode_weights):
            if role != "qkvg":
                ttnn.deallocate(decoder.decode_weights.pop(role))
        decoder.mesh_config = plan
        return decoder

    def _linear(self, x, role, **kwargs):
        if x.shape[1] == 1 and role == "qkvg":
            return OptimizedDecoder._linear(self, x, role, **kwargs)
        return super()._linear(x, role, **kwargs)


CANDIDATES["dram_qkvg"] = DramQkvg
