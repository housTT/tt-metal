# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coherent last-mile topology controls over the actual optimized default."""

import math
import os
from dataclasses import replace

import torch

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder


class FinalPackedMLPCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        weight = torch.cat([state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"]]).T.contiguous()
        reader = int(os.environ.get("ORNITH_FINAL_PACKED_READERS", "2"))
        roles = dict(decoder.optimization.role_configs)
        roles["gate_up"] = dict(cores=64, block_w=2, readers=reader)
        decoder.optimization = replace(decoder.optimization, role_configs=roles)
        decoder.projection_compute["gate_up"] = decoder.projection_compute["gate_proj"]
        decoder.w["gate_up"] = ttnn.from_torch(
            weight,
            device=decoder.device,
            dtype=decoder.w["gate_proj"].dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        dg = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
        k, n = weight.shape
        width = math.ceil(n / (32 * dg.x * dg.y * reader)) * 32 * reader
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
        )
        decoder.decode_weights["gate_up"] = ttnn.from_torch(
            weight,
            device=decoder.device,
            dtype=decoder.w["gate_proj"].dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
        )
        return decoder

    def _activate_mlp(self, ff_in, mode):
        if mode != "decode":
            return super()._activate_mlp(ff_in, mode)
        packed = self._linear(ff_in, "gate_up")
        b, _, width = packed.shape
        half = width // 2
        # Count slices and activation in the final down48/residual32 contract.
        mem = self._width_memory(half, self._role_config("down_proj")[0])
        gate = ttnn.slice(packed, [0, 0, 0], [b, 1, half], memory_config=mem)
        up = ttnn.slice(packed, [0, 0, half], [b, 1, width], memory_config=mem)
        ttnn.deallocate(packed)
        result = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result


class FinalTopologyCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import math

        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        decoder.topology = os.environ.get("ORNITH_TOPOLOGY", "gate_epilogue")
        weights = {}
        source_role = "qkvg" if decoder.is_full_attention else "gdn_packed"
        if decoder.topology == "separate_attention" and decoder.is_full_attention:
            cfg = decoder.cfg
            qg = state_dict["self_attn.q_proj.weight"].reshape(cfg.n_heads, 2, cfg.head_dim, cfg.dim)
            weights = {
                "separate_q": qg[:, 0].reshape(-1, cfg.dim).T,
                "separate_gate": qg[:, 1].reshape(-1, cfg.dim).T,
                "separate_k": state_dict["self_attn.k_proj.weight"].T,
                "separate_v": state_dict["self_attn.v_proj.weight"].T,
            }
        if decoder.topology == "packed_gdn" and not decoder.is_full_attention:
            weights["gdn_all"] = torch.cat(
                [state_dict[f"linear_attn.in_proj_{n}.weight"] for n in ("qkv", "a", "b", "z")]
            ).T
        if decoder.topology == "separate_gdn" and not decoder.is_full_attention:
            weights["separate_gdn_qkv"] = state_dict["linear_attn.in_proj_qkv.weight"].T
            weights["separate_gdn_ab"] = torch.cat(
                [state_dict[f"linear_attn.in_proj_{n}.weight"] for n in ("a", "b")]
            ).T
        dg = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))])
        for role, value in weights.items():
            decoder.w[role] = ttnn.from_torch(
                value.contiguous(),
                device=decoder.device,
                dtype=decoder.w[source_role].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
            )
            decoder.projection_compute[role] = decoder.projection_compute[source_role]
            if role in decoder.optimization.dram_roles:
                k, n = value.shape
                readers = decoder._role_config(role)[2]
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        grid,
                        [k, math.ceil(n / (32 * dg.x * dg.y * readers)) * 32 * readers],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                decoder.decode_weights[role] = ttnn.from_torch(
                    value.contiguous(),
                    device=decoder.device,
                    dtype=decoder.w[role].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(decoder.device),
                )
        return decoder

    def _linear(self, x, role, **kwargs):
        if role == "qkvg" and self.topology == "separate_attention":
            parts = [
                super(FinalTopologyCandidate, self)._linear(x, n)
                for n in ("separate_q", "separate_k", "separate_v", "separate_gate")
            ]
            result = ttnn.concat(
                parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG if x.shape[1] == 1 else ttnn.DRAM_MEMORY_CONFIG
            )
            for part in parts:
                ttnn.deallocate(part)
            return result
        return super()._linear(x, role, **kwargs)

    def _activate_mlp(self, ff_in, mode):
        if mode != "decode" or self.topology != "gate_epilogue":
            return super()._activate_mlp(ff_in, mode)
        gate = self._linear(ff_in, "gate_proj", activation="silu")
        up = self._linear(ff_in, "up_proj")
        result = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return result

    def _gdn_project(self, x):
        if self.topology == "separate_gdn":
            from ..tt.fused_decoder import _field

            qkv = self._linear(x, "separate_gdn_qkv")
            ab = self._linear(x, "separate_gdn_ab")
            heads = self.cfg.linear_num_value_heads
            a, b = _field(ab, 0, heads), _field(ab, heads, 2 * heads)
            ttnn.deallocate(ab)
            grid = self.device.compute_with_storage_grid_size()
            kwargs = {"core_grid": ttnn.CoreGrid(x=grid.x, y=grid.y)}
            if x.shape[1] == 1:
                kwargs["activation"] = "silu"
            z = self._linear(x, "gdn_z_epilogue", **kwargs)
            return qkv, z, a, b
        if self.topology != "packed_gdn":
            return super()._gdn_project(x)
        from ..tt.fused_decoder import _field

        cfg = self.cfg
        p = self._linear(x, "gdn_all")
        qkv = _field(p, 0, cfg.conv_dim)
        a = _field(p, cfg.conv_dim, cfg.conv_dim + cfg.linear_num_value_heads)
        b = _field(p, cfg.conv_dim + cfg.linear_num_value_heads, cfg.conv_dim + 2 * cfg.linear_num_value_heads)
        z = _field(
            p,
            cfg.conv_dim + 2 * cfg.linear_num_value_heads,
            cfg.conv_dim + 2 * cfg.linear_num_value_heads + cfg.linear_v_dim,
        )
        ttnn.deallocate(p)
        if x.shape[1] == 1:
            z = ttnn.silu(z)
        return qkv, z, a, b


class FinalPrefillMLPBlockCandidate(OptimizedDecoder):
    def _prefill_linear(self, x, role, **kwargs):
        if role not in ("gate_proj", "up_proj"):
            return super()._prefill_linear(x, role, **kwargs)
        grid = (8, 8)
        weight = self.w[role]
        k, n = weight.shape[-2], weight.shape[-1]
        m = math.prod(list(x.padded_shape)[:-1]) // 32
        pm, pn = math.ceil(m / grid[1]), math.ceil(n / 32 / grid[0])
        block_w = 8
        block_m = max(
            v for v in range(1, min(pm, int(os.environ.get("ORNITH_FINAL_PREFILL_M", "8"))) + 1) if pm % v == 0
        )
        block_n = max(v for v in range(1, min(pn, 48) + 1) if pn % v == 0)
        subblock_h = max(v for v in range(1, min(block_m, 2) + 1) if block_m % v == 0)
        subblock = max(v for v in range(1, min(8 // subblock_h, 4) + 1) if block_n % v == 0)
        kwargs.pop("core_grid", None)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=block_w,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=block_m,
            out_block_w=block_n,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock,
            transpose_mcast=False,
            fuse_batch=True,
        )
        out = ttnn.linear(
            x,
            weight,
            program_config=program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )
        return out


class FinalCacheCandidate(OptimizedDecoder):
    def allocate_kv_cache(self, num_blocks, dtype=ttnn.bfloat4_b):
        return super().allocate_kv_cache(num_blocks, dtype=dtype)


class FinalZApproxCandidate(OptimizedDecoder):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            decoder.projection_compute["gdn_z_epilogue"] = ttnn.init_device_compute_kernel_config(
                decoder.device.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, decoder.policy.attention_fidelity),
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        return decoder


class FinalPrefillGridCandidate(OptimizedDecoder):
    def _prefill_linear(self, x, role, **kwargs):
        grid = tuple(int(v) for v in os.environ.get("ORNITH_PREFILL_GRID", "11,10").split(","))
        weight = self.w[role]
        k, n = weight.shape[-2], weight.shape[-1]
        m = math.prod(list(x.padded_shape)[:-1]) // 32
        pm, pn = math.ceil(m / grid[1]), math.ceil(n / 32 / grid[0])
        block_w = int(os.environ.get("ORNITH_PREFILL_BLOCK", "8"))
        block_m = max(
            v for v in range(1, min(pm, int(os.environ.get("ORNITH_FINAL_PREFILL_M", "8"))) + 1) if pm % v == 0
        )
        block_n = max(
            v for v in range(1, min(pn, int(os.environ.get("ORNITH_PREFILL_OUT_N", "32"))) + 1) if pn % v == 0
        )
        subblock_h = max(
            v for v in range(1, min(block_m, int(os.environ.get("ORNITH_PREFILL_SUB_H", "2"))) + 1) if block_m % v == 0
        )
        subblock = max(
            v
            for v in range(1, min(8 // subblock_h, int(os.environ.get("ORNITH_PREFILL_SUB_W", "4"))) + 1)
            if block_n % v == 0
        )
        kwargs.pop("core_grid", None)
        program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=block_w,
            per_core_M=pm,
            per_core_N=pn,
            out_block_h=block_m,
            out_block_w=block_n,
            out_subblock_h=subblock_h,
            out_subblock_w=subblock,
            transpose_mcast=False,
            fuse_batch=True,
        )
        out = ttnn.linear(
            x,
            weight,
            program_config=program,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.projection_compute[role],
            **kwargs,
        )
        return out


class FinalLargePrefillMLPBlockCandidate(FinalPrefillGridCandidate):
    def _prefill_linear(self, x, role, **kwargs):
        if role not in ("gate_proj", "up_proj"):
            return OptimizedDecoder._prefill_linear(self, x, role, **kwargs)
        return super()._prefill_linear(x, role, **kwargs)
