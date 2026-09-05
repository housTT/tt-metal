# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual split QKV4/gate8 decode projections for the production TP4 family.

The packed BFP4 prefill weight remains unchanged. Decode performs exactly two
smaller matmuls sharing one activation conversion, then restores the public
packed QKVG output contract. This candidate does not retain the unused full
packed decode weight used during parent setup.
"""

import math
from dataclasses import replace
from types import SimpleNamespace

import ttnn

from ..tt.multichip_decoder import TP, _projection_weights, partition_state_dict


class MixedQKVG:
    """Compose before ProductionCandidate; both matmuls share input shards."""

    qkvg_dtype = "bfloat4_b"
    qkvg_cores = 8
    qkvg_block = 16
    qkvg_readers = 2
    gate_block = 16
    gate_readers = 1
    qkv_per_core_n = None
    gate_per_core_n = None

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        import torch

        decoder = super().from_state_dict(state_dict, **kwargs)
        if not decoder.is_full_attention:
            return decoder
        ttnn.deallocate(decoder.decode_weights.pop("qkvg"))
        local = SimpleNamespace(
            num_attention_heads=decoder.cfg.n_heads,
            head_dim=decoder.cfg.head_dim,
            hidden_size=decoder.cfg.dim,
        )
        packed = [
            _projection_weights(partition_state_dict(state_dict, decoder.global_hf_config, rank), local)["qkvg"]
            for rank in range(TP)
        ]
        qkv_width = (decoder.cfg.n_heads + 2 * decoder.cfg.n_kv_heads) * decoder.cfg.head_dim
        dram = decoder.device.dram_grid_size()
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        roles = dict(decoder.optimization.role_configs)
        decoder.mixed_projection_policy = {}
        for role, parts, dtype, block, readers, per_core_n in (
            (
                "qkv_split",
                [value[:, :qkv_width] for value in packed],
                ttnn.bfloat4_b,
                cls.qkvg_block,
                cls.qkvg_readers,
                cls.qkv_per_core_n,
            ),
            (
                "attn_gate_split",
                [value[:, qkv_width:] for value in packed],
                ttnn.bfloat8_b,
                cls.gate_block,
                cls.gate_readers,
                cls.gate_per_core_n,
            ),
        ):
            k, n = parts[0].shape
            width = math.ceil(n / (32 * dram.x * dram.y * readers)) * 32 * readers
            memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, [k, width], ttnn.ShardOrientation.ROW_MAJOR),
            )
            decoder.decode_weights[role] = ttnn.from_torch(
                torch.cat(parts, dim=1).contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=decoder.device,
                memory_config=memory,
                mesh_mapper=ttnn.ShardTensorToMesh(decoder.device, dim=1),
            )
            roles[role] = dict(cores=cls.qkvg_cores, block_w=block, readers=readers)
            if per_core_n is not None:
                roles[role]["per_core_n"] = per_core_n
            decoder.mixed_projection_policy[role] = dict(
                shape=[k, n], dtype=str(dtype), storage_width_per_bank=width, **roles[role]
            )
        decoder.optimization = replace(decoder.optimization, role_configs=roles)
        decoder.mesh_config = replace(decoder.mesh_config, local=decoder.optimization)
        return decoder

    def _linear(self, x, role, **kwargs):
        if role != "qkvg" or x.shape[1] != 1:
            return super()._linear(x, role, **kwargs)
        batch, _, width = x.shape
        folded = ttnn.reshape(x, [1, batch, width])
        memory = self._width_memory(width, self.qkvg_cores)
        working = ttnn.to_memory_config(folded, memory)
        outputs = []
        for name in ("qkv_split", "attn_gate_split"):
            cores, block, readers = self._role_config(name)
            n = self.decode_weights[name].shape[-1]
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=block,
                per_core_M=1,
                per_core_N=self.optimization.role_configs[name].get("per_core_n", math.ceil(n / (32 * cores))),
                num_workers_per_dram_bank=readers,
            )
            output = ttnn.linear(
                working,
                self.decode_weights[name],
                program_config=program,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.projection_compute[role],
                **kwargs,
            )
            interleaved = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(output)
            outputs.append(ttnn.reshape(interleaved, [batch, 1, n]))
        result = ttnn.concat(outputs, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        for output in outputs:
            ttnn.deallocate(output)
        if working.buffer_address() != x.buffer_address():
            ttnn.deallocate(working)
        return result
