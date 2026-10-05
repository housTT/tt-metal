# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY, DEFAULT_PORT
from models.autoports.convaiinnovations_laya.tt.modernbert_attention import TtnnModernBertAttention
from models.autoports.convaiinnovations_laya.tt.modernbert_mlp import TtnnModernBertMLP


class TtnnModernBertEncoderLayer:
    """h = h + attn(attn_norm(h)); h = h + mlp(mlp_norm(h)). Layer 0 has no attention norm."""

    def __init__(self, parameters, config, layer_idx, plan, device, policy=DEFAULT_POLICY, port=DEFAULT_PORT):
        self.eps = config.norm_eps
        self.layer_idx = layer_idx
        self.attn_norm = parameters["attn_norm"]
        self.mlp_norm = parameters["mlp_norm"]
        self.layer_type = config.layer_types[layer_idx]
        self.plan = plan
        self.port = port
        self.norm_config = policy.compute_config(device, "norm")
        self.attn = TtnnModernBertAttention(parameters["attn"], config, self.layer_type, plan, device, policy)
        self.mlp = TtnnModernBertMLP(parameters["mlp"], config, plan, device, policy, port)
        shard = plan.mlp_shard
        self.residual_fp32 = policy.residual_fp32
        self.resident = shard is not None and shard.norm is not None and port.resident_residual and not self.residual_fp32
        self.sharded_mlp_half = shard is not None and shard.norm is not None and not self.resident and not self.residual_fp32
        self._attn_mem = plan.attention_memory

    def _norm(self, x, weight, sharded_plan=None):
        if sharded_plan is not None:
            return ttnn.layer_norm(
                x,
                weight=weight,
                epsilon=self.eps,
                program_config=sharded_plan.norm,
                memory_config=sharded_plan.hidden_memory,
                compute_kernel_config=self.norm_config,
            )
        return ttnn.layer_norm(x, weight=weight, epsilon=self.eps, compute_kernel_config=self.norm_config)

    def _to_matmul_dtype(self, x, owned):
        if not self.residual_fp32:
            return x, owned
        x16 = ttnn.typecast(x, ttnn.bfloat16)
        if owned:
            ttnn.deallocate(x)
        return x16, True

    def _residual_add(self, h, branch):
        if self.residual_fp32:
            b32 = ttnn.typecast(branch, ttnn.float32)
            ttnn.deallocate(branch)
            branch = b32
        out = ttnn.add(h, branch)
        ttnn.deallocate(branch)
        return out

    def __call__(self, hidden_states, rotary, attn_mask):
        if self.resident:
            return self._resident(hidden_states, rotary, attn_mask)

        if self.attn_norm is None:
            normed, owned = self._to_matmul_dtype(hidden_states, False)
        else:
            normed, owned = self._to_matmul_dtype(self._norm(hidden_states, self.attn_norm), True)
        attn_out = self.attn(normed, rotary, attn_mask)
        if owned:
            ttnn.deallocate(normed)
        h = self._residual_add(hidden_states, attn_out)

        if self.sharded_mlp_half:
            return self._sharded_mlp_half(h)

        mlp_normed, _ = self._to_matmul_dtype(self._norm(h, self.mlp_norm), True)
        mlp_out = self.mlp(mlp_normed)
        ttnn.deallocate(mlp_normed)
        out = self._residual_add(h, mlp_out)
        ttnn.deallocate(h)
        return out

    def _sharded_mlp_half(self, hidden_states):
        plan = self.plan.mlp_shard
        h = ttnn.to_memory_config(hidden_states, plan.hidden_memory)
        ttnn.deallocate(hidden_states)
        normed = self._norm(h, self.mlp_norm, plan)
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        out = ttnn.add(h, mlp_out, memory_config=plan.hidden_memory)
        ttnn.deallocate(h)
        ttnn.deallocate(mlp_out)
        interleaved = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return interleaved

    def _resident(self, h, rotary, attn_mask):
        plan = self.plan.mlp_shard
        if self.attn_norm is None:
            normed_sh = h
        else:
            normed_sh = self._norm(h, self.attn_norm, plan)
        normed = ttnn.to_memory_config(normed_sh, self._attn_mem)
        if self.attn_norm is not None:
            ttnn.deallocate(normed_sh)
        attn_out = self.attn(normed, rotary, attn_mask)
        ttnn.deallocate(normed)

        attn_sh = ttnn.to_memory_config(attn_out, plan.hidden_memory)
        ttnn.deallocate(attn_out)
        h2 = ttnn.add(h, attn_sh, memory_config=plan.hidden_memory)
        ttnn.deallocate(h)
        ttnn.deallocate(attn_sh)

        normed2 = self._norm(h2, self.mlp_norm, plan)
        mlp_out = self.mlp(normed2)
        ttnn.deallocate(normed2)
        out = ttnn.add(h2, mlp_out, memory_config=plan.hidden_memory)
        ttnn.deallocate(h2)
        ttnn.deallocate(mlp_out)
        return out
