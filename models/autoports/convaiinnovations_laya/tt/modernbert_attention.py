# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY


class TtnnModernBertAttention:
    """linear(Wqkv) -> split heads -> rotary -> SDPA(mask, scale 1) -> concat heads -> linear(Wo)."""

    def __init__(self, parameters, config, layer_type, plan, device, policy=DEFAULT_POLICY, group="attn"):
        self.Wqkv = parameters["Wqkv"]
        self.Wo = parameters["Wo"]
        self.Wqkv_bias = parameters.get("Wqkv_bias")
        self.Wo_bias = parameters.get("Wo_bias")
        self.num_heads = config.num_attention_heads
        self.layer_type = layer_type
        self.compute_kernel_config = policy.compute_config(device, group)
        self.act_dtype = policy.act_dtype
        self.minimal_config = plan.minimal_config
        mem = plan.attention_memory
        self._mem = mem
        self._qkv_kwargs = {"compute_kernel_config": self.compute_kernel_config, "memory_config": mem}
        if plan.qkv_program_config is not None:
            self._qkv_kwargs["program_config"] = plan.qkv_program_config
        if self.Wqkv_bias is not None:
            self._qkv_kwargs["bias"] = self.Wqkv_bias
        self._sdpa_kwargs = {"compute_kernel_config": self.compute_kernel_config, "memory_config": mem}
        if plan.sdpa_program_config is not None:
            self._sdpa_kwargs["program_config"] = plan.sdpa_program_config
        self._split_kwargs = {"memory_config": mem}
        self._wo_kwargs = {"compute_kernel_config": self.compute_kernel_config, "memory_config": mem}
        if plan.wo_program_config is not None:
            self._wo_kwargs["program_config"] = plan.wo_program_config
        elif plan.down_core_grid is not None and not plan.wo_minimal:
            self._wo_kwargs["core_grid"] = plan.down_core_grid
        if self.Wo_bias is not None:
            self._wo_kwargs["bias"] = self.Wo_bias
        self._qkv_minimal = plan.qkv_minimal and self.minimal_config is not None
        self._wo_minimal = plan.wo_minimal and self.minimal_config is not None

    def _project(self, x, weight, bias, kwargs, minimal):
        if minimal:
            extra = {} if bias is None else {"bias_tensor": bias}
            return ttnn.experimental.minimal_matmul(
                x,
                weight,
                config=self.minimal_config,
                memory_config=self._mem,
                compute_kernel_config=self.compute_kernel_config,
                **extra,
            )
        return ttnn.linear(x, weight, **kwargs)

    def __call__(self, hidden_states, rotary, attn_mask):
        qkv = self._project(hidden_states, self.Wqkv, self.Wqkv_bias, self._qkv_kwargs, self._qkv_minimal)
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=False, **self._split_kwargs
        )
        ttnn.deallocate(qkv)

        if rotary is not None:
            rotated_q = rotary(query, self.layer_type)
            rotated_k = rotary(key, self.layer_type)
            ttnn.deallocate(query)
            ttnn.deallocate(key)
        else:
            rotated_q, rotated_k = query, key

        context = ttnn.transformer.scaled_dot_product_attention(
            rotated_q,
            rotated_k,
            value,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0,
            **self._sdpa_kwargs,
        )
        ttnn.deallocate(rotated_q)
        ttnn.deallocate(rotated_k)
        ttnn.deallocate(value)

        merged = ttnn.transformer.concatenate_heads(context, **self._split_kwargs)
        ttnn.deallocate(context)

        out = self._project(merged, self.Wo, self.Wo_bias, self._wo_kwargs, self._wo_minimal)
        ttnn.deallocate(merged)
        return out
