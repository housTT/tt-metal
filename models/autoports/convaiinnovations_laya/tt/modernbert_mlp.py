# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_POLICY, mlp_up_projection_program_config


class TtnnModernBertMLP:
    """GeGLU: Wo(gelu(x @ Wi_act) * (x @ Wi_gate)); block-sharded on one grid when the plan says so."""

    def __init__(self, parameters, config, plan, device, policy=DEFAULT_POLICY, port=None):
        self.width = plan.mlp_width
        weights = parameters[self.width]
        self.Wi_act = weights["Wi_act"]
        self.Wi_gate = weights["Wi_gate"]
        self.Wo = weights["Wo"]
        self.compute_kernel_config = policy.compute_config(device, "mlp")
        self.gelu_approx = policy.gelu_approx
        self.shard = plan.mlp_shard
        self.act_dtype = policy.act_dtype
        self.minimal_config = plan.minimal_config if plan.wo_minimal else None
        self._mem = plan.attention_memory
        self.gelu_separate = port is not None and port.gelu_separate
        fused = policy.gelu_activation(not self.gelu_separate)
        act_pc = gate_pc = None
        if self.shard is None and port is not None and port.interleaved_pad > 0 and self.width == port.interleaved_pad:
            act_pc = mlp_up_projection_program_config(
                device, plan.batch_size, plan.seq_len, config.hidden_size, self.width, not self.gelu_separate, policy, port
            )
            gate_pc = mlp_up_projection_program_config(
                device, plan.batch_size, plan.seq_len, config.hidden_size, self.width, False, policy, port
            )
        self.act_program_config = act_pc
        self._act_kwargs = {"compute_kernel_config": self.compute_kernel_config, "dtype": self.act_dtype}
        if act_pc is not None:
            self._act_kwargs["program_config"] = act_pc
        elif fused is not None:
            self._act_kwargs["activation"] = fused
        self._gate_kwargs = {"compute_kernel_config": self.compute_kernel_config, "dtype": self.act_dtype}
        if gate_pc is not None:
            self._gate_kwargs["program_config"] = gate_pc
        self._down_kwargs = {"compute_kernel_config": self.compute_kernel_config}
        if plan.mlp_down_program_config is not None:
            self._down_kwargs["program_config"] = plan.mlp_down_program_config
        elif plan.down_core_grid is not None and not plan.wo_minimal:
            self._down_kwargs["core_grid"] = plan.down_core_grid

    def __call__(self, hidden_states):
        if self.shard is not None:
            return self._sharded(hidden_states)

        activated = ttnn.linear(hidden_states, self.Wi_act, **self._act_kwargs)
        if self.gelu_separate:
            pre = activated
            activated = ttnn.gelu(pre, fast_and_approximate_mode=self.gelu_approx)
            ttnn.deallocate(pre)
        gate = ttnn.linear(hidden_states, self.Wi_gate, **self._gate_kwargs)
        gated = ttnn.mul(activated, gate)
        ttnn.deallocate(activated)
        ttnn.deallocate(gate)
        if self.minimal_config is not None:
            out = ttnn.experimental.minimal_matmul(
                gated, self.Wo, config=self.minimal_config, compute_kernel_config=self.compute_kernel_config
            )
        else:
            out = ttnn.linear(gated, self.Wo, **self._down_kwargs)
        ttnn.deallocate(gated)
        return out

    def _sharded(self, normed):
        plan = self.shard
        activated = ttnn.linear(
            normed,
            self.Wi_act,
            program_config=plan.act_matmul,
            memory_config=plan.intermediate_memory,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.act_dtype,
        )
        if self.gelu_separate:
            pre = activated
            activated = ttnn.gelu(pre, fast_and_approximate_mode=self.gelu_approx, memory_config=plan.intermediate_memory)
            ttnn.deallocate(pre)
        gate = ttnn.linear(
            normed,
            self.Wi_gate,
            program_config=plan.gate_matmul,
            memory_config=plan.intermediate_memory,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.act_dtype,
        )
        gated = ttnn.mul(activated, gate, memory_config=plan.intermediate_memory)
        ttnn.deallocate(activated)
        ttnn.deallocate(gate)
        out = ttnn.linear(
            gated,
            self.Wo,
            program_config=plan.down_matmul,
            memory_config=plan.hidden_memory,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(gated)
        return out
