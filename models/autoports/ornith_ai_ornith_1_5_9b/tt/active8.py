# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validated prefix-eight hybrid execution over authoritative B32 state."""

import copy

import torch

import ttnn

from .precision import load_precision

PREFIX = (True,) * 8 + (False,) * 24


def _precision_eligible(precision):
    selected = load_precision()
    mlp8 = copy.deepcopy(selected)
    for group in ("attention", "mlp_gate_up", "mlp_down"):
        mlp8["weight_groups"][group] = "bfloat8_b"
    gate_up_hifi2 = copy.deepcopy(mlp8)
    gate_up_hifi2["config_id"] = "mlp8_gate_up_hifi2_experiment"
    gate_up_hifi2["compute_fidelities"]["mlp_gate_up"] = "HiFi2"
    # GPQA precision arm: attention fidelity (which also covers the GDN
    # projections) raised to HiFi2 and the KV cache widened to BF16, on top
    # of the gate/up HiFi2 policy. Listed here so the policy keeps the active8
    # path; an unlisted policy silently falls back to the original trace.
    attn_hifi2_kv16 = copy.deepcopy(gate_up_hifi2)
    attn_hifi2_kv16["config_id"] = "mlp8_gate_up_hifi2_attn_hifi2_kv16"
    attn_hifi2_kv16["compute_fidelities"]["attention"] = "HiFi2"
    attn_hifi2_kv16["kv_cache_dtype"] = "bfloat16"
    # 2026-09-28: the policy every profile ships. Attention HiFi2 and gate/up
    # HiFi2 as above, a bfp8 LM head at HiFi2 in place of the bfp4/LoFi head,
    # and the bfp8 KV cache of the original selection (the bf16 cache cost
    # IFEval 89 -> 71 on one chip and 86 -> 71 on two). Same head geometry.
    head8_hifi2_kv8 = copy.deepcopy(gate_up_hifi2)
    head8_hifi2_kv8["config_id"] = "p150exp_c_head8_hifi2_kv8"
    head8_hifi2_kv8["compute_fidelities"]["attention"] = "HiFi2"
    head8_hifi2_kv8["weight_groups"]["lm_head"] = "bfloat8_b"
    head8_hifi2_kv8["compute_fidelities"]["lm_head"] = "HiFi2"
    head8_hifi2_kv8["kv_cache_dtype"] = "bfloat8_b"
    # Four-chip variant of the same policy on the 64-core head geometry: with the
    # bfp8 head, the 32-core geometry's prefill circular buffers overrun an L1
    # buffer of the TP4 path (2026-09-28).
    head8_hifi2_kv8_c64 = copy.deepcopy(head8_hifi2_kv8)
    head8_hifi2_kv8_c64["config_id"] = "p150x4_head8_hifi2_kv8_c64"
    head8_hifi2_kv8_c64["head_geometry"] = {"cores": 64, "columns": 32768, "in0_block_w": 1, "readers": 2}
    return precision in (selected, mlp8, gate_up_hifi2, attn_hifi2_kv16, head8_hifi2_kv8, head8_hifi2_kv8_c64)


def eligible(model, cache):
    """Only the full-model geometry and precision policies tested on silicon."""
    return (
        model.layer_indices == list(range(32))
        and tuple(model.mesh_device.shape) == (1, 4)
        and model.mesh_device.get_num_devices() == 4
        and model.mesh_device.arch() == ttnn.Arch.BLACKHOLE
        and (cache.batch_size, cache.context, cache.num_blocks) == (32, 262144, 4129)
        and model.page_block_size == 64
        and model.max_context == 262144
        and (model.dim, model.vocab_size) == (4096, 248320)
        and _precision_eligible(model.precision)
        and all(layer.mesh_config.residual == "replicated" for layer in cache.decode_layers)
    )


def buffers(layer):
    return [layer.recurrent_state, *layer.conv_state]


class Active8Mixer:
    def __init__(self, full, scratch, x8, indices, zero_output):
        self.full, self.scratch = full, scratch
        self.x8, self.indices, self.zero_output = x8, indices, zero_output
        self.targets = [full.recurrent_state] + [ttnn.reshape(value, [32, 1, 1, 2048]) for value in full.conv_state]
        self.sources = [scratch.recurrent_state] + [
            ttnn.reshape(value, [8, 1, 1, 2048]) for value in scratch.conv_state
        ]

    def forward(self, x):
        # Exact slice/mixer/indexed_fill/copyback operations of the measured gate.
        ttnn.slice(x, [0, 0, 0], [8, 1, self.full.cfg.dim], output_tensor=self.x8)
        for source, target in zip(buffers(self.full), buffers(self.scratch)):
            shape = list(source.shape)
            shape[0] = 8
            result = ttnn.slice(source, [0] * len(shape), shape, output_tensor=target)
            assert result.buffer_address() == target.buffer_address()
        output = self.scratch._gdn_decode(self.x8)
        for target, source in zip(self.targets, self.sources):
            merged = ttnn.indexed_fill(self.indices, target, source, dim=0)
            ttnn.copy(merged, target)
            ttnn.deallocate(merged)
        output4 = ttnn.reshape(output, [8, 1, 1, self.full.residual_width])
        assembled = ttnn.indexed_fill(self.indices, self.zero_output, output4, dim=0)
        ttnn.deallocate(output)
        return ttnn.reshape(assembled, [32, 1, self.full.residual_width])


class Active8State:
    """Scratch and proxies belong to one canonical cache and predate all traces."""

    def __init__(self, cache):
        self.cache = cache
        self.owners = []
        self.proxies = {}

    def prepare(self, model):
        indices = model.upload(torch.arange(8, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.owners.append(indices)
        for layer in self.cache.decode_layers:
            if layer.is_full_attention:
                continue
            scratch = copy.copy(layer)
            scratch.allocate_state(8)
            self.owners.extend(buffers(scratch))
            assert scratch.w is layer.w and scratch.ccl is layer.ccl
            assert scratch.recurrent_l1_intermediates == layer.recurrent_l1_intermediates
            x8 = model.upload(torch.zeros(8, 1, model.dim))
            self.owners.append(x8)
            zero = model.upload(torch.zeros(32, 1, 1, layer.residual_width))
            self.owners.append(zero)
            mixer = Active8Mixer(layer, scratch, x8, indices, zero)
            proxy = copy.copy(layer)
            proxy._gdn_decode = mixer.forward
            self.proxies[id(layer)] = proxy
        assert len(self.proxies) == 24
