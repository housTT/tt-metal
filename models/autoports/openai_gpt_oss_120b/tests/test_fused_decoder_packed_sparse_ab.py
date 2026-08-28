# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused A/B for the decode-only packed gate/up sparse-matmul rewrite.

This is intentionally a component experiment: the accepted functional MLP and
the candidate share the same router, down projection, routing weights, input,
and real GPT-OSS-120B dimensions.  Only the two gate/up sparse matmuls and two
bias adds are replaced with one host-packed sparse matmul and one bias add.
"""

import inspect
import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tt.functional_decoder import _FunctionalMLP
from models.autoports.openai_gpt_oss_120b.tt.fused_decoder import _FusedMLP
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu
from models.demos.gpt_oss.utils.general_utils import get_default_num_links
from models.demos.gpt_oss.utils.substate import substate


def _packed_sparse_decode(hidden_states, routing_weights, weights, config, program_config, packed_weight, packed_bias):
    batch_size = hidden_states.shape[1]
    seq_len = hidden_states.shape[2]
    sparsity = ttnn.to_layout(ttnn.unsqueeze_to_4D(routing_weights), ttnn.ROW_MAJOR_LAYOUT)
    output_tile = ttnn.Tile([32, 32])

    gate_up = ttnn.sparse_matmul(
        hidden_states,
        packed_weight,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], packed_weight.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=ttnn.bfloat8_b,
    )
    hidden_states.deallocate(True)
    gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, 1, 2 * config.intermediate_size))
    gate_up = ttnn.transpose(gate_up, 1, 2)
    gate_up = ttnn.reshape(gate_up, (batch_size, config.num_experts, 2 * config.intermediate_size))
    gate_up = ttnn.add(gate_up, packed_bias, output_tensor=gate_up)
    gate = ttnn.slice(
        gate_up,
        [0, 0, 0],
        [batch_size, config.num_experts, config.intermediate_size],
        [1, 1, 1],
    )
    up = ttnn.slice(
        gate_up,
        [0, 0, config.intermediate_size],
        [batch_size, config.num_experts, 2 * config.intermediate_size],
        [1, 1, 1],
    )
    gate_up.deallocate(True)

    down_input = apply_swiglu(gate, up, config)
    down_input = ttnn.transpose(down_input, 1, 0)
    down_input = ttnn.reshape(down_input, (1, config.num_experts, seq_len, weights.intermediate_size_per_device))
    down = ttnn.sparse_matmul(
        down_input,
        weights.down_proj,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        is_input_a_sparse=True,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], weights.down_proj.shape[-1], k=down_input.shape[-1]
        ),
        dtype=ttnn.bfloat8_b,
    )
    down_input.deallocate(True)
    sparsity.deallocate(True)

    next_states = ttnn.permute(down, (0, 2, 1, 3))
    next_states = ttnn.reshape(next_states, (batch_size, config.num_experts, config.hidden_size))
    next_states = ttnn.add(next_states, weights.down_proj_bias, output_tensor=next_states)
    routing_weights = ttnn.permute(routing_weights, (1, 0))
    routing_weights = ttnn.reshape(routing_weights, (batch_size, config.num_experts, 1))
    next_states = ttnn.mul(next_states, routing_weights, output_tensor=next_states)
    routing_weights.deallocate(True)
    next_states = ttnn.sum(next_states, dim=1)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    next_states = ttnn.unsqueeze_to_4D(next_states)
    return ttnn.reshape(
        next_states,
        (1, batch_size, seq_len, config.hidden_size),
        (1, batch_size, max(32, seq_len), config.hidden_size),
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PACKED_SPARSE_AB") != "1",
    reason="set GPT_OSS_120B_PACKED_SPARSE_AB=1 for the focused packed-sparse A/B",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_packed_gate_up_sparse_decode_ab(mesh_device, device_params, reset_seeds):
    del device_params, reset_seeds
    config = accepted._config()
    reference = accepted._reference_layer(config, layer_idx=0)
    mlp_state = substate(reference.state_dict(), "mlp")
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=1, ep=1, sp=1),
        prefill=ModeConfig(tp=1, ep=1, sp=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))
    baseline = _FunctionalMLP(mesh_device, config, mlp_state, ccl_manager, mesh_config, None)

    expert_state = substate(mlp_state, "experts")
    gate_up = expert_state["gate_up_proj"]
    gate_up_bias = expert_state["gate_up_proj_bias"]
    packed_weight_torch = torch.cat((gate_up[..., ::2], gate_up[..., 1::2]), dim=-1).reshape(
        1, config.num_local_experts, config.hidden_size, 2 * config.intermediate_size
    )
    packed_bias_torch = torch.cat((gate_up_bias[..., ::2], gate_up_bias[..., 1::2]), dim=-1).reshape(
        1, config.num_local_experts, 2 * config.intermediate_size
    )
    packed_weight = ttnn.as_tensor(
        packed_weight_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    packed_bias = ttnn.as_tensor(
        packed_bias_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    del packed_weight_torch, packed_bias_torch
    program_config = GPTOSSProgramConfig()

    generator = torch.Generator().manual_seed(120_511)
    hidden = torch.randn((1, 1, config.hidden_size), generator=generator) * 0.02
    with torch.no_grad():
        torch_output = reference.mlp(hidden)[0]
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    baseline_output = baseline(ttnn.clone(tt_hidden), is_decode=True)
    baseline_host = ttnn.to_torch(baseline_output)[0, 0, :1]
    baseline_output.deallocate(True)

    def accepted_component():
        return baseline(ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG), is_decode=True)

    accepted_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    accepted_traced_output = accepted_component()
    ttnn.end_trace_capture(mesh_device, accepted_trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, accepted_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    def candidate():
        candidate_input = ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        expert_indices, routing_weights = baseline.router(candidate_input, use_throughput_experts=False)
        expert_indices.deallocate(True)
        return _packed_sparse_decode(
            candidate_input,
            routing_weights,
            baseline.experts.weights,
            baseline.experts.config,
            program_config,
            packed_weight,
            packed_bias,
        )

    candidate_output = candidate()
    ttnn.synchronize_device(mesh_device)
    candidate_host = ttnn.to_torch(candidate_output)[0, 0, :1]
    passing_ab, detail_ab = comp_pcc(baseline_host.float(), candidate_host.float(), 0.995)
    passing_torch, detail_torch = comp_pcc(torch_output.float(), candidate_host.float(), 0.995)
    assert passing_ab, f"packed vs accepted failed: {detail_ab}"
    assert passing_torch, f"packed vs Torch failed: {detail_torch}"
    print(f"PACKED_SPARSE_PCC accepted={detail_ab} torch={detail_torch}")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = candidate()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = ttnn.to_torch(traced_output)[0, 0, :1].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = ttnn.to_torch(traced_output)[0, 0, :1]
    assert torch.equal(first, second), "packed sparse trace replay was not bitwise deterministic"

    repeats = int(os.environ.get("GPT_OSS_120B_PACKED_SPARSE_REPEATS", "20"))
    signpost("ACCEPTED_SPARSE_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, accepted_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    accepted_wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("ACCEPTED_SPARSE_DECODE_END")
    signpost("PACKED_SPARSE_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("PACKED_SPARSE_DECODE_END")
    print(
        f"PACKED_SPARSE_TRACE repeats={repeats} accepted_wall_ms={accepted_wall_ms:.6f} "
        f"packed_wall_ms={wall_ms:.6f} deterministic=true"
    )
    accepted_traced_output.deallocate(True)
    ttnn.release_trace(mesh_device, accepted_trace_id)
    ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_PACKED_SPARSE_AB") != "1",
    reason="set GPT_OSS_120B_PACKED_SPARSE_AB=1 for the focused packed-sparse A/B",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_fused_mlp_uses_indexed_sparse_decode(mesh_device, device_params, reset_seeds):
    """Exercise the delivered indexed Top-K decode path, including trace replay."""
    del device_params, reset_seeds
    decode_source = inspect.getsource(_FusedMLP._decode_single_user)
    assert "use_throughput_experts=True" in decode_source
    assert "indices=expert_indices_rm" in decode_source
    assert "nnz=" not in decode_source
    config = accepted._config()
    reference = accepted._reference_layer(config, layer_idx=0)
    mlp = _FusedMLP(mesh_device, config, substate(reference.state_dict(), "mlp"), None)

    generator = torch.Generator().manual_seed(120_511)
    hidden = torch.randn((1, 1, config.hidden_size), generator=generator) * 0.02
    with torch.no_grad():
        expected = reference.mlp(hidden)[0]
    tt_hidden = ttnn.from_torch(
        hidden.reshape(1, 1, 1, config.hidden_size),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    output = mlp(ttnn.clone(tt_hidden), is_decode=True)
    ttnn.synchronize_device(mesh_device)
    actual = ttnn.to_torch(output)[0, 0, :1]
    passing, detail = comp_pcc(expected.float(), actual.float(), 0.995)
    assert passing, f"delivered indexed sparse decode vs Torch failed: {detail}"

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = mlp(ttnn.clone(tt_hidden), is_decode=True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = ttnn.to_torch(traced_output)[0, 0, :1].clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(ttnn.to_torch(traced_output)[0, 0, :1], first)
    repeats = int(os.environ.get("GPT_OSS_120B_INDEXED_SPARSE_REPEATS", "20"))
    signpost("FUSED_MLP_INDEXED_SPARSE_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("FUSED_MLP_INDEXED_SPARSE_DECODE_END")
    ttnn.release_trace(mesh_device, trace_id)
    print(f"FUSED_MLP_INDEXED_SPARSE_DECODE {detail} wall_ms={wall_ms:.6f} deterministic=true")
