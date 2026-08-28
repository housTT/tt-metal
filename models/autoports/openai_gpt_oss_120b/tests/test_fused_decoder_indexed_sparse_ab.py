# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused A/B for indexed/gather sparse-matmul decode."""

import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_120b.tests import test_functional_decoder as accepted
from models.autoports.openai_gpt_oss_120b.tests.test_fused_decoder_packed_sparse_ab import _packed_sparse_decode
from models.autoports.openai_gpt_oss_120b.tt.functional_decoder import _FunctionalMLP
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.expert_configs import GPTOSSProgramConfig
from models.demos.gpt_oss.tt.experts.operations import apply_swiglu
from models.demos.gpt_oss.utils.general_utils import get_default_num_links
from models.demos.gpt_oss.utils.substate import substate


def _indexed_sparse_decode(
    hidden_states,
    router,
    config,
    program_config,
    packed_weight,
    packed_bias_table,
    down_weight,
    down_bias_table,
    unused_sparsity,
):
    # `nnz` must never be supplied in indexed mode.  The fixed loop count is
    # the four entries in this single-stick UINT16 ROW_MAJOR tensor.
    expert_indices, routing_scores = router(hidden_states, use_throughput_experts=True)
    expert_indices_rm = ttnn.to_layout(expert_indices, ttnn.ROW_MAJOR_LAYOUT)
    expert_indices.deallocate(True)
    expert_indices_rm = ttnn.reshape(expert_indices_rm, (1, 1, 1, config.num_experts_per_tok))
    embedding_indices = ttnn.typecast(expert_indices_rm, ttnn.uint32)
    output_tile = ttnn.Tile([32, 32])

    gate_up = ttnn.sparse_matmul(
        hidden_states,
        packed_weight,
        sparsity=unused_sparsity,
        indices=expert_indices_rm,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_gate_up_config(
            hidden_states.shape[2], packed_weight.shape[3], k=hidden_states.shape[-1]
        ),
        dtype=ttnn.bfloat8_b,
    )
    hidden_states.deallocate(True)
    gate_up = ttnn.reshape(gate_up, (1, config.num_experts_per_tok, 2 * config.intermediate_size))
    gate_up_bias = ttnn.embedding(
        embedding_indices,
        packed_bias_table,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    gate_up = ttnn.add(gate_up, gate_up_bias, output_tensor=gate_up)
    gate_up_bias.deallocate(True)
    gate = ttnn.slice(
        gate_up,
        [0, 0, 0],
        [1, config.num_experts_per_tok, config.intermediate_size],
        [1, 1, 1],
    )
    up = ttnn.slice(
        gate_up,
        [0, 0, config.intermediate_size],
        [1, config.num_experts_per_tok, 2 * config.intermediate_size],
        [1, 1, 1],
    )
    gate_up.deallocate(True)

    down_input = apply_swiglu(gate, up, config)
    down_input = ttnn.reshape(down_input, (1, config.num_experts_per_tok, 1, config.intermediate_size))
    down = ttnn.sparse_matmul(
        down_input,
        down_weight,
        sparsity=unused_sparsity,
        indices=expert_indices_rm,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=program_config.get_decode_down_config(
            down_input.shape[2], down_weight.shape[-1], k=down_input.shape[-1]
        ),
        dtype=ttnn.bfloat8_b,
    )
    down_input.deallocate(True)
    expert_indices_rm.deallocate(True)
    down = ttnn.reshape(down, (1, config.num_experts_per_tok, config.hidden_size))
    down_bias = ttnn.embedding(
        embedding_indices,
        down_bias_table,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    embedding_indices.deallocate(True)
    down = ttnn.add(down, down_bias, output_tensor=down)
    down_bias.deallocate(True)

    routing_scores_rm = ttnn.to_layout(routing_scores, ttnn.ROW_MAJOR_LAYOUT)
    routing_scores.deallocate(True)
    routing_scores_rm = ttnn.reshape(routing_scores_rm, (1, config.num_experts_per_tok, 1))
    down = ttnn.mul(down, routing_scores_rm, output_tensor=down)
    routing_scores_rm.deallocate(True)
    output = ttnn.sum(down, dim=1)
    output = ttnn.unsqueeze_to_4D(output)
    output = ttnn.unsqueeze_to_4D(output)
    return ttnn.reshape(
        output,
        (1, 1, 1, config.hidden_size),
        (1, 1, ttnn.TILE_SIZE, config.hidden_size),
    )


@pytest.mark.skipif(
    os.environ.get("GPT_OSS_120B_INDEXED_SPARSE_AB") != "1",
    reason="set GPT_OSS_120B_INDEXED_SPARSE_AB=1 for indexed sparse A/B",
)
@parametrize_mesh_with_fabric([(1, 1)])
def test_indexed_sparse_decode_ab(mesh_device, device_params, reset_seeds):
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
    # Logical B1 is physically tile-padded to 32 rows.  Avoid the separate
    # batch-32-only fused router and exercise the accepted standard router.
    baseline.router.use_fused_op = False

    expert_state = substate(mlp_state, "experts")
    gate_up = expert_state["gate_up_proj"]
    gate_up_bias = expert_state["gate_up_proj_bias"]
    packed_weight_torch = torch.cat((gate_up[..., ::2], gate_up[..., 1::2]), dim=-1).reshape(
        1, config.num_local_experts, config.hidden_size, 2 * config.intermediate_size
    )
    packed_bias_torch = torch.cat((gate_up_bias[..., ::2], gate_up_bias[..., 1::2]), dim=-1)
    packed_weight = ttnn.as_tensor(
        packed_weight_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    packed_bias_table = ttnn.as_tensor(
        packed_bias_torch,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    packed_bias_dense = ttnn.as_tensor(
        packed_bias_torch.reshape(1, config.num_local_experts, 2 * config.intermediate_size),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    down_bias_table = ttnn.as_tensor(
        expert_state["down_proj_bias"],
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    unused_sparsity = ttnn.as_tensor(
        torch.zeros((1, 1, 1, config.num_local_experts), dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    del packed_weight_torch, packed_bias_torch, gate_up, gate_up_bias
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

    def dense_candidate():
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
            packed_bias_dense,
        )

    def indexed_candidate():
        return _indexed_sparse_decode(
            ttnn.clone(tt_hidden, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            baseline.router,
            baseline.experts.config,
            program_config,
            packed_weight,
            packed_bias_table,
            baseline.experts.weights.down_proj,
            down_bias_table,
            unused_sparsity,
        )

    dense_output = dense_candidate()
    indexed_output = indexed_candidate()
    ttnn.synchronize_device(mesh_device)
    dense_host = ttnn.to_torch(dense_output)[0, 0, :1]
    indexed_host = ttnn.to_torch(indexed_output)[0, 0, :1]
    passed_ab, detail_ab = comp_pcc(dense_host.float(), indexed_host.float(), 0.995)
    passed_torch, detail_torch = comp_pcc(torch_output.float(), indexed_host.float(), 0.995)
    print(f"INDEXED_SPARSE_PCC dense={detail_ab} torch={detail_torch}")
    dense_output.deallocate(True)
    indexed_output.deallocate(True)

    repeats = int(os.environ.get("GPT_OSS_120B_INDEXED_SPARSE_REPEATS", "20"))
    dense_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    dense_traced_output = dense_candidate()
    ttnn.end_trace_capture(mesh_device, dense_trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(mesh_device, dense_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PACKED_DENSE_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, dense_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    dense_wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("PACKED_DENSE_DECODE_END")
    ttnn.release_trace(mesh_device, dense_trace_id)
    dense_traced_output.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    # TTNN trace buffers must not overlap independently captured candidates:
    # a later allocation can otherwise be corrupted when the earlier trace is
    # replayed.  Capture indexed only after packed has been fully released.
    indexed_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    indexed_traced_output = indexed_candidate()
    ttnn.end_trace_capture(mesh_device, indexed_trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, indexed_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = ttnn.to_torch(indexed_traced_output)[0, 0, :1].clone()
    ttnn.execute_trace(mesh_device, indexed_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = ttnn.to_torch(indexed_traced_output)[0, 0, :1]
    deterministic = torch.equal(first, second)

    signpost("INDEXED_SPARSE_DECODE")
    started = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, indexed_trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    indexed_wall_ms = 1000 * (time.perf_counter() - started) / repeats
    signpost("INDEXED_SPARSE_DECODE_END")
    print(
        f"INDEXED_SPARSE_TRACE repeats={repeats} dense_wall_ms={dense_wall_ms:.6f} "
        f"indexed_wall_ms={indexed_wall_ms:.6f} deterministic={deterministic}"
    )
    ttnn.release_trace(mesh_device, indexed_trace_id)
    indexed_traced_output.deallocate(True)

    assert deterministic, "indexed sparse trace replay was not bitwise deterministic"
    assert passed_ab, f"indexed vs packed-dense failed: {detail_ab}"
    assert passed_torch, f"indexed vs Torch failed: {detail_torch}"
