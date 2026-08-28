# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused A/B for the FullLocal post-compute score/combine graph."""

import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric

_HIDDEN_SIZE = 2880
_TOP_K = 4
_REPEATS = 100


def _trace_candidate(mesh_device, candidate):
    warmup_output = candidate()
    ttnn.synchronize_device(mesh_device)
    warmup_output.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        output = candidate()
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    first = ttnn.to_torch(output).clone()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    second = ttnn.to_torch(output)

    started = time.perf_counter()
    for _ in range(_REPEATS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    wall_ms = 1000 * (time.perf_counter() - started) / _REPEATS
    ttnn.release_trace(mesh_device, trace_id)
    return first, second, wall_ms


@pytest.mark.parametrize("tokens", [1, 2, 32])
@parametrize_mesh_with_fabric([(1, 1)])
def test_full_local_fused_score_reduce_ab(mesh_device, device_params, tokens, reset_seeds):
    """Compare the primitive score graph with the dedicated fused score/reduce op."""
    del device_params, reset_seeds
    generator = torch.Generator().manual_seed(81893 + tokens)
    slots = (torch.randn((_TOP_K, tokens, _HIDDEN_SIZE), generator=generator) * 0.02).to(torch.bfloat16)
    scores = torch.softmax(torch.randn((1, tokens, _TOP_K), generator=generator), dim=-1).to(torch.bfloat16)
    indices = torch.randint(0, 128, (tokens, 1, 1, _TOP_K), generator=generator, dtype=torch.int16).to(torch.uint16)
    expert_mapping = torch.zeros((1, 128), dtype=torch.uint16)
    expected = (slots.float() * scores[0].transpose(0, 1).unsqueeze(-1).float()).sum(dim=0)

    tt_slots = ttnn.from_torch(
        slots,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_scores = ttnn.from_torch(
        scores,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_indices = ttnn.from_torch(
        indices,
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_zero_indices = ttnn.from_torch(
        torch.zeros_like(indices),
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def primitive():
        scores_expert_major = ttnn.transpose(tt_scores, -2, -1)
        score_scale = ttnn.reshape(scores_expert_major, (_TOP_K, tokens, 1))
        weighted = ttnn.mul(tt_slots, score_scale)
        reduced = ttnn.sum(weighted, dim=0)
        weighted.deallocate(True)
        scores_expert_major.deallocate(True)
        return ttnn.reshape(reduced, (1, 1, tokens, _HIDDEN_SIZE))

    def fused(output_memory_config, reduce_indices):
        slots4d = ttnn.reshape(tt_slots, (_TOP_K, 1, tokens, _HIDDEN_SIZE))
        padded = ttnn.tilize_with_val_padding(
            slots4d,
            output_tensor_shape=(_TOP_K, 1, 32, _HIDDEN_SIZE),
            pad_value=0.0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        scores4d = ttnn.reshape(tt_scores, (tokens, 1, 1, _TOP_K))
        outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            padded,
            reduce_indices,
            tt_mapping,
            reduce_dim=0,
            split_size=_HIDDEN_SIZE,
            cluster_axis=0,
            output_memory_config=output_memory_config,
            scores_tensor=scores4d,
            num_shared_experts=0,
        )
        padded.deallocate(True)
        return outputs[0]

    primitive_first, primitive_second, primitive_ms = _trace_candidate(mesh_device, primitive)
    fused_dram_first, fused_dram_second, fused_dram_ms = _trace_candidate(
        mesh_device, lambda: fused(ttnn.DRAM_MEMORY_CONFIG, tt_indices)
    )
    fused_l1_first, fused_l1_second, fused_l1_ms = _trace_candidate(
        mesh_device, lambda: fused(ttnn.L1_MEMORY_CONFIG, tt_indices)
    )
    zero_dram_first, zero_dram_second, zero_dram_ms = _trace_candidate(
        mesh_device, lambda: fused(ttnn.DRAM_MEMORY_CONFIG, tt_zero_indices)
    )
    zero_l1_first, zero_l1_second, zero_l1_ms = _trace_candidate(
        mesh_device, lambda: fused(ttnn.L1_MEMORY_CONFIG, tt_zero_indices)
    )
    primitive_actual = primitive_first[0, 0, :tokens].float()
    fused_dram_actual = fused_dram_first[0, 0, :tokens].float()
    fused_l1_actual = fused_l1_first[0, 0, :tokens].float()
    primitive_passed, primitive_pcc = comp_pcc(expected, primitive_actual, 0.999)
    fused_dram_passed, fused_dram_pcc = comp_pcc(expected, fused_dram_actual, 0.999)
    fused_l1_passed, fused_l1_pcc = comp_pcc(expected, fused_l1_actual, 0.999)
    ab_passed, ab_pcc = comp_pcc(primitive_actual, fused_dram_actual, 0.999)
    print(
        "FULL_LOCAL_COMBINE_AB "
        f"tokens={tokens} primitive_pcc={primitive_pcc} fused_dram_pcc={fused_dram_pcc} "
        f"fused_l1_pcc={fused_l1_pcc} ab_pcc={ab_pcc} primitive_trace_ms={primitive_ms:.6f} "
        f"actual_dram_trace_ms={fused_dram_ms:.6f} actual_l1_trace_ms={fused_l1_ms:.6f} "
        f"zero_dram_trace_ms={zero_dram_ms:.6f} zero_l1_trace_ms={zero_l1_ms:.6f}"
    )
    assert primitive_passed
    assert fused_dram_passed
    assert fused_l1_passed
    assert ab_passed
    assert torch.equal(primitive_first, primitive_second)
    assert torch.equal(fused_dram_first, fused_dram_second)
    assert torch.equal(fused_l1_first, fused_l1_second)
    assert torch.equal(zero_dram_first, zero_dram_second)
    assert torch.equal(zero_l1_first, zero_l1_second)
    assert torch.equal(fused_dram_first, zero_dram_first)
    assert torch.equal(fused_l1_first, zero_l1_first)
