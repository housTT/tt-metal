# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Focused coverage for the all-local-expert unified routed FFN program."""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc


SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

MULTICHIP_MESH_PARAMS = [
    pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="1x4"),
]

TWO_CHIP_MESH_PARAMS = [
    pytest.param((1, 2), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="1x2"),
]


def _to_device(mesh_device, tensor, *, dtype, layout):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _reference_active_rows(dispatched, counts, offsets, local_to_global, weights):
    outputs = []
    for local_expert, global_expert in enumerate(local_to_global):
        count = counts[global_expert]
        if count == 0:
            continue
        start = offsets[global_expert]
        x = dispatched[start : start + count]
        gate = x @ weights[local_expert]["gate"]
        up = x @ weights[local_expert]["up"]
        outputs.append((F.silu(gate) * up) @ weights[local_expert]["down"])
    return torch.cat(outputs)


def _actual_active_rows(output, counts, offsets, local_to_global):
    outputs = []
    for global_expert in local_to_global:
        count = counts[global_expert]
        if count:
            start = offsets[global_expert]
            outputs.append(output[start : start + count])
    return torch.cat(outputs)


def _minimal_fused_args(mesh_device, *, counts_layout=ttnn.ROW_MAJOR_LAYOUT, index_layout=ttnn.ROW_MAJOR_LAYOUT):
    """Build one-expert inputs used by host-validation regressions."""
    torch.manual_seed(101)
    counts = torch.zeros((1, 32), dtype=torch.int32)
    counts[0, 0] = 32
    offsets = torch.arange(32, dtype=torch.int32).reshape(1, 32) * 32
    local_to_global = torch.arange(32, dtype=torch.int32).reshape(1, 32)
    weight = torch.randn(32, 32) * 0.04
    return (
        _to_device(mesh_device, torch.randn(32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        _to_device(mesh_device, offsets, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        _to_device(mesh_device, counts, dtype=ttnn.uint32, layout=counts_layout),
        _to_device(mesh_device, local_to_global, dtype=ttnn.uint32, layout=index_layout),
        [_to_device(mesh_device, weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)],
        [_to_device(mesh_device, weight + 0.01, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)],
        [_to_device(mesh_device, weight - 0.01, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)],
    )


def _run_fused_case(
    mesh_device,
    *,
    emb_dim,
    hidden_dim,
    max_tokens_per_expert,
    counts,
    offsets,
    local_to_global,
    input_dtype,
    input_layout,
    weight_seeds,
    zero_local_experts=(),
    assert_program_cache_reuse=False,
):
    """Run one fused-list shape and compare every replicated device shard."""
    num_experts = len(local_to_global)
    assert len(counts) == len(offsets) == num_experts
    total_tokens = max(offsets) + max_tokens_per_expert
    dispatched = torch.randn(total_tokens, emb_dim)

    # TILE input aliases the output. Fill zero-token experts' complete regions
    # with an exact sentinel so the regression checks that an expert skipped in
    # the middle of the fused loop performs no writes, not merely that its rows
    # are omitted from the PCC comparison.
    for local_expert in zero_local_experts:
        global_expert = local_to_global[local_expert]
        assert counts[global_expert] == 0
        start = offsets[global_expert]
        dispatched[start : start + max_tokens_per_expert] = 4.0

    counts_tt = _to_device(
        mesh_device, torch.tensor(counts, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    offsets_tt = _to_device(
        mesh_device, torch.tensor(offsets, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    local_to_global_tt = _to_device(
        mesh_device,
        torch.tensor(local_to_global, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    program_cache_entries = []
    active_outputs = []
    for weight_seed in weight_seeds:
        generator = torch.Generator().manual_seed(weight_seed)
        weights = [
            {
                "gate": torch.randn(emb_dim, hidden_dim, generator=generator) * 0.04,
                "up": torch.randn(emb_dim, hidden_dim, generator=generator) * 0.04,
                "down": torch.randn(hidden_dim, emb_dim, generator=generator) * 0.04,
            }
            for _ in range(num_experts)
        ]
        expected = _reference_active_rows(dispatched, counts, offsets, local_to_global, weights)

        dispatched_tt = _to_device(mesh_device, dispatched, dtype=input_dtype, layout=input_layout)
        before = [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(dispatched_tt)]
        gate_tt = [
            _to_device(mesh_device, expert["gate"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for expert in weights
        ]
        up_tt = [
            _to_device(mesh_device, expert["up"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for expert in weights
        ]
        down_tt = [
            _to_device(mesh_device, expert["down"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for expert in weights
        ]

        output_tt = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched_tt,
            offsets_tt,
            counts_tt,
            local_to_global_tt,
            gate_tt,
            up_tt,
            down_tt,
            max_dispatched_tokens_per_expert=max_tokens_per_expert,
        )
        program_cache_entries.append(mesh_device.num_program_cache_entries())
        outputs = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output_tt)]
        assert len(outputs) == mesh_device.get_num_devices()
        for device_id, output in enumerate(outputs):
            actual = _actual_active_rows(output, counts, offsets, local_to_global)
            passing, pcc = comp_pcc(expected, actual, 0.97)
            assert passing, f"device {device_id} all-expert fused output PCC below threshold: {pcc}"

            for local_expert in zero_local_experts:
                global_expert = local_to_global[local_expert]
                start = offsets[global_expert]
                stop = start + max_tokens_per_expert
                torch.testing.assert_close(output[start:stop], before[device_id][start:stop], rtol=0, atol=0)

        active_outputs.append(_actual_active_rows(outputs[0], counts, offsets, local_to_global).clone())

    if assert_program_cache_reuse:
        assert len(program_cache_entries) >= 2, "cache-reuse coverage requires at least two calls"
        assert len(set(program_cache_entries)) == 1, (
            "same-spec calls with replacement weights must not add a program-cache entry; "
            f"observed counts {program_cache_entries}"
        )
        assert not torch.equal(
            active_outputs[0], active_outputs[-1]
        ), "replacement expert weights must change the active output, proving cached base-address patching"


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_unified_routed_expert_moe_fuses_weight_lists_and_updates_cached_addresses(mesh_device, device_params):
    """Three local experts share one launch; the middle local expert has zero tokens.

    A second same-shape call replaces every weight tensor, exercising cached
    runtime-argument override for all three gate/up/down address arrays.
    """

    # Local experts map to non-identity global ids. This guards the fused loop's
    # consecutive local-id traversal independently from shared-buffer placement.
    # In local order their counts are active / zero / active, so the last expert
    # proves the loop resumes correctly after a skipped middle expert.
    local_to_global = [2, 0, 1]
    counts = [0, 48, 32]
    offsets = [64, 128, 0]
    torch.manual_seed(7)
    _run_fused_case(
        mesh_device,
        emb_dim=512,
        hidden_dim=352,
        max_tokens_per_expert=64,
        counts=counts,
        offsets=offsets,
        local_to_global=local_to_global,
        input_dtype=ttnn.bfloat8_b,
        input_layout=ttnn.TILE_LAYOUT,
        weight_seeds=(11, 29),
        zero_local_experts=(1,),
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", MULTICHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_unified_routed_expert_moe_row_major_production_layout_on_1x4(mesh_device, device_params):
    """The production ROW_MAJOR bf16 dispatch layout runs identically on every chip."""
    torch.manual_seed(17)
    _run_fused_case(
        mesh_device,
        emb_dim=512,
        hidden_dim=352,
        max_tokens_per_expert=64,
        counts=[0, 48, 32],
        offsets=[64, 128, 0],
        local_to_global=[2, 0, 1],
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_seeds=(41,),
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_unified_routed_expert_moe_e64_row_major_reuses_cache_and_patches_expert_63_weights(mesh_device, device_params):
    """E64 cache hits patch all weight bases, including the last expert."""
    num_experts = 64
    counts = [0] * num_experts
    counts[63] = 32
    torch.manual_seed(19)
    _run_fused_case(
        mesh_device,
        emb_dim=32,
        hidden_dim=32,
        max_tokens_per_expert=32,
        counts=counts,
        offsets=[expert * 32 for expert in range(num_experts)],
        local_to_global=list(range(num_experts)),
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_seeds=(43, 71),
        assert_program_cache_reuse=True,
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_unified_routed_expert_moe_partitions_more_than_64_bias_free_experts(mesh_device, device_params):
    """The public composite partitions 65 experts into fused groups of 64 + 1."""
    num_experts = 65
    counts = [0] * num_experts
    # Active work on both sides of the partition boundary proves that the
    # second program uses first_local_expert_id=64 rather than restarting at 0.
    counts[0] = counts[63] = counts[64] = 32
    torch.manual_seed(23)
    _run_fused_case(
        mesh_device,
        emb_dim=32,
        hidden_dim=32,
        max_tokens_per_expert=32,
        counts=counts,
        offsets=[expert * 32 for expert in range(num_experts)],
        local_to_global=list(range(num_experts)),
        input_dtype=ttnn.bfloat16,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weight_seeds=(47,),
    )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize(
    "bad_aux, expected_error",
    [
        ("counts", "counts must be ROW_MAJOR"),
        ("global_expert_idx_table", "global_expert_idx_table must be ROW_MAJOR"),
    ],
)
def test_unified_routed_expert_moe_rejects_tiled_count_and_index_vectors(
    mesh_device, device_params, bad_aux, expected_error, expect_error
):
    layouts = {
        "counts_layout": ttnn.TILE_LAYOUT if bad_aux == "counts" else ttnn.ROW_MAJOR_LAYOUT,
        "index_layout": ttnn.TILE_LAYOUT if bad_aux == "global_expert_idx_table" else ttnn.ROW_MAJOR_LAYOUT,
    }
    args = _minimal_fused_args(mesh_device, **layouts)
    with expect_error(RuntimeError, expected_error):
        ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            *args,
            max_dispatched_tokens_per_expert=32,
        )


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", TWO_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_unified_routed_expert_moe_cache_hit_rejects_cross_device_counts(mesh_device, device_params, expect_error):
    """A same-spec replacement tensor cannot bypass ownership checks on a cache hit."""
    owner_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
    other_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))
    args = list(_minimal_fused_args(owner_mesh))

    ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        *args,
        max_dispatched_tokens_per_expert=32,
    )
    ttnn.synchronize_device(owner_mesh)
    cache_entries = owner_mesh.num_program_cache_entries()

    other_counts = torch.zeros((1, 32), dtype=torch.int32)
    other_counts[0, 0] = 32
    args[2] = _to_device(
        other_mesh,
        other_counts,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with expect_error(RuntimeError, "counts must be on the same device as x"):
        ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            *args,
            max_dispatched_tokens_per_expert=32,
        )
    assert owner_mesh.num_program_cache_entries() == cache_entries
