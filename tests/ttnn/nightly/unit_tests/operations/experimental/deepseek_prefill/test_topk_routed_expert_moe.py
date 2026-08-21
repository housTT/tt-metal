# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device correctness coverage for router-index-native prefill MoE.

The references in this file deliberately start from the preserved top-k
``(weight, global expert id)`` pairs.  They therefore catch regressions that a
dense-routing reconstruction could hide: local-expert remapping, slot order,
valid-token padding, expert-region padding, and assignment-addressed workspace
placement are all checked explicitly.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import comp_pcc

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]

TWO_CHIP_MESH_PARAMS = [
    pytest.param((1, 2), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="1x2"),
]

FOUR_CHIP_MESH_PARAMS = [
    pytest.param((1, 4), {"fabric_config": ttnn.FabricConfig.DISABLED}, id="1x4"),
]

UINT32_MAX = (1 << 32) - 1


def _to_device(mesh_device, tensor, *, dtype, layout, mesh_mapper=None):
    if mesh_mapper is None:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _device_shards(tensor):
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def _as_u32(tensor):
    """Normalize Torch's signed/unsigned UINT32 host representations."""

    return torch.bitwise_and(tensor.reshape(-1).to(torch.int64), UINT32_MAX)


def _mapping_from_local_order(num_global_experts, local_to_global):
    """Return the native-map convention: local id or exact UINT32_MAX."""

    num_local_experts = len(local_to_global)
    mapping = torch.full((num_global_experts,), UINT32_MAX, dtype=torch.int64)
    for local_expert, global_expert in enumerate(local_to_global):
        mapping[global_expert] = local_expert
    return mapping


def _unique_random_topk(tokens, num_global_experts, topk, *, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(num_global_experts, generator=generator)[:topk] for _ in range(tokens)]).to(
        torch.int64
    )


def _force_expert_in_slot(topk_indices, expert, *, slot=0):
    """Force one expert into a slot without violating router top-k uniqueness."""

    result = topk_indices.clone()
    for row in result:
        existing = torch.nonzero(row == expert, as_tuple=False)
        if existing.numel():
            existing_slot = int(existing[0, 0])
            row[slot], row[existing_slot] = row[existing_slot].clone(), row[slot].clone()
        else:
            row[slot] = expert
    assert torch.all(torch.diff(torch.sort(result, dim=-1).values, dim=-1) != 0)
    return result


@dataclass(frozen=True)
class PlanReference:
    compact_x: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
    local_to_global: torch.Tensor
    assignments: torch.Tensor
    inverse: torch.Tensor
    valid: torch.Tensor


def _reference_plan(x, topk_indices, mapping, num_local_experts, valid_tokens):
    """Mirror the deterministic expert-major device planner."""

    tokens, hidden = x.shape
    topk = topk_indices.shape[-1]
    num_global_experts = mapping.numel()
    capacity = tokens * topk + 31 * num_local_experts

    local_to_global = torch.full((num_local_experts,), UINT32_MAX, dtype=torch.int64)
    for global_expert, local_expert in enumerate(mapping.tolist()):
        if local_expert < num_local_experts:
            assert local_to_global[local_expert] == UINT32_MAX
            local_to_global[local_expert] = global_expert
    assert torch.all(local_to_global != UINT32_MAX)

    counts = torch.zeros(num_global_experts, dtype=torch.int64)
    for token in range(valid_tokens):
        for global_expert in topk_indices[token].tolist():
            if mapping[global_expert] < num_local_experts:
                counts[global_expert] += 1

    offsets = torch.zeros(num_global_experts, dtype=torch.int64)
    region_end = 0
    for global_expert in local_to_global.tolist():
        offsets[global_expert] = region_end
        region_end += ((int(counts[global_expert]) + 31) // 32) * 32
    assert region_end <= capacity

    assignments = torch.full((capacity,), UINT32_MAX, dtype=torch.int64)
    inverse = torch.zeros(topk * tokens, dtype=torch.int64)
    valid = torch.zeros(topk * tokens, dtype=torch.int64)
    cursors = torch.zeros(num_local_experts, dtype=torch.int64)
    compact_x = torch.zeros(capacity, hidden, dtype=x.dtype)

    for token in range(valid_tokens):
        for slot, global_expert in enumerate(topk_indices[token].tolist()):
            local_expert = int(mapping[global_expert])
            if local_expert >= num_local_experts:
                continue
            packed_row = int(offsets[global_expert] + cursors[local_expert])
            cursors[local_expert] += 1
            assignment = token * topk + slot
            assignments[packed_row] = assignment
            inverse[slot * tokens + token] = packed_row
            valid[slot * tokens + token] = 1
            compact_x[packed_row] = x[token]

    return PlanReference(compact_x, counts, offsets, local_to_global, assignments, inverse, valid)


def _dispatch_inputs(mesh_device, x, topk_indices, mapping, *, mapping_mapper=None):
    return (
        _to_device(
            mesh_device,
            x.reshape(1, 1, *x.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        _to_device(
            mesh_device,
            topk_indices.reshape(1, 1, *topk_indices.shape).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
        ),
        _to_device(
            mesh_device,
            mapping.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapping_mapper,
        ),
    )


def _assert_dispatch_matches(outputs, reference, *, materialize_x=True, device_index=0):
    output_shards = [_device_shards(output) for output in outputs]
    if materialize_x:
        actual_x = output_shards[0][device_index].reshape(-1, reference.compact_x.shape[-1])
        torch.testing.assert_close(
            actual_x.float(),
            reference.compact_x.to(torch.bfloat16).float(),
            rtol=0,
            atol=0,
        )
    else:
        assert tuple(output_shards[0][device_index].shape) == (1, 1, 1, reference.compact_x.shape[-1])

    expected = (
        reference.counts,
        reference.offsets,
        reference.local_to_global,
        reference.assignments,
        reference.inverse,
        reference.valid,
    )
    for output_index, expected_tensor in enumerate(expected, start=1):
        actual = _as_u32(output_shards[output_index][device_index])
        torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)


def _planner_case(case):
    tokens, hidden, topk, num_global_experts = 64, 32, 8, 64
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(100 + len(case)))
    valid_tokens = tokens

    if case == "random":
        local_to_global = [1, 13, 42, 63]
        indices = _unique_random_topk(tokens, num_global_experts, topk, seed=11)
    elif case == "skew":
        local_to_global = [5, 17, 41, 63]
        indices = _unique_random_topk(tokens, num_global_experts, topk, seed=13)
        # Keep rows realistic (unique experts) while forcing a hot local expert.
        for token in range(tokens):
            row = indices[token]
            where_hot = torch.nonzero(row == 5)
            if where_hot.numel():
                row[int(where_hot[0])] = row[0]
            row[0] = 5
    elif case == "all-local-nonidentity":
        num_global_experts = 8
        local_to_global = [3, 0, 7, 2, 5, 1, 6, 4]
        indices = _unique_random_topk(tokens, num_global_experts, topk, seed=17)
    elif case == "zero-local":
        local_to_global = list(range(56, 64))
        indices = _unique_random_topk(tokens, 8, topk, seed=19)
    elif case == "expert63-nonidentity":
        local_to_global = [63, 5, 31, 0]
        indices = _unique_random_topk(tokens, num_global_experts, topk, seed=23)
        indices = _force_expert_in_slot(indices, 63)
    elif case == "padded-valid-tokens":
        local_to_global = [63, 5, 31, 0]
        indices = _unique_random_topk(tokens, num_global_experts, topk, seed=29)
        valid_tokens = 35
        # Every padded row looks local; none may enter counts or assignments.
        indices[valid_tokens:] = torch.tensor([63, 5, 31, 0, 63, 5, 31, 0])
    else:
        raise AssertionError(f"unknown planner case {case}")

    mapping = _mapping_from_local_order(num_global_experts, local_to_global)
    return x, indices, mapping, len(local_to_global), valid_tokens


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize(
    "case",
    [
        pytest.param("random", id="random"),
        pytest.param("skew", id="skew"),
        pytest.param("all-local-nonidentity", id="all-local-nonidentity"),
        pytest.param("zero-local", id="zero-local"),
        pytest.param("expert63-nonidentity", id="expert63-nonidentity"),
        pytest.param("padded-valid-tokens", id="padded-valid-tokens"),
    ],
)
def test_topk_local_dispatch_matches_cpu_plan(mesh_device, device_params, case):
    x, indices, mapping, num_local_experts, valid_tokens = _planner_case(case)
    x_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    outputs = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt,
        indices_tt,
        mapping_tt,
        num_local_experts,
        valid_tokens=valid_tokens,
    )
    reference = _reference_plan(x, indices, mapping, num_local_experts, valid_tokens)
    _assert_dispatch_matches(outputs, reference)


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_local_dispatch_cache_hit_patches_inputs_valid_tokens_and_all_output_addresses(mesh_device, device_params):
    tokens, hidden, topk, num_global_experts, num_local_experts = 64, 32, 8, 64, 4
    x_first = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(31))
    x_second = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(37))
    indices_first = _unique_random_topk(tokens, num_global_experts, topk, seed=41)
    indices_second = _unique_random_topk(tokens, num_global_experts, topk, seed=43)
    mapping_first = _mapping_from_local_order(num_global_experts, [0, 7, 22, 45])
    mapping_second = _mapping_from_local_order(num_global_experts, [63, 41, 13, 2])

    first_inputs = _dispatch_inputs(mesh_device, x_first, indices_first, mapping_first.reshape(1, -1))
    first = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        *first_inputs, num_local_experts, valid_tokens=tokens
    )
    ttnn.synchronize_device(mesh_device)
    cache_entries = mesh_device.num_program_cache_entries()

    second_inputs = _dispatch_inputs(mesh_device, x_second, indices_second, mapping_second.reshape(1, -1))
    second = ttnn.experimental.deepseek_prefill.topk_local_dispatch(*second_inputs, num_local_experts, valid_tokens=37)
    ttnn.synchronize_device(mesh_device)

    assert mesh_device.num_program_cache_entries() == cache_entries
    for first_output, second_output in zip(first, second):
        first_address = ttnn.get_device_tensors(first_output)[0].buffer_address()
        second_address = ttnn.get_device_tensors(second_output)[0].buffer_address()
        assert first_address != second_address, "live outputs must force cached writer-address replacement"

    reference = _reference_plan(x_second, indices_second, mapping_second, num_local_experts, 37)
    _assert_dispatch_matches(second, reference)


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", FOUR_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_topk_local_dispatch_preserves_unique_per_device_plans_on_1x4(mesh_device, device_params):
    tokens, hidden, topk, num_global_experts, num_local_experts = 64, 32, 8, 64, 16
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(47))
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=53)

    mappings = []
    for device_index in range(4):
        owned = list(reversed(list(range(device_index, num_global_experts, 4))))
        mappings.append(_mapping_from_local_order(num_global_experts, owned))
    stacked_mappings = torch.stack(mappings).to(torch.int32)
    mapping_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    x_tt, indices_tt, mapping_tt = _dispatch_inputs(
        mesh_device, x, indices, stacked_mappings, mapping_mapper=mapping_mapper
    )

    outputs = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt,
        indices_tt,
        mapping_tt,
        num_local_experts,
        valid_tokens=tokens,
    )
    for device_index, mapping in enumerate(mappings):
        reference = _reference_plan(x, indices, mapping, num_local_experts, tokens)
        _assert_dispatch_matches(outputs, reference, device_index=device_index)

    per_device_counts = [_as_u32(shard) for shard in _device_shards(outputs[1])]
    assert len({tuple(counts.tolist()) for counts in per_device_counts}) == 4


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", TWO_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_topk_local_dispatch_cache_hit_rejects_cross_device_mapping(mesh_device, device_params, expect_error):
    owner = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
    other = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))
    x, indices, mapping, num_local_experts, valid_tokens = _planner_case("random")
    x_tt, indices_tt, mapping_tt = _dispatch_inputs(owner, x, indices, mapping.reshape(1, -1))

    ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt, indices_tt, mapping_tt, num_local_experts, valid_tokens=valid_tokens
    )
    ttnn.synchronize_device(owner)
    cache_entries = owner.num_program_cache_entries()

    wrong_mapping = _to_device(
        other,
        mapping.reshape(1, -1).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    with expect_error(RuntimeError, "global_to_local_expert must be on the same device as x"):
        ttnn.experimental.deepseek_prefill.topk_local_dispatch(
            x_tt, indices_tt, wrong_mapping, num_local_experts, valid_tokens=valid_tokens
        )
    assert owner.num_program_cache_entries() == cache_entries


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", TWO_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_topk_local_dispatch_rejects_replicated_map_on_cache_miss_and_hit(mesh_device, device_params, expect_error):
    x, indices, mapping, num_local_experts, valid_tokens = _planner_case("random")
    x_tt, indices_tt, replicated_mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    message = r"global_to_local_expert must be device-unique \(sharded\) on a multi-device mesh"

    cache_entries_before_miss = mesh_device.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.topk_local_dispatch(
            x_tt,
            indices_tt,
            replicated_mapping_tt,
            num_local_experts,
            valid_tokens=valid_tokens,
        )
    assert mesh_device.num_program_cache_entries() == cache_entries_before_miss

    other_mapping = _mapping_from_local_order(mapping.numel(), [63, 41, 13, 2])
    unique_mapping_tt = _to_device(
        mesh_device,
        torch.stack([mapping, other_mapping]).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt,
        indices_tt,
        unique_mapping_tt,
        num_local_experts,
        valid_tokens=valid_tokens,
    )
    ttnn.synchronize_device(mesh_device)
    cache_entries_after_prime = mesh_device.num_program_cache_entries()

    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.topk_local_dispatch(
            x_tt,
            indices_tt,
            replicated_mapping_tt,
            num_local_experts,
            valid_tokens=valid_tokens,
        )
    assert mesh_device.num_program_cache_entries() == cache_entries_after_prime


def _combine_reference(packed_y, topk_weights, inverse, valid, *, assignment_addressed=False):
    tokens, topk = topk_weights.shape
    output = torch.zeros(tokens, packed_y.shape[-1], dtype=torch.float32)
    packed_y = packed_y.to(torch.bfloat16).float()
    topk_weights = topk_weights.to(torch.bfloat16).float()
    for token in range(tokens):
        for slot in range(topk):
            map_index = slot * tokens + token
            if int(valid[map_index]) != 1:
                continue
            row = token * topk + slot if assignment_addressed else int(inverse[map_index])
            output[token] += packed_y[row] * topk_weights[token, slot]
    return output


def _combine_case(case, *, tokens=64, hidden=1024, topk=8, assignment_addressed=False):
    generator = torch.Generator().manual_seed(59 + len(case))
    capacity = tokens * topk if assignment_addressed else tokens * topk + 32
    packed_y = torch.randn(capacity, hidden, generator=generator) * 0.08
    weights = torch.softmax(torch.randn(tokens, topk, generator=generator), dim=-1)
    # Assignment-addressed combine must not consult this legacy inverse map.
    inverse = torch.full((topk * tokens,), UINT32_MAX if assignment_addressed else 0, dtype=torch.int64)
    valid = torch.zeros(topk * tokens, dtype=torch.int64)
    cursor = 0
    for token in range(tokens):
        for slot in range(topk):
            is_local = case == "all-local" or (case == "random" and (token + 3 * slot) % 5 != 0)
            if is_local:
                if not assignment_addressed:
                    inverse[slot * tokens + token] = cursor
                valid[slot * tokens + token] = 1
                cursor += 1
    return packed_y, weights, inverse, valid


def _combine_inputs(mesh_device, packed_y, weights, inverse, valid):
    return (
        _to_device(
            mesh_device,
            packed_y.reshape(1, 1, *packed_y.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        _to_device(
            mesh_device,
            weights.reshape(1, 1, *weights.shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        _to_device(
            mesh_device,
            inverse.reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        _to_device(
            mesh_device,
            valid.reshape(1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
    )


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("assignment_addressed", [False, True], ids=["packed-row", "assignment-addressed"])
@pytest.mark.parametrize("case", ["random", "all-local", "zero-local"])
def test_topk_local_combine_matches_cpu_and_zero_local_is_exact(mesh_device, device_params, case, assignment_addressed):
    packed_y, weights, inverse, valid = _combine_case(case, assignment_addressed=assignment_addressed)
    tokens, topk = weights.shape
    packed_y_tt, weights_tt, inverse_tt, valid_tt = _combine_inputs(mesh_device, packed_y, weights, inverse, valid)
    output_tt = ttnn.experimental.deepseek_prefill.topk_local_combine(
        packed_y_tt,
        weights_tt,
        inverse_tt,
        valid_tt,
        tokens=tokens,
        topk=topk,
        assignment_addressed=assignment_addressed,
    )
    actual = _device_shards(output_tt)[0].reshape(tokens, -1).float()
    expected = _combine_reference(packed_y, weights, inverse, valid, assignment_addressed=assignment_addressed)
    if case == "zero-local":
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    else:
        passing, pcc = comp_pcc(expected, actual, 0.98)
        assert passing, f"top-k local combine PCC below threshold: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_local_combine_cache_hit_replaces_all_inputs_and_optional_output(mesh_device, device_params):
    tokens, topk = 64, 8
    first_host = _combine_case("random", tokens=tokens, topk=topk, assignment_addressed=True)
    second_host = _combine_case("all-local", tokens=tokens, topk=topk, assignment_addressed=True)
    # Make the second cache-hit payload independent, including its routing weights and slot mask.
    second_host = (
        -second_host[0],
        torch.flip(second_host[1], dims=(-1,)),
        second_host[2],
        torch.roll(second_host[3], shifts=1),
    )
    first_inputs = _combine_inputs(mesh_device, *first_host)
    second_inputs = _combine_inputs(mesh_device, *second_host)

    prototype = ttnn.experimental.deepseek_prefill.topk_local_combine(
        *first_inputs,
        tokens=tokens,
        topk=topk,
        assignment_addressed=True,
    )
    first_output = ttnn.empty_like(prototype)
    second_output = ttnn.empty_like(prototype)

    first = ttnn.experimental.deepseek_prefill.topk_local_combine(
        *first_inputs,
        tokens=tokens,
        topk=topk,
        assignment_addressed=True,
        output=first_output,
    )
    ttnn.synchronize_device(mesh_device)
    cache_entries = mesh_device.num_program_cache_entries()

    second = ttnn.experimental.deepseek_prefill.topk_local_combine(
        *second_inputs,
        tokens=tokens,
        topk=topk,
        assignment_addressed=True,
        output=second_output,
    )
    ttnn.synchronize_device(mesh_device)

    assert mesh_device.num_program_cache_entries() == cache_entries
    assert (
        ttnn.get_device_tensors(first)[0].buffer_address() == ttnn.get_device_tensors(first_output)[0].buffer_address()
    )
    assert (
        ttnn.get_device_tensors(second)[0].buffer_address()
        == ttnn.get_device_tensors(second_output)[0].buffer_address()
    )
    assert ttnn.get_device_tensors(first)[0].buffer_address() != ttnn.get_device_tensors(second)[0].buffer_address()

    second_actual = _device_shards(second)[0].reshape(tokens, -1).float()
    second_expected = _combine_reference(*second_host, assignment_addressed=True)
    passing, pcc = comp_pcc(second_expected, second_actual, 0.98)
    assert passing, f"cache-hit assignment-addressed combine PCC below threshold: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", TWO_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_topk_local_combine_rejects_cross_device_output_on_cache_miss_and_hit(mesh_device, device_params, expect_error):
    owner = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
    other = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))
    tokens, topk, hidden = 32, 2, 1024
    host = _combine_case("random", tokens=tokens, hidden=hidden, topk=topk, assignment_addressed=True)
    owner_inputs = _combine_inputs(owner, *host)
    wrong_output = _to_device(
        other,
        torch.empty(1, 1, tokens, hidden),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    message = "optional_output must be on the same device as packed_y"

    cache_entries_before_miss = owner.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.topk_local_combine(
            *owner_inputs,
            tokens=tokens,
            topk=topk,
            assignment_addressed=True,
            output=wrong_output,
        )
    assert owner.num_program_cache_entries() == cache_entries_before_miss

    ttnn.experimental.deepseek_prefill.topk_local_combine(
        *owner_inputs,
        tokens=tokens,
        topk=topk,
        assignment_addressed=True,
    )
    ttnn.synchronize_device(owner)
    cache_entries_after_prime = owner.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        ttnn.experimental.deepseek_prefill.topk_local_combine(
            *owner_inputs,
            tokens=tokens,
            topk=topk,
            assignment_addressed=True,
            output=wrong_output,
        )
    assert owner.num_program_cache_entries() == cache_entries_after_prime


def _make_expert_weights(mesh_device, num_local_experts, emb_dim, expert_dim, *, seed):
    generator = torch.Generator().manual_seed(seed)
    host = []
    for _ in range(num_local_experts):
        host.append(
            {
                "gate": torch.randn(emb_dim, expert_dim, generator=generator) * 0.025,
                "up": torch.randn(emb_dim, expert_dim, generator=generator) * 0.025,
                "down": torch.randn(expert_dim, emb_dim, generator=generator) * 0.025,
            }
        )
    gate = [_to_device(mesh_device, weights["gate"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for weights in host]
    up = [_to_device(mesh_device, weights["up"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for weights in host]
    down = [_to_device(mesh_device, weights["down"], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT) for weights in host]
    # The CPU oracle must start from the values actually admitted to the
    # device. Comparing against pre-quantization FP32 weights masks BF8 load or
    # replacement errors behind an artificially loose PCC threshold.
    quantized_host = []
    for expert in range(num_local_experts):
        quantized_host.append(
            {
                "gate": _device_shards(gate[expert])[0].float(),
                "up": _device_shards(up[expert])[0].float(),
                "down": _device_shards(down[expert])[0].float(),
            }
        )
    return quantized_host, gate, up, down


def _moe_reference(x, indices, weights, mapping, expert_weights, valid_tokens):
    tokens, _ = x.shape
    topk = indices.shape[-1]
    output = torch.zeros_like(x, dtype=torch.float32)
    x = x.to(torch.bfloat16).float()
    weights = weights.to(torch.bfloat16).float()
    for token in range(valid_tokens):
        for slot in range(topk):
            global_expert = int(indices[token, slot])
            local_expert = int(mapping[global_expert])
            if local_expert >= len(expert_weights):
                continue
            expert = expert_weights[local_expert]
            gate = x[token] @ expert["gate"]
            up = x[token] @ expert["up"]
            y = (F.silu(gate) * up) @ expert["down"]
            output[token] += weights[token, slot] * y
    return output


def _explicit_native_device_oracle(
    x_tile_tt,
    indices_tt,
    topk_weights_tt,
    mapping_tt,
    gate_tt,
    up_tt,
    down_tt,
    *,
    num_local_experts,
    valid_tokens,
    topk,
    output=None,
):
    """Spell out the same admitted primitives as the public native composite."""

    tokens = x_tile_tt.shape[-2]
    x_rm_tt = ttnn.to_layout(
        x_tile_tt,
        ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dispatch = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_rm_tt,
        indices_tt,
        mapping_tt,
        num_local_experts,
        valid_tokens=valid_tokens,
        materialize_x=False,
    )
    workspace = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x_rm_tt,
        dispatch[2],
        dispatch[1],
        dispatch[3],
        gate_tt,
        up_tt,
        down_tt,
        tokens,
        packed_assignment_ids=dispatch[4],
        topk=topk,
    )
    weights_rm = ttnn.to_layout(
        ttnn.typecast(topk_weights_tt, ttnn.bfloat16),
        ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    combined = ttnn.experimental.deepseek_prefill.topk_local_combine(
        workspace,
        weights_rm,
        dispatch[5],
        dispatch[6],
        tokens=tokens,
        topk=topk,
        assignment_addressed=True,
        output=output,
    )
    return combined, dispatch, workspace


def _composite_case_inputs(
    mesh_device,
    *,
    local_to_global,
    x_seed,
    index_seed,
    routing_seed,
    expert_seed,
    tokens=32,
    emb_dim=1024,
    expert_dim=32,
    topk=2,
    num_global_experts=64,
):
    mapping = _mapping_from_local_order(num_global_experts, local_to_global)
    x = torch.randn(tokens, emb_dim, generator=torch.Generator().manual_seed(x_seed)) * 0.08
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=index_seed)
    indices = _force_expert_in_slot(indices, 63)
    routing_weights = torch.softmax(
        torch.randn(tokens, topk, generator=torch.Generator().manual_seed(routing_seed)), dim=-1
    )
    _, gate_tt, up_tt, down_tt = _make_expert_weights(
        mesh_device, len(local_to_global), emb_dim, expert_dim, seed=expert_seed
    )
    x_rm_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    return {
        "x": ttnn.to_layout(x_rm_tt, ttnn.TILE_LAYOUT),
        "indices": indices_tt,
        "weights": _to_device(
            mesh_device,
            routing_weights.reshape(1, 1, tokens, topk),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
        ),
        "mapping": mapping_tt,
        "mapping_host": mapping,
        "gate": gate_tt,
        "up": up_tt,
        "down": down_tt,
        "tokens": tokens,
        "topk": topk,
        "num_local_experts": len(local_to_global),
    }


def _run_composite(case, *, output=None):
    return ttnn.experimental.deepseek_prefill.topk_routed_expert_moe(
        case["x"],
        case["indices"],
        case["weights"],
        case["mapping"],
        case["gate"],
        case["up"],
        case["down"],
        num_local_experts=case["num_local_experts"],
        valid_tokens=case["tokens"],
        max_dispatched_tokens_per_expert=case["tokens"],
        output=output,
    )


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_native_planner_only_workspace_and_composite_toy_smoke_match_cpu(mesh_device, device_params):
    """Toy CPU smoke; the separate device-oracle gate below requires PCC >= .9999."""

    tokens, emb_dim, expert_dim, topk, num_global_experts = 32, 1024, 32, 2, 4
    local_to_global = [2, 0, 3, 1]
    mapping = _mapping_from_local_order(num_global_experts, local_to_global)
    x = torch.randn(tokens, emb_dim, generator=torch.Generator().manual_seed(61)) * 0.08
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=67)
    routing_weights = torch.softmax(torch.randn(tokens, topk, generator=torch.Generator().manual_seed(71)), dim=-1)
    expert_weights, gate_tt, up_tt, down_tt = _make_expert_weights(
        mesh_device, len(local_to_global), emb_dim, expert_dim, seed=73
    )

    x_rm_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    dispatch = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_rm_tt,
        indices_tt,
        mapping_tt,
        len(local_to_global),
        valid_tokens=tokens,
        materialize_x=False,
    )
    plan = _reference_plan(x, indices, mapping, len(local_to_global), tokens)
    _assert_dispatch_matches(dispatch, plan, materialize_x=False)

    workspace_tt = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        x_rm_tt,
        dispatch[2],
        dispatch[1],
        dispatch[3],
        gate_tt,
        up_tt,
        down_tt,
        tokens,
        packed_assignment_ids=dispatch[4],
        topk=topk,
    )
    workspace = _device_shards(workspace_tt)[0]
    assert tuple(workspace.shape) == (1, 1, tokens * topk, emb_dim)
    assert workspace_tt.dtype == ttnn.bfloat16
    assert workspace_tt.layout == ttnn.ROW_MAJOR_LAYOUT

    topk_weights_tt = _to_device(
        mesh_device,
        routing_weights.reshape(1, 1, tokens, topk),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )
    x_tile_tt = _to_device(
        mesh_device,
        x.reshape(1, 1, tokens, emb_dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    output_tt = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe(
        x_tile_tt,
        indices_tt,
        topk_weights_tt,
        mapping_tt,
        gate_tt,
        up_tt,
        down_tt,
        num_local_experts=len(local_to_global),
        valid_tokens=tokens,
        max_dispatched_tokens_per_expert=tokens,
    )
    actual = _device_shards(output_tt)[0].reshape(tokens, emb_dim).float()
    expected = _moe_reference(x, indices, routing_weights, mapping, expert_weights, tokens)
    passing, pcc = comp_pcc(expected, actual, 0.95)
    assert passing, f"toy CPU smoke PCC below threshold: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_public_composite_matches_explicit_native_device_oracle_at_pcc_9999(mesh_device, device_params):
    tokens, emb_dim, expert_dim, topk, num_global_experts = 32, 1024, 32, 2, 64
    local_to_global = [63, 5, 31, 0]
    mapping = _mapping_from_local_order(num_global_experts, local_to_global)
    x = torch.randn(tokens, emb_dim, generator=torch.Generator().manual_seed(89)) * 0.08
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=97)
    indices = _force_expert_in_slot(indices, 63)
    routing_weights = torch.softmax(torch.randn(tokens, topk, generator=torch.Generator().manual_seed(101)), dim=-1)
    _, gate_tt, up_tt, down_tt = _make_expert_weights(mesh_device, len(local_to_global), emb_dim, expert_dim, seed=103)
    x_rm_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    x_tile_tt = ttnn.to_layout(x_rm_tt, ttnn.TILE_LAYOUT)
    topk_weights_tt = _to_device(
        mesh_device,
        routing_weights.reshape(1, 1, tokens, topk),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )

    oracle_tt, _, _ = _explicit_native_device_oracle(
        x_tile_tt,
        indices_tt,
        topk_weights_tt,
        mapping_tt,
        gate_tt,
        up_tt,
        down_tt,
        num_local_experts=len(local_to_global),
        valid_tokens=tokens,
        topk=topk,
    )
    composite_tt = ttnn.experimental.deepseek_prefill.topk_routed_expert_moe(
        x_tile_tt,
        indices_tt,
        topk_weights_tt,
        mapping_tt,
        gate_tt,
        up_tt,
        down_tt,
        num_local_experts=len(local_to_global),
        valid_tokens=tokens,
        max_dispatched_tokens_per_expert=tokens,
    )
    oracle = _device_shards(oracle_tt)[0].reshape(tokens, emb_dim).float()
    actual = _device_shards(composite_tt)[0].reshape(tokens, emb_dim).float()
    passing, pcc = comp_pcc(oracle, actual, 0.9999)
    assert passing, f"public top-k composite diverged from explicit native device oracle: {pcc}"


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_public_composite_cache_hit_replaces_routing_mapping_expert63_weights_and_output(
    mesh_device, device_params
):
    first_case = _composite_case_inputs(
        mesh_device,
        local_to_global=[63, 5, 31, 0],
        x_seed=107,
        index_seed=109,
        routing_seed=113,
        expert_seed=127,
    )
    second_case = _composite_case_inputs(
        mesh_device,
        local_to_global=[5, 63, 17, 42],
        x_seed=131,
        index_seed=137,
        routing_seed=139,
        expert_seed=149,
    )
    assert int(first_case["mapping_host"][63]) == 0
    assert int(second_case["mapping_host"][63]) == 1

    # Prime every primitive and use its output topology as the exact optional-
    # output allocation contract for the two cache-hit calls.
    prototype = _run_composite(first_case)
    first_output = ttnn.empty_like(prototype)
    second_output = ttnn.empty_like(prototype)
    first = _run_composite(first_case, output=first_output)
    ttnn.synchronize_device(mesh_device)
    cache_entries = mesh_device.num_program_cache_entries()

    second = _run_composite(second_case, output=second_output)
    ttnn.synchronize_device(mesh_device)
    assert mesh_device.num_program_cache_entries() == cache_entries

    first_address = ttnn.get_device_tensors(first)[0].buffer_address()
    second_address = ttnn.get_device_tensors(second)[0].buffer_address()
    assert first_address == ttnn.get_device_tensors(first_output)[0].buffer_address()
    assert second_address == ttnn.get_device_tensors(second_output)[0].buffer_address()
    assert first_address != second_address
    assert (
        ttnn.get_device_tensors(first_case["indices"])[0].buffer_address()
        != ttnn.get_device_tensors(second_case["indices"])[0].buffer_address()
    )
    assert (
        ttnn.get_device_tensors(first_case["weights"])[0].buffer_address()
        != ttnn.get_device_tensors(second_case["weights"])[0].buffer_address()
    )
    assert (
        ttnn.get_device_tensors(first_case["mapping"])[0].buffer_address()
        != ttnn.get_device_tensors(second_case["mapping"])[0].buffer_address()
    )
    assert (
        ttnn.get_device_tensors(first_case["gate"][0])[0].buffer_address()
        != ttnn.get_device_tensors(second_case["gate"][1])[0].buffer_address()
    )

    first_host = _device_shards(first)[0].float()
    second_host = _device_shards(second)[0].float()
    assert not torch.equal(first_host, second_host), "cache hit reused stale routing/mapping/expert/output data"


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", TWO_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_topk_public_composite_rejects_cross_device_output_on_combine_cache_miss_and_hit(
    mesh_device, device_params, expect_error
):
    owner = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
    other = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))
    case = _composite_case_inputs(
        owner,
        local_to_global=[63, 5, 31, 0],
        x_seed=151,
        index_seed=157,
        routing_seed=163,
        expert_seed=167,
    )
    wrong_output = _to_device(
        other,
        torch.empty(1, 1, case["tokens"], 1024),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )
    message = "optional_output must be on the same device as packed_y"

    # No combine with this signature has run yet: this reaches combine's
    # cache-miss validator after the preceding composite stages are enqueued.
    with expect_error(RuntimeError, message):
        _run_composite(case, output=wrong_output)
    ttnn.synchronize_device(owner)

    _run_composite(case)
    ttnn.synchronize_device(owner)
    cache_entries_after_prime = owner.num_program_cache_entries()
    with expect_error(RuntimeError, message):
        _run_composite(case, output=wrong_output)
    assert owner.num_program_cache_entries() == cache_entries_after_prime


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_native_production_t1024_workspace_allocation_is_under_40_mib(mesh_device, device_params):
    """Pin Ornith's real T=1024, K=8, H=2048 active workspace below 40 MiB."""

    tokens, topk, hidden, num_global_experts, num_local_experts = 1024, 8, 2048, 256, 64
    expected_bytes = tokens * topk * hidden * 2
    x_rm_bytes = tokens * hidden * 2
    output_bf8_bytes = tokens * hidden + (tokens // 32) * (hidden // 32) * 64
    capacity = tokens * topk + 31 * num_local_experts
    metadata_bytes = (2 * num_global_experts + num_local_experts + capacity + 2 * tokens * topk) * 4
    active_workspace_bytes = x_rm_bytes + expected_bytes + output_bf8_bytes + metadata_bytes
    assert expected_bytes == 33_554_432
    assert active_workspace_bytes == 40_085_504
    assert active_workspace_bytes < 40 * 1024 * 1024
    workspace = ttnn.empty(
        [1, 1, tokens * topk, hidden],
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    shard = ttnn.get_device_tensors(workspace)[0]
    assert shard.buffer_address() != 0
    assert shard.volume() * 2 == expected_bytes
    assert shard.buffer_page_size() == hidden * 2
    assert shard.buffer_aligned_page_size() * (tokens * topk) == expected_bytes
    ttnn.deallocate(workspace)


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_local_dispatch_malformed_duplicate_mapping_fails_closed(mesh_device, device_params):
    tokens, hidden, topk, num_global_experts, num_local_experts = 32, 32, 8, 64, 4
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(79))
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=83)
    mapping = _mapping_from_local_order(num_global_experts, [0, 1, 2, 3])
    mapping[4] = 3  # duplicate local id 3: planner must publish no usable work
    x_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    outputs = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt, indices_tt, mapping_tt, num_local_experts, valid_tokens=tokens
    )
    for output_index in (1, 2, 3, 5, 6):
        assert torch.count_nonzero(_as_u32(_device_shards(outputs[output_index])[0])) == 0
    assignments = _as_u32(_device_shards(outputs[4])[0])
    assert torch.all(assignments == UINT32_MAX)


@pytest.mark.skipif(not is_blackhole(), reason="topk_routed_expert_moe is Blackhole-only")
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_topk_local_dispatch_malformed_duplicate_expert_ids_fail_closed(mesh_device, device_params):
    tokens, hidden, topk, num_global_experts, num_local_experts = 32, 32, 8, 64, 4
    x = torch.randn(tokens, hidden, generator=torch.Generator().manual_seed(89))
    indices = _unique_random_topk(tokens, num_global_experts, topk, seed=97)
    indices = _force_expert_in_slot(indices, 0)
    indices[0, 1] = 0  # expert 0 count becomes tokens+1, the bounded-workspace hazard
    mapping = _mapping_from_local_order(num_global_experts, [0, 1, 2, 3])
    x_tt, indices_tt, mapping_tt = _dispatch_inputs(mesh_device, x, indices, mapping.reshape(1, -1))
    outputs = ttnn.experimental.deepseek_prefill.topk_local_dispatch(
        x_tt, indices_tt, mapping_tt, num_local_experts, valid_tokens=tokens
    )
    for output_index in (1, 2, 3, 5, 6):
        assert torch.count_nonzero(_as_u32(_device_shards(outputs[output_index])[0])) == 0
    assignments = _as_u32(_device_shards(outputs[4])[0])
    assert torch.all(assignments == UINT32_MAX)
