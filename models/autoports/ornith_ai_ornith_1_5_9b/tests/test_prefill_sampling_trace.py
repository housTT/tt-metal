# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual sampler orchestration under CPU-only TTNN allocation boundaries."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_generator_serving_contract.py")))


@pytest.fixture
def sampler():
    tt = HELPERS["tt"].__wrapped__()
    gen = HELPERS["gen"].__wrapped__(tt)
    gen._prefill_sampling_logits = None
    gen._prefill_sampling_saved = []
    gen._prefill_sampling_masks = []
    gen.sampling._penalties_active = True
    # Keep the full UINT32 range observable in the CPU boundary simulation.
    gen.sampling.tt_sampling.seeds_tt_tensor = torch.arange(2**32 - 100, 2**32 - 68, dtype=torch.int64)
    gen._logits = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    gen.mesh_device = SimpleNamespace(programs=10)
    gen.mesh_device.num_program_cache_entries = lambda: gen.mesh_device.programs
    gen._programs = 10
    gen._sampling_trace = 7
    gen._sample_device = lambda logits: pytest.fail("Serving prefill must not submit eager sampling")
    gen._prepare_prefill_sampling(gen._logits)
    gen._programs = gen.mesh_device.programs
    events, freed = [], []
    tt.deallocate = lambda tensor: freed.append(tensor)
    tt.clone = lambda tensor: pytest.fail("A request-local clone cannot survive sampler replay")

    def capture():
        events.append("capture")
        gen._logits = torch.full_like(gen._logits, -123)
        gen._programs = gen.mesh_device.programs

    gen._capture = capture

    def replay(mesh, trace, *, cq_id, blocking):
        assert trace == 7 and not blocking
        events.append(("sample", gen._logits.clone()))
        gen._inputs[0].copy_(torch.arange(200000, 200032, dtype=torch.int32).reshape(1, 1, 1, 32))
        gen.sampling.tt_sampling.seeds_tt_tensor.add_(1)
        if gen.sampling._penalties_active:
            for tensor in gen._sampler_history_tensors():
                tensor.add_(13)

    tt.execute_trace = replay
    return gen, tt, events, freed


@pytest.mark.parametrize("batch,rows", [(1, [0]), (32, [31]), (32, [0, 7, 31]), (32, list(range(32)))])
@pytest.mark.parametrize("penalties", [False, True])
@pytest.mark.parametrize("external,program_miss", [(False, False), (False, True), (True, True), (True, False)])
def test_prefill_replays_with_exact_partial_state_and_preserved_logits(
    sampler, batch, rows, penalties, external, program_miss
):
    gen, tt, events, freed = sampler
    gen.max_batch_size = batch
    gen.sampling._penalties_active = penalties
    targets = gen._prefill_sampling_targets()
    before = [tensor.clone() for tensor in targets]
    addresses = [tensor.data_ptr() for tensor in targets + gen._prefill_sampling_saved + gen._prefill_sampling_masks]
    supplied = gen._logits.clone().add_(77) if external else gen._logits
    expected_logits = supplied.clone()
    gen.mesh_device.programs += int(program_miss)
    replay = tt.execute_trace

    def checked_replay(*args, **kwargs):
        assert any(tensor is supplied for tensor in freed) == external
        assert gen._logits is not supplied if external or program_miss else gen._logits is supplied
        replay(*args, **kwargs)

    tt.execute_trace = checked_replay
    gen._sample_prefill_device(supplied, rows)
    assert events[-1][0] == "sample"
    assert torch.equal(events[-1][1], expected_logits), "Recapture lost populated canonical logits"
    assert ("capture" in events) == program_miss
    assert any(tensor is supplied for tensor in freed) == external
    for index, (target, old) in enumerate(zip(targets, before)):
        expected = old.clone()
        if index == 0:
            expected.reshape(-1)[rows] = torch.tensor([200000 + row for row in rows], dtype=torch.int32)
        elif index == 1:
            expected[rows] += 1
        elif penalties:
            expected[rows] += 13
        assert torch.equal(target, expected), f"Incorrect state for target {index}, rows {rows}"
    assert addresses == [
        tensor.data_ptr() for tensor in targets + gen._prefill_sampling_saved + gen._prefill_sampling_masks
    ]
    assert all(mask.dtype == torch.int32 for mask in gen._prefill_sampling_masks)
    assert gen.counters["prefill_sampling_replays"] == 1
    assert gen.counters["readbacks"] == 0


def test_late_program_miss_is_visible_instead_of_eager_fallback(sampler):
    gen, tt, _, _ = sampler
    original_copy = tt.copy

    def compile_copy(source, target):
        gen.mesh_device.programs += 1
        original_copy(source, target)

    tt.copy = compile_copy
    with pytest.raises(RuntimeError, match="warmed before trace replay"):  # allow-pytest.raises: CPU-only error.
        gen._sample_prefill_device(gen._logits, [0])
    assert gen.counters["prefill_sampling_replays"] == 0


def test_recapture_warm_reuses_persistent_buffers(sampler):
    gen, _, _, _ = sampler
    buffers = gen._prefill_sampling_saved + gen._prefill_sampling_masks + [gen._prefill_sampling_logits]
    addresses = [tensor.data_ptr() for tensor in buffers]
    gen.model.upload = lambda *args, **kwargs: pytest.fail("Recapture warm must reuse masks")
    gen._prepare_prefill_sampling(gen._logits)
    assert addresses == [tensor.data_ptr() for tensor in buffers]


def test_teardown_releases_traces_before_persistent_sampling_buffers(sampler):
    gen, tt, _, freed = sampler
    buffers = gen._prefill_sampling_saved + gen._prefill_sampling_masks + [gen._prefill_sampling_logits]
    events = []
    gen._release_traces = lambda: events.append("release_traces")
    gen._release_prefill_inputs = lambda: events.append("release_inputs")

    def deallocate(tensor):
        assert events == ["release_traces", "release_inputs"]
        freed.append(tensor)

    tt.deallocate = deallocate
    gen.teardown()
    assert len(freed) == len(buffers)
    assert all(any(tensor is value for value in freed) for tensor in buffers)
    assert gen._prefill_sampling_logits is None
    assert gen._prefill_sampling_saved == gen._prefill_sampling_masks == []


def test_admission_mask_refreshes_only_when_lane_membership_changes(sampler):
    gen, tt, _, _ = sampler
    copied = []
    original_copy = tt.copy_host_to_device_tensor

    def copy(source, target):
        copied.append(target)
        original_copy(source, target)

    tt.copy_host_to_device_tensor = copy
    for rows, writes in (([31], 2), ([31], 2), ([7, 31], 4), ([31, 7], 4)):
        gen._sample_prefill_device(gen._logits, rows)
        assert len(copied) == writes
        for mask in gen._prefill_sampling_masks:
            assert torch.nonzero(mask.reshape(-1)).reshape(-1).tolist() == sorted(rows)
    gen._sample_prefill_device(gen._logits, list(range(32)))
    assert len(copied) == 4
    assert gen._prefill_sampling_rows == (7, 31)
    # Trace recapture and repeat warmup reuse the unchanged mask allocations.
    gen._capture()
    gen._prepare_prefill_sampling(gen._logits)
    gen._sample_prefill_device(gen._logits, [31, 7])
    assert len(copied) == 4


def test_initial_full_admission_needs_no_mask_upload(sampler):
    gen, tt, _, _ = sampler
    tt.copy_host_to_device_tensor = lambda *args: pytest.fail("Initial all-lane masks are already resident")
    gen._sample_prefill_device(gen._logits, list(range(32)))
    assert gen._prefill_sampling_rows == tuple(range(32))


@pytest.mark.parametrize("penalties", [False, True])
def test_full_physical_admission_skips_all_preservation_work(sampler, penalties):
    gen, tt, events, _ = sampler
    gen.max_batch_size = 32
    gen.sampling._penalties_active = penalties
    gen._prefill_sampling_rows = (7, 31)
    targets = gen._prefill_sampling_targets()
    before = [tensor.clone() for tensor in targets]

    def forbidden(*args, **kwargs):
        pytest.fail("All32 physical lanes need no backup, mask refresh, or restoration")

    tt.copy = tt.copy_host_to_device_tensor = forbidden
    gen._restore_prefill_sampling = gen._prefill_sampling_targets = forbidden
    gen._sample_prefill_device(gen._logits, list(reversed(range(32))))
    assert len(events) == 1 and events[0][0] == "sample"
    assert gen._prefill_sampling_rows == (7, 31), "Skipping masks must retain their actual membership cache"
    assert gen._inputs[0].reshape(-1).tolist() == list(range(200000, 200032))
    assert torch.equal(targets[1], before[1] + 1)
    for actual, old in zip(targets[2:], before[2:]):
        assert torch.equal(actual, old + 13 if penalties else old)


@pytest.mark.parametrize("rows", [[0], [0] * 32])
def test_less_than_32_unique_lanes_still_restores_unadmitted_state(sampler, rows):
    gen, _, _, _ = sampler
    gen.max_batch_size = 1 if len(rows) == 1 else 32
    original_restore = gen._restore_prefill_sampling
    restored = []

    def restore(targets):
        restored.append(True)
        original_restore(targets)

    gen._restore_prefill_sampling = restore
    seed = gen.sampling.tt_sampling.seeds_tt_tensor.clone()
    gen._sample_prefill_device(gen._logits, rows)
    assert restored == [True]
    seed[0] += 1
    assert torch.equal(seed, gen.sampling.tt_sampling.seeds_tt_tensor)
