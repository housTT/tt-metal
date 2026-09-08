# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Source-executed adapter mode transitions; no TTNN import or device access."""

import copy
import runpy
from pathlib import Path

import pytest
import torch

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_prefill_penalty_admission.py")))


@pytest.fixture
def case():
    state = HELPERS["admission"].__wrapped__()
    state.adapter.allow_host_sampling = True
    state.adapter._last_device_sampling = False
    state.adapter._device_rows[:] = False
    state.gen.refresh_serving_inputs = lambda *args: state.calls.append("refresh_inputs")
    state.gen.decode_forward = lambda *args, **kwargs: state.calls.append(("decode", kwargs)) or torch.tensor([0] * 4)

    def reset_output(packed):
        counts = torch.stack([torch.bincount(row[row >= 0], minlength=3) for row in packed])
        state.gen.sampling.tt_penalties.output_counts.copy_(counts)
        state.gen.sampling.tt_penalties.output_counts_gathered.copy_(counts)
        state.gen.sampling.tt_penalties.output_mask.copy_((counts > 0).to(torch.int64))
        state.calls.append("restore_output_history")

    state.gen.sampling.reset_output_state = reset_output
    state.gen.sampling.reset_prompt_tokens = lambda packed: state.calls.append("restore_prompt_history")
    return state


def setup(state, presence=(2.0, 0.0, 0.0, 0.0)):
    params = HELPERS["params"](state, presence_penalty=list(presence))
    state.adapter._sampling(params, prompt_tokens=torch.tensor([[2]] * 4), output_tokens=torch.tensor([[0]] * 4))
    state.calls.clear()
    return params


def decode(state, params, *, histories=True, positions=(3, -1, -1, -1), remap=None, generated=(0, 0, 1)):
    kwargs = {}
    if histories:
        kwargs.update(prompt_tokens=torch.tensor([[2]] * 4), output_tokens=torch.tensor([list(generated)] * 4))
    return state.adapter.decode_forward(
        tokens=torch.ones(4, 1, dtype=torch.int64),
        start_pos=torch.tensor(positions),
        page_table=torch.zeros(4, 32, dtype=torch.int32),
        kv_cache=state.gen.kv_cache,
        sampling_params=params,
        read_from_device=False,
        slot_remap=remap,
        **kwargs,
    )


def counts(state):
    return state.gen.sampling.tt_penalties.output_counts[:4].clone()


def test_unchanged_parameters_restore_host_continuation_history(case):
    params = setup(case)
    before = counts(case)
    decode(case, params)
    assert not torch.equal(counts(case), before)
    assert counts(case)[0].tolist() == [2, 1, 0]
    assert "restore_prompt_history" in case.calls
    assert "restore_output_history" in case.calls


def test_penalized_host_resume_rejects_missing_histories_before_replay(case):
    params = setup(case)
    # CPU-only adapter boundary check; no device state exists to recover.
    with pytest.raises(ValueError, match="Resuming.*histories"):  # allow-pytest.raises
        decode(case, params, histories=False)
    assert not any(isinstance(call, tuple) and call[0] == "decode" for call in case.calls)


def test_neutral_host_resume_does_not_reconfigure(case):
    params = setup(case, (0.0,) * 4)
    decode(case, params, histories=False)
    assert "configure" not in case.calls
    assert "restore_output_history" not in case.calls


def test_steady_device_decode_preserves_device_authoritative_history(case):
    params = setup(case)
    case.adapter._last_device_sampling = True
    case.adapter._device_rows[0] = True
    before = counts(case)
    decode(case, params)
    assert torch.equal(counts(case), before)
    assert "configure" not in case.calls
    assert "refresh_inputs" not in case.calls


def test_fresh_device_prefill_keeps_its_new_history_after_older_host_decode(case):
    params = setup(case)
    case.adapter._device_rows[0] = True
    case.adapter._prefilled_rows[0] = True
    before = counts(case)
    decode(case, params)
    assert torch.equal(counts(case), before)
    assert "configure" not in case.calls


def test_fresh_host_prefill_restores_its_host_sampled_first_token(case):
    params = setup(case)
    case.adapter._last_device_sampling = None
    case.adapter._prefilled_rows[0] = True
    decode(case, params, generated=(1,))
    assert counts(case)[0].tolist() == [0, 1, 0]
    assert "restore_output_history" in case.calls


def test_neutral_host_row_does_not_force_fresh_device_penalty_row_history(case):
    params = setup(case, (0.0, 2.0, 0.0, 0.0))
    case.adapter._device_rows[1] = True
    case.adapter._prefilled_rows[1] = True
    before = counts(case)
    decode(case, params, histories=False, positions=(3, 3, -1, -1))
    assert torch.equal(counts(case), before)
    assert "configure" not in case.calls


def test_inactive_penalized_host_slot_does_not_force_history(case):
    params = setup(case, (0.0, 2.0, 0.0, 0.0))
    decode(case, params, histories=False)
    assert "configure" not in case.calls


def test_changed_parameters_restore_host_history(case):
    setup(case)
    changed = HELPERS["params"](case, presence_penalty=[1.0, 0.0, 0.0, 0.0])
    decode(case, changed)
    assert counts(case)[0].tolist() == [2, 1, 0]
    assert "restore_output_history" in case.calls


def test_remap_then_host_resume_restores_new_slot_order(case):
    setup(case)
    case.gen.remap_serving_slots = lambda remap: case.calls.append(("remap", remap.tolist()))
    changed = HELPERS["params"](case, presence_penalty=[0.0, 2.0, 0.0, 0.0])
    decode(case, changed, positions=(-1, 3, -1, -1), remap=[1, 0, 2, 3])
    assert case.calls[0] == ("remap", [1, 0, 2, 3])
    assert counts(case)[1].tolist() == [2, 1, 0]
    assert "restore_output_history" in case.calls


def test_device_to_host_step_skips_device_sampler_and_preserves_history(case):
    setup(case)
    case.adapter._last_device_sampling = True
    case.adapter._device_rows[0] = True
    before = counts(case)
    decode(case, None)
    assert torch.equal(counts(case), before)
    assert case.calls[-1][0] == "decode"
    assert case.calls[-1][1]["sample_on_device"] is False


def seed_case(state):
    path = Path(__file__).parents[1] / "doc/full_model/test_generator_host_contract.py"
    helpers = runpy.run_path(str(path))
    common, manager = helpers["common_seed_contract"]()
    common["copy"] = copy  # The real SeedManager remap uses copy.copy.
    initialize = helpers["NAMESPACE"]["_request_seed_values"]
    gen = state.gen
    gen._configured_seeds, gen._sampling_key = [11] * 32, None
    gen._seed_values = torch.zeros(32, dtype=torch.int64)
    gen.sampling._penalties_active = False
    gen.sampling.reset_sampling_params = lambda params: None
    gen._configure_sampling = type(gen)._configure_sampling.__get__(gen)

    def reset_rows(rows):
        values = initialize(manager, gen._configured_seeds, rows)
        gen._seed_values[rows] = torch.tensor(values)
        gen.sampling.tt_sampling.seeds_tt_tensor[rows] = torch.tensor(values)
        state.calls.append(("reset_seed_rows", list(rows)))

    gen._reset_request_seeds = reset_rows

    def prefill(tokens, **kwargs):
        gen._prepare_prompt_sampling(tokens, kwargs["prompt_lens"], kwargs["slots"], kwargs["start_pos"])
        gen._live = True
        return torch.zeros(4, 3)

    gen.prefill_forward, gen.logits_from = prefill, lambda out: out
    state.adapter._last_device_sampling = None
    return manager, common["_hash_request_seed_to_device_seed"]


def seed_prefill(state, *, params=None, start=0):
    return state.adapter.prefill_forward(
        tokens=torch.tensor([[0]]),
        page_table=torch.zeros(1, 32, dtype=torch.int32),
        kv_cache=state.gen.kv_cache,
        prompt_lens=[1],
        empty_slots=[0],
        start_pos=[start],
        sampling_params=params,
    )


def seeded(state, seeds):
    return state.params(temperature=[1.0] * 4, top_k=[4] * 4, top_p=[1.0] * 4, seed=seeds)


def test_host_first_request_initializes_its_seed_once_on_first_device_step(case):
    manager, seed_hash = seed_case(case)
    seed_prefill(case)
    before = case.gen.sampling.tt_sampling.seeds_tt_tensor.clone()
    assert int(before[0]) == seed_hash(11, 0)
    params = seeded(case, [99, None, None, None])
    case.calls.clear()
    decode(case, params, histories=False)
    seeds = case.gen.sampling.tt_sampling.seeds_tt_tensor
    assert int(seeds[0]) == seed_hash(99, 0)
    assert torch.equal(seeds[1:], before[1:])
    assert manager.seeds[0] == 99
    assert not case.adapter._pending_device_seeds.any()
    assert ("reset_seed_rows", [0]) in case.calls
    case.calls.clear()
    decode(case, params, histories=False)
    assert "configure" not in case.calls
    assert not any(isinstance(call, tuple) and call[0] == "reset_seed_rows" for call in case.calls)


def test_pending_host_seed_follows_slot_remap(case):
    manager, seed_hash = seed_case(case)
    seed_prefill(case)

    def remap(values):
        order = values.tolist() + list(range(4, 32))
        tensor = case.gen.sampling.tt_sampling.seeds_tt_tensor
        tensor.copy_(tensor[order].clone())
        manager.apply_slot_remap(order)

    case.gen.remap_serving_slots = remap
    decode(case, seeded(case, [None, 99, None, None]), histories=False, positions=(-1, 3, -1, -1), remap=[1, 0, 2, 3])
    assert int(case.gen.sampling.tt_sampling.seeds_tt_tensor[1]) == seed_hash(99, 0)
    assert manager.seeds[1] == 99
    assert not case.adapter._pending_device_seeds.any()
    assert ("reset_seed_rows", [1]) in case.calls


def test_host_first_continuation_prefill_initializes_seed_before_device_sampling(case):
    manager, seed_hash = seed_case(case)
    seed_prefill(case)
    case.calls.clear()
    seed_prefill(case, params=seeded(case, [99, None, None, None]), start=1)
    assert int(case.gen.sampling.tt_sampling.seeds_tt_tensor[0]) == seed_hash(99, 0)
    assert manager.seeds[0] == 99
    assert not case.adapter._pending_device_seeds[0]
    assert case.calls.count(("reset_seed_rows", [0])) == 1


def test_fresh_device_prefill_uses_normal_seed_admission_and_clears_old_pending_flag(case):
    _, seed_hash = seed_case(case)
    seed_prefill(case)
    case.calls.clear()
    seed_prefill(case, params=seeded(case, [99, None, None, None]), start=0)
    assert int(case.gen.sampling.tt_sampling.seeds_tt_tensor[0]) == seed_hash(99, 0)
    assert not case.adapter._pending_device_seeds[0]
    assert case.calls.count(("reset_seed_rows", [0])) == 1
