# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.autoports.qwen_qwen3_5_35b_a3b.tt.generator import SAMPLER_ROWS, Qwen35Generator, _greedy_params


class _SeedSink:
    max_batch_size = SAMPLER_ROWS

    def __init__(self):
        self.writes = []

    def _next_unseeded_device_seed(self):
        return len(self.writes) + 1

    def write_device_seed_values(self, values):
        self.writes.append(list(values))


class _Sampler:
    def __init__(self):
        self.params = []
        self.samples = 0
        self.seed_manager = _SeedSink()

    def reset_sampling_params(self, params):
        self.params.append(params)

    def sample(self, logits, **kwargs):
        self.samples += 1


def _generator():
    gen = Qwen35Generator.__new__(Qwen35Generator)
    gen.sampler = _Sampler()
    gen.counters = Counter()
    gen._sampling_mode = None
    gen._greedy_lanes = True
    gen.tt_tokens = SimpleNamespace()
    return gen


def test_scalar_sampling_params_reach_every_lane():
    gen = _generator()
    gen._set_sampling(replace(_greedy_params(), temperature=0.8, top_k=32, top_p=0.9))
    formatted = gen.sampler.params[-1]
    assert list(formatted.top_k) == [32] * SAMPLER_ROWS
    assert list(formatted.top_p) == [0.9] * SAMPLER_ROWS
    assert len(set(formatted.temperature)) == 1
    assert not gen._greedy_lanes


def test_greedy_lanes_skip_seed_writes_and_sampled_lanes_refresh_seeds():
    gen = _generator()
    gen._set_sampling(_greedy_params())
    assert gen._greedy_lanes
    gen._sample(None)
    assert gen.sampler.seed_manager.writes == []
    gen._set_sampling(replace(_greedy_params(), temperature=1.0, top_k=32, top_p=1.0))
    gen._sample(None)
    gen._sample(None)
    writes = gen.sampler.seed_manager.writes
    assert len(writes) == 2 and len(writes[0]) == SAMPLER_ROWS and writes[0] != writes[1]
    assert gen.sampler.samples == 3


def test_explicit_request_seed_rejected():
    gen = _generator()
    with pytest.raises(ValueError):
        gen._set_sampling(replace(_greedy_params(), seed=7))


@pytest.mark.parametrize("field,value", [("presence_penalty", 0.5), ("frequency_penalty", 0.5),
                                         ("repetition_penalty", 1.2), ("enable_log_probs", True),
                                         ("num_logprobs", 1)])
def test_unprepared_sampler_variants_rejected(field, value):
    gen = _generator()
    with pytest.raises(ValueError):
        gen._set_sampling(replace(_greedy_params(), **{field: value}))
    assert gen.sampler.params == []


def test_sampling_refresh_tracks_every_field():
    gen = _generator()
    params = replace(_greedy_params(), temperature=1.0, top_k=32, top_p=0.9)
    gen._set_sampling(params)
    gen._set_sampling(params)
    assert gen.counters["sampling_param_refreshes"] == 1
    gen._set_sampling(replace(params, top_p=0.8))
    gen._set_sampling(replace(params, top_p=0.8, presence_penalty=None))
    assert gen.counters["sampling_param_refreshes"] == 2


class _Pool:
    scratch_block = 99


def _decode_generator():
    gen = _generator()
    gen.kv_pool = _Pool()
    gen._live_slots = set()
    gen.decode_batches = (1, 4, 32)
    gen.writes = {}
    gen._check_kv = lambda kv: None
    gen._write_tokens = lambda tokens: gen.writes.update(tokens=tokens)
    gen._write_positions = lambda batch, positions: gen.writes.update(positions=positions)
    gen._write_page_table = lambda batch, table: gen.writes.update(table=table)
    gen._replay = lambda batch, sample=True: None
    gen.read_tokens = lambda batch: torch.zeros(batch, dtype=torch.int64)
    return gen


def test_negative_position_marks_inactive_row():
    gen = _decode_generator()
    table = torch.arange(8, dtype=torch.int32).reshape(4, 2)
    gen.decode_forward(torch.zeros(4, 1, dtype=torch.int64), torch.tensor([5, -1, 7, -1]), page_table=table,
                       sampling_params=_greedy_params())
    assert gen.writes["positions"] == [5, 0, 7, 0]
    assert gen.writes["table"][1].tolist() == [99, 99] and gen.writes["table"][3].tolist() == [99, 99]
    assert gen.writes["table"][0].tolist() == [0, 1] and gen.writes["table"][2].tolist() == [4, 5]
    assert table[1].tolist() == [2, 3]


def test_decode_rejects_slot_map_other_than_row_order():
    gen = _decode_generator()
    args = (torch.zeros(4, 1, dtype=torch.int64), [1, 1, 1, 1])
    gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params(),
                       request_slots=[0, 1, 2, 3])
    with pytest.raises(ValueError):
        gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params(),
                           request_slots=[5, 1, 2, 3])
    gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params(),
                       slot_remap=torch.arange(4, dtype=torch.int32))
    with pytest.raises(ValueError):
        gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params(),
                           slot_remap=torch.tensor([5, 1, 2, 3]))
    with pytest.raises(TypeError):
        gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params(), read_from_device=False)


def test_decode_rejects_unprepared_batch():
    gen = _decode_generator()
    with pytest.raises(ValueError, match="not prepared"):
        gen.decode_forward(torch.zeros(3, 1, dtype=torch.int64), [1, 1, 1], page_table=torch.zeros(3, 1),
                           sampling_params=_greedy_params())
    assert gen.writes == {}


def test_inactive_row_on_live_slot_rejected_until_released():
    gen = _decode_generator()
    gen._live_slots = {0, 1, 2, 3}
    args = (torch.zeros(4, 1, dtype=torch.int64), torch.tensor([5, -1, 7, 8]))
    with pytest.raises(ValueError, match=r"\[1\]"):
        gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params())
    assert gen.writes == {}
    gen.release_slots([1])
    gen.decode_forward(*args, page_table=torch.zeros(4, 1), sampling_params=_greedy_params())
    assert gen.writes["positions"] == [5, 0, 7, 8]


class _Model:
    max_batch_size = 32

    def __init__(self):
        self.prefills = []
        self.resets = []

    def reset_request_state(self, slots):
        self.resets.append(slots)

    def prefill(self, ids, *, start_pos, page_table, request_slots, return_all_logits):
        self.prefills.append((request_slots, start_pos))
        return SimpleNamespace(deallocate=lambda force: None)


def _prefill_generator():
    gen = _decode_generator()
    gen.model = _Model()
    gen.prepare = lambda: None
    gen._write_prefill_table = lambda row: None
    gen.tt_prefill_table = None
    gen._logits_to_host = lambda logits, rows: torch.zeros(rows, 4)
    return gen


def test_prefill_empty_slots_select_request_slots():
    gen = _prefill_generator()
    tokens = torch.zeros(2, 3, dtype=torch.int64)
    gen.prefill_forward(tokens, page_table=torch.zeros(2, 1), prompt_lens=[3, 2], empty_slots=[5, 9])
    assert [p[0] for p in gen.model.prefills] == [(5,), (9,)]
    assert gen.model.resets == [(5,), (9,)]
    assert gen._live_slots == {5, 9}
    gen.prefill_forward(tokens, page_table=torch.zeros(2, 1), prompt_lens=[3, 2], empty_slots=[5, 9],
                        request_slots=[5, 9])
    with pytest.raises(ValueError):
        gen.prefill_forward(tokens, page_table=torch.zeros(2, 1), prompt_lens=[3, 2], empty_slots=[5, 9],
                            request_slots=[0, 1])
    for slots in ([5], [5, 5], [5, 32]):
        with pytest.raises(ValueError):
            gen.prefill_forward(tokens, page_table=torch.zeros(2, 1), prompt_lens=[3, 2], empty_slots=slots)
    with pytest.raises(TypeError):
        gen.prefill_forward(tokens, page_table=torch.zeros(2, 1), prompt_lens=[3, 2], unknown_slots=[5, 9])
    assert len(gen.model.prefills) == 4
