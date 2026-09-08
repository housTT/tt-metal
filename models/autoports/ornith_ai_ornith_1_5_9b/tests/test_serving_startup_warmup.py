# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Actual adapter/canonical parameter source with CPU boundaries; no TTNN import."""

import runpy
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HELPERS = runpy.run_path(str(Path(__file__).with_name("test_prefill_penalty_admission.py")))


def fixture(batch, mrope):
    case = HELPERS["admission"].__wrapped__()
    adapter, gen = case.adapter, case.gen
    adapter.max_batch_size = gen.max_batch_size = batch
    adapter.max_model_len = 262144
    adapter._warmed = False
    adapter.uses_mrope = mrope
    adapter._last_device_sampling = None
    adapter._device_rows = torch.zeros(batch, dtype=torch.bool)
    adapter._prefilled_rows = torch.zeros(batch, dtype=torch.bool)
    adapter._pending_device_seeds = torch.zeros(batch, dtype=torch.bool)
    defaults = case.params(temperature=0.0, top_k=1, top_p=1.0)
    adapter._params = {field.name: [getattr(defaults, field.name)] * batch for field in fields(defaults)}
    gen._live = False
    gen._model_trace = gen._sampling_key = None
    gen._inputs = [torch.zeros(32, dtype=torch.int64), torch.zeros(batch), torch.zeros(batch)]
    gen.owns_cache = False
    gen.use_prefill_trace = True
    gen._prefill_key = gen._prefill_inputs = gen._previous_prefill_table = None
    gen.model = SimpleNamespace(prefill_chunk=2048)
    gen._device = lambda value, dtype: value.clone()
    gen._history_rows = 0
    gen.counters = defaultdict(int)
    gen.sampling.reset_sampling_params = lambda params: case.calls.append(("canonical_params", params))
    gen.sampling.reset_output_state = lambda *args: None
    # Restore the real canonical key/format logic overridden by the admission fixture.
    del gen._configure_sampling

    def release():
        gen._model_trace = None

    def ensure(*, preserve_cache=True):
        if gen._model_trace is None:
            case.calls.append(("capture", gen._prefill_key))
            gen.counters["cache_snapshots"] += int(preserve_cache)
            gen.counters["captures"] += 1
            gen._model_trace = gen.counters["captures"]

    gen._release_traces = release
    gen.ensure_traces = ensure
    gen.recurrent = torch.zeros(batch, 3)
    gen.seed_state = torch.zeros(32)

    def prefill(tokens, *, slots, prompt_lens, page_table, start_pos=None, **kwargs):
        starts = [0] * len(slots) if start_pos is None else start_pos
        traced = gen._prepare_prefill_trace(prompt_lens, slots, starts, page_table)
        gen.ensure_traces()
        gen.counters["prefill_replays" if traced else "prefill_eager_calls"] += 1
        gen._live = True
        gen.recurrent[slots] += 1
        gen.seed_state[slots] += 1
        return torch.tensor([17] * len(slots))

    def refresh(tokens, positions, mask):
        mask = torch.as_tensor(mask).bool()
        gen._inputs[0][:batch][mask] = torch.as_tensor(tokens).reshape(-1)[mask]
        gen._inputs[1][:] = torch.as_tensor(positions)
        gen._inputs[2][:] = torch.as_tensor(positions).clamp_min(0)

    def decode(*args, **kwargs):
        assert kwargs["sample_on_device"] is True and kwargs["read_from_device"] is False
        gen.counters["decode_replays"] += 1
        gen.recurrent += 1
        gen.seed_state += 1
        gen._inputs[0] += 1
        gen._inputs[1] += 1
        gen._inputs[2] += 1
        return gen._inputs[0]

    def reset(*, clear_kv):
        assert clear_kv
        case.calls.append("reset")
        gen._live = False
        gen.recurrent.zero_()

    def reset_seeds():
        gen.seed_state.zero_()

    def write_tokens(values):
        gen._inputs[0].zero_()
        gen._inputs[0][:batch] = torch.tensor(values)

    def write_positions(values):
        gen._inputs[1][:] = torch.tensor(values)
        gen._inputs[2][:] = torch.tensor(values)

    gen.prefill_forward, gen.refresh_serving_inputs, gen.decode_forward = prefill, refresh, decode
    gen.reset, gen._reset_seeds = reset, reset_seeds
    gen._reset_output_history = lambda: setattr(gen, "_history_rows", 0)
    gen._write_tokens, gen._write_positions = write_tokens, write_positions
    gen.read_output_async = lambda output, **kwargs: (output.clone(), "event")
    gen.tokens_from = lambda value: value.reshape(-1)[:batch].to(torch.int64)
    namespace = adapter.warmup_model_prefill.__func__.__globals__
    namespace["ttnn"] = SimpleNamespace(
        event_synchronize=lambda event: case.calls.append(("wait", event)),
        int32=torch.int32,
        uint32=torch.int32,
        deallocate=lambda tensor: None,
    )
    return case


@pytest.mark.parametrize("batch", [1, 4, 32])
@pytest.mark.parametrize("mrope", [False, True])
def test_startup_admits_and_finishes_dummy_request_then_restores_empty_state(batch, mrope):
    case = fixture(batch, mrope)
    adapter, gen = case.adapter, case.gen
    cache = gen.kv_cache
    inputs = list(gen._inputs)
    for enable_trace in (False, True):
        adapter.warmup_model_prefill(cache, enable_trace=enable_trace)
        adapter.warmup_model_decode(cache, enable_trace=enable_trace)
    assert gen.counters["decode_replays"] == 1
    assert adapter.counters["async_reads"] == 1
    assert case.calls.index(("wait", "event")) < case.calls.index("reset")
    assert not gen._live and adapter._last_device_sampling is None
    assert not adapter._device_rows.any() and not adapter._prefilled_rows.any()
    assert not adapter._pending_device_seeds.any()
    assert gen.recurrent.count_nonzero() == 0 and gen.seed_state.count_nonzero() == 0
    assert gen._history_rows == 0 and all(tensor.count_nonzero() == 0 for tensor in gen._inputs)
    assert all(actual is original for actual, original in zip(gen._inputs, inputs))
    assert gen.kv_cache is cache
    assert adapter._sampling_key is not None and gen._sampling_key is not None
    assert gen.counters["cache_snapshots"] == 0
    captured_keys = [entry[1] for entry in case.calls if isinstance(entry, tuple) and entry[0] == "capture"]
    assert len(captured_keys) == 1
    assert (captured_keys[0] is not None) == (batch == 1)
    if batch == 1:
        assert captured_keys[0][1] == 128
        assert gen.counters["prefill_replays"] == 1
    # The actual plugin may supply unrestricted k and an explicit seed; these
    # normalize to the same greedy trace and must not snapshot empty native KV.
    captures = gen.counters["captures"]
    request_params = case.params(temperature=0.0, top_k=248320, top_p=1.0, seed=7, num_logprobs=-2)
    adapter.prefill_forward(
        tokens=torch.zeros(1, 128, dtype=torch.int32),
        page_table=torch.zeros(batch, 32, dtype=torch.int32),
        kv_cache=cache,
        prompt_lens=[128],
        empty_slots=[0],
        sampling_params=request_params,
    )
    assert gen.counters["captures"] == captures
    assert gen.counters["cache_snapshots"] == 0


def test_first_nonmatching_request_after_startup_retains_external_trace_without_snapshot():
    case = fixture(1, False)
    adapter, gen = case.adapter, case.gen
    adapter.warmup_model_prefill(gen.kv_cache)
    assert not gen._live
    key, inputs, trace = gen._prefill_key, gen._prefill_inputs, gen._model_trace
    params = case.params(temperature=0.0, top_k=1, top_p=1.0, num_logprobs=-2)
    for length, traced in ((131, False), (128, True)):
        before = dict(gen.counters)
        adapter.prefill_forward(
            tokens=torch.zeros(1, length, dtype=torch.int32),
            page_table=torch.zeros(1, 32, dtype=torch.int32),
            kv_cache=gen.kv_cache,
            prompt_lens=[length],
            empty_slots=[0],
            sampling_params=params,
        )
        assert gen.counters["prefill_replays"] - before.get("prefill_replays", 0) == int(traced)
        assert gen._prefill_key == key and gen._prefill_inputs is inputs and gen._model_trace == trace
        assert gen.counters["cache_snapshots"] == 0 and gen.counters["captures"] == 1
