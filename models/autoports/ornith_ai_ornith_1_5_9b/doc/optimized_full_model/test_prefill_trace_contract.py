# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for prefill trace ownership, invalidation and validation."""

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_fixtures = runpy.run_path(str(Path(__file__).with_name("test_trace_lifecycle.py")))
generator_fixture = _fixtures["generator_fixture"]
sampling_params = _fixtures["sampling_params"]


def prefill_fixture():
    gen, runtime, teardown = generator_fixture()
    gen.use_prefill_trace = True
    gen.owns_cache = True
    gen.max_batch_size = 1
    gen._live = False
    gen.counters = dict.fromkeys(
        (
            "prefill_captures",
            "prefill_trace_misses",
            "prefill_replays",
            "prefill_token_refreshes",
            "prefill_page_table_refreshes",
            "prefill_eager_calls",
            "prefill_sampling_replays",
            "sampling_replays",
        ),
        0,
    )
    gen.kv_cache = SimpleNamespace(prefill_layers=[SimpleNamespace(reset_state=lambda: None)])
    gen.model = SimpleNamespace(prefill_chunk=2048)
    runtime.uint32 = runtime.int32 = None

    def allocate(value, dtype):
        assert not runtime.live, "persistent prefill inputs allocated behind a trace"
        return value.clone()

    gen._device = allocate
    gen._prefill_trace_forward = lambda output: None
    return gen, runtime, teardown


def test_four_trace_recapture_rebinds_output_and_releases_every_old_trace():
    gen, runtime, teardown = prefill_fixture()
    gen._prefill_inputs = [torch.zeros(1, 128), torch.tensor([[0, 1]])]
    bound_outputs = []
    gen._prefill_trace_forward = lambda output: bound_outputs.append(output)
    gen._capture()
    for _ in range(7):
        previous = runtime.live.copy()
        output = gen._logits
        gen.mesh_device.programs += 1
        gen._ensure_replay_safe()
        assert len(runtime.live) == 4
        assert previous.isdisjoint(runtime.live)
        assert previous.issubset(runtime.released)
        assert bound_outputs[-1] is gen._logits
        assert gen._logits is not output
        assert runtime.created[-1] == gen._prefill_trace
    gen._configure_sampling(sampling_params())
    assert not runtime.live
    gen._capture()
    teardown()
    assert not runtime.live
    assert gen._prefill_inputs is gen._prefill_key is None


def test_shape_eviction_releases_traces_before_replacing_inputs_and_live_miss_falls_back():
    gen, runtime, teardown = prefill_fixture()
    freed = []
    runtime.deallocate = lambda tensor: freed.append(tensor)
    for length in (128, 131, 128):
        old = gen._prefill_inputs
        assert gen._prepare_prefill_trace([length], [0], [0], torch.tensor([[0, 1]]))
        assert not runtime.live
        if old is not None:
            assert all(any(value is tensor for value in freed) for tensor in old)
        assert gen._prefill_inputs[0].shape == (1, length)
        gen._capture()
    assert gen.counters["prefill_trace_misses"] == 3
    gen._live = True
    previous, inputs = runtime.live.copy(), gen._prefill_inputs
    assert not gen._prepare_prefill_trace([131], [0], [0], torch.tensor([[1, 0]]))
    assert gen._prefill_inputs is inputs and runtime.live == previous
    assert gen._prepare_prefill_trace([128], [0], [0], torch.tensor([[1, 0]]))
    assert gen._prefill_inputs is inputs and runtime.live == previous
    teardown()


@pytest.mark.parametrize(
    "changes,length,rows,starts,all_logits",
    [
        ({"use_prefill_trace": False}, 128, [0], [0], False),
        ({"sampling_mode": "host"}, 128, [0], [0], False),
        ({"max_batch_size": 32}, 128, [0], [0], False),
        ({}, 128, [1], [0], False),
        ({}, 128, [0], [127], False),
        ({}, 2049, [0], [0], False),
        ({}, 128, [0], [0], True),
    ],
)
def test_unsupported_trace_specs_preserve_eager_fallback(changes, length, rows, starts, all_logits):
    gen, runtime, _ = prefill_fixture()
    for name, value in changes.items():
        setattr(gen, name, value)
    assert not gen._prepare_prefill_trace([length], rows, starts, torch.tensor([[0, 1]]), return_all_logits=all_logits)
    assert gen._prefill_inputs is None
    assert not runtime.created


@pytest.mark.parametrize("state_already_reset", [False, True])
@pytest.mark.parametrize("owns_cache", [False, True])
def test_prompt_preparation_recaptures_before_prefill_and_public_output_is_owned(state_already_reset, owns_cache):
    gen, runtime, teardown = prefill_fixture()
    gen.owns_cache = owns_cache
    events = []
    resets = []
    gen.kv_cache.prefill_layers[0].reset_state = lambda: resets.append(True)
    table = torch.tensor([[1, 0]])
    assert gen._prepare_prefill_trace([131], [0], [0], table)
    gen._capture()
    gen.ensure_traces = lambda: None
    gen.model.validate_prefill = lambda *args, **kwargs: ([0], [0], table)

    def prepare(*args):
        events.append("prepare")
        gen.mesh_device.programs += 1

    def write(value, target, counter):
        target.copy_(value)
        gen.counters[counter] += 1

    def replay(mesh, trace, **kwargs):
        if trace == gen._prefill_trace:
            events.append("prefill")
            gen._logits.fill_(gen._prefill_inputs[0].sum())
        else:
            assert trace == gen._sampling_trace
            events.append("sample")

    gen._prepare_prompt_sampling = prepare
    gen._write = write
    runtime.execute_trace = replay
    gen._prefill_trace_forward = lambda output: events.append("capture")
    kwargs = dict(page_table=table, kv_cache=gen.kv_cache, prompt_lens=[131], return_device_logits=True)
    if state_already_reset:
        gen.kv_cache.prefill_layers[0].reset_state()
    borrowed = gen._prefill([[7] * 131], **kwargs, borrow_logits=True, state_already_reset=state_already_reset)
    assert len(resets) == 1, "prefill duplicated the caller's reset or failed to reset fresh state"
    assert events == ["prepare", "capture", "prefill"]
    assert borrowed is gen._logits and borrowed.item() == 917
    gen._sample_first_token(borrowed)
    assert events[-1] == "sample"
    assert gen.counters["prefill_sampling_replays"] == 1
    owned = gen.prefill_forward([[8] * 131], **kwargs)
    assert len(resets) == 2, "public prefill must still reset its own fresh state"
    assert owned is not gen._logits and owned.data_ptr() != gen._logits.data_ptr()
    assert owned.item() == 1048
    gen._logits.zero_()
    assert owned.item() == 1048
    teardown()


def test_first_sampler_program_miss_does_not_recapture_populated_logits():
    gen, _, teardown = prefill_fixture()
    gen._capture()
    observed = []
    logits = gen._logits
    gen._sample_device = lambda tensor: observed.append(tensor)
    gen.mesh_device.programs += 1
    gen._sample_first_token(logits)
    assert observed == [logits]
    assert gen._logits is logits
    assert gen.counters["prefill_sampling_replays"] == 0
    teardown()


def model_validator():
    source = Path(__file__).resolve().parents[2] / "tt/model.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "OrnithModel"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "validate_prefill")
    namespace = dict(torch=torch, num_blocks_for_context=lambda context, block: (context + block - 1) // block)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    model = SimpleNamespace(page_block_size=64, vocab_size=248320)
    return lambda *args, **kwargs: namespace["validate_prefill"](model, *args, **kwargs)


@pytest.mark.parametrize(
    "override,error",
    [
        ({"slots": [1]}, "valid cache rows"),
        ({"prompt_lens": [132]}, "logical token IDs"),
        ({"tokens": [[248320] * 131]}, "logical token IDs"),
        ({"page_table": torch.tensor([[0]])}, "attention read window"),
        ({"page_table": torch.tensor([[0, 1, 2, 4]])}, "unallocated physical block"),
        ({"start_pos": [126]}, "cache context"),
        ({"prompt_lens": [0]}, "cache context"),
    ],
)
def test_shared_validation_rejects_invalid_prefill_before_device_work(override, error):
    validate = model_validator()
    kwargs = dict(
        tokens=[[100] * 131],
        page_table=torch.tensor([[3, 2, 1, 0]]),
        kv_cache=SimpleNamespace(batch_size=1, context=256, kv=[(torch.empty(4, 1), None)]),
        prompt_lens=[131],
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match=error):
        validate(**kwargs)


def test_shared_validation_keeps_native_context_and_noncontiguous_mixed_slots():
    validate = model_validator()
    width = 262144 // 64
    table = torch.arange(32 * width).reshape(32, width).flip(1)
    rows, starts, actual = validate(
        [[248319] * 131, [100]],
        page_table=table,
        kv_cache=SimpleNamespace(batch_size=32, context=262144, kv=[(torch.empty(32 * width, 1), None)]),
        prompt_lens=[131, 1],
        slots=[31, 2],
        start_pos=[127, 262143],
    )
    assert rows == [31, 2] and starts == [127, 262143]
    assert torch.equal(actual, table)


@pytest.mark.parametrize("length", [128, 131])
@pytest.mark.parametrize("live", [False, True])
def test_external_cache_selects_exact_fresh_shape_and_keeps_resident_inputs(length, live):
    gen, runtime, teardown = prefill_fixture()
    gen.owns_cache = False
    table = torch.arange(4096).reshape(1, -1)
    cache = gen.kv_cache
    assert gen._prepare_prefill_trace([length], [0], [0], table)
    gen._capture()
    inputs, traces = gen._prefill_inputs, runtime.live.copy()
    gen._live = live
    assert gen._prepare_prefill_trace([length], [0], [0], table.flip(1))
    assert gen.kv_cache is cache and gen._prefill_key[0] == id(cache)
    assert gen._prefill_inputs is inputs and runtime.live == traces
    assert gen._prefill_inputs[0].shape == (1, length)
    assert not gen._prepare_prefill_trace([131 if length == 128 else 128], [0], [0], table)
    assert gen._prefill_inputs is inputs and runtime.live == traces
    assert gen.counters["prefill_trace_misses"] == 1
    teardown()


def test_external_prefill_rejects_different_cache_before_validation_or_capture():
    gen, runtime, _ = prefill_fixture()
    gen.owns_cache = False
    gen.model.validate_prefill = lambda *args, **kwargs: pytest.fail("identity must be checked first")
    with pytest.raises(ValueError, match="Bind caller-owned cache"):
        gen.prefill_forward([[100]], page_table=torch.tensor([[0]]), kv_cache=object(), prompt_lens=[1])
    assert not runtime.created
