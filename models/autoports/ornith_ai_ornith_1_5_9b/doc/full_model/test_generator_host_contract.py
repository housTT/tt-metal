"""Execute generator orchestration with fake TT boundaries, never import TTNN."""

import ast
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[2] / "tt/generator.py"
TREE = ast.parse(SOURCE.read_text())
# Load actual functions/class while excluding imports that initialize TT runtime.
NAMESPACE = {"torch": torch, "time": time, "Generator": object, "ttnn": SimpleNamespace(deallocate=lambda tensor: None)}
exec(
    compile(
        ast.Module(
            body=[node for node in TREE.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))], type_ignores=[]
        ),
        str(SOURCE),
        "exec",
    ),
    NAMESPACE,
)


def params(**kwargs):
    return SimpleNamespace(temperature=kwargs.pop("temperature", 0.0), top_k=kwargs.pop("top_k", 1), **kwargs)


@pytest.mark.parametrize(
    "sample_request", [params(temperature=0.8, top_k=20), params(presence_penalty=0.5), params(enable_log_probs=True)]
)
def test_host_mode_rejects_unsupported_implicit_sampling(sample_request):
    with pytest.raises(ValueError, match="host_sample"):
        NAMESPACE["_validate_host_sampling"](sample_request, None)


def test_host_callback_receives_sampling_history_and_validates_ids():
    sample_request = params(temperature=0.8, top_k=20, seed=7)
    observed = []

    def sample(logits, **kwargs):
        observed.append(kwargs)
        return [3, 2]

    NAMESPACE["_validate_host_sampling"](sample_request, sample)
    result = NAMESPACE["_host_predictions"](
        torch.zeros(2, 4),
        host_sample=sample,
        sampling_params=sample_request,
        step=2,
        prompts=[[1], [2, 1]],
        predictions=[[0, 1], [2, 3]],
    )
    assert result == [3, 2]
    assert observed == [
        {
            "sampling_params": sample_request,
            "step": 2,
            "prompt_token_ids": [[1], [2, 1]],
            "generated_token_ids": [[0, 2], [1, 3]],
        }
    ]
    for bad in ([4, 0], [1], [1.0, 2.0], [True, False]):
        with pytest.raises(ValueError, match="host_sample"):
            NAMESPACE["_host_predictions"](
                torch.zeros(2, 4),
                host_sample=lambda *a, **kw: bad,
                sampling_params=sample_request,
                step=0,
                prompts=[[1], [2]],
                predictions=[],
            )


def fake_generator():
    generator = NAMESPACE["OrnithGenerator"].__new__(NAMESPACE["OrnithGenerator"])
    generator.sampling_mode = "host"
    generator.host_sample = None
    generator.model = SimpleNamespace(
        logits_to_host=lambda tensor, users: tensor, hf_config=SimpleNamespace(eos_token_id=[2, 3])
    )
    generator.tokenizer = SimpleNamespace(eos_token_id=3)
    generator.kv_cache = SimpleNamespace(context=100)
    generator.page_table = None
    generator.max_batch_size = 1
    generator.counters = {}
    generator.perf = {}
    generator.reset = lambda: None
    generator.ensure_traces = lambda: None
    generator._ensure_replay_safe = lambda: None
    generator._write_positions = lambda values: None
    generator._refresh_table = lambda table: None
    generator.writes = []
    generator._write_tokens = lambda values: generator.writes.append(values)
    generator.prefill_forward = lambda *a, **kw: torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    generator._replay = lambda: setattr(generator, "_logits", torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    return generator


def test_generate_default_eos_fixed_window_and_teacher_forcing():
    generator = fake_generator()
    assert generator.generate([1], 3) == [2]
    assert generator.perf["decode_steps"] == 2
    assert generator.generate([1], 3, stop_on_eos=False) == [2, 3, 3]
    seen = []

    def teacher(step, token):
        seen.append((step, token))
        return 1

    assert generator.generate([1], 3, next_input=teacher) == [2, 3, 3]
    assert seen == [(0, 2), (1, 3), (2, 3)]


def test_generate_explicit_host_sampling_controls_feedback():
    generator = fake_generator()
    sample_request = params(temperature=0.7, top_k=4, seed=7)
    seen = []

    def sample(logits, **kwargs):
        assert kwargs["sampling_params"] is sample_request
        seen.append(kwargs["generated_token_ids"])
        return [0]

    assert generator.generate([1], 3, sampling_params=sample_request, host_sample=sample) == [0, 0, 0]
    assert seen == [[[]], [[0]], [[0, 0]]]
    assert generator.writes == [[0], [0]]


def test_live_sampling_reconfigure_restores_every_warmup_mutation(monkeypatch):
    gen = fake_generator()
    gen.sampling_mode = "device"
    gen._live = True
    gen._model_trace = 17
    gen._logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    gen._inputs = [torch.tensor([3]), torch.tensor([4]), torch.tensor([4]), torch.tensor([[0]])]
    histories = [torch.ones(2, 4, dtype=torch.int32) * i for i in range(3)]
    gen.sampling = SimpleNamespace(
        _penalties_active=True,
        tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.tensor([81])),
        tt_penalties=SimpleNamespace(
            output_mask=histories[0], output_counts=histories[1], output_counts_gathered=histories[2]
        ),
    )
    formatted = SimpleNamespace(presence_penalty=[0.1], frequency_penalty=[0.0], repetition_penalty=[1.0])
    monkeypatch.setitem(NAMESPACE, "format_sampling_params", lambda params, size: formatted)
    fake_tt = SimpleNamespace(
        clone=lambda tensor: tensor.clone(), copy=lambda src, dst: dst.copy_(src), deallocate=lambda tensor: None
    )
    monkeypatch.setitem(NAMESPACE, "ttnn", fake_tt)
    state = gen._inputs + [gen._logits, gen.sampling.tt_sampling.seeds_tt_tensor] + histories
    expected = [tensor.clone() for tensor in state]
    calls = []
    gen.teardown = lambda: calls.append("release")
    gen._configure_sampling = lambda value: calls.append("params")

    def warm(logits):
        for tensor in [gen._inputs[0], logits, gen.sampling.tt_sampling.seeds_tt_tensor] + histories:
            tensor.add_(7)

    gen._sample_device = warm
    gen._capture = lambda: calls.append("capture")
    gen.configure_sampling(params())
    assert calls == ["release", "params", "capture"]
    assert all(torch.equal(a, b) for a, b in zip(expected, state))
    gen.sampling._penalties_active = False
    with pytest.raises(ValueError, match="requires prompt_token_ids"):
        gen.configure_sampling(params())


def test_partial_prefill_sampler_preserves_other_rows(monkeypatch):
    gen = fake_generator()
    gen._inputs = [torch.arange(32).reshape(1, 1, 1, 32)]
    histories = [torch.arange(128).reshape(32, 4).clone() for _ in range(3)]
    gen.sampling = SimpleNamespace(
        _penalties_active=True,
        tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.arange(32)),
        tt_penalties=SimpleNamespace(
            output_mask=histories[0], output_counts=histories[1], output_counts_gathered=histories[2]
        ),
    )
    gen.model.upload = lambda value, **kwargs: value

    def where(condition, yes, no, output_tensor=None):
        value = torch.where(condition.bool(), yes, no)
        if output_tensor is not None:
            output_tensor.copy_(value)
        return value

    fake_tt = SimpleNamespace(
        clone=lambda tensor: tensor.clone(),
        copy=lambda src, dst: dst.copy_(src),
        deallocate=lambda tensor: None,
        uint32=None,
        int32=None,
        TILE_LAYOUT=None,
        ROW_MAJOR_LAYOUT=None,
        reshape=torch.reshape,
        to_layout=lambda tensor, layout: tensor.clone(),
        where=where,
    )
    monkeypatch.setitem(NAMESPACE, "ttnn", fake_tt)
    state = [gen._inputs[0], gen.sampling.tt_sampling.seeds_tt_tensor] + histories
    before = [tensor.clone() for tensor in state]

    def sample(logits):
        for tensor in state:
            tensor.add_(100)

    gen._sample_device = sample
    gen._sample_prefill_device(torch.zeros(32, 4), [2])
    for index, (old, new) in enumerate(zip(before, state)):
        if index < 2:
            old, new = old.flatten(), new.flatten()
        assert torch.equal(old[:2], new[:2])
        assert torch.equal(old[3:], new[3:])
        assert torch.equal(new[2], old[2] + 100)


def test_ttft_includes_request_reset_and_reports_prefill_boundary(monkeypatch):
    gen = fake_generator()
    timestamps = iter([10.0, 14.0, 18.0, 20.0, 23.0])
    events = []

    def clock():
        value = next(timestamps)
        events.append(("clock", value))
        return value

    gen.reset = lambda: events.append(("reset", None))
    monkeypatch.setitem(NAMESPACE, "time", SimpleNamespace(perf_counter=clock))
    gen.generate([1], 3)
    assert events[:2] == [("clock", 10.0), ("reset", None)]
    assert gen.perf["ttft_s"] == 8.0
    assert gen.perf["request_setup_s"] == 4.0
    assert gen.perf["prefill_only_s"] == 4.0
    assert gen.perf["decode_s"] == 3.0


def common_seed_contract():
    """Load the actual host-only shared seed/format definitions, excluding TT imports."""
    import random
    import secrets
    from dataclasses import dataclass, fields, replace
    from typing import List

    source = SOURCE.parents[3] / "common/sampling/generator.py"
    selected = {"SamplingParams", "format_sampling_params", "_hash_request_seed_to_device_seed", "SeedManager"}
    namespace = {
        "torch": torch,
        "random": random,
        "secrets": secrets,
        "dataclass": dataclass,
        "fields": fields,
        "replace": replace,
        "List": List,
        "MAX_UINT32": 2**32 - 1,
        "DEVICE_SEED_MAX": 1_000_000,
        "_UINT64_MASK": 2**64 - 1,
        "clamp": lambda value, low, high: max(low, min(high, value)),
    }
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    manager = namespace["SeedManager"](SimpleNamespace(_sampling_dp=1), salt_duplicate_seeds=False)
    return namespace, manager


def test_explicit_seed_lists_repeat_without_duplicate_slot_salts():
    common, manager = common_seed_contract()
    configured = [11, 22, 11] + [None] * 29
    initialize = NAMESPACE["_request_seed_values"]
    first = initialize(manager, configured, list(range(32)), reset_all=True)
    second = initialize(manager, configured, list(range(32)), reset_all=True)
    assert first[:3] == second[:3]
    assert first[0] == first[2] == common["_hash_request_seed_to_device_seed"](11, 0)
    assert manager.seed_counters[:3] == [1, 1, 1]


def test_scalar_seed_targets_only_lane_zero_with_real_common_formatter():
    common, manager = common_seed_contract()
    request_params = common["SamplingParams"](temperature=[1.0] * 4, top_k=32, top_p=1.0, seed=99)
    formatted = common["format_sampling_params"](request_params, 32)
    assert formatted.seed == [99] + [None] * 31
    values = NAMESPACE["_request_seed_values"](manager, formatted.seed, list(range(32)), reset_all=True)
    assert manager.seeds == [99] + [None] * 31
    assert values[0] == common["_hash_request_seed_to_device_seed"](99, 0)
    assert len(set(values[1:])) > 1


def test_unseeded_requests_get_fresh_lane_entropy(monkeypatch):
    _, manager = common_seed_contract()
    entropy = iter(range(1000, 1064))
    used = []

    def next_entropy():
        value = next(entropy)
        used.append(value)
        return value

    monkeypatch.setattr(manager, "_next_unseeded_rng_seed", next_entropy)
    initialize = NAMESPACE["_request_seed_values"]
    first = initialize(manager, [None] * 32, list(range(32)), reset_all=True)
    second = initialize(manager, [None] * 32, list(range(32)), reset_all=True)
    assert used == list(range(1000, 1064))
    assert first != second
    assert len(set(first)) > 1 and len(set(second)) > 1
    assert all(1 <= value <= 1_000_000 for value in first + second)


def test_new_or_reused_seeded_slot_matches_fresh_and_preserves_ongoing_manager():
    common, manager = common_seed_contract()
    initialize = NAMESPACE["_request_seed_values"]
    configured = [7, 8, 7] + [None] * 29
    fresh = initialize(manager, configured, [0], reset_all=True)[0]
    manager.seed_counters[0] = 123
    joined = initialize(manager, configured, [2])[0]
    assert joined == fresh
    assert manager.seed_counters[0] == 123
    manager.seed_counters[2] = 88
    reused = initialize(manager, configured, [2])[0]
    assert reused == joined == common["_hash_request_seed_to_device_seed"](7, 0)
    assert manager.seed_counters[0] == 123


def test_continuation_prefill_does_not_reset_request_seed():
    gen = fake_generator()
    gen.sampling_mode = "device"
    gen.sampling = SimpleNamespace(_penalties_active=False)
    calls = []
    gen._reset_request_seeds = lambda rows: calls.append(rows)
    gen._prepare_prompt_sampling([[1]], [1], [2], [5])
    assert calls == []
    gen._prepare_prompt_sampling([[1]], [1], [2], None)
    assert calls == [[2]]


def test_generator_configuration_consumes_formatted_lane_seeds(monkeypatch):
    common, _ = common_seed_contract()
    gen = fake_generator()
    gen._model_trace = None
    gen._sampling_key = None
    gen.sampling = SimpleNamespace(reset_sampling_params=lambda value: None)
    monkeypatch.setitem(NAMESPACE, "format_sampling_params", common["format_sampling_params"])
    sample_params = common["SamplingParams"](temperature=[0.8] * 4, top_k=32, top_p=1.0, seed=99)
    gen._configure_sampling(sample_params)
    assert gen._configured_seeds == [99] + [None] * 31


def test_request_seed_device_merge_preserves_ongoing_counters(monkeypatch):
    common, manager = common_seed_contract()
    gen = fake_generator()
    gen._configured_seeds = [7, 8, 7] + [None] * 29
    gen._seed_values = torch.ones(32, dtype=torch.int32)
    target = torch.arange(100, 132, dtype=torch.int64)
    before = target.clone()
    gen.sampling = SimpleNamespace(seed_manager=manager, tt_sampling=SimpleNamespace(seeds_tt_tensor=target))
    gen.model.upload = lambda tensor, **kwargs: tensor
    fake_tt = SimpleNamespace(
        copy=lambda src, dst: dst.copy_(src),
        deallocate=lambda tensor: None,
        uint32=None,
        int32=None,
        TILE_LAYOUT=None,
        ROW_MAJOR_LAYOUT=None,
        reshape=torch.reshape,
        to_layout=lambda tensor, layout: tensor.clone(),
        where=lambda condition, yes, no: torch.where(condition.bool(), yes, no),
    )
    monkeypatch.setitem(NAMESPACE, "ttnn", fake_tt)
    gen._reset_request_seeds([2])
    expected = before.clone()
    expected[2] = common["_hash_request_seed_to_device_seed"](7, 0)
    assert torch.equal(target, expected)
