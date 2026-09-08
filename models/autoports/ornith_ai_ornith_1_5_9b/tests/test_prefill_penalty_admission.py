# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Execute real adapter/generator admission methods with CPU-only TT boundaries."""

import ast
from collections import defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def admission():
    namespace = {"torch": torch, "dataclass": dataclass, "fields": fields, "replace": replace, "List": List}
    common = ROOT.parents[1] / "common/sampling"
    definitions(common / "_utils.py", {"clamp"}, namespace)
    definitions(common / "generator.py", {"SamplingParams", "format_sampling_params"}, namespace)
    namespace["Generator"] = object
    namespace["ttnn"] = SimpleNamespace(
        clone=lambda value: value.clone(), copy=lambda value, target: target.copy_(value), deallocate=lambda value: None
    )
    definitions(ROOT / "tt/generator.py", {"OrnithGenerator"}, namespace)
    definitions(ROOT / "tt/generator_vllm.py", {"TTOrnithForCausalLM"}, namespace)
    params_type = namespace["SamplingParams"]
    gen = namespace["OrnithGenerator"].__new__(namespace["OrnithGenerator"])
    gen.max_batch_size, gen.sampling_mode, gen._live = 4, "device", True
    gen._model_trace, gen._logits = 1, torch.arange(4).float()
    gen._inputs = [torch.arange(32), torch.arange(4), torch.arange(4), torch.arange(128).reshape(4, 32)]
    gen.sampling = SimpleNamespace(
        _penalties_active=False,
        tt_sampling=SimpleNamespace(seeds_tt_tensor=torch.arange(32) + 10000),
        tt_penalties=SimpleNamespace(
            output_mask=torch.arange(96).reshape(32, 3),
            output_counts=torch.arange(96).reshape(32, 3) + 100,
            output_counts_gathered=torch.arange(96).reshape(32, 3) + 200,
        ),
    )
    gen.kv_cache = object()
    calls = []
    gen._release_traces = lambda: calls.append("release")
    gen._capture = lambda: calls.append("capture")
    tracked = [gen._inputs[0], gen._logits, gen.sampling.tt_sampling.seeds_tt_tensor] + gen._sampler_history_tensors()

    def configure(params):
        calls.append("configure")
        formatted = namespace["format_sampling_params"](params, 32)
        gen.sampling._penalties_active = any(
            any(value != default for value in getattr(formatted, name))
            for name, default in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0))
        )

    gen._configure_sampling = configure
    gen._sample_device = lambda logits: [value.add_(17) for value in tracked]
    gen.prefill_forward = lambda *args, **kwargs: calls.append(("prefill", kwargs)) or torch.tensor([17])
    adapter = namespace["TTOrnithForCausalLM"].__new__(namespace["TTOrnithForCausalLM"])
    adapter.generator, adapter.max_batch_size, adapter.page_table_blocks = gen, 4, 32
    adapter.allow_host_sampling, adapter.uses_mrope = False, False
    adapter._prefilled_rows, adapter._device_rows = torch.zeros(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool)
    adapter._pending_device_seeds = torch.zeros(4, dtype=torch.bool)
    adapter._sampling_key, adapter.counters = None, defaultdict(int)
    defaults = params_type(temperature=0.0, top_k=1, top_p=1.0)
    adapter._params = {field.name: [getattr(defaults, field.name)] * 4 for field in fields(defaults)}
    return SimpleNamespace(adapter=adapter, gen=gen, params=params_type, calls=calls, tracked=tracked)


def params(case, **kwargs):
    return case.params(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, **kwargs)


@pytest.mark.parametrize("other_rows_active", [True, False])
@pytest.mark.parametrize(
    "field,value", [("presence_penalty", 0.5), ("frequency_penalty", 0.4), ("repetition_penalty", 1.1)]
)
def test_adapter_admits_new_penalized_prefill_after_neutral_request(admission, other_rows_active, field, value):
    case = admission
    case.adapter._device_rows[:] = other_rows_active
    before = [tensor.clone() for tensor in case.tracked]
    case.adapter.prefill_forward(
        tokens=torch.tensor([[7, 8, 9]]),
        page_table=torch.tensor([[1]]),
        kv_cache=case.gen.kv_cache,
        prompt_lens=[3],
        empty_slots=[2],
        start_pos=[0],
        sampling_params=case.params(temperature=0.0, top_k=1, top_p=1.0, **{field: value}),
    )
    assert case.calls[-1][0] == "prefill"
    assert case.calls[-1][1]["slots"] == [2]
    assert all(torch.equal(actual, expected) for actual, expected in zip(case.tracked, before))


def test_generator_admits_fresh_penalty_lane_without_erasing_ongoing_state(admission):
    case = admission
    before = [tensor.clone() for tensor in case.tracked]
    case.gen.configure_sampling(params(case, presence_penalty=[0.0, 0.0, 0.5, 0.0]), fresh_slots=[2])
    assert case.calls == ["release", "configure", "capture"]
    assert all(torch.equal(actual, expected) for actual, expected in zip(case.tracked, before))


def test_continuation_prefill_cannot_claim_fresh_history(admission):
    case = admission
    with pytest.raises(ValueError, match="requires prompt_token_ids"):  # allow-pytest.raises: CPU-only error.
        case.adapter.prefill_forward(
            tokens=torch.tensor([[7, 8, 9]]),
            page_table=torch.tensor([[1]]),
            kv_cache=case.gen.kv_cache,
            prompt_lens=[3],
            empty_slots=[2],
            start_pos=[3],
            sampling_params=case.params(temperature=0.0, top_k=1, top_p=1.0, presence_penalty=0.5),
        )
    assert case.calls == []


@pytest.mark.parametrize("fresh", [None, [], [1]])
def test_existing_or_continuing_penalty_lane_still_requires_real_histories(admission, fresh):
    case = admission
    with pytest.raises(ValueError, match="requires prompt_token_ids"):  # allow-pytest.raises: CPU-only error.
        case.gen.configure_sampling(params(case, presence_penalty=[0.0, 0.0, 0.5, 0.0]), fresh_slots=fresh)
    assert case.calls == []


def test_fresh_lane_does_not_authorize_penalties_for_another_live_lane(admission):
    case = admission
    with pytest.raises(ValueError, match="requires prompt_token_ids"):  # allow-pytest.raises: CPU-only error.
        case.gen.configure_sampling(params(case, frequency_penalty=[0.1, 0.0, 0.5, 0.0]), fresh_slots=[2])
    assert case.calls == []


@pytest.mark.parametrize("fresh", [[2, 2], [-1], [4], [2.5]])
def test_invalid_fresh_slots_fail_before_any_sampler_mutation(admission, fresh):
    case = admission
    with pytest.raises(ValueError, match="fresh_slots"):  # allow-pytest.raises: CPU-only error.
        case.gen.configure_sampling(params(case, presence_penalty=[0.0, 0.0, 0.5, 0.0]), fresh_slots=fresh)
    assert case.calls == []
