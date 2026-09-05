# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual generator trace ownership without importing TTNN."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[2] / "tt/generator.py"


class TraceRuntime:
    def __init__(self):
        self.live = set()
        self.created = []
        self.released = []
        self.capturing = None

    def begin_trace_capture(self, mesh, *, cq_id):
        assert self.capturing is None
        trace = len(self.created) + 1
        self.created.append(trace)
        self.live.add(trace)
        self.capturing = trace
        return trace

    def end_trace_capture(self, mesh, trace, *, cq_id):
        assert self.capturing == trace
        self.capturing = None

    def release_trace(self, mesh, trace):
        assert self.capturing is None
        assert trace in self.live, f"trace {trace} released more than once"
        self.live.remove(trace)
        self.released.append(trace)

    clone = staticmethod(torch.clone)
    copy = staticmethod(lambda src, dst: dst.copy_(src))
    deallocate = staticmethod(lambda tensor: None)


def sampling_params():
    return SimpleNamespace(
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        enable_log_probs=[False],
        num_logprobs=[0],
        top_k=[20],
        top_p=[0.95],
        temperature=[0.8],
        seed=[1234],
    )


def generator_fixture():
    runtime = TraceRuntime()
    tree = ast.parse(SOURCE.read_text())
    namespace = dict(torch=torch, Generator=object, ttnn=runtime, format_sampling_params=lambda value, size: value)
    exec(
        compile(
            ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[]),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    cls = namespace["OrnithGenerator"]
    gen = cls.__new__(cls)
    gen.mesh_device = SimpleNamespace(programs=10)
    gen.mesh_device.num_program_cache_entries = lambda: gen.mesh_device.programs
    gen.sampling_mode = "device"
    gen._model_trace = gen._sampling_trace = gen._sampling_history_trace = None
    gen._prefill_trace = gen._prefill_key = gen._prefill_inputs = None
    gen._programs = None
    gen._sampling_key = ()
    gen._forward = lambda: torch.tensor([17.0])
    gen._sample_device = lambda logits: None
    gen._append_output_history = lambda: None
    gen.sampling = SimpleNamespace(reset_trace=lambda: None, reset_sampling_params=lambda params: None)
    saved_teardown = gen.teardown
    # The qualitative runner deliberately defers public cleanup across requests.
    gen.teardown = lambda: None
    return gen, runtime, saved_teardown


def test_recapture_releases_old_traces_when_public_cleanup_is_deferred():
    gen, runtime, saved_teardown = generator_fixture()
    gen._capture()
    for _ in range(7):
        previous = runtime.live.copy()
        gen.mesh_device.programs += 1
        gen._ensure_replay_safe()
        assert len(runtime.live) == 3, f"superseded traces remain live: {sorted(runtime.live)}"
        assert previous.isdisjoint(runtime.live)
        assert previous.issubset(runtime.released)
    saved_teardown()
    saved_teardown()
    assert not runtime.live
    assert sorted(runtime.created) == sorted(runtime.released)


def test_sampling_key_change_releases_traces_despite_public_override():
    gen, runtime, _ = generator_fixture()
    gen._capture()
    gen._configure_sampling(sampling_params())
    assert not runtime.live, "sampling-key change left stale traces live"
    assert gen._model_trace is gen._sampling_trace is gen._sampling_history_trace is None
    gen._capture()
    previous = runtime.live.copy()
    gen._configure_sampling(sampling_params())
    assert runtime.live == previous, "unchanged sampling key unnecessarily released traces"


def test_live_reconfiguration_releases_before_warm_and_preserves_state():
    gen, runtime, saved_teardown = generator_fixture()
    gen._capture()
    gen._live = True
    gen.max_batch_size = 1
    gen._inputs = [torch.tensor([3]), torch.tensor([4]), torch.tensor([4]), torch.tensor([[0]])]
    gen.sampling._penalties_active = False
    gen.sampling.tt_sampling = SimpleNamespace(seeds_tt_tensor=torch.tensor([81]))
    gen.sampling.tt_penalties = SimpleNamespace(
        output_mask=torch.tensor([1]), output_counts=torch.tensor([2]), output_counts_gathered=torch.tensor([3])
    )
    state = [gen._inputs[0], gen._logits, gen.sampling.tt_sampling.seeds_tt_tensor] + gen._sampler_history_tensors()
    before = [value.clone() for value in state]
    warms = []

    def sample(logits):
        if runtime.capturing is None:
            assert not runtime.live, "sampler warmup ran behind an unreleased trace"
            warms.append(True)
            for value in state:
                value.add_(7)

    gen._sample_device = sample
    gen.configure_sampling(sampling_params())
    assert warms == [True]
    assert len(runtime.live) == 3
    assert all(torch.equal(old, new) for old, new in zip(before, state))
    saved_teardown()
    assert not runtime.live
