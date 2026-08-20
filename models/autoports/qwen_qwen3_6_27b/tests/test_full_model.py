# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Contract and opt-in device smoke tests for the Qwen3.6 full model."""

from __future__ import annotations

import inspect
import gc
import os
import time
import types
from pathlib import Path

import pytest
import torch
import ttnn
from tracy import signpost
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

from models.autoports.qwen_qwen3_6_27b.tt.generator import Generator
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.model import (
    LazySafetensorState,
    MAX_BATCH_SIZE,
    PADDED_VOCAB_SIZE,
    QwenFullModel,
    TP_SIZE,
    VocabParallelLMHead,
    _resolve_checkpoint,
)
from models.common.readiness_check.contract import Generator as ReadinessGenerator
from models.common.readiness_check.metrics_json import runtime_metadata, write_metrics_json
from models.common.sampling.generator import SamplingGenerator, SamplingParams
from models.common.sampling.tt_sampling import TTSampling


MESH_DEVICE = [4]
DEVICE_PARAMS = [
    {
        "trace_region_size": 1_500_000_000,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    }
]


def test_full_model_static_contracts():
    assert issubclass(Generator, ReadinessGenerator)
    assert MAX_BATCH_SIZE == 32
    assert PADDED_VOCAB_SIZE == 262_144
    generate = inspect.signature(Generator.generate)
    assert "enable_trace" in generate.parameters
    assert "next_input" in generate.parameters
    decode = inspect.signature(Generator.decode_forward)
    assert decode.parameters["read_from_device"].default is True
    assert callable(Generator.read_decode_output)
    assert callable(Generator.process_decode_output_host)
    source = inspect.getsource(QwenFullModel.decode_device)
    assert "MultichipDecoder" not in source  # layers were selected at construction
    assert "to_torch" not in source
    assert "argmax" not in source
    assert "plus_one" in source
    sampling_args_source = inspect.getsource(QwenFullModel._sampling_args)
    assert "trace_seeded_sampling=True" in sampling_args_source
    sample_source = inspect.getsource(SamplingGenerator.sample)
    assert "self.trace_seeded_sampling" in sample_source
    snapshot_source = inspect.getsource(Generator._snapshot_linear_state)
    assert snapshot_source.count("ttnn.clone(item[0]") == 1
    assert snapshot_source.count("ttnn.clone(item[1]") == 1

    # A physical four-device P300c ring must not be treated like a logical
    # submesh sliced from a larger Galaxy topology.
    sampler = object.__new__(TTSampling)
    sampler.mesh_device = types.SimpleNamespace(get_num_devices=lambda: 4)
    sampler.tt_ccl = types.SimpleNamespace(get_num_links=lambda axis: 2)
    sampler.num_argmax_gather_links = 1
    sampler.ag_topology = ttnn.Topology.Ring
    sampler.allow_small_ring_argmax = True
    assert sampler._get_force_argmax_all_gather_config(None) == (1, ttnn.Topology.Ring)
    sampler.allow_small_ring_argmax = False
    assert sampler._get_force_argmax_all_gather_config(None) == (1, ttnn.Topology.Linear)
    sampler.num_gather_links = 1
    sampler.allow_small_ring_sampling = True
    assert sampler._get_sampling_all_gather_config(None, 1) == (1, ttnn.Topology.Ring)
    sampler.allow_small_ring_sampling = False
    assert sampler._get_sampling_all_gather_config(None, 1) == (1, ttnn.Topology.Linear)


def test_small_ring_full_logits_gather_uses_proven_async_protocol(monkeypatch):
    calls = []
    sampler = object.__new__(TTSampling)
    sampler.mesh_device = types.SimpleNamespace(get_num_devices=lambda: 4)
    sampler.tt_ccl = types.SimpleNamespace(
        get_num_links=lambda axis: 2,
        get_and_cycle_ag_semaphore_handles=lambda axis: "ag-semaphore",
    )
    sampler.ag_topology = ttnn.Topology.Ring
    sampler.allow_small_ring_sampling = True
    sampler.argmax_chunks_per_sync = 10
    sampler.argmax_num_workers_per_link = 1
    sampler._line_all_gather = None

    def fake_all_gather_async(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return "gathered"

    monkeypatch.setattr(ttnn.experimental, "all_gather_async", fake_all_gather_async)
    result = sampler._perform_proven_ring_all_gather(
        "input",
        dim=3,
        cluster_axis=None,
        memory_config="dram",
        num_links=1,
    )

    assert result == "gathered"
    assert calls == [
        (
            "input",
            {
                "persistent_output_buffer": None,
                "dim": 3,
                "multi_device_global_semaphore": "ag-semaphore",
                "num_links": 1,
                "memory_config": "dram",
                "cluster_axis": None,
                "topology": ttnn.Topology.Ring,
                "chunks_per_sync": 10,
                "num_workers_per_link": 1,
                "num_buffers_per_channel": 2,
            },
        )
    ]
    forward_source = inspect.getsource(TTSampling.forward)
    full_ring_branch = forward_source.split("if self.use_full_logits_ring_sampling:", 1)[1]
    assert "self._perform_proven_ring_all_gather" in full_ring_branch
    assert "self.tt_full_indices_tensor_chunks" in forward_source
    assert "logits_chunks = ttnn.split" in forward_source
    assert "ttnn.concat(topk_values_chunks, dim=3)" in forward_source
    assert "ttnn.concat(topk_indices_chunks, dim=3)" in forward_source
    assert "if self.use_full_logits_ring_sampling:" in forward_source


class _SeedManagerSpy:
    def __init__(self):
        self.reset_calls = []
        self.advance_calls = []
        self.explicit_seed_advance_calls = []
        self.seeds = [None] * MAX_BATCH_SIZE
        self.seed_counters = [0] * MAX_BATCH_SIZE
        self._seed_active = False
        self._reseted = False
        self._needs_skip = False
        self._active_request_seed = False

    def reset_seed(self, seeds, slots):
        self.reset_calls.append((list(seeds), tuple(slots)))
        for index, slot in enumerate(slots):
            self.seeds[slot] = seeds[index]
            self.seed_counters[slot] = 0
        self._seed_active = any(seed is not None for seed in self.seeds)
        self._reseted = True

    def get_new_values(self, slots):
        self.advance_calls.append(tuple(slots))
        if self._seed_active:
            self.explicit_seed_advance_calls.append(tuple(slots))
            for slot in slots:
                if self.seeds[slot] is not None:
                    self.seed_counters[slot] += 1
        elif self._reseted:
            self._reseted = False
            self._needs_skip = True
        elif self._needs_skip:
            self._needs_skip = False

    def reset_request_state(self):
        self.seeds = [None] * MAX_BATCH_SIZE
        self.seed_counters = [0] * MAX_BATCH_SIZE
        self._seed_active = False
        self._reseted = False
        self._needs_skip = False
        self._active_request_seed = False


class _SamplingSpy:
    def __init__(self):
        self.seed_manager = _SeedManagerSpy()
        self.prompt_resets = []
        self.output_resets = []

    def reset_sampling_params(self, params):
        self.params = params

    def reset_trace(self):
        pass

    def reset_prompt_tokens(self, tokens):
        self.prompt_resets.append(tokens.clone())

    def reset_output_state(self, tokens=None):
        self.output_resets.append(None if tokens is None else tokens.clone())


class _HostSamplingFakeModel:
    vocab_size = 4
    max_seq_len = 32

    def __init__(self):
        self.sampling = _SamplingSpy()
        self.decode_logits = [
            torch.tensor([[[[2.0, 1.9, -4.0, -5.0]]]]),
            torch.tensor([[[[1.8, 2.0, -4.0, -5.0]]]]),
        ]

    def allocate_state(self, batch_size):
        assert batch_size == 1
        return types.SimpleNamespace(prompt_lens=(), token_buffer=object())

    def prefill(self, tokens, *, prompt_lens, state, return_all_logits):
        assert prompt_lens == [tokens.shape[-1]]
        assert not return_all_logits
        state.prompt_lens = tuple(prompt_lens)
        return torch.tensor([[[8.0, 7.9, -4.0, -5.0]]])

    def prepare_decode_state(self, state, tokens):
        del state, tokens

    def decode_device(self, state):
        del state
        return self.decode_logits.pop(0)


def _make_host_sampling_generator():
    generator = object.__new__(Generator)
    generator.model = _HostSamplingFakeModel()
    generator.host_sampling_compatibility = True
    generator._state = None
    generator._model_trace_id = None
    generator._trace_logits = None
    generator._trace_state = None
    generator._sampling_trace_output = None
    generator._host_sampling_rng = None
    generator._device_sampling_slots = ()
    generator._device_output_history = []
    generator._device_output_history_by_slot = []
    generator._device_sampling_request_active = False
    generator._previous_page_table = None
    generator._previous_page_table_object = None
    generator.tokenizer = types.SimpleNamespace(eos_token_id=-1)
    generator._logits_to_host = lambda logits: logits
    generator.copied_tokens = []
    generator._copy_replicated = lambda host, target, **kwargs: generator.copied_tokens.append(
        int(host.reshape(-1)[0].item())
    )
    return generator


def test_public_sampling_params_apply_from_token_zero_and_host_decode():
    params = SamplingParams(temperature=1.0, top_k=2, top_p=1.0, seed=0)
    first_generator = _make_host_sampling_generator()
    observed = []
    first = first_generator.generate(
        [3],
        3,
        host_sampling=True,
        enable_trace=False,
        sampling_params=params,
        token_observer=lambda step, token: observed.append((step, token)),
        stop_on_eos=False,
    )
    second = _make_host_sampling_generator().generate(
        [3], 3, host_sampling=True, enable_trace=False, sampling_params=params, stop_on_eos=False
    )

    # Seed zero deliberately selects token 1 instead of the prefill argmax 0;
    # equality proves the request seed governs the full host sampling stream.
    assert first[0] == 1
    assert first == second
    assert observed == list(enumerate(first))
    assert all(token in (0, 1) for token in first)
    assert first_generator.copied_tokens == first[1:]

    # Prefill is a host-logit boundary in both modes, so stochastic parameters
    # must govern token zero before the optimized device decode path starts.
    device_mode_first = _make_host_sampling_generator().generate(
        [3], 1, host_sampling=False, enable_trace=True, sampling_params=params, stop_on_eos=False
    )
    assert device_mode_first == [1]


def test_device_sampling_initializes_seed_and_penalty_history_then_advances_per_token():
    generator = _make_host_sampling_generator()
    params = SamplingParams(
        temperature=0.8,
        top_k=4,
        top_p=0.9,
        presence_penalty=0.2,
        frequency_penalty=0.1,
        repetition_penalty=1.1,
        seed=17,
    )
    formatted = generator._configure_sampling(params)
    generator._initialize_device_sampling_request(
        formatted,
        prompt_token_ids=[7, 8, 7],
        first_output_token=9,
    )
    generator._advance_device_sampling_seed()
    generator._advance_device_sampling_seed()

    sampling = generator.model.sampling
    assert sampling.seed_manager.reset_calls == [(formatted.seed, (0,))]
    assert sampling.seed_manager.advance_calls == [(0,), (0,)]
    assert torch.equal(sampling.prompt_resets[-1], torch.tensor([[7, 8, 7]]))
    assert torch.equal(sampling.output_resets[-1], torch.tensor([[9]]))


def test_low_level_device_sampling_reset_clears_seeded_rows_before_smaller_unseeded_request():
    generator = _make_host_sampling_generator()
    state = types.SimpleNamespace(
        prompt_lens=(),
        active_slots=(),
        token_buffer=torch.zeros(2, dtype=torch.long),
        kv_cache=[],
        linear_state=[],
    )
    generator._state = state
    generator.mesh_device = object()
    generator._resolve_state = lambda **kwargs: state
    generator._refresh_page_table = lambda *args, **kwargs: None
    generator.model.prepare_decode_state = (
        lambda resolved, tokens, positions=None: setattr(
            resolved, "active_slots", tuple(range(tokens.numel()))
        )
    )
    sampled_batches = iter(
        (
            torch.tensor([11, 12]),
            torch.tensor([13, 14]),
            torch.tensor([15]),
            torch.tensor([16]),
            torch.tensor([17]),
        )
    )
    generator.model.decode_device = lambda resolved: torch.zeros(2, 4)
    generator.model.sampling.sample = lambda *args, **kwargs: next(sampled_batches)
    generator._tokens_to_host = lambda tensor: tensor.reshape(-1).to(torch.long)

    params = SamplingParams(
        temperature=[0.8, 0.9],
        top_k=[4, 8],
        top_p=[0.9, 0.8],
        presence_penalty=[0.2, 0.1],
        frequency_penalty=[0.1, 0.05],
        repetition_penalty=[1.1, 1.05],
        seed=[17, 29],
    )
    first = generator.decode_forward(
        torch.tensor([[9], [10]]),
        torch.tensor([17, 23]),
        page_table=None,
        kv_cache=state,
        state=state,
        enable_trace=False,
        sampling_mode="device",
        sampling_params=params,
        sampling_request_start=True,
        prompt_token_ids=[[1, 2, 3], [4]],
    )
    second = generator.decode_forward(
        first.reshape(2, 1),
        torch.tensor([18, 24]),
        page_table=None,
        kv_cache=state,
        state=state,
        enable_trace=False,
        sampling_mode="device",
    )
    assert first.tolist() == [11, 12]
    assert second.tolist() == [13, 14]
    assert generator._device_sampling_slots == (0, 1)
    assert generator._device_output_history_by_slot == [[9, 11, 13], [10, 12, 14]]
    assert generator.model.sampling.seed_manager.advance_calls == [(0, 1), (0, 1)]
    assert generator.model.sampling.seed_manager.seed_counters[:2] == [2, 2]
    formatted = generator.model.sampling.params
    assert formatted.seed[:3] == [17, 29, None]
    assert formatted.presence_penalty[:3] == [0.2, 0.1, 0.0]
    assert torch.equal(
        generator.model.sampling.prompt_resets[-1], torch.tensor([[1, 2, 3], [4, -1, -1]])
    )
    assert torch.equal(generator.model.sampling.output_resets[-1], torch.tensor([[9], [10]]))

    generator.reset()
    assert not generator._device_sampling_request_active
    assert not generator._device_sampling_slots
    assert not generator._device_output_history_by_slot
    seed_manager = generator.model.sampling.seed_manager
    assert seed_manager.seeds == [None] * MAX_BATCH_SIZE
    assert seed_manager.seed_counters == [0] * MAX_BATCH_SIZE
    assert not seed_manager._seed_active
    assert not seed_manager._reseted
    assert not seed_manager._needs_skip
    assert not seed_manager._active_request_seed
    unseeded = generator.decode_forward(
        torch.tensor([[9]]),
        torch.tensor([17]),
        page_table=None,
        kv_cache=state,
        state=state,
        enable_trace=False,
        sampling_mode="device",
        sampling_params=SamplingParams(temperature=0.8, top_k=4, top_p=0.9),
        sampling_request_start=True,
        prompt_token_ids=[[1, 2, 3]],
    )
    assert unseeded.tolist() == [15]
    assert seed_manager.reset_calls[-1][1] == (0,)
    assert seed_manager.seeds == [None] * MAX_BATCH_SIZE
    assert seed_manager.seed_counters == [0] * MAX_BATCH_SIZE
    assert not seed_manager._seed_active
    assert not seed_manager._reseted
    assert seed_manager._needs_skip  # normal one-time unseeded init transition

    unseeded_second = generator.decode_forward(
        unseeded.reshape(1, 1),
        torch.tensor([18]),
        page_table=None,
        kv_cache=state,
        state=state,
        enable_trace=False,
        sampling_mode="device",
    )
    unseeded_third = generator.decode_forward(
        unseeded_second.reshape(1, 1),
        torch.tensor([19]),
        page_table=None,
        kv_cache=state,
        state=state,
        enable_trace=False,
        sampling_mode="device",
    )
    assert unseeded_second.tolist() == [16]
    assert unseeded_third.tolist() == [17]
    assert seed_manager.seeds == [None] * MAX_BATCH_SIZE
    assert seed_manager.seed_counters == [0] * MAX_BATCH_SIZE
    assert not seed_manager._seed_active
    assert not seed_manager._reseted
    assert not seed_manager._needs_skip
    assert not seed_manager._active_request_seed
    assert seed_manager.explicit_seed_advance_calls == [(0, 1), (0, 1)]


def test_penalty_history_update_is_captured_and_not_dispatched_after_trace(monkeypatch):
    """Penalty bookkeeping must be part of the sampler trace.

    Dispatching ``update_output_tokens`` from Python after sampler replay creates
    temporary device buffers while the model and sampler traces are resident.
    """
    trace_active = False
    updates = []
    executions = []

    class FakePenalties:
        def apply(self, logits):
            return logits

        def update_output_tokens(self, tokens):
            updates.append((tokens, trace_active))

    class FakeSampling:
        force_argmax_sampling = False

        def __call__(self, logits, *, tt_out_tok=None):
            del logits
            return tt_out_tok or "sampled-token", "log-probs"

    class FakeSeedManager:
        @staticmethod
        def has_active_request_seed():
            return False

    sampler = object.__new__(SamplingGenerator)
    sampler.mesh_device = "mesh"
    sampler.cq_id = 0
    sampler._penalties_active = True
    sampler._log_probs_active = False
    sampler.trace_seeded_sampling = True
    sampler._sampling_debug_enabled = False
    sampler._trace_states = {}
    sampler.tt_penalties = FakePenalties()
    sampler.tt_sampling = FakeSampling()
    sampler.seed_manager = FakeSeedManager()

    def begin_trace(*args, **kwargs):
        nonlocal trace_active
        del args, kwargs
        trace_active = True
        return "trace-id"

    def end_trace(*args, **kwargs):
        nonlocal trace_active
        del args, kwargs
        trace_active = False

    monkeypatch.setattr(ttnn, "begin_trace_capture", begin_trace)
    monkeypatch.setattr(ttnn, "end_trace_capture", end_trace)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "execute_trace", lambda *args, **kwargs: executions.append("replay"))

    logits = types.SimpleNamespace(shape=(1, 1, 32, 262_144))
    token_buffer = types.SimpleNamespace(shape=(1, 1, 1, 32))
    output = sampler.capture_trace(logits, tt_out_tok=token_buffer)
    assert output == (token_buffer, "log-probs")
    # Compile-only warmup must not mutate request history; the sole update is
    # recorded in the trace and therefore runs on-device at replay time.
    assert updates == [(token_buffer, True)]

    updates.clear()
    replayed = sampler.sample(logits, enable_trace=True, tt_out_tok=token_buffer)
    assert replayed == output
    assert executions == ["replay"]
    assert updates == [], "replay must not dispatch allocating penalty ops from Python"


def test_host_sampling_top_p_filters_before_multinomial():
    generator = _make_host_sampling_generator()
    params = SamplingParams(temperature=1.0, top_k=4, top_p=0.5, seed=0)
    generator._reset_host_sampling_rng(params)
    # Token zero alone crosses p=0.5 and must be retained; all tail tokens are
    # excluded even though seed zero selects token one in the unfiltered case.
    token = generator._sample_host_logits(
        torch.tensor([2.0, 1.9, -4.0, -5.0]), params, token_history=[]
    )
    assert token == 0


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_reduced_full_model_prefill_decode_and_split_trace(mesh_device):
    if os.environ.get("QWEN36_RUN_FULL_MODEL_SMOKE") != "1":
        pytest.skip("set QWEN36_RUN_FULL_MODEL_SMOKE=1")
    generator = Generator(
        mesh_device=mesh_device,
        override_num_layers=4,
        max_seq_len=256,
    )
    prompt = [151644, 872, 198]
    logits = generator.prefill_forward(
        torch.tensor([prompt]),
        page_table=None,
        kv_cache=None,
        prompt_lens=[len(prompt)],
        return_all_logits=True,
    )
    assert logits.shape == (1, len(prompt), generator.model.vocab_size)
    assert torch.isfinite(logits).all()
    predictions = generator.generate(prompt, 3, enable_trace=True, stop_on_eos=False)
    assert len(predictions) == 3
    assert all(0 <= token < generator.model.vocab_size for token in predictions)
    assert generator._model_trace_id is not None
    assert generator.model.sampling._trace_states
    assert generator.model.sampling.tt_sampling.force_argmax_sampling
    persistent_inputs = (
        generator._state.token_buffer,
        generator._state.current_positions,
        generator._state.rotary_positions,
    )
    assert generator._tokens_to_host(generator._state.token_buffer)[0].item() == predictions[-1]
    positions = generator._tokens_to_host(generator._state.current_positions)
    assert positions[0].item() == len(prompt) + 2
    assert torch.all(positions[1:] == -1)
    assert all(
        current is original
        for current, original in zip(
            (
                generator._state.token_buffer,
                generator._state.current_positions,
                generator._state.rotary_positions,
            ),
            persistent_inputs,
        )
    )

    changed_page_table = torch.arange(
        generator._state.page_table_host.shape[1] - 1, -1, -1, dtype=torch.int32
    ).unsqueeze(0)
    generator._refresh_page_table(generator._state, changed_page_table)
    first_snapshot = generator._previous_page_table
    stable_page_table = changed_page_table.clone()
    generator._refresh_page_table(generator._state, stable_page_table)
    assert generator._previous_page_table is first_snapshot
    page_table_reads = 0
    original_page_table_to_host = generator._page_table_to_host

    def count_page_table_reads(self, value):
        nonlocal page_table_reads
        page_table_reads += 1
        return original_page_table_to_host(value)

    generator._page_table_to_host = types.MethodType(count_page_table_reads, generator)
    generator._refresh_page_table(generator._state, stable_page_table)
    assert page_table_reads == 0
    generator._refresh_page_table(generator._state, stable_page_table, changed=True)
    assert page_table_reads == 1
    page_table_device = generator._tokens_to_host(generator._state.page_table).reshape(
        generator._state.page_table_host.shape
    )
    assert torch.equal(
        page_table_device[0, : changed_page_table.shape[1]], changed_page_table[0]
    )
    generator.reset()
    assert not generator._state.prompt_lens
    assert not generator._state.active_slots
    assert generator._model_trace_id is None
    assert not generator.model.sampling._trace_states

    # A second request must recapture rather than replaying tensor bindings or
    # state from the first request, and repeated reset must be deterministic.
    second_prompt = [151644, 3838, 374, 220, 19]
    second = generator.generate(second_prompt, 3, enable_trace=True, stop_on_eos=False)
    assert generator._model_trace_id is not None
    generator.reset()
    second_control = generator.generate(second_prompt, 3, enable_trace=True, stop_on_eos=False)
    assert second == second_control

    # Seeded stochastic decode uses the common device sampler's request state:
    # prompt and token-zero penalty history are installed once, then the
    # explicit seed advances once per device-decoded token.  Resetting the
    # request must reproduce the same stream on the proven physical Ring.
    stochastic = SamplingParams(
        temperature=0.8,
        top_k=16,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.05,
        repetition_penalty=1.05,
        seed=17,
    )
    generator.reset()
    stochastic_first = generator.generate(
        second_prompt, 3, enable_trace=True, sampling_params=stochastic, stop_on_eos=False
    )
    assert generator.model.sampling.seed_manager.seed_counters[0] == 2
    print(
        "stochastic_seeded_first "
        f"tokens={stochastic_first} seed_counter="
        f"{generator.model.sampling.seed_manager.seed_counters[0]} "
        f"topology={generator.model.sampling.tt_sampling._get_sampling_all_gather_config(None, 1)[1]}"
    )
    assert generator._device_output_history == stochastic_first
    assert generator.model.sampling._penalties_active
    prompt_mask = ttnn.to_torch(
        generator.model.sampling.tt_penalties.prompt_mask,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1),
    )
    assert torch.all(prompt_mask[0, second_prompt] == 1)
    output_counts = generator._tokens_to_host(
        generator.model.sampling.tt_penalties.output_counts_gathered
    ).reshape(MAX_BATCH_SIZE, -1)
    expected_counts = torch.bincount(
        torch.tensor(stochastic_first), minlength=generator.model.vocab_size
    )
    sampled_ids = torch.unique(torch.tensor(stochastic_first))
    assert torch.equal(output_counts[0, sampled_ids], expected_counts[sampled_ids])
    assert generator.model.sampling.tt_sampling._get_sampling_all_gather_config(None, 1) == (
        1,
        ttnn.Topology.Ring,
    )
    generator.reset()
    stochastic_second = generator.generate(
        second_prompt, 3, enable_trace=True, sampling_params=stochastic, stop_on_eos=False
    )
    assert stochastic_first == stochastic_second
    print(
        "stochastic_seeded_replay "
        f"tokens={stochastic_second} seed_counter="
        f"{generator.model.sampling.seed_manager.seed_counters[0]} reproducible=true"
    )

    # The ordinary unseeded top-k/top-p mode retains the common sampler trace;
    # three tokens force both capture and replay while Ring remains selected.
    unseeded = SamplingParams(temperature=0.8, top_k=16, top_p=0.9, seed=None)
    generator.reset()
    unseeded_tokens = generator.generate(
        second_prompt, 3, enable_trace=True, sampling_params=unseeded, stop_on_eos=False
    )
    print(f"stochastic_unseeded tokens={unseeded_tokens} trace_capture_replay=true")
    stochastic_trace = next(
        slot
        for key, slot in generator.model.sampling._trace_states.items()
        if not key.force_argmax
    )
    assert stochastic_trace["id"] is not None
    assert generator._tokens_to_host(generator._state.token_buffer)[0].item() == unseeded_tokens[-1]
    assert generator._tokens_to_host(generator._state.current_positions)[0].item() == len(second_prompt) + 2

    # Switching back to force-argmax must preserve valid device feedback and
    # return to the semantically greedy trace key.
    generator._configure_greedy_sampling()
    greedy_out = generator._decode_traced_device(generator._state)
    greedy_token = generator._tokens_to_host(greedy_out)[0].item()
    assert 0 <= greedy_token < generator.model.vocab_size
    generator.teardown()
    del generator
    gc.collect()

    # Public prefill owns non-aligned padding and supports mixed prompt lengths
    # while keeping inactive decode rows outside the recurrent linear state.
    mixed = Generator(mesh_device=mesh_device, override_num_layers=4, max_seq_len=512)
    boundary_prompt = [151644, 872] + [198] * 63
    mixed_tokens = torch.tensor(
        [boundary_prompt + [0, 0], boundary_prompt + [220, 19], boundary_prompt + [0, 0]],
        dtype=torch.long,
    )
    mixed_logits = mixed.prefill_forward(
        mixed_tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=[65, 67, 65],
        return_all_logits=True,
    )
    assert mixed_logits.shape == (3, 67, mixed.model.vocab_size)
    assert torch.equal(mixed_logits[0, :65], mixed_logits[2, :65])
    assert mixed._state.active_slots == (0, 1, 2)
    for layer_idx, layer in enumerate(mixed.model.layers):
        if layer.layer_kind == "linear_attention":
            assert mixed._state.linear_state[layer_idx][0].shape[0] == 3
    mixed_first_tokens = torch.tensor(
        [
            torch.argmax(mixed_logits[user, length - 1]).item()
            for user, length in enumerate((65, 67, 65))
        ]
    )
    mixed_sampled_device = mixed.decode_forward(
        mixed_first_tokens.reshape(3, 1),
        torch.tensor([65, 67, 65]),
        page_table=mixed._state.page_table_host[:3],
        kv_cache=mixed._state,
        state=mixed._state,
        enable_trace=True,
        sampling_mode="device",
        sampling_params=SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None),
        sampling_request_start=True,
        prompt_token_ids=[boundary_prompt[:65], boundary_prompt + [220, 19], boundary_prompt[:65]],
        read_from_device=False,
    )
    mixed_sampled_async = mixed.read_decode_output(mixed_sampled_device, async_read=True)
    mixed_sampled = mixed.process_decode_output_host(mixed_sampled_async, is_tokens=True)
    assert mixed_sampled.shape == (3,)
    assert mixed_sampled[0].item() == mixed_sampled[2].item()
    assert len(mixed._device_output_history_by_slot) == 3
    assert all(len(history) == 2 for history in mixed._device_output_history_by_slot)
    assert mixed._device_output_history_by_slot[0] == mixed._device_output_history_by_slot[2]
    mixed_positions = mixed._tokens_to_host(mixed._state.current_positions)
    assert mixed_positions[:3].tolist() == [66, 68, 66]
    assert torch.all(mixed_positions[3:] == -1)
    mixed.teardown()
    del mixed
    gc.collect()

    # Low-level serving can enter decode from externally owned state without
    # a preceding wrapper prefill; explicit positions are authoritative.
    direct = Generator(mesh_device=mesh_device, override_num_layers=4, max_seq_len=256)
    direct_state = direct.model.allocate_state(batch_size=1)
    direct_device_token = direct.decode_forward(
        torch.tensor([[151644]]),
        torch.tensor([17]),
        page_table=None,
        kv_cache=direct_state,
        state=direct_state,
        enable_trace=False,
        sampling_mode="device",
        sampling_params=SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None),
        sampling_request_start=True,
        prompt_token_ids=[[151644]],
        read_from_device=False,
    )
    direct_host_token = direct.read_decode_output(direct_device_token, async_read=True)
    direct_token = direct.process_decode_output_host(direct_host_token, is_tokens=True)
    assert direct_token.shape == (1,)
    assert direct._device_output_history_by_slot[0] == [151644, direct_token.item()]
    direct_positions = direct._tokens_to_host(direct_state.current_positions)
    assert direct_positions[0].item() == 18
    assert direct_state.prompt_lens == (17,)
    direct.teardown()
    del direct
    gc.collect()

    # Fixed-slot batch-32 decode uses explicitly disjoint physical cache
    # blocks and returns one token per active row without host logits.
    batched = Generator(mesh_device=mesh_device, override_num_layers=4, max_seq_len=2048)
    batch_page_table = torch.arange(MAX_BATCH_SIZE, dtype=torch.int32).reshape(MAX_BATCH_SIZE, 1)
    batch_state = batched.model.allocate_state(
        batch_size=MAX_BATCH_SIZE, page_table=batch_page_table
    )
    batch_tokens = torch.arange(151644, 151644 + MAX_BATCH_SIZE).reshape(MAX_BATCH_SIZE, 1)
    batch_out = batched.decode_forward(
        batch_tokens,
        torch.zeros(MAX_BATCH_SIZE, dtype=torch.int32),
        page_table=batch_page_table,
        kv_cache=batch_state,
        state=batch_state,
        enable_trace=True,
        sampling_mode="device",
        sampling_params=SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None),
        sampling_request_start=True,
        prompt_token_ids=[[int(token)] for token in batch_tokens.reshape(-1)],
    )
    assert batch_out.shape == (MAX_BATCH_SIZE,)
    assert torch.all((0 <= batch_out) & (batch_out < batched.model.vocab_size))
    batch_positions = batched._tokens_to_host(batch_state.current_positions)
    assert batch_positions.tolist() == [1] * MAX_BATCH_SIZE
    batched.teardown()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_real_weight_lm_head_alignment(mesh_device):
    """Autofix isolation: TP vocabulary ordering must match the HF head exactly."""
    if os.environ.get("QWEN36_RUN_LM_HEAD_DIAGNOSTIC") != "1":
        pytest.skip("set QWEN36_RUN_LM_HEAD_DIAGNOSTIC=1")
    checkpoint = LazySafetensorState(_resolve_checkpoint(None))
    weight = checkpoint.tensor("lm_head.weight")
    head = VocabParallelLMHead(
        weight, mesh_device, hidden_size=weight.shape[1], vocab_size=weight.shape[0]
    )
    torch.manual_seed(3600)
    hidden = torch.randn(1, 1, MAX_BATCH_SIZE, weight.shape[1], dtype=torch.bfloat16) * 0.1
    tt_hidden = ttnn.from_torch(
        hidden,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    actual = ttnn.to_torch(
        head(tt_hidden, decode=True, mask_invalid=False),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1),
    )[..., : weight.shape[0]]
    reference = torch.nn.functional.linear(hidden.float(), weight.float())
    row = 0
    overlap = len(
        set(torch.topk(actual[0, 0, row].float(), 100).indices.tolist())
        & set(torch.topk(reference[0, 0, row], 100).indices.tolist())
    )
    pcc = torch.corrcoef(
        torch.stack([actual[0, 0, row].float(), reference[0, 0, row]])
    )[0, 1].item()
    print(f"lm_head_pcc={pcc:.8f} top100_overlap={overlap}/100")
    assert pcc >= 0.99
    assert overlap >= 95


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_reduced_full_model_max_context_decode(mesh_device, monkeypatch):
    """Exercise full-wrapper state/page/RoPE geometry at the advertised final position."""
    if os.environ.get("QWEN36_RUN_CONTEXT_WRAPPER") != "1":
        pytest.skip("set QWEN36_RUN_CONTEXT_WRAPPER=1")
    generator = Generator(
        mesh_device=mesh_device,
        override_num_layers=4,
        max_seq_len=262_144,
    )
    state = generator.model.allocate_state(batch_size=1)
    assert state.num_blocks == 4096
    assert state.page_table_host.shape == (MAX_BATCH_SIZE, 4096)
    capture_positions = []
    original_begin_trace = ttnn.begin_trace_capture

    def checked_begin_trace(mesh, *args, **kwargs):
        current = generator._tokens_to_host(state.current_positions)[0].item()
        capture_positions.append(current)
        assert current == 262_143, "trace setup advanced beyond the supported context"
        return original_begin_trace(mesh, *args, **kwargs)

    monkeypatch.setattr(ttnn, "begin_trace_capture", checked_begin_trace)
    sampled = generator.decode_forward(
        torch.tensor([[151644]]),
        torch.tensor([262_143]),
        page_table=state.page_table_host[:1],
        kv_cache=state,
        state=state,
        enable_trace=True,
        sampling_mode="device",
        sampling_params=SamplingParams(temperature=1.0, top_k=1, top_p=0.0, seed=None),
        sampling_request_start=True,
        prompt_token_ids=[[151644]],
    )
    assert sampled.shape == (1,)
    # One begin is the model trace and one is the common sampler trace.  Both
    # must see the original logical position at their capture boundary.
    assert capture_positions == [262_143, 262_143]
    positions = generator._tokens_to_host(state.current_positions)
    assert positions[0].item() == 262_144
    generator.teardown()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 0, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
def test_reduced_stack_first_divergence(mesh_device):
    """Autofix isolation: report cumulative hidden-state PCC after each layer."""
    if os.environ.get("QWEN36_RUN_STACK_DIAGNOSTIC") != "1":
        pytest.skip("set QWEN36_RUN_STACK_DIAGNOSTIC=1")
    model = QwenFullModel(mesh_device=mesh_device, override_num_layers=4, max_seq_len=256)
    checkpoint = LazySafetensorState(model.checkpoint_path)
    tokens = torch.tensor([[248045, 846, 198] + [198] * 29], dtype=torch.long)
    reference = torch.nn.functional.embedding(
        tokens, checkpoint.tensor("model.language_model.embed_tokens.weight")
    ).to(torch.bfloat16)
    actual = model._embed_host_tokens(tokens[0])
    embedded_host = ttnn.to_torch(
        actual, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )[0, 0, : tokens.shape[1]].float()
    embedding_pcc = torch.corrcoef(
        torch.stack([reference.float().reshape(-1), embedded_host.reshape(-1)])
    )[0, 1].item()
    print(f"embedding_pcc={embedding_pcc:.8f}")
    assert embedding_pcc >= 0.999
    original_embedding = reference.float().clone()
    state = model.allocate_state(batch_size=1)
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)
    cos_host, sin_host = model.hf_rope(reference, positions)
    cos = model._to_replicated(cos_host.unsqueeze(1))
    sin = model._to_replicated(sin_host.unsqueeze(1))
    page_row = model._to_replicated(
        state.page_table_host[:1], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    reference_cache = DynamicCache(config=model.config)
    causal_mask = torch.full(
        (1, 1, tokens.shape[1], tokens.shape[1]), torch.finfo(torch.float32).min
    )
    causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)
    for layer_idx, layer in enumerate(model.layers):
        layer_state = checkpoint.layer(layer_idx)
        prefix = f"model.language_model.layers.{layer_idx}."
        reference_layer = Qwen3_5DecoderLayer(model.config, layer_idx).to(torch.bfloat16).eval()
        reference_layer.load_state_dict(
            {key.removeprefix(prefix): value for key, value in layer_state.items()}, strict=True
        )
        with torch.no_grad():
            reference = reference_layer(
                reference,
                position_embeddings=(cos_host, sin_host),
                attention_mask=(
                    torch.ones((1, tokens.shape[1]), dtype=torch.bool)
                    if layer.layer_kind == "linear_attention"
                    else causal_mask
                ),
                position_ids=positions,
                past_key_values=reference_cache,
            )
        if layer.layer_kind == "linear_attention":
            actual = layer.prefill_forward(
                actual,
                logical_seq_len=tokens.shape[1],
                linear_state=layer.allocate_linear_state(batch_size=1),
            )
        else:
            actual = layer.prefill_forward(
                actual,
                logical_seq_len=tokens.shape[1],
                cos=cos,
                sin=sin,
                page_table=page_row,
                kv_cache=state.kv_cache[layer_idx],
            )
        actual_host = ttnn.to_torch(
            actual, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )[0, 0, : tokens.shape[1]].float()
        pcc = torch.corrcoef(
            torch.stack([reference.float().reshape(-1), actual_host.reshape(-1)])
        )[0, 1].item()
        print(f"layer={layer_idx} kind={layer.layer_kind} cumulative_pcc={pcc:.8f}")
        if layer_idx == 0:
            actual_input_pcc = torch.corrcoef(
                torch.stack([actual_host.reshape(-1), original_embedding.reshape(-1)])
            )[0, 1].item()
            reference_input_pcc = torch.corrcoef(
                torch.stack([reference.float().reshape(-1), original_embedding.reshape(-1)])
            )[0, 1].item()
            print(
                f"layer0_actual_vs_input_pcc={actual_input_pcc:.8f} "
                f"layer0_reference_vs_input_pcc={reference_input_pcc:.8f} "
                f"input_std={original_embedding.std().item():.8f} "
                f"actual_std={actual_host.std().item():.8f} "
                f"reference_std={reference.float().std().item():.8f}"
            )
        assert pcc >= 0.98


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_real_embedding_singlechip_layer0(mesh_device):
    """Autofix isolation: distinguish base decoder math from TP4 integration."""
    if os.environ.get("QWEN36_RUN_SINGLECHIP_STACK_DIAGNOSTIC") != "1":
        pytest.skip("set QWEN36_RUN_SINGLECHIP_STACK_DIAGNOSTIC=1")
    checkpoint = LazySafetensorState(_resolve_checkpoint(None))
    config = AutoConfig.from_pretrained(checkpoint.checkpoint, local_files_only=True).text_config
    layer_state = checkpoint.layer(0)
    decoder = FunctionalDecoder.from_state_dict(
        layer_state, hf_config=config, layer_idx=0, mesh_device=mesh_device
    )
    if os.environ.get("QWEN36_DIAGNOSTIC_HOST_INVERSE") == "1":
        def _host_inverse(self, base_attention):
            base_host = ttnn.to_torch(base_attention).float()
            identity = torch.eye(base_host.shape[-1]).reshape(1, 1, *base_host.shape[-2:])
            inverse = torch.linalg.inv(identity - base_host)
            return ttnn.from_torch(
                inverse,
                device=mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        decoder._linear_chunk_inverse = types.MethodType(_host_inverse, decoder)
    reference_layer = Qwen3_5DecoderLayer(config, 0).to(torch.bfloat16).eval()
    prefix = "model.language_model.layers.0."
    reference_layer.load_state_dict(
        {key.removeprefix(prefix): value for key, value in layer_state.items()}, strict=True
    )
    tokens = torch.tensor([[248045, 846, 198] + [198] * 29], dtype=torch.long)
    hidden = torch.nn.functional.embedding(
        tokens, checkpoint.tensor("model.language_model.embed_tokens.weight")
    ).to(torch.bfloat16)
    with torch.no_grad():
        reference_norm = reference_layer.input_layernorm(hidden)
        reference_mixed = reference_layer.linear_attn(
            reference_norm,
            cache_params=DynamicCache(config=config),
            attention_mask=torch.ones((1, tokens.shape[1]), dtype=torch.bool),
        )
        reference = reference_layer(
            hidden,
            position_embeddings=(
                torch.zeros(1, tokens.shape[1], 64, dtype=torch.bfloat16),
                torch.zeros(1, tokens.shape[1], 64, dtype=torch.bfloat16),
            ),
            attention_mask=torch.ones((1, tokens.shape[1]), dtype=torch.bool),
            past_key_values=DynamicCache(config=config),
        )
    tt_hidden = ttnn.from_torch(
        hidden.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_norm = ttnn.rms_norm(
        tt_hidden, epsilon=decoder.rms_norm_eps, weight=decoder.input_norm
    )
    actual_mixed = decoder._linear_prefill(
        tt_norm,
        linear_state=decoder.allocate_linear_state(batch_size=1),
        logical_seq_len=tokens.shape[1],
    )
    actual = decoder._finish_layer(tt_hidden, actual_mixed)
    norm_host = ttnn.to_torch(tt_norm)[0, 0].float()
    mixed_host = ttnn.to_torch(actual_mixed)[0, 0].float()
    norm_pcc = torch.corrcoef(
        torch.stack([reference_norm.float().reshape(-1), norm_host.reshape(-1)])
    )[0, 1].item()
    mixed_pcc = torch.corrcoef(
        torch.stack([reference_mixed.float().reshape(-1), mixed_host.reshape(-1)])
    )[0, 1].item()
    actual_host = ttnn.to_torch(actual)[0, 0].float()
    pcc = torch.corrcoef(
        torch.stack([reference.float().reshape(-1), actual_host.reshape(-1)])
    )[0, 1].item()
    print(
        f"singlechip_layer0_norm_pcc={norm_pcc:.8f} mixed_pcc={mixed_pcc:.8f} "
        f"mixed_actual_std={mixed_host.std().item():.8f} "
        f"mixed_reference_std={reference_mixed.float().std().item():.8f} "
        f"full_pcc={pcc:.8f} "
        f"actual_std={actual_host.std().item():.8f} reference_std={reference.float().std().item():.8f}"
    )
    assert norm_pcc >= 0.999
    assert mixed_pcc >= 0.98
    assert pcc >= 0.98


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_reduced_token_out_latency_breakdown(mesh_device):
    """Opt-in sampler A/B and split-trace latency breakdown on real TP4 shapes."""
    if os.environ.get("QWEN36_RUN_TOKEN_OUT_BENCHMARK") != "1":
        pytest.skip("set QWEN36_RUN_TOKEN_OUT_BENCHMARK=1")
    profiling = os.environ.get("QWEN36_PROFILE_TOKEN_OUT") == "1"
    profiling_prefill = os.environ.get("QWEN36_PROFILE_PREFILL") == "1"
    if profiling and profiling_prefill:
        raise ValueError("profile prefill and token-out in separate hardware runs")
    metrics_json = os.environ.get("QWEN36_TOKEN_OUT_METRICS_JSON")
    if (profiling or profiling_prefill) and metrics_json:
        raise ValueError("QWEN36_TOKEN_OUT_METRICS_JSON requires the non-profiling A/B benchmark path")
    layer_indices_env = os.environ.get("QWEN36_BENCH_LAYER_INDICES")
    layer_indices = (
        [int(value) for value in layer_indices_env.split(",")]
        if layer_indices_env
        else None
    )
    generator = Generator(
        mesh_device=mesh_device,
        override_num_layers=(
            None if layer_indices is not None else int(os.environ.get("QWEN36_BENCH_LAYERS", "4"))
        ),
        override_layer_indices=layer_indices,
        max_seq_len=256,
    )
    if profiling_prefill:
        representative_prompt = [151644, 872] + [198] * 126
        # Compile and warm the exact logical/physical prefill shape before
        # flushing setup rows from the reduced full-model profile.
        generator.generate(representative_prompt, 1, enable_trace=True, stop_on_eos=False)
        generator.reset()
        # Compile emits enough profiled device work to fill the small per-RISC
        # buffers.  Flush it before the ordinary warm iteration so Tracy can
        # retain a one-to-one host/device op ledger for post-processing.
        ttnn.ReadDeviceProfiler(mesh_device)
        generator.generate(representative_prompt, 1, enable_trace=True, stop_on_eos=False)
        generator.reset()
        ttnn.ReadDeviceProfiler(mesh_device)
        signpost("QWEN36_FULL_MODEL_PREFILL_START")
        generator.generate(representative_prompt, 1, enable_trace=True, stop_on_eos=False)
        ttnn.synchronize_device(mesh_device)
        signpost("QWEN36_FULL_MODEL_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh_device)
        generator.teardown()
        return
    prompt = [151644, 872, 198]
    generator.generate(prompt, 3, enable_trace=True, stop_on_eos=False)
    state = generator._state
    iterations = 1 if profiling else 10

    def timed(callback):
        start = time.perf_counter()
        for _ in range(iterations):
            callback()
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) * 1_000 / iterations

    model_ms = timed(
        lambda: ttnn.execute_trace(
            mesh_device, generator._model_trace_id, cq_id=0, blocking=False
        )
    )
    sampler_ms = timed(
        lambda: generator.model.sampling.sample(
            generator._trace_logits,
            enable_trace=True,
            tt_out_tok=state.token_buffer,
        )
    )

    def token_out():
        ttnn.execute_trace(mesh_device, generator._model_trace_id, cq_id=0, blocking=False)
        generator.model.sampling.sample(
            generator._trace_logits,
            enable_trace=True,
            tt_out_tok=state.token_buffer,
        )

    token_out_ms = timed(token_out)
    print(
        f"sampler_mode=distributed_local_argmax_ring_greedy model_trace_ms={model_ms:.3f} "
        f"sampler_trace_ms={sampler_ms:.3f} token_out_ms={token_out_ms:.3f} "
        f"token_out_t/s/u={1000.0 / token_out_ms:.3f}"
    )
    if os.environ.get("QWEN36_BENCH_TEACHER_BOUNDARY") == "1":
        token_host = torch.zeros((1, 1, 1, MAX_BATCH_SIZE), dtype=torch.int32)

        def forced_boundary(*, synchronize_mesh: bool):
            token_out()
            token_host[0, 0, 0, 0] = generator._tokens_to_host(state.token_buffer)[0].item()
            if synchronize_mesh:
                ttnn.synchronize_device(mesh_device)
            generator._copy_replicated(
                token_host,
                state.token_buffer,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        unsynchronized_ms = timed(lambda: forced_boundary(synchronize_mesh=False))
        synchronized_ms = timed(lambda: forced_boundary(synchronize_mesh=True))
        print(
            f"teacher_boundary_unsynchronized_ms={unsynchronized_ms:.3f} "
            f"teacher_boundary_synchronized_ms={synchronized_ms:.3f}"
        )
    if profiling:
        # Flush compilation/warmup records so the small profiler DRAM buffer
        # contains the one signposted production trace pair, not setup work.
        ttnn.ReadDeviceProfiler(mesh_device)
        signpost("QWEN36_FULL_MODEL_TOKEN_OUT_START")
        token_out()
        ttnn.synchronize_device(mesh_device)
        signpost("QWEN36_FULL_MODEL_TOKEN_OUT_END")
        ttnn.ReadDeviceProfiler(mesh_device)
        assert generator.model.sampling.tt_sampling.force_argmax_sampling
        generator.teardown()
        return
    assert generator.model.sampling.tt_sampling.force_argmax_sampling
    sampling = generator.model.sampling.tt_sampling

    # The comparison is outside the measured trace pair.  Release both
    # resident traces before either eager sampler can allocate temporaries.
    generator.model.sampling.reset_trace()
    ttnn.release_trace(mesh_device, generator._model_trace_id)
    generator._model_trace_id = None
    host_greedy_token = int(
        torch.argmax(generator._logits_to_host(generator._trace_logits)[0, 0, 0]).item()
    )
    selected_out = generator.model.sampling.sample(
        generator._trace_logits,
        enable_trace=False,
        tt_out_tok=state.token_buffer,
    )
    ttnn.synchronize_device(mesh_device)
    selected_token = generator._tokens_to_host(selected_out)[0].item()
    assert selected_token == host_greedy_token

    # The comparison module/trace is mutually exclusive with the selected
    # split traces; release them before allocating the rejected alternative.
    from models.common.modules.sampling.sampling_1d import Sampling1D

    alternative = Sampling1D.from_model_args(
        mesh_device,
        None,
        generator.model._sampling_args(),
    )
    alternative_out, _ = alternative.decode_forward(
        generator._trace_logits,
        k=sampling.k_tensor,
        p=sampling.p_tensor,
        temp=sampling.temp_tensor,
        tt_out_tok=state.token_buffer,
    )
    ttnn.synchronize_device(mesh_device)
    alternative_token = generator._tokens_to_host(alternative_out)[0].item()
    assert alternative_token == selected_token

    alternative_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    alternative.decode_forward(
        generator._trace_logits,
        k=sampling.k_tensor,
        p=sampling.p_tensor,
        temp=sampling.temp_tensor,
        tt_out_tok=state.token_buffer,
    )
    ttnn.end_trace_capture(mesh_device, alternative_trace_id, cq_id=0)
    alternative_ms = timed(
        lambda: ttnn.execute_trace(
            mesh_device, alternative_trace_id, cq_id=0, blocking=False
        )
    )
    print(
        f"sampler_mode=sampling_1d_greedy sampler_trace_ms={alternative_ms:.3f} "
        f"selected_token={selected_token}"
    )
    metrics_report = None
    if metrics_json:
        selected_links, selected_topology = sampling._get_force_argmax_all_gather_config(
            sampling._get_sampling_cluster_axis()
        )
        alternative_topology = alternative.config.ag_topology
        metrics_report = {
            "schema_version": 1,
            "check": "qwen3_6_token_out_latency_breakdown",
            "runtime": runtime_metadata(
                mesh_device,
                cli={
                    "layers": generator.model.num_layers,
                    "layer_indices": list(generator.model.layer_indices),
                    "iterations": iterations,
                    "fabric_config": "FABRIC_1D_RING",
                    "trace_region_size": DEVICE_PARAMS[0]["trace_region_size"],
                },
            ),
            "selected": {
                "sampler_mode": "distributed_local_argmax_ring_greedy",
                "model_trace_ms": model_ms,
                "sampler_trace_ms": sampler_ms,
                "combined_trace_ms": token_out_ms,
                "token_out_t/s/u": 1000.0 / token_out_ms,
                "selected_token": int(selected_token),
                "host_greedy_token": int(host_greedy_token),
                "all_gather_links": selected_links,
                "topology": str(selected_topology),
                "candidate_value_shape": [1, 1, MAX_BATCH_SIZE, TP_SIZE * 32],
                "candidate_index_shape": [1, 1, TP_SIZE * 32, MAX_BATCH_SIZE],
                "full_vocab_all_gather": False,
            },
            "alternative": {
                "sampler_mode": "sampling_1d_greedy",
                "sampler_trace_ms": alternative_ms,
                "selected_token": int(alternative_token),
                "topology": str(alternative_topology),
            },
        }
    ttnn.release_trace(mesh_device, alternative_trace_id)
    del alternative

    # Representative caller-visible token-out contract: prompt 128, generate
    # 128, and include the sampled-ID readback performed by Generator.generate.
    # Warm the exact prefill/decode shapes first; the earlier short-prompt A/B
    # does not compile the seq-128 prefill programs and therefore cannot make
    # this TTFT measurement warm by itself.
    representative_prompt = [151644, 872] + [198] * 126

    # Same-process inherited-policy TTFT control.  Warm the exact shape, then
    # measure the old allocate/concat/copy recurrent-state path before
    # selecting the persistent-state default below.
    generator.model.reuse_prefill_state = False
    generator.reset()
    warm_tokens = generator.generate(
        representative_prompt,
        2,
        enable_trace=True,
        stop_on_eos=False,
    )
    assert len(warm_tokens) == 2
    generator.reset()
    baseline_observed_times = []
    baseline_start = time.perf_counter()
    baseline_tokens = generator.generate(
        representative_prompt,
        2,
        enable_trace=True,
        stop_on_eos=False,
        token_observer=lambda step, token: baseline_observed_times.append(time.perf_counter()),
    )
    assert len(baseline_tokens) == 2 and len(baseline_observed_times) == 2
    baseline_ttft_ms = (baseline_observed_times[0] - baseline_start) * 1_000

    generator.model.reuse_prefill_state = True
    generator.reset()
    warm_tokens = generator.generate(
        representative_prompt,
        2,
        enable_trace=True,
        stop_on_eos=False,
    )
    assert len(warm_tokens) == 2
    # The observer is read-only and therefore does not introduce host token
    # feedback like the teacher-forcing `next_input` callback.
    generator.reset()
    observed_times = []
    observed_tokens = []

    def observe_token(step, token):
        observed_times.append(time.perf_counter())
        observed_tokens.append((int(step), int(token)))

    representative_start = time.perf_counter()
    representative_tokens = generator.generate(
        representative_prompt,
        128,
        enable_trace=True,
        stop_on_eos=False,
        token_observer=observe_token,
    )
    assert len(representative_tokens) == 128
    assert len(observed_tokens) == 128
    assert observed_tokens == list(enumerate(representative_tokens))
    representative_ttft_ms = (observed_times[0] - representative_start) * 1_000
    # Token zero is selected at the prefill boundary. The 127 intervals after
    # that callback are autonomous traced decode plus caller-visible readback.
    representative_decode_tokens = len(observed_times) - 1
    representative_decode_ms = (observed_times[-1] - observed_times[0]) * 1_000
    representative_ms_per_token = representative_decode_ms / representative_decode_tokens
    representative_tpsu = 1_000.0 / representative_ms_per_token
    print(
        "sampler_mode=representative_caller_visible_greedy prompt_tokens=128 "
        f"generated_tokens=128 measured_decode_tokens={representative_decode_tokens} "
        f"baseline_ttft_ms={baseline_ttft_ms:.3f} "
        f"ttft_ms={representative_ttft_ms:.3f} "
        f"decode_ms_per_token={representative_ms_per_token:.3f} "
        f"token_out_t/s/u={representative_tpsu:.3f}"
    )
    if metrics_report is not None:
        metrics_report["representative_token_out"] = {
            "sampler_mode": "caller_visible_distributed_local_argmax_ring_greedy",
            "prompt_tokens": len(representative_prompt),
            "generated_tokens": len(representative_tokens),
            "measured_decode_tokens": representative_decode_tokens,
            "includes_sampled_id_readback": True,
            "host_token_feedback": False,
            "same_run_inherited_ttft_ms": baseline_ttft_ms,
            "ttft_ms": representative_ttft_ms,
            "decode_total_ms": representative_decode_ms,
            "decode_ms_per_token": representative_ms_per_token,
            "token_out_t/s/u": representative_tpsu,
            "first_token": int(representative_tokens[0]),
            "last_token": int(representative_tokens[-1]),
        }
    generator.teardown()
    if metrics_report is not None:
        write_metrics_json(Path(metrics_json).expanduser().resolve(), metrics_report)


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_distributed_greedy_sampler_trace(mesh_device):
    """Focused compact-greedy eager/capture/replay and token-feedback gate."""
    if os.environ.get("QWEN36_RUN_DISTRIBUTED_ARGMAX") != "1":
        pytest.skip("set QWEN36_RUN_DISTRIBUTED_ARGMAX=1")
    generator = Generator(mesh_device=mesh_device, override_num_layers=1, max_seq_len=256)
    tokens = generator.generate([151644, 872, 198], 4, enable_trace=True, stop_on_eos=False)
    sampling = generator.model.sampling.tt_sampling
    assert sampling.force_argmax_sampling
    assert sampling._allow_distributed_force_argmax
    assert generator._tokens_to_host(generator._state.token_buffer)[0].item() == tokens[-1]

    generator.model.sampling.reset_trace()
    ttnn.release_trace(mesh_device, generator._model_trace_id)
    generator._model_trace_id = None
    host_token = int(torch.argmax(generator._logits_to_host(generator._trace_logits)[0, 0, 0]).item())
    sampled, _ = generator.model.sampling.sample(
        generator._trace_logits,
        enable_trace=False,
        tt_out_tok=generator._state.token_buffer,
    )
    ttnn.synchronize_device(mesh_device)
    device_token = int(generator._tokens_to_host(sampled)[0].item())
    assert device_token == host_token
    print(
        "distributed_greedy "
        f"token={device_token} logits_shape={list(generator._trace_logits.shape)} "
        "candidate_value_shape=[1,1,32,128] candidate_index_shape=[1,1,128,32]"
    )
    generator.teardown()


@pytest.mark.parametrize("mesh_device", MESH_DEVICE, indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_candidate_local_top1_sampler(mesh_device):
    """Autofix isolation for local-top1 plus tiny TP candidate gathers."""
    if os.environ.get("QWEN36_RUN_LOCAL_TOP1") != "1":
        pytest.skip("set QWEN36_RUN_LOCAL_TOP1=1")
    generator = Generator(mesh_device=mesh_device, override_num_layers=1, max_seq_len=256)
    generator.generate([151644, 872, 198], 3, enable_trace=True, stop_on_eos=False)
    logits = generator._trace_logits
    token_buffer = generator._state.token_buffer
    generator._release_split_traces()

    offsets = torch.arange(TP_SIZE, dtype=torch.int32).reshape(1, 1, 1, TP_SIZE)
    offsets = offsets.expand(1, 1, MAX_BATCH_SIZE, TP_SIZE) * (PADDED_VOCAB_SIZE // TP_SIZE)
    tt_offsets = ttnn.from_torch(
        offsets,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    def local_top1():
        local_values = ttnn.max(logits, dim=-1, keepdim=True)
        local_indices = ttnn.argmax(logits, dim=-1, keepdim=True)
        local_indices = ttnn.typecast(local_indices, ttnn.uint32)
        local_indices = ttnn.add(local_indices, tt_offsets)
        values = ttnn.all_gather(
            local_values,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )
        indices = ttnn.all_gather(
            local_indices,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )
        winner = ttnn.argmax(values, dim=-1, keepdim=True)
        winner = ttnn.typecast(winner, ttnn.uint16)
        winner = ttnn.to_layout(winner, ttnn.TILE_LAYOUT)
        tokens = ttnn.gather(indices, 3, index=winner)
        tokens = ttnn.permute(tokens, (0, 1, 3, 2))
        tokens = ttnn.to_layout(tokens, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.copy(tokens, token_buffer)

    candidate = local_top1()
    ttnn.synchronize_device(mesh_device)
    host_token = int(torch.argmax(generator._logits_to_host(logits)[0, 0, 0]).item())
    assert generator._tokens_to_host(candidate)[0].item() == host_token
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    local_top1()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    start = time.perf_counter()
    for _ in range(20):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    latency_ms = (time.perf_counter() - start) * 1000 / 20
    print(f"local_top1_sampler_trace_ms={latency_ms:.3f} selected_token={host_token}")
    assert generator._tokens_to_host(token_buffer)[0].item() == host_token
    ttnn.release_trace(mesh_device, trace_id)
    generator.teardown()
