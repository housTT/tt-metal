# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only protocol tests for the Qwen3.8 vLLM adapter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tt import generator as generator_module
from models.autoports.qwen_qwen3_8_flash_next.tt import generator_vllm as adapter_module
from models.autoports.qwen_qwen3_8_flash_next.tt import model as model_module
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import (
    Qwen38Generator,
    _ServingDecodeHost,
    _ServingDecodeOutput,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import Qwen4ExpForConditionalGeneration
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel, _physical_compressed_ids_host


class _GeneratorSpy:
    def __init__(self) -> None:
        self.prefill_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.decode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.read_calls = []
        self.process_calls = []
        self.release_calls = []
        self.prefill_result = object()
        self.decode_result = object()
        self.read_result = object()
        self.process_result = torch.tensor([7])
        self.model = SimpleNamespace()

    def prefill_forward(self, *args, **kwargs):
        self.prefill_calls.append((args, kwargs))
        return self.prefill_result

    def decode_forward(self, *args, **kwargs):
        self.decode_calls.append((args, kwargs))
        return self.decode_result

    def read_decode_output(self, value, async_read=False):
        self.read_calls.append((value, async_read))
        return self.read_result

    def process_decode_output_host(self, value, is_tokens=False):
        self.process_calls.append((value, is_tokens))
        return self.process_result

    def release_virtual_slots(self, released_slots):
        self.release_calls.append(released_slots)


def _adapter():
    generator = _GeneratorSpy()
    adapter = Qwen4ExpForConditionalGeneration(generator, max_batch_size=2, max_seq_len=262_144)
    adapter._vllm_kv_cache = object()
    return adapter, generator


def _sampling_params():
    return SimpleNamespace(
        temperature=[1.0],
        top_p=[0.0],
        top_k=[1],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[None],
        enable_log_probs=[False],
        num_logprobs=[-2],
    )


def test_physical_compressed_page_map_is_built_once_from_host_table() -> None:
    pages = torch.tensor([[7, 2, 11]], dtype=torch.int32)
    result = _physical_compressed_ids_host(
        pages,
        compressed_blocks=35,
        compressed_per_page=16,
    )
    assert result.shape == (1, 35)
    assert result[0, :16].tolist() == list(range(112, 128))
    assert result[0, 16:32].tolist() == list(range(32, 48))
    assert result[0, 32:].tolist() == [176, 177, 178]
    # The compressed-key cache is stored one 32-row tile per page: 16 keys, 16 padding rows.
    tiled = _physical_compressed_ids_host(pages, compressed_blocks=35, compressed_per_page=16, rows_per_page=32)
    assert tiled[0, :16].tolist() == list(range(7 * 32, 7 * 32 + 16))
    assert tiled[0, 16:32].tolist() == list(range(2 * 32, 2 * 32 + 16))
    assert tiled[0, 32:].tolist() == [11 * 32, 11 * 32 + 1, 11 * 32 + 2]


def test_prefill_program_cache_guard_releases_only_when_trace_replay_would_be_unsafe() -> None:
    generator = object.__new__(Qwen38Generator)
    entries = [17]
    releases = []

    def release_trace():
        releases.append(entries[0])
        generator.model._trace_ready = False

    generator.model = SimpleNamespace(
        _trace_ready=True,
        _virtual_prefill_trace_invalidations=0,
        mesh_device=SimpleNamespace(num_program_cache_entries=lambda: entries[0]),
        release_decode_traces=release_trace,
    )
    generator._serving_program_cache_initialization_uncertain = False

    snapshot = generator._snapshot_live_trace_program_cache()
    assert snapshot == 17
    assert generator._finish_live_trace_prefill(snapshot, failed=False) is False
    assert releases == []

    entries[0] = 18
    assert generator._finish_live_trace_prefill(snapshot, failed=False) is True
    assert releases == [18]
    assert generator.model._trace_ready is False
    assert generator.model._virtual_prefill_trace_invalidations == 1

    generator.model._trace_ready = True
    snapshot = generator._snapshot_live_trace_program_cache()
    assert generator._finish_live_trace_prefill(snapshot, failed=True) is True
    assert releases == [18, 18]
    assert generator.model._virtual_prefill_trace_invalidations == 2


def test_failed_enqueue_forces_future_trace_release_without_cache_count_growth() -> None:
    generator = object.__new__(Qwen38Generator)
    entries = [23]
    releases = []

    def release_trace():
        releases.append(entries[0])
        generator.model._trace_ready = False

    generator.model = SimpleNamespace(
        _trace_ready=True,
        _virtual_prefill_trace_invalidations=0,
        mesh_device=SimpleNamespace(num_program_cache_entries=lambda: entries[0]),
        release_decode_traces=release_trace,
    )
    # A failed first enqueue can leave an existing cache entry with lazy
    # kernel binaries.  Its next hit keeps the same entry count, so persistent
    # uncertainty must release before eager work rather than trust a delta.
    generator._serving_program_cache_initialization_uncertain = True

    assert generator._snapshot_live_trace_program_cache() is None
    assert releases == [23]
    assert generator.model._virtual_prefill_trace_invalidations == 1


def test_failed_guard_marks_program_cache_initialization_uncertain() -> None:
    generator = object.__new__(Qwen38Generator)
    entries = [18]
    releases = []

    def release_trace():
        releases.append(entries[0])
        generator.model._trace_ready = False

    generator.model = SimpleNamespace(
        _trace_ready=True,
        _virtual_prefill_trace_invalidations=0,
        mesh_device=SimpleNamespace(num_program_cache_entries=lambda: entries[0]),
        release_decode_traces=release_trace,
    )
    generator._serving_program_cache_initialization_uncertain = False
    snapshot = generator._snapshot_live_trace_program_cache()

    assert generator._finish_live_trace_prefill(snapshot, failed=True) is True
    assert generator._serving_program_cache_initialization_uncertain is True
    assert releases == [18]

    # The flag outlives request-slot poison cleanup and makes a later live
    # trace conservative before an unchanged cache-hit workload.
    generator.model._trace_ready = True
    assert generator._snapshot_live_trace_program_cache() is None
    assert releases == [18, 18]
    assert generator.model._virtual_prefill_trace_invalidations == 2


def test_virtual_slot_release_is_inside_program_cache_guard() -> None:
    generator = object.__new__(Qwen38Generator)
    entries = [31]
    trace_releases = []

    def release_slot(slot, request, *, generation):
        assert (slot, request, generation) == (0, "request", 7)
        entries[0] += 1

    def release_trace():
        trace_releases.append(entries[0])
        generator.model._trace_ready = False

    generator.model = SimpleNamespace(
        _trace_ready=True,
        _virtual_prefill_trace_invalidations=0,
        mesh_device=SimpleNamespace(num_program_cache_entries=lambda: entries[0]),
        release_virtual_slot=release_slot,
        release_decode_traces=release_trace,
    )
    generator._serving_program_cache_initialization_uncertain = False
    generator._serving_virtual_decode_poisoned = False
    generator._serving_preempted_sampling = {}
    generator._serving_virtual_slots = {
        0: SimpleNamespace(
            request_id="request",
            external_generation=3,
            lease=SimpleNamespace(generation=7),
            sampling_signature=None,
        )
    }

    generator.release_virtual_slots([("request", 0, 3, "finished")])

    assert trace_releases == [32]
    assert generator.model._virtual_prefill_trace_invalidations == 1
    assert generator._serving_virtual_slots == {}


def test_static_protocol_and_full_context_contract() -> None:
    cls = Qwen4ExpForConditionalGeneration
    assert adapter_module.MAX_NUM_SEQS == 1
    assert cls.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_request_specific_rope": False,
        "supports_device_sampling_penalties": False,
        # DEVSTACK-294 mitigation: stochastic rows go to the host sampler.
        "device_sampling_max_top_k": 0,
        "force_host_seeded_sampling": True,
        "supports_virtual_state_slots": False,
        "supports_intermediate_prefill_device_sampling": True,
    }
    assert cls.get_max_tokens_all_users(max_model_len=262_144, max_num_seqs=1) == 262_144
    try:
        cls.get_max_tokens_all_users(max_model_len=262_144, max_num_seqs=2)
    except ValueError as error:
        assert "max_num_seqs=1" in str(error)
    else:
        raise AssertionError("adapter accepted a concurrent serving configuration")
    for name in (
        "tokens",
        "page_table",
        "kv_cache",
        "start_pos",
        "reset_batch",
        "request_ids",
        "state_slot_ids",
        "state_slot_generations",
        "unpadded_batch_size",
        "released_state_slots",
    ):
        assert name in inspect.signature(cls.decode_forward).parameters


def test_initialize_uses_selected_precision_and_tp4_ep4(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        precision_config_path = str(adapter_module.DEFAULT_PRECISION_CONFIG_PATH.resolve())

    class FakeGenerator:
        model = FakeModel()

        def close(self):
            raise AssertionError("selected precision should not close")

    def fake_build(snapshot, mesh, **kwargs):
        captured.update(snapshot=snapshot, mesh=mesh, **kwargs)
        return FakeGenerator()

    monkeypatch.setattr(adapter_module, "build_generator", fake_build)
    mesh = SimpleNamespace(get_num_devices=lambda: 4)
    result = Qwen4ExpForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=adapter_module.MODEL_ID),
        mesh,
        max_batch_size=1,
        max_seq_len=262_144,
    )
    assert result.max_seq_len == 262_144
    assert captured["max_batch"] == 1
    assert captured["virtual_slot_capacity"] == 1
    assert captured["max_seq_len"] == 262_144
    assert captured["precision_config"] == adapter_module.DEFAULT_PRECISION_CONFIG_PATH


def test_allocate_cache_delegates_one_exact_vllm_owned_object() -> None:
    adapter, generator = _adapter()
    adapter._vllm_kv_cache = None
    cache = object()
    calls = []
    generator.model.allocate_vllm_attention_cache = lambda shape: calls.append(shape) or cache
    result = adapter.allocate_kv_cache((4097, 1, 64, 256), torch.bfloat16, 12)
    assert result is cache
    assert adapter._vllm_kv_cache is cache
    assert calls == [(4097, 1, 64, 256)]


def test_vllm_cache_adoption_includes_qsa_compressed_attention_state(
    monkeypatch,
) -> None:
    class FakeTensor:
        def __init__(self, shape, dtype="bf16"):
            self.shape = tuple(shape)
            self.dtype = dtype

    created = []
    released = []

    def fake_zeros(shape, *, dtype, **_kwargs):
        tensor = FakeTensor(shape, dtype)
        created.append(tensor)
        return tensor

    monkeypatch.setattr(model_module.ttnn, "zeros", fake_zeros)
    monkeypatch.setattr(model_module, "dtype_object", lambda _name: "bfp8")
    monkeypatch.setattr(model_module, "_deallocate", released.append)

    old = [FakeTensor((1,)) for _ in range(4)]
    layer = SimpleNamespace(
        shapes=SimpleNamespace(
            layer_idx=3,
            layer_type="qwen_sparse_attention",
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_compress_ratio=4,
        ),
        kv_cache=tuple(old[:2]),
        indexer_cache=old[2],
        fused_index_key_cache=old[3],
        max_num_blocks=4096,
    )
    model = object.__new__(Qwen38FullModel)
    model.layers = [layer]
    model.max_seq_len = 262_144
    model.max_num_blocks = 4096
    model.mesh_device = object()
    model.precision_config = {"kv_cache": {"dtype": "bfp8"}}
    model._trace_ready = False
    model._vllm_kv_cache = None
    model._attention_cache_owner = "model"
    model.attention_cache_lifecycle = {
        "standalone_allocations": 4,
        "vllm_adoptions": 0,
        "standalone_tensors_released": 0,
    }

    cache = model.allocate_vllm_attention_cache((4097, 1, 64, 256))

    assert len(cache) == 1 and len(cache[0]) == 4
    assert cache[0][3].shape == (4097, 1, 32, 128)  # one full tile per page, 16 keys + 16 padding rows
    assert layer.fused_index_key_cache is cache[0][3]
    assert released == old
    assert model._attention_cache_owner == "vllm"


def test_prefill_preserves_non_aligned_logical_length_and_cache_identity() -> None:
    adapter, generator = _adapter()
    tokens = torch.arange(96).reshape(1, 96)
    pages = torch.arange(4096, dtype=torch.int32).reshape(1, 4096)
    result = adapter.prefill_forward(
        tokens=tokens,
        page_table=pages,
        kv_cache=adapter._vllm_kv_cache,
        prompt_lens=[67],
        sampling_params=_sampling_params(),
        empty_slots=[0],
        request_ids=["nonaligned"],
        state_slot_ids=[0],
        state_slot_generations=[1],
        unpadded_batch_size=1,
        start_pos=torch.tensor([0]),
        enable_trace=True,
    )
    assert result is generator.prefill_result
    args, kwargs = generator.prefill_calls[0]
    assert args[0] is tokens
    assert kwargs["page_table"] is pages
    assert kwargs["kv_cache"] is adapter._vllm_kv_cache
    assert kwargs["prompt_lens"] == [67]
    assert kwargs["start_pos"].tolist() == [0]
    assert kwargs["intermediate_prefill_mask"] is None
    assert kwargs["on_device_sampling"] is True
    assert kwargs["request_ids"] == ["nonaligned"]
    assert kwargs["state_slot_ids"] is None
    assert kwargs["state_slot_generations"] is None


def test_prefill_metrics_count_completed_logical_rows_not_grouped_calls() -> None:
    adapter, _ = _adapter()
    common = {
        "tokens": torch.arange(128).reshape(2, 64),
        "page_table": torch.zeros(2, 4096, dtype=torch.int32),
        "kv_cache": adapter._vllm_kv_cache,
        "prompt_lens": [64, 64],
        "sampling_params": _sampling_params(),
        "empty_slots": [0, 1],
        "request_ids": ["a", "b"],
        "state_slot_ids": [0, 1],
        "state_slot_generations": [1, 1],
        "unpadded_batch_size": 2,
        "start_pos": torch.tensor([0, 0]),
    }
    adapter.prefill_forward(**common, intermediate_prefill_mask=torch.tensor([True, True]))
    assert adapter._completed_requests == 0
    adapter.prefill_forward(**common, intermediate_prefill_mask=torch.tensor([False, False]))
    assert adapter._completed_requests == 2


def test_vllm_prefill_cache_pages_follow_logical_tokens_not_compute_padding(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        model_module,
        "_upload_replicated",
        lambda value, *_args, **_kwargs: value.clone(),
    )
    model = object.__new__(Qwen38FullModel)
    model.mesh_device = object()
    state = SimpleNamespace(page_table_host=torch.arange(4096, dtype=torch.int32).reshape(1, 4096))

    cases = {
        1: ((0,),),
        63: ((0,),),
        64: ((0,),),
        65: ((0, 1),),
        67: ((0, 1),),
        127: ((0, 1),),
        129: ((0, 1), (2,)),
    }
    for logical_tokens, expected_pages in cases.items():
        plan = []
        for start in range(0, logical_tokens, 128):
            plan.append((start, min(128, logical_tokens - start), 128))
        layer = SimpleNamespace(prefill_chunk_plan=lambda _length, plan=plan: plan)
        full, chunks = model._prefill_page_inputs(layer, state, 0, logical_tokens)
        assert torch.equal(full, state.page_table_host)
        assert tuple(tuple(int(value) for value in chunk.reshape(-1)) for chunk in chunks) == expected_pages

    layer = SimpleNamespace(
        prefill_chunk_plan=lambda _length: [(0, 128, 128), (128, 1, 128)]
    )
    _, chunks = model._prefill_page_inputs(layer, state, 0, 129, start_pos=1024)
    assert tuple(tuple(int(value) for value in chunk.reshape(-1)) for chunk in chunks) == (
        (16, 17),
        (18,),
    )

    reset_source = inspect.getsource(FunctionalDecoder._reset_user_state)
    assert "ttnn.copy" in reset_source
    assert "ttnn.multiply" not in reset_source
    assert "host_expert_cache.reset" not in inspect.getsource(Qwen38FullModel.reset_batch_state)


def test_batch_one_generator_prefill_advances_only_new_chunk() -> None:
    calls = []

    class FakeModel:
        max_batch = 1
        vocab_size = 32
        _trace_ready = False

        def new_batch_state(self, lengths, *, request_ids, page_table, active_mask):
            del active_mask
            return SimpleNamespace(
                prompt_lens=torch.as_tensor(lengths, dtype=torch.int32).clone(),
                computed_lens=torch.zeros(1, dtype=torch.int32),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                active_slots=(0,),
                page_table_host=torch.as_tensor(page_table).clone(),
                token_input=object(),
            )

        def prefill_forward(self, tokens, **kwargs):
            calls.append((tokens.clone(), dict(kwargs)))
            kwargs["state"].computed_lens = kwargs["prompt_lens"].clone()
            return object()

    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator._serving_device_feedback_current = False
    pages = torch.arange(4096, dtype=torch.int32).reshape(1, 4096)
    tokens = torch.arange(1537, dtype=torch.int64).reshape(1, 1537)

    generator.prefill_forward(
        tokens,
        page_table=pages,
        prompt_lens=[1024],
        start_pos=[0],
        intermediate_prefill_mask=[True],
        request_ids=["request"],
        read_from_device=False,
    )
    generator.prefill_forward(
        tokens,
        page_table=pages,
        prompt_lens=[1537],
        start_pos=[1024],
        intermediate_prefill_mask=[False],
        request_ids=["request"],
        read_from_device=False,
    )

    assert [call[1]["start_pos"].tolist() for call in calls] == [[0], [1024]]
    assert [call[1]["is_final_chunk"] for call in calls] == [False, True]
    assert generator.state.computed_lens.tolist() == [1537]

    with pytest.raises(ValueError, match="committed offsets"):
        generator.prefill_forward(
            tokens,
            page_table=pages,
            prompt_lens=[1537],
            start_pos=[1536],
            intermediate_prefill_mask=[False],
            request_ids=["request"],
            read_from_device=False,
        )


def test_decode_reset_once_then_steady_async_delegation() -> None:
    adapter, generator = _adapter()
    common = dict(
        tokens=torch.tensor([[13]]),
        start_pos=torch.tensor([67]),
        page_table=torch.zeros((1, 4096), dtype=torch.int32),
        kv_cache=adapter._vllm_kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=_sampling_params(),
        request_ids=["active"],
        state_slot_ids=[0],
        state_slot_generations=[1],
        unpadded_batch_size=1,
    )
    adapter.decode_forward(reset_batch=False, **common)
    adapter.decode_forward(reset_batch=False, **common)
    assert generator.decode_calls[0][1]["reset_batch"] is False
    assert generator.decode_calls[1][1]["reset_batch"] is False
    assert all(call[1]["serving_mode"] is True for call in generator.decode_calls)

    read = adapter.read_decode_output(generator.decode_result, async_read=True)
    assert read is generator.read_result
    processed = adapter.process_decode_output_host(read, is_tokens=True)
    assert processed is generator.process_result
    assert generator.read_calls == [(generator.decode_result, True)]
    assert generator.process_calls == [(read, True)]
    adapter.release_virtual_state_slots([("active", 0, 1, "finished")])
    assert generator.release_calls == [[("active", 0, 1, "finished")]]


def test_generator_steady_decode_falls_back_to_device_token_for_ple() -> None:
    ple_inputs = []

    class FakeModel:
        max_batch = 1
        vocab_size = 32

        def _require_kv_cache_identity(self, cache):
            assert cache is kv_cache

        def update_page_table(self, state, table):
            assert state is batch_state
            assert table is page_table

        def set_sampling_params(self, **kwargs):
            assert kwargs["top_k"] == [1]

        def sampled_tokens_to_torch(self, token, state):
            assert token is batch_state.token_input
            return torch.tensor([77])

        def decode_token_out_traced(self, state, values):
            ple_inputs.append(values.clone())
            return object(), sampled_output

    kv_cache = object()
    page_table = object()
    sampled_output = object()
    batch_state = SimpleNamespace(token_input=object(), compact_token_readbacks=0)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator._serving_device_feedback_current = True
    generator._serving_pending_token_host = None
    generator._serving_sampling_signature = None
    generator.host_sampling_compatibility_calls = 0
    generator.async_feedback_host_reuses = 0
    generator.async_feedback_device_fallbacks = 0

    result = generator._decode_forward_serving(
        torch.tensor([[999]]),
        start_pos=torch.tensor([68]),
        page_table=page_table,
        kv_cache=kv_cache,
        state=batch_state,
        read_from_device=False,
        enable_trace=True,
        sampling_params=_sampling_params(),
        reset_batch=False,
        slot_remap=torch.tensor([0]),
    )
    assert isinstance(result, _ServingDecodeOutput)
    assert result.device is sampled_output
    assert ple_inputs[0].reshape(-1).tolist() == [77]
    assert generator.async_feedback_host_reuses == 0
    assert generator.async_feedback_device_fallbacks == 1


def test_generator_steady_decode_reuses_async_host_token_for_ple() -> None:
    ple_inputs = []

    class FakeModel:
        max_batch = 1
        vocab_size = 32

        def _require_kv_cache_identity(self, cache):
            assert cache is kv_cache

        def update_page_table(self, state, table):
            assert state is batch_state
            assert table is page_table

        def set_sampling_params(self, **kwargs):
            assert kwargs["top_k"] == [1]

        def sampled_tokens_to_torch(self, _token, _state):
            raise AssertionError("the already-enqueued async host token must be reused")

        def decode_token_out_traced(self, state, values):
            ple_inputs.append(values.clone())
            return object(), sampled_output

    kv_cache = object()
    page_table = object()
    sampled_output = object()
    batch_state = SimpleNamespace(token_input=object(), compact_token_readbacks=0)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator._serving_device_feedback_current = True
    generator._serving_pending_token_host = _ServingDecodeHost(
        "tokens", torch.tensor([77], dtype=torch.int32), 1, batch_state
    )
    generator._serving_sampling_signature = None
    generator.host_sampling_compatibility_calls = 0
    generator.async_feedback_host_reuses = 0
    generator.async_feedback_device_fallbacks = 0

    result = generator._decode_forward_serving(
        torch.tensor([[999]]),
        start_pos=torch.tensor([68]),
        page_table=page_table,
        kv_cache=kv_cache,
        state=batch_state,
        read_from_device=False,
        enable_trace=True,
        sampling_params=_sampling_params(),
        reset_batch=False,
        slot_remap=torch.tensor([0]),
    )
    assert isinstance(result, _ServingDecodeOutput)
    assert result.device is sampled_output
    assert ple_inputs[0].reshape(-1).tolist() == [77]
    assert generator._serving_pending_token_host is None
    assert generator.async_feedback_host_reuses == 1
    assert generator.async_feedback_device_fallbacks == 0


def test_generator_async_output_supplies_next_virtual_ple_token_once(monkeypatch) -> None:
    events = []
    conversions = []
    completion = object()
    host_tensor = object()
    state = SimpleNamespace(token_input=object(), compact_token_readbacks=0)

    monkeypatch.setattr(generator_module.ttnn, "event_synchronize", lambda event: events.append(event))
    monkeypatch.setattr(generator_module.ttnn, "get_device_tensors", lambda value: (value,))
    monkeypatch.setattr(
        generator_module.ttnn,
        "to_torch",
        lambda value: conversions.append(value) or torch.tensor([77], dtype=torch.int32),
    )

    class FakeModel:
        def sampled_tokens_to_torch(self, token, current_state):
            assert token is state.token_input and current_state is state
            return torch.tensor([88], dtype=torch.int64)

    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.async_feedback_host_reuses = 0
    generator.async_feedback_device_fallbacks = 0
    virtual = generator_module._ServingVirtualSlot(
        "req-a",
        4,
        SimpleNamespace(slot_id=0, request_id="req-a", generation=11),
        True,
    )
    generator._serving_virtual_slots = {0: virtual}
    host = _ServingDecodeHost("tokens", host_tensor, 1, state, completion)
    output = _ServingDecodeOutput("tokens", object(), 1, state, ((0, "req-a", 4),))

    generator._attach_virtual_token_host(output, host)
    assert virtual.pending_token_host == (host, 0)
    assert generator._consume_virtual_ple_token(virtual, state).tolist() == [77]
    # Plugin finalization formats the same host object without another wait,
    # conversion, or compact device read.
    assert host.to_torch(generator.model, is_tokens=True).tolist() == [77]
    assert events == [completion]
    assert conversions == [host_tensor]
    assert state.compact_token_readbacks == 1
    assert generator.async_feedback_host_reuses == 1
    assert generator.async_feedback_device_fallbacks == 0
    assert virtual.pending_token_host is None

    stale = _ServingDecodeOutput("tokens", object(), 1, state, ((0, "req-a", 3),))
    generator._attach_virtual_token_host(stale, host)
    assert virtual.pending_token_host is None
    assert generator._consume_virtual_ple_token(virtual, state).tolist() == [88]
    assert generator.async_feedback_device_fallbacks == 1


def test_generator_virtual_decode_microbatches_real_rows_and_ignores_padding() -> None:
    calls = []
    page_rows = []
    ple_rows = []
    devices = {0: object(), 1: object()}

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 2
        vocab_size = 32

        def _require_kv_cache_identity(self, cache):
            assert cache is kv_cache

        def activate_virtual_slot(self, slot, request, *, generation):
            calls.append(("activate", slot, request, generation))
            self.active_slot = slot
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def commit_virtual_slot(self, slot, request, *, generation):
            calls.append(("commit", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def virtual_slot_token(self, slot, request, *, generation):
            calls.append(("token", slot, request, generation))
            return devices[slot]

        def update_page_table(self, state, table):
            page_rows.append((self.active_slot, table.clone()))

        def set_sampling_params(self, **kwargs):
            calls.append(
                (
                    "sampling",
                    self.active_slot,
                    int(torch.as_tensor(kwargs["top_k"]).reshape(-1)[0]),
                )
            )

        def sampled_tokens_to_torch(self, token, state):
            assert token is state.token_input
            return torch.tensor([70 + self.active_slot])

        def decode_token_out_traced(self, state, values):
            ple_rows.append((self.active_slot, state.request_ids, values.clone()))
            return object(), state.token_input

    kv_cache = object()
    state = SimpleNamespace(token_input=object(), request_ids=("unset",), compact_token_readbacks=0)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = state
    generator._serving_device_feedback_current = False
    generator._serving_sampling_signature = None
    generator._serving_virtual_slots = {
        0: generator_module._ServingVirtualSlot(
            "req-a",
            4,
            SimpleNamespace(slot_id=0, request_id="req-a", generation=11),
            True,
        ),
        1: generator_module._ServingVirtualSlot(
            "req-b",
            7,
            SimpleNamespace(slot_id=1, request_id="req-b", generation=13),
            True,
        ),
    }
    generator.host_sampling_compatibility_calls = 0

    output = generator._decode_forward_serving(
        torch.tensor([[999], [998]]),
        start_pos=torch.tensor([68, 105]),
        page_table=torch.tensor([[10, 11], [20, 21]]),
        kv_cache=kv_cache,
        state=state,
        read_from_device=False,
        enable_trace=True,
        sampling_params=SimpleNamespace(
            temperature=[1.0, 0.8],
            top_p=[0.0, 0.9],
            top_k=[1, 8],
            presence_penalty=[0.0, 0.0],
            frequency_penalty=[0.0, 0.0],
            repetition_penalty=[1.0, 1.0],
            seed=[101, 202],
            enable_log_probs=[False, False],
            num_logprobs=[-2, -2],
        ),
        reset_batch=True,
        slot_remap=None,
        request_ids=["req-a", "req-b"],
        state_slot_ids=[0, 1],
        state_slot_generations=[4, 7],
        unpadded_batch_size=2,
    )

    assert isinstance(output, _ServingDecodeOutput)
    assert output.device == (devices[0], devices[1])
    assert output.rows == 2
    assert [item[0] for item in calls if item[0] in {"activate", "commit"}] == [
        "activate",
        "commit",
        "activate",
        "commit",
    ]
    assert [(slot, row.tolist()) for slot, row in page_rows] == [
        (0, [[10, 11]]),
        (1, [[20, 21]]),
    ]
    assert [(slot, requests, values.reshape(-1).tolist()) for slot, requests, values in ple_rows] == [
        (0, ("req-a",), [70]),
        (1, ("req-b",), [71]),
    ]


def test_generator_virtual_decode_supports_multi_active_host_compatibility(
    monkeypatch,
) -> None:
    calls = []
    outputs = {0: object(), 1: object()}

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 2
        vocab_size = 4

        def activate_virtual_slot(self, slot, request, *, generation):
            self.active_slot = slot
            calls.append(("activate", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def commit_virtual_slot(self, slot, request, *, generation):
            calls.append(("commit", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def update_page_table(self, _state, table):
            calls.append(("page", self.active_slot, table.clone()))

        def copy_tokens(self, _state, values):
            calls.append(("token", self.active_slot, values.clone()))

        def replay_model_only_traced(self, _state, values):
            calls.append(("model-only", self.active_slot, values.clone()))
            return outputs[self.active_slot]

        def advance_positions_traced(self):
            calls.append(("advance", self.active_slot))

        def logits_to_torch(self, output):
            slot = next(slot for slot, candidate in outputs.items() if candidate is output)
            calls.append(("read-logits", slot))
            return torch.full((1, 1, self.vocab_size), float(10 + slot))

        def decode_token_out_traced(self, *_args, **_kwargs):
            raise AssertionError("host compatibility must not run the device sampler")

    position_uploads = []
    monkeypatch.setattr(generator_module.ttnn, "from_torch", lambda value, **_kwargs: value.clone())
    monkeypatch.setattr(
        generator_module.ttnn,
        "copy_host_to_device_tensor",
        lambda value, target: position_uploads.append((value.clone(), target)),
    )
    state = SimpleNamespace(
        current_pos=object(),
        page_table_host=torch.zeros((1, 2), dtype=torch.int32),
        position_host_copies=0,
        request_ids=("unset",),
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.host_sampling_compatibility_calls = 0
    generator._serving_device_feedback_current = True
    generator._serving_virtual_decode_poisoned = False
    generator._serving_virtual_slots = {
        0: generator_module._ServingVirtualSlot(
            "req-a",
            4,
            SimpleNamespace(slot_id=0, request_id="req-a", generation=11),
            True,
        ),
        1: generator_module._ServingVirtualSlot(
            "req-b",
            7,
            SimpleNamespace(slot_id=1, request_id="req-b", generation=13),
            True,
        ),
    }

    output = generator._decode_virtual_slots(
        torch.tensor([[101], [202]]),
        start_pos=torch.tensor([9, 19]),
        page_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        kv_cache=object(),
        state=state,
        read_from_device=False,
        enable_trace=True,
        sampling_params=None,
        reset_batch=False,
        request_ids=["req-a", "req-b"],
        state_slot_ids=[0, 1],
        state_slot_generations=[4, 7],
        unpadded_batch_size=2,
    )

    assert isinstance(output, _ServingDecodeOutput)
    assert output.kind == "logits"
    assert torch.equal(
        output.device,
        torch.tensor([[[10.0, 10.0, 10.0, 10.0]], [[11.0, 11.0, 11.0, 11.0]]]),
    )
    assert torch.equal(
        generator.process_decode_output_host(output, is_tokens=False),
        output.device,
    )
    assert [call[0] for call in calls] == [
        "activate",
        "page",
        "token",
        "model-only",
        "advance",
        "read-logits",
        "commit",
        "activate",
        "page",
        "token",
        "model-only",
        "advance",
        "read-logits",
        "commit",
    ]
    assert [values.reshape(-1).tolist() for values, _target in position_uploads] == [
        [9],
        [19],
    ]
    assert state.position_host_copies == 2
    assert generator.host_sampling_compatibility_calls == 1
    assert generator._serving_device_feedback_current is False
    assert all(not virtual.device_feedback_current for virtual in generator._serving_virtual_slots.values())


def test_generator_virtual_prefill_supports_multi_active_host_compatibility(
    monkeypatch,
) -> None:
    calls = []
    reusable_logits = object()

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 2
        vocab_size = 4
        _default_page_table_host = torch.zeros((1, 3), dtype=torch.int32)

        def _require_kv_cache_identity(self, cache):
            assert cache is kv_cache

        def assign_virtual_slot(self, slot, request):
            calls.append(("assign", slot, request))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=slot + 10)

        def begin_virtual_prefill(self, slot, request, *, generation):
            self.active_slot = slot
            calls.append(("begin", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def finish_virtual_prefill(self, slot, request, *, generation):
            calls.append(("finish", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def new_batch_state(self, lengths, *, request_ids, page_table):
            return SimpleNamespace(
                token_input=object(),
                current_pos=object(),
                prompt_lens=torch.as_tensor(lengths, dtype=torch.int32),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                page_table_host=page_table.clone(),
                compact_token_readbacks=0,
            )

        def reset_batch_state(self, _state):
            calls.append(("reset", self.active_slot))

        def prefill_forward(self, row, *, state, prompt_lens, kv_cache, return_all_logits):
            del state, prompt_lens, return_all_logits
            assert kv_cache is not None
            self.current_logit = int(row[0, 0])
            calls.append(("prefill", self.active_slot, self.current_logit))
            return reusable_logits

        def logits_to_torch(self, output):
            assert output is reusable_logits
            calls.append(("read-logits", self.active_slot, self.current_logit))
            return torch.full((1, 1, self.vocab_size), float(self.current_logit))

        def sample_logits(self, *_args, **_kwargs):
            raise AssertionError("host compatibility must use the shared host sampler")

    kv_cache = object()
    monkeypatch.setattr(
        generator_module.ttnn,
        "deallocate",
        lambda output: calls.append(("deallocate", output is reusable_logits)),
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator.host_sampling_compatibility_calls = 0
    generator._serving_virtual_slots = {}
    generator._serving_preempted_sampling = {}
    generator._serving_sampling_signature = None
    generator._serving_device_feedback_current = True
    generator._serving_virtual_decode_poisoned = False

    output = generator._prefill_forward_virtual(
        torch.tensor([[11, 12], [21, 22]]),
        page_table=torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.int32),
        kv_cache=kv_cache,
        prompt_lens=[2, 2],
        start_pos=[0, 0],
        intermediate_prefill_mask=[False, False],
        request_ids=["req-a", "req-b"],
        state_slot_ids=[0, 1],
        state_slot_generations=[3, 5],
        unpadded_batch_size=2,
        read_from_device=True,
        return_all_logits=False,
        on_device_sampling=False,
        sampling_params=None,
    )

    assert torch.equal(
        output,
        torch.tensor([[[11.0, 11.0, 11.0, 11.0]], [[21.0, 21.0, 21.0, 21.0]]]),
    )
    assert [call[0] for call in calls] == [
        "assign",
        "begin",
        "prefill",
        "finish",
        "read-logits",
        "deallocate",
        "assign",
        "begin",
        "reset",
        "prefill",
        "finish",
        "read-logits",
        "deallocate",
    ]
    assert generator.host_sampling_compatibility_calls == 1
    assert generator._serving_device_feedback_current is False
    assert all(not virtual.device_feedback_current for virtual in generator._serving_virtual_slots.values())


def test_generator_multi_host_prefill_preflights_later_stale_owner(expect_error) -> None:
    calls = []
    model = SimpleNamespace(
        max_batch=1,
        virtual_slot_capacity=2,
        _require_kv_cache_identity=lambda _cache: None,
        _default_page_table_host=torch.zeros((1, 2), dtype=torch.int32),
        begin_virtual_prefill=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = model
    generator.state = None
    generator.host_sampling_compatibility_calls = 0
    generator._serving_virtual_decode_poisoned = False
    generator._serving_virtual_slots = {
        0: generator_module._ServingVirtualSlot(
            "valid",
            3,
            SimpleNamespace(slot_id=0, request_id="valid", generation=10),
            False,
        ),
        1: generator_module._ServingVirtualSlot(
            "new-owner",
            9,
            SimpleNamespace(slot_id=1, request_id="new-owner", generation=11),
            False,
        ),
    }

    with expect_error(RuntimeError, "stale virtual prefill owner"):
        generator._prefill_forward_virtual(
            torch.tensor([[1, 2], [3, 4]]),
            page_table=torch.zeros((2, 2), dtype=torch.int32),
            kv_cache=object(),
            prompt_lens=[2, 2],
            start_pos=[1, 1],
            intermediate_prefill_mask=[False, False],
            request_ids=["valid", "new-owner"],
            state_slot_ids=[0, 1],
            state_slot_generations=[3, 8],
            unpadded_batch_size=2,
            read_from_device=True,
            return_all_logits=False,
            on_device_sampling=False,
            sampling_params=None,
        )

    assert calls == []
    assert generator.host_sampling_compatibility_calls == 0
    assert generator._serving_virtual_decode_poisoned is False


def test_generator_multi_host_prefill_preflights_page_width(expect_error) -> None:
    calls = []
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(
        max_batch=1,
        virtual_slot_capacity=2,
        _require_kv_cache_identity=lambda _cache: None,
        _default_page_table_host=torch.zeros((1, 2), dtype=torch.int32),
        assign_virtual_slot=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    generator.state = None
    generator.host_sampling_compatibility_calls = 0
    generator._serving_virtual_decode_poisoned = False
    generator._serving_virtual_slots = {}

    with expect_error(ValueError, "page-table width"):
        generator._prefill_forward_virtual(
            torch.tensor([[1, 2], [3, 4]]),
            page_table=torch.zeros((2, 3), dtype=torch.int32),
            kv_cache=object(),
            prompt_lens=[2, 2],
            start_pos=[0, 0],
            intermediate_prefill_mask=[False, False],
            request_ids=["a", "b"],
            state_slot_ids=[0, 1],
            state_slot_generations=[1, 1],
            unpadded_batch_size=2,
            read_from_device=True,
            return_all_logits=False,
            on_device_sampling=False,
            sampling_params=None,
        )

    assert calls == []
    assert generator.host_sampling_compatibility_calls == 0
    assert generator._serving_virtual_decode_poisoned is False


def test_generator_multi_host_prefill_fail_stops_after_partial_execution(
    monkeypatch,
    expect_error,
) -> None:
    calls = []

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 2
        vocab_size = 4
        _default_page_table_host = torch.zeros((1, 2), dtype=torch.int32)

        def _require_kv_cache_identity(self, _cache):
            return None

        def assign_virtual_slot(self, slot, request):
            calls.append(("assign", slot))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=slot + 10)

        def begin_virtual_prefill(self, slot, request, *, generation):
            self.active_slot = slot
            calls.append(("begin", slot))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def finish_virtual_prefill(self, slot, request, *, generation):
            calls.append(("finish", slot))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def new_batch_state(self, lengths, *, request_ids, page_table):
            return SimpleNamespace(
                prompt_lens=torch.as_tensor(lengths),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                page_table_host=page_table.clone(),
            )

        def reset_batch_state(self, _state):
            return None

        def prefill_forward(self, *_args, **_kwargs):
            calls.append(("prefill", self.active_slot))
            if self.active_slot == 1:
                raise RuntimeError("injected row-1 failure")
            return object()

        def logits_to_torch(self, _output):
            return torch.zeros((1, 1, self.vocab_size))

    monkeypatch.setattr(generator_module.ttnn, "deallocate", lambda _value: None)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator.host_sampling_compatibility_calls = 0
    generator._serving_virtual_slots = {}
    generator._serving_preempted_sampling = {}
    generator._serving_virtual_decode_poisoned = False
    kwargs = dict(
        page_table=torch.zeros((2, 2), dtype=torch.int32),
        kv_cache=object(),
        prompt_lens=[2, 2],
        start_pos=[0, 0],
        intermediate_prefill_mask=[False, False],
        request_ids=["a", "b"],
        state_slot_ids=[0, 1],
        state_slot_generations=[1, 1],
        unpadded_batch_size=2,
        read_from_device=True,
        return_all_logits=False,
        on_device_sampling=False,
        sampling_params=None,
    )

    with expect_error(RuntimeError, "injected row-1 failure"):
        generator._prefill_forward_virtual(torch.tensor([[1, 2], [3, 4]]), **kwargs)
    assert generator._serving_virtual_decode_poisoned is True
    calls_after_failure = list(calls)

    with expect_error(RuntimeError, "fail-stopped"):
        generator._prefill_forward_virtual(torch.tensor([[1, 2], [3, 4]]), **kwargs)
    assert calls == calls_after_failure


def test_generator_virtual_prefill_reuses_one_physical_state_for_two_slots(
    monkeypatch,
) -> None:
    calls = []
    slot_tokens = {0: object(), 1: object()}

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 2
        vocab_size = 32
        _default_page_table_host = torch.zeros((1, 4), dtype=torch.int32)

        def _require_kv_cache_identity(self, cache):
            assert cache is kv_cache

        def assign_virtual_slot(self, slot, request):
            calls.append(("assign", slot, request))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=slot + 10)

        def begin_virtual_prefill(self, slot, request, *, generation):
            calls.append(("begin", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def finish_virtual_prefill(self, slot, request, *, generation):
            calls.append(("finish", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def virtual_slot_token(self, slot, request, *, generation):
            return slot_tokens[slot]

        def new_batch_state(self, lengths, *, request_ids, page_table):
            state = SimpleNamespace(
                token_input=object(),
                current_pos=object(),
                page_table=object(),
                prompt_lens=torch.as_tensor(lengths, dtype=torch.int32),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                page_table_host=page_table.clone(),
                compact_token_readbacks=0,
            )
            calls.append(("new-state", id(state), tuple(request_ids)))
            return state

        def reset_batch_state(self, state):
            calls.append(
                (
                    "reset-state",
                    id(state),
                    state.request_ids,
                    state.page_table_host.clone(),
                )
            )

        def set_sampling_params(self, **kwargs):
            calls.append(("sampling", int(torch.as_tensor(kwargs["top_k"]).reshape(-1)[0])))

        def prefill_forward(self, row, *, state, prompt_lens, kv_cache, return_all_logits):
            calls.append(
                (
                    "prefill",
                    id(state),
                    state.request_ids,
                    int(prompt_lens[0]),
                    state.page_table_host.clone(),
                    row.clone(),
                )
            )
            return object()

        def sample_logits(self, logits, state):
            return state.token_input

    kv_cache = object()
    monkeypatch.setattr(generator_module.ttnn, "deallocate", lambda _value: None)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator._serving_virtual_slots = {}
    generator._serving_sampling_signature = None
    generator._serving_device_feedback_current = False

    output = generator._prefill_forward_virtual(
        torch.tensor([[1, 2, 0], [3, 4, 5]]),
        page_table=torch.tensor([[1, 2, 0, 0], [7, 8, 9, 0]], dtype=torch.int32),
        kv_cache=kv_cache,
        prompt_lens=[2, 3],
        request_ids=["req-a", "req-b"],
        state_slot_ids=[0, 1],
        state_slot_generations=[3, 5],
        unpadded_batch_size=2,
        read_from_device=False,
        return_all_logits=False,
        on_device_sampling=True,
        sampling_params=SimpleNamespace(
            temperature=[1.0, 1.0],
            top_p=[0.0, 0.0],
            top_k=[1, 1],
            presence_penalty=[0.0, 0.0],
            frequency_penalty=[0.0, 0.0],
            repetition_penalty=[1.0, 1.0],
            seed=[None, None],
            enable_log_probs=[False, False],
            num_logprobs=[-2, -2],
        ),
    )

    assert isinstance(output, _ServingDecodeOutput)
    assert output.device == (slot_tokens[0], slot_tokens[1])
    new_state = next(call for call in calls if call[0] == "new-state")
    reset_state = next(call for call in calls if call[0] == "reset-state")
    assert reset_state[1] == new_state[1]
    prefills = [call for call in calls if call[0] == "prefill"]
    assert [call[1] for call in prefills] == [new_state[1], new_state[1]]
    assert [(call[2], call[3]) for call in prefills] == [
        (("req-a",), 2),
        (("req-b",), 3),
    ]
    assert [call[0] for call in calls] == [
        "assign",
        "begin",
        "new-state",
        "prefill",
        "sampling",
        "finish",
        "assign",
        "begin",
        "reset-state",
        "prefill",
        "sampling",
        "finish",
    ]


def test_generator_virtual_decode_rejects_stale_generation_before_device_execution(expect_error) -> None:
    model = SimpleNamespace(
        max_batch=1,
        virtual_slot_capacity=2,
        _require_kv_cache_identity=lambda _cache: None,
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = model
    generator._serving_virtual_slots = {
        0: generator_module._ServingVirtualSlot(
            "new-owner",
            9,
            SimpleNamespace(slot_id=0, request_id="new-owner", generation=3),
            True,
        )
    }

    with expect_error(RuntimeError, "stale virtual state owner"):
        generator._decode_virtual_slots(
            torch.tensor([[1]]),
            start_pos=torch.tensor([1]),
            page_table=None,
            kv_cache=object(),
            state=SimpleNamespace(),
            read_from_device=False,
            enable_trace=True,
            sampling_params=_sampling_params(),
            reset_batch=False,
            request_ids=["new-owner"],
            state_slot_ids=[0],
            state_slot_generations=[8],
            unpadded_batch_size=1,
        )


def test_generator_multi_host_decode_preflights_later_stale_row(expect_error) -> None:
    calls = []
    model = SimpleNamespace(
        max_batch=1,
        virtual_slot_capacity=2,
        activate_virtual_slot=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = model
    generator.host_sampling_compatibility_calls = 0
    generator._serving_virtual_decode_poisoned = False
    generator._serving_virtual_slots = {
        0: generator_module._ServingVirtualSlot(
            "valid",
            3,
            SimpleNamespace(slot_id=0, request_id="valid", generation=10),
            True,
        ),
        1: generator_module._ServingVirtualSlot(
            "new-owner",
            9,
            SimpleNamespace(slot_id=1, request_id="new-owner", generation=11),
            True,
        ),
    }

    with expect_error(RuntimeError, "stale virtual state owner"):
        generator._decode_virtual_slots(
            torch.tensor([[1], [2]]),
            start_pos=torch.tensor([10, 20]),
            page_table=None,
            kv_cache=object(),
            state=SimpleNamespace(),
            read_from_device=False,
            enable_trace=True,
            sampling_params=None,
            reset_batch=False,
            request_ids=["valid", "new-owner"],
            state_slot_ids=[0, 1],
            state_slot_generations=[3, 8],
            unpadded_batch_size=2,
        )

    assert calls == []
    assert generator.host_sampling_compatibility_calls == 0
    assert generator._serving_virtual_decode_poisoned is False


def test_generator_virtual_release_is_generation_guarded_and_idempotent() -> None:
    released = []
    model = SimpleNamespace(
        release_virtual_slot=lambda slot, request, *, generation: released.append((slot, request, generation))
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = model
    lease = SimpleNamespace(slot_id=1, request_id="survivor", generation=12)
    generator._serving_virtual_slots = {1: generator_module._ServingVirtualSlot("survivor", 5, lease, True)}

    # Late completion/release from the prior lifetime cannot touch the owner.
    generator.release_virtual_slots([("old-owner", 1, 4, "finished")])
    assert released == []
    assert 1 in generator._serving_virtual_slots

    generator.release_virtual_slots([("survivor", 1, 5, "finished"), ("survivor", 1, 5, "finished")])
    assert released == [(1, "survivor", 12)]
    assert generator._serving_virtual_slots == {}


def test_vllm_greedy_encoding_delegates_to_canonical_argmax() -> None:
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(set_sampling_params=Mock())
    generator._serving_sampling_signature = None

    generator._apply_serving_sampling_params(
        SimpleNamespace(
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            enable_log_probs=False,
            top_k=[0],
            top_p=[1.0],
            temperature=[0.0],
            seed=[None],
        )
    )

    kwargs = generator.model.set_sampling_params.call_args.kwargs
    assert torch.equal(kwargs["top_k"], torch.tensor([1], dtype=torch.int32))
    assert torch.equal(kwargs["top_p"], torch.tensor([0.0]))
    assert torch.equal(kwargs["temperature"], torch.tensor([1.0]))


def test_canonical_sampler_converts_user_temperature_to_kernel_multiplier(
    monkeypatch,
) -> None:
    writes = []
    monkeypatch.setattr(
        model_module,
        "_copy_host_to_device",
        lambda value, target, **kwargs: writes.append((value.clone(), target, kwargs)),
    )
    model = SimpleNamespace(
        max_batch=1,
        _sampling_force_argmax=True,
        _trace_ready=False,
        _sampling_seed_rngs=None,
        sampling_k=object(),
        sampling_p=object(),
        sampling_temp=object(),
    )

    Qwen38FullModel.set_sampling_params(
        model,
        top_k=10,
        top_p=1.0,
        temperature=2.0,
        seeds=[123],
    )

    assert torch.equal(writes[-1][0], torch.tensor([0.5]))
    assert model._sampling_force_argmax is False


def test_canonical_sampler_routes_top1_at_any_temperature_to_argmax(
    monkeypatch,
) -> None:
    writes = []
    monkeypatch.setattr(
        model_module,
        "_copy_host_to_device",
        lambda value, target, **kwargs: writes.append((value.clone(), target, kwargs)),
    )
    model = SimpleNamespace(
        max_batch=1,
        _sampling_force_argmax=False,
        _trace_ready=False,
        _sampling_seed_rngs=None,
        sampling_k=object(),
        sampling_p=object(),
        sampling_temp=object(),
    )

    Qwen38FullModel.set_sampling_params(
        model,
        top_k=1,
        top_p=0.95,
        temperature=2.0,
        seeds=[123],
    )

    assert model._sampling_force_argmax is True
    assert model._sampling_seed_rngs is None
    assert torch.equal(writes[-1][0], torch.tensor([0.5]))


def test_canonical_sampler_synthesizes_seeds_for_unseeded_stochastic_requests(monkeypatch) -> None:
    """DEVSTACK-294: stochastic sampling must never run on the static default seed buffer."""

    monkeypatch.setattr(model_module, "_copy_host_to_device", lambda value, target, **kwargs: None)
    model = SimpleNamespace(
        max_batch=1,
        _sampling_force_argmax=True,
        _trace_ready=False,
        _sampling_seed_rngs=None,
        sampling_k=object(),
        sampling_p=object(),
        sampling_temp=object(),
    )
    Qwen38FullModel.set_sampling_params(model, top_k=20, top_p=0.95, temperature=1.0)
    assert model._sampling_force_argmax is False
    assert model._sampling_seed_rngs is not None and len(model._sampling_seed_rngs) == 1


def test_canonical_sampler_refuses_to_advance_without_seed_streams() -> None:
    model = SimpleNamespace(_sampling_force_argmax=False, _sampling_seed_rngs=None)
    try:
        Qwen38FullModel._advance_sampling_seeds(model)
    except RuntimeError as error:
        assert "seed streams" in str(error)
    else:
        raise AssertionError("stochastic sampling advanced without per-row seed streams")
    greedy = SimpleNamespace(_sampling_force_argmax=True, _sampling_seed_rngs=None)
    Qwen38FullModel._advance_sampling_seeds(greedy)


def test_compile_hooks_route_sampling_params_through_serving_bridge() -> None:
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(
        max_batch=1,
        set_sampling_params=Mock(),
        _require_kv_cache_identity=lambda kv_cache: None,
        _default_page_table_host=torch.zeros(1, 1, dtype=torch.int32),
    )
    generator._serving_sampling_signature = None
    generator._apply_serving_sampling_params = Mock()
    params = SimpleNamespace(top_k=[20], top_p=[0.95], temperature=[1.0], seed=[None])
    generator.compile_decode(tokens=[1], start_pos=[0], sampling_params=params)
    generator._apply_serving_sampling_params.assert_called_once_with(params, reset_seed=True)
    generator.model.set_sampling_params.assert_not_called()


def test_vllm_unseeded_device_sampling_refreshes_once_per_request(monkeypatch) -> None:
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(max_batch=1, set_sampling_params=Mock())
    generator._serving_sampling_signature = None
    random_values = iter((10, 20))
    monkeypatch.setattr(generator_module.secrets, "randbelow", lambda _limit: next(random_values))
    params = SimpleNamespace(
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        enable_log_probs=False,
        top_k=[20],
        top_p=[0.95],
        temperature=[1.0],
        seed=[None],
    )

    generator._apply_serving_sampling_params(params, reset_seed=True)
    generator._apply_serving_sampling_params(params)
    generator._apply_serving_sampling_params(params, reset_seed=True)

    calls = generator.model.set_sampling_params.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["seeds"] == [11]
    assert calls[1].kwargs["seeds"] == [21]


def test_virtual_chunked_prefill_replays_to_chunk_end_without_logits_readback(
    monkeypatch,
) -> None:
    calls = []

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 3
        vocab_size = 16
        _default_page_table_host = torch.zeros((1, 4), dtype=torch.int32)

        def _require_kv_cache_identity(self, _cache):
            return None

        def assign_virtual_slot(self, slot, request):
            calls.append(("assign", slot, request))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=5)

        def begin_virtual_prefill(self, slot, request, *, generation):
            calls.append(("begin", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def finish_virtual_prefill(self, slot, request, *, generation):
            calls.append(("finish", slot, request, generation))
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def new_batch_state(self, lengths, *, request_ids, page_table):
            return SimpleNamespace(
                token_input=object(),
                prompt_lens=torch.as_tensor(lengths, dtype=torch.int32),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                page_table_host=page_table.clone(),
            )

        def reset_batch_state(self, _state):
            calls.append(("reset",))

        def prefill_forward(self, row, **_kwargs):
            calls.append(("prefill-width", row.shape[1]))
            return object()

        def set_sampling_params(self, **kwargs):
            calls.append(("sampling", kwargs["seeds"]))

        def sample_logits(self, _logits, state):
            calls.append(("sample",))
            return state.token_input

    monkeypatch.setattr(
        generator_module.ttnn,
        "deallocate",
        lambda _value: calls.append(("deallocate",)),
    )
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator._serving_virtual_slots = {}
    generator._serving_preempted_sampling = {}
    generator._serving_sampling_signature = None
    generator._serving_device_feedback_current = False
    params = SimpleNamespace(
        temperature=[1.0],
        top_p=[0.9],
        top_k=[8],
        seed=[123],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        enable_log_probs=[False],
        num_logprobs=[-2],
    )

    first = generator._prefill_forward_virtual(
        torch.tensor([[1, 2, 0, 0]]),
        page_table=None,
        kv_cache=object(),
        prompt_lens=[2],
        start_pos=[0],
        intermediate_prefill_mask=[True],
        request_ids=["chunked"],
        state_slot_ids=[2],
        state_slot_generations=[7],
        unpadded_batch_size=1,
        read_from_device=True,
        return_all_logits=False,
        on_device_sampling=True,
        sampling_params=params,
    )
    assert first.tolist() == [0]
    assert ("sample",) not in calls
    assert not any(call[0] == "sampling" for call in calls)

    generator._prefill_forward_virtual(
        torch.tensor([[1, 2, 3, 4]]),
        page_table=None,
        kv_cache=object(),
        prompt_lens=[4],
        start_pos=[2],
        intermediate_prefill_mask=[False],
        request_ids=["chunked"],
        state_slot_ids=[2],
        state_slot_generations=[7],
        unpadded_batch_size=1,
        read_from_device=False,
        return_all_logits=False,
        on_device_sampling=True,
        sampling_params=params,
    )
    assert [call for call in calls if call[0] == "prefill-width"] == [
        ("prefill-width", 2),
        ("prefill-width", 4),
    ]
    assert [call for call in calls if call[0] == "sample"] == [("sample",)]
    assert not hasattr(generator.model, "logits_to_torch")


def test_stale_prefill_generation_cannot_evict_newer_owner(expect_error) -> None:
    released = []
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(release_virtual_slot=lambda *args, **kwargs: released.append((args, kwargs)))
    generator._serving_virtual_slots = {
        1: generator_module._ServingVirtualSlot(
            "new", 12, SimpleNamespace(slot_id=1, request_id="new", generation=4), True
        )
    }

    with expect_error(RuntimeError, "stale virtual prefill owner"):
        generator._preflight_virtual_prefill_claims(("old",), (1,), (11,))
    with expect_error(RuntimeError, "stale virtual prefill owner"):
        generator._claim_virtual_slot(1, "old", 11)
    assert released == []
    assert generator._serving_virtual_slots[1].request_id == "new"


@pytest.mark.parametrize("seed", [123, None])
def test_preemption_preserves_seeded_and_unseeded_device_rng(monkeypatch, seed) -> None:
    calls = []
    rng_marker = ("opaque-random-state", seed)

    class FakeModel:
        max_batch = 1
        virtual_slot_capacity = 3
        vocab_size = 16
        _default_page_table_host = torch.zeros((1, 2), dtype=torch.int32)

        def _require_kv_cache_identity(self, _cache):
            return None

        def export_virtual_slot_sampling_rng_state(self, *args, **kwargs):
            calls.append(("export", args, kwargs))
            return rng_marker

        def release_virtual_slot(self, *args, **kwargs):
            calls.append(("release", args, kwargs))

        def assign_virtual_slot(self, slot, request):
            return SimpleNamespace(slot_id=slot, request_id=request, generation=9)

        def begin_virtual_prefill(self, slot, request, *, generation):
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def finish_virtual_prefill(self, slot, request, *, generation):
            return SimpleNamespace(slot_id=slot, request_id=request, generation=generation)

        def new_batch_state(self, lengths, *, request_ids, page_table):
            return SimpleNamespace(
                token_input=object(),
                prompt_lens=torch.as_tensor(lengths),
                active_mask=torch.ones(1, dtype=torch.bool),
                request_ids=tuple(request_ids),
                page_table_host=page_table.clone(),
            )

        def prefill_forward(self, *_args, **_kwargs):
            return object()

        def set_sampling_params(self, **kwargs):
            calls.append(("set", kwargs["seeds"]))

        def restore_virtual_slot_sampling_rng_state(self, *args, **kwargs):
            calls.append(("restore", args[2]))

        def sample_logits(self, _logits, state):
            calls.append(("sample",))
            return state.token_input

    monkeypatch.setattr(generator_module.ttnn, "deallocate", lambda _value: None)
    monkeypatch.setattr(generator_module.secrets, "randbelow", lambda _limit: 456)
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator.state = None
    generator._serving_sampling_signature = None
    generator._serving_device_feedback_current = False
    generator._serving_preempted_sampling = {}
    old = generator_module._ServingVirtualSlot(
        "request",
        4,
        SimpleNamespace(slot_id=2, request_id="request", generation=3),
        True,
        sampling_signature=("old",),
    )
    generator._serving_virtual_slots = {2: old}
    generator.release_virtual_slots([("request", 2, 4, "preempted")])

    params = SimpleNamespace(
        temperature=[1.0],
        top_p=[0.9],
        top_k=[8],
        seed=[seed],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        enable_log_probs=[False],
        num_logprobs=[-2],
    )
    generator._prefill_forward_virtual(
        torch.tensor([[1, 2]]),
        page_table=None,
        kv_cache=object(),
        prompt_lens=[2],
        start_pos=[0],
        intermediate_prefill_mask=[False],
        request_ids=["request"],
        state_slot_ids=[2],
        state_slot_generations=[5],
        unpadded_batch_size=1,
        read_from_device=False,
        return_all_logits=False,
        on_device_sampling=True,
        sampling_params=params,
    )

    assert [call[0] for call in calls].index("restore") < [call[0] for call in calls].index("sample")
    assert ("restore", rng_marker) in calls
    set_call = next(call for call in calls if call[0] == "set")
    assert set_call[1] == ([123] if seed == 123 else [457])


def test_later_row_sampling_error_is_atomic_before_any_decode_execution(expect_error) -> None:
    calls = []
    generator = object.__new__(Qwen38Generator)
    generator.model = SimpleNamespace(max_batch=1, virtual_slot_capacity=3)
    generator._serving_virtual_decode_poisoned = False
    generator._serving_virtual_slots = {
        slot: generator_module._ServingVirtualSlot(
            request,
            generation,
            SimpleNamespace(slot_id=slot, request_id=request, generation=20 + slot),
            True,
        )
        for slot, request, generation in ((0, "a", 3), (1, "b", 4), (2, "c", 5))
    }
    generator.model.activate_virtual_slot = lambda *args, **kwargs: calls.append(("activate", args, kwargs))
    state = SimpleNamespace(page_table_host=torch.zeros((1, 2), dtype=torch.int32))
    params = SimpleNamespace(
        temperature=[1.0, 1.0, 1.0],
        top_p=[0.9, 0.9, 0.9],
        top_k=[8, 99, 8],
        seed=[1, 2, 3],
        presence_penalty=[0.0] * 3,
        frequency_penalty=[0.0] * 3,
        repetition_penalty=[1.0] * 3,
        enable_log_probs=[False] * 3,
        num_logprobs=[-2] * 3,
    )

    with expect_error(ValueError, "top_k"):
        generator._decode_virtual_slots(
            torch.tensor([[1], [2], [3]]),
            start_pos=torch.tensor([9, 10, 11]),
            page_table=torch.zeros((3, 2), dtype=torch.int32),
            kv_cache=object(),
            state=state,
            read_from_device=False,
            enable_trace=True,
            sampling_params=params,
            reset_batch=False,
            request_ids=["a", "b", "c"],
            state_slot_ids=[0, 1, 2],
            state_slot_generations=[3, 4, 5],
            unpadded_batch_size=3,
        )
    assert calls == []
    assert generator._serving_virtual_decode_poisoned is False


def test_adapter_has_no_sampling_or_full_logits_policy() -> None:
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            Qwen4ExpForConditionalGeneration.prefill_forward,
            Qwen4ExpForConditionalGeneration.decode_forward,
            Qwen4ExpForConditionalGeneration.read_decode_output,
            Qwen4ExpForConditionalGeneration.process_decode_output_host,
        )
    )
    for forbidden in (
        "torch.argmax",
        "ttnn.argmax",
        "torch.topk",
        "ttnn.topk",
        "logits_to_torch",
    ):
        assert forbidden not in source
