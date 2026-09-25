# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only contract tests for the Qwen3.6 vLLM translation adapter.

These tests deliberately construct the adapter with ``object.__new__`` and a
spy generator.  Importing this module must not open a TT device, load weights,
or allocate a cache.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import torch

from models.autoports.qwen_qwen3_6_27b.tt import generator_vllm as adapter_module
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import (
    Qwen3_5ForConditionalGeneration,
)


class _GeneratorSpy:
    def __init__(self) -> None:
        self.prefill_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.decode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.sampling_state_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.read_calls: list[tuple[Any, bool]] = []
        self.process_calls: list[tuple[Any, bool]] = []
        self.prefill_result = object()
        self.decode_result = object()
        self.read_result = object()
        self.process_result = torch.tensor([7, 11], dtype=torch.long)

    def prefill_forward(self, *args: Any, **kwargs: Any) -> Any:
        self.prefill_calls.append((args, kwargs))
        return self.prefill_result

    def decode_forward(self, *args: Any, **kwargs: Any) -> Any:
        self.decode_calls.append((args, kwargs))
        return self.decode_result

    def apply_decode_sampling_state(self, *args: Any, **kwargs: Any) -> None:
        self.sampling_state_calls.append((args, kwargs))

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        self.read_calls.append((tt_out, async_read))
        return self.read_result

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> Any:
        self.process_calls.append((tt_out, is_tokens))
        return self.process_result


def _adapter() -> tuple[Qwen3_5ForConditionalGeneration, _GeneratorSpy]:
    adapter = object.__new__(Qwen3_5ForConditionalGeneration)
    generator = _GeneratorSpy()
    adapter.generator = generator
    adapter._vllm_kv_cache = object()
    adapter._expanded_kv_cache = object()
    adapter._state = object()
    adapter._decode_started = False
    return adapter, generator


def _sampling_params() -> SimpleNamespace:
    return SimpleNamespace(
        temperature=[1.0, 1.0],
        top_p=[0.0, 0.0],
        top_k=[1, 1],
        presence_penalty=[0.0, 0.0],
        frequency_penalty=[0.0, 0.0],
        repetition_penalty=[1.0, 1.0],
        seed=[None, None],
        enable_log_probs=[False, False],
        num_logprobs=[-1, -1],
    )


def test_static_capabilities_and_protocol_surface() -> None:
    cls = Qwen3_5ForConditionalGeneration
    assert isinstance(cls, type)
    assert cls.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    initialize = inspect.signature(cls.initialize_vllm_model)
    for name in ("hf_config", "mesh_device", "max_batch_size", "max_seq_len", "tt_data_parallel"):
        assert name in initialize.parameters
    decode = inspect.signature(cls.decode_forward)
    for name in (
        "tokens",
        "page_table",
        "kv_cache",
        "start_pos",
        "enable_trace",
        "read_from_device",
        "sampling_params",
        "reset_batch",
        "slot_remap",
    ):
        assert name in decode.parameters
    assert callable(cls.allocate_kv_cache)
    assert callable(cls.read_decode_output)
    assert callable(cls.process_decode_output_host)
    assert cls.get_max_tokens_all_users(max_model_len=1, max_num_seqs=32) == 262_144


def test_initialize_loads_the_selected_precision_policy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            self.model = SimpleNamespace(
                precision_policy=SimpleNamespace(
                    path=adapter_module.DEFAULT_PRECISION_CONFIG.resolve()
                )
            )

    monkeypatch.setattr(adapter_module, "Generator", FakeGenerator)
    mesh = SimpleNamespace(get_num_devices=lambda: 4)
    result = Qwen3_5ForConditionalGeneration.initialize_vllm_model(
        hf_config=SimpleNamespace(_name_or_path=""),
        mesh_device=mesh,
        max_batch_size=32,
        max_seq_len=262_144,
    )

    assert result.max_batch_size == 32
    assert result.max_seq_len == 262_144
    assert captured["mesh_device"] is mesh
    assert captured["max_seq_len"] == 262_144
    assert captured["precision_config_path"] == adapter_module.DEFAULT_PRECISION_CONFIG


def test_allocate_cache_binds_exact_external_attention_cache() -> None:
    adapter, generator = _adapter()
    adapter.max_seq_len = 262_144
    allocated: list[tuple[int, int, Any]] = []

    class FakeLayer:
        def __init__(self, layer_idx: int) -> None:
            self.layer_idx = layer_idx

        def allocate_paged_kv_cache(self, *, num_blocks: int, dtype: Any) -> Any:
            pair = (f"k{self.layer_idx}", f"v{self.layer_idx}")
            allocated.append((self.layer_idx, num_blocks, dtype))
            return pair

    state = object()
    state_calls: list[dict[str, Any]] = []

    def allocate_state(**kwargs: Any) -> Any:
        state_calls.append(kwargs)
        return state

    generator.model = SimpleNamespace(
        precision_policy=SimpleNamespace(kv_cache={"dtype": "bfp8"}),
        num_layers=64,
        layers=[FakeLayer(index) for index in range(64)],
        allocate_state=allocate_state,
    )

    compact = adapter.allocate_kv_cache(
        (97, 1, 64, 256), dtype=torch.bfloat16, num_layers=16
    )

    assert [layer for layer, _, _ in allocated] == list(range(3, 64, 4))
    assert all(blocks == 97 for _, blocks, _ in allocated)
    assert all(dtype is adapter_module.ttnn.bfloat8_b for _, _, dtype in allocated)
    assert compact is adapter._vllm_kv_cache
    assert len(compact) == 16
    assert adapter._expanded_kv_cache[3] is compact[0]
    assert adapter._expanded_kv_cache[63] is compact[-1]
    assert all(
        adapter._expanded_kv_cache[index] is None
        for index in range(64)
        if index not in range(3, 64, 4)
    )
    assert len(state_calls) == 1
    assert state_calls[0]["external_kv_cache"] is adapter._expanded_kv_cache
    assert state_calls[0]["num_blocks"] == 97
    assert state_calls[0]["linear_state_capacity"] == 32
    assert adapter._state is state
    assert generator._state is state


def test_adapter_has_no_sampling_or_full_logits_readback_path() -> None:
    """The shim may translate state, but it must not implement model policy."""

    source = "\n".join(
        inspect.getsource(method)
        for method in (
            Qwen3_5ForConditionalGeneration.prefill_forward,
            Qwen3_5ForConditionalGeneration.decode_forward,
            Qwen3_5ForConditionalGeneration.read_decode_output,
            Qwen3_5ForConditionalGeneration.process_decode_output_host,
        )
    )
    for forbidden in (
        "torch.argmax",
        "ttnn.argmax",
        "_logits_to_host",
        "_sample_host_logits",
        "torch.topk",
        "ttnn.topk",
    ):
        assert forbidden not in source


def test_prefill_delegates_non_aligned_logical_lengths_and_external_cache() -> None:
    adapter, generator = _adapter()
    tokens = torch.arange(2 * 96, dtype=torch.long).reshape(2, 96)
    page_table = torch.tensor([[7, 9], [12, 15]], dtype=torch.int32)
    kv_cache = adapter._vllm_kv_cache
    sampling_params = _sampling_params()

    result = adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=False,
        prompt_lens=[65, 67],
        start_pos=torch.zeros(2, dtype=torch.int32),
        sampling_params=sampling_params,
        empty_slots=[3, 8],
    )

    assert result is generator.prefill_result
    assert len(generator.prefill_calls) == 1
    args, kwargs = generator.prefill_calls[0]
    delegated_tokens = args[0] if args else kwargs["tokens"]
    assert delegated_tokens is tokens
    assert kwargs["page_table"] is page_table
    assert kwargs["kv_cache"] is adapter._expanded_kv_cache
    assert kwargs["state"] is adapter._state
    assert kwargs["state_slots"] == [3, 8]
    assert list(kwargs["prompt_lens"]) == [65, 67]
    # Padding/chunk/page alignment is generator-owned; the adapter preserves
    # both the caller's storage and the logical lengths.
    assert delegated_tokens.shape == (2, 96)
    assert kwargs["page_table_changed"] is True


def test_text_only_mrope_protocol_returns_neutral_deltas() -> None:
    adapter, generator = _adapter()
    tokens = torch.ones((2, 65), dtype=torch.long)

    result, rope_deltas = adapter.prefill_forward(
        tokens=tokens,
        page_table=torch.zeros((2, 2), dtype=torch.int32),
        kv_cache=adapter._vllm_kv_cache,
        prompt_lens=[65, 65],
        sampling_params=_sampling_params(),
        empty_slots=[0, 1],
        pixel_values=[None, None],
        image_grid_thw=[None, None],
    )

    assert result is generator.prefill_result
    assert torch.equal(rope_deltas, torch.zeros(2, dtype=torch.int64))


def test_async_steady_decode_ignores_stale_host_inputs_and_delegates_split() -> None:
    adapter, generator = _adapter()
    adapter._decode_started = True
    stale_tokens = torch.tensor([[101], [202]], dtype=torch.long)
    stale_positions = torch.tensor([31, 47], dtype=torch.int32)
    page_table = torch.tensor([[0, 3], [8, 11]], dtype=torch.int32)
    kv_cache = adapter._vllm_kv_cache
    sampling_params = _sampling_params()

    result = adapter.decode_forward(
        tokens=stale_tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=stale_positions,
        enable_trace=True,
        read_from_device=False,
        sampling_params=sampling_params,
        prompt_tokens=torch.tensor([[1, 2], [3, 4]]),
        output_tokens=torch.tensor([[5], [6]]),
        reset_batch=False,
        slot_remap=torch.arange(32, dtype=torch.int32),
    )

    assert result is generator.decode_result
    assert len(generator.decode_calls) == 1
    args, kwargs = generator.decode_calls[0]
    delegated_tokens = args[0] if args else kwargs["tokens"]
    delegated_positions = args[1] if len(args) > 1 else kwargs["start_pos"]
    assert delegated_tokens is stale_tokens
    assert delegated_positions is stale_positions
    assert kwargs["page_table"] is page_table
    assert kwargs["kv_cache"] is adapter._expanded_kv_cache
    assert kwargs["state"] is adapter._state
    assert kwargs["enable_trace"] is True
    assert kwargs["read_from_device"] is False
    # The host tensors are deliberately stale.  The adapter enters the
    # canonical serving mode without a reset; the generator therefore retains
    # its authoritative device token/current-position chain.
    assert kwargs["serving_mode"] is True
    assert kwargs["reset_batch"] is False
    assert kwargs["page_table_changed"] is False
    assert adapter._decode_started is True

    read = adapter.read_decode_output(result, async_read=True)
    assert read is generator.read_result
    assert generator.read_calls == [(result, True)]
    processed = adapter.process_decode_output_host(read, is_tokens=True)
    assert processed is generator.process_result
    assert generator.process_calls == [(read, True)]


def test_reset_and_slot_remap_force_one_authoritative_refresh() -> None:
    adapter, generator = _adapter()
    adapter._decode_started = True
    remap = torch.tensor([2, 0, 1] + list(range(3, 32)), dtype=torch.int32)
    params = _sampling_params()

    adapter.decode_forward(
        tokens=torch.tensor([[10], [20], [30]]),
        page_table=torch.tensor([[0], [1], [2]], dtype=torch.int32),
        kv_cache=adapter._vllm_kv_cache,
        start_pos=torch.tensor([65, 81, 97], dtype=torch.int32),
        enable_trace=True,
        read_from_device=False,
        sampling_params=params,
        prompt_tokens=torch.tensor([[1], [2], [3]]),
        output_tokens=torch.tensor([[4], [5], [6]]),
        reset_batch=True,
        slot_remap=remap,
    )

    assert len(generator.decode_calls) == 1
    _, decode_kwargs = generator.decode_calls[0]
    assert decode_kwargs["reset_batch"] is True
    assert decode_kwargs["page_table_changed"] is True
    assert torch.equal(decode_kwargs["slot_remap"], remap)
    # Sampling/slot state and token selection both remain in the canonical
    # generator; the adapter submits exactly one reset/remap decode call.
    assert generator.sampling_state_calls == []


def test_decode_refreshes_page_table_only_when_a_new_page_starts() -> None:
    adapter, generator = _adapter()
    adapter._decode_started = True
    common = {
        "tokens": torch.tensor([[17]]),
        "page_table": torch.tensor([[0, 1, 2]], dtype=torch.int32),
        "kv_cache": adapter._vllm_kv_cache,
        "enable_trace": True,
        "read_from_device": False,
        "sampling_params": _sampling_params(),
        "reset_batch": False,
    }

    adapter.decode_forward(start_pos=torch.tensor([127], dtype=torch.int32), **common)
    adapter.decode_forward(start_pos=torch.tensor([128], dtype=torch.int32), **common)

    assert generator.decode_calls[0][1]["page_table_changed"] is False
    assert generator.decode_calls[1][1]["page_table_changed"] is True


def test_optional_host_sampling_is_explicit_and_not_the_async_device_path() -> None:
    adapter, generator = _adapter()
    adapter._decode_started = True

    adapter.decode_forward(
        tokens=torch.tensor([[17]]),
        page_table=torch.tensor([[0]], dtype=torch.int32),
        kv_cache=adapter._vllm_kv_cache,
        start_pos=torch.tensor([9], dtype=torch.int32),
        enable_trace=True,
        read_from_device=False,
        sampling_params=None,
        reset_batch=False,
    )

    _, kwargs = generator.decode_calls[0]
    # Host sampling has no device token feedback, so host inputs are
    # authoritative even when the previous device-sampled step was current.
    assert kwargs["sampling_mode"] == "host"
    assert kwargs["reset_batch"] is True
    assert kwargs["serving_mode"] is True


def test_async_read_envelope_is_delegated_to_the_canonical_formatter() -> None:
    adapter, generator = _adapter()
    host_handles = (object(), None)
    read_events = [object()]

    result = adapter.process_decode_output_host((host_handles, read_events), is_tokens=True)

    assert torch.equal(result, generator.process_result)
    assert generator.process_calls == [((host_handles, read_events), True)]
