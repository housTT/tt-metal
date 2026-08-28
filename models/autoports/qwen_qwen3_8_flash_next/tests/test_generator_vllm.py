# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only protocol tests for the Qwen3.8 vLLM adapter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch

from models.autoports.qwen_qwen3_8_flash_next.tt import generator as generator_module
from models.autoports.qwen_qwen3_8_flash_next.tt import generator_vllm as adapter_module
from models.autoports.qwen_qwen3_8_flash_next.tt import model as model_module
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import Qwen38Generator, _ServingDecodeOutput
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import Qwen4ExpForConditionalGeneration
from models.autoports.qwen_qwen3_8_flash_next.tt.model import Qwen38FullModel


class _GeneratorSpy:
    def __init__(self) -> None:
        self.prefill_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.decode_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.read_calls = []
        self.process_calls = []
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


def _adapter():
    generator = _GeneratorSpy()
    adapter = Qwen4ExpForConditionalGeneration(generator, max_batch_size=1, max_seq_len=262_144)
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


def test_static_protocol_and_full_context_contract() -> None:
    cls = Qwen4ExpForConditionalGeneration
    assert cls.model_capabilities == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "supports_request_specific_rope": False,
        "supports_device_sampling_penalties": False,
        "device_sampling_max_top_k": 32,
    }
    assert cls.get_max_tokens_all_users(max_model_len=262_144, max_num_seqs=1) == 262_144
    for name in ("tokens", "page_table", "kv_cache", "start_pos", "reset_batch", "slot_remap"):
        assert name in inspect.signature(cls.decode_forward).parameters


def test_initialize_uses_selected_precision_and_proven_tp2(monkeypatch) -> None:
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
    mesh = SimpleNamespace(get_num_devices=lambda: 2)
    result = Qwen4ExpForConditionalGeneration.initialize_vllm_model(
        SimpleNamespace(_name_or_path=adapter_module.MODEL_ID),
        mesh,
        max_batch_size=1,
        max_seq_len=262_144,
    )
    assert result.max_seq_len == 262_144
    assert captured["max_batch"] == 1
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


def test_vllm_cache_adoption_includes_qsa_compressed_attention_state(monkeypatch) -> None:
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
    assert cache[0][3].shape == (4097, 1, 16, 128)
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
        start_pos=torch.tensor([0]),
        enable_trace=True,
    )
    assert result is generator.prefill_result
    args, kwargs = generator.prefill_calls[0]
    assert args[0] is tokens
    assert kwargs["page_table"] is pages
    assert kwargs["kv_cache"] is adapter._vllm_kv_cache
    assert kwargs["prompt_lens"] == [67]
    assert kwargs["on_device_sampling"] is True


def test_vllm_prefill_cache_pages_follow_logical_tokens_not_compute_padding(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "_upload_replicated", lambda value, *_args, **_kwargs: value.clone())
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

    reset_source = inspect.getsource(FunctionalDecoder._reset_user_state)
    assert "ttnn.copy" in reset_source
    assert "ttnn.multiply" not in reset_source
    assert "host_expert_cache.reset" not in inspect.getsource(Qwen38FullModel.reset_batch_state)


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
        slot_remap=torch.tensor([0]),
    )
    adapter.decode_forward(reset_batch=False, **common)
    adapter.decode_forward(reset_batch=False, **common)
    assert generator.decode_calls[0][1]["reset_batch"] is True
    assert generator.decode_calls[1][1]["reset_batch"] is False
    assert all(call[1]["serving_mode"] is True for call in generator.decode_calls)

    read = adapter.read_decode_output(generator.decode_result, async_read=True)
    assert read is generator.read_result
    processed = adapter.process_decode_output_host(read, is_tokens=True)
    assert processed is generator.process_result
    assert generator.read_calls == [(generator.decode_result, True)]
    assert generator.process_calls == [(read, True)]


def test_generator_steady_decode_reads_device_token_only_for_ple() -> None:
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
    batch_state = SimpleNamespace(token_input=object())
    generator = object.__new__(Qwen38Generator)
    generator.model = FakeModel()
    generator._serving_device_feedback_current = True
    generator._serving_sampling_signature = None
    generator.host_sampling_compatibility_calls = 0

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


def test_canonical_sampler_converts_user_temperature_to_kernel_multiplier(monkeypatch) -> None:
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


def test_canonical_sampler_routes_top1_at_any_temperature_to_argmax(monkeypatch) -> None:
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
    for forbidden in ("torch.argmax", "ttnn.argmax", "torch.topk", "ttnn.topk", "logits_to_torch"):
        assert forbidden not in source
