# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.autoports.openai_gpt_oss_120b.tt.generator import Generator
from models.autoports.openai_gpt_oss_120b.tt.generator_vllm import (
    ENV_CHUNK_WARMUP,
    ENV_KV_POOL_BLOCKS,
    FULL_CONTEXT_BLOCKS,
    KV_POOL_BLOCKS_DEFAULT,
    PAGE_SIZE,
    SLIDING_RING_BLOCKS,
    SLIDING_RING_TOKENS,
    TTGptOssForCausalLM,
    _sampling_key,
    _sampling_params_to_host_values,
    kv_pool_blocks,
    ring_page_table,
    sliding_layer_indices,
)
from models.autoports.openai_gpt_oss_120b.tt.model import (
    HF_CONTEXT_LENGTH,
    FullModelArgs,
    FullModelCapacityError,
    Model,
)
from models.common.sampling.sampling_params import SamplingParams
from models.tt_transformers.tt.common import get_block_size
from models.tt_transformers.tt.generator import (
    assemble_decode_trace_vector,
    prefilled_decode_rows,
    slice_prefill_page_tables_per_layer,
)


class _FakeModel:
    def __init__(self, n_layers=2):
        self.n_layers = n_layers
        self.page_table_updates = []
        self.decode_batch_sizes = []

    def update_persistent_per_layer_page_tables(self, tables):
        self.page_table_updates.append([table.clone() for table in tables])

    def activate_decode_batch_size(self, batch_size):
        self.decode_batch_sizes.append(batch_size)


class _FakeGenerator:
    def __init__(self):
        self.prefill_calls = []
        self.decode_calls = []
        self.prefill_warmup_calls = []
        self.decode_warmup_calls = []
        self.decode_prepare_calls = []
        self.decode_capture_calls = []
        self.trace_release_calls = 0
        self.trace_live = True
        self.teardown_calls = 0
        self._lifetime_decode_trace_releases_for_prefill_compile = 0
        self.release_for_prefill_compile = False
        self.already_warmed_up_prefill = True
        self._inner = SimpleNamespace(
            model=[SimpleNamespace(sampling=None)],
            trace_ids_decode={},
            trace_inputs_decode={},
            trace_output_decode={},
        )

    def prefill_forward(self, tokens, **kwargs):
        self.prefill_calls.append((tokens.clone(), kwargs))
        if self.release_for_prefill_compile:
            self._lifetime_decode_trace_releases_for_prefill_compile += 1
        return torch.tensor([17])

    def decode_forward(self, tokens, start_pos, **kwargs):
        self.decode_calls.append((tokens.clone(), start_pos.clone(), kwargs))
        return object()

    def warmup_model_prefill(self, **kwargs):
        self.prefill_warmup_calls.append(kwargs)

    def warmup_model_decode(self, **kwargs):
        self.decode_warmup_calls.append(kwargs)
        return "decode-warmup"

    def prepare_model_decode_trace(self, **kwargs):
        self.decode_prepare_calls.append(kwargs)
        return {
            "width": kwargs["max_batch_size"],
            "on_device_sampling": kwargs.get("on_device_sampling", True),
        }

    def capture_prepared_model_decode_trace(self, prepared):
        self.decode_capture_calls.append(prepared)
        self.trace_live = True

    def release_decode_traces_for_host_sampling(self):
        if not self.trace_live:
            return False
        self.trace_release_calls += 1
        self.trace_live = False
        return True

    def teardown(self):
        self.teardown_calls += 1


def _adapter(n_layers=2):
    model = _FakeModel(n_layers)
    adapter = TTGptOssForCausalLM(
        model,
        SimpleNamespace(),
        mesh_device=SimpleNamespace(shape=(1, 4)),
        max_batch_size=32,
        max_model_len=HF_CONTEXT_LENGTH,
        hf_config=SimpleNamespace(head_dim=64),
    )
    adapter.generator = _FakeGenerator()
    return adapter


def _greedy(batch=2):
    return SamplingParams(
        temperature=[0.0] * batch,
        top_k=[1] * batch,
        top_p=[1.0] * batch,
        presence_penalty=[0.0] * batch,
        frequency_penalty=[0.0] * batch,
        repetition_penalty=[1.0] * batch,
        seed=[None] * batch,
        enable_log_probs=[False] * batch,
        num_logprobs=[0] * batch,
    )


def _sampled(batch=2):
    return SamplingParams(
        temperature=[0.7] * batch,
        top_k=[20] * batch,
        top_p=[0.9] * batch,
        presence_penalty=[0.0] * batch,
        frequency_penalty=[0.0] * batch,
        repetition_penalty=[1.0] * batch,
        seed=[None] * batch,
        enable_log_probs=[False] * batch,
        num_logprobs=[0] * batch,
    )


def test_capacity_contract_requires_p150x4_and_full_context(expect_error):
    kwargs = {
        "model_name": "openai/gpt-oss-120b",
        "tt_data_parallel": 1,
        "max_model_len": HF_CONTEXT_LENGTH,
        "max_num_seqs": 32,
    }
    assert TTGptOssForCausalLM.get_max_tokens_all_users(num_devices=4, **kwargs) == (KV_POOL_BLOCKS_DEFAULT - 32) * 64
    for num_devices in (1, 2):
        with expect_error(FullModelCapacityError, "P150x4"):
            TTGptOssForCausalLM.get_max_tokens_all_users(num_devices=num_devices, **kwargs)
    with expect_error(ValueError, "context_contract.json"):
        TTGptOssForCausalLM.get_max_tokens_all_users(
            num_devices=4,
            **{**kwargs, "max_model_len": HF_CONTEXT_LENGTH - 1},
        )
    with expect_error(ValueError, "1..32 concurrent sequences"):
        TTGptOssForCausalLM.get_max_tokens_all_users(
            num_devices=4,
            **{**kwargs, "max_num_seqs": 33},
        )


def test_kv_pool_blocks_knob_lands_on_the_requested_pool(monkeypatch, expect_error):
    assert kv_pool_blocks(32) == KV_POOL_BLOCKS_DEFAULT
    monkeypatch.setenv(ENV_KV_POOL_BLOCKS, "8066")
    assert kv_pool_blocks(32) == 8066
    assert (
        TTGptOssForCausalLM.get_max_tokens_all_users(
            model_name="openai/gpt-oss-120b",
            num_devices=4,
            tt_data_parallel=1,
            max_model_len=HF_CONTEXT_LENGTH,
            max_num_seqs=32,
        )
        == (8066 - 32) * PAGE_SIZE
    )
    monkeypatch.setenv(ENV_KV_POOL_BLOCKS, str(FULL_CONTEXT_BLOCKS + 31))
    with expect_error(ValueError, "outside"):
        kv_pool_blocks(32)
    monkeypatch.setenv(ENV_KV_POOL_BLOCKS, str(32 * (FULL_CONTEXT_BLOCKS + 1) + 1))
    with expect_error(ValueError, "outside"):
        kv_pool_blocks(32)


def test_kv_cache_spec_names_only_the_full_attention_layers():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    hf_config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"] * 3,
        sliding_window=128,
        head_dim=64,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda parallel_config: 2,
        get_head_size=lambda: 64,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(block_size=64, cache_dtype="auto"),
        parallel_config=None,
    )
    spec = TTGptOssForCausalLM.get_kv_cache_spec(vllm_config)
    assert sorted(spec) == [f"model.layers.{i}.self_attn" for i in (1, 3, 5)]
    assert all(isinstance(entry, FullAttentionSpec) for entry in spec.values())
    assert TTGptOssForCausalLM._HYBRID_KV_CACHE_GROUPS_ENABLED is False
    assert sliding_layer_indices(hf_config, 6) == [0, 2, 4]
    assert sliding_layer_indices(SimpleNamespace(), 5) == [0, 2, 4]


def test_ring_page_table_is_cyclic_and_masks_the_padded_tail():
    table = ring_page_table([3, 0], 26, valid_blocks=[26, 2])
    assert SLIDING_RING_BLOCKS == 12 and SLIDING_RING_TOKENS == 768
    assert table[0].tolist() == [36 + column % 12 for column in range(26)]
    assert table[1].tolist() == [0, 1] + [-1] * 24
    assert ring_page_table([1], 3).tolist() == [[12, 13, 14]]
    assert ring_page_table([1], 3, base=4640).tolist() == [[4652, 4653, 4654]]


def test_ring_tables_sit_above_the_pool_after_allocation():
    adapter = _adapter(n_layers=2)
    adapter._ring_block_base = 4640
    tables = adapter._normalise_page_tables(None, torch.zeros(2, 14, dtype=torch.int32))
    assert tables[0][0].tolist() == [4640 + column % 12 for column in range(14)]
    assert tables[0][1].tolist() == [4652 + column % 12 for column in range(14)]
    assert tables[1].shape == (2, 14) and int(tables[1].max()) == 0


def test_decode_routes_ring_tables_for_sliding_layers_by_slot():
    adapter = _adapter(n_layers=4)
    page_table = torch.arange(64, dtype=torch.int32).reshape(4, 16) + 100
    adapter.decode_forward(
        tokens=torch.zeros(4, 1, dtype=torch.int32),
        start_pos=torch.tensor([65, 97, 12, 3]),
        page_table=page_table,
        page_tables_per_layer=None,
        kv_cache=object(),
        sampling_params=_greedy(4),
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )
    tables = adapter.model.page_table_updates[-1]
    assert torch.equal(tables[1], page_table) and torch.equal(tables[3], page_table)
    expected = ring_page_table([0, 1, 2, 3], 16)
    assert torch.equal(tables[0], expected) and torch.equal(tables[2], expected)
    assert tables[0].shape == page_table.shape


def test_slot_remap_moves_ring_ownership_and_refreshes_tables(expect_error):
    adapter = _adapter(n_layers=2)
    page_table = torch.zeros(32, 8, dtype=torch.int32)
    common = {
        "tokens": torch.zeros(32, 1, dtype=torch.int32),
        "start_pos": torch.tensor([5] * 32),
        "page_table": page_table,
        "kv_cache": object(),
        "sampling_params": _greedy(32),
        "enable_trace": True,
        "read_from_device": False,
        "page_table_state_id": 7,
    }
    adapter.decode_forward(reset_batch=True, **common)
    assert adapter.serving_counters["page_table_refreshes"] == 1
    adapter.decode_forward(reset_batch=False, **common)
    assert adapter.serving_counters["page_table_reuses"] == 1
    remap = torch.tensor([3, 0, 1, 2] + list(range(4, 32)), dtype=torch.int32)
    adapter.decode_forward(reset_batch=True, slot_remap=remap, **common)
    assert adapter._ring_of_slot[:4] == [3, 0, 1, 2]
    assert adapter.serving_counters["page_table_refreshes"] == 2
    assert adapter.model.page_table_updates[-1][0][:4, :4].tolist() == ring_page_table([3, 0, 1, 2], 4).tolist()
    with expect_error(ValueError, "permutation"):
        adapter._apply_ring_slot_remap(torch.tensor([0, 0] + list(range(2, 32))))


def test_prefill_ring_tables_follow_empty_slots_and_prompt_lengths():
    adapter = _adapter(n_layers=2)
    page_table = torch.arange(32, dtype=torch.int32).reshape(2, 16) + 500
    adapter.prefill_forward(
        tokens=torch.zeros(2, 300, dtype=torch.int32),
        page_table=page_table,
        kv_cache=object(),
        prompt_lens=[300, 70],
        page_tables_per_layer=None,
        sampling_params=_greedy(2),
        empty_slots=[7, 2],
    )
    tables = adapter.generator.prefill_calls[-1][1]["page_tables_per_layer"]
    assert torch.equal(tables[1], page_table)
    assert tables[0].tolist() == ring_page_table([7, 2], 16, valid_blocks=[5, 2]).tolist()
    assert tables[0][0, :6].tolist() == [84, 85, 86, 87, 88, -1]
    assert tables[0][1, :3].tolist() == [24, 25, -1]


def test_device_sampling_capability_declares_exact_top_k_limit():
    assert TTGptOssForCausalLM.model_capabilities["supports_sample_on_device"]
    assert TTGptOssForCausalLM.model_capabilities["max_device_sampling_top_k"] == 32
    assert TTGptOssForCausalLM.model_capabilities["accepts_tensor_sampling_params"]
    assert TTGptOssForCausalLM.model_capabilities["accepts_serving_state_ids"]


def test_tensor_sampling_params_convert_seed_sentinel_without_mutating_input():
    from dataclasses import make_dataclass

    tensor_params_type = make_dataclass(
        "TensorSamplingParams",
        [
            (field, object)
            for field in (
                "temperature",
                "top_k",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "seed",
                "enable_log_probs",
                "num_logprobs",
            )
        ],
        frozen=True,
    )
    params = tensor_params_type(
        torch.tensor([0.0, 0.7]),
        torch.tensor([1, 20]),
        torch.tensor([1.0, 0.9]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([-1, 7]),
        torch.tensor([False, False]),
        torch.tensor([0, 0]),
    )

    converted = _sampling_params_to_host_values(params)

    assert converted.temperature == pytest.approx([0.0, 0.7])
    assert converted.seed == [None, 7]
    assert torch.equal(params.seed, torch.tensor([-1, 7]))


def test_adapter_exposes_the_decode_buckets_it_prepares():
    assert TTGptOssForCausalLM.tt_supported_decode_batch_sizes == (1, 4, 8, 32)


def test_vllm_initialization_disables_duplicate_seed_salting_before_sampling_construction(
    monkeypatch,
):
    captured = {}

    def fake_from_checkpoint(cls, mesh_device, **kwargs):
        del cls, mesh_device
        captured.update(kwargs)
        return _FakeModel(), SimpleNamespace(
            salt_duplicate_seeds=kwargs["salt_duplicate_seeds"],
            tensor_cache_path=Path("/tmp/gpt-oss-120b-vllm-test"),
        )

    monkeypatch.setattr(Model, "from_checkpoint", classmethod(fake_from_checkpoint))

    adapter = TTGptOssForCausalLM.initialize_vllm_model(
        hf_config=SimpleNamespace(head_dim=64),
        mesh_device=SimpleNamespace(shape=(1, 4)),
        max_batch_size=32,
        max_seq_len=HF_CONTEXT_LENGTH,
        n_layers=1,
    )

    assert captured["create_kv_cache"] is False
    assert captured["salt_duplicate_seeds"] is False
    assert adapter.model_args.salt_duplicate_seeds is False


def test_persistent_capture_shutdown_flushes_runtime_evidence_before_trace_release():
    adapter = _adapter()
    generator = adapter.generator
    events = []
    adapter._write_serving_capability = lambda: events.append("snapshot")
    generator.teardown = lambda: events.append("teardown")

    adapter.release_persistent_capture()
    adapter.release_persistent_capture()

    assert events == ["snapshot", "teardown"]
    assert adapter.generator is None


def test_nonaligned_prefill_delegates_to_canonical_generator():
    adapter = _adapter()
    page_table = torch.zeros(2, 2048, dtype=torch.int32)
    tokens = torch.arange(2 * 97, dtype=torch.int64).reshape(2, 97)
    result = adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        page_tables_per_layer=[page_table, page_table],
        kv_cache=object(),
        prompt_lens=[65, 97],
        sampling_params=_greedy(),
        empty_slots=[3, 7],
        enable_trace=True,
    )
    assert result.tolist() == [17]
    _, call = adapter.generator.prefill_calls[-1]
    assert call["prompt_lens"] == [65, 97]
    assert call["empty_slots"] == [3, 7]
    assert call["sampling_params"] == _greedy()
    assert torch.equal(call["page_tables_per_layer"][0], ring_page_table([3, 7], 2048, valid_blocks=[2, 2]))
    assert call["page_tables_per_layer"][1] is page_table
    assert call["enable_trace"] is False
    assert adapter.serving_counters["prefill_calls"] == 1
    assert adapter.model.page_table_updates == []


def _resumed_prefill(adapter, start_pos, prompt_lens=(9000, 8700), empty_slots=(1, 4)):
    page_table = torch.zeros(2, 2048, dtype=torch.int32)
    tokens = torch.arange(2 * 9000, dtype=torch.int64).reshape(2, 9000)
    adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        page_tables_per_layer=None,
        kv_cache=object(),
        prompt_lens=list(prompt_lens),
        sampling_params=_greedy(),
        empty_slots=list(empty_slots),
        start_pos=start_pos,
    )
    _, call = adapter.generator.prefill_calls[-1]
    return call


def test_chunk_continuation_on_a_warm_ring_resumes_at_the_aligned_start():
    adapter = _adapter()
    adapter._slot_prefill_end[1] = 8192
    call = _resumed_prefill(adapter, [8192, 300])
    assert call["start_pos"] == [8192, 0]
    assert call["fill_start_pos"] == [8192, 256]
    assert call["ring_cold"] == [False, False]
    assert adapter._slot_prefill_end[1] == 9000 and adapter._slot_prefill_end[4] == 8700


def test_cached_prefix_on_a_cold_ring_restarts_the_reprefill_window_earlier():
    adapter = _adapter()
    call = _resumed_prefill(adapter, [8192, 300])
    assert call["start_pos"] == [8192 - 2560, 0]
    assert call["fill_start_pos"] == [8192, 256]
    assert call["ring_cold"] == [True, False]
    adapter.release_request(1)
    adapter._slot_prefill_end[4] = 8700
    call = _resumed_prefill(adapter, torch.tensor([8192, 8700]), prompt_lens=(9000, 8760))
    assert call["start_pos"] == [8192 - 2560, 8704 - 512]
    assert call["fill_start_pos"] == [8192, 8704 - 64]
    assert call["ring_cold"] == [True, False]


def test_new_request_flag_forces_a_cold_resume_even_when_the_marker_matches():
    adapter = _adapter()
    adapter._slot_prefill_end[1] = 8192
    page_table = torch.zeros(2, 2048, dtype=torch.int32)
    tokens = torch.arange(2 * 9000, dtype=torch.int64).reshape(2, 9000)
    adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        page_tables_per_layer=None,
        kv_cache=object(),
        prompt_lens=[9000, 8700],
        sampling_params=_greedy(),
        empty_slots=[1, 4],
        start_pos=[8192, 8192],
        new_request_rows=[True, False],
    )
    _, call = adapter.generator.prefill_calls[-1]
    assert call["start_pos"] == [8192 - 2560, 8192]
    assert call["ring_cold"] == [True, False]
    adapter._slot_prefill_end[1] = 9000
    adapter._slot_prefill_end[4] = 8700
    adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        page_tables_per_layer=None,
        kv_cache=object(),
        prompt_lens=[9000, 8700],
        sampling_params=_greedy(),
        empty_slots=[1, 4],
        start_pos=[9000, 8700],
        new_request_rows=[False, False],
    )
    _, call = adapter.generator.prefill_calls[-1]
    assert call["start_pos"] == [8704, 8192]
    assert call["ring_cold"] == [False, False]


def test_prefill_without_cached_tokens_sends_no_resume_plan():
    adapter = _adapter()
    call = _resumed_prefill(adapter, torch.tensor([0, 0]))
    assert call["start_pos"] is None and call["fill_start_pos"] is None and call["ring_cold"] is None
    call = _resumed_prefill(adapter, torch.tensor([100, 511]))
    assert call["start_pos"] == [0, 0]
    assert call["fill_start_pos"] == [64, 448]
    assert call["ring_cold"] == [False, False]


def test_ring_slot_remap_moves_the_prefill_end_marker_with_the_ring():
    adapter = _adapter()
    adapter._slot_prefill_end[:3] = [100, 200, 300]
    adapter._apply_ring_slot_remap([2, 0, 1])
    assert adapter._slot_prefill_end[:3] == [300, 100, 200]
    assert adapter._ring_of_slot[:3] == [2, 0, 1]


def test_chunk_continuation_programs_are_keyed_by_chunk_length():
    fresh = Generator._prefill_program_signature(9000, "device_sampling")
    chunk = Generator._prefill_program_signature(9000, "device_sampling", 8192)
    assert fresh == (16384, 9024, 8992, "device_sampling")
    assert chunk == (1024, 9024, 800, "device_sampling:chunk")
    later = Generator._prefill_program_signature(17192, "device_sampling", 16384)
    assert later == (1024, 17216, 800, "device_sampling:chunk")
    assert later != chunk
    cold = Generator._prefill_program_signature(9000, "device_sampling", 5632, 8192, True)
    assert cold == (4096, 9024, 3360, "device_sampling:chunk:cold:fill2560")
    partial = Generator._prefill_program_signature(9000, "device_sampling", 0, 4992)
    assert partial == (16384, 9024, 8992, "device_sampling:fill4992")
    assert Generator._prefill_program_signature(808, "device_sampling") == (1024, 832, 800, "device_sampling")


def test_prefill_compile_release_restores_every_prepared_decode_bucket():
    adapter = _adapter()
    adapter.generator.release_for_prefill_compile = True
    restored = []
    adapter._restore_device_decode_traces = restored.append
    page_table = torch.zeros(1, 2048, dtype=torch.int32)
    kv_cache = object()

    result = adapter.prefill_forward(
        tokens=torch.arange(3, dtype=torch.int64).reshape(1, 3),
        page_table=page_table,
        page_tables_per_layer=[page_table, page_table],
        kv_cache=kv_cache,
        prompt_lens=[3],
        sampling_params=_greedy(1),
        empty_slots=[0],
    )

    assert result.tolist() == [17]
    assert restored == [kv_cache]


def test_prefill_warmup_state_is_proxied_but_serving_prefill_is_untraced():
    adapter = _adapter()
    assert adapter.already_warmed_up_prefill
    adapter.already_warmed_up_prefill = False
    assert not adapter.generator.already_warmed_up_prefill

    adapter.warmup_model_prefill(
        kv_cache=object(),
        enable_trace=True,
        can_sample_on_device=True,
    )
    assert adapter.generator.prefill_warmup_calls[-1]["enable_trace"] is False
    tokens, kwargs = adapter.generator.prefill_calls[0]
    assert tokens.shape == (1, 128)
    assert kwargs["prompt_lens"] == [128]
    assert kwargs["sampling_params"].top_k == 1
    assert kwargs["enable_trace"] is False
    assert len(kwargs["page_tables_per_layer"]) == adapter.model.n_layers
    assert all(table.shape == (1, 2) for table in kwargs["page_tables_per_layer"])


def test_prefill_warmup_compiles_the_chunk_continuation_shapes(monkeypatch):
    monkeypatch.setenv(ENV_CHUNK_WARMUP, "8192:32768")
    adapter = _adapter()
    adapter.warmup_model_prefill(kv_cache=object(), enable_trace=False, can_sample_on_device=True)
    continuations = [(tokens, kwargs) for tokens, kwargs in adapter.generator.prefill_calls if kwargs.get("start_pos")]
    assert [kwargs["start_pos"] for _, kwargs in continuations] == [[8192], [16384], [24576]]
    assert [kwargs["prompt_lens"] for _, kwargs in continuations] == [[16384], [24576], [32768]]
    tokens, kwargs = continuations[-1]
    assert tokens.shape == (1, 32768)
    assert kwargs["sampling_params"].top_k == 1
    assert kwargs["enable_trace"] is False
    assert kwargs["empty_slots"] == [0]
    assert kwargs["page_table"].shape == (1, 512)
    sliding, full = kwargs["page_tables_per_layer"]
    assert torch.equal(full, torch.arange(512, dtype=torch.int32).reshape(1, 512))
    assert sliding.shape == (1, 512)
    assert int(sliding.min()) >= 0
    assert torch.equal(sliding[0, :SLIDING_RING_BLOCKS], sliding[0, SLIDING_RING_BLOCKS : 2 * SLIDING_RING_BLOCKS])


def test_prefill_warmup_skips_chunk_continuations_when_disabled(monkeypatch):
    monkeypatch.setenv(ENV_CHUNK_WARMUP, "")
    adapter = _adapter()
    adapter.warmup_model_prefill(kv_cache=object(), enable_trace=False, can_sample_on_device=True)
    assert not any(kwargs.get("start_pos") for _, kwargs in adapter.generator.prefill_calls)


def test_get_block_size_never_indexes_a_device_tensor():
    class ShapeOnlyTensor:
        shape = (4096, 2, 64, 64)

        def __getitem__(self, index):
            raise AssertionError(f"device tensor indexing is forbidden: {index}")

    tensor = ShapeOnlyTensor()
    assert get_block_size([tensor, tensor]) == 64
    assert get_block_size([[tensor, tensor], [tensor, tensor]]) == 64


def test_steady_decode_reuses_sampling_and_page_table_state():
    adapter = _adapter()
    tables = [
        torch.zeros(2, 2048, dtype=torch.int32),
        torch.ones(2, 2048, dtype=torch.int32),
    ]
    params = _greedy()
    slot_remap = torch.tensor([1, 0], dtype=torch.int32)
    common = {
        "page_table": tables[0],
        "page_tables_per_layer": tables,
        "kv_cache": object(),
        "sampling_params": params,
        "enable_trace": True,
        "read_from_device": False,
    }

    adapter.decode_forward(
        tokens=torch.tensor([[11], [12]]),
        start_pos=torch.tensor([65, 97]),
        slot_remap=slot_remap,
        reset_batch=True,
        removal_only_reset=True,
        **common,
    )
    adapter.decode_forward(
        # Intentionally stale host-side values: the canonical split path owns
        # token feedback and current-position progression after reset.
        tokens=torch.tensor([[11], [12]]),
        start_pos=torch.tensor([65, 97]),
        slot_remap=None,
        reset_batch=False,
        **common,
    )

    first = adapter.generator.decode_calls[0][2]
    steady = adapter.generator.decode_calls[1][2]
    assert first["sampling_params"] == params
    assert not first["reuse_sampling_state"]
    assert first["force_host_tokens"] is False
    assert "removal_only_reset" not in first
    assert steady["sampling_params"] is None
    assert steady["reuse_sampling_state"]
    assert steady["slot_remap"] is None
    assert len(adapter.model.page_table_updates) == 1
    assert adapter.serving_counters["page_table_refreshes"] == 1
    assert adapter.serving_counters["page_table_reuses"] == 1
    assert adapter.serving_counters["sampling_state_pushes"] == 1
    assert adapter.serving_counters["sampling_state_reuses"] == 1

    # The plugin owns the drained reset signal. A remap without that signal is
    # not allowed to turn async-lagging host tokens into the authority here.
    adapter.decode_forward(
        tokens=torch.tensor([[13], [14]]),
        start_pos=torch.tensor([66, 98]),
        slot_remap=slot_remap,
        reset_batch=False,
        **common,
    )
    remapped = adapter.generator.decode_calls[-1][2]
    assert remapped["reset_batch"] is False
    assert remapped["force_host_tokens"] is False
    assert remapped["reuse_sampling_state"] is True
    # The remap moved the sliding-window rings with their requests, which is a
    # page-table refresh of the ring layers even though vLLM's tables are unchanged.
    assert adapter._ring_of_slot[:2] == [0, 1]
    assert len(adapter.model.page_table_updates) == 2

    changed_tables = [tables[0], tables[1].clone()]
    changed_tables[1][0, 0] = 3
    adapter.decode_forward(
        tokens=torch.tensor([[13], [14]]),
        start_pos=torch.tensor([66, 98]),
        page_table=changed_tables[0],
        page_tables_per_layer=changed_tables,
        kv_cache=object(),
        sampling_params=params,
        enable_trace=True,
        read_from_device=False,
    )
    assert len(adapter.model.page_table_updates) == 3


def test_steady_decode_state_ids_skip_sampling_materialization(monkeypatch):
    adapter = _adapter()
    page_table = torch.zeros(2, 8, dtype=torch.int32)
    params = _greedy()
    materializations = 0
    original = _sampling_params_to_host_values

    def counted_materialization(value):
        nonlocal materializations
        materializations += 1
        return original(value)

    monkeypatch.setattr(
        "models.autoports.openai_gpt_oss_120b.tt.generator_vllm." "_sampling_params_to_host_values",
        counted_materialization,
    )
    common = {
        "tokens": torch.tensor([[11], [12]]),
        "start_pos": torch.tensor([65, 97]),
        "page_table": page_table,
        "page_tables_per_layer": [page_table, page_table],
        "page_table_state_id": 3,
        "sampling_state_id": 5,
        "kv_cache": object(),
        "sampling_params": params,
        "enable_trace": True,
        "read_from_device": False,
    }

    adapter.decode_forward(reset_batch=True, **common)
    adapter.decode_forward(reset_batch=False, **common)

    assert materializations == 1
    assert adapter.generator.decode_calls[-1][2]["sampling_params"] is None
    assert adapter.serving_counters["page_table_reuses"] == 1


def test_repeated_layer_page_tables_are_compared_and_cloned_once_per_group(monkeypatch):
    adapter = _adapter(n_layers=4)
    first = torch.zeros(2, 8, dtype=torch.int32)
    second = torch.ones(2, 8, dtype=torch.int32)
    params = _greedy()
    common = {
        "tokens": torch.tensor([[11], [12]]),
        "start_pos": torch.tensor([65, 97]),
        "page_table": first,
        "kv_cache": object(),
        "sampling_params": params,
        "enable_trace": True,
        "read_from_device": False,
    }
    adapter.decode_forward(
        page_tables_per_layer=[first, first, second, second],
        reset_batch=True,
        **common,
    )
    assert adapter._last_page_tables[0] is adapter._last_page_tables[2]
    assert torch.equal(adapter._last_page_tables[1], first)
    assert torch.equal(adapter._last_page_tables[3], second)

    equal_calls = 0
    original_equal = torch.equal

    def counted_equal(left, right):
        nonlocal equal_calls
        equal_calls += 1
        return original_equal(left, right)

    monkeypatch.setattr(torch, "equal", counted_equal)
    first_next = first.clone()
    second_next = second.clone()
    adapter.decode_forward(
        page_table=first_next,
        page_tables_per_layer=[first_next, first_next, second_next, second_next],
        reset_batch=False,
        **{key: value for key, value in common.items() if key != "page_table"},
    )

    assert equal_calls == 3
    assert adapter.serving_counters["page_table_reuses"] == 1


@pytest.mark.parametrize("bucket", [1, 32])
def test_selected_policy_sampled_greedy_sampled_keeps_one_device_trace(bucket):
    adapter = _adapter()
    kv_cache = object()
    table = torch.zeros(32, 2048, dtype=torch.int32)

    positions = torch.tensor([65] + ([-1] * 31) if bucket == 1 else [65, 66] + ([-1] * 30))
    common = {
        "tokens": torch.zeros(32, 1, dtype=torch.int32),
        "start_pos": positions,
        "page_table": table,
        "page_tables_per_layer": [table, table],
        "kv_cache": kv_cache,
        "enable_trace": True,
        "read_from_device": False,
    }
    for index, params in enumerate((_sampled(32), _greedy(32), _sampled(32))):
        adapter.decode_forward(
            sampling_params=params,
            reset_batch=index == 0,
            **common,
        )

    calls = adapter.generator.decode_calls[-3:]
    assert [call[0].shape[0] for call in calls] == [bucket] * 3
    assert [call[2]["sampling_params"] for call in calls] == [
        _sampled(32),
        _greedy(32),
        _sampled(32),
    ]
    assert all(call[2].get("skip_trace_precompile", False) is False for call in calls)
    assert all(call[2]["reuse_sampling_state"] is False for call in calls)
    assert adapter.generator.trace_release_calls == 0
    assert adapter.generator.decode_capture_calls == []
    assert adapter.serving_counters["device_sampled_decodes"] == 3
    assert adapter.serving_counters["host_sampled_decodes"] == 0


def test_host_sampling_is_an_explicit_eager_compatibility_path():
    adapter = _adapter()
    page_table = torch.zeros(32, 2048, dtype=torch.int32)

    adapter.decode_forward(
        tokens=torch.arange(32, dtype=torch.int32).reshape(32, 1),
        start_pos=torch.tensor([65, 97] + [-1] * 30),
        page_table=page_table,
        kv_cache=object(),
        sampling_params=None,
        enable_trace=True,
        read_from_device=True,
    )

    call = adapter.generator.decode_calls[-1][2]
    assert call["sampling_mode"] == "host"
    assert call["enable_trace"] is False
    assert adapter.serving_counters["host_sampled_decodes"] == 1
    assert adapter.serving_counters["device_sampled_decodes"] == 0
    assert adapter.generator.trace_release_calls == 1
    assert adapter._host_sampling_active


def test_host_sampling_after_b1_trace_activates_serving_width():
    adapter = _adapter()
    page_table = torch.zeros(32, 2048, dtype=torch.int32)
    kv_cache = object()
    adapter.decode_forward(
        tokens=torch.zeros(32, 1, dtype=torch.int32),
        start_pos=torch.tensor([65] + [-1] * 31),
        page_table=page_table,
        page_tables_per_layer=[page_table, page_table],
        kv_cache=kv_cache,
        sampling_params=_greedy(32),
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )
    assert adapter._active_decode_bucket == 1

    adapter.decode_forward(
        tokens=torch.zeros(32, 1, dtype=torch.int32),
        start_pos=torch.tensor([66] + [-1] * 31),
        page_table=page_table,
        page_tables_per_layer=[page_table, page_table],
        kv_cache=kv_cache,
        sampling_params=None,
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )

    assert adapter._active_decode_bucket == 32
    assert adapter.model.decode_batch_sizes[-1] == 32
    assert adapter.generator.decode_calls[-1][0].shape == (32, 1)


def test_singleton_host_sampling_slices_warmup_padded_hybrid_page_tables():
    adapter = _adapter()
    page_table = torch.zeros(1, 16, dtype=torch.int32)
    padded_per_layer = torch.zeros(32, 16, dtype=torch.int32)

    adapter.decode_forward(
        tokens=torch.zeros(1, 1, dtype=torch.int32),
        start_pos=torch.tensor([65]),
        page_table=page_table,
        page_tables_per_layer=[padded_per_layer, padded_per_layer],
        kv_cache=object(),
        sampling_params=None,
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )

    assert adapter._active_decode_bucket == 1
    assert [table.shape for table in adapter.model.page_table_updates[-1]] == [
        (1, 16),
        (1, 16),
    ]
    assert adapter.generator.decode_calls[-1][2]["page_table"].shape == (1, 16)


def test_host_prefill_releases_then_device_prefill_recaptures_traces(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr("ttnn.synchronize_device", lambda *_: None)
    page_table = torch.zeros(1, 2048, dtype=torch.int32)
    common = {
        "tokens": torch.arange(65, dtype=torch.int64).reshape(1, 65),
        "page_table": page_table,
        "page_tables_per_layer": [page_table, page_table],
        "kv_cache": object(),
        "prompt_lens": [65],
    }

    adapter.prefill_forward(sampling_params=None, **common)
    assert adapter.generator.trace_release_calls == 1
    assert adapter._host_sampling_active
    assert adapter.serving_counters["host_trace_releases"] == 1

    adapter._prepared_decode_buckets = {
        1: {"width": 1, "on_device_sampling": True},
        32: {"width": 32, "on_device_sampling": True},
    }
    adapter.prefill_forward(sampling_params=_greedy(1), **common)

    assert [(call["width"], call["on_device_sampling"]) for call in adapter.generator.decode_capture_calls] == [
        (1, True),
        (32, True),
    ]
    assert not adapter._host_sampling_active
    assert adapter.serving_counters["device_trace_recaptures"] == 1

    # A later host-only decode is authoritative even if its immediately
    # preceding prefill was device-routed. It must quiesce the recaptured
    # traces before entering the eager full-logits compatibility path.
    decode_table = torch.zeros(32, 2048, dtype=torch.int32)
    adapter.decode_forward(
        tokens=torch.zeros(32, 1, dtype=torch.int32),
        start_pos=torch.tensor([65] + [-1] * 31),
        page_table=decode_table,
        page_tables_per_layer=[decode_table, decode_table],
        kv_cache=common["kv_cache"],
        sampling_params=None,
        enable_trace=True,
        read_from_device=False,
    )
    assert adapter.generator.trace_release_calls == 2
    assert adapter._host_sampling_active
    assert adapter.generator.decode_calls[-1][2]["enable_trace"] is False

    # The inverse transition is equally decode-authoritative: restore both
    # prepared buckets before the next on-device traced replay.
    adapter.decode_forward(
        tokens=torch.zeros(32, 1, dtype=torch.int32),
        start_pos=torch.tensor([65] + [-1] * 31),
        page_table=decode_table,
        page_tables_per_layer=[decode_table, decode_table],
        kv_cache=common["kv_cache"],
        sampling_params=_greedy(32),
        # The runner may not mark a route-only transition as a batch reset.
        # Recapture must nevertheless force one authoritative restage before
        # replaying its persistent model/sampler inputs.
        reset_batch=False,
        enable_trace=True,
        read_from_device=False,
    )
    assert adapter.serving_counters["device_trace_recaptures"] == 2
    assert not adapter._host_sampling_active
    recaptured = adapter.generator.decode_calls[-1][2]
    assert recaptured["reset_batch"] is True
    assert recaptured["force_host_tokens"] is True
    assert recaptured["reuse_sampling_state"] is False
    assert not adapter._device_trace_recapture_requires_reset


def test_device_recapture_reset_guard_survives_failed_submission(monkeypatch, expect_error):
    adapter = _adapter()
    adapter._device_trace_recapture_requires_reset = True
    page_table = torch.zeros(32, 16, dtype=torch.int32)
    delegated = {}

    def fail_decode(_tokens, _start_pos, **kwargs):
        delegated.update(kwargs)
        raise RuntimeError("injected decode submission failure")

    monkeypatch.setattr(adapter.generator, "decode_forward", fail_decode)

    with expect_error(RuntimeError, "injected decode submission failure"):
        adapter.decode_forward(
            tokens=torch.zeros(32, 1, dtype=torch.int32),
            start_pos=torch.tensor([65] + [-1] * 31),
            page_table=page_table,
            page_tables_per_layer=[page_table, page_table],
            kv_cache=object(),
            sampling_params=_greedy(32),
            reset_batch=False,
            enable_trace=True,
            read_from_device=False,
        )

    assert delegated["reset_batch"] is True
    assert delegated["force_host_tokens"] is True
    assert adapter._device_trace_recapture_requires_reset


def test_single_active_device_decode_uses_width_one_trace_bucket():
    adapter = _adapter()
    first = torch.arange(32 * 16, dtype=torch.int32).reshape(32, 16)
    second = first + 1000

    adapter.decode_forward(
        tokens=torch.arange(32, dtype=torch.int32).reshape(32, 1),
        start_pos=torch.tensor([65] + [-1] * 31),
        page_table=first,
        page_tables_per_layer=[first, second],
        kv_cache=object(),
        sampling_params=_greedy(32),
        slot_remap=None,
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )

    tokens, positions, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (1, 1)
    assert positions.tolist() == [65]
    assert call["page_table"].shape == (1, 16)
    assert [table.shape for table in adapter.model.page_table_updates[-1]] == [
        (1, 16),
        (1, 16),
    ]
    assert call["force_host_tokens"] is False
    assert adapter._active_decode_bucket == 1
    assert adapter.model.decode_batch_sizes[-1] == 1


def test_slot_remap_and_host_sampling_keep_the_full_decode_width():
    adapter = _adapter()
    page_table = torch.zeros(32, 16, dtype=torch.int32)
    positions = torch.tensor([65] + [-1] * 31)

    for sampling_params, slot_remap in (
        (_greedy(32), torch.arange(32, dtype=torch.int32)),
        (None, None),
    ):
        adapter.decode_forward(
            tokens=torch.zeros(32, 1, dtype=torch.int32),
            start_pos=positions,
            page_table=page_table,
            page_tables_per_layer=[page_table, page_table],
            kv_cache=object(),
            sampling_params=sampling_params,
            slot_remap=slot_remap,
            enable_trace=True,
            read_from_device=False,
        )
        tokens, _, call = adapter.generator.decode_calls[-1]
        assert tokens.shape == (32, 1)
        assert call["page_table"].shape == (32, 16)


def test_decode_trace_bucket_switches_only_at_drained_reset_boundaries():
    adapter = _adapter()
    page_table = torch.zeros(32, 16, dtype=torch.int32)
    singleton = torch.tensor([65] + [-1] * 31)
    batched = torch.tensor([66, 40] + [-1] * 30)
    common = {
        "tokens": torch.zeros(32, 1, dtype=torch.int32),
        "page_table": page_table,
        "page_tables_per_layer": [page_table, page_table],
        "kv_cache": object(),
        "sampling_params": _greedy(32),
        "enable_trace": True,
        "read_from_device": False,
    }

    adapter.decode_forward(start_pos=singleton, reset_batch=True, **common)
    assert adapter._active_decode_bucket == 1

    # Admission without the plugin's reset/drain signal must keep B1 sticky;
    # the newly visible row is consumed only after the safe boundary below.
    adapter.decode_forward(start_pos=batched, reset_batch=False, **common)
    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (1, 1)
    assert adapter._active_decode_bucket == 1
    assert call["force_host_tokens"] is False

    adapter.decode_forward(start_pos=batched, reset_batch=True, **common)
    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (32, 1)
    assert adapter._active_decode_bucket == 32
    assert call["reset_batch"] is True
    assert call["force_host_tokens"] is True
    assert call["reuse_sampling_state"] is False

    # Condensation can expose a singleton before the next reset; B32 remains
    # active so stale B1 token/current-position/page-table state cannot replay.
    adapter.decode_forward(start_pos=singleton, reset_batch=False, **common)
    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (32, 1)
    assert adapter._active_decode_bucket == 32
    assert call["force_host_tokens"] is False

    adapter.decode_forward(start_pos=singleton, reset_batch=True, **common)
    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (1, 1)
    assert adapter._active_decode_bucket == 1
    assert call["force_host_tokens"] is True


def test_removal_only_reset_keeps_surviving_request_in_active_decode_bucket():
    adapter = _adapter()
    page_table = torch.zeros(32, 16, dtype=torch.int32)
    common = {
        "tokens": torch.zeros(32, 1, dtype=torch.int32),
        "page_table": page_table,
        "page_tables_per_layer": [page_table, page_table],
        "kv_cache": object(),
        "sampling_params": _greedy(32),
        "enable_trace": True,
        "read_from_device": False,
    }

    adapter.decode_forward(
        start_pos=torch.tensor([65, 40] + [-1] * 30),
        reset_batch=True,
        **common,
    )
    assert adapter._active_decode_bucket == 32

    adapter.decode_forward(
        start_pos=torch.tensor([66] + [-1] * 31),
        reset_batch=True,
        removal_only_reset=True,
        slot_remap=None,
        **common,
    )

    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (32, 1)
    assert adapter._active_decode_bucket == 32
    assert "removal_only_reset" not in call
    assert call["slot_remap"] is None


def test_removal_only_reset_follows_a_narrower_plugin_wire_width():
    """32 -> 8: the plugin pads the shrinking batch to the next declared bucket.

    The wire tensors arrive 8 wide, so the adapter must switch to the B8 trace
    (the B32 persistent inputs cannot take 8-row host tensors).
    """
    adapter = _adapter()
    wide_table = torch.zeros(32, 16, dtype=torch.int32)
    adapter.decode_forward(
        tokens=torch.zeros(32, 1, dtype=torch.int32),
        start_pos=torch.tensor([65, 40, 12, 9, 8, 7] + [-1] * 26),
        page_table=wide_table,
        page_tables_per_layer=[wide_table, wide_table],
        kv_cache=object(),
        sampling_params=_greedy(32),
        enable_trace=True,
        read_from_device=False,
        reset_batch=True,
    )
    assert adapter._active_decode_bucket == 32

    narrow_table = torch.zeros(8, 16, dtype=torch.int32)
    adapter.decode_forward(
        tokens=torch.zeros(8, 1, dtype=torch.int32),
        start_pos=torch.tensor([66, 41, 13, 10, 9] + [-1] * 3),
        page_table=narrow_table,
        page_tables_per_layer=[narrow_table, narrow_table],
        kv_cache=object(),
        sampling_params=_greedy(8),
        enable_trace=True,
        read_from_device=False,
        reset_batch=True,
        removal_only_reset=True,
        slot_remap=None,
    )
    tokens, _, call = adapter.generator.decode_calls[-1]
    assert tokens.shape == (8, 1)
    assert adapter._active_decode_bucket == 8
    assert call["force_host_tokens"] is True


def test_decode_bucket_warmup_compiles_before_capturing(monkeypatch):
    adapter = _adapter()
    monkeypatch.setattr("ttnn.synchronize_device", lambda *_: None)

    result = adapter.warmup_model_decode(
        kv_cache=object(),
        enable_trace=True,
        max_batch_size=32,
        num_blocks=16,
        can_sample_on_device=True,
    )

    assert result == "decode-warmup"
    assert [
        (
            call["max_batch_size"],
            call["enable_trace"],
            call.get("skip_trace_precompile", False),
        )
        for call in adapter.generator.decode_warmup_calls
    ] == [
        (1, False, False),
        (4, False, False),
        (8, False, False),
        (32, False, False),
        (1, True, True),
        (4, True, True),
        (8, True, True),
        (32, True, True),
    ]
    assert set(adapter.generator._inner._bucket_trace_store) == {1, 4, 8, 32}
    assert [call["max_batch_size"] for call in adapter.generator.decode_prepare_calls] == [1, 4, 8, 32]
    assert [call["width"] for call in adapter.generator.decode_capture_calls] == [1, 4, 8, 32]
    assert adapter._active_decode_bucket == 32
    assert adapter._device_trace_recapture_requires_reset


def test_decode_bucket_activation_toggles_every_decode_norm():
    class _RopeSetup:
        cos_matrix = object()
        sin_matrix = object()

        @staticmethod
        def get_both_trans_mats():
            return {"decode": object()}

    model = Model.__new__(Model)
    model.norm = SimpleNamespace(enable_decode_sharding=False)
    model.layers = []
    model._decode_rope_setups = {width: _RopeSetup() for width in (1, 32)}
    model._decode_layer_transformation_mats = {width: [object(), object()] for width in (1, 32)}
    model._decode_layer_kv_memory_configs = {width: [object(), object()] for width in (1, 32)}
    for _ in range(2):
        decoder = SimpleNamespace(
            input_layernorm=SimpleNamespace(enable_decode_sharding=False),
            post_attention_layernorm=SimpleNamespace(enable_decode_sharding=False),
        )
        model.layers.append(
            SimpleNamespace(
                decoder=decoder,
                self_attn=SimpleNamespace(transformation_mats={"decode": None}, kv_mem_cfg=None),
            )
        )

    Model.activate_decode_batch_size(model, 1)
    assert model.norm.enable_decode_sharding
    assert all(layer.decoder.input_layernorm.enable_decode_sharding for layer in model.layers)
    assert all(layer.decoder.post_attention_layernorm.enable_decode_sharding for layer in model.layers)

    model.norm.enable_decode_sharding = False
    Model.activate_decode_batch_size(model, 32)
    assert model.norm.enable_decode_sharding
    assert all(layer.decoder.input_layernorm.enable_decode_sharding for layer in model.layers)
    assert all(layer.decoder.post_attention_layernorm.enable_decode_sharding for layer in model.layers)


def test_decode_trace_warmup_excludes_the_duplicate_host_model_trace():
    class _Inner:
        def __init__(self):
            self.calls = []

        def warmup_model_decode(self, **kwargs):
            self.calls.append(kwargs)
            return "decode-warmup"

    generator = Generator.__new__(Generator)
    generator._inner = _Inner()
    generator._outer_cache = lambda cache: cache

    for enable_trace in (False, True):
        assert (
            generator.warmup_model_decode(
                kv_cache=object(),
                enable_trace=enable_trace,
                max_batch_size=32,
                num_blocks=2048,
                can_sample_on_device=True,
            )
            == "decode-warmup"
        )

    assert generator._inner.calls[0]["include_host_sampling"] is True
    assert generator._inner.calls[1]["include_host_sampling"] is False
    assert generator._inner.calls[1]["skip_trace_precompile"] is False


def test_prefill_compile_release_preserves_decode_bucket_namespaces(monkeypatch):
    bucket_one = ({}, {True: object()}, {True: object()})
    bucket_thirty_two = ({}, {True: object()}, {True: object()})
    inner = SimpleNamespace(
        model=[],
        model_args=[],
        _bucket_trace_store={1: bucket_one, 32: bucket_thirty_two},
        trace_ids_decode=bucket_one[0],
        trace_inputs_decode=bucket_one[1],
        trace_output_decode=bucket_one[2],
        mode=object(),
        prev_page_table=object(),
        _prev_on_device_sampling=True,
        _slots_prefilled_since_decode={0},
        _defer_trace_recording=True,
        _pending_decode_trace=object(),
    )
    generator = Generator.__new__(Generator)
    generator._inner = inner
    generator.mesh_device = object()
    generator._prepared_device_sampling_params = object()
    generator.trace_evidence = SimpleNamespace(decode_trace_releases_for_prefill_compile=0)
    generator._lifetime_decode_trace_releases_for_prefill_compile = 0
    monkeypatch.setattr("ttnn.synchronize_device", lambda *_: None)

    generator._release_decode_traces_for_prefill_compile()

    assert inner._bucket_trace_store == {1: bucket_one, 32: bucket_thirty_two}
    assert inner.trace_ids_decode is bucket_one[0]
    assert all(not store for bucket in inner._bucket_trace_store.values() for store in bucket)


def test_row_sharded_trace_tokens_are_reconstructed_for_condense():
    expected = torch.arange(32)
    shards = list(torch.chunk(expected, 4))
    assert torch.equal(assemble_decode_trace_vector(shards, 32), expected)

    replicated = [expected.clone() for _ in range(4)]
    assert torch.equal(assemble_decode_trace_vector(replicated, 32), expected)
    assert assemble_decode_trace_vector(shards[:3], 32) is None


def test_penalties_or_seed_disable_fixed_sampling_reuse():
    neutral = _greedy()
    assert TTGptOssForCausalLM._sampling_state_reusable(neutral)
    assert _sampling_key(neutral) == _sampling_key(_greedy())
    for field, value in (
        ("seed", [123, None]),
        ("presence_penalty", [0.1, 0.0]),
        ("frequency_penalty", [0.0, 0.1]),
        ("repetition_penalty", [1.0, 1.1]),
        ("enable_log_probs", [True, False]),
    ):
        changed = {**vars(neutral), field: value}
        assert not TTGptOssForCausalLM._sampling_state_reusable(SamplingParams(**changed))


def test_sequential_hybrid_prefill_slices_each_unique_table_by_request_row():
    first = torch.tensor([[10, 100], [11, 101]], dtype=torch.int32)
    second = torch.tensor([[20, 200], [21, 201]], dtype=torch.int32)
    tables = [first, second, first]

    sliced = slice_prefill_page_tables_per_layer(tables, 1, False)

    assert sliced[0].tolist() == [[11, 101]]
    assert sliced[1].tolist() == [[21, 201]]
    assert sliced[2] is sliced[0]
    assert slice_prefill_page_tables_per_layer(tables, 1, True) is tables
    assert first.tolist() == [[10, 100], [11, 101]]
    assert second.tolist() == [[20, 200], [21, 201]]


def test_sequential_hybrid_prefill_ignores_unowned_stale_table_tail():
    first = torch.tensor([[137, 5, 99], [211, 6, 100]], dtype=torch.int32)
    second = torch.tensor([[138, 7, 101], [212, 8, 102]], dtype=torch.int32)

    sliced = slice_prefill_page_tables_per_layer(
        [first, second, first],
        0,
        False,
        valid_seq_len=3,
        block_sizes=[64, 64, 64],
    )

    assert [table.tolist() for table in sliced] == [[[137]], [[138]], [[137]]]
    assert sliced[2] is sliced[0]


def test_fresh_prefill_slots_map_to_post_remap_decode_rows():
    remap = torch.tensor([3, 0, 1, 2], dtype=torch.int32)

    assert prefilled_decode_rows({3}, remap, chunk_index=0, chunk_size=4) == {0}
    assert prefilled_decode_rows({3}, None, chunk_index=0, chunk_size=4) == {3}


def test_production_model_installs_sampling_policy_before_base_sampler_construction():
    source = inspect.getsource(Model.__init__)
    assert source.index("self.args = args") < source.index("super().__init__(")
    args_source = inspect.getsource(FullModelArgs.__init__)
    assert "self.padded_vocab_size = compute_per_device_vocab" in args_source


def test_adapter_has_no_independent_token_sampling_or_feedback_loop():
    source = Path(inspect.getsourcefile(TTGptOssForCausalLM))
    text = source.read_text(encoding="utf-8")
    assert ".argmax(" not in text
    assert "torch.topk(" not in text
    assert "ttnn.topk(" not in text
    assert "Generator(" in text
    assert 'cache_owner="vllm"' in text


def test_cold_ring_resume_near_the_context_end_keeps_the_padded_chunk_inside_it():
    adapter = _adapter()
    start, fill_start, cold = adapter._resume_plan(130560, 3, 130816)
    assert (start, fill_start, cold) == (126976, 130560, True)
    assert start + 4096 == HF_CONTEXT_LENGTH
    start, fill_start, cold = adapter._resume_plan(130560, 3, 131072)
    assert start + 4096 <= HF_CONTEXT_LENGTH and start % 512 == 0 and cold
    adapter._slot_prefill_end[3] = 122880
    assert adapter._resume_plan(122880, 3, 130816) == (122880, 122880, False)
    assert adapter._resume_plan(8192, 3, 16384) == (8192 - 2560, 8192, True)
