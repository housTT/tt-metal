# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host contract tests; device stale-state proof lives in the stage probe."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator_vllm import TTOrnithForCausalLM
from models.common.sampling import SamplingParams


@pytest.fixture
def adapter():
    model = SimpleNamespace(mesh_device=Mock(), page_block_size=64)
    obj = TTOrnithForCausalLM(model, max_batch_size=4, max_model_len=262144, hf_config=SimpleNamespace())
    obj.generator = Mock(kv_cache=object(), counters={})
    obj._device_rows[:] = True
    obj._last_device_sampling = True
    yield obj
    obj.teardown()


def test_steady_async_ignores_stale_host_tokens_and_positions(adapter):
    table = torch.zeros(4, 4096, dtype=torch.int32)
    adapter.decode_forward(
        tokens=torch.full((4, 1), 999),
        start_pos=torch.full((4,), 123),
        page_table=table,
        kv_cache=adapter.generator.kv_cache,
        sampling_params=object(),
        read_from_device=False,
    )
    adapter.generator.refresh_serving_inputs.assert_not_called()
    adapter.generator.configure_sampling.assert_not_called()
    call = adapter.generator.decode_forward.call_args
    assert call.args == (None, None)
    assert call.kwargs["page_table"] is table
    assert call.kwargs["sample_on_device"] is True
    assert call.kwargs["read_from_device"] is False


def test_new_prefill_refreshes_only_its_row(adapter):
    adapter._prefilled_rows[2] = True
    adapter.decode_forward(
        tokens=torch.zeros(4, 1),
        start_pos=torch.tensor([10, 20, 131, -1]),
        page_table=torch.zeros(4, 4096, dtype=torch.int32),
        kv_cache=adapter.generator.kv_cache,
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
        reset_batch=True,
        read_from_device=False,
    )
    mask = adapter.generator.refresh_serving_inputs.call_args.args[2]
    assert mask.tolist() == [False, False, True, False]


def test_prefill_preserves_logical_nonaligned_length_and_slot(adapter):
    adapter.prefill_forward(
        tokens=torch.ones(1, 131, dtype=torch.int32),
        prompt_lens=[131],
        empty_slots=[2],
        page_table=torch.tensor([[9, 8, 7]], dtype=torch.int32),
        kv_cache=adapter.generator.kv_cache,
        sampling_params=SamplingParams(temperature=0.0, top_k=1, top_p=1.0),
    )
    kwargs = adapter.generator.prefill_forward.call_args.kwargs
    assert kwargs["prompt_lens"] == [131]
    assert kwargs["slots"] == [2]
    assert kwargs["page_table"][2, :3].tolist() == [9, 8, 7]
    assert kwargs["kv_cache"] is adapter.generator.kv_cache


def test_cache_identity_is_required(adapter):
    with pytest.raises(ValueError, match="exact vLLM cache"):  # allow-pytest.raises: CPU-only error.
        adapter.decode_forward(tokens=None, start_pos=None, page_table=None, kv_cache=object())


def test_host_sampling_requires_explicit_compatibility(adapter):
    with pytest.raises(ValueError, match="ORNITH_VLLM_ALLOW_HOST_SAMPLING"):  # allow-pytest.raises: CPU-only error.
        adapter._device_sampling(None)
    adapter.allow_host_sampling = True
    assert adapter._device_sampling(None) is False
