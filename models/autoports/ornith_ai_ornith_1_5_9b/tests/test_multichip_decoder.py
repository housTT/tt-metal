# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the unchanged logical/cache/HF contract on the four-chip TP decoder.

The host adapter checks all replicated output ranks. Persistent-state snapshots
stay on-device, preserving distinct local heads instead of broadcasting rank0.
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder, fabric_router_config
from . import test_functional_decoder as H
from .test_contract_extensions import (
    test_trace_changed_page_table,
    test_trace_repeated_identical_state,
    test_unaligned_continuation_to_capacity,
)
from .test_functional_decoder import (
    test_batched_decode_ragged_positions,
    test_batched_prefill_decode_pcc,
    test_decode_pcc,
    test_determinism_repeated_inputs,
    test_forward_with_poisoned_free_pool,
    test_full_context_chunk_size_invariance,
    test_full_context_prefill_and_decode,
    test_long_context_pcc,
    test_no_host_fallback_in_forward,
    test_permuted_page_table,
    test_prefill_continuation,
    test_prefill_pcc,
    test_real_weights_pcc,
    test_traced_decode_pcc,
    test_unaligned_max_context,
)
from .test_optimization_experiments import recorded_activations

__all__ = [
    "test_trace_changed_page_table",
    "test_trace_repeated_identical_state",
    "test_unaligned_continuation_to_capacity",
    "test_batched_decode_ragged_positions",
    "test_batched_prefill_decode_pcc",
    "test_decode_pcc",
    "test_determinism_repeated_inputs",
    "test_forward_with_poisoned_free_pool",
    "test_full_context_chunk_size_invariance",
    "test_full_context_prefill_and_decode",
    "test_long_context_pcc",
    "test_no_host_fallback_in_forward",
    "test_permuted_page_table",
    "test_prefill_continuation",
    "test_prefill_pcc",
    "test_real_weights_pcc",
    "test_traced_decode_pcc",
    "test_unaligned_max_context",
]

pytestmark = [
    pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "trace_region_size": 32 * 1024 * 1024,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": fabric_router_config(),
            }
        ],
        indirect=True,
    ),
]


def state_buffers(decoder):
    return (
        [decoder.k_cache, decoder.v_cache]
        if decoder.is_full_attention
        else [decoder.recurrent_state, *decoder.conv_state]
    )


@pytest.fixture(autouse=True)
def multichip_contract(monkeypatch, request):
    monkeypatch.setenv("ORNITH_WEIGHTS", "real")
    decoder_class = MultichipDecoder
    candidate = os.environ.get("ORNITH_MULTICHIP_CANDIDATE", "default")
    if candidate != "default":
        from .optimized_multichip_candidates import CANDIDATES

        decoder_class = CANDIDATES[candidate]
    monkeypatch.setattr(H, "FunctionalDecoder", decoder_class)
    layer = getattr(request.node, "callspec", None)
    layer = layer.params.get("layer_idx", H.FULL_LAYER) if layer else H.FULL_LAYER
    source = recorded_activations(layer)[0]

    def real_inputs(batch, seq_len, *, seed=0):
        indices = (torch.arange(seq_len)[None, :] + seed + torch.arange(batch)[:, None] * 137) % source.shape[0]
        return source[indices].clone()

    monkeypatch.setattr(H, "make_activations", real_inputs)
    original_read = ttnn.to_torch

    def read(value, *args, **kwargs):
        if kwargs.get("mesh_composer") is not None:
            return original_read(value, *args, **kwargs)
        parts = ttnn.get_device_tensors(value)
        result = [original_read(part, *args, **kwargs) for part in parts]
        # Public residuals are replicated. State/head-shaped tensors are local
        # and deliberately have different values on each rank.
        if len(result) == 4 and value.shape[-1] == 4096 and len(value.shape) == 3:
            if not all(torch.equal(result[0], part) for part in result[1:]):
                from .multichip_long_diagnostic import summarize

                artifact = (
                    Path(__file__).resolve().parents[1] / "doc/multichip_decoder" / f"replica_failure_{time.time_ns()}"
                )
                torch.save(result, artifact.with_suffix(".pt"))
                artifact.with_suffix(".json").write_text(json.dumps(summarize(result), indent=2) + "\n")
                pytest.fail(f"residual replicas differ; raw evidence: {artifact}")
        return result[0]

    def snapshot(decoder):
        return [ttnn.clone(buf) for buf in state_buffers(decoder)]

    def restore(decoder, saved):
        for source, destination in zip(saved, state_buffers(decoder)):
            ttnn.copy(source, destination)

    monkeypatch.setattr(ttnn, "to_torch", read)
    monkeypatch.setattr(H, "_snapshot_state", snapshot)
    monkeypatch.setattr(H, "_restore_state", restore)


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 2, 3, 7, 8, 12, 13, 16, 17, 24, 31])
def test_local_head_batch_boundaries(mesh_device, layer_idx, batch):
    H.test_batched_prefill_decode_pcc(mesh_device, layer_idx, batch)
