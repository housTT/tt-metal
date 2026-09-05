# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the complete prior decoder contract against the optimized decoder.

The autouse fixture binds the shared harness to the optimized class, checks every
constructed instance, and rejects use of the functional block implementation.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from ..tt.fused_decoder import FusedDecoder as OriginalFusedDecoder
from . import test_functional_decoder as H
from .optimization_candidates import selected_candidate
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
    test_perf_decode_traced,
    test_perf_prefill,
    test_permuted_page_table,
    test_prefill_continuation,
    test_prefill_pcc,
    test_real_weights_pcc,
    test_traced_decode_pcc,
    test_unaligned_max_context,
)
from .test_optimization_experiments import recorded_activations

__all__ = [
    "H",
    "test_prefill_pcc",
    "test_decode_pcc",
    "test_batched_prefill_decode_pcc",
    "test_batched_decode_ragged_positions",
    "test_permuted_page_table",
    "test_prefill_continuation",
    "test_unaligned_max_context",
    "test_determinism_repeated_inputs",
    "test_forward_with_poisoned_free_pool",
    "test_no_host_fallback_in_forward",
    "test_traced_decode_pcc",
    "test_real_weights_pcc",
    "test_synthetic_weights_diagnostic",
    "test_full_context_prefill_and_decode",
    "test_full_context_chunk_size_invariance",
    "test_long_context_pcc",
    "test_perf_prefill",
    "test_perf_decode_traced",
    "test_trace_changed_page_table",
    "test_trace_repeated_identical_state",
    "test_native_context_decode_oracle",
    "test_unaligned_continuation_to_capacity",
]


pytestmark = H.pytestmark


@pytest.fixture(autouse=True)
def optimized_path(monkeypatch, request):
    # This checkpoint-specific precision gate defaults to the actual weights.
    # The explicitly named synthetic diagnostic still requests its own source.
    monkeypatch.setenv("ORNITH_WEIGHTS", os.environ.get("ORNITH_WEIGHTS", "real"))
    FusedDecoder = selected_candidate()
    original_build = FusedDecoder.from_state_dict
    built = []
    calls = []
    original_block = FusedDecoder._block

    def optimized_block(self, *args, **kwargs):
        assert original_block is not OriginalFusedDecoder._block
        calls.append(kwargs["mode"])
        return original_block(self, *args, **kwargs)

    monkeypatch.setattr(FusedDecoder, "_block", optimized_block)
    if os.environ.get("ORNITH_OPT_INPUTS", "recorded") == "recorded" and "synthetic_weights" not in request.node.name:
        layer = getattr(request.node, "callspec", None)
        layer = layer.params.get("layer_idx", H.FULL_LAYER) if layer else H.FULL_LAYER
        source = recorded_activations(layer)[0]

        def real_inputs(batch, seq_len, *, seed=0):
            # Recorded rows are selected cyclically for long and batched contract
            # tests; they are not claimed to be an HF rollout of that long prefix.
            indices = (torch.arange(seq_len)[None, :] + seed + torch.arange(batch)[:, None] * 137) % source.shape[0]
            return source[indices].clone()

        monkeypatch.setattr(H, "make_activations", real_inputs)

    def build(cls, *args, **kwargs):
        decoder = original_build(*args, **kwargs)
        assert type(decoder) is FusedDecoder
        built.append(decoder.kind)
        return decoder

    def forbidden(*args, **kwargs):
        raise AssertionError("functional block fallback in optimized test")

    monkeypatch.setattr(H, "FunctionalDecoder", FusedDecoder)
    monkeypatch.setattr(FusedDecoder, "from_state_dict", classmethod(build))
    monkeypatch.setattr(FunctionalDecoder, "_block", forbidden)
    yield
    assert built, "test did not construct the optimized decoder"
    assert calls, "test did not execute the optimized block"


@pytest.mark.long
@pytest.mark.timeout(1800)
def test_native_context_decode_oracle(mesh_device):
    """HF vs trace at position 262143 over an exact-shape permuted cache in the selected dtype.

    Historical KV is a deterministic test fixture (not a claimed HF prefill).
    The query and full decoder use real layer-3 checkpoint weights. Together with
    full-length prefill and chunk invariance this independently checks rounded
    decode read-window coverage, page addressing, and high-position RoPE.
    """
    cfg = H.hf_config()
    context = cfg.max_position_embeddings
    heads, dim, page = cfg.num_key_value_heads, cfg.head_dim, 64
    blocks = H.num_blocks_for_context(context)
    decoder, _, _ = H.build_decoder(mesh_device, H.FULL_LAYER, "real", max_context=context)
    rng = torch.Generator().manual_seed(970)
    table_host = torch.randperm(blocks, generator=rng).to(torch.int32).reshape(1, blocks)
    table = H.to_device(mesh_device, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    history = []
    for target in (decoder.k_cache, decoder.v_cache):
        values = torch.randn(1, heads, context, dim, generator=rng).to(torch.bfloat16) * 0.125
        history.append(values[:, :, : context - 1].float())
        logical = values.reshape(heads, blocks, page, dim).permute(1, 0, 2, 3).contiguous()
        physical = torch.empty_like(logical)
        physical[table_host[0].long()] = logical
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(physical, dtype=target.dtype, layout=ttnn.TILE_LAYOUT), target)
        del values, logical, physical
    cache = H.R._new_cache(cfg)
    cache.update(history[0], history[1], H.FULL_LAYER)
    x = H.make_activations(1, 1, seed=971)
    position = torch.tensor([context - 1])
    with torch.no_grad():
        golden = H.R.reference_decode(H.reference_layer(H.FULL_LAYER, "real"), cfg, x.float(), position, cache)
    del history, cache
    x_buf = H.to_device(mesh_device, x)
    pos, rot = H.decode_inputs(mesh_device, position)

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos, rot_idxs=rot, page_table=table)

    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    actual = ttnn.to_torch(out)
    value = H.pcc(golden, actual)
    logger.info(f"native-context exact-cache HF/traced decode position={context - 1} PCC={value:.8f}")
    assert torch.isfinite(actual).all()
    assert value >= H.PCC_BAR
    ttnn.release_trace(mesh_device, trace)


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_synthetic_weights_diagnostic(mesh_device, layer_idx):
    """Report random-weight PCC; require finite, exactly replayable optimized output.

    The stage's explicit precision-selection contract uses real model weights.
    Synthetic random matrices have a different BFP4 error distribution and do
    not veto the real-weight winner. All real-weight PCC tests retain 0.995.
    """
    seq_len = 300
    prompt = H.make_activations(1, seq_len, seed=43)
    token = H.make_activations(1, 1, seed=44)
    golden, decoded = H.run_reference(layer_idx, "synthetic", prompt, decode_x=[token], decode_steps=1)
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "synthetic")
    out = decoder.prefill_forward(H.to_device(mesh_device, prompt), page_table=table)
    prefill = ttnn.to_torch(out)
    assert torch.isfinite(prefill).all()
    ttnn.deallocate(out)
    x = H.to_device(mesh_device, token)
    pos, rot = H.decode_inputs(mesh_device, torch.tensor([seq_len]))
    saved = H._snapshot_state(decoder)

    def forward():
        return decoder.decode_forward(x, current_pos=pos, rot_idxs=rot, page_table=table)

    out = forward()
    eager = ttnn.to_torch(out)
    assert torch.isfinite(eager).all()
    ttnn.deallocate(out)
    H._restore_state(decoder, saved)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for _ in range(3):
            H._restore_state(decoder, saved)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            assert torch.equal(eager, ttnn.to_torch(out))
    finally:
        ttnn.release_trace(mesh_device, trace)
    logger.info(
        f"SYNTHETIC DIAGNOSTIC layer={layer_idx} prefill PCC={H.pcc(golden,prefill):.8f} "
        f"decode PCC={H.pcc(decoded[0],eager):.8f}; finite/exact replay required; "
        "precision acceptance uses real-weight PCC>=0.995"
    )
