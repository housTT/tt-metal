# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compact residual resources, batch policy, and exact restored trace coverage."""

import json

import pytest
import torch

import ttnn

from . import test_functional_decoder as H
from .test_optimized_decoder import optimized_path  # noqa: F401 (autouse contract fixture)
from .test_optimized_trace_regression import all_equal, differences

pytestmark = H.pytestmark


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", [1, 8, 13, 16, 17, 24, 31])
def test_batch_boundary_hf(mesh_device, layer_idx, batch):
    H.test_batched_prefill_decode_pcc(mesh_device, layer_idx, batch)


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", [4, 8, 12, 13, 16, 32])
@pytest.mark.parametrize("public_memory", ["dram", "sharded"])
def test_batch_restored_trace(mesh_device, layer_idx, batch, public_memory):
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real", batch=batch, max_context=256)
    host_prefix = H.make_activations(batch, 63, seed=101)
    host_token = H.make_activations(batch, 1, seed=103)
    positions = torch.full((batch,), 63)
    with torch.no_grad():
        golden_prefix, reference_state = H.R.reference_prefill(
            H.reference_layer(layer_idx, "real"), H.hf_config(), host_prefix.float(), start_pos=0
        )
        golden_token = H.R.reference_decode(
            H.reference_layer(layer_idx, "real"), H.hf_config(), host_token.float(), positions, reference_state
        )
    prefix = H.to_device(mesh_device, host_prefix)
    out = decoder.prefill_forward(prefix, page_table=table)
    actual_prefix = ttnn.to_torch(out)
    prefill_pcc = [H.pcc(golden_prefix[user], actual_prefix[user]) for user in range(batch)]
    assert min(prefill_pcc) >= H.PCC_BAR
    ttnn.deallocate(out)
    token = H.to_device(mesh_device, host_token)
    if public_memory == "sharded":
        token = ttnn.to_memory_config(token, decoder._width_memory(decoder.cfg.dim, 32, batch * 32))
    pos, rot = H.decode_inputs(mesh_device, torch.full((batch,), 63))
    immutable = {"token": ttnn.to_torch(token), "prefix": ttnn.to_torch(prefix)}
    saved = H._snapshot_state(decoder)

    def state_buffers():
        return (
            [decoder.k_cache, decoder.v_cache]
            if decoder.is_full_attention
            else [decoder.recurrent_state, *decoder.conv_state]
        )

    buffers = state_buffers()
    addresses = [buf.buffer_address() for buf in buffers]

    def restore():
        H._restore_state(decoder, saved)
        assert all_equal(differences(saved, H._snapshot_state(decoder)))
        assert addresses == [buf.buffer_address() for buf in state_buffers()]

    def forward():
        return decoder.decode_forward(token, current_pos=pos, rot_idxs=rot, page_table=table)

    out = forward()
    eager, state = ttnn.to_torch(out), H._snapshot_state(decoder)
    decode_pcc = [H.pcc(golden_token[user], eager[user]) for user in range(batch)]
    assert min(decode_pcc) >= H.PCC_BAR
    ttnn.deallocate(out)
    restore()
    out = forward()
    record = {
        "eager": differences(eager, ttnn.to_torch(out)),
        "eager_state": differences(state, H._snapshot_state(decoder)),
    }
    ttnn.deallocate(out)
    restore()
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    try:
        for i in range(4):
            restore()
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            record[f"trace_{i}"] = differences(eager, ttnn.to_torch(out))
            record[f"state_{i}"] = differences(state, H._snapshot_state(decoder))
        restore()
        for _ in range(32):
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        restore()
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        record["poststress"] = differences(eager, ttnn.to_torch(out))
        record["poststress_state"] = differences(state, H._snapshot_state(decoder))
        record["immutable"] = differences(immutable, {"token": ttnn.to_torch(token), "prefix": ttnn.to_torch(prefix)})
        print(
            "BATCH_TRACE "
            + json.dumps(
                {
                    "batch": batch,
                    "min_prefill_pcc": min(prefill_pcc),
                    "min_decode_pcc": min(decode_pcc),
                    "kind": decoder.kind,
                    "public_memory": public_memory,
                    "state_addresses": addresses,
                    "state_memory": str(buffers[0].memory_config()),
                    "l1_intermediates": getattr(decoder, "recurrent_l1_intermediates", False),
                    **record,
                }
            ),
            flush=True,
        )
        assert all_equal(record)
    finally:
        ttnn.release_trace(mesh_device, trace)


@pytest.mark.parametrize("layer_idx", [H.FULL_LAYER], ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", [17, 19, 23, 29, 31])
def test_prime_batch_rope_hf(mesh_device, layer_idx, batch):
    H.test_batched_prefill_decode_pcc(mesh_device, layer_idx, batch)


@pytest.mark.parametrize("layer_idx", [H.FULL_LAYER], ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", [17, 19, 23, 29, 31])
@pytest.mark.parametrize("public_memory", ["dram", "sharded"])
def test_prime_batch_rope_trace(mesh_device, layer_idx, batch, public_memory):
    test_batch_restored_trace(mesh_device, layer_idx, batch, public_memory)


@pytest.mark.parametrize("layer_idx", [H.FULL_LAYER], ids=lambda i: H.LAYER_IDS[i])
@pytest.mark.parametrize("batch", range(1, 33))
def test_all_full_batches(mesh_device, layer_idx, batch):
    test_batch_restored_trace(mesh_device, layer_idx, batch, "dram")
