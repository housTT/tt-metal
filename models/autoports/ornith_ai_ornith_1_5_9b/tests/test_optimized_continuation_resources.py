# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Borrowed input ownership across single-token public prefill continuation."""

import pytest
import torch

import ttnn

from . import test_functional_decoder as H
from .test_optimized_decoder import optimized_path  # noqa: F401 (autouse contract fixture)

pytestmark = H.pytestmark


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_continuation_borrowed_input(mesh_device, layer_idx):
    host = H.make_activations(1, 128, seed=1021)
    token_host = H.make_activations(1, 1, seed=1022)
    golden, reference_decode = H.run_reference(layer_idx, "real", host, decode_x=[token_host], decode_steps=1)
    decoder, table, _ = H.build_decoder(mesh_device, layer_idx, "real", max_context=256)
    prefix = H.to_device(mesh_device, host[:, :127])
    continuation = H.to_device(mesh_device, host[:, 127:])
    addresses = prefix.buffer_address(), continuation.buffer_address()
    first = decoder.prefill_forward(prefix, page_table=table)
    second = decoder.prefill_forward(continuation, start_pos=127, page_table=table)
    actual = torch.cat([ttnn.to_torch(first), ttnn.to_torch(second)], dim=1)
    assert H.pcc(golden, actual) >= H.PCC_BAR
    assert addresses == (prefix.buffer_address(), continuation.buffer_address())
    assert torch.equal(ttnn.to_torch(prefix), host[:, :127])
    assert torch.equal(ttnn.to_torch(continuation), host[:, 127:])
    assert list(second.shape) == [1, 1, H.hf_config().hidden_size]
    assert second.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    ttnn.deallocate(first)
    ttnn.deallocate(second)

    pos, rot = H.decode_inputs(mesh_device, torch.tensor([128]))
    token = H.to_device(mesh_device, token_host)
    out = decoder.decode_forward(token, current_pos=pos, rot_idxs=rot, page_table=table)
    assert H.pcc(reference_decode[0], ttnn.to_torch(out)) >= H.PCC_BAR
    for tensor in (out, token, pos, rot, prefix, continuation):
        ttnn.deallocate(tensor)
