# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace input ownership and optional YaRN contracts for the actual 9B decoder."""

import copy

import pytest
import torch
from loguru import logger

import ttnn

from ..tt.model_config import OrnithDecoderConfig
from ..tt.rope import OrnithRope
from . import test_functional_decoder as H

pytestmark = H.pytestmark


def test_trace_changed_page_table(mesh_device):
    """One captured graph switches between disjoint requests via its stable page table."""
    cfg = H.hf_config()
    source = H.default_weight_source()
    blocks = H.num_blocks_for_context(1024)
    decoder, table, _ = H.build_decoder(mesh_device, H.FULL_LAYER, source, max_context=1024, num_blocks=blocks * 2)
    # Both tables have identical shape and entirely disjoint nonidentity physical pages.
    rng = torch.Generator().manual_seed(881)
    pages = torch.randperm(blocks * 2, generator=rng).to(torch.int32)
    tables = [pages[i * blocks : (i + 1) * blocks].reshape(1, blocks) for i in range(2)]
    table = H.to_device(mesh_device, tables[0], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    lengths = [63, 129]
    decode_x = [H.make_activations(1, 1, seed=883 + i) for i in range(2)]
    goldens = []
    for i, length in enumerate(lengths):
        prompt = H.make_activations(1, length, seed=887 + i)
        _, ref_decode = H.run_reference(H.FULL_LAYER, source, prompt, decode_x=[decode_x[i]], decode_steps=1)
        goldens.append(ref_decode[0])
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(tables[i], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), table
        )
        ttnn.deallocate(decoder.prefill_forward(H.to_device(mesh_device, prompt), page_table=table))
    x_buf = H.to_device(mesh_device, decode_x[0])
    pos_buf, rot_buf = H.decode_inputs(mesh_device, torch.tensor([lengths[0]]))

    def refresh(i):
        for host, target, dtype, layout in (
            (decode_x[i], x_buf, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            (torch.tensor([lengths[i]], dtype=torch.int32), pos_buf, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            (torch.tensor([[lengths[i]]], dtype=torch.int32), rot_buf, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            (tables[i], table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
        ):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dtype, layout=layout), target)

    def forward():
        return decoder.decode_forward(x_buf, current_pos=pos_buf, rot_idxs=rot_buf, page_table=table)

    refresh(0)
    ttnn.deallocate(forward())
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    results = []
    for i in [0, 1, 0]:
        refresh(i)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        actual = ttnn.to_torch(out)
        value = H.pcc(goldens[i], actual)
        logger.info(f"changed-page-table replay request={i} position={lengths[i]} PCC={value:.6f}")
        assert value >= H.PCC_BAR
        results.append(actual)
    assert torch.equal(results[0], results[2])
    assert not torch.equal(results[0], results[1])
    ttnn.release_trace(mesh_device, trace)


@pytest.mark.parametrize("layer_idx", H.LAYERS, ids=lambda i: H.LAYER_IDS[i])
def test_trace_repeated_identical_state(mesh_device, layer_idx):
    decoder, page_table, _ = H.build_decoder(mesh_device, layer_idx, H.default_weight_source())
    ttnn.deallocate(
        decoder.prefill_forward(H.to_device(mesh_device, H.make_activations(1, 63, seed=901)), page_table=page_table)
    )
    x = H.to_device(mesh_device, H.make_activations(1, 1, seed=902))
    pos, rot = H.decode_inputs(mesh_device, torch.tensor([63]))
    saved = H._snapshot_state(decoder)

    def forward():
        return decoder.decode_forward(x, current_pos=pos, rot_idxs=rot, page_table=page_table)

    ttnn.deallocate(forward())
    H._restore_state(decoder, saved)
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = forward()
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    results = []
    for _ in range(3):
        H._restore_state(decoder, saved)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        results.append(ttnn.to_torch(out))
    assert all(torch.equal(results[0], item) for item in results[1:])
    logger.info(f"trace determinism layer={layer_idx}: three identical outputs from identical restored state")
    ttnn.release_trace(mesh_device, trace)


def test_yarn_million_rope(mesh_device):
    cfg = copy.deepcopy(H.hf_config())
    native = cfg.max_position_embeddings
    cfg.max_position_embeddings = 1_000_000
    cfg.rope_parameters = dict(
        cfg.rope_parameters, rope_type="yarn", factor=1_000_000 / native, original_max_position_embeddings=native
    )
    parsed = OrnithDecoderConfig.from_hf_config(cfg)
    rope = OrnithRope(mesh_device, parsed, max_context=cfg.max_position_embeddings)
    positions = torch.tensor([[0, native - 1, native, 999_999]], dtype=torch.int32)
    cos_ref, sin_ref = H.R.reference_position_embeddings(cfg, positions.long())
    indices = H.to_device(mesh_device, positions, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = rope.decode_forward(indices)
    for name, ref, actual in (("cos", cos_ref, cos), ("sin", sin_ref, sin)):
        actual = ttnn.to_torch(actual)
        value = H.pcc(ref, actual)
        assert value > 0.99999
        assert torch.allclose(ref, actual.float(), atol=0.005, rtol=0.005)
        logger.info(f"YaRN million-token RoPE {name} PCC={value:.8f}")


@pytest.mark.long
@pytest.mark.timeout(1800)
def test_native_context_decode_oracle(mesh_device):
    """HF vs trace at position 262143 over an exact-shape permuted BF16 cache.

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
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(physical, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), target)
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


def test_unaligned_continuation_to_capacity(mesh_device):
    """Continue an unaligned prompt through the final legal cache page."""
    context, split = 4096, 63
    source = H.default_weight_source()
    x = H.make_activations(1, context, seed=982)
    golden, _ = H.run_reference(H.FULL_LAYER, source, x)
    decoder, table, _ = H.build_decoder(mesh_device, H.FULL_LAYER, source, max_context=context)
    first = decoder.prefill_forward(H.to_device(mesh_device, x[:, :split]), page_table=table)
    second = decoder.prefill_forward(H.to_device(mesh_device, x[:, split:]), start_pos=split, page_table=table)
    actual = torch.cat([ttnn.to_torch(first), ttnn.to_torch(second)], dim=1)
    value = H.pcc(golden, actual)
    logger.info(f"unaligned continuation to capacity context={context} split={split} PCC={value:.8f}")
    assert value >= H.PCC_BAR
