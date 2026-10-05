# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.reference.modernbert import ModernBertRotaryEmbedding
from models.autoports.convaiinnovations_laya.reference.modernbert import build_masks as torch_build_masks
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import pcc
from models.autoports.convaiinnovations_laya.tt.model_config import (
    ACTIVATIONS_DTYPE,
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    bucket_plan,
)
from models.autoports.convaiinnovations_laya.tt.modernbert_attention import TtnnModernBertAttention
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, deallocate_masks
from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, fold_q_scale, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
ATTENTION_PCC = 0.999
LAYER_FOR = {FULL_ATTENTION: 0, SLIDING_ATTENTION: 1}
BATCH, SEQ = 2, 512


@pytest.fixture(scope="module")
def tt_params(module_device, parts, laya_config):
    device = module_device
    params = prepare_weights(parts["encoder"], laya_config, device, layers=[0, 1])
    yield params
    deallocate_weights(params)


@pytest.fixture(scope="module")
def batch_inputs(torch_encoder):
    b = encoder_inputs(batch_size=BATCH, seq_len=SEQ, fill=True)
    with torch.no_grad():
        _, hidden = torch_encoder(b["input_ids"], b["attention_mask"], output_hidden_states=True)
    xs = {i: torch_encoder.layers[i].attn_norm(hidden[i]) for i in (0, 1)}
    return b, xs


@pytest.fixture(scope="module")
def tt_env(module_device, laya_config):
    device = module_device
    plan = bucket_plan(device, laya_config, BATCH, SEQ)
    rotary = TtnnModernBertRotary(laya_config, device, SEQ, batch_size=BATCH, attention_memory=plan.attention_memory)
    builder = TtnnMaskBuilder(laya_config, device, SEQ, BATCH)
    yield plan, rotary, builder
    rotary.deallocate()
    builder.deallocate()


def _torch_rope(config, layer_type, seq_len):
    hd = config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters[layer_type]["rope_theta"]
    return ModernBertRotaryEmbedding(hd, theta)(torch.arange(seq_len).unsqueeze(0), torch.float32)


def _reference(config, ref, layer_idx, layer_type, x, attention_mask):
    masks = torch_build_masks(config, attention_mask, SEQ, x.dtype, config.local_attention // 2)
    with torch.no_grad():
        return ref.layers[layer_idx].attn(x, _torch_rope(config, layer_type, SEQ), masks[layer_type])


def _device_masks(builder, attention_mask):
    pad = builder.upload_pad_row(attention_mask)
    masks = builder.build(pad)
    ttnn.deallocate(pad)
    return masks


def _run(device, laya_config, attn_params, layer_type, plan, rotary, x, mask, module_layer_type=None):
    module = TtnnModernBertAttention(attn_params, laya_config, module_layer_type or layer_type, plan, device)
    tt_x = ttnn.from_torch(x, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    out = module(tt_x, rotary, mask)
    got = ttnn.to_torch(out).float().reshape(x.shape)
    ttnn.deallocate(out)
    ttnn.deallocate(tt_x)
    return got


def _real_pcc(expected, got, attention_mask):
    real = attention_mask == 1
    return pcc(expected[real], got[real])


@pytest.mark.parametrize("layer_type", [FULL_ATTENTION, SLIDING_ATTENTION])
def test_ttnn_attention_matches_reference_padded_batch(device, tt_params, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log, layer_type):
    b, xs = batch_inputs
    plan, rotary, builder = tt_env
    idx = LAYER_FOR[layer_type]
    x = xs[idx]
    expected = _reference(laya_config, torch_encoder, idx, layer_type, x, b["attention_mask"])
    masks = _device_masks(builder, b["attention_mask"])
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], layer_type, plan, rotary, x, masks[layer_type])
    deallocate_masks(masks)
    p_real = _real_pcc(expected, got, b["attention_mask"])
    p_all = pcc(expected, got)
    record(pcc_log, test="attention", layer_type=layer_type, layer=idx, batch=BATCH, seq=SEQ, lengths=b["lengths"].tolist(), pcc_real=p_real, pcc_all=p_all)
    assert p_real >= ATTENTION_PCC, f"attention {layer_type} PCC {p_real:.8f} < {ATTENTION_PCC}"


def test_negative_control_band_removed(device, tt_params, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log):
    b, xs = batch_inputs
    plan, rotary, builder = tt_env
    idx = LAYER_FOR[SLIDING_ATTENTION]
    expected = _reference(laya_config, torch_encoder, idx, SLIDING_ATTENTION, xs[idx], b["attention_mask"])
    masks = _device_masks(builder, b["attention_mask"])
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], SLIDING_ATTENTION, plan, rotary, xs[idx], masks[FULL_ATTENTION])
    deallocate_masks(masks)
    p = _real_pcc(expected, got, b["attention_mask"])
    record(pcc_log, test="NC sliding without band", pcc=p, threshold=ATTENTION_PCC)
    assert p < ATTENTION_PCC


def test_negative_control_band_32(device, tt_params, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log):
    b, xs = batch_inputs
    plan, rotary, _ = tt_env
    idx = LAYER_FOR[SLIDING_ATTENTION]
    expected = _reference(laya_config, torch_encoder, idx, SLIDING_ATTENTION, xs[idx], b["attention_mask"])
    narrow = TtnnMaskBuilder(laya_config, device, SEQ, BATCH, half_window=32)
    masks = _device_masks(narrow, b["attention_mask"])
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], SLIDING_ATTENTION, plan, rotary, xs[idx], masks[SLIDING_ATTENTION])
    deallocate_masks(masks)
    narrow.deallocate()
    p = _real_pcc(expected, got, b["attention_mask"])
    record(pcc_log, test="NC band +/-32", pcc=p, threshold=ATTENTION_PCC)
    assert p < ATTENTION_PCC


def test_negative_control_wrong_theta(device, tt_params, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log):
    b, xs = batch_inputs
    plan, rotary, builder = tt_env
    idx = LAYER_FOR[FULL_ATTENTION]
    expected = _reference(laya_config, torch_encoder, idx, FULL_ATTENTION, xs[idx], b["attention_mask"])
    masks = _device_masks(builder, b["attention_mask"])
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], FULL_ATTENTION, plan, rotary, xs[idx], masks[FULL_ATTENTION], module_layer_type=SLIDING_ATTENTION)
    deallocate_masks(masks)
    p = _real_pcc(expected, got, b["attention_mask"])
    record(pcc_log, test="NC theta swapped", pcc=p, threshold=ATTENTION_PCC)
    assert p < ATTENTION_PCC


def test_negative_control_pad_mask_dropped(device, tt_params, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log):
    b, xs = batch_inputs
    plan, rotary, builder = tt_env
    idx = LAYER_FOR[FULL_ATTENTION]
    expected = _reference(laya_config, torch_encoder, idx, FULL_ATTENTION, xs[idx], b["attention_mask"])
    masks = _device_masks(builder, torch.ones_like(b["attention_mask"]))
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], FULL_ATTENTION, plan, rotary, xs[idx], masks[FULL_ATTENTION])
    deallocate_masks(masks)
    real0 = b["attention_mask"][0] == 1
    p = pcc(expected[0][real0], got[0][real0])
    record(pcc_log, test="NC pad mask dropped (padded row)", pcc=p, threshold=ATTENTION_PCC, real_tokens=int(real0.sum()))
    assert p < ATTENTION_PCC


def test_negative_control_qkv_permuted(device, tt_params, parts, torch_encoder, laya_config, batch_inputs, tt_env, pcc_log):
    b, xs = batch_inputs
    plan, rotary, builder = tt_env
    idx = LAYER_FOR[FULL_ATTENTION]
    expected = _reference(laya_config, torch_encoder, idx, FULL_ATTENTION, xs[idx], b["attention_mask"])
    w = parts["encoder"][f"layers.{idx}.attn.Wqkv.weight"]
    h = w.shape[0] // 3
    permuted = torch.cat([w[h : 2 * h], w[:h], w[2 * h :]], dim=0)
    bad = ttnn.from_torch(
        fold_q_scale(permuted, 64).transpose(0, 1).contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    masks = _device_masks(builder, b["attention_mask"])
    attn_params = {"Wqkv": bad, "Wo": tt_params["layers"][idx]["attn"]["Wo"]}
    got = _run(device, laya_config, attn_params, FULL_ATTENTION, plan, rotary, xs[idx], masks[FULL_ATTENTION])
    deallocate_masks(masks)
    ttnn.deallocate(bad)
    p = _real_pcc(expected, got, b["attention_mask"])
    record(pcc_log, test="NC Q/K permuted", pcc=p, threshold=ATTENTION_PCC)
    assert p < ATTENTION_PCC


def test_band_is_invisible_at_seq_64(device, tt_params, torch_encoder, laya_config, pcc_log):
    seq_len = 64
    b = encoder_inputs(batch_size=1, seq_len=512)
    ids = b["input_ids"][:, :seq_len]
    att = torch.ones_like(ids)
    idx = LAYER_FOR[SLIDING_ATTENTION]
    with torch.no_grad():
        x = torch_encoder.layers[idx].attn_norm(torch_encoder.layers[0](torch_encoder.embeddings(ids), _torch_rope(laya_config, FULL_ATTENTION, seq_len), None))
        masks_t = torch_build_masks(laya_config, att, seq_len, x.dtype, laya_config.local_attention // 2)
        expected = torch_encoder.layers[idx].attn(x, _torch_rope(laya_config, SLIDING_ATTENTION, seq_len), masks_t[SLIDING_ATTENTION])
    plan = bucket_plan(device, laya_config, 1, seq_len)
    rotary = TtnnModernBertRotary(laya_config, device, seq_len, attention_memory=plan.attention_memory)
    builder = TtnnMaskBuilder(laya_config, device, seq_len, 1)
    masks = _device_masks(builder, att)
    got = _run(device, laya_config, tt_params["layers"][idx]["attn"], SLIDING_ATTENTION, plan, rotary, x, masks[FULL_ATTENTION])
    deallocate_masks(masks)
    builder.deallocate()
    rotary.deallocate()
    p = pcc(expected, got)
    record(pcc_log, test="band invisible at seq 64", pcc=p)
    assert p >= ATTENTION_PCC
