# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.reference.modernbert import ModernBertRotaryEmbedding
from models.autoports.convaiinnovations_laya.reference.modernbert import build_masks as torch_build_masks
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import max_abs_err, outlier_report, pcc
from models.autoports.convaiinnovations_laya.tt.model_config import ACTIVATIONS_DTYPE, DEFAULT_PORT, bucket_plan
from models.autoports.convaiinnovations_laya.tt.modernbert_layer import TtnnModernBertEncoderLayer
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, deallocate_masks
from models.autoports.convaiinnovations_laya.tt.modernbert_rope import TtnnModernBertRotary
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
LAYER_PCC = 0.999
LAYERS = [0, 1, 16, 27]
SEQ = 512


@pytest.fixture(scope="module")
def tt_params(module_device, parts, laya_config):
    device = module_device
    params = prepare_weights(parts["encoder"], laya_config, device, layers=LAYERS)
    yield params
    deallocate_weights(params)


@pytest.fixture(scope="module")
def ref_states(torch_encoder):
    out = {}
    for batch in (1, 2):
        b = encoder_inputs(batch_size=batch, seq_len=SEQ, fill=batch > 1)
        with torch.no_grad():
            _, hidden = torch_encoder(b["input_ids"], b["attention_mask"], output_hidden_states=True)
        out[batch] = (b, hidden)
    return out


def _layer_reference(config, ref, layer_idx, hidden_in, attention_mask):
    layer = ref.layers[layer_idx]
    hd = config.hidden_size // config.num_attention_heads
    theta = config.rope_parameters[layer.attention_type]["rope_theta"]
    pos = ModernBertRotaryEmbedding(hd, theta)(torch.arange(SEQ).unsqueeze(0), torch.float32)
    masks = torch_build_masks(config, attention_mask, SEQ, hidden_in.dtype, config.local_attention // 2)
    with torch.no_grad():
        return layer(hidden_in, pos, masks[layer.attention_type])


def _run_layer(device, config, params, layer_idx, hidden_in, attention_mask, port=DEFAULT_PORT, attn_norm_override=None):
    batch = hidden_in.shape[0]
    plan = bucket_plan(device, config, batch, SEQ, port=port)
    rotary = TtnnModernBertRotary(config, device, SEQ, batch_size=batch, attention_memory=plan.attention_memory, port=port)
    builder = TtnnMaskBuilder(config, device, SEQ, batch)
    pad = builder.upload_pad_row(attention_mask)
    masks = builder.build(pad)
    module = TtnnModernBertEncoderLayer(params["layers"][layer_idx], config, layer_idx, plan, device, port=port)
    if attn_norm_override is not None:
        module.attn_norm = attn_norm_override
    tt_h = ttnn.from_torch(hidden_in, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    if module.resident:
        sh = ttnn.to_memory_config(tt_h, plan.mlp_shard.hidden_memory)
        ttnn.deallocate(tt_h)
        tt_h = sh
    out = module(tt_h, rotary, masks[module.layer_type])
    got = ttnn.to_torch(out).float().reshape(hidden_in.shape)
    ttnn.deallocate(out)
    deallocate_masks(masks)
    ttnn.deallocate(pad)
    builder.deallocate()
    rotary.deallocate()
    return got, plan


@pytest.mark.parametrize("layer_idx", LAYERS)
@pytest.mark.parametrize("batch", [1, 2])
def test_ttnn_layer_matches_reference(device, tt_params, torch_encoder, laya_config, ref_states, pcc_log, layer_idx, batch):
    b, hidden = ref_states[batch]
    expected = _layer_reference(laya_config, torch_encoder, layer_idx, hidden[layer_idx], b["attention_mask"])
    got, plan = _run_layer(device, laya_config, tt_params, layer_idx, hidden[layer_idx], b["attention_mask"])
    real = b["attention_mask"] == 1
    p = pcc(expected[real], got[real])
    record(
        pcc_log,
        test="layer",
        layer=layer_idx,
        layer_type=laya_config.layer_types[layer_idx],
        batch=batch,
        seq=SEQ,
        resident=plan.mlp_shard is not None,
        pcc=p,
        pcc_all=pcc(expected, got),
        max_abs_err=max_abs_err(expected[real], got[real]),
        input_outliers=outlier_report(hidden[layer_idx]),
        output_outliers=outlier_report(expected),
    )
    assert p >= LAYER_PCC, f"layer {layer_idx} batch {batch} PCC {p:.8f} < {LAYER_PCC}"


def test_layer0_has_no_attn_norm(torch_encoder, tt_params):
    import torch.nn as nn

    assert isinstance(torch_encoder.layers[0].attn_norm, nn.Identity)
    assert isinstance(torch_encoder.layers[1].attn_norm, nn.LayerNorm)
    assert tt_params["layers"][0]["attn_norm"] is None
    assert tt_params["layers"][1]["attn_norm"] is not None


def test_negative_control_norm_at_layer0(device, tt_params, torch_encoder, laya_config, ref_states, pcc_log):
    b, hidden = ref_states[1]
    expected = _layer_reference(laya_config, torch_encoder, 0, hidden[0], b["attention_mask"])
    got, _ = _run_layer(device, laya_config, tt_params, 0, hidden[0], b["attention_mask"], attn_norm_override=tt_params["layers"][1]["attn_norm"])
    p = pcc(expected, got)
    record(pcc_log, test="NC norm applied at layer 0", pcc=p, threshold=LAYER_PCC)
    assert p < LAYER_PCC
