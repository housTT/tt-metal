# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.tests.conftest import DOC_DIR, encoder_inputs, policy_from_env, port_from_env, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import max_abs_err, outlier_report, pcc
from models.autoports.convaiinnovations_laya.tt.model_config import DEFAULT_PORT
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, deallocate_masks
from models.autoports.convaiinnovations_laya.tt.modernbert_model import TtnnModernBertModel
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
MODEL_PCC = 0.99
OUT = os.environ.get("LAYA_LAYER_PCC_OUT", os.path.join(DOC_DIR, "functional_decoder", "layer_pcc.json"))
SHAPES = [
    (1, 512, "default"),
    (1, 512, "fill_only"),
    (2, 512, "default"),
    (2, 512, "interleaved"),
    (4, 512, "default"),
    (8, 512, "default"),
    (1, 1024, "default"),
]
PORT = port_from_env()
POLICY = policy_from_env()
PORTS = {"default": PORT, "interleaved": PORT.with_(geglu_plan="interleaved"), "fill_only": PORT}


@pytest.fixture(scope="module")
def tt_params(module_device, parts, laya_config):
    device = module_device
    params = prepare_weights(parts["encoder"], laya_config, device, POLICY)
    yield params
    deallocate_weights(params)


@pytest.fixture(scope="module")
def layer_rows():
    rows = []
    yield rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)


def _ids(ids, device):
    return ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _run(device, laya_config, tt_params, b, batch, seq_len, per_layer, port=PORT):
    model = TtnnModernBertModel(tt_params, laya_config, device, seq_len, batch, policy=POLICY, port=port)
    builder = TtnnMaskBuilder(laya_config, device, seq_len, batch)
    pad = builder.upload_pad_row(b["attention_mask"])
    masks = builder.build(pad)
    captured = {}

    def hook(idx, t):
        captured[idx] = ttnn.to_torch(t).float()

    out = model(_ids(b["input_ids"], device), masks, layer_hook=hook if per_layer else None)
    got = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    deallocate_masks(masks)
    ttnn.deallocate(pad)
    builder.deallocate()
    model.deallocate()
    return got, captured, model.plan


@pytest.mark.parametrize("batch,seq_len,variant", SHAPES)
def test_ttnn_encoder_matches_fp32(device, tt_params, torch_encoder, laya_config, pcc_log, layer_rows, batch, seq_len, variant):
    b = encoder_inputs(batch_size=batch, seq_len=seq_len, fill=batch > 1 or variant == "fill_only")
    with torch.no_grad():
        expected, hidden = torch_encoder(b["input_ids"], b["attention_mask"], output_hidden_states=True)
    got, captured, plan = _run(device, laya_config, tt_params, b, batch, seq_len, per_layer=True, port=PORTS[variant])
    got = got.reshape(expected.shape)
    real = b["attention_mask"] == 1
    worst = (1.0, None)
    for idx in sorted(captured):
        ref_i = hidden[idx + 1] if idx + 1 < len(hidden) - 1 else None
        if ref_i is None:
            continue
        g = captured[idx].reshape(ref_i.shape)
        p_i = pcc(ref_i[real], g[real])
        row = {
            "batch": batch,
            "seq": seq_len,
            "variant": variant,
            "policy": POLICY.name,
            "layer": idx,
            "layer_type": laya_config.layer_types[idx],
            "pcc": p_i,
            "max_abs_err": max_abs_err(ref_i[real], g[real]),
            "ref_outliers": outlier_report(ref_i),
            "tt_outliers": outlier_report(g),
        }
        layer_rows.append(row)
        if p_i < worst[0]:
            worst = (p_i, idx)
    p = pcc(expected[real], got[real])
    per_row = [pcc(expected[i][real[i]], got[i][real[i]]) for i in range(batch)]
    per_row_l19 = None
    if 19 in captured:
        g19 = captured[19].reshape(hidden[20].shape)
        per_row_l19 = [pcc(hidden[20][i][real[i]], g19[i][real[i]]) for i in range(batch)]
    layer_rows.append(
        {
            "batch": batch,
            "seq": seq_len,
            "variant": variant,
            "policy": POLICY.name,
            "layer": "final_norm",
            "pcc": p,
            "pcc_per_row": per_row,
            "pcc_per_row_after_layer19": per_row_l19,
            "pcc_all_positions": pcc(expected, got),
            "max_abs_err": max_abs_err(expected[real], got[real]),
            "worst_layer": {"layer": worst[1], "pcc": worst[0]},
            "lengths": b["lengths"].tolist(),
            "plan": {"attention_memory": "L1" if plan.attention_memory == ttnn.L1_MEMORY_CONFIG else "DRAM", "mlp_sharded": plan.mlp_shard is not None},
        }
    )
    record(pcc_log, test="encoder", batch=batch, seq=seq_len, variant=variant, policy=POLICY.name, port=PORTS[variant].describe(), pcc=p, per_row=per_row, per_row_after_layer19=per_row_l19, worst_layer=worst[1], worst_layer_pcc=worst[0], max_abs_err=max_abs_err(expected[real], got[real]))
    assert p >= MODEL_PCC, f"encoder PCC {p:.8f} < {MODEL_PCC} (worst layer {worst[1]} at {worst[0]:.6f})"


def test_ttnn_encoder_vs_bf16_reference(device, tt_params, torch_encoder, laya_config, pcc_log):
    b = encoder_inputs(batch_size=1, seq_len=512)
    ref16 = torch_encoder.to(torch.bfloat16)
    try:
        with torch.no_grad():
            expected = ref16(b["input_ids"], b["attention_mask"]).float()
    finally:
        torch_encoder.to(torch.float32)
    got, _, _ = _run(device, laya_config, tt_params, b, 1, 512, per_layer=False)
    p = pcc(expected, got.reshape(expected.shape))
    record(pcc_log, test="encoder vs bf16 reference (informational)", batch=1, seq=512, pcc=p)
    assert torch.isfinite(got).all()


def test_layer_type_pattern(laya_config):
    expected = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)]
    assert list(laya_config.layer_types) == expected
    assert len(laya_config.layer_types) == 28
    assert sum(t == "full_attention" for t in laya_config.layer_types) == 10
