# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.autoports.convaiinnovations_laya.tests import head_reference as HR
from models.autoports.convaiinnovations_laya.tests.conftest import encoder_inputs, policy_from_env, port_from_env, record
from models.autoports.convaiinnovations_laya.tests.pcc_utils import max_abs_err, pcc
from models.autoports.convaiinnovations_laya.tt.laya_head import TtnnLayaHead
from models.autoports.convaiinnovations_laya.tt.laya_model import TtnnLayaModel
from models.autoports.convaiinnovations_laya.tt.model_config import (
    ACTIVATIONS_DTYPE,
    DEFAULT_POLICY,
    FULL_ATTENTION,
    POLICIES,
    bucket_plan,
)
from models.autoports.convaiinnovations_laya.tt.modernbert_masks import TtnnMaskBuilder, deallocate_masks
from models.autoports.convaiinnovations_laya.tt.weights import deallocate_weights, prepare_head_weights

pytestmark = pytest.mark.use_module_device({"l1_small_size": 79104})
HEAD_PCC = 0.999
SCORER_PCC = 0.99
BATCH, SEQ = 2, 512
PORT = port_from_env()
POLICY = policy_from_env()


@pytest.fixture(scope="module")
def head_params(module_device, state_dict, laya_config):
    device = module_device
    params = prepare_head_weights(state_dict, laya_config, device)
    yield params
    deallocate_weights({"type_emb": params["type_emb"], "layers": params["layers"], "scorer": params["scorer"]})


@pytest.fixture(scope="module")
def encoder_out(torch_encoder):
    b = encoder_inputs(batch_size=BATCH, seq_len=SEQ, fill=True)
    with torch.no_grad():
        out = torch_encoder(b["input_ids"], b["attention_mask"])
    return b, out


@pytest.fixture(scope="module")
def tt_env(module_device, laya_config):
    device = module_device
    plan = bucket_plan(device, laya_config, BATCH, SEQ, port=PORT)
    builder = TtnnMaskBuilder(laya_config, device, SEQ, BATCH)
    yield plan, builder
    builder.deallocate()


def _masks(builder, att):
    pad = builder.upload_pad_row(att)
    masks = builder.build(pad)
    ttnn.deallocate(pad)
    return masks


def _standardize(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


def _qtype(qtype, device):
    return ttnn.from_torch(qtype.reshape(-1, 1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def test_type_embedding_exact_for_three_types(device, head_params, parts):
    for t in range(3):
        q = _qtype(torch.tensor([t, t]), device)
        emb = ttnn.embedding(q, head_params["type_emb"], layout=ttnn.TILE_LAYOUT)
        got = ttnn.to_torch(emb).float()[:, 0, :]
        ttnn.deallocate(emb)
        ttnn.deallocate(q)
        want = parts["type_emb"][t].to(torch.bfloat16).float()
        assert torch.equal(got[0], want) and torch.equal(got[1], want)


def test_head_layers_match_explicit_math(device, head_params, parts, laya_config, encoder_out, tt_env, pcc_log):
    b, enc = encoder_out
    plan, builder = tt_env
    expected = HR.head_layers(parts, enc + parts["type_emb"][b["qtype"]][:, None, :], b["attention_mask"])
    head = TtnnLayaHead(head_params, laya_config, plan, device)
    masks = _masks(builder, b["attention_mask"])
    x = ttnn.from_torch(enc, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    q = _qtype(b["qtype"], device)
    h = head.add_type_embedding(x, q)
    h = head.run_layers(h, masks[FULL_ATTENTION])
    got = ttnn.to_torch(h).float().reshape(expected.shape)
    ttnn.deallocate(h)
    ttnn.deallocate(q)
    deallocate_masks(masks)
    real = b["attention_mask"] == 1
    p = pcc(expected[real], got[real])
    p_std = pcc(_standardize(expected)[real], _standardize(got)[real])
    record(pcc_log, test="head layers", batch=BATCH, seq=SEQ, pcc=p, pcc_standardized=p_std, max_abs_err=max_abs_err(expected[real], got[real]))
    assert p >= HEAD_PCC, f"head layers PCC {p:.8f} < {HEAD_PCC}"
    assert p_std >= SCORER_PCC, f"head layers standardized PCC {p_std:.8f} < {SCORER_PCC}"


@pytest.mark.parametrize("policy_name", ["bf8w_hifi3", "bf8w_hifi3_head_bf16"])
def test_scorer_matches_explicit_math(device, state_dict, parts, laya_config, encoder_out, tt_env, pcc_log, policy_name):
    b, enc = encoder_out
    plan, _ = tt_env
    policy = POLICIES[policy_name]
    params = prepare_head_weights(state_dict, laya_config, device, policy)
    h_ref = HR.head_layers(parts, enc + parts["type_emb"][b["qtype"]][:, None, :], b["attention_mask"])
    expected = HR.scorer(parts, h_ref)
    head = TtnnLayaHead(params, laya_config, plan, device, policy)
    x = ttnn.from_torch(h_ref, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    logits = head.scorer(x)
    got = ttnn.to_torch(logits).float().reshape(expected.shape)
    ttnn.deallocate(logits)
    ttnn.deallocate(x)
    deallocate_weights({"type_emb": params["type_emb"], "layers": params["layers"], "scorer": params["scorer"]})
    real = b["attention_mask"] == 1
    p = pcc(expected[real], got[real])
    markers = HR.gather_markers(expected, b["marker_pos"], b["marker_mask"]), HR.gather_markers(got, b["marker_pos"], b["marker_mask"])
    mk = b["marker_mask"]
    record(
        pcc_log,
        test="scorer (isolated, fp32 input)",
        policy=policy_name,
        fp32_out=policy.scorer_fp32_out,
        pcc_real=p,
        max_abs_err_real=max_abs_err(expected[real], got[real]),
        marker_max_abs_err=max_abs_err(markers[0][mk], markers[1][mk]),
        marker_logits_ref=markers[0][mk].tolist(),
        marker_logits_tt=markers[1][mk].tolist(),
    )
    assert p >= SCORER_PCC


def _device_head_logits(device, head, builder, enc, att, qtype):
    masks = _masks(builder, att)
    x = ttnn.from_torch(enc, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    q = _qtype(qtype, device)
    h = head.run_layers(head.add_type_embedding(x, q), masks[FULL_ATTENTION])
    logits = head.scorer(h)
    got = ttnn.to_torch(logits).float().reshape(enc.shape[:2])
    ttnn.deallocate(logits)
    ttnn.deallocate(h)
    ttnn.deallocate(q)
    deallocate_masks(masks)
    return got


def _clean_reference(parts, b, enc):
    h = HR.head_layers(parts, enc + parts["type_emb"][b["qtype"]][:, None, :], b["attention_mask"])
    return HR.scorer(parts, h)


def test_weak_controls_relu_and_head_pad_mask_are_recorded(device, head_params, parts, laya_config, encoder_out, tt_env, pcc_log):
    """The plan's two head controls move the fp32 reference's own scorer logits by about 0.01, so no implementation can
    fail them; they are measured on device and on the reference and recorded, not gated."""
    b, enc = encoder_out
    plan, builder = tt_env
    real = b["attention_mask"] == 1
    clean_ref = _clean_reference(parts, b, enc)
    h_in = enc + parts["type_emb"][b["qtype"]][:, None, :]
    ref_gelu = HR.scorer(parts, HR.head_layers(parts, h_in, b["attention_mask"], activation="gelu"))
    ref_nopad = HR.scorer(parts, HR.head_layers(parts, h_in, torch.ones_like(b["attention_mask"])))
    head = TtnnLayaHead(head_params, laya_config, plan, device)
    for layer in head.layers:
        layer.relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0)
    dev_gelu = _device_head_logits(device, head, builder, enc, b["attention_mask"], b["qtype"])
    head = TtnnLayaHead(head_params, laya_config, plan, device)
    dev_nopad = _device_head_logits(device, head, builder, enc, torch.ones_like(b["attention_mask"]), b["qtype"])
    mk = b["marker_mask"]
    g = lambda l: HR.gather_markers(l, b["marker_pos"], mk)[mk]
    record(
        pcc_log,
        test="weak controls (recorded)",
        relu_to_gelu={
            "reference_intrinsic_pcc": pcc(clean_ref[real], ref_gelu[real]),
            "reference_intrinsic_marker_max_abs": max_abs_err(g(clean_ref), g(ref_gelu)),
            "device_control_vs_clean_reference_pcc": pcc(clean_ref[real], dev_gelu[real]),
            "device_control_vs_reference_control_pcc": pcc(ref_gelu[real], dev_gelu[real]),
        },
        head_pad_mask_dropped={
            "reference_intrinsic_pcc": pcc(clean_ref[real], ref_nopad[real]),
            "reference_intrinsic_marker_max_abs": max_abs_err(g(clean_ref), g(ref_nopad)),
            "device_control_vs_clean_reference_pcc": pcc(clean_ref[real], dev_nopad[real]),
            "device_control_vs_reference_control_pcc": pcc(ref_nopad[real], dev_nopad[real]),
        },
    )
    assert torch.isfinite(dev_gelu).all() and torch.isfinite(dev_nopad).all()


def test_negative_control_head_layers_skipped(device, head_params, parts, laya_config, encoder_out, tt_env, pcc_log):
    b, enc = encoder_out
    plan, builder = tt_env
    real = b["attention_mask"] == 1
    clean_ref = _clean_reference(parts, b, enc)
    head = TtnnLayaHead(head_params, laya_config, plan, device)
    x = ttnn.from_torch(enc, dtype=ACTIVATIONS_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
    q = _qtype(b["qtype"], device)
    h = head.add_type_embedding(x, q)
    logits = head.scorer(h)
    got = ttnn.to_torch(logits).float().reshape(enc.shape[:2])
    ttnn.deallocate(logits)
    ttnn.deallocate(h)
    ttnn.deallocate(q)
    p = pcc(clean_ref[real], got[real])
    ref_skip = HR.scorer(parts, enc + parts["type_emb"][b["qtype"]][:, None, :])
    record(pcc_log, test="NC head layers skipped", pcc_scorer_logits=p, reference_intrinsic_pcc=pcc(clean_ref[real], ref_skip[real]), threshold=SCORER_PCC)
    assert p < SCORER_PCC


def test_negative_control_wrong_question_type(device, head_params, parts, laya_config, encoder_out, tt_env, pcc_log):
    b, enc = encoder_out
    plan, builder = tt_env
    real = b["attention_mask"] == 1
    clean_ref = _clean_reference(parts, b, enc)
    wrong = (b["qtype"] + 1) % 3
    head = TtnnLayaHead(head_params, laya_config, plan, device)
    got = _device_head_logits(device, head, builder, enc, b["attention_mask"], wrong)
    p = pcc(clean_ref[real], got[real])
    ref_wrong = HR.scorer(parts, HR.head_layers(parts, enc + parts["type_emb"][wrong][:, None, :], b["attention_mask"]))
    record(pcc_log, test="NC wrong question type", pcc_scorer_logits=p, reference_intrinsic_pcc=pcc(clean_ref[real], ref_wrong[real]), threshold=SCORER_PCC)
    assert p < SCORER_PCC


@pytest.mark.parametrize("batch", [1, 8])
def test_full_device_model_end_to_end(device, state_dict, parts, torch_encoder, laya_config, pcc_log, batch):
    b = encoder_inputs(batch_size=batch, seq_len=SEQ, fill=batch > 1)
    with torch.no_grad():
        enc = torch_encoder(b["input_ids"], b["attention_mask"])
    logits_ref, cls_ref, _ = HR.decision_head(parts, enc, b["attention_mask"], b["qtype"])
    model = TtnnLayaModel(device, laya_config, state_dict=state_dict, policy=POLICY, port=PORT, row_buckets=(1, 8), seq_buckets=(512,))
    try:
        out = model.forward(b["input_ids"], b["attention_mask"], b["qtype"])
    finally:
        model.close()
    real = b["attention_mask"] == 1
    p_logits = pcc(logits_ref[real], out["logits"][real])
    mk = b["marker_mask"]
    m_ref = HR.gather_markers(logits_ref, b["marker_pos"], mk)[mk]
    m_tt = HR.gather_markers(out["logits"], b["marker_pos"], mk)[mk]
    p_cls = pcc(cls_ref, out["cls"])
    record(
        pcc_log,
        test="TtnnLayaModel end to end",
        policy=POLICY.name,
        port=PORT.describe(),
        batch=batch,
        bucket=list(out["bucket"]),
        pcc_logits_real=p_logits,
        pcc_markers=pcc(m_ref, m_tt),
        marker_max_abs_err=max_abs_err(m_ref, m_tt),
        pcc_cls=p_cls,
        device_ms=out["device_ms"],
        nan=bool(torch.isnan(out["logits"]).any()),
    )
    assert not torch.isnan(out["logits"]).any()
    assert p_logits >= SCORER_PCC and p_cls >= SCORER_PCC
