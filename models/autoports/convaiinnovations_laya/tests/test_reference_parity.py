# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from models.autoports.convaiinnovations_laya import common
from models.autoports.convaiinnovations_laya.reference import laya_reference as lr
from models.autoports.convaiinnovations_laya.reference.modernbert import ModernBertModel
from models.autoports.convaiinnovations_laya.tests.pcc_utils import max_abs_err, pcc

PARITY_PCC = 0.999999
SEQ_LEN = 512
BATCH = 4
EXPECTED_ENCODER_TENSORS = 170
EXPECTED_TOTAL_TENSORS = 206
LOGIT_TOL = 1e-4
ACT_LOGIT_TOL = 1e-2


def build_reference(config, ablate=None):
    ref = ModernBertModel(config, ablate=ablate)
    ref.eval()
    return ref


def load_into_reference(ref, hf_model):
    return ref.load_state_dict(hf_model.state_dict(), strict=True)


def forward_args(batch):
    return (batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])


@pytest.fixture(scope="module")
def config():
    return common.load_config()


@pytest.fixture(scope="module")
def hf_model():
    return common.load_torch_model(attn_implementation="eager")


@pytest.fixture(scope="module")
def batch():
    return common.build_batch(SEQ_LEN, BATCH, seed=0)


@pytest.fixture(scope="module")
def ref():
    return lr.LayaReference()


@pytest.fixture(scope="module")
def vendored_agent():
    api = common.vendor_module("rl_agent_api")
    return api.RLAgent(common.model_dir(), device="cpu")


def test_config_pins(config):
    assert common.rope_theta(config, common.FULL_ATTENTION) == 160000.0
    assert common.rope_theta(config, common.SLIDING_ATTENTION) == 10000.0
    assert common.sliding_window_half(config) == 64
    assert config.local_attention == 128
    assert config.num_hidden_layers == 28
    assert config.hidden_size == 1024
    assert config.num_attention_heads == 16
    assert config.intermediate_size == 2624
    assert config.vocab_size == 50368
    assert config.pad_token_id == 50283
    assert all((lt == common.FULL_ATTENTION) == (i % 3 == 0) for i, lt in enumerate(config.layer_types))
    rl = common.load_rl_config()
    assert (rl["max_len"], rl["head_max_len"], rl["head_layers"]) == (512, 192, 2)


def test_state_dict_maps_exactly(config, hf_model):
    ref = build_reference(config)
    load_into_reference(ref, hf_model)
    hf_keys = set(hf_model.state_dict().keys())
    ref_keys = set(ref.state_dict().keys())
    assert ref_keys == hf_keys, f"missing={hf_keys - ref_keys} extra={ref_keys - hf_keys}"
    assert len(hf_keys) == EXPECTED_ENCODER_TENSORS
    assert "layers.0.attn_norm.weight" not in hf_keys
    assert "layers.1.attn_norm.weight" in hf_keys
    sd = common.load_state_dict(dtype=None)
    assert len(sd) == EXPECTED_TOTAL_TENSORS
    assert set(common.encoder_state_dict(sd)) == hf_keys
    assert {str(v.dtype) for v in sd.values()} == {"torch.float16", "torch.float32"}
    assert sd["temperature"].dtype == torch.float32
    prefixes = {k.split(".")[0] for k in sd}
    assert prefixes == {"encoder", "head", "scorer", "act_head", "type_emb", "temperature"}


def test_batch_is_real_and_padded(batch, config):
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert ids.shape == (BATCH, SEQ_LEN)
    assert mask.sum() < BATCH * SEQ_LEN
    assert (ids[:, 0] == config.cls_token_id).all()
    tok = common.load_tokenizer()
    for r in range(BATCH):
        L = int(mask[r].sum())
        assert (ids[r, L:] == config.pad_token_id).all()
        for j in range(int(batch["marker_mask"][r].sum())):
            pos = int(batch["marker_pos"][r, j])
            assert pos < L
            assert int(ids[r, pos]) == tok.mask_token_id


def test_reference_matches_hf(config, hf_model, batch):
    ref = build_reference(config)
    load_into_reference(ref, hf_model)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    real = mask.bool()
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        ref_out, ref_hidden = ref(ids, mask, output_hidden_states=True)
    assert len(hf_out.hidden_states) == config.num_hidden_layers + 1 == len(ref_hidden)
    final = pcc(hf_out.last_hidden_state[real], ref_out[real])
    print(f"\n[padded S={SEQ_LEN} B={BATCH}] final PCC real={final:.10f} all={pcc(hf_out.last_hidden_state, ref_out):.10f} max_abs_err={max_abs_err(hf_out.last_hidden_state, ref_out):.3e}")
    worst = (1.0, None)
    for i, (h, r) in enumerate(zip(hf_out.hidden_states, ref_hidden)):
        p = pcc(h[real], r[real])
        if p < worst[0]:
            worst = (p, i)
        assert p >= PARITY_PCC, f"hidden state {i} PCC {p:.10f} < {PARITY_PCC}"
    print(f"worst hidden-state PCC {worst[0]:.10f} at index {worst[1]}")
    assert final >= PARITY_PCC


def test_hf_eager_vs_sdpa(hf_model, batch):
    sdpa = common.load_torch_model(attn_implementation="sdpa")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.no_grad():
        a = hf_model(input_ids=ids, attention_mask=mask).last_hidden_state
        b = sdpa(input_ids=ids, attention_mask=mask).last_hidden_state
    real = mask.bool()
    print(f"\n[eager vs sdpa fp32] PCC={pcc(a[real], b[real]):.10f} max_abs_err={max_abs_err(a[real], b[real]):.3e}")
    assert pcc(a[real], b[real]) >= PARITY_PCC


NEGATIVE_CONTROLS = [
    ("NC1_swap_geglu_gate", {"swap_gate": True}),
    ("NC2_wrong_window_65", {"half_window_override": 65 // 2}),
    ("NC3_norm_at_layer0", {"norm_at_layer0": True}),
    ("NC4_qkv_permuted", {"qkv_permute": True}),
    ("NC5_single_rope_theta", {"single_theta": True}),
    ("NC6_band_removed", {"half_window_override": 10**6}),
]


@pytest.mark.parametrize("name,ablate", NEGATIVE_CONTROLS)
def test_negative_control_detects_break(config, hf_model, batch, name, ablate):
    ref = build_reference(config, ablate=ablate)
    try:
        load_into_reference(ref, hf_model)
    except RuntimeError as e:
        print(f"\n[{name}] detected via strict state_dict rejection: {str(e)[:120]}")
        return
    ids, mask = batch["input_ids"][:2], batch["attention_mask"][:2]
    real = mask.bool()
    with torch.no_grad():
        hf_out = hf_model(input_ids=ids, attention_mask=mask).last_hidden_state
        ref_out = ref(ids, mask)
    p = pcc(hf_out[real], ref_out[real])
    print(f"\n[{name}] PCC={p:.10f} (must be < {PARITY_PCC})")
    assert p < PARITY_PCC, f"{name} did not change the output"


def test_laya_reference_loads_strictly(ref):
    rep = ref.load_report
    assert rep["n_tensors"] == EXPECTED_TOTAL_TENSORS
    assert rep["missing"] == [] and rep["unexpected"] == [] and rep["bad_shape"] == []
    assert rep["source_dtypes"] == ["torch.float16", "torch.float32"]
    s = ref.shapes()
    assert (s["hidden_size"], s["num_hidden_layers"], s["num_attention_heads"], s["head_dim"]) == (1024, 28, 16, 64)
    assert (s["head_layers"], s["head_ffn"], s["n_act"], s["max_len"], s["head_max_len"]) == (2, 4096, 2, 512, 192)


def test_laya_reference_bit_identical_to_vendored(ref, vendored_agent, batch):
    args = forward_args(batch)
    valid = batch["marker_mask"]
    logits, act = ref.forward(*args)
    assert logits.dtype == torch.float32 and act.dtype == torch.float32
    assert logits.shape == (BATCH, valid.shape[1]) and act.shape == (BATCH, 2)
    assert torch.isfinite(logits).all() and torch.isfinite(act).all()
    assert (logits[~valid] == lr.NEG_LOGIT).all()
    logits2, act2 = ref.forward(*args)
    assert torch.equal(logits, logits2) and torch.equal(act, act2)
    logits3, act3, hidden = ref.forward_with_hidden(*args)
    assert torch.equal(logits, logits3) and torch.equal(act, act3)
    assert len(hidden["encoder"]) == 29 and len(hidden["head"]) == 2
    with torch.no_grad():
        lv, av = vendored_agent.model(*args)
    lv, av = lv.float(), av.float()
    d_logit = float((lv - logits)[valid].abs().max())
    d_act = float((av - act).abs().max())
    print(f"\n[LayaReference eager vs vendored RLAgent sdpa] max|dlogit|={d_logit:.3e} max|dact|={d_act:.3e}")
    assert d_logit < LOGIT_TOL and d_act < ACT_LOGIT_TOL
    twin = lr.LayaReference(attn_implementation=vendored_agent.model.encoder.config._attn_implementation)
    lt, at = twin.forward(*args)
    assert torch.equal(lt, lv), "LayaReference(sdpa) is not bit-identical to the vendored DecisionModel"
    assert torch.equal(at, av), "LayaReference(sdpa) act logits are not bit-identical to the vendored DecisionModel"


def test_explicit_head_matches_fused(ref, batch):
    args = forward_args(batch)
    valid = batch["marker_mask"]
    real = batch["attention_mask"].bool()
    logits, act, hidden = ref.forward_with_hidden(*args)
    le, ae = ref.forward_explicit_head(*args)
    d_logit = float((le - logits)[valid].abs().max())
    d_act = float((ae - act).abs().max())
    print(f"\n[explicit head vs fused] max|dlogit|={d_logit:.3e} max|dact|={d_act:.3e} PCC={pcc(le[valid], logits[valid]):.10f}")
    assert d_logit < LOGIT_TOL and d_act < ACT_LOGIT_TOL
    assert pcc(le[valid], logits[valid]) >= PARITY_PCC
    x = hidden["after_type_emb"]
    for j, layer in enumerate(ref.model.head.layers):
        with torch.inference_mode():
            y = lr.head_layer_explicit(layer, x, batch["attention_mask"])
            y_nomask = lr.head_layer_explicit(layer, x, torch.ones_like(batch["attention_mask"]))
        fused = hidden["head"][j]
        p = pcc(y[real], fused[real])
        print(f"[head layer {j}] PCC real={p:.10f} max_abs={max_abs_err(y[real], fused[real]):.3e} pad-mask-dropped max_abs={max_abs_err(y_nomask[real], fused[real]):.3e}")
        assert p >= PARITY_PCC
        assert max_abs_err(y_nomask[real], fused[real]) > 1e-3
        x = fused
    with torch.inference_mode():
        s_mod = ref.model.scorer(x).squeeze(-1)
        s_exp = lr.scorer_explicit(ref.model.scorer, x)
    assert torch.allclose(s_mod, s_exp, atol=1e-4, rtol=1e-5)


def test_host_postprocessing_matches_vendored_system_one(ref, vendored_agent):
    case = common.gate_subset(common.load_typed_decisions())[0]
    captured = {}

    def hook(mod, args, out):
        captured["args"] = [a.detach().clone() for a in args]
        captured["out"] = [o.detach().clone() for o in out]

    h = vendored_agent.model.register_forward_hook(hook)
    try:
        with torch.no_grad():
            res = vendored_agent.system_one(case["state"], case["questions"])
    finally:
        h.remove()
    items = ref.encode(case["state"], case["questions"])
    ids_t, att_t, mpos_t, mmask_t, qtype_t = captured["args"]
    for r, it in enumerate(items):
        L = int(att_t[r].sum())
        assert ids_t[r, :L].tolist() == it["ids"]
        assert mpos_t[r, : len(it["markers"])].tolist() == it["markers"]
        assert int(qtype_t[r]) == it["qtype"]
    logits_t, act_t = captured["out"]
    mirror = lr.decode_items(items, logits_t.float().numpy(), lr.act_probs(act_t), vendored_agent.temperature, vendored_agent.temperature_by_options, shape="hub", clamp=False)
    assert mirror == res["answers"]
    pip_shape = lr.decode_items(items, logits_t.float().numpy(), lr.act_probs(act_t), vendored_agent.temperature, vendored_agent.temperature_by_options, shape="pip")
    for qid, ans in pip_shape.items():
        assert ans["type"] == res["answers"][qid]["type"]
        assert 0.0 <= ans["answer_confidence"] <= 1.0
        if ans["type"] != "noul":
            assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-3


def test_temperature_clamp_rule():
    rl = common.load_rl_config()
    temps, table = rl["temperature"], rl["temperature_by_options"]
    assert lr.temperature_for(temps, table, 0, 12, clamp=False) == pytest.approx(0.10058280825614929)
    assert lr.temperature_for(temps, table, 0, 12, clamp=True) == 0.5
    for qt, k in ((0, 2), (0, 4), (0, 6), (1, 4), (2, 2)):
        assert lr.temperature_for(temps, table, qt, k, clamp=True) == lr.temperature_for(temps, table, qt, k, clamp=False)
    assert lr.clamp_temperature(True) == 1.0 and lr.clamp_temperature(None) == 1.0 and lr.clamp_temperature(float("nan")) == 1.0
    assert lr.clamp_temperature(7.0) == 5.0 and lr.clamp_temperature(0.01) == 0.5 and lr.clamp_temperature(1.3) == 1.3
    z = torch.tensor([2.0, 1.0, 0.5] + [0.0] * 9).numpy()
    q = {"t": "choice", "crit": {f"o{i}": None for i in range(12)}}
    raw = lr.decode_answer(q, z, [1.0, 0.0], temps, table, shape="pip", clamp=False)
    clamped = lr.decode_answer(q, z, [1.0, 0.0], temps, table, shape="pip", clamp=True)
    assert raw["choice"] == clamped["choice"] == "o0"
    assert raw["probabilities"]["o0"] > clamped["probabilities"]["o0"]


def test_padding_invariance_fp32(ref):
    items = common.gate_items(ref.tok)[:5]
    nat = ref.collate(items)
    pad = ref.collate(items, seq_len=SEQ_LEN)
    ln, an = ref.forward(*forward_args(nat))
    lp, ap = ref.forward(*forward_args(pad))
    valid = nat["marker_mask"]
    d = float((ln - lp)[valid].abs().max())
    print(f"\n[natural {nat['input_ids'].shape[1]} vs padded {SEQ_LEN}] max|dlogit|={d:.3e} max|dact|={float((an - ap).abs().max()):.3e}")
    assert d < LOGIT_TOL
    assert (ln.argmax(-1) == lp.argmax(-1)).all()
