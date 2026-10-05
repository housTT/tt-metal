# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from models.autoports.convaiinnovations_laya.server import decode
from models.autoports.convaiinnovations_laya.server.engine import (
    Buckets,
    LimitError,
    RequestError,
    check_question,
    pad_batch,
    plan_chunks,
    to_internal,
)
from models.autoports.convaiinnovations_laya.vendor.rl_common import QTYPES, confidence_from_probs, temp_bucket

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "recorded_logits.json")
VENDOR_CFG = os.path.join(os.path.dirname(HERE), "vendor", "rl_agent_config.json")


def load_fixture():
    if not os.path.isfile(FIXTURE):
        pytest.skip("recorded logits missing: %s" % FIXTURE)
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def fixture():
    return load_fixture()


@pytest.fixture(scope="module")
def cfg():
    with open(VENDOR_CFG, encoding="utf-8") as fh:
        return json.load(fh)


def hub_postprocess(cfg, logits_row, act_row, qt, k):
    temperature = cfg.get("temperature", [1.0, 1.0, 1.0])
    by_options = cfg.get("temperature_by_options", {})
    z = np.asarray(logits_row[:k], dtype=np.float32) / by_options.get(temp_bucket(qt, k), temperature[qt])
    p = np.exp(z - z.max())
    p = p / p.sum()
    act = torch.softmax(torch.tensor(act_row, dtype=torch.float32), -1).numpy()
    return p, float(act[0])


def test_temperature_buckets_match_vendored_lookup(cfg):
    raw = decode.Temperatures(cfg, clamp=False)
    for qt in (0, 1, 2):
        for k in (2, 3, 5, 6, 10, 11, 20):
            expected = cfg["temperature_by_options"].get(temp_bucket(qt, k), cfg["temperature"][qt])
            assert raw.for_question(qt, k) == pytest.approx(expected)


def test_temperature_clamp_follows_pip(cfg):
    clamped = decode.Temperatures(cfg, clamp=True)
    for value in clamped.temperature + list(clamped.by_options.values()):
        assert decode.TEMP_MIN <= value <= decode.TEMP_MAX
    assert clamped.by_options["choice:11+"] == decode.TEMP_MIN
    assert "choice:11+" in clamped.describe()["clamped_buckets"]
    assert clamped.for_question(QTYPES["choice"], 3) == pytest.approx(cfg["temperature_by_options"]["choice:3-5"])


def test_decode_matches_hub_math_on_recorded_logits(fixture):
    temps = decode.Temperatures(fixture["cfg"], clamp=False)
    logits = np.asarray(fixture["logits"], dtype=np.float32)
    act = np.asarray(fixture["act_logits"], dtype=np.float32)
    act_probs = torch.softmax(torch.tensor(act), -1).numpy()
    for r, item in enumerate(fixture["items"]):
        q = to_internal(fixture["questions"][item["qid"]])
        k = len(item["markers"])
        p_ref, act_ref = hub_postprocess(fixture["cfg"], logits[r], act[r], item["qtype"], k)
        a = decode.decode_answer(q, logits[r], act_probs[r], k, temps)
        assert a["action"]["act_probability"] == pytest.approx(round(act_ref, 4), abs=1e-4)
        probs = np.array(list(a["probabilities"].values())) if "probabilities" in a else np.array([1 - a["noul"], a["noul"]])
        assert np.allclose(probs, np.round(p_ref, 4), atol=1e-4)
        assert abs(probs.sum() - 1.0) <= 1e-3
        if q["t"] == "choice":
            assert a["choice"] == list(q["crit"].keys())[int(p_ref.argmax())]
            assert a["confidence"] == pytest.approx(round(confidence_from_probs(p_ref, k), 4), abs=1e-4)
        elif q["t"] == "score":
            assert a["score"] == pytest.approx(round(float((np.arange(k) * p_ref).sum()), 4), abs=1e-4)
            assert a["confidence"] == pytest.approx(round(confidence_from_probs(p_ref, k), 4), abs=1e-4)
            assert list(a["legend"].keys()) == [str(i) for i in range(k)]
        else:
            assert a["noul"] == pytest.approx(round(float(p_ref[1]), 4), abs=1e-4)
            assert a["confidence"] == pytest.approx(round(max(float(p_ref[1]), 1 - float(p_ref[1])), 4), abs=1e-4)
        assert a["answer_confidence"] == pytest.approx(round(float(p_ref.max()), 4), abs=1e-4)


def test_decode_ignores_masked_marker_slots(fixture):
    temps = decode.Temperatures(fixture["cfg"], clamp=False)
    logits = np.asarray(fixture["logits"], dtype=np.float32)
    act_probs = torch.softmax(torch.tensor(fixture["act_logits"]), -1).numpy()
    kmax = logits.shape[1]
    for r, item in enumerate(fixture["items"]):
        k = len(item["markers"])
        if k < kmax:
            assert np.all(logits[r, k:] <= -1e3)
        a = decode.decode_answer(to_internal(fixture["questions"][item["qid"]]), logits[r], act_probs[r], k, temps)
        n = len(a["probabilities"]) if "probabilities" in a else 2
        assert n == k


def test_recorded_answers_match_decode_answers(fixture):
    recorded_path = os.path.join(HERE, "fixtures", "recorded_answers.json")
    if not os.path.isfile(recorded_path):
        pytest.skip("recorded answers missing")
    with open(recorded_path, encoding="utf-8") as fh:
        recorded = json.load(fh)
    temps = decode.Temperatures(fixture["cfg"], clamp=True)
    ids = [item["qid"] for item in fixture["items"]]
    internal = {qid: to_internal(fixture["questions"][qid]) for qid in ids}
    act_probs = torch.softmax(torch.tensor(fixture["act_logits"]), -1).numpy()
    answers = decode.decode_answers(ids, internal, fixture["items"], np.asarray(fixture["logits"], dtype=np.float32), act_probs, temps)
    for qid in ids:
        rec = recorded["answers"][qid]
        got = answers[qid]
        assert got["type"] == rec["type"]
        for key in ("choice", "score", "noul", "confidence", "answer_confidence"):
            if key in rec:
                if isinstance(rec[key], float):
                    assert got[key] == pytest.approx(rec[key], abs=2e-3)
                else:
                    assert got[key] == rec[key]


def test_usage_fields_from_items(fixture):
    ids = [item["qid"] for item in fixture["items"]]
    n_tokens = sum(len(item["ids"]) for item in fixture["items"])
    usage = decode.usage_for_state(ids, fixture["items"], n_tokens)
    assert usage["input_tokens"] == n_tokens
    assert usage["output_tokens"] == 0
    assert usage["state_tokens"] == fixture["items"][0]["state_stats"]["state_tokens"]
    assert usage["state_tokens_dropped"] == max(i["state_stats"]["state_tokens_dropped"] for i in fixture["items"])
    assert usage["truncated"] == (usage["state_tokens_dropped"] > 0)
    assert set(usage["truncated_questions"]) == {qid for qid, i in zip(ids, fixture["items"]) if i["state_stats"]["truncated"]}


def test_pad_batch_rule():
    b = {
        "input_ids": torch.tensor([[5, 6, 7], [8, 9, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "marker_pos": torch.tensor([[1, 2], [1, 0]]),
        "marker_mask": torch.tensor([[True, True], [True, False]]),
        "qtype": torch.tensor([2, 0]),
    }
    ids, att, mpos, mmask, qt = pad_batch(b, rows=4, seq=8, pad_id=50283, keep_one_token=True)
    assert ids.shape == (4, 8) and att.shape == (4, 8) and mpos.shape == (4, 2) and mmask.shape == (4, 2) and qt.shape == (4,)
    assert ids[:2, :3].tolist() == b["input_ids"].tolist()
    assert (ids[:, 3:] == 50283).all() and (ids[2:] == 50283).all()
    assert att[:2, :3].tolist() == b["attention_mask"].tolist()
    assert att[2:, 0].tolist() == [1, 1] and att[2:, 1:].sum() == 0 and att[:2, 3:].sum() == 0
    assert mmask[2:].tolist() == [[True, False], [True, False]]
    assert (mpos[2:] == 0).all() and (qt[2:] == 0).all()
    _, att2, _, _, _ = pad_batch(b, rows=4, seq=8, pad_id=50283, keep_one_token=False)
    assert att2[2:].sum() == 0


def test_buckets_and_chunk_plan():
    buckets = Buckets([1, 2, 4, 8], [128, 256, 512])
    assert buckets.rows_for(3) == 4 and buckets.rows_for(8) == 8 and buckets.seq_for(1) == 128 and buckets.seq_for(300) == 512
    with pytest.raises(LimitError):
        buckets.rows_for(9)
    with pytest.raises(RequestError):
        buckets.seq_for(513)
    lengths = [500, 100, 120, 260, 90, 510, 130, 140, 150]
    plan = plan_chunks(lengths, buckets, max_rows=8, max_batch_tokens=1024)
    seen = sorted(i for idx, _, _ in plan for i in idx)
    assert seen == list(range(len(lengths)))
    for idx, rows, seq in plan:
        assert rows == buckets.rows_for(len(idx))
        assert seq == buckets.seq_for(max(lengths[i] for i in idx))
        assert rows * seq <= 1024 or len(idx) == 1
    plan_rows = plan_chunks([10] * 9, buckets, max_rows=4, max_batch_tokens=10**9)
    assert [len(idx) for idx, _, _ in plan_rows] == [4, 4, 1]
    derived = Buckets.from_shapes({"warm_shapes": [[1, 512], [8, 512], [4, 256]]})
    assert derived.rows == [1, 4, 8] and derived.seqs == [256, 512]


@pytest.mark.parametrize(
    "qdef",
    [
        {"type": "pick", "instructions": "x", "criteria": ["a"]},
        {"type": "choice", "criteria": ["a"]},
        {"type": "choice", "instructions": "", "criteria": ["a"]},
        {"type": "choice", "instructions": "x"},
        {"type": "choice", "instructions": "x", "criteria": []},
        {"type": "choice", "instructions": "x", "criteria": ["a", "a"]},
        {"type": "choice", "instructions": "x", "criteria": [None]},
        {"type": "score", "instructions": "x", "criteria": {"a": "b"}},
        {"type": "score", "instructions": "x", "criteria": ["a", None]},
        {"type": "noul", "instructions": "x", "criteria": {"yes": "y"}},
        {"type": "noul", "instructions": "x", "criteria": ["a"]},
        {"type": "noul", "instructions": "x", "labels": {"false": "B", "true": "A"}},
        {"type": "choice", "instructions": "x", "criteria": ["a", "b"], "option_order": [0, 0]},
        "not a dict",
    ],
)
def test_check_question_rejects_malformed(qdef):
    with pytest.raises(RequestError):
        check_question("q", qdef)


def test_check_question_accepts_valid_and_to_internal():
    check_question("c", {"type": "choice", "instructions": "x", "criteria": ["a", "b"], "option_order": [1, 0]})
    check_question("s", {"type": "score", "instructions": {"k": "v"}, "criteria": ["lo", "hi"]})
    check_question("n", {"type": "noul", "instructions": "x", "criteria": {"True": "yes"}})
    q = to_internal({"type": "choice", "instructions": {"k": "v"}, "criteria": ["a", "b"], "option_order": [1, 0]})
    assert q["crit"] == {"a": None, "b": None} and q["ins"] == '{"k": "v"}' and q["option_order"] == [1, 0]
    assert to_internal({"type": "noul", "instructions": "x", "criteria": {"True": "yes"}})["crit"] == {"true": "yes"}


def test_min_confidence_validation_and_gate():
    assert decode.check_min_confidence(0.3) == 0.3
    assert decode.check_min_confidence({"choice:3-5": 0.6, "default": 0.1}) == {"choice:3-5": 0.6, "default": 0.1}
    for bad in (-0.1, 1.5, True, "0.5", float("nan"), {}, {"x": 2.0}):
        with pytest.raises(ValueError):
            decode.check_min_confidence(bad)
    results = [
        {
            "answers": {
                "a": {"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}, "answer_confidence": 0.7},
                "b": {"type": "noul", "noul": 0.55, "answer_confidence": 0.55},
                "c": {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.5},
                "d": {"type": "score", "probabilities": {"0": 0.5, "1": 0.5}},
            }
        }
    ]
    decode.apply_confidence_gate(results, None)
    assert "abstention" not in results[0]["answers"]["a"]
    decode.apply_confidence_gate(results, 0.6)
    a, b, c, d = (results[0]["answers"][k] for k in "abcd")
    assert a["abstention"] == decode.GATE_PASSED and "low_confidence" not in a
    assert b["abstention"] == decode.GATE_ABSTAINED and b["low_confidence"] is True
    assert c["abstention"] == decode.GATE_ABSTAINED and c["abstention_threshold"] == 0.6
    assert d["abstention"] == decode.GATE_UNEVALUATED and "low_confidence" not in d
    decode.apply_confidence_gate(results, 0.0)
    assert all(v["abstention"] == decode.GATE_PASSED and "low_confidence" not in v for k, v in results[0]["answers"].items() if k != "d")
    assert results[0]["answers"]["d"]["abstention"] == decode.GATE_UNEVALUATED
    decode.apply_confidence_gate(results, {"choice:2": 0.9, "default": 0.0})
    assert results[0]["answers"]["a"]["abstention"] == decode.GATE_ABSTAINED
    assert results[0]["answers"]["a"]["abstention_threshold"] == 0.9
    assert results[0]["answers"]["b"]["abstention"] == decode.GATE_PASSED
    gone = [{"answers": {"z": {"type": "choice", "probabilities": {"x": 1.0}, "answer_confidence": float("nan")}}}]
    decode.apply_confidence_gate(gone, 0.5)
    assert gone[0]["answers"]["z"]["abstention"] == decode.GATE_UNEVALUATED


def test_confidence_helpers():
    p = np.array([0.5, 0.25, 0.25])
    assert decode.answer_confidence(p, 3) == 0.5
    assert math.isclose(confidence_from_probs(p, 3), 1 - (-(p * np.log(p)).sum()) / math.log(3))
    assert decode.unpermute_probs(np.array([0.1, 0.9]), [1, 0]).tolist() == [0.9, 0.1]
    assert decode.render_criterion({"a": 1}) == '{"a": 1}' and decode.render_criterion("x") == "x"
