# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("LAYA_CPU_THREADS", "4")
os.environ["LAYA_BACKEND"] = "cpu"
os.environ.setdefault("LAYA_MODEL_DIR", "/home/hous/dev/laya/state/laya_models/laya")
os.environ.setdefault("LAYA_RAW_FORWARD", "1")

from models.autoports.convaiinnovations_laya.server import app as server_app
from models.autoports.convaiinnovations_laya.server.app import (
    BATCH_HEADER,
    DEVICE_HEADER,
    LATENCY_HEADER,
    Q_CHOICE,
    Q_NOUL,
    SANITY_FILE,
    STATE_EN,
    create_app,
)
from models.autoports.convaiinnovations_laya.server.engine import Engine, collate_items, encode_state, to_internal

MODEL_DIR = os.environ["LAYA_MODEL_DIR"]
needs_weights = pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")), reason="weights missing: %s" % MODEL_DIR
)
SHORT_STATE = {"ticket": "Payout failed twice this week, please fix it today."}
TONE = {"type": "score", "instructions": "What is the customer's tone?", "criteria": ["angry", "neutral", "happy"]}


@pytest.fixture(scope="session")
def cpu_engine():
    if not os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")):
        pytest.skip("weights missing: %s" % MODEL_DIR)
    return Engine.from_env()


@pytest.fixture(scope="session")
def client(cpu_engine):
    with TestClient(create_app(engine=cpu_engine, raw_forward=True, demo=True, sanity=True)) as c:
        yield c


def probs_of(answer):
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def test_health_while_loading():
    with TestClient(create_app(load=False, sanity=False, demo=False)) as c:
        assert c.get("/health").status_code == 503
        r = c.get("/v1/health")
        assert r.status_code == 503 and r.json()["ready"] is False
        assert c.post("/v1/systemone", json={"state": "x", "questions": {"q": Q_NOUL}}).status_code == 503


@needs_weights
def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["ready"] is True and r.json()["backend"] == "cpu"
    h = client.get("/v1/health").json()
    assert h["status"] == "ok" and h["backend"] == "cpu"
    assert h["seq_buckets"] and h["row_buckets"] and h["max_len"] == 512 and h["head_max_len"] == 192
    assert h["raw_forward"] is True and h["precision"] == "fp32"
    assert h["temperatures"]["clamp"] is True
    assert h["sanity"] is not None and h["sanity"]["choice"]
    if os.path.isfile(SANITY_FILE):
        assert h["sanity"]["ok"] is True


@needs_weights
def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data[0]["id"] == "convaiinnovations/laya" and data[0]["backend"] == "cpu"
    assert LATENCY_HEADER in r.headers


@needs_weights
def test_state_en_example(client):
    r = client.post("/v1/systemone", json={"state": STATE_EN, "questions": {"routing": Q_CHOICE, "urgent": Q_NOUL}})
    assert r.status_code == 200, r.text
    for h in (LATENCY_HEADER, DEVICE_HEADER, BATCH_HEADER):
        assert h in r.headers
    assert float(r.headers[LATENCY_HEADER]) > 0 and float(r.headers[DEVICE_HEADER]) > 0
    assert r.headers[BATCH_HEADER].count("x") == 1
    body = r.json()
    assert body["model"] == "laya-rl-agent"
    a = body["answers"]["routing"]
    assert a["type"] == "choice" and a["choice"] in Q_CHOICE["criteria"]
    assert list(a["probabilities"]) == list(Q_CHOICE["criteria"])
    assert abs(sum(a["probabilities"].values()) - 1.0) <= 1e-3
    assert a["answer_confidence"] == pytest.approx(max(a["probabilities"].values()), abs=1e-4)
    assert 0.0 <= a["confidence"] <= 1.0 and 0.0 <= a["action"]["act_probability"] <= 1.0
    n = body["answers"]["urgent"]
    assert n["type"] == "noul" and 0.0 <= n["noul"] <= 1.0
    assert n["confidence"] == pytest.approx(max(n["noul"], 1 - n["noul"]), abs=1e-4)
    assert n["answer_confidence"] == n["confidence"]
    u = body["usage"]
    assert u["output_tokens"] == 0 and u["input_tokens"] > u["state_tokens"] > 0
    assert u["truncated"] is False and u["state_tokens_dropped"] == 0 and u["truncated_questions"] == []
    assert "abstention" not in a
    if os.path.isfile(SANITY_FILE):
        with open(SANITY_FILE, encoding="utf-8") as fh:
            ref = json.load(fh)["answers"]["routing"]
        assert a["choice"] == ref["choice"]
        assert max(abs(a["probabilities"][k] - ref["probabilities"][k]) for k in ref["probabilities"]) <= 1e-3


@needs_weights
def test_score_question_and_truncation(client):
    long_state = {"ticket": "The customer is furious about the double charge. " * 80}
    r = client.post("/v1/systemone", json={"state": long_state, "questions": {"tone": TONE}, "max_len": 128})
    assert r.status_code == 200, r.text
    body = r.json()
    s = body["answers"]["tone"]
    assert s["type"] == "score" and 0.0 <= s["score"] <= 2.0
    assert s["legend"] == {"0": "angry", "1": "neutral", "2": "happy"}
    assert abs(sum(s["probabilities"].values()) - 1.0) <= 1e-3
    assert body["usage"]["truncated"] is True and body["usage"]["state_tokens_dropped"] > 0
    assert body["usage"]["truncated_questions"] == ["tone"]


@needs_weights
def test_batch_and_chunking(client, cpu_engine):
    states = [SHORT_STATE, {"ticket": "Can I get a demo of the enterprise plan?"}, {"ticket": "The API returns 500 on every call since the deploy."}]
    body = {"states": states, "questions": {"routing": Q_CHOICE}}
    r = client.post("/v1/systemone/batch", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["results"]) == 3
    assert out["total_usage"]["input_tokens"] == sum(x["usage"]["input_tokens"] for x in out["results"])
    assert r.headers[BATCH_HEADER].startswith("4x")
    saved = cpu_engine.limits.max_rows
    cpu_engine.limits.max_rows = 2
    try:
        r2 = client.post("/v1/systemone/batch", json=body)
    finally:
        cpu_engine.limits.max_rows = saved
    assert r2.status_code == 200, r2.text
    assert len(r2.headers[BATCH_HEADER].split(",")) == 2
    for a, b in zip(out["results"], r2.json()["results"]):
        pa, pb = a["answers"]["routing"], b["answers"]["routing"]
        assert pa["choice"] == pb["choice"]
        assert max(abs(pa["probabilities"][k] - pb["probabilities"][k]) for k in pa["probabilities"]) <= 1e-3


@needs_weights
def test_min_confidence(client):
    q = {"routing": Q_CHOICE}
    r = client.post("/v1/systemone", json={"state": SHORT_STATE, "questions": q, "min_confidence": 0.999})
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["routing"]
    assert a["low_confidence"] is True and a["abstention"] == "abstained" and a["abstention_threshold"] == 0.999
    r = client.post("/v1/systemone", json={"state": SHORT_STATE, "questions": q, "min_confidence": 0.0})
    a = r.json()["answers"]["routing"]
    assert a["abstention"] == "passed" and "low_confidence" not in a and a["abstention_threshold"] == 0.0
    r = client.post("/v1/systemone", json={"state": SHORT_STATE, "questions": q, "min_confidence": {"choice:3-5": 0.999}})
    assert r.json()["answers"]["routing"]["abstention"] == "abstained"


@needs_weights
@pytest.mark.parametrize(
    "body",
    [
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": ["a"]}}},
        {"state": "x", "questions": {"q": {"type": "pick", "instructions": "x"}}},
        {"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["a", None]}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "x", "labels": {"false": "B", "true": "A"}}}},
        {"state": "x", "questions": {"q": Q_NOUL}, "min_confidence": 1.5},
        {"state": "x", "questions": {"q": Q_NOUL}, "max_len": 100000},
        {"state": "x", "questions": {"q": Q_NOUL}, "max_len": "512"},
        {"state": "x", "questions": {"q": Q_NOUL}, "hooks": ["x"]},
    ],
)
def test_422_on_malformed_questions(client, body):
    r = client.post("/v1/systemone", json=body)
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], str)


@needs_weights
def test_413_on_too_many(client):
    r = client.post("/v1/systemone/batch", json={"states": ["x"] * 65, "questions": {"q": Q_NOUL}})
    assert r.status_code == 413
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q%d" % i: Q_NOUL for i in range(65)}})
    assert r.status_code == 413
    r = client.post("/v1/systemone", json={"state": "y" * 50001, "questions": {"q": Q_NOUL}})
    assert r.status_code == 413
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["o%d" % i for i in range(101)]}}})
    assert r.status_code == 413


@needs_weights
def test_400_on_bad_envelope(client):
    assert client.post("/v1/systemone", json={"questions": {"q": Q_NOUL}}).status_code == 400
    assert client.post("/v1/systemone", json={"state": "x", "questions": []}).status_code == 400
    assert client.post("/v1/systemone", json={"state": "x"}).status_code == 400
    assert client.post("/v1/systemone/batch", json={"states": [], "questions": {"q": Q_NOUL}}).status_code == 400
    r = client.post("/v1/systemone", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


@needs_weights
def test_empty_questions(client):
    r = client.post("/v1/systemone", json={"state": "x", "questions": {}})
    assert r.status_code == 200
    assert r.json() == {"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


@needs_weights
def test_raw_forward_matches_wire_path(client, cpu_engine):
    questions = {"routing": Q_CHOICE}
    ids = list(questions)
    internal = {q: to_internal(questions[q]) for q in ids}
    items = encode_state(cpu_engine.tok, SHORT_STATE, ids, internal, cpu_engine.max_len, cpu_engine.head_max_len)
    b = collate_items([items], cpu_engine.pad_id)
    payload = {k: b[k].tolist() for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")}
    r = client.post("/v1/forward", json=payload)
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["logits"]) == 1 and len(out["logits"][0]) == 3 and len(out["act_logits"][0]) == 2
    assert BATCH_HEADER in r.headers and out["batch"] == r.headers[BATCH_HEADER]
    wire = client.post("/v1/systemone", json={"state": SHORT_STATE, "questions": questions}).json()["answers"]["routing"]
    temps = cpu_engine.temps
    from models.autoports.convaiinnovations_laya.server.decode import scaled_softmax

    p = scaled_softmax(out["logits"][0], temps.for_question(0, 3))
    for k, v in zip(Q_CHOICE["criteria"], p):
        assert abs(wire["probabilities"][k] - v) <= 2e-3
    bad = dict(payload)
    bad["qtype"] = [7]
    assert client.post("/v1/forward", json=bad).status_code == 422


@needs_weights
def test_raw_forward_disabled_without_flag(cpu_engine):
    with TestClient(create_app(engine=cpu_engine, raw_forward=False, demo=False, sanity=False)) as c:
        assert c.post("/v1/forward", json={}).status_code in (404, 405)
        assert c.get("/demo/").status_code == 404


@needs_weights
def test_api_key(cpu_engine):
    with TestClient(create_app(engine=cpu_engine, api_key="secret", demo=False, sanity=False)) as c:
        assert c.get("/v1/models").status_code == 401
        assert c.post("/v1/systemone", json={"state": "x", "questions": {"q": Q_NOUL}}).status_code == 401
        assert c.get("/health").status_code == 200
        h = c.get("/v1/health").json()
        assert h["api_key_required"] is True and "seq_buckets" not in h
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.get("/v1/health", headers={"Authorization": "Bearer secret"}).json()["seq_buckets"]


def test_module_level_app_exists():
    assert server_app.app is not None
    paths = {r.path for r in server_app.app.routes}
    assert {"/health", "/v1/health", "/v1/models", "/v1/systemone", "/v1/systemone/batch"} <= paths
