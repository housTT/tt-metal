import math
import os
import threading

import pytest

ADAPTER = (
    "/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0"
)
BASE = (
    "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404"
)
os.environ["KEV_FAKE_ENGINE"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("KEV_RUN", ADAPTER if os.path.isdir(ADAPTER) else "jaredpalmer/kev-9b")
os.environ.setdefault("HF_MODEL", BASE if os.path.isdir(BASE) else "Qwen/Qwen3.5-9B-Base")
os.environ.setdefault("KEV_FAKE_WORKERS", "2")

from fastapi.testclient import TestClient

from models.autoports.jaredpalmer_kev_9b.tt import server as srv

DEPARTMENT = {
    "returns": "Exchanges, refunds, wrong or damaged items",
    "shipping": "Delivery status, delays, lost packages",
    "billing": "Charges, invoices, payment problems",
}
QUICKSTART = {
    "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    "model": "kev-latest",
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?", "criteria": DEPARTMENT},
        "escalate": {"type": "noul", "instructions": "Does this need urgent human attention?"},
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


@pytest.fixture(scope="module")
def client():
    with TestClient(srv.app) as c:
        yield c


def post(client, body, **kw):
    r = client.post("/v1/systemone", json=body, **kw)
    assert r.headers["x-typesafe-request-id"]
    return r.status_code, r.json()


def test_choice_noul_score_round_trip(client):
    code, r = post(client, QUICKSTART)
    assert code == 200
    assert set(r) == {"model", "answers", "usage", "latency_ms"}
    assert r["model"] == "kev-latest"
    assert set(r["usage"]) == {"input_tokens", "output_tokens"}
    assert r["usage"]["input_tokens"] > 0 and r["usage"]["output_tokens"] > 0 and r["latency_ms"] >= 0
    c = r["answers"]["department"]
    assert set(c) == {"type", "choice", "confidence", "probabilities"} and c["type"] == "choice"
    assert c["choice"] in DEPARTMENT and set(c["probabilities"]) == set(DEPARTMENT)
    assert math.isclose(sum(c["probabilities"].values()), 1.0, abs_tol=0.03) and 0 <= c["confidence"] <= 1
    assert c["choice"] == max(c["probabilities"], key=c["probabilities"].get)
    n = r["answers"]["escalate"]
    assert n == {"type": "noul", "noul": n["noul"]} and 0 <= n["noul"] <= 1
    s = r["answers"]["frustration"]
    assert set(s) == {"type", "score", "legend", "probabilities", "confidence"} and s["type"] == "score"
    assert s["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"} and set(s["probabilities"]) == {
        "0",
        "1",
        "2",
    }
    assert 0 <= s["score"] <= 2 and 0 <= s["confidence"] <= 1
    assert math.isclose(s["score"], sum(int(k) * v for k, v in s["probabilities"].items()), abs_tol=0.05)


def test_request_id_is_echoed(client):
    r = client.post("/v1/systemone", json=QUICKSTART, headers={"x-typesafe-request-id": "abc123"})
    assert r.status_code == 200 and r.headers["x-typesafe-request-id"] == "abc123"
    assert r.headers["server-timing"].startswith("app;dur=")


def test_object_state_and_noul_criteria(client):
    code, r = post(
        client,
        {
            "state": {"document": "I was charged twice. Please fix this ASAP."},
            "model": "jev-latest",
            "questions": {
                "billing": {
                    "type": "noul",
                    "instructions": "Is this ticket about billing?",
                    "criteria": {"true": "Explicitly about charges", "false": "Not about charges"},
                },
                "urgency": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["today"]},
            },
        },
    )
    assert code == 200 and r["model"] == "jev-latest"
    assert r["answers"]["urgency"] == {
        "type": "score",
        "score": 0.0,
        "legend": {"0": "today"},
        "probabilities": {"0": 1.0},
        "confidence": 1.0,
    }


def test_same_request_is_deterministic_and_cached(client):
    first = post(client, QUICKSTART)[1]
    second = post(client, QUICKSTART)[1]
    assert first["answers"] == second["answers"]
    card = client.get("/v1/models").json()["models"][0]
    assert card["prefix_cache"]["hits"] >= 1


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    cards = r.json()["models"]
    assert [c["name"] for c in cards] == ["kev-latest", "jev-latest"]
    for c in cards:
        assert c["description"] and c["release_date"]
        assert c["backend"] == "fake" and c["max_state_tokens"] == 65536 and c["truncate_states"] is False
        assert set(c["prefix_cache"]) == {"size", "hits", "misses", "cached_states"}
        assert len(c["workers"]) == 2


def test_health(client):
    for path in ("/health", "/v1/health"):
        r = client.get(path)
        assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["workers"] == 2


def test_validation_422(client):
    assert (
        post(
            client,
            {"state": "x", "model": "m", "questions": {"q": {"type": "score", "instructions": "i", "criteria": []}}},
        )[0]
        == 422
    )
    assert (
        post(client, {"state": "x", "model": "m", "questions": {"q": {"type": "bogus", "instructions": "i"}}})[0] == 422
    )
    assert post(client, {"state": "x", "model": "m", "questions": {}})[0] == 422
    assert (
        post(
            client,
            {
                "state": "x",
                "model": "m",
                "questions": {
                    "q": {"type": "choice", "instructions": "i", "criteria": {f"o{i}": None for i in range(256)}}
                },
            },
        )[0]
        == 422
    )


def test_oversize_state_422(client):
    code, r = post(client, {**QUICKSTART, "state": "lorem " * 70000})
    assert code == 422
    assert "over the 65,536-token limit" in r["detail"] and "KEV_TRUNCATE_STATES=1" in r["detail"]


def test_prefix_cache_eviction_and_revisit():
    settings = srv.Settings.from_env()
    settings.fake_workers = 1
    settings.prefix_cache = 2
    s = srv.Server.start(settings)
    try:
        w = s.workers[0]
        assert w.cache_size == 2 and w.engine.snapshot_slots == 2
        reqs = [
            srv.SystemOneRequest.model_validate({**QUICKSTART, "state": f"Ticket {i}. {QUICKSTART['state']}"})
            for i in range(3)
        ]
        fresh = [s.answer(r)["answers"] for r in reqs]
        hits, misses = w.hits, w.misses
        assert (hits, misses, len(w.cache)) == (0, 4, 2)
        assert s.answer(reqs[2])["answers"] == fresh[2] and (w.hits, w.misses) == (hits + 1, misses)
        assert s.answer(reqs[0])["answers"] == fresh[0] and (w.hits, w.misses) == (hits + 1, misses + 1)
        assert s.answer(reqs[1])["answers"] == fresh[1] and (w.hits, w.misses) == (hits + 1, misses + 2)
        assert s.answer(reqs[0])["answers"] == fresh[0] and (w.hits, w.misses) == (hits + 2, misses + 2)
        assert len(w.cache) == 2 and sorted(slot for _, slot in w.cache.values()) == [0, 1] and w.free_slots == []
    finally:
        s.close()


def test_failed_prefill_returns_slot(expect_error):
    settings = srv.Settings.from_env()
    settings.fake_workers = 1
    settings.prefix_cache = 1
    s = srv.Server.start(settings)
    try:
        w = s.workers[0]
        real = w.engine.prefill_state
        state = {"n": 0}

        def flaky(ids, slot=0):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("device hiccup")
            return real(ids, slot=slot)

        w.engine.prefill_state = flaky
        bad = srv.SystemOneRequest.model_validate({**QUICKSTART, "state": "A brand new state that is not cached."})
        with expect_error(RuntimeError, "device hiccup"):
            s.answer(bad)
        assert w.free_slots == [0] and len(w.cache) == 0
        ok = s.answer(bad)
        assert ok["answers"]["department"]["type"] == "choice" and len(w.cache) == 1 and w.free_slots == []
        other = srv.SystemOneRequest.model_validate({**QUICKSTART, "state": "Another state after the failure."})
        assert s.answer(other)["answers"]["department"]["type"] == "choice" and len(w.cache) == 1
    finally:
        s.close()


def test_truncate_mode_marks_responses():
    settings = srv.Settings.from_env()
    settings.truncate = True
    settings.fake_workers = 1
    s = srv.Server.start(settings)
    try:
        body = s.answer(srv.SystemOneRequest.model_validate({**QUICKSTART, "state": "lorem " * 70000}))
        assert body["truncated"] is True
        assert body["usage"]["state_tokens_used"] == 65536 and body["usage"]["state_tokens"] > 65536
        assert set(body["usage"]) == {"input_tokens", "output_tokens", "state_tokens", "state_tokens_used"}
        short = s.answer(srv.SystemOneRequest.model_validate(QUICKSTART))
        assert short["truncated"] is False and short["usage"]["state_tokens"] == short["usage"]["state_tokens_used"]
    finally:
        s.close()


def test_auth(client, monkeypatch):
    monkeypatch.setattr(srv, "API_KEY", "secret")
    assert client.post("/v1/systemone", json=QUICKSTART).status_code == 401
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/health").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.post("/v1/systemone", json=QUICKSTART, headers={"authorization": "Bearer secret"}).status_code == 200


def test_sync_predict_on_labelled_record(client):
    record = {
        "state": "The weather is nice today.",
        "questions": {
            "a": {"type": "noul", "instructions": "Is the weather nice?", "label": True, "src": "x"},
            "b": {
                "type": "choice",
                "instructions": "Season?",
                "criteria": {"summer": None, "winter": None},
                "label": "summer",
                "src": "x",
            },
        },
    }
    probs, stats = srv.server().predict(record)
    assert len(probs) == 2 and len(probs[0]) == 2 and len(probs[1]) == 2
    assert all(math.isclose(sum(p), 1.0, abs_tol=1e-4) for p in probs)
    assert stats["tokens"] > stats["state_tokens_used"] >= 1 and "latency_ms" in stats and stats["worker"] in (0, 1)


def test_concurrent_requests_spread_over_workers(client):
    codes = []

    def one(i):
        codes.append(post(client, {**QUICKSTART, "state": f"Ticket {i}. {QUICKSTART['state']}"})[0])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes == [200] * 12
    workers = client.get("/v1/models").json()["models"][0]["workers"]
    assert sum(w["requests"] for w in workers) >= 12 and all(w["queued"] == 0 for w in workers)
