# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import hashlib
import math
import os

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from models.autoports.contrastive_lm_clm_v0_1_8b.clm.cache import VectorArena
from models.autoports.contrastive_lm_clm_v0_1_8b.clm.engine import (
    DEFAULT_MODEL,
    RAW_MODEL,
    RELEASE,
    EmbedderLike,
    Engine,
)
from models.autoports.contrastive_lm_clm_v0_1_8b.clm.heads import HIDDEN, PROJ_DIM
from models.autoports.contrastive_lm_clm_v0_1_8b.clm.schema import build_pairs
from models.autoports.contrastive_lm_clm_v0_1_8b.server.app import LATENCY_HEADER, create_app
from models.autoports.contrastive_lm_clm_v0_1_8b.tt import heads as tt_heads

CKPT = tt_heads.DEFAULT_CHECKPOINT
PARAMS_PER_HEAD = 4096 * 1536 + 1536 + 1536 * 1536 + 1536 + 2 * 1536 + 1536 * 512 + 512
README_STATE = "Customer: my invoice was charged twice and nobody answers the phone!"
README_QUESTIONS = {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"},
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}
RANK_STATE = "What causes tides on Earth?"
RANK_CANDIDATES = ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."]

needs_ckpt = pytest.mark.skipif(not os.path.isfile(CKPT), reason=f"checkpoint missing: {CKPT}")


def unit_vector(seed: bytes) -> np.ndarray:
    n = int.from_bytes(hashlib.blake2b(seed, digest_size=8).digest(), "little")
    v = np.random.default_rng(n).standard_normal(HIDDEN, dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    max_tokens = 2048
    name = "fake-qwen3-8b"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vecs = np.stack([unit_vector(t.encode("utf-8")) for t in texts]).astype(np.float32)
        return vecs, sum(max(1, len(t) // 4) for t in texts)

    def healthy(self):
        return True


class FakeEmbedderWithIds(FakeEmbedder):
    def embed_ids(self, id_lists):
        self.calls += 1
        vecs = np.stack([unit_vector(b"ids:" + ",".join(map(str, ids)).encode()) for ids in id_lists]).astype(
            np.float32
        )
        return vecs, sum(len(ids) for ids in id_lists)


class FakeEmbedderArrayIds(FakeEmbedderWithIds):
    max_tokens = 8

    def embed_ids(self, id_lists):
        vecs, _ = super().embed_ids(id_lists)
        return vecs

    def stats(self):
        return {"calls": self.calls}


class FakeTokenizer:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


@pytest.fixture(scope="module")
def heads():
    if not os.path.isfile(CKPT):
        pytest.skip(f"checkpoint missing: {CKPT}")
    return tt_heads.load_heads(CKPT, device="cpu")


@pytest.fixture(scope="module")
def engine():
    if not os.path.isfile(CKPT):
        pytest.skip(f"checkpoint missing: {CKPT}")
    return Engine(FakeEmbedderWithIds(), checkpoint=CKPT, device="cpu", action_cache="64MiB")


@pytest.fixture(scope="module")
def client(engine):
    with TestClient(create_app(engine, ui=True)) as c:
        yield c


def probs_sum(d: dict) -> float:
    return sum(d.values())


@needs_ckpt
def test_heads_load_checkpoint(heads):
    info = tt_heads.head_info(heads)
    assert info["n_params"] == 2 * PARAMS_PER_HEAD == 18_887_680
    assert abs(info["n_params"] - 2 * 9_440_000) < 20_000
    assert info["scale"] == 100.0
    assert info["proj_dim"] == PROJ_DIM == 512
    assert info["hidden"] == HIDDEN == 4096
    cfg = info["cfg"]
    assert (cfg["width"], cfg["depth"], cfg["activation"], cfg["layernorm"], cfg["residual"]) == (
        1536,
        3,
        "gelu",
        True,
        False,
    )
    assert cfg["projection_dim"] == 512
    assert heads.device == "cpu"


@needs_ckpt
def test_heads_project_normalized(heads, expect_error):
    x = np.stack([unit_vector(f"proj{i}".encode()) for i in range(3)])
    zs = tt_heads.project_states(heads, x)
    za = tt_heads.project_actions(heads, x)
    assert zs.shape == za.shape == (3, 512)
    assert zs.dtype == np.float32
    assert np.allclose(np.linalg.norm(zs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(za, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(zs, za)
    one = tt_heads.project_states(heads, x[0])
    assert one.shape == (1, 512)
    assert np.allclose(one[0], zs[0], atol=1e-6)
    s = tt_heads.scores(heads, x[:1], x)
    assert s.shape == (1, 3)
    assert np.all(np.abs(s) <= 100.0 + 1e-4)
    with expect_error(ValueError, "expected embeddings of shape"):
        tt_heads.project_states(heads, np.zeros((2, 5), dtype=np.float32))


def test_build_pairs_readme_example():
    pairs = build_pairs(README_STATE, README_QUESTIONS)
    assert list(pairs) == ["urgency", "department", "frustration"]
    state, keys, texts = pairs["urgency"]
    assert state == "Customer: my invoice was charged twice and nobody answers the phone!\n\nIs this urgent?"
    assert keys == ["false", "true"]
    assert texts == ["false: No. This is false: Is this urgent?", "true: Yes. This is true: Is this urgent?"]
    state, keys, texts = pairs["department"]
    assert (
        state
        == "Customer: my invoice was charged twice and nobody answers the phone!\n\nWhich team should handle this?"
    )
    assert keys == ["billing", "technical"]
    assert texts == ["Charges, invoices, refunds", "Bugs and outages"]
    state, keys, texts = pairs["frustration"]
    assert (
        state
        == "Customer: my invoice was charged twice and nobody answers the phone!\n\nHow frustrated is the customer?"
    )
    assert keys == ["0", "1", "2"]
    assert texts == ["Calm", "Frustrated", "Very angry"]


def test_fake_embedder_satisfies_protocol():
    emb = FakeEmbedder()
    assert isinstance(emb, EmbedderLike)
    vecs, tokens = emb.embed(["a", "a", "b"])
    assert vecs.shape == (3, HIDDEN) and vecs.dtype == np.float32
    assert np.array_equal(vecs[0], vecs[1]) and not np.array_equal(vecs[0], vecs[2])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert tokens == 3


@needs_ckpt
def test_engine_models_match_upstream(engine):
    models = engine.models()
    assert [m["name"] for m in models] == [DEFAULT_MODEL, RAW_MODEL] == ["clm-latest", "clm-raw"]
    assert all(m["release_date"] == RELEASE == "2026-09-19" for m in models)
    assert all(m["description"] for m in models)
    assert engine.has("clm-latest") and engine.has("clm-raw") and not engine.has("nope")


@needs_ckpt
def test_engine_answers_readme_example(engine):
    out = engine.answer(README_STATE, README_QUESTIONS)
    assert out["model"] == "clm-latest"
    assert out["usage"]["billing_units"] == 3
    assert out["usage"]["output_tokens"] == 0
    assert out["usage"]["input_tokens"] > 0
    a = out["answers"]
    assert set(a) == {"urgency", "department", "frustration"}
    assert a["urgency"]["type"] == "noul" and 0.0 <= a["urgency"]["noul"] <= 1.0
    dep = a["department"]
    assert dep["type"] == "choice" and dep["choice"] in ("billing", "technical")
    assert set(dep["probabilities"]) == {"billing", "technical"}
    assert math.isclose(probs_sum(dep["probabilities"]), 1.0, abs_tol=1e-9)
    assert 0.0 <= dep["confidence"] <= 1.0
    fr = a["frustration"]
    assert fr["type"] == "score" and 0.0 <= fr["score"] <= 2.0
    assert fr["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert math.isclose(probs_sum(fr["probabilities"]), 1.0, abs_tol=1e-9)
    again = engine.answer(README_STATE, README_QUESTIONS)
    assert again["answers"] == a


@needs_ckpt
def test_engine_rank_and_temperature(engine, expect_error):
    ranked = engine.rank(RANK_STATE, RANK_CANDIDATES)
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert sorted(r["candidate"] for r in ranked) == sorted(RANK_CANDIDATES)
    probs = [r["prob"] for r in ranked]
    assert probs == sorted(probs, reverse=True)
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
    hot = engine.rank(RANK_STATE, RANK_CANDIDATES, temperature=4.0)
    cold = engine.rank(RANK_STATE, RANK_CANDIDATES, temperature=0.25)
    assert hot[0]["prob"] < ranked[0]["prob"] < cold[0]["prob"]
    p = np.array([dict((r["candidate"], r["prob"]) for r in ranked)[c] for c in RANK_CANDIDATES])
    for t, res in ((4.0, hot), (0.25, cold)):
        q = np.array([dict((r["candidate"], r["prob"]) for r in res)[c] for c in RANK_CANDIDATES])
        expect = p ** (1.0 / t)
        assert np.allclose(q, expect / expect.sum(), atol=1e-6)
    with expect_error(ValueError, "temperature must be in"):
        engine.rank(RANK_STATE, RANK_CANDIDATES, temperature=0.0)
    with expect_error(ValueError, "questions must not be empty"):
        engine.answer(README_STATE, {})


@needs_ckpt
def test_engine_raw_model(engine, expect_error):
    out = engine.answer(README_STATE, README_QUESTIONS, model="clm-raw")
    assert out["model"] == "clm-raw"
    assert math.isclose(probs_sum(out["answers"]["department"]["probabilities"]), 1.0, abs_tol=1e-9)
    with expect_error(KeyError, "unknown model"):
        engine.answer(README_STATE, README_QUESTIONS, model="clm-nope")


def test_vector_arena_cpu_hits():
    arena = VectorArena("cpu", "64MiB")
    assert str(arena.device) == "cpu"
    assert arena.flat.numel() * 4 == 64 << 20
    pool = arena.reserve(512, 0.875)
    raw = arena.reserve(HIDDEN, 0.125)
    assert pool is not None and raw is not None
    floats = (64 << 20) // 4
    assert pool.capacity == int(floats * 0.875) // 512 == 28_672
    assert raw.capacity == int(floats * 0.125) // HIDDEN == 512
    assert arena.stats()["pools"]["512"]["reserved_mb"] == round(28_672 * 512 * 4 / 10**6, 1)
    calls = []

    def compute(texts):
        calls.append(list(texts))
        return torch.stack([torch.from_numpy(unit_vector(t.encode())[:512]) for t in texts])

    first = arena.get("ns", 512, ["a", "b", "a"], compute)
    assert first.shape == (3, 512) and calls == [["a", "b"]]
    second = arena.get("ns", 512, ["b", "a"], compute)
    assert len(calls) == 1
    assert torch.equal(second[0], first[1]) and torch.equal(second[1], first[0])
    st = arena.stats()
    assert st["pools"]["512"]["hits"] == 2 and st["pools"]["512"]["misses"] == 2
    assert st["hit_rate"] > 0
    arena.get("other", 512, ["a"], compute)
    assert len(calls) == 2


@needs_ckpt
def test_engine_arena_caches_repeated_call(engine):
    emb = engine.embedder
    state = "arena state: the robot is in room 7 with the door closed"
    questions = {"go": {"type": "choice", "instructions": "Where next?", "criteria": {"n": "north", "s": "south"}}}
    before = emb.calls
    first = engine.answer(state, questions)
    assert emb.calls > before and first["usage"]["input_tokens"] > 0
    mid = emb.calls
    second = engine.answer(state, questions)
    assert emb.calls == mid
    assert second["usage"]["input_tokens"] == 0
    assert second["answers"] == first["answers"]
    st = engine.arena.stats()
    assert st["hit_rate"] > 0
    assert set(st["pools"]) == {"512", "4096"}


@needs_ckpt
def test_app_health_and_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and j["ready"] is True and j["embedder"] is True
    assert j["models"] == ["clm-latest", "clm-raw"]
    assert set(j["cache"]["pools"]) == {"512", "4096"}
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert [m["name"] for m in r.json()["models"]] == ["clm-latest", "clm-raw"]
    assert all(m["release_date"] == "2026-09-19" for m in r.json()["models"])
    assert float(r.headers[LATENCY_HEADER]) >= 0.0


@needs_ckpt
def test_app_systemone(client):
    r = client.post("/v1/systemone", json={"state": README_STATE, "model": "clm-latest", "questions": README_QUESTIONS})
    assert r.status_code == 200
    assert float(r.headers[LATENCY_HEADER]) >= 0.0
    j = r.json()
    assert j["model"] == "clm-latest" and set(j["answers"]) == {"urgency", "department", "frustration"}
    assert math.isclose(probs_sum(j["answers"]["department"]["probabilities"]), 1.0, abs_tol=1e-9)
    assert j["usage"]["billing_units"] == 3
    r = client.post("/v1/systemone", json={"state": README_STATE, "questions": README_QUESTIONS, "model": "clm-nope"})
    assert r.status_code == 422
    r = client.post("/v1/systemone", json={"questions": README_QUESTIONS})
    assert r.status_code == 422
    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "bogus"}}})
    assert r.status_code == 422
    r = client.post("/v1/systemone", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


@needs_ckpt
def test_app_rank(client):
    r = client.post("/v1/rank", json={"context": RANK_STATE, "question": None, "answers": RANK_CANDIDATES})
    assert r.status_code == 200
    assert float(r.headers[LATENCY_HEADER]) >= 0.0
    j = r.json()
    assert j["model"] == "clm-latest"
    assert [x["rank"] for x in j["ranked"]] == [1, 2, 3]
    assert math.isclose(sum(x["prob"] for x in j["ranked"]), 1.0, abs_tol=1e-9)
    r = client.post("/v1/rank", json={"context": RANK_STATE, "answers": []})
    assert r.status_code == 422
    r = client.post("/v1/rank", json={"context": RANK_STATE, "answers": ["ok", ""]})
    assert r.status_code == 422


@needs_ckpt
def test_app_embeddings_float_and_base64(client, engine):
    texts = ["hello world", README_STATE]
    r = client.post("/v1/embeddings", json={"model": "qwen3-8b", "input": texts})
    assert r.status_code == 200
    assert float(r.headers[LATENCY_HEADER]) >= 0.0
    j = r.json()
    assert j["object"] == "list" and j["model"] == "qwen3-8b"
    assert [d["index"] for d in j["data"]] == [0, 1]
    assert all(d["object"] == "embedding" and len(d["embedding"]) == HIDDEN for d in j["data"])
    flt = np.array([d["embedding"] for d in j["data"]], dtype=np.float32)
    assert np.allclose(np.linalg.norm(flt, axis=1), 1.0, atol=1e-5)
    expect, tokens = engine.embedder.embed(texts)
    assert np.allclose(flt, expect, atol=1e-6)
    assert j["usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}
    r = client.post("/v1/embeddings", json={"input": texts, "encoding_format": "base64"})
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "fake-qwen3-8b"
    b64 = np.stack([np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32) for d in j["data"]])
    assert b64.shape == (2, HIDDEN)
    assert np.array_equal(b64, expect)
    r = client.post("/v1/embeddings", json={"input": "hello world"})
    assert r.status_code == 200 and len(r.json()["data"]) == 1
    assert np.allclose(np.array(r.json()["data"][0]["embedding"], dtype=np.float32), expect[0], atol=1e-6)


@needs_ckpt
def test_app_embeddings_token_ids(client, engine):
    r = client.post("/v1/embeddings", json={"input": [[1, 2, 3], [4, 5]]})
    assert r.status_code == 200
    j = r.json()
    expect, tokens = engine.embedder.embed_ids([[1, 2, 3], [4, 5]])
    got = np.array([d["embedding"] for d in j["data"]], dtype=np.float32)
    assert np.allclose(got, expect, atol=1e-6)
    assert j["usage"]["prompt_tokens"] == tokens == 5
    r = client.post("/v1/embeddings", json={"input": [7, 8, 9], "truncate_prompt_tokens": 2})
    assert r.status_code == 200
    expect, tokens = engine.embedder.embed_ids([[8, 9]])
    assert np.allclose(np.array(r.json()["data"][0]["embedding"], dtype=np.float32), expect[0], atol=1e-6)
    assert r.json()["usage"]["prompt_tokens"] == 2


@needs_ckpt
def test_app_embeddings_token_ids_decoded_without_embed_ids():
    eng = Engine(FakeEmbedder(), checkpoint=CKPT, device="cpu", action_cache="0")
    assert eng.arena is None
    with TestClient(create_app(eng, ui=False, tokenizer=FakeTokenizer(), warmup=False)) as c:
        r = c.post("/v1/embeddings", json={"input": [[10, 20, 30]]})
        assert r.status_code == 200
        expect, _ = eng.embedder.embed(["10 20 30"])
        assert np.allclose(np.array(r.json()["data"][0]["embedding"], dtype=np.float32), expect[0], atol=1e-6)
        assert c.get("/").status_code == 404
        assert c.get("/health").json()["cache"] is None


@needs_ckpt
def test_app_embeddings_array_returning_embed_ids_and_default_truncation():
    emb = FakeEmbedderArrayIds()
    eng = Engine(emb, checkpoint=CKPT, device="cpu", action_cache="0")
    with TestClient(create_app(eng, ui=False, warmup=False)) as c:
        ids = list(range(100, 112))
        r = c.post("/v1/embeddings", json={"input": ids})
        assert r.status_code == 200
        expect = FakeEmbedderWithIds().embed_ids([ids[-8:]])[0]
        assert np.allclose(np.array(r.json()["data"][0]["embedding"], dtype=np.float32), expect[0], atol=1e-6)
        assert r.json()["usage"] == {"prompt_tokens": 8, "total_tokens": 8}
        r = c.post("/v1/embeddings", json={"input": [[1, 2, 3]], "truncate_prompt_tokens": 50})
        assert r.status_code == 200 and r.json()["usage"]["prompt_tokens"] == 3
        r = c.post("/v1/embeddings", json={"input": ["hello world"], "truncate_prompt_tokens": 2})
        assert r.status_code == 200
        assert np.allclose(
            np.array(r.json()["data"][0]["embedding"], dtype=np.float32),
            FakeEmbedder().embed(["hello world"])[0][0],
            atol=1e-6,
        )
        h = c.get("/health").json()
        assert h["embedder_stats"] == {"calls": emb.calls} and h["cache"] is None


@needs_ckpt
def test_app_embeddings_rejects_bad_input(client):
    for body in (
        {"input": []},
        {"input": [1, "a"]},
        {"input": [[]]},
        {"input": "x", "encoding_format": "hex"},
        {"input": "x", "truncate_prompt_tokens": 0},
        {"input": "x", "dimensions": 128},
        {"model": "qwen3-8b"},
    ):
        r = client.post("/v1/embeddings", json=body)
        assert r.status_code == 422, body
        assert LATENCY_HEADER in r.headers


@needs_ckpt
def test_app_playground_static(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CLM Playground" in r.text and 'href="app.css?v=' in r.text and 'src="app.js?v=' in r.text
    assert r.headers["Cache-Control"] == "no-cache"
    r = client.get("/app.js")
    assert r.status_code == 200 and r.headers["Cache-Control"] == "no-cache"
    assert "/v1/systemone" in r.text


@needs_ckpt
def test_app_api_key(engine):
    with TestClient(create_app(engine, api_key="secret", ui=False, warmup=False)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.post("/v1/embeddings", json={"input": "x"}).status_code == 401
        r = c.post("/v1/embeddings", json={"input": "x"}, headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200
