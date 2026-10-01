# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("CLM_RUN_HF") != "1", reason="set CLM_RUN_HF=1 to load Qwen3-8B on CPU")

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
README_ANSWERS = {"urgency": 0.41022, "billing": 0.93878, "frustration": 1.98386}


@pytest.fixture(scope="module")
def embedder():
    from models.autoports.contrastive_lm_clm_v0_1_8b.reference.hf_embedder import HfQwen3Embedder

    t0 = time.perf_counter()
    emb = HfQwen3Embedder()
    print(f"\nhf embedder loaded in {time.perf_counter() - t0:.1f} s: {emb.info()}")
    return emb


def test_hf_embedder_two_texts(embedder):
    texts = ["Hello world", README_STATE]
    t0 = time.perf_counter()
    vecs, tokens = embedder.embed(texts)
    dt = time.perf_counter() - t0
    print(f"embed 2 texts: {dt:.2f} s, tokens={tokens}")
    assert vecs.shape == (2, 4096) and vecs.dtype == np.float32
    assert np.all(np.isfinite(vecs))
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert tokens == len(embedder.tokenize(texts[0])) + len(embedder.tokenize(texts[1]))
    cos = float(vecs[0] @ vecs[1])
    print(f"cos(hello, readme) = {cos:.4f}")
    assert cos < 0.999


def test_hf_embedder_identical_texts_identical_vectors(embedder):
    vecs, _ = embedder.embed([README_STATE, README_STATE])
    assert np.array_equal(vecs[0], vecs[1])
    again, _ = embedder.embed([README_STATE])
    assert np.array_equal(again[0], vecs[0])


def test_hf_embedder_truncates_to_last_tokens(embedder):
    ids = list(range(1000, 1000 + embedder.max_tokens + 50))
    kept = embedder.truncate(ids)
    assert len(kept) == embedder.max_tokens - 1
    assert kept[-1] == ids[-1]
    assert embedder.tokenize("") == embedder.tokenize(" ")


def test_hf_embedder_readme_example_through_engine(embedder):
    from models.autoports.contrastive_lm_clm_v0_1_8b.clm.engine import Engine
    from models.autoports.contrastive_lm_clm_v0_1_8b.tt.heads import DEFAULT_CHECKPOINT

    if not os.path.isfile(DEFAULT_CHECKPOINT):
        pytest.skip(f"checkpoint missing: {DEFAULT_CHECKPOINT}")
    engine = Engine(embedder, checkpoint=DEFAULT_CHECKPOINT, device="cpu", action_cache="64MiB")
    t0 = time.perf_counter()
    out = engine.answer(README_STATE, README_QUESTIONS)
    dt = time.perf_counter() - t0
    a = out["answers"]
    got = {
        "urgency": a["urgency"]["noul"],
        "billing": a["department"]["probabilities"]["billing"],
        "frustration": a["frustration"]["score"],
    }
    print(
        f"readme example in {dt * 1000:.0f} ms, input_tokens={out['usage']['input_tokens']} (README: 106 cold): "
        f"got {got} vs README {README_ANSWERS} (RTX 4090, vLLM bf16, texts not reproducible from the published commit)"
    )
    assert out["usage"]["billing_units"] == 3 and out["usage"]["input_tokens"] > 0
    assert a["department"]["choice"] == "billing"
    assert a["department"]["probabilities"]["billing"] > 0.5
    assert a["frustration"]["score"] > 1.0
    assert 0.0 <= a["urgency"]["noul"] <= 1.0
    again = engine.answer(README_STATE, README_QUESTIONS)
    assert again["answers"] == a and again["usage"]["input_tokens"] == 0
