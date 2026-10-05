# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import re

import pytest

from models.autoports.convaiinnovations_laya.server.demo import DEMO_DIR, DEMO_FILES, load_feed, load_presets, validate_feed, validate_preset
from models.autoports.convaiinnovations_laya.tests.test_server_cpu import client, cpu_engine, needs_weights, probs_of

DASHES = re.compile("[\\u2013\\u2014]")


def test_demo_files_present_and_dash_free():
    for name in DEMO_FILES + ("presets.json", "feed_cases.json"):
        path = os.path.join(DEMO_DIR, name)
        assert os.path.isfile(path), path
        if name == "feed_cases.json":
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        assert not DASHES.search(text), "em or en dash in %s" % name
    with open(os.path.join(DEMO_DIR, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    external = re.compile(r'(src|href)="https?://|url\(\s*["\']?https?://|@import\s+["\']?https?://|fetch\(\s*["\']https?://')
    for name in DEMO_FILES:
        with open(os.path.join(DEMO_DIR, name), encoding="utf-8") as fh:
            assert not external.search(fh.read()), "external asset reference in %s" % name
    assert 'src="demo.js"' in html and 'href="demo.css"' in html


def test_presets_and_feed_validate():
    presets = load_presets()
    assert len(presets) >= 5
    ids = [p["id"] for p in presets]
    assert len(ids) == len(set(ids))
    for p in presets:
        validate_preset(p)
    assert any(p.get("gold") for p in presets)
    feed = load_feed()
    validate_feed(feed)
    assert feed["attribution"]["dataset"] == "LocalLLaMA/typed-decisions"
    assert feed["attribution"]["revision"] == "d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18"
    workflows = {c["workflow"] for c in feed["cases"]}
    assert len(workflows) == 4
    assert all(len(c["questions"]) == 5 for c in feed["cases"])


@needs_weights
def test_demo_routes(client):
    r = client.get("/demo", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/demo/"
    r = client.get("/demo/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "demo.js?v=" in r.text and "demo.css?v=" in r.text
    assert r.headers["cache-control"] == "no-cache"
    for name in ("demo.js", "demo.css"):
        s = client.get("/demo/" + name)
        assert s.status_code == 200 and s.headers["cache-control"] == "no-cache"
    p = client.get("/demo/presets.json")
    assert p.status_code == 200 and isinstance(p.json(), list)
    f = client.get("/demo/feed.json")
    assert f.status_code == 200 and 40 <= len(f.json()["cases"]) <= 100
    assert client.get("/demo/missing.txt").status_code == 404


@needs_weights
def test_every_preset_decides(client):
    for preset in load_presets():
        body = {"state": preset["state"], "questions": preset["questions"]}
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200, "%s: %s" % (preset["id"], r.text)
        answers = r.json()["answers"]
        assert set(answers) == set(preset["questions"]), preset["id"]
        for qid, a in answers.items():
            assert a["type"] == preset["questions"][qid]["type"]
            p = probs_of(a)
            assert abs(sum(p) - 1.0) <= 1e-3, (preset["id"], qid, p)
            assert all(0.0 <= x <= 1.0 for x in p)
            assert 0.0 <= a["answer_confidence"] <= 1.0
        gold = preset.get("gold")
        if gold:
            assert set(gold) == set(answers)
            for qid, g in gold.items():
                if a := answers.get(qid):
                    if a["type"] == "choice":
                        assert set(a["probabilities"]) == set(g["probabilities"])


@needs_weights
def test_feed_case_decides_with_gold_keys(client):
    case = load_feed()["cases"][0]
    r = client.post("/v1/systemone", json={"state": case["state"], "questions": case["questions"]})
    assert r.status_code == 200, r.text
    answers = r.json()["answers"]
    for qid, g in case["gold"].items():
        a = answers[qid]
        if a["type"] == "choice":
            assert g["label"] in a["probabilities"]
        elif a["type"] == "score":
            assert str(g["label"]) in a["probabilities"]
        else:
            assert str(g["label"]).lower() in ("true", "false")
