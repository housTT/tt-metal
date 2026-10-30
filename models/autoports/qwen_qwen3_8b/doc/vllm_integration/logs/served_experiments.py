#!/usr/bin/env python3
"""Served experiments on a live server: prefix-cache hits near the context end (with decode replays after
each), the served greedy path on the rendered qualitative prompt ids against the generator control, and
concurrent host-sampled seeded requests in different submission orders. Writes one JSON."""

from __future__ import annotations

import bz2
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from transformers import AutoTokenizer

SERVER = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
MODEL = "Qwen/Qwen3-8B"
SNAP = "/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)


def comp(prompt, **kw):
    p = {"model": MODEL, "prompt": prompt, "max_tokens": 4, "temperature": 0.0}
    p.update(kw)
    t = time.perf_counter()
    r = requests.post(f"{SERVER}/v1/completions", json=p, timeout=900)
    ms = round((time.perf_counter() - t) * 1000)
    body = r.json()
    if "choices" not in body:
        return {"status": r.status_code, "ms": ms, "error": json.dumps(body)[:200]}
    return {"status": r.status_code, "ms": ms, "text": body["choices"][0]["text"]}


def div(a, b):
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), n)


record = {}
book = tok.encode(
    bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt").read()[:300000], add_special_tokens=False
)
requests.post(f"{SERVER}/reset_prefix_cache")
near = {}
near["cold_40000"] = comp(book[:40000])
near["hit_40000_plus_956"] = comp(book[:40956])
near["repeat_40956"] = comp(book[:40956])
near["hit_40000_plus_900"] = comp(book[:40900])
requests.post(f"{SERVER}/reset_prefix_cache")
near["cold_36869"] = comp(book[:36869])
near["hit_36869_plus_4087"] = comp(book[:40956])
requests.post(f"{SERVER}/reset_prefix_cache")
near["cold_38000"] = comp(book[:38000])
near["hit_38000_plus_2933"] = comp(book[:40933], max_tokens=27)
requests.post(f"{SERVER}/reset_prefix_cache")
near["cold_32"] = comp(book[:32])
near["hit_32_plus_39924"] = comp(book[:39956])
record["near_context_hits"] = near
print(json.dumps(near, indent=1), flush=True)

rendered = json.load(open("models/autoports/qwen_qwen3_8b/doc/optimized_full_model/qualitative/rendered_prompts.json"))
control = {
    e["id"]: e["token_ids"]
    for e in json.load(
        open("models/autoports/qwen_qwen3_8b/doc/vllm_integration/qualitative_control/chip1/tt_outputs.json")
    )
}
served = []
for e in rendered:
    ctrl = control[e["id"]]
    a = comp(e["token_ids"], max_tokens=128)
    b = comp(e["token_ids"], max_tokens=128, top_k=1, top_p=1.0)
    r = requests.post(
        f"{SERVER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": e["prompt"]}],
            "max_tokens": 128,
            "temperature": 0.0,
        },
        timeout=600,
    ).json()
    chat_text = r["choices"][0]["message"]["content"]
    ids_a = tok.encode(a["text"], add_special_tokens=False)
    ids_b = tok.encode(b["text"], add_special_tokens=False)
    ids_c = tok.encode(chat_text, add_special_tokens=False)
    rendered_text = tok.apply_chat_template(
        [{"role": "user", "content": e["prompt"]}], tokenize=False, add_generation_prompt=True
    )
    row = {
        "id": e["id"],
        "ids_greedy_vs_control": div(ids_a, ctrl),
        "ids_topk1_vs_control": div(ids_b, ctrl),
        "chat_vs_control": div(ids_c, ctrl),
        "chat_vs_ids_greedy": div(ids_c, ids_a),
        "rendered_ids_match": tok.encode(rendered_text, add_special_tokens=False) == e["token_ids"],
        "control_tokens": len(ctrl),
    }
    served.append(row)
    print(json.dumps(row), flush=True)
record["served_vs_control"] = served

A = dict(prompt="Random: ", temperature=2.0, top_k=100, seed=42, max_tokens=5)
B = dict(prompt="Other: ", temperature=2.0, top_k=100, seed=7, max_tokens=5)
C = dict(prompt="Letter: ", temperature=0.01, seed=42, max_tokens=3)
pairs = {
    "solo_A": [comp(**A)["text"] for _ in range(2)],
    "solo_B": [comp(**B)["text"] for _ in range(2)],
    "solo_C": [comp(**C)["text"] for _ in range(2)],
}
for order in ("AB", "BA", "ABC", "CBA", "ACB", "AAB"):
    outs = []
    for rep in range(2):
        with ThreadPoolExecutor(max_workers=3) as pool:
            fs = [pool.submit(comp, **{"A": A, "B": B, "C": C}[c]) for c in order]
            outs.append([f.result().get("text", f.result().get("error")) for f in fs])
    pairs[order] = {"runs": outs, "identical": outs[0] == outs[1]}
    print(order, outs, outs[0] == outs[1], flush=True)
record["seeded_pairs"] = pairs
json.dump(record, open(OUT, "w"), indent=2)
print("EXPERIMENTS_DONE", flush=True)
