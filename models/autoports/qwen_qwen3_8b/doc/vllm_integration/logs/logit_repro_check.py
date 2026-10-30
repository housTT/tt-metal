#!/usr/bin/env python3
"""Logit reproducibility through the live vLLM server.

One short prompt, greedy with top-20 logprobs (host-sampled on one chip because logprobs are requested),
alone and beside 3, 10 and 20 greedy fillers (decode buckets 1, 8, 16 and 24), two runs each. For every
generated position the top-20 logprob vector is compared run to run (same batch shape) and across batch
shapes against the solo run: identical token sets, maximum absolute logprob difference, and the top-1 /
top-2 margin. Also two runs of the mixed-params batch shape with the host-routed seeded request
(top_k 100, temperature 2.0, seed 42) to see whether host-sampled seeded output reproduces when the
batch shape is fixed.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

SERVER = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
MODEL = "Qwen/Qwen3-8B"
PROMPT = "The three laws of thermodynamics are"


def completion(prompt, **kw):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 6, "temperature": 0.0}
    payload.update(kw)
    r = requests.post(f"{SERVER}/v1/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]


def with_fillers(n, **kw):
    with ThreadPoolExecutor(max_workers=n + 1) as pool:
        fillers = [pool.submit(completion, f"Filler prompt number {i} says", max_tokens=6) for i in range(n)]
        main = pool.submit(completion, PROMPT, logprobs=20, **kw)
        res = main.result()
        for f in fillers:
            f.result()
    return res


def vectors(choice):
    lp = choice["logprobs"]
    return [dict(sorted(d.items())) for d in lp["top_logprobs"]], lp["tokens"]


def compare(a, b):
    va, ta = vectors(a)
    vb, tb = vectors(b)
    rows = []
    for i in range(min(len(va), len(vb))):
        keys = set(va[i]) | set(vb[i])
        same_set = set(va[i]) == set(vb[i])
        diffs = [abs(va[i].get(k, -1e9) - vb[i].get(k, -1e9)) for k in keys if k in va[i] and k in vb[i]]
        sa = sorted(va[i].values(), reverse=True)
        rows.append(
            {
                "position": i,
                "token_a": ta[i],
                "token_b": tb[i],
                "same_top20_set": same_set,
                "max_abs_logprob_diff": max(diffs) if diffs else None,
                "top1_top2_margin_a": round(sa[0] - sa[1], 4) if len(sa) > 1 else None,
            }
        )
    return rows


def main():
    record = {"prompt": PROMPT, "shapes": {}, "seeded_host_routed": {}}
    runs = {}
    for n in (0, 3, 10, 20):
        runs[n] = [with_fillers(n) if n else completion(PROMPT, logprobs=20) for _ in range(2)]
        record["shapes"][n] = {
            "texts": [r["text"] for r in runs[n]],
            "run_to_run": compare(runs[n][0], runs[n][1]),
            "vs_solo_run0": compare(runs[0][0], runs[n][0]),
        }
        print(
            n,
            [r["text"] for r in runs[n]],
            "run-to-run max diff",
            max((x["max_abs_logprob_diff"] or 0) for x in record["shapes"][n]["run_to_run"]),
            "vs solo max diff",
            max((x["max_abs_logprob_diff"] or 0) for x in record["shapes"][n]["vs_solo_run0"]),
            flush=True,
        )
    seeded = dict(temperature=2.0, top_k=100, seed=42, max_tokens=5)
    for n in (0, 3):
        outs = []
        for rep in range(2):
            if n:
                with ThreadPoolExecutor(max_workers=n + 1) as pool:
                    fillers = [
                        pool.submit(completion, f"Filler prompt number {i} says", max_tokens=5) for i in range(n)
                    ]
                    main_ = pool.submit(completion, "Random: ", **seeded)
                    outs.append(main_.result()["text"])
                    for f in fillers:
                        f.result()
            else:
                outs.append(completion("Random: ", **seeded)["text"])
        record["seeded_host_routed"][n] = {"texts": outs, "identical": outs[0] == outs[1]}
        print("seeded host-routed with", n, "fillers:", outs, flush=True)
    summary = {
        "greedy_texts_identical_across_all": len({t for s in record["shapes"].values() for t in s["texts"]}) == 1,
        "max_run_to_run_logprob_diff": max(
            (x["max_abs_logprob_diff"] or 0) for s in record["shapes"].values() for x in s["run_to_run"]
        ),
        "max_cross_shape_logprob_diff": max(
            (x["max_abs_logprob_diff"] or 0) for s in record["shapes"].values() for x in s["vs_solo_run0"]
        ),
        "all_top20_sets_identical_run_to_run": all(
            x["same_top20_set"] for s in record["shapes"].values() for x in s["run_to_run"]
        ),
        "all_top20_sets_identical_cross_shape": all(
            x["same_top20_set"] for s in record["shapes"].values() for x in s["vs_solo_run0"]
        ),
    }
    record["summary"] = summary
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
