"""Runaway rate: fraction of requests that hit the token cap instead of stopping."""
import glob
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = sys.argv[1]
LABEL = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 96
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 4096

# real AIME prompts, reused from a recorded eval run
f = sorted(
    glob.glob(
        "/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad/ab2_apc_off/**/samples_aime25_*.jsonl",
        recursive=True,
    )
)[-1]
rows = [json.loads(l) for l in open(f)]
rows.sort(key=lambda r: r["doc_id"])
PROMPTS = [json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])[0]["content"] for r in rows]


def ask(i):
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": CAP,
        "stream": False,
        "seed": 42,
    }
    t = time.time()
    try:
        r = json.load(
            urllib.request.urlopen(
                urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                timeout=1800,
            )
        )
        ch = r["choices"][0]
        return {
            "i": i,
            "doc": i % len(PROMPTS),
            "finish": ch["finish_reason"],
            "ctok": r["usage"]["completion_tokens"],
            "dt": round(time.time() - t, 1),
        }
    except Exception as e:
        return {
            "i": i,
            "doc": i % len(PROMPTS),
            "finish": "ERROR",
            "ctok": 0,
            "dt": round(time.time() - t, 1),
            "err": str(e)[:120],
        }


t0 = time.time()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    res = list(ex.map(ask, range(N)))
ok = [r for r in res if r["finish"] == "stop"]
cap = [r for r in res if r["finish"] == "length"]
err = [r for r in res if r["finish"] == "ERROR"]
toks = sorted(r["ctok"] for r in res if r["ctok"])
out = {
    "label": LABEL,
    "n": N,
    "conc": CONC,
    "cap": CAP,
    "runaway_rate": round(len(cap) / N, 4),
    "stopped": len(ok),
    "hit_cap": len(cap),
    "errors": len(err),
    "median_ctok": toks[len(toks) // 2] if toks else 0,
    "p90_ctok": toks[int(len(toks) * 0.9)] if toks else 0,
    "wall_s": round(time.time() - t0, 1),
}
print(json.dumps(out))
json.dump(
    {"summary": out, "rows": res},
    open(
        f"/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad/runaway_{LABEL}.json",
        "w",
    ),
    indent=1,
)
