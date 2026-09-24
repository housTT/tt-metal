"""First-token degeneration probe.

Fires N requests at a server and classifies each reply by its FIRST tokens only
(max_tokens=64), so one trial takes seconds and many trials fit in minutes.

  burst   : all N requests launched at once  -> N prefills land in one engine step
  stagger : N requests launched GAP seconds apart -> one prefill per step, decode batch grows

A reply is "degenerate" when one token accounts for >= 50% of the first 48 generated
tokens (the ' to to to' signature seen from token 0 in the eval samples).

usage: firsttok_probe.py BASE_URL LABEL MODE N TRIALS [GAP]
"""
import glob
import json
import sys
import threading
import time
import urllib.request
from collections import Counter

BASE, LABEL, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4])
TRIALS = int(sys.argv[5])
GAP = float(sys.argv[6]) if len(sys.argv) > 6 else 1.5
SP = "/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad"

f = sorted(glob.glob(f"{SP}/ab2_apc_off/**/samples_aime25_*.jsonl", recursive=True))[-1]
rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["doc_id"])
PROMPTS = [json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])[0]["content"] for r in rows]


def ask(i):
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 64,
        "stream": False,
        "seed": 42,
    }
    t = time.time()
    try:
        r = json.load(
            urllib.request.urlopen(
                urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                timeout=600,
            )
        )
        m = r["choices"][0]["message"]
        text = (m.get("reasoning_content") or "") + (m.get("content") or "")
        words = text.split()[:48]
        top = Counter(words).most_common(1)[0][1] if words else 0
        degen = bool(words) and top >= max(6, len(words) // 2)
        return {
            "i": i,
            "order": i,
            "degen": degen,
            "ctok": r["usage"]["completion_tokens"],
            "head": text[:60].replace("\n", " "),
            "dt": round(time.time() - t, 2),
        }
    except Exception as e:
        return {
            "i": i,
            "order": i,
            "degen": None,
            "ctok": 0,
            "head": f"ERR {str(e)[:60]}",
            "dt": round(time.time() - t, 2),
        }


def trial(t):
    res = [None] * N

    def go(i):
        res[i] = ask(i)

    threads = []
    for i in range(N):
        th = threading.Thread(target=go, args=(i,))
        th.start()
        threads.append(th)
        if MODE == "stagger":
            time.sleep(GAP)
    for th in threads:
        th.join()
    return res


t0 = time.time()
allres = []
for t in range(TRIALS):
    r = trial(t)
    allres.extend(r)
    d = sum(1 for x in r if x["degen"])
    e = sum(1 for x in r if x["degen"] is None)
    print(
        json.dumps(
            {"trial": t, "degen": d, "n": N, "errors": e, "degen_by_order": [x["order"] for x in r if x["degen"]]}
        ),
        flush=True,
    )
ok = [x for x in allres if x["degen"] is not None]
d = sum(1 for x in ok if x["degen"])
by_order = Counter(x["order"] for x in ok if x["degen"])
summary = {
    "label": LABEL,
    "mode": MODE,
    "n_per_trial": N,
    "trials": TRIALS,
    "gap_s": GAP if MODE == "stagger" else 0,
    "degen_rate": round(d / max(1, len(ok)), 4),
    "degen": d,
    "n": len(ok),
    "errors": len(allres) - len(ok),
    "degen_by_launch_order": dict(sorted(by_order.items())),
    "wall_s": round(time.time() - t0, 1),
}
print("SUMMARY " + json.dumps(summary), flush=True)
json.dump({"summary": summary, "rows": allres}, open(f"{SP}/firsttok_{LABEL}.json", "w"), indent=1)
