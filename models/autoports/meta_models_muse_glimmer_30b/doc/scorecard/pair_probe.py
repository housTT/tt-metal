"""Pair probe: N=2 bursts with controlled prompt-length relationships.

Arms (T trials each, both requests launched together, max_tokens=64):
  same      : the identical prompt twice           -> identical padded length, identical content
  samelen   : two DIFFERENT prompts whose token counts pad to the same 32-row tile
  difflen   : two prompts whose token counts pad to DIFFERENT tiles
Reports, per arm, how often each launch position opens the channel correctly (' to=self').
usage: pair_probe.py BASE_URL TRIALS
"""
import glob
import json
import sys
import threading
import urllib.request

BASE = sys.argv[1]
T = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SP = "/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad"
f = sorted(glob.glob(f"{SP}/ab2_apc_off/**/samples_aime25_*.jsonl", recursive=True))[-1]
rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["doc_id"])
PROMPTS = [json.loads(r["arguments"]["gen_args_0"]["arg_0"][0])[0]["content"] for r in rows]
TOK = BASE.rsplit("/v1/", 1)[0] + "/tokenize"


def ntok(p):
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": p}],
        "add_generation_prompt": True,
    }
    r = json.load(
        urllib.request.urlopen(
            urllib.request.Request(TOK, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=60
        )
    )
    return r["count"]


lens = [ntok(p) for p in PROMPTS]
tile = [((n + 31) // 32) * 32 for n in lens]
print("prompt token counts:", lens)
print("padded tiles       :", tile)


def ask(i):
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": PROMPTS[i]}],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 64,
        "stream": False,
        "seed": 42,
    }
    r = json.load(
        urllib.request.urlopen(
            urllib.request.Request(BASE, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600
        )
    )
    m = r["choices"][0]["message"]
    return ((m.get("reasoning_content") or "") + (m.get("content") or "")).startswith(" to=self")


def pair(a, b):
    out = [None, None]

    def go(k, i):
        out[k] = ask(i)

    ta = threading.Thread(target=go, args=(0, a))
    tb = threading.Thread(target=go, args=(1, b))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    return out


# choose pairs
same = (0, 0)
samelen = next(((i, j) for i in range(len(PROMPTS)) for j in range(i + 1, len(PROMPTS)) if tile[i] == tile[j]), None)
difflen = next(
    (
        (i, j)
        for i in range(len(PROMPTS))
        for j in range(i + 1, len(PROMPTS))
        if tile[i] != tile[j] and abs(tile[i] - tile[j]) >= 64
    ),
    None,
)
arms = {"same": same, "samelen": samelen, "difflen": difflen}
print("pairs:", {k: (v, (tile[v[0]], tile[v[1]]) if v else None) for k, v in arms.items()})

for name, pr in arms.items():
    if pr is None:
        print(f"{name}: no pair available")
        continue
    ok0 = ok1 = 0
    for t in range(T):
        r = pair(*pr)
        ok0 += r[0]
        ok1 += r[1]
    print(
        json.dumps(
            {
                "arm": name,
                "pair": pr,
                "tiles": [tile[pr[0]], tile[pr[1]]],
                "trials": T,
                "first_ok": ok0,
                "second_ok": ok1,
                "both_ok": None,
                "first_ok_rate": round(ok0 / T, 2),
                "second_ok_rate": round(ok1 / T, 2),
            }
        ),
        flush=True,
    )
print("PAIR_DONE")
