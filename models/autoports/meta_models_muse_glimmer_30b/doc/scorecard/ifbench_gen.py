"""Generate IFBench responses from the served model, in the scorer's input format.

usage: ifbench_gen.py BASE_URL OUT.jsonl [--seed S] [--temperature T] [--limit N] [--conc C]

Writes one {"prompt": <exact prompt string>, "response": <message.content>} line per
IFBench_test prompt. The prompt string is copied verbatim from data/IFBench_test.jsonl
because run_eval.py keys responses on exact string equality. ``response`` is the
chat-completions ``content`` field: the server's reasoning parser has already removed
the analysis channel, which is what the IFBench paper asks for with reasoning models.
A sidecar OUT.meta.json records empties, cap hits and errors.
"""
import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("base_url")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=64)
ap.add_argument("--max_tokens", type=int, default=32768)
ap.add_argument("--conc", type=int, default=32)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--data", default=str(pathlib.Path(__file__).parent / "IFBench/data/IFBench_test.jsonl"))
a = ap.parse_args()

prompts = [json.loads(l)["prompt"] for l in open(a.data)]
if a.limit:
    prompts = prompts[: a.limit]


def ask(prompt):
    body = {
        "model": "meta-models/Muse-Glimmer-30B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": a.temperature,
        "max_tokens": a.max_tokens,
        "stream": False,
        "seed": a.seed,
    }
    if a.temperature > 0:
        body["top_p"] = a.top_p
        body["top_k"] = a.top_k
    last = None
    for attempt in range(3):
        try:
            r = json.load(
                urllib.request.urlopen(
                    urllib.request.Request(a.base_url, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                    timeout=3600,
                )
            )
            ch = r["choices"][0]
            return {
                "prompt": prompt,
                "response": ch["message"].get("content") or "",
                "_finish": ch.get("finish_reason"),
                "_ctok": r.get("usage", {}).get("completion_tokens"),
                "_reasoning_chars": len(ch["message"].get("reasoning_content") or ""),
            }
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    return {"prompt": prompt, "response": "", "_finish": "ERROR", "_ctok": 0, "_error": str(last)[:200]}


t0 = time.time()
with ThreadPoolExecutor(max_workers=a.conc) as ex:
    rows = list(ex.map(ask, prompts))
out = pathlib.Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps({"prompt": r["prompt"], "response": r["response"]}) + "\n")
meta = {
    "n": len(rows),
    "seed": a.seed,
    "temperature": a.temperature,
    "max_tokens": a.max_tokens,
    "empty": sum(1 for r in rows if not r["response"].strip()),
    "capped": sum(1 for r in rows if r["_finish"] == "length"),
    "errors": sum(1 for r in rows if r["_finish"] == "ERROR"),
    "median_completion_tokens": sorted(r["_ctok"] or 0 for r in rows)[len(rows) // 2],
    "seconds": round(time.time() - t0, 1),
}
json.dump(meta, open(str(out) + ".meta.json", "w"), indent=1)
print("IFBENCH_GEN " + json.dumps(meta))
