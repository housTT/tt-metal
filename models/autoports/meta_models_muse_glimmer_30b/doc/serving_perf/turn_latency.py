#!/usr/bin/env python3
"""Per-turn TTFT against conversation length, for an agent-shaped transcript.

Prefix caching is disabled on this backend, so every turn re-prefills the whole
conversation.  This measures what that costs as an agent loop grows its history.

The transcript is shaped like a real agent run -- user, assistant tool_call,
tool result, assistant text -- so the chat template's multi-turn and tool-role
paths are exercised, not just a long single prompt.

Each shape is sent TWICE and only the second is timed.  tt-metal JIT-compiles
per prefill shape; without the warmup the first call at a new length absorbs
compilation and TTFT is meaningless.
"""
import json
import time

import requests

URL = "http://127.0.0.1:20000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
ROUNDS = [1, 2, 4, 8, 16, 32]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a source file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    }
]

# ~2000 tokens of plausible file content per round.
FILLER = (
    "    def resolve_layer_kind(config, idx):\n"
    "        # sliding layers alternate with full-attention layers\n"
    "        return LAYER_KIND_SLIDING if idx % 4 else LAYER_KIND_FULL\n"
) * 60


def transcript(rounds: int) -> list:
    msgs = [{"role": "system", "content": "You are a coding agent. Be terse."}]
    for i in range(rounds):
        msgs.append({"role": "user", "content": f"Read src/mod_{i}.py and note the layer kinds."})
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/mod_{i}.py"})},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"call_{i}", "content": f"# src/mod_{i}.py\n{FILLER}"})
        msgs.append({"role": "assistant", "content": f"mod_{i} alternates sliding and full."})
    msgs.append({"role": "user", "content": "In one word, which layer kind dominates?"})
    return msgs


def one(msgs, timed: bool):
    body = {
        "model": MODEL,
        "messages": msgs,
        "tools": TOOLS,
        "tool_choice": "none",
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    prompt_tokens = None
    with requests.post(URL, json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                prompt_tokens = d["usage"]["prompt_tokens"]
            ch = d.get("choices") or []
            if ch and ttft is None:
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return ttft, total, prompt_tokens


def main():
    print(f"{'rounds':>7} {'prompt tok':>11} {'TTFT ms':>10} {'total ms':>10} {'ms/1k tok':>10}", flush=True)
    print("-" * 52, flush=True)
    rows = []
    for r in ROUNDS:
        msgs = transcript(r)
        try:
            one(msgs, timed=False)  # warm this prefill shape
            ttft, total, ptok = one(msgs, timed=True)  # measure
        except Exception as e:
            print(f"{r:>7}  FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        if ttft is None:
            ttft = total
        per1k = ttft / (ptok / 1000.0) if ptok else 0.0
        rows.append(
            dict(rounds=r, prompt_tokens=ptok, ttft_ms=ttft * 1000, total_ms=total * 1000, ms_per_1k=per1k * 1000)
        )
        print(f"{r:>7} {ptok:>11,} {ttft*1000:>10,.1f} {total*1000:>10,.1f} {per1k*1000:>10.1f}", flush=True)
    out = "/home/ttuser/dev/muse-glimmer/logs/turn_latency.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nwrote {out}", flush=True)
    if len(rows) >= 2:
        cum = sum(r["ttft_ms"] for r in rows) / 1000.0
        print(f"cumulative TTFT across the {len(rows)} measured turns: {cum:,.1f} s", flush=True)


if __name__ == "__main__":
    main()
