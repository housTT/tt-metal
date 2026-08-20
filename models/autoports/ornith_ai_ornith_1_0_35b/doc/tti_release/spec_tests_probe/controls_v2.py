"""Round-2 controls for the TTI VLLMParamConformanceTest failures (Ornith-1.0-35B autoport).

Round 1 (controls.json) compared only `message.content`, which for these prompts is the two-word
final answer. That was too weak to support a reproducibility claim, and it left
`test_non_uniform_seeding` unreplayed. This round records the FULL generated text - reasoning trace
plus content - for every repeat, and adds the missing test.

Read-only against the live external autoport server on port 8100.
"""
import hashlib
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.0-35B"
REPRO_PROMPT = [{"role": "user", "content": "What is the capital of France? Be concise."}]
BASE_PROMPT = [{"role": "user", "content": "Tell me a short joke."}]
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def call(payload, timeout=3600):
    body = dict(payload)
    body["model"] = MODEL
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t
    m = r["choices"][0]["message"]
    reasoning = m.get("reasoning") or ""
    content = m.get("content")
    full = reasoning + (content or "")
    return {
        "elapsed_s": round(dt, 2),
        "finish_reason": r["choices"][0]["finish_reason"],
        "completion_tokens": r["usage"]["completion_tokens"],
        "content": content,
        "reasoning_chars": len(reasoning),
        # the whole generation, hashed, so repeats are compared on everything the model emitted
        "full_generation_sha256": hashlib.sha256(full.encode()).hexdigest(),
        "full_generation_chars": len(full),
    }


def repeat(label, payload, n=3):
    runs = [call(payload) for _ in range(n)]
    hashes = {r["full_generation_sha256"] for r in runs}
    contents = {r["content"] for r in runs}
    return {
        "runs": runs,
        "n": n,
        "full_generation_identical_across_runs": len(hashes) == 1,
        "final_content_identical_across_runs": len(contents) == 1,
        "completion_tokens_seen": sorted(r["completion_tokens"] for r in runs),
        "max_elapsed_s": max(r["elapsed_s"] for r in runs),
        "any_run_exceeds_suite_30s_read_timeout": any(r["elapsed_s"] > 30 for r in runs),
    }


out = {
    "_what_this_measures": (
        "Whether repeated identical requests to the shipped max_num_seqs=32 server reproduce the "
        "WHOLE generation, not just the short final answer, and whether the suite's 30 s client "
        "read timeout is reachable. Round 1 compared only message.content and was too weak."
    ),
    # test_determinism_parameters[temperature-0.0]: greedy, suite payload
    "determinism_temperature_0": repeat("det", {"messages": REPRO_PROMPT, "temperature": 0.0}),
    # test_seed_reproducibility: suite payload
    "seed_reproducibility_seed42_temp0.5": repeat("seed", {"messages": REPRO_PROMPT, "seed": 42, "temperature": 0.5}),
    # test_non_uniform_seeding: the row round 1 never replayed. The suite asserts all seed=0
    # requests agree and seed!=0 requests differ; it failed with 'NoneType' has no attribute
    # 'strip', i.e. content was None. Replay both with thinking on and thinking off.
    "non_uniform_seeding_seed0_thinking_on": repeat(
        "s0on", {"messages": BASE_PROMPT, "seed": 0, "temperature": 0.7, "max_tokens": 64}, n=3
    ),
    "non_uniform_seeding_seed0_thinking_off": repeat(
        "s0off",
        {"messages": BASE_PROMPT, "seed": 0, "temperature": 0.7, "max_tokens": 64, **NO_THINK},
        n=3,
    ),
    # greedy with thinking off, so the comparison is not dominated by a long sampled think trace
    "determinism_temperature_0_thinking_off": repeat(
        "detoff", {"messages": REPRO_PROMPT, "temperature": 0.0, **NO_THINK}, n=3
    ),
}
json.dump(out, sys.stdout, indent=1)
print()
