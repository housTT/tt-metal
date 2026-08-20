"""Controls for the TTI VLLMParamConformanceTest failures on the Ornith-1.0-35B autoport.

Each failing conformance row is replayed against the same live server with the suite's own
payload, plus a paired control that isolates the cause. Written for the tti-release stage;
run against the external autoport server on port 8100.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL = "ornith-ai/Ornith-1.0-35B"


def call(payload, timeout=3600):
    body = dict(payload)
    body["model"] = MODEL
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t
    ch = r["choices"][0]
    msg = ch["message"]
    return {
        "elapsed_s": round(dt, 2),
        "finish_reason": ch["finish_reason"],
        "completion_tokens": r["usage"]["completion_tokens"],
        "content_is_none": msg.get("content") is None,
        "content": msg.get("content"),
        "reasoning_chars": len(msg.get("reasoning") or ""),
    }


NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
REPRO_PROMPT = [{"role": "user", "content": "What is the capital of France? Be concise."}]
out = {}

# --- test_coherence_verbatim_echo (suite payload: max_tokens 32, temperature 0) -------------
sentence = "The quick brown fox jumps over the lazy dog."
echo_prompt = [
    {
        "role": "user",
        "content": "Repeat the following sentence exactly, with no extra words, no quotes, and no commentary: "
        + sentence,
    }
]
out["coherence_echo__suite_payload"] = call({"messages": echo_prompt, "max_tokens": 32, "temperature": 0})
out["coherence_echo__thinking_disabled"] = call(
    {"messages": echo_prompt, "max_tokens": 32, "temperature": 0, **NO_THINK}
)
out["coherence_echo__thinking_on_large_budget"] = call({"messages": echo_prompt, "max_tokens": 4096, "temperature": 0})

# --- test_penalties (suite payload: max_tokens 1024, temperature 0.1, seed 1234) -------------
pen_prompt = [{"role": "user", "content": "Write a very repetitive story."}]
out["penalties_base__suite_payload"] = call(
    {"messages": pen_prompt, "temperature": 0.1, "max_tokens": 1024, "seed": 1234}
)
out["penalties_base__thinking_disabled"] = call(
    {"messages": pen_prompt, "temperature": 0.1, "max_tokens": 1024, "seed": 1234, **NO_THINK}
)

# --- test_stop (suite payload: stop ["Stop"], max_tokens 1024) ------------------------------
stop_prompt = [{"role": "user", "content": "Count to 5 and then say 'StopIt'."}]
out["stop__suite_payload"] = call({"messages": stop_prompt, "stop": ["Stop"], "max_tokens": 1024})
r = call({"messages": stop_prompt, "stop": ["Stop"], "max_tokens": 1024, **NO_THINK})
r["stop_seq_absent_from_content"] = (r["content"] is not None) and ("Stop" not in r["content"])
out["stop__thinking_disabled"] = r

# --- test_determinism_parameters[temperature-0.0] (suite payload, suite timeout 30 s) -------
det = [call({"messages": REPRO_PROMPT, "temperature": 0.0}) for _ in range(2)]
out["determinism_temp0__run1"] = det[0]
out["determinism_temp0__run2"] = det[1]
out["determinism_temp0__verdict"] = {
    "both_exceed_suite_30s_read_timeout": det[0]["elapsed_s"] > 30 and det[1]["elapsed_s"] > 30,
    "contents_identical": det[0]["content"] == det[1]["content"],
}

# --- test_seed_reproducibility (suite payload: seed 42, temperature 0.5) --------------------
sd = [call({"messages": REPRO_PROMPT, "seed": 42, "temperature": 0.5}) for _ in range(2)]
out["seed_reproducibility__run1"] = sd[0]
out["seed_reproducibility__run2"] = sd[1]
out["seed_reproducibility__verdict"] = {
    "both_exceed_suite_30s_read_timeout": sd[0]["elapsed_s"] > 30 and sd[1]["elapsed_s"] > 30,
    "contents_identical": sd[0]["content"] == sd[1]["content"],
}

json.dump(out, sys.stdout, indent=1)
print()
