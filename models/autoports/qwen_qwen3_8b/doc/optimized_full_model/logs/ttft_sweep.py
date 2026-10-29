"""Warmed batch-1 TTFT of the full model across prompt lengths at and around every prefill bucket boundary.

TTFT here is ``generate(prompt, 1)`` wall time: input preparation, every prefill chunk, final norm, LM head,
on-device greedy sampling and the first-token readback. Each length is run once warm and then three times;
the best and median are recorded. The same numbers calibrate batched prefill (single-row cost per bucket).
Usage: python ttft_sweep.py <output.json> [lengths comma separated]
"""
import json
import os
import statistics
import subprocess
import sys
import time

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

out_path = sys.argv[1]
default = (
    "32,64,96,128,129,160,192,256,257,384,512,513,768,1024,1025,1536,2048,2049,3072,4096,4097,6144,8192,8193,16384"
)
lengths = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else default).split(",")]
layers = int(os.environ.get("QWEN3_FULL_MODEL_LAYERS", "36"))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
t0 = time.perf_counter()
gen = build_generator("models/autoports/qwen_qwen3_8b", mesh, num_layers=layers)
build_s = time.perf_counter() - t0
tok = gen.tokenizer
import bz2

text = bz2.open("models/tt_transformers/tests/tale-of-two-cities.txt.bz2", "rt", encoding="utf-8").read(400000)
ids = tok.encode(text, add_special_tokens=False)
rows = []
for n in lengths:
    prompt = ids[1000 : 1000 + n]
    assert len(prompt) == n
    gen.reset()
    gen.generate(prompt, 1, enable_trace=True, stop_on_eos=False)
    times = []
    for _ in range(3):
        gen.reset()
        t = time.perf_counter()
        first = gen.generate(prompt, 1, enable_trace=True, stop_on_eos=False)
        times.append((time.perf_counter() - t) * 1000)
    from models.autoports.qwen_qwen3_8b.tt.model import prefill_compute_len

    row = {
        "prompt_len": n,
        "compute_len": prefill_compute_len(n),
        "ttft_ms_best": round(min(times), 2),
        "ttft_ms_median": round(statistics.median(times), 2),
        "first_token": first[0],
    }
    rows.append(row)
    print("TTFT " + json.dumps(row), flush=True)
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
payload = {
    "workload": "batch 1, book text, greedy split sampling, generate(prompt, 1) wall time = TTFT incl. first-token readback",
    "build_s": round(build_s, 1),
    "prepare_counters": gen.counter_snapshot(),
    "rows": rows,
    "provenance": {
        "checkout_git_sha": sha,
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "num_layers": layers,
        "terminal_policy": gen.model.terminal_policy.describe(),
        "warmup_prefill_lengths": list(gen.warmup_prefill_lengths),
        "environment": {k: v for k, v in sorted(os.environ.items()) if k.startswith("QWEN3_")},
    },
}
with open(out_path, "w") as fh:
    json.dump(payload, fh, indent=2)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("SWEEP_OK", flush=True)
