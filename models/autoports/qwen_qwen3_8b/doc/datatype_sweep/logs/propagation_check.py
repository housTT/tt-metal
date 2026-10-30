"""Build a reduced generator from each precision config and prove the live runtime consumes every field.

Usage: propagation_check.py <num_layers> <output.json> <config.json> [<config.json> ...]
Each config is built through ``build_generator`` (the normal path, ``QWEN3_PRECISION_CONFIG`` set per
config), ``describe_runtime`` reads dtypes and fidelities back from the device tensors and compute
configs, ``check_propagation`` compares them with the config, and one short greedy generation runs the
built graphs. The output JSON records the resolved runtime per config and any mismatch.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator
from models.autoports.qwen_qwen3_8b.tt.precision import check_propagation, describe_runtime

layers = int(sys.argv[1])
out_path = Path(sys.argv[2])
configs = [Path(p) for p in sys.argv[3:]]
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
results = {
    "checkout_git_sha": sha,
    "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
    "num_layers": layers,
    "configs": {},
}
for cfg in configs:
    os.environ["QWEN3_PRECISION_CONFIG"] = str(cfg)
    t0 = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 * 1024 * 1024)
    gen = build_generator(
        "models/autoports/qwen_qwen3_8b",
        mesh,
        num_layers=layers,
        decode_buckets=(1,),
        warmup_prefill_lengths=(32, 128),
        warmup_chunk_starts=(),
        warmup_multi_row=False,
        num_blocks=256,
    )
    runtime = describe_runtime(gen)
    problems = check_propagation(gen)
    ids = gen.tokenizer.encode("The capital of France is", add_special_tokens=False)
    out = gen.generate(ids, 8, next_input=None, enable_trace=True, stop_on_eos=False)
    counters = gen.counter_snapshot()
    gen.teardown()
    ttnn.close_mesh_device(mesh)
    results["configs"][gen.precision_config["config_id"]] = {
        "config_path": str(cfg),
        "runtime": runtime,
        "problems": problems,
        "generated_tokens": out,
        "generated_text": gen.tokenizer.decode(out),
        "counters": counters,
        "seconds": round(time.time() - t0, 1),
    }
    print(f"{gen.precision_config['config_id']}: problems={problems} text={gen.tokenizer.decode(out)!r}", flush=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
failed = {k: v["problems"] for k, v in results["configs"].items() if v["problems"]}
print("PROPAGATION_FAIL" if failed else "PROPAGATION_OK", json.dumps(failed), flush=True)
