#!/usr/bin/env python3
"""Same-regime qualitative control: the stage-9 generator (this code, this chip), greedy, bucket 1, on the
rendered chat prompts the HF control used, for comparison with the served greedy completions."""

import json
import os
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

RENDERED = Path("models/autoports/qwen_qwen3_8b/doc/optimized_full_model/qualitative/rendered_prompts.json")
OUT = Path(sys.argv[1])
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
OUT.mkdir(parents=True, exist_ok=True)
rendered = json.loads(RENDERED.read_text())
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
gen = build_generator("models/autoports/qwen_qwen3_8b", mesh)
results = []
for entry in rendered:
    t0 = time.time()
    out = gen.generate(entry["token_ids"], MAX_NEW, next_input=None, enable_trace=True)
    results.append(
        {
            "id": entry["id"],
            "prompt": entry["prompt"],
            "token_ids": out,
            "num_tokens": len(out),
            "text": gen.tokenizer.decode(out, skip_special_tokens=False),
            "seconds": time.time() - t0,
        }
    )
    print(f"prompt {entry['id']} done {len(out)} tokens in {time.time() - t0:.1f}s", flush=True)
(OUT / "tt_outputs.json").write_text(json.dumps(results, indent=2) + "\n")
(OUT / "provenance.json").write_text(
    json.dumps(
        {
            "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "max_new_tokens": MAX_NEW,
            "counters": gen.counter_snapshot(),
            "precision_config": gen.precision_config["config_id"],
        },
        indent=2,
    )
    + "\n"
)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("TT_QUALITATIVE_DONE", flush=True)
