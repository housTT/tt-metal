"""TT generator run of the shared qualitative prompt suite on the selected precision config (same rendered chat prompts as the HF control)."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

SRC = Path("models/autoports/qwen_qwen3_8b/doc/optimized_full_model/qualitative")
OUT = Path("models/autoports/qwen_qwen3_8b/doc/datatype_sweep/qualitative")
OUT.mkdir(parents=True, exist_ok=True)
for name in ("qualitative_prompt_format.json", "rendered_prompts.json", "hf_outputs.json"):
    shutil.copy(SRC / name, OUT / name)
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 128
rendered = json.loads((OUT / "rendered_prompts.json").read_text())
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
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
(OUT / "tt_counters.json").write_text(json.dumps(gen.counter_snapshot(), indent=2) + "\n")
(OUT / "tt_provenance.json").write_text(
    json.dumps(
        {
            "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
            "precision_config_id": gen.precision_config["config_id"],
            "precision_config_path": str(gen.precision_config_path),
            "max_new_tokens": MAX_NEW,
        },
        indent=2,
    )
    + "\n"
)
gen.teardown()
ttnn.close_mesh_device(mesh)
print("TT_QUALITATIVE_DONE", flush=True)
