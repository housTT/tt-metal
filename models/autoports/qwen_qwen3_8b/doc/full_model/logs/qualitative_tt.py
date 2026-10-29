"""TT generator run of the shared qualitative prompt suite with the same rendered chat prompts as the HF control."""
import json
import sys
import time
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import build_generator

OUT = Path("models/autoports/qwen_qwen3_8b/doc/full_model/qualitative")
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
gen.teardown()
ttnn.close_mesh_device(mesh)
print("TT_QUALITATIVE_DONE", flush=True)
