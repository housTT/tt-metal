import json, sys, time
from pathlib import Path
import ttnn
from models.autoports.qwen_qwen3_5_35b_a3b.tt.generator import build_generator
from models.autoports.qwen_qwen3_5_35b_a3b.tests.mesh_util import open_mesh

mesh_n, slots, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
mesh = open_mesh(mesh_n)
result = dict(mesh=mesh_n, request_slots=slots)
try:
    t0 = time.perf_counter()
    gen = build_generator(Path("/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b"), mesh,
                          max_batch_size=slots, decode_batches=(1, slots))
    result["build_seconds"] = time.perf_counter() - t0
    result["model_slots"] = gen.model.max_batch_size
    result["layer_slots"] = sorted({len(l._states) for l in gen.model.linear_layers})
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    result["dram"] = dict(allocated=view.total_bytes_allocated_per_bank * view.num_banks,
                          free=view.total_bytes_free_per_bank * view.num_banks,
                          largest_free_per_bank=view.largest_contiguous_bytes_free_per_bank)
    prompt = gen.tokenizer("The capital of France is", add_special_tokens=False).input_ids
    result["b1"] = gen.tokenizer.decode(gen.generate(prompt, 8))
    result["ok"] = True
except Exception as e:
    result["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
    raise
finally:
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)
    ttnn.close_mesh_device(mesh)
