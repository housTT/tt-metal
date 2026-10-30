"""Token budget of the selected build through Qwen3Generator.max_tokens_all_users: with the mesh (live DRAM), with the model only, with neither."""
import json
import os
import subprocess

import ttnn
from models.autoports.qwen_qwen3_8b.tt.generator import Qwen3Generator, build_generator

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
gen = build_generator("models/autoports/qwen_qwen3_8b", mesh, prepare=False)
view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
result = {
    "config_id": gen.precision_config["config_id"],
    "weight_bytes": gen.model.weight_bytes(),
    "kv_bytes_per_token": gen.model.kv_bytes_per_token(gen.kv_cache_dtype),
    "max_tokens_with_mesh": Qwen3Generator.max_tokens_all_users(
        gen.model, mesh_device=mesh, kv_cache_dtype=gen.kv_cache_dtype
    ),
    "max_tokens_with_mesh_note": "read after the generator allocated its own 1280-block KV pool and sampler state, so lower than the contract's weights-only budget",
    "max_tokens_model_only": Qwen3Generator.max_tokens_all_users(gen.model, kv_cache_dtype=gen.kv_cache_dtype),
    "max_tokens_no_model_no_mesh": Qwen3Generator.max_tokens_all_users(),
    "dram_total_bytes": int(view.total_bytes_per_bank) * int(view.num_banks),
    "dram_allocated_bytes": int(view.total_bytes_allocated_per_bank) * int(view.num_banks),
    "dram_free_bytes": int(view.total_bytes_free_per_bank) * int(view.num_banks),
    "checkout_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
}
gen.teardown()
ttnn.close_mesh_device(mesh)
with open("models/autoports/qwen_qwen3_8b/doc/datatype_sweep/token_budget_check.json", "w") as fh:
    json.dump(result, fh, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "weight_bytes"}), flush=True)
print("TOKEN_BUDGET_OK", flush=True)
