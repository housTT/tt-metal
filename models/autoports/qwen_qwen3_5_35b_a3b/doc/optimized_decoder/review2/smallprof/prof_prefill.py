import os, sys, torch
from pathlib import Path
from transformers import AutoConfig
import ttnn
from tracy import signpost
from models.autoports.qwen_qwen3_5_35b_a3b.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
layer, length = int(sys.argv[1]), int(sys.argv[2])
ck = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(ck, local_files_only=True).text_config
state = load_layer_state_dict(ck, layer)
x = torch.load(f"/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage03_inputs/layer{layer}.pt", weights_only=True)["activations"][:, :length].contiguous()
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
put = lambda t, dt=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=mesh, dtype=dt, layout=lay, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
try:
    model = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
    kw = {}
    if cfg.layer_types[layer] == "full_attention":
        pages = (length + 31) // 32 + 8
        kc = ttnn.zeros((pages + 3, cfg.num_key_value_heads, 32, cfg.head_dim), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        model.bind_paged_kv_cache(kc, ttnn.zeros_like(kc))
        kw = dict(page_table=put(torch.arange(pages, dtype=torch.int32).reshape(1, pages), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT))
    px = put(x)
    model.reset_request_state(range(1)); ttnn.synchronize_device(mesh)
    model.prefill_forward(px, **kw).deallocate(True); ttnn.synchronize_device(mesh); ttnn.ReadDeviceProfiler(mesh)
    model.reset_request_state(range(1)); ttnn.synchronize_device(mesh); ttnn.ReadDeviceProfiler(mesh)
    signpost("PERF_PREFILL")
    model.prefill_forward(px, **kw).deallocate(True); ttnn.synchronize_device(mesh)
    signpost("PERF_PREFILL_END")
    ttnn.ReadDeviceProfiler(mesh)
finally:
    ttnn.close_mesh_device(mesh)
