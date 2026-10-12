import argparse, json, os, time
from pathlib import Path

import torch
from transformers import AutoConfig

from models.autoports.qwen_qwen3_5_35b_a3b.tests.layer_weights import load_layer_state_dict
from models.autoports.qwen_qwen3_5_35b_a3b.tt.multichip_decoder import MultichipDecoder

p = argparse.ArgumentParser()
p.add_argument("--mesh", type=int, required=True)
p.add_argument("--layer", type=int, required=True)
p.add_argument("--length", type=int, required=True)
p.add_argument("--chunk", type=int, default=4096)
p.add_argument("--steps", type=int, default=32)
p.add_argument("--align", type=int, default=1)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
checkpoint = Path("/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B")
cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True).text_config
state = load_layer_state_dict(checkpoint, args.layer)
import ttnn

if args.mesh == 1:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16 << 20,
                                 physical_device_ids=[int(os.environ.get("QDEV", "3"))])
else:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if args.mesh == 4 else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh), trace_region_size=16 << 20)
result = dict(vars(args), output=str(args.output))
try:
    def put(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t.contiguous(), device=mesh, dtype=dtype, layout=layout,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))

    model = MultichipDecoder.from_state_dict(state, hf_config=cfg, layer_idx=args.layer, mesh_device=mesh)
    full = cfg.layer_types[args.layer] == "full_attention"
    torch.manual_seed(0)
    kw, dkw = {}, {}
    if full:
        page = 32
        blocks = (args.length + args.steps + page - 1) // page
        blocks = -(-blocks // args.align) * args.align
        kc = ttnn.zeros((blocks + 1, model.config.n_kv_heads, page, cfg.head_dim), dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vc = ttnn.zeros_like(kc)
        model.bind_paged_kv_cache(kc, vc, scratch_block=blocks)
        pt = put(torch.randperm(blocks, dtype=torch.int32).reshape(1, blocks), dtype=ttnn.int32,
                 layout=ttnn.ROW_MAJOR_LAYOUT)
        kw = dict(page_table=pt)
    chunks = []
    for start in range(0, args.length, args.chunk):
        size = min(args.chunk, args.length - start)
        x = put(torch.randn(1, size, cfg.hidden_size, dtype=torch.bfloat16) * 0.1)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        y = model.prefill_forward(x, start_pos=start, **kw)
        ttnn.synchronize_device(mesh)
        chunks.append(dict(start=start, size=size, us=(time.perf_counter() - t0) * 1e6))
        if start == 0 or start + size >= args.length:
            host = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
            chunks[-1]["finite"] = bool(torch.isfinite(host).all())
        y.deallocate(True)
        x.deallocate(True)
        print(json.dumps(chunks[-1]), flush=True)
    result["prefill_chunks"] = chunks
    result["prefill_total_us"] = sum(c["us"] for c in chunks)
    dx = put(torch.randn(1, 1, cfg.hidden_size, dtype=torch.bfloat16) * 0.1)
    if full:
        dkw = dict(page_table=pt, position_tensor=put(torch.tensor([args.length], dtype=torch.int32),
                                                       dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT))
    out = model.decode_forward(dx, **dkw)
    result["decode_finite"] = bool(torch.isfinite(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()).all())
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = model.decode_forward(dx, **dkw)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    result["decode_trace_us"] = sorted(samples)[len(samples) // 2]
    result["decode_samples_us"] = samples
    print(json.dumps({k: v for k, v in result.items() if k not in ("prefill_chunks", "decode_samples_us")}), flush=True)
finally:
    args.output.write_text(json.dumps(result, indent=1))
    ttnn.close_mesh_device(mesh)
