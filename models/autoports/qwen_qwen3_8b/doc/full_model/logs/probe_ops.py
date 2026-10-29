import time

import torch

import ttnn
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    DecodeMatmulRole,
    dram_sharded_program_config,
    dram_sharded_weight_memory_config,
    width_sharded_memory_config,
)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
H, V, VP = 4096, 151936, 152576

tok_host = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
tok_host[0, 0, 0, 0] = 5
tokens = ttnn.from_torch(tok_host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
emb_w = ttnn.from_torch(
    torch.randn(V, H).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
)
e = ttnn.embedding(tokens, emb_w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("embedding out", e.shape, e.layout, flush=True)
e4 = ttnn.unsqueeze_to_4D(e)
print("unsqueeze", e4.shape, flush=True)
res_cfg = width_sharded_memory_config(H, DEFAULT_POLICY.residual_cores, grid)
x = ttnn.to_memory_config(e4, res_cfg)
print("residual sharded", x.shape, x.memory_config().memory_layout, flush=True)

role = DecodeMatmulRole("bfp8", "hifi2", False, 32, 4, 1)
import os

SW = int(os.environ.get("SPLIT", "16384"))
splits = [SW] * (VP // SW) + ([VP % SW] if VP % SW else [])
print("splits", splits, flush=True)
weights = []
for w in splits:
    weights.append(
        ttnn.from_torch(
            torch.randn(1, 1, H, w).to(torch.bfloat16) * 0.02,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=dram_sharded_weight_memory_config(H, w, 8),
        )
    )
ckc = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
outs = []
for w, wt in zip(splits, weights):
    o = ttnn.linear(
        x,
        wt,
        program_config=dram_sharded_program_config(role, H, w, 8),
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat16,
        memory_config=width_sharded_memory_config(w, 32, grid),
    )
    outs.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
logits = ttnn.concat(outs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
print("logits", logits.shape, logits.dtype, flush=True)

pos = ttnn.from_torch(
    torch.tensor([7, -1], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh
)
ttnn.plus_one(pos, skip_negative_entries=True)
print("plus_one", ttnn.to_torch(pos).tolist(), flush=True)

sliced = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, 32, V], memory_config=ttnn.DRAM_MEMORY_CONFIG)
rm = ttnn.untilize(sliced, use_multicore=True)
tok_out = ttnn.argmax(rm, dim=-1, output_tensor=tokens, keepdim=False)
print("argmax into tokens", tok_out.shape, ttnn.to_torch(tokens)[0, 0, 0, :4].tolist(), flush=True)
ref = torch.argmax(ttnn.to_torch(sliced).float(), dim=-1)[0, 0, :4].tolist()
print("torch argmax", ref, flush=True)

t0 = time.perf_counter()
trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
e = ttnn.embedding(tokens, emb_w, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
e4 = ttnn.unsqueeze_to_4D(e)
x = ttnn.to_memory_config(e4, res_cfg)
outs = []
for w, wt in zip(splits, weights):
    o = ttnn.linear(
        x,
        wt,
        program_config=dram_sharded_program_config(role, H, w, 8),
        compute_kernel_config=ckc,
        dtype=ttnn.bfloat16,
        memory_config=width_sharded_memory_config(w, 32, grid),
    )
    outs.append(ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG))
tlogits = ttnn.concat(outs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
ttnn.plus_one(pos, skip_negative_entries=True)
ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
print("captured in", round(time.perf_counter() - t0, 2), flush=True)
for _ in range(3):
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
t0 = time.perf_counter()
for _ in range(10):
    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
print(
    "lm head trace replay ms",
    round((time.perf_counter() - t0) * 100, 3),
    "pos",
    ttnn.to_torch(pos).tolist(),
    flush=True,
)
ttnn.release_trace(mesh, trace_id)
ttnn.close_mesh_device(mesh)
print("PROBE_OK", flush=True)
