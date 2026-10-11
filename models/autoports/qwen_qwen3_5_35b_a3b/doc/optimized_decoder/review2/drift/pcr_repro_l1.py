import os, sys, torch, ttnn
T = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[int(os.environ.get("QDEV", "0"))])
R = ttnn.ReplicateTensorToMesh(mesh)
torch.manual_seed(0)
restored = torch.randn(1, 1, T, 8, 2048).bfloat16()
w = torch.softmax(torch.randn(T, 8), -1).reshape(1, 1, T, 8, 1).bfloat16()
rt = ttnn.from_torch(restored, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=R)
if os.environ.get("DEVW"):
    vt = ttnn.from_torch(w.reshape(1, 1, T, 8), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=R,
                         memory_config=ttnn.L1_MEMORY_CONFIG if os.environ.get("WL1") else ttnn.DRAM_MEMORY_CONFIG)
    wt = ttnn.reshape(ttnn.to_layout(vt, ttnn.ROW_MAJOR_LAYOUT), (1, 1, T, 8, 1))
    print("weights spec", wt.shape, wt.padded_shape, wt.memory_config(), flush=True)
else:
    wt = ttnn.from_torch(w, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=R)
get = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
ref = (restored.float() * w.float()).sum(3)
outs = []
for i in range(N):
    if os.environ.get("DEVW") == "2":
        wt = ttnn.reshape(ttnn.to_layout(vt, ttnn.ROW_MAJOR_LAYOUT), (1, 1, T, 8, 1))
    o = ttnn.experimental.deepseek_prefill.post_combine_reduce(rt, wt, expert_dim=3, output_memory_config=ttnn.DRAM_MEMORY_CONFIG)
    outs.append(get(o).float().reshape(T, 2048))
bad = [i for i in range(1, N) if not torch.equal(outs[0], outs[i])]
rows = set()
for i in bad:
    rows |= set(torch.nonzero((outs[i] - outs[0]).abs().amax(-1) > 0).flatten().tolist())
err = (outs[0] - ref.reshape(T, 2048)).abs().max().item()
print(f"T={T} runs={N} mismatching_runs={len(bad)} distinct_rows={len(rows)} sample_rows={sorted(rows)[:8]} max_abs_err_vs_fp32={err:.4g}", flush=True)
ttnn.close_mesh_device(mesh)
