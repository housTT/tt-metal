import json
import sys

import torch
import ttnn

out = sys.argv[1]
repeats = int(sys.argv[2])
chips = int(sys.argv[3]) if len(sys.argv) > 3 else 1
if chips == 1:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[3])
else:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if chips == 4 else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips), trace_region_size=0)
torch.manual_seed(0)
DRAM = ttnn.DRAM_MEMORY_CONFIG
rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)


def put(x, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=DRAM,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False)
rows = []
dummy = ttnn.from_torch(torch.randn(chips, 1, 128, 2048).to(torch.bfloat16), device=mesh, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)) if chips > 1 else None
topology = ttnn.Topology.Ring if chips == 4 else ttnn.Topology.Linear
for experts, width, tokens, grid, input_rm in ((64, 512, 128, (4, 4), True), (64, 512, 4096, (8, 8), True)):
    gate = torch.randn(experts, 2048, width) * 0.02
    up = torch.randn(experts, 2048, width) * 0.02
    down = torch.randn(experts, width, 2048) * 0.02
    fused = put(torch.cat((up, gate), dim=-1).unsqueeze(0), ttnn.bfloat4_b)
    stacked_down = put(down.unsqueeze(0), ttnn.bfloat8_b)
    gate4 = [put(gate[e], ttnn.bfloat4_b) for e in range(experts)]
    up4 = [put(up[e], ttnn.bfloat4_b) for e in range(experts)]
    gate8 = [ttnn.typecast(t, ttnn.bfloat8_b) for t in gate4]
    up8 = [ttnn.typecast(t, ttnn.bfloat8_b) for t in up4]
    down8 = [put(down[e], ttnn.bfloat8_b) for e in range(experts)]
    ids = put(torch.arange(experts, dtype=torch.int32), **rm)
    g = torch.Generator().manual_seed(tokens + experts)
    chosen = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(tokens)])
    local = chosen[chosen < experts]
    counts = torch.bincount(local.flatten(), minlength=experts).to(torch.int32)
    aligned = ((counts + 31) // 32) * 32
    offsets = (torch.cumsum(aligned, 0) - aligned).to(torch.int32)
    capacity = int(aligned.sum()) + 32
    x = torch.randn(1, 1, capacity, 2048).to(torch.bfloat16)
    xin = put(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if input_rm else put(x, ttnn.bfloat8_b)
    counts_t, offsets_t = put(counts, **rm), put(offsets, **rm)
    valid = torch.zeros(capacity, dtype=torch.bool)
    for e in range(experts):
        valid[int(offsets[e]):int(offsets[e]) + int(counts[e])] = True
    for mode in ("list_bfp8", "list_mixed", "stacked_mixed"):
        args = {"list_bfp8": (gate8, up8, down8), "list_mixed": (gate4, up4, down8),
                "stacked_mixed": ([fused], [fused], [stacked_down])}[mode]
        kw = dict(input_m_tiles=tokens // 32, core_grid=ttnn.CoreCoord(*grid), dtype=ttnn.bfloat16,
                  expert_region_offsets=offsets_t, read_x_at_offset=True, compute_kernel_config=compute)
        if mode == "stacked_mixed":
            kw.update(stacked_intermediate=width, gate_column_offset=width // 32, up_column_offset=0)
        results = []
        for _ in range(repeats):
            output = ttnn.zeros((1, 1, capacity, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=DRAM)
            if dummy is not None:
                ttnn.all_reduce(dummy, cluster_axis=1, topology=topology, num_links=2)
            ttnn.experimental.deepseek_prefill.moe_fused_swiglu(xin, *args, counts_t, ids, output=output, **kw)
            results.append(torch.stack([ttnn.to_torch(s).float()[0, 0][valid] for s in ttnn.get_device_tensors(output)]))
            output.deallocate(True)
        bad = [i for i in range(1, repeats) if not torch.equal(results[0], results[i])]
        max_abs = max([float((results[0] - results[i]).abs().max()) for i in bad], default=0.0)
        cross_device = sum(1 for r in results if not all(torch.equal(r[0], r[d]) for d in range(r.shape[0])))
        row = dict(cross_device_mismatch_repeats=cross_device, experts=experts, width=width, tokens=tokens, grid=grid, mode=mode, repeats=repeats,
                   differing_repeats=len(bad), max_abs=max_abs, valid_rows=int(valid.sum()))
        print(json.dumps(row), flush=True)
        rows.append(row)
    for t in gate4 + up4 + gate8 + up8 + down8 + [fused, stacked_down]:
        t.deallocate(True)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
