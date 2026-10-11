import json
import sys
import time

import torch
import ttnn

out = sys.argv[1]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[3])
torch.manual_seed(0)
rows = []
DRAM = ttnn.DRAM_MEMORY_CONFIG


def put(x, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=DRAM)


def timed(fn, repeats=5):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / repeats * 1e6


compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False)
for experts, width in ((256, 512), (256, 256), (64, 512)):
    host_gate = torch.randn(experts, 2048, width) * 0.02
    host_up = torch.randn(experts, 2048, width) * 0.02
    host_down = torch.randn(experts, width, 2048) * 0.02
    fused = put(torch.cat((host_up, host_gate), dim=-1).unsqueeze(0), ttnn.bfloat4_b)
    stacked_down = put(host_down.unsqueeze(0), ttnn.bfloat8_b)
    gate4 = [put(host_gate[e], ttnn.bfloat4_b) for e in range(experts)]
    up4 = [put(host_up[e], ttnn.bfloat4_b) for e in range(experts)]
    gate8 = [ttnn.typecast(t, ttnn.bfloat8_b) for t in gate4]
    up8 = [ttnn.typecast(t, ttnn.bfloat8_b) for t in up4]
    down8 = [put(host_down[e], ttnn.bfloat8_b) for e in range(experts)]
    rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    ids = put(torch.arange(experts, dtype=torch.int32), **rm)
    for tokens, grid in ((512, (4, 4)), (4096, (8, 8))):
        g = torch.Generator().manual_seed(tokens)
        chosen = torch.stack([torch.randperm(256, generator=g)[:8] for _ in range(tokens)])
        local = chosen[chosen < experts]
        counts = torch.bincount(local.flatten(), minlength=experts).to(torch.int32)
        aligned = ((counts + 31) // 32) * 32
        offsets = (torch.cumsum(aligned, 0) - aligned).to(torch.int32)
        capacity = int(aligned.sum()) + 32
        xin = put(torch.randn(1, 1, capacity, 2048).to(torch.bfloat16), ttnn.bfloat8_b)
        counts_t, offsets_t = put(counts, **rm), put(offsets, **rm)
        results = {}
        for mode in ("list_bfp8", "list_mixed", "stacked_mixed"):
            output = ttnn.zeros((1, 1, capacity, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                memory_config=DRAM)
            kw = dict(input_m_tiles=tokens // 32, core_grid=ttnn.CoreCoord(*grid), dtype=ttnn.bfloat16, output=output,
                      expert_region_offsets=offsets_t, read_x_at_offset=True, compute_kernel_config=compute)
            if mode == "list_bfp8":
                args = (gate8, up8, down8)
            elif mode == "list_mixed":
                args = (gate4, up4, down8)
            else:
                args = ([fused], [fused], [stacked_down])
                kw.update(stacked_intermediate=width, gate_column_offset=width // 32, up_column_offset=0)
            row = {"experts": experts, "width": width, "tokens": tokens, "grid": grid, "mode": mode}
            try:
                fn = lambda: ttnn.experimental.deepseek_prefill.moe_fused_swiglu(xin, *args, counts_t, ids, **kw)
                fn()
                results[mode] = ttnn.to_torch(output).float()
                row["us"] = timed(fn)
                if mode != "list_bfp8":
                    ref = results["list_bfp8"]
                    row["identical_to_list_bfp8"] = bool(torch.equal(ref, results[mode]))
                    row["max_abs_diff"] = float((ref - results[mode]).abs().max())
                    row["finite"] = bool(torch.isfinite(results[mode]).all())
            except Exception as error:
                row["error"] = str(error).splitlines()[0][:300]
            print(json.dumps(row), flush=True)
            rows.append(row)
    for t in gate4 + up4 + gate8 + up8 + down8 + [fused, stacked_down]:
        t.deallocate(True)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
