import json
import sys
import time

import torch
import ttnn

out = sys.argv[1]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, physical_device_ids=[3])
rows = []
torch.manual_seed(0)


def put(x, dtype, layout=ttnn.TILE_LAYOUT, memory=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, memory_config=memory)


def timed(fn, repeats):
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / repeats * 1e6


def traced(fn, repeats=10):
    fn()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e6 / repeats)
    ttnn.release_trace(mesh, tid)
    samples.sort()
    return samples[len(samples) // 2]


def routes(tokens, experts=256, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(experts, generator=g)[:k] for _ in range(tokens)])


def sparse_config(n, cores, k_block):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8), in0_block_w=k_block,
        out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=n // 32 // cores,
        per_core_M=1, per_core_N=n // 32 // cores, fuse_batch=False, fused_activation=None, mcast_in0=True)


compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False)
x = put(torch.randn(1, 1, 1, 2048).to(torch.bfloat16), ttnn.bfloat16, memory=ttnn.L1_MEMORY_CONFIG)

for width, experts, label in ((512, 256, "x1_baseline"), (256, 256, "x2_expert_tp"), (128, 256, "x4_expert_tp")):
    gate_up = put(torch.randn(1, experts, 2048, 2 * width) * 0.02, ttnn.bfloat4_b)
    down = put(torch.randn(1, experts, width, 2048) * 0.02, ttnn.bfloat8_b)
    ids = ttnn.from_torch(routes(1)[0].to(torch.int32).reshape(1, 1, 1, 8), device=mesh, dtype=ttnn.uint16,
                          layout=ttnn.ROW_MAJOR_LAYOUT)
    placeholder = put(torch.zeros(1, 1, 1, experts), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    for gu_cores, dn_cores, gu_k, dn_k in ((32, 32, 32, 16), (16, 32, 32, 16), (8, 32, 32, 8), (8, 16, 64, 4),
                                           (4, 32, 64, 4), (16, 64, 16, 4)):
        row = {"case": "decode_indexed", "label": label, "width": width, "gate_up_cores": gu_cores,
               "down_cores": dn_cores, "gu_k": gu_k, "down_k": dn_k}
        try:
            if (2 * width // 32) % gu_cores or 64 % dn_cores or (width // 32) % dn_k or 64 % gu_k:
                raise ValueError("geometry does not divide")

            def run():
                both = ttnn.sparse_matmul(x, gate_up, sparsity=placeholder, nnz=None, indices=ids,
                                          program_config=sparse_config(2 * width, gu_cores, gu_k),
                                          compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG,
                                          dtype=ttnn.bfloat16, output_tile=ttnn.Tile([32, 32]))
                act = ttnn.swiglu(ttnn.reshape(both, (1, 8, 1, 2 * width)), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                return ttnn.sparse_matmul(act, down, sparsity=placeholder, nnz=None, indices=ids,
                                          program_config=sparse_config(2048, dn_cores, dn_k),
                                          compute_kernel_config=compute, memory_config=ttnn.L1_MEMORY_CONFIG,
                                          dtype=ttnn.bfloat16, is_input_a_sparse=True, output_tile=ttnn.Tile([32, 32]))

            row["trace_us"] = traced(run)
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:200]
        print(json.dumps(row), flush=True)
        rows.append(row)
    gate_up.deallocate(True)
    down.deallocate(True)


def dispatch_case(label, tokens, experts_local, width, expert_offset, grid, chosen):
    local = chosen[(chosen >= expert_offset) & (chosen < expert_offset + experts_local)] - expert_offset
    counts = torch.bincount(local.flatten(), minlength=experts_local).to(torch.int32)
    aligned = ((counts + 31) // 32) * 32
    offsets = torch.cumsum(aligned, 0) - aligned
    capacity = int(tokens * 8 + experts_local * 32)
    gate = [put(torch.randn(2048, width) * 0.02, ttnn.bfloat8_b) for _ in range(experts_local)]
    up = [put(torch.randn(2048, width) * 0.02, ttnn.bfloat8_b) for _ in range(experts_local)]
    down = [put(torch.randn(width, 2048) * 0.02, ttnn.bfloat8_b) for _ in range(experts_local)]
    rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    counts_t, offsets_t = put(counts, **rm), put(offsets.to(torch.int32), **rm)
    ids_t = put(torch.arange(experts_local, dtype=torch.int32), **rm)
    xin = put(torch.randn(1, 1, capacity, 2048).to(torch.bfloat16), ttnn.bfloat8_b)
    output = ttnn.empty((1, 1, capacity, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    row = {"case": "prefill_dispatch", "label": label, "tokens": tokens, "local_experts": experts_local,
           "width": width, "grid": grid, "local_routes": int(local.numel()), "max_count": int(counts.max())}
    try:
        def run():
            return ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                xin, gate, up, down, counts_t, ids_t, input_m_tiles=tokens // 32, core_grid=ttnn.CoreCoord(*grid),
                dtype=ttnn.bfloat16, output=output, expert_region_offsets=offsets_t, read_x_at_offset=True,
                compute_kernel_config=compute)

        row["us"] = timed(run, 5)
    except Exception as error:
        row["error"] = str(error).splitlines()[0][:200]
    print(json.dumps(row), flush=True)
    rows.append(row)
    for t in gate + up + down + [xin, output]:
        t.deallocate(True)


for tokens in (512, 4096):
    chosen = routes(tokens, seed=tokens)
    big = [8, 8] if tokens >= 2048 else [4, 4]
    dispatch_case("x1_baseline", tokens, 256, 512, 0, big, chosen)
    for grid in ([4, 8], [4, 4], [2, 8], [8, 8]):
        dispatch_case("x4_expert_tp", tokens, 256, 128, 0, grid, chosen)
    for grid in ([8, 8], [4, 8], [4, 4]):
        dispatch_case("x2_expert_tp", tokens, 256, 256, 0, grid, chosen)
    for grid in ([8, 8], [4, 4], [8, 4]):
        dispatch_case("x4_expert_parallel_dev0", tokens, 64, 512, 0, grid, chosen)
        dispatch_case("x2_expert_parallel_dev0", tokens, 128, 512, 0, grid, chosen)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
