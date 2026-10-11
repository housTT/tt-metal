import json
import sys
import time

import torch
import ttnn

out = sys.argv[1]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, physical_device_ids=[3])
rows = []
torch.manual_seed(0)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def put(x, dtype, layout=ttnn.TILE_LAYOUT, memory=DRAM):
    return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, memory_config=memory)


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


def config(n, cores, k_block):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(min(8, cores), (cores + 7) // 8), in0_block_w=k_block,
        out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=n // 32 // cores,
        per_core_M=1, per_core_N=n // 32 // cores, fuse_batch=False, fused_activation=None, mcast_in0=True)


compute = ttnn.init_device_compute_kernel_config(mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi,
                                                 fp32_dest_acc_en=False, packer_l1_acc=False, math_approx_mode=False)
experts = 64
host_x = torch.randn(1, 1, 1, 2048).to(torch.bfloat16)
x = put(host_x, ttnn.bfloat16, memory=L1)
host_gu = torch.randn(1, experts, 2048, 1024) * 0.02
host_dn = torch.randn(1, experts, 512, 2048) * 0.02
gate_up = put(host_gu, ttnn.bfloat4_b)
down = put(host_dn, ttnn.bfloat8_b)

for active in (0, 1, 2, 3, 4, 8):
    chosen = torch.randperm(experts)[:active]
    mask = torch.zeros(1, 1, 1, experts)
    mask[..., chosen] = 1
    scores = torch.zeros(1, 1, 1, experts)
    scores[..., chosen] = torch.rand(active) + 0.1
    sparsity = put(mask, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    score_t = put(scores, ttnn.bfloat16)
    for gu_cores, dn_cores, gu_k, dn_k, memory in ((32, 32, 32, 16, L1), (16, 64, 16, 4, L1), (32, 32, 32, 16, DRAM)):
        row = {"case": "ep_decode_mask", "local_experts": experts, "active": active, "gu_cores": gu_cores,
               "dn_cores": dn_cores, "gu_k": gu_k, "dn_k": dn_k, "memory": "l1" if memory is L1 else "dram"}
        try:
            nan_fill = ttnn.full((1, experts, 32, 2048), float("nan"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=mesh, memory_config=memory)
            ttnn.deallocate(nan_fill)

            def run():
                both = ttnn.sparse_matmul(x, gate_up, sparsity=sparsity, nnz=None,
                                          program_config=config(1024, gu_cores, gu_k), compute_kernel_config=compute,
                                          memory_config=memory, dtype=ttnn.bfloat16, output_tile=ttnn.Tile([32, 32]))
                act = ttnn.swiglu(ttnn.reshape(both, (1, experts, 1, 1024)), dim=-1, memory_config=memory)
                dn = ttnn.sparse_matmul(act, down, sparsity=sparsity, nnz=None, program_config=config(2048, dn_cores, dn_k),
                                        compute_kernel_config=compute, memory_config=memory, dtype=ttnn.bfloat16,
                                        is_input_a_sparse=True, output_tile=ttnn.Tile([32, 32]))
                weighted = ttnn.multiply(dn, ttnn.permute(score_t, (0, 3, 2, 1)), memory_config=memory)
                return ttnn.experimental.fast_reduce_nc(weighted, dims=[1], memory_config=memory), dn

            result, dn = run()
            dense = ttnn.to_torch(dn).float()[0, :, 0, :]
            inactive = [e for e in range(experts) if mask[0, 0, 0, e] == 0]
            row["inactive_nonzero"] = int((dense[inactive] != 0).any(dim=-1).sum()) if inactive else 0
            row["inactive_nonfinite"] = int((~torch.isfinite(dense[inactive])).any(dim=-1).sum()) if inactive else 0
            ref = torch.zeros(2048)
            for e in chosen.tolist():
                gu = host_x.float()[0, 0, 0] @ host_gu[0, e]
                act_ref = torch.nn.functional.silu(gu[512:]) * gu[:512]
                ref += scores[0, 0, 0, e] * (act_ref @ host_dn[0, e])
            got = ttnn.to_torch(result).float().reshape(-1)[:2048]
            row["max_abs_vs_fp32"] = float((got - ref).abs().max())
            row["ref_abs_max"] = float(ref.abs().max())
            row["trace_us"] = traced(lambda: run())
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:220]
        print(json.dumps(row), flush=True)
        rows.append(row)

gate = [put(host_gu[0, e, :, 512:], ttnn.bfloat8_b) for e in range(experts)]
up = [put(host_gu[0, e, :, :512], ttnn.bfloat8_b) for e in range(experts)]
dwn = [put(host_dn[0, e], ttnn.bfloat8_b) for e in range(experts)]
rm = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
ids_t = put(torch.arange(experts, dtype=torch.int32), **rm)
for active in (1, 2, 4, 8):
    chosen = torch.randperm(experts)[:active]
    counts = torch.zeros(experts, dtype=torch.int32)
    counts[chosen] = 1
    offsets = (torch.cumsum(counts * 32, 0) - counts * 32).to(torch.int32)
    counts_t, offsets_t = put(counts, **rm), put(offsets, **rm)
    xin = put(torch.randn(1, 1, 32 * 8, 2048).to(torch.bfloat16), ttnn.bfloat16)
    output = ttnn.empty((1, 1, 32 * 8, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        memory_config=DRAM)
    for grid in ((8, 2), (4, 4), (8, 8)):
        row = {"case": "ep_decode_fused_swiglu", "active": active, "grid": grid}
        try:
            row["trace_us"] = traced(lambda: ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                xin, gate, up, dwn, counts_t, ids_t, input_m_tiles=1, core_grid=ttnn.CoreCoord(*grid),
                dtype=ttnn.bfloat16, output=output, expert_region_offsets=offsets_t, read_x_at_offset=True,
                compute_kernel_config=compute))
        except Exception as error:
            row["error"] = str(error).splitlines()[0][:220]
        print(json.dumps(row), flush=True)
        rows.append(row)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
print("DONE")
