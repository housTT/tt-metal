import json
import sys
import time

import torch
import ttnn

out = sys.argv[1]
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 << 20, physical_device_ids=[3])
torch.manual_seed(0)
L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG


def put(x, dtype, layout=ttnn.TILE_LAYOUT, memory=DRAM):
    return ttnn.from_torch(x, device=mesh, dtype=dtype, layout=layout, memory_config=memory)


def traced(fn, repeats=20):
    fn()
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    samples = []
    for _ in range(15):
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
x = put(torch.randn(1, 1, 1, 2048).to(torch.bfloat16), ttnn.bfloat16, memory=L1)
ids = ttnn.from_torch(torch.randperm(256)[:8].to(torch.int32).reshape(1, 1, 1, 8), device=mesh, dtype=ttnn.uint16,
                      layout=ttnn.ROW_MAJOR_LAYOUT)
placeholder = put(torch.zeros(1, 1, 1, 256), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
rows = []
for width, label in ((128, "x4"), (256, "x2")):
    gate_up = put(torch.randn(1, 256, 2048, 2 * width) * 0.02, ttnn.bfloat4_b)
    down = put(torch.randn(1, 256, width, 2048) * 0.02, ttnn.bfloat8_b)
    act = put(torch.randn(1, 8, 1, width).to(torch.bfloat16), ttnn.bfloat16, memory=L1)
    for cores in (1, 2, 4, 8, 16):
        for k in (8, 16, 32, 64):
            if (2 * width // 32) % cores or 64 % k:
                continue
            row = dict(mesh=label, role="gate_up", n=2 * width, cores=cores, k=k)
            try:
                row["trace_us"] = traced(lambda: ttnn.sparse_matmul(
                    x, gate_up, sparsity=placeholder, nnz=None, indices=ids, program_config=config(2 * width, cores, k),
                    compute_kernel_config=compute, memory_config=L1, dtype=ttnn.bfloat16, output_tile=ttnn.Tile([32, 32])))
            except Exception as error:
                row["error"] = str(error).splitlines()[0][:160]
            print(json.dumps(row), flush=True)
            rows.append(row)
    for cores in (8, 16, 32, 64):
        for k in (1, 2, 4, 8):
            if 64 % cores or (width // 32) % k:
                continue
            row = dict(mesh=label, role="down", n=2048, cores=cores, k=k)
            try:
                row["trace_us"] = traced(lambda: ttnn.sparse_matmul(
                    act, down, sparsity=placeholder, nnz=None, indices=ids, program_config=config(2048, cores, k),
                    compute_kernel_config=compute, memory_config=L1, dtype=ttnn.bfloat16, is_input_a_sparse=True,
                    output_tile=ttnn.Tile([32, 32])))
            except Exception as error:
                row["error"] = str(error).splitlines()[0][:160]
            print(json.dumps(row), flush=True)
            rows.append(row)
    gate_up.deallocate(True)
    down.deallocate(True)
json.dump(rows, open(out, "w"), indent=1)
ttnn.close_mesh_device(mesh)
