import time

import torch

import ttnn

HD, NH, NKV, HID, INTER = 128, 32, 8, 4096, 12288
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=256 * 1024 * 1024)
compute = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def tt(t):
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def timed_trace(fn, iters=4, reps=3):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(iters):
        fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (reps * iters) * 1e6
    ttnn.release_trace(mesh, tid)
    return dt


torch.manual_seed(0)
wg, wu, wd = tt(torch.randn(HID, INTER) * 0.02), tt(torch.randn(HID, INTER) * 0.02), tt(torch.randn(INTER, HID) * 0.02)
wqkv = tt(torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02)
wo = tt(torch.randn(HID, HID) * 0.02)
ones = tt(torch.ones(1, 1, 1, HID))
print("rows | qkv lin/min | o+res lin/fused | down+res lin/fused | gate+up+mul lin/min2", flush=True)
for rows in (32, 64, 96, 128, 192, 256, 512):
    x = tt(torch.randn(1, 1, rows, HID) * 0.5)
    res = tt(torch.randn(1, 1, rows, HID))
    act = tt(torch.randn(1, 1, rows, INTER) * 0.5)
    r = []
    r.append(
        timed_trace(lambda: ttnn.linear(x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG))
    )
    r.append(
        timed_trace(
            lambda: ttnn.experimental.minimal_matmul(
                x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
    )
    r.append(
        timed_trace(
            lambda: ttnn.add(
                res,
                ttnn.linear(x, wo, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    )
    r.append(
        timed_trace(
            lambda: ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, wo, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
    )
    r.append(
        timed_trace(
            lambda: ttnn.add(
                res,
                ttnn.linear(act, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    )
    r.append(
        timed_trace(
            lambda: ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                act, wd, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        )
    )

    def gl():
        g = ttnn.linear(x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def gm():
        g = ttnn.experimental.minimal_matmul(
            x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        u = ttnn.experimental.minimal_matmul(
            x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    r.append(timed_trace(gl))
    r.append(timed_trace(gm))
    print(
        f"{rows:5d} | {r[0]:7.0f}/{r[1]:7.0f} | {r[2]:7.0f}/{r[3]:7.0f} | {r[4]:7.0f}/{r[5]:7.0f} | {r[6]:7.0f}/{r[7]:7.0f}",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
print("DONE")
