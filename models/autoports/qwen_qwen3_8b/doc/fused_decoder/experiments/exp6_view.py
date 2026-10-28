import time

import torch

import ttnn

HID, INTER = 4096, 12288
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=64 * 1024 * 1024)
compute = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def interleave_gate_up(gate, up):
    k, n = gate.shape
    return torch.stack([gate.reshape(k, n // 32, 32), up.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)


def timed_trace(fn, iters=10, reps=3):
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
x = tt(torch.randn(1, 1, 32, HID) * 0.5)
wg_t = torch.randn(HID, INTER) * 0.02
wu_t = torch.randn(HID, INTER) * 0.02
wg, wu, wgu = tt(wg_t), tt(wu_t), tt(interleave_gate_up(wg_t, wu_t))
wd = tt(torch.randn(INTER, HID) * 0.02)


def base():
    g = ttnn.linear(x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    u = ttnn.linear(x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def via_view():
    gu = ttnn.linear(x, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.reshape(gu, (1, INTER // 32, 32, 64))
    g = ttnn.slice(v, [0, 0, 0, 0], [1, INTER // 32, 32, 32], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    u = ttnn.slice(v, [0, 0, 0, 32], [1, INTER // 32, 32, 64], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a = ttnn.reshape(a, (1, 1, 32, INTER))
    return ttnn.linear(a, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)


gu = ttnn.linear(x, wgu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
v = ttnn.reshape(gu, (1, INTER // 32, 32, 64))
print("view same buffer:", gu.buffer_address() == v.buffer_address(), flush=True)
b = ttnn.to_torch(base())
w = ttnn.to_torch(via_view())
print("equal", torch.equal(b, w), "maxdiff", float((b.float() - w.float()).abs().max()), flush=True)
for rep in range(3):
    print(f"rep{rep} base {timed_trace(base):.1f} via_view {timed_trace(via_view):.1f}", flush=True)
ttnn.close_mesh_device(mesh)
print("DONE")
