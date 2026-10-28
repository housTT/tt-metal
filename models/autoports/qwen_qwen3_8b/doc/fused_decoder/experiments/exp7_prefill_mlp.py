import time

import torch

import ttnn

HD, NH, NKV, HID, INTER = 128, 32, 8, 4096, 12288
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
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


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def timed_trace(fn, iters=2, reps=3):
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


def interleave_gate_up(gate, up):
    k, n = gate.shape
    return torch.stack([gate.reshape(k, n // 32, 32), up.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)


torch.manual_seed(0)
wg_t = torch.randn(HID, INTER) * 0.02
wu_t = torch.randn(HID, INTER) * 0.02
wd_t = torch.randn(INTER, HID) * 0.02
wg, wu, wd, wgu = tt(wg_t), tt(wu_t), tt(wd_t), tt(interleave_gate_up(wg_t, wu_t))
wqkv = tt(torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02)
wo = tt(torch.randn(HID, HID) * 0.02)
ones = tt(torch.ones(1, 1, 1, HID))
silu = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
for rows in (1024, 4096):
    print(f"\n===== rows={rows} =====", flush=True)
    x = tt(torch.randn(1, 1, rows, HID) * 0.5)
    res = tt(torch.randn(1, 1, rows, HID))
    act = tt(torch.randn(1, 1, rows, INTER) * 0.5)

    def mlp_linear():
        g = ttnn.linear(x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_min2():
        g = ttnn.experimental.minimal_matmul(
            x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        u = ttnn.experimental.minimal_matmul(
            x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_min2_act():
        g = ttnn.experimental.minimal_matmul(
            x, wg, fused_activation=silu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        u = ttnn.experimental.minimal_matmul(
            x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.mul(g, u, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def mlp_fused():
        return ttnn.experimental.minimal_matmul(
            x, wgu, fuse_swiglu=True, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    b = ttnn.to_torch(mlp_linear())
    for name, fn in (("min2", mlp_min2), ("min2_act", mlp_min2_act), ("fused_interleaved", mlp_fused)):
        try:
            o = ttnn.to_torch(fn())
            print(
                f"{name}: pcc_vs_linear {pcc(o, b):.7f} maxdiff {float((o.float() - b.float()).abs().max()):.4g}",
                flush=True,
            )
        except Exception as e:
            print(name, "FAIL:", str(e).splitlines()[0][:300], flush=True)
    for rep in range(2):
        line = f"rep{rep} gate/up/mul: linear {timed_trace(mlp_linear):.0f}"
        for name, fn in (("min2", mlp_min2), ("min2_act", mlp_min2_act), ("fused_interleaved", mlp_fused)):
            try:
                line += f" {name} {timed_trace(fn):.0f}"
            except Exception as e:
                line += f" {name} FAIL"
        print(line, flush=True)

    def qkv_lin():
        return ttnn.linear(x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def qkv_min():
        return ttnn.experimental.minimal_matmul(
            x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def o_lin():
        return ttnn.add(
            res,
            ttnn.linear(x, wo, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def o_fused():
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x, wo, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def d_lin():
        return ttnn.add(
            res,
            ttnn.linear(act, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def d_fused():
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            act, wd, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for name, a, b_ in (("qkv", qkv_lin, qkv_min), ("o+res", o_lin, o_fused), ("down+res", d_lin, d_fused)):
        try:
            ra, rb = ttnn.to_torch(a()), ttnn.to_torch(b_())
            print(
                f"{name}: pcc {pcc(ra, rb):.7f} maxdiff {float((ra.float() - rb.float()).abs().max()):.4g} linear {timed_trace(a):.0f} fused/minimal {timed_trace(b_):.0f}",
                flush=True,
            )
        except Exception as e:
            print(name, "FAIL:", str(e).splitlines()[0][:300], flush=True)
ttnn.close_mesh_device(mesh)
print("DONE")
