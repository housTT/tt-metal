import time

import torch

import ttnn

HD, NH, NKV, HID, INTER = 128, 32, 8, 4096, 12288
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=256 * 1024 * 1024)
grid = mesh.compute_with_storage_grid_size()
compute = ttnn.init_device_compute_kernel_config(
    mesh.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=mem)


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


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


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    k, n = gate.shape
    t = torch.stack([gate.reshape(k, n // 32, 32), up.reshape(k, n // 32, 32)], dim=2)
    return t.reshape(k, 2 * n)


torch.manual_seed(0)
wg_t = torch.randn(HID, INTER) * 0.02
wu_t = torch.randn(HID, INTER) * 0.02
wd_t = torch.randn(INTER, HID) * 0.02
wo_t = torch.randn(HID, HID) * 0.02
wqkv_t = torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02
wg, wu, wd, wo, wqkv = tt(wg_t), tt(wu_t), tt(wd_t), tt(wo_t), tt(wqkv_t)
wgu = tt(interleave_gate_up(wg_t, wu_t))
ones = tt(torch.ones(1, 1, 1, HID))
ref_fp32 = lambda x: (torch.nn.functional.silu(x.float() @ wg_t) * (x.float() @ wu_t))

for rows in (32, 1024):
    print(f"\n===== rows={rows} =====", flush=True)
    x_t = torch.randn(1, 1, rows, HID) * 0.5
    x = tt(x_t)
    res_t = torch.randn(1, 1, rows, HID)
    res = tt(res_t)
    act_t = torch.randn(1, 1, rows, INTER) * 0.5
    act = tt(act_t)
    xb = x_t.to(torch.bfloat16).float()

    def gu_base():
        g = ttnn.linear(x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.linear(x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def gu_fused():
        return ttnn.experimental.minimal_matmul(
            x, wgu, fuse_swiglu=True, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def gu_fused_default_compute():
        return ttnn.experimental.minimal_matmul(x, wgu, fuse_swiglu=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    golden = ref_fp32(xb[0, 0])
    try:
        b = ttnn.to_torch(gu_base())[0, 0].float()
        f = ttnn.to_torch(gu_fused())[0, 0].float()
        print(
            f"swiglu base pcc_vs_fp32 {pcc(b, golden):.7f} fused pcc_vs_fp32 {pcc(f, golden):.7f} fused_vs_base pcc {pcc(f, b):.7f} maxdiff {(f - b).abs().max():.4g}",
            flush=True,
        )
        for rep in range(2):
            print(
                f"rep{rep} swiglu base {timed_trace(gu_base):.1f} fused(HiFi4) {timed_trace(gu_fused):.1f} fused(default) {timed_trace(gu_fused_default_compute):.1f}",
                flush=True,
            )
    except Exception as e:
        print("swiglu FAIL:", str(e).splitlines()[0][:400], flush=True)

    # residual fused into the projection
    def o_base():
        o = ttnn.linear(x, wo, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(res, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def o_fused():
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x, wo, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def o_minimal_plus_add():
        o = ttnn.experimental.minimal_matmul(
            x, wo, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        return ttnn.add(res, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    golden_o = res_t.to(torch.bfloat16).float()[0, 0] + xb[0, 0] @ wo_t
    try:
        b = ttnn.to_torch(o_base())[0, 0].float()
        f = ttnn.to_torch(o_fused())[0, 0].float()
        print(
            f"o+res base pcc_vs_fp32 {pcc(b, golden_o):.7f} fused pcc_vs_fp32 {pcc(f, golden_o):.7f} fused_vs_base {pcc(f, b):.7f} maxdiff {(f - b).abs().max():.4g}",
            flush=True,
        )
        for rep in range(2):
            print(
                f"rep{rep} o+res base {timed_trace(o_base):.1f} addcmul_fused {timed_trace(o_fused):.1f} minimal+add {timed_trace(o_minimal_plus_add):.1f}",
                flush=True,
            )
    except Exception as e:
        print("o+res FAIL:", str(e).splitlines()[0][:400], flush=True)

    # down projection + residual
    def d_base():
        o = ttnn.linear(act, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.add(res, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def d_fused():
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            act, wd, 1.0, res, ones, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    golden_d = res_t.to(torch.bfloat16).float()[0, 0] + act_t.to(torch.bfloat16).float()[0, 0] @ wd_t
    try:
        b = ttnn.to_torch(d_base())[0, 0].float()
        f = ttnn.to_torch(d_fused())[0, 0].float()
        print(
            f"down+res base pcc_vs_fp32 {pcc(b, golden_d):.7f} fused pcc_vs_fp32 {pcc(f, golden_d):.7f} fused_vs_base {pcc(f, b):.7f} maxdiff {(f - b).abs().max():.4g}",
            flush=True,
        )
        for rep in range(2):
            print(
                f"rep{rep} down+res base {timed_trace(d_base):.1f} addcmul_fused {timed_trace(d_fused):.1f}", flush=True
            )
    except Exception as e:
        print("down+res FAIL:", str(e).splitlines()[0][:400], flush=True)

    # qkv projection: linear vs minimal_matmul
    def qkv_base():
        return ttnn.linear(x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def qkv_min():
        return ttnn.experimental.minimal_matmul(
            x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    golden_q = xb[0, 0] @ wqkv_t
    try:
        b = ttnn.to_torch(qkv_base())[0, 0].float()
        f = ttnn.to_torch(qkv_min())[0, 0].float()
        print(
            f"qkv linear pcc_vs_fp32 {pcc(b, golden_q):.7f} minimal pcc_vs_fp32 {pcc(f, golden_q):.7f} min_vs_lin {pcc(f, b):.7f} maxdiff {(f - b).abs().max():.4g}",
            flush=True,
        )
        for rep in range(2):
            print(f"rep{rep} qkv linear {timed_trace(qkv_base):.1f} minimal {timed_trace(qkv_min):.1f}", flush=True)
    except Exception as e:
        print("qkv FAIL:", str(e).splitlines()[0][:400], flush=True)

ttnn.close_mesh_device(mesh)
print("DONE")
