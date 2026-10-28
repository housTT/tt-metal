import itertools
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


def interleave_gate_up(gate, up):
    k, n = gate.shape
    return torch.stack([gate.reshape(k, n // 32, 32), up.reshape(k, n // 32, 32)], dim=2).reshape(k, 2 * n)


torch.manual_seed(0)
x_t = torch.randn(1, 1, 32, HID) * 0.5
x = tt(x_t)
act = tt(torch.randn(1, 1, 32, INTER) * 0.5)
res = tt(torch.randn(1, 1, 32, HID))
ones = tt(torch.ones(1, 1, 1, HID))
wg_t = torch.randn(HID, INTER) * 0.02
wu_t = torch.randn(HID, INTER) * 0.02
wqkv = tt(torch.randn(HID, HD * (NH + 2 * NKV)) * 0.02)
wo = tt(torch.randn(HID, HID) * 0.02)
wd = tt(torch.randn(INTER, HID) * 0.02)
wgu = tt(interleave_gate_up(wg_t, wu_t))
wg, wu = tt(wg_t), tt(wu_t)

print("===== baselines (ttnn.linear) =====", flush=True)
base = {
    "qkv": timed_trace(
        lambda: ttnn.linear(x, wqkv, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ),
    "o": timed_trace(lambda: ttnn.linear(x, wo, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
    "down": timed_trace(
        lambda: ttnn.linear(act, wd, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ),
}


def gu_base():
    g = ttnn.linear(x, wg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    u = ttnn.linear(x, wu, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG)


base["swiglu"] = timed_trace(gu_base)
print(base, flush=True)

print("\n===== matmul_decode probe =====", flush=True)
for name, a, w in (("qkv", x, wqkv), ("o", x, wo), ("down", act, wd)):
    try:
        out = ttnn.experimental.matmul_decode(a, w)
        ref = ttnn.linear(a, w, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        p = pcc(ttnn.to_torch(out), ttnn.to_torch(ref))
        t = timed_trace(lambda: ttnn.experimental.matmul_decode(a, w))
        print(
            f"matmul_decode {name}: pcc {p:.6f} us {t:.1f} (linear {base[name]:.1f}) mem {out.memory_config().memory_layout}",
            flush=True,
        )
    except Exception as e:
        print(f"matmul_decode {name} FAIL:", str(e).splitlines()[0][:300], flush=True)

print("\n===== minimal_matmul decode config sweep =====", flush=True)
grids = [(grid.x, grid.y), (8, 8), (11, 8), (8, 10)]
configs = []
for (gx, gy), kb, nb, sw in itertools.product(grids, (2, 4, 8), (1, 2, 4, 8), (1, 2, 4)):
    if nb % sw != 0:
        continue
    configs.append((gx, gy, kb, nb, sw))
results = {"qkv": [], "swiglu": [], "down": []}
for gx, gy, kb, nb, sw in configs:
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=1,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=1,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )
    tag = f"grid{gx}x{gy} K{kb} N{nb} sw{sw}"
    for name, fn in (
        (
            "qkv",
            lambda: ttnn.experimental.minimal_matmul(
                x, wqkv, config=cfg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        ),
        (
            "swiglu",
            lambda: ttnn.experimental.minimal_matmul(
                x,
                wgu,
                fuse_swiglu=True,
                config=cfg,
                compute_kernel_config=compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        ),
        (
            "down",
            lambda: ttnn.experimental.minimal_matmul(
                act, wd, config=cfg, compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        ),
    ):
        if name == "swiglu" and nb % 2 != 0:
            continue
        try:
            t = timed_trace(fn, iters=5, reps=2)
            results[name].append((t, tag))
        except Exception as e:
            results[name].append((float("inf"), tag + " FAIL " + str(e).splitlines()[0][:120]))
for name in results:
    results[name].sort(key=lambda r: r[0])
    print(f"-- {name}: linear baseline {base[name]:.1f} us", flush=True)
    for t, tag in results[name][:6]:
        print(f"   {t:8.1f} us  {tag}", flush=True)
    fails = [tag for t, tag in results[name] if t == float("inf")]
    print(f"   failures: {len(fails)}", (fails[0] if fails else ""), flush=True)

ttnn.close_mesh_device(mesh)
print("DONE")
