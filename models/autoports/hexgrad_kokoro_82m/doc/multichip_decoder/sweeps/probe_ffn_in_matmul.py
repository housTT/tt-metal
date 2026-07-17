import sys as _s

_s.meta_path = [m for m in _s.meta_path if "editable" not in getattr(type(m), "__module__", "")]
_s.path = [p for p in _s.path if "model-bringup" not in p]
import time

import torch

import ttnn

mm = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)
dev = ttnn.open_device(device_id=0)
H = 768
I = 2048


def bench(M, make_pc, label, gx, gy, ibw, sbw, use_grid=False):
    a = ttnn.from_torch(torch.randn(1, 1, M, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    w = ttnn.from_torch(torch.randn(H, I), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(I), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    try:
        if use_grid:
            fn = lambda: ttnn.linear(
                a,
                w,
                bias=b,
                compute_kernel_config=mm,
                core_grid=ttnn.CoreGrid(y=8, x=10),
                dtype=ttnn.bfloat16,
                activation="gelu",
            )
        else:
            pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=ibw,
                out_subblock_h=1,
                out_subblock_w=sbw,
                per_core_M=(M // 32) // gy,
                per_core_N=(I // 32 + gx - 1) // gx,
                transpose_mcast=False,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),
            )
            fn = lambda: ttnn.linear(a, w, bias=b, compute_kernel_config=mm, program_config=pc, dtype=ttnn.bfloat16)
        for _ in range(3):
            r = fn()
            ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(200):
            r = fn()
        ttnn.synchronize_device(dev)
        print(f"M={M} {label:32s} {(time.perf_counter()-t0)/200*1e6:7.1f} us")
    except Exception as e:
        print(f"M={M} {label:32s} ERR {repr(e)[:90]}")


for M in (128, 64, 32):
    bench(M, None, "core_grid(default)", 0, 0, 0, 0, use_grid=True)
    mt = M // 32
    for gy in [g for g in (1, 2, 4) if mt % g == 0]:
        bench(M, None, f"g=8x{gy} ibw=8 sbw=2", 8, gy, 8, 2)
        bench(M, None, f"g=8x{gy} ibw=12 sbw=2", 8, gy, 12, 2)
        bench(M, None, f"g=8x{gy} ibw=24 sbw=2", 8, gy, 24, 2)
        bench(M, None, f"g=8x{gy} ibw=8 sbw=4", 8, gy, 8, 4)
ttnn.close_device(dev)
print("DONE")
