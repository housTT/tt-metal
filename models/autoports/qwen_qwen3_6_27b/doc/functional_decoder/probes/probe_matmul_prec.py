"""A/B the compute-kernel config used by the delta-rule matmuls on the real attn0 matrices."""
import math

import torch

import ttnn

torch.manual_seed(0)
C = 64
B = 64
# strictly lower triangular, entries of the magnitude real weights produce (|a| <= 0.74)
A = torch.zeros(B, 1, C, C)
tri = torch.tril(torch.ones(C, C), diagonal=-1).bool()
A[:, 0][:, tri] = torch.rand(B, int(tri.sum())) * 1.4 - 0.7

I64 = torch.eye(C, dtype=torch.float64)
Ad = A[:, 0].to(torch.float64)
inv_ref = torch.linalg.solve_triangular(I64 - Ad, I64.expand(B, C, C).contiguous(), upper=False)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)


def run(dtype, fidelity, fp32_acc, packer_acc):
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer_acc,
    )
    tt = ttnn.from_torch(A, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    eye = ttnn.from_torch(
        torch.eye(C).reshape(1, 1, C, C),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # single matmul error
    sq = ttnn.matmul(tt, tt, dtype=dtype, compute_kernel_config=cfg)
    sq_err = float((ttnn.to_torch(sq).to(torch.float64) - (Ad @ Ad).reshape(B, 1, C, C)).abs().max())
    ttnn.deallocate(sq)
    inv = ttnn.add(tt, eye)
    power = tt
    for _ in range(int(math.log2(C)) - 1):
        power = ttnn.matmul(power, power, dtype=dtype, compute_kernel_config=cfg)
        inv = ttnn.matmul(inv, ttnn.add(power, eye), dtype=dtype, compute_kernel_config=cfg)
    got = ttnn.to_torch(inv).reshape(B, C, C).to(torch.float64)
    err = float((got - inv_ref).abs().max())
    print(
        f"dtype={str(dtype):22s} fid={str(fidelity).split('.')[-1]:6s} fp32_acc={fp32_acc!s:5s} "
        f"packer={packer_acc!s:5s} single_matmul_err={sq_err:.3e} inv_err={err:.3e} "
        f"inv_absmax={float(got.abs().max()):.3e}",
        flush=True,
    )


print(f"reference inv absmax={float(inv_ref.abs().max()):.4e}", flush=True)
for dtype in (ttnn.float32, ttnn.bfloat16):
    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2):
        for fp32_acc in (True, False):
            for packer in (True, False):
                try:
                    run(dtype, fid, fp32_acc, packer)
                except Exception as exc:  # noqa: BLE001
                    print(f"dtype={dtype} fid={fid} fp32_acc={fp32_acc} packer={packer} FAILED {exc}", flush=True)
ttnn.close_mesh_device(mesh)
