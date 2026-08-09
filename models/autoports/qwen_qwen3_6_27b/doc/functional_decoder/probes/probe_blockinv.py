"""Measure TTNN accuracy of block-recursive unit-lower-triangular inversion on real attn0."""
import math

import torch

import ttnn

A = torch.load("/tmp/attn0_real.pt").float()  # [B, 64, 64] strictly lower triangular
B, N, _ = A.shape
I64 = torch.eye(N, dtype=torch.float64)
exact = torch.linalg.solve_triangular(I64 - A.double(), I64.expand(B, N, N).contiguous(), upper=False)
print(f"batch={B} exact |inv|max={float(exact.abs().max()):.4f}", flush=True)

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
DT = ttnn.float32


def to_dev(t):
    return ttnn.from_torch(t, dtype=DT, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def mm(a, b):
    return ttnn.matmul(a, b, dtype=DT, compute_kernel_config=CFG)


EYES = {}


def eye(n):
    if n not in EYES:
        EYES[n] = to_dev(torch.eye(n).reshape(1, 1, n, n))
    return EYES[n]


def doubling(a, n):
    """(I - a)^-1 as prod_j (I + a^(2^j)); accurate only while a is small/nilpotent fast."""
    inv = ttnn.add(a, eye(n))
    power = a
    for _ in range(int(math.log2(n)) - 1):
        power = mm(power, power)
        inv = mm(inv, ttnn.add(power, eye(n)))
    return inv


def block_inv(a, n, base):
    """Recursive 2x2 block inversion down to ``base``; both diagonal blocks in one batch."""
    if n <= base:
        return doubling(a, n)
    h = n // 2
    lead = int(a.shape[0])
    a11 = ttnn.slice(a, [0, 0, 0, 0], [lead, 1, h, h])
    a22 = ttnn.slice(a, [0, 0, h, h], [lead, 1, n, n])
    a21 = ttnn.slice(a, [0, 0, h, 0], [lead, 1, n, h])
    diag = ttnn.concat([a11, a22], dim=0)
    xd = block_inv(diag, h, base)
    x11 = ttnn.slice(xd, [0, 0, 0, 0], [lead, 1, h, h])
    x22 = ttnn.slice(xd, [lead, 0, 0, 0], [2 * lead, 1, h, h])
    x21 = mm(x22, mm(a21, x11))
    zero = ttnn.zeros((lead, 1, h, h), dtype=DT, layout=ttnn.TILE_LAYOUT, device=mesh)
    top = ttnn.concat([x11, zero], dim=-1)
    bottom = ttnn.concat([x21, x22], dim=-1)
    return ttnn.concat([top, bottom], dim=-2)


dev_a = to_dev(A.reshape(B, 1, N, N))
for label, fn in (
    ("doubling@64", lambda: doubling(dev_a, N)),
    ("block base=32", lambda: block_inv(dev_a, N, 32)),
    ("block base=16", lambda: block_inv(dev_a, N, 16)),
    ("block base=8", lambda: block_inv(dev_a, N, 8)),
):
    got = ttnn.to_torch(fn()).reshape(B, N, N).double()
    err = (got - exact).abs()
    print(
        f"{label:16s} |got|max={float(got.abs().max()):.4e} max_abs_err={float(err.max()):.4e} "
        f"mean_abs_err={float(err.mean()):.4e}",
        flush=True,
    )
ttnn.close_mesh_device(mesh)
