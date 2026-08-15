# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The routed ``ttnn.sparse_matmul`` geometry sweep, re-run at the **per-device** operating point.

The single-chip sweep (``doc/optimized_decoder/logs/probe_sparse_matmul.py``) selected
``SPARSE_CORES_PER_ACTIVE``, ``SPARSE_MIN_CORES``/``SPARSE_MAX_CORES`` and
``SPARSE_GATE_UP_IN0_BLOCK_W`` at ``E = 256`` and ``active ∈ {8, 32, 64, 162}``. Expert parallelism
moves this stage to ``E = 64`` per device and ``active ≈ 4`` at batch-1 decode / ``≈ 41`` for a
32-token prefill group — points that sweep does not contain. README section 5.6 re-swept every *dense*
role for exactly this reason ("the local shape is different"), and the routed matmuls are the single
dominant op in all four profiler captures, so leaving them on an extrapolated geometry would be the
larger omission of the two. Review round 1 of this stage raised it.

This is the single-chip probe with one change: ``E`` is a flag instead of a constant, so the same
candidate ladder can be measured at the per-device expert count. Everything else — the curated
candidate set, the timing method (min of ``--repeats`` means of ``--iters``, with the spread), the
BFP4/LoFi default policy, the deallocate-inside-the-loop rule that keeps L1 candidates measurable —
is copied unchanged so the two sweeps are directly comparable.

Note what does **not** change under EP: ``Nt`` is 32 tiles for the packed gate/up and 64 for the down
projection either way, because ``moe_intermediate_size`` (512) and ``dim`` (2048) are not sharded.
The candidate ladder is therefore identical to the single-chip one and only the operating point
moves. That is the whole question this probe answers: does the same geometry still win when each
call sweeps 4 experts out of 64 instead of 8 out of 256?

    python .../doc/multichip_decoder/logs/probe_sparse_matmul_local.py --experts 64 --active 4
    python .../doc/multichip_decoder/logs/probe_sparse_matmul_local.py --experts 64 --active 41

The ``--nnz`` static-count arm of the single-chip probe is deliberately **not** carried over: it
wedged the device the one time it was run (``doc/optimized_decoder/triage/``), and this stage has no
reason to re-run it.
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt import multichip_decoder as MC
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import (
    SPARSE_CORES_PER_ACTIVE,
    SPARSE_GATE_UP_IN0_BLOCK_W,
    SPARSE_MIN_CORES,
    _largest_divisor_at_most,
    _sparse_cores,
)

TILE = 32
H = 2048
I = 512

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def grids_for(cores, grid):
    """Rectangles of exactly ``cores`` cores that fit ``grid``; the op requires an exact tile."""
    out = []
    for y in range(1, grid.y + 1):
        if cores % y == 0 and cores // y <= grid.x:
            out.append((cores // y, y))
    return out


def prog_config(cx, cy, in0_block_w, per_core_M, per_core_N, out_block_w, sub_h, sub_w):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_M,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def time_op(fn, iters, warmup=3, repeats=3):
    """Min-of-``repeats`` mean-of-``iters`` microseconds, and the spread across repeats."""
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(fn.mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(fn.mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return min(samples), max(samples) - min(samples)


def shipped_choice(role, active, experts, nt):
    """What each rule picks for ``role`` here, as ``(inherited_realised, multichip_realised, cap)``.

    Both the inherited target and this stage's rescaled one are reduced to the largest divisor of
    ``Nt`` at or below the target, which is what ``_sparse_matmul_config`` does and therefore what
    actually runs. Printing the *target* instead would have hidden the whole finding: at
    ``active = 63`` the inherited target is 31, which realises as 16 cores, not 31.
    """
    inherited = _largest_divisor_at_most(nt, max(1, _sparse_cores(role, active)))
    scale = SPARSE_CORES_PER_ACTIVE[role] if MC.SPARSE_SCALE_CORES_BY_TP else 1
    multichip = _largest_divisor_at_most(nt, max(1, _sparse_cores(role, active * scale)))
    cap = None
    if role == "gate_up":
        cap = SPARSE_GATE_UP_IN0_BLOCK_W[multichip > SPARSE_MIN_CORES]
    return inherited, multichip, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dtype", default="bfloat4_b")
    ap.add_argument("--act-dtype", default="bfloat8_b")
    ap.add_argument("--fidelity", default="LoFi")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--role", default="both", choices=["gate_up", "down", "both"])
    ap.add_argument(
        "--experts",
        type=int,
        default=None,
        help="expert-axis extent. Default is the per-device count under this stage's EP: "
        "num_experts // DEFAULT_TP = 64.",
    )
    ap.add_argument(
        "--active",
        type=int,
        default=4,
        help="non-zero experts in the sparsity tensor, per device per group. 4 is the measured "
        "batch-1 decode count; 41 is the expected distinct union for a 32-token prefill group, "
        "E*(1-(1-1/E)^(32*top_k/tp)) at E=64 — the draws are divided by tp because only that "
        "fraction lands on this device, which review round 2 found missing (it gave 63). "
        "run_evidence.sh sweeps 4/8/16/32/63 so the whole decode-to-prefill range is bracketed.",
    )
    args = ap.parse_args()

    experts = args.experts if args.experts is not None else 256 // MC.DEFAULT_TP
    wd, ad, fid = DTYPES[args.weight_dtype], DTYPES[args.act_dtype], FIDELITIES[args.fidelity]

    # A 1x1 mesh: this op is entirely device-local under EP (no collective is involved), so measuring
    # it on one chip is the same measurement and leaves the fabric out of the timing.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    print(
        f"# grid {grid.x}x{grid.y}  weight={args.weight_dtype} act={args.act_dtype} "
        f"fidelity={args.fidelity} active={args.active}/{experts}"
    )
    for role, nt in (("gate_up", 2 * I // TILE), ("down", H // TILE)):
        inherited, multichip, cap = shipped_choice(role, args.active, experts, nt)
        print(
            f"# SHIPPED role={role} active={args.active} Nt={nt} inherited_realised_cores={inherited} "
            f"multichip_realised_cores={multichip} in0_block_w={cap}"
        )
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        gen = torch.Generator().manual_seed(7)
        sparse_t = torch.zeros(1, 1, 1, experts)
        sparse_t[0, 0, 0, torch.randperm(experts, generator=gen)[: min(args.active, experts)]] = 1.0
        sparsity = ttnn.from_torch(
            sparse_t,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tile = ttnn.Tile([TILE, TILE])

        def upload(t, dtype, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, memory_config=mem)

        roles = []
        if args.role in ("gate_up", "both"):
            roles.append(
                (
                    "gate_up",
                    upload(torch.randn(1, 1, TILE, H, generator=gen) * 0.1, ttnn.bfloat16),
                    upload(torch.randn(1, experts, H, 2 * I, generator=gen) * 0.02, wd),
                    False,
                    H,
                    2 * I,
                )
            )
        if args.role in ("down", "both"):
            roles.append(
                (
                    "down",
                    upload(torch.randn(1, experts, TILE, I, generator=gen) * 0.1, ad),
                    upload(torch.randn(1, experts, I, H, generator=gen) * 0.02, wd),
                    True,
                    I,
                    H,
                )
            )

        for name, a, b, a_sparse, k, n in roles:
            kt, nt = k // TILE, n // TILE
            best = None
            for per_core_N in [d for d in divisors(nt) if nt // d <= grid.x * grid.y][:4]:
                cores = nt // per_core_N
                shapes = grids_for(cores, grid)
                for cx, cy in shapes[:1] + shapes[-1:] if len(shapes) > 1 else shapes:
                    for in0_block_w in divisors(kt):
                        blocks = {(per_core_N, min(per_core_N, 8)), (per_core_N, min(per_core_N, 4)), (1, 1)}
                        if per_core_N >= 2:
                            blocks.add((per_core_N, 2))
                        for out_block_w, sub_w in sorted(blocks):
                            if out_block_w > per_core_N or sub_w > out_block_w or per_core_N % out_block_w:
                                continue
                            if out_block_w % sub_w:
                                continue
                            for mem_name, mem in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
                                cfg = prog_config(cx, cy, in0_block_w, 1, per_core_N, out_block_w, 1, sub_w)

                                def run(cfg=cfg, mem=mem):
                                    return ttnn.sparse_matmul(
                                        a,
                                        b,
                                        sparsity=sparsity,
                                        nnz=None,
                                        memory_config=mem,
                                        output_tile=output_tile,
                                        program_config=cfg,
                                        is_input_a_sparse=a_sparse,
                                        compute_kernel_config=ckc,
                                        dtype=ad,
                                    )

                                run.mesh = mesh
                                tag = (
                                    f"experts={experts} active={args.active} role={name} cores={cores}({cx}x{cy}) "
                                    f"in0_block_w={in0_block_w} per_core_N={per_core_N} "
                                    f"out_block_w={out_block_w} sub_w={sub_w} mem={mem_name}"
                                )
                                try:
                                    us, spread = time_op(run, args.iters, repeats=args.repeats)
                                except Exception as exc:  # noqa: BLE001 - illegal geometries are data
                                    print(f"SPARSEL {tag} FAILED {str(exc).splitlines()[0][:110]}", flush=True)
                                    continue
                                print(f"SPARSEL {tag} us={us:.1f} spread={spread:.1f}", flush=True)
                                if best is None or us < best[0]:
                                    best = (round(us, 1), tag)
            print(f"SPARSELBEST role={name} {best}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
