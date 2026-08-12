# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Program-config / dtype / memory geometry sweep for the two routed-expert ``ttnn.sparse_matmul``
calls, at Ornith-1.0-35B's exact decode shapes.

The two calls are 43 % of the optimized traced decode window, so OPT-004 requires a geometry sweep
under the dtype/fidelity policy that is actually being selected — not one measured under a different
precision. Every candidate here therefore takes ``--weight-dtype``/``--fidelity`` and the sweep is
run once per policy that matters.

Shapes (decode, one 32-row token group, ``num_experts`` = 256, ``num_experts_per_tok`` = 8):

* gate/up (packed): ``[1, 1, 32, 2048] x [1, 256, 2048, 1024]``
* down:             ``[1, 256, 32, 512] x [1, 256, 512, 2048]``  (``is_input_a_sparse=True``)

Timing is a warmed loop of ``--iters`` launches with one synchronize at the end, so a row is the
op's steady-state device+dispatch cost at that geometry, comparable across rows.

    python .../logs/probe_sparse_matmul.py --weight-dtype bfloat4_b --fidelity LoFi
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn

TILE = 32
E = 256
TOPK = 8
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


def _bind(run, nnz, mesh):
    def call():
        return run(nnz=nnz)

    call.mesh = mesh
    return call


def time_op(fn, iters, warmup=3, repeats=3):
    """Min-of-``repeats`` mean-of-``iters`` microseconds, and the spread across repeats.

    The spread is reported because this sweep's decisions turn on differences of a few percent — review
    round 5 found the shipped grid *orientation* 1-3 % behind the other rectangle at four of eight
    points, and without a spread there was no way to tell a real 2 % from noise. Now there is.
    """
    for _ in range(warmup):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(fn.mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            # Freed inside the loop, not collected: an L1 output at these shapes is ~8 MB, so holding
            # `iters` of them turns every L1 candidate into a bank-allocation failure and silently
            # deletes the L1 arm of the sweep.
            ttnn.deallocate(fn())
        ttnn.synchronize_device(fn.mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return min(samples), max(samples) - min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dtype", default="bfloat4_b")
    ap.add_argument("--act-dtype", default="bfloat8_b")
    ap.add_argument("--fidelity", default="LoFi")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--role", default="both", choices=["gate_up", "down", "both"])
    ap.add_argument(
        "--nnz",
        action="store_true",
        help="DANGEROUS. Also time the static-nnz fixed-count path (nnz == --active). The factory "
        "comments claim the sender validates count_nonzero(sparsity) against nnz on device and fails "
        "loudly rather than deadlocking (sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp, "
        "tt-metal #45943) — at Ornith's shapes it does NOT: with an exactly matching count "
        "(count_nonzero == nnz == 8) the very first candidate hung the device inside "
        "SparseMatmulDeviceOperation and needed a tt-smi reset. Evidence: "
        "doc/optimized_decoder/triage/. Do not enable this without being ready to reset.",
    )
    ap.add_argument(
        "--active",
        type=int,
        default=TOPK,
        help="non-zero experts in the sparsity tensor: 8 is the batch-1 decode group, ~162 the "
        "expected distinct union of a 32-token prefill group (256 draws from 256 experts)",
    )
    args = ap.parse_args()

    wd = DTYPES[args.weight_dtype]
    ad = DTYPES[args.act_dtype]
    fid = FIDELITIES[args.fidelity]

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    print(
        f"# grid {grid.x}x{grid.y}  weight={args.weight_dtype} act={args.act_dtype} "
        f"fidelity={args.fidelity} active={args.active}/{E}"
    )
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        gen = torch.Generator().manual_seed(7)
        # sparsity: exactly --active experts, matching what the router produces for that phase
        sparse_t = torch.zeros(1, 1, 1, E)
        sparse_t[0, 0, 0, torch.randperm(E, generator=gen)[: args.active]] = 1.0
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
                    upload(torch.randn(1, E, H, 2 * I, generator=gen) * 0.02, wd),
                    False,
                    H,
                    2 * I,
                )
            )
        if args.role in ("down", "both"):
            roles.append(
                (
                    "down",
                    upload(torch.randn(1, E, TILE, I, generator=gen) * 0.1, ad),
                    upload(torch.randn(1, E, I, H, generator=gen) * 0.02, wd),
                    True,
                    I,
                    H,
                )
            )

        for name, a, b, a_sparse, k, n in roles:
            kt, nt = k // TILE, n // TILE
            best = None
            # Curated rather than exhaustive: every extra candidate costs a kernel build, so the
            # sweep covers the axes OPT-004 names (logical core count via per_core_N, grid shape,
            # in0_block_w up the full divisor ladder of Kt, output block/subblock width, and output
            # placement) without the full cross product of all four at once.
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

                                def run(cfg=cfg, mem=mem, nnz=None):
                                    return ttnn.sparse_matmul(
                                        a,
                                        b,
                                        sparsity=sparsity,
                                        nnz=nnz,
                                        memory_config=mem,
                                        output_tile=output_tile,
                                        program_config=cfg,
                                        is_input_a_sparse=a_sparse,
                                        compute_kernel_config=ckc,
                                        dtype=ad,
                                    )

                                run.mesh = mesh
                                tag = (
                                    f"active={args.active} role={name} cores={cores}({cx}x{cy}) in0_block_w={in0_block_w} "
                                    f"per_core_N={per_core_N} out_block_w={out_block_w} sub_w={sub_w} mem={mem_name}"
                                )
                                try:
                                    us, spread = time_op(run, args.iters, repeats=args.repeats)
                                except Exception as exc:  # noqa: BLE001 - illegal geometries are data
                                    print(f"SPARSE {tag} FAILED {str(exc).splitlines()[0][:110]}", flush=True)
                                    continue
                                print(f"SPARSE {tag} nnz=inferred us={us:.1f} spread={spread:.1f}", flush=True)
                                if args.nnz:
                                    # DANGEROUS: see --nnz's help. This wedged the device the one
                                    # time it was run; it is kept so the finding is reproducible, and
                                    # it is deliberately NOT part of run_evidence.sh.
                                    try:
                                        us_static, sp_static = time_op(
                                            _bind(run, args.active, mesh), args.iters, repeats=args.repeats
                                        )
                                        print(
                                            f"SPARSE {tag} nnz={args.active} us={us_static:.1f} "
                                            f"spread={sp_static:.1f}",
                                            flush=True,
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        print(
                                            f"SPARSE {tag} nnz={args.active} FAILED {str(exc).splitlines()[0][:90]}",
                                            flush=True,
                                        )
                                if best is None or us < best[0]:
                                    best = (round(us, 1), tag)
            print(f"SPARSEBEST role={name} {best}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
