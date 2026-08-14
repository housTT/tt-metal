# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode program-config sweep for the dense matmuls at their **per-device** multichip shapes.

The single-chip stage tuned ``DECODE_MATMUL_GEOMETRY`` at the unsharded widths. Under TP=4 every one
of those roles is 4x narrower (and ``o_proj``/``gdn_out`` are 4x shallower instead), which moves both
axes of the tuning: the core-count target is reduced to a divisor of ``Nt`` inside
``_decode_1d_matmul_config``, and ``Nt`` has changed by 4x. This sweep re-derives the winner per role
at the local shape, in the family the layer ships (``mcast_in0``, DRAM-interleaved weight, L1 output),
under the weight dtype and fidelity the precision policy selects.

Runs on a 1x1 mesh: a per-device program config is a per-device question, and holding the sweep off
the 4-chip mesh keeps it independent of fabric state.

    python .../doc/multichip_decoder/logs/probe_dense_matmul.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
FIDELITIES = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}

#: ``(role, M, K, N, weight dtype, fidelity)`` at batch-1 decode, per device, TP=4.
#: ``single_chip`` is the same role's unsharded shape, for the record.
ROLES = [
    ("attn_in", 32, 2048, 2560, "bfloat8_b", "HiFi2", (2048, 9216)),
    ("o_proj", 32, 1024, 2048, "bfloat8_b", "HiFi2", (4096, 2048)),
    ("gdn_in", 32, 2048, 3136, "bfloat8_b", "HiFi2", (2048, 12352)),
    ("gdn_out", 32, 1024, 2048, "bfloat8_b", "HiFi2", (4096, 2048)),
    ("shared_in", 32, 2048, 288, "bfloat8_b", "HiFi2", (2048, 1056)),
    ("shared_down", 32, 128, 2048, "bfloat8_b", "HiFi2", (512, 2048)),
    ("router", 32, 2048, 256, "bfloat16", "HiFi4", (2048, 256)),
    ("expert_select", 32, 256, 64, "bfloat16", "HiFi4", None),
]

CORE_TARGETS = [4, 8, 16, 24, 32, 48, 64, 88, 110]
IN0_CAPS = [1, 2, 4, 8, 16, 32]


def largest_divisor_at_most(value, cap):
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def mcast1d_config(grid, cores, m, k, n, fp32_acc, in0_cap):
    cols = min(grid.x, cores)
    rows = math.ceil(cores / cols)
    if rows > grid.y:
        rows = grid.y
    m_t, k_t, n_t = math.ceil(m / TILE), math.ceil(k / TILE), math.ceil(n / TILE)
    per_core_n = math.ceil(n_t / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_t % i == 0 and i * sub_w <= cap)
    return (
        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cols, rows),
            in0_block_w=largest_divisor_at_most(k_t, in0_cap),
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=m_t,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        ),
        cols * rows,
        largest_divisor_at_most(k_t, in0_cap),
        per_core_n,
    )


def time_call(mesh, fn, iters, warmup=3, repeats=3):
    for _ in range(warmup):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    samples = []
    for _ in range(repeats):
        start = time.time()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        samples.append((time.time() - start) / iters * 1e6)
    return min(samples), max(samples) - min(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--roles", default="")
    args = ap.parse_args()

    wanted = set(filter(None, args.roles.split(",")))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    print(f"# DENSE decode program-config sweep, per-device TP=4 shapes. grid {grid.x}x{grid.y}")
    print(f"# columns: role M K N family cores in0_block_w per_core_N us spread")
    try:
        gen = torch.Generator().manual_seed(11)
        for name, m, k, n, wdt, fidname, single in ROLES:
            if wanted and name not in wanted:
                continue
            wd, fid = DTYPES[wdt], FIDELITIES[fidname]
            ckc = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=fid,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            x = ttnn.from_torch(
                torch.randn(1, 1, m, k, generator=gen) * 0.1,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            w = ttnn.from_torch(
                torch.randn(1, 1, k, n, generator=gen) * 0.02,
                dtype=wd,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            print(f"# --- {name} M={m} K={k} N={n} weight={wdt} fidelity={fidname} single_chip={single}")

            def default():
                return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)

            try:
                us, spread = time_call(mesh, default, args.iters, repeats=args.repeats)
                print(f"DENSE {name} {m} {k} {n} default - - - {us:.2f} {spread:.2f}")
            except Exception as exc:  # noqa: BLE001
                print(f"DENSE {name} {m} {k} {n} default - - - FAIL {type(exc).__name__}")

            for cores in CORE_TARGETS:
                for cap in IN0_CAPS:
                    built = mcast1d_config(grid, cores, m, k, n, False, cap)
                    if built is None:
                        continue
                    cfg, realised, block_w, per_core_n = built

                    def run(cfg=cfg):
                        return ttnn.linear(
                            x,
                            w,
                            compute_kernel_config=ckc,
                            program_config=cfg,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )

                    try:
                        us, spread = time_call(mesh, run, args.iters, repeats=args.repeats)
                        print(
                            f"DENSE {name} {m} {k} {n} mcast1d {realised} {block_w} {per_core_n} "
                            f"{us:.2f} {spread:.2f}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"DENSE {name} {m} {k} {n} mcast1d {realised} {block_w} {per_core_n} "
                            f"FAIL {type(exc).__name__}"
                        )
            ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
