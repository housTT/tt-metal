# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Program-config family sweep for the dense decode matmuls of the Ornith-1.0-35B decoder.

Three families are compared at each role's real decode shape, under the weight dtype and math
fidelity the precision policy actually selects:

``default``
    what the fused stage shipped — ``ttnn.linear`` with a compute-kernel config and no program
    config, i.e. ttnn's own heuristic, DRAM-interleaved weight, DRAM-interleaved output.
``mcast1d``
    an explicit ``MatmulMultiCoreReuseMultiCast1DProgramConfig`` (``mcast_in0``) on a named core
    grid with an L1 output and a DRAM-interleaved weight. This is the family
    ``models/demos/blackhole/qwen36`` reports as the winner for skinny Blackhole decode matmuls.
``dram_sharded``
    ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` with the weight DRAM width-sharded
    over the DRAM banks and the activation and output L1 width-sharded — the layout
    ``tech_reports/LLMs/llms.md`` prescribes for small-M large-weight decode matmuls.

Roles are the packed attention in-projection, the attention output projection, the packed DeltaNet
in-projection, the DeltaNet output projection, the shared expert's packed ``gate|up|router`` matmul
and its down projection, and the 256-way router.

    python .../logs/probe_dense_matmul.py --weight-dtype bfloat8_b --fidelity HiFi2
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn

TILE = 32
DRAM_BANKS = 8

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

#: (name, M, K, N) at batch-1 decode, M tile-padded exactly as the layer runs it.
ROLES = [
    ("attn_in", 32, 2048, 9216),
    ("o_proj", 32, 4096, 2048),
    ("gdn_in", 32, 2048, 12352),
    ("gdn_out", 32, 4096, 2048),
    ("shared_in", 32, 2048, 1056),
    ("shared_down", 32, 512, 2048),
    ("router", 32, 2048, 256),
]


def divisors(n, cap=None):
    return [d for d in range(1, (cap or n) + 1) if n % d == 0]


def largest_divisor_at_most(value, cap):
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def time_call(mesh, fn, iters, warmup=3, repeats=3):
    """Min-of-``repeats`` mean-of-``iters`` microseconds, and the spread across repeats.

    These rows differ by fractions of a microsecond and the shipped geometry is chosen from them, so the
    spread is measured rather than assumed: review round 5 pointed out that five of the seven shipped
    dense roles sit 0.2-0.3 us behind another candidate, which is only meaningful against a spread.
    """
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


def mcast1d_config(grid, cores, m, k, n, fp32_acc, in0_block_w=None):
    cols = min(grid.x, cores)
    rows = math.ceil(cores / cols)
    if rows > grid.y:
        return None
    m_t, k_t, n_t = math.ceil(m / TILE), math.ceil(k / TILE), math.ceil(n / TILE)
    per_core_n = math.ceil(n_t / (cols * rows))
    cap = 4 if fp32_acc else 8
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_t % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=largest_divisor_at_most(k_t, in0_block_w or 32),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_t,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def dram_weight_config(k, n):
    padded_n = TILE * DRAM_BANKS * math.ceil(n / (TILE * DRAM_BANKS))
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_BANKS - 1, 0))}),
        (k, padded_n // DRAM_BANKS),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def act_shard_config(grid, k, cores):
    """L1 width-shard of a ``[32, k]`` activation over exactly ``cores`` cores.

    The rectangle has to be exact — a partially filled row would leave a core without a shard — so
    the widest legal ``cols <= grid.x`` that divides ``cores`` is chosen rather than
    ``min(grid.x, cores)``, which silently dropped every core count that is not a multiple of the
    grid width (16, 32, 64 on an 11-wide grid).
    """
    k_t = k // TILE
    if k_t % cores:
        return None
    cols = max((c for c in range(1, grid.x + 1) if cores % c == 0 and cores // c <= grid.y), default=0)
    if not cols:
        return None
    rows = cores // cols
    return ttnn.create_sharded_memory_config(
        shape=(TILE, k // cores),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def dram_sharded_config(m, k, n, cores, in0_block_w):
    k_t, n_t = k // TILE, math.ceil(n / TILE)
    n_padded_t = DRAM_BANKS * math.ceil(n_t / DRAM_BANKS)
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / TILE),
        per_core_N=n_padded_t // DRAM_BANKS,
        fused_activation=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dtype", default="bfloat8_b")
    ap.add_argument("--fidelity", default="HiFi2")
    ap.add_argument("--fp32-acc", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--roles", default="")
    ap.add_argument("--families", default="default,mcast1d,dram_sharded")
    ap.add_argument("--in0-blocks", default="32", help="comma-separated in0_block_w caps for mcast1d")
    args = ap.parse_args()

    wd = DTYPES[args.weight_dtype]
    fid = FIDELITIES[args.fidelity]
    wanted = set(filter(None, args.roles.split(",")))
    families = set(args.families.split(","))

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
    grid = mesh.compute_with_storage_grid_size()
    print(f"# grid {grid.x}x{grid.y} weight={args.weight_dtype} fidelity={args.fidelity} fp32_acc={args.fp32_acc}")
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=fid,
            math_approx_mode=False,
            fp32_dest_acc_en=args.fp32_acc,
            packer_l1_acc=True,
        )
        gen = torch.Generator().manual_seed(11)
        for name, m, k, n in ROLES:
            if wanted and name not in wanted:
                continue
            x_t = torch.randn(1, 1, m, k, generator=gen) * 0.1
            w_t = torch.randn(1, 1, k, n, generator=gen) * 0.02
            x = ttnn.from_torch(
                x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            w = ttnn.from_torch(
                w_t, dtype=wd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            def default():
                return ttnn.linear(x, w, compute_kernel_config=ckc)

            if "default" in families:
                print(
                    f"DENSE role={name} m={m} k={k} n={n} family=default "
                    + "us={:.1f} spread={:.1f}".format(*time_call(mesh, default, args.iters, repeats=args.repeats)),
                    flush=True,
                )

            for cores in (8, 16, 24, 32, 48, 64, 80, 96, 110) if "mcast1d" in families else ():
                if cores > grid.x * grid.y:
                    continue
                for in0_cap in [int(v) for v in args.in0_blocks.split(",")]:
                    cfg = mcast1d_config(grid, cores, m, k, n, args.fp32_acc, in0_cap)
                    if cfg is None:
                        continue
                    for mem_name, mem in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):

                        def run(cfg=cfg, mem=mem):
                            return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=cfg, memory_config=mem)

                        tag = f"DENSE role={name} m={m} k={k} n={n} family=mcast1d cores={cores} in0_block_w={cfg.in0_block_w} per_core_N={cfg.per_core_N} out={mem_name}"
                        try:
                            print(
                                f"{tag} "
                                + "us={:.1f} spread={:.1f}".format(
                                    *time_call(mesh, run, args.iters, repeats=args.repeats)
                                ),
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - illegal geometries are data
                            print(f"{tag} FAILED {str(exc).splitlines()[0][:100]}", flush=True)
            ttnn.deallocate(w)

            # DRAM-sharded family: weight width-sharded over the DRAM banks, activation and output
            # L1 width-sharded on the same logical core count.
            w_ds = (
                ttnn.from_torch(
                    w_t, dtype=wd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram_weight_config(k, n)
                )
                if "dram_sharded" in families
                else None
            )
            for cores in (8, 16, 32, 64) if "dram_sharded" in families else ():
                act_cfg = act_shard_config(grid, k, cores)
                if act_cfg is None:
                    continue
                k_per_core = (k // cores) // TILE
                for in0_block_w in [d for d in divisors(k_per_core) if d >= 1][-3:]:
                    cfg = dram_sharded_config(m, k, n, cores, in0_block_w)
                    tag = f"DENSE role={name} m={m} k={k} n={n} family=dram_sharded act_cores={cores} in0_block_w={in0_block_w} per_core_N={cfg.per_core_N}"
                    try:
                        x_sh = ttnn.to_memory_config(x, act_cfg)
                    except Exception as exc:  # noqa: BLE001
                        print(f"{tag} FAILED-shard {str(exc).splitlines()[0][:90]}", flush=True)
                        continue

                    def run(cfg=cfg, x_sh=x_sh):
                        return ttnn.linear(
                            x_sh,
                            w_ds,
                            compute_kernel_config=ckc,
                            program_config=cfg,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        )

                    try:
                        print(
                            f"{tag} "
                            + "us={:.1f} spread={:.1f}".format(*time_call(mesh, run, args.iters, repeats=args.repeats)),
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"{tag} FAILED {str(exc).splitlines()[0][:100]}", flush=True)
                    ttnn.deallocate(x_sh)
            if w_ds is not None:
                ttnn.deallocate(w_ds)
            ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
