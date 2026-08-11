# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Chase the one ``Bound=SLOW`` decode row that is a material share of the step.

The committed optimized decode reports classify every BFP8 projection ``Bound=DRAM`` at 85-90 % of
the roofline, but the two **BFP4** gate/up rows come back ``Bound=SLOW`` at 54 % of the DRAM roofline
*and* 54 % of the FLOP roofline - neither ceiling - on the 12 compute cores the DRAM-sharded matmul
allocates itself.  Together they are about a third of the traced ``full_attention`` decode step, so
"SLOW" here is worth a search rather than a footnote.

The hypothesis the numbers suggest: quartering the weight bytes moved that matmul off the bandwidth
ceiling, and 12 cores is then not enough compute.  If that is right, a program config that puts the
same matmul on more cores should beat it - even if it gives up the DRAM-sharded reader.  This probe
measures every legal way to run that exact matmul:

* the shipped DRAM-sharded config, as the baseline;
* ``ttnn.linear``'s own heuristic on an interleaved weight;
* an explicit 1D multicast config over several core grids, with the weight interleaved and
  width-sharded in DRAM;
* an explicit 2D config, for completeness;

and repeats the whole thing at BFP8 so the comparison separates "BFP4 is compute-bound" from "this
shape prefers a different config".

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_bfp4_gateup.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import torch

import ttnn

TILE = 32
REPEATS = 25
#: ``(label, M, K, N)`` - the shapes this probe chases, all from the committed decode reports.
SHAPES = [
    ("mlp_gate / mlp_up (split)", 32, 5120, 17408),
    ("mlp_gate_up (packed)", 32, 5120, 34816),
    ("mlp_down", 32, 17408, 5120),
]
DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
TILE_BYTES = {ttnn.bfloat4_b: 576, ttnn.bfloat8_b: 1088, ttnn.bfloat16: 2048}
_DEVICE = None


def _median_stdev(samples):
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def _divisors(value):
    return [d for d in range(1, value + 1) if value % d == 0]


def _largest_at_most(value, limit):
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def dram_sharded_cfg(k, n):
    dram = _DEVICE.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    padded = math.ceil(n / (TILE * dram.x)) * TILE * dram.x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded // dram.x), ttnn.ShardOrientation.ROW_MAJOR),
    )


def width_sharded_l1(rows, width, cores):
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // cores),
        core_grid=ttnn.num_cores_to_corerangeset(cores, _DEVICE.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def pcc(golden, actual):
    a = golden.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return float((a @ b) / (a.norm() * b.norm()))


def time_call(fn):
    out = fn()
    ttnn.synchronize_device(_DEVICE)
    ttnn.deallocate(out)
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(_DEVICE)
        samples.append((time.perf_counter() - start) * 1e6)
        ttnn.deallocate(out)
    return _median_stdev(samples)


def run_shape(label, m, k, n, dt_name, results):
    dtype = DTYPES[dt_name]
    torch.manual_seed(0)
    act_t = torch.randn(1, 1, m, k) * 0.05
    w_t = torch.randn(1, 1, k, n) * 0.02
    golden = (act_t.to(torch.float64) @ w_t.to(torch.float64)).float()
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    act_dram = ttnn.from_torch(
        act_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w_inter = ttnn.from_torch(
        w_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w_shard = ttnn.from_torch(w_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=dram_sharded_cfg(k, n))
    k_tiles, n_tiles, m_tiles = k // TILE, n // TILE, m // TILE

    def record(candidate, median, stdev, value, extra=None, error=None):
        row = {
            "sweep": "bfp4_gateup",
            "shape": label,
            "M": m,
            "K": k,
            "N": n,
            "dtype": dt_name,
            "candidate": candidate,
            "median_us": median,
            "stdev_us": stdev,
            "pcc": value,
            "error": error,
        }
        row.update(extra or {})
        results.append(row)
        print(
            f"  {label:26s} {dt_name:5s} {candidate:44s} "
            f"{median if median is not None else float('nan'):8.1f} us +-{stdev if stdev else 0:6.1f} "
            f"pcc={value if value is not None else float('nan'):.6f}" + (f"  ERROR {error}" if error else ""),
            flush=True,
        )

    def measure(candidate, fn, extra=None):
        try:
            median, stdev = time_call(fn)
            out = fn()
            value = pcc(golden, ttnn.to_torch(out).float())
            ttnn.deallocate(out)
            record(candidate, median, stdev, value, extra)
        except Exception as exc:  # noqa: BLE001 - a blocker is a result
            record(candidate, None, None, None, extra, str(exc)[:180])

    # 1. the shipped DRAM-sharded form, at the shipped 32-core activation shard
    cores = 32
    if k_tiles % cores == 0 and n_tiles % cores == 0:
        per_core_n = n_tiles // cores
        best = 1
        for candidate in _divisors(k_tiles // cores):
            l1 = 2 * candidate * m_tiles * 2048 + 2 * candidate * per_core_n * TILE_BYTES[dtype]
            l1 += m_tiles * per_core_n * 2048
            if l1 <= 350_000:
                best = candidate
        act_sharded = ttnn.to_memory_config(act_dram, width_sharded_l1(m, k, cores))
        out_cfg = width_sharded_l1(m, n, cores)
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=best, per_core_M=m_tiles, per_core_N=per_core_n, fused_activation=None
        )
        measure(
            "dram-sharded (shipped)",
            lambda: ttnn.linear(
                act_sharded,
                w_shard,
                program_config=pc,
                memory_config=out_cfg,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ck,
            ),
            {"cores": cores, "in0_block_w": best, "per_core_N": per_core_n},
        )
    else:
        act_sharded = None

    # 2. the heuristic on an interleaved weight
    measure(
        "interleaved, ttnn.linear heuristic",
        lambda: ttnn.linear(act_dram, w_inter, dtype=ttnn.bfloat16, compute_kernel_config=ck),
    )

    # 3. explicit 1D multicast over several grids, weight interleaved and DRAM width-sharded
    for grid in ((8, 8), (11, 10), (8, 4), (11, 5)):
        total = grid[0] * grid[1]
        per_core_n = math.ceil(n_tiles / total)
        block_w = _largest_at_most(k_tiles, 8)
        subblock_w = _largest_at_most(per_core_n, 4)
        for weight_label, weight in (("interleaved", w_inter), ("dram-sharded", w_shard)):
            pc1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=grid,
                in0_block_w=block_w,
                out_subblock_h=1,
                out_subblock_w=subblock_w,
                out_block_h=m_tiles,
                out_block_w=per_core_n,
                per_core_M=m_tiles,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            measure(
                f"1D mcast {grid[0]}x{grid[1]} w={weight_label}",
                lambda pc1d=pc1d, weight=weight: ttnn.linear(
                    act_dram,
                    weight,
                    program_config=pc1d,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ck,
                ),
                {"cores": total, "in0_block_w": block_w, "per_core_N": per_core_n},
            )

    # 4. explicit 2D config, weight DRAM width-sharded (the prefill form, at one tile row)
    banks = _DEVICE.dram_grid_size().x
    if n_tiles % banks == 0:
        per_core_n = n_tiles // banks
        pc2d = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(banks, m_tiles),
            in0_block_w=_largest_at_most(k_tiles, 8),
            out_subblock_h=1,
            out_subblock_w=_largest_at_most(per_core_n, 4),
            out_block_h=1,
            out_block_w=per_core_n,
            per_core_M=1,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
        )
        measure(
            f"2D {banks}x{m_tiles} w=dram-sharded",
            lambda: ttnn.linear(
                act_dram,
                w_shard,
                program_config=pc2d,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ck,
            ),
            {"cores": banks * m_tiles, "per_core_N": per_core_n},
        )

    for tensor in (act_dram, w_inter, w_shard):
        if tensor.is_allocated():
            ttnn.deallocate(tensor)
    if act_sharded is not None and act_sharded.is_allocated():
        ttnn.deallocate(act_sharded)


def main() -> int:
    global _DEVICE
    _DEVICE = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for label, m, k, n in SHAPES:
            for dt_name in ("bfp4", "bfp8"):
                run_shape(label, m, k, n, dt_name, results)
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        ttnn.close_mesh_device(_DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
