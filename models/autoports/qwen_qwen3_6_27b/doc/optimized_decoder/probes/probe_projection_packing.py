# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Packed versus separate shared-input projections, at **this stage's** dtypes and layout.

OPT-001 asks for a packed candidate whenever several projections consume the same activation, and
this decoder has two such pairs that stage 2 kept separate:

* ``wqkv`` (``5120 x 8192``) + ``wgate`` (``5120 x 6144``) - ``full_attention``;
* ``in_proj_qkv`` (``5120 x 10240``) + ``in_proj_z`` (``5120 x 6144``) - ``linear_attention``.

Stage 2 measured both and rejected packing, but it measured them with **bfloat16** weights, a
DRAM-interleaved layout and ``ttnn.linear``'s own heuristic on both sides.  This stage changed all
three, so the comparison has to be redone rather than inherited - the packed form's cost is two
slices of the merged output, and the separate form's cost is one extra matmul launch plus a second
read of the activation, and reducing the weight bytes moves those two terms by different factors.

The comparison is made on the *pair*, at the real shapes, at the shipped weight dtypes, on the
shipped decode layout (width-sharded L1 activation, DRAM width-sharded weights, the shipped core
count) and at the 2048-row prefill layout, and it charges the packed arm for the slices it needs.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_projection_packing.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    DEFAULT_GEOMETRY,
    DEFAULT_POLICY,
    _largest_divisor_at_most,
)

TILE = 32
REPEATS = 25
#: ``(label, K, N_a, N_b, dtype_a, dtype_b, out_dtype)`` - the two pairs, with the shipped dtypes.
PAIRS = [
    ("full_attention wqkv + wgate", 5120, 8192, 6144, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat16),
    ("linear_attention in_proj_qkv + in_proj_z", 5120, 10240, 6144, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.float32),
]
_DEVICE = None


def _median_stdev(samples):
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def dram_sharded_weight_cfg(k: int, n: int) -> ttnn.MemoryConfig:
    dram = _DEVICE.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    padded = math.ceil(n / (TILE * dram.x)) * TILE * dram.x
    spec = ttnn.ShardSpec(grid, (k, padded // dram.x), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def width_sharded_l1(rows: int, width: int, cores: int) -> ttnn.MemoryConfig:
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // cores),
        core_grid=ttnn.num_cores_to_corerangeset(cores, _DEVICE.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def decode_pc(k: int, n: int, rows: int, cores: int, weight_bytes: int, out_bytes: int):
    k_tiles = k // TILE
    per_core_m, per_core_n = rows // TILE, (n // TILE) // cores
    best = 1
    for candidate in [d for d in range(1, k_tiles // cores + 1) if (k_tiles // cores) % d == 0]:
        l1 = 2 * candidate * per_core_m * 2048 + 2 * candidate * per_core_n * weight_bytes
        l1 += per_core_m * per_core_n * out_bytes
        if l1 <= 350_000:
            best = candidate
    return (
        ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=best, per_core_M=per_core_m, per_core_N=per_core_n, fused_activation=None
        ),
        best,
    )


def prefill_pc(k: int, n: int, rows: int, fp32: bool):
    banks = _DEVICE.dram_grid_size().x
    k_tiles, n_tiles = k // TILE, n // TILE
    m_tiles = max(1, math.ceil(rows / TILE))
    x = banks
    y = _largest_divisor_at_most(m_tiles, _DEVICE.compute_with_storage_grid_size().y)
    per_core_m, per_core_n = math.ceil(m_tiles / y), math.ceil(n_tiles / x)
    out_block_w, out_block_h = per_core_n, 1
    for candidate in [d for d in range(1, per_core_m + 1) if per_core_m % d == 0]:
        if candidate * out_block_w <= 160:
            out_block_h = candidate
    budget = 2 if fp32 else 4
    sw = _largest_divisor_at_most(out_block_w, budget)
    sh = _largest_divisor_at_most(out_block_h, max(1, budget // sw))
    # L1-aware ``in0_block_w``, the same bound the model uses: a float32 output block plus a BFP8
    # weight block puts 8 over L1 for the 10240-wide ``in_proj_qkv``, and a flat cap of 8 here made
    # that pair unmeasurable rather than slow.
    weight_bytes = 1088
    out_bytes = 4096 if fp32 else 2048
    block_w = 1
    for candidate in [d for d in range(1, min(k_tiles, 8) + 1) if k_tiles % d == 0]:
        l1 = 2 * candidate * out_block_h * 2048 + 2 * candidate * out_block_w * weight_bytes
        l1 += out_block_h * out_block_w * out_bytes
        if l1 <= 1_100_000:
            block_w = candidate
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(x, y),
        in0_block_w=block_w,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


def time_call(fn):
    outs = fn()
    ttnn.synchronize_device(_DEVICE)
    for tensor in outs:
        ttnn.deallocate(tensor)
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(_DEVICE)
        samples.append((time.perf_counter() - start) * 1e6)
        for tensor in outs:
            ttnn.deallocate(tensor)
    return _median_stdev(samples)


def run_pair(label, k, n_a, n_b, dt_a, dt_b, out_dtype, rows, decode: bool, results: list):
    cores = DEFAULT_GEOMETRY.cores
    torch.manual_seed(0)
    act_t = torch.randn(1, 1, rows, k) * 0.05
    wa_t = torch.randn(1, 1, k, n_a) * 0.02
    wb_t = torch.randn(1, 1, k, n_b) * 0.02
    packed_t = torch.cat([wa_t, wb_t], dim=-1)
    n_packed = n_a + n_b
    fp32 = out_dtype == ttnn.float32
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2 if fp32 else ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )
    weight_bytes = {ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576, ttnn.bfloat16: 2048}[dt_a]
    out_bytes = 4096 if fp32 else 2048

    def upload(tensor, dtype, cfg):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=cfg)

    def record(candidate, median, stdev, extra=None, error=None):
        row = {
            "sweep": "projection_packing",
            "pair": label,
            "phase": "decode" if decode else "prefill",
            "rows": rows,
            "candidate": candidate,
            "median_us": median,
            "stdev_us": stdev,
            "error": error,
        }
        row.update(extra or {})
        results.append(row)
        print(
            f"  {label:42s} {'dec' if decode else 'pre':3s} {candidate:34s} "
            f"{median if median is not None else float('nan'):9.1f} us +-{stdev if stdev else 0:6.1f}"
            + (f"  {extra}" if extra else "")
            + (f"  ERROR {error}" if error else ""),
            flush=True,
        )

    if decode:
        act = upload(act_t, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
        sharded_in = ttnn.to_memory_config(act, width_sharded_l1(rows, k, cores))
        wa = upload(wa_t, dt_a, dram_sharded_weight_cfg(k, n_a))
        wb = upload(wb_t, dt_b, dram_sharded_weight_cfg(k, n_b))
        wp = upload(packed_t, dt_a, dram_sharded_weight_cfg(k, n_packed))
        pca, ibw_a = decode_pc(k, n_a, rows, cores, weight_bytes, out_bytes)
        pcb, ibw_b = decode_pc(k, n_b, rows, cores, weight_bytes, out_bytes)
        pcp, ibw_p = decode_pc(k, n_packed, rows, cores, weight_bytes, out_bytes)
        out_a, out_b = width_sharded_l1(rows, n_a, cores), width_sharded_l1(rows, n_b, cores)
        out_p = width_sharded_l1(rows, n_packed, cores)

        def separate():
            a = ttnn.linear(
                sharded_in, wa, program_config=pca, memory_config=out_a, dtype=out_dtype, compute_kernel_config=ck
            )
            b = ttnn.linear(
                sharded_in, wb, program_config=pcb, memory_config=out_b, dtype=out_dtype, compute_kernel_config=ck
            )
            return [a, b]

        def packed():
            merged = ttnn.linear(
                sharded_in, wp, program_config=pcp, memory_config=out_p, dtype=out_dtype, compute_kernel_config=ck
            )
            a = ttnn.slice(merged, [0, 0, 0, 0], [1, 1, rows, n_a])
            b = ttnn.slice(merged, [0, 0, 0, n_a], [1, 1, rows, n_packed])
            ttnn.deallocate(merged)
            return [a, b]

        for candidate, fn, extra in (
            ("separate (shipped)", separate, {"in0_block_w": [ibw_a, ibw_b]}),
            ("packed + 2 slices", packed, {"in0_block_w": ibw_p, "packed_N": n_packed}),
        ):
            try:
                median, stdev = time_call(fn)
                record(candidate, median, stdev, extra)
            except Exception as exc:  # noqa: BLE001 - a blocker is a result
                record(candidate, None, None, extra, str(exc)[:200])
        for tensor in (act, sharded_in, wa, wb, wp):
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        return

    act = upload(act_t, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
    wa = upload(wa_t, dt_a, dram_sharded_weight_cfg(k, n_a))
    wb = upload(wb_t, dt_b, dram_sharded_weight_cfg(k, n_b))
    wp = upload(packed_t, dt_a, dram_sharded_weight_cfg(k, n_packed))
    pca, pcb = prefill_pc(k, n_a, rows, fp32), prefill_pc(k, n_b, rows, fp32)
    pcp = prefill_pc(k, n_packed, rows, fp32)

    def separate_prefill():
        a = ttnn.linear(
            act,
            wa,
            program_config=pca,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=out_dtype,
            compute_kernel_config=ck,
        )
        b = ttnn.linear(
            act,
            wb,
            program_config=pcb,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=out_dtype,
            compute_kernel_config=ck,
        )
        return [a, b]

    def packed_prefill():
        merged = ttnn.linear(
            act,
            wp,
            program_config=pcp,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=out_dtype,
            compute_kernel_config=ck,
        )
        a = ttnn.slice(merged, [0, 0, 0, 0], [1, 1, rows, n_a])
        b = ttnn.slice(merged, [0, 0, 0, n_a], [1, 1, rows, n_packed])
        ttnn.deallocate(merged)
        return [a, b]

    for candidate, fn in (("separate (shipped)", separate_prefill), ("packed + 2 slices", packed_prefill)):
        try:
            median, stdev = time_call(fn)
            record(candidate, median, stdev)
        except Exception as exc:  # noqa: BLE001
            record(candidate, None, None, None, str(exc)[:200])
    for tensor in (act, wa, wb, wp):
        if tensor.is_allocated():
            ttnn.deallocate(tensor)


def main() -> int:
    global _DEVICE
    _DEVICE = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        print(f"shipped decode cores: {DEFAULT_GEOMETRY.cores}; policy: {DEFAULT_POLICY.name}")
        for pair in PAIRS:
            run_pair(*pair, rows=32, decode=True, results=results)
        for pair in PAIRS:
            run_pair(*pair, rows=2048, decode=False, results=results)
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        ttnn.close_mesh_device(_DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
