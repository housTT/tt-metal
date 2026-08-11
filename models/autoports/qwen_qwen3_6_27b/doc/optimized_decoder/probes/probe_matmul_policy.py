# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Model-free sweep of the decoder's dominant matmul roles: dtype x fidelity x geometry.

Every role below is a real ``(M, K, N)`` from the committed fused-stage reports, at the real
decode row count (32 tile-padded rows) and the real 2048-token prefill row count.  For each
role the sweep measures:

* the *shipped* fused-stage form - ``ttnn.linear`` with no program config, bfloat16 weights,
  HiFi4 - as the baseline;
* bfloat8_b and bfloat4_b weights at LoFi / HiFi2 / HiFi4;
* for decode, the DRAM-sharded program config over several activation-shard core counts, which
  is what sets the legal ``in0_block_w`` (OPT-004 / OPT-014: precision crossed with geometry).

It reports median and stdev over repeated launches and the PCC of every candidate against a
float32 torch reference of the same random inputs, so a candidate that is fast because it is
wrong cannot look good here.  Real-weight evidence is the layer suite's job; this probe exists
to find the legal/fast envelope and to catch op-contract blockers cheaply.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_matmul_policy.py
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
REPEATS = 20

# (role, M, K, N, out_dtype) - M is the tile-padded row count of the measured pass.
DECODE_ROLES = [
    ("mlp_gate_up", 32, 5120, 34816, ttnn.bfloat16),
    ("mlp_down", 32, 17408, 5120, ttnn.bfloat16),
    ("wqkv", 32, 5120, 8192, ttnn.bfloat16),
    ("wgate", 32, 5120, 6144, ttnn.bfloat16),
    ("o_proj", 32, 6144, 5120, ttnn.bfloat16),
    ("in_proj_qkv", 32, 5120, 10240, ttnn.float32),
    ("in_proj_z", 32, 5120, 6144, ttnn.bfloat16),
]

PREFILL_ROLES = [
    ("mlp_gate_up", 2048, 5120, 34816, ttnn.bfloat16),
    ("mlp_down", 2048, 17408, 5120, ttnn.bfloat16),
    ("wqkv", 2048, 5120, 8192, ttnn.bfloat16),
    ("o_proj", 2048, 6144, 5120, ttnn.bfloat16),
    ("in_proj_qkv", 2048, 5120, 10240, ttnn.float32),
]

FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}

DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def kernel_cfg(fidelity: str, fp32_acc: bool):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=FIDELITIES[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=True,
    )


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return float((a @ b) / (a.norm() * b.norm()))


def largest_divisor(value: int, limit: int = 64) -> int:
    for candidate in range(min(value, limit), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def dram_sharded_weight_cfg(device, k: int, n: int) -> ttnn.MemoryConfig:
    dram = device.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    banks = dram.x
    padded = math.ceil(n / (TILE * banks)) * TILE * banks
    spec = ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def width_sharded_l1(device, rows: int, width: int, cores: int) -> ttnn.MemoryConfig:
    grid = device.compute_with_storage_grid_size()
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // cores),
        core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def time_op(fn, repeats: int = REPEATS):
    fn()
    ttnn.synchronize_device(_DEVICE)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(_DEVICE)
        samples.append((time.perf_counter() - start) * 1e6)
        ttnn.deallocate(out)
    return statistics.median(samples), statistics.stdev(samples) if len(samples) > 1 else 0.0


_DEVICE = None


def run_role(role, m, k, n, out_dtype, *, decode: bool, results: list):
    device = _DEVICE
    torch.manual_seed(0)
    act_t = torch.randn(1, 1, m, k, dtype=torch.float32) * 0.05
    w_t = torch.randn(1, 1, k, n, dtype=torch.float32) * 0.02
    golden = (act_t.to(torch.float64) @ w_t.to(torch.float64)).to(torch.float32)

    act = ttnn.from_torch(
        act_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    def record(candidate, median, stdev, value, cores=None, ibw=None, per_core_n=None, error=None):
        results.append(
            {
                "phase": "decode" if decode else "prefill",
                "role": role,
                "m": m,
                "k": k,
                "n": n,
                "candidate": candidate,
                "cores": cores,
                "in0_block_w": ibw,
                "per_core_N": per_core_n,
                "median_us": median,
                "stdev_us": stdev,
                "pcc": value,
                "error": error,
            }
        )
        print(
            f"  {role:14s} {'dec' if decode else 'pre':3s} {candidate:34s} "
            f"cores={cores if cores else '-':>4} ibw={ibw if ibw else '-':>3} "
            f"{median if median is not None else float('nan'):9.1f} us "
            f"+-{stdev if stdev is not None else 0:6.1f}  pcc={value if value is not None else float('nan'):.6f}"
            + (f"  ERROR {error}" if error else ""),
            flush=True,
        )

    for dt_name, dtype in DTYPES.items():
        weight = ttnn.from_torch(
            w_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for fid in ("LoFi", "HiFi2", "HiFi4"):
            cfg = kernel_cfg(fid, fp32_acc=out_dtype == ttnn.float32)
            try:
                median, stdev = time_op(lambda: ttnn.linear(act, weight, dtype=out_dtype, compute_kernel_config=cfg))
                out = ttnn.linear(act, weight, dtype=out_dtype, compute_kernel_config=cfg)
                value = pcc(golden, ttnn.to_torch(out).to(torch.float32))
                ttnn.deallocate(out)
                record(f"interleaved {dt_name}/{fid}", median, stdev, value)
            except Exception as exc:  # noqa: BLE001 - op-contract blockers are the point
                record(f"interleaved {dt_name}/{fid}", None, None, None, error=str(exc)[:160])
        ttnn.deallocate(weight)

    if not decode:
        ttnn.deallocate(act)
        return

    # DRAM-sharded decode geometry x precision.
    k_tiles = k // TILE
    core_candidates = [c for c in (8, 10, 16, 20, 32, 40, 64) if k_tiles % c == 0 and c <= 110]
    for dt_name in ("bf16", "bfp8", "bfp4"):
        dtype = DTYPES[dt_name]
        w_cfg = dram_sharded_weight_cfg(device, k, n)
        try:
            weight = ttnn.from_torch(w_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_cfg)
        except Exception as exc:  # noqa: BLE001
            record(f"dram-sharded {dt_name}", None, None, None, error=f"weight: {str(exc)[:140]}")
            continue
        for cores in core_candidates:
            in_cfg = width_sharded_l1(device, m, k, cores)
            out_cfg = width_sharded_l1(device, m, n, cores) if n % cores == 0 else None
            if out_cfg is None:
                continue
            ibw_max = k_tiles // cores
            for ibw in sorted(
                {1, 2, 4, largest_divisor(ibw_max, 32), ibw_max}
                & set(d for d in range(1, ibw_max + 1) if ibw_max % d == 0)
            ):
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=ibw,
                    per_core_M=m // TILE,
                    per_core_N=(n // TILE) // cores,
                    fused_activation=None,
                )
                fid = "LoFi" if dt_name != "bf16" else "HiFi4"
                cfg = kernel_cfg(fid, fp32_acc=out_dtype == ttnn.float32)
                try:
                    sharded = ttnn.to_memory_config(act, in_cfg)

                    def call():
                        return ttnn.linear(
                            sharded,
                            weight,
                            program_config=pc,
                            memory_config=out_cfg,
                            dtype=out_dtype,
                            compute_kernel_config=cfg,
                        )

                    median, stdev = time_op(call)
                    out = call()
                    value = pcc(golden, ttnn.to_torch(out).to(torch.float32))
                    ttnn.deallocate(out)
                    ttnn.deallocate(sharded)
                    record(f"dram-sharded {dt_name}/{fid}", median, stdev, value, cores, ibw, (n // TILE) // cores)
                except Exception as exc:  # noqa: BLE001
                    record(
                        f"dram-sharded {dt_name}/{fid}",
                        None,
                        None,
                        None,
                        cores,
                        ibw,
                        (n // TILE) // cores,
                        error=str(exc)[:160],
                    )
        ttnn.deallocate(weight)
    ttnn.deallocate(act)


def main() -> int:
    global _DEVICE
    _DEVICE = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    print(f"compute grid: {_DEVICE.compute_with_storage_grid_size()}  dram grid: {_DEVICE.dram_grid_size()}")
    results: list = []
    try:
        print("== decode roles (32 tile-padded rows)")
        for role, m, k, n, out_dtype in DECODE_ROLES:
            run_role(role, m, k, n, out_dtype, decode=True, results=results)
        print("== prefill roles (2048 rows)")
        for role, m, k, n, out_dtype in PREFILL_ROLES:
            run_role(role, m, k, n, out_dtype, decode=False, results=results)
    finally:
        print("PROBE_JSON " + json.dumps(results))
        ttnn.close_mesh_device(_DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
