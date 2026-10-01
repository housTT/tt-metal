import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

WEIGHT_DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
SHAPES = {
    "mlp_gate_up": (4096, 12288, None),
    "mlp_down": (12288, 4096, ttnn.bfloat8_b),
    "gdn_in_proj": (4096, 12352, ttnn.bfloat8_b),
    "attn_qkv": (4096, 8192, ttnn.bfloat8_b),
    "o_proj": (4096, 4096, ttnn.bfloat8_b),
}
TILE = 32


def pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-8)).item()


def time_op(dev, fn, reps):
    out = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.deallocate(out)
        t = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) * 1000)
    return out, statistics.median(ts)


def ckc(fidelity, fp32_acc, l1_acc=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=l1_acc
    )


def grid_configs(M, K, N, grid, fp32_acc):
    gx, gy = grid
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    cfgs = []
    for cols in sorted({gx, 8, 10, 11, 12} & set(range(1, gx + 1)), reverse=True):
        for rows in sorted({gy, 8, 10} & set(range(1, gy + 1)), reverse=True):
            per_core_M = math.ceil(Mt / rows)
            per_core_N = math.ceil(Nt / cols)
            for in0 in (1, 2, 4, 8, 16):
                if Kt % in0:
                    continue
                for h, w in ((1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 2), (1, 8), (4, 1), (2, 1)):
                    if h * w > (4 if fp32_acc else 8) or per_core_M % h or per_core_N % w:
                        continue
                    cfgs.append(((cols, rows), in0, h, w, per_core_M, per_core_N))
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--ms", default="128,512,2048")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-configs", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--gate-up-dtype", choices=sorted(WEIGHT_DTYPES), default="bfp8")
    a = ap.parse_args()
    SHAPES["mlp_gate_up"] = SHAPES["mlp_gate_up"][:2] + (WEIGHT_DTYPES[a.gate_up_dtype],)
    dev = ttnn.open_device(device_id=a.device_id, l1_small_size=24576, num_command_queues=2, trace_region_size=0)
    dev.enable_program_cache()
    grid = dev.compute_with_storage_grid_size()
    grid = (grid.x, grid.y)
    logger.info(f"compute grid {grid}")
    tuning = tpc.prefill_tuning(1)
    results = []
    torch.manual_seed(0)
    for name in a.shapes.split(","):
        K, N, wdtype = SHAPES[name]
        w_t = torch.randn(K, N) * 0.02
        w = ttnn.from_torch(
            w_t, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        w_ref = ttnn.to_torch(w).float()
        for M in [int(x) for x in a.ms.split(",")]:
            x_t = torch.randn(1, M, K)
            x = ttnn.from_torch(
                x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            ref = ttnn.to_torch(x).float()[0] @ w_ref
            rows = []

            def record(label, fn, extra=None):
                try:
                    out, ms = time_op(dev, fn, a.reps)
                except Exception as e:
                    rows.append({"config": label, "error": str(e).splitlines()[0][:200]})
                    logger.warning(f"{name} M={M} {label}: {str(e).splitlines()[0][:160]}")
                    return
                p = pcc(ttnn.to_torch(out).float()[0], ref)
                ttnn.deallocate(out)
                flops = 2 * M * K * N / (ms / 1000) / 1e12
                rows.append(
                    {
                        "config": label,
                        "ms": round(ms, 3),
                        "tflops": round(flops, 1),
                        "pcc": round(p, 6),
                        **(extra or {}),
                    }
                )
                logger.info(f"{name} M={M} {label}: {ms:.3f} ms {flops:.1f} TFLOPs pcc {p:.5f}")

            for fid, fname in ((ttnn.MathFidelity.LoFi, "LoFi"), (ttnn.MathFidelity.HiFi2, "HiFi2")):
                for fp32 in (True, False):
                    c = ckc(fid, fp32)
                    record(
                        f"default {fname} fp32acc={fp32}",
                        lambda c=c: ttnn.linear(x, w, compute_kernel_config=c, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                    )
                    pc = tpc.create_prefill_matmul_program_config(M, K, N, grid_size=grid, tuning=tuning)
                    record(
                        f"prefill_progcfg {fname} fp32acc={fp32} in0={pc.in0_block_w} sub={pc.out_subblock_h}x{pc.out_subblock_w} pcM={pc.per_core_M} pcN={pc.per_core_N}",
                        lambda c=c, pc=pc: ttnn.linear(
                            x, w, compute_kernel_config=c, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc
                        ),
                    )
                    if M <= 512:
                        record(
                            f"default {fname} fp32acc={fp32} L1 out",
                            lambda c=c: ttnn.linear(x, w, compute_kernel_config=c, memory_config=ttnn.L1_MEMORY_CONFIG),
                        )
            c = ckc(ttnn.MathFidelity.LoFi, True)
            cfgs = grid_configs(M, K, N, grid, True)
            cfgs = sorted(cfgs, key=lambda t: (-t[1], -(t[2] * t[3])))[: a.max_configs]
            for (cols, rws), in0, h, w_, pcm, pcn in cfgs:
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(cols, rws),
                    in0_block_w=in0,
                    out_subblock_h=h,
                    out_subblock_w=w_,
                    per_core_M=pcm,
                    per_core_N=pcn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )
                record(
                    f"2d grid={cols}x{rws} in0={in0} sub={h}x{w_} pcM={pcm} pcN={pcn} LoFi fp32acc",
                    lambda pc=pc: ttnn.linear(
                        x, w, compute_kernel_config=c, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=pc
                    ),
                )
            try:
                record(
                    "minimal_matmul LoFi fp32acc",
                    lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=c),
                )
            except Exception as e:
                rows.append({"config": "minimal_matmul", "error": str(e)[:200]})
            ttnn.deallocate(x)
            ok = [r for r in rows if "ms" in r]
            best = min(ok, key=lambda r: r["ms"]) if ok else None
            results.append(
                {"shape": name, "M": M, "K": K, "N": N, "weight_dtype": str(wdtype), "rows": rows, "best": best}
            )
            logger.info(f"BEST {name} M={M}: {best}")
            Path(a.out).write_text(json.dumps({"grid": grid, "results": results}, indent=1) + "\n")
        ttnn.deallocate(w)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
