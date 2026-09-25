# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Every remaining piece of ``tt-perf-report`` advice on the ``linear_attention`` recurrence rows.

The committed optimized reports mark every weight projection ``✅ Optimized`` except the two BFP4
gate/up rows (chased in ``probe_bfp4_gateup.py``).  What is left is a cluster of advice on the three
float32 gated-delta-rule recurrence matmuls and on ``in_proj_ab``, all of which stage 2 owns and this
stage inherited:

* ``HiFi2 is sufficient for BFP8 multiplication and has 2x the throughput of HiFi4``;
* ``in0_block_w=1 is small, try in0_block_w=2 or above``;
* ``Output subblock 1x1 is small, try out_subblock_h * out_subblock_w >= 2 if possible``;
* ``If possible place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)``;
* ``Try a DRAM-sharded program config`` (on the batch-32 rows).

The skill's advice policy is to try each one and record before/after, so this probe does exactly
that on the real recurrence shape - a batched ``[batch * num_v_heads, 32, 128] x [.., 128, 128]``
float32 matmul - at both decode regimes: the shipped explicit ``core_grid``, the same at HiFi2 and
LoFi, explicit block program configs with a larger ``in0_block_w`` and output subblock, and the
activation placed in L1.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_recurrence_advice.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import torch

import ttnn

TILE = 32
REPEATS = 25
#: ``num_v_heads`` and the per-head state dims of this model's gated delta net.
NV, DK, DV = 48, 128, 128
_DEVICE = None


def _median_stdev(samples):
    return statistics.median(samples), (statistics.stdev(samples) if len(samples) > 1 else 0.0)


def kernel_cfg(fidelity, fp32_acc=True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
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


def run(batch, results):
    """The state read: ``[1, batch*NV, 1(->32), DK] x [1, batch*NV, DK, DV]`` in float32."""
    heads = batch * NV
    torch.manual_seed(0)
    k_t = torch.randn(1, heads, 1, DK) * 0.05
    state_t = torch.randn(1, heads, DK, DV) * 0.05
    golden = (k_t.to(torch.float64) @ state_t.to(torch.float64)).float()

    def upload(tensor, memory_config):
        return ttnn.from_torch(
            tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=memory_config
        )

    k_dram = upload(k_t, ttnn.DRAM_MEMORY_CONFIG)
    k_l1 = upload(k_t, ttnn.L1_MEMORY_CONFIG)
    state_dram = upload(state_t, ttnn.DRAM_MEMORY_CONFIG)

    def record(candidate, median, stdev, value, extra=None, error=None):
        row = {
            "sweep": "recurrence_advice",
            "batch": batch,
            "head_problems": heads,
            "candidate": candidate,
            "median_us": median,
            "stdev_us": stdev,
            "pcc": value,
            "error": error,
        }
        row.update(extra or {})
        results.append(row)
        print(
            f"  batch={batch:<3d} heads={heads:<5d} {candidate:52s} "
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

    # The shipped form: explicit core_grid, HiFi4, float32 destination accumulation, in0 in DRAM.
    grid = ttnn.CoreGrid(y=10, x=4) if heads >= 1536 else ttnn.CoreGrid(y=6, x=4)
    for label, fidelity in (
        ("HiFi4 (shipped)", ttnn.MathFidelity.HiFi4),
        ("HiFi2 (report advice)", ttnn.MathFidelity.HiFi2),
        ("LoFi", ttnn.MathFidelity.LoFi),
    ):
        cfg = kernel_cfg(fidelity)
        measure(
            f"core_grid {grid.y}x{grid.x}, {label}, in0 DRAM",
            lambda cfg=cfg: ttnn.matmul(
                k_dram, state_dram, dtype=ttnn.float32, compute_kernel_config=cfg, core_grid=grid
            ),
            {"fidelity": label, "core_grid": [grid.y, grid.x], "in0_memory": "dram"},
        )
    # "If possible place input 0 in L1".
    measure(
        f"core_grid {grid.y}x{grid.x}, HiFi4, in0 L1 (report advice)",
        lambda: ttnn.matmul(
            k_l1,
            state_dram,
            dtype=ttnn.float32,
            compute_kernel_config=kernel_cfg(ttnn.MathFidelity.HiFi4),
            core_grid=grid,
        ),
        {"fidelity": "HiFi4", "core_grid": [grid.y, grid.x], "in0_memory": "l1"},
    )
    # "in0_block_w=1 is small" / "Output subblock 1x1 is small": an explicit block program config.
    # ``MatmulMultiCoreReuseProgramConfig`` is the batched-reuse form; K is 4 tiles and N is 4 tiles,
    # so in0_block_w can be up to 4 and the output subblock up to the whole 1x4 output block.
    for block_w in (1, 2, 4):
        for subblock_w in (1, 2, 4):
            if subblock_w > 4 // 1:
                continue
            pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=block_w,
                out_subblock_h=1,
                out_subblock_w=subblock_w,
                per_core_M=1,
                per_core_N=DV // TILE,
            )
            measure(
                f"MatmulMultiCoreReuse in0_block_w={block_w} subblock 1x{subblock_w}",
                lambda pc=pc: ttnn.matmul(
                    k_dram,
                    state_dram,
                    dtype=ttnn.float32,
                    compute_kernel_config=kernel_cfg(ttnn.MathFidelity.HiFi4),
                    program_config=pc,
                ),
                {"in0_block_w": block_w, "out_subblock_w": subblock_w},
            )
    # "Try a DRAM-sharded program config" - the report's advice on the batch-32 rows.  The "weight" of
    # this matmul is the carried recurrent state, so this needs it width-sharded across the DRAM banks
    # and the activation width-sharded in L1.  Measured rather than rejected by argument.
    dram = _DEVICE.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
    row = {
        "sweep": "recurrence_advice",
        "batch": batch,
        "head_problems": heads,
        "candidate": "batched DRAM-sharded program config (report advice)",
    }
    try:
        padded = ((DV + TILE * dram.x - 1) // (TILE * dram.x)) * TILE * dram.x
        state_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (heads * DK, padded // dram.x), ttnn.ShardOrientation.ROW_MAJOR),
        )
        state_sharded = ttnn.from_torch(
            state_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=_DEVICE, memory_config=state_cfg
        )
        # The op requires a width-sharded **L1** activation as well as the DRAM width-sharded "weight".
        # The first attempt handed it the interleaved copy and got ``input_tensor_a.is_sharded()``, which
        # is an API requirement rather than a verdict on the candidate, so this is the same candidate
        # given the layout the op asks for.  ``K`` is 4 tiles here, so the activation spreads over at
        # most 4 cores.
        act_cores = min(4, DK // TILE)
        act_cfg = ttnn.create_sharded_memory_config(
            shape=(heads * TILE, DK // act_cores),
            core_grid=ttnn.num_cores_to_corerangeset(
                act_cores, _DEVICE.compute_with_storage_grid_size(), row_wise=True
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_sharded = ttnn.to_memory_config(k_dram, act_cfg)
        # ...and a width-sharded L1 *output* config, which is the third requirement this op states one at
        # a time: ``input_tensor_a.is_sharded()``, then ``output_mem_config.is_sharded()``.  Each is an
        # API precondition rather than a verdict on the candidate, so each gets adapted rather than
        # treated as the answer.
        out_cfg = ttnn.create_sharded_memory_config(
            shape=(heads * TILE, DV // act_cores),
            core_grid=ttnn.num_cores_to_corerangeset(
                act_cores, _DEVICE.compute_with_storage_grid_size(), row_wise=True
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
            in0_block_w=DK // TILE // act_cores, per_core_M=1, per_core_N=DV // TILE, fused_activation=None
        )

        def run():
            return ttnn.matmul(
                k_sharded,
                state_sharded,
                dtype=ttnn.float32,
                compute_kernel_config=kernel_cfg(ttnn.MathFidelity.HiFi4),
                program_config=pc,
                memory_config=out_cfg,
            )

        out = run()
        row["pcc"] = pcc(golden, ttnn.to_torch(out).float())
        row["median_us"], row["stdev_us"] = time_call(run)
        ttnn.deallocate(out)
        ttnn.deallocate(state_sharded)
        if k_sharded.is_allocated():
            ttnn.deallocate(k_sharded)
    except Exception as exc:  # noqa: BLE001 - a blocker is the result
        row["error"] = str(exc)[:220]
        row["median_us"] = None
        row["stdev_us"] = None
        row["pcc"] = None
    results.append(row)
    print(
        f"  batch={batch:<3d} heads={heads:<5d} {row['candidate']:52s} "
        + ("blocked" if row.get("error") else f"pcc={row['pcc']:.6f}")
        + (f"  ERROR {row['error']}" if row.get("error") else ""),
        flush=True,
    )

    for tensor in (k_dram, k_l1, state_dram):
        if tensor.is_allocated():
            ttnn.deallocate(tensor)


def main() -> int:
    global _DEVICE
    _DEVICE = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        print(f"recurrent state bytes per batch slot: {NV * DK * DV * 4 / 1e6:.1f} MB")
        for batch in (1, 32):
            print(f"  state at batch {batch}: {batch * NV * DK * DV * 4 / 1e6:.1f} MB")
            run(batch, results)
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        ttnn.close_mesh_device(_DEVICE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
