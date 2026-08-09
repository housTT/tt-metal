# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``O7``: is packing ``wqkv`` with ``wgate`` actually blocked, or just untried?

The first pass rejected it on an L1 argument: packing widens N, which raises ``per_core_N`` for
the DRAM-sharded decode matmul toward the value at which the packed MLP gate/up overflows.  That
argument does not survive its own data - the BFP4 gate runs fine at ``per_core_N = 17`` while the
packed gate/up fails at 34, and a packed ``wqkv|wgate`` is only 14.  So it is measured here
instead, model-free, on the real shapes and dtype:

* separate: ``5120 x 8192`` (QKV) + ``5120 x 6144`` (gate), the shipped path;
* packed:   ``5120 x 14336``, one matmul, plus the two width slices its consumers need.

Both at BFP8/LoFi, both decode (M=32, DRAM-sharded) and prefill (M=2048, 2D), PCC-gated against
the separate path so a faster-but-wrong config cannot win.

Emits ``PACKATTN `` JSON lines.
"""
from __future__ import annotations

import json
import math
import sys
import time

sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal")
sys.path.insert(0, "/home/ttuser/dev/qwen/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes")

import torch  # noqa: E402
import ttnn  # noqa: E402
from matmul_sweep import B16, B8, LOFI, TILE, dram_sharded_weight_cfg, time_op, width_sharded_l1  # noqa: E402

from models.autoports.qwen_qwen3_6_27b.tt import optimized_decoder as O  # noqa: E402

K, N_QKV, N_GATE = 5120, 8192, 6144
ITERS = 20


def emit(**p):
    print("PACKATTN " + json.dumps(p, sort_keys=True, default=str), flush=True)


def pcc(a, b):
    a = a.to(torch.float64).flatten(); a = a - a.mean()
    b = b.to(torch.float64).flatten(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=LOFI, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)
    try:
        torch.manual_seed(0)
        w_qkv = torch.randn(1, 1, K, N_QKV) * 0.02
        w_gate = torch.randn(1, 1, K, N_GATE) * 0.02
        w_packed = torch.cat([w_qkv, w_gate], dim=-1)

        for m in (32, 2048):
            x_host = torch.randn(1, 1, m, K) * 0.02
            x_dram = ttnn.from_torch(x_host, dtype=B16, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
            golden = (x_host.reshape(m, K) @ torch.cat([w_qkv, w_gate], dim=-1).reshape(K, -1))

            if m == 32:
                dec = O.OptimizedDecoder.__new__(O.OptimizedDecoder)
                dec.topology = O.DEFAULT_TOPOLOGY
                for label, widths in (("separate", (N_QKV, N_GATE)), ("packed", (N_QKV + N_GATE,))):
                    total = 0.0
                    outs = []
                    ok = True
                    for n in widths:
                        host = w_packed[..., :n] if label == "packed" else (
                            w_qkv if n == N_QKV else w_gate)
                        weight = ttnn.from_torch(host, dtype=B8, layout=ttnn.TILE_LAYOUT,
                                                 device=device,
                                                 memory_config=dram_sharded_weight_cfg(K, n))
                        plan = O._decode_matmul_plan("w", K, n, 32, 0)
                        act = ttnn.from_torch(x_host, dtype=B16, layout=ttnn.TILE_LAYOUT,
                                              device=device,
                                              memory_config=plan.input_memory_config)
                        row = {"phase": "decode", "family": label, "k": K, "n": n,
                               "cores": plan.cores, "in0_block_w": plan.program_config.in0_block_w,
                               "per_core_N": plan.program_config.per_core_N}
                        try:
                            got = ttnn.linear(act, weight, program_config=plan.program_config,
                                              memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                              dtype=B16, compute_kernel_config=ckc)
                            outs.append(ttnn.to_torch(got).float().reshape(m, -1)[:, :n])
                            ttnn.deallocate(got)
                            row["us"] = time_op(
                                lambda: ttnn.linear(act, weight, program_config=plan.program_config,
                                                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                                    dtype=B16, compute_kernel_config=ckc), device)
                            total += row["us"]
                        except Exception as exc:
                            lines = str(exc).splitlines()
                            row["error"] = lines[2] if len(lines) > 2 else str(exc)
                            ok = False
                        emit(**row)
                        ttnn.deallocate(act)
                        ttnn.deallocate(weight)
                    if ok:
                        merged = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
                        emit(phase="decode", family=label, total_us=round(total, 1),
                             pcc=pcc(golden, merged[:, : golden.shape[-1]]))
            else:
                for label, widths in (("separate", (N_QKV, N_GATE)), ("packed", (N_QKV + N_GATE,))):
                    total = 0.0
                    outs = []
                    ok = True
                    for n in widths:
                        host = w_packed[..., :n] if label == "packed" else (
                            w_qkv if n == N_QKV else w_gate)
                        weight = ttnn.from_torch(host, dtype=B8, layout=ttnn.TILE_LAYOUT,
                                                 device=device,
                                                 memory_config=dram_sharded_weight_cfg(K, n))
                        dec = O.OptimizedDecoder.__new__(O.OptimizedDecoder)
                        dec.topology = O.DEFAULT_TOPOLOGY
                        pcs = dec._prefill_program_configs(m, K, weight)
                        row = {"phase": "prefill", "family": label, "k": K, "n": n}
                        for pc in pcs:
                            try:
                                got = ttnn.linear(x_dram, weight, program_config=pc, dtype=B16,
                                                  compute_kernel_config=ckc)
                                outs.append(ttnn.to_torch(got).float().reshape(m, -1)[:, :n])
                                ttnn.deallocate(got)
                                row.update(in0_block_w=pc.in0_block_w, out_block_h=pc.out_block_h,
                                           per_core_N=pc.per_core_N)
                                row["us"] = time_op(
                                    lambda: ttnn.linear(x_dram, weight, program_config=pc, dtype=B16,
                                                        compute_kernel_config=ckc), device)
                                total += row["us"]
                                break
                            except Exception:
                                continue
                        if "us" not in row:
                            row["error"] = "no legal program config"
                            ok = False
                        emit(**row)
                        ttnn.deallocate(weight)
                    if ok:
                        merged = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
                        emit(phase="prefill", family=label, total_us=round(total, 1),
                             pcc=pcc(golden, merged[:, : golden.shape[-1]]))
            ttnn.deallocate(x_dram)

        # ---- the consumer path, which is what an earlier revision estimated rather than measured.
        #
        # Separate: `qkv` is width-sharded in L1 and has to reach `nlp_create_qkv_heads_decode`
        # interleaved, so it pays one sharded-to-interleaved on 8192 columns; `gate` stays in
        # `o_proj`'s activation shard and pays nothing.
        # Packed: the single 14336-wide output pays the same conversion on 1.75x the columns and
        # then two width slices to separate the halves.
        m = 32
        x_host = torch.randn(1, 1, m, K) * 0.02
        for label, width in (("separate", N_QKV), ("packed", N_QKV + N_GATE)):
            plan = O._decode_matmul_plan("w", K, width, 32, 0)
            sharded = ttnn.from_torch(torch.randn(1, 1, m, width) * 0.02, dtype=B16,
                                      layout=ttnn.TILE_LAYOUT, device=device,
                                      memory_config=width_sharded_l1(m, width, plan.grid))
            row = {"phase": "decode_consumers", "family": label, "width": width}
            row["us_sharded_to_interleaved"] = time_op(
                lambda: ttnn.sharded_to_interleaved(sharded, ttnn.L1_MEMORY_CONFIG), device)
            if label == "packed":
                inter = ttnn.sharded_to_interleaved(sharded, ttnn.L1_MEMORY_CONFIG)
                row["us_slice_qkv"] = time_op(
                    lambda: ttnn.slice(inter, [0, 0, 0, 0], [1, 1, m, N_QKV]), device)
                row["us_slice_gate"] = time_op(
                    lambda: ttnn.slice(inter, [0, 0, 0, N_QKV], [1, 1, m, width]), device)
                ttnn.deallocate(inter)
            row["us_total"] = round(sum(v for k, v in row.items() if k.startswith("us_")), 2)
            emit(**row)
            ttnn.deallocate(sharded)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
