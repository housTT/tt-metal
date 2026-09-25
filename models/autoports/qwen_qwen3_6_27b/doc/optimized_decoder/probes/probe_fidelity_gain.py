# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does math fidelity apply a systematic *gain* to a matmul, and does it grow with mantissa width?

Model-free, one matmul, no weights and no attention.  This exists because the full-context real-weight
runs kept producing a result that reads backwards: every arm that *raises* operand precision makes the
output magnitude worse, monotonically.  At 262143 tokens on real weights the `full_attention` prefill
tail comes back scaled by 0.969 with the shipped BFP4/BFP8 weights, 0.966 with bfloat16 attention
weights, and 0.927 with bfloat16 MLP weights - while the tail *PCC* goes the other way and improves.

The hypothesis that predicts all three: LoFi and HiFi2 feed the FPU a truncated operand mantissa, and
truncation rounds magnitudes **toward zero**, so it is a systematic gain loss rather than symmetric
noise - and the more mantissa bits the operand actually has, the more of them get truncated away.  A
BFP4 weight has almost nothing to lose; a bfloat16 weight has eight bits to lose.

So this measures, for one fixed pair of inputs, `<golden, device> / <golden, golden>` - the best-fit
gain - across the fidelity settings and the weight dtypes, with everything else held constant.  If the
hypothesis holds, LoFi/HiFi2 gain is below 1.0 and falls as the weight dtype widens, and HiFi4 is ~1.0
for every dtype.  PCC is reported beside it to show the two are independent: a configuration can be more
correlated and less correctly scaled at the same time.

The reduction depth is swept too, because a per-element truncation bias accumulates over K: a 5120-deep
reduction should show more of it than a 512-deep one.

    python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/probes/probe_fidelity_gain.py
"""

from __future__ import annotations

import json
import sys

import torch

import ttnn

M = 2048
DEPTHS = (512, 5120, 17408)
N = 1024
DTYPES = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
FIDELITIES = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def gain(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    return float((a @ b) / (a @ a))


def pcc(golden: torch.Tensor, actual: torch.Tensor) -> float:
    a = golden.to(torch.float64).flatten()
    b = actual.to(torch.float64).flatten()
    a, b = a - a.mean(), b - b.mean()
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    results: list = []
    try:
        for depth in DEPTHS:
            torch.manual_seed(0)
            # bfloat16 on the host first, so the golden and the device see bit-identical activations and
            # the only difference left is the kernel's arithmetic.
            act = (torch.randn(1, 1, M, depth) * 0.05).to(torch.bfloat16)
            weight = (torch.randn(1, 1, depth, N) * 0.02).to(torch.bfloat16)
            golden = (act.to(torch.float64) @ weight.to(torch.float64)).float()
            tt_act = ttnn.from_torch(
                act, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for dtype_name, dtype in DTYPES.items():
                tt_weight = ttnn.from_torch(
                    weight, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                for fid_name, fidelity in FIDELITIES.items():
                    for fp32_acc in (False, True):
                        row = {
                            "sweep": "fidelity_gain",
                            "K": depth,
                            "weight_dtype": dtype_name,
                            "fidelity": fid_name,
                            "fp32_dest_acc": fp32_acc,
                        }
                        try:
                            cfg = ttnn.WormholeComputeKernelConfig(
                                math_fidelity=fidelity,
                                math_approx_mode=False,
                                fp32_dest_acc_en=fp32_acc,
                                packer_l1_acc=True,
                            )
                            out = ttnn.linear(tt_act, tt_weight, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                            host = ttnn.to_torch(out).float()
                            row["gain"] = round(gain(golden, host), 6)
                            row["pcc"] = round(pcc(golden, host), 6)
                            ttnn.deallocate(out)
                        except Exception as exc:  # noqa: BLE001 - a blocker is a result
                            row["error"] = f"{type(exc).__name__}: {exc}"[:160]
                        results.append(row)
                        print(
                            f"  K={depth:<6d} {dtype_name:5s} {fid_name:6s} "
                            f"fp32_acc={str(fp32_acc):5s} gain {row.get('gain', float('nan')):.6f} "
                            f"pcc {row.get('pcc', float('nan')):.6f}"
                            + (f"  ERROR {row['error']}" if row.get("error") else ""),
                            flush=True,
                        )
                ttnn.deallocate(tt_weight)
            ttnn.deallocate(tt_act)
    finally:
        for row in results:
            print("PROBEROW " + json.dumps(row, sort_keys=True, default=str), flush=True)
        ttnn.close_mesh_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
