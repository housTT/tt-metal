# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Remeasure compatible topology/precision controls on the selected local geometry."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
doc = root / "doc/multichip_decoder"
parser = argparse.ArgumentParser()
parser.add_argument("--label", default="selected32")
parser.add_argument("--layers", nargs="+", type=int, default=[0, 3])
parser.add_argument("--start-at", default="replicated")
args = parser.parse_args()
cases = [("replicated", []), ("sharded", ["--residual", "sharded"])]
for variant in ("ag_mm", "fused_ag_mm", "fused_mm_rs", "fused_norm_ag_mm"):
    cases.append((variant, ["--variant", variant, "--residual", "sharded"]))
for variant in ("packed_mlp", "persistent", "ccl_bfp8"):
    for residual in ("replicated", "sharded"):
        cases.append((f"{variant}_{residual}", ["--variant", variant, "--residual", residual]))
for group in ("attention", "mlp"):
    for residual in ("replicated", "sharded"):
        cases.append((f"acts8_{group}_{residual}", ["--activation-group", group, "--residual", residual]))
    for dtype in ("bfloat4_b", "bfloat8_b"):
        for fidelity in ("LoFi", "HiFi2"):
            if dtype == "bfloat4_b" and fidelity == "LoFi":
                continue
            policy = (
                {"attention": dtype, "attention_fidelity": fidelity}
                if group == "attention"
                else {"mlp_gate_up": dtype, "mlp_down": dtype, "mlp_fidelity": fidelity}
            )
            cases.append(
                (
                    f"{group}_{dtype}_{fidelity}",
                    ["--policy", json.dumps(policy), "--local-config", json.dumps({"large_prefill_block_w": 8})],
                )
            )
start = [name for name, _ in cases].index(args.start_at)
for name, extra in cases[start:]:
    for layer in args.layers:
        command = [
            sys.executable,
            str(doc / "record_run.py"),
            f"{args.label}_{name}_layer{layer}",
            "timeout",
            "180",
            sys.executable,
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe",
            "--layer",
            str(layer),
            "--length",
            "2048",
            *extra,
        ]
        subprocess.run(command, check=True)
