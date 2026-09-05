# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check whether precision permits faster QKVG blocks on the real batch32 case."""

import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
doc = root / "doc/multichip_decoder"
for dtype, fidelity, block in [
    ("bfloat4_b", "HiFi2", 16),
    ("bfloat4_b", "HiFi2", 8),
    ("bfloat8_b", "LoFi", 16),
    ("bfloat8_b", "LoFi", 8),
    ("bfloat8_b", "HiFi2", 16),
]:
    name = f"qkvg_accuracy_{dtype}_{fidelity}_b{block}"
    policy = {"attention": dtype, "attention_fidelity": fidelity}
    subprocess.run(
        [
            sys.executable,
            str(doc / "record_run.py"),
            name,
            "timeout",
            "180",
            sys.executable,
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batch32_diagnostic",
            "--name",
            name,
            "--only",
            "tp4",
            "--qkvg-block",
            str(block),
            "--policy",
            json.dumps(policy),
        ],
        check=True,
    )

for block in (16, 8):
    name = f"qkvg_accuracy_fp32_LoFi_b{block}"
    subprocess.run(
        [
            sys.executable,
            str(doc / "record_run.py"),
            name,
            "timeout",
            "180",
            sys.executable,
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batch32_diagnostic",
            "--name",
            name,
            "--only",
            "tp4",
            "--qkvg-block",
            str(block),
            "--fp32-attention",
        ],
        check=True,
    )
