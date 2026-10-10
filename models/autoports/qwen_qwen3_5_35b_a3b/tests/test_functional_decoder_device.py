# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape device regression; synthetic mode needs no checkpoint download."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("weight_kind", ["synthetic", "real"])
def test_functional_decoder(layer_idx, weight_kind, tmp_path):
    checkpoint = Path(__file__).resolve().parents[5] / "weights" / "Qwen3.5-35B-A3B"
    if weight_kind == "real" and not checkpoint.is_dir():
        pytest.skip("Pinned real checkpoint is not installed; synthetic target-shape test still runs")
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
            "--layer",
            str(layer_idx),
            "--length",
            "129",
            "--steps",
            "8",
            "--trace",
            "--audit",
            "--weights",
            weight_kind,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "result.json"),
        ],
        env=env,
        check=True,
    )
