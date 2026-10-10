# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the delivered fused defaults with pinned real and target-shape weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("weight_kind", ["synthetic", "real"])
def test_fused_decoder(layer_idx, weight_kind, tmp_path):
    checkpoint = Path(__file__).resolve().parents[5] / "weights" / "Qwen3.5-35B-A3B"
    if weight_kind == "real" and not checkpoint.is_dir():
        pytest.skip("Pinned real checkpoint is not installed")
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
            "--decoder",
            "fused",
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
            str(output),
        ],
        env=dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"),
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["decoder_class"] == "FusedDecoder"
    required = {
        "moe_gate",
        "shared_minimal",
        "compact_reduce",
        "compact_routing",
        "prefill_split",
        "gdn_grouped" if layer_idx == 0 else "qkv",
    }
    assert required <= set(result["fusions"])
    if layer_idx == 3:
        assert "indexed_rope" in result["fusions"]
    assert result["runtime_components"] == {
        "attention": "_FusedGDN" if layer_idx == 0 else "_FusedAttention",
        "router": "_DedicatedRouter",
        "experts": "_PackedPrefillExperts",
        "decode_experts": "_CompactExperts",
        "shared": "_FusedShared",
    }
