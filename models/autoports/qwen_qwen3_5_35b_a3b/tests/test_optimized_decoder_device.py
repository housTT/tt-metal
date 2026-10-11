# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the delivered optimized defaults with pinned real and target-shape weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("length", [129, 1000])
@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("weight_kind", ["synthetic", "real"])
def test_optimized_decoder(layer_idx, weight_kind, length, tmp_path):
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
            "optimized",
            "--layer",
            str(layer_idx),
            "--length",
            str(length),
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
    assert result["decoder_class"] == "OptimizedDecoder"
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
        "attention": "_OptimizedGDN" if layer_idx == 0 else "_OptimizedAttention",
        "router": "_OptimizedRouter",
        "experts": "_OptimizedPrefillExperts",
        "decode_experts": "_OptimizedSparseExperts",
        "shared": "_OptimizedShared",
    }
    selected = result["optimization"]
    assert selected["expert_backend"] == "sparse" and selected["sparse_indexed"]
    assert selected["prefill_backend"] == "dispatch" and selected["dispatch_rank"] == "matmul"
    assert list(selected["dispatch_grid_large"]) == [2048, 8, 8] and selected["dispatch_dtype"] == "bfp8"
    assert selected["dispatch_tiles_from"] == 2048
    assert selected["prefill_moe_batch"] and selected["prefill_masked_tail"]
    assert selected["gate_pack"] and selected["decode_norm_cores"] == (0 if layer_idx == 0 else 16)
    assert selected["kv_dtype"] == ("bf16" if layer_idx == 0 else "bfp8")
