# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MESHES = [int(m) for m in os.environ.get("QWEN35_MESHES", "1,2,4").split(",")]
DISPATCH = {1: "stacked", 2: "expert_parallel", 4: "expert_parallel"}


@pytest.mark.parametrize("length", [129, 1000])
@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("mesh", MESHES)
def test_multichip_decoder(mesh, layer_idx, length, tmp_path):
    checkpoint = Path(__file__).resolve().parents[5] / "weights" / "Qwen3.5-35B-A3B"
    if not checkpoint.is_dir():
        pytest.skip("Pinned real checkpoint is not installed")
    env = dict(os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0")
    env.pop("TT_VISIBLE_DEVICES", None)
    if mesh == 2:
        env["TT_VISIBLE_DEVICES"] = os.environ.get("QWEN35_X2_DEVICES", "0,1")
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "models.autoports.qwen_qwen3_5_35b_a3b.tests.run_layer",
            "--decoder",
            "multichip",
            "--mesh",
            str(mesh),
            "--layer",
            str(layer_idx),
            "--length",
            str(length),
            "--steps",
            "8",
            "--trace",
            "--audit",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        env=env,
        check=True,
    )
    result = json.loads(output.read_text())
    assert result["decoder_class"] == "MultichipDecoder"
    assert result["tensor_parallel"] == mesh and result["mesh_shape"] == [1, mesh]
    assert result["output_replicas_identical"]
    components = result["runtime_components"]
    assert components["attention"] == ("_OptimizedGDN" if layer_idx == 0 else "_OptimizedAttention")
    assert components["decode_experts"] == "_OptimizedSparseExperts"
    assert components["attention_reduce"] == components["feed_forward_reduce"] == (mesh > 1)
    selected = result["optimization"]
    assert selected["dispatch_weights"] == DISPATCH[mesh]
    assert selected["expert_backend"] == "sparse" and selected["sparse_indexed"]
    assert selected["prefill_backend"] == "dispatch" and selected["kv_dtype"] == ("bf16" if layer_idx == 0 else "bfp8")
    assert result["prefill_pcc_min"] >= 0.995 and result["decode_pcc_min"] >= 0.995
