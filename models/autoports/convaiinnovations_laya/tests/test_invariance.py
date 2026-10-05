# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from models.autoports.convaiinnovations_laya.tests.decision_agreement import GATE_MAX_ABS_DP, N_QUESTIONS
from models.autoports.convaiinnovations_laya.tests.decision_agreement import main as run_agreement
from models.autoports.convaiinnovations_laya.tests.run_fidelity import DOC_DIR

POLICY = os.environ.get("LAYA_POLICY") or "bf8w_hifi3_erf"
MESH = os.environ.get("LAYA_MESH_SHAPE") or "1x1"
LIVE = os.environ.get("LAYA_INVARIANCE_LIVE") == "1"
EVIDENCE = os.path.join(DOC_DIR, "decision_agreement.json")


@pytest.fixture(scope="module")
def agreement(tmp_path_factory):
    if LIVE:
        out = str(tmp_path_factory.mktemp("invariance") / "decision_agreement.json")
        run_agreement(["--policy", POLICY, "--mesh", MESH, "--out", out])
        path = out
    else:
        if not os.path.exists(EVIDENCE):
            pytest.skip(f"{EVIDENCE} missing; run tests/decision_agreement.py or set LAYA_INVARIANCE_LIVE=1")
        path = EVIDENCE
    with open(path) as f:
        return json.load(f)


def test_sixteen_questions_in_five_placements(agreement):
    assert agreement["summary"]["questions"] == N_QUESTIONS
    assert agreement["summary"]["placements"] == ["alone", "b2", "b4", "mixed_b8", "b64"]
    assert agreement["buckets"]["alone"][0] == 1 or agreement["mesh"] != "1x1"
    assert agreement["buckets"]["b2"][0] * agreement["engine"]["num_devices"] == 2 or agreement["mesh"] != "1x1"
    assert agreement["buckets"]["b4"][0] * agreement["engine"]["num_devices"] == 4 or agreement["mesh"] != "1x1"
    assert agreement["buckets"]["b64"][0] * agreement["engine"]["num_devices"] == 64


def test_same_argmax_alone_and_in_batch(agreement):
    g = agreement["gates"]["same_argmax"]
    assert g["pass"], g


def test_max_abs_dp_alone_vs_in_batch(agreement):
    g = agreement["gates"]["max_abs_dp_alone_vs_in_batch"]
    assert g["value"] <= GATE_MAX_ABS_DP, g
    for p, v in agreement["summary"]["max_abs_dp_alone_vs"].items():
        assert v <= GATE_MAX_ABS_DP, (p, v)
