# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os

import pytest

from models.autoports.convaiinnovations_laya.tests.run_fidelity import DOC_DIR, GATES
from models.autoports.convaiinnovations_laya.tests.run_fidelity import main as run_fidelity

POLICY = os.environ.get("LAYA_POLICY") or "bf8w_hifi3_erf"
MESH = os.environ.get("LAYA_MESH_SHAPE") or "1x1"
LIVE = os.environ.get("LAYA_FIDELITY_LIVE") == "1"


def _evidence_path():
    exact = os.path.join(DOC_DIR, f"fidelity_{POLICY}_{MESH}.json")
    if os.path.exists(exact):
        return exact
    found = sorted(glob.glob(os.path.join(DOC_DIR, f"fidelity_{POLICY}_*.json")))
    return found[-1] if found else None


@pytest.fixture(scope="module")
def fidelity(tmp_path_factory):
    if LIVE:
        out = str(tmp_path_factory.mktemp("fidelity") / f"fidelity_{POLICY}_{MESH}.json")
        run_fidelity(
            [
                "--policy",
                POLICY,
                "--mesh",
                MESH,
                "--items",
                "gate",
                "--hidden-cases",
                os.environ.get("LAYA_HIDDEN_CASES", "8"),
                "--out",
                out,
            ]
        )
        path = out
    else:
        path = _evidence_path()
        if path is None:
            pytest.skip(
                f"no fidelity evidence for {POLICY} under {DOC_DIR}; run tests/run_fidelity.py or set LAYA_FIDELITY_LIVE=1"
            )
    with open(path) as f:
        return json.load(f)


def test_gate_subset_has_200_decisions(fidelity):
    assert fidelity["gate_subset"]["n"] == 200


def test_confident_argmax_agreement(fidelity):
    g = fidelity["gates"]["confident_argmax_agreement"]
    assert g["value"] >= GATES["confident_agreement"], g


def test_median_max_abs_dp(fidelity):
    g = fidelity["gates"]["median_max_abs_dp"]
    assert g["value"] <= GATES["median_max_abs_dp"], g


def test_scorer_logit_pcc(fidelity):
    g = fidelity["gates"]["scorer_logit_pcc"]
    assert g["value"] >= GATES["scorer_pcc"], g


def test_hidden_state_pcc(fidelity):
    g = fidelity["gates"].get("hidden_state_pcc")
    if g is None:
        pytest.skip("hidden states were not compared in this run")
    assert g["encoder_pooled"] >= GATES["hidden_pcc"] and g["head_pooled"] >= GATES["hidden_pcc"], g


def test_no_nan(fidelity):
    assert fidelity["gates"]["no_nan"]["pass"], fidelity["gates"]["no_nan"]


def test_act_argmax_agreement_is_reported_not_gated(fidelity):
    r = fidelity["reported_not_gated"]
    assert "act_argmax_agreement" in r and "p95_max_abs_dp" in r and "plain_argmax_agreement" in r
