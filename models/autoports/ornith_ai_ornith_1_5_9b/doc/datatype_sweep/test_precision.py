# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only rejection and layer exception tests for the policy contract."""

import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "tt/precision.py"
spec = importlib.util.spec_from_file_location("ornith_precision", SOURCE)
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


def test_baseline_isolation_and_layer_exceptions():
    config = precision.load_precision("baseline")
    config["layer_exceptions"] = {"3": {"weight_groups": {"attention": "bfloat8_b"}}}
    loaded = precision.load_precision(config)
    assert precision.layer_precision(loaded, 3)["weight_groups"]["attention"] == "bfloat8_b"
    assert precision.layer_precision(loaded, 0)["weight_groups"]["attention"] == "bfloat4_b"
    assert precision.load_precision("baseline")["layer_exceptions"] == {}


@pytest.mark.parametrize(
    "key,value",
    [
        ("activation_dtype", "bfloat8_b"),
        ("residual_dtype", "float32"),
        ("max_context", 2048),
        ("kv_cache_dtype", "uint32"),
        ("ccl_dtype", "ignored"),
    ],
)
def test_invalid_contract_rejected(key, value):
    config = precision.load_precision("baseline")
    config[key] = value
    with pytest.raises(ValueError):
        precision.load_precision(config)


def test_unknown_layer_group_rejected():
    config = precision.load_precision("baseline")
    config["layer_exceptions"] = {"0": {"weight_groups": {"typo": "bfloat4_b"}}}
    with pytest.raises(ValueError):
        precision.load_precision(config)


def test_default_loads_selected_artifact(tmp_path, monkeypatch):
    import json

    config = precision.load_precision("baseline")
    config["config_id"] = "selected-test"
    config["weight_groups"]["lm_head"] = "bfloat8_b"
    artifact = tmp_path / "selected_precision_config.json"
    artifact.write_text(json.dumps(config))
    monkeypatch.setattr(precision, "SELECTED", artifact)
    assert precision.load_precision() == config
    assert precision.load_precision("baseline")["weight_groups"]["lm_head"] == "bfloat16"


def test_missing_selected_artifact_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(precision, "SELECTED", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        precision.load_precision()


@pytest.mark.parametrize("override", [{31: {}}, {"031": {}}, {"31": {"weight_groups": {"lm_head": "bfloat8_b"}}}])
def test_unused_layer_exceptions_rejected(override):
    config = precision.load_precision("baseline")
    config["layer_exceptions"] = override
    with pytest.raises(ValueError):
        precision.load_precision(config)


def test_legacy_policy_preserves_k1_geometry():
    config = precision.load_precision("baseline")
    del config["head_geometry"]
    assert precision.load_precision(config)["head_geometry"]["in0_block_w"] == 1
