# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the Qwen3.6 selected precision policy contract."""

from __future__ import annotations

import json

import pytest

from models.autoports.qwen_qwen3_6_27b.tt.precision import (
    DEFAULT_PRECISION_CONFIG,
    PRECISION_CONFIG_ENV,
    load_precision_policy,
)


def _write(path, data):
    path.write_text(json.dumps(data) + "\n")
    return path


def test_default_selected_policy_is_complete(monkeypatch):
    monkeypatch.delenv(PRECISION_CONFIG_ENV, raising=False)
    policy = load_precision_policy()

    assert policy.path == DEFAULT_PRECISION_CONFIG.resolve()
    assert policy.config_id == "full_down_bfp4_lofi"
    assert policy.weight_dtype("mlp_down", layer_kind="linear_attention", layer_idx=0) == "bfp4"
    assert policy.weight_dtype("mlp_down", layer_kind="full_attention", layer_idx=3) == "bfp4"
    assert policy.fidelity("mlp_down", layer_kind="full_attention", layer_idx=3) == "lofi"
    assert policy.activation_residual == {
        "activations": "bf16",
        "residual": "bf16",
        "norm": "bf16",
    }
    assert set(policy.ccl) == {
        "linear_attention_attention",
        "linear_attention_mlp",
        "full_attention_attention",
        "full_attention_mlp",
    }
    assert policy.kv_cache["dtype"] == "bfp8"
    assert policy.kv_cache["layout"] == "tile_dram_paged"
    assert policy.kv_cache["page_block_size"] == 64
    assert policy.logits_sampling["logits_dtype"] == "bf16"
    assert policy.logits_sampling["sampling_dtype"] == "bf16"


def test_environment_override_and_base_config_are_consumed(tmp_path, monkeypatch):
    base = json.loads(DEFAULT_PRECISION_CONFIG.read_text())
    base["config_id"] = "base"
    _write(tmp_path / "base.json", base)
    candidate = _write(
        tmp_path / "candidate.json",
        {
            "base_config": "base.json",
            "config_id": "candidate",
            "layer_exceptions": {
                "3": {"mlp_down": "bfp8", "mlp_down_fidelity": "hifi2"}
            },
            "kv_cache": {"dtype": "bf16"},
        },
    )
    monkeypatch.setenv(PRECISION_CONFIG_ENV, str(candidate))

    policy = load_precision_policy()
    assert policy.path == candidate.resolve()
    assert policy.weight_dtype("mlp_down", layer_kind="full_attention", layer_idx=3) == "bfp8"
    assert policy.weight_dtype("mlp_down", layer_kind="full_attention", layer_idx=7) == "bfp4"
    assert policy.fidelity("mlp_down", layer_kind="full_attention", layer_idx=3) == "hifi2"
    assert policy.kv_cache["dtype"] == "bf16"


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    [
        ("activation_residual", "activations", "bfp8", "activation_residual.activations"),
        ("logits_sampling", "sampling_dtype", "bfp8", "logits_sampling.sampling_dtype"),
        ("kv_cache", "dtype", "bfp4", "kv_cache.dtype"),
        ("kv_cache", "dtype", "BFP8", "canonical lowercase"),
        ("ccl", "linear_attention_mlp", "BfP8", "canonical lowercase"),
        ("kv_cache", "layout", "row_major", "kv_cache.layout"),
        ("kv_cache", "page_block_size", 32, "kv_cache.page_block_size"),
    ],
)
def test_unsupported_runtime_fields_are_rejected(tmp_path, section, field, value, match):
    data = json.loads(DEFAULT_PRECISION_CONFIG.read_text())
    data[section][field] = value
    path = _write(tmp_path / "invalid.json", data)
    with pytest.raises(ValueError, match=match):
        load_precision_policy(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data.update({"ignored_root_policy": "bf16"}), "root keys"),
        (
            lambda data: data["weight_groups"]["mlp_down"].update(
                {"full_attentoin": "bf16"}
            ),
            "weight_groups.mlp_down keys",
        ),
        (
            lambda data: data["compute_fidelities"].update(
                {
                    "mlp_down": {
                        "linear_attention": "lofi",
                        "full_attention": "lofi",
                        "full_attentoin": "hifi4",
                    }
                }
            ),
            "compute_fidelities.mlp_down keys",
        ),
    ],
)
def test_unknown_root_and_nested_fields_are_rejected(tmp_path, mutation, match):
    data = json.loads(DEFAULT_PRECISION_CONFIG.read_text())
    mutation(data)
    path = _write(tmp_path / "unknown.json", data)
    with pytest.raises(ValueError, match=match):
        load_precision_policy(path)
