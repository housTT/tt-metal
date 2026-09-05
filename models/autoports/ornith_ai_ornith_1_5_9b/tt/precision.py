# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validated full-model precision policy shared by all model construction paths."""

import copy
import json
from pathlib import Path

SELECTED = Path(__file__).resolve().parents[1] / "doc/datatype_sweep/selected_precision_config.json"
BASELINE = {
    "config_id": "baseline_bfp4_lofi_qkvg8_lofi_head16_hifi4",
    "weight_groups": {
        "attention": "bfloat4_b",
        "mlp_gate_up": "bfloat4_b",
        "mlp_down": "bfloat4_b",
        "decode_qkvg": "bfloat8_b",
        "lm_head": "bfloat16",
        "embedding": "bfloat16",
        "norm": "bfloat16",
    },
    "compute_fidelities": {"attention": "LoFi", "mlp_gate_up": "LoFi", "mlp_down": "LoFi", "lm_head": "HiFi4"},
    "layer_exceptions": {},
    "activation_dtype": "bfloat16",
    "residual_dtype": "bfloat16",
    "ccl_dtype": "native",
    "kv_cache_dtype": "bfloat8_b",
    "logits_dtype": "bfloat16",
    "sampling_dtype": "bfloat16",
    "token_dtype": "uint32",
    "recurrent_dtype": "float32",
    "max_context": 262144,
    "head_geometry": {"cores": 64, "columns": 32768, "in0_block_w": 1, "readers": 2},
    "matmul_flags": {
        "projection_fp32_dest_acc_en": False,
        "head_fp32_dest_acc_en": True,
        "math_approx_mode": False,
        "packer_l1_acc": True,
    },
}


def load_precision(value=None):
    if value is None:
        value = SELECTED
    if value == "baseline":
        value = BASELINE
    result = copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())
    result.setdefault("head_geometry", copy.deepcopy(BASELINE["head_geometry"]))
    if set(result) != set(BASELINE):
        raise ValueError(f"Precision policy fields differ: {set(result) ^ set(BASELINE)}")
    for field in (
        "activation_dtype",
        "residual_dtype",
        "logits_dtype",
        "sampling_dtype",
        "token_dtype",
        "recurrent_dtype",
        "max_context",
        "matmul_flags",
    ):
        if result[field] != BASELINE[field]:
            raise ValueError(f"Unsupported {field}: {result[field]}; required runtime contract {BASELINE[field]}")
    if set(result["weight_groups"]) != set(BASELINE["weight_groups"]) or set(result["compute_fidelities"]) != set(
        BASELINE["compute_fidelities"]
    ):
        raise ValueError("Incomplete weight/fidelity groups")
    for group, dtype in result["weight_groups"].items():
        if dtype not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            raise ValueError(f"Unsupported {group} dtype {dtype}")
        if group in ("embedding", "norm") and dtype != "bfloat16":
            raise ValueError(f"{group} requires BF16")
    if any(v not in ("LoFi", "HiFi2", "HiFi4") for v in result["compute_fidelities"].values()):
        raise ValueError("Unsupported fidelity")
    if result["kv_cache_dtype"] not in ("bfloat16", "bfloat8_b", "bfloat4_b"):
        raise ValueError("Unsupported KV dtype")
    if result["ccl_dtype"] not in ("native", "bfloat8_b"):
        raise ValueError("CCL supports native producer dtype or BFP8 transfer")
    geometry = result["head_geometry"]
    if (
        set(geometry) != set(BASELINE["head_geometry"])
        or geometry["cores"] not in (16, 32, 64)
        or geometry["columns"] != 32768
        or geometry["readers"] not in (1, 2)
        or not isinstance(geometry["in0_block_w"], int)
        or geometry["in0_block_w"] < 1
        or (128 // geometry["cores"]) % geometry["in0_block_w"]
    ):
        raise ValueError("Unsupported head geometry")
    for index, override in result["layer_exceptions"].items():
        if (
            not isinstance(index, str)
            or not index.isdigit()
            or index != str(int(index))
            or not 0 <= int(index) < 32
            or set(override) - {"weight_groups", "compute_fidelities"}
        ):
            raise ValueError("Invalid layer exception")
        if set(override.get("weight_groups", {})) - {"attention", "mlp_gate_up", "mlp_down", "decode_qkvg"} or set(
            override.get("compute_fidelities", {})
        ) - {"attention", "mlp_gate_up", "mlp_down"}:
            raise ValueError("Layer exceptions must name per-layer projection groups")
        merged = copy.deepcopy(result)
        merged["layer_exceptions"] = {}
        for key, fields in override.items():
            merged[key].update(fields)
        load_precision(merged)
    return result


def layer_precision(policy, index):
    result = copy.deepcopy(policy)
    for key, fields in policy["layer_exceptions"].get(str(index), {}).items():
        result[key].update(fields)
    return result
