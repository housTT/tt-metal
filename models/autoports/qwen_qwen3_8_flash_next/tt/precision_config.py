# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mechanically consumed precision policy for Qwen3.8-Flash-Next.

The datatype-sweep artifact is deliberately runtime configuration, not only
documentation.  The selected file is loaded by the ordinary full-model
constructor; ``QWEN38_PRECISION_CONFIG`` selects one sweep candidate without
changing source or bypassing ``build_generator``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import ttnn


DEFAULT_PRECISION_CONFIG_PATH = (
    Path(__file__).parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"
)

_DTYPES = {
    "bf16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfp4": ttnn.bfloat4_b,
    "fp32": ttnn.float32,
}
_LAYOUTS = {
    "row_major": ttnn.ROW_MAJOR_LAYOUT,
    "tile": ttnn.TILE_LAYOUT,
}
_PROJECTION_POLICIES = {
    "bf16_hifi4": ("bf16", "hifi4"),
    "bf16_hifi2": ("bf16", "hifi2"),
    "bfp8_hifi2": ("bfp8", "hifi2"),
    "bfp8_lofi": ("bfp8", "lofi"),
    "bfp4_hifi2": ("bfp4", "hifi2"),
    "bfp4_lofi": ("bfp4", "lofi"),
}
_EXPERT_POLICIES = {
    "expert_bfp4_hifi2_g40b16_d40b5": ("bfp4", "hifi2"),
    "expert_bfp4_lofi_g40b16_d40b5": ("bfp4", "lofi"),
}
_LM_HEAD_POLICIES = {
    "bf16_hifi2": ("bf16", "hifi2", "bf16"),
    "bf16_lofi": ("bf16", "lofi", "bf16"),
    "bfp8_hifi2": ("bfp8", "hifi2", "bfp8"),
    "bfp8_lofi": ("bfp8", "lofi", "bfp8"),
    "bfp4_lofi": ("bfp4", "lofi", "bf16"),
}


def dtype_object(name: str):
    try:
        return _DTYPES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3.8 dtype {name!r}") from exc


def layout_object(name: str):
    try:
        return _LAYOUTS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported Qwen3.8 layout {name!r}") from exc


def _require(mapping: dict, path: str):
    value = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"precision config is missing {path!r}")
        value = value[part]
    return value


def _validate_policy_pair(config: dict, group: str) -> None:
    value = _require(config, f"weight_groups.{group}")
    policy = str(_require(value, "policy"))
    try:
        expected = _PROJECTION_POLICIES[policy]
    except KeyError as exc:
        raise ValueError(f"unsupported {group} policy {policy!r}") from exc
    actual = (str(_require(value, "dtype")), str(_require(value, "compute_fidelity")))
    if actual != expected:
        raise ValueError(f"{group} policy {policy!r} means {expected}, not {actual}")


def validate_precision_config(config: dict) -> dict:
    """Validate the complete selected/candidate policy and return a copy."""

    value = copy.deepcopy(config)
    if int(_require(value, "schema_version")) != 1:
        raise ValueError("Qwen3.8 precision config schema_version must be 1")
    if not str(_require(value, "config_id")):
        raise ValueError("precision config_id cannot be empty")

    for group in ("shared_projection", "gdn_projection", "qsa_input", "attention_output"):
        _validate_policy_pair(value, group)

    expert = _require(value, "weight_groups.routed_expert")
    expert_policy = str(_require(expert, "policy"))
    if expert_policy not in _EXPERT_POLICIES:
        raise ValueError(f"unsupported routed expert policy {expert_policy!r}")
    expert_pair = (str(_require(expert, "dtype")), str(_require(expert, "compute_fidelity")))
    if expert_pair != _EXPERT_POLICIES[expert_policy]:
        raise ValueError(f"routed expert policy {expert_policy!r} means {_EXPERT_POLICIES[expert_policy]}")

    lm_head = _require(value, "weight_groups.lm_head")
    lm_policy = str(_require(lm_head, "policy"))
    if lm_policy not in _LM_HEAD_POLICIES:
        raise ValueError(f"unsupported LM-head policy {lm_policy!r}")
    lm_actual = (
        str(_require(lm_head, "dtype")),
        str(_require(lm_head, "compute_fidelity")),
        str(_require(value, "logits_sampling.logits_dtype")),
    )
    if lm_actual != _LM_HEAD_POLICIES[lm_policy]:
        raise ValueError(f"LM-head policy {lm_policy!r} means {_LM_HEAD_POLICIES[lm_policy]}, not {lm_actual}")

    for path in (
        "weight_groups.embedding.dtype",
        "weight_groups.final_hyper_down_up.dtype",
        "weight_exceptions.norms.dtype",
        "weight_exceptions.router_topk_outputs.dtype",
        "activations.model_input_dtype",
        "activations.residual_dtype",
        "activations.matmul_output_dtype",
        "activations.ple_dtype",
        "ccl.payload_dtype",
        "kv_cache.dtype",
        "kv_cache.update_dtype",
        "logits_sampling.sampling_dtype",
        "host_backed.expert.source_dtype",
        "host_backed.expert.host_packed_dtype",
        "host_backed.expert.device_staging_dtype",
        "host_backed.expert.execution_weight_dtype",
        "host_backed.ple.table_dtype",
        "host_backed.ple.host_assembly_dtype",
        "host_backed.ple.device_staging_dtype",
        "host_backed.ple.execution_dtype",
    ):
        dtype_object(str(_require(value, path)))
    for path in (
        "weight_groups.embedding.layout",
        "kv_cache.layout",
        "host_backed.expert.host_packed_layout",
        "host_backed.expert.device_staging_layout",
        "host_backed.ple.device_staging_layout",
    ):
        layout_object(str(_require(value, path)))

    if str(_require(value, "activations.model_input_dtype")) != "bf16":
        raise ValueError("the current Qwen3.8 embedding/ingress contract requires BF16 model inputs")
    if str(_require(value, "activations.matmul_output_dtype")) != "bf16":
        raise ValueError("the current Qwen3.8 matmul-output contract requires BF16")
    if str(_require(value, "activations.ple_dtype")) != "bf16":
        raise ValueError("the exact PLE execution boundary currently requires BF16")
    if str(_require(value, "activations.ple_dtype")) != str(
        _require(value, "host_backed.ple.execution_dtype")
    ):
        raise ValueError("PLE activation and host-backed execution dtypes must match")
    if str(_require(value, "weight_exceptions.norms.dtype")) != "bf16" or str(
        _require(value, "weight_exceptions.norms.compute_fidelity")
    ) != "hifi4":
        raise ValueError("the current RMSNorm contract requires BF16 weights and HiFi4")
    if str(_require(value, "weight_exceptions.router_topk_outputs.dtype")) != str(
        _require(value, "activations.matmul_output_dtype")
    ):
        raise ValueError("router/top-k output dtype must match the router matmul output dtype")
    if str(_require(value, "activations.residual_dtype")) not in {"bf16", "bfp8"}:
        raise ValueError("residual_dtype must be bf16 or bfp8")
    if str(_require(value, "ccl.payload_dtype")) not in {"bf16", "bfp8"}:
        raise ValueError("CCL payload_dtype must be bf16 or bfp8")
    if str(_require(value, "ccl.topology")) != "linear":
        raise ValueError("the fixed P300 precision contract requires linear CCL topology")
    if int(_require(value, "ccl.num_links")) not in {1, 2}:
        raise ValueError("CCL num_links must be 1 or 2")
    if int(_require(value, "ccl.packet_bytes")) != 8192:
        raise ValueError("the measured P300 router contract requires 8192-byte packets")

    cache_policy = str(_require(value, "kv_cache.policy"))
    cache_dtype = str(_require(value, "kv_cache.dtype"))
    if (cache_policy, cache_dtype) not in {("bf16", "bf16"), ("bfp8", "bfp8")}:
        raise ValueError("KV policy/dtype must be bf16/bf16 or bfp8/bfp8")
    if str(_require(value, "kv_cache.update_dtype")) != "bf16":
        raise ValueError("paged update tensors must remain BF16 for both cache policies")
    if str(_require(value, "kv_cache.layout")) != "tile" or int(_require(value, "kv_cache.page_block_size")) != 64:
        raise ValueError("Qwen3.8 KV cache requires TILE layout and page_block_size 64")

    if str(_require(value, "logits_sampling.sampling_mode")) != "device":
        raise ValueError("selected precision configs must use device sampling")
    if str(_require(value, "logits_sampling.greedy_strategy")) != "full_vocabulary_argmax":
        raise ValueError("selected precision configs require the traced full-vocabulary device argmax")

    host_expert = _require(value, "host_backed.expert")
    if str(_require(host_expert, "source_dtype")) != "bf16":
        raise ValueError("checkpoint expert source must remain BF16")
    for field in ("host_packed_dtype", "device_staging_dtype", "execution_weight_dtype"):
        if str(_require(host_expert, field)) != expert_pair[0]:
            raise ValueError(f"host expert {field} must preserve routed expert dtype {expert_pair[0]}")
    if str(_require(host_expert, "host_packed_layout")) != "tile" or str(
        _require(host_expert, "device_staging_layout")
    ) != "tile":
        raise ValueError("expert host/device packing requires TILE layout")
    if int(_require(host_expert, "slots_per_layer")) < 10:
        raise ValueError("expert slots must cover exact top-10 routing")
    if int(_require(host_expert, "packed_capacity_per_layer")) != 512:
        raise ValueError("full-model ranking fixes packed expert capacity at all 512 experts")
    if not bool(_require(host_expert, "prepack_all")):
        raise ValueError("full-model datatype ranking requires the fixed fully warm packed-host regime")

    host_ple = _require(value, "host_backed.ple")
    for field in ("table_dtype", "host_assembly_dtype", "device_staging_dtype", "execution_dtype"):
        if str(_require(host_ple, field)) != "bf16":
            raise ValueError(f"exact PLE {field} must remain BF16")
    if str(_require(host_ple, "table_layout")) != "row_major_mmap":
        raise ValueError("PLE table_layout must be row_major_mmap")
    if str(_require(host_ple, "device_staging_layout")) != "tile":
        raise ValueError("PLE device staging layout must be TILE")
    if int(_require(host_ple, "row_cache_capacity")) != 8192:
        raise ValueError("datatype ranking fixes the PLE row-cache capacity at 8192")
    if int(_require(host_ple, "prefill_chunk_rows")) != 128:
        raise ValueError("datatype ranking fixes PLE prefill chunks at 128 rows")

    exceptions = _require(value, "layer_exceptions")
    if not isinstance(exceptions, dict):
        raise ValueError("layer_exceptions must be a mapping from zero-based layer id to overrides")
    for layer, overrides in exceptions.items():
        if not str(layer).isdigit() or not 0 <= int(layer) < 48 or not isinstance(overrides, dict):
            raise ValueError(f"invalid layer exception {layer!r}")
        unknown = set(overrides) - {"routed_expert", "shared_projection", "gdn_projection", "qsa_input", "attention_output", "kv_cache"}
        if unknown:
            raise ValueError(f"unknown layer exception field(s) {sorted(unknown)}")
    return value


def load_precision_config(path: str | Path | None = None) -> tuple[dict, Path]:
    selected = Path(path or os.getenv("QWEN38_PRECISION_CONFIG", DEFAULT_PRECISION_CONFIG_PATH)).resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"Qwen3.8 precision config does not exist: {selected}")
    return validate_precision_config(json.loads(selected.read_text())), selected


def layer_policy(config: dict, layer_idx: int) -> dict[str, object]:
    """Return the exact decoder kwargs for one measured layer."""

    groups = config["weight_groups"]
    result = {
        "optimization_policy": groups["routed_expert"]["policy"],
        "shared_projection_policy": groups["shared_projection"]["policy"],
        "gdn_projection_policy": groups["gdn_projection"]["policy"],
        "qsa_input_policy": groups["qsa_input"]["policy"],
        "attention_output_policy": groups["attention_output"]["policy"],
        "cache_policy": config["kv_cache"]["policy"],
        "cache_update_dtype": config["kv_cache"]["update_dtype"],
        "residual_dtype": config["activations"]["residual_dtype"],
        "matmul_output_dtype": config["activations"]["matmul_output_dtype"],
        "ple_activation_dtype": config["activations"]["ple_dtype"],
        "norm_weight_dtype": config["weight_exceptions"]["norms"]["dtype"],
        "norm_compute_fidelity": config["weight_exceptions"]["norms"]["compute_fidelity"],
        "router_output_dtype": config["weight_exceptions"]["router_topk_outputs"]["dtype"],
        "collective_payload_dtype": config["ccl"]["payload_dtype"],
        "collective_num_links": int(config["ccl"]["num_links"]),
    }
    overrides = config["layer_exceptions"].get(str(int(layer_idx)), {})
    key_map = {
        "routed_expert": "optimization_policy",
        "shared_projection": "shared_projection_policy",
        "gdn_projection": "gdn_projection_policy",
        "qsa_input": "qsa_input_policy",
        "attention_output": "attention_output_policy",
        "kv_cache": "cache_policy",
    }
    for name, selected in overrides.items():
        result[key_map[name]] = selected
    return result


__all__ = [
    "DEFAULT_PRECISION_CONFIG_PATH",
    "dtype_object",
    "layer_policy",
    "layout_object",
    "load_precision_config",
    "validate_precision_config",
]
