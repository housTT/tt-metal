# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision-policy loading for the Qwen/Qwen3.6-27B full-model path."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PRECISION_CONFIG = MODEL_DIR / "doc/datatype_sweep/selected_precision_config.json"
PRECISION_CONFIG_ENV = "QWEN36_PRECISION_CONFIG"

_DTYPES = {"bf16", "bfp8", "bfp4"}
_FIDELITIES = {"lofi", "hifi2", "hifi4"}


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{field} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


class PrecisionPolicy:
    """Validated, layer-aware view of a selected or candidate precision config."""

    def __init__(self, data: dict[str, Any], path: Path):
        self.data = data
        self.path = path
        required_root_keys = {
            "schema_version",
            "model_id",
            "config_id",
            "status",
            "weight_groups",
            "layer_exceptions",
            "compute_fidelities",
            "activation_residual",
            "ccl",
            "kv_cache",
            "logits_sampling",
        }
        actual_root_keys = set(data)
        allowed_root_keys = required_root_keys | {"runtime_loading"}
        missing_root_keys = required_root_keys - actual_root_keys
        extra_root_keys = actual_root_keys - allowed_root_keys
        if missing_root_keys or extra_root_keys:
            raise ValueError(
                "precision config root keys are invalid; "
                f"missing={sorted(missing_root_keys)}, extra={sorted(extra_root_keys)}"
            )
        if data.get("model_id") != MODEL_ID:
            raise ValueError(f"precision config model_id must be {MODEL_ID!r}")
        self.config_id = str(data.get("config_id", ""))
        if not self.config_id:
            raise ValueError("precision config requires config_id")
        self.weight_groups = _require_mapping(data.get("weight_groups"), "weight_groups")
        self.layer_exceptions = _require_mapping(data.get("layer_exceptions"), "layer_exceptions")
        self.compute_fidelities = _require_mapping(
            data.get("compute_fidelities"), "compute_fidelities"
        )
        self.activation_residual = _require_mapping(
            data.get("activation_residual"), "activation_residual"
        )
        self.ccl = _require_mapping(data.get("ccl"), "ccl")
        self.kv_cache = _require_mapping(data.get("kv_cache"), "kv_cache")
        self.logits_sampling = _require_mapping(data.get("logits_sampling"), "logits_sampling")
        self.runtime_loading = _require_mapping(data.get("runtime_loading", {}), "runtime_loading")
        self._validate()

    @staticmethod
    def _dtype(value: Any, field: str, *, allowed: set[str] = _DTYPES) -> str:
        raw_value = str(value)
        value = raw_value.lower()
        if value not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}, got {raw_value!r}")
        if raw_value != value:
            raise ValueError(f"{field} must use canonical lowercase spelling, got {raw_value!r}")
        return value

    @staticmethod
    def _fidelity(value: Any, field: str) -> str:
        raw_value = str(value)
        value = raw_value.lower()
        if value not in _FIDELITIES:
            raise ValueError(f"{field} must be one of {sorted(_FIDELITIES)}, got {raw_value!r}")
        if raw_value != value:
            raise ValueError(f"{field} must use canonical lowercase spelling, got {raw_value!r}")
        return value

    def _validate(self) -> None:
        _require_exact_keys(
            self.weight_groups,
            {
                "embedding",
                "attention_input",
                "attention_output",
                "mlp_gate",
                "mlp_up",
                "mlp_down",
                "final_norm",
                "lm_head",
            },
            "weight_groups",
        )
        _require_exact_keys(
            self.compute_fidelities,
            {"projection", "mlp_gate", "mlp_up", "mlp_down", "lm_head"},
            "compute_fidelities",
        )
        _require_exact_keys(
            self.activation_residual,
            {"activations", "residual", "norm"},
            "activation_residual",
        )
        _require_exact_keys(
            self.ccl,
            {
                "linear_attention_attention",
                "linear_attention_mlp",
                "full_attention_attention",
                "full_attention_mlp",
            },
            "ccl",
        )
        _require_exact_keys(self.kv_cache, {"dtype", "layout", "page_block_size"}, "kv_cache")
        _require_exact_keys(
            self.logits_sampling,
            {
                "lm_head_weight_dtype",
                "logits_dtype",
                "sampling_dtype",
                "greedy_accumulation",
                "stochastic_assumption",
            },
            "logits_sampling",
        )
        for field in ("embedding", "final_norm"):
            self._dtype(
                self.weight_groups.get(field),
                f"weight_groups.{field}",
                allowed={"bf16"},
            )
        for field in ("attention_input", "attention_output"):
            self._dtype(self.weight_groups.get(field), f"weight_groups.{field}")
        self._dtype(
            self.weight_groups.get("lm_head"),
            "weight_groups.lm_head",
            allowed={"bf16", "bfp8"},
        )
        for role in ("gate", "up", "down"):
            by_kind = _require_mapping(
                self.weight_groups.get(f"mlp_{role}"), f"weight_groups.mlp_{role}"
            )
            _require_exact_keys(
                by_kind,
                {"linear_attention", "full_attention"},
                f"weight_groups.mlp_{role}",
            )
            for kind in ("linear_attention", "full_attention"):
                self._dtype(by_kind.get(kind), f"weight_groups.mlp_{role}.{kind}")
        for field in ("projection", "mlp_gate", "mlp_up", "mlp_down", "lm_head"):
            value = self.compute_fidelities.get(field)
            if isinstance(value, dict):
                _require_exact_keys(
                    value,
                    {"linear_attention", "full_attention"},
                    f"compute_fidelities.{field}",
                )
                for kind in ("linear_attention", "full_attention"):
                    self._fidelity(value.get(kind), f"compute_fidelities.{field}.{kind}")
            else:
                self._fidelity(value, f"compute_fidelities.{field}")
        # The current graph's public tensor contract is BF16. Reject unsupported
        # activation policies instead of silently accepting ignored JSON fields.
        for field in ("activations", "residual", "norm"):
            self._dtype(
                self.activation_residual.get(field),
                f"activation_residual.{field}",
                allowed={"bf16"},
            )
        for field in (
            "linear_attention_attention",
            "linear_attention_mlp",
            "full_attention_attention",
            "full_attention_mlp",
        ):
            self._dtype(self.ccl.get(field), f"ccl.{field}", allowed={"bf16", "bfp8"})
        self._dtype(self.kv_cache.get("dtype"), "kv_cache.dtype", allowed={"bf16", "bfp8"})
        if self.kv_cache.get("layout") != "tile_dram_paged":
            raise ValueError("kv_cache.layout must be 'tile_dram_paged'")
        if self.kv_cache.get("page_block_size") != 64:
            raise ValueError("kv_cache.page_block_size must be 64 for the measured paged-cache kernels")
        if self.logits_sampling.get("lm_head_weight_dtype") != self.weight_groups["lm_head"]:
            raise ValueError("logits_sampling.lm_head_weight_dtype must match weight_groups.lm_head")
        self._dtype(self.logits_sampling.get("logits_dtype"), "logits_sampling.logits_dtype", allowed={"bf16"})
        self._dtype(self.logits_sampling.get("sampling_dtype"), "logits_sampling.sampling_dtype", allowed={"bf16"})
        if self.logits_sampling.get("greedy_accumulation") != "local_bf16_max_argmax":
            raise ValueError("unsupported logits_sampling.greedy_accumulation")
        if self.logits_sampling.get("stochastic_assumption") != "bf16_logits_with_fp32_probability_math_where_required":
            raise ValueError("unsupported logits_sampling.stochastic_assumption")
        allowed_overrides = {
            "attention_input",
            "attention_output",
            "mlp_gate",
            "mlp_up",
            "mlp_down",
            "projection_fidelity",
            "mlp_gate_fidelity",
            "mlp_up_fidelity",
            "mlp_down_fidelity",
        }
        for layer, overrides in self.layer_exceptions.items():
            try:
                layer_idx = int(layer)
            except ValueError as error:
                raise ValueError(f"layer_exceptions key must be an integer, got {layer!r}") from error
            if not 0 <= layer_idx < 64:
                raise ValueError(f"layer exception {layer_idx} is outside [0, 63]")
            overrides = _require_mapping(overrides, f"layer_exceptions.{layer}")
            unknown = set(overrides) - allowed_overrides
            if unknown:
                raise ValueError(f"unsupported layer exception fields for layer {layer_idx}: {sorted(unknown)}")
            for field, value in overrides.items():
                if field.endswith("_fidelity"):
                    self._fidelity(value, f"layer_exceptions.{layer}.{field}")
                else:
                    self._dtype(value, f"layer_exceptions.{layer}.{field}")
        if self.runtime_loading:
            _require_exact_keys(
                self.runtime_loading,
                {
                    "default_path",
                    "candidate_override_env",
                    "full_model_constructor_kwarg",
                    "default_consumers",
                    "vllm_contract",
                },
                "runtime_loading",
            )

    def _layer_override(self, layer_idx: int, field: str) -> Any | None:
        return self.layer_exceptions.get(str(layer_idx), {}).get(field)

    def weight_dtype(self, field: str, *, layer_kind: str | None = None, layer_idx: int | None = None) -> str:
        if layer_idx is not None:
            override = self._layer_override(layer_idx, field)
            if override is not None:
                return self._dtype(override, f"layer_exceptions.{layer_idx}.{field}")
        value = self.weight_groups[field]
        if isinstance(value, dict):
            if layer_kind is None:
                raise ValueError(f"{field} requires layer_kind")
            value = value[layer_kind]
        return self._dtype(value, f"weight_groups.{field}")

    def fidelity(self, field: str, *, layer_kind: str | None = None, layer_idx: int | None = None) -> str:
        override_field = f"{field}_fidelity"
        if layer_idx is not None:
            override = self._layer_override(layer_idx, override_field)
            if override is not None:
                return self._fidelity(override, f"layer_exceptions.{layer_idx}.{override_field}")
        value = self.compute_fidelities[field]
        if isinstance(value, dict):
            if layer_kind is None:
                raise ValueError(f"{field} fidelity requires layer_kind")
            value = value[layer_kind]
        return self._fidelity(value, f"compute_fidelities.{field}")

    def summary(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "path": str(self.path),
            "weight_groups": self.weight_groups,
            "layer_exceptions": self.layer_exceptions,
            "compute_fidelities": self.compute_fidelities,
            "activation_residual": self.activation_residual,
            "ccl": self.ccl,
            "kv_cache": self.kv_cache,
            "logits_sampling": self.logits_sampling,
            "runtime_loading": self.runtime_loading,
        }


def load_precision_policy(path: str | Path | None = None) -> PrecisionPolicy:
    configured = path or os.environ.get(PRECISION_CONFIG_ENV) or DEFAULT_PRECISION_CONFIG
    resolved = Path(configured).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Qwen3.6 precision config not found: {resolved}; set {PRECISION_CONFIG_ENV} for a candidate"
        )
    return PrecisionPolicy(_load_config_data(resolved), resolved)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_data(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"cyclic precision base_config chain: {(*seen, path)}")
    data = json.loads(path.read_text())
    base_config = data.pop("base_config", None)
    if base_config is None:
        return data
    base_path = (path.parent / base_config).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"precision base_config not found: {base_path}")
    return _deep_merge(_load_config_data(base_path, (*seen, path)), data)


__all__ = [
    "DEFAULT_PRECISION_CONFIG",
    "PRECISION_CONFIG_ENV",
    "PrecisionPolicy",
    "load_precision_policy",
]
