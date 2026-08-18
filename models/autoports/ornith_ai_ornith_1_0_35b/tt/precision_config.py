# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The precision policy as a **file**, and the resolution rule the model builds from.

``$datatype-sweep`` requires the selected precision configuration to be an artifact that the real
construction path consumes, not a document beside it. So:

* :data:`SELECTED_PRECISION_CONFIG_PATH` -
  ``doc/datatype_sweep/selected_precision_config.json`` - is the model's **default** precision
  policy. ``OrnithModel.from_pretrained`` (and therefore ``build_generator``, every readiness
  runner, the benchmark harness and any vLLM adapter that goes through them) resolves ``policy=None``
  through :func:`resolve_policy`, which loads that file. There is no second copy of the selected
  values in code: delete a field from the JSON and the build raises.
* A caller who wants a different policy passes ``policy=`` a registered name
  (``"optimized"``, ``"fused-parity"``, ``"bfp4-projections"``, ``"bfp4-projections-lofi"``), a path
  to another JSON file, a plain dict in the same schema, or a ``PrecisionPolicy`` object. The
  environment variable ``ORNITH_PRECISION_POLICY`` does the same thing without touching a call site,
  which is the "simple change back to the safe baseline" the skill asks for:

      ORNITH_PRECISION_POLICY=optimized python ... # the pre-sweep decoder-stage policy
      ORNITH_PRECISION_POLICY=fused-parity python ... # the bfloat16 correctness floor

The JSON schema is grouped the way the skill asks for it - weight groups, layer exceptions, compute
fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, logits/sampling dtype - and
:func:`policy_from_dict` / :func:`policy_to_dict` are exact inverses over every
:class:`PrecisionPolicy` field. :func:`policy_to_dict` is what writes the artifact, so a field added
to the dataclass without a schema entry fails loudly rather than being silently dropped.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY, POLICIES, PrecisionPolicy

MODEL_ROOT = Path(__file__).resolve().parent.parent

#: The selected precision configuration. This path is the model's default policy.
SELECTED_PRECISION_CONFIG_PATH = MODEL_ROOT / "doc" / "datatype_sweep" / "selected_precision_config.json"

#: Environment override, read at build time. A registered policy name or a path to a JSON config.
POLICY_ENV_VAR = "ORNITH_PRECISION_POLICY"

DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
    "uint32": ttnn.uint32,
    "int32": ttnn.int32,
}
DTYPE_NAMES = {v: k for k, v in DTYPES.items()}

FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
FIDELITY_NAMES = {v: k for k, v in FIDELITIES.items()}


def _dtype_name(value):
    if value is None:
        return None
    if value not in DTYPE_NAMES:
        raise ValueError(f"unserializable dtype {value!r}; known: {sorted(DTYPES)}")
    return DTYPE_NAMES[value]


def _dtype(value):
    if value is None:
        return None
    if value not in DTYPES:
        raise ValueError(f"unknown dtype {value!r}; known: {sorted(DTYPES)}")
    return DTYPES[value]


def _fidelity_name(value):
    if value is None:
        return None
    if value not in FIDELITY_NAMES:
        raise ValueError(f"unserializable math fidelity {value!r}; known: {sorted(FIDELITIES)}")
    return FIDELITY_NAMES[value]


def _fidelity(value):
    if value is None:
        return None
    if value not in FIDELITIES:
        raise ValueError(f"unknown math fidelity {value!r}; known: {sorted(FIDELITIES)}")
    return FIDELITIES[value]


#: How each :class:`PrecisionPolicy` field is spelled in the JSON: ``(section, key, kind)``.
#: ``kind`` is ``"dtype"``, ``"fidelity"`` or ``"plain"``. Every field of the dataclass must appear
#: here exactly once - :func:`_check_schema_covers_the_dataclass` asserts it at import time, so a
#: policy field added later cannot be silently dropped from the artifact.
SCHEMA: dict[str, tuple[str, str, str]] = {
    "name": ("", "config_id", "plain"),
    # ---- weight groups ----
    "expert_gate_up_dtype": ("weight_groups", "routed_expert_gate_up", "dtype"),
    "expert_down_dtype": ("weight_groups", "routed_expert_down", "dtype"),
    "proj_dtype": ("weight_groups", "dense_projections", "dtype"),
    "shared_dtype": ("weight_groups", "shared_expert", "dtype"),
    "router_dtype": ("weight_groups", "router", "dtype"),
    "lm_head_dtype": ("weight_groups", "lm_head", "dtype"),
    # ---- compute fidelities ----
    "expert_fidelity": ("compute_fidelities", "routed_experts", "fidelity"),
    "expert_fp32_acc": ("compute_fidelities", "routed_experts_fp32_dest_acc", "plain"),
    "expert_packer_l1_acc": ("compute_fidelities", "routed_experts_packer_l1_acc", "plain"),
    "proj_fidelity": ("compute_fidelities", "dense_projections", "fidelity"),
    "proj_fp32_acc": ("compute_fidelities", "dense_projections_fp32_dest_acc", "plain"),
    "proj_packer_l1_acc": ("compute_fidelities", "dense_projections_packer_l1_acc", "plain"),
    "shared_fidelity": ("compute_fidelities", "shared_expert", "fidelity"),
    "shared_fp32_acc": ("compute_fidelities", "shared_expert_fp32_dest_acc", "plain"),
    "shared_packer_l1_acc": ("compute_fidelities", "shared_expert_packer_l1_acc", "plain"),
    "router_fidelity": ("compute_fidelities", "router", "fidelity"),
    "router_fp32_acc": ("compute_fidelities", "router_fp32_dest_acc", "plain"),
    "state_fidelity": ("compute_fidelities", "deltanet_state", "fidelity"),
    "state_fp32_acc": ("compute_fidelities", "deltanet_state_fp32_dest_acc", "plain"),
    "sdpa_fidelity": ("compute_fidelities", "sdpa", "fidelity"),
    "sdpa_fp32_acc": ("compute_fidelities", "sdpa_fp32_dest_acc", "plain"),
    "lm_head_fidelity": ("compute_fidelities", "lm_head", "fidelity"),
    # ---- activations ----
    "expert_act_dtype": ("activations", "routed_expert_output", "dtype"),
    "residual_dtype": ("activations", "residual_stream", "dtype"),
    # ---- collectives ----
    "ccl_dtype": ("ccl", "payload_dtype", "dtype"),
    # ---- KV cache ----
    "kv_cache_dtype": ("kv_cache", "dtype", "dtype"),
    # ---- logits / sampling ----
    "logits_dtype": ("logits_sampling", "logits_dtype", "dtype"),
    # ---- layer exceptions ----
    "layer_exceptions": ("layer_exceptions", "", "plain"),
    # ---- resolved program legality ----
    "prefill_sdpa_chunk": ("prefill", "sdpa_q_k_chunk", "plain"),
}


def _check_schema_covers_the_dataclass() -> None:
    declared = {f.name for f in fields(PrecisionPolicy)}
    missing = declared - set(SCHEMA)
    extra = set(SCHEMA) - declared
    if missing or extra:
        raise RuntimeError(
            f"precision-config SCHEMA is out of step with PrecisionPolicy: missing {sorted(missing)}, "
            f"unknown {sorted(extra)}. Every policy field must round-trip through the selected-config "
            "artifact, or a selected value could be recorded and never consumed."
        )


_check_schema_covers_the_dataclass()


def policy_to_dict(policy: PrecisionPolicy, **extra) -> dict:
    """Serialize a policy to the selected-config schema. ``extra`` adds top-level provenance keys."""
    out: dict = {"schema": "ornith-precision-config/1"}
    out.update(extra)
    for field_name, (section, key, kind) in SCHEMA.items():
        value = getattr(policy, field_name)
        if kind == "dtype":
            value = _dtype_name(value)
        elif kind == "fidelity":
            value = _fidelity_name(value)
        if field_name == "layer_exceptions":
            out.setdefault("layer_exceptions", [])
            for layer_idx, exc_field, exc_value in value:
                exc_section, exc_key, exc_kind = SCHEMA[exc_field]
                encoded = exc_value
                if exc_kind == "dtype":
                    encoded = _dtype_name(exc_value)
                elif exc_kind == "fidelity":
                    encoded = _fidelity_name(exc_value)
                out["layer_exceptions"].append(
                    {
                        "layer": int(layer_idx),
                        "field": exc_field,
                        "section": exc_section,
                        "key": exc_key,
                        "value": encoded,
                    }
                )
            continue
        if not section:
            out[key] = value
        else:
            out.setdefault(section, {})[key] = value
    return out


def policy_from_dict(payload: dict) -> PrecisionPolicy:
    """Rebuild a policy from the selected-config schema. Inverse of :func:`policy_to_dict`."""
    kwargs: dict = {}
    for field_name, (section, key, kind) in SCHEMA.items():
        if field_name == "layer_exceptions":
            # As strict as every other section: an omitted key is a partial file, not "no
            # exceptions". `null` and `[]` both mean "none", explicitly.
            if "layer_exceptions" not in payload:
                raise ValueError(
                    "precision config is missing layer_exceptions (policy field 'layer_exceptions'). "
                    "Write [] for none; a missing key would silently drop a selected layer exception."
                )
            exceptions = []
            for item in payload["layer_exceptions"] or []:
                exc_field = item["field"]
                _, _, exc_kind = SCHEMA[exc_field]
                value = item["value"]
                if exc_kind == "dtype":
                    value = _dtype(value)
                elif exc_kind == "fidelity":
                    value = _fidelity(value)
                exceptions.append((int(item["layer"]), exc_field, value))
            kwargs["layer_exceptions"] = tuple(exceptions)
            continue
        container = payload if not section else payload.get(section)
        if container is None or key not in container:
            raise ValueError(
                f"precision config is missing {section + '.' if section else ''}{key} "
                f"(policy field {field_name!r}). The selected config must be complete: a partial file "
                "would silently take a dataclass default that no measurement selected."
            )
        value = container[key]
        if kind == "dtype":
            value = _dtype(value)
        elif kind == "fidelity":
            value = _fidelity(value)
        kwargs[field_name] = value
    return PrecisionPolicy(**kwargs)


def load_policy_file(path) -> PrecisionPolicy:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"precision config {path} not found. The datatype-sweep stage writes "
            f"{SELECTED_PRECISION_CONFIG_PATH.relative_to(MODEL_ROOT.parent.parent.parent)} and the model "
            "builds from it; pass policy='optimized' (or set ORNITH_PRECISION_POLICY=optimized) to build "
            "the pre-sweep decoder-stage policy instead."
        )
    return policy_from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_selected_policy() -> PrecisionPolicy:
    """The selected precision config, from the artifact. This is the model's default."""
    return load_policy_file(SELECTED_PRECISION_CONFIG_PATH)


def resolve_policy(policy=None) -> PrecisionPolicy:
    """Resolve a ``policy=`` argument to a :class:`PrecisionPolicy`.

    ``None``      the selected config artifact, unless ``ORNITH_PRECISION_POLICY`` is set.
    ``str``       a registered policy name, or a path to a JSON precision config.
    ``dict``      an inline config in the selected-config schema.
    ``Path``      a JSON precision config.
    """
    if policy is None:
        env = os.environ.get(POLICY_ENV_VAR)
        policy = env if env else SELECTED_PRECISION_CONFIG_PATH
    if isinstance(policy, PrecisionPolicy):
        return policy
    if isinstance(policy, dict):
        return policy_from_dict(policy)
    if isinstance(policy, Path):
        return load_policy_file(policy)
    if isinstance(policy, str):
        if policy in POLICIES:
            return POLICIES[policy]
        candidate = Path(policy)
        if candidate.suffix == ".json" or candidate.exists():
            return load_policy_file(candidate)
        raise ValueError(
            f"unknown precision policy {policy!r}; registered names: {sorted(POLICIES)}, or a path to a "
            "JSON precision config"
        )
    raise TypeError(f"cannot resolve precision policy from {type(policy).__name__}")


def write_policy_file(path, policy: PrecisionPolicy, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = policy_to_dict(policy, **extra)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "DTYPES",
    "FIDELITIES",
    "POLICY_ENV_VAR",
    "SELECTED_PRECISION_CONFIG_PATH",
    "DEFAULT_POLICY",
    "load_policy_file",
    "load_selected_policy",
    "policy_from_dict",
    "policy_to_dict",
    "resolve_policy",
    "write_policy_file",
]
