# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit YaRN configuration and evidence-gated serving context selection."""

import copy
import math


def configure_context(text_config, max_context, rope_parameters=None):
    """Copy the pinned text config; alter only context and explicit RoPE settings.

    This enables capacity/accuracy probes, not a claim that a device can fit the
    requested context. Serving separately requires a validated profile entry.
    """
    native = int(text_config.max_position_embeddings)
    if isinstance(max_context, bool) or not isinstance(max_context, int) or max_context < 1:
        raise ValueError("max_context must be a positive integer")
    result = copy.deepcopy(text_config)
    rope = dict(result.rope_parameters)
    if rope_parameters is not None:
        for key in ("rope_theta", "partial_rotary_factor", "mrope_section", "mrope_interleaved"):
            if key in rope_parameters and rope_parameters[key] != rope.get(key):
                raise ValueError(f"YaRN must preserve the checkpoint's {key}")
        rope.update(rope_parameters)
    kind = rope.get("rope_type", "default")
    if kind == "default":
        if max_context > native:
            raise ValueError("Context beyond the native window requires explicit YaRN")
    elif kind == "yarn":
        factor = rope.get("factor")
        original = rope.get("original_max_position_embeddings")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 1:
            raise ValueError("YaRN factor must be finite and greater than one")
        if original != native or isinstance(original, bool):
            raise ValueError("YaRN original_max_position_embeddings must match the pinned native context")
        if max_context > math.floor(native * factor):
            raise ValueError("Requested context exceeds the explicit YaRN factor")
        result.max_position_embeddings = max(native, max_context)
    else:
        raise ValueError(f"Unsupported rope_type {kind!r}")
    result.rope_parameters = rope
    return result


def serving_rope_parameters(hf_config):
    """Read legacy top-level or current text-level Transformers fields.

    Legacy rope_scaling remains on the top-level Qwen3.5 config in some tested
    vLLM/Transformers combinations. Do not silently discard that explicit opt-in.
    """
    text = getattr(hf_config, "text_config", hf_config)
    candidates = []
    for config in (hf_config, text):
        for name in ("rope_scaling", "rope_parameters"):
            value = getattr(config, name, None)
            if value and value.get("rope_type", value.get("type", "default")) != "default":
                value = dict(value)
                if "type" in value and "rope_type" in value and value["type"] != value["rope_type"]:
                    raise ValueError("Conflicting legacy and current rope_type")
                value["rope_type"] = value.pop("type", value.get("rope_type"))
                candidates.append(value)
    if not candidates:
        return None
    result = {}
    for candidate in candidates:
        for key, value in candidate.items():
            if key in result and result[key] != value:
                raise ValueError(f"Conflicting serving RoPE field {key!r}")
            result[key] = value
    return result


def validated_serving_context(hf_config, contract, requested_context, num_devices):
    """Return allocated context and override only for a proven serving profile."""
    if num_devices not in (1, 2, 4):
        raise ValueError("Expected a one-, two-, or four-chip profile")
    rope = serving_rope_parameters(hf_config)
    if rope is None:
        context = int(contract["current_supported_context"])
    else:
        profile = "p150" if num_devices == 1 else f"p150x{num_devices}"
        entry = contract.get("optional_yarn", {}).get("profiles", {}).get(profile, {})
        if entry.get("status") != "pass":
            raise ValueError(f"YaRN lacks validated capacity and accuracy evidence for {profile}")
        proven_rope = entry.get("rope_parameters", {})
        # Every caller override must match the evidence, including optional YaRN
        # beta/attention factors and the checkpoint's partial rotary geometry.
        if any(proven_rope.get(key) != value for key, value in rope.items()):
            raise ValueError("Requested YaRN configuration differs from the validated profile")
        context = int(entry["max_context"])
    if requested_context is not None and not 1 <= requested_context <= context:
        raise ValueError(f"Requested context exceeds the validated {context}-token contract")
    return context, rope
