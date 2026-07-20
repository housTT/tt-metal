# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Selected precision policy loader for Kokoro-82M's plbert full model.

The datatype-sweep stage (stage 07) selects a weight/activation/CCL/compute-
fidelity policy and records it in ``doc/datatype_sweep/selected_precision_config.json``.
This module makes that JSON the **consumed source of truth**: ``build_generator``
(and therefore ``run_prefill_check`` / ``run_teacher_forcing`` and the future vLLM
adapter) constructs the model from this file by default, so the served/measured
runtime path uses exactly the swept-and-selected policy rather than whatever the
dataclass defaults happen to be.

The JSON's ``construct.policy`` / ``construct.opt`` objects are the exact
constructor kwargs for :class:`PrecisionPolicy` / :class:`OptConfig`. The rest of
the JSON is human-facing documentation of the full policy (weight groups, layer
exceptions, fidelities, activation/residual/CCL/KV-cache/logits dtypes).

Safe-baseline override: passing ``policy=`` / ``opt=`` explicitly to
``build_generator`` bypasses the file, and deleting the file falls back to the
dataclass defaults (which equal the selected baseline), so returning to a known
config is a one-line change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from models.demos.audio.kokoro.tt.optimized_decoder import PrecisionPolicy
from models.demos.audio.kokoro.tt.optimized_multichip_decoder import OptConfig

SELECTED_CONFIG_RELPATH = "doc/datatype_sweep/selected_precision_config.json"


def selected_config_path(model_dir=None) -> Path:
    base = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parents[1]
    return base / SELECTED_CONFIG_RELPATH


def load_selected(model_dir=None) -> Tuple[PrecisionPolicy, OptConfig, Optional[dict]]:
    """Return (PrecisionPolicy, OptConfig, raw_json) from the selected config.

    Falls back to dataclass defaults (== selected baseline) when the file is
    absent, so the model is always constructible.
    """
    path = selected_config_path(model_dir)
    if not path.exists():
        return PrecisionPolicy(), OptConfig(), None
    raw = json.loads(path.read_text())
    construct = raw.get("construct", {})
    policy = PrecisionPolicy(**construct.get("policy", {}))
    opt = OptConfig(**construct.get("opt", {}))
    return policy, opt, raw
