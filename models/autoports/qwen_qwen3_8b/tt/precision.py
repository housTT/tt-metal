# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Precision config artifact: one JSON that names every dtype and compute-fidelity choice of the Qwen3-8B runtime.

``build_generator`` reads ``doc/datatype_sweep/selected_precision_config.json`` by default
(``QWEN3_PRECISION_CONFIG=<path>`` selects another file) and constructs the decoder policies, the
terminal policy and the KV-cache dtype from it. ``describe_runtime`` reads the dtypes and fidelities
back from the live device tensors and compute-kernel configs, which is the propagation evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import ttnn
from models.autoports.qwen_qwen3_8b.tt.model import DEFAULT_TERMINAL_POLICY, TerminalPolicy
from models.autoports.qwen_qwen3_8b.tt.optimized_decoder import (
    DEFAULT_POLICY,
    DTYPE_NAMES,
    FIDELITY_NAMES,
    DecodeMatmulRole,
    OptimizedPolicy,
)

MODEL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = MODEL_DIR / "doc" / "datatype_sweep" / "selected_precision_config.json"
ENV_CONFIG_PATH = "QWEN3_PRECISION_CONFIG"
WEIGHT_GROUPS = ("qkv", "o_proj", "gate_proj", "up_proj", "down_proj")
ROLE_FIELDS = ("dtype", "fidelity", "fp32_acc", "readers", "in0_block_w")
DTYPE_BY_TTNN = {value: key for key, value in DTYPE_NAMES.items()}
FIDELITY_BY_TTNN = {value: key for key, value in FIDELITY_NAMES.items()}


def _role_dict(role: DecodeMatmulRole) -> dict:
    return {
        "dtype": role.dtype,
        "fidelity": role.fidelity,
        "fp32_acc": role.fp32_acc,
        "readers": role.readers,
        "in0_block_w": role.in0_block_w,
    }


def baseline_config() -> dict:
    """The stage-7 policy as a precision config (the safe baseline every sweep candidate starts from)."""
    policy = DEFAULT_POLICY
    terminal = DEFAULT_TERMINAL_POLICY
    return {
        "config_id": "c0_stage7_baseline",
        "description": "stage-7 shipped policy: BFP8 LoFi QKV/O/gate/down, BFP4 LoFi up, BF16 KV cache, LoFi BFP8 LM head",
        "weight_groups": {
            "qkv": _role_dict(policy.qkv),
            "o_proj": _role_dict(policy.o_proj),
            "gate_proj": _role_dict(policy.gate_up),
            "up_proj": _role_dict(policy.up_role),
            "down_proj": _role_dict(policy.down),
        },
        "layer_exceptions": {},
        "norm": {"dtype": "bf16", "fidelity": policy.norm_fidelity, "fp32_acc": True},
        "sdpa": {"fidelity": policy.sdpa_fidelity, "fp32_acc": True},
        "prefill": {"fp32_acc": policy.prefill_fp32_acc, "bfp8_fidelity": policy.prefill_bfp8_fidelity},
        "activation_dtype": "bf16",
        "residual_dtype": "bf16",
        "ccl_dtype": None,
        "ccl_note": "one chip (stages 4 and 5 skipped by the operator): no collectives, no CCL activations",
        "kv_cache_dtype": "bf16",
        "embedding_dtype": terminal.embedding_dtype,
        "lm_head": {
            "dtype": terminal.lm_head_dtype,
            "decode_fidelity": terminal.lm_head_decode_fidelity,
            "decode_fp32_acc": False,
            "prefill_fidelity": terminal.lm_head_prefill_fidelity,
            "prefill_fp32_acc": terminal.lm_head_prefill_fp32_acc,
            "readers": terminal.lm_head_readers,
            "in0_block_w": terminal.lm_head_in0_block_w,
        },
        "final_norm": {"dtype": terminal.final_norm_dtype, "fidelity": terminal.final_norm_fidelity, "fp32_acc": True},
        "logits": {
            "decode_dtype": "bf16",
            "decode_note": "LM head decode output [1, 1, 32, padded_vocab] BF16 with the padded columns at bf16 min",
            "prefill_dtype": terminal.prefill_logits_dtype,
        },
        "sampling": {
            "dtype": "bf16",
            "path": "models/common/sampling SamplingGenerator over the BF16 logits tile (top-k path, k=1 for greedy)",
        },
    }


def resolve_config_path(path: str | Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    env = os.environ.get(ENV_CONFIG_PATH)
    if env:
        return Path(env)
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return None


def load_precision_config(path: str | Path | None = None) -> tuple[dict, Path | None]:
    """``(config, path)``; ``path`` is ``None`` when the built-in baseline is used because no file exists."""
    resolved = resolve_config_path(path)
    if resolved is None:
        return baseline_config(), None
    config = json.loads(Path(resolved).read_text())
    missing = [key for key in ("config_id", "weight_groups", "kv_cache_dtype", "lm_head") if key not in config]
    if missing:
        raise ValueError(f"precision config {resolved} lacks {missing}")
    return config, Path(resolved)


def _role(spec: dict, base: DecodeMatmulRole) -> DecodeMatmulRole:
    unknown = set(spec) - set(ROLE_FIELDS)
    if unknown:
        raise ValueError(f"unknown weight-group fields {sorted(unknown)}")
    if "dtype" in spec and spec["dtype"] not in DTYPE_NAMES:
        raise ValueError(f"unknown dtype {spec['dtype']}")
    if "fidelity" in spec and spec["fidelity"] not in FIDELITY_NAMES:
        raise ValueError(f"unknown fidelity {spec['fidelity']}")
    return replace(base, **spec)


def decoder_policy(config: dict, layer_idx: int | None = None) -> OptimizedPolicy:
    """The ``OptimizedPolicy`` for one layer: the weight groups with that layer's exceptions applied."""
    groups = {name: dict(config["weight_groups"].get(name, {})) for name in WEIGHT_GROUPS}
    if layer_idx is not None:
        for name, spec in config.get("layer_exceptions", {}).get(str(layer_idx), {}).items():
            if name not in WEIGHT_GROUPS:
                raise ValueError(f"layer {layer_idx} exception names unknown group {name}")
            groups[name].update(spec)
    base = DEFAULT_POLICY
    norm = config.get("norm", {})
    sdpa = config.get("sdpa", {})
    prefill = config.get("prefill", {})
    return replace(
        base,
        qkv=_role(groups["qkv"], base.qkv),
        o_proj=_role(groups["o_proj"], base.o_proj),
        gate_up=_role(groups["gate_proj"], base.gate_up),
        up=_role(groups["up_proj"], base.up_role),
        down=_role(groups["down_proj"], base.down),
        norm_fidelity=norm.get("fidelity", base.norm_fidelity),
        sdpa_fidelity=sdpa.get("fidelity", base.sdpa_fidelity),
        prefill_fp32_acc=prefill.get("fp32_acc", base.prefill_fp32_acc),
        prefill_bfp8_fidelity=prefill.get("bfp8_fidelity", base.prefill_bfp8_fidelity),
    )


def layer_policies(config: dict, num_layers: int) -> dict[int, OptimizedPolicy]:
    """Per-layer policies for the layers named in ``layer_exceptions``."""
    out = {}
    for key in config.get("layer_exceptions", {}):
        idx = int(key)
        if 0 <= idx < num_layers:
            out[idx] = decoder_policy(config, idx)
    return out


def terminal_policy(config: dict) -> TerminalPolicy:
    lm = config["lm_head"]
    final_norm = config.get("final_norm", {})
    logits = config.get("logits", {})
    base = DEFAULT_TERMINAL_POLICY
    return replace(
        base,
        embedding_dtype=config.get("embedding_dtype", base.embedding_dtype),
        final_norm_dtype=final_norm.get("dtype", base.final_norm_dtype),
        final_norm_fidelity=final_norm.get("fidelity", base.final_norm_fidelity),
        lm_head_dtype=lm.get("dtype", base.lm_head_dtype),
        lm_head_decode_fidelity=lm.get("decode_fidelity", base.lm_head_decode_fidelity),
        lm_head_prefill_fidelity=lm.get("prefill_fidelity", base.lm_head_prefill_fidelity),
        lm_head_prefill_fp32_acc=lm.get("prefill_fp32_acc", base.lm_head_prefill_fp32_acc),
        prefill_logits_dtype=logits.get("prefill_dtype", base.prefill_logits_dtype),
        lm_head_in0_block_w=lm.get("in0_block_w", base.lm_head_in0_block_w),
        lm_head_readers=lm.get("readers", base.lm_head_readers),
    )


def kv_cache_dtype(config: dict) -> ttnn.DataType:
    name = config["kv_cache_dtype"]
    if name not in DTYPE_NAMES:
        raise ValueError(f"unknown kv_cache_dtype {name}")
    return DTYPE_NAMES[name]


def activation_dtype(config: dict) -> ttnn.DataType:
    name = config.get("activation_dtype", "bf16")
    if name != config.get("residual_dtype", "bf16"):
        raise ValueError("activation_dtype and residual_dtype must match (one residual layout)")
    if name != "bf16":
        raise ValueError("only BF16 activations are supported by the sharded decode path")
    return DTYPE_NAMES[name]


def _compute_fidelity(config: Any) -> dict:
    return {
        "fidelity": FIDELITY_BY_TTNN.get(config.math_fidelity, str(config.math_fidelity)),
        "fp32_acc": bool(config.fp32_dest_acc_en),
        "packer_l1_acc": bool(config.packer_l1_acc),
    }


def describe_runtime(generator: Any) -> dict:
    """Dtypes and fidelities read back from the live model: weight tensors, compute configs, caches."""
    model = generator.model
    layers = []
    for layer in model.layers:
        w = layer.weights
        layers.append(
            {
                "layer_idx": layer.layer_idx,
                "weight_dtypes": {
                    "qkv": DTYPE_BY_TTNN[w.qkv.dtype],
                    "o_proj": DTYPE_BY_TTNN[w.o_proj.dtype],
                    "gate_proj": DTYPE_BY_TTNN[w.gate_proj.dtype],
                    "up_proj": DTYPE_BY_TTNN[w.up_proj.dtype],
                    "down_proj": DTYPE_BY_TTNN[w.down_proj.dtype],
                },
                "decode_compute": {name: _compute_fidelity(cfg) for name, cfg in layer.decode_compute.items()},
                "decode_readers": {
                    name: layer.decode_program_config[name].num_workers_per_dram_bank
                    for name in layer.decode_program_config
                },
                "prefill_compute": {
                    layer._prefill_role[key]: _compute_fidelity(layer._prefill_compute[key])
                    for key in layer._prefill_role
                },
                "norm_compute": _compute_fidelity(layer.norm_compute_config),
                "sdpa_compute": _compute_fidelity(layer.sdpa_compute_config),
                "weight_bytes_per_token": layer.weight_bytes_per_token(),
            }
        )
    key_cache, value_cache = generator.kv_cache[0]
    return {
        "config_id": generator.precision_config.get("config_id"),
        "config_path": str(generator.precision_config_path) if generator.precision_config_path else None,
        "layers": layers,
        "distinct_layer_policies": sorted(
            {json.dumps({k: v for k, v in d.items() if k != "layer_idx"}, sort_keys=True) for d in layers}
        ),
        "kv_cache_dtype": DTYPE_BY_TTNN[key_cache.dtype],
        "kv_value_cache_dtype": DTYPE_BY_TTNN[value_cache.dtype],
        "kv_cache_layers": len(generator.kv_cache),
        "embedding_dtype": DTYPE_BY_TTNN[model.embedding.dtype],
        "final_norm_dtype": DTYPE_BY_TTNN[model.final_norm.dtype],
        "final_norm_compute": _compute_fidelity(model.norm_compute_config),
        "lm_head_dtype": sorted({DTYPE_BY_TTNN[w.dtype] for w in model.lm_head_weights}),
        "lm_head_decode_compute": _compute_fidelity(model.lm_head_decode_compute),
        "lm_head_prefill_compute": _compute_fidelity(model.lm_head_prefill_compute),
        "lm_head_readers": model.lm_head_program_configs[0].num_workers_per_dram_bank,
        "prefill_logits_dtype": model.terminal_policy.prefill_logits_dtype,
        "decode_logits_dtype": "bf16",
        "sampling_vocab": generator.sampling.args.vocab_size if hasattr(generator.sampling, "args") else None,
        "activation_dtype": DTYPE_BY_TTNN[model.layers[0].activation_dtype],
        "ccl_dtype": None,
    }


def expected_runtime(config: dict, num_layers: int) -> dict:
    """What ``describe_runtime`` must report for ``config``; used by the propagation check."""
    expected_layers = []
    for idx in range(num_layers):
        policy = decoder_policy(config, idx)
        expected_layers.append(
            {
                "layer_idx": idx,
                "weight_dtypes": {
                    "qkv": policy.qkv.dtype,
                    "o_proj": policy.o_proj.dtype,
                    "gate_proj": policy.gate_up.dtype,
                    "up_proj": policy.up_role.dtype,
                    "down_proj": policy.down.dtype,
                },
                "decode_fidelity": {
                    "qkv": policy.qkv.fidelity,
                    "o_proj": policy.o_proj.fidelity,
                    "gate_up": policy.gate_up.fidelity,
                    "up": policy.up_role.fidelity,
                    "down": policy.down.fidelity,
                },
                "decode_readers": {
                    "qkv": policy.qkv.readers,
                    "o_proj": policy.o_proj.readers,
                    "gate_up": policy.gate_up.readers,
                    "up": policy.up_role.readers,
                    "down": policy.down.readers,
                },
            }
        )
    terminal = terminal_policy(config)
    return {
        "layers": expected_layers,
        "kv_cache_dtype": config["kv_cache_dtype"],
        "lm_head_dtype": terminal.lm_head_dtype,
        "lm_head_decode_fidelity": terminal.lm_head_decode_fidelity,
        "lm_head_prefill_fidelity": terminal.lm_head_prefill_fidelity,
        "prefill_logits_dtype": terminal.prefill_logits_dtype,
    }


def check_propagation(generator: Any) -> list[str]:
    """Mismatches between the config the generator was built from and the live runtime; empty when consumed."""
    runtime = describe_runtime(generator)
    expected = expected_runtime(generator.precision_config, generator.model.num_layers)
    problems = []
    for exp, got in zip(expected["layers"], runtime["layers"]):
        if exp["weight_dtypes"] != got["weight_dtypes"]:
            problems.append(f"layer {exp['layer_idx']} weight dtypes {got['weight_dtypes']} != {exp['weight_dtypes']}")
        got_fid = {name: got["decode_compute"][name]["fidelity"] for name in exp["decode_fidelity"]}
        if exp["decode_fidelity"] != got_fid:
            problems.append(f"layer {exp['layer_idx']} decode fidelity {got_fid} != {exp['decode_fidelity']}")
        if exp["decode_readers"] != got["decode_readers"]:
            problems.append(f"layer {exp['layer_idx']} readers {got['decode_readers']} != {exp['decode_readers']}")
    for key in ("kv_cache_dtype", "prefill_logits_dtype"):
        if runtime[key] != expected[key]:
            problems.append(f"{key} {runtime[key]} != {expected[key]}")
    if runtime["lm_head_dtype"] != [expected["lm_head_dtype"]]:
        problems.append(f"lm_head_dtype {runtime['lm_head_dtype']} != {expected['lm_head_dtype']}")
    if runtime["lm_head_decode_compute"]["fidelity"] != expected["lm_head_decode_fidelity"]:
        problems.append("lm_head decode fidelity mismatch")
    if runtime["lm_head_prefill_compute"]["fidelity"] != expected["lm_head_prefill_fidelity"]:
        problems.append("lm_head prefill fidelity mismatch")
    return problems


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ENV_CONFIG_PATH",
    "activation_dtype",
    "baseline_config",
    "check_propagation",
    "decoder_policy",
    "describe_runtime",
    "expected_runtime",
    "kv_cache_dtype",
    "layer_policies",
    "load_precision_config",
    "terminal_policy",
]
