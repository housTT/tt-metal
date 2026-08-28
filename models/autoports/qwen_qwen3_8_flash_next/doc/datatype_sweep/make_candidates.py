# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generate the reproducible coarse datatype/fidelity candidate matrix."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE_SEED = ROOT / "baseline_precision_config.json"
BASE = json.loads(BASELINE_SEED.read_text())


def set_path(value, path, selected):
    target = value
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = selected


CANDIDATES = {
    "baseline_optimized_bfp4lofi_bfp8hifi2": {},
    "canonical_accuracy_bf16cache": {
        "weight_groups.routed_expert.compute_fidelity": "hifi2",
        "weight_groups.routed_expert.policy": "expert_bfp4_hifi2_g40b16_d40b5",
        "weight_groups.shared_projection.compute_fidelity": "hifi2",
        "weight_groups.shared_projection.policy": "bfp8_hifi2",
        "weight_groups.lm_head.dtype": "bf16",
        "weight_groups.lm_head.policy": "bf16_hifi2",
        "logits_sampling.logits_dtype": "bf16",
        "logits_sampling.sampling_dtype": "bf16",
        "kv_cache.policy": "bf16",
        "kv_cache.dtype": "bf16",
    },
    "expert_bfp4_hifi2": {
        "weight_groups.routed_expert.compute_fidelity": "hifi2",
        "weight_groups.routed_expert.policy": "expert_bfp4_hifi2_g40b16_d40b5",
    },
    "shared_bfp8_hifi2": {
        "weight_groups.shared_projection.compute_fidelity": "hifi2",
        "weight_groups.shared_projection.policy": "bfp8_hifi2",
    },
    "shared_bfp4_lofi": {
        "weight_groups.shared_projection.dtype": "bfp4",
        "weight_groups.shared_projection.compute_fidelity": "lofi",
        "weight_groups.shared_projection.policy": "bfp4_lofi",
    },
    "shared_bfp4_hifi2": {
        "weight_groups.shared_projection.dtype": "bfp4",
        "weight_groups.shared_projection.compute_fidelity": "hifi2",
        "weight_groups.shared_projection.policy": "bfp4_hifi2",
    },
    "gdn_bfp8_lofi": {
        "weight_groups.gdn_projection.compute_fidelity": "lofi",
        "weight_groups.gdn_projection.policy": "bfp8_lofi",
    },
    "qsa_bfp8_hifi2": {
        "weight_groups.qsa_input.dtype": "bfp8",
        "weight_groups.qsa_input.policy": "bfp8_hifi2",
        "weight_groups.attention_output.dtype": "bfp8",
        "weight_groups.attention_output.policy": "bfp8_hifi2",
    },
    "qsa_bfp8_lofi": {
        "weight_groups.qsa_input.dtype": "bfp8",
        "weight_groups.qsa_input.compute_fidelity": "lofi",
        "weight_groups.qsa_input.policy": "bfp8_lofi",
        "weight_groups.attention_output.dtype": "bfp8",
        "weight_groups.attention_output.compute_fidelity": "lofi",
        "weight_groups.attention_output.policy": "bfp8_lofi",
    },
    "lm_head_bfp8_lofi": {
        "weight_groups.lm_head.compute_fidelity": "lofi",
        "weight_groups.lm_head.policy": "bfp8_lofi",
    },
    "lm_head_bf16_hifi2": {
        "weight_groups.lm_head.dtype": "bf16",
        "weight_groups.lm_head.compute_fidelity": "hifi2",
        "weight_groups.lm_head.policy": "bf16_hifi2",
        "logits_sampling.logits_dtype": "bf16",
        "logits_sampling.sampling_dtype": "bf16",
    },
    "qsa_bfp8_hifi2_lm_head_bf16_hifi2": {
        "weight_groups.qsa_input.dtype": "bfp8",
        "weight_groups.qsa_input.policy": "bfp8_hifi2",
        "weight_groups.attention_output.dtype": "bfp8",
        "weight_groups.attention_output.policy": "bfp8_hifi2",
        "weight_groups.lm_head.dtype": "bf16",
        "weight_groups.lm_head.compute_fidelity": "hifi2",
        "weight_groups.lm_head.policy": "bf16_hifi2",
        "logits_sampling.logits_dtype": "bf16",
        "logits_sampling.sampling_dtype": "bf16",
    },
    "qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2": {
        "weight_groups.shared_projection.compute_fidelity": "hifi2",
        "weight_groups.shared_projection.policy": "bfp8_hifi2",
        "weight_groups.qsa_input.dtype": "bfp8",
        "weight_groups.qsa_input.policy": "bfp8_hifi2",
        "weight_groups.attention_output.dtype": "bfp8",
        "weight_groups.attention_output.policy": "bfp8_hifi2",
        "weight_groups.lm_head.dtype": "bf16",
        "weight_groups.lm_head.compute_fidelity": "hifi2",
        "weight_groups.lm_head.policy": "bf16_hifi2",
        "logits_sampling.logits_dtype": "bf16",
        "logits_sampling.sampling_dtype": "bf16",
    },
    "kv_bf16_control": {
        "kv_cache.policy": "bf16",
        "kv_cache.dtype": "bf16",
    },
    "ccl_bfp8": {
        "ccl.payload_dtype": "bfp8",
    },
    "residual_bfp8": {
        "activations.residual_dtype": "bfp8",
    },
}


def render_outputs() -> dict[Path, str]:
    output = ROOT / "candidates"
    manifest = []
    rendered = {}
    for config_id, updates in CANDIDATES.items():
        value = copy.deepcopy(BASE)
        value["config_id"] = config_id
        for path, selected in updates.items():
            set_path(value, path, selected)
        path = output / f"{config_id}.json"
        rendered[path] = json.dumps(value, indent=2, sort_keys=True) + "\n"
        manifest.append({"config_id": config_id, "path": str(path.relative_to(ROOT)), "updates": updates})
    rendered[ROOT / "candidate_matrix.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail without writing if the retained matrix differs from deterministic baseline-seed rendering",
    )
    args = parser.parse_args()
    rendered = render_outputs()
    if args.check:
        mismatches = [str(path.relative_to(ROOT)) for path, text in rendered.items() if not path.is_file() or path.read_text() != text]
        if mismatches:
            raise RuntimeError(f"candidate matrix is not reproducible from {BASELINE_SEED.name}: {mismatches}")
        print(json.dumps({"reproducible": True, "artifacts": len(rendered)}, sort_keys=True))
        return
    (ROOT / "candidates").mkdir(parents=True, exist_ok=True)
    for path, text in rendered.items():
        path.write_text(text)


if __name__ == "__main__":
    main()
