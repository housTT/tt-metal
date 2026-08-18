# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Write the sweep's candidate precision configs to ``doc/datatype_sweep/candidates/``.

One JSON per candidate, in the same schema as ``selected_precision_config.json``, so a candidate is
run by exactly the mechanism that will later carry the selection: ``build_generator(policy=<path>)``.
Nothing here is measured - this file is the *matrix*, and ``sweep_one.py`` measures it.

The matrix is `$datatype-sweep`'s default search applied to this model's real decode cost:

* the routed-expert matmuls are already BFP4/LoFi (the decoder stage selected them), so this stage
  owes them the **opposite** comparison - BFP4+HiFi2 - to prove LoFi is the right fidelity rather
  than the inherited one;
* the dense projections and the shared expert are BFP8/HiFi2. Both get BFP8+LoFi (same dtype,
  cheaper fidelity), BFP4+HiFi2 and BFP4+LoFi;
* the LM head inherits the dense group. The optimized full-model stage measured BFP4 there and
  handed the frontier point to this stage; it gets both fidelities too;
* KV cache, CCL payload, residual stream and logits are the yes/no dtype switches.

``prefill_sdpa_chunk`` is carried explicitly on every candidate rather than left to the name-keyed
table, so no candidate silently takes the conservative fallback and measures a slower prefill for a
reason that has nothing to do with its dtype.
"""

from __future__ import annotations

import json
from pathlib import Path

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.optimized_decoder import DEFAULT_POLICY, PREFILL_SDPA_CHUNK
from models.autoports.ornith_ai_ornith_1_0_35b.tt.precision_config import write_policy_file

SWEEP_DIR = Path("models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep")
CANDIDATE_DIR = SWEEP_DIR / "candidates"

BASE = DEFAULT_POLICY.replace(prefill_sdpa_chunk=PREFILL_SDPA_CHUNK["optimized"])

LOFI, HIFI2, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
BF16, BFP8, BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b

#: ``(config_id, {policy field: value}, rationale)``.
CANDIDATES = [
    (
        "S00-baseline-optimized",
        {},
        "the decoder stage's selected policy, carried unchanged by the optimized full model. The "
        "sweep's baseline and the reference point for every row below",
    ),
    # ---- LM head (the model's one extra dense projection, largest full-model-only decode op) ----
    (
        "C01-lmhead-bfp4-hifi2",
        {"lm_head_dtype": BFP4},
        "BFP4 LM head at the dense group's inherited HiFi2. The optimized full-model stage measured "
        "this arm (1.351 vs 1.435 ms of reduced model trace) and handed it over as a frontier point",
    ),
    (
        "C02-lmhead-bfp4-lofi",
        {"lm_head_dtype": BFP4, "lm_head_fidelity": LOFI},
        "the BFP4+LoFi arm the skill requires for every material BFP4 matmul group",
    ),
    (
        "C03-lmhead-bfp8-lofi",
        {"lm_head_fidelity": LOFI},
        "same dtype, cheaper fidelity: the BFP8+LoFi vs BFP8+HiFi2 comparison the skill requires for "
        "dominant decode projection groups",
    ),
    # ---- dense token-mixer projections ----
    (
        "C04-proj-bfp8-lofi",
        {"proj_fidelity": LOFI},
        "dense projections keep BFP8 and drop to LoFi. The BFP8 fidelity comparison for the largest "
        "dense group in the layer",
    ),
    (
        "C05-proj-bfp4-hifi2",
        {"proj_dtype": BFP4},
        "the decoder stage's `bfp4-projections` policy, which it measured faster and rejected on a "
        "layer-level HF-golden ladder rather than on full-model accuracy",
    ),
    (
        "C06-proj-bfp4-lofi",
        {"proj_dtype": BFP4, "proj_fidelity": LOFI},
        "the required BFP4+LoFi arm for the dense projection group",
    ),
    # ---- shared expert ----
    (
        "C07-shared-bfp8-lofi",
        {"shared_fidelity": LOFI},
        "shared expert keeps BFP8 and drops to LoFi",
    ),
    (
        "C08-shared-bfp4-lofi",
        {"shared_dtype": BFP4, "shared_fidelity": LOFI},
        "the required BFP4+LoFi arm for the shared-expert group",
    ),
    (
        "C09-shared-bfp4-hifi2",
        {"shared_dtype": BFP4},
        "the BFP4+HiFi2 comparison for the shared-expert group",
    ),
    # ---- routed experts: the fidelity the decoder stage inherited, tested the other way ----
    (
        "C10-experts-bfp4-hifi2",
        {"expert_fidelity": HIFI2},
        "the routed experts are already BFP4+LoFi, so the comparison this stage owes them is the "
        "*higher* fidelity arm: proof that LoFi is selected rather than assumed",
    ),
    # ---- yes/no dtype switches ----
    (
        "C11-kv-bfp4",
        {"kv_cache_dtype": BFP4},
        "narrower paged KV cache. Changes memory capacity, so it needs its own context contract",
    ),
    (
        "C12-kv-bf16",
        {"kv_cache_dtype": BF16, "prefill_sdpa_chunk": 128},
        "the wider cache, as the high-precision control for the BFP8 cache the decoder stage "
        "selected. 128 is the largest chunked-SDPA q/k that builds with a bfloat16 cache",
    ),
    (
        "C13-ccl-bfp8",
        {"ccl_dtype": BFP8},
        "halve the CCL payload: both per-layer collectives carry bfloat8_b instead of bfloat16 out "
        "of the token mixer and bfloat8_b out of the MoE",
    ),
    (
        "C14-residual-bfp8",
        {"residual_dtype": BFP8},
        "the inter-layer residual stream in bfloat8_b, which also narrows what both RMSNorms read",
    ),
    (
        "C15-logits-bfp8",
        {"logits_dtype": BFP8},
        "the 62464-wide logits tensor in bfloat8_b, which is what the split sampler's local top-k reads",
    ),
]


def main():
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for config_id, overrides, rationale in CANDIDATES:
        policy = BASE.replace(name=config_id, **overrides)
        path = CANDIDATE_DIR / f"{config_id}.json"
        write_policy_file(path, policy, stage="datatype-sweep", candidate_rationale=rationale)
        index.append(
            {"config_id": config_id, "path": str(path), "overrides": sorted(overrides), "rationale": rationale}
        )
        print(f"wrote {path}")
    (CANDIDATE_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
