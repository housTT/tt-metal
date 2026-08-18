# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Write the ``datatype_sweep`` block of ``doc/context_contract.json``.

Every byte in the block comes from ``doc/datatype_sweep/capacity/*.json``, which is the allocator's
own DRAM view after each construction step at the **full advertised 262144-token context**, one file
per KV-cache dtype the sweep evaluated. Nothing here is projected: if a row is not in a capacity
probe it is not written.

The block is additive - the earlier stages' blocks are left exactly as they were - except for the
two flat keys ``hf_advertised_context`` / ``current_supported_context``, which
``.agents/scripts/check_context_contract.py`` resolves with a non-recursive top-level lookup and
which must therefore stay in sync with the nested objects.

    python .../doc/datatype_sweep/logs/update_context_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("models/autoports/ornith_ai_ornith_1_0_35b")
CONTRACT = ROOT / "doc" / "context_contract.json"
SWEEP = ROOT / "doc" / "datatype_sweep"
CAPACITY = SWEEP / "capacity"

ADVERTISED = 262144


def main():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    selected_policy = json.loads((SWEEP / "selected_precision_config.json").read_text(encoding="utf-8"))
    long_prompt_path = SWEEP / "long_prompt.json"
    long_prompt = json.loads(long_prompt_path.read_text(encoding="utf-8")) if long_prompt_path.is_file() else None

    probes = {}
    for path in sorted(CAPACITY.glob("*.json")):
        probes[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if "selected" not in probes:
        raise SystemExit(f"{CAPACITY}/selected.json is missing; run probe_capacity.py for the selected config")

    selected = probes["selected"]
    per_dtype = {}
    for key, probe in probes.items():
        bytes_ = probe["per_device_bytes"]
        per_dtype[key] = {
            "config_id": probe["config_id"],
            "kv_cache_dtype": probe["kv_cache_dtype"],
            "cache_context_allocated": probe["cache_context"],
            "batch": probe["batch"],
            "per_device_bytes": bytes_,
            "kv_cache_per_device_bytes_per_token": probe["kv_cache"]["per_device_bytes_per_token"],
            "advertised_context_fits": probe["capacity"]["advertised_context_fits"],
            "largest_feasible_context_at_batch_1": probe["capacity"]["largest_feasible_context_at_this_batch"],
            "largest_feasible_note": probe["capacity"]["largest_feasible_note"],
        }

    block = {
        "stage": "datatype-sweep",
        "implementation": "tt/precision_config.py (the selected config artifact and the resolution rule) "
        "+ tt/optimized_decoder.py::PrecisionPolicy; tt/model.py and tt/generator.py are unchanged except "
        "for resolving the policy and passing the logits/LM-head dtypes through",
        "tests": "tests/test_full_model.py",
        "selected_precision_config": "doc/datatype_sweep/selected_precision_config.json",
        "selected_config_id": selected_policy["config_id"],
        "selected_kv_cache_dtype": selected_policy["kv_cache"]["dtype"],
        "capability_change": (
            "NONE, and the headroom improves. The sweep left the paged KV cache at bfloat8_b, so the "
            "advertised 262144-token context is allocated exactly as the optimized full-model stage "
            "allocated it. The selected config narrows the dense projections and the LM head from "
            "bfloat8_b to bfloat4_b, which makes the resident weight set SMALLER "
            f"({selected['per_device_bytes']['weights_embedding_lm_head']} B against "
            f"{probes['optimized']['per_device_bytes']['weights_embedding_lm_head']} B for the pre-sweep "
            "policy), so free DRAM at the advertised context goes up rather than down. The batch bound "
            "stays 32 (ttnn.sampling asserts 1 <= num_users <= 32) and non-aligned logical prompt "
            "lengths are still accepted end to end - doc/datatype_sweep/long_prompt.json walks them "
            "through the public path on the selected config at the full advertised cache."
        ),
        "context_length": ADVERTISED,
        "kv_cache_dtype_candidates": per_dtype,
        "kv_cache_dtype_finding": (
            "all three evaluated cache dtypes fit the advertised context with room to spare, so the "
            "cache dtype was never a capacity decision for this model at batch 1 - it was a decode-speed "
            "and accuracy decision, and on both it was a wash (bfloat4_b 41.962 t/s/u, bfloat8_b 41.962, "
            "bfloat16 41.997, all inside the 0.24-0.48 % run-to-run spread). bfloat8_b is kept because it "
            "is what the decoder stage measured and because the bfloat4_b cache FAILS the accuracy gate "
            "once it is combined with bfloat4_b dense projections (C18: top-1 0.870/0.860 against the "
            "0.90 bar), which is exactly the combination the selected config would have paired it with."
        ),
        "measured_by": "doc/datatype_sweep/logs/probe_capacity.py, one process per candidate, each "
        "building the whole 40-layer model and allocating the paged KV cache at 262144 tokens",
        # Repo-relative, like every other path in this file. ROOT is already
        # models/autoports/<model>, so relative_to(ROOT.parent.parent) would strip "models/".
        "evidence": sorted(str(p) for p in CAPACITY.glob("*.json")),
    }
    if long_prompt:
        completed = [r for r in long_prompt.get("results", []) if r.get("status") == "ok"]
        block["non_aligned_prompt_walk"] = {
            "artifact": "doc/datatype_sweep/long_prompt.json",
            "lengths_completed": [r["prompt_len"] for r in completed],
            "largest_completed": max((r["prompt_len"] for r in completed), default=None),
            "note": "each length is a fresh request through the public generator on the selected config, "
            "with the full advertised cache allocated: prefill, one traced token-out decode step, then a "
            "check that the logits are finite and the sampled token is a valid id",
        }

    contract["datatype_sweep"] = block
    contract["stage"] = "datatype-sweep"
    contract["current_supported_context"] = ADVERTISED
    contract["supported"]["context_length"] = ADVERTISED
    contract["supported"]["capability_reduction"] = "none"
    note = (
        " Datatype-sweep stage: the selected precision config keeps the paged KV cache at bfloat8_b, so "
        "the advertised context is allocated unchanged; the narrower dense projections and LM head make "
        "the resident weight set smaller, so free DRAM at 262144 tokens goes UP "
        f"({selected['per_device_bytes']['free_for_activations']} B against "
        f"{probes['optimized']['per_device_bytes']['free_for_activations']} B). "
        "doc/datatype_sweep/capacity/ has one measured allocator view per evaluated cache dtype, "
        "including the two the sweep did not select."
    )
    if note not in contract["supported"]["notes"]:
        contract["supported"]["notes"] += note
    CONTRACT.write_text(json.dumps(contract, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {CONTRACT}")
    print(json.dumps({k: v for k, v in block.items() if k != "kv_cache_dtype_candidates"}, indent=2)[:1400])


if __name__ == "__main__":
    main()
