# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Recompute ``doc/context_contract.json``'s capacity rows for the full model.

Every byte figure here is either (a) read out of ``doc/full_model/footprint.json``, which is the
allocator's own DRAM view after each construction stage, or (b) arithmetic that is printed next to
the measurement it must agree with. Nothing is asserted that the probe did not see.

    python .../doc/full_model/logs/probe_footprint.py --cache-context 262144 \
        --output .../doc/full_model/footprint.json
    python .../doc/full_model/logs/update_context_contract.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("models/autoports/ornith_ai_ornith_1_0_35b")
CONTRACT = ROOT / "doc" / "context_contract.json"
FOOTPRINT = ROOT / "doc" / "full_model" / "footprint.json"
FOOTPRINT_B32 = ROOT / "doc" / "full_model" / "footprint_batch32.json"

VOCAB = 248320
DIM = 2048
TP = 4
FULL_ATTENTION_LAYERS = 10
LINEAR_ATTENTION_LAYERS = 30

#: bfloat8_b is one byte per element plus one shared exponent byte per 16-element block.
BFP8_BYTES_PER_ELEM = 1 + 1 / 16


def main():
    contract = json.loads(CONTRACT.read_text())
    measured = json.loads(FOOTPRINT.read_text()) if FOOTPRINT.is_file() else None
    measured_b32 = json.loads(FOOTPRINT_B32.read_text()) if FOOTPRINT_B32.is_file() else None

    # The projection lives in the multichip-decoder stage's block; the optimized-multichip stage
    # that follows it changed no per-layer byte count, so this is the current one.
    projection = contract["multichip_decoder"]["full_model_projection"]
    rope_pair_bytes = contract["capacity_evidence"]["full_context_footprint_per_layer_bytes"]["rope_cos_sin_tables"]
    shared_rope_saving = (FULL_ATTENTION_LAYERS - 1) * rope_pair_bytes

    layers_bytes = projection["per_device_all_layers_bytes"] - shared_rope_saving
    embedding_bytes = VOCAB * DIM * 2
    lm_head_bytes = int(DIM * (VOCAB // TP) * BFP8_BYTES_PER_ELEM)
    final_norm_bytes = DIM * 2
    resident = layers_bytes + embedding_bytes + lm_head_bytes + final_norm_bytes

    # This stage's own probe, not the functional-decoder stage's inherited figure: the two differ
    # (33,978,715,136 against 34,091,302,912) because the trace region this stage reserves at
    # open_mesh_device is carved out of the DRAM bank region. Mixing them would compute headroom
    # against DRAM that is not available to the model.
    allocatable = (
        measured["per_device_bytes"]["allocatable_total"]
        if measured
        else contract["capacity_evidence"]["measured_allocatable_dram_bytes"]
    )
    kv_all_layers = projection["per_device_kv_cache_all_layers_bytes"]
    kv_bytes_per_token = kv_all_layers / contract["hf_advertised_context"]

    # What batch 32 actually bounds, from the measured batch-32 build rather than an estimate. The
    # batch-32 probe runs at cache_context 8192, so its kv_cache_and_per_batch_state row is the KV
    # for 32 x 8192 tokens plus the per-batch state; subtracting the first leaves the second, and
    # what is then left of DRAM is the KV ceiling.
    b32_ceiling_per_user = None
    b32_ceiling_arithmetic = None
    if measured_b32:
        b32 = measured_b32["per_device_bytes"]
        b32_kv = int(measured_b32["batch"]) * int(measured_b32["cache_context"]) * kv_bytes_per_token
        b32_state = b32["kv_cache_and_per_batch_state"] - b32_kv
        spare = b32["allocatable_total"] - b32["weights_embedding_lm_head"] - b32["trace_and_sampler"] - b32_state
        b32_ceiling_per_user = int(spare / kv_bytes_per_token / int(measured_b32["batch"]))
        b32_ceiling_arithmetic = (
            f"({b32['allocatable_total']} allocatable - {b32['weights_embedding_lm_head']} weights - "
            f"{b32['trace_and_sampler']} traces/sampler - {int(b32_state)} per-batch state) / "
            f"{kv_bytes_per_token:.0f} B per token / 32 users = {b32_ceiling_per_user} tokens per user, with "
            "zero left for activations; footprint_batch32.json is the measured build this comes from"
        )

    entry = {
        "stage": "full-model",
        "implementation": "tt/model.py + tt/generator.py (the decoder layer is tt/multichip_decoder.py, unchanged)",
        "tests": "tests/test_full_model.py",
        "capability_change": (
            "NONE. The advertised 262144-token context is what the model builds and what "
            "tests/test_full_model.py::test_context_contract_is_the_advertised_one asserts. The batch bound "
            "stays 32 - now set by ttnn.sampling, which runs one core per user and asserts 1 <= num_users <= 32, "
            "the same number the decoder stage advertised. Non-aligned logical prompt lengths are still "
            "accepted end to end, now through the public generator: test_prefill_accepts_any_logical_prompt_length "
            "covers 1, 7, 31, 33, 63, 129, 250, 1000, 2049 and 3000, test_full_stack_non_aligned_long_prompt "
            "runs 5003 through the complete 40-layer stack, and doc/full_model/logs/probe_long_prompt.py walks "
            "5003 / 8191 / 16381 / 32749 / 65521 / 131071 / 262143 through the same public path at the full "
            "advertised cache - the last of those is one token short of the advertised context, prefills in "
            "163.9 s with finite logits and a valid sampled token, and leaves 24.15 GiB of DRAM free "
            "(doc/full_model/long_prompt.json)."
        ),
        "context_length": contract["hf_advertised_context"],
        "capability_note": (
            "The full model adds three tensors to the decoder stack - a replicated bfloat16 token embedding, a "
            "column-parallel bfloat8_b LM head and the final norm - and REMOVES nine duplicate RoPE table pairs "
            "by sharing one across the ten full_attention layers, which this file's "
            "multichip_decoder.full_model_projection explicitly flagged as its own conservatism. Net per-device resident set at the full "
            "advertised context and batch 1 is smaller than that projection, not larger."
        ),
        "per_device_bytes_at_advertised_context_batch1": {
            "decoder_layers_incl_paged_kv": layers_bytes,
            "decoder_layers_note": (
                f"multichip_decoder.full_model_projection.per_device_all_layers_bytes "
                f"({projection['per_device_all_layers_bytes']}) minus the nine "
                f"duplicate RoPE table pairs this stage shares ({shared_rope_saving} = 9 x {rope_pair_bytes})"
            ),
            "token_embedding": embedding_bytes,
            "token_embedding_note": f"{VOCAB} x {DIM} bfloat16, REPLICATED - the residual contract is replicated",
            "lm_head": lm_head_bytes,
            "lm_head_note": (
                f"{DIM} x {VOCAB // TP} bfloat8_b per device (column-parallel over the vocabulary, "
                f"{VOCAB} padded to {VOCAB}); dtype is the decoder policy's dense-projection group"
            ),
            "final_norm": final_norm_bytes,
            "total_resident": resident,
            "allocatable_dram_bytes": allocatable,
            "allocatable_dram_bytes_source": (
                "doc/full_model/footprint.json (this stage's probe, with the 200 MB trace region already "
                "reserved); capacity_evidence.measured_allocatable_dram_bytes is the functional-decoder "
                "stage's figure without a trace region and is 112,587,776 B larger"
            ),
            "free_for_activations": allocatable - resident,
            "headroom_ratio": round(allocatable / resident, 3),
            "headroom_ratio_note": (
                "allocatable / the ARITHMETIC total_resident above. doc/full_model/README.md quotes 3.2x, which "
                "is free_for_activations / the MEASURED resident set in measured_footprint; the two ratios have "
                "different numerators and denominators and both are stated where they appear"
            ),
        },
        "capacity_reduction": "none",
        "reduction_reason": None,
        "batch_contract": {
            "primary": 1,
            "max_batch_size": 32,
            "max_batch_reason": (
                "ttnn.sampling runs one core per user and asserts 1 <= num_users <= 32 "
                "(sampling_device_operation); the decoder stage's advertised decode batch bound is also 32, so "
                "neither narrows the other."
            ),
            "tested": [1, 4, 32],
            "tested_note": (
                "batch 4 covers mixed prompt lengths, fixed slots and an inactive row (position -1), and "
                "test_batch_one_and_batch_four_agree_on_the_same_prompt asserts slot 0 of a batch-4 model "
                "predicts what the batch-1 model predicts. batch 32 - the advertised bound - is exercised end "
                "to end by test_batch_32_prefill_and_decode (32 mixed prompt lengths, prefill plus a traced "
                "decode step, every row's position advanced), and doc/full_model/footprint_batch32.json is the "
                "measured 40-layer build at batch 32. No batch was found to be infeasible."
            ),
        },
        "batch_times_context_bound": {
            "kv_bytes_per_token_per_device": round(kv_bytes_per_token, 1),
            "kv_bytes_per_token_note": (
                f"{FULL_ATTENTION_LAYERS} full_attention layers x bfloat8_b paged K and V; "
                f"{kv_all_layers} B holds {contract['hf_advertised_context']} tokens"
            ),
            "batch32_kv_ceiling_tokens_per_user": b32_ceiling_per_user,
            "batch32_kv_ceiling_arithmetic": b32_ceiling_arithmetic,
            "note": (
                "Paged KV is allocated by the caller through the generator's cache_context, exactly as "
                "--num-gpu-blocks is in vLLM. This bounds the PRODUCT of batch and context, not the model's "
                "advertised context: at batch 1 the cache for the full 262144 tokens fits with 24.17 GiB "
                f"free (measured, footprint.json) and a real 262143-token non-aligned prompt runs through the "
                f"whole stack (long_prompt.json). At batch 32 the ceiling is {b32_ceiling_per_user} tokens per "
                "user if EVERY remaining byte goes to KV and nothing is left for activations; the arithmetic is "
                "in batch32_kv_ceiling_arithmetic. This is an allocation choice, not a capability reduction - "
                "the model, the page table, the positions and the tests all still address 262144."
            ),
        },
        "internal_shape_policy_delta": {
            "public_prefill_seq_len": "unchanged: any 1 <= seq_len <= supported_context, no divisibility requirement",
            "who_owns_the_padding": (
                "the model. tt/model.py chunks the prompt into prefill_chunk blocks across the whole stack "
                "(rather than running the whole prompt through one layer at a time), pads each block's physical "
                "length to 128, relies on the decoder's logical_len masking for the tail, fills the paged cache, "
                "keeps positions coherent and slices the returned logits back to the logical length."
            ),
            "decode_token_buffer": (
                "[1, 1, 1, 32] uint32 ROW_MAJOR - ttnn.sampling emits one token per sampler row and needs a "
                "rank-4 preallocated output, so the same buffer is the sampler's output and the decode graph's "
                "token input. That identity is what makes token feedback device-side."
            ),
            "lm_head_output": (
                "[1, 1, 32, padded_vocab/tp] per device, sampler-ready with no gather; padded_vocab is "
                f"align_up({VOCAB}, 32*tp) = {VOCAB}, so nothing is padded and no invalid-vocab mask is needed."
            ),
        },
        "shared_repo_changes": [
            "models/common/sampling/tt_sampling.py: opt-in args.topk_num_groups grouped local top-k (default 1)",
            "models/common/readiness_check/hf_model.py: new; resolve the HF reference class from config.architectures",
            "models/common/readiness_check/generate.py, run_autoregressive.py: use it; normalise apply_chat_template",
        ],
    }

    if measured:
        entry["measured_footprint"] = {
            "source": "doc/full_model/footprint.json, from doc/full_model/logs/probe_footprint.py",
            "cache_context": measured["cache_context"],
            "batch": measured["batch"],
            "per_device_bytes": measured["per_device_bytes"],
            "arithmetic_vs_measured_note": (
                "The arithmetic above counts long-lived model tensors only; the measurement additionally "
                "carries the sampler's index tables and whatever the allocator has not returned yet, so the "
                "measured resident set is the larger of the two and is the one to plan against."
            ),
        }

    if measured_b32:
        entry["measured_footprint_batch32"] = {
            "source": "doc/full_model/footprint_batch32.json",
            "cache_context": measured_b32["cache_context"],
            "batch": measured_b32["batch"],
            "per_device_bytes": measured_b32["per_device_bytes"],
            "note": (
                "The whole 40-layer stack builds, allocates both per-batch state packs and captures its decode "
                "traces at batch 32: 8.31 GiB resident, 23.33 GiB free. ttnn.conv1d's weight preparation refuses "
                "every prefill block length at this batch and logs caught 'Out of Memory' L1 messages while "
                "probing; that is the decoder stage's documented FIR fallback, is handled by "
                "_prepare_conv1d_weights_local, and is not a failure."
            ),
        }

    contract["full_model"] = entry
    # The top-level labels name the stage that last owned this file; the per-stage blocks above keep
    # their own history. Leaving them at the decoder stage's values made the file read as if the
    # full model had never touched it.
    contract["stage"] = "full-model"
    contract["target"] = (
        "the whole 40-layer text model - token embedding, the optimized multichip decoder stack, the final "
        "norm, a column-parallel LM head and on-device split sampling - on a 1x4 Blackhole (p300c) mesh under "
        "FABRIC_1D_RING; the per-stage blocks below keep each earlier stage's own target"
    )
    contract["current_supported_context"] = contract["hf_advertised_context"]
    contract["supported"]["context_length"] = contract["hf_advertised_context"]
    contract["supported"]["capability_reduction"] = "none"
    # Idempotent: the documented reproduce block re-runs this script, and an unconditional append
    # would add the sentence again every time.
    note = (
        "Full-model stage: tt/model.py defaults max_context to text_config.max_position_embeddings and "
        "tests/test_full_model.py::test_context_contract_is_the_advertised_one asserts it; see "
        "full_model.per_device_bytes_at_advertised_context_batch1 for the recomputed weight-plus-KV capacity."
    )
    existing = contract["supported"].get("notes", "")
    while note in existing:
        existing = existing.replace(note, "").strip()
    contract["supported"]["notes"] = f"{existing} {note}".strip()

    CONTRACT.write_text(json.dumps(contract, indent=1) + "\n")
    print(json.dumps(entry, indent=2))
    print(f"\nwrote {CONTRACT}")


if __name__ == "__main__":
    main()
