# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Recompute advertised-context capacity for each swept KV-cache representation."""

from __future__ import annotations

import json
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent
CONTEXT = DOC / "context_contract.json"
DRAM_BYTES = 34_225_520_640
CONTEXT_TOKENS = 262_144
QSA_LAYERS = 12
BLOCK_SIZE = 64
BLOCKS = CONTEXT_TOKENS // BLOCK_SIZE
KV_HEADS_PER_DEVICE = 1
HEAD_DIM = 256
INDEX_HEADS_PER_DEVICE = 1
INDEX_HEAD_DIM = 128
TILE_ELEMENTS = 1024
TILE_BYTES = {"bfp8": 1088, "bf16": 2048}
COMPRESSED_INDEX_BF16_BYTES_PER_LAYER = (CONTEXT_TOKENS // 4) * INDEX_HEAD_DIM * 2
OPTIMIZED_BASELINE_TOTAL_BYTES = 10_005_165_144
OPTIMIZED_BASELINE_CACHE_BYTES = 2_340_421_632
OPTIMIZED_BASELINE_NON_EXPERT_WEIGHT_BYTES = 4_460_891_136
OPTIMIZED_BASELINE_BATCH32_PLAN_BYTES = 23_393_673_304
FIXED_NON_WEIGHT_AND_NON_CACHE_BYTES = (
    OPTIMIZED_BASELINE_TOTAL_BYTES
    - OPTIMIZED_BASELINE_CACHE_BYTES
    - OPTIMIZED_BASELINE_NON_EXPERT_WEIGHT_BYTES
)
DECODER_WITHOUT_QSA_BYTES = 3_479_858_176 - 637_599_744
QSA_INPUT_BF16_BYTES = 448_266_240
QSA_ATTENTION_OUTPUT_BF16_BYTES = 188_743_680
QSA_NORM_BYTES = 393_216 + 196_608
ENDPOINT_WITHOUT_LM_HEAD_BYTES = 981_032_960 - 337_715_200
LM_HEAD_BF16_BYTES = 635_699_200


def cache_bytes(dtype: str) -> dict[str, int]:
    raw_elements_per_layer = (
        2 * BLOCKS * KV_HEADS_PER_DEVICE * BLOCK_SIZE * HEAD_DIM
        + BLOCKS * INDEX_HEADS_PER_DEVICE * BLOCK_SIZE * INDEX_HEAD_DIM
    )
    raw_bytes = raw_elements_per_layer // TILE_ELEMENTS * TILE_BYTES[dtype] * QSA_LAYERS
    compressed_bytes = COMPRESSED_INDEX_BF16_BYTES_PER_LAYER * QSA_LAYERS
    return {
        "raw_main_k_v_and_index_bytes_per_device": raw_bytes,
        "compressed_bf16_index_bytes_per_device": compressed_bytes,
        "total_bytes_per_device": raw_bytes + compressed_bytes,
    }


def _bfp8_bytes(bf16_bytes: int) -> int:
    return bf16_bytes // 2048 * 1088


def selected_weight_bytes(config: dict) -> dict[str, int]:
    qsa_input = QSA_INPUT_BF16_BYTES
    if config["weight_groups"]["qsa_input"]["dtype"] == "bfp8":
        qsa_input = _bfp8_bytes(qsa_input)
    attention_output = QSA_ATTENTION_OUTPUT_BF16_BYTES
    if config["weight_groups"]["attention_output"]["dtype"] == "bfp8":
        attention_output = _bfp8_bytes(attention_output)
    decoder = DECODER_WITHOUT_QSA_BYTES + qsa_input + attention_output + QSA_NORM_BYTES
    lm_head = (
        LM_HEAD_BF16_BYTES
        if config["weight_groups"]["lm_head"]["dtype"] == "bf16"
        else _bfp8_bytes(LM_HEAD_BF16_BYTES)
    )
    endpoint = ENDPOINT_WITHOUT_LM_HEAD_BYTES + lm_head
    return {
        "decoder_non_expert_weight_bytes_per_device": decoder,
        "full_text_endpoint_weight_bytes_per_device": endpoint,
        "non_expert_weight_allowance_bytes_per_device": decoder + endpoint,
    }


def main() -> None:
    selected = json.loads((ROOT / "selected_precision_config.json").read_text())
    selected_dtype = selected["kv_cache"]["dtype"]
    selected_bf16 = copy.deepcopy(selected)
    selected_bf16["config_id"] = f"{selected['config_id']}_kv_bf16_capacity"
    selected_bf16["kv_cache"]["policy"] = "bf16"
    selected_bf16["kv_cache"]["dtype"] = "bf16"
    (ROOT / "selected_kv_bf16_capacity_config.json").write_text(
        json.dumps(selected_bf16, indent=2, sort_keys=True) + "\n"
    )
    weights = selected_weight_bytes(selected)
    output_dir = ROOT / "context_contract_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = {}
    for dtype in ("bfp8", "bf16"):
        construction_path = ROOT / "full_runs" / f"context_{dtype}" / "advertised_context_construction.json"
        construction = json.loads(construction_path.read_text())
        sizes = cache_bytes(dtype)
        planned = (
            FIXED_NON_WEIGHT_AND_NON_CACHE_BYTES
            + weights["non_expert_weight_allowance_bytes_per_device"]
            + sizes["total_bytes_per_device"]
        )
        payload = {
            "schema_version": 1,
            "model_id": "Qwen/Qwen3.8-Flash-Next",
            "kv_cache_dtype": dtype,
            "hf_advertised_context_tokens": CONTEXT_TOKENS,
            "full_model_supported_context_tokens": CONTEXT_TOKENS,
            "capability_reduction": None,
            "hardware": "P300 Blackhole board, dies 0 and 1",
            "mesh": "1x2 FABRIC_1D",
            "cache_geometry": {
                "qsa_layers": QSA_LAYERS,
                "page_blocks": BLOCKS,
                "page_block_size": BLOCK_SIZE,
                "main_k_v_heads_per_device": KV_HEADS_PER_DEVICE,
                "main_head_dim": HEAD_DIM,
                "index_heads_per_device": INDEX_HEADS_PER_DEVICE,
                "index_head_dim": INDEX_HEAD_DIM,
                "tile_elements": TILE_ELEMENTS,
                "tile_bytes": TILE_BYTES[dtype],
            },
            "cache_capacity": sizes,
            "selected_precision_weight_capacity": weights,
            "full_stack_capacity": {
                "fixed_non_weight_and_non_cache_bytes_per_device": FIXED_NON_WEIGHT_AND_NON_CACHE_BYTES,
                "planned_total_bytes_per_device": planned,
                "dram_bytes_per_device": DRAM_BYTES,
                "headroom_bytes_per_device": DRAM_BYTES - planned,
                "fits": planned <= DRAM_BYTES,
            },
            "construction_evidence": f"../full_runs/context_{dtype}/advertised_context_construction.json",
            "construction_evidence_config_id": construction["config_id"],
            "selected_config_id": selected["config_id"],
            "construction_capacity_equivalent_to_selected": True,
            "construction_capacity_equivalence_reason": (
                f"The construction policy and final selected-derived {dtype.upper()} capacity candidate have "
                "identical weight dtypes, KV geometry, KV dtype, and endpoint footprint. They differ only in "
                "shared-projection compute fidelity (HiFi2 versus LoFi), which does not change any allocated "
                "tensor bytes."
            ),
            "non_aligned_evidence": f"../smokes/{'kv_bf16_control' if dtype == 'bf16' else 'baseline'}.xml",
            "method": (
                "Full 48-layer construction allocated every QSA cache at 262,144 tokens. "
                "Capacity charges TILE-physical raw K/V/index pages and the BF16 compressed-index working set."
            ),
        }
        (output_dir / f"kv_{dtype}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        candidates[dtype] = payload

    if not candidates[selected_dtype]["full_stack_capacity"]["fits"]:
        raise RuntimeError(f"selected {selected_dtype} KV cache does not fit advertised context")
    contract = json.loads(CONTEXT.read_text())
    chosen = candidates[selected_dtype]
    capacity = contract["full_stack_residency"]
    capacity["selected_kv_cache_dtype"] = selected_dtype
    capacity["max_context_cache_bytes_per_device"] = chosen["cache_capacity"]["total_bytes_per_device"]
    capacity.update(weights)
    capacity["host_backed_planned_total_bytes_per_device"] = chosen["full_stack_capacity"][
        "planned_total_bytes_per_device"
    ]
    capacity["host_backed_headroom_bytes_per_device"] = chosen["full_stack_capacity"]["headroom_bytes_per_device"]
    capacity["status"] = "host_backed_fits"
    capacity["reason"] = (
        "The selected datatype policy charges exact TILE-physical non-expert weights: "
        f"{weights['decoder_non_expert_weight_bytes_per_device']:,} decoder bytes/die after BFP8 QSA input/output, "
        f"{weights['full_text_endpoint_weight_bytes_per_device']:,} endpoint bytes/die with a BF16 LM head, "
        f"and {chosen['cache_capacity']['total_bytes_per_device']:,} bytes/die for the selected {selected_dtype.upper()} "
        f"maximum-context K/V/index cache. Exact host-backed top-10 expert slots, PLE staging, runtime state, endpoints, "
        f"and the 1 GiB trace/runtime reserve remain charged. The complete plan fits with "
        f"{chosen['full_stack_capacity']['headroom_bytes_per_device']:,} bytes/die headroom; no context reduction is required."
    )
    contract["multichip_parallelism"]["qsa_main_cache_per_device_at_batch1"] = (
        f"[4096, 1, 64, 256] {selected_dtype.upper()} K and V at context 262144"
    )
    contract["multichip_parallelism"]["qsa_index_cache_per_device_at_batch1"] = (
        f"Replicated raw [4096, 1, 64, 128] {selected_dtype.upper()} plus BF16 compressed index cache"
    )
    contract["datatype_sweep_context_candidates"] = {
        dtype: {
            "artifact": f"datatype_sweep/context_contract_candidates/kv_{dtype}.json",
            "max_context_cache_bytes_per_device": value["cache_capacity"]["total_bytes_per_device"],
            "planned_total_bytes_per_device": value["full_stack_capacity"]["planned_total_bytes_per_device"],
            "headroom_bytes_per_device": value["full_stack_capacity"]["headroom_bytes_per_device"],
            "fits": value["full_stack_capacity"]["fits"],
            "full_model_supported_context_tokens": CONTEXT_TOKENS,
            "construction_evidence": f"datatype_sweep/full_runs/context_{dtype}/advertised_context_construction.json",
            "construction_evidence_config_id": value["construction_evidence_config_id"],
            "construction_capacity_equivalent_to_selected": True,
        }
        for dtype, value in candidates.items()
    }
    contract["selected_datatype_sweep_kv_cache_dtype"] = selected_dtype
    contract["evidence"]["datatype_sweep_selected_precision"] = "datatype_sweep/selected_precision_config.json"
    contract["evidence"]["datatype_sweep_context_bfp8"] = "datatype_sweep/context_contract_candidates/kv_bfp8.json"
    contract["evidence"]["datatype_sweep_context_bf16"] = "datatype_sweep/context_contract_candidates/kv_bf16.json"
    batch32 = contract["full_model_eager_batch32_seq4096_capacity"]
    weight_delta = weights["non_expert_weight_allowance_bytes_per_device"] - OPTIMIZED_BASELINE_NON_EXPERT_WEIGHT_BYTES
    batch32["base_full_model_max_context_batch1_plan_bytes_per_device"] = candidates["bfp8"][
        "full_stack_capacity"
    ]["planned_total_bytes_per_device"]
    batch32["planned_total_bytes_per_device"] = OPTIMIZED_BASELINE_BATCH32_PLAN_BYTES + weight_delta
    batch32["headroom_bytes_per_device"] = DRAM_BYTES - batch32["planned_total_bytes_per_device"]
    CONTEXT.write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"selected_kv_cache_dtype": selected_dtype, "candidates": candidates}, sort_keys=True))


if __name__ == "__main__":
    main()
