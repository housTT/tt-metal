# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Project the selected precision policy and measured host path into the host contract."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent
CONTRACT_PATH = DOC / "host_weight_contract.json"
CONTEXT_PATH = DOC / "context_contract.json"


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    selected = _load("selected_precision_config.json")
    teacher = _load("post_selection/teacher_forcing_selected/candidate_result.json")
    token_out = _load("post_selection/token_out/full_model_performance.json")
    smoke = _load("post_selection/precision_smoke/precision_propagation_smoke.json")
    context = json.loads(CONTEXT_PATH.read_text())
    contract = json.loads(CONTRACT_PATH.read_text())
    config_id = selected["config_id"]
    if any(payload.get("config_id") != config_id for payload in (teacher, token_out, smoke)):
        raise RuntimeError("host contract inputs do not all use the selected precision config")
    for payload in (teacher, token_out):
        propagation = payload["precision_propagation"]
        if not propagation["all_fields_consumed"] or propagation["consumed_leaf_count"] != 61:
            raise RuntimeError("selected policy is not fully consumed by a measured path")
    if not smoke["all_fields_consumed"] or smoke["consumed_leaf_count"] != 61:
        raise RuntimeError("selected precision smoke does not consume all 61 policy leaves")

    expert = selected["host_backed"]["expert"]
    ple = selected["host_backed"]["ple"]
    checks = teacher["precision_propagation"]["checks"]
    required_upload_checks = (
        "host_backed.expert.source_dtype",
        "host_backed.expert.host_packed_dtype",
        "host_backed.expert.host_packed_layout",
        "host_backed.expert.device_staging_dtype",
        "host_backed.expert.device_staging_layout",
        "host_backed.expert.execution_weight_dtype",
        "host_backed.ple.table_dtype",
        "host_backed.ple.table_layout",
        "host_backed.ple.host_assembly_dtype",
        "host_backed.ple.device_staging_dtype",
        "host_backed.ple.device_staging_layout",
        "host_backed.ple.execution_dtype",
    )
    upload_policy_preserved = all(checks[name]["passed"] for name in required_upload_checks)
    if not upload_policy_preserved:
        raise RuntimeError("expert or PLE upload path did not preserve the selected representation")

    full_stack = context["full_stack_residency"]
    capacity = contract["full_stack_capacity"]
    for key in (
        "decoder_non_expert_weight_bytes_per_device",
        "full_text_endpoint_weight_bytes_per_device",
        "non_expert_weight_allowance_bytes_per_device",
        "max_context_cache_bytes_per_device",
    ):
        capacity[key] = full_stack[key]
    capacity["planned_total_bytes_per_device"] = full_stack["host_backed_planned_total_bytes_per_device"]
    capacity["headroom_bytes_per_device"] = full_stack["host_backed_headroom_bytes_per_device"]
    capacity["fits"] = full_stack["status"] == "host_backed_fits"

    metrics = token_out["metrics"]
    host = token_out["host_service_all_totals"]
    contract["datatype_sweep_selected_policy"] = {
        "config_id": config_id,
        "config_artifact": "datatype_sweep/selected_precision_config.json",
        "weight_groups": selected["weight_groups"],
        "weight_exceptions": selected["weight_exceptions"],
        "layer_exceptions": selected["layer_exceptions"],
        "activations": selected["activations"],
        "ccl": selected["ccl"],
        "kv_cache": selected["kv_cache"],
        "logits_sampling": selected["logits_sampling"],
        "expert_representations": {
            "source": expert["source_dtype"],
            "host_packed": f"{expert['host_packed_dtype']}_{expert['host_packed_layout']}",
            "device_staging": f"{expert['device_staging_dtype']}_{expert['device_staging_layout']}",
            "execution": f"{expert['execution_weight_dtype']}_{expert['device_staging_layout']}",
        },
        "ple_representations": {
            "table": f"{ple['table_dtype']}_{ple['table_layout']}",
            "host_assembly": ple["host_assembly_dtype"],
            "device_staging": f"{ple['device_staging_dtype']}_{ple['device_staging_layout']}",
            "execution": ple["execution_dtype"],
        },
        "upload_policy_preserved": upload_policy_preserved,
        "precision_propagation": {
            "artifact": "datatype_sweep/post_selection/teacher_forcing_selected/candidate_result.json",
            "all_fields_consumed": True,
            "consumed_leaf_count": 61,
            "required_upload_checks": list(required_upload_checks),
        },
        "traced_teacher_forcing": {
            "top1_percent": teacher["top1_percent"],
            "top5_percent": teacher["top5_percent"],
            "top100_percent": teacher["top100_percent"],
            "ttft_seconds": teacher["ttft_seconds"],
            "decode_seconds_per_token": teacher["decode_seconds_per_token"],
            "decode_tokens_per_second_per_user": teacher["decode_tokens_per_second_per_user"],
            "model_only_trace_replays": teacher["model_only_trace_replays"],
        },
        "post_selection_token_out": {
            "artifact": "datatype_sweep/post_selection/token_out/full_model_performance.json",
            "workload": token_out["workload"],
            "ttft_seconds": metrics["ttft_seconds"],
            "decode_seconds_per_token": metrics["decode_seconds_per_token"],
            "decode_tokens_per_second_per_user": metrics["decode_tokens_per_second_per_user"],
            "traced": metrics["traced"],
            "teacher_forced": metrics["teacher_forced"],
            "expert_misses": host["expert_misses"],
            "expert_h2d_bytes": host["expert_h2d_bytes"],
            "expert_h2d_seconds": host["expert_h2d_seconds"],
            "ple_lookup_calls": host["ple_lookup_calls"],
            "ple_table_rows_read": host["ple_table_rows_read"],
            "ple_lookup_seconds": host["ple_lookup_seconds"],
            "ple_device_h2d_bytes": host["ple_device_h2d_bytes"],
            "prohibited_host_work": token_out["runtime_fallback_audit"]["prohibited_host_work"],
        },
    }
    contract["full_model_eager_batch32_seq4096_capacity"]["planned_total_bytes_per_device"] = context[
        "full_model_eager_batch32_seq4096_capacity"
    ]["planned_total_bytes_per_device"]
    contract["full_model_eager_batch32_seq4096_capacity"]["headroom_bytes_per_device"] = context[
        "full_model_eager_batch32_seq4096_capacity"
    ]["headroom_bytes_per_device"]
    contract["validation"]["datatype_sweep_artifacts"] = [
        "datatype_sweep/sweep_results.json",
        "datatype_sweep/selected_precision_config.json",
        "datatype_sweep/post_selection/precision_smoke/precision_propagation_smoke.json",
        "datatype_sweep/post_selection/teacher_forcing_selected/candidate_result.json",
        "datatype_sweep/post_selection/token_out/full_model_performance.json",
        "datatype_sweep/full_runs/context_bfp8/advertised_context_construction.json",
        "datatype_sweep/full_runs/context_bf16/advertised_context_construction.json",
    ]
    contract["measured_limitations"] = [
        item
        for item in contract["measured_limitations"]
        if not item.startswith("Selected token-out is")
        and not item.startswith("The datatype-sweep selected 128+128 warmed token-out path is")
    ]
    contract["measured_limitations"].append(
        "The datatype-sweep selected 128+128 warmed token-out path is "
        f"{metrics['decode_seconds_per_token'] * 1000:.6f} ms/token "
        f"({metrics['decode_tokens_per_second_per_user']:.6f} tokens/s/user). "
        "Traced teacher forcing, not token-out, is the dtype Pareto and selection metric."
    )
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n")
    context["remaining_risk"] = re.sub(
        r"Selected token-out is [0-9.]+ ms/token\.",
        f"Selected token-out is {metrics['decode_seconds_per_token'] * 1000:.6f} ms/token.",
        context["remaining_risk"],
    )
    context["evidence"]["datatype_sweep_selected_token_out"] = (
        "datatype_sweep/post_selection/token_out/full_model_performance.json"
    )
    CONTEXT_PATH.write_text(json.dumps(context, indent=2) + "\n")
    print(json.dumps({"config_id": config_id, "upload_policy_preserved": upload_policy_preserved}, sort_keys=True))


if __name__ == "__main__":
    main()
