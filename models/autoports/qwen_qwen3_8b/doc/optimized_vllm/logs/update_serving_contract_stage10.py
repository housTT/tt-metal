#!/usr/bin/env python3
"""Update doc/full_model/serving_contract.json with the stage-10 serving evidence (rows that changed)."""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path("models/autoports/qwen_qwen3_8b/doc/full_model/serving_contract.json")
OV = "doc/optimized_vllm"
RV = "readiness_vllm"

UPDATES = {
    "prefill_forward(tokens, page_table, kv_cache, enable_trace, prompt_lens, start_pos, sampling_params, empty_slots)": {
        "status": "supported",
        "evidence": f"{RV}/vllm_prefill_burst_benchmark.json",
        "notes": f"served: {RV}/sampling_tests.log, {RV}/prefix_cache_check.json (start_pos at the cached length), {RV}/mixed_shape_trace_reuse.json (multi-row calls, nonzero starts), {RV}/vllm_prefill_burst_benchmark.json (32 x 128 in one batched call); generator level: doc/full_model/batched_prefill_equals_solo_*.stage10.json, readiness/prefill_resume_check.json; fresh rows that share a prefill bucket run as one batched pass (tt/generator.py _batched_prefill_groups), every other row on the single-row path of the same call; model_runner.py submit_prefill",
    },
    "prefill_forward: several rows per call with per-row start_pos and prompt_lens ends": {
        "status": "supported",
        "evidence": "doc/full_model/batched_prefill_equals_solo_mixed_lengths.stage10_tracked.json",
        "notes": "rows 5, 31, 32, 33, 127, 128, 129, 257, 640, 1000, 1024 in both orders: batched groups (32 x 3 rows, 128 x 2 rows) beside single-row rows in one call, every row bit equal in logits and K/V pages (layers 0 and 35) to its solo result, output in call order; doc/full_model/multi_row_prefill.stage10.json (stage-6 contract through the batched path)",
    },
    "batched_prefill": {
        "status": "supported",
        "evidence": f"{RV}/vllm_prefill_burst_benchmark.json",
        "notes": f"equal-padded-length design: fresh rows grouped by prefill bucket (32, 64, 128, 256, 512), row buckets 2/4/8/16/32 with dummy rows, dense work once over rows x bucket (<= 4096, a prepared single-row bucket) on the 2D/1D multicast kernel table (tt/optimized_decoder.py BATCHED_PREFILL_KERNEL_TABLE), attention and cache fill per row inside one op each (batched causal SDPA, paged_fill_cache batch_idx_tensor); batched equals solo bit for bit (doc/full_model/batched_prefill_equals_solo_*.stage10*.json, {OV}/logs/batched_prefill_probe_L36_chip2_v2.json, 22 shapes); 32 x 128 burst TTFT against the stage-9 serial reference in {RV}/vllm_prefill_burst_benchmark.json serial_reference; every (rows, bucket) shape warmed in prepare() before the first capture, program cache flat on the served workload ({RV}/mixed_shape_trace_reuse.json); override QWEN3_BATCHED_PREFILL=0",
    },
    "decode_forward: prompt_tokens / output_tokens (device penalties state)": {
        "status": "supported",
        "evidence": f"{RV}/sampling_tests.log",
        "notes": f"tt/generator.py _apply_sampling_params: prompt mask rebuilt at prefill from the call's tokens and reloaded from prompt_tokens on a reload decode step, output counts reloaded from output_tokens on a reload step, counted on device inside the penalties-on sampling trace on steady steps; {OV}/logs/penalty_probe_L2_chip1.json (plugin call pattern at the generator level); model_runner.py prompt_tokens/output_tokens population",
    },
    "supports_device_penalties": {
        "status": "supported",
        "evidence": f"{RV}/sampling_tests.log",
        "notes": f"model_capabilities['supports_device_penalties'] = True; the common SamplingGenerator's TTPenalties path (presence, frequency, repetition) with a penalties-on sampling trace per decode bucket captured in warmup_model_decode ({RV}/server.log 'device+penalties'); test_tt_penalties.py through the device sampler; {OV}/logs/penalty_probe_L2_chip1.json",
    },
    "presence_penalty / frequency_penalty / repetition_penalty": {
        "status": "supported",
        "evidence": f"{RV}/sampling_tests.log",
        "notes": f"applied on device (supports_device_penalties True); test_tt_penalties.py results in {RV}/sampling_tests.log; generator-level {OV}/logs/penalty_probe_L2_chip1.json",
    },
}


def main() -> None:
    data = json.loads(PATH.read_text())
    rows = data["rows"]
    names = [r.get("name") or r.get("item") for r in rows] if isinstance(rows, list) else list(rows)
    updated = []
    for name, update in UPDATES.items():
        if isinstance(rows, list):
            row = next((r for r in rows if (r.get("name") or r.get("item")) == name), None)
        else:
            row = rows.get(name)
        if row is None:
            raise SystemExit(f"row not found: {name}")
        row.update(update)
        updated.append(name)
    data.setdefault("vllm_integration", {})["optimized_vllm"] = {
        "stage": 10,
        "batched_prefill": "on by default (QWEN3_BATCHED_PREFILL=0 disables)",
        "prefill_kernel_table": "stage10 (QWEN3_PREFILL_KERNEL_TABLE=stage7 restores)",
        "documents": f"{OV}/README.md, {OV}/work_log.md",
    }
    PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {len(updated)} rows")


if __name__ == "__main__":
    main()
