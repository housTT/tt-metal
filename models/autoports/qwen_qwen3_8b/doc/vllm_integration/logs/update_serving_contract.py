#!/usr/bin/env python3
"""Update doc/full_model/serving_contract.json with the stage-9 serving evidence (rows that changed)."""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path("models/autoports/qwen_qwen3_8b/doc/full_model/serving_contract.json")
VLLM = "doc/vllm_integration"
RV = "readiness_vllm"

UPDATES = {
    "initialize_vllm_model(hf_config, mesh_device, max_batch_size, ...)": {
        "status": "supported",
        "evidence": f"tt/generator_vllm.py Qwen3ForCausalLM.initialize_vllm_model -> build_generator(prepare=False, num_blocks=1); propagation check of the selected precision config at load ({RV}/server.log 'propagation check OK'); loader.py:38",
    },
    "allocate_kv_cache(shape, dtype, num_layers)": {
        "status": "supported",
        "evidence": f"tt/generator_vllm.py allocate_kv_cache -> Qwen3Model.allocate_kv_cache(num_blocks, dtype) with the selected KV dtype, then Qwen3Generator.adopt_cache so warmup and capture run over the vLLM cache; scratch_block=num_blocks passed on every call ({RV}/server.log 'vLLM KV cache'); model_runner.py:500",
    },
    "get_max_tokens_all_users(model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs)": {
        "status": "supported",
        "evidence": f"tt/generator_vllm.py get_max_tokens_all_users -> Qwen3Generator.max_tokens_all_users(model, mesh_device, kv_cache_dtype=<selected config>) from live free DRAM after the owned pool was released ({RV}/server.log 'max_tokens_all_users='); worker.py:541",
    },
    "prefill_forward(tokens, page_table, kv_cache, enable_trace, prompt_lens, start_pos, sampling_params, empty_slots)": {
        "status": "supported",
        "evidence": f"served: {RV}/sampling_tests.log, {RV}/prefix_cache_check.json (start_pos at the cached length), {RV}/mixed_shape_trace_reuse.json (multi-row calls, nonzero starts); generator level: doc/full_model/multi_row_prefill.json, readiness/prefill_resume_check.json; model_runner.py:2770-2840; continuation rows compile per (start, length) after warmup ({VLLM}/README.md limitations, stage-10 runtime-indexed prefill)",
    },
    "decode_forward(tokens, page_table, kv_cache, start_pos, sampling_params, enable_trace, read_from_device)": {
        "status": "supported",
        "evidence": f"served through the async split: decode_forward(read_from_device=False) -> DecodeOutput, read_decode_output(async_read=True), process_decode_output_host ({RV}/server.log 'deferred_read=True', {RV}/vllm_benchmark.json); generator level: doc/full_model/split_sampling_trace.tracked.json, batch_decode_*.json; async_decode.py:625",
    },
    "decode_forward: reset_batch / slot_remap": {
        "status": "supported",
        "evidence": f"reset_batch reloads tokens and positions from the host (the plugin drains pending async steps first) and re-registers seeds; slot_remap applied to the seed manager (tt/generator.py decode_forward); served: {RV}/sampling_tests.log (test_request_isolation, seeded tests), {RV}/mixed_shape_trace_reuse.json reset_batch_steps",
    },
    "read_decode_output(tt_out, async_read=True) / process_decode_output_host": {
        "status": "supported",
        "evidence": f"model_capabilities['supports_async_decode'] = True; tt/generator.py read_decode_output (tensor.cpu(blocking=False) + ttnn.record_event on CQ 0) and process_decode_output_host; served with --async-scheduling ({RV}/server.log 'async_scheduling=True', every decode line 'deferred_read=True'); stale-token/current-position/page-table coverage: doc/full_model/split_sampling_trace.stage9_tracked.json plus the served greedy repeats in {RV}/mixed_shape_trace_reuse.json",
    },
    "warmup_model_prefill / warmup_model_decode (two-phase)": {
        "status": "supported",
        "evidence": f"phase 1: Qwen3Generator.prepare() over the adopted vLLM cache (warmup_model_prefill); phase 2: capture_decode_traces in warmup_model_decode(enable_trace=True), five device decode and sampling traces captured before the first request ({RV}/server.log 'warmup_model_decode(enable_trace=True): captured'); the generator also enforces prepare() at every forward entry (_require_prepared); model_runner.py:3478-3489",
    },
    "supports_async_decode": {
        "status": "supported",
        "evidence": f"model_capabilities['supports_async_decode'] = True; see the read_decode_output row; {RV}/vllm_benchmark.json measured with async scheduling on",
    },
    "supports_prefix_caching": {
        "status": "supported",
        "evidence": f"readiness/prefill_resume_check.json (pass at lowered thresholds 0.90/0.93, both controls fail) and readiness/prefill_resume_check_stage9_default_thresholds.json (default thresholds: only the 16352 hit window top-1 31/32, stage-7 margin probe); served: {RV}/prefix_cache_check.json, {RV}/vllm_run_summary.json (apc_log_enabled true), sampling and qualitative verdicts from the same caching-enabled server ({RV}/sampling_tests.log, {RV}/vllm_qualitative_outputs.json); continuation prefill compiles per start offset after warmup ({VLLM}/README.md limitations)",
    },
    "max_device_top_k": {
        "status": "supported",
        "evidence": f"32 (SamplingGenerator max_top_k; tt/generator.py MAX_TOP_K); the checkpoint's generation_config.json defaults (temperature 0.6, top_k 20, top_p 0.95) stay below the ceiling and route to the device sampler ({RV}/server.log 'overridden by the model's generation_config.json'); no out-of-vocabulary sampled id in any served run ({VLLM}/work_log.md)",
    },
    "max_num_seqs up to 32": {
        "status": "supported",
        "evidence": f"served with --max-num-seqs 32: {RV}/vllm_ci_serving_benchmark.json (32 prompts), {RV}/vllm_prefill_burst_benchmark.json (32 concurrent), {RV}/sampling_tests.log (--tt-max-num-seqs 32); generator level: doc/full_model/batch_decode_32.json",
    },
    "prompt lengths not divisible by chunk/tile/page sizes": {
        "status": "supported",
        "evidence": f"served: {RV}/mixed_shape_trace_reuse.json (129, 130, 131, 5, 33, 257, 1025, 4097 and 5000-token prompts), {RV}/prefix_cache_check.json (2305 tokens), the qualitative chat prompts; generator level: doc/full_model/multi_row_prefill.json",
    },
    "max_model_len 40960": {
        "status": "supported",
        "evidence": f"served with --max_model_len 40960 ({RV}/vllm_run_summary.json, {RV}/server.log); doc/context_contract.json (40960 retained)",
    },
}


def main() -> None:
    data = json.loads(PATH.read_text())
    seen = set()
    for row in data["rows"]:
        item = row.get("item")
        if item in UPDATES:
            update = UPDATES[item]
            row["status"] = update["status"]
            row["evidence"] = update["evidence"]
            row.pop("declaration", None)
            row.pop("reason", None)
            seen.add(item)
    missing = set(UPDATES) - seen
    if missing:
        raise SystemExit(f"rows not found: {missing}")
    data.setdefault("vllm_integration", {})
    data["vllm_integration"] = {
        "registration": "EXTRA_MODELS_DIR bundle models/autoports/qwen_qwen3_8b/vllm/qwen3_8b/vllm_metadata.json -> TTQwen3ForCausalLM = models.autoports.qwen_qwen3_8b.tt.generator_vllm:Qwen3ForCausalLM (read before the plugin's built-in map)",
        "serving_stack_deviation": "deprecated fork lineage used unchanged on the operator's instruction; see doc/vllm_integration/work_log.md section 2",
        "server_command": "doc/vllm_integration/logs/serve.sh (in-tree runner, --enable-prefix-caching --async-scheduling, VLLM_SERVER_DEV_MODE=1)",
    }
    PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {len(seen)} rows")


if __name__ == "__main__":
    main()
