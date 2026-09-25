## Tenstorrent Model Release Summary: Qwen/Qwen3.8-Flash-Next on P300

### Metadata: Qwen/Qwen3.8-Flash-Next on P300

```json
{
    "model_name": "Qwen/Qwen3.8-Flash-Next",
    "device": "P300",
    "generated_at": "2026-08-31T13:50:47+00:00",
    "report_id": "Qwen__Qwen3.8-Flash-Next_2026-08-31T135047+0000",
    "workflow": "benchmarks",
    "server_mode": "API",
    "run_command": "python run.py --workflow benchmarks --runtime-model-spec-json /home/ttuser/dev/qwen3.8-flash-next/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/tti_release/specs/smoke_runtime_model_spec.json",
    "runtime_model_spec_json": "/home/ttuser/dev/qwen3.8-flash-next/tti-release/qwen3_8_flash_next/tt-inference-server/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-31_09-50-33_id_autoport_Qwen3.8-Flash-Next_p300_smoke_dtBMUnul.json",
    "model_id": "id_autoport_Qwen3.8-Flash-Next_p300_smoke",
    "model_repo": "Qwen/Qwen3.8-Flash-Next",
    "inference_engine": "vLLM",
    "tt_metal_commit": "60f1562e8ecf709bd778cb86c32dbfa5a06f8cb1",
    "vllm_commit": "a48857ac68b17c31303e4809f348caaebbf10f74",
    "model_impl": "autoport-qwen-qwen3-8-flash-next"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `FUNCTIONAL`
- Benchmarks: 🟨 `NA` (0/1 passed, 1 NA)
- Evals: 🟨 `NA` (no blocks present)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### vLLM Benchmark Targets — ISL 8 / OSL 8, concurrency 1 for Qwen/Qwen3.8-Flash-Next on P300

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 1            | 8   | 8   | 1261.1    | 1261.1        | 1261.1        | 220.8     | 2806.8    | 2.8               | 0.356          |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Decode Target | Tput Decode Ratio | Tput Decode Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 6e+04       | 0.02102    | ✅ PASS    | 0.10             | 45.29           | ✅ PASS         | 0.1                | 28.5              | ✅ PASS           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.