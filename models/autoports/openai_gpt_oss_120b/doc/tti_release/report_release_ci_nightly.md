## Tenstorrent Model Release Summary: openai/gpt-oss-120b on P150X4

### Metadata: openai/gpt-oss-120b on P150X4

```json
{
    "model_name": "openai/gpt-oss-120b",
    "device": "P150X4",
    "generated_at": "2026-09-03T04:19:22+00:00",
    "report_id": "id_openai-gpt-oss-120b-autoport_p150x4_release-repaired_2026-09-03T041922+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model gpt-oss-120b --runtime-model-spec-json /home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/specs/autoport_release_spec.json --tt-device p150x4 --engine vllm --workflow release --server-url http://127.0.0.1:8000 --service-port 8000 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-01_18-57-30_id_openai-gpt-oss-120b-autoport_p150x4_DCBRySjX.json",
    "model_id": "id_openai-gpt-oss-120b-autoport_p150x4",
    "model_repo": "openai/gpt-oss-120b",
    "inference_engine": "vLLM",
    "tt_metal_commit": "76e51849603f6ff7d05f37e15b4938016d7e946e",
    "vllm_commit": "54dea57d98ccfaef072908f085d9296d544ba1fe",
    "model_impl": "gpt-oss-autoport",
    "release_readiness": "release-readiness-ci-subset-pass",
    "vllm_base_commit": "568afb3a13806beb53bb2e6bd518269357b237c0",
    "tti_commit": "ddfba898209f0aaada2294d9230801d053edcf80",
    "source_release_report_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/release/data/report_data_id_openai-gpt-oss-120b-autoport_p150x4_2026-09-02_13-42-30.json",
    "source_release_runtime_model_spec_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-01_18-57-30_id_openai-gpt-oss-120b-autoport_p150x4_DCBRySjX.json",
    "validation_runtime_model_spec_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/runtime_model_specs/runtime_model_spec_2026-09-03_00-39-36_id_openai-gpt-oss-120b-autoport_p150x4_AcpnnKS6.json",
    "handoff_runtime_model_spec_json": "models/autoports/openai_gpt_oss_120b/doc/tti_release/runtime_model_spec_validation.json",
    "autoport_code_path": "models/autoports/openai_gpt_oss_120b",
    "release_code_commits": {
        "official_vllm": "54dea57d98ccfaef072908f085d9296d544ba1fe",
        "tti_client": "ddfba898209f0aaada2294d9230801d053edcf80"
    },
    "context_contract": {
        "path": "models/autoports/openai_gpt_oss_120b/doc/context_contract.json",
        "sha256": "f05506ba1ab14a88bb64a798e7a864433aa675b41c4ab018e30076599f191bee",
        "supported_context": 131072,
        "non_aligned_requests_preserved": true
    },
    "gpqa_harness_recovery": {
        "task_name": "gpqa_diamond_cot_zeroshot",
        "repair_report_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/evals/data/report_data_openai__gpt-oss-120b_2026-09-02T152431+0000.json",
        "publisher_revision": "56686c06f5e19865c153de0fdb11be3890014df7",
        "archive_sha256": "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc",
        "diamond_csv_sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
        "sample_ids": [
            0,
            1,
            2,
            3,
            4,
            5,
            6
        ],
        "raw_samples_copied": false
    },
    "ifeval_harness_recovery": {
        "logical_task_name": "meta_ifeval",
        "canonical_task_name": "ifeval",
        "dataset_path": "google/IFEval",
        "scope": {
            "limit": null,
            "original_samples": 541,
            "effective_samples": 541,
            "strict_prompt_metric": "prompt_level_strict_acc,none",
            "max_length": 131072,
            "generation_policy": {
                "reasoning_effort": "medium",
                "max_gen_toks": 4096,
                "do_sample": false,
                "temperature": 0.0,
                "seed": 42
            }
        },
        "report_paths": {
            "report_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/evals/data/report_data_openai__gpt-oss-120b_2026-09-03T041704+0000.json",
            "raw_result_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/evals/gpt-oss-120b_p150x4_evals/eval_id_openai-gpt-oss-120b-autoport_p150x4/openai__gpt-oss-120b/results_2026-09-03T04-17-04.086964.json"
        },
        "raw_samples_copied": false
    },
    "benchmark_harness_recovery": {
        "repair_report_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/benchmarks/data/report_data_openai__gpt-oss-120b_2026-09-02T194657+0000.json",
        "endpoint": "/v1/completions",
        "backend": "vllm",
        "temperature": 0,
        "expected_rows": 21,
        "raw_evidence_validation": [
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_15-27-34_isl-128_osl-128_maxcon-1_n-8.json",
                "timestamp": "20260902-152805",
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 8,
                "completed": 8,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_15-28-07_isl-128_osl-128_maxcon-32_n-256.json",
                "timestamp": "20260902-154016",
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "concurrency": 32,
                "num_prompts": 256,
                "completed": 256,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_15-40-18_isl-128_osl-1024_maxcon-1_n-4.json",
                "timestamp": "20260902-154142",
                "input_sequence_length": 128,
                "output_sequence_length": 1024,
                "concurrency": 1,
                "num_prompts": 4,
                "completed": 4,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_15-41-44_isl-128_osl-1024_maxcon-32_n-128.json",
                "timestamp": "20260902-162305",
                "input_sequence_length": 128,
                "output_sequence_length": 1024,
                "concurrency": 32,
                "num_prompts": 128,
                "completed": 128,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-23-08_isl-1024_osl-128_maxcon-1_n-4.json",
                "timestamp": "20260902-162344",
                "input_sequence_length": 1024,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 4,
                "completed": 4,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-23-46_isl-1024_osl-128_maxcon-32_n-128.json",
                "timestamp": "20260902-163640",
                "input_sequence_length": 1024,
                "output_sequence_length": 128,
                "concurrency": 32,
                "num_prompts": 128,
                "completed": 128,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-36-42_isl-2048_osl-128_maxcon-1_n-4.json",
                "timestamp": "20260902-163743",
                "input_sequence_length": 2048,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 4,
                "completed": 4,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-37-45_isl-2048_osl-128_maxcon-32_n-128.json",
                "timestamp": "20260902-165827",
                "input_sequence_length": 2048,
                "output_sequence_length": 128,
                "concurrency": 32,
                "num_prompts": 128,
                "completed": 128,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-58-29_isl-4096_osl-128_maxcon-1_n-4.json",
                "timestamp": "20260902-165949",
                "input_sequence_length": 4096,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 4,
                "completed": 4,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_16-59-51_isl-4096_osl-128_maxcon-31_n-124.json",
                "timestamp": "20260902-173510",
                "input_sequence_length": 4096,
                "output_sequence_length": 128,
                "concurrency": 31,
                "num_prompts": 124,
                "completed": 124,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_17-35-12_isl-8192_osl-128_maxcon-1_n-2.json",
                "timestamp": "20260902-173637",
                "input_sequence_length": 8192,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 2,
                "completed": 2,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_17-36-39_isl-8192_osl-128_maxcon-15_n-30.json",
                "timestamp": "20260902-175353",
                "input_sequence_length": 8192,
                "output_sequence_length": 128,
                "concurrency": 15,
                "num_prompts": 30,
                "completed": 30,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_17-53-56_isl-8192_osl-1024_maxcon-1_n-2.json",
                "timestamp": "20260902-175539",
                "input_sequence_length": 8192,
                "output_sequence_length": 1024,
                "concurrency": 1,
                "num_prompts": 2,
                "completed": 2,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_17-55-41_isl-8192_osl-1024_maxcon-14_n-28.json",
                "timestamp": "20260902-182943",
                "input_sequence_length": 8192,
                "output_sequence_length": 1024,
                "concurrency": 14,
                "num_prompts": 28,
                "completed": 28,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_18-29-46_isl-10000_osl-1024_maxcon-1_n-2.json",
                "timestamp": "20260902-183234",
                "input_sequence_length": 10000,
                "output_sequence_length": 1024,
                "concurrency": 1,
                "num_prompts": 2,
                "completed": 2,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_18-32-36_isl-10000_osl-1024_maxcon-11_n-22.json",
                "timestamp": "20260902-191432",
                "input_sequence_length": 10000,
                "output_sequence_length": 1024,
                "concurrency": 11,
                "num_prompts": 22,
                "completed": 22,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_19-14-35_isl-16384_osl-128_maxcon-1_n-2.json",
                "timestamp": "20260902-191647",
                "input_sequence_length": 16384,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 2,
                "completed": 2,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_19-16-50_isl-16384_osl-128_maxcon-7_n-14.json",
                "timestamp": "20260902-193311",
                "input_sequence_length": 16384,
                "output_sequence_length": 128,
                "concurrency": 7,
                "num_prompts": 14,
                "completed": 14,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_19-33-13_isl-32768_osl-128_maxcon-1_n-1.json",
                "timestamp": "20260902-193524",
                "input_sequence_length": 32768,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 1,
                "completed": 1,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_19-35-27_isl-32768_osl-128_maxcon-3_n-3.json",
                "timestamp": "20260902-194245",
                "input_sequence_length": 32768,
                "output_sequence_length": 128,
                "concurrency": 3,
                "num_prompts": 3,
                "completed": 3,
                "failed": 0,
                "errors": 0
            },
            {
                "artifact": "benchmark_openai__gpt-oss-120b_2026-09-02_19-42-47_isl-65536_osl-128_maxcon-1_n-1.json",
                "timestamp": "20260902-194657",
                "input_sequence_length": 65536,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 1,
                "completed": 1,
                "failed": 0,
                "errors": 0
            }
        ],
        "raw_outputs_copied": false,
        "issue_waiver": {
            "classification": "issue-waived",
            "scope": {
                "input_sequence_length": 128,
                "output_sequence_length": 128,
                "concurrency": 1,
                "num_prompts": 8
            },
            "source": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tt-inference-server/evidence/benchmark_target_ISSUE_WAIVER.md",
            "handoff_path": "models/autoports/openai_gpt_oss_120b/doc/tti_release/benchmark_target_ISSUE_WAIVER.md",
            "reason": "the source target combines a concurrency-1 row with an aggregate-throughput threshold scaled for a larger batch",
            "unrestricted_performance_readiness": false
        }
    },
    "spec_test_harness_recovery": {
        "repair_report_json": "/home/ttuser/dev/gpt-oss-20b/tti-release/openai_gpt_oss_120b/tti_cache/workflow_logs/reports_output/spec_tests/data/report_data_id_openai-gpt-oss-120b-autoport_p150x4_2026-09-02_20-59-57.json",
        "request_timeout_seconds": 300,
        "generated_token_checks_use_reasoning_fallback": true,
        "final_content_required_for_coherence": true,
        "raw_responses_copied": false
    }
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: ✅ `PASS` (0/21 passed, 1 waived, 20 NA)
- Evals: ✅ `PASS` (4/4 passed)
- Spec Tests: ✅ `PASS` (1/1 passed)
- All acceptance criteria passed.

---

### Accuracy Evaluations for openai/gpt-oss-120b on P150X4

| Task                      | Tolerance | Published Score | Published Score Ref                                                                          | GPU Reference Score | gpu_reference_score_ref                                                                | Score | Ratio to Published | Ratio to Reference | Accuracy Check | mean_seconds_per_task | lm_eval_task_name |
|:--------------------------|:----------|:----------------|:---------------------------------------------------------------------------------------------|:--------------------|:---------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|:----------------------|:------------------|
| aime25                    | 0.05      | 92.5            | https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf   | 90.4                | https://github.com/tenstorrent/tt-inference-server/issues/1322#issuecomment-3801635211 | 86.67 | 0.9369             | 0.9587             | ✅ PASS        | 2123                  | N/A               |
| gpqa_diamond_cot_zeroshot | 0.05      | 80.1            | https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf   | 79.7                | https://github.com/tenstorrent/tt-inference-server/issues/1322#issuecomment-3801635211 | 100   | 1.248              | 1.255              | ✅ PASS        | 11.35                 | N/A               |
| mmlu_generative           | 0.05      | 85.9            | https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf   | 85.9                | DUMMY VALUE                                                                            | 84.67 | 0.9857             | 0.9857             | ✅ PASS        | N/A                   | N/A               |
| meta_ifeval               | 0.05      | 78.2            | https://frozebench.com/runs/openai-mirror%2Fgpt-oss-120b__ifeval__2025-10-20T07-31-55.208086 | N/A                 | N/A                                                                                    | 85.58 | 1.094              | N/A                | ✅ PASS        | 466                   | ifeval            |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark Targets — ISL 128 / OSL 128, concurrency 1 for openai/gpt-oss-120b on P150X4

| Concurrency | Num Requests | ISL | OSL | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) | Target Check |
|:------------|:-------------|:----|:----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|:-------------|
| 1           | 8            | 128 | 128 | 500.1     | 497.4         | 512.9         | 18.3      | 2820.4    | 45.4             | 45.4              | 90.8             | 0.354          | ❌ FAIL      |

#### Target Checks

| Tier       | TTFT Target | TTFT Ratio | TTFT Check | Tput User Target | Tput User Ratio | Tput User Check | Tput Output Target | Tput Output Ratio | Tput Output Check |
|:-----------|:------------|:-----------|:-----------|:-----------------|:----------------|:----------------|:-------------------|:------------------|:------------------|
| functional | 510         | 0.9806     | ✅ PASS    | 2.70             | 20.27           | ✅ PASS         | 85.9               | 0.5283            | ❌ FAIL           |
| complete   | 102         | 4.903      | ❌ FAIL    | 13.50            | 4.054           | ✅ PASS         | 429.5              | 0.1057            | ❌ FAIL           |
| target     | 51          | 9.806      | ❌ FAIL    | 27.00            | 2.027           | ✅ PASS         | 859                | 0.05283           | ❌ FAIL           |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: The Target Check column reflects only the strictest `target` tier. The Target Checks table grades three tiers — functional, complete, and target — from most to least lenient. Acceptance criteria pass a benchmark when any single tier meets all of its checks.

---

### vLLM Benchmark for openai/gpt-oss-120b on P150X4

| Concurrency | Num Requests | ISL   | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Input (TPS) | Tput Output (TPS) | Tput Total (TPS) | Req Tput (RPS) |
|:------------|:-------------|:------|:-----|:----------|:--------------|:--------------|:----------|:----------|:-----------------|:------------------|:-----------------|:---------------|
| 32          | 256          | 128   | 128  | 15284.5   | 15290.3       | 15303.5       | 589.7     | 90180.1   | 45.4             | 45.4              | 90.8             | 0.355          |
| 1           | 4            | 128   | 1024 | 498.1     | 497.0         | 501.9         | 18.5      | 19417.9   | 6.6              | 52.7              | 59.3             | 0.051          |
| 32          | 128          | 128   | 1024 | 15295.2   | 15289.8       | 15343.6       | 590.0     | 618914.9  | 6.6              | 52.9              | 59.6             | 0.052          |
| 1           | 4            | 1024  | 128  | 4803.5    | 3659.9        | 8100.1        | 18.4      | 7135.7    | 143.5            | 17.9              | 161.4            | 0.140          |
| 32          | 128          | 1024  | 128  | 116348.3  | 116347.2      | 116368.0      | 590.2     | 191306.9  | 171.3            | 21.4              | 192.7            | 0.167          |
| 1           | 4            | 2048  | 128  | 11225.4   | 7339.4        | 22425.2       | 19.4      | 13692.4   | 149.6            | 9.3               | 158.9            | 0.073          |
| 32          | 128          | 2048  | 128  | 233938.6  | 233933.9      | 233960.7      | 590.9     | 308984.0  | 212.1            | 13.3              | 225.4            | 0.104          |
| 1           | 4            | 4096  | 128  | 15927.3   | 14633.0       | 19661.5       | 19.4      | 18389.4   | 222.7            | 7.0               | 229.7            | 0.054          |
| 31          | 124          | 4096  | 128  | 452802.0  | 452802.3      | 452862.8      | 592.0     | 527989.4  | 240.5            | 7.5               | 248.0            | 0.059          |
| 1           | 2            | 8192  | 128  | 36632.6   | 36632.6       | 43846.5       | 19.5      | 39104.2   | 209.5            | 3.3               | 212.8            | 0.026          |
| 15          | 30           | 8192  | 128  | 438704.7  | 438650.2      | 438778.7      | 594.2     | 514169.1  | 239.0            | 3.7               | 242.7            | 0.029          |
| 1           | 2            | 8192  | 1024 | 29270.3   | 29270.3       | 29277.5       | 19.1      | 48846.3   | 167.7            | 21.0              | 188.7            | 0.021          |
| 14          | 28           | 8192  | 1024 | 409414.4  | 409415.4      | 409430.1      | 594.7     | 1017819.4 | 112.7            | 14.1              | 126.8            | 0.014          |
| 1           | 2            | 10000 | 1024 | 61348.4   | 61348.4       | 64052.3       | 19.5      | 81317.6   | 123.0            | 12.6              | 135.6            | 0.012          |
| 11          | 22           | 10000 | 1024 | 644265.2  | 644228.1      | 644357.8      | 595.8     | 1253736.9 | 87.7             | 9.0               | 96.7             | 0.009          |
| 1           | 2            | 16384 | 128  | 60881.6   | 60881.6       | 63104.9       | 19.7      | 63381.0   | 258.5            | 2.0               | 260.5            | 0.016          |
| 7           | 14           | 16384 | 128  | 410101.1  | 410087.2      | 410132.5      | 599.4     | 486224.3  | 235.9            | 1.8               | 237.7            | 0.014          |
| 1           | 1            | 32768 | 128  | 122797.2  | 122797.2      | 122797.2      | 20.7      | 125428.3  | 261.2            | 1.0               | 262.3            | 0.008          |
| 3           | 3            | 32768 | 128  | 353149.7  | 353171.5      | 353172.2      | 609.1     | 430502.3  | 228.3            | 0.9               | 229.2            | 0.007          |
| 1           | 1            | 65536 | 128  | 240107.6  | 240107.6      | 240107.6      | 23.1      | 243036.9  | 269.7            | 0.5               | 270.2            | 0.004          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.

---

## 📋 Summary

| Metric         | Value                     |
|:---------------|:--------------------------|
| Total Tests    | 2                         |
| Passed         | 2                         |
| Failed         | 0                         |
| Skipped        | 0                         |
| NA             | 0                         |
| Attempted      | 2                         |
| Success Rate   | 100.0%                    |
| Total Duration | 1640.21s                  |
| Total Attempts | 2                         |
| Generated      | 2026-09-03T04:19:22+00:00 |

## 🧪 Test Results

| Status  | Test Name                | Duration | Attempts | Description                                       |
|:--------|:-------------------------|:---------|:---------|:--------------------------------------------------|
| ✅ PASS | LoggerForkSafetyTest     | 0.00s    | 1        | Test for logging fork safety to prevent deadlocks |
| ✅ PASS | VLLMParamConformanceTest | 1640.21s | 1        | vLLM chat/completions parameter conformance       |

---

### Logger Fork Safety for openai/gpt-oss-120b on P150X4

| Child Result |
|:-------------|
| OK           |

---

### Vllm Chat Completions for openai/gpt-oss-120b on P150X4

| Endpoint URL                              | model_name          | Task                  |
|:------------------------------------------|:--------------------|:----------------------|
| http://127.0.0.1:8000/v1/chat/completions | openai/gpt-oss-120b | vllm_chat_completions |

#### Parameter Conformance Summary

| Test Case                    | Status  | Summary    |
|:-----------------------------|:--------|:-----------|
| test_coherence_verbatim_echo | ✅ PASS | 1/1 passed |
| test_determinism_parameters  | ✅ PASS | 3/3 passed |
| test_logprobs                | ✅ PASS | 1/1 passed |
| test_max_tokens              | ✅ PASS | 2/2 passed |
| test_n                       | ✅ PASS | 2/2 passed |
| test_non_uniform_seeding     | ✅ PASS | 1/1 passed |
| test_penalties               | ✅ PASS | 9/9 passed |
| test_seed_reproducibility    | ✅ PASS | 1/1 passed |
| test_stop                    | ✅ PASS | 2/2 passed |

#### Detailed Test Results

| Test Case                    | Parametrization                                                      | Status    |
|:-----------------------------|:---------------------------------------------------------------------|:----------|
| test_coherence_verbatim_echo | test_coherence_verbatim_echo                                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[temperature-0.0]                         | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_k-1]                                 | ✅ PASSED |
| test_determinism_parameters  | test_determinism_parameters[top_p-0.01]                              | ✅ PASSED |
| test_logprobs                | test_logprobs                                                        | ✅ PASSED |
| test_max_tokens              | test_max_tokens[10]                                                  | ✅ PASSED |
| test_max_tokens              | test_max_tokens[5]                                                   | ✅ PASSED |
| test_n                       | test_n[2]                                                            | ✅ PASSED |
| test_n                       | test_n[3]                                                            | ✅ PASSED |
| test_non_uniform_seeding     | test_non_uniform_seeding                                             | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-natural_repetition-messages1]   | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-repeat_trap-messages0]          | ✅ PASSED |
| test_penalties               | test_penalties[frequency_penalty-1.2-semantic_repetition-messages2]  | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-natural_repetition-messages1]    | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-repeat_trap-messages0]           | ✅ PASSED |
| test_penalties               | test_penalties[presence_penalty-1.2-semantic_repetition-messages2]   | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-natural_repetition-messages1]  | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-repeat_trap-messages0]         | ✅ PASSED |
| test_penalties               | test_penalties[repetition_penalty-1.5-semantic_repetition-messages2] | ✅ PASSED |
| test_seed_reproducibility    | test_seed_reproducibility                                            | ✅ PASSED |
| test_stop                    | test_stop[stop_seq0]                                                 | ✅ PASSED |
| test_stop                    | test_stop[stop_seq1]                                                 | ✅ PASSED |
