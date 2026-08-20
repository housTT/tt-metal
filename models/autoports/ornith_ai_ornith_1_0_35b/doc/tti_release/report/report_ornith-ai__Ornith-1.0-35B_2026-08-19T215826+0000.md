## Tenstorrent Model Release Summary: ornith-ai/Ornith-1.0-35B on P300X2

### Metadata: ornith-ai/Ornith-1.0-35B on P300X2

```json
{
    "model_name": "ornith-ai/Ornith-1.0-35B",
    "device": "P300X2",
    "generated_at": "2026-08-19T21:58:26+00:00",
    "report_id": "ornith-ai__Ornith-1.0-35B_2026-08-19T215826+0000",
    "workflow": "release",
    "server_mode": "API",
    "run_command": "python run.py --model Ornith-1.0-35B --runtime-model-spec-json /home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/ornith_autoport_release_spec.json --tt-device p300x2 --workflow release --service-port 8100 --no-auth --skip-system-sw-validation --limit-samples-mode ci-nightly",
    "runtime_model_spec_json": "/home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/tti_cache_release/workflow_logs/runtime_model_specs/runtime_model_spec_2026-08-19_20-19-16_id_ornith-1-0-35b-autoport_Ornith-1.0-35B_p300x2_yLM6S8pX.json",
    "model_id": "id_ornith-1-0-35b-autoport_Ornith-1.0-35B_p300x2",
    "model_repo": "ornith-ai/Ornith-1.0-35B",
    "inference_engine": "vLLM",
    "tt_metal_commit": "278e0ba",
    "vllm_commit": "5380fd4",
    "model_impl": "ornith-1-0-35b-autoport"
}
```

### Acceptance Criteria

- Acceptance status: ✅ `PASS`
- Model status: `EXPERIMENTAL`
- Benchmarks: 🟨 `NA` (0/20 passed, 20 NA)
- Evals: 🟨 `NA` (0/2 passed, 2 NA)
- Spec Tests: 🟨 `NA` (no blocks present)
- All acceptance criteria passed.

---

### Accuracy Evaluations for ornith-ai/Ornith-1.0-35B on P300X2

| Task            | Tolerance | gpu_reference_score_ref                                                                                     | Score | Ratio to Published | Ratio to Reference | Accuracy Check |
|:----------------|:----------|:------------------------------------------------------------------------------------------------------------|:------|:-------------------|:-------------------|:---------------|
| ifeval          | 0.05      | none: no published IFEval score on the Ornith-1.0-35B model card and no GPU reference run for this autoport | 82.08 | N/A                | N/A                | 🟨 NA          |
| r1_gpqa_diamond | 0.05      | none: no published GPQA score on the Ornith-1.0-35B model card and no GPU reference run for this autoport   | 50    | N/A                | N/A                | 🟨 NA          |

Note: The ratio to published scores defines if eval ran roughly correctly, as the exact methodology of the model publisher cannot always be reproduced. For this reason the accuracy check is based first on being equivalent to the GPU reference within a +/- tolerance. If a value GPU reference is not available, the accuracy check is based on the direct ratio to the published score.

---

### vLLM Benchmark for ornith-ai/Ornith-1.0-35B on P300X2

| Concurrency | Num Requests | ISL    | OSL  | TTFT (ms) | P50 TTFT (ms) | P99 TTFT (ms) | TPOT (ms) | E2EL (ms) | Tput Decode (TPS) | Req Tput (RPS) |
|:------------|:-------------|:-------|:-----|:----------|:--------------|:--------------|:----------|:----------|:------------------|:---------------|
| 1           | 8            | 128    | 128  | 157.3     | 157.3         | 160.7         | 138.2     | 17704.9   | 7.2               | 0.057          |
| 32          | 256          | 128    | 128  | 4494.6    | 4625.3        | 4774.2        | 144.6     | 22855.8   | 179.2             | 1.400          |
| 1           | 4            | 128    | 1024 | 163.4     | 161.8         | 168.9         | 138.3     | 141621.3  | 7.2               | 0.007          |
| 32          | 128          | 128    | 1024 | 4454.1    | 4593.2        | 4602.4        | 142.8     | 150580.1  | 217.6             | 0.212          |
| 1           | 4            | 1024   | 128  | 437.6     | 434.0         | 452.5         | 138.5     | 18023.0   | 7.1               | 0.056          |
| 32          | 128          | 1024   | 128  | 13117.5   | 13529.6       | 13578.8       | 146.3     | 31702.5   | 129.2             | 1.009          |
| 1           | 4            | 2048   | 128  | 813.4     | 812.6         | 845.7         | 138.7     | 18434.1   | 6.9               | 0.054          |
| 32          | 128          | 2048   | 128  | 24555.6   | 25574.2       | 25872.5       | 152.4     | 43905.7   | 93.3              | 0.729          |
| 1           | 4            | 4096   | 128  | 1618.3    | 1615.2        | 1686.4        | 139.4     | 19326.4   | 6.6               | 0.052          |
| 32          | 128          | 4096   | 128  | 49739.3   | 51144.2       | 51560.3       | 156.5     | 69615.3   | 58.8              | 0.460          |
| 1           | 2            | 8192   | 128  | 3190.2    | 3190.2        | 3244.9        | 140.7     | 21056.6   | 6.1               | 0.048          |
| 31          | 62           | 8192   | 128  | 90829.0   | 100043.0      | 100275.3      | 220.0     | 118763.0  | 33.4              | 0.261          |
| 1           | 2            | 16384  | 128  | 6482.7    | 6482.7        | 6606.4        | 143.3     | 24679.4   | 5.2               | 0.041          |
| 16          | 32           | 16384  | 128  | 99203.6   | 104931.4      | 106196.4      | 255.5     | 131649.3  | 14.8              | 0.115          |
| 1           | 1            | 32768  | 128  | 13682.1   | 13682.1       | 13682.1       | 148.5     | 32545.4   | 3.9               | 0.031          |
| 8           | 8            | 32768  | 128  | 98712.4   | 110795.2      | 110797.2      | 244.8     | 129803.1  | 7.9               | 0.062          |
| 1           | 1            | 65536  | 128  | 29351.3   | 29351.3       | 29351.3       | 158.9     | 49526.2   | 2.6               | 0.020          |
| 4           | 4            | 65536  | 128  | 94024.2   | 115575.2      | 115577.1      | 329.0     | 135806.3  | 3.8               | 0.029          |
| 1           | 1            | 131072 | 128  | 66597.6   | 66597.6       | 66597.6       | 179.5     | 89395.8   | 1.4               | 0.011          |
| 2           | 2            | 131072 | 128  | 99346.0   | 99346.0       | 131110.6      | 435.0     | 154590.4  | 1.7               | 0.013          |

Note: Columns without a percentile label (e.g. P50, P95, P99) report the mean value across the benchmark run.

Note: No perf targets are configured for these sweep points, so these rows are reported for information only and are not graded.
