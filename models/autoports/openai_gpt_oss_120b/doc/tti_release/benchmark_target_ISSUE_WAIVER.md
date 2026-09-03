# GPT-OSS-120B P150x4 benchmark target issue waiver

Date: 2026-09-02

Status: current Stage 11 release note; applies only to the TTI benchmark row
`ISL=128, OSL=128, max_concurrency=1, num_prompts=8`.

## Classification

`issue-waived` — exactly five target subchecks remain failed on this one row.
Three are derived from an internally inconsistent aggregate-throughput target;
two are genuine misses of the higher `complete` and `target` TTFT tiers. TTI's
acceptance policy treats all five tiers as informational for this model's
`EXPERIMENTAL` status. This waiver does not apply to request failures, missing
metrics, shortened inputs or outputs, other rows, or accuracy/API gates.

## Current repaired row

The no-Docker TTI repair used the generated autoport server and the same raw
completion workload as optimized-vLLM: `/v1/completions`, greedy temperature
zero, and exact 128 input plus 128 output tokens. Its authoritative raw result
completed 8/8 requests with zero failures/errors and measured:

- mean TTFT: 500.1041 ms;
- mean TPOT: 18.2700 ms;
- aggregate output throughput: 45.3798 tok/s;
- all realized input lengths: 128;
- all realized output lengths: 128.

Evidence:
`tti_cache/workflow_logs/reports_output/benchmarks/gpt-oss-120b_p150x4_benchmarks/llm/benchmark_openai__gpt-oss-120b_2026-09-02_15-27-34_isl-128_osl-128_maxcon-1_n-8.json`.

## Control and target defect

The completed optimized-vLLM autoport control used the same model, hardware,
endpoint family, logical lengths, concurrency, and temperature. It measured
508.22 ms TTFT and 46.966 tok/s in the saved JSON; the final optimized-vLLM
headline rerun measured 47.316 tok/s. The repaired TTI row therefore reproduces
the current autoport baseline within normal run variance.

Evidence:

- `models/autoports/openai_gpt_oss_120b/doc/optimized_vllm/artifacts/after_final_clean/vllm_benchmark.json`;
- `models/autoports/openai_gpt_oss_120b/doc/optimized_vllm/README.md`.

TTI's current source entry assigns all of these values to the same
concurrency-one record:

- `tput_user=27` tok/s;
- aggregate `tput=859` tok/s;
- functional aggregate threshold `85.9` tok/s.

At concurrency one, aggregate output throughput and per-user output throughput
must be approximately equal. The 859/27 ratio is about 31.8, showing that the
aggregate value is a roughly B32 target attached to the B1 row. Even the
functional aggregate threshold is 1.89x the verified optimized B1 control.
Changing the autoport cannot make a concurrency-one aggregate result satisfy a
B32 aggregate target while preserving the requested workload.

The exact failed checks are:

- `functional.tput`: 45.3798 tok/s versus 85.9 tok/s;
- `complete.tput`: 45.3798 tok/s versus 429.5 tok/s;
- `target.tput`: 45.3798 tok/s versus 859 tok/s;
- `complete.ttft`: 500.1041 ms versus 102 ms;
- `target.ttft`: 500.1041 ms versus 51 ms.

The three aggregate-throughput checks inherit the malformed B32-like source
target described above. The two TTFT checks do not: they are genuine misses of
aspirational higher-performance tiers and are disclosed rather than reclassified
as a source-data error. Functional TTFT passes at 500.1041 ms versus its 510 ms
threshold, and all three per-user throughput checks pass.

Source evidence:
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json`
in the TTI checkout at commit
`f07a31d2a2f908aa04098685034e7a5bde7554ea`.

## Resolution boundary

The row's functional TTFT and all per-user throughput checks pass. The five
failures enumerated above are informational under TTI's committed
`ModelStatusTypes.EXPERIMENTAL` policy, whose `required_target_tiers` is empty;
they are not silently converted to passes. The row is issue-waived only for
this Stage 11 CI-subset readiness decision. TTI should split the B1
latency/per-user reference from the B32 aggregate-throughput reference or
replace its aggregate target with a valid B1 value. The two higher TTFT tiers
remain unmet even after that source repair.

All other 20 benchmark rows are ungraded (`NA`) and have complete, non-missing
metrics plus exact raw count/length evidence; they are not claimed as
target-qualified performance rows.

This release note is the linked waiver evidence for the final merged report and
`RUN_NOTES.md`. It does not claim unrestricted performance readiness.
