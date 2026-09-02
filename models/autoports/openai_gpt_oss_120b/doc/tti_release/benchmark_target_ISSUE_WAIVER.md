# GPT-OSS-120B P150x4 benchmark target issue waiver

Date: 2026-09-02

Status: current Stage 11 release note; applies only to the TTI benchmark row
`ISL=128, OSL=128, max_concurrency=1, num_prompts=8`.

## Classification

`issue-waived` — the TTI source attaches an internally inconsistent aggregate
throughput target to a concurrency-one row. This waiver does not apply to
request failures, missing metrics, shortened inputs or outputs, other rows, or
accuracy/API gates.

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

## Control and target defect

The completed optimized-vLLM autoport control used the same model, hardware,
endpoint family, logical lengths, concurrency, and temperature. It measured
508.22 ms TTFT and 46.966 tok/s in the saved JSON; the final optimized-vLLM
headline rerun measured 47.316 tok/s. The repaired TTI row therefore reproduces
the current autoport baseline within normal run variance.

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

Source evidence is
`reference_config/benchmarking/benchmark_targets/model_performance_reference.json`
in the TTI checkout at base commit
`f07a31d2a2f908aa04098685034e7a5bde7554ea`. The optimized controls are in
`doc/optimized_vllm/artifacts/after_final_clean/vllm_benchmark.json` and
`doc/optimized_vllm/README.md`.

## Resolution boundary

The row's functional TTFT and per-user throughput checks pass. Only target
fields derived from the malformed aggregate reference remain failed. The row
is issue-waived until TTI splits the B1 latency/per-user reference from the B32
aggregate-throughput reference or replaces the aggregate target with a valid
B1 value. All other 20 benchmark rows are ungraded (`NA`) and have complete,
non-missing metrics plus exact raw count/length evidence; they are not claimed
as target-qualified performance rows.

This is the linked waiver evidence for the final merged report and
`RUN_NOTES.md`. It does not claim unrestricted performance readiness.
