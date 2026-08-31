# GPT-OSS 120B vLLM integration

## Serving result

**Primary P150x4 single-user serving, 1 request x (128 prompt -> 128 output),
concurrency 1:** **514.11 ms TTFT P50/P99** and **46.64 decode t/s/u** from
`21.44 ms` mean/P50/P99 TPOT. ITL P50/P99 is `20.72/34.22 ms`; output
throughput is `39.54 tok/s`. The request completed 128/128 tokens. These are the
headline vLLM numbers and were measured first on a clean server.

**Secondary CI serving burst, 32 requests x (100 prompt -> 100 output),
unbounded client admission, server `max-num-seqs=32`:** 32/32 completed; TTFT
P50/P99 `15.269/15.270 s`, TPOT mean/P50/P99
`589.40/589.39/589.59 ms`, ITL P50/P99 `589.39/595.41 ms`, and aggregate
output throughput **43.47 tok/s**. Its TPOT-derived `1.70 t/s/u` is not a
headline decode result because burst admission and interleaved work affect
TPOT.

Status: **serves through the shared official vLLM + standalone TT plugin path
on P150x4**. The final full sampling profile passed **73 passed, 1 skipped** in
`1502.77 s`. P150 and P150x2 fail the resident-model physical-capacity gate
even at zero context (`69.752 GiB/device` and `36.575 GiB/device` required,
respectively). P150x4 serves the unreduced checkpoint/context-contract length
of **131,072 tokens**.

## Final configuration

| Item | Value |
| --- | --- |
| Model revision | `openai/gpt-oss-120b` / `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` |
| Hardware | four Blackhole p300c boards as P150x4 mesh `(1, 4)` |
| Registry alias | `TTGptOss120BForCausalLM` -> `tt/generator_vllm.py:TTGptOssForCausalLM` |
| Resident layers | 36/36 |
| `max_model_len` | 131072, identical to `doc/context_contract.json` |
| `max-num-seqs` | 32 |
| KV cache | vLLM-owned hybrid paged cache; page size 64; BFP8 |
| Scheduling | async; adapter declares `supports_async_decode=True` |
| Tracing | decode enabled with B1/B32 buckets; prefill eager |
| Sampling | `sample_on_device_mode=all`; canonical full-model split sampler/token-out feedback |
| Optional host mode | explicit compatibility route for unsupported shared-test parameters only |
| Fabric / trace region | `FABRIC_1D_RING` / 750,000,000 bytes |

The adapter translates vLLM scheduler state and delegates model construction,
prefill, decode, warmup, trace capture/replay, split sampling, and token-output
collection to the full-model generator in `tt/generator.py`. It has no
independent sampler, host argmax, generic top-k greedy fallback, full-logits
readback, or Python readback/writeback token-feedback loop. vLLM allocates and
owns the serving cache; the adapter binds those per-layer buffers directly and
does not construct a hidden standalone cache.

The selected `ds00_baseline` precision policy is consumed unchanged: BFP8
attention weights, projection inputs, attention CCL, LM head, and KV cache;
BFP4 expert weights; BF16 residual/router/norm weights, expert intermediates,
and expert CCL; LoFi decode attention/expert math, HiFi2 prefill
attention/router/LM-head math, and HiFi4 SDPA. There are no layer exceptions.
Policy SHA-256:
`b8e1e655581ffca37dd8b841285940c494a5e43d6b5deec3ebc10078ddb52c57`.

## Commands

Every device command first sourced `.agents/scripts/gpt_oss_workspace_env.sh`.
An import-origin check verified TTNN, vLLM, the plugin, and the adapter all
resolved below `/home/ttuser/dev/gpt-oss-20b`; device work stopped unless it
printed `ORIGIN_CHECK=PASS`. The official vLLM checkout remained unmodified.

```bash
VLLM_SYSTEM_START_DATE=2026-08-31 \
python_env/bin/python -m vllm.entrypoints.openai.api_server \
  --model openai/gpt-oss-120b --block-size 64 --max-num-seqs 32 --port 8000 \
  --max-model-len 131072 \
  --additional-config '{"tt":{"sample_on_device_mode":"all","trace_region_size":750000000,"fabric_config":"FABRIC_1D_RING"}}' \
  --async-scheduling --disable-log-stats \
  --structured-outputs-config '{"reasoning_parser":"openai_gptoss","enable_in_reasoning":false}'
```

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages sampling --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b --max-num-seqs 32 --sampling-profile full

python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages qualitative --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b

python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages benchmark --server-url http://127.0.0.1:8000 \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b --max-num-seqs 32 \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-num-requests 1 --benchmark-concurrency 1 \
  --benchmark-temperature 0 --ci-benchmark-prompt-len 100 \
  --ci-benchmark-output-len 100 --ci-benchmark-num-requests 32
```

The benchmark artifacts preserve the exact underlying `vllm bench serve`
commands. The primary measurement was rerun first on a clean server so no
prior wide serving workload influenced it.

## Correctness and qualitative result

The 65-token direct `/v1/completions` request returned HTTP 200 with 65 prompt
and eight completion tokens. It is divisible by none of 32, 64, or 128, proving
valid prompt lengths need not match tile, page, chunk, or trace sizes.

The full sampling profile covers greedy and stochastic sampling, seeds and
mixed batches, penalties, logprobs, bad/allowed token controls, optional host
compatibility parameters, full-capacity structured output, and async decode.
Adapter tests cover stale-token/current-position/page-table refresh and reuse,
trace bucket changes, cache ownership, and absence of a parallel sampling
path. GPT-OSS responses that legitimately use the separate reasoning field are
handled explicitly in shared tests; the measured path remains on-device.

All 12 greedy/sampled outputs from six prompts were read. They are coherent and
on-topic, with no pathological repetition, gibberish, wrong-language drift, or
request contamination. Longer responses remain coherent when cut at the
explicit 256-token limit. Low reasoning effort is explicit for this qualitative
workload so GPT-OSS reaches user-visible final content within that limit. The
machine degeneracy check also passed with no findings.

## Performance interpretation

The selected datatype-sweep teacher-forcing result is `61.04 t/s/u`; it is
reported only as a lower-bound reference because teacher forcing explicitly
refreshes caller-owned tokens and is not the serving token-out loop. The vLLM
result is not presented as equivalent to that different workload. The measured
vLLM decode path has no host sampler, host argmax, full-logits materialization,
or per-token Python feedback; remaining latency is the shared scheduler,
P150x4 model/cache, asynchronous collection, and canonical traced token-out
path rather than an avoidable adapter-side decode fallback.

## Evidence

- `readiness_vllm/vllm_benchmark.json`, `vllm_result.json`, and
  `vllm_benchmark.log`: primary summary, raw result, and command output.
- `readiness_vllm/vllm_ci_serving_benchmark.json`,
  `vllm_ci_serving_result.json`, and `vllm_ci_serving_benchmark.log`: secondary
  burst evidence.
- `readiness_vllm/sampling_tests.log`: final 73-pass full sampling profile.
- `readiness_vllm/vllm_qualitative_outputs.json`,
  `qualitative_verdict.json`, and `degenerate_check.json`: outputs and human/
  machine verdicts.
- `readiness_vllm/nonaligned_prompt_request.json`: exact 65-token gate.
- `readiness_vllm/vllm_serving_capability.json`: static cache, precision,
  trace, and capability snapshot. Request-path behavior is backed by the
  sampling/benchmark logs and adapter/plugin tests.
- `readiness_vllm/final_server.log` and `final_benchmark_server.log`: final
  sampling/qualitative and clean benchmark server logs.
- `readiness_vllm/runtime_cleanup_audit.md`: fallback, shutdown, process, and
  post-run device-health audit.
- `readiness_vllm/final_origin_check.log`: final import-origin proof for
  TTNN, official vLLM, standalone plugin, and the GPT-OSS 120B adapter.
- `readiness_vllm/final_process_audit.txt` and
  `final_tt_smi_status.txt`: final no-leftover-process and device-health
  snapshots.
- `readiness_vllm/adapter_readiness_host_tests.log` and
  `host_tests_after_format.log`: focused adapter/readiness host coverage and
  post-format sampling/trace allocation coverage.
- `readiness_vllm/plugin_host_tests.log`, `black_check_ttmetal.log`,
  `black_check_plugin.log`, `git_diff_check_ttmetal.log`, and
  `git_diff_check_plugin.log`: host plugin, formatting, and whitespace checks.
- `readiness_vllm/stage_09_check.log`: final multigoal stage check output.
- `readiness_vllm/configure_only_build.log.gz`: CI-image configure-only build
  evidence; the wrapper reported missing Garage credentials, so a full cold
  compile was not attempted.
- `doc/vllm_integration/stage_review.md`: independent stage-review finding,
  remediation, and clean-pass rereview record.

## Limitations

- Production serving requires P150x4; P150/P150x2 fail hard resident-state
  capacity checks rather than advertising a fictitious reduced context.
- Prefix caching is disabled; the alternating hybrid KV groups remain
  vLLM-owned.
- Host sampling is explicit and optional for compatibility tests. It is not
  used for the reported performance workload.
