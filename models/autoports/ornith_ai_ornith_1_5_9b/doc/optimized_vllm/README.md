# Optimized vLLM serving

**Primary vLLM TTFT:35.25ms; decode:87.70 tokens/s/user** for128 input/128
output/1 request, concurrency1 and max-num-seqs1, native context262144, TP4.
The same warmed workload before optimization measured49.96ms and87.88t/s/user:
TTFT improved29.4%; decode remained within0.3% of the current standalone traced
128-input/128-output/B1 result87.96t/s/user (29.01ms TTFT).

| Primary workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output t/s | Decode t/s/user |
| --- | --- | --- | --- | --- | --- |
| Before: 128 input /128 output /1 request, concurrency1, max-num-seqs1 | 49.956/49.956 | 11.379/11.379 | 11.360/11.656 | 85.598 | 87.884 |
| After: 128 input /128 output /1 request, concurrency1, max-num-seqs1 | 35.249/35.249 | 11.403/11.403 | 11.365/11.659 | 86.273 | 87.700 |

All rows use the same native262144 context, TP4 mesh, selected precision,
greedy sampling and TT configuration described below. Decode t/s/user is
1000/mean TPOT; aggregate throughput includes prefill. With one request, request
P50/P99 coincide; ITL quantiles describe its streaming intervals. These are
median-TTFT repeat selections, not independently selected best metrics.
[All repeats and exact artifacts](primary_comparison.json) and
[the current full-model comparison](final_full_model_tokenout.json) retain raw
precision/runtime and timing evidence. H1 alone measured35.04ms/87.68t/s/user
for the identical128/128/1 workload; H2's additional latency change is within
repeat variation, so its benefit is the required traced sampling and persistent
buffer contract, not a separately claimed speedup.

All stage correctness and performance gates pass. [Independent review](STAGE_REVIEW.md) returned `clean-pass`; local checkpoint receipts are recorded in [the work log](work_log.md).

The measured path is the shared `models.common.readiness_check.run_vllm_server`
runner, real TT plugin, and `tt/generator_vllm.py`, using all 32 layers and the
selected `head4_lofi_last8_c32_k4_r2` datatype policy. Hardware is four Blackhole
chips on two P300c boards, TP4 mesh `[1,4]`, software profile `P150x4`.

## Secondary CI serving capacity

| CI capacity workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output t/s | Secondary1000/TPOT t/s/user |
| --- | --- | --- | --- | --- | --- |
| Before:100 input/100 output/32 requests, unlimited admission, max-num-seqs32 | 2038.402/2039.860 | 75.466/88.498 | 67.641/305.210 | 338.607 | 13.251 |
| After:100 input/100 output/32 requests, unlimited admission, max-num-seqs32 | 2012.919/2014.100 | 75.366/88.242 | 67.666/302.043 | 339.863 | 13.269 |

Same native262144 context, TP4, selected precision and TT configuration as
the primary server, with max-num-seqs32 fixed for both CI measurements. All
32 requests completed all100 output tokens. These capacity/nightly-parity
metrics are secondary; their TPOT-derived rate is not the headline decode rate.
[CI repeats](ci_comparison.json) and [complete summary](perf_summary.json) retain
all measurements, including the diagnostic128/128/1 run at capacity32.

## Reproduction

Use the pinned task environment `../state/serving-env/bin/python` and checkpoint
`../upstream`, revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
[Dependency pins](../vllm_integration/serving_source_pins.json) and
[the dependency lock](../vllm_integration/serving_runtime_requirements.lock)
remain unchanged. From the checkout root:

```bash
USER=hous ../state/serving-env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/run_server.py \
  --label after_b1 --max-num-seqs 1 --async-scheduling

# In the same serialized serving lane, attach the benchmark client:
USER=hous ../state/serving-env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/benchmark_repeats.py \
  --label after_b1_metrics --max-num-seqs 1
```

Stop the runner with SIGTERM and wait for mesh closure before the next server.
For CI capacity use separate `after_b32`/`after_b32_metrics` labels and
`--max-num-seqs 32`. Each benchmark set has one discarded warmup and three
measured repeats. The headline selects median TTFT; every metric in that row
comes from the same repeat. Raw results retain individual streaming intervals.

All before/after servers use native `max_model_len=262144`, page64,
`sample_on_device_mode=all`, async scheduling, trace region100000000,
L1-small32768, `FABRIC_1D_RING`, and payload8192. Primary is greedy raw
completion128 input/128 output/1 request/concurrency1; CI is100 input/100
output/32 requests/unlimited admission. Performance servers disable host
sampling compatibility. The expanded commands, environments, and source hashes
are recorded in each `.command.json`.

## Changes and boundaries

The generator now reuses its B1 prefill trace with the vLLM-owned KV pool.
Startup preselects the128-token family before capture over an empty pool.
A different logical length retains ordinary model prefill and does not
rebuild or snapshot the live external pool. Non-aligned lengths remain valid.

First-token serving sampling now replays the same canonical split-sampling
trace used by full-model decode. Persistent logits staging handles transient
prefill outputs and recapture of the canonical logits buffer. Persistent
token/seed/history backups preserve untouched physical sampler lanes using
exact INT32 selection. Admission masks upload only when slot membership
changes. Full32 admission skips preservation because every physical lane is
admitted; partial/B1 admission preserves all untouched lanes. A late program-cache miss raises instead of choosing eager sampling.

Decode submits persistent model and sampling traces with `blocking=False`.
The plugin exercises `read_from_device=False`, receives a device token buffer,
queues a nonblocking copy of one replicated token shard, then waits and formats
that copy after the async boundary. Device feedback advances token, position,
and RoPE; stale host values do not overwrite them. Page tables refresh only on
changed scheduler contents. No full-logits readback or host argmax is used by
the measured device-sampling path.

Model math and selected precision are unchanged: BFP4/LoFi body and head,
BFP8/LoFi decode QKVG, layer31 BFP8 exceptions, BF16 residual/logits, native CCL,
FP32 recurrent state, and BFP8 paged KV. The terminal retains vocabulary-sharded
LMHead1D and canonical greedy split sampling; force-argmax and generic greedy
sampling were not introduced.

## Correctness and capability

- `h1_external_full32.json`: 18 exact traced/eager comparisons of complete
  logits, recurrent/conv state, and the next decode step; native external pool,
  resident128/131, changed tokens/pages, live shape misses and cache sentinels.
- `h2_sampling_b1.json` and `h2_sampling_b32_final.json`: real reduced-model prefill
  plus exact sampler-shape controls for greedy, seeded, penalties and mode
  transitions; partial/full admission, all-rank UINT32/history preservation,
  transient/canonical logits, recapture, persistent addresses and clean teardown.
- `after_async_worker_contract_v2.json`: deferred stale-input replay, changed and
  unchanged page tables, physical-page writes, slot remapping, and synchronous
  equivalence with zero steady-state token/position/RoPE refreshes.
- `after_native_capacity_v2.json`: all32 native external262143/262144 prefill,
  last-valid-position async decode, resident sampler/trace memory and clean
  trace teardown. The later all32-admission-only shortcut changes no B1 allocation.
- `after_b1_nonaligned.json` and `after_b32_nonaligned.json`: full-model API
 131/65 repeats and concurrent A/B/A match prior token controls.
- `h2_reduced_nonaligned.json`: real plugin requests of logical131 and65,
  repeats, and concurrent A/B/A. Reduced outputs prove the serving contract;
  they are not model-quality or headline-performance evidence.

Worker Watcher10 and native trace-allocation tracking include program caches
in device-contract runs. Ethernet watcher instrumentation is excluded after
the exact firmware-size failure and successful reset/list/mesh recovery
recorded in [the work log](work_log.md). Performance servers have no watcher
or allocation-tracker instrumentation.

## Sampling, cleanup and validation

The final canonical full sampling profile passed **72 tests with1 expected
skip** at max-num-seqs32/native262144/TP4. The skip is the framework API
capability test for all-vocabulary chat logprobs. This suite uses the explicit
host compatibility flag for host-only/logprob cases; headline benchmarks do
not. [Final test log](final_sampling.log), [runner command](final_sampling.runner.log)
and [server manifest](final_b32_compat.command.json) record the exact run.

The final B1/B32 benchmark servers and compatibility server all closed the mesh
and exited0. [Process audit](final_process_cleanup.json) found no vLLM or
EngineCore owner; the bounded device list returned all four chips with exit0.
The model-stage check and all hooks passed;87 focused CPU regressions passed.
Changes are Python/docs only, so no C++ build was needed.

## Qualitative behavior

The final shared suite uses the pinned chat template, six prompts and256-token
generation budgets, greedy0 and sampled0.7/top_p0.9, max-num-seqs1/native262144.
All six greedy texts exactly match the previous controlled integration output.
All twelve texts were read directly; the vLLM-scope degeneracy check passed.
The scope is serving regression and coherence. Several outputs exhaust the
budget during thinking; haiku syllable counting remains a controlled selected-
precision limitation. It is not a blanket task-accuracy pass.
[The final reading](qualitative_review.md), [hashed verdict](qualitative_review.json),
[prompt metadata](qualitative_prompt_format.json), and
[extended predecessor controls](../vllm_integration/qualitative_extended_control_report.md)
make those limits explicit.

## Scope and artifacts

No Tracy, tt-perf-report, device/adapter profiler, or ReadDeviceProfiler was
used. Device-time fields are unavailable by design. Earlier decoder/multichip
optimization evidence supplies device-operation context; current serving
benchmarks and contract checks determine this stage's result.

Explicit host compatibility remains available for canonical logprob and
host-only sampling tests using `--allow-host-sampling`. Such tests are separate
from performance measurements. Prefix caching, TP1/TP2, million-token YaRN,
and32 simultaneous native-length requests are not established by this stage.

[The checklist](checklist.md) maps relevant Optimize items to serving evidence;
[the work log](work_log.md) records hypotheses, rejected options, exact commands,
failed invocations and corrections, the independent `clean-pass`, and local
checkpoint receipts. No commits were pushed.


## Raw evidence restoration

Byte-sensitive logs, raw benchmark/qualitative JSON and the large all-rank
prefill hash report are preserved losslessly in a split gzip archive. Plain
originals remain locally; restore them before following raw-file links in a
fresh checkout. The split keeps every committed file below the repository's
500 KiB limit. All member and part hashes are in
[raw_evidence_core_manifest.json](raw_evidence_core_manifest.json).

```bash
cat models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/raw_evidence_core.tar.gz.part* | \
  tar -xz -C models/autoports/ornith_ai_ornith_1_5_9b
```

The final sampling/cleanup/check logs and matching selected readiness outputs
are in [raw_evidence_final.tar.gz](raw_evidence_final.tar.gz), verified by
[its manifest](raw_evidence_final_manifest.json). Restore this after the core
archive (and after any predecessor-stage archives):

```bash
tar -xzf models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/raw_evidence_final.tar.gz \
  -C models/autoports/ornith_ai_ornith_1_5_9b
```
