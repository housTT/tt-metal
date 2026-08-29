# Qwen3.8-Flash-Next optimized vLLM serving

Full vLLM sampling passes (72 passed, one expected all-vocabulary-logprobs
skip), both qualitative suites are coherent and non-degenerate, and the real
TT plugin path serves through `tt/generator_vllm.py`. On the primary warmed
single-user workload—P300 1x2, 128 input / 128 output / 1 request, concurrency
1, `max_num_seqs=2`, `max_model_len=262144`, explicit greedy temperature 0,
full sampling profile, `sample_on_device_mode=all`, decode-only trace—the
definitive optimized repeat measures **4484.084 ms TTFT** and **269.661 ms
TPOT / 3.708356 decode t/s/u**. The same retained source also measured
267.016 ms / 3.745088 t/s/u in an earlier exact repeat; the two-repeat mean is
268.339 ms / 3.726631 t/s/u. Relative to the 268.980 ms / 3.717743 t/s/u
baseline, single-user decode is neutral within -0.25% to +0.74% run variation.

## Primary single-user vLLM result

All numbers in this table are the same warmed 128-input / 128-output /
1-request / concurrency-1 workload and the frozen server configuration above.
TPOT-derived decode t/s/u is `1000 / mean TPOT`; aggregate output throughput
includes TTFT and is not substituted for decode t/s/u.

| Primary 128/128/1, concurrency 1 | TTFT p50/p99 (ms) | TPOT mean/p99 (ms) | ITL p50/p99 (ms) | Aggregate output (token/s) | TPOT decode (t/s/u) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 4007.856 / 4007.856 | 268.980 / 268.980 | 271.870 / 303.712 | 3.353537 | 3.717743 |
| **After, definitive final artifact** | **4484.084 / 4484.084** | **269.661 / 269.661** | **272.196 / 303.694** | **3.304815** | **3.708356** |
| After, same-source repeat A | 4146.631 / 4146.631 | 267.016 / 267.016 | 266.566 / 301.020 | 3.363287 | 3.745088 |

The definitive repeat is 0.25% lower in decode t/s/u and has 11.88% higher
TTFT than baseline; repeat A is 0.74% higher in decode t/s/u and 3.46% higher
in TTFT. The result is therefore not claimed as a single-user model-kernel
speedup. The accepted change is an async/host-service/memory improvement that
holds primary decode approximately flat while improving the secondary burst
and making the serving contract materially cleaner.

The comparable optimized full-model traced token-out workload is P300 1x2,
128 input / 128 output / batch 1 at 231.594 ms/token / 4.317901 t/s/u. The
definitive vLLM TPOT is 16.44% higher (14.12% lower t/s/u), while repeat A is
15.30% higher (13.27% lower t/s/u). vLLM remains about as fast as the full
generator while including scheduler admission, plugin handoff, compact expert
and PLE host service, sampling, token feedback, virtual state, and readback.

## Secondary CI serving-burst result

These are capacity/nightly-parity metrics only: P300 1x2, 100 input / 100
output / 32 requests, unbounded client concurrency, server `max_num_seqs=2`,
physical decode batch 1 with two virtual slots, `max_model_len=262144`, explicit
greedy temperature 0, full profile, and the same TT config. Burst TPOT is not
the headline decode t/s/u.

| CI burst 100/100/32 | TTFT p50/p99 (ms) | TPOT mean/p99 (ms) | ITL p50/p99 (ms) | Aggregate output (token/s) | TPOT-derived capacity context (t/s/u) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 495900.588 / 984280.694 | 608.992 / 633.142 | 622.197 / 675.489 | 3.059929 | 1.642057 |
| **After, definitive final artifact** | **486420.899 / 965326.408** | **594.470 / 622.156** | **605.783 / 642.479** | **3.123112** | **1.682172** |
| After, same-source repeat A | 488699.107 / 965687.389 | 594.145 / 624.915 | 607.484 / 636.812 | 3.127670 | 1.683091 |

The definitive 100/100/32 burst improves TTFT p50/p99 by 1.91%/1.93%, mean
TPOT by 2.38%, ITL p50/p99 by 2.64%/4.89%, aggregate output throughput by
2.06%, and TPOT-derived capacity context by 2.44%.

## Exact retained path

The vLLM TT plugin registers architecture `Qwen4ExpForConditionalGeneration`
to `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm`. That thin
adapter delegates prefill, decode, sampling, cache lifecycle, trace replay, and
host-store behavior to `Qwen38Generator` and `Qwen38FullModel`; vLLM owns the
attention KV object. The selected datatype-sweep config is
`qsa_bfp8_hifi2_lm_head_bf16_hifi2`: exact host-packed and executed BFP4/LoFi
routed experts, BF16/HiFi2 LM head, BFP8 KV cache, BF16 activations/CCL, and
the full-model greedy split sampler.

`decode_forward(..., read_from_device=False)` returns device token tensors.
The plugin calls `read_decode_output(..., async_read=True)`, which enqueues only
the compact token D2H and records a CQ0 completion event. Host conversion is
locked and cached once. The next exact `(virtual slot, request id, generation)`
PLE lookup reuses that same compact result instead of reading the persistent
device token again; plugin host formatting shares the cached conversion.
`supports_async_decode=True` is justified by the live counter signatures:

| Async/trace evidence | Primary 128/128/1 | CI burst 100/100/32 |
| --- | ---: | ---: |
| Canonical token-feedback steps | 127 | 3168 |
| Async compact-host reuses | 126 | 3136 |
| Initial device-token fallbacks | 1 | 32 |
| Nonblocking trace replays | 126 | 3167 |
| Initial captures / prefill invalidation-recaptures | 1 / 0 | 0 / 1 |
| Model-only trace replays | 0 | 0 |
| Host sampling compatibility calls | 0 | 0 |
| Sampling-seed host copies in greedy window | 0 | 0 |

Token, current-position/RoPE, page-table, cache, and sampler inputs are stable
device tensors. Device sampling feeds `tt_out_tok` directly back into the
persistent token input, and position/RoPE advances on device. Page tables are
copied only when scheduler state changes. The real reduced physical-B1 /
virtual-B2 test changes tokens and positions, exercises changed and unchanged
page tables, rejects a stale generation before replay, cancels/reuses slots,
and matches isolated A/B controls for non-aligned prompt lengths 63 and 67.

Greedy sampling remains the optimized full-model vocabulary-sharded split
argmax path. `sample_on_device_mode=all` performs no adapter or host top-1,
force-argmax, eager generic sampling, or full-logits readback. The full plugin
sampling suite covers greedy/top-1, top-k, temperature, deterministic seeds,
unseeded variety, mixed parameters, logprobs, penalties, structured/plain
isolation, and negative seeds: 72 pass, with only the intentionally unsupported
all-vocabulary chat-logprobs case skipped.

## Host expert and PLE service

Both end-to-end campaigns prepacked all 24,576 exact experts into the same
68,080,435,200-byte host store, used ten persistent device slots per layer,
serial miss waves, configured depth 1, PLE row cache 8192, the same 12-request
warm-up sequence, and then primary followed by CI on one server. Cache
residency remains request-dependent, but cold/prepacked versus warmed state and
all capacity/configuration knobs are matched.

The accepted expert change writes each exact owner shard directly to its
persistent slot coordinate. It removes upload scratch and owner D2D, saving
2,764,800 bytes/layer/rank and 132,710,400 bytes/rank over 48 layers (8.33% of
the prior expert device footprint). A conservative owner ledger skips the
exact-zero peer D2D while ownership is unchanged and restores it on owner
flips. The focused completed ten-expert service improves from 6.820373 GB/s,
4.051894 ms p50 / 4.077123 ms p95 to **12.496113 GB/s, 2.171385 ms p50 /
2.699363 ms p95**. Twenty waves plus forced flips validate all ten slots,
both ranks, and both packed projections.

| Exact host window | Primary before, 128/128/1 | Primary after, 128/128/1 | CI before, 100/100/32 | CI after, 100/100/32 |
| --- | ---: | ---: | ---: | ---: |
| Expert cache hits / misses | 24,868 / 42,796 | 22,968 / 44,695 | 181,758 / 1,506,457 | 182,710 / 1,505,505 |
| Owner H2D bytes / enqueue time | 118.322 GB / 9.501 s | 123.573 GB / 8.795 s | 4165.052 GB / 329.659 s | 4162.420 GB / 283.602 s |
| Owner slot D2D bytes | 118.322 GB implicit staging copy | **0** | 4165.052 GB implicit staging copy | **0** |
| Exact peer-zero D2D bytes | 118.322 GB | 61.807 GB | 4165.052 GB | 2084.640 GB |
| Peer-zero resets / skips | all misses / 0 | 22,355 / 22,340 | all misses / 0 | 753,993 / 751,512 |
| Expert completion syncs | 0 | 0 | 0 | 0 |
| Compact route-read + TT exposed stall | 24.021 s | 24.569 s | 621.834 s | 639.758 s |
| Expert service | 33.102 s | 32.891 s | 941.364 s | 917.919 s |
| PLE selected / unique rows | 4096 / 4096 | 4096 / 4096 | 101,904 / 101,896 | 101,904 / 101,896 |
| PLE table bytes read | 1.254 MB | 1.234 MB | 26.575 MB | 26.715 MB |
| PLE lookup / H2D enqueue | 0.703 s / 0.012 s | 1.489 s / 0.015 s | 8.993 s / 0.363 s | 16.237 s / 0.350 s |
| PLE exposed service | 0.385 s | 0.760 s | 5.244 s | 7.976 s |
| PLE completion syncs | 0 | 0 | 0 | 0 |
| Total model submit window | 33.591 s | 33.756 s | 949.184 s | 928.506 s |

Expert H2D enqueue time falls 7.43% in the primary and 13.97% in CI despite
slightly different route/cache-hit histories. Primary total submit is +0.49%;
CI total submit is -2.18%. PLE lookup is noisier and slower in the optimized
process, so it is reported rather than hidden. The exact decode-specialized
PLE row-dedup candidate improved isolated CPU lookup 6.89x cold / 1.14x hot,
but regressed real primary TPOT to 270.037506 ms / 3.703189 t/s/u and was
reverted. The retained PLE contract remains exact mmap row coalescing,
per-shard lookup, BF16 staging, nonblocking upload, EOS-aware two-token history,
cancellation, and request isolation.

## Serving and context gates

| Gate | Workload / result | Artifact |
| --- | --- | --- |
| Full plugin sampling | Live TT server, full profile: 72 pass / 1 expected skip | `after/sampling_tests.log` |
| Generic qualitative | Six prompts, greedy and sampled: coherent/non-degenerate | `after/vllm_qualitative_outputs.json`, `after/degenerate_output.json` |
| Prompt-correct qualitative | Three chat-template prompts with matching full-model HF controls: pass | `after/qualitative_tt_chat.json` |
| Non-aligned serving | Exact lengths 1, 63, 64, 65, 67, 127, 129, two deterministic repeats each | `after/non_aligned_prompt_check.json` |
| Real async/stale/isolation | Physical B1 / virtual B2, representative layers 0/1/3, lengths 63/67: pass | `async_feedback_virtual_b2_tt.xml` |
| Expert slots/uploads | 20 ten-expert waves, all-slot exactness, forced owner flips: pass | `direct_target_completed_cache_tt.xml` |
| Active cancellation/lifecycle | Overlap, active peer cancellation, four releases, identical follow-ups: pass | `after/host_serving_lifecycle.json` |
| Static/contract final | 62 pass, 2 explicit hardware-gated skips already covered above | `after/final_static.xml` |
| Context/stage gate | `262144 == doc/context_contract.json`; no degenerate output | `after/stage_gate.log` |
| Shutdown | No vLLM/EngineCore process, no device holders, both DRAM/heartbeats healthy | `after/process_cleanup_audit.json` |

Advertised and served context remains 262,144. There is no prompt, benchmark,
evaluation, or server context reduction. Direct-slot H2D reduces the current
maximum-context batch-1 capacity plan to 9,871,840,344 bytes/device and raises
headroom to 24,353,680,296 bytes/device. Internal 64-token pages and 128-token
prefill chunks do not restrict valid non-aligned request lengths.

## Optimize checklist and rejected paths

The vLLM-relevant `$optimize` audit is complete without serving profiler
collection, as required for this stage:

| Area | Evidence/decision |
| --- | --- |
| Same-harness measurement | Identical runner, generation mode, sampling, model length, sequence cap, mesh, TT config, workload shapes, and warm-up order; raw before/after JSON retained |
| Runtime path | Real registered adapter delegates to the selected full model; no stock/HF fallback |
| Trace | Persistent inputs, decode-only split trace, nonblocking replay, one declared initial capture, stale-input and page-table change/skip tests |
| Async | Device return, minimal compact deferred read, exact reuse/fallback signatures in both benchmark windows |
| Sampling | Full-model greedy split sampling; full plugin suite; zero host compatibility/argmax/full-logits/model-only-trace evidence |
| Movement | Direct coordinate H2D; no upload scratch, owner D2D, per-miss fence, PLE fence, unnecessary reshard/conversion, or standalone KV assumption |
| Host stores | Exact expert generations and peer-zero placement; PLE coalescing/staging/history/isolation; cold/warm and capacity declarations retained |
| Full-model comparison | 269.661 ms vLLM versus 231.594 ms full-model token-out for comparable 128+128 B1 work |
| Profiler policy | No Tracy, `tt-perf-report`, live-server device profiler, adapter profiler, or `ReadDeviceProfiler` was collected |

Rejected options include staging depths 2/10, owner partition/coalescing/thread
policies inherited from optimized full-model evidence, and the new PLE decode
dedup candidate. Force-argmax, slow generic sampling, host greedy/top-1,
full-logits readback, and aligned-only prefill were not accepted as candidates.

## Limitations and AutoFix

- Exact host prepack costs about 236-244 seconds in these server runs and
  consumes 68.080 GB host RAM. PLE is a 102.400 GB mmap table; pinned host
  staging is unavailable in the installed CPU Torch build.
- Physical decode remains batch 1 with two virtual slots. The 100/100/32 CI
  burst demonstrates capacity/nightly parity but its high admission TTFT and
  TPOT-derived value are not single-user decode claims.
- Primary TTFT varies materially between identical-source runs, and the
  retained source does not show a statistically robust single-user latency
  win. The host-memory, async, exactness, and CI improvements are the accepted
  outcome.
- The focused `test_host_backed_layer0_decode_matches_optimized_reference`
  reports PCC 0.863740265 on optimized source, forced peer-zero source, an
  explicit post-service synchronization, and clean starting HEAD. AutoFix
  therefore failed to find a causal stage fix; the pre-existing regression is
  recorded in `AUTODEBUG.md` and `AUTOFIX.md`, while all-slot cache exactness
  and live serving gates pass.
- Server shutdown emits nanobind interpreter-reference leak diagnostics, but
  the launcher exits zero and the final audit finds no process or device
  holder. No model-owned KV residue or standalone-cache assumption is present.

Commands and chronology are in `work_log.md`. Structured raw evidence lives
under `before/`, `after/`, and `candidates/`; `serving_host_metrics.json`
contains the exact marker-selected async, cache, PLE, trace, lifecycle, and
prohibited-host-work assertions.

The required independent review is recorded in `stage_review.md`: verdict
`clean-pass`, no Required Work. Its non-blocking concerns and anomaly ledger
remain part of the handoff rather than being omitted from the selected stage
evidence.

The multi-megabyte raw `before/server.log.gz` and `after/server.log` are
retained locally in this checkout and were part of the independent review, but
are intentionally unversioned under the repository's 500 KB artifact policy.
The selected benchmark JSON, derived host/async metrics, sampling output,
exactness XML, lifecycle/context checks, and compact benchmark logs are
versioned here.
