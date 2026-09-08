# Optimized vLLM serving: Gemma 4 26B A4B IT

Warmed primary **128 input / 128 output / 1 request / concurrency 1** through the real vLLM TT plugin:

| Profile | Workload in/out/requests | TTFT ms, before → after | Decode t/s/u, before → after |
| --- | --- | ---: | ---: |
| P150 | 128/128/1 | 211.36 → **209.12** | 36.22 → **37.24** |
| P150x2 | 128/128/1 | 167.29 → **167.80** | 43.96 → **46.44** |
| P150x4 | 128/128/1 | 145.74 → **143.91** | 48.20 → **52.06** |

Decode t/s/u is **1000 / mean TPOT**, not aggregate throughput or a CI-burst value. These are independently measured P150/P150x2/P150x4 profiles on the established **P300C Blackhole 1/2/4-chip proxies**. They are not measurements on physical P150 cards.

**Validation status:** all three full serving gates, artifact reconciliation, final degeneracy/context checks and shutdown/reopen checks pass. Independent [stage review](STAGE_REVIEW.md) returns **clean-pass**, with no required work. Local checkpoint SHAs are recorded in [work_log.md](work_log.md).

## Matched measurement contract

All six runs use `models.common.readiness_check.run_vllm_server`, the autoport `tt/generator_vllm.py`, all 30 layers, the unchanged `selected_canonical_profile_policy`, trace mode `all`, `sample_on_device_mode=all`, async scheduling, 32 scheduler slots, block size 64, and Gemma4 tool/reasoning parsers. The resolved checkpoint is `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Benchmark requests explicitly set greedy temperature zero and ignore EOS. Every workload has one **successful** explicit warmup before its timed requests.

| Profile | Mesh configuration | `max_model_len` | TT configuration |
| --- | --- | ---: | --- |
| P150 | N150, 1 chip | 50,624 | trace region 220,000,000 bytes |
| P150x2 | N300, 1×2 submesh of 2×2 parent | 262,144 | same trace region; FABRIC_2D; offset [0,0] |
| P150x4 | P300x2, 1×4 mesh | 262,144 | same trace region; FABRIC_1D_RING |

The scheduler prefill budget remains 2,048 tokens. This chunks requests internally; it does not lower logical context. [Context contract](../context_contract.json) and [independent lifetime accounting](context_lifetime_audit.md) preserve the profile limits while accounting for retained traces, real serving KV allocations and all-row prefill logits. The latter is source-derived accounting, not a measured allocator high-water mark.

The original adapter was restored from `6eb0427423392d7c6a7f87a511be892b8bf677ae` for the warmed baseline, then the optimized adapter was restored and hash-checked. Both sides use identical fixed worker, router and benchmark-client code. [Source-swap ledger](baseline_source_swap.json) and each run manifest retain exact argv and hashes. Initial `before` and P150 `after` benchmarks accidentally used zero warmups; they remain unchanged as **cold diagnostics**, excluded from these tables. [Warmup diagnosis](AUTODEBUG_benchmark_warmup.md) and [fail-closed harness fix](AUTOFIX_benchmark_warmup.md) explain the correction.

## Primary single-user measurements

One timed request follows one warmup per profile. TTFT and TPOT p50/p99 therefore equal their means; ITL percentiles cover the 127 inter-token intervals. This small mandated workload does not establish confidence intervals. TTFT is essentially unchanged; the measured improvement is removal of repeated decode setup.

| Profile | Workload | Phase | TTFT mean ms | TPOT mean ms | ITL mean / p50 / p99 ms | Aggregate output tok/s | Decode t/s/u |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| P150 | 128/128/1, C1 | before | 211.358 | 27.608 | 27.608 / 26.829 / 27.160 | 34.429 | 36.222 |
| P150 | 128/128/1, C1 | after | 209.124 | 26.850 | 26.850 / 26.832 / 27.185 | 35.366 | 37.244 |
| P150x2 | 128/128/1, C1 | before | 167.289 | 22.750 | 22.750 / 21.489 / 25.420 | 41.873 | 43.957 |
| P150x2 | 128/128/1, C1 | after | 167.799 | 21.532 | 21.532 / 21.485 / 24.384 | 44.098 | 46.443 |
| P150x4 | 128/128/1, C1 | before | 145.735 | 20.746 | 20.746 / 19.156 / 30.405 | 46.029 | 48.201 |
| P150x4 | 128/128/1, C1 | after | 143.907 | 19.209 | 19.209 / 19.149 / 23.117 | 49.540 | 52.058 |

## Comparison with selected full-model token-out decode

The serving primary includes token return through the plugin and HTTP client. The selected full-model control includes model, final norm, LM head, split sampling and device feedback, with five warmups and 128 timed tokens at positions 134–261, and no timed host token reads. These are comparable token-out workloads with distinct host boundaries; this is not a same-run device/network decomposition.

| Profile | Serving workload | Serving TPOT ms | Full-model workload | Full-model ms/token | Serving/control latency ratio |
| --- | --- | ---: | --- | ---: | ---: |
| P150 | 128/128/1, C1 | 26.850 | B1, prompt 128, 5 warmups + 128 timed token-out steps | 26.842 | 1.0003× |
| P150x2 | 128/128/1, C1 | 21.532 | B1, prompt 128, 5 warmups + 128 timed token-out steps | 21.491 | 1.0019× |
| P150x4 | 128/128/1, C1 | 19.209 | B1, prompt 128, 5 warmups + 128 timed token-out steps | 19.156 | 1.0028× |

All three serving results are within 0.3% of their own selected traced controls. The [selected TP4 control](../datatype_sweep/artifacts/selected_token_out/token_out_trace_tp4.json) supersedes the older full-model result; no teacher-forcing throughput is substituted. Device-time, roofline and profiler fields are deliberately null in [perf_summary.json](perf_summary.json). No Tracy, tt-perf-report, live-server/adapter profiler or ReadDeviceProfiler was used.

## Secondary CI serving-burst capacity

The CI/nightly-parity workload is **100 input / 100 output / 32 requests**, unbounded client concurrency and 32 server slots. One successful B1 warmup precedes the burst; this does not prewarm every intermediate batch shape. All 32 requests and all requested output tokens completed on each side. Capacity is essentially unchanged. These values are not headline decode t/s/u.

| Profile | Workload | Phase | TTFT mean ms | TPOT mean ms | ITL mean / p50 / p99 ms | Aggregate output tok/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| P150 | 100/100/32 | before | 4938.205 | 272.208 | 272.208 / 245.342 / 622.641 | 100.349 |
| P150 | 100/100/32 | after | 4949.860 | 272.270 | 272.270 / 245.379 / 623.713 | 100.293 |
| P150x2 | 100/100/32 | before | 3950.120 | 225.249 | 225.249 / 199.521 / 643.827 | 121.899 |
| P150x2 | 100/100/32 | after | 3951.984 | 225.442 | 225.442 / 199.551 / 648.664 | 121.801 |
| P150x4 | 100/100/32 | before | 3434.381 | 215.540 | 215.540 / 186.541 / 789.761 | 129.165 |
| P150x4 | 100/100/32 | after | 3452.552 | 215.490 | 215.490 / 186.556 / 785.050 | 129.096 |

## Trace, sampling and runtime decisions

- Retain the canonical model/sampling trace pair across warmed prefill of a compatible execution shape. Cold prefill program-cache growth retires old traces before replay; host compatibility, remapping and physical batch-shape changes also retire them. No allocation-tracker bypass or new corruptible scope was added.
- Restore temperature-zero semantics after plugin normalization so greedy requests use the existing full-model split-sampling key. Request seed epochs no longer force greedy recapture. Sampling still uses vocab-sharded local top-32 and semantic k1; no adapter argmax, full-logits readback or eager sampler enters the measured path.
- Reuse persistent token, position, RoPE, page-table, cache and sampler inputs. Token output aliases the next token input; position/RoPE advance on device. Refresh only at scheduler boundaries. Compare actual shared host page groups once and upload changed layers only, while keeping all per-layer device buffers distinct.
- `decode_forward(..., read_from_device=False)` returns a device tensor. Both trace replays use `blocking=False`; the plugin subsequently enqueues token-only `cpu(blocking=False)`, records an event, synchronizes that event at finalization and formats host output. Real async scheduling and overlap checks exercise this contract.
- Preserve genuine B1 and the existing padded lane space for larger active batches. Nonaligned prompts and page crossings remain supported. The adapter consumes vLLM-owned caches; no standalone-cache substitution is introduced.

[Serving contract audit](serving_contract_audit.md), [page-table audit](page_table_audit.md), [trace proof](trace_contract_summary.json) and [inherited device checklist](inherited_device_contract.md) record scope and evidence. Model arithmetic, precision, decoder geometry and collective topology are unchanged. The selected datatype sweep remains authoritative; no new decoder sweep or selected-policy profiler is claimed.

Rejected options: unconditional trace retention failed when cold prefill programs occupied trace workspace; key-stable greedy plus a program-cache growth guard fixes that safely. Force-argmax, host sampling and an aligned-only shortcut are not adopted. Reduced active-row execution for larger batches remains excluded by the existing correctness evidence. [AutoDebug record](AUTODEBUG.md) retains failed controls and successful repairs.

The [TP2 warning audit](runtime_warning_audit.md) derives an inherited global-versus-local output-grid mismatch. Matmul directly creates its computed output; the warning branch adds no fallback or corrective copy. Matched benchmark warning counts are identical. The separate multiply/down layout is preserved; no unsupported claim of an all-L1 or uniformly 11-core decoder is made.

## Correctness and cleanup evidence

The final real-adapter regression independently passes 18 cold/warmed/control requests on every mesh, using two real layer kinds and the real terminal path. Exact logits/tokens match explicit recapture controls; changed/unchanged pages, changed token/current-position, intentionally stale host feedback, persistent addresses and B2/B3 transitions are checked. All three tests pass through full process exit with allocation tracking and watcher assertions/NoC checks enabled. Watcher NOINLINE and waypoint-only omission fit the hard ERISC code-size limit; assertions, NoC and Ethernet checks remain enabled. These reduced functional probes are not serving performance or maximum-context measurements.

Each profile independently passes 72 plugin sampling tests, with the configured all-vocabulary logprobs-cap skip. Feature, nonaligned prompt, tool/reasoning parsing, async and profile-local logit gates pass on all three. All 36 qualitative outputs were read: all 18 greedy outputs exactly match their respective prior profile. Long explanations/stories truncate at the shared 256-token cap, as in controls; answer completeness is not claimed. The sampled TP4 thermodynamics explanation uses overbroad efficiency wording. A [targeted original/current control](AUTOFIX_thermodynamics.md) continues its actual retained prefix at 247 input / 256 output tokens with identical device-sampling settings: all three pairs match exactly and reproduce the factual simplification. This controls inherited TT behavior at that prefix; it does not establish HF attribution, scientific correctness, or equality of the original full stochastic trajectory. Profile-local numeric controls remain unchanged.

Optional unsupported seeded/penalty/logprob/structured features retain the integration’s explicit compatibility path. Full-logit diagnostic checks are labelled separately; their passing results are not evidence of all-device execution for those features. The measured greedy workloads and supported top-32 sampled qualitative requests use device sampling.

Worker shutdown is explicit and idempotent, using the existing nested-mesh close helper. Router teardown clears completed NoC tags before firmware handoff. [Cleanup evidence](cleanup_evidence.json), [worker experiment](worker_cleanup_experiment.md), and [kernel experiment](kernel_cleanup_experiment.md) retain host tests, failed controls, clean serving shutdown and immediate mesh reopen without reset. Inherited nanobind interpreter-exit diagnostics are controlled against the baseline; no retained server/device ownership is inferred from them.

The mandatory `.github/scripts/copilot-build.sh` was attempted but could not access `/var/run/docker.sock`; full repository compilation is **unverified due to that environment failure**. The changed router kernel compiled through device JIT and passed the targeted watcher/process-lifetime runs. Source/documentation pre-commit checks pass; raw capture artifacts are preserved without formatting rewrites. No performance improvement is attributed to the cleanup fix. The final degeneracy/context gate passes after fixing profile recognition in aggregate stage reports; [checker regression evidence](AUTOFIX_context_reports.md) proves that other context floors remain enforced.

## Reproduction and artifact map

Run from `/home/hous/dev/tt-metal`. The driver refuses to overwrite existing manifests, so use fresh phase names for additional experiments. The baseline helper also verifies the candidate hash, requires a stopped server, preserves a source backup and restores the candidate in `finally`. The following are the commands used for the accepted warmed measurements and profile quality gates:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_before_warmed.py
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_profiles.py --phase after_warmed --profiles P150
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_profiles.py --phase after_warmed --profiles P150x2 --full-gates
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_profiles.py --phase after_warmed --profiles P150x4 --full-gates
```

P150 quality was collected by the earlier `--phase after --profiles P150 --full-gates` run using identical final adapter/model/generator/worker/router bytes. Only its cold performance is excluded. `summarize_results.py` checks source/config pairing, successful warmups, complete outputs and profile-local gates before producing the final summary.

| Evidence | Location |
| --- | --- |
| Exact commands, hashes, normalized metrics and raw benchmark JSON | `../../readiness_vllm/<profile>/optimized_vllm/{before_warmed,after_warmed}/` |
| P150 final quality evidence | `../../readiness_vllm/P150/optimized_vllm/after/` |
| Profile qualitative/control reviews | `qualitative_<profile>_review.json` |
| Final focused hardware assertions and request records | `trace_reuse_probe/final_router_fix/`, `.log`, `.xml` |
| Host adapter and worker checks | `host_adapter_tests.log`, `worker_cleanup_pytest.log` |
| Complete decisions, failed attempts and commands | [work_log.md](work_log.md) |
| Independent review and local commit SHAs | [STAGE_REVIEW.md](STAGE_REVIEW.md), [work_log.md](work_log.md) |

Large server/watcher logs are stored losslessly as `.log.xz`; use `xz -dc` to
read the original text and line numbers. [Archive index](artifact_compression.json)
records original paths, byte counts and uncompressed SHA-256. Embedded runner
paths and historical line references name those original decompressed logs.
