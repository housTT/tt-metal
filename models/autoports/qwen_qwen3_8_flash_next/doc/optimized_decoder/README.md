# Qwen3.8-Flash-Next optimized decoder

This stage delivers the single-P300c optimized decoder for
`Qwen/Qwen3.8-Flash-Next`. The runtime is
`../../tt/optimized_decoder.py`; the dedicated tests instantiate the exact
`OptimizedDecoder` type and reject a functional/fused fallback or host
conversion in the measured path.

## Result

The selected precision policy is real-weight BFP4/LoFi for routed experts,
BFP8/LoFi for shared projections, BFP8/HiFi2 for GDN projections, BF16/HiFi2
for QSA input and attention output, and BF16 for recurrence-sensitive state and
the paged KV cache. Batch-one decode executes the ten on-device row-0 top-k
expert IDs with indexed sparse matmuls and compact L1 intermediates. Prefill
and batched decode retain the exact dynamic-union path. Explicit 2D prefill and
1D decode programs, L1 outputs, tuned native SDPA, legal packed same-input
projections, and DRAM-sharded QSA-input and attention-output decode matmuls
complete the selected topology.

The performance table is like-for-like: identical real checkpoint weights,
deterministic nonzero checkpoint-token embeddings, seven independent warmed
samples, sequence-128 prefill, and 100 traced replays per decode sample.

| Layer | Kind | fused prefill ms | optimized prefill ms | win | fused traced decode ms | optimized traced decode ms | win |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | GDN | 31.864180 | 22.504434 | 29.374% | 4.090832 | 1.090618 | 73.340% |
| 1 | PLE + GDN | 36.066132 | 25.594588 | 29.034% | 4.554379 | 1.455579 | 68.040% |
| 3 | QSA | 56.996406 | 48.573433 | 14.778% | 5.777473 | 2.866913 | 50.378% |

Real-weight Hugging Face-reference PCC remains above the 0.995 functional bar
for every meaningful layer kind:

| Layer | fused prefill | optimized prefill | delta | fused decode | optimized decode | delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.99842572 | 0.99824739 | -0.00017833 | 0.99702853 | 0.99601841 | -0.00101012 |
| 1 | 0.99911219 | 0.99837846 | -0.00073373 | 0.99988902 | 0.99922472 | -0.00066430 |
| 3 | 0.99668270 | 0.99611741 | -0.00056529 | 0.99977344 | 0.99915755 | -0.00061589 |

The bounded deltas come from the selected expert/projection precision and the
layer-0-only 55-core padded GDN packed-qkv geometry. Recurrence state,
normalization, attention activations, and cache dtype remain BF16/FP32 where
the fused contract requires them.

An exact-final BFP8 KV-cache candidate also passed layer-3 PCC
(0.99606544/0.99814188) and deterministic trace replay. Its seven-run,
100-replay traced-decode median was 2.867980 ms versus 2.866913 ms for BF16;
the candidate and control ranges did not overlap. BFP8's 48.451365 ms prefill
median was 0.122068 ms faster, but BF16 is retained because it is the fastest
correct traced-decode policy.

## Preserved contract

`../context_contract.json` is unchanged: maximum context 262,144, page size
64, internal prefill chunk 128, and tested decode batch 32. Public logical
sequence lengths are not required to align to a tile or chunk. Coverage
includes 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 2047, 2048, 2049,
262,143, and 262,144; shuffled and per-user page tables; deterministic trace
replay; full and near-maximum contexts for GDN, PLE+GDN, and QSA; and QSA
traced decode at position 262,143. The exact promoted layer-3 default was
retested at 262,144 and 262,143 after adding both setup-time sharded weight
copies. No dtype or layout change reduces advertised capacity.

## Profiler conclusion

The exact-final layer-3 reports are under
`tracy_dram_qsa_attn_final/layer3_qsa/`. The prefill window has 220 rows,
48.269088 ms device time, 0.517290 ms gaps, and a 6.9%/35 GB/s modeled DRAM
roofline. The decode window has 234 rows, 2.552567 ms device time, 0.418620 ms
gaps, and an 11.3%/58 GB/s modeled DRAM roofline. DRAM-sharded QSA input is
151.843 us and DRAM-sharded attention output is 66.498 us. Relative to the
correct QSA-only capture, the attention-output promotion saves 8.615 us in its
matmul and 6.555 us net in the device-plus-gap window; seven-run clean latency
confirms the overall win.

Prefill routed unions are observed as `[148, 143, 140, 140]` for layer 3.
`tt-perf-report` accepts only one integer per group, so its scalar 143 models
572 active rows versus the exact 571 (a documented one-row conservative
rounding). Decode is exact `active=10/512` because the indexed path consumes
exactly ten compact row-0 experts.

## Evidence map

- `final_real_fused_perf_count7.xml` and
  `final_real_optimized_dram_qsa_attn_perf_count7.xml` contain comparable
  real-weight, nonzero-input samples and workload/routing hashes.
- `final_dram_qsa_attn_correctness_trace_alloc.xml` is the 35-test normal gate
  with trace-allocation tracking. `final_dram_qsa_attn_long_context.xml`
  contains the three exact-default layer-3 capacity cases.
- `final_dram_qsa_attn_stress.xml` contains nine repeated deterministic trace
  passes. `final_dram_qsa_attn_watcher.xml` and
  `watcher_dram_qsa_attn_final/generated/watcher/watcher.log` are the final
  watcher-clean run.
- `tracy_final_indexed/layer0_gdn/`, `tracy_final_indexed/layer1_ple_gdn/`,
  and `tracy_dram_qsa_attn_final/layer3_qsa/` contain signpost-filtered
  prefill/decode ops CSVs, report CSVs, stacked CSVs, and plots.
- `autofix_*` and `candidate_*` artifacts record controlled topology,
  precision, sharding, SDPA, program, movement, and recovery trials.
  `autofix_cache_bfp8_exact_final_{real_pcc_trace_l3,perf_l3_count7}.xml`
  closes the exact-final KV-cache dtype comparison.
  `triage/width_sharded_attempt2/` contains the hang investigation.
- `work_log.md` records commands, the operation-topology audit, candidate
  decisions, roofline/accounting, hashes, and the `$optimize` checklist.

Raw Tracy databases remain reproducibly in the recorded `/tmp` paths; only
compact CSV/report artifacts are checked in. No applicable single-device
decoder optimization is deferred. Multichip, full-model, and vLLM work are
deliberately outside this goal.
