# AutoFix: final cumulative decoder topology

Date: 2026-08-28

This pass tested the final-stage review findings as hypotheses. Production
remains capacity-selected: configured capacity 1 uses BFP8/HiFi2; configured
capacities 2--32 use BFP4/LoFi, the 15-core DRAM-sharded packed QKV projection,
and the 32-core output projection. The pass also repaired decode calls whose
logical batch is smaller than the decoder's configured capacity.

## DRAM10 versus DRAM15

Three exact-checkpoint, prompt-derived capacity-2 A/B replicates covered the
sliding and full layer kinds, final 32-core output topology, unchanged
0.95/0.99 PCC bars, first/second determinism, and post-1000-replay determinism.
The order was DRAM15-first, DRAM10-first, DRAM15-first.

| Replicate | Sliding DRAM15 / DRAM10 ms | Full DRAM15 / DRAM10 ms | Equal-kind sum DRAM15 / DRAM10 ms |
| --- | ---: | ---: | ---: |
| 1 | 0.778251 / 0.770836 | 0.727213 / 0.727257 | 1.505464 / 1.498093 |
| 2 | 0.778011 / 0.771072 | 0.726912 / 0.727637 | 1.504923 / 1.498709 |
| 3 | 0.778281 / 0.770826 | 0.727121 / 0.727294 | 1.505402 / 1.498120 |

The model has 18 sliding and 18 full layers, so the representative mix weights
the two kinds equally. DRAM10 won that batch-2 mix by a median 0.484%, but a
configured-capacity-2 decoder may legally receive a logical-batch-1 call. On
that exact real-prompt contract, DRAM10 sliding decode reached only 0.989158
PCC, below the unchanged 0.99 bar; DRAM15 passed at 0.990655. The corresponding
full rows both passed (0.997948 and 0.997896). DRAM15 therefore remains the
production policy for configured capacities 2--32. Capacity-1-only evidence
was not used to decide this result.

The first combined larger-capacity process exposed a real runtime issue before
the policy comparison: KV-update input sharding inherited the configured
capacity even when the logical batch was smaller. The decoder now builds the
decode K/V input shard grid from the explicit runtime batch. The repaired
capacity-2 exact test reached the policy comparison above. Two exact
configured-32 retries were bounded and retained separately: one was killed at
120B expert materialization and the other was stopped after reaching about
226 GB host RSS, both before candidate execution. They are resource failures,
not policy results.

A lightweight configured-32/logical-1 semantic gate replaced those unsafe
retries. Its initial random-PCC diagnostics are retained but do not veto a
real-weight-qualified BFP4 policy. The corrected gate passes both layer kinds:
finite prefill/decode outputs have the expected shapes, only the selected
physical K/V page changes, unselected pages remain byte-identical, and traced
replay is deterministic.

## Cumulative output and packing decisions

The eventual DRAM15 policy was swept cumulatively with legal 64/32/16/8-core
output projections. All rows passed correctness and determinism. Relative to
the 32-core reference, the sliding/full whole-trace deltas were +4.611/+4.898%,
-0.041/-0.029% for the identical 32-core configuration (timer noise),
+0.248/+0.250%, and +3.027/+3.206%. The 32-core output topology remains final.
An earlier DRAM10 output sweep is retained as diagnostic evidence but did not
select production.

Separate-Q/K/V controls now use the same final 32-core output projection.
The initial DRAM-sharded separate-control legality dismissal was wrong: TTNN
legally accepts three independently configured DRAM-sharded
matmuls. The cumulative control uses the same 15-core, `in0_block_w=6` input
sharding, per-projection `per_core_N=(9,2,2)`, Q/K/V DRAM weight shards of
512/64/64 elements per bank, converts each result to interleaved DRAM, then
concatenates and applies the common bias. It also retains the final 32-core
output projection.

The original whole-decoder A/B was bounded three times before candidate
execution because exact expert materialization reached 226--239 GB host RSS.
Those logs are infrastructure diagnostics, not candidate evidence. The
replacement low-memory gate loads only exact q/k/v/o, sink, and norm tensors,
uses exact prompt embedding rows, and injects the same stateless zero-MLP
object into both otherwise complete decoder shells. Layer 1 is explicitly an
embedding-input proxy, not a saved post-layer-0 activation. It directly
compares separate outputs with packed outputs at the unchanged 0.95 prefill /
0.99 decode bars, checks selected-only paged-cache mutation, and checks
first/second/post-1000 trace determinism. All four nodes passed:

| Capacity/policy | Kind | Direct prefill/decode PCC | Packed / separate shell ms | Measured delta | Anchored whole estimate |
| --- | --- | ---: | ---: | ---: | ---: |
| 1, BFP8/HiFi2 | sliding | 1.000000 / 0.999944 | 0.223314 / 0.257823 | +0.034508 ms | +6.376% |
| 1, BFP8/HiFi2 | full | 1.000000 / 0.999955 | 0.223208 / 0.257758 | +0.034550 ms | +6.377% |
| 2, BFP4/LoFi DRAM15 | sliding | 1.000000 / 1.000000 | 0.200318 / 0.227760 | +0.027442 ms | +3.526% |
| 2, BFP4/LoFi DRAM15 | full | 1.000000 / 1.000000 | 0.197476 / 0.227730 | +0.030253 ms | +4.161% |

The anchored values add the measured shell delta to the earlier exact packed
whole-trace measurement; they are estimates, while direct PCC and measured
shell deltas are the decision evidence. Separate is 13.70--15.48% slower in
the measured shell and the anchored whole estimate is 3.526--6.377% slower.
Because every gap exceeds 1% and packed wins both layer kinds, reverse/third
ordering is not applicable. Packed remains final on measured evidence, not an
API-legality inference.

Each node ran separately through the serialized safe wrapper; `NODE` was
`sliding-batch1`, `full-batch1`, `sliding-batch2`, then `full-batch2`:

```bash
TT_VISIBLE_DEVICES=0,1 \
GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY=1 \
GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY_REPEATS=1000 \
GPT_OSS_120B_DRAM_SEPARATE_QKV_LOW_MEMORY_ORDER=packed,separate \
scripts/run_safe_pytest.sh \
  "models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_low_memory_exact_attention_packed_vs_dram_separate_qkv[blackhole-1x1-${NODE}]" \
  -q -s
```

## Final harness and gates

The warmed performance harness now materializes automatic configured/logical
batch 2, passes the logical batch through prefill, decode, and trace capture,
and emits the existing prefill/decode signposts. It asserts the instantiated TT
weight dtypes, LoFi fidelity, DRAM-sharded QKV program and input memory config,
and final output grid/block geometry. Without Tracy, its 1000-replay
qualification measured 0.687493 ms sliding and 0.779065 ms full; warmed
sequence-128 prefill measured 41.094766 and 41.420308 ms. The subsequent
marker-clean ten-replay Tracy captures assert the same instantiated policy and
report 0.628083/0.705526 ms sliding device/wall and 0.728872/0.799062 ms full
device/wall. Their rows directly show BFP4, LoFi, DRAM sharding, and final32;
the two-user compulsory-traffic roofline is 0.285296 ms.

Post-repair gates passed: production logical batch 32 (2/2), configured-32 /
logical-1 semantic coverage (2/2), the final-source ordinary integrated suite
(12 passed, 183 opt-in skips), and a focused watcher run covering non-aligned
sliding/full optimized paths (2/2). A broader watcher attempt is retained: its
advertised-context SDPA
program exceeded watcher's reduced kernel-config capacity (81984 requested vs
70656 available) after four earlier tests passed. The same ordinary integrated
path passes without watcher; the focused watcher process is clean. Final
`tt-smi -s` enumerates all four boards.

## Exact artifacts

All paths below are under `evidence/logs/`.

- `optimized_autofix_stage_review_capacity2_dram10_vs_15_replicate{1_15_first,2_10_first,3_15_first}.log.gz`
- `optimized_autofix_stage_review_dram10_vs_15_logical_batch1_capacity2_repaired.log.gz`
- `optimized_autofix_stage_review_dram10_vs_15_logical_batch1_larger_capacity.log.gz`
- `optimized_autofix_stage_review_dram15_logical_batch1_capacity32_sliding.log.gz`
- `optimized_autofix_stage_review_logical_batch1_capacity32_lightweight{,_repaired,_semantic}.log.gz`
- `optimized_autofix_stage_review_dram{10,15}_output_geometry_sweep_reference_first.log.gz`
- `optimized_autofix_stage_review_packed_vs_separate_final_packed_first.log.gz`
- `optimized_autofix_stage_review_dram_separate_qkv_packed_first.log.gz`
- `optimized_autofix_stage_review_dram_separate_qkv_batch1_sliding_{packed_first,shared_mlp_packed_first}.log.gz`
- `optimized_autofix_stage_review_dram_separate_qkv_low_memory_batch{1,2}_{sliding,full}_packed_first.log.gz`
- `optimized_autofix_stage_review_dram_separate_qkv_low_memory_final_host_checks.log.gz`
- `optimized_autofix_stage_review_automatic_capacity2_profile_harness.log.gz`
- `optimized_final_source_capacity2_same_run_profile_{sliding,full}_repeats10.log.gz`
- `optimized_final_source_capacity2_tt_perf_report_generation.log.gz`
- `optimized_final_source_{integrated,batch32,watcher_focused,post_tt_smi}.log.gz`
- `optimized_autofix_stage_review_final_{batch32,integrated,watcher,watcher_focused,post_tt_smi}.log.gz`

The evidence manifest is regenerated after this report and the README/work log
updates. No context limit, KV-cache dtype, or advertised capability changed, so
`doc/context_contract.json` remains unchanged.
