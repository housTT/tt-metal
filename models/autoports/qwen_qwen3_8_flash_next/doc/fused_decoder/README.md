# Qwen3.8-Flash-Next fused decoder

This directory is the completed single-chip graph-fusing stage for
`Qwen/Qwen3.8-Flash-Next` at checkpoint revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. The implementation is
`../../tt/fused_decoder.py`. It subclasses the completed functional decoder to
reuse its public state/cache contract, but every measured entry point
dynamically dispatches to `FusedDecoder`; there is no functional fallback
switch. The source-contract test asserts the exact type and the promoted fused
operators.

The stage covers representative layer 0 (Gated DeltaNet), layer 1 (PLE plus
Gated DeltaNet), and layer 3 (Qwen Sparse Attention), including the real
512-expert/top-10 sparse MoE. It does not contain optimized-decoder,
multichip-decoder, full-model, generator, or vLLM work.

## Acceptance result

All required correctness, capability, reliability, and performance gates pass
on local P300c Blackhole chip 0. The functional acceptance bar is PCC >= 0.995.

| Layer | Kind | Functional prefill PCC | Fused prefill PCC | Functional decode PCC | Fused traced-decode PCC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Gated DeltaNet | 0.99871051 | 0.99842572 | 0.99997765 | 0.99702853 |
| 1 | PLE + Gated DeltaNet | 0.99910986 | 0.99911219 | 0.99991739 | 0.99988902 |
| 3 | QSA | 0.99679226 | 0.99668270 | 0.99988198 | 0.99977344 |

The material layer-0 decode delta is caused by the promoted BF16 boundary of
`qkv_causal_conv1d_silu`; the exact FP32 FIR remained correct but was about
1 ms slower in 128-token prefill. The QSA delta comes from native GQA SDPA,
HF partial RoPE, persistent compressed index keys, and reordered device
arithmetic. All remain above the unchanged functional bar and repeated trace
replays are bitwise deterministic.

Like-for-like host values below are medians of seven complete runs. Both
runners use zero input, sequence length 128, identical cache geometry and
warmup, ten nonblocking trace replays, then one synchronization. Tracy is
collected separately and is not used as host latency.

| Layer | Functional prefill ms | Fused prefill ms | Functional traced decode ms | Fused traced decode ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 24.047593 | 17.015016 | 5.730349 | 3.883350 |
| 1 | 27.598879 | 20.517265 | 6.701143 | 4.321484 |
| 3 | 408.372929 | 43.610069 | 11.341126 | 5.539352 |

The final graph beats every distinct correct rejected candidate. In particular,
the exact `indexer_score_dsa` rewrite measured 43.660651/5.588385 ms versus the
final 43.610069/5.539352 ms. The earlier 43.666625/5.540206 observation is the
same persistent-compressed-cache path that was subsequently simplified and
promoted, not a distinct rejected graph; the authoritative final seven-run
median is numerically lower. Candidate commands, reversible patches/source
hashes, PCC, timing, and decisions are in `candidates/` and indexed by
`candidates/README.md`.

## Preserved public contract

`../context_contract.json` is unchanged:

- maximum logical context: 262144 tokens;
- public prefill chunk: 128 tokens, with internal padding/output slicing;
- QSA page size: 64 tokens;
- caller-provided paged prefill/decode page tables;
- maximum tested decode batch: 32.

The seven long-context tests execute exact 262144-token and non-aligned
262143-token prefill for layers 0/1/3 plus traced QSA decode at position
262143. Non-long coverage also exercises logical lengths 1, 31, 33, 65, and
127, shuffled and per-user page tables, underfilled selection, two-user
prefill-to-decode, and batch 32. No public alignment restriction was added.
No cache dtype, cache layout, page geometry, batch, or advertised capacity was
reduced, so no context-contract edit was necessary.

Batch one uses the fastest legal persistent/sharded paths. TTNN contracts
require two explicit correctness fallbacks for larger batches: V is converted
from the decode splitter's height-sharded output before padding, and index K is
repeated to four index heads because the matmul's head broadcast is only legal
for batch one. The batch-32 tests prove both fallbacks.

## Final fused graph

The retained graph includes:

- packed shared-LHS projections for hyperconnections, MoE/router/shared
  expert, routed gate/up, GDN, PLE, and QSA;
- one group-major A-sparse/B-dense expert-down matmul, with routing/shared
  scalar gates moved before the down projections;
- top-k-on-logits followed by selected-top-k softmax, which is algebraically
  identical to global-softmax/top-k/renormalize for this unbiased router;
- KDA fused four-tap causal convolution + SiLU + QKV split in GDN prefill,
  split persistent FIR/PLE decode state, fused chunk GDN, ternary arithmetic,
  and setup-folded normalization scales/biases;
- dedicated prefill/decode QKV split, `rotary_embedding_hf`, fused K/V paged
  update, decode SDPA, native GQA prefill SDPA, and persistent normalized,
  RoPE-applied compressed index keys;
- cancellation of inverse QSA V/Q/head permutes and concat-heads: the SDPA
  result is reshaped directly to the output-projection input;
- setup-time compressed address/static RoPE precomputation and removal of dead
  or redundant PLE/QSA reshapes, repeats, and masks.

`graph_inventory.md` gives the complete primitive-to-final topology and an
explicit disposition for every applicable `graph-fusing` pattern, including
all dedicated-op contracts that could not represent this model.

## Tracy and tt-perf-report

Each `tracy/<layer>/<mode>_ops.csv` is the copied raw Tracy device-op table.
The sibling `<mode>_perf_report.csv`, `<mode>_perf_report_stacked.csv`, and PNG
are generated with signpost-filtered `tt-perf-report`. `provenance_manifest.md`
records exact source and artifact hashes.

| Layer | Functional prefill us | Fused prefill us | Functional decode us | Fused decode us |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 23790 | 16736.260 | 5396 | 3848.370 |
| 1 | 27269 | 20161.470 | 5922 | 4269.910 |
| 3 | 408094 | 43401.980 | 11081 | 5338.730 |

The report conclusions are:

- the KDA causal-convolution promotion removes about 1 ms from both GDN
  prefill layer kinds despite its required device BF16 row-major boundary;
- the QSA profile no longer has the former two-to-24 K/V expansion, padded
  two-head reshape, inverse head permutes, or concat-heads operation;
- decode retains V sharding through the batch-one cache update, uses one fused
  K/V paged update, and uses the dedicated decode SDPA;
- the filtered measured windows contain no host/from-torch/to-torch operation;
  explicit layout movement is limited to target-op contracts. Layer-3 decode
  has three interleaved-to-sharded rows, two sharded-to-interleaved rows and
  one 0.34 us reshard inside the legal decode Q/K normalization/SDPA bridge.
  The maximal-sharded candidate proved that removing it violates RMSNorm's
  input contract. Other tilize/untilize rows are operator-owned boundaries for
  KDA, sparse metadata, gather, paged cache, top-k/scatter, or SDPA—not host
  fallback or an avoidable round trip.

## Reliability evidence

- `final_correctness_trace_alloc.xml`: 35 passed in 55.413 seconds.
- `final_stress.xml`: 9 deterministic replay cases passed in 23.073 seconds.
- `final_watcher.xml`: 35 passed in 94.476 seconds, separately from Tracy.
- `watcher_final_v5/generated/watcher/watcher.log`: no assert, fatal, hang,
  timeout, NoC/kernel/sanitizer error; dump completed and device 0 detached.
- `final_long_context.xml`: seven advertised-context tests.
- post-run `tt-smi` reports zero corrected/uncorrected DRAM errors; no reset or
  recovery was needed.

The successful processes emit existing nanobind teardown diagnostics and the
intentional single-visible-chip P300 warning. They occur after passing tests
and clean device close, outside measured execution.

## Reproduction

Source the pinned environment first:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
```

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py \
  -m 'not long_context'

pytest -q -s --long-context -m long_context \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py

QWEN38_FUSED_PERF_DECODE_REPLAYS=10 pytest -q -s --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder_perf.py

pytest -q -s --count=3 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_fused_decoder.py::test_decode_trace_replay_and_determinism
```

Exact watcher, Tracy, and report commands are in `work_log.md`.

## Limitations

- Performance uses the stage's zero-input synthetic harness; correctness uses
  real checkpoint tensors. Sparse expert utilization remains unmodeled by
  `tt-perf-report`, so the CSVs are latency/topology evidence rather than a
  routed-token FLOP-efficiency claim.
- The 95.37 GiB PLE n-gram table remains the functional stage's accepted
  caller-preprocessing boundary. PLE projection, convolution, gating, and
  state are measured in this decoder.
- Direct `sparse_sdpa` is blocked in this scoped file because it needs a
  persistent combined row-major KV cache, whereas arbitrary paged fill/update
  requires the existing tiled caches. Supporting it requires a TTNN row-major
  paged writer or a tiled-cache sparse reader.
