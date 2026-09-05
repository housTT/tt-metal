# Gemma 4 26B A4B optimized decoder

This is the single-device optimized-decoder stage for
`google/gemma-4-26B-A4B-it`. It starts from fused-decoder checkpoint
`6e67efeb6251e655cb27c39153db5af1da90d68b` and does not begin multichip,
full-model, or vLLM work.

## Delivered runtime

`OptimizedDecoder` owns both material paths; tests statically reject a
functional fallback and dynamically require optimized counters. The selected
Blackhole P300C policy is:

- B1 decode keeps residual, norm, attention, dense MLP, and expert inputs on a
  coherent 22-core width-sharded path. QKV uses its fused-equivalence-safe
  G22/block2/subblock 1x3 path. Sliding O and both layer kinds' packed dense
  gate/up plus down use one-reader DRAM-sharded matmuls; full O retains
  G22/block6/subblock 1x4. Padding is private to the optimized kernels.
- B32 decode uses correctness-safe G8 attention and BFP8 packed expert
  gate/up. B1 experts use packed BFP4/LoFi gate/up block22 and BFP8/LoFi down
  block11. Separate expert GeGLU remains available and uses accurate GELU
  fused into multiply, but is slower.
- Prefill runs only routed experts in logical 32-token chunks, with a separate
  legal tail configuration for arbitrary public sequence lengths. Attention
  retains the correct automatic TTNN matmuls after measured 2D candidates
  either lost fused equivalence, exceeded L1 capacity, or regressed latency.
- Decode routing creates the scatter result directly in row-major form and
  feeds that metadata to sparse matmul. A 256-byte zero base is allocated once
  before trace capture at a stable address. This removes two untilizes and one
  unary operation per replay without adding a hot-path conversion.
- Router projection, shared FFN normalization, expert score scaling, and the
  final scalar are folded. The router remains FP32; reduced precision fails
  numerical gates.
- Paged fill/update, paged SDPA, natural/shared cache views, sliding-ring wrap,
  and the caller-owned cache contract are preserved.

Frozen hashes:

- `optimized_decoder.py`: `feebc8cb2f20ad9ba81c7d0f50f8323694d6d91ebb31cb9072e18b0f6b0a9c45`
- `test_optimized_decoder.py`: `01ed0de36451891c6c41968baeeae34f1cb77acf4e3c6db0c322506fb83bd349`

## Correctness and cache policy

The acceptance threshold is PCC 0.995. Final real-checkpoint evidence includes:

| Case | Sliding | Full |
| --- | ---: | ---: |
| prefill vs HF | 0.999252011 | 0.998321241 |
| decode vs HF | 0.996587866 | 0.995966596 |
| fused vs optimized prefill | 0.999698712 | 0.997576330 |
| fused vs optimized decode | 0.996685143 | 0.995892446 |
| B32 minimum-user traced decode | 0.995802051 | 0.999781521 |
| eager/replay and repeat replay | 1.000000000 | 1.000000000 |

BF16 is the only accepted optimized KV-cache policy. It passes cache-consuming
decode after nonaligned prefill at sliding length 1025 and full length 33 for
both natural and shared full-attention views; minimum HF PCC is 0.997232031
sliding and 0.995421141 full. BFP8 remains an explicitly gated experimental
candidate, not a recommended or accepted path: corresponding minima are
0.994624887 and 0.988673202, below the 0.995 gate, despite deterministic
eager/replay output.

The final suite also covers B1/B32 traces, A/B/A mutable buffers, B2 prefill,
1,104 wrap-crossing replays, and logical lengths 1/31/32/33, 63/64/65,
127/128/129, and 1023/1024/1025.

## Context and allocation

Both layer kinds pass traced decode at current position 262,143 and physical
nonaligned prefill of 262,143 tokens. The advertised 262,144-token context,
page tables, block geometry, cache ownership, and BF16 cache dtype are
unchanged.

Measured persistent state is 2,158,283,008 bytes for a representative sliding
layer and 2,130,482,432 bytes for a representative full layer. Each includes
the selected 256-byte routing base, batch-aware packed expert weights, and
deduplicated attention aliases plus only the selected DRAM-sharded copies. The
setup-only interleaved packed-dense source is released after conversion.
Caller-owned KV cache and transient activations are excluded. The 25/5
layer-kind projection is 64,609,487,360 bytes; full-stack placement remains
outside this stage.

## Warmed performance

Measurements use one P300C, sequence/current position 1024. Prefill is warmed
host latency; decode is the mean of 1,000 warmed trace replays. The frozen
final-test-hash record is `candidate_runs/final_reviewfix2_perf_v6.json`.

| Workload | Fused baseline | Selected BF16 | Reduction |
| --- | ---: | ---: | ---: |
| sliding prefill B1 | 278.380476 ms | 96.365912 ms | 65.38% |
| full prefill B1 | 279.611919 ms | 107.636814 ms | 61.50% |
| sliding decode B1 | 1.309336 ms | 0.811915 ms | 37.99% |
| full decode B1 | 1.487052 ms | 0.859545 ms | 42.20% |
| sliding decode B32 | 19.500527 ms | 12.837788 ms | 34.17% |
| full decode B32 | 19.317333 ms | 12.541034 ms | 35.08% |
| sliding prefill B32 | - | 3036.028061 ms | serving contract |
| full prefill B32 | - | 3441.924553 ms | serving contract |

The final selection improves the prior correct review-fix default from
0.841313/0.864549 ms to 0.811915/0.859545 ms. A faster sliding DRAM-sharded QKV
candidate was rejected because direct fused decode PCC was 0.987622. The
rejected BFP8-cache timings do not qualify because cache-consuming PCC fails.

## Operation and movement audit

The fresh selected profile has 615 prefill ops for each layer kind and 70/73
decode ops for sliding/full. Relative to the original 72/75 topology, routing
removes two
`UntilizeWithUnpaddingDeviceOperation` rows and one `UnaryDeviceOperation` row
per replay, while packed dense removes one matmul and adds two device slices.
The resulting net count is two below the original; acceptance is based on the
measured latency win, not op count. The remaining two routing untilizes are required by top-k's
values/indices contract; the remaining score tilize is required by its tile
consumer. There is one QKV matmul, one packed gate/up sparse matmul, and one
expert-down sparse matmul; no Torch conversion, host fallback, repeated input
projection, or activation reshard loop occurs in the measured windows.

| Region | Candidate coverage | Decision |
| --- | --- | --- |
| attention | G8/G22/G32, blocks/subblocks, padding, DRAM readers/shards | sliding O reader-1 DRAM-sharded; QKV/full O G22; B32 G8 |
| dense MLP | separate/packed, R11/R22, BFP4/BFP8, DRAM-sharded | packed BFP8/LoFi reader-1 DRAM-sharded on R22 |
| experts | packed/separate, BFP4/BFP8, block11/22/44, N/subblocks, L1 | B1 packed BFP4 block22; B32 packed BFP8 block11 |
| prefill attention | 2D grids/blocks, per-role isolation, input L1 | automatic path retained |
| router/routing | FP32/BFP8, input L1, row/tile metadata | FP32 plus selected row-major metadata |
| cache | BF16/BFP8, nonaligned prefill then traced consumption | BF16 only accepted |

Notable review-remediation trials:

- Packed expert block44 is legal and correct but ties inconsistently with
  block22. A fair separate accurate-GeGLU path is about 5% slower.
- The first R22 matrix was invalidated because loader flags were not retained;
  tests now assert counters from requested environment intent. Genuine
  reader-1/2/3 runs select sliding O and packed dense/down at reader-1. QKV
  reader-1 was faster locally but failed direct fused decode PCC at 0.987622.
- Full O reader-1 first hit an L1/CB overlap at its default local K block 32; a
  legal block-8 adaptation passed
  correctness but lost at 0.876961 ms. Readers 2/3 also lost.
- Full O G8x4 passes correctness but is slower. QKV G8x4 fails full-attention
  downstream decode PCC; sliding G8x8 is locally faster but fails direct
  fused-equivalence decode PCC. Full O input-L1 passes correctness but regresses
  prefill from 107.524 ms to 108.162 ms and adds movement.
- BFP8 cache, B32 BFP4 experts, dense BFP4, expert-down BFP4, attention LoFi,
  and router-input L1 are all rejected by recorded numerical or whole-layer
  performance gates.

## Profiler and advice closure

Raw captures:

- `generated/profiler/gemma4_optimized_reviewfix2_sliding/reports/2026_09_05_14_24_18/ops_perf_results_2026_09_05_14_24_18.csv`
- `generated/profiler/gemma4_optimized_reviewfix2_full/reports/2026_09_05_14_24_38/ops_perf_results_2026_09_05_14_24_38.csv`

Advice-enabled human-readable tables plus CSV/summary/PNG outputs are under
`tt_perf_report/final_reviewfix2_v5_{sliding,full}`.

| Kind/window | Ops | Device sum | Gaps | Span | Modeled DRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding prefill | 615 | 95.822331 ms | 0.747437 ms | 96.569768 ms | sparse model omitted |
| full prefill | 615 | 107.114693 ms | 0.594407 ms | 107.709100 ms | sparse model omitted |
| sliding decode | 70 | 0.772742 ms | 0.068347 ms | 0.841089 ms | 160 GB/s (31.2%) |
| full decode | 73 | 0.827871 ms | 0.065011 ms | 0.892882 ms | 128 GB/s (25.1%) |

The same-profile five-replay decode host timings are 0.867675/0.918276 ms for
sliding/full, leaving 0.026586/0.025394 ms outside the device span. A
conservative 123,469,824/106,348,544-byte model gives 0.241152/0.207712 ms
floors at 512 GB/s and 146.798/119.107 GB/s effective bandwidth from span.
Independent 1,000-replay timings, not these short profiler runs, rank candidates.

All advice is closed with measurements: selected O and packed-dense/down use
the winning DRAM-sharded reader-1 programs; QKV DRAM sharding fails the
cumulative numerical gate; full O legal variants lose; larger expert geometry
did not win; prefill grid/block variants either failed the cumulative gate or
lost; and input-L1 variants added a copy and regressed.
No decoder optimization is deferred to a later pipeline stage.

## Verification

- Complete final suite: `44 passed, 15 skipped in 121.22s`. Skips are explicit
  context/performance/serving/BFP8-candidate gates run separately.
- Context: `4 passed in 183.75s`.
- Canonical performance: `4 passed in 48.50s`.
- Serving B32 prefill: `2 passed in 26.52s`.
- Watcher/stress: `11 passed in 97.58s`; no watcher error, healthy DRAM on all
  four P300C devices, and no remaining workload process.
- Persistent accounting: `5 passed in 11.68s`, including the selected routing
  buffer.
- Fresh Tracy and `tt-perf-report` captures passed for both layer kinds.
- Python compilation, scoped pre-commit hooks, JSON validation, and whitespace
  checks are required before commit. This stage is Python-only, so AGENTS.md
  requires no C++ build.

Exact commands, the full candidate ledger, initial independent review,
AutoDebug/AutoFix remediation, and checklist evidence are in `work_log.md`,
`STAGE_REVIEW.md`, `AUTODEBUG.md`, and `AUTOFIX.md`.
