# Qwen3.8-Flash-Next optimized full model

| Batch-1 P300 1x2 headline | Before | Selected frozen source | Change |
| --- | ---: | ---: | ---: |
| Warm repeated prompt-128 TTFT | 11.316 s | **5.461 s** | **-51.7%** |
| First-request prompt-128 TTFT, model construction/prepack excluded | 123.910 s | **7.856 s** | **-93.7%** |
| Traced token-out decode, 126 measured tokens | 644.562 ms/token, 1.551 t/s/u | **231.490 ms/token, 4.320 t/s/u** | **-64.1%, +178.4%** |
| Traced teacher forcing, 98 measured rows | — | **289.150 ms/token, 3.458 t/s/u** | full-logits D2H included; sampling, token feedback, and compact-token readback excluded |
| Instrumented traced token-out, 126 measured tokens | — | **261.457 ms/token, 3.825 t/s/u** | per-token completed-wall/layer/host deltas enabled; reported separately |

All final selected-path results above use frozen source digest
`e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`
on Blackhole dies 0 and 1. Token-out includes all 48 layers, exact expert and
PLE host service, device greedy sampling, `tt_out_tok`, compact caller token
readback, device position/RoPE advance, changed-only page-table logic, and
persistent CCL buffers. Exact prepacking materializes 24,576 experts / 68.080
GB in 240.979 s at model construction and is excluded from request TTFT. vLLM
work was not started.

## Selected full path

The target is a fixed `1x2` `FABRIC_1D` P300 mesh. The optimized generator
covers embeddings, all decoder layers, hyperconnection and final norms,
vocabulary-sharded LM head, sampler-ready logits, greedy/top-k/top-p sampling,
KV/recurrence/expert/PLE caches, split traces, collectives, residual layouts,
host boundaries, and request orchestration.

The inherited decoder policy remains intact:

- inter-layer residual ABI `S` is mesh-sharded BF16 `[1,1,4*M,1280]` on
  dimension 3, DRAM-interleaved per local shard, with one ingress partition
  and one final gather;
- routed experts are exact EP2 top-10 BFP4/LoFi, shared projections are
  BFP8/LoFi, GDN is BFP8/HiFi2, and QSA input/output is BF16/HiFi2;
- QSA raw KV/index caches remain BFP8 and compressed index state remains BF16;
- CCL partials remain BF16, synchronous, two-link, packet size 8192;
- selected QSA decode 1D sharding and inherited projection, program, and
  kernel configurations remain selected.

No replicated inter-layer stream, broad datatype frontier, larger rejected
expert cache, async CCL, fused matmul-reduce-scatter, or rejected GDN policy
was substituted. The retained GDN LoFi artifact is not a current full-path
A/B because its route/miss work differs and it has no full-model accuracy
gate. Pareto precision selection remains owned by `$datatype-sweep`.

## Accepted optimizations

| Change | Exactness condition | Result |
| --- | --- | ---: |
| Nonblocking front/back CQ0 trace segments | producer/consumer order remains at route/token boundaries | 644.562 -> 605.189 ms/token |
| Owner-only expert H2D plus immutable peer zero D2D | identical packed owner shard and exact-zero non-owner | 605.189 -> 454.810 ms/token |
| Deferred expert and PLE completion | CQ0 completion is observed at the next exact route/token boundary | zero per-miss and per-PLE fences |
| Deterministic full host prepack | all 512 exact experts/layer exist before requests | removes 27.806 s source pack from the prior 126-token lazy window |
| BFP8/HiFi2 LM-head weights | exact vocabulary order and sampler contract | 1.693 -> 0.972 ms/row, -42.6% |

Full-model construction preloads all packed experts by default. Focused
cold/warm tests keep lazy packing to expose identical cold and warm regimes.
Routes, misses, slot generations, EOS/history semantics, and request isolation
are unchanged; there is no route prediction or approximate expert reuse.

## Performance and host boundary

The frozen no-timeline workload is the reportable warmed result: prompt 128,
generate 128, 126 measured traced transitions, TTFT 7.856 s, and 231.490
ms/token / 4.320 t/s/u. The optional timeline collector is deliberately kept
separate because detailed Python per-layer/per-token accounting adds overhead.

| Instrumented 128+128 decode field | Mean per token / total |
| --- | ---: |
| Completed token-out wall | 261.457 ms/token |
| Model submit | 259.063 ms/token |
| Layer boundary | 258.795 ms/token |
| Expert service | 252.136 ms/token |
| Compact route read / TT stall | 100.052 ms/token |
| Cache, control, and DMA submission | 151.830 ms/token |
| Nonblocking trace submission | 0.446 ms/token |
| PLE lookup and staging | 5.656 ms/token |
| Compact caller read/completion | 2.366 ms/token |
| Expert hits / misses | 18,873 / 41,607 |
| Owner H2D / exact-zero D2D | 115.035 / 115.035 GB |
| Decode source pack | 0 s |
| PLE rows / bytes read | 680 / 217,600 |
| Expert / PLE completion syncs | 0 / 0 |

The frozen lazy cold/warm prompt-128 prefill test is exact: cold is 127.019 s
with 10,994 packed misses, 118.091 s source packing, and 1,816 PLE rows read;
warm is 5.461 s with 10,994 packed hits, zero source pack, and zero PLE table
reads. Both choose token 248046 and transfer 30.396 GB of owner expert data.

### Decoder-stack and host lower bounds

The preserved decoder-stack lower bound is
`35*2.082220 + 1*2.613159 + 12*3.027236 = 111.817691 ms/token`.
The frozen profiler adds a conservative selected terminal floor of 1.387908
ms: 0.896805 ms for the four BFP8 LM-head rows plus 0.491103 ms for traced
greedy sampling. Instrumented PLE service is 5.655693 ms/token.

The measured workload moves 912,976,457 owner H2D bytes/token. Two different
bounds must not be conflated:

| Bound component | ms/token |
| --- | ---: |
| Decoder stack | 111.817691 |
| Selected LM head + greedy sampling floor | 1.387908 |
| Owner H2D at raw x4+x4 ceiling, 15.753846 GB/s aggregate | 57.952608 |
| PLE lookup/staging | 5.655693 |
| **Optimistic physical lower bound** | **176.813899** |
| **Observed no-timeline token-out** | **231.489929** |
| **Gap to optimistic bound** | **54.676029 ms, 30.923% of bound** |

The raw-link bound intentionally omits peer-zero D2D plus index/cache control.
The focused pure concurrent-owner path reaches only 6.497517 GB/s, equivalent
to 140.511592 ms/token of H2D staging. Adding that measured implementation
cost to stack, terminal, and PLE gives 259.372884 ms, which exceeds observed
token-out by 27.883 ms and proves that transfer already overlaps TT work; it
is a diagnostic estimate, not a physical lower bound.

Because the optimistic gap exceeds 15%, AutoFix split the remaining boundary.
Exact staging depths 2 and 10, owner partitioning, owner coalescing, and a
prestarted owner-thread pool were all A/B tested. The selected serial depth-1
service is 4.052 ms p50; every grouping/threading policy is slower, and depth
10 adds 1.194 GB/rank without a p50 win. Pure dual-owner submission reaches
only 41.24% of the two-link raw ceiling. AutoFix therefore failed to find a
safe Python-level speedup. The next credible lever is one native batched TTNN
H2D submission/event per physical owner, first gated on materially exceeding
6.50 GB/s; it is outside the current Python cache API.

## LM head, logits, and sampling

Rank 0 owns vocabulary IDs `0..124159`, rank 1 owns `124160..248319`, and each
rank preserves splits `32768,32768,32768,25856`. Selected BFP8/HiFi2 endpoint
weights occupy 981,032,960 bytes/device. No replicated vocabulary or pre-head
full-vocabulary gather exists.

| LM-head candidate | Mean row | Accuracy | Verdict |
| --- | ---: | --- | --- |
| BF16/HiFi2 interleaved | 1.693 ms | PCC .999988, top-5 5, top-100 99 | reference |
| **BFP8/HiFi2 interleaved** | **0.972 ms** | **PCC .999922, top-5 5, top-100 98** | **selected** |
| BFP8/HiFi2 DRAM s1/c40 | — | static CB 5,815,680 B > L1 | compile reject |
| BFP8/HiFi2 DRAM s4/c40 | — | CB region overlaps L1 allocation | compile reject |
| BFP8/HiFi2 DRAM s5/c40 | 1.331 ms | PCC .999923, exact greedy, valid top-k/top-p | correct but 36.24% slower |
| BFP4/LoFi -> BF16 sampler boundary | 0.726 ms | PCC .976245, top-5 4, top-100 67 | accuracy reject |

Greedy stays on device. Both generic local-top32/k=1 and specialized full-
vocabulary argmax match host argmax; specialized argmax is selected because it
is faster, 0.664 vs 0.899 ms in the frozen A/B. The traced greedy sampling
window is about 0.491 ms: local-vocabulary all-gather 0.401 ms, untilize/unpad
0.039 ms, and device argmax 0.051 ms. `TopKDeviceOperation` in token-out is the
expert router, not terminal sampling. Traced top-k/top-p retains persistent
seed control, local-candidate offsets, direct device token feedback, and device
state advance.

## Traces and serving contract

Model, sampling, and position traces keep `tt_out_tok`, position/RoPE,
KV/recurrence state, changed-only page tables, sampling controls, and CCL
buffers persistent. Replay segments are nonblocking. Only declared compact
expert/PLE lookup-control, exact weight/PLE DMA, and caller-visible compact
token readback are host-side.

The generator keeps explicit cache, page-table, position, prompt-length,
batch, slot, and request state. Fixed slots, prompt lengths 1 and 33, 30
inactive rows, distinct page tables, reset/reuse, and PLE history isolation
pass at eager batch 32/context 4096. Advertised context remains 262,144. The
optimized generator accepts the non-aligned 201-token AIME prompt; static
lengths 1, 31-33, 63-65, 127-129, 2047-2049, 262143, and 262144 also pass.
The split/mixed JUnit contains four passes and one environment-gated sampler
A/B skip; `final_frozen_sampler_strategy_ab.xml` reruns that exact skipped node
with its required environment and passes, so the combined contract is 5/5.

## Accuracy and qualitative evidence

| Frozen selected gate | Top-1 | Top-5 | Top-100 | Result |
| --- | ---: | ---: | ---: | --- |
| AIME24 prefill, non-aligned 201-token prompt | 100% | 100% | 100% | pass |
| AIME24 traced teacher forcing, 99 rows | 91.919% | 100% | 100% | pass |

Teacher forcing measures 98 traced rows at 289.150 ms/token / 3.458 t/s/u and
explicitly reads full logits to the host; it excludes sampling, device token
feedback, and compact caller token readback. The separate 100-token free-run
is coherent and non-degenerate, with 12.273 s TTFT and 224.634 ms/token /
4.452 t/s/u. The fresh shared explanation, coding, and summarization suite also
passes with exact chat templates and traced on-device sampling.

## Profiler conclusions

Frozen selected-policy Tracy captures were isolated from Watcher and processed
with `tt-perf-report` advice enabled.

| Representative full-path window | Ops | Device time | Host gaps | Sampling | LM head |
| --- | ---: | ---: | ---: | ---: | ---: |
| Layer 0 GDN + endpoints | 195 | 3.213 ms | 125.283 ms | 0.491 ms | 0.895 ms |
| Layer 1 PLE+GDN + endpoints | 265 | 3.632 ms | 153.876 ms | 0.491 ms | 0.899 ms |
| Layer 3 QSA + endpoints | 285 | 4.574 ms | 143.112 ms | 0.492 ms | 0.897 ms |
| Warm layer-0 prefill + endpoints | 201 | 5.736 ms | 6.368 ms | n/a | included |

The layer-0 decode report reaches 139 GB/s and 27.2% modeled DRAM roofline.
The largest gaps surround exact expert-slot service; reduced profiler windows
are lazy-packed, so these gaps are diagnostic rather than additive full-stack
timing. Every LM row receives DRAM-sharded advice, but the focused exact
DRAM-sharded A/B is slower. There is no missing-device-data or profiler-buffer
overflow signature. Raw CSVs, compact tables, hashes, commands, and source
provenance are under `final_frozen_profiler/` and `profiler_provenance.txt`.

Applicable `$multichip` and `$optimize` checks are evidenced:

| Area | Selected decision | Evidence |
| --- | --- | --- |
| Mesh/layout | TP2 fractured residual, EP2 routed experts, TP2 shared expert, contiguous TP2 vocabulary | batch-32 state gate, full token-out, representative profiles |
| CCL | BF16 synchronous two-link packet-8192 with persistent buffers | full-48 traces, profiler CCL rows, fallback audit |
| Matmuls/dtypes | inherited decoder policy; BFP8/HiFi2 endpoint only | real endpoint frontier, AIME, profiler |
| Programs/kernels | inherited selected QSA/GDN/expert configs and four LM splits | rejection ledger, DRAM A/B, raw operation CSVs |
| Host/data movement | owner H2D, local zero D2D, deferred CQ0 completion, exact prepack | completed bandwidth matrix and per-token timeline |

The full-48 selected token-out also passes Watcher and trace-allocation
tracking in a process separate from Tracy, with no Watcher, NoC, CB, assert,
hang, or allocation fault. Instrumented Watcher timings are not performance
evidence.

## Capacity, artifacts, and limitations

No capability was reduced. Batch-1 context 262144 plans 10,005,165,144
bytes/device with 24,220,355,496 bytes headroom. Batch-32 sequence 4096 plans
23,393,673,304 bytes/device with 10,831,847,336 bytes headroom. The serving
contract remains traced batch 1 and eager batch 32.

Primary frozen artifacts are `final_frozen_performance_no_timeline/`,
`final_frozen_performance/`, `final_frozen_aime_prefill_teacher_autoreg.xml`,
`final_frozen_accuracy/`, `final_frozen_qualitative/`,
`final_frozen_cold_warm.xml`, `final_frozen_split_sampling_mixed_contracts.xml`,
`final_frozen_sampler_strategy_ab.xml`,
`final_frozen_batch32_and_context_capacity.xml`,
`final_frozen_full48_watcher_alloc_tracker.xml`,
`final_frozen_static_contracts.xml`, `final_frozen_lm_head_bfp8_hifi2.xml`,
the `autofix_lm_head_*` and `autofix_cache_service_*` matrices, and
`final_frozen_profiler/`. `before_after.csv`, `optimization_matrix.csv`,
`lower_bound.csv`, and `work_log.md` provide compact provenance.

Measured limitations are explicit: exact prepack costs about 241 s and 68.080
GB host RAM; traced token-out is batch 1 while eager serving state passes batch
32; CPU-only Torch lacks a pinned allocator; pure dual-owner submission reaches
only 41.24% of the raw two-link ceiling; and the CCL API emits a future
deprecation warning. The remaining host-submission opportunity requires a
native batched TTNN H2D boundary. No vLLM work was started.

AutoFix evidence is in `AUTOFIX.md`; independent review iterations are retained
beside this README. The final fresh-context rereview in `STAGE_REREVIEW.md`
returned `clean-pass` with no required work.
