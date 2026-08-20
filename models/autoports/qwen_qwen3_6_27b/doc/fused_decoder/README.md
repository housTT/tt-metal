# Qwen3.6-27B fused decoder

This stage adds the graph-fused single-device decoder in `tt/fused_decoder.py`. It preserves the functional decoder's prefill/decode API, paged KV cache, recurrent state, determinism, batch-32 behavior, non-aligned logical lengths, and 262,144-token context contract. `doc/context_contract.json` is unchanged: packed weights replace and deallocate their unpacked copies, and cache dtype, layout, capacity, and public limits are unchanged.

The tests monkeypatch the functional test module to instantiate `FusedDecoder` and assert runtime fusion counters. They cannot pass by silently constructing `FunctionalDecoder`. The frozen source and test hashes used for final correctness, profiling, and Watcher evidence are in `source_manifest.sha256`.

## Selected graph

| Region | Selected runtime graph |
|---|---|
| Full Q/K/V and heads | Host-repacked `[Q,K,V,gate]`, one shared-LHS linear, dedicated prefill/decode head creation, gate slice; unpacked weights deallocated |
| Full prefill RoPE | Dedicated `rotary_embedding` over the partial rotary width plus passthrough concat |
| Attention and MLP gates | Sigmoid/SiLU folded into the consuming binary multiply |
| Linear input projections | Four shared-LHS projections packed into one linear; 48-wide beta/a tails tile-aligned |
| Linear Q/K replication | Direct `repeat_interleave`, selected by measured end-to-end latency |
| Linear Q/K L2 norm | Dedicated RMSNorm plus exact constant scale |
| DeltaNet transpose matmuls | `transpose_a` / `transpose_b` attributes instead of material transposes |
| Linear core gate | SiLU folded into the consuming multiply |
| Decode beta gate | Sigmoid folded into the consuming multiply |

The linear core normalization intentionally retains the stable spelled reduction. Dedicated RMSNorm, including FP32 destination accumulation, becomes nonfinite after a 262,144-token recurrent prefix. That candidate is therefore not a valid fusion even though it is correct at short sequence lengths.

## Correctness

The acceptance threshold is PCC >= 0.995. Final real-weight results from `correctness/final/pytest.log` are:

| Gate | Fused PCC/result |
|---|---:|
| Linear non-aligned prefill, 65 tokens | 0.998477649 |
| Linear first decode after 65 | 0.999461964 |
| Linear traced decode after 65 | 0.999649418 |
| Linear repeated trace determinism | 1.000000000 |
| Full non-aligned prefill, 33 tokens | 0.998800717 |
| Full decode after 33 | 0.997618106 |
| Full forced-chunk prefill, 257 logical / 384 physical | 0.998416664 |
| Full traced decode at position 262,143 | 0.999814352 |
| Native-position repeated trace determinism | 1.000000000 |
| Linear batch-32 minimum decode PCC | 0.999417208 |
| Full batch-32 minimum decode PCC | 0.997578817 |
| Linear native 262,144 prefill and traced decode | finite, pass |
| Full 32,769 and 262,144 prefill | finite, pass |

The complete non-profiler suite passed 12/12 in 102.91 seconds. It includes non-aligned 33/65-token paths, forced chunk tails, both layer kinds, paged cache updates, repeated trace replay, both batch-32 paths, native linear prefill/decode, full decode at the last native position, and full prefill at both 32,769 and 262,144 tokens.

## Performance

Times are the sum of `DEVICE KERNEL DURATION [ns]` strictly between warmed signposts. Decode is warmed trace replay.

| Layer | Phase | Functional | Fused | Improvement |
|---|---|---:|---:|---:|
| Linear | prefill | 6,046.000 us / 177 ops | 5,605.562 us / 159 ops | 440.438 us / 7.29% |
| Linear | traced decode | 2,901.000 us / 88 ops | 2,728.252 us / 71 ops | 172.748 us / 5.96% |
| Full | prefill | 2,594.000 us / 51 ops | 2,362.104 us / 29 ops | 231.896 us / 8.94% |
| Full | traced decode | 2,639.000 us / 50 ops | 2,334.152 us / 45 ops | 304.848 us / 11.55% |

Raw profiler CSVs are under `perf/raw_*/reports/ops_perf_results.csv`. Signpost-filtered `tt-perf-report` CSVs, summaries, and plots are under `perf/reports/`. `perf/summary.csv` is the final before/after table. `candidates/index.csv` and each candidate bundle retain the exact source snapshot, PCC/failure log, raw signpost CSV, report, and disposition where applicable.

## Data movement audit

There is no `torch`, `from_torch`, `to_torch`, explicit tilize/untilize, reshard, or host fallback in a measured method. Host-side Torch is used only once during construction to pack weights.

The final full prefill graph has no layout conversions. Full decode has two required `ShardedToInterleaved` and two `InterleavedToSharded` operations: dedicated decode head creation emits height-sharded Q/K, while per-head RMSNorm rejects that layout, so Q/K must move to interleaved memory for normalization/RoPE and then return to the paged-cache/SDPA shard contract. Linear prefill reports 11 tilize and 12 untilize-family composite operations; linear decode reports four tilize and five untilize-family operations. These arise inside TTNN composites for direct head replication, the stateful 10,240-channel convolution window, and extraction of the final recurrent-state row. The following alternatives were measured or contract-tested:

- padding beta/a to tile boundaries: selected at 5,605.562 us / 159 prefill ops and 2,728.252 us / 71 decode ops after native-context validation;
- replacing direct Q/K repetition with reshape/concat: slower measured graph; rejected;
- grouped `conv1d`: repository tests mark groups above 5,120 as OOM, while this graph requires 10,240 BF16 groups;
- alternate final-row reduction: requires a larger broadcast/reduction graph than the retained slice;
- sharded SDPA into `nlp_concat_heads_decode`: hardware fatal, `Sharded output not supported for GQA`.

No explicit `ReshardDeviceOperation` occurs in any of the four final signpost windows.

## Exhaustive graph-fusing assessment

| Pattern | Outcome |
|---|---|
| Dedicated activation | Retained existing convolution SiLU; folded attention, MLP, core, and beta activations into binary consumers. |
| Softmax / stable softmax / SDPA | Full attention already uses dedicated causal, chunked, and paged SDPA; DeltaNet has no softmax subgraph. |
| RMSNorm | Full norms retained. Linear L2 norm fused. Linear core dedicated RMSNorm rejected at native context, including FP32 accumulation. |
| Distributed RMSNorm | Not applicable to the single-device stage. |
| Split/create QKV heads | Retained after host-reordering Q/gate rows into tile-aligned `[Q,K,V,gate]`; dedicated prefill and decode head creation produces a major latency win. |
| Concat heads prefill | Correct at the final PCC but 2,366.470 us versus 2,356.778 us on its identical base; permute/reshape retained. |
| Concat heads decode | Tried with sharded SDPA; GQA sharded output is unsupported on this path. |
| RoPE | Dedicated partial prefill RoPE retained. Per-lane decode-axis formulation was correct (decode PCC 0.997824526) but slower at 2,346.213 us versus 2,339.680 us on its identical base. |
| TopK | Not present. |
| RepVGG / spatial mean | No matching convolution branch or spatial reduction. |
| Shared-LHS matmul | Packed full Q/K/V/gate and all four linear inputs retained; packed MLP is PCC-correct but slower in all four phases. |
| Permute/reshape identity | Remaining permutations change semantic head/token order; none is an identity. |
| Convolution bias/scale/activation | Stateful depthwise rolling window has no viable TTNN fused contract at 10,240 groups; SiLU is already dedicated. |
| Matmul/linear activation | Selective packed-output activations are invalid; adjacent activation folds are performed at binary consumers. |
| Binary input activation | Retained for attention sigmoid, MLP SiLU, linear core SiLU, and decode beta sigmoid. |
| Matmul bias | DeltaNet `dt_bias` precedes softplus and is not a matmul bias. |
| Transpose plus matmul | Retained transpose attributes for DeltaNet matmuls. |
| Slice after matmul | Required to distribute packed projection outputs; pushing slices before the matmul recreates separate projections. |
| BatchNorm / pad consumer | No BatchNorm or compatible pooling/conv pad consumer; public non-aligned lengths retain internal padding and tail slicing. |
| Reduction plus reshape / scaled sum | Required reductions keep semantic singleton dimensions; L2 normalization is the applicable fused replacement. |
| Decode RoPE reshape folding | Correct per-lane formulation measured and rejected on latency. |

No remaining compatible dedicated operation, structural simplification, shared-input merge, or adjacent attribute fold improves the correct graph on this hardware.

## Watcher and artifacts

The separate Watcher run passed six representative final fused-path tests in 22.31 seconds: non-aligned full/linear paths, paged full trace replay, forced chunked full prefill, and batch-32 coverage for both layer kinds. Scanning `watcher/final/generated/watcher/watcher.log` found no Watcher error/assert/hang/timeout, NoC error/timeout, kernel assert, or device hang. Watcher and Tracy were never enabled together; the complete 12-test non-Watcher suite supplies native-capacity and stress coverage.

Exact commands, native-context failure isolation, artifact paths, and commit/review records are in `work_log.md`.
