# Gemma 4 26B-A4B-it fused decoder

This stage adds the single-device TTNN `FusedDecoder` for
`google/gemma-4-26B-A4B-it`. It preserves the completed functional decoder's
prefill, decode, paged-KV-cache, logical-length, determinism, layer-kind, and
262,144-token context contracts. It does not begin optimized-decoder,
multichip-decoder, full-model, or vLLM work.

## Selected graph

The default path selects every candidate that improved the measured graph:

- dense `[up, gate]` weights share one projection, with fast GELU fused into
  the following binary multiply;
- router feature and hidden scales are folded into its FP32 projection;
- the equal unweighted portion of the dense, router, and expert pre-FFN
  normalizations is computed once; the two learned norm weights are folded
  into their projection weights;
- expert routing scale is folded into each expert down weight;
- expert `[up, gate]` weights share one sparse projection, with accurate GELU
  fused into the following binary multiply;
- the packed expert projection is zero-padded from 1,408 to 1,536 columns,
  changing its sparse program from 4 to 48 cores; and
- each 1,024-token prefill uses 32-token expert groups and one final list
  concat rather than 31 growing concats; and
- the learned final residual scale is fused into the residual add.

The padded tensor increases resident expert gate/up weight storage by 9.1%
relative to an unpadded packed tensor; original tensors and temporary folded
tensors are deallocated after construction. Capacity testing below proves this
does not reduce the advertised contract. Candidate environment switches are
retained only for reproducible A/Bs. The exact resolved default policy is
asserted by `test_selected_fusion_defaults` and embedded in every final JSON.

Tests wrap every material override, assert selected path counters including
dense and expert activation counters, and include direct tests between
distinct `FunctionalDecoder` and `FusedDecoder` instances with distinct KV
caches. A functional fallback cannot satisfy the suite.

## Correctness

All thresholds remain the functional acceptance bar, PCC 0.995.

| Representative layer | Phase | Functional PCC | Final fused PCC | Delta |
| --- | --- | ---: | ---: | ---: |
| layer 0, sliding attention | prefill | 0.999163 | 0.999307 | +0.000144 |
| layer 0, sliding attention | decode | 0.999739 | 0.999716 | -0.000023 |
| layer 5, full attention | prefill | 0.998457 | 0.998004 | -0.000453 |
| layer 5, full attention | decode | 0.999860 | 0.999900 | +0.000040 |

Direct functional-vs-fused isolation covers dense-only, router-only,
expert-only, and the selected combined graph for both meaningful layer kinds.
Prefill PCC ranges from 0.999652 to 1.0 and traced-decode PCC ranges from
0.999866 to 1.0. The selected combined results are:

| Layer kind | Direct prefill PCC | Direct traced-decode PCC |
| --- | ---: | ---: |
| sliding attention | 0.999652 | 0.999866 |
| full attention | 0.999700 | 0.999902 |

Full attention passes with its natural KV cache and the shared physical-cache
view. Traced decode passes at batch 1 and 32 for both layer kinds. Mutable
stable-buffer tests prove repeat determinism; batch-2 prefill passes; logical
boundary suites cover 1/31/32/33, cache-page edges, and
1,023/1,024/1,025 without a public alignment restriction. The sustained
sliding-cache test covers 1,104 trace replays across the 1,024-token ring wrap
and reports PCC 1.0 at positions 1,023, 1,024, 1,025, and 1,103.

## Performance

Measurements use one Blackhole P300 chip, a 1x1 mesh, real weights, batch 1,
and sequence length 1,024. Prefill is warmed. Decode is TTNN-traced, warmed,
and replayed. Device totals are signpost-sliced `tt-perf-report` 1.2.9 output
from final source-hash-matched Tracy captures.

| Layer kind | Phase | Functional device | Final fused device | Reduction | Ops before/after |
| --- | --- | ---: | ---: | ---: | ---: |
| sliding attention | prefill | 1,242.489 ms | 277.751 ms | 77.6% | 557 / 456 |
| sliding attention | traced decode | 3.012 ms | 1.269 ms | 57.9% | 74 / 65 |
| full attention | prefill | 1,243.618 ms | 278.891 ms | 77.6% | 557 / 456 |
| full attention | traced decode | 3.207 ms | 1.451 ms | 54.7% | 76 / 67 |

The matching unprofiled screen uses 20 additional warmups and 200 measured
trace replays:

| Layer kind | Functional prefill/decode | Final fused prefill/decode | Reduction |
| --- | ---: | ---: | ---: |
| sliding attention | 1,243.063 / 3.052 ms | 278.476 / 1.311 ms | 77.6% / 57.1% |
| full attention | 1,244.202 / 3.230 ms | 279.607 / 1.495 ms | 77.5% / 53.7% |

The dominant packed sparse projection is now 3.54 ms per 32-token prefill
group and 222 microseconds in decode, using 48 cores and about 323/318 GB/s.
The final prefill graph has 64 sparse matmuls and one concat; decode has two
sparse matmuls. There is no Torch conversion, `from_torch`, `to_torch`, host
fallback, or reshard in the measured decoder path. Remaining type/layout
operations are required by attention heads or router top-k/scatter and the
sparse routing-mask contract; none is a removable round trip introduced here.

See `perf/README.md` for exact capture commands, raw paths, signposts, durable
CSV/text reports, and source hashes.

## Context contract

`doc/context_contract.json` remains unchanged. With the final padded/folded
weight layout, physical real-weight prefill passes at both 262,143 and 262,144
tokens for sliding and full attention:

| Length | Sliding elapsed | Full elapsed |
| ---: | ---: | ---: |
| 262,143 | 71.999 s | 157.489 s |
| 262,144 | 72.000 s | 157.482 s |

Traced decode passes at current position 262,143 with a permuted page table,
device-initialized history, cache-sentinel readback, finite output, and repeat
PCC 1.0 for both layer kinds. Internal 32-token expert grouping remains
invisible to callers; the largest non-aligned advertised-context input passes.

## Exhaustive fusion assessment

| Graph-fusing pattern | Assessment and evidence |
| --- | --- |
| Dedicated/composite operation | Dense composite `ttnn.geglu` was tested, but the selected fast GELU through `BinaryNgDeviceOperation` removed the standalone GELU and improved full traced decode. Attention already uses dedicated QKV-head, head-concat, RMSNorm, HF rotary, paged SDPA, top-k, numeric-stable softmax, and paged-cache operations. |
| Shared-LHS projections | Selected packed dense and packed expert gate/up projections. |
| Constant folding | Selected router feature/hidden-scale folding, expert routing-scale folding, dense/expert learned norm-weight folding, and the final learned scalar folded into residual add. |
| Common subexpression | Selected one unweighted RMSNorm feeding dense, router, and expert consumers; isolated PCC and combined perf passed. |
| Producer/consumer activation | Selected fast dense GELU and accurate expert GELU through `BinaryNgDeviceOperation` input activations. The matched fast expert composite candidate passed PCC but was slower and less accurate on sliding attention. |
| Add plus scalar activation | Selected `MUL_UNARY_SFPU` as an activation on final residual add; it removes the separate scalar multiply while preserving PCC and improving decode. |
| Structural concat simplification | Selected one list concat for 32 expert groups, removing 31 growing concats. |
| Sparse geometry/padding | Tested widths 1,408/1,536/1,792/2,048 and K-block widths 1/2/4/8/11. Width 1,536 and block 4 won three repeated screens: 278.65 ms prefill average versus 280.16/283.48 for block 8/11, with decode differences within 0.003 ms. |
| Larger grouped sparse execution | With adapted `per_core_M`, 64/128/256/1,024-token groups became runnable but regressed prefill to 392.30/619.98/1,072.95/3,800.76 ms. The selected 32-token group is 278.48 ms and is not a public restriction. |
| Add folded into RMSNorm | `residual_input_tensor` passed correctness but regressed the then-selected traced decode from 2.456 to 2.494 ms. |
| Multiply/reduce to matmul | Batched routing matmul passed at 0.999312/0.999722 PCC but regressed prefill from 704.606 to 721.214 ms. |
| Rotary fusion | Available fused Q/K rotary implements Llama adjacent-pair/transform-matrix rotation; Gemma uses HF rotate-half. It is not semantically substitutable. |
| Conv/mean/pool or spatial patterns | No convolution, spatial reduction, pooling, or equivalent subgraph exists. |
| Host/layout elimination | Source and final CSV audits find no host conversion/fallback/reshard. Retained layout boundaries are consumer-required as described above. |

The matched fast expert candidate used the same surrounding graph at that
comparison point. Its
sliding/full results were 279.989/1.320 and 281.256/1.497 ms versus accurate
binary activation at 278.484/1.319 and 279.685/1.496 ms; sliding PCC was also
lower (0.999209/0.999659 versus 0.999309/0.999711). It is rejected.

Candidate artifacts under `candidate_runs/` record exact environment
overrides, source/test hashes, PCC where applicable, and host timings. Final
artifacts additionally record the resolved policy. Historical candidates
intentionally retain their historical hashes; `definitive_final_v2_*` and the
canonical timing JSONs are final-source comparisons.

## Verification summary

- regular fused suite: 25 passed, 6 opt-in gates skipped;
- direct functional/fused isolation: 8 passed;
- advertised-context traced decode: 2 passed;
- physical prefill capacity: 262,143 and 262,144, both layer kinds, 4 passed;
- ring-cache stress: 1,104 trace replays with all repeat probes at PCC 1.0;
- watcher: 2 real-weight tests passed, 1,220-line final log, clean scan;
- Tracy and all four `tt-perf-report` slices passed;
- Python and documentation files pass repository pre-commit hooks;
- no build is required because changes are Python, JSON, CSV, text, and Markdown.

Exact commands, rejected candidates, review remediation, and local commit SHAs
are in `work_log.md`.
