# Gemma 4 26B A4B multichip decoder

`tt/multichip_decoder.py` is the TP1/TP2/TP4 decoder-layer baseline for
`google/gemma-4-26B-A4B-it`. It subclasses and reuses the optimized decoder,
adds rank-local tensor loading, paged-KV ownership, and profile-specific
collectives, and deliberately stops before full-model or vLLM integration.

## Supported profiles

| Target | Accepted P300C QB2 proxy | Fabric and collective | Local dense/expert widths |
| --- | --- | --- | --- |
| P150 | one device | no CCL | 2112 / 704 |
| P150x2 | 1x2 compute submesh of a 2x2 parent | `FABRIC_2D` parent; Linear, one link | 1056 / 352 |
| P150x4 | 1x4 mesh | `FABRIC_1D_RING`; Ring, two links | 544 / 192 |

TP4 load-time pads dense 2112 to 2176 and expert 704 to 768. QKV and gate/up
are column-parallel; O and down projections are row-parallel with a BF16
hidden all-reduce. Norms, router, residual, page table, and current positions
are replicated. Sliding KV heads are sharded 8/4/2. Full attention has only two
KV heads, so TP4 duplicates each head on the pair of ranks whose Q heads use
it. Sparse MoE remains gate-selected top-8; it is not converted to dense expert
execution.

Decode uses three rotating persistent asynchronous all-reduce resources on
TP2/TP4. TP2's odd 11-tile local expert width uses `per_core_n=1`; TP1/TP4 use
2. TP4's selected decode DRAM roles are O, packed dense gate/up, and dense
down, each with one reader per DRAM bank. Sliding O reuses one sharded BF16
weight with `in0_block_w=2` for logical B1 and `in0_block_w=4` for B32; this
dual-program policy is required to clear both HF and optimized-baseline gates.
Two- and three-reader candidates are
unsupported on a multi-device mesh by the inherited primitive and now fail
early instead of reaching its `MeshDevice` fatal. Packed expert gate/up is a
separate retained decode tensor for every
layer; the selected runtime policy enables it only on TP4. TP1 disables it by
a hard capacity limit, and TP2 disables it because the 8,088,453,120 B retained
copy would violate the conservative 262,144-token full-stack contract.
Report-driven QKV DRAM, wider-QKV-block, and
router-L1 candidates were measured and rejected.

## Correctness and context

All representative real-weight prefill/decode comparisons clear PCC 0.995:

| Profile | sliding prefill / decode | full prefill / decode |
| --- | --- | --- |
| TP1 | 0.998077 / 0.999262 | 0.997276 / 0.998433 |
| TP2, final no-packed capacity policy | 0.997943 / 0.999121 | 0.998801 / 0.999741 |
| TP4 | 0.997090 / 0.999319 | 0.998620 / 0.999875 |

TP4 also passes direct HF-oracle comparisons at 0.998460/0.999532 for sliding
prefill/decode and 0.998500/0.999705 for full, logical S=33 prefill followed by
bit-exact repeated decode trace replay, paged-cache replica checks, the
262,143-token real-weight prefill probe, and traced decode at current position
262,143. TP2 and TP4 retain the 262,144-token contract. TP1 is restricted to
50,624 by the 32 GiB full-stack bound; 50,623 is its tighter nonaligned case,
and 50,625 fails the byte contract before device allocation. See
`../context_contract.json` and `artifacts/` for the machine evidence.
Real-weight sliding/full probes pass at both 50,623 and 50,624; the earlier
passing 53,343/53,344 probes are retained only as superseded shape evidence.

The validated decoder policy keeps sliding-attention projections in BF16 and
uses BFP8 for full-attention, dense, and expert projections, except that TP1
full-attention expert gate/up weights use BFP4. Activations, CCL, and KV cache
are BF16, while the router is FP32. TP1, TP2, and TP4 sliding layers select the
optimized R22 residual path, all four graph folds, and row-major routing. TP1
and TP2 full layers use the same policy. The TP4 full layer is a measured B32
correctness exception: R0 residual, raw router and FFN norm, folded expert
scale, fused final scalar, packed expert decode, and non-row-major routing.
The direct per-layer canonical gate remains above 0.995 for every profile.
The final no-packed TP2 policy also passes direct HF at
0.999024/0.999566 (sliding) and 0.998534/0.999711 (full), while identical-input
optimized/TP2 decode boundaries pass the stricter 0.99 stacked gate at
0.991223/0.995297. Only the deliberately divergent final routed chain uses its
separate 0.98 stress threshold.

The target 1x4 stack is also validated across the TP4-only policy transition:
layer 0 sliding R22/folded/row-major output feeds layer 5 full R0/raw-router/
non-row-major input with one shared three-slot persistent CCL bundle. Same-input
prefill PCC is 0.997792/0.998964 and decode PCC is 0.999117/0.995288; the
chained prefill layer-5 PCC is 0.991588 and the deliberately divergent routed
decode chain is 0.986078 at its 0.98 stress gate. Twenty trace replays are bit
exact across ranks, with local cache geometry, full-KV rank-pair duplication,
replicated page tables/current positions, and the R22-to-R0 public boundary all
checked explicitly. See `AUTOFIX_TP4_STACK.md`.

The stage-review repair accounts for actual prefill lifetimes. Chunked SDPA
does not bound the block: `_attention_prefill` first creates full-length QKV,
the attention helpers retain every output chunk through `ttnn.concat` (where
both accumulated chunks and the new concat output coexist), and the inherited
block retains residual/branch tensors through attention and FFN. At
262,144 tokens the conservative BF16 live-tensor peaks are 22,817,013,760 B on
TP1, 15,651,045,376 B on TP2, and 17,001,611,264 B on TP4. No deallocation
credit is assumed.

Full-stack placement continues to mandate BFP8_B embedding and LM-head storage
downstream. Physical 1,088-byte tile accounting gives 1,568,669,696 B before
TP sharding; final norm is 180,224 B/device. The reserve is now the configured
64 MiB trace region plus the conservative live peak plus the prior
profile-specific allocator slack, not a fixed 1 GiB chunk estimate.

| Profile/context | packed expert copy | decoder + KV + CCL | operating reserve | full-stack total | 32 GiB headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1 / 50,624 contiguous maximum | disabled | 27,284,654,080 B | 5,505,196,544 B | 34,358,700,544 B | 1,037,824 B |
| TP2 / 262,144 | disabled | 16,486,360,064 B | 16,300,113,920 B | 33,570,989,056 B | 788,749,312 B |
| TP4 / 262,144 | 4,411,883,520 B | 15,251,053,568 B | 17,776,508,928 B | 33,419,910,144 B | 939,828,224 B |

The TP1 total uses 50,623, the worst nonaligned length below the inclusive
50,624 limit; the aligned limit itself leaves 389,822,464 B. At advertised
262,144 TP1 would require 56,398,546,944 B, exceeding 32 GiB by
22,038,808,576 B under the correctness-proven precision policy. Broader BFP4
expert-down candidates failed PCC, so the implementation enforces the honest
per-profile context limit. The terminal contract is recorded here but remains
unimplemented and requires downstream logit accuracy evidence.

Real-weight BFP4 attention trials were completed independently for QKV and O
under cache-consuming TP4 traced decode. QKV scored 0.984498 sliding and
0.978292 full; O scored 0.991448 sliding and 0.998744 full. The first three
results fail the 0.995 decoder gate. The passing full-only O result was only
0.39% nominally faster over 30 replays and requires another retained decode
tensor, so the selected BF16-sliding/BFP8-full attention policy is unchanged.
Runtime assertions prove the BFP4 tensors reached the intended matmuls; see
`AUTOFIX_BFP4_ATTENTION.md` and `artifacts/bfp4_attention_summary.json`.

## Measured decoder-layer performance

The verified unprofiled batch-32 trace remains the throughput proxy. Mandatory
TP1/TP2 profiles were additionally rerun on the frozen source using the same
optimized one-chip S=33 baseline, five warmups, and 30 trace replays:

| Profile | Layer | optimized 1-chip | selected profile | speedup | TP efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| TP1 | sliding | 0.7657 ms | 0.9299 ms | 0.823x | 82.34% |
| TP1 | full | 0.8121 ms | 1.0486 ms | 0.774x | 77.44% |
| TP2 | sliding | 0.7657 ms | 0.8166 ms | 0.938x | 46.88% |
| TP2 | full | 0.8121 ms | 0.9140 ms | 0.888x | 44.42% |

These are honest layer-level scaling results: neither small-batch TP1 nor TP2
beats the optimized baseline. Replay and residual replicas were bit-exact.
The compact profiler accounting below is the final selected TP4 capture.

| Decode regime | Layer | optimized 1-chip | TP4 | speedup | TP efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| B32, position 32, zero sliding history | sliding | 12.1992 ms | 8.8175 ms | 1.384x | 34.59% |
| B32, S=32 prefill, position 32 | full | 12.1998 ms | 9.1740 ms | 1.330x | 33.25% |
| B1, S=33, verified | sliding | 0.7657 ms | 0.6521 ms | 1.174x | 29.35% |
| B1, S=33, verified | full | 0.8121 ms | 0.9524 ms | 0.853x | 21.32% |

The final residual-topology AutoFix tested a complete K-fractured sliding
layer, not only a synthetic collective. Its dynamic indexed top-8 expert plus
next-QKV micro-chain was correct, watcher-clean, and 1.741x faster, but the
complete real-weight paged decoder required six reduce-scatters, six norm-stat
all-gathers, and one router all-reduce. It remained correct (PCC 0.999805 and
20 bit-exact stress replays) but the warning-clean rerun measured median/p95
latencies of 0.844279/0.847750 ms versus 0.652086/0.657987 ms for the selected
replicated-residual layer: 29.47% slower by median. Both paths used five
warmups and 30 individually timed blocking replays, with non-overlapping trace
lifetimes and release-before-readback. The temporary product candidate was
therefore removed. A restored-source S=33 run measured
0.651610 ms with 30 bit-exact replays. Human-readable and CSV profiler evidence
is under `artifacts/fractured_sparse/tt_perf_report/`; the full decision is in
`AUTOFIX_FRACTURED_RESIDUAL.md`.

The reduced B1 profiler run is a separate S=1024 accounting regime:

| Layer | optimized prefill | TP4 prefill | speedup / TP efficiency | TP4 decode e2e | per-device op sum | critical remainder | critical all-reduce share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 96.3659 ms | 80.0302 ms | 1.204x / 30.10% | 760.727 us | 629.075-633.008 us | 127.719 us / 16.79% | 56.238 us; 8.88% of ops / 7.39% e2e |
| full | 107.6368 ms | 88.8432 ms | 1.212x / 30.29% | 1077.322 us | 942.862-947.451 us | 129.871 us / 12.05% | 57.739 us; 6.09% of ops / 5.36% e2e |

Per-device sums come from a host-only `--no-merge-devices` pass over the local
raw `ops.csv` captures and are divided by three trace iterations (86 sliding
and 87 full ops per replay/device). Those large intermediates are ignored by
the repository; `perf_summary.json` retains their hashes and the compact
ledger retains their derived values. The
remainder is e2e minus the slowest device's summed op durations; it is not
assigned to one cause. Each device runs three all-reduces per decode. The
ordinary merged reports instead retain the maximum-duration device per op and
average collective durations, so their `Device` column is not a per-device
ledger. Readback gaps also invalidate their whole-layer `Total %` columns,
although device times and per-row DRAM/compute roofline models remain usable.

`perf_summary.json` records the full precision, representative roofline rows,
input hashes, cross-checkout baseline provenance, and exclusions. The compact
CSV derivation is `tt_perf_report/selected_accounting.csv`. This profiler
capture is verified at selected multichip source SHA256 `34225bd3...` and test
SHA256 `8338455c...`. The later capacity repair changed the final source SHA256
to `7279e13a...` by adding the TP1 prefill guard and selecting packed-expert
retention only for TP4. Neither change alters the profiled TP4 hot path. The
frozen final source passed `py_compile`; an eight-case TP1/TP2/TP4 watcher run
passed on the same hot path, and the four TP1/TP2 timings above were rerun on
the exact final source.

## Evidence map

- `mesh_plan.md`: tensor mappings, topology choice, padding, and rejected alternatives.
- `capacity_projection.json`: profile-by-profile full-stack and retained-copy
  capacity ledger.
- `perf_summary.json`: exact warmed timings, PCC against the optimized baseline, and profiler limitations.
- `AUTODEBUG*.md`, `AUTOFIX*.md`, `AUTOTRIAGE.md`: repair history and ruled-out hypotheses.
- `work_log.md`: commands and gate chronology.
- `artifacts/`: JUnit, correctness, context, trace, and candidate timing evidence.

`artifact_manifest.json` is the authority for selected versus superseded
evidence. Its final JUnits include `direct_and_stacked_selected_final.xml`,
`hf_oracle_selected_final.xml`, `batch32_selected_final.xml`,
`trace_selected_final.xml`, `context_cache_selected_final.xml`,
`final_profiles_after_capacity.xml`, `warmed_required_profiles_final.xml`,
`p150_prefill_capacity_{50623,50624}.xml`,
`watcher_final_capacity_policy.xml`, and the two
`profiler_{sliding,full}_b1_final_selected.xml` captures. The final review
repairs add `stacked_tp4_reference_capture.xml`,
`stacked_tp4_mixed_trace.xml`, `bfp4_attention_selected.xml`, and the passing
fractured-candidate decision gates recorded by the manifest. The selected machine
data include `pcc_tp{1,2,4}_layer{0,5}.json`,
`trace_{sliding,full}_attention_batch1.json`,
`trace_tp{1,2}_{sliding,full}_attention_batch1.json`,
`multichip_batch32_layer{0,5}.json`, `stacked_tp2_mixed_trace.json`,
`stacked_tp4_mixed_trace.json`, `bfp4_attention_summary.json`, and the Gate-D
complete-layer rejection plus restored-incumbent trace records.
The B32 artifacts use a 256-token cache per user and decode position 32; they
are throughput/PCC gates, not S=1024 measurements.

This stage validates two-layer decoder-stack boundaries but does not assemble
the complete 30-layer model, embedding, LM head, generator, or serving path.
Those belong to later full-model and vLLM stages; the full-model stage must
honor and accuracy-gate the BFP8_B terminal placement above.
