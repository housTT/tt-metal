# GPT-OSS 120B optimized multichip decoder

Status: implementation, local validation, fully enabled watcher validation, and
independent stage rereview complete with `clean-pass`.

This stage optimizes the completed `openai/gpt-oss-120b` decoder in place for
P150, P150x2, and P150x4. The measured multi-device results are the real TP2
and TP4 decoder paths on a 1x4 Blackhole mesh. Full-model assembly, generation,
and vLLM are intentionally outside this stage.

## Final cumulative contract

| Contract | Final default |
| --- | --- |
| Public residual | Logical `[1, 1, batch, 2880]` and replicated BF16 in both phases; decode uses an L1-interleaved boundary, while prefill retains the inherited DRAM-interleaved boundary |
| Inter-layer movement | No gather, reshard, all-reduce, reduce-scatter, or all-gather between decoder layers |
| Input ownership | Decode borrows the caller's residual; the layer does not clone it to DRAM or deallocate it |
| Norms | Both decode RMSNorms use a 10-core L1 width shard; outputs remain in L1 |
| Attention | Packed rank-local QKV, paged SDPA decode, row-parallel O; BFP8 weights and KV; decode projections LoFi; prefill QKV HiFi2 and O LoFi |
| Attention CCL | Decode uses a BFP8 physical-hidden ring reduction; TP4 reduces 2944 and slices internally to logical 2880. The inherited 127-token prefill path uses BF16 fused reduce-scatter plus all-gather. |
| TP2 O projection | Decode-only 16-core DRAM-width-sharded BFP8 weight, internally padded from 2880 to 3072 and sliced after the collective |
| TP4 O projection | Decode uses the native 16-core L1 path at physical width 2944; the DRAM-sharded decode family was slower end to end |
| MoE | Router-selected top-4 active experts only; packed gate/up and row-parallel down use `ttnn.sparse_matmul` |
| Expert precision | BFP4/LoFi weights, BF16 expert collective and final layer residual |
| Sparse geometry | Gate/up: 45 cores, `in0_block_w=30`, TP2 subblock `1x2`, TP4 `1x1`; decode down: 15 cores, `1x6`; prefill down: 45 cores, `1x2` |
| Trace | Warmed decode is captured/replayed; current position, RoPE, page table, input, and cache state are refreshed |
| Context | Page size 64; 131072-token decoder-layer contract unchanged; non-aligned logical lengths are padded/masked/sliced internally |

In decode, the mid-layer post-attention residual is BFP8 because the attention
result is the output buffer of the residual add. The final expert reduction is
BF16, so the decode output and next-layer boundary return to BF16 L1; the next
decode layer starts with its 10-core sharded norm directly from that tensor.
Prefill remains BF16/DRAM across its attention residual, norms, expert
collective, final residual, and next-layer boundary.

The default policy name is
`p150_1d_tp_replicated_residual_mixed_ccl_lofi_sparse_decode45x15_prefill45x45_tp2_subblock2_dram_output`.
Candidate selection exists only behind the test-only
`GPT_OSS_120B_MULTICHIP_CANDIDATE` variable; production construction has no
runtime fallback.

## Final correctness and latency

These are the authoritative final-default medians from 100 traced replays per
sample and five samples. Prefill uses logical sequence length 127. PCC is
against the accepted P150 producer artifact at the same checkpoint revision.

| Target | Layer kind | Prefill before -> after (ms) | Prefill change | Prefill PCC | Traced decode before -> after (ms) | Decode change | Decode PCC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 | sliding | 36.594683 -> 36.594683 | control | 1.0 | 0.530660340 -> 0.530660340 | control | 1.0 |
| P150 | full | 31.935862 -> 31.935862 | control | 1.0 | 0.528483690 -> 0.528483690 | control | 1.0 |
| P150x2 | sliding | 47.748477 -> 21.988971 | -53.95% | 0.993102916 | 0.752601540 -> 0.428405910 | -43.08% | 0.999176688 |
| P150x2 | full | 47.438676 -> 21.679801 | -54.30% | 0.992071333 | 0.752338440 -> 0.425538640 | -43.44% | 0.997517369 |
| P150x4 | sliding | 25.895578 -> 14.374113 | -44.49% | 0.992664780 | 0.652817770 -> 0.396413630 | -39.28% | 0.998653710 |
| P150x4 | full | 25.544376 -> 14.041818 | -45.03% | 0.992419902 | 0.653009110 -> 0.396353620 | -39.30% | 0.997784401 |

The 1000-replay, three-sample stress rerun reproduced 0.428461/0.425393 ms for
TP2 sliding/full and 0.396258/0.396140 ms for TP4 sliding/full. All four cases passed cache
reconstruction, two trace-input refreshes, exact rank replication, and both
meaningful layer kinds. Batch-2 high-position coverage also passed all four.
Raw values are in `performance_summary.csv` and the final/stress logs.

## Operation topology and coherent-family decisions

`operation_topology_audit.md` records the starting sequence and was written
before implementation changes. Material families were measured as follows.

| Family | Evidence | Decision |
| --- | --- | --- |
| Residual/norm layout | Decode's old per-layer DRAM clone and interleaved norms versus borrowed L1 residual and two 10-core sharded norms | Keep the borrowed decode L1 boundary. Final decode norm rows are 5.5--6.4 us. Prefill retains its inherited BF16 DRAM boundary; neither phase inserts an inter-layer collective. |
| Replicated versus lower-movement residual | Physical replicated 0.112159 ms; async RS+AG 0.117813 ms; persistent async 0.113429 ms; reduce-scattered residual -> distributed norm -> fused AG+QKV 0.135073 ms, or 0.131030 ms persistent | Keep physical replicated. The sharded family was measured through the next norm/QKV without an immediate restore and remained 16.8% slower. |
| Attention collective placement | Decode logical/physical ring variants, async RS+AG, delayed gather, fused output MMRS, and persistent buffers | Decode physical-width BFP8 reduction wins the complete layer. Integrated TP4 fused MMRS regressed to 0.414726 ms from about 0.3966 ms. Prefill retains its inherited BF16 fused RS+AG path. |
| Packed versus separate QKV | Separate TP4 sliding 24.916073 ms prefill, 0.500701600 ms decode, PCC 0.981589952 | Keep packed QKV. |
| Packed versus separate gate/up | Separate TP4 sliding 23.720292 ms prefill, 0.506294350 ms decode | Keep packed gate/up. Both retain top-4 indexed sparse execution. |
| Activation/CCL dtype | Decode expert BFP8 0.464593 ms; decode attention BFP4 trace-refresh PCC 0.885471; expert/global BFP4 prefill PCC 0.929322 | Keep BFP8 for the custom decode attention CCL and BF16 for the expert CCL in both phases. The CCL-dtype sweep did not alter inherited prefill attention: at the measured logical length 127 it selects BF16, as the final profiler rows prove. |
| Persistent CCL buffers | Async 0.117813 -> 0.113429 ms; sharded fused 0.135073 -> 0.131030 ms | Persistence helps rejected families, but neither beats the physical-replicated path. Its synchronous API exposes no reusable output-buffer argument. |
| Expert collective padding | Logical width 0.054117 ms versus runtime-padded 0.096323 ms | Keep logical 2880 expert reduction. |
| Prefill expert geometry | Down 15 cores: TP2 24.305916/23.956320 ms, TP4 15.470742/15.594027; down 30: 27.096822/26.747726 and 14.700000/14.436056; down 45: 22.139411/21.656049 and 14.346312/14.010952 | Keep 45-core down for prefill on both meshes; decode remains 15-core. Every comparison ran both layer kinds through the complete layer. |
| Router config/placement | Explicit 4x4 config was 21.911335/21.542912 ms TP2 and 14.216203/14.010310 TP4 but reduced PCC in three cases; adding DRAM-to-L1 was neutral/slower. L1-only was mixed: 21.956250/21.700090 and 14.455629/14.062288. | Keep automatic config and DRAM input. The sub-1.1% timing variation was not coherent enough to trade away PCC; explicit config and L1 advice are closed with full-layer evidence. |

TP2 fused output projection was not rejected on the first error: rank, weight
layout, program config, 3072 padding, and native 2880 were tried. Both adapted
forms hung on this Blackhole runtime; triage and bounded reset/mesh-smoke
recovery were performed. TP4 is the passing minimal control.

## Precision, fidelity, and projection choices

| Candidate | Result | Decision |
| --- | --- | --- |
| Attention BFP8/HiFi2 decode projections | Correct, slower than LoFi | Reject for decode; prefill packed QKV remains HiFi2 |
| Attention BFP8/HiFi4 decode projections | TP4 decode 0.479017 ms | Reject |
| Attention BFP8/LoFi | Decode projections and prefill O pass all four TP2/TP4 cases; prefill packed QKV remains HiFi2 | Keep the phase-specific fidelity policy |
| Attention BFP4/LoFi | Real-weight TP4 prefill PCC 0.829483 | Reject on model-visible correctness |
| Experts BFP8/LoFi | TP4 prefill 19.239024 ms, decode 0.414414 ms, decode PCC 0.996777 | Reject on latency |
| Experts BF16 | TP4 prefill 27.932937 ms, decode 0.468665 ms, decode PCC 0.996784 | Reject on latency |
| Experts BFP4/LoFi | Runtime rows prove gate/up and down use BFP4 | Keep |
| Router BFP8 | TP4 decode 0.396340 ms, refresh minimum 0.965473 | No material gain |
| Router BFP4 | TP4 decode 0.396170 ms, prefill PCC 0.985043 | Reject |

BFP4 attention was tested on the final packed topology with real checkpoint
weights. BFP4 experts were crossed with the geometry sweep. KV remains BFP8;
its paged fill/update and replay contract remains accepted.

## Dominant matmul search

Final rows are in
`artifacts/20260829_profiler_after_review/*/decode_perf_report.csv`. Times below are
representative sliding-layer device rows.

| Role | Final shape and policy | Geometry | TP2 / TP4 row (us) | Search conclusion |
| --- | --- | --- | ---: | --- |
| QKV | `32x2880x(2560/1280)`, BF16 x BFP8 -> BF16, LoFi | 10 cores, block 9, subblock `1x8` / `1x4`, L1 width-sharded input | 28.080 / 19.036 | Packed wins; DRAM-sharded and three separate projections lose whole-layer latency. |
| O | TP2 `32x2048x3072`, TP4 `32x1024x2944`, BF16 x BFP8, LoFi | TP2 16-core DRAM-width-sharded; TP4 native 16-core, block 2, subblock `1x6` | 18.292 / 14.454 | TP2 core sweep selected 16; TP4 tried 16/8/4/2 and retained native. |
| Router | `32x2880x128`, BFP8 x BF16 -> BFP8, HiFi2 | 10 cores, block 9, subblock `1x1` | 7.151 / 7.153 | Lower router dtypes did not win. |
| Gate/up | active `4/128 x 32 x 2880 x (2880/1440)`, BFP8 x BFP4 -> BF16, LoFi | 45 cores, block 30; TP2 `1x2`, TP4 `1x1` | 72.563 / 40.158 | Tried 9/12/15/30/45 cores and legal subblocks. TP4 `per_core_N=1` makes width 2 illegal. |
| Down | active `4/128 x 32 x (1440/736) x 2880`, BF16 x BFP4 -> BF16, LoFi | 15 cores, TP2/TP4 block 9/23, subblock `1x6` | 52.758 / 31.808 | Tried 15/18/30/45/48-core families; 15 with the largest legal subblock wins. |

The first 30-core gate/up and 48-core down attempts exposed validation limits;
shape, core orientation, divisibility, and subblocks were adapted before the
sweep continued. TP4's final gate/up `1x1` is called out because
`tt-perf-report` advises wider: its local output is one tile per core, so width
2 exceeds `per_core_N`. TP2 has two output tiles per core and uses `1x2`.

The routed sparse op has no DRAM-sharded program factory. Source inspection
shows `ttnn.sparse_matmul` selects the interleaved
`SparseMatmulMultiCoreReuseMcast1DProgramFactory`; the exhaustive supported
core/block/subblock search therefore keeps sparse intermediates in L1.

## Profiler accounting

Profiler captures use the final default and one traced replay. Tracy inflates
wall latency, so same-capture accounting and non-profiled headline are both
shown.

| Target/layer | Device rows (ms) | Profiled wall (ms) | Same-capture overhead/gap (ms) | Non-profiled final (ms) |
| --- | ---: | ---: | ---: | ---: |
| TP2 sliding | 0.389214 | 0.499321 | 0.110108 | 0.428406 |
| TP2 full | 0.387509 | 0.510702 | 0.123193 | 0.425539 |
| TP4 sliding | 0.360311 | 0.497397 | 0.137086 | 0.396414 |
| TP4 full | 0.362191 | 0.498419 | 0.136228 | 0.396354 |

The multi-device CSV merger reports 76--108 ms of op-to-op gaps because it
sorts host timestamps from multiple ranks into one stream. That is not layer
latency and is excluded. Summed device rows and signposted trace wall are used.
Without Tracy, wall minus device rows is about 34--39 us, dominated by replay
synchronization and measurement fences; the decode has no CPU fallback.

At decode position 127, a conservative mandatory per-device DRAM-read estimate
is 43.319 MB on TP2 and 21.992 MB on TP4. It counts packed QKV, O, router, four
active gate/up experts, four active down experts, and local K/V reads at stored
dtypes. At the profiler's 512 GB/s per-device model, lower bounds are 0.0846 ms
and 0.0430 ms. Device time is 4.6x / 8.4x these bounds because it also contains
SDPA, CCL, layouts, norms, routing, top-k, elementwise work, and small-M
underutilization. `tt-perf-report` reports about 99 GB/s overall on TP2.

## Validation and commands

Hardware commands were serialized through `scripts/run_safe_pytest.sh`. The
real-weight matrix used this form for TP2/TP4 and sliding/full:

```bash
env GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR="$PWD/models/autoports/openai_gpt_oss_120b/doc/optimized_multichip_decoder/artifacts/20260829_baseline" \
  GPT_OSS_120B_MULTICHIP_RUN_ID=20260829_optimized_multichip_baseline \
  GPT_OSS_120B_MULTICHIP_TRACE_REPEATS=100 \
  GPT_OSS_120B_MULTICHIP_TRACE_SAMPLES=5 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_real_weight_multichip_against_baseline_artifact \
  -q -s
```

Stress used repeats/samples `1000`/`3`; batch-2 used
`test_real_weight_multichip_batch2_high_position_against_baseline_artifact`.
Profiler runs used `scripts/run_safe_pytest.sh --profile` with both counts 1,
then:

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --active-experts 4 --csv decode_perf_report.csv --no-summary --no-color raw_ops.csv
```

Watcher and profiler were separate. Host-only policy/plan/fallback checks use:

```bash
python_env/bin/pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py
```

The fully enabled watcher initially exposed two independent fabric issues. The
normal O3 watcher image exceeded ACTIVE_ETH's 26-KiB kernel-config region, so
the accepted watcher command uses the repository-supported
`TT_METAL_FABRIC_OPT_LEVEL=Os` override; no watcher feature is disabled. The
model then exposed sticky write-capable NoC packet tags during router teardown.
After `$autofix` added a packet-tag clear following the write and atomic
barriers, TP2/TP4 sliding/full passed in one process, every mesh reopen reported
`disabled features: None`, and process/device teardown was clean. The matrix,
negative controls, and exact diagnosis are in
`autofix/AUTOFIX_watcher_active_eth.md`.

The retained router change is C++. Its device kernel was JIT-compiled and run
on the four-device hardware by the focused smoke, isolated TP2 case, and final
four-case watcher matrix. The final-source wrapper configure check passed with
`--build-dir build_copilot_optimized_multichip_final --configure-only`. The
prescribed full CI-wrapper compile remains unverified because Garage
credentials are absent and the wrapper reported a cold-cache build; the build
disposition and configure log are recorded in the work log.

## Artifacts and limitations

- `artifacts/20260829_baseline/`: local immutable P150 producer tensors. The
  `.pt` inputs exceed the repository's 500-KB file policy and are intentionally
  not commit payloads; the producer commands and identifying revision are
  recorded so they can be regenerated.
- `artifacts/20260829_candidates/`: full-layer candidate logs.
- `artifacts/20260829_topology/`: CCL/residual/fusion probes and retries.
- `artifacts/20260829_final/`: authoritative 100x5 default and batch-2 logs.
- `artifacts/20260829_stress/`: 1000x3 replay logs.
- `artifacts/20260829_profiler_after_review/`: final-default gzip-compressed raw profiler CSVs,
  plain `tt-perf-report` tables/CSVs, provenance, and compressed command logs.
- `artifacts/20260829_watcher/`: strict watcher attempts and accepted run.
- `autofix/AUTOFIX_prefill_replication.md`: refuted prefill hypothesis and
  device-state recovery.
- `autofix/AUTOFIX_watcher_active_eth.md`: admission and packet-tag teardown
  diagnosis, negative controls, retained fix, and fully enabled final matrix.

TP2 fused matmul-reduce-scatter hangs for both adapted shapes on this runtime.
The lowest observed refresh PCC is 0.964227 for TP2 sliding, and batch-2
high-position full-attention decode reaches 0.962896/0.965031 on TP2/TP4; all
are above their explicit acceptance floors but leave limited precision margin.
P150/P150x2 full 36-layer residency remains a later full-model capacity issue.
Full-model/generator/vLLM work was not started. The TP2 decode-only DRAM output
copy adds 6,690,816 bytes/device/layer (0.224327 GiB for 36 layers) but does not
change KV allocation or the 131072-token decoder context.
