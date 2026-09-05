# Optimized decoder work log

## Scope and frozen state

- Model: `google/gemma-4-26B-A4B-it`
- Starting fused-decoder checkout: `6e67efeb6251e655cb27c39153db5af1da90d68b`
- Hardware: one Blackhole P300C, 1x1 mesh, device 0
- Decoder SHA-256: `feebc8cb2f20ad9ba81c7d0f50f8323694d6d91ebb31cb9072e18b0f6b0a9c45`
- Test SHA-256: `01ed0de36451891c6c41968baeeae34f1cb77acf4e3c6db0c322506fb83bd349`
- Scope: `tt/optimized_decoder.py`, its tests, and optimized-decoder/context
  documentation only. No multichip, full-model, or vLLM work was started.

Hardware commands were serialized. Watcher and profiler were never enabled in
the same run. The stage is Python-only, so `AGENTS.md` does not require a C++
build.

## Initial measured topology

| Region | Existing ops | Opportunity audited | Final state |
| --- | --- | --- | --- |
| attention | packed QKV, head split, paged fill/update, SDPA, O | precision/fidelity, residual/attention grids, blocks/subblocks, readers, cache dtype | B1 R22 QKV/full-O plus sliding reader-1 DRAM O; B32 G8; composite SDPA; BF16 cache |
| dense MLP | same-input gate/up, GeGLU, down | packed/separate, BFP4/BFP8, LoFi/HiFi2, DRAM/L1 sharding | reader-1 DRAM-sharded packed BFP8/LoFi gate/up and down; fused GELU multiply |
| experts | same-input gate/up, down, routing/reduction | active sparse execution, packing, BFP4/BFP8, block/N/subblocks, L1 placement | packed active-expert decode; separate chunked active-expert prefill |
| router | FP32 projection, top-k, softmax/scatter | fold, dtype/fidelity, input placement | projection folded; FP32 retained; final L1-input advice loses |
| residual/norm | interleaved adds/norms | coherent width-sharded family | R22 chain selected over R11 and nonresidual controls |
| movement | head/cache and expert boundaries | redundant copies, reshard loops, host fallback | unused copies removed; required/cache/watcher boundaries remain |

The operation audit was performed before local op tuning. Candidate artifacts
are immutable JSON/XML pairs under `candidate_runs/`; the exact resolved policy,
source/test hashes, command label, hardware, PCC, and timings are stamped into
each JSON result.

## Selected runtime configuration

The B1 residual stream is BF16 L1 width-sharded across G22 (11x2) on entry,
through both residual adds, all decoder norms, attention QKV/O boundaries,
dense MLP, and final residual. It exits once to DRAM-interleaved to preserve the
public decoder contract. The layer norms use
`LayerNormShardedMultiCoreProgramConfig`, 11x2, `block_h=1`, and the R22
geometry's legal `block_w/subblock_w`. The profiler shows 6.036 us for the
input norm and 6.814/6.584/6.838 us for representative subsequent sharded
norms; residual adds output to the same R22 memory config.

The only deliberate mid-chain crossings are:

1. R22 QKV output to L1-interleaved before the head creator, selecting the
   watcher-safe interleaved factory.
2. Required head/cache layout forms around paged update/SDPA.
3. R22 MoE input to L1-interleaved for `sparse_matmul`, and its reduced output
   back to R22 for the following norm/add.

There is no Torch conversion, device/host readback, fallback, or immediate
activation-layout restore loop inside the measured runtime. Five device-side
tile/row-major conversions remain around top-k/scatter/sparse routing metadata;
the final profile records 12.683 us of device work plus 7.267 us of associated
gaps in sliding decode. They are tracked separately from host fallback.

Dominant per-role configuration:

| Role | Final dtype/fidelity | Grid/source and geometry | Memory | Final decode row |
| --- | --- | --- | --- | ---: |
| sliding QKV | BF16 x BF16, HiFi2 | G22, K shard 4 tiles, block2, M1/N12, subblock1x3 | R22 L1 width-sharded, weight DRAM interleaved | 123.502 us |
| sliding O | BF16 x BF16, HiFi2 | G8 input, block16, M1/N11, one reader/bank | L1 width-sharded, weight DRAM-sharded | 52 us |
| full QKV | BF16 x BFP8, HiFi2 | G22, K shard 4 tiles, block2, M1/N15, subblock1x3 | R22 L1 width-sharded, weight DRAM interleaved | 86.121 us |
| full O | BF16 x BFP8, HiFi2 | G22, K shard 12 tiles, block6, M1/N4, subblock1x4 | L1 width-sharded, weight DRAM interleaved | 68.543 us |
| dense packed gate/up | BF16 x BFP8, LoFi | G8 input, block4, M1/N6, one reader/bank | R22 L1 output, weight DRAM-sharded | 31 us |
| dense down | BF16 x BFP8, LoFi | G8 input, block3, M1/N4, one reader/bank | R22 L1 output, weight DRAM-sharded | 19/20 us |
| router | FP32 x FP32, HiFi4 | four-core matmul, block2 | DRAM-interleaved input/output | 37.412 us |
| B1 expert gate/up | BF16 x BFP4, LoFi | active=8/128, 48 cores, block22, M1/N1 | L1 input/output, DRAM weights | 84.907 us |
| expert down | BF16 x BFP8, LoFi | active=8/128, 88 cores, block11, M1/N1 | sparse L1 input/output, DRAM weights | 51.653 us |

The 1x1 output-subblock rows are op/shape constrained and were not accepted
from defaults alone: expert block/subblock/N families and attention G8/G22/G32
families were swept; padded DRAM-reader alternatives were measured. The router
L1 suggestion was also tried after the final report and regressed the complete
layer.

Sparse routing passes exact `nnz=8`, matching top-k and validated routing
construction. Down uses `is_input_a_sparse=True`; scores are multiplied into
expert outputs before reduction. Decode and prefill never execute all 128
experts densely.

## Candidate ledger

### Residual, packing, and graph topology

- R11 and R22 were measured as coherent whole-layer layout families. R22 won.
- Fresh legal nonresidual controls were 1.106792 ms sliding and 1.142292 ms
  full after the watcher repair; R22 remained faster.
- Four graph folds were isolated with 0000/0001/0010/0100/1000 and cumulative
  1111 correctness/perf runs. Router projection, shared FFN norm, expert score,
  and final scalar folds are all selected.
- Packed QKV is retained. Decode expert gate/up packing beat the legal separate
  path after activation/split overhead. Prefill expert packing measured
  107.967/121.280 ms versus 96.821/108.434 ms separate and was rejected.
- Dense packing is retained in the final R22 B1 path: true reader-1 dispatch
  measures 0.834892/0.858662 ms versus separate reader-1 at
  0.838778/0.861295 ms. Prefill and B32 retain separate originals.

### Precision/fidelity and geometry

- Precision was isolated independently for QKV, O, dense gate/up, dense down,
  expert gate/up, expert down, and cache. Real checkpoint evidence, not random
  PCC, made rejection decisions.
- Dense BFP4 gate/up and down failed real-weight gates at 0.988444 and 0.991786.
- Expert BFP4 gate/up passed in the mixed policy; block11/block22/N2 measured
  about 0.89328/0.88434/0.88889 ms sliding, selecting block22 N1. BFP4 expert
  down passed but lost to BFP8 down in the complete mixed policy. All-expert
  BFP4 failed the stronger trace/full gates.
- Sliding BF16/LoFi failed direct-fused and boundary acceptance. BF16/HiFi2
  passed. Full residual LoFi failed HF decode at PCC 0.985942; HiFi2 passed.
- QKV G22 block4/block2 passed at 0.852665/0.906593 and
  0.850448/0.903690 ms, selecting block2. G32 block3 failed full PCC 0.980698.
- Full O G22 block12/block6/G32 block8 measured 0.881815/0.876916/0.896347 ms,
  selecting G22 block6.
- The B1 G22+BFP4 policy does not propagate to batch 32. G22+BFP4 and
  G22+BFP8 had aggregate PCC 0.988527/0.991454 but severe user-15 minima
  0.771948/0.782364. G8+BFP4 had minimum-user PCC 0.989871. G8+BFP8 passed at
  0.995802 and is the selected B32 policy.
- The initial BFP8 bulk-fill/structural evidence was insufficient. The review
  remediation below adds cache-consuming traced numerical gates and rejects
  BFP8; BF16 is the only accepted cache policy.

### DRAM sharding, readers, placement, and advice

- The early nonresidual QKV/O 1/2/3-reader candidates were made legal with
  inert padding and tested with matching dtype/fidelity; the multi-reader
  candidates lost in that historical topology.
- Early nonresidual packed-dense 2/3-reader candidates measured 1.225/1.272 ms
  and dense-down 2/3-reader candidates 1.202/1.234 ms against the 1.142 ms
  legal control. Those runs did not settle the final R22 question. The genuine
  final R22 reader matrix and selected DRAM-sharded copies are recorded in the
  authoritative v5/v6 section below.
- Sparse expert input/output L1 placement is selected. Prefill expert L1 input
  lost and was rejected. Required sparse scores and weight tensors remain DRAM.
- The pre-review report advised moving the FP32 router input to L1. The adapted
  candidate completed its runtime/performance gate but regressed B1 sliding/full from
  0.849307/0.872289 to 0.879544/0.902738 ms and B32 from
  12.840555/12.532409 to 12.862084/12.563001 ms; it was reverted. This
  placement-only experiment did not receive a separate numerical PCC claim.
- Remaining report suggestions for QKV/O DRAM sharding, reader count, and
  subblocks are closed by the geometry matrices above. No applicable advice or
  material decoder optimization is deferred.

## AutoDebug, AutoTriage, and AutoFix closure

The first R22 watcher gate identified a terminal runtime-argument over-read in
the sharded QKV head-split reader. `AUTODEBUG.md`, stage `AUTOTRIAGE.md`, and
`AUTOFIX.md` retain the source/runtime ledger. Converting the projection output
once to L1-interleaved selects the safe reader; focused watcher tests and the
final watcher suite pass. The repair costs about 9–10 us and remains faster
than legal nonresidual controls.

One combined long-context run later stalled during the 262,143-token sliding
prefill. Captured `triage/context_capacity_tt_triage.txt` showed `cq_prefetch`
waiting on five tagged pinned-host NoC reads and `cq_dispatch` waiting for the
downstream DRAM-write payload. There was no active model op, allocator OOM, ARC
failure, or DRAM error. After a clean reset, an unpinned control passed in
40.90s, a fresh pinned repeat passed in 41.38s, and the final combined four-case
context gate passed in 183.11s. Pinning was not a reproducible cause, so no
runtime workaround was kept.

AutoFix also found the persistent accounting omitted the distinct B32 packed
expert allocation. The helper now covers B1/B32 packed/unpacked expert weights
and attention aliases, skips released wrappers, and deduplicates by buffer ID.
Host regressions and five selected live accounting cases pass. These pre-review
values (superseded by the final 256-byte routing-base accounting below) were
2,116,256,768 bytes sliding, 2,111,524,864 bytes full, and 63,464,043,520 bytes
for the 25/5 projected layer mix.

## Pre-review frozen-source correctness and performance

PCC acceptance is 0.995 for every meaningful layer kind. The historical
pre-review results were:

| Case | Sliding | Full |
| --- | ---: | ---: |
| HF prefill | 0.999252011 | 0.998321241 |
| HF decode | 0.996587866 | 0.995966596 |
| fused vs optimized prefill | 0.999698712 | 0.997576330 |
| fused vs optimized decode | 0.996685143 | 0.995892446 |
| B32 trace aggregate/min user | 0.999300220 / 0.995802051 | 0.999781521 / 0.999781521 |
| batch-2 prefill | 0.998279693 | 0.998235973 |
| lowest logical-boundary prefill | 0.995123240 | 0.997922075 |

| Workload | Fused | Final BF16 cache | Final BFP8 cache |
| --- | ---: | ---: | ---: |
| B1 sliding prefill | 278.380476 ms | 96.218797 ms | 96.436061 ms |
| B1 full prefill | 279.611919 ms | 107.555402 ms | 107.664378 ms |
| B1 sliding trace decode | 1.309336 ms | 0.849307 ms | 0.841330 ms |
| B1 full trace decode | 1.487052 ms | 0.872289 ms | 0.866062 ms |
| B32 sliding trace decode | 19.500527 ms | 12.840555 ms | — |
| B32 full trace decode | 19.317333 ms | 12.532409 ms | — |

## Pre-review commands and artifacts (historical)

All pytest hardware commands used
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'` and
`GEMMA4_RANGE_DOWNLOAD=1`.

Complete default suite:

```bash
GEMMA4_OPT_CANDIDATE_ID=final_batch_aware_default_suite_formatted \
GEMMA4_OPT_EXACT_COMMAND='final formatted batch-aware complete optimized decoder suite, fallback exceptions enabled' \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_results.xml
```

Result: `33 passed, 12 skipped in 115.99s`. Skips are the explicitly gated
context/performance/serving cases below.

Context contract:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
GEMMA4_OPT_CANDIDATE_ID=final_formatted_context_capacity \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_advertised_context or optimized_prefill_capacity' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/context_capacity_results.xml
```

Result: `4 passed in 183.11s`.

Pre-review canonical performance and BFP8 candidate timing:

```bash
GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
GEMMA4_OPT_CANDIDATE_ID=final_formatted_perf \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_decoder_perf_profile \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/perf_results.xml

GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
GEMMA4_OPT_KV_CACHE_DTYPE=bfp8 \
GEMMA4_OPT_CANDIDATE_ID=final_formatted_bfp8_cache_perf \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'test_optimized_decoder_perf_profile and batch1' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/bfp8_cache_perf_results.xml
```

Results: `4 passed in 48.16s`; `2 passed in 13.23s`.

Serving-prefill contract:

```bash
GEMMA4_OPT_SERVING_PREFILL_PERF=1 \
GEMMA4_OPT_CANDIDATE_ID=final_formatted_serving_prefill \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k serving_batch32_prefill_perf \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/serving_prefill_results.xml
```

Result: `2 passed in 26.23s`, 3036.312916/3442.461965 ms sliding/full.

Watcher and stress (profiler disabled):

```bash
TT_METAL_WATCHER=10 GEMMA4_OPT_CANDIDATE_ID=final_formatted_watcher \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode or bfp8_nonaligned_prefill_cache_consuming_decode or (traced_decode_batch_contract and batch32) or trace_mutable_stable_buffers or bounded_modulo_decode_stress' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/watcher_results.xml
```

Result: `11 passed in 100.58s`. A post-run `tt-smi -ls --local` at
2026-09-05T11:13:49Z showed four healthy P300C devices, healthy DRAM, DDR
`0x55555555`, and zero corrected/uncorrected GDDR errors. See
`post_watcher_health_summary.json`.

Final Tracy used the same command form once per layer/cache pair, changing the
node ID and capture/candidate names:

```bash
GEMMA4_FUNCTIONAL_DECODER_PERF=1 GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=5 \
GEMMA4_OPT_DECODE_DEVICE_PROFILE=1 GEMMA4_OPT_CANDIDATE_ID=<candidate> \
python_env/bin/python -m tracy -r -p -o <capture-dir> -m pytest \
'models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::<exact-batch1-node-id>'
```

Candidates/captures are `final_formatted_profiler_{sliding,full}` at
`generated/profiler/gemma4_optimized_batch_aware_formatted_{sliding,full}` and
the corresponding `_bfp8` names with `GEMMA4_OPT_KV_CACHE_DTYPE=bfp8`.
All four passed. The exact host numbers and provenance are in the four
candidate JSON files.

Advice-enabled reports were generated from each capture's
`reports/*/ops_perf_results_*.csv` with:

```bash
python_env/bin/tt-perf-report <ops-csv> \
--start-signpost <exact-PERF_PREFILL_layer-kind-or-OPTIMIZED_DECODE_TRACE_REPLAY> \
--end-signpost <matching-end> [--active-experts 8-for-decode-only] \
--csv <phase.csv> --stacked-csv <phase_summary.csv>
```

The then-current CSV/PNG outputs are in
`tt_perf_report/final_formatted_{sliding,full}`
and `tt_perf_report/final_formatted_{sliding,full}_bfp8`.

Prefill sparse `nnz` is the route union over a 32-token chunk and is not fixed
at eight. Final prefill reports therefore omit the decode-only
`--active-experts 8` override and label utilization unknown unless joined to a
captured per-chunk union. Decode uses exact `nnz=8` and retains the override.

`final_perf_results.xml` is a historical four-case run. The pre-review frozen
formatted run was `perf_results.xml` (48.156 s). Top-level host-timing JSON
filenames are mutable test outputs and then contained the reverted router-L1
candidate; the corresponding pre-review snapshots are the immutable entries in
`candidate_runs/final_formatted_perf.json`.

## Pre-review profiler accounting

| Cache/kind | Ops | Device work | Gaps | Span | Same-run host | Canonical host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 sliding | 72 | 0.813272 ms | 0.073286 ms | 0.886558 ms | 0.906826 ms | 0.849307 ms |
| BF16 full | 75 | 0.838203 ms | 0.069945 ms | 0.908148 ms | 0.940990 ms | 0.872289 ms |
| BFP8 sliding | 72 | 0.804855 ms | 0.072788 ms | 0.877643 ms | 0.908886 ms | 0.841330 ms |
| BFP8 full | 75 | 0.834798 ms | 0.070270 ms | 0.905068 ms | 0.927989 ms | 0.866062 ms |

BF16 prefill contains 615 ops and spans 96.569241/107.694445 ms; BFP8 cache
adds two fill typecasts (617 ops) and spans 96.578587/107.728887 ms. The Tracy
five-replay host values are used only to reconcile the signposted device
window; the independent 1,000-replay rows are the canonical latency ranking.

The conservative bytes are 123,469,824 sliding and 106,348,544 full, giving
0.241152/0.207712 ms floors at 512 GB/s. Effective bandwidth including gaps is
139.27/117.10 GB/s with BF16 cache and 140.68/117.50 GB/s with BFP8 cache.
`profiler_summary.json` retains the complete final ledger. The final code beats
the strongest correct candidate and every fused baseline row; fewer ops alone
was never an acceptance criterion.

## Superseded first-review remediation evidence

This section records the first remediation freeze for chronology. Its
`reviewfix_*` R22 reader rows were later proven to have unused candidate
weights because two loader flags were discarded; they are not final candidate
evidence. The authoritative final v5 section below supersedes all timing,
hash, allocation, and profile claims in this section.

The initial `STAGE_REVIEW.md` verdict was `more-work-needed`. `$autofix`
consumed a fresh AutoDebug report and closed every finding with isolated
experiments before rerunning cumulative gates.

- H1: BF16 nonaligned prefill followed by cache-consuming decode passes at
  sliding length 1025 and full length 33 for natural/shared views, with minimum
  PCC 0.997232031/0.995421141. The same actual BFP8 cache path is deterministic
  but measures 0.994624887/0.988673202 and is rejected normally, not xfailed.
- H2: packed expert block44 is legal/correct but does not stably beat block22;
  fair separate accurate-GeGLU measures 0.893834/0.915895 ms and loses by about
  5%.
- H3 (invalidated): the `reviewfix_r22_*` files allocated candidate weights but
  their host counters remained zero. They are retained only as evidence of the
  loader defect and must not be used for correctness or performance decisions.
- H4: prefill QKV/O 2D grid/block candidates were adapted beyond initial
  legality failures. Full QKV and the locally faster sliding candidate fail
  downstream fused-equivalence PCC; correct full O and input-L1 regress.
- H4 movement: direct row-major routing metadata is correct, trace-stable, and
  selected. It removes two untilizes and one unary per replay, reducing decode
  op count from 72/75 to 69/72.
- H5: stale hashes, BFP8 claims, profile commands, timing identity, and health
  prose were replaced with final immutable artifacts.

Final executable gates and artifacts:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix_default_suite_v3 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_results_reviewfix_v3.xml

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix_perf_v3 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_decoder_perf_profile \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/perf_results_reviewfix_v3.xml

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 \
GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix_context_capacity_v2 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_advertised_context_traced_decode or optimized_prefill_capacity_probe' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/context_results_reviewfix_v2.xml

TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix_watcher_v2 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_real_weights_prefill_decode or (optimized_nonaligned_prefill_cache_consuming_decode and kv_bf16) or (optimized_traced_decode_batch_contract and batch32) or optimized_trace_mutable_stable_buffers or optimized_bounded_modulo_decode_stress' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/watcher_results_reviewfix_v2.xml
```

Results: complete suite `44 passed, 15 skipped in 121.66s`; canonical timing
`4 passed in 48.22s`; context `4 passed in 183.24s`; watcher `11 passed in
97.96s`; serving B32 prefill `2 passed in 26.52s`; persistent accounting `5
passed in 11.54s`. Watcher and profiler were never enabled together. A fresh
post-watcher `tt-smi -s` showed all four P300C devices with healthy DRAM and no
remaining workload process.

Final host timing is 96.317813/107.524409 ms prefill and
0.841313/0.864549 ms B1 traced decode for sliding/full. B32 traced decode is
12.837672/12.537022 ms. Fresh selected-profile device spans are
96.587428/107.726896 ms prefill and 0.872457/0.896397 ms decode. Prefill reports
omit the incorrect fixed-eight route assumption; decode uses exact eight-route
metadata. Same-profile five-replay decode host timings are
0.905540/0.933437 ms; host-minus-device-span is 0.033083/0.037040 ms. With the
conservative 123,469,824/106,348,544-byte model, the 512 GB/s floors are
0.241152/0.207712 ms and measured span corresponds to 141.520/118.640 GB/s.
The final advice closure is in `profiler_summary.json`. Human-readable tables,
CSV, summaries, and plots are in `tt_perf_report/final_reviewfix_{sliding,full}`.

The selected routing base adds 256 bytes per layer. Live page accounting gives
2,116,257,024/2,111,525,120 bytes. The long context, watcher, serving, and Tracy
runs predate only the test-helper enumeration of this already-present buffer;
the runtime source hash is exact and unchanged. The complete suite and
canonical timing were rerun under the final test hash.

## Final v5 loader repair, candidate matrix, and frozen evidence

The independent rereview found that `FunctionalDecoder.from_state_dict`
accepted but discarded two optimized-only constructor flags. Candidate weights
were allocated, while `decoder.r22_dram_sharded` and
`decoder.r22_packed_dense_gate_up` stayed false and the intended paths did not
execute. The repair restores both flags after the fused loader returns, tracks
the public logical decode batch separately from tile-padded composite tensors,
and makes tests assert requested/default R22 counter deltas from environment
intent rather than trusting instance flags. The original `reviewfix_r22_*`
files are invalid unused-path evidence. Only `reviewfix2_r22_*` files inform
the final decision.

True one/two/three-reader B1 decode measurements, in sliding/full ms:

| Candidate | Reader 1 | Reader 2 | Reader 3 | Decision |
| --- | ---: | ---: | ---: | --- |
| QKV | 0.812760 / 0.880103 | 0.868099 / 0.878559 | 0.849659 / 0.867514 | reject: reader 1 sliding direct-fused PCC 0.987622; full loses |
| O | 0.814941 / L1 invalid | 0.832050 / 0.866755 | 0.844712 / 0.879254 | select sliding reader 1 only |
| separate dense | 0.838778 / 0.861295 | 0.860420 / 0.882572 | 0.847404 / 0.869799 | correct but loses packed reader 1 |
| packed dense | 0.834892 / 0.858662 | 0.858390 / 0.880213 | 0.848393 / 0.870567 | select reader 1 both kinds |

The full O reader-1 failure was not accepted as a first API error. Reducing its
local K block from the default 32 to 8 passes both full cache views, but measures
0.876961 ms and loses. Sliding O plus packed dense/down composes correctly at
0.811630 ms. The no-env final reproduction is 0.811773/0.859455 ms and records
nonzero O/packed/down counters; full intentionally records zero O/QKV DRAM
counters. Direct fused prefill/decode PCC is 0.999698712/0.996685143 sliding
and 0.997576330/0.995892446 full. The setup-only interleaved packed-dense
source is released after DRAM sharding; prefill/B32 retain their required
separate originals.

Final source identity:

- decoder: `feebc8cb2f20ad9ba81c7d0f50f8323694d6d91ebb31cb9072e18b0f6b0a9c45`
- tests: `01ed0de36451891c6c41968baeeae34f1cb77acf4e3c6db0c322506fb83bd349`

The v5 context, watcher, allocation, serving, and Tracy artifacts were captured
with test hash `cc236183671b8a0e18e8c16f2a30754a7678facb1042e07548756ecabf97848d`.
Pre-commit subsequently collapsed one `dense_roles` ternary from three lines
to one. Independent byte reconstruction recovered that prior hash exactly, and
the two files have identical AST dumps. The v6 complete and canonical
performance gates below execute the final formatted hash; the runtime hash is
unchanged across every gate.

Final commands:

```bash
GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix2_default_suite_v6 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/final_results_reviewfix2_v6.xml

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=1000 \
GEMMA4_OPTIMIZED_PREFILL_BATCH32_PERF=1 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix2_perf_v6 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k test_optimized_decoder_perf_profile \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/perf_results_reviewfix2_v6.xml

GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 \
GEMMA4_PREFILL_CAPACITY_LENGTH=262143 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix2_context_capacity_v5 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_advertised_context_traced_decode or optimized_prefill_capacity_probe' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/context_results_reviewfix2_v5.xml

TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_OPT_CANDIDATE_ID=final_reviewfix2_watcher_v5 \
python_env/bin/python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'optimized_real_weights_prefill_decode or (optimized_nonaligned_prefill_cache_consuming_decode and kv_bf16) or (optimized_traced_decode_batch_contract and batch32) or optimized_trace_mutable_stable_buffers or optimized_bounded_modulo_decode_stress' \
--junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/watcher_results_reviewfix2_v5.xml
```

The complete suite passed `44 passed, 15 skipped in 121.22s`; the skips are
the separately run context/performance/serving/BFP8 gates. Performance passed
`4 passed in 48.50s`, context `4 passed in 183.75s`, watcher/stress `11 passed
in 97.58s`, serving prefill `2 passed in 26.52s`, and allocation `5 passed in
11.68s`. Post-watcher `tt-smi -s` reports four healthy P300C DRAM statuses and
no remaining workload process.

Final warmed host latency is 96.365912/107.636814 ms prefill,
0.811915/0.859545 ms B1 traced decode, and 12.837788/12.541034 ms B32 traced
decode. Fused B1 baselines are 278.380476/279.611919 ms prefill and
1.309336/1.487052 ms decode. Serving B32 prefill is
3036.028061/3441.924553 ms.

Fresh Tracy captures are
`generated/profiler/gemma4_optimized_reviewfix2_{sliding,full}`; exact raw CSV
paths are in `profiler_summary.json`. Rendered tables/CSV/summaries/plots are
in `tt_perf_report/final_reviewfix2_v5_{sliding,full}`. Prefill has 615 ops and
96.569768/107.709100 ms device span. Decode has 70/73 ops and
0.841089/0.892882 ms span; same-profile host is 0.867675/0.918276 ms. The
123,469,824/106,348,544-byte conservative model gives 0.241152/0.207712 ms
floors at 512 GB/s and 146.798/119.107 GB/s effective span bandwidth. Prefill
sparse rows correctly retain unknown route-union utilization; decode uses
exact `active=8/128`.

Persistent bytes are 2,158,283,008 sliding, 2,130,482,432 full, and
64,609,487,360 for the 25/5 projected layer mix. The final context run proves
the advertised 262,144-token contract remains feasible, so no capability is
reduced.

## Optimize checklist

- [x] Optimized-path functional, fused/HF PCC, both layer kinds, paged BF16
  cache, trace determinism, mutable buffers, batch 1/2/32, and wrap stress
  pass; static and runtime fallback audits are clean. BFP8 was exercised and
  rejected by its cache-consuming numerical gate.
- [x] Decode is traced. Activations are generally BF16 R22 L1 width-sharded
  across norms, residual adds, attention, dense MLP, and O boundaries; required
  cache/MoE boundaries and the watcher-safe head split are measured exceptions.
- [x] Prefill is DRAM-interleaved, uses large 2D matmul/sparse programs, and
  accepts non-aligned logical lengths through internal chunk/tail programs.
- [x] Initial operation topology, same-input projections, movement, packing,
  composite ops, precision constraints, and candidate actions are recorded.
- [x] Multi-device/collective layout, fused CCL-matmul, and persistent CCL
  checklist items are not applicable to this explicitly single-device decoder
  layer; no collective exists in the measured graph.
- [x] Coherent lower-movement R11/R22 residual families and a legal nonresidual
  control were measured without disguising restore costs. R22 wins.
- [x] Final BF16 default reproduces the selected best correct candidate and
  beats the strongest correct fused/control candidates in traced warmed decode.
- [x] Final profiler rows verify actual activation/weight dtype and fidelity for
  every dominant attention, dense, router, and expert matmul.
- [x] Composite scaled-dot-product attention and paged cache ops are retained.
- [x] Packed QKV, dense gate/up, and expert gate/up families were compared with
  legal separate/packed candidates including split, activation, and movement
  overhead. Only measured winners remain in the final path.
- [x] Important memory, program, and compute-kernel configurations are explicit.
- [x] Dominant QKV, O, dense gate/up/down, router, expert gate/up/down roles were
  swept for applicable grid, block, per-core N, subblock, memory placement,
  weight precision, and LoFi/HiFi2/HiFi4 fidelity.
- [x] Attention BFP4, dense BFP4 gate/up and down, expert BFP4 gate/up and down,
  BFP8, and cache BFP8 were tested with real weights/activations. Synthetic PCC
  did not veto a real-weight win.
- [x] Shard specs use clean tile division where possible; inert padding enabled
  G22 QKV/full-O candidates without imposing a public alignment contract.
- [x] Blackhole DRAM-sharded and one/two/three-reader candidates were adapted
  and tested for material legal roles; one reader is the measured winner.
- [x] Routed MoE uses exact active-expert `sparse_matmul`, exact `nnz`, separately
  tuned gate/up and sparse down, score weighting/reduction, and L1 intermediates.
- [x] LM-head, logits, sampling, token feedback, and LM-head DRAM-sharding items
  are not applicable to a decoder-layer-only stage and were not started.
- [x] Context capacity and corrected persistent allocation are reconciled; the
  advertised contract is unchanged.
- [x] Prefill/decode tt-perf reports with advice, roofline/device/host accounting,
  repeated runs, final default reproduction, and watcher-clean evidence exist.
- [x] Batch capability is preserved through batch 32 with a measured
  correctness-safe batch-aware policy.
- [x] Independent final `$stage-review` clean-pass; see
  `STAGE_REVIEW_FINAL.md`.
- [ ] Local stage-owned commit; SHA recorded below and never pushed.

## Local commits

Independent review: `STAGE_REVIEW_FINAL.md`, verdict `clean-pass`.
Stage-owned commit pending.
