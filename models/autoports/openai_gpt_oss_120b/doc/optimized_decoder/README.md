# OpenAI GPT-OSS 120B optimized decoder

Status: optimized-decoder implementation and evidence complete. Operator recovery restored the
four-board inventory, the full 80-node exact-activation matrix ran, and the
final capacity-split production path passed batch 1/2/32, integrated, watcher,
same-run profiler/roofline, and post-run health gates. The decoder remains a
distinct optimized TTNN path; its tests cannot silently substitute the
functional decoder.

The implementation preserves the fused decoder's public prefill/decode API,
paged BFP8 KV cache, deterministic trace replay, non-aligned logical lengths,
batch-32 allocation, and 131072-token advertised context. No change to
`doc/context_contract.json` was necessary.

## Delivered runtime policy

- Omitted/`None` policy selection is based on configured `max_batch_size`, not
  the logical batch of an individual call. Configured batch 1 uses BFP8
  attention projection weights with HiFi2 decode projection math; configured
  batch 2 through 32 uses BFP4 attention projection weights with LoFi decode
  projection math, BF16 output activations, and the promoted DRAM-sharded QKV
  `15c_ibw6_pcn10` program. The packed
  same-input QKV projection remains packed. The capacity-1 interleaved path
  folds bias into `ttnn.linear`; the DRAM-sharded path performs the required
  device-side add after returning to interleaved DRAM. Final-topology exact
  A/Bs give separate Q/K/V the same 32-core output projection. Three separate
  DRAM-sharded Q/K/V matmuls are legal when each projection has its own program
  config: the measured control uses common 15-core/`in0_block_w=6` input
  sharding and `per_core_N=(9,2,2)`. A low-memory exact-attention gate reached
  that path for both layer kinds and capacities 1/2 without loading experts.
  Direct prefill/decode PCC against packed is 1.000000/0.999944--1.000000,
  selected-only paged-cache mutation and 1000-replay determinism pass, and the
  separate shell is 13.70--15.48% slower. Adding the measured shell delta to
  the prior exact packed whole traces estimates a 3.526--6.377% whole-decoder
  penalty. Packed therefore remains final on measured evidence. The layer-1
  low-memory input is explicitly an exact embedding-row proxy, not a saved
  post-layer-0 activation.
- BFP8 and BF16 remain explicit measured controls. At configured batch 2,
  automatic DRAM-QKV BFP4/LoFi measures 0.778263 ms sliding and 0.727294 ms
  full, improving the former non-DRAM automatic's
  0.788169--0.788481/0.744771--0.745105 ms. It beats explicit BFP8 by 2.920%
  sliding and 10.937% full in the final integrated run. At configured
  batch 1, BFP4/LoFi regresses sliding decode by 3.67%, so automatic selection
  retains BFP8/HiFi2 there. Synthetic/random failures remain opt-in diagnostics
  and do not veto exact checkpoint prompt-activation wins.
- All final production policies and retained BF16/BFP8/BFP4 attention-precision
  controls use the same decode output-projection topology: SDPA output is
  width-sharded over an `8x4` grid with shard
  `(32, 128)`, then a 1D multicast linear uses `in0_block_w=4`,
  `per_core_N=3`, `out_subblock_w=3`, `mcast_in0=True`, and
  `fuse_batch=True`. This complete whole-trace path, including the reshard,
  reduced traced decode by 5.23% sliding and 5.11% full relative to the
  previous best correct optimized path.
- Both decode RMSNorms use a ten-core L1 width-sharded program at batch 1/2.
  At batch 32, both use the canonical interleaved RMSNorm. The input norm was
  already isolated as trace-unstable at a full tile of users; after promoting
  the 32-core output projection, the post-attention sharded L1 norm introduced
  a second lifetime overlap. Disabling only that capacity-bound sharding
  restored bitwise replay while retaining the promoted projection and every
  other optimized component.
- The router's small BF16 weight and bias stay resident in L1. Decode retains
  the fused stage's exact-revision FullLocal active-expert MoE path: BFP4
  expert weights, LoFi compute, packed gate/up, four active experts, and the
  fused score reducer. Other layers/batches retain the fused indexed path.
- Prefill retains the fused stage's large packed projection program configs,
  paged SDPA, internal logical-length padding/chunking, and unified routed
  expert operation.

This is intentionally a `1x1` decoder. P150x2/P150x4 qualification in this
stage means the same single-device path runs correctly when those boards are
present on the host; tensor/mesh parallelism belongs to the later multichip
decoder stage.

## Projection remediation and configuration evidence

Stage review found that the first explicit output-projection trial was not a
legal correctness candidate. Its `per_core_N=90 // 64=1` covered only 64 of
the required 90 output tiles, explaining its 0.838096 decode PCC. Four
ceiling-covered geometries were then tested on real weights. The first legal
64-core run exposed TTNN's requirement that sharded input A use
`fuse_batch=True`; that API/config error was repaired and the candidate was
rerun rather than rejected.

| Output-projection geometry | Sliding prefill/decode PCC | Warmed prefill / traced decode | Decision |
| --- | --- | ---: | --- |
| 64 cores, `ibw2/pcn2/sb2` | 0.973261 / 0.992752 | 35.795665 / 0.561594 ms | Correct; slower than the 32/16-core candidates. |
| 32 cores, `ibw4/pcn3/sb3` | 0.973261 / 0.992055 | 35.626650 / 0.535187 ms | Best; promoted to every production/control policy. |
| 16 cores, `ibw8/pcn6/sb6` | 0.973261 / 0.992520 | 35.791649 / 0.536127 ms | Correct, but 0.18% slower than 32 cores. |
| 8 cores, `ibw16/pcn12/sb6` | 0.973261 / 0.993205 | 35.780629 / 0.566041 ms | Correct, but slower than the prior 0.564745 ms path. |

The prior DRAM-sharded QKV PCC failures were traced to fused bias in this
matmul family. The repaired candidate omits bias from the DRAM-sharded linear,
returns the result to interleaved DRAM, and performs an in-place device add
before packed-head creation. That raised the 90-core BFP8 decode PCC from
approximately 0.916125 to 0.993434.

The earlier statement that DRAM QKV was completely swept and rejected is
superseded. Those measurements used random activations, omitted the promoted
32-core output projection, and did not cross BFP8/BFP4 with both HiFi2/LoFi at
all five geometries. The source now defines that exact 20-row cumulative
cross-product, and every row uses `32c_ibw4_pcn3_sb3`. The opt-in test compares
each candidate with the applicable capacity-selected automatic baseline in the
same process at configured batch 1 and 2, both representative layer kinds,
using exact prompt-derived activations, unchanged 0.95/0.99 PCC bars, warmed
prefill, and first/second plus post-1000 traced determinism.

All 80 nodes executed after operator recovery. BFP4/LoFi 15-core was the
fastest candidate clearing every PCC bar, with -4.724/-4.712% batch-1 and
-1.458/-2.561% batch-2 sliding/full deltas; reversed order reproduced strict
wins. The 10-core BFP4 rows were rejected because batch-1 sliding decode PCC
was 0.989157931 below 0.99. The sequence-128 production gate then retained
BFP8 at capacity 1 because DRAM15 sliding measured 0.541827 ms versus the
correct BFP8 0.535201 ms. `AUTOFIX_dram_qkv_final32c_matrix.md` records the
complete matrix, recovery history, commands, and exact artifacts. The promoted
DRAM policy is exported; rejected geometries remain private.

## Correctness

Exact checkpoint revision:
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.

| Representative layer | Fused prefill PCC | Optimized prefill PCC | Fused decode PCC | Optimized decode PCC |
| --- | ---: | ---: | ---: | ---: |
| layer 0, sliding | 0.978123157 | 0.973260504 | 0.990064440 | 0.992054854 |
| layer 1, full | 0.990890 | 0.990510676 | 0.955781 | 0.956693412 |
| layer 1, full, batch 2, explicit BF16 control | 0.992492223 | 0.992492223 | 0.992720098 | 0.992070182 |
| layer 0, sliding, batch 2, automatic BFP4/LoFi DRAM15, prompt-derived | n/a | 0.993718903 | n/a | 0.993277385 |
| layer 1, full, batch 2, automatic BFP4/LoFi DRAM15, prompt-derived | n/a | 0.994043785 | n/a | 0.993977524 |

All rows pass the unchanged functional acceptance assertions. The first three
rows are historical batch-1/BF16 controls. The final two rows are the current
multibatch automatic production path and clear the unchanged 0.95 prefill and
0.99 batch-decode bars.

The final batch-2 policy A/B additionally used exact checkpoint embedding rows
for layer 0 and an exact HF layer-0 pass to produce layer-1 activations for two
fixed 33-token text sequences. Both policies passed unchanged PCC bars and
remained bitwise equal after 1000 traced replays. Sliding automatic BFP4/LoFi
DRAM15 versus explicit BFP8 decode PCC was 0.993277385/0.997049920 and traced
latency was 0.778263/0.800991 ms. Full-layer PCC was
0.993977524/0.998445515 and latency was 0.727294/0.806837 ms. The preceding
final-32-core attention-precision cross-product covered BFP8/BFP4, HiFi2/LoFi
decode projections, and BF16/BFP8 output activations in both orders; all rows
passed, and BFP4/LoFi without output-activation quantization was the fastest
non-DRAM precision policy for both layer kinds. Historical
exact-weight/random-activation and synthetic failures remain opt-in diagnostics
at unchanged bars and do not select production precision.

Additional qualification passed:

- non-aligned tile/page/window boundary cases for both layer kinds;
- logical lengths 4095, 4096, and 4097 for both layer kinds;
- prefill lengths 131071 and 131072 and traced decode at position 131071;
- a two-order batch-2 automatic-BFP4/LoFi versus explicit-BFP8 A/B with exact
  prompt-derived layer-0 and layer-1 activations, unchanged PCC bars, and
  bitwise equality after 1000 trace replays;
- opt-in exact-weight/random-input and synthetic BFP8 diagnostics which retain
  the unchanged 0.99 bar without selecting production precision;
- automatic-BFP4/LoFi batch-32 paged-cache allocation and bitwise-equal trace
  replay for sliding and full attention at 131072 tokens per user;
- repeated 1000-replay performance/determinism runs;
- repaired DRAM-QKV bias followed by a full dtype/fidelity/geometry sweep, and
  legal 64/32/16/8-core output-projection correctness/performance sweeps;
- a focused watcher run with both non-aligned optimized layer kinds and no
  watcher errors. The broad `--dev` suite passed four earlier checks before
  advertised-context paged SDPA could not compile because assert
  instrumentation grows its program to 81984 bytes, beyond the 70656-byte
  TENSIX kernel-config buffer. That failure is retained as tooling-limit
  evidence and was not used for latency.

The post-repair default integrated module run passed 8 tests and skipped 183 explicit
opt-in candidate, diagnostic, performance, and expensive capacity gates. It
included both real-activation layer kinds and 1000 trace replays. The gated
runs were executed separately and are retained under `evidence/logs/`.

## Performance

Warmed wall time is the optimization decision metric. Prefill is one warmed
sequence-128 call; decode is a captured trace replayed 1000 times. Lower is
better.

| Layer kind | Fused prefill wall | Optimized prefill wall | Prefill change | Fused traced decode wall | Optimized traced decode wall | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sliding | 37.081505 ms | 35.609699 ms | 3.97% lower | 0.719116 ms | 0.535223 ms | 25.57% lower |
| full | 36.431644 ms | 30.277082 ms | 16.89% lower | 0.728915 ms | 0.547673 ms | 24.86% lower |

At configured batch capacity 2, the final integrated prompt-derived 33-token
A/B measured sliding automatic DRAM-QKV BFP4/LoFi versus explicit BFP8 traced
decode at 0.778263/0.800991 ms and full-layer decode at 0.727294/0.806837 ms.
Automatic BFP4/LoFi is selected because both policies clear the production
real-activation bars while the DRAM policy is 2.920% faster sliding and 10.937%
faster full. At capacity 1,
omitted policy reproduces the best BFP8 result at 0.535201 ms sliding and
0.547660 ms full; the measured all-capacity BFP4 alternative regressed sliding
to 0.554860 ms and was rejected.

The final candidate beats the best correct fused traced-decode baseline for
both layer kinds and improves on the previous best correct optimized candidate
by 5.23% sliding and 5.11% full. Fresh marker-clean, ten-replay Tracy captures
of the exact promoted source independently confirm the win:

| Layer kind | Fused → optimized prefill device | Optimized prefill ops / host | Fused → optimized decode device | Optimized decode ops / host |
| --- | ---: | ---: | ---: | ---: |
| sliding | 36.220 → 34.982 ms | 66 / 0 | 0.678 → 0.504847 ms | 36 / 0 |
| full | 35.558 → 29.655 ms | 66 / 0 | 0.689 → 0.517103 ms | 36 / 0 |

The final trace contains no Torch conversion, host fallback, or host op. Its
three interleaved-to-sharded operations total 3.24/3.50 us for sliding/full
decode and feed legal sharded RMSNorm/head contracts. The one device-side
SDPA-output reshard added by the promoted topology averages 1.45/1.49 us and
feeds the 32-core output linear; the complete 36-op trace is still faster than
both the fused and previous optimized traces.

The final `tt-perf-report` identifies prefill UnifiedRoutedExpert as
89.00-90.64% of device time. Decode is dominated by FullLocal MoE
(45.14-46.57%), three matmuls (20.08-20.60%), head creation (8.62-8.85%), and
SDPA (4.15-4.34%).

The promoted capacity-2 graph has its own marker-clean ten-replay Tracy
capture; it is not inferred from the capacity-1 graph. Runtime rows directly
show BFP4 QKV/output weights, LoFi projection math, ten DRAM-sharded packed-QKV
rows, and ten 32-core output rows with `in0_block_w=4` and output subblock
`1x3`. There are 57 device operations per replay and no host rows:

| Layer kind | Prefill device / same-run wall / ops | Decode device / same-run wall / ops |
| --- | ---: | ---: |
| sliding | 40.157148 / 41.894460 ms / 74 | 0.628083 / 0.705526 ms / 57 |
| full | 41.044686 / 42.394656 ms / 74 | 0.728872 / 0.799062 ms / 57 |

The profile wall includes instrumentation and does not replace the
uninstrumented 1000-replay capacity-2 decision measurements (0.687493 ms
sliding and 0.779065 ms full in the sequence-128 harness, or the exact-prompt
DRAM15-versus-BFP8 A/B above). Capacity-2 prefill is 87.68--87.74% unified
routed experts. Decode is 51.86--58.39% FullLocal MoE, 9.36--10.85% projection
matmuls, 6.17--7.14% head creation, and 3.12--3.73% SDPA.

The required theoretical/device/wall accounting comes from the same final-source
profile invocation for each layer kind. A 100-replay attempt was rejected
because profiler DRAM buffers filled and dropped markers. Ten replays retained
all markers and 36 decode operations per replay; the table uses the summed
device rows divided by ten and the wall timer printed by that same pytest run.

| Layer kind | 512 GB/s compulsory-traffic roofline | Tracy device (10-replay mean) | Same-run wall (10-replay mean) | Wall-device gap | Effective compulsory GB/s, device/wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| sliding | 0.183868 ms | 0.504847 ms (2.75x) | 0.555876 ms (3.02x) | 0.051029 ms (9.18% wall) | 186.5 / 169.4 |
| full | 0.183868 ms | 0.517103 ms (2.81x) | 0.567259 ms (3.09x) | 0.050155 ms (8.84% wall) | 182.1 / 166.0 |

The roofline is a conservative lower bound from 94,140,416 compulsory physical
bytes per layer-token at context 128. TT BFP8 and BFP4 tiles occupy 1088 and
576 bytes, including their tile headers. The count is 15,667,200 packed-QKV
weight bytes, 12,533,760 output-projection bytes, 737,280 router bytes,
65,028,096 bytes for four active experts including the FullLocal ring and bias
padding, 139,264 KV-read bytes, and 34,816 physical KV-write-tile bytes. It
intentionally excludes intermediate spill/reload traffic, so it can understate
the required device time and overstate achievable whole-path bandwidth. The
same-run wall-device gaps include profiled dispatch/runtime overhead and are
not substituted for the uninstrumented 1000-replay decision metric above.

`tt-perf-report` separately prints 52-53 GB/s (10.1-10.4% of its 512 GB/s peak)
for the operations its performance model can account for. The fused
`MoEComputeDeviceOperation` row has blank bandwidth/FLOP modeling, so that
number excludes the largest 45.14-46.57% operation and is not a whole-decoder
bandwidth estimate. Exact component formulas and same-run comparison rows are
retained in `tracy/decode_roofline.csv`; the marker-clean and rejected-overflow
profile transcripts are retained under `evidence/logs/`.

For the two-user capacity-2 trace, the conservative compulsory count is
146,071,552 physical bytes: 8,294,400 BFP4 packed-QKV bytes, 6,635,520 BFP4
output-projection bytes, 737,280 router bytes, 130,056,192 bytes for eight
active-expert executions, 278,528 KV-read bytes, and 69,632 physical KV-write
bytes. At 512 GB/s its roofline is 0.285296 ms. The same profiler invocations
give 0.628083/0.705526 ms device/wall for sliding (2.20x/2.47x roofline) and
0.728872/0.799062 ms for full (2.55x/2.80x). Exact formulas are retained in
`tracy/capacity2/decode_roofline.csv`.

Capacity-2 `tt-perf-report` advice was exercised rather than accepted from a
single API result. Decode's output block/subblock is already the measured
32-core winner; 64/16/8-core final-policy controls were 4.61--4.90%,
0.248--0.250%, and 3.03--3.21% slower. Its HiFi2 accuracy suggestion was
covered by the real-weight BFP4 HiFi2/LoFi matrix; LoFi clears the unchanged
bars and wins whole-trace latency. Prefill's L1-input suggestion is unchanged
from the capacity-1 report: the sequence activation plus projection working
set cannot remain wholly in L1, while the inherited large configs already use
the reported good blocks. The unconfigured router is only 0.22--0.24% of the
capacity-2 prefill device path, so even eliminating it cannot be material to
the warmed end-to-end result.

## Evidence

- `work_log.md`: full topology audit, candidate ledger, commands, profiler
  advice, device record, and completed optimize checklist.
- `tracy/{sliding,full}/ops_perf_results.csv.gz`: exact losslessly compressed
  raw Tracy captures.
- `tracy/{sliding,full}/{prefill,decode}_perf_report.csv`: advice-bearing
  `tt-perf-report` tables; `_no_advice.csv` files are machine-friendly copies.
- `tracy/{sliding,full}/{prefill,decode}_summary.csv` and `.png`: grouped
  device-time summaries.
- `tracy/decode_roofline.csv`: physical bytes/token derivation and theoretical,
  device, and wall comparison.
- `tracy/capacity2/{sliding,full}/`: raw BFP4/LoFi DRAM15 captures plus
  advice/no-advice prefill/decode tables and summary CSV/PNG files;
  `tracy/capacity2/decode_roofline.csv` is the two-user same-run accounting.
- `evidence/logs/*.log.gz`: exact safe-run transcripts for accepted, rejected,
  correctness, stress, watcher, profiler, AutoFix, inventory, and health runs.
  The 32 projection-remediation artifacts are enumerated exactly in
  `work_log.md`.
- `AUTODEBUG*.md`: fresh-context root-cause reports used by the AutoFix loop.
- `STAGE_REVIEW.md`: final independent `clean-pass` verdict and scope audited.
- `artifacts.sha256`: checksums for the retained evidence and profiler files.

Python and documentation are the only changed file types, so the repository's
`AGENTS.md` does not require a C++ build. Formatting and focused test results
are recorded in `work_log.md`.

## Final-topology stage-review AutoFix

The final review repair rechecked the cumulative topology rather than inferring
across earlier single-variable sweeps. DRAM10 was 0.484% faster by median over
the model's equal 18-sliding/18-full capacity-2 layer mix, but failed the valid
logical-batch-1 sliding decode contract when the decoder was configured for
capacity 2 (0.989158 PCC versus the unchanged 0.99 bar). DRAM15 passed at
0.990655, so configured capacities 2--32 remain DRAM15. This pass also repaired
runtime KV-update sharding so explicit logical batch, rather than configured
capacity, controls the decode K/V input shard grid.

Under DRAM15, the cumulative 64/32/16/8 output sweep reconfirmed final32: every
distinct alternative was slower for both layer kinds. Packed QKV also earned
retention under identical final32 topology. The corrected DRAM-sharded
separate-Q/K/V control is legal but its measured exact-attention shell is
13.70--15.48% slower; the anchored whole-trace estimate is 3.526--6.377%
slower. All A/B rows, bounded heavy-harness resource failures, and corrected
configured-32/logical-1 semantic evidence are recorded in
`AUTOFIX_stage_review_final_topology.md`.

The automatic performance harness now executes and signposts configured/logical
batch 2 while asserting instantiated BFP4/LoFi, DRAM15, and final32 objects. Its
uninstrumented 1000-replay qualification passed both kinds. Final logical batch
32, integrated, focused watcher, and post-health gates also pass. A broad
watcher diagnostic exceeded watcher's reduced kernel-config buffer only at the
advertised-context SDPA case; it is retained separately from the clean focused
watcher run.
