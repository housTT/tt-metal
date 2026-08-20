# Qwen3.6-27B optimized multichip decoder

## Result

This stage optimizes the existing Qwen/Qwen3.6-27B decoder on the target
`1x4` Blackhole P300c tensor-parallel mesh. It does not add a single-chip,
replicated-model, full-model, or vLLM path. Qwen3.6-27B is dense, so no MoE
expert path applies.

The final default keeps the accepted replicated BF16/TILE/DRAM residual at
every decoder-layer boundary and replaces decode's composite `ttnn.all_reduce`
with mesh-scoped, preallocated `ttnn.experimental.all_reduce_async`. Three
stable L1 slots and semaphores are shared by all layers on a mesh. Decode uses
two ring links; variable-length prefill uses one. Full-attention gate/up weights
move from BFP8 to BFP4; its down projection remains BFP8. Linear-attention MLP
weights remain BFP4. Linear-attention MLP collectives use a BFP8 payload; all
attention collectives and the full-attention MLP collective remain BF16. The
replicated layer output is restored to BF16. All selected decode projection and
MLP kernels use LoFi.

The final runtime has two in-layer row-parallel collective boundaries and no
gather, reshard, or all-reduce between decoder layers. A following layer must
consume and produce replicated `[B,1,S,5120]` BF16/TILE/DRAM residuals. This is
the contract full-model bringup must preserve.

## Final performance and correctness

Times are medians of three fresh final-default runs with real checkpoint
weights. Prefill is warmed and decode is warmed trace replay. The initial-stage
baseline and the same-session old-policy prefill control are both shown because
host-side prefill gaps varied over the long hardware session.

| Layer kind | Initial prefill (us) | Old-policy prefill control (us) | Final prefill (us) | Initial traced decode (us) | Final traced decode (us) | Decode improvement |
|---|---:|---:|---:|---:|---:|---:|
| Linear attention | 6535.723 | 6485.116 | 6680.525 | 774.659 | 718.857 | 7.20% |
| Full attention | 1778.039 | 2012.143 | 2008.705 | 553.373 | 476.422 | 13.91% |

The final persistent CCL itself improves the exact current-source,
same-precision non-persistent control from 765.223 to 721.692 us for linear
decode (5.69%) and from 524.213 to 476.388 us for full decode (9.12%). Minor
differences from the three-run medians are run-to-run noise. The final linear
prefill is 3.01% slower than the contemporaneous old-policy control, while full
prefill is 0.17% faster; prefill did not provide the decode win and is reported
without hiding that regression. The authoritative no-override files are
`autofix/fused_persistent/authoritative_default_run{1,2,3}.log`. Each records
the command environment, selected branch, links, payload and persistent-buffer
dtypes. Older `final/final_perf_run_*` and `final/selected_perf_run_*` files
predate the final mixed policy and are retained only as candidate provenance.

| Final correctness gate | Result |
|---|---|
| Linear logical length 65 | prefill PCC 0.997960; decode 0.997461; traced decode 0.998331; repeat 1.0 |
| Full logical length 33 | prefill PCC 0.9966856504; decode 0.9974546379 |
| Full paged traced decode | PCC 0.9954127239; repeat 1.0 |
| Forced chunked prefill | PCC 0.9965537 |
| Batch 32 | both layer kinds pass; linear per-user decode PCC 0.998508--0.998741, full 0.996858--0.997723; replicated boundaries 1.0 |
| Advertised context | 262144-token prefill passes for both kinds; full position 262143 traced decode PCC 0.9994974156, repeat 1.0; final mixed-default linear prefill plus final-position trace passes in 692.46 s with exact CCL provenance |
| Broad acceptance | 8 passed, 4 intentional opt-in topology/perf tests skipped, 14 deselected; separate fallback/batch selection 3 passed |

The opt-in same-mesh sequential-layer test was run separately. It proves that
linear and full layers reuse the identical three buffers and pass correctness.
No public input alignment restriction was introduced: logical lengths 65 and
33, forced chunking, page-table/cache updates, output slicing, and native
262144-token context all remain model-owned.
The authoritative final linear context artifact is
`final/context/linear_native_mixed_default.log`; the older `linear_native.log`
predates the mixed CCL selection and remains only as historical provenance.

## Operation-topology audit

| Material topology | Starting sequence / cost | Candidate family and contract | Dtype / layout constraints | Action and evidence |
|---|---|---|---|---|
| Attention packed projections | One packed QKV/gate matmul consumes the same residual | Keep packed | BFP8 weights; local head outputs | Retained; splitting would add same-input matmuls and movement |
| Linear GDN projections | Packed QKV plus packed Z/beta/a consumers | Keep existing packed inputs and local head/state ownership | BFP8 weights; DRAM-sharded decode matmuls | Retained; prior lower-movement GDN adapter expanded the graph to 344 ops / 6482.619 us |
| MLP gate/up | Two same-input matmuls | Pack into one physical projection, then slice | Must preserve full-layer 4352 logical / 4608 padded rows | Correct but slower: linear 766.393 vs 765.026 us; full 525.240 vs 524.519 us; rejected |
| Row projection boundary | Local row matmul then composite all-reduce | Explicit persistent async all-reduce | BF16 except linear-MLP BFP8 payload; exact 8/16-core sharded inputs; stable L1 output | Selected; exact collective 2.34--2.52x faster and whole layers 5.69--9.12% faster |
| Collective link placement | One link for both phases | Two links for decode, one for prefill | Ring topology | Selected. Decode improved; three-run two-link prefill did not robustly improve both layer kinds |
| Residual movement | Replicated residual after both row boundaries | Reduce-scatter, fractured residual add, distributed norm, compatible next projection | 1280-wide device-local residual | Correct but slower end-to-end for both kinds; rejected without immediate restoration |
| Fused row matmul + CCL | Matmul then all-reduce | Fused matmul+reduce-scatter carried fractured through residual/norm and into fused gather+matmul consumer | MinimalMatmul rank-4 weights, persistent buffers/semaphores | Exact ops pass PCC; coherent layers slower (linear 960.261 vs 730.243 us, full 710.184 vs 485.937 us); rejected after adapted retries |
| Fused gather + matmul | Gathered normalized activation then column matmul | `all_gather_matmul_async` with fractured downstream contract | Rank-4 weights; persistent L1 or DRAM output | L1 retry proved static-CB overlap; DRAM/interleaved retries made no progress. `$autofix` and focused hang triage retained; rejected |
| Activation / CCL precision | BF16 activations and boundary payload | Explicit persistent BFP8 global and isolated attention/MLP partial activations/payloads | True BFP8 input/output/persistent buffers; restore BF16 residual after boundary | Exact and real-layer correctness pass; three interleaved 50-replay cycles select BFP8 only for the linear MLP boundary; all other boundaries retain BF16 |
| Projection precision/fidelity | BFP8 projections, automatic/LoFi decode kernels | BFP4 input/output and HiFi2 | Real weights and traced path | BFP4 input fails PCC; BFP4 output fails full-layer PCC; HiFi2 passes but is slower |
| MLP precision/fidelity | BFP4 linear MLP; BFP8 full MLP | Per-role BFP4 and HiFi2 | Real weights; full down projection isolated | Full gate/up BFP4 selected; full down BFP4 fails PCC 0.991911; HiFi2 is slower |
| Persistent resources | Composite CCL allocates/converts per call | One mesh-shared pool with three logical slots | 3.1640625 MiB/device, preallocated before first decode graph | Selected; stable identity, exact allocation, and sequential-layer trace replay pass |

Every selected row boundary uses `cluster_axis=1`, Ring topology, a sharded L1
input/output memory config, one prefill link and two decode links. The TTNN API
defaults are retained for chunks-per-sync, workers-per-link, and buffers-per-
channel because this `all_reduce_async` overload does not expose those knobs;
the output buffer and global semaphore are explicit and persistent, while no
separate intermediate-buffer argument exists. For the physical padded decode
payload, a four-device ring moves an expected 1.5 payloads/device: 1,966,080
bytes at BF16 and 1,044,480 bytes at BFP8. Thus only the linear MLP boundary
uses the lower-traffic figure; the cast back to BF16 occurs after the collective
and before residual addition.

The final topology does not reject lower movement by paying an immediate return
to the old contract. The fractured candidate carried the residual through
residual addition and distributed RMSNorm into the next projection. Fresh
whole-layer results were linear 762.198 us replicated versus 865.142 us
fractured and full 521.526 versus 627.390 us. The fused candidate likewise
kept both row outputs fractured and adapted the next consumers. Distributed
norm and fused-op overhead outweighed the bytes saved.

## Candidate evidence

`candidates/index.csv` is the compact candidate ledger. Important conclusions:

- The original composite BFP8 CCL results (779.233/560.886 us globally) were
  not used for final selection after review exposed their topology confound.
  `$autofix` added true BFP8 input/output persistent buffers and reran a
  same-source family. Exact 8/16-core PCC is 0.9999418. Persistent BF16 measured
  721.502/476.934 us linear/full; attention BFP8 722.918/479.149, MLP BFP8
  718.964/479.494, and global BFP8 719.703/480.809. All real-weight correctness
  and trace gates pass. Because a common dtype policy is unnecessary, a second
  interleaved three-cycle family compared linear BF16 (median 721.843 us,
  range 0.385), linear-MLP-only BFP8 (719.125 us, range 0.639), and linear-global
  BFP8 (719.971 us, range 0.373). MLP-only BFP8 beat its matched BF16 control in
  every cycle by 2.351--3.277 us and is selected for linear layers; global BFP8
  is rejected and full-layer boundaries remain BF16.
- Two links improved decode but were noisy for prefill. The final default uses
  one prefill link and two decode links. Against the contemporaneous old-policy
  prefill medians, the authoritative mixed-policy median is 3.01% slower for
  linear and 0.17% faster for full; two-link prefill was not robust across kinds.
- Full gate/up BFP4 passed accepted real-weight PCC. Full down BFP4 failed
  non-aligned PCC at 0.991911 and was rejected.
- Full gate/up BFP4/LoFi geometry retained block 10; the selected full down row
  is BFP8/LoFi with block 9. Smaller gate/up blocks 5, 2, 1 measured whole-layer
  decode at 485.014, 513.238, and 609.578 us; down blocks 3 and 1 measured
  479.853 and 531.982 us.
- The fused family was retried with logical-row slicing, rank-4 zero-copy
  weights, L1 and DRAM outputs, and then `$autofix`. Exact fused RS PCC was
  about 0.999962 and exact fused gather-matmul PCC about 0.999849. The final
  rejection rests on coherent whole-layer regression plus recorded L1 and
  host-progress blockers, not a first API error.

## Precision-locked decode geometry

This stage inherits the completed multichip decoder's exact-shape geometry and
retunes every row whose precision changed. The prior linear sweep is retained
at `../multichip_decoder/autofix/bfp4_geometry/AUTOFIX.md` and `sweep.log`; the
current full sweep is under `candidates/full_mlp_blocks/`. All rows use real
TP-local dimensions, DRAM width-sharded weights, BF16 width-sharded L1
activations, LoFi, and accepted correctness.

| Layer / role | Shape MxKxN | Weight / fidelity | Grid and output shard | `in0_block_w` candidates | Selected / measured evidence |
|---|---|---|---|---|---|
| Linear gate/up | 32x5120x4352 | BFP4 / LoFi | 8 cores, `[32,544]` | 20, 10, 5, 4, 2, 1 | 10; exact-row 57.787 us vs 94.687/59.041/61.638/73.956/125.220 us |
| Linear down | 32x4352x5120 | BFP4 / LoFi | 8 cores, `[32,640]` | 17, 1 | 17; exact-row 60.470 vs 114.866 us |
| Linear padded alternative | 32x5120x4608 and 32x4608x5120 | BFP4 / LoFi | 16 cores | compatible padded programs | rejected; real-layer PCC passed, dominant rows regressed to 45.7/45.8/46.3 us from about 43 us |
| Full gate/up | 32x5120x4608 | BFP4 / LoFi | 16 cores, `[32,288]` | 10, 5, 2, 1 | 10; final profiler rows 46/46 us; smaller-block whole layers 485.014/513.238/609.578 us versus default about 476 us |
| Full down | 32x4608x5120 | BFP8 / LoFi | 16 cores, `[32,320]` | 9, 3, 1 | 9; final profiler row 56 us; smaller-block whole layers 479.853/531.982 us versus default about 476 us |
| Attention/linear output | 32x1536x5120 | BFP8 / LoFi | 16 cores, `[32,320]` | selected block 3 | final profiler row 22 us and 362 GB/s; marked DRAM-bound |
| Full packed QKV/gate | 32x5120x3584 | BFP8 / LoFi | 16 cores, `[32,224]` | selected block 10 | final profiler row 44 us and 422 GB/s; marked optimized |
| Linear packed input | 32x5120x4352 | BFP8 / LoFi | 8 cores, `[32,544]` | selected block 20 | final profiler row 52 us and 425 GB/s; marked DRAM-bound |

The linear exact sweep's eight legal block candidates and its only useful
16-core padding family exhaust the divisible TP-local geometries. Padding to
5120/32 cores adds 17.6% inert intermediate work; 6144/64 cores adds about 69%,
so both are dominated by the already slower 4608/16-core measurement. For the
full layer, local 4608 dimensions make the selected 16-core grid exact; this
stage explicitly swept the larger legal block values under the final per-role
dtype. The final profiler rows verify the claimed BFP4/BFP8 weights and LoFi
at runtime, rather than relying on constructor policy alone.

## Profiler and roofline accounting

Advice-enabled `tt-perf-report` text, CSV, generation logs, summaries, and raw
signposted Tracy CSVs are under `profiler/{linear,full}`. They identify the
device as Blackhole with 110 workers and use 512 GB/s DRAM bandwidth per chip,
2.048 TB/s aggregate.

The three required values below come from the same instrumented capture. The
report rounds device time and inter-op gaps to whole microseconds; the residual
is capture E2E minus those rounded values. It includes Tracy/device-profiler
instrumentation, Python signpost/harness work, trace replay/synchronization, and
rounding not attributed to a device op or inter-op gap by `tt-perf-report`.

| Layer kind | Required stored weight bytes/device/token | Aggregate roofline | Device time | Reported inter-op gaps | Other profiler/runtime residual | Same-run profiled E2E |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 69,632,000 plus recurrent-state traffic | at least 136.0 us | 543 us | 200 us | about 155.7 us | 898.716 us |
| Full | 79,462,400 plus small KV reads | at least 155.2 us | 401 us | 81 us | about 151.5 us | 633.489 us |

The separately measured, uninstrumented final-default medians are 718.857 us
linear and 476.422 us full. They are the headline latency results, not inputs to
the profiler-run reconciliation above. The raw profiler capture predates the
final linear-MLP BFP8 boundary and therefore shows BF16 for that one collective;
it otherwise profiles the selected math, layout, trace, and persistent-async
topology. The exact final mixed CCL runtime is established by the authoritative
default logs and the interleaved variance family. The difference between the
instrumented and uninstrumented E2E regimes is profiler overhead and run-regime
perturbation; device time from one regime is not subtracted from E2E in the other.

The full decode report has 50 device ops and 183 GB/s modeled throughput; the
linear report has 78 ops and 123 GB/s. The report's slow BFP4 rows and HiFi
advice were acted on: legal block geometry was swept under BFP4/LoFi, and
HiFi2 was correct but slower (linear 823.805 us, full 578.071 us). BFP8 down
and input rows are marked optimized. The recurrent GDN's small DRAM matmuls
retain the prior stage's measured geometry; batch-32 recurrent state is too
large for an L1-resident state contract. Decode dispatch gaps were reduced by
trace-safe persistent buffers. Prefill stays dynamic because the public path
owns arbitrary logical lengths and chunking; its host gaps are included in the
reported warmed end-to-end result.

## Context, memory, and runtime gates

The new pool contains two `[1,1,32,20480]` BF16/TILE/L1 buffers at 1,310,720
bytes each and one BFP8/TILE/L1 buffer at 696,320 bytes, for a fixed 3,317,760
bytes (3.1640625 MiB) per device and mesh. It is not allocated per layer and
fits inside the existing 4 GiB
trace/activation/CCL/fragmentation reserve. KV-cache dtype/layout remains BFP8
and head-sharded. The advertised 262144 context and batch 32 remain unchanged;
`doc/context_contract.json` records the contract and native-context evidence.

The source/runtime fallback audit found no Torch conversion, host layout
fallback, or hidden reshard in the multichip runtime. Watcher and profiler were
run separately. The closure watcher used the required `TT_METAL_WATCHER=10`,
passed the final mixed-default linear non-aligned and full traced tests (full
PCC 0.9954127239, repeat 1.0), produced 1092 clean lines, and detached all four devices without
error/assert/hang/stuck/mismatch signatures. The earlier level-1 watcher log is
retained only as provenance.

One adapted fused gather-matmul DRAM retry stopped making host progress. The
focused AutoTriage attempt could not attach because Inspector data was absent
and the process exited before capture. The failure is therefore classified as
a host-progress blocker, not a proven kernel deadlock. All four P300c devices
were visible afterward and a minimal 1x4 mesh open/close passed; no reset was
needed. Evidence is under `triage/fused_agmm_hang/`.

## Reproduction and artifacts

Run TT commands from `/tmp` with this checkout on `PYTHONPATH`; running from the
repository directory can select mismatched JIT artifacts. Representative
commands and every material artifact path are in `work_log.md`. The final
acceptance log is `final/acceptance_mixed_default.log`, the separate fallback
and batch-32 log is `final/fallback_batch32_mixed_default.log`, context logs are under
`final/context/`, watcher evidence is under `final/watcher/`, and the persistent
and fused investigations are under `autofix/fused_persistent/`.

## Limitations

The optimized path is validated for the exact 1x4 P300c ring and TP=4. It keeps
the replicated inter-layer residual because every coherent fractured/fused
alternative measured slower on this hardware/runtime. The retained environment
overrides exist for reproducibility and candidate replay; the no-override
default is the final measured path. This stage intentionally stops at the
decoder and does not construct a full model, generator, or vLLM integration.
The process-lifetime pool strongly retains its mesh object, manager, semaphores,
and tensors because this TTNN MeshDevice has no documented weak-reference
lifecycle. Same-mesh sequential reuse is validated; applications that close and
reopen meshes in one Python process should use a fresh process until TTNN exposes
an explicit pool teardown contract.

## Anomaly ledger

| Observed anomaly | Evidence and affected path | Control / likely subsystem | Investigation | Resolution |
|---|---|---|---|---|
| Original BFP8 CCL appeared much slower | Composite BFP8 logs; row-boundary payload | Persistent BF16; experiment topology | Added true BFP8 persistent buffers, exact probes, isolated/global correctness/perf, then an interleaved layer-specific variance family | Fixed experiment; linear-MLP BFP8 selected, other boundaries BF16 |
| Profiler arithmetic did not match headline E2E | Raw reports and capture logs | Instrumented versus uninstrumented run regimes | Reconciled roofline/device/gaps/residual/E2E within capture and separated headline medians | Fixed reporting |
| Watcher used level 1 | Earlier watcher command | Skill requires level 10 | Repeated risk-focused suite at 10 separately from profiler | Fixed; 2 passed and clean dump |
| Batch-32 PCC bands differ by layer kind | Acceptance lines for 64 users | Linear versus full attention | Reported ranges independently | Fixed reporting; both accepted |
| Fused AGMM DRAM retry stopped making progress | `triage/fused_agmm_hang/` | Selected persistent replicated control | AutoFix plus AutoTriage, rank/layout/buffer retries, health postflight | Controlled rejected family; no unsupported deadlock claim |
