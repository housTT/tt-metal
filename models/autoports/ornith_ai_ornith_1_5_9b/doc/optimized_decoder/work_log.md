# Optimized decoder work log

Stage started from local fused checkpoint bc8f514f30 (runtime checkpoint 0e0fcfc6f3).
Scope: optimized_decoder.py, stage tests and docs only. No later stages.
Hardware: four Blackhole chips on P300c boards; measured mesh is one chip.
Initial `timeout 60 tt-smi -ls --local` succeeds with all four chips.
Hardware commands are serialized. Watcher and profiler runs remain separate.

## Initial operation-topology audit

Derived from fused runtime and `doc/fused_decoder/tracy/*/decode_perf_report.txt`; the previous stage suppressed advice, so this stage regenerated advice-enabled baseline reports under `tracy/baseline/` before geometry tuning.
Times below are approximate per replay from the existing four-replay captures.

| Current path | Cost / constraints | Candidate and planned action |
| --- | --- | --- |
| Two residual RMSNorms in DRAM, one core each | 37–38 us each, BF16 | Width-sharded L1 residual and explicit norm configs; carry layout through residual adds |
| Linear packed QKV/A/B, separate Z | 169 + 83 us, BF16/HiFi4; Z SiLU epilogue | Per-group BFP8/BFP4 fidelity trials; compare packing Z with complete split/activation costs |
| Full packed Q/K/V/gate | BF16/HiFi4, width10240 | Preserve packing; test reduced precision, DRAM-sharded geometry and legal separate control |
| Dense MLP separate gate/up + fused multiply-SiLU + down | 251 + 254 + 12 + 237 us; BF16/HiFi4 | Prioritize BFP4/LoFi gate/up and down, cross working shard geometries, compare packed versus tuned separate |
| Linear/full output projection | Large DRAM-bound weights | Independent precision/fidelity, DRAM sharding and geometry/readers |
| Full Q/K norm DRAM handoff, native partial RoPE, disjoint K/V shards | Cache ownership/head layout must remain valid | Sharded norm adaptation, explicit SDPA configurations, BFP8 cache with correct fill/update dtypes |
| Linear FIR, FP32 state decay/read/outer/write/read | Three state matmuls; final read ~61 us on four cores | Tune exact batched matrix geometry and movement while preserving FP32 state and round boundaries |
| Prefill packed projections + Conv1d + chunk delta / paged SDPA | T2048 baseline; arbitrary logical lengths internally padded | Explicit large 2D programs, composite-op/data-movement advice after dominant decode search |
| Collectives / experts / LM head | Absent: single-chip dense decoder | Not applicable; do not implement later-stage paths |

## Preserved starting contracts

Both layer kinds (real layers0/3), BF16 residuals/norms, FP32 recurrent state,
BF16 cache with64-token pages and page-table width rounded to32 entries,
logical batches1–32, native context262144, arbitrary valid logical lengths.
Fused paired B1/T2048/position2048 baseline: linear26.180386ms prefill,
1.462064ms trace; full23.498724ms prefill,1.263523ms trace. These historical
numbers guide the audit; new before/after comparisons will use the same harness.
No optimization acceptance or checklist closure is claimed yet.

## Initial precision screens

`record_run.py` records command argv, environment, source snapshots/hashes and logs.
The first test collection failed before opening a device because UID1002 has no
passwd entry and torch tried to derive a cache path. Reusing the preceding stage's
explicit `TORCHINDUCTOR_CACHE_DIR` fixes collection; no dependencies were installed.
`bfp8_screen_v2`:6 passed. These use real weights with synthetic scaled activations
and are screening only. `record_activations.py` records checkpoint embeddings for
natural/code/math token sequences and pinned HF layers0–2; the manifest identifies
actual inputs to layers0 and3. Tensor files are reproducible, local and excluded
from git; source/manifest are retained.

`bfp8_recorded`, `bfp4_gate_recorded`, `bfp4_mlp_recorded`: both real-input paired
checks pass. BFP8/HiFi2 trace medians1.138141/0.920619ms (linear/full), versus
paired fused1.462444/1.263416ms. BFP4 gate/up plus LoFi MLP:1.056295/0.839426ms;
minimum real HF PCC0.998882. BFP4 down also passes; gate/up/down geometry search
therefore includes the all-BFP4/LoFi candidate. No synthetic precision veto.
Raw KV/recurrent-state PCC is diagnostic for precision changes; cache-consuming
outputs and exact own-eager/trace replay are acceptance checks.

## Sharded MLP adaptation and trace investigation

`mlp_dram_c8_b16` and `_v2` fail on Python API differences (Shape slices and
CoreRange coordinates); both fixed using explicit list conversion and legal
grid dimensions. `_v3` reaches the BFP4/LoFi matmul and records requested L1
1602560bytes versus1572864bytes at gate/up Kblock16. Kblock8 runs but fails
exact eager/trace replay; this is unresolved work, not a rejection of sharding.
Fresh AutoFix investigator `/root/debug_sharded_trace` is inspecting source
and provenance before focused experiments. No hardware hang/reset occurred.

`bfp4_attention_recorded` tests BFP4/LoFi on the existing packed attention
family with BFP4 MLP. Linear layer prefill HF PCC0.993531 fails0.995, while
decode0.995758 and repeated cache-consuming output0.995474 pass. This is a
real-input output loss (not merely cache PCC); separate projection/phase
policies remain to investigate. Full attention was not executed after the
linear failure and remains pending.

## Current profiler-guided priorities

`profile_bfp4_full_decode_v2` uses recorded layer3 activations, four warmed
traced replays, position128, logical batch1. The preceding unsuffixed command
failed collection because Tracy flattened a spaced pytest selector. Retrying
with the single selector `decode-full_attention` passes and produces complete
ops CSV. `render_perf.py full_attention decode --capture bfp4_decode_v2
--iterations 4 --label bfp4_interleaved` preserves advice and CSV/table output.

Runtime rows prove BF16 activation × BFP4 weight / LoFi for packed QKVG,
output, both gate/up matmuls and down. Approximate row times are103,75,102,102,
217us respectively; norms37/38us and SDPA18us. Down remains especially slow
and all projection geometry work remains open. Raw K/V PCC0.9910/0.9872 in
the corresponding whole-layer comparison is diagnostic: real follow-on layer
output PCC0.997533 passes, so cache PCC does not veto that attention policy.

Tool advice limitations identified: the original fused report renderer used
`--no-advice`; this stage regenerated advice-enabled baseline tables. Some
FP32 state rows receive generic advice that HiFi2 suffices for BFP8 even though
the actual row is FP32. Reduced-weight SLOW rows with block2 can receive only
“look good” advice, omitting DRAM-sharding/geometry work. We follow the actual
row topology/bandwidth plus the optimize checklist, and retain these as
potential tt-perf-report improvements rather than treating them as closure.

`geometry_gate_bfp4`:45 exact real-input micro candidates,40 pass and5
record L1 capacity/overlap exceptions. All successful rows have exact eager/
trace equality. Fastest measured c64/block2/readers3:73.578924us including
boundary conversion; c32/block2/readers3:76.208016us. Wider working grids
c8/block16 and c4/block32 are included; failures have requested/available bytes
in JSON. These are micro results only; full-layer integration is pending.

## AutoFix verified residual-add cause

Source report: `AUTODEBUG_sharded_trace.md`. Fresh verifier
`/root/fix_sharded_trace` separately tests current control, residual-only,
DRAM-MLP-only and combined candidates. DRAM-MLP-only and isolated two-norm
controls pass; residual-only/combined fail restored eager equality. Boundary
probes show identical GDN outputs, residual-add operands and persistent states,
but differing post-attention residuals. Exact mixed BF16/FP32 width-sharded
addition reproduces nondeterminism; homogeneous and interleaved controls
localize it to that operation contract.

The narrow `_residual_add` adaptation promotes operands to FP32 only at a
mixed-dtype sharded addition, then rounds the sum once to the residual dtype.
This preserves the FP32 GDN update through addition. Restored eager outputs,
four immediate replays and post-224-replay state/output comparisons now pass
for both layer kinds. `autofix_original_fixed.log` repeats the uninstrumented
original pair: linear0.948965ms and full0.716283ms traced decode, HF output
PCCs above0.9984. Focused Watcher evidence and final AutoFix report follow.

A precision-materialization discrepancy is still being investigated: the
interleaved setup uses device typecast whereas DRAM weights are packed from
original host weights. The latter's better PCC must be isolated before the
former can reject an attention precision policy. This is ongoing work.

## Host packing and full geometry integration

`host_bfp4_all_recorded` isolates original-host packing from device-side
BF16-to-BFP4 conversion. All attention and MLP weights use BFP4/LoFi. Linear
HF prefill/decode/stress PCC: 0.999197/0.999488/0.999331; full attention:
0.998764/0.999061. Thus the earlier device-conversion failure does not reject
BFP4 attention. Canonical materialization will pack original host weights.

`geometry_all_host_bfp4` sweeps every material role on both kinds with real
recorded inputs, all legal divisor blocks for 4/8/16/32/64 input cores and
one/two/three readers. `geometry_layer{0,3}.json` contains every measurement,
compute config, PCC, exact trace check, bandwidth calculation and L1 exception.
Different roles prefer different readers and grids. `best_geometry_config.json`
records the integrated micro winners (including gate/up64 cores with3 readers,
down64 cores/block6 with2 readers).

`all_dram_best_geometry_c64` validates the integrated configuration against HF
and exact restored trace. Warmed traced decode: linear0.628192ms versus fused
1.461628ms; full0.456406ms versus1.263757ms. Prefill10.833815/7.740169ms.
HF prefill/decode PCC linear0.999197/0.999378, full0.998764/0.999059;
linear32-step stress0.999126. This becomes the best correct comparison point.
Residual grids, fidelity controls and coherent topology comparisons remain
under active measurement; these are candidate numbers, not final defaults.

## Coherent topology, memory and precision controls

`precision_residual_plan.json` and each matching `logs/*.provenance.json`
record all controls. On the original integrated DRAM geometry, BF4 attention
HiFi2 decode0.646224/0.473753ms loses to LoFi0.628192/0.456406ms; BF4 MLP
HiFi2 loses at0.693540/0.522776ms. BF8/LoFi attention and MLP controls also
pass but are slower (0.710739/0.525628 and0.784626/0.614098ms).
Residual grids8/16/32/64 are compared under this same BF4/LoFi policy.

`topology_geometry_plan.json` sweeps packed gate/up, packed QKV/A/B/Z and
separate Q/K/V/gate using the same dtype/fidelity and real recorded activation
path. `topology_adapted_plan.json` integrates the best geometries. Packed MLP
0.632318/0.461445ms loses after split/activation costs; explicitly slicing into
the down-matmul input shards improves it to0.629385/0.458367ms, still behind
the separate path. Packed GDN0.629116ms is also behind separate Z with its
SiLU epilogue. Separate full-attention projections0.483225ms lose to packed
QKV/gate0.456406ms. Separate gate with matmul SiLU epilogue loses to binary
SiLU/multiply0.636748/0.465181ms. These are legal, tuned comparisons.

`movement_plan.json` initially exposed two exact API contracts: RMSNorm rejects
height sharding, and Addcmul requires matching input dtypes. Neither is a
rejection. Retrying Q/K norm via width shards passes; width8 saves about4us.
Retaining the original output dtype while trying BFP8 projection inputs passes
but loses (attention0.642580/0.468947ms, MLP0.636045/0.463212ms).
Tiled rotary indices pass but lose at0.460506ms. Shared MLP input resharding
with residual32 passes at0.625507/0.453133ms.

BFP8 KV passes real HF prefill/decode at0.998757 minimum PCC and improves
prefill7.1969ms and decode0.447721ms. SDPA grids and chunks are swept in
`sdpa_plan.json`; context contract and long/batch tests are pending final
integration, so this is not yet a completed capacity claim.

Explicit recurrent reads, L1 state and L1 intermediates pass exact restored
trace and32-step real-HF stress. `state_plan.json` measures block1/2/4,
subblocks1/2/4,32/64/110-core grids and HiFi4/HiFi2/LoFi. Current best:
grid8x4/block1/subblock4/HiFi4 at0.545592ms (HF minimum0.999197).

`extra_geometry_plan.json` includes block1 controls for all material roles and
legal12/24/48/96-core down-projection geometries. Both kinds prefer down48
cores/block8/readers2 at71.47us versus initial64/block6/readers2 at74.7us.
Nonrectangular96-core layouts are attempted, not dismissed by the rectangular
helper. `initial_best_geometry_config.json` preserves the initial configuration;
`best_geometry_config.json` now incorporates the48-core down candidate and
residual32 for subsequent integration. Full per-run resolved configs are in
provenance, including earlier runs before this update.

## Prefill program and combined-path candidates

Explicit 2D programs were swept across 8x8, 8x4, 4x8 and 11x10 grids,
K blocks 4/8/16/32, output blocks, subblocks, and optional padded output
widths. `prefill_l1_adapted_plan.json` records smaller output blocks after
large-block L1 failures; a first allocation error did not reject the family.
The 8x8 grid with Kblock8, outM8/outN32, subH2 reaches linear10.5886ms
and full7.4481ms. Output padding1024/2048 does not help linear
(10.7678/10.8002ms). The independent full-attention Kblock16/outN16
candidate passes at7.4127ms; Kblock32/outN8 still fails L1. These use the
then-current projection path; combined-path confirmation remains required.

`combined_decode_first` integrates lower output movement, width-sharded
head norms, BFP8 cache/SDPA8x8 Kchunk256, explicit recurrent reads with L1
state/intermediates, and a shared MLP input reshard. Real-input paired trace
medians are0.536840ms linear and0.427923ms full. `combined_output` further
uses BF16 GDN output projection, reaching0.523923ms linear; minimum HF
PCC0.999197. These are the strongest B1 candidates and final defaults must
preserve their benefit. Larger-batch resource verification is in AutoFix.

The native KDA Conv1d+SiLU candidate passes B1 real-input semantics and
slightly improves prefill to10.7618ms. Batch32 is being rechecked after
repairing an independent residual/state L1 capacity failure. Native padded
chunk-delta decode and chunk-delta compute/memory probes are prepared in
`post_repair_candidate_plan.json`; they are not yet accepted or rejected.
Source inspection finds the phased chunk-GDN factory fixes its own HiFi4
FP32 compute policy, so constructor intent alone cannot establish a fidelity
change. Runtime evidence will identify the actual kernels.

## Batch resource and attention handoff verification

Fresh `AUTODEBUG_batch_l1.md` separates two allocation causes: public
[B,1,4096] tile padding expands each user's residual row, and persistent
FP32 recurrent state competes with norm/matmul circular buffers. Compact
[1,B,4096] residual arithmetic preserves public shapes while using one tile
row through batch32. Setup selects persistent L1 state through300KiB/bank
(B16 on110 banks), and L1 large intermediates through224KiB/bank (B12).
Borrowed width-sharded public inputs are included in those bounds. Higher
batches retain FP32 state and use DRAM; no logical capability is reduced.

`autofix_batch_residual_all` passes64 exact component cases: every batch1–32
with DRAM and sharded operands. `autofix_batch_kda32` passes real-weight KDA
prefill/decode at PCC0.999025/0.998989. The repaired combined B1 control
`autofix_batch_b1_pair` reproduces linear0.536406ms/full0.427921ms with
passing HF/stress checks; it retains the original FP32 GDN projection output
so the resource repair does not depend on the optional BF16 output policy.

Expanded coverage found two separate full-attention issues. Prime batches
put grouped rotary data on one core; retaining joined/partial tensors can
exhaust that bank. Explicit temporary lifetimes and interleaving partials
before grouped concat pass17/19/23/29/31 HF checks. Separately, B12 Q uses
a6x2 rotary grid while SDPA reads first-B cores from its8x8 grid; exact
replay fails despite identical KV/input tensors. A fresh source report and
focused Q handoff controls are in progress. This finding remains required
work until exact restored replay and Watcher verification pass.

## Resumed composite and prefill closure

The previous turn ended after two completed, provenance-backed rows; no device
process remained. `post_repair_remaining_plan.json` resumes only unstarted rows.
`separate_gdn_best` passes at10.8676ms prefill/.631463ms decode and loses to
packed QKV/A/B with separate Z. `native_delta_padded_decode` uses a neutral
32-token padded native chunk to express one decode update; it passes all HF
and restored trace gates but loses at.682695ms versus.523923ms.

HiFi2 and LoFi constructor controls for phased chunk-GDN have identical PCC
and similar prefill10.779/10.798ms: source hardcodes the actual phase compute
policy, so this is not evidence of runtime LoFi. L1 output first requests
8388608 bytes with only31744bytes/bank free;512-token physical chunk adapts
but still hits a CB overlap (999424 allocation versus1115136 static end).
The minimum128-token physical chunk control is pending.

Recurrent rank-one broadcast multiplication passes real output checks but
loses at.575673ms. Direct TILE head split also passes but prefill slows to
11.9041ms; it does not remove layout cost effectively. Native flat Q/K
normalization instead passes real B1 andB32 (the prior fused rejection used
synthetic inputs), improves prefill to9.2137ms, and keeps decode.524301ms.

Integrated flat-Q/K with2D prefill8x8/Kblock8/outM8/outN32/subH2 reaches
8.87993ms linear and6.91004ms full; adding native KDA Conv1d+SiLU reaches
8.71741ms linear. HF minimum remains.9991286 linear and.9987051 full.
Prefill L1 input placement passes but loses10.7918/7.1369ms versus
10.4749/6.9792ms for the matching non-flat combined DRAM controls. Larger
outN48 fails L1; smaller outN32 is the proven adaptation. Full-only
Kblock16/outN16 reaches6.9173ms, effectively tied with common Kblock8.
Final configuration selection and actual default verification remain pending.

## Final runtime promotion and default verification

The final candidate combines native flat-Q/K GDN prefill, KDA Conv1d+SiLU
(channel chunk1024), 8x8 2D prefill matmuls (Kblock8, outM<=8, outN<=32,
subH<=2), separate gate/up in both phases, and the selected decode geometry.
Separate prefill MLP wins8.4043/6.3969ms against packed8.8799/6.9100ms; adding
KDA1024 reaches8.0842/6.3871ms. KDA512 gives8.1226/6.4087ms.

Promoted all compatible winners into `tt/optimized_decoder.py`, with no test
imports or environment dispatch in runtime. Original checkpoint tensors are
packed directly to BFP4 during construction; measured forward paths perform no
host weight conversion. Removed the now-unused packed gate/up allocation.
Historical experiments use `tests/optimization_baseline.py` and archived source
snapshots, keeping their original policies independent of final defaults.

`logs/final_default_pair_v1.log` and `final_default_measurements.json` reproduce
the final default with no policy/config/variant overrides. Linear prefill
26.1322→8.10375ms and traced decode1.46228→0.524046ms; full prefill
23.4900→6.40362ms and traced decode1.26308→0.427555ms. The selected candidate
decode was0.523917/0.427635ms: final differences are+0.025%/-0.019%, within
repeat noise (well below0.5%); no candidate performance is substituted for the
reported default. Final HF PCC prefill/decode is0.9991286/0.9994463 linear
and0.9987051/0.9990599 full. Linear32-step stress PCC is0.9991993. Restored
trace output is exactly equal to the same optimized eager result. State/cache
PCC against BF16 fused tensors remains diagnostic (minimum0.9944095), while
cache-consuming HF outputs satisfy the unchanged0.995 acceptance threshold.

The initial reader profiler test exits pytest successfully but Tracy processing
fails; source investigation found profiler DRAM overflow and separately logged
reader3 API errors. Neither the process exit nor those error rows count as
complete reader evidence. See `AUTODEBUG_reader_profiler.md` and subsequent
repair artifacts. Final contract/profile/review gates are still underway.

## Final continuation repair and initial independent review

The first final short suite exposed retained L1 outputs during full-attention
unaligned continuation, a concat ELF local-storage failure, and a subsequent
host wait on an unfilled LLRT binary-cache entry. Captured live triage before
terminating PID107747; bounded list/reset/list succeeded with all four chips
visible. Exact recovery logs are `recovery_final_continuation_*.log.gz`.
See `AUTOTRIAGE_final_continuation.md` and
`AUTOFIX_final_continuation.md` for controls, source contracts and artifacts.

Immediate DRAM collection of leading decode outputs fixes both observed
resource failures. Bounded concat alone still fails L1; spill alone passes.
Extra Python concat batching was rejected after257/1024-input DRAM controls
passed and source inspection found TTNN's existing47-input recursive bound.
Only public optimized prefill orchestration changes; ordinary decode retains
its sharded output. Fifteen selected contract/performance tests and five
separate Watcher10 tests passed, including borrowed-input ownership and
continuation to exact capacity. Their helper decode timings use position128,
not the headline2048-position comparison.

`STAGE_REVIEW_initial.md` independently verified145 paired rows,98 complete
log/source archives, activation hashes and geometry evidence. Verdict is
more-work-needed while final gates are active. Its additional tuning finding
requires packed decode MLP to be combined with final residual32/down48 and
output layouts. `final_combined_topology_plan.json` closes that comparison
under the actual final default, including reader1/2/3 and an adjacent separate
control. It also compares final packed-GDN/separate-attention controls and
separate-prefill MLP output block48 with adapted M8/M4/M2 controls.

## Final combined topology comparison

The independent review's packed-MLP concern is closed by actual-final-default
candidates in `final_combined_topology_plan.json`. With residual32/down48,
shared input, output retention, all selected composites and precision, packed
gate/up readers1/2/3 give linear decode0.603110/0.531472/0.538388ms and full
decode0.504940/0.434217/0.441350ms. Every candidate passes real HF and exact
restored replay, but all lose the separate default (~0.524/~0.428ms).

Final all-packed GDN gives0.535471ms and8.23478ms prefill, also losing the
selected QKV/A/B plus separate Z. Separate full-attention projections retain
their measured disadvantage under final norms/SDPA/output layout.

For separate prefill MLP, outM8/outN48 reproduces the expected L1 failure.
Adapted outM4/outN48 passes at8.10217/6.42680ms and leaves decode unchanged;
this is effectively tied linear and slightly slower full than8.10/6.40ms
standard blocks, so does not justify another role-specific configuration.
OutM2/outN48 passes but slows to8.76531/7.01938ms. The rejection includes
successful layout/program adaptations, not merely the first allocation error.

## Final cache precision and reader closure

All paged fill/update and SDPA APIs in this checkout explicitly accept BFP4
cache tensors. The actual final BFP4-cache candidate passes HF prefill/decode
PCC0.99781955/0.99860275, exact restored replay, and improves full-attention
prefill6.3858→6.2268ms and traced decode0.427753→0.424050ms. A same-cache
higher-precision attention control (BFP8/HiFi2 projection weights) also passes
0.998509/0.999250 but loses6.9396ms/0.496143ms. BFP4 KV is promoted for
final contract validation; capability documentation will reflect measured
native-context results and the smaller cache allocation.

`AUTOFIX_reader_profiler.md` and `reader_comparison_summary.md/.csv/.json`
close reader geometry and collection. All66 profiled and66 unprofiled
alternating cases pass, minimum PCC0.999771 and exact eager/trace equality.
Capacity1000 loses markers; capacity8000 fixes collection independently of
periodic drains, which also fix default-capacity collection. Narrow-output
reader3 is legal after common output width adaptation to per_core_N18, but
loses to reader2. Every selected per-role reader wins both actual matmul time
and unprofiled trace time. Production reader2 keeps its legal per_core_N16.
Raw joined ops, host metadata and device counters are compressed and linked
from the report; no failed row is silently omitted from a completed group.

## Cache acceptance closure and final SFPU control

Broader real-input validation rejected the provisional BFP4 KV selection above.
`AUTOFIX_cache4_batch.md`, `cache_precision_summary.json`, and
`test_cache_precision_regression.py` preserve the diagnosis and controls.
Real users 8 and 26 fail PCC 0.995 in batch 32 and as identical extracted batch-1
inputs. BFP8 passes every user; a BF16/HiFi4 attention control also passes with
BFP4 cache. Exact identity/permuted-page probes, independent cache updates,
and an attention oracle over actual dequantized cache contents rule out a
batch mapping, update-address, or trace-state bug. This is real-input numerical
sensitivity, so the global production KV policy is restored to BFP8.

K8/V4 passes accuracy after replacing the unsupported mixed-format fused update
with two independent updates. Its traced decode is 0.432902 ms versus the
correct K8/V8 control's 0.427814 ms (1.19% slower); prefill is 6.297976 versus
6.395061 ms. The primary decode target therefore retains K8/V8 without a
batch-specific fallback. BFP4 KV's isolated speed win does not satisfy the
unchanged real-weight accuracy bar. The diagnostic mixed pair's fused baseline
interceptor detail is recorded in the AutoFix report; this policy decision uses
only its unaffected optimized-policy timings.

The final Z SiLU approximation control also passes with identical real PCC,
but traces at 0.524314 ms versus exact-mode 0.524160 ms; prefill 8.101293 versus
8.108838 ms is within noise. Keep the existing exact SFPU mode. Evidence:
`final_sfpu_plan.json`, `logs/final_z_{approx,exact_control}.*`.

Production code is now frozen for `run_final_gates.py release_v2 short long
batch watcher watcher_batch pair`; final Watcher and profiler runs remain
separate. The fresh independent final reviewer is auditing code and historical
candidate evidence while these final gates execute.

## Final production contract validation

`final_release_v2_short`: 73 passed, 9 long tests deselected, 177.22 seconds.
`final_release_v2_long`: 9 passed, 73 short tests deselected, 445.49 seconds.
Both use real weights with no candidate configuration environment overrides.
Both kinds complete prefill at 262143 and 262144, and traced decode at262143
matches eager PCC1.0. At native context, changing chunk2048 to1024 yields tail
PCC1.0 for both kinds. Real-HF8001-token PCC is linear0.999115/0.999328 and
full0.998694/0.998385. The native full-cache decode oracle passes0.99821247.
The context contract now records BFP8 cache bytes, duplicated BFP4 projection
storage, sharded layouts and bounded L1 state without reducing capability.

Native full-attention chunk-size validation spent364.7seconds on its first
two-prefill comparison; the subsequent262143 and262144 capacity calls finish
normally. Host/inspector inspection showed ongoing work, not a deadlock.
These are correctness-run durations, not warmed performance measurements.
The two pytest warnings are SWIG type metadata deprecations; the subset-MMIO
message describes opening one chip on a multi-chip board, not missing devices.

Python pre-commit checks pass. The first all-artifact formatting pass fixed
trailing/terminal whitespace in generated report text and a triage summary;
compressed raw logs/ops remain unchanged. `render_perf.py` now applies the
same terminal-newline normalization when generating new reports. Final
all-artifact checks run again after final reporting.

`final_release_v2_batch`: all85 cases passed. Its66 exact trace/resource rows
cover12 linear and54 full-attention cases. Minimum per-user HF PCC is linear
0.9987220442 prefill /0.9984290488 decode, full0.9982672048 /0.9960662493.
Every restored eager/trace/post-stress output, persistent state and borrowed
input check is exact; full-attention batch1–32 is covered explicitly. The
context contract now links this final batch evidence.

Final Watcher gates pass:73 correctness cases in234.86seconds, plus3 targeted
B12/prime-B31 cases in48.88seconds. `final_release_v2_pair` passes both kinds
with final production defaults. Linear fused→optimized prefill26.208289→
8.118329ms, traced decode1.462297→0.524132ms; full23.504981→6.396126ms and
1.263194→0.427323ms. Real-HF PCC remains linear0.999129/0.999446,
full0.998705/0.999060, with linear32-step stress0.999199. Final default timing
is now in `final_default_measurements.json`; all six gate commands/results
are in `final_gates_summary.json`. Final optimized profiling is running
separately, after every Watcher process closed.

## Final-profile large-prefill advice closure

The first final profile marks every decode matmul optimized, and actual rows
confirm BFP4/LoFi and all selected readers. Its prefill table still recommends
increasing64-core projection grids and tracing ~100us dispatch gaps. These
were treated as work. The earlier11x10 trial used the old packed-MLP path;
a coherent final separate-MLP comparison exposes a real win.

`final_prefill_grid_plan.json` compares final11x10,10x8,8x10 and8x8.
11x10/K8 reaches7.637058ms linear and6.030137ms full, with identical PCC
and unchanged decode. 10x8 gives7.984963/6.216814ms;8x10 gives8.011043/
6.105711ms. Its adapted M1 version loses13.625716/11.570226ms.

`final_prefill_grid_blocks_plan.json` selects11x10/K16/maxN32 at
7.400897/5.891442ms. K4 loses8.002006/6.474917ms; K16/maxN8 loses
7.624855/6.085542ms; adjacent K8 control gives7.601089/6.113326ms.
K32 initially requires2253824B of circular buffers against1572864B L1.
Smaller output blocks are tried in `final_prefill_grid_adapted_plan.json`:
K32/M1/maxN32 passes11.248151/9.507804ms, K32/M1/maxN8 passes11.599783/
9.950120ms, and K64/M1/maxN8 passes11.807900/10.186330ms. K128/maxN8
overlaps L1, so minimalM1/N1 is tried for both kinds separately and passes
20.511717/17.383599ms. Thus larger K is rejected by successful adapted
measurements, not the first L1 error.

Production now uses named large_prefill_grid=(11,10),
large_prefill_block_w=16 and large_prefill_min_seq=2048. Shorter physical
sequence tensors retain8x8/K8. This is an internal program choice, not a
logical sequence restriction. Decode and all short-prefix batch resource
paths are unchanged. The reviewer corrected an initial rerun-scope assumption:
start_pos0/length63 pads to128 and uses prefill; it is not leading decode.
The explicit>=2048 branch preserves that exact validated configuration.

`release_v3 short long watcher pair prefill_trace watcher_prefill_trace`
refreshes every affected gate, including native2048-vs1024 chunk handoff.
The85-case resource matrix and3 targeted Watcher batch tests from release_v2
remain applicable because their physical prefill sequences are below2048
and decode is unchanged. Fixed-shape prefill tracing is exercised separately
to measure and validate the dispatch-gap advice. Final MLP output-width35
versus7 blocking also receives a role-specific legal adaptation control.

## Final register-subblock advice closure

The11x10 profile exposes MLP1x1 subblocks because per-coreM7/N35 selects
output block7x7 while the old subblock cap2x4 admits only1x1. This is a
new actionable program consequence, not a reason to accept the SLOW label.
`final_prefill_subblock_plan.json` measures both legal orientations under
the actual final BFP4/LoFi/K16 policy. MLP-only7x1 passes7.137765/5.555920ms,
MLP-only1x7 passes7.039277/5.415721ms; all-role7x1 gives7.316848/5.605795ms,
and all-role1x7 wins6.998278/5.400382ms. All HF PCC values are unchanged.

The cap8 control is also legal: it only changes linear QKV/AB blockN24 from
subblockwidth6 to8; other roles keepwidth6/7. It passes at7.023706ms,
within noise of cap7 and with no measured advantage. Select the lower
median cap7. `large_prefill_subblock=(1,7)` is an explicit upper-bound
policy: actual subblocks divide each output block, so projections use1x6
and MLP gate/up uses1x7. Short-prefill2x4 bounds and all decode programs
remain unchanged.

The MLP full-width35 output-block control is separately closed:
M7/N35 exceeds L1; legalM1/N35 passes but loses8.829108/7.096671ms.
The intermediate v3 prefill-trace control passes exact state/output and
refreshed real inputs, including Watcher. It measures26.134505→26.083287ms
fused versus7.370257→7.356451ms optimized linear, and23.498769→23.486145ms
fused versus5.908189→5.848345ms optimized full. The final subblock path gets
its own refreshed trace evidence and pair; those numbers replace the headline.

`release_v4 short long watcher pair prefill_trace watcher_prefill_trace`
runs against the frozen subblock promotion. The fresh reviewer independently
verified that the runtime diff only changes the>=2048 subblock divisor choice;
the previously complete short-prefix batch matrix and targeted batch Watcher
remain applicable. The ledger now contains206 paired rows before the v4 pair.


## Frozen v4 production verification

Runtime SHA256 `01a3e3d084f6ca039ab78cc9545607f509afd4aff6852d47cb4f9ca1cc7f1608`
passes refreshed short73 (171.39s), long9 (113.81s), Watcher73 (187.66s),
pair2 (13.36s), prefill-trace2 (13.68s) and Watcher-prefill-trace2 (15.41s).
Together with unchanged-path batch85 and Watcher-batch3, the final gate index
contains249 passing cases. `final_gates_summary.json` links each exact command,
environment, source archive and compressed log. Native chunk/HF8001/cache-oracle
PCC is unchanged from v3. Watcher records contain no device assertions.

The final production-default pair measures linear prefill26.258161→6.983336ms
and traced decode1.461685→0.524047ms; full prefill23.509834→5.444302ms
and decode1.263289→0.427504ms. HF prefill/decode remains linear0.998982/
0.999388 and full0.998614/0.999037; linear32-step stress0.999229. These
final numbers replace intermediate controls in the README. The regenerated
candidate ledger contains208 paired rows, including final resolved configs.

The v4 fixed-shape prefill trace control measures fused26.197454→26.151917ms
versus optimized7.028883→6.888515ms linear, and fused23.486760→23.456102ms
versus optimized5.455464→5.409857ms full. Thus the final control saves140us/
46us. The existing prefill path captures directly and passes exact output,
state, input ownership and refreshed real-input replay checks; its separate
Watcher run passes both kinds. No native-size prefill-trace allocation claim
is inferred from this2048-token control.


## Final profiler and accounting closure

`run_profiles.py optimized_release_v4` completes all four captures with rc0,
separately from Watcher. `tracy/{linear_attention,full_attention}/
optimized_release_v4/` contains both advice-enabled tables/CSV, immutable
compressed source ops and decode_accounting.json. All actual projection
rows carry BFP4/LoFi; decoder readers match the selected geometry. Prefill
reports show110-core/K16 and1x6/1x7 subblocks. Large-grid and small-subblock
advice is now resolved by final-policy measured winners and legal adaptations.
The remaining L1/fidelity hints have successful/slower controls; dispatch
advice has final prefill capture plus existing nonblocking decode windows.

The final profiles reconcile linear bandwidth floor0.240768ms, kernel sum
0.499069ms, device FW span0.544265ms and host0.562476ms; full floor0.241544ms,
kernels0.411805ms, FW0.457337ms and host0.471041ms. Host-minus-FW is18.211us/
13.704us. These are same-run profiler measurements, not values substituted
into the unprofiled headline. Physical bytes, clock calibration and lower-bound
limitations are explicit in the accounting JSON and README tables. The
before/after HF table uses the corresponding real measured profile outputs;
linear four-step decode is0.999972699→0.999281869 and full0.999987019→
0.999036870, both above0.995. This closes the accounting/precision evidence
without interpreting the renderer's known derived-core/FLOPs anomaly as
hardware utilization.


The task-owned Tracy WASM server PID104909 on port18940 was stopped after
all captures (`final_profile_server_cleanup`). Final `tt-smi -ls --local`
returns0 and lists all four Blackhole devices (`final_device_health_list`).
An initial `tt-smi -l` invocation returned1 because it selected the interactive
UI without a TTY; the corrected list command succeeds. This was a CLI-mode
error, not a device failure, and required no reset. No hardware job remains.


`final_host_checks_v4` runs repository pre-commit hooks over every stage-owned
untracked file plus the tracked context contract and returns0. Formatting,
Python lint, YAML and large-file checks pass. No C++/CMake source changes were
made, so no build is required. Final runtime remains SHA256
`01a3e3d084f6ca039ab78cc9545607f509afd4aff6852d47cb4f9ca1cc7f1608`.


## Independent stage review

The fresh xhigh reviewer `/root/final_stage_review` returns **clean-pass**
with no required work in [STAGE_REVIEW.md](STAGE_REVIEW.md). The report
independently checks the original goal and skill checklist, frozen source,
249 acceptance cases,208 paired candidate rows, real activation provenance,
raw profiler programs/accounting, context capacity and the anomaly repairs.
The final-ready verdict applies to runtime SHA256
`01a3e3d084f6ca039ab78cc9545607f509afd4aff6852d47cb4f9ca1cc7f1608`.
All review findings have been fixed or controlled and rereviewed; no decoder
optimization task is deferred. Only stage-owned local checkpoint creation
and SHA bookkeeping follow; no push is authorized or performed.
