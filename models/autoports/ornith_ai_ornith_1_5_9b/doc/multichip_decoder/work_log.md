# Multichip decoder work log

Stage in progress. No clean-pass, final performance, or completed capability claim.
Branch hous/ornith-1.5-9b; starting HEAD483920f536; optimized checkpointd085eb6d1a.
Only multichip decoder, tests and documentation are stage-owned. No full-model
or vLLM implementation is in scope. Local commits follow independent clean-pass.

## Planning and initial hardware

Read model-local AGENTS, multichip/tt-device-usage, LLM report3.3, optimized,
fused and functional decoder implementation/contracts and old35B head/cache/
RoPE/reference tests. Hardware listing `timeout60 tt-smi -ls --local` succeeds
with four P300c Blackhole chips. `logs/mesh_smoke.log` proves FABRIC_1D_RING
mesh1x4 open/close, physical/logical degree2, order[1,0,3,2], compute11x10,
DRAM8x1. Unknown B850M-C motherboard warning uses PCI bus as tray ID; actual
ring topology is discovered and mesh closes successfully.

`mesh_plan.md` precedes implementation. Native262144 stays the target.
`memory_capacity_plan.json` and context_contract's new planned section estimate
6296862720 bytes/device including full stack projections, BFP8 KV, conservative
BF16 embeddings/head, unshared RoPE and1GiB reserve. This is not a measured
allocation or completed multichip context validation.

## Initial probe

`record_run.py` archives exact command, environment, source and logs.
`initial_linear` failed before device open: torch compiler cache import error.
Using the preceding stage's persistent TORCHINDUCTOR_CACHE_DIR fixes import;
no dependency installation or runtime-model change was needed.

`initial_linear_v2` ran optimized baseline replicated across4 chips: prefill
passes, decode errors at get_worker_noc_hop_distance's unit-MeshDevice assert
in multi-reader DRAM matmul. This is not model PCC evidence. Mesh closed; reset
`timeout180 tt-smi -r` succeeds (`reset_initial_linear_v2.log`). AutoFix fresh
xhigh source diagnosis is delegated to reader_mesh_diagnosis with no hardware
access. Baseline comparison harness now opens a genuine1x1 mesh, closes it,
then opens1x4 for TP. No concurrent hardware lanes exist.

## Initial replicated control and sharded failure

`initial_linear_v3` and `initial_full` pass real-weight128-token comparisons
against sequential actual1x1 optimized runs. Each of four ranks matches:
linear0.9999763147/0.9999771174; full0.9999829421/0.9999720258
(prefill/decode). These controls use reader1 and packed local heads.

`replicated_linear_trace_v2` measures2048-token linear prefill6.968691ms
single-chip versus3.474822ms TP4; warmed traced decode0.523932ms versus
0.411376ms. Both exact restored eager/trace and repeat-eager checks pass on
all ranks. PCC is0.99996308 prefill/0.99998964 decode. This is initial
untuned replicated-residual evidence, not final optimization signoff.

`sharded_linear_v1` completes prefill but fails exact trace equality. The
follow-up `sharded_linear_diagnostic` originally saved tensor values to
trace_mismatch.pt. A later control overwrote that filename: the original tensor
is no longer available. The surviving artifact is explicitly named
`sharded_link1_gather_control_trace_mismatch.pt` and belongs to that later
control. AutoDebug recorded the original eager count of747 infinities before
the overwrite;
NaN PCC arises from those infinities, not merely a small trace mismatch.
See AUTODEBUG_sharded_trace.md. Fresh AutoFix hypothesis experiment agent
sharded_repair owns the exclusive hardware lane while parent works tests/docs.

Reset after diagnostic completed successfully, but unusually slowly. Parent
mistakenly started replicated_linear_trace_v1 before reset completion, noticed
at topology discovery and terminated only its child PID351561 with SIGTERM.
That run returned-15 and is invalid. Reset itself returns0; list returns0 with
all four devices; mesh_smoke_after_reset.log proves ring open/close. No locks
were cleared, no reboot occurred. This orchestration error is recorded and
subsequent device ownership is explicit; v2 above ran after successful smoke.

## Native prefill memory refinement

The initial1GiB activation reserve was too small for the public decoder's
full-length replicated input/output contract. At262144*4096*2 bytes, each
residual is2GiB. Input, accumulated chunk outputs and final concatenation
can overlap (6GiB). The updated plan reserves8GiB including2GiB for trace,
chunk temporaries, small constants and allocator overhead. This is a
conservative full-stack estimate; capability is unchanged and runtime
native-context validation remains required.

### Broader contracts and first profiler capture (2026-09-05 05:06 UTC)

`contracts_initial_short` passed 53 cases before batch32 full-attention traced
HF PCC failed (one user0.99480519<0.995). The completed optimized stage passes
that identical input. Fresh AutoDebug and AutoFix reports are
`AUTODEBUG_batch32.md` / `AUTOFIX_batch32.md`. Do not classify this as a waived
precision threshold or a trace-only failure. The investigation proves eager,
restored eager and replay exact; rank-local snapshots remain immutable.
Changing only QKVG K-block4 to2 restores exact packed projection agreement with
the genuine single-chip baseline. The original unchanged narrow case and scoped
worker watcher rerun pass. Final all-contract rerun remains pending.

After the initial failure, serialized `reset_after_contracts_initial`,
`list_after_contracts_initial`, `mesh_after_contracts_initial` all exited0.
No stale process killed, no locks removed, no second reset needed.

Initial `run_profiles.py initial` completed all four real-input captures with
HF PCC checks: linear/full attention × prefill/decode. Sources, commands and
compressed console logs are `logs/profile_initial_*`; human-readable tables,
CSVs and provenance are `tracy/<kind>/initial/`. These are pre-tuning profiles.
The original merged decode report includes ~307ms before the first measured op:
the per-device gap begins before PERF_DECODE and includes profiler draining.
Raw kernel durations are sane. `render_perf.py` now also emits each device's
window/report and `*_rank_accounting.json`: it removes only that first
pre-window gap, preserves all kernel durations and intra-window gaps, and never
adds simultaneous device times. The raw merged report and raw CSV remain intact.
Linear decode per-rank kernels ~366us, intra-window gaps~67us, matmuls~181–183us,
collectives~43–46us, explicit movement~49–50us. This agrees with ~0.411ms
uninstrumented timing within profiler overhead. Geometry and topology search
remain necessary; percentage columns from the uncorrected merged report are not
accepted performance evidence.

Added decoder-only stack composition and native-capacity reservation probes;
syntax checked, hardware validation still pending. They do not implement a full
model, embeddings, logits, generation or serving.

### Whole-layer topology and geometry comparisons

All `topology_*`, `tuned_*`, and `precision_*` paired probes run genuine1x1
optimized baseline then1x4 TP, with real checkpoint weights and recorded HF
inputs. `candidate_measurements.csv/.md` are regenerated by
`analyze_experiments.py` from compressed logs and provenance; they include
rejected historical candidates, not just accepted timings.

`geometry_layer0/3` completed220 exact-replay local geometry candidates using
actual decoder-produced rank-local inputs; `geometry_search.md` records every
legal divisor/core pair. Matmul reader1 is fixed by the independently proven
reader2/3 mesh API blocker. More activation shard cores do not increase this
op's fixed8 compute workers. Wider interleaved-weight1D matmuls were also tried
as a different family; default64-core whole-layer timing did not beat the
DRAM-sharded control.

The first combined geometry set (`contracts_geometry_v1`) again failed the
same batch32 user after53 passes. Followup AutoFix isolated MLP gate block16;
only gate block4 is retained, with all other faster geometry preserved.
Restoring output projection or down alone did not fix the gate. Gate block8
also fails. QKVG block8/16/32 controls fail the same per-user test, while
QKVG32cores/block2 remains bitwise equal to the original passing projection.
See appended `AUTOFIX_batch32.md` and `batch32_geometry_comparison.json`.
Ten nearby traced and batched checks pass; full final suite remains pending.
Recovery was serialized by the repair agent; exact names/exit statuses are
in that report and the `batch32_geometry_*` provenance.

All coherent topology families now have shape-faithful whole-layer controls:
replicated native all-reduce; hidden-sharded RS consumed directly by residual
and distributed norms; local-activation AG plus column-sharded WO/down;
fused AG-MM for those projections; fused MM-RS; explicitly persistent RS/AG
buffers consumed as borrowed outputs; and hidden norm AG fused with QKV/gate,
sharing the gathered input with Z/up. Packed local MLP gate/up is also measured.
No per-layer test-boundary gather is included in timed hidden-sharded forwards.
BFP8 CCL-only payload trials pass under both replicated and sharded contracts.

Fused AG-MM initially collided with L1 CBs at local-K-wide blocks. Reducing
its block to4 fixed it; both layer kinds then pass. `reset_after_fused_ag_mm`,
`list_after_fused_ag_mm`, `mesh_after_fused_ag_mm` all exit0. The later fused
norm AG candidate required SiLU in the sharded matmul program config rather
than its generic activation argument. After that API adaptation, both kinds
pass (`tuned_fused_norm_ag_mm_silu_layer*`). Recovery
`reset/list/mesh_after_fused_norm_ag` all exit0. No hangs, kills or lock cleanup
occurred in these two recoveries; the failures were host validation errors.

Current tuned native controls: linear decode0.3925ms vs0.5242ms single;
full decode0.3203ms vs0.4272ms single. Packed MLP~0.3940/0.3210ms,
persistent replicated~0.4099/0.3378ms, persistent sharded~0.5025/0.4335ms,
fused norm AG-MM~0.4691/0.4296ms. These are paired candidate measurements;
final numerical/context/watcher review remains required. BFP8 CCL adds small
conversion cost and has not improved whole-layer latency. Attention/MLP
activation and fidelity controls are running next.


## Final layout and precision controls

Attention/MLP BFP8 input conversions, BFP8 weights, and HiFi2 controls all
passed B1 real-weight checks but increased decode latency. The BFP8 attention
weight control first exceeded prefill L1 CB capacity at block16; block8 adapted
the candidate and both layer kinds/fidelities then ran successfully. Serialized
`reset_after_attention8`, `list_after_attention8`, `mesh_after_attention8` all
exit0. No hang or process kill occurred. Exact candidates are in
`candidate_measurements.csv` with source archives and commands.

The final layout sweep tested residual grids16/64, conv chunks256/1024,
SDPA32/110 cores and chunks128/512, wide interleaved projections32/110 cores,
and8192-byte fabric packets. All16 runs exit0. The32-core interleaved decode
candidate improves linear latency0.3925 ->0.3810ms and full0.3203 ->0.3184ms.
Packet8192 independently improves the DRAM-sharded control to0.3902/0.3176ms.
The combined32-core/8192-packet path is now undergoing the complete contract
suite. Role K blocks retain the measured QKVG2/gate4 numerical constraints.
The previous DRAM-sharded path remains selectable with `decode_grid=None` for
controls. Prefill is unchanged; prefill variation in decode-only controls is
measurement variation, not attributed to the decode change.


`contracts_wide32_short` passes89 tests (8 native/long deselected), including
the previous batch32 per-user regression. `final_stack` passes direct
linear/full/linear composition with zero boundary conversions, nonaligned131
prefill and8 changed-input/position trace steps. Every rank prefill PCC is
0.99996213; minimum8-step decode PCC0.99996541; restored eager/trace is exact.
The selected interleaved path now frees unused duplicate decode weights at
setup, reducing projection persistent memory from1961754624 to975568896 bytes
per device (all32layers). The capacity plan was updated before its reservation
run. The previous DRAM-sharded control is explicit `decode_grid=None`.

`final_long` passes native linear262143 +traced position262143 and native
linear chunk2048-vs1024 invariance PCC0.999972, then fails at the8001-token
linear prefill because output replicas differ. This is required work, not a
relaxed bitwise gate. Fresh AutoDebug is inspecting source while root performs
serialized reset/list/mesh recovery. The remaining long/cache/capacity commands
were not launched after the failure.


`final_native_cache` passes: HF PCC0.99809617 at position262143, exact eager
and trace, all output replicas equal, one distinct local BFP8 KV head per chip,
randomly permuted4096pages. Historical KV uses exactly representable binary
fractions; this is an independent decode oracle, not HF native prefill.

`final_capacity` reserved6385827840 DRAM bytes/device plus24 recurrent L1
states, then failed to return from native full-attention readback. Root captured
`triage/capacity.txt` with the full `tools/tt-triage.py --llm-output` set and
`triage/capacity_stable.txt` with callstacks/running-ops/fast-dispatch/ETH.
Both bounded120s captures exited0. Summary flags do not capture all raw
integrity warnings; AutoTriage reads the raw reports. Model workers were idle,
dispatch/prefetch awaited host commands. A read-only ptrace register/frame
capture of busy hostTID450521 identifies `completion_queue_wait_front` ->
`copy_completion_queue_data_into_user_space` -> mesh readback executor. The
ELF executable segment's vaddr-offset correction is recorded explicitly in
`triage/capacity_host_pc.json`. No debugger/dependency was installed.

After preserving evidence, root SIGTERM'd only testPID450445; recorded test
returncode-15. No locks were removed. `reset_after_capacity`,
`list_after_capacity`, `mesh_after_capacity` all exit0; all4 devices returned
and the ring opened/closed cleanly. Hardware lane then passed exclusively to
AutoFix `long_capacity_repair`. `AUTOTRIAGE_capacity.md` classifies the
read-completion stall and proposes focused aligned-read controls; it does not
claim a proven source fix. Current repairs must preserve the same checked
logical tokens and all-rank comparisons.

### AutoFix long-output/capacity controls (2026-09-05 06:12–06:23 UTC)

Read `AUTODEBUG_long_replica.md` and `AUTOTRIAGE_capacity.md`. After root's
recorded capacity reset/list/smoke, isolated8001 raw replicas and physical-chunk
clones are finite and bitwise identical; host physical assembly equals final
concat on all4 ranks. Exact original8-long suite passes297.41s. Original
capacity passes once with phase markers and twice after restoring its original
source; no aligned-envelope intervention was run. Worker watcher then passes
capacity+8001 both kinds (3 cases,187.43s, exit0; ETH disabled). Raw watcher
retained for final fixture only; console covers all3. Production SHA remains
7b63bbb22ee64f76025d8363af0c60248296925609fc199f620007d12d8c7c0c. No baseline/C++
or production repair; original anomaly recovered but source-level cause
unproven. Keep strict gates and failure-only unique raw capture. Full commands,
PCC, hypotheses, limitations and artifacts: `AUTOFIX_long_capacity.md`. Root
received the hardware lane after all processes closed at06:23:25 UTC.


## Interleaved projection selection

`geometry_interleaved_layer0/3` measures92 legal K-block configurations on the
selected8x4 interleaved-weight decoder, at BFP4/LoFi with identical recorded
local inputs and all-rank exact trace checks. Larger legal K divisors are
included through128 (96 for down,32 for row attention outputs). Final role
candidates: GDNpacked32,Z32,GDNout8,O8,gate8,up8,down6. QKVG's fastest16 is
numerically rejected;4/8 controls also fail the same batch32 user even after
MLP tuning. QKVG2 +gate8/up8/down6 passes minimum per-user HF PCC0.99519507;
gate4/up8/down6 also passes0.99533285 but is slower. The earlier DRAM-family
need for gate4 does not apply to the measured combined interleaved MLP.
Exact wide32_batch_* logs/provenance record all7 accuracy controls; diagnostic
exit0 means the comparison completed, not that a below-threshold candidate
passed. No failed numerical candidate is selected.

`run_final_candidates.py` then measures44 paired whole-layer runs with this
local geometry and8192-byte packets: replicated/sharded residuals, all coherent
CCL/fusion/persistent/packing families, activation and CCL precision, and
attention/MLP dtype/fidelity. All44 commands complete0. Replicated default
traced decode is0.3709ms linear /0.3093ms full, versus paired single-chip
0.5243/0.4270ms. Packed local MLP and persistent collectives remain slower.
Final gate/review/profile acceptance is still pending. A final QKVG accuracy
tradeoff check tests whether higher precision permits the larger faster block;
it preserves LoFi candidates and does not assume a precision fix.


## Decode QKV precision and geometry refinement

Real batch32 controls in `run_accuracy_tradeoffs.py` show BFP4 HiFi2 and
FP32 accumulation do not rescue the fast QKV blocks. BFP8 LoFi does.
A decode-only raw-HF BFP8 QKV copy retains BFP4 prefill and other projections;
`decode_qkvg8_b16_accuracy` passes minimum per-user HF PCC0.99644738,
and `decode_qkvg8_b16_layer3` measures0.293185ms traced decode. The full
attention BFP8 policy is slower and changes prefill; it is not selected.
The extra copy adds11141120 bytes per full-attention layer/device.
The plan now budgets1064697856 projection bytes across all32 layers/device,
with native262144 capability unchanged and final reservation rerun pending.

`geometry_qkvg8_wide_layer3` sweeps all8 legal blocks under BFP8 LoFi.
The first DRAM sweep reaches an L1 CB collision at cores4/block32
(CB end1307648 vs lowest allocation1284416); smaller configurations pass.
The diagnostic retained all local inputs in L1, so a revised capture stores
inputs in DRAM and restores only the role being measured. Its first attempt
(`geometry_qkvg8_dram_sparse_capture_layer3`) fails before geometry because
clone cannot change sharded/interleaved layout. The corrected probe uses
to_memory_config plus clone only when already in DRAM. This is test setup,
not a production-path error. `reset/list/mesh_after_capture_clone` all exit0.
No process was killed or lock removed. `geometry_qkvg8_dram_capture_v2_layer3`
and `geometry_qkvg8_wide_capture_v2_layer3` both pass; the DRAM rerun excludes
the already failing block32 and measures all other legal core/block choices.
The strongest DRAM component candidate is32 input cores/block4 (~42.53us),
now being compared in the complete decoder against interleaved32 block8/16.

`finalq8_dram32_b4_accuracy` passes all96 per-user comparisons across three
steps (minimum0.99687457), exact restored eager/trace and immutable snapshots.
Mixed DRAM32/block4 is selected over DRAM4/block4 and interleaved8/16.
Production now has explicit decode_qkvg_dram=True; the geometry and fused
input-gather probes request an interleaved weight contract when needed.
`finalmixed_*` reruns compatible full-attention topology/precision families;
linear-attention code/precision is unchanged from the44 selected32 runs.
The packed-MLP control initially retains a DRAM-sharded gate_up duplicate
which the new decode-weight selection sends to standard multicast matmul.
It fails with "Only L1 buffers can have an associated circular buffer".
The control now shares its interleaved packed weight when decode_grid is set,
matching the standard matmul contract. This is candidate setup only.
Serialized reset/list/mesh recovery is recorded as *_after_mixed_packed.

The adapted mixed-policy matrix completes22 full-attention families across
`finalmixed_*` and `finalmixed_v2_*` (all completed candidates exit0).
Replicated default reproduces0.283588ms; directly consumed sharded residual
0.392927ms, fused norm/input-gather MM0.382070ms, packed MLP0.299471ms,
persistent CCL0.297634ms, BFP8 CCL0.296217ms. MLP HiFi2 is within timing
noise (~0.2831ms) and offers no material advantage; LoFi remains selected.
Higher precision and lower-precision activation controls do not win.

`python doc/multichip_decoder/run_validation.py release_mixed` (path relative
to the autoport root; actual invocation uses repo-relative full path) starts
the full97 contracts plus capacity and native oracle under worker watcher.
TT_METAL_WATCHER_APPEND=1 and a unique TT_METAL_LOGS_PATH preserve every
fixture. The script archives all watcher files even if a gate fails, then only
on success runs direct-stack and actual1x1-vsTP4 paired timings. Profiler is
separate. A fresh xhigh stage-review subagent is independently inspecting
source/evidence while root owns hardware. It requests a corrected-capture
retry for DRAM QKV cores4/block32; this is queued after validation rather
than treating the earlier excess-input L1 collision as definitive.

Stage Python files pass all applicable repo pre-commit hooks and py_compile
(19 files at that point). No C++/CMake change: no build is required. Final
artifact-wide checks and checkpoint commits remain pending.


## Final mixed-path gates and review controls

`release_mixed_watcher_contracts`:99 passed,2 Python/SWIG warnings,815.61s;
exit0. Full native262143/262144 contracts and final-position trace pass for
both kinds; chunk2048-vs1024 tail PCC.999972 linear/.999954 full. Both host
fallback guard families are verified to fire and then all measured forwards
pass. Native permuted4096-page BFP8 cache oracle PCC.99836834. The updated
capacity reservation holds6473908224 DRAM bytes/device plus24 recurrent L1
states while executing native full attention, and passes. Context remains
262144. See validation_summary.json and release_mixed_watcher_manifest.json.
Watcher log and kernel-name maps append across all fixtures; other inspector
metadata is a final snapshot. Raw11.52MB watcher log has no error signatures.
ETH is explicitly disabled; the full-instrumentation limitation above remains.

`release_mixed_stack` passes131 logical prefill and8 changed-input/position
trace steps through linear/full/linear; minimum baseline PCC.99996213 prefill
and.99992172 decode, exact restored eager/trace, zero boundary conversions.
`release_mixed_timing_layer0/3` reproduce traced decode.371098/.283307ms,
paired single-chip.523857/.427430ms. Source and outputs are archived.

Independent review requested same-policy wider QKV grids and the corrected
DRAM4/block32 probe. `geometry_qkvg8_grid64_layer3` and grid110 both sweep
all8 legal blocks at BFP8 LoFi, exact replay and local PCC>.995. Best47.26us
(block4) and45.31us(block8), versus selected DRAM32/block4~42.53us.
`geometry_qkvg8_dram4_b32_corrected_layer3` now FITS:47.71us, exact replay.
The original L1 collision was caused by diagnostic retained-input pressure;
it is not used as a production blocker. Whole-layer QKV-only controls
`review_q64_b4_layer3`, `review_q110_b8_layer3`, `review_dram4_b32_layer3`
all pass and measure.289691/.287984/.291866ms, slower than default.
Both review geometry findings are resolved by measurements. BF16 tile units
in mesh_plan corrected:2048 bytes/tile, four tiles per8192-byte payload.

Final benchmark mesh setup now uses l1_small_size24576, matching the existing
99-case contract fixture and profiler. This prevents native CCL semaphore
allocation in the general L1 heap. Historical geometry/timing controls used0
and preserve that provenance; selected default, nearest wider-grid control,
and direct stack are reproduced as release_l1small_* /review_l1small_q110.
Production source is unchanged by this harness setup consistency update.

The reviewer found the remaining local-prefill geometry controls were mixed
with precision changes. Final-policy8×8 and11×10 grids at K blocks8/16 are
therefore measured directly.8×8/block8 passes both kinds but is slower
(3.6901/2.9323ms).8×8/block16 fails specifically at FP32 gdn_out: statically
allocated CBs1717248 bytes exceed physical L11572864. This is an exact
capacity error, not a PCC rejection. A test-only per-role override keeps that
projection at block8 while retaining block16 elsewhere; full attention has
BF16 O input and is tested unchanged. Reset/list/mesh recovery is recorded
as *_after_prefill64_b16. No process was killed or lock removed.


Final adapted8×8/block16 prefill passes at3.6067ms linear and2.9241ms full,
both slower. The11×10/block8 three-sample result was close to block16, so
review_prefill_stable_b{8,16}_layer{0,3} measures15 calls after one warmup.
Selected block16 wins both:3.363816 vs3.448008ms linear,2.677883 vs2.801300ms
full. All four runs pass baseline PCC, all-rank and restored trace gates.
These longer paired samples supply the final summary table: prefill speedup
2.0658×/2.0182× and decode1.4124×/1.5039× for linear/full. No production
change was required. run_validation.py now reproduces this sample count.

All four final profiler captures and per-rank tt-perf-report tables pass.
PERF_ANALYSIS.md reconciles host/device times without summing independent
chip clocks, records actual dtype/fidelity/subblocks and examines compute,
DRAM, CCL and movement. Optimistic weight/KV bandwidth floors are60.19μs
linear and70.05μs full; omitted traffic/compute/communication are explicit.
The source-derived QKV subblock is1×5, distinct from the public storage
per_core_N3. The reviewer’s geometry/accounting findings are now addressed.

Final repository checks: pre-commit run --files with all 1096 stage-owned tracked/visible untracked paths passes (logs/final_precommit.txt). First pass trimmed report/triage text whitespace; second pass is clean. Python compilation passes for all 19 stage Python files; JSON parses and git diff --check passes. Production SHA256 remains cf37a95b2b77afa4b6a0131643b444cc26b8f83c42b3ee0938273079d0ab3702, identical to final hardware evidence. Python/tests/docs only: no C++ or CMake build is required.


## Stage closure

Independent xhigh reviewer multichip_stage_review returns clean-pass in
STAGE_REVIEW.md. All findings were resolved and rereviewed; no required work
remains. The reviewer independently verified final source/log archives,
watcher/profiler hashes, rank accounting and paired speedup/efficiency.
Root also rechecked all4 final raw profiler hashes, all11 watcher archives
and8 original baseline Python files, with no mismatch. Context/capacity and
validation status now record complete. Native262144 is preserved; all-feature
Ethernet watcher and later full-model/serving validation retain the explicitly
recorded bounds. The stage implements only the decoder, tests and docs.

Repository: /home/hous/dev/ornith-1.5-9b/tt-metal.
Branch: hous/ornith-1.5-9b. Parent:483920f536f64462ba6c3dc785c59b8ff002cdc1.
The local stage checkpoint includes implementation, tests, review and compact
evidence. A subsequent documentation commit records its SHA. Nothing is pushed.

Local stage checkpoint: `d4a512f69b72badb4579b0af0b0fe652ed86af7b` (`Add validated TP4 Ornith 1.5 9B multichip decoder`),1097 stage-owned files. Commit hooks pass. The initial staged git diff whitespace check flags standard CSV CRLF line endings; CSV data is preserved byte-for-byte for provenance. `git -c core.whitespace=cr-at-eol show --format= --check HEAD` passes. This documentation-only follow-up records the checkpoint SHA; no push occurred.
