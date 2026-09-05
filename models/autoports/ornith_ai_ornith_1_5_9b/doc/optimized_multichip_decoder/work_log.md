# Optimized multichip decoder work log

Status: in progress; no optimized-stage pass or performance improvement claimed.
Start: 2026-09-05, branch hous/ornith-1.5-9b, HEAD 65abe7f69d.
Target: ornith-ai/Ornith-1.5-9B, pinned revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Hardware: four Blackhole chips on two P300c boards, logical 1x4 ring.
Scope: optimize completed multichip decoder in place; no full-model/vLLM work.
Initial git status clean. Read model AGENTS, optimize, tt-device-usage,
autofix and stage-review skills, and LLM report section 4. Existing multichip
README, PERF_ANALYSIS, candidate and geometry evidence are the prior controls.
`timeout 60 tt-smi -ls --local` exited 0; all four P300c chips visible.
No reset, process termination, or hardware recovery was needed.

## Initial operation-topology audit

Evidence: ../multichip_decoder/PERF_ANALYSIS.md and advice-enabled
../multichip_decoder/tracy/{linear_attention,full_attention}/final/decode_device*_report.txt,
plus tt/multichip_decoder.py and inherited optimized/fused decoder implementations.
These are prior-stage measurements, not this pass's before/after results.

| Boundary / repeated operations | Prior material cost | Compatible replacement and constraints | Action / evidence status |
|---|---|---|---|
| GDN input QKVAB and Z consume the same normalized 4096-wide input | 19.4 + 12.5 us projection rows; separate fused-SiLU Z | Pack all fields from raw HF per-rank weights, tile-align A/B; retain BFP4/LoFi and compare added Z SiLU/slices | New packed GDN experiment required |
| Full attention packed QKVG, heads, RoPE, BFP8 paged cache, SDPA | QKV DRAM plus layout transitions; logical batch1 with32 padded rows | Preserve packed head ordering and local1 KV head; attention BFP4/LoFi must retain real batch32 correctness | Reproduce baseline and inspect prior batch32 localization before precision changes |
| GDN/attention row projection then native all-reduce | Two layer reductions together39–50 us decode, ~0.70ms prefill | RS + carried hidden-sharded residual/distributed norm; gathered-input/output-sharded projection; fused AG-MM or MM-RS | Compare whole compatible layer families, no timed boundary restore |
| Both residual norms/adds | Width-sharded replicated BF16 decode; DRAM prefill | Carry local L1 shards; compare distributed hidden shards with stats gather and delayed activation gather | Prior sharded family slower; rerun current cumulative policy |
| MLP gate/up same input + fused SiLU-multiply + down | Two26.4 us gate/up and25.4 us down rows | BFP4/LoFi packed versus tuned separate with all slices/activation/layout costs | Reproduce both families and preserve separate precision groups |
| Decode projection input to interleaved L1 and row collective layouts | 35–43us total movement | Direct sharded inputs/outputs, phase-specific working shards; DRAM one/two/three readers | Prior reader2/3 mesh-coordinate blocker needs current AutoFix adjudication; geometry prior evidence retained |
| Repeated decode CCL allocations | Native versus async RS/AG | Persistent outputs/intermediates with stable ownership and same dtype/links/topology | Rerun persistent coherent families; test changed-input trace and watcher separately |
| Communication and activation precision | BF16 residual/CCL, FP32 GDN recurrence | BFP8 attention, MLP, CCL independently on replicated and hidden-sharded families | Cross with kept topology; runtime dtypes must agree with policy |

Initial cumulative contract: TP4; replicated BF16 residual4096, L1 width-sharded
32 cores in decode, DRAM prefill; BFP4/LoFi projections except raw-HF BFP8/LoFi
full-attention decode QKVG (DRAM32-core/block4/reader1); BFP8 local-head paged
KV64; FP32 GDN state; split MLP gate8/up8/down6 on8x4 compute; GDN QKVAB/Z
block32 and output8. Native262144, arbitrary logical lengths, batches1–32.
Layer outputs feed the next decoder directly. No inter-layer conversion is
permitted solely to restore a helper's preferred contract.

## Fresh baseline and topology experiments

`record_run.py before_layer{0,3} timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer {0,3} --length 2048 --prefill-iterations 16`:
TP4 linear prefill3.463762ms/decode0.371151ms; full prefill2.858084ms/decode0.283669ms.
TTNN PCC linear0.999963081/0.999987989, full0.999977247/0.999695988;
exact all-rank restored eager/trace. The 1x1 phase is a numerical control only;
this stage compares the TP4 rows. `run_profiles.py before` captures four separate
prefill/decode reports with watcher unset, advice enabled and source provenance.

Raw-HF packed GDN QKVABZ passes at0.369156ms; packed plus shared interleaved MLP
input passes at0.365455ms. Shared MLP alone passes0.367545ms linear and0.280776ms
full, removing the inherited8-core input reshard that the selected multicast
matmul immediately undoes. These are candidates, not final default evidence.

Fused Z SiLU into mixed-FP32/BF16 multiply fails real decode PCC(-0.085274)
while prefill and restored trace/eager agreement pass. Rejected from selection
pending fresh AutoDebug localization; no numerical-instability explanation is
claimed. `packing_matrix.json` stops on that failed gate. Geometry continuation
uses the correct separate-Z-SiLU candidate in `packing_geometry_matrix.json`.

AutoFix reader investigation independently confirms current binary's unit-mesh
hop-distance assertion for reader2/3. Native per-coordinate factory adaptation
is being investigated in an isolated worktree. No binary/build changes occur
while the parent hardware lane is active.

## Cumulative family failure and recovery

`family_ag_mm_replicated_layer0` failed exact restored eager agreement after
trace capture; trace PCC -0.242638 and max difference2.72e28. The same family
was previously measured only with a sharded residual. No latency or correctness
claim is accepted from this failure. The matrix stopped; task processes exited
and closed devices. A bounded `tt-smi -r` is recorded as
`reset_after_ag_mm_failure`; post-reset list and controlled original/cumulative
replicated/sharded probes will localize rather than reject on the first failure.

## Reader mesh AutoFix investigation

Reader subagent independently verified the current native R2/R3 hop-overload
blocker and prepared the minimal coordinate-aware patch in isolated worktree
`/tmp/ornith-reader-mesh-autofix`. Patch and report are
`reader_mesh.patch` and `AUTOFIX_reader_mesh.md`. Four host C++ syntax checks
pass (utility, factory, matmul operation/adapter concepts, nanobind). Required
`copilot-build.sh` exits 1 because Docker is unavailable. No TTNN import, device
access, object/library build, install, or live implementation edit occurred in
this investigation. Parent build and runtime verification remain pending.

## Native reader fix build and Z localization

`reader_native_build`: `timeout 1200 cmake --build build_Release --target ttnncpp ttnn -j 2` passed all14 incremental targets with the installed compiler/cache.
This environment is already a build container; no compiler/dependency installation
was needed. Mandated `.github/scripts/copilot-build.sh --build-ttnn-tests` fails
because Docker is absent (`reader_copilot_build`), not because source failed to
compile. Normal `cmake --install build_Release --component tar` and component
`tt_pybinds` succeeded; subsequent provenance captures actual runtime SHA256s.
Hardware reset/list/1x4 mesh smoke all passed before build/install.

`z_fusion_boundary` uses real captured core/Z and verifies failure at the fused
mixed-dtype multiply: BF16 RHS activation produces huge/nonfinite values;
FP32 RHS and swapped BF16 LHS with explicit FP32 output pass. Swapped-LHS
projection PCC0.999999675, not claimed bitwise identical. The source diagnosis
identifies missing RHS unpacker format configuration. A model-local swapped
operand adaptation is now being tested as `packed_gdn_fused_z_lhs`; the original
failing RHS candidate remains diagnostic evidence, never a default.

Reader follow-through: parent integrated the exact patch, native build and both
install components exited0, and real TP4 layer3 QKVG reader2 at context2048
passed (reader2_qkvg_layer3, eager/trace exact, decode PCC0.9996959877 all ranks).
Native sources and installed artifacts were captured in reader_native_sources.json.gz
and reader_native_artifacts.json. New tests/multichip_reader_probe.py provides
raw-HF per-rank1/2/3 comparisons, alternating timing windows, same-input trace
and fresh-buffer/changed-input regressions. py_compile and pre-commit pass;
this subagent did not execute its hardware path. See AUTOFIX_reader_mesh.md.

Reader3 follow-up: first all-role probe reached a distinct GDN packed writer
assert after R1/R2 eager/trace passed. Source arithmetic identifies reader23
starting at output tile69 after4×17=68 storage tiles are exhausted (N66tiles,
24readers×3tiles=72). Added probe-only --per-core-n; candidate18 pads output
capacity to72 with unchanged input4cores/block32 and unchanged weights.
Source-predicted related role overrides and exact commands are in the AutoFix
addendum and reader_storage_geometry.json. Runtime adjudication remains parent-owned.

## Reader tail storage and collective localization

`reader_sweep_layer0` passes construction of readers1/2 then reader3 GDN packed
fails because24 readers×3 output tiles=72 exceed output4 cores×17=68 tiles;
the last padding-only reader starts beyond storage. `reader_gdn_n18` adapts
per_core_N to18 under all reader counts, preserving input4 cores/K-block32,
raw weights and logical N2112. All three now pass PCC, repeated cache rebinding,
changed-input replay, and alternating timing windows: R1 26.270us, R2 26.327us,
R3 27.189us. This exact-shape adaptation resolves the first validation error;
no generic reader default changes.

`ag_boundary_layer0` localizes corruption before trace to BF16 gather3072→12288
on rank3. Smaller gathers pass. Source audit finds default Blackhole worker
selection extends beyond inherited8x8 semaphore initialization. Caller worker-grid
and full-device semaphore controls are being tested. First caller-grid control
fails only because diagnostic used singular sub_core_grid; installed API uses
sub_core_grids. Corrected retry is required and this error is not a rejection.

## Full-grid semaphore repair

Both `ag_boundary_core_grid_layer0_v2` and `ag_boundary_full_grid_layer0` pass exact whole-layer restored eager and all isolated/producer gathers. This verifies the missing semaphore coverage hypothesis independently of trace and producer lifetime. Model-local MeshCCLManager now initializes the actual 11x10 grid. Original failed topology families and async links1/2 are queued for retest; historical two-link rejection is being reconsidered.

`reader_sweep_adapted_layer0` passes all seven real projection roles with readers1/2/3 and padded output storage. R2 gate/up/down traced microseconds27.52/27.49/27.71 versus R1 32.94/32.88/33.66; whole-layer DRAM selection still requires measurement.

## Corrected topology and packed projection evidence

All16 `ccl_fixed_matrix.json` cases pass. Original/cumulative replicated gather, fused gather, fused-norm sharded, and async links1/2 all pass both kinds after full semaphore coverage. Linear default-packed/native0.3657ms beats async1/2 0.3870/0.3895ms, replicated gather0.4825ms, fused gather0.4889ms, and stack-compatible fusednorm/sharded0.4495ms. These are measured rejections, not first-error exclusions. Source review found the fused-norm gather offset(0,8) also lies outside inherited8x8. `multichip_probe` now explicitly restores/replays/reads after the intervening eager run, closing a coverage gap; final topology comparisons use this check.

All13 packed projection geometry cases pass. Packing QKVAB/Z in both phases improves warmed prefill (~3.35ms) and traced decode; best block8 on8x4 (~0.3602ms) vs8x2/8x8/11x10 alternatives. Final default must be rerun with16prefill samples and batch/native gates.

`dram_mlp_r1_layer0` failed an experiment setup mismatch: unused DRAM-sharded GDN output weight was selected by interleaved multicast, yielding L1 circular-buffer validation. Candidate now removes unused DRAM copies before forward. All4 corrected whole-layer cases pass: R1/R2 MLP linear0.3806/0.3605ms, full0.2935/0.2723ms (fullQKVG reader2). Shared width-sharded MLP input/output is preserved in these measurements.

## Production-path integration controls

All38 all-modes topology/activation cases pass after the semaphore repair, including restored trace-after-eager readback. Lower-movement families retain sharded residual1024 through the full layer; host gather occurs only outside measured execution. Shared/native remains faster.

The production implementation now has opt-in pack_gdn/decode_dram_roles controls; defaults are not yet promoted. It packs from raw per-rank HF weights, removes unused separate weights, uses the proven compact mixed-dtype Z gate, and preserves explicit DRAM role selection. The inherited DRAM program accepts an optional per_core_n output-storage override without changing any default. Current production controls pass: linear packed/MLP-R2 ~0.3589ms, full MLP-R2 with QKVG-R1/R2/R3 ~0.27135/0.27217/0.27497ms. The coherent current family therefore prefers QKVG-R1, despite an earlier isolated whole-layer R2 comparison. QKVG BFP4 at adapted32/16/8 cores is under batch validation before any default selection.

`production_cache4_layer3` passes outputPCC (prefill0.998539/decode0.999104) and exact replay at0.272232ms, but fails the unchanged0.99 state/cache PCC gate against the BFP8 control. This is not a partition-bug diagnosis or an eligible speed result. Real-weight batched HF controls are queued to determine model-visible precision impact; future failure logs print the individual state score.

All4 production BFP8QKV/BFP8KV batch cases pass; all4 cases also pass each adapted BFP4QKV geometry32/block4,16/block8,8/block16 with MLPDRAM readers2. BFP4QKV8/block16 gives0.266119ms atcontext2048; precision selection remains cumulative and must pass fullstress/nativegates. BFP4KV with BFP8QKV also passes batch4/32 HF checks, despite the cross-dtype state99 failure. New AutoDebug/AutoFix is investigating the validation attribution and exact TP4 cache write/read contract; no threshold has been lowered.

The seven additional fewer-core reader runs all pass51 cases: MLP core4 does not beat8core/R2; QKVG BFP8 core8/block8 improves local latency and is being measured alongside BFP4 under the same whole-layer geometry/readers. PackedGDN reader1/2/3 and every proposed32/64core case pass, including explicitly padded storage.

All12 finalQKV production comparisons pass at common output per_core_N12, crossing BFP4/BFP8, readers1/2/3, blocks8/16 and input8cores. BFP4/R2/block16 wins0.266045ms (un-padded defaultN10 control0.266119ms), versus bestBFP8/R1/block16 0.271374ms. R3 remains slower even after output-storage adaptation. All four GDN DRAM coherent candidates pass: packed0.361181, output-only0.363800, wholemixer0.365592, separate0.367297ms versus packedinterleaved/MLP-R2 0.358922ms. These earned rejections preserve native residual boundaries.

`precision_attention8_replicated_layer0` hit a prefill GDN-output L1 circular-buffer size1618944>1572864 bytes with FP32 activations/BFP8 weights and block16. The precise failing projection was localized; the retry halves only that prefill role K-block to8. This first resource error does not reject BFP8 or defer precision work.

### Final-family and cache adjudication (2026-09-05 09:38 UTC)

The exact B32/T2048 user31 current production candidate passed HF PCC0.9989239221 on every rank (`z_user31_hf_current_v2`), including finite/nonconstant rank-local core/Z/merged boundaries. See the completed Z AutoFix report for archived-source and legacy-grid controls; the historical constants are not assigned an unsupported causal explanation.

`remaining_op_matrix_v3.json` completed all pending SDPA-grid and residual-grid candidates. The oversized chunk1024 SDPA failure is undergoing an explicit live-buffer/layout adaptation, not rejected at its first L1 validation error. `final_topology_matrix.json` compares the selected packed GDN, DRAM-reader2 MLP and QKV4 core8/block16 family under replicated and retained-sharded residual contracts.

Matched cache diagnostics `cache_precision_qkv8_b1`, `cache_precision_qkv4_b1`, `cache_precision_qkv4_b1_permuted`, and `cache_precision_qkv4_b32` prove identical producers and exact physical fill/update/untouched-row behavior. Both K4 and V4 fail the unchanged state0.99 gate; at B32, QKV4+KV4 also fails HF decode users8/26 at0.993657745/0.994208792 while KV8 passes (minimum0.996814237). Remaining permuted B32 diagnostic is queued. No gate is relaxed and neither mixed K4/V8 nor K8/V4 can satisfy the unchanged per-payload state gate.

The candidate memory plan now accounts for all32 additional DRAM MLP copies and the smaller QKV4 decode copy:1,702,232,064 projection bytes/device;13,553,532,928 conservative total estimated bytes/device;7,111,442,432 reserved DRAM bytes/device in the upcoming native-capacity test. This is a decoder capacity reservation, not full-model execution. Final context revalidation remains pending.

### Coherent final families and adapted SDPA complete

All38 cases in `final_topology_matrix.json` pass output/state gates and exact restored trace after an intervening eager decode. Native replicated controls are0.358910ms linear and0.266112ms full-attention; the retained-hidden-shard families and fused/packed/persistent/activation/CCL alternatives are slower. See `optimization_evidence.md` for paired family tables and byte/layout contracts.

SDPA1024 first adaptation released dead buffers and moved Q/gate toDRAM. It passed initial eager/trace but failed the subsequent restored eager decode while the captured trace/output remained live: liveL1 at1,514,496 versus staticCBend1,526,144. The second adaptation capped reduction workers8, saving81,920 static scratch bytes and passed all probe gates at0.276570ms. Matched low-live-buffer256 controls were0.267339ms (cap16) and0.271272ms (cap8), versus selected unadapted256 at0.266112ms. Retain selected256; material rejection has successful adapted timing, not only an API failure. `AUTOFIX_sdpa_chunk1024.md` records commands and exact phase correction.

All five matched KV diagnostics are complete. Permuted B32 reproduces failed KV4 output/state accuracy while exact implementation checks and identical producer hashes pass. RetainKV8; see `AUTOFIX_cache_precision.md`.

### Final default v1 gate failure: batch32 changed-input trace

`final_default_v1_watcher_contracts` fails full-attention B32 `test_traced_decode_pcc` on a real recorded input at PCC0.9942707597 <0.995. Native capacity reservation, native cache oracle (PCC0.9981916521), full-attention chunk invariance(PCC0.999954) and native traced position262143 already pass. The suite stops; final defaults are not accepted and no final timing/profile is claimed. A fresh xhigh source-only AutoFix agent investigates while root runs matched QKV8 and QKV4 controls on the exact failing selector. Watcher logs are preserved by the validation runner. No threshold change or precision-only conclusion is made from the first failure.

### QKV trace AutoFix localization

Both QKV8 controls pass the exact failing96 user-step gates, including the matched8-core/block16/reader2 geometry. `qkv_trace_b4_b32` reproduces onlyuser31/step0 atPCC0.9942707597 on allranks and all eager/trace/replay-after-eager phases; snapshots remainunchanged and final caches areexact. Extracted user31 failsalone at0.9943646417, rulingout a batch-conditional workaround. Input hashes and all8 prefill cache payload hashes matchQKV8 andHiFi2 controls exactly. HiFi2 is numericallyidentical toLoFi; HiFi4+FP32 accumulation improves to0.9948409 butstillfails, and32-core/block4/reader1 reaches0.99454552 butstillfails. `qkv_trace_gate8_b32` restoresonly thegatefield fromBFP8 and passesall96 user-step comparisons.

`AUTODEBUG_qkv_trace.md` / `AUTOFIX_qkv_trace.md` preserve the fresh independent diagnosis. Gate substitution uses two full packed matmuls onlyforlocalization and supplies no performance claim. A true splitQKV4/gate8 implementation is being prepared to compare against correctpackedQKV8 (~0.2713ms). Allthresholds remainunchanged.

### Packed QKV8 restored after completed AutoFix

All14 actual splitQKV4/gate8 geometry/reader candidates pass; best0.275439845ms loses to packedQKV8 C32/block4/reader1 at0.271286342ms (C8/block16/reader1 ties0.271270063ms). The default returns to the establishedC32/block4/reader1 QKV8 contract for allbatches, retaining packedGDN and DRAMMLP reader2. No runtime error fallback or batch-specific precision workaround was added. The updated plan includes13,595,475,968 estimated bytes/device and7,153,385,472 reserved DRAM bytes/device for upcoming nativecapacity revalidation.

Before final closure, packedMLP gets explicit gate_up1/2/3-reader, core-grid and K-block trials. Existing ProductionPackedMLP copied gate_proj config (reader2/block8) into gate_up; the experiment now honors an explicit gate_up config so tuning is effective. The selected QKV8 full-attention topology families will be measured again; prior linear-family data remains the identical projection path.


## Packed decode MLP promotion and runner continuation

The precision-locked packed gate/up sweep now includes 22 geometry cases:
4/8/16/32 input cores, wider legal K blocks, and all three reader counts.
`packed_mlp_final_matrix.json` adds seven 32-core configurations and three
matched separate/packed R2 pairs for each layer kind. R2 at32/K4 ties linear
and loses slightly on full attention. `packed_mlp_r3_confirm_matrix.json`
adds two matched pairs per kind and retained-sharded controls. R3 at32/K4
reproducibly wins about3us per layer. Production now packs gate/up only for
decode, uses32/K4/R3 and down8/K6/R2, retains separate prefill weights, and
owns compact batch folding before sharded output slices. The two decoder
weight dictionaries alias one packed allocation; unused separate decode
copies are removed. Weight bytes and the262144 context contract are unchanged.
`packed_promoted_batch_smoke` passes10 real-HF cases including B1/B4/B32 traced
outputs for both kinds and B4/B32 prefill/decode. Default probes reproduce
0.355676ms linear and0.268672ms full decode; these are promotion probes, not
final stage timing. Production code and candidate/source-summary formatting
passes pre-commit. The candidate index now calls local gates `probe_pass`,
avoiding an implication that historical QKV4 probes passed full-stage gates.

At continuation on2026-09-05 the runner container had been recreated. The
old exec session72929 no longer existed and no hardware workload survived.
The next case `packedfinal_persistent_replicated_layer0` had an empty log and
incomplete provenance; preserve it plus `interrupted_runs.json`, and rerun as
`packedfinal_persistent_replicated_layer0_resumed`. This is an interrupted
measurement, not a numerical or hardware failure. `timeout60 tt-smi -ls --local`
returned all four p300c chips; recorded `resumed_mesh_smoke` opened/closed the
native1x4 fabric ring successfully. No reset or reboot was requested. Resume
only unfinished cases using `packed_default_families_resumed.json`, then run
`validated_full_attention_families.json`. All earlier completed records remain
immutable. Final watcher/capacity gates, same-default timing and profiler
collection, fresh independent review and local commits remain required.


## Final cumulative families and advice closure

After the resume, all32 `packed_default_families_resumed.json` cases and19
`validated_full_attention_families.json` controls pass. The former retain the
production packed MLP; the latter explicitly exercise separate MLP controls,
with their named packed alternative. Lower-movement families retain1024-wide
residuals through decoder consumers, rather than restoring replication inside
timing. All34 `packed_final_precision.json` runs pass: packed fused-norm AG/MM,
per-group HiFi2, packed MLP8 LoFi/HiFi2, async link counts, BFP8 packet4352,
and paired prefill L1 inputs. No option displaces the default. L1 input0 advice
was tested for all prefill projections with one shared gate/up conversion:
linear3.333419→3.452045ms; full2.667687→2.790130ms, unchanged correctness.
BFP8 packet4352 advice was tried in both residual layouts and both kinds;
passing results remain slower than the final BF16 collective family.

`run_packed_readers.py` then passes two alternating-order R1/R2/R3 micro runs
with exact real TP4 gate/up shape4096x6144 and independently uploaded raw-HF
BFP4 weights, plus two separate signposted Tracy captures. Warmed microseconds
R1/R2/R3 are59.0998/44.6689/41.5453 linear and59.0985/44.5434/41.5854 full.
Changed-input, fresh-binding and trace checks pass. `render_reader_profiles.py`
produces24 reader/rank rows, advice-enabled per-rank text/CSV, compressed raw
ops and same-run host/kernel/gap values. Stored-weight GB/s is labeled as an
inferred storage rate, not a measured traffic counter;512GB/s is the tool's
Blackhole model peak. Raw reader traces stay under ignored `tracy/reader_raw/`.

`run_validation.py final_default_v2` starts the101-case worker-watcher suite,
native capacity/cache tests, explicit async watcher checks, direct composition,
and16-iteration warmed final timing on the default path. No performance claim
uses the rejected QKV4 policy. Full final profiler collection and independent
review remain pending until this run finishes. `warning_ledger.md` now records
trace-allocation risk controls, packet advice, topology metadata and runner
interruption classifications without suppressing warnings or weakening gates.

## Final default gates and profile accounting

`python D/run_validation.py final_default_v2` completes successfully:
101 tests pass in477.90s with worker watcher10; both explicit async watcher
probes pass; native262144 allocation includes7,153,385,472 reserved DRAM
bytes/device plus24 recurrent layers in L1. The separate constructed-history
cache oracle passes at262143 with HF PCC0.9983894214 and exact trace output.
Direct three-decoder composition (layers0,3,0; logical prefill131; eight trace
steps) has zero B1 boundary conversions, exact eager/trace output and minimum
PCC0.9999203986. No host fallback, state/trace, ragged/prime batch, non-aligned,
poisoned pool, or refreshed prefill trace gate fails or is waived.

The final default unprofiled measurements are recorded in `measurements.json`:
linear prefill3.463762→3.443891ms, decode0.371151→0.355672ms;
full prefill2.858084→2.690952ms, decode0.283669→0.268965ms.
The reproducible optimization claim is4.17%/5.18% lower traced decode latency.
Linear prefill varies across synchronized host windows; its final median is
not replaced by an earlier faster candidate. Full-attention prefill has
unchanged math/topology, so the observed median difference is not attributed
to an algorithmic prefill optimization. Final probe prefill/decode PCC is
0.999963081/0.999990048 linear and0.999977247/0.999697387 full.

`python D/run_profiles.py after` passes all four separate advice-enabled
Tracy/HF runs with watcher unset. `python D/summarize_profiles.py --before
before --after after` verifies eight profiles/32 rank windows and writes
`performance_accounting.{json,md,csv}`. Final same-profile host time is
385.308/314.762us linear/full; max independent rank spans368.621/303.530us.
Kernel and all interior gap times reconcile to those spans. Optimistic active
projection-weight/minimum-tiled-KV read floors are60.192/70.050us at the
explicit512GB/s-per-chip model assumption, excluding other traffic and overhead.
These are model bounds, not measured bus utilization. Raw rows prove packed
GDN and gate/up BFP4/LoFi, gate/up reader3, down reader2, and QKVG BFP8/LoFi
reader1. Reported kernel core counts are kept separate from storage grids.

`python D/verify_artifacts.py --final-label final_default_v2 --final-label
profile_after_` verifies466 completed provenance records,933 archives and36,591
archived source entries with zero errors. All10 selected final runtime records
match the current TTNN Python/native source and installed binaries. The one
runner-interrupted empty record is preserved and counted separately. No
historical failed candidate counts as a passing stage gate.

Final pre-commit initially normalized trailing whitespace/EOF in generated
human-readable profiler tables. Raw ops/log/source archives were untouched;
checksum verification remains clean. Both renderers now produce normalized
text directly. The final repeat is recorded in `precommit_final_after_format.log`.
Independent stage-review is the last signoff before local checkpoint commits.

## Independent review and completion

The fresh xhigh reviewer returns **clean-pass** in [STAGE_REVIEW.md](STAGE_REVIEW.md),
with no required work. It independently verifies final runtime hashes, the
101-test gate, async watcher checks, direct composition, native capacity,
coherent optimization families, native build evidence and all eight profiles.
The final full stage pre-commit run passes 1728 files; `git diff --check` and
artifact verification pass. Only completion metadata and checkpoint logging
changed after the reviewed implementation was frozen. No full-model or vLLM
work was performed, and no push or PR was created.

## Local checkpoints

Repository: `tt-metal`; branch: `hous/ornith-1.5-9b`. Stage-owned code, tests
and compact evidence are isolated from the ignored raw profiler/tensor dumps.
Checkpoint SHA is appended immediately after the local commit.
