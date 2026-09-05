# Datatype sweep work log

## Contract and starting state — 2026-09-05

Stage: datatype-sweep, after completed optimized-full-model and before vLLM.
Target: ornith-ai/Ornith-1.5-9B, pinned revision
489cb97981b8654bcfcf30ce1f94ed1b62e07b53. Four Blackhole chips on two physical
P300c boards, mesh1x4 (`p150x4` software profile). No vLLM integration or push.
The worktree was clean at start. Native context remains262144, batch1 native
capacity; no advertised capability reduction is authorized by testing cost.

Acceptance: full-model top1>=90%, top5>=98%, top100=100%, main AIME24 chat
reference100 generated tokens. Only trace-verified teacher-forcing decode
ranks candidates. Post-selection warmed token-out no-readback is a separate
serving-comparison metric. Qualitative evidence follows `$qualitative-check`.

Device listing: `timeout 60 tt-smi -ls --local` exits0 and lists chips0..3,
Blackhole/p300c. The original baseline then opens/closes the1x4 mesh normally.
No reset, lock clearing or process termination was needed. All device work is
serialized under the supervising runner's existing hardware lock. Workspace,
logs and session state are on persistent `/home/hous/dev/ornith-1.5-9b` storage.
The generic runner description of no hardware does not match this model-specific
reservation; device availability is verified directly.

## Baseline refresh

`logs/baseline_teacher_v1.provenance.json` records the exact original-runtime
command, source snapshot, commit, environment, binary/log hashes and exit0.
AIME24 chat100 teacher forcing:94%/100%/100%,82.70t/s/u,789.13ms cold-request
TTFT. This is not warmed token-out TTFT. The reference already contains100
positions and stays pinned; no new HF-main resolution is performed.

The selected optimized baseline already uses BFP4/LoFi decoder projections,
BFP8/LoFi decode QKVG, BFP8 paged KV, BF16 head with HiFi4/FP32 accumulation,
BF16 residual/activation stream and FP32 recurrent state. Native CCL retains
its producer dtype (BF16 ordinary projections; FP32 where GDN explicitly requests
FP32 output). Sampling logits remain BF16 and token/history storage UINT32.

## Runtime plumbing and candidate matrix

`tt/precision.py` validates a complete policy and defaults to the selected
artifact once present; `precision_config="baseline"` restores the safe baseline.
Both `OrnithModel.from_pretrained` and `build_generator` consume it. Layer
exceptions are explicit index-keyed partial group overrides. Unsupported fixed
activation/residual/logits/sampling/state formats and compute flags raise before
weight loading. Decoder weight/fidelity, decode QKVG, head, KV allocation and CCL
transfer choices are passed into their actual runtime constructors/operations.

`run_candidate.py` calls the unchanged common prefill and teacher-forcing checks
using one normal constructed generator. It records actual allocated weight/cache
dtypes, compute configs, and model replay counters. Timing is the common runner's
callback-bounded traced teacher forcing. It records accuracy failures without
turning them into runtime failures. Reduced131-token smoke checks precede cache
and CCL format changes.

Initial candidate files under `configs/`: baseline; canonical BFP8/LoFi and
BFP8/HiFi2; BFP4/HiFi2 all, attention-only and MLP-only controls; KV BF16/BFP4;
BFP8 CCL; BFP4 decode QKVG; head BFP8/BFP4 each with LoFi/HiFi2. Baseline supplies
BFP4+LoFi for every decoder group. Head changes will also require the shared
prompt suite because prior-stage French controls were sensitive to head policy.

Historical checkpoint status: in progress at this point; final selection and review disposition follow below.

## Initial fidelity controls and host checks

Both canonical BFP8 policies pass: LoFi99%/100%/100% at71.086t/s/u;
HiFi298%/100%/100% at70.135t/s/u. All-BFP4/HiFi2 is93%/100%/100%
at79.096t/s/u. Attention-onlyHiFi2 is93% at81.452t/s/u; MLP-onlyHiFi2
is94% at80.017t/s/u. All are slower than the refreshed82.472t/s/u baseline.
These are full32-layer results, with no synthetic PCC rejection.

`reference_provenance.json` verifies pinned reference hash and161 chat prompt
plus100 generated tokens. `logs/host_tests_v2.log`:46 host-only tests pass,
including the seven new policy validation checks and39 preserved generator/
trace/prefill tests. Initial host_tests_v1 used a nonexistent prior-stage test
path and collected zero tests; corrected in v2. Initial formatting fixes were
applied by repository pre-commit; final formatting is rerun after implementation
settles. No C++/CMake changes; no build is needed.

The first baseline and canonical-LoFi runtime summaries used object repr for
compute configs. Subsequent runs serialize actual math_fidelity, approximation,
FP32 accumulation and packer properties; the final baseline/selection reruns
will use that richer summary. All runs retain immutable source snapshots.

## Cache/CCL and QKVG controls

`kv16_smoke_v1`, `kv4_smoke_v1`, `ccl8_smoke_v1` each pass real-layer0/3
logical131 prefill, repeated greedy8 and seven measured decode replays.
Allocated KV formats and compute configs are recorded. Full-model runs:
KV BF1693%/100%/100% at82.691t/s/u; KV BFP491%/100%/100% at82.672;
CCL BFP894%/100%/100% at80.040. QKVG BFP4/LoFi94%/100%/100%
at82.570. Differences below approximately0.5% are provisional timing ties;
close finalists must be repeated, with the simpler/safer policy preferred
when noise prevents distinguishing them.

`context_candidates.json` recomputes all8 full-attention layer KV bytes for
BF16/BFP8/BFP4, including tile headers and the unchanged page64/local-head256
contract. Every format fits the advertised262144 budget analytically. These
calculations are not a claim of native execution for an unselected candidate.
Selected construction must still pass the native262143/262144 execution probe.

## LM-head performance and qualitative AutoFix

All four reduced-head fidelity candidates meet AIME100 gates: BFP8/LoFi92%
at84.994, BFP8/HiFi291% at84.684, BFP4/LoFi92% at84.574, BFP4/HiFi292%
at84.724t/s/u; every top5/top100 is100%. These are candidate results, not
selected serving headlines. The modest BFP4/BFP8 ordering differs from older
terminal-only evidence, so matched warmed token-out controls are required.

The shared six-prompt suite plus AIME was rerun with BFP8/LoFi head and exact
cached pinned HF controls: `qualitative_head8_lofi_v1/`. Prompt4 incorrectly
labels Bonjour informal, while current copied HF and prior qualified TT control
do not. This reproduces the prior head-policy sensitivity and disqualifies
unconditional promotion of the raw accuracy/performance winner. A fresh xhigh
AutoFix hypothesis agent inspects source, hashes, actual outputs and prior oracle
evidence without touching devices. Parent owns all serialized experiments.

The first/last layer BFP8 exceptions are measured controls, not assumed fixes:
head8_lofi_edges8 AIME94/100/100 at83.961; head4_lofi_edges8 AIME94/100/100
at84.137. Full shared-suite validation is pending. No stale/fixed-head claim is
made from the prior frozen-hidden oracle; the current autoregressive prefix may
differ. See `AUTOFIX_french.md` for the independent hypothesis ledger.

## Minimal exception and matched token-out controls

First-onlyBFP8 head8 AIME93/100/100 at84.607t/s/u; last-only93/100/100
at84.576. Both shared seven-output suites were run with exact HF controls.
First-only corrects French, while last-only retains the incorrect informal
label. AutoFix records the branch at generated29 before the wrong label34.
The first-only haiku ends immediately after a drafted `(5` at token128, so a
256-token original-prompt HF/TT continuation control is required to distinguish
an unfinished draft from an actual uncorrected count. No quality pass is claimed
while that control is pending.

Matched full32/native262144-cache token-out controls use the exact predecessor
benchmark: prompt128, generate128, five warm requests, three127-step plain
replay windows, no loop host refresh/read/wait and two boundary synchronizations.
All repeated tokens match. Measurements, distinct from teacher forcing:

| Explicit policy | Median warm TTFT ms | Median plain token-out t/s/u |
| --- | ---: | ---: |
| Baseline BF16/HiFi4 head |29.820|83.376|
| Raw BFP8/LoFi head |29.371|85.742|
| Raw BFP4/LoFi head |29.452|85.605|
| BFP8/LoFi head, first layer BFP8 |29.432|85.324|

Exact `tokenout_*.json` and `logs/tokenout_*.provenance.json` retain all samples,
workload shapes, actual runtime policies and counters. Current token-out agrees
with teacher-forcing's slight BFP8-over-BFP4 ranking; prior reduced terminal
measurements used different geometry and do not supersede these matched full
paths. These explicit candidate controls precede final selection; a fresh default
artifact construction benchmark remains required after selection.

## Extended haiku control: first-only policy rejected

`qualitative_first8_haiku256_v1` uses the original shared haiku chat prompt and
fresh pinned CPU HF256 versus TT256. The process exits0 and closes the mesh.
TT repeatedly counts “Patterns in the data flow” as5 syllables with a malformed
breakdown; HF correctly counts its different draft. The initial cutoff was
uncertain, but the continuation confirms an uncorrected numerical-language
error within the observed reasoning. The first-only policy is therefore not a
full-suite winner despite correcting French. AutoFix updates its verdict.
The first/last BFP8 control remains qualified by its full shared128/100 suite;
remaining faster candidates receive current full-suite controls.

HF logs explicitly report a CPU torch fallback for missing optional fast-path
libraries. This is the reference implementation, not a TT inference fallback;
no dependency installation is attempted. Its CPU-active generation completes
normally, so this is not a device hang and no triage/reset is appropriate.

## Runtime advisory classification

UMD reports unknown motherboardB850M-C and uses PCI bus ID as tray ID. The
expected four-chip ring opens, executes and closes on every measured run; this
metadata fallback has no missing device/link symptom. No hardware recovery was
needed. TT allocator's generic “buffers may be corrupted” advisory occurs during
the preserved trace lifecycle, as in the optimized-full-model predecessor.
Selected native execution is run with trace allocation tracking to validate the
actual lifetimes; the warning itself is not evidence of corruption and is not
silently suppressed. Watcher and profiler are never combined; these timings use
plain traces, no device-profiler instrumentation.


## Pre-review precision selection and default-path reproduction

Selected `head4_lofi_last8`: BFP4/LoFi body/head; BFP8 attention and MLP
projections only in layer31; BFP8/LoFi decode QKVG; BFP8 KV; BF16
activation/residual/logits; native producer CCL and FP32 recurrent state.
`selected_precision_config.json` is required by default construction; explicit
`precision_config="baseline"` remains available. No numeric runtime path was
changed after final source formatting. Missing artifact raises immediately.

Repeated paired teacher-forcing medians: baseline82.507, head8 edges83.943,
head4 edges83.894, head4 last-LoFi84.371, head4 last-HiFi284.262t/s/u.
All five selected/HiFi2 samples pass92/100/100. HiFi2 produces exactly the same
seven bounded qualitative outputs as LoFi; fidelity difference is within noise.
Final selected-artifact five-run reproduction returns92/100/100 and84.246t/s/u
with386.687ms teacher TTFT from the same median sample (first/cold teacher
request following prefill readiness). Prefill96/100/100. All99 decode steps
are actual model trace replays. Ten selected repeated samples together have
median84.300 and range84.174..84.517t/s/u. Charts use the median across all
repeated samples for each config. Raw single runs remain in JSON/CSV unchanged.

Historical pending qualitative decisions above are resolved in
`quality_decisions.json` and `AUTOFIX_french.md`. Raw BFP4/BFP8 heads atLoFi
orHiFi2 fail the French control. Head8 first-only fails the extended256-token
haiku control; head4 first-only miscounts a completed draft line. Head8 last-only
fails French. Both edge policies and head4 last-only LoFi/HiFi2 pass the recorded
bounded shared suite. Faster numeric-only candidates are therefore not eligible.

`post_selection_tokenout_v1.json` loads the default selected config with no
precision/head override: native262144 cache, prompt128/generate128, five warmed
requests plus three127-step plain replay windows. Median85.232539t/s/u and
29.448763ms warmed TTFT. Every window has127 model and sampling trace replays,
zero loop input refresh/readback/wait/synchronization; two boundary waits and
one subsequent validation read are separate. This is2.23% faster than the
matched baseline83.375768t/s/u. This is the pre-review post-selection token-out value; the geometry follow-up
and revised default measurement below supersede it for later serving comparisons.

## Selected native capability and anomaly resolution

`selected_native_v1.json` and provenance exit0: normal default model, full32,
B1,262144 cache; logical262143/262144 prefill both execute in43.95s, last
decode at262143 advances262144. Logical131 and2048 prefill traces are resident
before native execution. Final allocation/device: DRAM5107231744 bytes,
L124471040 bytes, TRACE26542080 bytes. `doc/context_contract.json` is recomputed
and preserves262144. Analytical KV16/KV4 alternatives remain explicitly separate
from this selected BFP8 execution; neither is selected or requires a reduction.

Observed anomaly: generic allocator trace-buffer corruption advisory.
Evidence: untracked benchmark logs and predecessor lifecycle controls.
Affected path: preserved persistent trace lifecycle.
Control: selected native run sets `TT_METAL_TRACE_ALLOC_TRACKING=1` and completes
both maximum windows and final decode without a tracker violation.
Resolution: controlled; no stale allocation or corruption identified.

Observed anomaly: native probe AICLK settles1343MHz versus1350 target (within5%).
Evidence: `logs/selected_native_v1.log` line14. Affected path: capacity-only
execution, not Pareto or token-out ranking. Control: the runtime accepts the
settled clock and all capacity/trace checks complete. Resolution: controlled
capacity-run advisory; no frequency-corrected speed claim is made from that run.

## Independent-review follow-up

Fresh xhigh `$stage-review` inspects the live stage. The selected default
seven-output suite reproduces all preselection tokens and pinned controls.
Review asks for head geometry controls at selected BFP4/LoFi: the old BF16
K2 L1 blocker may not apply after shrinking weight tiles. Parent tests full-model
K2 and a fresh AutoDebug/AutoFix source agent adapts the prior real-activation
geometry harness. This finding remains work until controls resolve it.

### Selected-precision geometry source handoff

Fresh source-only AutoDebug writes `AUTODEBUG_head_geometry.md` before the
adapted stage harness. BFP4 32x32 tiles are576 bytes including exponents and
DRAM alignment, reducing C64/K2/R2 predicted static end from1299456 to734208.
The exact existing C16/C32/C64 and R1/R2/R3 family contains27 legal K/grid
points,20 source-feasible including the selected C64/K1/R2 baseline. R1/K2
exceeds the resident-plus-fixed-norm frontier; R1/K4/K8 and R2/K8 are also
source-pruned. The remaining19 alternatives are listed in
`head_geometry_plan.json` without any new speed claim.

`probe_head_geometry.py` explicitly loads the selected BFP4/LoFi policy,
retains BF16 input/output and FP32 destination accumulation, reuses the real
frozen hidden and common two-chunk head, and preserves three-reader per-chunk
padding/trimming. It records full-logit metrics and hashes rather than score
tensor files. C64/K1/R2 baseline and8x4 normalization are explicit even if
production defaults change. Parent owns all hardware execution and any proven
integration. Source-only agent ran Python compile, py310 Black check, and
all27 legal-point arithmetic checks with no torch/TTNN imports; no C++ build
is needed for this Python/docs harness. Device checks remain pending here.

### C16 one-reader dynamic L1 rejection

Parent-run C16/K1/R1 fails before candidate eager completion: predicted and
observed static end1123328 versus dynamic frontier1023232, collision100096.
The required serial-trace retry fails identically after releasing baseline
trace/output, so retained-baseline-trace interference is refuted. The frontier
equals normalized address1301760 minus16384 input bytes and two simultaneous
131072-byte common-head output shards. Both runs close devices normally and
retain exit1 receipts. `geometry_rejections.json` records their hashes and
explicit control/source classification for evidence audit. No implementation
change is made; remaining geometry points continue under the parent.

### Complete selected-head-precision geometry matrix

Parent completed all20 source-feasible points:19 pass and C16/K1/R1 fails
with the independently confirmed common-head allocation bound. Its serial
retry brings total device receipts to21. Together with seven exact source
exclusions, all27 legal points are resolved. `head_geometry_results.json`
and `head_geometry_README.md` retain every geometry, timing, exclusion,
numerical group, replay check and immutable receipt hash. Summary generation
asserts all27 classifications, all19 passing precision/residency contracts,
the12 timing records per passing run (three rounds x64 replays for both
geometries/modes), log hashes, eager/replay checks and group-logit exactness.

C32/K4/R2 is the focused BFP4/LoFi geometry winner: head0.436340 ms and
terminal0.449184 ms versus paired C64/K1/R2 baseline0.814679/0.827854 ms.
This is45.74% less component terminal latency on four Blackhole chips on
physical P300c boards, not a full-model throughput claim. All passing K1
points preserve baseline logits; K2/K4/K8 differ with max error0.125, while
all successful grids/readers within each K group have identical complete
248320-logit hashes. Every eager/replay check and frozen-row top-1/top-5
comparison passes. Parent now owns full32 complete-policy/geometry comparisons,
qualitative review, watcher/default/native validation and final end-to-end
selection. No production or hardware changes by the source/report agent.


## Full-model geometry/precision comparison

The measured head-geometry fix is threaded through `head_geometry` in the
precision artifact and checked against actual TTNN program/shard parameters.
Legacy policies preserve C64/K1/two readers. C32 can directly reuse the final
norm tensor, with ownership-aware deallocation. Selected/source fields no longer
silently accept unused global per-layer overrides or noncanonical layer keys.

All new full-model geometry rows use the main AIME24 chat100 reference and five
traced repetitions, full32, batch1. LoFi last8 C32/K4/two readers passes92/100/100
at87.287860t/s/u; HiFi2 same geometry passes92/100/100 at86.563855; LoFi last8
C64/K2 passes92/100/100 at86.981867. BFP8 head C64/K2 is slower: raw92/100/100
at85.882427, edge exceptions94/100/100 at85.220734. These are measured full-model
comparisons, not projected savings from component time. BFP4 raw C32/K4 passes
93/100/100 at87.391352, so the layer31 exception needs a fresh quality control.
`qualitative_head4_c32_k4_v1` reproduces the erroneous informal Bonjour label;
its exact HF control does not. The current last8 suite is reviewed separately.

Host schema/generator/trace checks:52 pass in `logs/host_tests_geometry_policy_v2.log`.
The initial geometry-policy host command omitted the existing explicit
TORCHINDUCTOR_CACHE_DIR and failed during Torch import because uid1002 lacks a
passwd entry; the corrected command uses the established persistent cache path
and passes. This is a host environment failure, not a TT model test failure.
Source formatting passes in `logs/format_final_source_v3.log`. No C++ or CMake
files changed, so no build is required.


## Revised selected default: final execution receipts

Final artifact: `head4_lofi_last8_c32_k4_r2`. The layer31 BFP8 exception remains
necessary: current rawK4 French still fails, selectedK4 suite passes. All fields,
including actual32-core/K4/two-reader head geometry, propagate from the required
artifact; final default commands have no precision/head override.

`selected_default_repeat5_v2.json`:92/100/100 across five traced runs,
87.114742t/s/u and35.515059ms teacher TTFT from the same warm median sample.
Combined ten selected-geometry samples median87.219974, range87.068..87.322.
Final reproduced performance is the headline, not the slightly higher explicit
candidate87.287860. Prefill readiness remains96/100/100.

`post_selection_tokenout_v2.json` supersedes v1:88.036676t/s/u median plain
no-readback decode,29.001868ms median warmed TTFT. Same native262144-cache,
prompt128/generate128 benchmark, five warm requests, three127-replay windows,
zero loop readback/host refresh/wait; two boundary syncs and one later validation
read are separate. The unadjusted measured throughput gain over baseline
83.375768 is5.59%. Later reports/vLLM comparisons must use this v2 token-out
number, not teacher forcing or the earlier85.233 token-out result.

`selected_native_v2.json` with trace allocation tracking passes logical262143
and262144, last-position decode advancing262144, and131/2048 resident prefill
traces. Final/device DRAM5107215360 bytes, L124471040, TRACE26411008.
`selected_batch32_watcher_v2.json` adds full32-layers,32 active requests,
mixed131/127/3, exactcross-slot/permuted-page logits, seven traced decode steps
and no loop input refresh. Watcher10 (Ethernet watcher disabled) plus allocation
tracking pass and close normally. These instrumented checks do not rank speed.
The recomputed context contract preserves all advertised native capability.

Observed anomaly: startup AICLK advisories also occur in final timing runs.
Evidence: final teacher/token-out logs report1337MHz, baseline token-out1343MHz,
selected native1343MHz, all against unchanged requested1350MHz policy and within
runtime5% tolerance. Affected path: device initialization; clocks are not sampled
inside timed windows. Control/comparison: identical request policy/hardware and
warmed trace workloads, repeated teacher/token-out windows and independent
component/full-model geometry gains. Resolution: controlled operating-policy
variation, reported unadjusted measurements; no clock-normalized improvement is
claimed. The6MHz reported startup difference is~0.45%, below the measured5.59%
end-to-end gain and insufficient to explain the geometry improvement.


## Final host/evidence closure

`logs/host_tests_final_v3.log`:52 policy/generator/trace/prefill tests pass.
`logs/degeneracy_selected_v2.log` and `qualitative_selected_v2/degeneracy.json`:
strict shared-suite artifact/degeneracy checks pass. Manual HF-controlled reviews
remain authoritative for semantic errors that this structural checker cannot see.
`evidence_audit.json` verifies86 serialized immutable run receipts,35 full-model
rows/28 distinct policy-geometry configs, JSON/CSV agreement, final production
source hashes, selected actual policy and native/token-out counters. Two nonzero
geometry receipts are allowed only by exact exit/artifact/provenance hashes in
`geometry_rejections.json`. The initial audit assumed the later assertion-marker
field existed in every historical row; the corrected audit preserves early raw
ledgers and records their source-backed schema explicitly instead of rewriting
old evidence. Final selected rows have full propagation assertions/kernel fields.

Authored source/docs/config pre-commit passes; raw generated text/JSON and logs
remain byte-for-byte unchanged. Python-only changes require no C++ build.
Final independent stage-review and local commit receipts are recorded below.


## Stage review and local checkpoint

Fresh independent xhigh reviewer `/root/review_datatype` returns **clean-pass**
with no required work in `STAGE_REVIEW.md`. The initial more-work-needed finding
reopened BFP4 head geometry; the measured C32/K4 fix, full-model comparisons,
current default performance/accuracy/quality, native capacity and B32 Watcher
controls close it. Stale historical-geometry wording was corrected and rereviewed.

Stage complete: all requested precision, accuracy, performance, capability,
qualitative and artifact gates pass. Applicable authored pre-commit checks pass
(`logs/format_closure_v5.log`). At commit time only trailing-whitespace and
end-of-file-fixer hooks are skipped for immutable generated evidence; authored
source/docs/config were checked separately. No C++/CMake build was needed.
Only this model's runtime precision plumbing, datatype-sweep artifacts and context
contract are included. No vLLM integration, remote push or publication occurred.
Local implementation/evidence commit SHA is recorded by the receipt commit below.


| Repository | Branch | Implementation/evidence commit | Review |
| --- | --- | --- | --- |
| `/home/hous/dev/ornith-1.5-9b/tt-metal` | `hous/ornith-1.5-9b` | `f37cf869ea6f86d640efba6b186b73dc0fbf5c80` | clean-pass |

The implementation commit includes all stage-owned runtime and evidence changes.
Its hooks pass, with only the two documented immutable-evidence whitespace hooks
skipped. The post-commit86-run artifact/source audit also passes. This receipt-only
commit records the implementation SHA; both implementation and receipt SHAs are
stored in `/home/hous/dev/ornith-1.5-9b/state/datatype-sweep-local-commits.json`.
No push is performed.
