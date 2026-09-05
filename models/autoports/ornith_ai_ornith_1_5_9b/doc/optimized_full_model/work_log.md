# Optimized full model work log

Started 2026-09-05 from full-model clean-pass, local HEAD `2e4b8f828c`.
Target: pinned Ornith-1.5-9B revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53, native 1x4 TP ring on four Blackhole chips on P300c boards. No vLLM integration or datatype frontier search.

Device discovery: `timeout 60 tt-smi -ls --local` returned all four chips.
Hardware runs serialized. Existing source-backed runtime is available despite generic runner instructions; no dependencies installed.

Baseline: `logs/baseline_v1.provenance.json` records exact command, source and binary hashes; `baseline_v1.json` records warmed full32-layer prompt128/generate128/native262144 measurements. Prior full-model and optimized-multichip policy/rejection ledger retained.

Performance work starts with output collection, terminal layout and full-path resource lifetime. AutoFix source investigation of a traced output-history buffer is running in a fresh xhigh agent.

## Candidate results and AutoFix integration

- `norm_layout_v1`: harness failure, saved real hidden is a dict; corrected
  key access. Mesh closed; `tt-smi -r` exit0 and health log shows four chips.
  `norm_layout_v2` passes exact eager/trace parity; L1 norm saves29us terminal,
  changes logits (max0.75); no selection before all-layer quality gates.
- `norm_smoke_v1` passes reduced real layers0/3,131 non-aligned tokens,8 outputs.
- `norm_teacher_v1`:94/100/100%; `norm_prefill_v1`:95/100/100% top1/5/100.
- `sampler_greedy_v1`: semantic greedy split0.57246ms, force-argmax2.74421ms;
  both eager/trace exactly match CPU greedy for32 distinct rows. Split retained.
- `head_fidelity_v1`: real-hidden BF16/HiFi4 versus HiFi2 versus LoFi,
  alternating order, same selected K1/32768/2readers and sharded norm. Times
  1.3743–1.3746 /1.3726–1.3727 /1.3740–1.3743ms. At most2us terminal saving,
  under0.02% of whole decode, with changed logits. Retain qualified HiFi4;
  no material speedup or inherent lower-fidelity defect is claimed.
- `reference_provenance_check.json` verifies exact pinned reference hash,
  tokenizer/config/template/index hashes, AIME24/chat/100tokens/K100 metadata.
  Same-day completed-stage HF controls are reusable; TT gates are rerun here.
- `output_history_probe_v1`: exact UINT32 indexed_fill/copy/index advance passed
  five windows on every rank, with worker watcher, no profiler. This verified
  the AutoDebug hypothesis before generator integration.
- `generation_contract_v1`: harness-only tokenizer BatchEncoding error before
  prefill. Corrected render-text then encode-token-IDs; bounded reset exit0,
  four-chip health logged. This is not a model failure.
- `generation_contract_v2`: allocation tracker rejected one retained scratch
  buffer before the first collected model replay. AutoFix is checking exact
  ownership/lifetime; tracker remains enabled. Reset log preserved.

## AutoTriage and resumed output validation

`history_tracker_diagnostic_v1` proves retained `gen._history_scratch` buffer8461
was the sole unsafe allocation. The generator now deallocates this fully
rewritten indexed_fill temporary after the recorded copy, within capture.
No allocation-tracker suppression or corruptible marker was added.

`generation_contract_v3` then spent minutes with active host CPU and no visible
harness progress messages. Live AutoTriage captured existing evidence before
termination. `AUTOTRIAGE_history.md` and `triage/history_v3_*` show changing
watcher operation IDs/DONE states before triage, refuting a continuous initial
kernel hang. `verify_before_replay` runs `gc.collect()` twice per decode: the
long paired control test would perform2136 full GCs. GC cost was not measured
separately, so its attribution remains a supported explanation, not a timing fact.

The live capture caused a distinct unrecoverable-by-waiting halt: this checkout's
triage intentionally does not CONTINUE Blackhole cores. See exact source lines
and downstream CCL/dispatch waits in the report. PID91123 was stopped withSIGTERM;
record wrapper exited241(child-15). All evidence retained. Bounded list/reset/list
completed with all four chips. No reboot, second reset, lock clearing, or operator
intervention was needed. The first separate mesh-smoke command omitted the
required TORCHINDUCTOR_CACHE_DIR environment and failed at torch import before
hardware access; a correctly configured smoke remains required. Subsequent
serialized checks use the recorded environment.

The parity harness now writes per-case progress and incremental JSON. The short
watcher/allocation-tracked mode covers greedy/sampled/greedy and device-output
API parity; the separate long mode retains128/260-token and output-window boundary
coverage without the expensive allocation tracker. This changes instrumentation,
not correctness assertions or model execution contracts.

## Output-history closure and trace-lifecycle repair

Configured mesh smoke `mesh_smoke_after_triage_v2.log` passed. The short
`generation_contract_quick_v1` passed with worker watcher and allocation tracker;
the normal long `generation_contract_long_v1` passed exact collected-versus-control
parity for greedy8/128/260, seeded top-k20/top-p0.95/temperature0.8 length128,
and greedy8 again, plus the public device-output API. The128-output case has127
model/sampler/history replays and one final read/wait;260 outputs have259 replays,
three window reads and two index resets at window boundaries. No per-token host
input refresh occurs in autonomous decode. Reduced layers validate orchestration,
not language quality.

`qualitative_norm_v1` produced all six shared-suite outputs but failed while
recapturing for the seventh (AIME) request:108,429,312 trace bytes exceeded the
100,000,000-byte region. The suite preserves weights by replacing public teardown
with a no-op; `_capture` had newly delegated internal trace release to that hook.
AutoFix verified the resulting leak with mocked repeated capture, configuration
change, and live warmup experiments. It separates private trace release from
public teardown; trace capacity is not increased. The required same-lifecycle
hardware check and original qualitative rerun follow.

After the failed qualitative run closed devices, bounded list/reset/list passed
with four chips. `reset_after_qual_trace_leak.log`,
`health_after_qual_trace_leak.log`, and `mesh_after_qual_trace_leak.log` retain
recovery; no process kill, second reset, stale-lock deletion or reboot was needed.

## Final-path candidates and accuracy

`terminal_contract_v2` qualifies native L1 padding plus common LMHead1D with
bitwise real-hidden eager/trace logits and1.3677–1.3679ms terminal time.
`ccl_persistence_v2` qualifies ownership-adapted preallocated embedding and
candidate-gather buffers. `final_contract_quick_v1` passes greedy/sample/greedy
and public device-output parity with watcher/allocation tracking after integration.
`qualitative_final_v1` completes all six shared prompts and AIME24, fixing the
original seventh-request leak. All HF/TT text was read directly; no unexplained
language/register error or mechanical degeneration remains. The128-token shared
windows and100-token AIME window mostly end in reasoning, as do HF controls;
final task completion is not claimed. `qualitative_review.json` and
`degenerate_output.json` retain classifications and zero findings.

`prefill_final_v1`:95/100/100% top1/5/100. `teacher_final_v1`:94/100/100%,
80.99 t/s/u with the readiness teacher-forcing callback boundary.
`full_batch32_final_v1` passes all32 layers and32 mixed prompt slots, exact
cross-slot and permuted-page full logits/tokens, worker watcher and allocation
tracking. The19 host generator/lifecycle tests pass in `logs/host_tests_final.log`.

`perf_final_v1` preserved its three valid no-readback windows (81.72–81.73t/s/u)
and logits-only85.80t/s, but its `perf` dict aliased mutable generator state and
was overwritten by later one-token preparations. It is not a collected-output
headline artifact. The harness now snapshots that dict and asserts its decode
step count after later probes. `perf_final_v2` correctly records127 collected
steps,81.674t/s/u and one final read/wait.

## New-request KV clearing experiment

`request_reset_probe_v1` (real layers0/3,B4,worker watcher) and
`request_reset_full_v1` (all32 layers,B1,native cache) both pass lengths
128,131,63,65,127,129,2047,2049,3. Each control poisons all KV pages with large
finite stale values, permutes physical pages, then checks exact generated tokens
and final logits against the clearing path. Their TTFT columns are diagnostic,
not paired timing claims: new shapes compile in the first control, and poisoning
is queued before the candidate's timer.

The separate fully warmed/synchronized alternating reset comparison in
`perf_final_v2.request_reset_trials` holds prompt128/gen8/native cache fixed.
Clearing TTFTs:47.0733,44.2638,44.0603ms. Retaining overwritten/unused KV:
44.0683,43.6069,43.6716ms. Every output is identical; medians44.2638 versus43.6716ms.
Select `generate` request reset with `clear_kv=False`; explicit public `reset()`
keeps `clear_kv=True`. All hybrid/recurrent/conv state is still reset, and prefill
rewrites each active prefix. Final default performance and remaining final-state
gates are rerun after this selection. No KV dtype, page geometry or capability
limit changes.


## Matched warmed controls and head input geometry

`baseline_repeated_v2` temporarily restored exact completed-stage model/generator
source at2e4b8f828c; the selected files were restored and byte-verified afterward.
Five warmed128/128/native-cache requests yielded medianTTFT47.065316ms and
paired81.551810t/s/u. `selected_repeated_v2` before the head-grid change yielded
47.104038ms and81.675862t/s/u: TTFT was effectively unchanged, so the earlier
short alternating reset trial is not presented as a repeatable request-level gain.
`perf_context2048_selected_v1` measured80.883859t/s/u at prompt2048 and nativecache.
`native_context_selected_v1` passed all32-layer262143/262144 prefill and last-position
decode with allocation snapshots. These artifacts precede the final head-grid
selection and are labeled accordingly.

Fresh AutoDebug inspected the native head program factory and compared BF16/HiFi4
input16/32/64-core families with full221952-byte/bank resident L1 reservation.
The legal three-reader family pads each32768-column chunk to33024 and trims each
chunk independently before returning sampler-width65536. This adapts the contract
instead of rejecting three readers on divisibility alone. The64-core K1/two-reader
family is materially faster with bit-identical real-hidden logits; full-model
integration and final-default validation follow the isolated watcher control.
`AUTODEBUG_head_geometry.md` and `head_geometry_*.json` preserve exact candidates.

`logs/host_tests_selected.log`:19 host generator tests passed with pytest
`--noconftest` and plugin autoload disabled; the test module loads the actual
orchestration AST with fake TT boundaries and never imports TTNN.


Head geometry closure: `AUTOFIX_head_geometry.md` and
`head_geometry_summary.json` record seven precision-locked geometry experiments
plus the failed/repaired watcher lifecycle controls. Selected C64/K1/R2 uses
8x8 input shards [32,64] and output per_core_N16 after unchanged8x4 norm.
Paired terminal medians1.371676 →1.137086 ms and head1.358203 →1.124876 ms;
all248320 real-hidden logits are bit-identical. C64/K2/R2 still collides by30464
L1 bytes; padded3-reader candidates are legal but slower than the winner.
Corrected serial-trace watcher_v2 passes worker watcher and allocation tracking
at221952 persistent L1 bytes/bank. The first watcher failure was a probe-only
cross-trace allocation conflict; no suppression, reset, or model workaround.
Hardware returned to parent for integration/full gates. The final probe now
constructs the32-core baseline explicitly so later model defaults cannot change
its comparison; that maintenance edit has host-only validation.


## Final64-core-head default gates

`selected_head_contract_quick_v1` passes worker watcher/allocation tracking.
`generation_release_v1` passes exact greedy8/128/260 and seeded128 output history
against per-token-readback controls, plus public device-output API.
`full_batch32_release_v1` passes all32 layers/B32 mixed and permuted pages with
worker watcher/tracker; `trace_batch32_release_v1`, `scheduler_release_v1`, and
`cache_release_v1` pass explicit scheduling, inactive rows, changed-only state,
live sampling, partial continuation, external cache and host compatibility.
The scheduler's intentional UINT32-predicate negative control remains recorded
beside the passing selected INT32-predicate implementation; it is not a fallback.

`prefill_release_v1`:95/100/100%; `teacher_release_v1`:94/100/100%,82.573559t/s/u
traced teacher-forcing decode. `qualitative_release_v1` completes the same six
shared prompts and AIME100 through the actual generator. Every HF/TT output was
read; all seven TT completions are byte-identical to the earlier selected-path
`qualitative_final_v1`. The exact prompt/HF metadata is retained, and
`qualitative_release_v1/degenerate_output.json` has zero findings. The bounded
reasoning windows are not a claim that final answers or requested artifacts
complete. Python pre-commit passes in `logs/precommit_python_release.log`.


## Independent review remediation in progress

The reviewer found an unnecessary last-decoder L1→DRAM conversion before the
terminal reconverted to L1. Removed the conversion from decode_forward; the
terminal now receives the returned layout directly after logical reshape.
An exact real-hidden old-boundary/direct control and impacted final reruns follow.
The reviewer also requested the missing fixed64-core K1/readers3 padded control,
which AutoFix is measuring directly against selected64-core K1/readers2.

Evidence-link correction: `logs/host_tests_selected.log` contains16 host-contract
tests, not the19 previously stated. The three lifecycle tests were omitted from
that invocation. `logs/host_tests_release_v2.log` reruns both modules together:
19 pass with `--noconftest`, plugin autoload disabled, and no TTNN import.


Stage-review head reader-family closure: the new `--baseline-cores 64` control
holds C64/K1 fixed and compares R2 with legally padded R3 at full221952-byte L1
residency. `head_geometry_c64_k1_r3_vs_c64_k1_r2_v1` passes exact logits and
repeated/trace checks. R2/R3 terminal medians1.137138/1.309113ms and head
1.125082/1.296014ms retain R2 as the winner. No new watcher is necessary for the
unchanged selection. Hardware returned; AUTOFIX addendum and compact summary
preserve the previously missing eighth geometry experiment.


`terminal_boundary_v1` proves bit-identical real-hidden logits before/after the
removed hop atB1 L1 andB4/B32 DRAM, eager and traced, with221952 L1 bytes/bank
reserved. PairedB1 terminal averages1.140972→1.134565ms; B4/B32 are equivalent
within sub-microsecond variation because their input was alreadyDRAM.
`terminal_boundary_watcher_v1` repeats all12 controls under worker watcher and
allocation tracker (2 replays each), no suppression. Prefill code is unchanged;
prior all-layerB32/scheduler/cache evidence is supported by the exact DRAM-input
boundary control. ImpactedB1 final teacher/qual/perf/native gates are rerun.

Probe maintenance after default promotion: probe_ccl_persistence now constructs
its disabled nonpersistent control explicitly instead of calling the newly
persistent superclass; probe_terminal_contract keeps explicit8x4 normalization
and reshards for the64-core common head. Their historical timings remain tied
to immutable archived sources, not relabeled as measurements of these maintenance
edits. The new terminal-boundary control measures the final default directly.


The direct-terminal final default passes `selected_head_contract_quick_v2` under
watcher/tracker and `teacher_release_v2` at94/100/100%,82.598988t/s/u. All seven
`qualitative_release_v2` HF/TT/prompt files exactly match directly inspectedv1;
the final degeneracy check has zero findings. `perf_release_v2` retains all five
warm samples: medianTTFT47.853787ms versus baseline47.065316ms (small observed
increase; no TTFT improvement claimed), token-out83.308610 versus81.551810t/s/u
(+2.154%). Plain no-readback median83.372643t/s/u; logits-only87.616889t/s.
`perf_context2048_release_v2` gives82.481166t/s/u with one final output read.
No per-token token/position/RoPE/table refresh or readback appears in plain replay.


`native_context_release_v2` passes final32-layer262143/262144 prefill,43.926495s
and43.852780s respectively, with last-position decode262143→262144. Final
post-prefill per-device DRAM5446347776 bytes, L124442880, TRACE13762560.
`doc/context_contract.json` preserves all predecessor evidence and native262144,
adding this exact final probe and persistent history/CCL/trace contract. No
capability reduction or32 simultaneousnative-context claim is made.


Before profiling, bounded health/reset/health and configured mesh open/close all
passed (`health_before_profile_release`, `reset_before_profile_release`,
`health_after_profile_reset_release`, `mesh_after_profile_reset_release`). Four
chips are visible. This also resets after the isolated head allocation/lifecycle
negative probes; no live workload was interrupted, no kill, stale-lock deletion,
second reset or reboot was needed. Profiler runs are separate from watcher.


## Final reduced profiler closure

`tracy_decode_release_v1` and `tracy_prefill_release_v1` both exit0; matching
`render_*_release_v1` commands produce canonical merged/per-rank text/CSV.
Only real layers0/3 plus complete embedding/terminal/sampling path are profiled.
Same decode-run floor/device/host:1.176782/2.431520/2.443457ms/token,11.94us
outside the slowest device.119 model plus37 sampling/history ops per replay;
raw all-rank2496 decode and576 prefill signposted rows are preserved. Sampler/
history0.585688ms is under5% of full-model token-out and has no genericTopK,
argmax, full-vocabulary gather or host feedback. Finaldtype/program/CCL rows
match the selected policy. Prefill's combined eager window is correctly labeled
prefill_and_sampling; its raw hostJSON collection flag only selects decode mode.

`tracy/README.md`, runtime-contract JSON, trace split and head-roofline
classification retain all advice and row metadata. Head weight bandwidth is
96.8% of the installed512GB/s model. Raw~144% head FLOPs is the known eight-worker
report heuristic; native R2 uses16 workers, so corrected modeled utilization is
~72%. MLP R3 uses24 workers, also explicitly classified. No raw row is edited
except removing each first pre-signpost gap in normalized per-rank windows.
The standalone layer+terminal budget12.454681ms exceeds final prompt2048
12.123980ms; no unexplained positive10–15% gap remains. Full32-layer device time
is not invented: perf_summary.json keeps the direct reduced same-run triplet and
full-model end-to-end timing separate, as required by the no-full-stack-profile
rule. The pre-existing viewer process predates this stage; devices are closed.


Prefill-gap AutoFix: `AUTODEBUG_prefill_gaps.md` and `AUTOFIX_prefill_gaps.md`
classify the144-op rank1 profile (3.043997ms kernels +2.071031ms gaps) without
calling between-op gaps mandatory waits. Native host bodies sum0.176745ms but
span4.933206ms; unchanged uninstrumented public128 prefill/sample median is
4.872266ms versus profiled5.263950ms. The exact prepared-body trace passes B1
logical128/131 with changed tokens, reversed pages, allrank hybrid state and
next-decode logits: request medians4.224427/4.628720ms. Prepared eager6.799340/
7.241485ms is slower than public and is not used to inflate a production gain.
One128 trace sample7.201118ms is retained. Separate watcher/allocation tracking
passes both shapes; no suppression or recovery. Hardware returned; integration
must coordinate prefill/decode/sampler trace lifetime, public owned outputs and
full capability gates. `prefill_gaps_summary.json` retains all evidence.


## Reusable prefill integration validation

The generator now owns one eligible prefill shape (fresh owned-cache B1,
logical1..2048) and releases/recaptures prefill/model/plain-sampler/history-sampler
as one family. Device prefill uses persistent token/page inputs and copies
sampler-ready logits into the canonical decode output. Public device-logits
returns are owned clones; high-level generate borrows the canonical buffer and
replays first-token sampling before its caller-visible read. Seed/penalty setup
precedes capture safety checks. Native long prompts, return-all-logits, mixed
batches, continuations, external caches and live shape misses retain validated
eager prefill. No capability or selected dtype/residual policy was reduced.

`host_prefill_integration_v1`:38 host tests pass. `prefill_integration_quick_v1`:
reduced real0/3 greedy8/sampled8/greedy8 and public device replay exact against
the readback control under watcher10, ETH watcher disabled, allocation tracking1.
No suppression, reset or device fault. Runtime source is archived in each receipt.

`perf_prefill_trace_release_v1`: all32 layers/native262144 B1, prompt128/gen128;
warmed TTFT samples33.734852,32.412824,32.250163,32.395030,32.505822ms.
Median32.412824ms pairs with83.316625 token-out t/s/u (one final history read),
plain-no-read median83.377152 and logits-only87.610384. The exact HEAD baseline
remains47.065316ms/81.551810t/s/u. Every warm request proves prefill replay1,
first-sampler replay1, captures0, eager-prefill0, changed page writes0; all127
decode steps prove two traces, history collection and no input refresh.
These are full-model measurements; remaining integrated edges/accuracy/capacity
and final profiler/review remain work. Explicit eager-prefill control is running.

`perf_prefill_eager_control_v1` also passes: explicitly disabling only reusable
prefill tracing gives median46.737011ms and paired83.321479t/s/u, confirming
that the warmed TTFT reduction comes from prefill/first-sample dispatch reuse.
First request after model construction is a separate tradeoff: baseline455.968ms,
new trace738.012ms with two initial prefill captures. This excludes model loading
and is not total cold-start cost. One cached shape means shape changes rebuild
all four traces; the warmed31.1% gain does not describe first-use arbitrary shapes.


`prefill_integration_quick_v2`:16 exact paired cases with worker watcher and
allocation tracker, including logical1/2048, changed tokens/pages, live shape
miss/reuse, owned public logits freed before decode, live seeded/penalty changes,
and four-trace release/rebuild/zero-byte cleanup. `prefill_integration_full32_v1`
repeats13 exact pairs with all32 layers/native cache under the same instrumentation.
The full report's per-rank state hashes exceed the repository500KB file limit;
complete traced/eager lane JSON.gz plus summary are checkpointed in its named
directory, with whole raw JSON/gzip retained and hashed in this workspace.
`prefill_integration_long_v1` repeats all16 reduced cases without tracker timing
distortion and extends sampled/penalized requests to128 and final greedy to260;
all final logits, hybrid hashes, tokens and history-window controls are exact.

`prefill_prefill_trace_release_v1`:95/100/100 top1/top5/top100.
`teacher_prefill_trace_release_v1`:94/100/100,99 traced teacher-forcing decode
steps82.377515t/s/u. Its797.505ms readiness cold-request TTFT is not the warmed
headline. `perf_context2048_prefill_trace_v1`:110.941748ms warmed TTFT,
82.480658 token-out t/s/u, one prefill and first-sampler replay and zero recaptures.
`precommit_prefill_python_v1` passes all applicable Python hooks; this stage does
not modify C++/CMake and needs no build. Native capacity and remaining final
state/qualitative/profiler gates are still running.


`native_context_prefill_trace_release_v1` passes262143prefill/decode-to262144
and262144prefill (43.867097/43.866066s) with the maximum eligible2048 prefill
trace kept resident. Final per-device DRAM5,447,773,184bytes, L1 24,471,040bytes,
TRACE26,542,080bytes within its100MB reservation. `full_batch32_prefill_trace_release_v1`,
`scheduler_prefill_trace_release_v1` and `cache_prefill_trace_release_v1` exit0,
preserving mixed prompts, slots, inactive/live state, continuation and external caches.

Review finding: `qualitative_prefill_trace_release_v1` omitted the old harness's
--sharded-final-norm flag, whose store_true defaultFalse overrode the selected
model defaultTrue. It ran the supported DRAM-norm control, not the measured default;
four text windows differ from selected v2. Preserve this wrong-policy receipt and
outputs, exclude them from final default evidence. The harness now uses
BooleanOptionalAction/defaultTrue, matching the model, and will be rerun with an
explicit --sharded-final-norm flag. No runtime precision/layout fallback occurred.

Review also found duplicated warm B1 hybrid reset:96 recurrent/conv mesh multiply
calls in generate.reset(clear_kv=False), followed by the same96 in private prefill.
A scoped skip-second-only control is being authored before any runtime promotion;
public reset and low-level prefill semantics must remain unchanged.

`prefill_reset_full_v1` exits1 before either reset intervention: the diagnostic
fixture combined scalar temperature with32-list seeds. Common
format_sampling_params wraps all fields when temperature is scalar, yielding a
nested seed list and int(list) in SeedManager. This is a parameter-format/probe
construction failure, not a reset-policy rejection. Runtime and devices are clean;
the fixture will use a coherent explicit temperature lane list and be rerun under
a new immutable label. No common sampler source is changed for this probe.


Final reset promotion uses a private per-call state_already_reset=False default;
generate passesTrue after its existing reset, skipping only the second eligible
traced B1 clear. No model, graph, cache allocation or public/eager behavior changes.
`prefill_integration_full32_v2` passes13 exact full32/native pairs under watcher10
and allocation tracker1, with zero TRACE after both teardowns. `host_prefill_reset_v1`
passes39 checks (the expected unknown-timeout config warning comes from deliberately
disabled pytest plugin autoload); applicable Python hooks pass in
`precommit_prefill_final_python_v1`. Native/B32/external/public-prefill receipts
above still exercise unchanged paths; independent review checked the three-hunk diff.

`perf_prefill_trace_release_v2`: five warmed native full32/128/128 TTFT samples
29.750481,29.550729,29.720443,29.523698,29.586688ms. Median29.586688ms pairs with
83.314796 token-out t/s/u; plain median83.377370, logits-only87.612478.
Versus the exact completed full-model baseline:37.13696% lower TTFT and2.16180%
higher token-out throughput. First request after construction743.741207ms remains
separate from warm timing and excludes loading. Every warm request proves prefill
and first-sampler replay1, captures/eager calls/page updates0; all127 decode steps
use model+history sampler traces, one final output read/wait and no input refresh.


## Final evidence closure

`perf_context2048_prefill_trace_v2`: one warmed request108.333826ms TTFT and
82.482439t/s/u (12.123793ms/token), within the10.687848ms layer-stack estimate
plus1.766833ms measured terminal/embedding/sampling budget. `teacher_prefill_trace_release_v2`
keeps94/100/100 top1/top5/top100,99 traced decode steps82.488863t/s/u; its cold
readiness TTFT789.652ms is not the warmed headline. `prefill_integration_long_v2`
passes all16 exact pairs with final private reset policy and260-token history.

`qualitative_prefill_trace_release_v2` uses explicit --sharded-final-norm and
records actual selected norm/head/trace/cache policy. All seven prompt/HF/TT text
files were directly reread and match prior selected qualitative_release_v2 bytes.
French greeting/register and AIME walking-time reasoning remain correct within
the fixed128/100-token windows; completed final answers are not claimed. The
shared file-path degeneracy checker (`degenerate_prefill_trace_release_v2`) exits0
with no findings. An initial checker --help attempted via package -m encountered
this host's missing UID username while importing Torch; the checker source explicitly
requires direct file invocation, which was used for the actual successful check.
No additional device process was opened by that failed help request.

`tracy_decode_prefill_trace_v1`, `tracy_prefill_prefill_trace_v1` and matching
render commands exit0; `profile_details_prefill_trace_v1` and summarize_final_v3
produce final accounting. Reports under tracy/prefill_trace_release retain all
2496 decode and580 prefill signposted rows across four independent device clocks.
Decode maxrank2.430452ms versus same-runhost2.440547ms leaves10.095us outside
device;119 model and37 sampler/history operations per token. Rank0 sampling/history
0.587166ms remains under5% of full token-out; no genericTopK, ArgMax, full-vocab
all-gather or per-token host input/readback path is present.

The final prefill signpost now spans actual warmed generate128/1, including
request setup, first-token read and post-first-token position/page/history setup:
4.634666ms wall,4.465412ms TTFT, maxdevice4.343019ms. Its8 eager request-boundary
ops precede103 prefill-trace and34 sampler-trace ops. The .981019ms gap attributed
to eager rows is not a stalled captured graph: .757561ms precedes seed tilize and
includes configuration, RNG admission, validation and uploads. Only about .114ms
subsequent seed-merge gaps belong to the four UINT32 merge operations. Potential
future host-admission/upload coalescing must preserve both RNG admissions and
unseeded entropy consumption; no .981ms savings or physical-mandatory-cost claim
is made. These minor request-boundary opportunities do not dominate token-out.

Final profiles confirm selected dtype/fidelity/matmul/CCL contracts. Head weight
bandwidth is497.18GB/s (97.11% of installed512GB/s model); native16 compute workers
correct the report's eight-worker raw FLOPs heuristic to71.93%. Raw reports stay
unchanged, with only pre-signpost first-gap exclusion in normalized rank windows.
The expected motherboard/viewer/Pandas/allocation advisories are preserved; all
required custom captures/timing rows exist, separate watcher/tracker gates pass,
and no profiler overflow or hardware recovery occurred in these final runs.


## Final checkpoint checks — 2026-09-05

`precommit_checkpoint_v2` runs repository pre-commit hooks on all542 applicable
staged files outside immutable generated qualitative directories and raw text;
all applicable hooks pass. Python formatters, size check, Metalium includes and
global Torch-import checks pass. All781 initially staged files are below500000
bytes (32,189,108 total bytes). This stage changes Python and documentation;
no C++/CMake build is needed. The final implementation still matches the measured
source hashes; the existing39 host tests remain applicable.

The first lint attempt, `precommit_checkpoint_v1`, only added final newlines to45
generated qualitative metadata JSON files. Each change was verified to be exactly
the original bytes plus one newline with identical parsed JSON, then reverted to
preserve raw artifact hashes. Raw prompt/completion text is also preserved exactly.
`git diff --cached --check` reports five expected trailing spaces in captured HF
AIME completion text; the same check excluding immutable qualitative text passes.
No runtime or authored Python was changed by either lint run. Exact commands,
exit codes and source snapshots are in the corresponding provenance files.

The installed local commit hook runs the remaining checks across the complete
checkpoint with `SKIP=trailing-whitespace,end-of-file-fixer`; those two mutating
hooks already passed for authored files in the scoped lint run. This preserves
captured text and metadata byte-for-byte without changing repository hook policy.


## Independent review and local checkpoint — 2026-09-05

The fresh-context xhigh reviewer returns **clean-pass** in
[STAGE_REVIEW.md](STAGE_REVIEW.md), with no required work remaining. The reviewer
independently checked the final runtime, exact controls, actual generated text,
source/log hashes, per-chip CSV accounting and the final acceptance contract.
All findings were fixed or resolved with controls and rereviewed. The report
explicitly retains cold/shape-rebuild costs, qualitative token-window limits,
native capacity versus long-context accuracy, and small future seed-admission
opportunities. Context capability is unchanged and stage status is complete.

Final runtime SHA256 remains:

- generator.py: `dc91559ec9e08b5a412ee59e02c9831be2abd82c7000634b6ad54078d697c736`
- model.py: `3b741a166737b8d76eda9ab2117070f5f84d41d406b97a79914a5128745116ab`

Only stage-owned model Python, context documentation and optimized-full-model
evidence are checkpointed. No vLLM integration, datatype frontier, C++/CMake
changes or push is included. The implementation/evidence commit SHA is appended
in the following documentation receipt commit; that receipt's own SHA is recorded
outside the worktree and in the final handoff to avoid a self-referential commit.
