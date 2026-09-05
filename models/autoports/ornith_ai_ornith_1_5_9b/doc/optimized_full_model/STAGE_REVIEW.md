# Stage Review

Verdict: clean-pass

Independent review of optimized-full-model for pinned
ornith-ai/Ornith-1.5-9B, revision489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Reviewed the live hous/ornith-1.5-9b worktree above completed full-model
HEAD2e4b8f828cb0d6d83da5e608df5290f622c5de98, before the stage checkpoint.
The supplied stage name is authoritative; no orchestration number was supplied.
This reviewer performed local read-only inspection and artifact analysis, with
no TTNN import, device access, hardware reservation, reset, server or vLLM run.
Only this review file was written.

## Required Work

None. The findings raised during independent review have been fixed or resolved
with controls, and the final evidence supports this stage's acceptance contract.
Local checkpointing and recording its SHA follow this verdict under the stage
workflow; this verdict does not certify a later serving or release stage.

## Other Concerns

- The warm improvement is specific to reuse of the measured prompt shape.
  Five full-model B1/native-cache requests give median TTFT47.065316→29.586688ms
  and paired token-out81.551810→83.314796t/s/u. First request after model
  construction increases455.968→743.741ms, excluding model loading. One cached
  prefill shape means a shape change rebuilds the four-trace family. The README
  now reports this cost beside the warm result.
- Request seed-upload coalescing remains a small, unmeasured optimization
  opportunity. Both existing host admissions must be preserved: explicit seeds
  reset to the same first draw, while an unseeded second admission consumes fresh
  entropy. Public partial-prefill lane preservation remains necessary. The
  measured0.981019ms attributed to eager-row gaps is not all removable seed-merge
  work;0.757561ms precedes the first merge operation and includes other host
  preparation. No physically mandatory-wait or0.981ms-saving claim is accepted.

## Hard-Check Gaps

- Native262144 execution validates capacity and cache/position operation with
  synthetic token IDs. It does not establish long-context task accuracy. B32
  correctness is at context2048, not32 simultaneous native-length requests.
- Shared qualitative evidence is bounded to128 generated tokens, with100 for
  AIME. HF and TT commonly remain in reasoning. Completed haiku, story, code,
  translation response or AIME-answer success is not inferred from these windows.
- Full32-layer device profiling is deliberately absent under the optimize skill.
  The report preserves the real reduced-path same-run floor/device/wall triplet
  and separately measured full-model wall time. No full-stack device time is
  synthesized. Fixed-window EOS handling and future serving/API behavior remain
  outside this stage's claim.

These are explicit evidence limits, not missing requirements of the supplied
optimized-full-model contract.

## Anomaly Ledger

- Observed anomaly: the last decode boundary still moved hidden state to DRAM
  despite the claimed direct L1 terminal path.
  Evidence: inspected model.decode_forward and the original audit claim;
  [boundary control](terminal_boundary_v1.json) and
  [watcher control](terminal_boundary_watcher_v1.json).
  Affected path: final decoder→norm/head.
  Control or comparison: old hop versus direct B1 L1 and B4/B32 DRAM inputs,
  all248320 logical logits, eager and traced.
  Likely subsystem: unnecessary tensor-memory conversion.
  Investigation performed: independent source finding, explicit real-hidden
  comparisons, watcher/allocation checks and final runtime rows.
  Resolution: fixed; the final decode source directly reshapes the returned
  hidden state, and final profiler rows show the selected L1 terminal sequence.

- Observed anomaly: the head search lacked an isolated64-core K1 two-reader
  versus three-reader comparison; an earlier comparison changed K as well.
  Evidence: [head AutoFix](AUTOFIX_head_geometry.md) and
  [fixed-K reader control](head_geometry_c64_k1_r3_vs_c64_k1_r2_v1.json).
  Affected path: dominant BF16/HiFi4 LM-head chunks.
  Control or comparison: same real hidden state, precision and resident L1
  reservation; R3 adapts physical columns to33024 and slices logical32768.
  Likely subsystem: DRAM reader/output-shard geometry.
  Investigation performed: checked raw exactness/timing,16/32/64-core controls,
  and the separately measured30464-byte K2/R2 L1 collision.
  Resolution: controlled; adapted R3 is bit-exact but15.1% slower. C64/K1/R2
  remains selected. Larger-K rejection is backed by the actual allocation
  contract rather than an unadapted API failure.

- Observed anomaly: earlier output-history storage and repeated shared-runner
  recapture exposed allocation-tracker/lifetime failures and trace-region growth.
  Evidence: [output AutoFix](AUTOFIX_output_collection.md),
  [lifecycle investigation](AUTODEBUG_trace_lifecycle.md), tracker diagnostics,
  and [final whole-model integration](prefill_integration_full32_v2/summary.json).
  Affected path: persistent token collection and trace ownership.
  Control or comparison: exact UINT32/history-window tests, eight recaptures,
  repeated generation, changed inputs and explicit teardown.
  Likely subsystem: temporary buffers retained behind older captures and
  internal cleanup routed through an overridden public teardown hook.
  Investigation performed: inspected capture/deallocation order, private release,
  common CCL ownership and final full32 watcher/allocation-tracker results.
  Resolution: fixed; scratch is released inside capture, internal release owns
  all four traces, prefill is captured last into canonical logits, and both
  full-model comparison lanes finish with zero allocated TRACE bytes. No tracker
  suppression or corruptible-buffer exception is introduced.

- Observed anomaly: the original eager-prefill profile had2.071031ms of gaps;
  narrow native host-function durations did not explain them.
  Evidence: [gap investigation](AUTODEBUG_prefill_gaps.md),
  [capture control](AUTOFIX_prefill_gaps.md),
  [integration proof](AUTOFIX_prefill_integration.md), and final profiler CSVs.
  Affected path: prefill/first-sampler dispatch and request setup.
  Control or comparison: unchanged public uninstrumented window, prepared eager
  versus trace at128/131, changed tokens/reversed pages, exact hybrid state and
  next-decode logits, followed by full-model controls.
  Likely subsystem: avoidable host submission work, with instrumentation and
  ordinary request preparation also contributing.
  Investigation performed: rejected the mandatory-wait explanation, inspected
  the integrated bounded shape cache and four-trace lifetime, and rederived
  final warm performance and per-rank accounting.
  Resolution: fixed/controlled; reusable prefill and first sampling are selected.
  Remaining request-boundary gaps are separately classified. The final public
  generate128/1 profile includes more work than the historical private-prefill
  window and is not presented as a direct same-scope comparison.

- Observed anomaly: high-level generation cleared the same B1 hybrid buffers
  twice, adding96 redundant mesh multiply calls.
  Evidence: [reset AutoFix](AUTOFIX_prefill_reset.md) and
  [paired full-model control](prefill_reset_full_v2.json).
  Affected path: request reset followed by fresh traced prefill.
  Control or comparison: skip only reset call2 within a warmed request; twelve
  original/changed/reversed-page cases compare tokens, logits, all-rank hybrid
  state, feedback, positions, RNG and penalties.
  Likely subsystem: duplicated request-boundary work.
  Investigation performed: inspected shared B1 layer objects and common seed
  semantics; rederived alternating-arm timing; reviewed the final private flag.
  Resolution: fixed;2.74–2.85ms saved in the isolated128/131 control. The private
  per-call flag skips only the traced B1 duplicate. Public prefill/reset, eager
  paths, graph and persistent allocations remain unchanged. Final full32
  watcher, long-generation, teacher and performance checks pass.

- Observed anomaly: four new qualitative completions differed from the selected
  control despite exact prefill integration results.
  Evidence: actual text and command provenance in
  qualitative_prefill_trace_release_v1; inspected run_qualitative.py.
  Affected path: qualitative harness final-normalization selection.
  Control or comparison: the command omitted the sharded-norm flag and the
  harness's old default explicitly selected DRAM normalization. The corrected
  [final suite](qualitative_prefill_trace_release_v2/qualitative_review.json)
  uses selected sharded normalization.
  Likely subsystem: harness default/command mismatch, not unexplained trace drift.
  Investigation performed: directly read the changed text, compared token IDs
  and archived commands, and checked the corrected runtime-policy metadata.
  Resolution: fixed; the harness defaults to the selected norm and records actual
  policy. All seven final TT completion files are byte-identical to the previously
  reviewed selected control. The wrong-policy run is retained and excluded from
  final selected-path evidence.

- Observed anomaly: raw head modeled FLOPs exceed100%, and some decoder rows
  retain SLOW/generic-fidelity advice.
  Evidence: final raw matmul attributes and
  [head classification](tracy/prefill_trace_release/head_roofline_classification.json),
  plus the predecessor's precision-locked coherent-family ledger.
  Affected path: performance-report interpretation and geometry selection.
  Control or comparison: report heuristic assumes8 workers; native R2 uses16
  and R3 uses24. Final head weight bandwidth is497.182GB/s,97.106% of the installed
  512GB/s model; corrected modeled compute utilization is71.930%.
  Likely subsystem: reporting heuristic and generic advice.
  Investigation performed: checked source-backed worker counts, raw dtype/fidelity/
  program rows, adapted geometry controls and retained decoder-family measurements.
  Resolution: controlled; raw rows remain intact. Generic advice does not override
  the user's selected real-model policy or its measured rejection ledger.

- Observed anomaly: deliberate UINT32-predicate negative controls fail; a reset
  probe's mixed scalar-temperature/list-seed fixture failed before its arms;
  tooling/logs contain allocation and environment advisories.
  Evidence: scheduler_prefill_trace_release_v1.json, failed reset-probe receipt,
  corrected reset control, runtime_audit.md and work_log.md.
  Affected path: lane-merge control, harness construction and tool diagnostics.
  Control or comparison: INT32 predicates preserve the complete UINT32 values;
  coherent per-lane sampling parameters pass the common formatter; final worker
  watcher/allocation checks and direct degeneracy-check invocation pass.
  Likely subsystem: known predicate dispatch, common parameter normalization,
  allocation diagnostics and host tooling.
  Investigation performed: read the negative and positive outputs and relevant
  common source; distinguished historical failed fixtures/help invocations from
  final hardware gates. The documented ETH watcher exclusion is retained without
  claiming Ethernet watcher coverage; final profiling is separate from watcher.
  Resolution: controlled; failed receipts and advisories are preserved, with no
  unexplained current model-output or device-health failure.

## Scope Inspected

- Goal/skill paths: stage_contract.md; ../../AGENTS.md; repository AGENTS.md;
  .agents/skills/stage-review, multichip, optimize, full-model,
  tt-enable-tracing, qualitative-check and tt-device-usage SKILL.md files.
- Code: ../../tt/model.py and generator.py, their complete stage diff, changed
  ../full_model/test_generator_host_contract.py, new trace/lifecycle/reset tests,
  stage runners/probes/profiling/manifest scripts, relevant common LMHead1D,
  LazyWeight, sampling/SeedManager and native program/report source. Decoder
  implementation files retain their predecessor hashes and policy.
- Correctness: final AIME prefill95/100/100 and teacher94/100/100 over100 reference
  positions; final full32 watcher13 exact comparisons; reduced long/edge16 exact
  comparisons including1/2048, sampled/penalized128 and greedy260; B32 mixed
  131/127/3 prompts, inactive slots, live reconfiguration, seed/partial-prefill
  controls, external-cache warmup, continuation127+3 and explicit reset.
- Capability: native_context_prefill_trace_release_v1 executes262143 and262144
  prefill with a maximum2048 prefill trace resident. Last-position decode advances
  262143→262144. Final recorded allocations are5,447,773,184 DRAM,
  24,471,040 L1 and26,542,080 TRACE bytes/device. The final generator differs from
  that receipt only by the reviewed private duplicate-reset guard; the tested
  public long/native, external-cache and B32 execution branches are unchanged.
- Output inspection: actual HF/TT text for all six shared prompts and AIME,
  rendered chat prompts, exact checkpoint/template/reference metadata, wrong-policy
  control and corrected final text hashes. French register/translation reasoning,
  thermodynamics, Fibonacci and AIME setup are coherent within the stated windows.
- Performance: rederived five-run median TTFT and paired throughput from
  baseline_repeated_v2.json and perf_prefill_trace_release_v2.json. Checked
  plain83.377370t/s/u, logits-only87.612478t/s and teacher82.488863t/s/u as separate
  workloads. The single warm2048 request is108.333826ms/82.482439t/s/u. Every warm
  headline request proves prefill1/sampler1/capture0/eager0; its127 decode steps
  have no input refresh and one history read/wait.
- Profiling: independently summed all eight final per-rank CSV windows, checked
  normalized hashes and all2496 decode/580 prefill rows. Reduced decode is
  1.176782ms optimistic bandwidth floor,2.430452250ms slowest-device time and
  2.440547018ms wall time. Sampler/history is0.587165500ms on rank0, below5% of
  full-model token-out. Final raw rows preserve the selected decoder precision,
  BFP8 decode QKVG, FP32 recurrent math and BF16/HiFi4 head; no generic TopK,
  ArgMax, full-vocabulary gather or host feedback appears in measured decode.
  Public-prefill wall4.634666024ms, TTFT4.465411999ms and device4.343019ms retain
  their distinct boundaries. Per-phase gap sums include incoming handoffs.
- Budget:24×0.355672+8×0.268965=10.687848ms standalone stack; terminal/sampler/
  embedding gives12.454681ms versus12.123793ms measured at context2048. The
  standalone DRAM/trace-boundary caveat is explicit; this is an additive planning
  estimate, not a strict physical lower bound. No positive10–15% unexplained
  token-out excess remains.
- Evidence integrity: verified key decompressed log hashes, compressed source
  snapshot hashes and source contents against receipts; final runtime hashes
  match the final full32, performance, teacher, qualitative and profiler runs.
  The baseline runtime snapshot matches HEAD2e4b8f828c exactly. Full32 split JSON
  lanes/summary reconstruct their complete raw report. The inspected1078-entry
  manifest contains776 checkpoint and302 workspace-only artifacts; all indexed
  files exist and hashes matched before addition of this review/final lint
  receipts. Manifest refresh after those additions is a packaging step.
- Verification logs:39 host tests and applicable Python pre-commit hooks pass.
  No C++ build is required for these Python/documentation changes. Generated
  qualitative metadata bytes are retained as evidence; the final EOF-hook handling
  preserves their original hashes rather than silently rewriting the receipts.
- Commands run by this reviewer: read-only rg, sed/cat, git diff/show/status,
  and Python standard-library JSON/CSV/gzip/hashlib/statistics analysis. Hardware
  results above were independently inspected, not rerun by the reviewer.

Final reviewed runtime SHA256 values:

    generator.py dc91559ec9e08b5a412ee59e02c9831be2abd82c7000634b6ad54078d697c736
    model.py     3b741a166737b8d76eda9ab2117070f5f84d41d406b97a79914a5128745116ab

## Residual Risk

This pass covers the supplied TP4 optimized-full-model stage on four Blackhole
chips on two P300c boards. It does not establish TP1/TP2 full-model performance,
32-way native-context capacity, completed long-form answer quality, million-token
YaRN execution, production serving/streaming, or future datatype-frontier choices.
Those broader bringup/release requirements remain assigned to their later stages.
Within the reviewed contract, no required gate is failing, no selected runtime
fallback contradicts the claim, and no material anomaly remains unexplained.
