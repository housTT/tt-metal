# Stage Review

Verdict: clean-pass

Independent review of `datatype-sweep` for `ornith-ai/Ornith-1.5-9B`, after
optimized full model and before vLLM, completed 2026-09-05. The live worktree
was reviewed on branch `hous/ornith-1.5-9b`, based on
`0b926aa04f5e31137d7d55b34a1d710fe5a24c65`. The initial precision-specific
head-geometry finding is resolved by measured controls, implementation changes
and fresh default-path qualification. This final verdict supersedes the earlier
in-progress report.

Selected policy: **head4_lofi_last8_c32_k4_r2**. Final default teacher forcing
is **92% top-1 / 100% top-5 / 100% top-100**, **87.114742 traced t/s/u** and
**35.515059 ms TTFT** from the same median sample. Separate warmed token-out
performance is **88.036676 t/s/u**, **29.001868 ms TTFT**, at batch 1 with
native 262144 cache, prompt 128 and generation 128. These are distinct regimes.
The selected artifact SHA-256 at review is
`146ac866165118e0b4d8d55f4819ea3f4f9a0692141eb13add50ad3a4ba4ba3e`.

## Required Work

None.

The geometry omission is closed: all 27 legal points in the existing
C16/C32/C64, legal-K, R1/R2/R3 family have measured or exact source-backed
dispositions. There are 19 device passes, one allocation rejection confirmed
by a serial-trace control, and seven exact source exclusions. The selected
BFP4/LoFi C32/K4/R2 geometry then passes full-model accuracy, quality, default
reproduction, native-context and batch-32 watcher checks. Different geometries
have distinct configuration IDs; their timing samples are not pooled.

## Other Concerns

None requiring follow-up for this stage. The initially permissive layer-exception
schema now rejects global head/embedding/norm overrides and noncanonical keys.
The selected policy uses the valid string key `31` and layer-local projection
groups. Normal construction consumes the artifact's head geometry, weight and
fidelity groups, layer exception, KV and CCL choices, and fixed dtype/compute
assumptions. Explicit `precision_config="baseline"` preserves the safe baseline,
including its historical C64/K1 head.

## Hard-Check Gaps

- Early baseline and canonical-LoFi compute summaries contain object reprs, and
  nine historical rows lack the later propagation-assertion flag. Their actual
  tensor ledgers and immutable construction sources supply the supporting
  evidence. Later baseline and final selected rows explicitly record actual
  math fidelity and compute flags. No contradictory dtype or fidelity was found.
- Historical geometry fields are normalized from preserved constructor defaults
  and the recorded K2 override. Final rows record and assert the actual program,
  shard shape and complete geometry policy directly.
- This review precedes local checkpoint commits, as the stage-review workflow
  requires. The parent must record the stage-owned commit receipts after this
  clean review before declaring the overall stage complete. This report does
  not claim those commits already exist. No vLLM adapter or push is in scope.

## Anomaly Ledger

### Precision-specific head geometry

Observed anomaly: The old BF16 C64/K2/R2 L1 rejection was used to retain K1 after
changing the head to BFP4/LoFi, although smaller weight tiles reopen the geometry.

Evidence: `../optimized_full_model/head_geometry_summary.json`,
`AUTODEBUG_head_geometry.md`, `AUTOFIX_head_geometry.md`,
`head_geometry_results.json`, `head_geometry_README.md`, and all `geometry_c*`
receipts. C32/K4/R2 measures 0.449184 ms terminal time versus its paired
C64/K1/R2 baseline of 0.827854 ms on the same frozen real model hidden.

Affected path: Common `LMHead1D`, including final norm, input layout and both
head chunks.

Control or comparison: Actual pinned model weights, BFP4/LoFi, BF16 input/output,
FP32 accumulation, fixed norm and 221952 persistent L1 bytes/bank. R3 controls
adapt per-chunk weight padding and output trimming. Larger K changes logits;
same-K successful grid/reader variants have identical complete logit hashes.

Likely subsystem: Weight-tile storage, circular buffers, input/output layout
and accumulation grouping.

Investigation performed: Reviewed native arithmetic, adapted probe source,
all measured dispositions, complete-policy full-model comparisons, and final
normal construction. C16/K1/R1 fails identically after releasing the baseline
trace: static end 1123328 versus frontier 1023232, explained by the input and
two simultaneously live common-head output shards. This rejects that fixed
implementation contract without claiming a universal matmul limitation.

Resolution: Fixed. Artifact-selected C32/K4/R2 is integrated with ownership-aware
norm-buffer reuse and actual runtime assertions. All final v2 gates pass.

### Precision-sensitive generated text

Observed anomaly: Several reduced-head policies pass numerical gates but label
Bonjour informal; first-layer-only alternatives miscount haiku syllables.

Evidence: Actual neighboring suite outputs, `quality_decisions.json`,
`AUTOFIX_french.md`, the 256-token first-layer haiku control, and current
`qualitative_head4_c32_k4_v1/prompt_4/tt_completion.txt`.

Affected path: Autoregressive generation under the complete precision policy.

Control or comparison: Exact pinned HF chat controls and the qualified
last-layer exception. Raw BFP4/C32/K4 remains numerically competitive at
93/100/100 and 87.391352 traced t/s/u but still says Bonjour is informal. The
selected last-layer exception describes Bonjour as good day and Salut as
informal hello, with appropriate question forms.

Likely subsystem: Complete decoder/head precision sensitivity. The old frozen
hidden oracle has a different prefix and does not establish a universal
terminal-only, cache or kernel defect for these runs.

Investigation performed: Read all seven actual final selected outputs and their
HF controls, current raw K4 outputs, relevant neighboring failures and extended
haiku control. Independently verified original tokenizer rendering, prompt IDs,
100/128-token budgets, pinned HF metadata and exact final TT token identity to
the qualified explicit C32/K4 candidate. Read the final qualitative review.

Resolution: Controlled rejection of failing candidates; final selected bounded
suite passes. Most outputs remain reasoning prefixes at the requested budget;
no completed-answer, executable-function or final-haiku claim is made.

### Trace allocation advisory

Observed anomaly: Untracked benchmark and qualitative runs emit the generic
warning that allocations made with an active trace may be corrupted.

Evidence: Final v2 logs; `selected_native_v2` and
`selected_batch32_watcher_v2` provenance and results.

Affected path: Persistent model, prefill and sampling trace lifecycle.

Control or comparison: Native 262143/262144 execution and final-position decode
with allocation tracking; all 32 active requests with mixed 131/127/3 prompts,
worker watcher and allocation tracking. Watcher Ethernet checks are explicitly
disabled; profiler instrumentation is absent from these checks.

Likely subsystem: Generic allocation advisory, not a demonstrated corruption.

Investigation performed: Checked flags, normal completion, actual windows,
allocation views, seven replay steps, exact cross-slot and permuted-page logits,
and current production source hashes. No tracker or watcher violation appears.

Resolution: Controlled. All final instrumented checks pass and close normally.

### Clock and inventory advisories

Observed anomaly: Startup AICLK settles at 1337 or 1343 MHz under the unchanged
1350 MHz request; motherboard discovery falls back to PCI bus IDs.

Evidence: Final timing/native/watcher logs, baseline token-out log and work log.

Affected path: Initialization and inventory metadata; clocks are not sampled
inside the timed windows.

Control or comparison: Same four Blackhole chips on two physical P300c boards,
same requested clock policy, repeated warmed measurements and independent
component/full-model controls. The expected mesh executes and closes normally.

Likely subsystem: Runtime clock tolerance and physical-inventory metadata.

Investigation performed: Checked warning scope, recorded comparison regime and
absence of missing-device/link symptoms. Performance is reported unadjusted.

Resolution: Controlled operating-policy variation. No clock-normalized speed
claim is made; close timing differences retain this limitation.

### Host harness setup failures

Observed anomaly: An initial host test command used an absent test path; a later
command omitted the established Torch cache environment and failed on uid/passwd
lookup during import.

Evidence: Historical host logs and work log; corrected
`logs/host_tests_final_v3.log` reports 52 passing tests.

Affected path: Host validation setup, before useful test execution.

Control or comparison: Correct paths and the existing persistent cache directory.

Likely subsystem: Host command/environment configuration.

Investigation performed: Read failure classification and final test/format logs.

Resolution: Fixed. Failed setup attempts are not counted as passing tests.

## Scope Inspected

- Goal/skills: Original stage contract supplied by the parent;
  `.agents/skills/{stage-review,datatype-sweep,tt-device-usage,qualitative-check}/SKILL.md`.
- Core artifacts: Final README/work log, selected config, selection summary,
  all 35 full-model JSON/CSV rows across 28 precision/geometry policies, candidate
  configs, runtime ledgers, reference provenance and actual reference hash,
  both final Pareto charts, geometry plan/results/rejections and AutoFix reports.
- Final execution: `selected_default_repeat5_v2.json`,
  `post_selection_tokenout_v2.json`, `selected_native_v2.json`,
  `selected_batch32_watcher_v2.json`, `qualitative_selected_v2/`, and
  `../context_contract.json`. The selected policy, native windows and measured
  allocations match the context contract exactly; native context remains 262144.
- Controls: Original baseline refresh and matched token-out controls, repeated
  fidelity/policy alternatives, all final selected text with pinned HF,
  current raw K4 failure, earlier French/haiku failures and extended controls.
- Source: `tt/{precision,model,generator,optimized_decoder,multichip_decoder}.py`;
  stage candidate, qualitative, capacity, performance, summary, provenance,
  validation and geometry runners; common head and predecessor geometry/batch
  helpers. The five final runtime jobs match current production source hashes.
- Host receipts: `logs/host_tests_final_v3.log` (52 passes),
  `logs/format_final_v4.log` (applicable scoped hooks pass),
  `logs/degeneracy_selected_v2.log` (no degenerate output),
  `logs/evidence_audit_final_v3.log` and `evidence_audit.json` (pass).
  Runtime changes are Python-only; no C++/CMake build is required.
- Reviewer commands: Read-only `git status`, `git diff`, `rg`, `cat`, `sed`,
  `tail`, `wc`; standard-library artifact scripts for raw counts, medians,
  policy/geometry normalization, CSV agreement, provenance/source snapshot
  hashes and serialized run intervals; tokenizer-only
  `USE_TORCH=0 python_env/bin/python` prompt/control audits; image inspection.
  An initial ledger audit encountered historical missing fields and was corrected
  for the older schema. Final independent audit verified 86 serialized immutable
  runs, including the two explicitly classified nonzero allocation receipts.
  The reviewer did not run tests, import TTNN, open devices, start servers or
  modify implementation files. Only this review report was written.

## Residual Risk

- Accuracy is token ranking over one 100-position AIME chat reference, not
  dataset-wide mathematical answer accuracy. Shared-suite text checks cover
  recorded 128/100-token budgets; long completed answers remain unestablished.
- Native execution establishes batch-1 capacity and non-aligned support, not
  native-length HF parity, 32 simultaneous native-length requests or million-token
  YaRN execution. Batch 32 is validated at short mixed contexts.
- Startup clocks vary within the runtime's accepted operating policy and are
  not measured during timing windows. Reported throughput is unadjusted, and
  small differences should not be treated as universal speed separations.
- The selected configuration is the fastest evaluated qualified configuration
  in the measured search, not a proof of a global optimum over all policies.
  Future vLLM integration must consume the same public construction path and
  establish its own serving-path gates.
