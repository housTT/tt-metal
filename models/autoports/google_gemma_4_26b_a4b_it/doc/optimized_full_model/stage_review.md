# Stage Review

Verdict: clean-pass

## Required Work

- None. All required findings from the initial review and subsequent independent
  rereview are closed by inspected implementation, completed controls and
  reconciled final artifacts.

## Other Concerns

- The decoder-stack lower bound uses inherited S=1024 layer measurements,
  whereas the generator benchmark starts with prompt 128 and advances through
  positions `[134,262)`. Its arithmetic is correct and useful as an inherited
  budget, not a same-context measured full-stack decomposition.
- Processed profiler CSVs and watcher files beneath `generated/` match ignore
  rules. The owner has explicitly committed to including stage-owned evidence
  in the local checkpoint. This review precedes the post-clean-pass commit;
  it does not attest that the commit has already been created.

## Hard-Check Gaps

- The original hard checks accepted an incomplete profiler replay. The final
  raw capture was independently checked against both replay inventories:
  every device records 195 model and 12 sampling operations in each session.
- Construction and representative-layer probes did not reveal the complete
  resident model's transient fragmentation. The new full 30-layer P150
  boundary control now closes this specific capacity gap.
- Generator synchronization counters exclude the benchmark's external terminal
  synchronization, which is included in elapsed time. Source confirms no
  per-token synchronization in the no-readback loop.
- Long-context probes establish capacity, shape, position, finite-output and
  overflow behavior, not a full-stack near-limit numerical-accuracy campaign.
  The requested AIME24 numerical gates are separately satisfied.

## Anomaly Ledger

- Observed anomaly: the BF16 embedding initially appeared to require reducing
  P150 context to 49,664.
  Evidence: the original complete 50,623-token probe and candidate 49,792
  fail in `_full_chunked_prefill_attention` at `ttnn.concat(outputs, dim=2)`.
  At 49,792 the requested allocation is 101,974,016 bytes per bank, with
  151,142,336 bytes free but a 101,326,592-byte contiguous maximum.
  Affected path: complete-model long-prefill full attention on P150.
  Control or comparison:
  `final/full_stack_context_tp1/lifetime_fix_50624/` passes all layers 0–29,
  prompt 50,623, final traced position 50,624 and safe following-position
  rejection in 117.360 s, with allocation tracking and strict fallback.
  Likely subsystem: the consumed full Q allocation stayed live while a
  same-sized concatenated attention output was allocated.
  Investigation performed: followed the exception into attention source,
  audited remaining Q consumers and separate chunk-output ownership, compared
  total and contiguous free memory, and inspected the isolated lifetime fix,
  final JSON and JUnit. Q is released after every chunk consumes it and before
  concat; attention math, dtype and layout are unchanged.
  Resolution: fixed. The inherited 50,624-token contract is restored in code,
  tests and context documentation. P150 remains correctly marked unable to
  satisfy HF's 262,144-token advertised context; TP2/TP4 preserve that context.
  Final construction and post-boundary memory figures match their artifacts.

- Observed anomaly: the selected profiler's second replay omitted operations
  and misattributed the large-index TopK.
  Evidence: the superseded capture had 195→151 model operations and 12→10
  sampler operations per device, missing position updates, full-attention
  cache updates, final norm and other operations. Device-report core counts
  also exposed loss before `tt-perf-report` processing.
  Affected path: reduced P150x4 steady-decode performance accounting.
  Control or comparison: canonical `final/profiler_final/raw_ops.csv.xz`
  has 195 model plus 12 sampler operations in both replay sessions on every
  device; the selected window contains 828 rows, 207 per device.
  Likely subsystem: accumulated profiler-buffer exhaustion and reporting
  attribution.
  Investigation performed: compared operation IDs, trace/session IDs, shapes
  and core counts; inspected extraction/report source; re-summed raw and
  processed rows. The corrected run uses a guarded pre-window profiler dump
  and 20,000-program support count. Both PlusOne operations, all four cache
  updates, all five embeddings and all 21 norms are present per device.
  Resolution: fixed. Processed sums independently reproduce 7,657.06325 us
  prefill and 2,591.75575 us decode; host windows are separately reported as
  9.657513 and 3.581892 ms. Trace 1's 279.48 us large-index operation consumes
  the local `[1,1,32,65536]` vocabulary shard and belongs to sampling.
  The two generic TopK operations totaling 44.67 us belong to model-trace
  MoE routing. Final documentation now makes this distinction.

- Observed anomaly: watcher aborts on sampler UINT32 index all-gather.
  Evidence: `final/watcher/reduced_bf16_tp2/console.log.xz` records the BRISC
  included-line-279 assertion, 4,096-byte pages and a 4,352-byte packet.
  Affected path: TP2 sampler transfer during trace setup.
  Control or comparison: fixed reduced TP2/TP4 controls and the complete
  `final/watcher/full_stack_fixed/` all-profile sweep pass, three tests in
  187.31 s with periodic watcher polling.
  Likely subsystem: unused scatter-header initialization receives one chunk
  although the selected transfer uses unicast.
  Investigation performed: inspected both helper constructors, their
  `use_scatter_write` transfer branches, the included fabric assertion and
  failing/fixed watcher output. The new compile-time guard matches the
  transfer condition and does not alter valid data transfer.
  Resolution: fixed for the exercised model paths.

- Observed anomaly: logits-only subtraction used initial position 32.
  Evidence: superseded `final/logits_only/` versus token-out `[134,262)`.
  Affected path: terminal-cost decomposition.
  Control or comparison: `final/logits_only_matched/` uses initial 128,
  device position advancement, five warmups and 128 measured replays over
  `[134,262)` on all profiles.
  Likely subsystem: benchmark configuration.
  Investigation performed: checked all three JSONs, JUnit and the benchmark
  source; recomputed increments of 1.524875, 0.754086 and 0.454791 ms.
  Resolution: fixed. These increments include sampling, token feedback and
  the second trace replay; they are not the entire terminal path.

- Observed anomaly: prefill-to-logits was labeled generator TTFT.
  Evidence: low-level TP4 106.326 ms versus actual public TTFT 138.561 ms.
  Affected path: reported first-token boundary.
  Control or comparison: `final/public_generator_bfp8_control/` and
  `final/public_generator/` use the same 128-token prompt, 128 generated
  tokens, all 30 layers and warmed public path on all profiles.
  Likely subsystem: metric naming and missing baseline.
  Investigation performed: checked all six public JSONs against the runner
  and separated low-level prefill from host-visible first-token timing.
  Resolution: fixed. Public TTFT BFP8/BF16 pairs are
  188.517594/182.831042, 148.851433/147.120236 and
  139.204826/138.561434 ms for TP1/TP2/TP4.

- Observed anomaly: AIME24 free-running TT/HF agreement is 9/100 tokens;
  several shared-suite outputs stop mid-answer.
  Evidence: actual HF/TT completion text and token arrays under
  `final/qualitative/aime24_autoregressive/` and
  `final/qualitative/shared_readiness_suite/`.
  Affected path: greedy text generation.
  Control or comparison: AIME outputs both give coherent English algebra;
  the fresh exact-template six-prompt suite gives coherent, task-aligned
  completions in both implementations. Fixed 64/100-token budgets truncate
  both controls. The haiku end marker also appears in HF.
  Likely subsystem: ordinary autoregressive choice divergence and bounded
  output budgets, not mechanical feedback corruption.
  Investigation performed: read every completion pair, prompt metadata and
  token IDs; verified the retained shared-prompt source hash and pinned model
  revision. Checked task adherence, language, repetition and leakage.
  Resolution: controlled. The newly refreshed shared suite closes the
  previous missing-suite finding; the Fibonacci output is a matched
  budget-limited introduction, not a claim of executed code correctness.

- Observed anomaly: qualitative throughput is approximately 2.79 tokens/s
  while normal TP4 public generation is approximately 49.44.
  Evidence: allocation-tracked qualitative invocation versus normal timing.
  Affected path: diagnostic quality runs versus performance measurements.
  Control or comparison: normal public and token-out runs agree closely;
  qualitative timing is not claimed as optimized throughput.
  Likely subsystem: diagnostic allocation tracking.
  Investigation performed: checked run configuration, counters and boundaries.
  Resolution: controlled measurement distinction.

## Scope Inspected

- Goal/skill paths: supplied stage 07 optimized-full-model contract and
  `.agents/skills/{stage-review,multichip,optimize,tt-device-usage,full-model,tt-enable-tracing,qualitative-check}/SKILL.md`;
  relevant multi-device/performance sections of `tech_reports/LLMs/llms.md`;
  supplied `AGENTS.md` and build wrapper. Stage-review was reread for this
  independent rereview; no additional reviewer was spawned.
- Artifact paths: README, work log, topology, performance summary, provenance,
  context contract; baseline/final profiles; accuracy, matched logits-only,
  public BFP8/BF16 controls, sampler, capacity, long-boundary candidate/fixed
  logs, nonaligned/batch32/mixed-state evidence, shared-suite/AIME outputs,
  raw/processed profiler and watcher files; inherited optimized-multichip
  precision/layout/performance/rejection ledger and earlier terminal trials.
- Code paths: `tt/model.py`, `tt/generator.py`,
  `tests/test_full_model_contract.py`, relevant functional/optimized and
  multichip decoder paths, selected precision policy, sampling implementation,
  both changed CCL helper headers and included fabric assertion; relevant
  Tracy extraction and `tt-perf-report` source.
- Commands run: read-only Git status/diff/check-ignore, `rg`, `sed`, `jq`,
  compressed-log inspection, bounded Docker daemon diagnostic and small
  Python JSON/CSV/XML/hash/reference analyses. Rechecked all nine declared
  final source hashes and the canonical profiler, final-profile, watcher,
  matched-logits, fixed-context and shared-suite metadata hashes. All match.
  Parsed all 81 stage JSONs and inspected the final 14-pass host-unit JUnit.
  No TT device access, reset, server, vLLM or hardware test was performed by
  this reviewer. The only reviewer write is this report.
- Snapshot identifiers: branch `hous/gemma-4-26b-a4b-it`, live stage worktree
  starting at `6740ab49d22`; final model SHA
  `ffe1815c68bd79249a0f98ae260583d44fa7a14d555202db531625d986b0d40d`;
  multichip decoder SHA
  `3f7f9d8f3d17b464730985dd017816d7c0563ea5f47637797e70f35d247382da`;
  canonical compressed profiler SHA
  `6a0c420653dbeb6567f3cfc5d70bfbdf87ad593ec0a8bbb6b24444ef962b1fe3`.
  Remaining source identities are in the verified provenance manifest.

## Residual Risk

- Independently verified token-out results are 37.2565, 46.5319 and 49.7467
  tokens/s for TP1/TP2/TP4, with matched baseline/final regimes and speedups
  1.2183, 1.1306 and 1.0690. The inherited stack-budget gaps are 14.62%,
  8.61% and 4.68%. All six refreshed accuracy rows have 100/100 top-5 and
  top-100 matches. The new lifetime release affects only long-prefill full
  attention, not these short-prompt accuracy/performance paths.
- Inspected code/artifacts preserve fixed slots, mixed prompts, inactive rows,
  changed-only tables, stable trace inputs, semantic greedy and sampled
  top-k/top-p, device token/position/RoPE feedback and nonblocking replay.
  Decoder dtype/fidelity, BF16 KV/activation/CCL, persistent collectives and
  inter-layer layout/rejection ledger remain intact. The measured correct
  sampler alternatives support the selected split contract; no malformed
  sampled path is hidden by force-argmax. No datatype frontier or vLLM path
  was introduced.
- Host C++ compilation remains explicitly unverified because Docker daemon
  socket access is denied, not because the CLI is missing. Repository
  instructions allow this genuine environmental limitation to be disclosed.
  Successful device JIT/watcher gates cover exercised helper paths, not
  every template/topology combination. Accuracy preceding the constructor
  guard is reasonable evidence for unchanged valid transfer behavior, with
  that limit.
- Ethernet watcher instrumentation is disabled under the inherited fabric
  binary-buffer limitation. Worker watcher evidence is real and separate
  from profiling. Reduced-model instrumented host/device intervals are not
  substituted for the normal full-stack throughput results.
- P150x2/P150x4 retain representative sliding/full-layer boundary execution
  plus complete-model allocation evidence and substantial headroom. P150's
  previously tight combined resident/transient boundary is now directly
  exercised with the complete model. This stage preserves, rather than
  reopens, the inherited context and datatype frontiers.
