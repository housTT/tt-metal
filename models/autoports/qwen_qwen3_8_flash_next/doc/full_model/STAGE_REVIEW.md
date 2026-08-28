# Stage Review

Verdict: clean-pass

## Required Work

None.

## Other Concerns

- The accepted watcher runs still emit deprecated CCL-argument warnings. They
  are runtime/API migration debt, not evidence of a model fallback or device
  integrity failure: the focused and all-48 watcher tests pass with zero
  watcher assertions, and the generic active-trace allocation warning is
  separately resolved by tracker-enabled reruns.
- Exact host-backed service is the dominant token-out cost. The final
  prompt-128/generate-128 artifact measures 0.608555 s/token (1.643238 t/s/u),
  while a representative final submit includes 0.397370 s expert service.
  This is an explicit measured limitation of the exact bounded host-store
  policy, not hidden CPU model math. Sampling is approximately 0.535 ms and is
  not dominant.

## Hard-Check Gaps

- This checkout has no
  `models/common/readiness_check/check_degenerate_output.py`. The stage records
  that absence and uses its local token-degeneracy checks plus direct human
  review of exact HF/TT text and raw tokens. The replacement covers dominant
  token fraction, adjacent repeats, repeated four-grams, language, coherence,
  topic drift, and the shared qualitative suite; it does not conceal visible
  output wrongness.
- A full 262,144-token, all-48-layer prefill was not repeated in this stage.
  The supported-context claim is instead tied to byte-exact full-model
  capacity arithmetic, a successful all-48 construction at 262,144, inherited
  public per-layer full/near-max non-aligned prefill evidence, a max-position
  QSA trace, and a non-aligned 201-token public full-model generator run. No
  context reduction is advertised.

## Prior-Finding Closure

- Active-trace allocation warning: closed. `AUTODEBUG_TRACE_ALLOC.md` traces
  the warning to the intentional multi-live-trace allocation protocol and
  records the exact tracker commands. Both
  `reduced_split_trace_alloc_tracker.xml` and
  `full48_tokenout_trace_alloc_tracker.xml` pass with
  `TT_METAL_TRACE_ALLOC_TRACKING=1`; neither raises the runtime tracker's
  unsafe-live-allocation error. Source inspection confirms retained crossings
  are regenerated before consumption and explicitly marked corruptible, while
  state snapshots are restored and freed before replay.
- Missing shared qualitative suite: closed. `qualitative_prompt_format.json`
  records the exact revision, `Qwen2Tokenizer`, chat-template hash, prompt
  mode, rendering method, suite hash, prompt ids, and 128-token greedy policy.
  `qualitative_shared_suite.refpt` contains the rendered prompts, prompt token
  ids, and fresh exact HF controls. `qualitative_shared_suite_final.json`
  contains all three traced TT runs, raw HF/TT tokens, text, divergence,
  metrics, and fallback audits. `QUALITATIVE_REVIEW.md` reviews explanation,
  coding, and summarization concretely. Explanation/coding are truncated at
  the allowed 128-token ceiling on both HF and TT; summarization completes
  correctly. No output shows wrong language, topic drift, prompt corruption,
  or a mechanical loop.
- Non-greedy trace and seed control: closed.
  `non_greedy_split_trace_final.xml` passes a real reduced GDN/PLE/QSA stack
  with top-k 4, top-p 0.95, temperature 0.8, and seed 12345. The test proves
  current top-k membership, changing device seeds, direct `tt_out_tok`
  feedback, positions 4 through 7, unchanged and changed page-table behavior,
  zero token/position refresh after capture, and sampled -> greedy -> sampled
  trace release/rebuild. The per-token seed publication is explicit compact
  sampling control into the existing persistent seed buffer; it is audited and
  absent from the canonical greedy measurements.
- Full-model batch evidence: closed.
  `full48_batch32_eager_fixed_slots.xml` constructs the real all-48 model at
  batch 32/context 4096 and passes active endpoint slots 0 and 31, 30 inactive
  rows, logical prompt lengths 1 and 33, distinct/flipped page rows, two
  on-device-sampled eager tokens, repeat/reset with new request IDs, exact
  position/page ownership, inactive EOS histories, active PLE history carry,
  deterministic active outputs, and no prohibited host-work audit flags. The
  conservative plan is 23,558,946,904 bytes/die with 10,666,573,736 bytes/die
  headroom. Batch-1 segmented tracing and batch-32 eager low-level capability
  are stated separately.
- Reproducibility/provenance concerns: closed. The work log's HF command now
  uses only supported CLI arguments; current profiler provenance reports
  0.608555 s/token, 0.397370 s expert service, and 0.000887 s PLE service; the
  host contract distinguishes the final 13-pass current-source matrix from the
  inherited 32-pass matrix; and
  `aime24_autoregressive_100_report_final.json` retains exact 100-token HF/TT
  ids, metrics, degeneracy counters, and the complete fallback audit.

## Contract Findings

- Full HF path and generator: `tt/model.py` loads token embedding, all 48
  optimized host-backed `MultichipDecoder` layers, final hyperconnection
  norm/mixer, untied vocabulary-sharded `LMHead1D`, and `Sampling1D` without
  transient full expert/PLE residency. `tt/generator.py` provides the standard
  `build_generator(model_dir, mesh_device, **kwargs)` entry point, explicit
  `enable_trace`, model-owned cache state, low-level compile/prefill/decode and
  token-out methods, high-level generation/chat, and explicit host-sampling
  compatibility mode. No vLLM implementation or registration exists.
- Optimized multichip contract: stack ingress fractures once to BF16
  `[1,1,4*M,1280]`; every layer preserves that ABI; stack exit gathers once.
  The inherited expert BFP4/LoFi, shared BFP8/LoFi, GDN BFP8/HiFi2, QSA
  BF16/HiFi2, BFP8 KV/index-cache policy, selected 1D QSA roles, BF16 CCL
  partials, two links, 8192-byte payload, and `FABRIC_1D` remain in the actual
  model construction. The rejected `gdn_qkv_b_a@0:55` geometry is localized by
  real progressing-HF evidence rather than replaced with a single-chip,
  replicated, CPU-projection, or broad precision fallback.
- Context: `context_contract.json` and `host_weight_contract.json` agree on
  10,170,438,744 planned bytes/die at the advertised 262,144-token context
  against 34,225,520,640 bytes/die, leaving 24,055,081,896 bytes/die. The sum
  includes 4,758,875,136 non-expert weight bytes, 1,459,814,400 bounded expert
  slots/staging, 2,340,421,632 KV/index-cache bytes, 469,630,976 runtime state,
  67,135,576 endpoint runtime bytes, 819,200 PLE staging bytes, and a
  1,073,741,824 trace reserve. The arithmetic re-derives exactly and the
  all-48 construction artifact passes.
- Non-aligned/mixed prompts: public prefill accepts ragged or padded logical
  lengths and owns internal 128-token chunk padding, page inputs, cache fill,
  masking/tails, positions, and slicing. Existing full-model evidence includes
  the non-aligned 201-token AIME chat prompt; reduced real-weight evidence
  covers lengths 1 and 33 plus an inactive fixed row; the batch-32 gate repeats
  the mixed endpoint-slot case on all 48 layers.
- Accuracy and generation: the fresh AIME reference is exact revision/tokenizer
  chat-template evidence with 201 prompt tokens, 100 greedy reference tokens,
  and `[100,100]` HF top-token ids. The final teacher-forcing artifact reports
  prefill 100/100/100% top-1/top-5/top-100 and 99 decode rows at
  91.9192/100/100%. Free-running TT first diverges at token 5, but exact text
  and raw-token artifacts show a fluent, English, on-topic continuation with
  no adjacent repeats or mechanical collapse; this is controlled by the
  passing top-5/top-100 gate and qualitative outputs, not dismissed.
- Canonical split sampling: both common sampling families are compared and
  `Sampling1D` is selected for direct `LMHead1D` shard consumption, persistent
  per-call parameters, TP2 topology, and `tt_out_tok`. The real A/B measures
  exact device full-vocabulary argmax at 0.665837 ms versus semantically greedy
  local-top32 at 0.906958 ms; both equal host argmax. No custom sampler was
  written. Capture/replay code uses stable token/current-position/page-table
  buffers, traced model/terminal/sampler/position segments, device token
  feedback, device position increment, and no greedy host argmax/full-logit
  readback/Python feedback reconstruction.
- Host stores: expert routing/top-k and expert projection remain TT work. The
  host boundary reads compact exact route IDs, mmap-loads and packs only exact
  misses, transfers through fixed rank-local staging, validates generations,
  and executes projections from stable slots on TT. PLE host work is exact
  EOS-aware n-gram hashing, sparse real-row mmap lookup, and stable input DMA;
  PLE projection/gating/convolution/recurrence remains TT. Static and hardware
  artifacts cover misses, hits, evictions, capacity-one thrash, stale-slot and
  failed-upload protection, real rows/hash parity, EOS/history carry,
  cold/warm chunked prefill, reset/cancel, repeated decode, and mixed-request
  isolation. The cold/warm artifact records equal output token 248046, 10,994
  cold packed misses versus 10,994 warm hits, zero warm source packing, and
  zero warm PLE table reads.
- Trace and runtime integrity: the current full-48 trace passes watcher after
  the generic Linear all-gather endpoint fix, and the reduced reproducer passes
  the same watcher contract. The tracker reruns close the allocation warning.
  Runtime audits separate expert/PLE lookup-control/DMA, caller-visible compact
  token readback, and explicit sampled-mode seed control from forbidden expert
  or PLE host projection, activation/KV/recurrence round trips, optimized-path
  host sampling, token-feedback reconstruction, per-token position upload, and
  unchanged-page-table refresh.
- Performance: the README begins with the representative full-48 batch-1
  prompt-128/generate-128 TTFT (104.640 s), trace capture (4.227 s), and
  trace-verified token-out throughput (1.643 t/s/u), and separately labels the
  AIME teacher-forcing compatibility throughput (1.953 t/s/u). Compact
  per-representative `tt-perf-report` outputs and hashes cover GDN, PLE+GDN,
  QSA, terminal LM head, and sampling without relying on the rejected
  profiler-overflow run.

## Anomaly Ledger

- Observed anomaly: active-trace allocation warning in untracked watcher runs.
  Evidence: `reduced_split_trace_watcher_fixed.log` and
  `full48_tokenout_watcher_fixed.log`.
  Affected path: reduced and full-48 split token-out capture/replay.
  Control or comparison: tracker-enabled reruns of both original tests pass
  capture, replay, and teardown without an unsafe-live-allocation error.
  Likely subsystem: intentional younger snapshot/crossing allocations under a
  multi-live-trace protocol.
  Investigation performed: source-level AutoDebug, lifetime review, focused
  reduced tracker run, then original full-48 tracker run.
  Resolution: controlled.

- Observed anomaly: TT AIME greedy output first diverges from HF at token 5.
  Evidence: `aime24_autoregressive_100_report_final.json`.
  Affected path: free-running all-48 traced device-greedy generation.
  Control or comparison: prefill and 99-row teacher-forcing top-5/top-100 are
  100%; both HF and TT completions remain fluent English and on-topic; token
  degeneracy counters and direct review find no feedback loop or collapse.
  Likely subsystem: acceptable low-precision rank-order differences amplified
  by autoregressive feedback.
  Investigation performed: raw tokens/text, divergence index, top-k metrics,
  trace counters, and fallback audit inspected directly.
  Resolution: controlled.

- Observed anomaly: two shared-suite answers end inside visible reasoning or
  answer formulation at 128 tokens.
  Evidence: `qualitative_shared_suite_final.json` and
  `QUALITATIVE_REVIEW.md`.
  Affected path: explanation and coding qualitative prompts.
  Control or comparison: the exact HF controls also exhaust the same allowed
  128-token window; TT remains coherent/on-topic/non-degenerate, and the third
  summarization prompt completes correctly.
  Likely subsystem: checkpoint chat template's visible xhigh-reasoning budget,
  not token feedback, cache, language drift, or sampling corruption.
  Investigation performed: exact rendered prompts, raw HF/TT tokens, decoded
  text, and degeneration counters reviewed prompt by prompt.
  Resolution: controlled.

- Observed anomaly: earlier accepted-stage artifacts include failed or stale
  controls (initial static capacity mismatch, assertion-only RoPE-width
  failure, pre-fix watcher assertion, and combined-profiler overflow).
  Evidence: retained non-final XML/logs plus the failure/fix ledger.
  Affected path: evidence provenance, not the final source claim.
  Control or comparison: explicitly named current-source final/rereview
  artifacts pass; the compact profiler captures are isolated by representative
  layer and contain no missing rows.
  Likely subsystem: iterative test assertions, pre-fix CCL endpoint handling,
  and profiler buffer capacity.
  Investigation performed: final artifact names, timestamps, JUnit status,
  hashes, and documentation provenance cross-checked.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths:
  - Original Qwen/Qwen3.8-Flash-Next full-model user contract supplied to the
    reviewer.
  - `.agents/skills/stage-review/SKILL.md`
  - `.agents/skills/full-model/SKILL.md`
  - `.agents/skills/host-weight-cache/SKILL.md`
  - `.agents/skills/tt-device-usage/SKILL.md`
  - `.agents/skills/tt-enable-tracing/SKILL.md`
  - `.agents/skills/qualitative-check/SKILL.md`
  - `.agents/skills/autofix/SKILL.md`
- Artifact paths:
  - `doc/full_model/README.md`, `work_log.md`, `AUTOFIX.md`,
    `AUTODEBUG_TRACE_ALLOC.md`, `QUALITATIVE_REVIEW.md`, and
    `profiler_provenance.txt`
  - `doc/context_contract.json` and `doc/host_weight_contract.json`
  - Fresh AIME `.refpt`/metadata, final teacher-forcing XML, final exact
    autoregressive JSON/XML, shared-suite `.refpt`/format/final JSON/XML
  - Final batch-1 perf, cold/warm, sampler A/B, non-greedy split trace,
    batch-32, advertised-context construction, watcher, allocation-tracker,
    static-rereview, and compact per-representative profiler reports
- Code paths:
  - `tt/model.py`, `tt/generator.py`, `tt/host_weight_cache.py`,
    `tt/multichip_decoder.py`, `tt/fused_decoder.py`, and
    `tt/optimized_decoder.py`
  - `demo/full_model.py`, `demo/generate_hf_reference.py`, and
    `demo/generate_qualitative_reference.py`
  - `tests/test_full_model.py`, `tests/test_full_model_perf.py`,
    `tests/test_host_weight_cache.py`, and relevant inherited decoder tests
  - `ttnn/.../all_gather_async/device/kernels/minimal_default_writer.cpp`
- Commands run:
  - Complete `sed` reads of the required skills and selected source/docs
  - `git status --short`, branch/HEAD inspection, focused `git diff`, and
    `git diff --check`
  - `find`/`rg` inventory and contradiction searches over stage evidence
  - Read-only Python/JQ parsing of JSON, JUnit XML, `.refpt`, exact generated
    text/tokens, capacity arithmetic, evidence-path existence, and compact
    profiler report hashes
  - `python -m py_compile` for the full-model test module

## Residual Risk

- This independent review did not open TT devices, run hardware, start a
  server, reset hardware, or run vLLM. Runtime judgments rely on existing
  current-source artifacts and source inspection.
- The worktree is live and uncommitted at review time. Stage-owned changes are
  separable by explicit paths from unrelated untracked multichip experiment
  artifacts. Per the stage-review workflow, the stage owner must create local
  checkpoint commit(s), record the branch/SHA in `work_log.md`, and never push.
- Exact host-backed prefill and decode are functionally complete but slow; no
  performance target beyond evidence-backed reporting was supplied. The
  measured path is exact, traced for all TT compute segments, and has no
  sampler-dominance or hidden-host-compute waiver.
