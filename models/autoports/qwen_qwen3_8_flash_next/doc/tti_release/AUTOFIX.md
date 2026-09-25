# AutoFix log: Stage 11 TTI release harness

## Scope and rules

- Generated implementation under evaluation: `models/autoports/qwen_qwen3_8_flash_next`.
- TTI checkout: `tt-inference-server` tag `v0.20.0`, base SHA
  `6ab1de736f303b899f84ae07d184d33f9889946e`.
- Server mode: existing OpenAI-compatible autoport server on port 8021;
  `docker_server=false`, `local_server=false`.
- Hardware and model implementation were held constant while each TTI harness
  hypothesis was tested. No infrastructure failure was reclassified as a
  model or accuracy failure.

## Proven repairs

### External runtime spec rejected before loading

- Symptom: both TTI parsers rejected `Qwen3.8-Flash-Next` as an unknown catalog
  choice, so the external JSON was never loaded.
- Fresh-context evidence: `AUTODEBUG.md`.
- Isolated control: unknown models without a runtime JSON still fail.
- Repair: make catalog validation conditional on the absence of an external
  spec in both parsers; hydrate model identity and runtime configuration from
  that spec before legacy required-argument checks.
- Result: focused parser/source-of-truth tests pass and TTI reaches the live
  autoport server.

### Spec `cli_args` overwritten by CLI defaults

- Symptom: the top-level driver ignored `RuntimeConfig.from_json()` and
  replaced the spec's port, workflow, and server-mode settings with CLI
  defaults.
- Isolated control: conflicting CLI placeholders versus authoritative JSON.
- Repair: load `ModelSpec` and `RuntimeConfig` together from the external JSON,
  preserve its path, and normalize from that resolved runtime.
- Result: generated runtime specs retain port 8021, `docker_server=false`,
  `local_server=false`, the selected workflow, and the autoport implementation.

### Loopback external server polled on port 80

- Symptom: the first TTI smoke waited on `http://127.0.0.1/health` instead of
  port 8021 because an explicit loopback URL was classified as remote.
- Isolated control: URL helper tests for loopback, remote URLs, and explicit
  ports.
- Repair: classify localhost and loopback URLs as local for service-port
  resolution.
- Result: no-Docker smoke reached port 8021 and passed its non-aligned 8/8
  benchmark.

### Runtime-only release had no eval policy

- Symptom: release validation required an eval registry entry but the generated
  model is intentionally absent from TTI's packaged model catalog.
- Isolated control: registry tests prove no `MODEL_SPECS` entry was introduced.
- Repair: add a runtime-only Qwen evaluation policy containing mandatory
  `meta_ifeval` and `meta_gpqa_cot`; reuse Meta's model-independent Llama 3.1
  dataset artifacts while retaining the actual Qwen model identity.
- Result: both tasks are selected by the strict release workflow without
  redirecting to a stock implementation.

### False experimental waiver

- Symptom: an early diagnostic run produced no eval results, but an
  `EXPERIMENTAL` status waived both failures and reported acceptance.
- Isolated control: the same report evaluated under `FUNCTIONAL` status fails
  with two blockers.
- Repair: the release spec uses `FUNCTIONAL`; neither mandatory Meta task has
  a known-issue waiver.
- Result: the next diagnostic release correctly exited 1 with two blockers.

### IFEval could not import `pkg_resources`

- Symptom: `lm-eval==0.4.4` IFEval utilities import `pkg_resources`, which is
  absent from setuptools 81 and later.
- Isolated control: import failed with setuptools 84 and succeeds after the
  compatibility pin.
- Repair: pin `setuptools<81` in the Meta evaluation environment.
- Result: a real one-sample IFEval request completed and produced parsed metric
  rows.

### Meta prompts produced an invalid chat request

- Symptom: `local-chat-completions` received an already-rendered Llama prompt
  string as `messages`; the OpenAI-compatible server correctly returned HTTP
  400 because `messages` was not a list.
- Rejected workaround: using `/v1/completions` would execute Llama control
  tokens against Qwen and invalidate prompt-format evidence.
- Repair: strip only the exact outer Llama user/assistant wrapper from the
  prepared Meta prompts, enable lm-eval chat templating so it builds a
  structured user message, and let the live server apply Qwen's checkpoint
  chat template.
- Guard: unexpected source prompt wrappers raise instead of silently changing
  task content.
- Result: real one-sample IFEval and GPQA requests both returned HTTP success
  and parsed results. Unit coverage proves the adapter is idempotent and rejects
  an unexpected wrapper.

### Fractional nightly policy exceeded the release window

- Symptom: the initial strict policy interpreted `0.05` against 541 IFEval and
  448 GPQA rows, selecting 28 and 23 samples. IFEval completed all 28 in
  1h48m50s. The first uncapped GPQA request then exceeded 5.7K tokens and 30
  minutes while remaining healthy, making 23 samples mathematically
  incompatible with the six-hour run window.
- Rejected workaround: context, prompt, and completion budgets were not
  shortened; no sample was rejected or aligned to an internal tile/page size.
- Repair: define this host-backed `ci-nightly` policy explicitly as 10 IFEval
  and 5 GPQA samples. This matches the intended bounded-sample character of a
  reasoning nightly while preserving every selected request's full 262,144
  context and 32,768-token GPQA output allowance.
- Accuracy guard: the adjustment was made before any GPQA result was returned
  or scored, so it is independent of answer correctness. Both mandatory tasks
  remain present and unwaived.

### Meta chat concurrency silently collapsed to serial execution

- Symptom: the first five-sample GPQA release request completed at the full
  32,768-token allowance in 2h37m, while server telemetry showed exactly one
  running request and the emitted lm-eval command contained no
  `num_concurrent` argument. Five worst-case samples could not finish within
  the release timeout.
- Root cause: a version-stale Meta-task compatibility branch for lm-eval 0.4.3
  replaced `max_concurrent=2` with `batch_size=2` and cleared concurrency. The
  pinned lm-eval 0.4.4 chat adapter then rejects batching and silently restores
  `batch_size=1`, making the release serial.
- Rejected workaround: no prompt, context, completion length, or sample was
  shortened, and the length-finished response was not treated as an automatic
  accuracy failure before TTI could score it.
- Repair: Meta tasks now retain `max_concurrent` and use individual request
  batches. The generated GPQA command is proven to carry
  `num_concurrent=2,batch_size=1`, matching the autoport server's two virtual
  serving slots.
- Isolated verification: 98 focused parser, URL, runtime-eval, and prompt
  adaptation tests pass; direct command construction asserts both concurrency
  fields. The strict release rerun is the serving integration proof.

## Verification in progress

The strict `ci-nightly` release rerun is the integration proof for the repairs
above. Final report paths, task scores, benchmark gates, cleanup, and review
status are recorded in `RUN_NOTES.md` after completion.

## Infrastructure recovery complete

- After the serial GPQA diagnostic was stopped, the original server shut down
  cleanly. Relaunch with the spec-declared `qwen3` reasoning parser reached
  fabric initialization but device 2 did not complete its remote-Ethernet
  handshake.
- The bounded `tt-smi list/reset/list` sequence was run twice. All four
  Blackhole devices remained visible, but the exact server launcher still
  failed at the same device-2 fabric handshake after the second reset.
- Raw mesh probes that lacked the launcher's mesh-graph binding failed earlier
  in topology mapping and are not treated as model or device evidence.
- The physical QuietBox reboot completed, followed by two bounded
  `list/reset/list` sequences. An implicit-placement retry still opened all
  four local devices and reproduced the device-2 router timeout.
- The release context contract requires Blackhole dies 0 and 1. The failing
  worker log showed `TT_VISIBLE_DEVICES=None`; explicit
  `TT_VISIBLE_DEVICES=0,1` restricted topology discovery to `{0, 1}`, brought
  both fabric routers through strict initialization, and opened the required
  1x2 mesh. This is a placement correction, not a reduced mesh or fabric
  workaround, and it matches the runtime release spec.
- The corrected server reached health on port 8021 after a 340-second cold
  start. Its log identifies the exact generated class
  `models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm.Qwen4ExpForConditionalGeneration`,
  full 262,144-token context, `reasoning_parser=qwen3`, and devices 0-1.

## Reasoning-parser null-content guard fixed

- A bounded request that exhausted its output allowance before closing the
  reasoning section returned HTTP 200 with `finish_reason=length` and null
  final content. Hidden reasoning is intentionally not evaluated.
- A bounded request with checkpoint-supported thinking disabled returned HTTP
  200, `finish_reason=stop`, and nonempty final content, proving the parser and
  content channel work. Only hashes and lengths are retained in evidence.
- Hypothesis: pinned `lm-eval==0.4.4` returns OpenAI chat
  `message.content=null` unchanged and the GPQA `RegexFilter` then applies a
  regular expression to `None`.
- Focused experiment: the installed `LocalChatCompletion.parse_generations`
  returned `[None]` for a synthetic reasoning-only response; the installed
  `RegexFilter` raised `TypeError`. Ordinary final content remained unchanged.
  This verified the hypothesis without reading any model-generated reasoning.
- Repair: the TTI-owned EVALS_META setup hook now applies an idempotent,
  version- and source-guarded compatibility patch after dependency
  installation. Only `content is None` becomes the empty string; non-null
  content is unchanged and reasoning fields are never read or scored.
- Verification: the actual installed parser returns `""` for null content,
  preserves normal content, and the real regex filter selects its unanswered
  fallback. The focused external-runtime/eval suite passes (`141 passed`),
  scoped Ruff E/F/I checks pass, and `git diff --check` passes. The corrected
  strict release rerun remains the integration proof.

## Resume-13 hardware recovery verification

- With no stage server or TT client alive, bounded `tt-smi list/reset/list`
  completed and all four local Blackhole devices remained visible.
- The exact release topology then opened and closed successfully with
  `TT_VISIBLE_DEVICES=0,1`, P300 mesh graph binding, a 1x2 mesh,
  `FABRIC_1D`, strict initialization, 8192-byte fabric packets, 1 GiB trace
  region, and 24,576-byte small L1. This clears the earlier recoverable remote
  Ethernet state before the durable server relaunch.
