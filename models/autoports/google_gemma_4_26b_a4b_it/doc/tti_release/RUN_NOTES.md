# Gemma 4 26B A4B IT TTI release

Status as of 2026-09-09 20:44 UTC: **release-workflow-fail; Stage 11 incomplete**.
P150x2 nightly_v3 finished rc1 after 901 text and 600 structured requests, with
112 invalid constrained responses, failed GPQA data access, and failed agentic
setup. Missing mandatory Meta fixtures and approved reference targets also block
release. The external P150x2 autoport server is closed for focused boundary diagnosis;
current task ownership is recorded in ACTIVE_WORK_STATE.json. A failed release
report exists under P150x2/nightly_v3/tti/reports_output/release/; reports-only
presentation regeneration is underway. No profile-wide readiness pass, final
clean-pass review, stage commit, or push is claimed.

The following checkpoint summaries are historical; later dated entries and
ACTIVE_WORK_STATE.json record subsequent changes.

Current checkpoint (2026-09-09 01:28 UTC): all three corrected no-trace smokes
pass. P150's complete v2 API suite passes 21/21 cases, and legacy passes 22/22.
Each suite has ten mechanical cases covering 50 requests and 10,936 returned
completion tokens. Reports capture 10,942 audited draws at assertion time;
the reviewer found 10,949 in each suite's durable records after asynchronous
drain. These are different snapshots, not conflicting final counts.
The sampler audit is request-scoped; exact live controls
prove the repaired tie ordering and unmarked-request bypass. The legacy
report's false benchmark/eval-scope warning was repaired and regenerated from
saved data, with zero blockers or waivers. Independent scoped review confirms
the API/report evidence; see `P150/api_audit_v3/workflows/` and
`AUTOFIX_api_conformance.md`.

P150's fresh `nightly_server` launch passed binding and completed all six N1
structured diagnostics. The constrained JSON-unique row failed at its unchanged
128-token allowance; exact ordered-schema replay reproduced that text. An
explicitly diagnostic 256-token request completed at165 with the original128
prefix. HF float32 CPU with the exact prompt/schema and greedy128 also truncated
inside the optional UUID field. The texts differ, but the failure class exists
in both implementations. This is no waiver; see `AUTODEBUG_structured_live.md`
and the unpublished `STRUCTURED_OUTPUT_ISSUE_DRAFT.md`.

P150 stopped gracefully. P150x2's first launcher attempt captured an empty
command-line hash immediately after Popen and rejected the later identity.
Cleanup completed without survivors or forced termination. The bounded startup
handshake repair passes all64 host controls; the fresh `api_release_v2` server
bound successfully. Another independent reviewer turn was rejected with
`agent thread limit reached`, so final independent review remains outstanding.
The P150x2 server and full `nightly_v2` client are active; the HF control exited.
The task Docker transport remains active, with an empty pre-agentic container
baseline. Historical cleanup files below are not final stage cleanup.

The first full release invocation stopped at an unnecessary aggregate HF_TOKEN
guard. A focused test proved standalone external evals/benchmarks were allowed
but their release aggregate was not. Adding RELEASE to the client workflow set
passes all15 secret-handling controls and preserves Docker authentication
requirements. The actual full retry now records the gated GPQA failure at the
eval loader and continues normally. See `AUTOFIX_release_auth.md`. The incomplete
standalone benchmark component was stopped; its three completed raw rows and
executed helper source are retained. The reviewed full-release helper was
restored exactly before the retry.

Full nightly specs now exist under `release_specs/`, with actual loader checks
selecting 15/19/19 text rows plus six structured rows per profile. These retain
all contexts, use external serving on port 8000, and have **empty** numerical
target fields. `release_specs/manifest.json` records exact commands and missing
mandatory Meta tasks. Prepared specs are not an executed or passing release.
No smoke reference was carried into them. TTI's separate eval client
environments are prepared; no task containers have run. The installed R1 GPQA
task loader confirms authenticated dataset access is required, after both
offline environment flags are removed (`r1_gpqa_task_preflight.json`).

Agentic Docker access initially failed for both normal CLI and Python SDK.
The authorized `sudo -n docker system dial-stdio` transport returned HTTP 200;
`docker_client_proxy.py` exposes that transport through a temporary user-only
Unix socket (0700 directory, 0600 socket). Read-only ordinary CLI and SDK checks
now pass with zero containers; see `agentic_docker_proxy_check.json`. Host
permissions are unchanged. The proxy serves task-sandbox clients only; it does
not replace the external autoport inference server. Its process/socket are
stage-owned and must be stopped/removed during final cleanup.

Strict structured scoring, termination, dialect-aware schema availability and
portable manifest evidence received a scoped clean-pass; see
`STAGE_REVIEW_REMEDIATION.md`. P150's N1 diagnostics completed; the unchanged
N100 release rows remain pending. Exact-model
release targets and authorized Meta/GPQA
fixtures remain external dependencies. No failed row is waived and no stage
commit has been made.

Stage 11 evaluates `google/gemma-4-26B-A4B-it` through the generated
`models/autoports/google_gemma_4_26b_a4b_it/tt/generator_vllm.py`.
Stage 10 completed with independent clean-pass; its implementation checkpoint is
`5cc0391d415f371a69150da91dc74076a179a791`, followed by ledger commit
`b1741a6f239`. The starting worktree was clean. The vLLM checkpoint is
`7b24b0e5904dac2f3859d9ab36577dbdfb5b7d55`.

## Environment and preflight

- Server mode: external autoport vLLM server, no Docker or TTI local-server launcher.
- Host/context: `qb2-120-p11t02`, current experiment checkout
  `/home/hous/dev/tt-metal`; four visible P300C Blackhole chips provide the
  established 1/2/4-chip P150/P150x2/P150x4 proxies.
- `timeout 60 tt-smi -ls --local` returned 0 and listed all four chips.
  Evidence: `device_health.log`. No reset, lock deletion, or recovery was needed.
- Context contract remains 50,624 / 262,144 / 262,144 tokens. P150's inherited
  physical-capacity limit is documented in `../context_contract.json`.
- TTI checkout: `/home/hous/dev/tti-release-gemma4/tt-inference-server`.
  Initial upstream HEAD was `39f6da8c014445cf70b99beed13a47cfd70f080e`.
  Selected repo tag `v0.18.0` (`d5913e81`), which contains the exact model's
  development reference recipe and the `6e396b4` custom-spec fix. Its Gemma 4
  family release image uses version 0.18.0. No Docker image was launched.
- Checked this tag's `python3 run.py --help`; supported spelling is
  `--tt-device` and `--runtime-model-spec-json`. Saved help under the work root.
- Persistent resume/evidence remains in this checkout and the existing
  `bringup/artifacts/multigoal-runs/gemma4-26b-a4b-p150` directory.

## Smoke and implementation wiring

Autoport implementation check: target is
`models/autoports/google_gemma_4_26b_a4b_it`; custom smoke spec and server launch
identify this path. The original TTI-produced runtime specs and worker import logs were validated;
see `implementation_proof.json`. The P150 `no_trace` rerun completed its tiny benchmark and failed three API checks;
the remaining topology reruns completed separately and passed.

P150 uses the exact Stage 10 launch flags: N150, 32 sequences, block size 64,
50,624 context, trace region 220,000,000 bytes, async scheduling, sampling `all`,
and Gemma4 tool/reasoning parsers. Exact argv and key environment are saved in
`P150/server_launch.json`. `TT_GEMMA4_TEXT_VER` selects the autoport;
`TT_METAL_HOME` is the current checkout, `HF_HOME` is the existing user cache,
`HF_HUB_OFFLINE=1` applies only to serving, and fallback exceptions stay enabled.

P150 health returned HTTP 200 and an 8-token chat request completed. The feature
smoke passed a 29-token nonaligned prompt, parsed `get_weather(city=Paris)`, and
separated reasoning from the correct arithmetic answer. See
`P150/openai_smoke.json` and `P150/openai_feature_checks.json`.

Prompt-format decision: instruct/chat, using the checkpoint's native template
through `/v1/chat/completions`. Exact checkpoint revision is
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. The shared qualitative suite completed on all profiles as documented below.
Mandatory eval-format validation still awaits authorized fixtures.

The tiny TTI benchmark uses a separate spec with 8 input / 8 output tokens,
one request, concurrency one, loose targets, embedded `docker_server=false`,
`local_server=false`, port 8000 and `disable_trace_capture=true`.
`ONLY_BENCHMARK_TARGETS=1` selects that single smoke row; it is not a release cap.
The exact invocation is:

```bash
ONLY_BENCHMARK_TARGETS=1 CACHE_ROOT=/home/hous/dev/tti-release-gemma4/smoke_P150 \
HF_HOME=/home/hous/.cache/huggingface python3 run.py --dev-mode \
  --model gemma-4-26B-A4B-it \
  --runtime-model-spec-json /home/hous/dev/tti-release-gemma4/P150_smoke_spec.json \
  --tt-device p150 --workflow benchmarks --service-port 8000 --no-auth \
  --skip-system-sw-validation --disable-trace-capture
```

Client dependencies are provisioned in TTI's own workflow environments; the
working server environment is unchanged. Initial log: work-root `smoke_P150.log`.

## Outstanding release gates

TTI v0.18.0's v2 custom-spec propagation was repaired and the original smoke
passed. The exact model's recipe still needs mandatory `meta_ifeval` /
`meta_gpqa_cot` integration and authorized fixtures. No full release has been
launched, no final report exists, and no failed row is waived. Full API/spec,
eval, benchmark and agentic execution, exact canonical quality controls,
independent clean-pass, and local stage checkpoints remain required work.

## Verified harness repairs and external eval dependency

The initial P150 benchmark failed before networking because v2 discarded the
custom runtime ModelSpec. AutoFix reproduced the exact failure, fixed the load
boundary, and passed 41 command-factory tests. The original 8/8/1 command then
exited 0 with completed=1, failed=0 and all requested tokens. The written runtime
spec matches the autoport/context/no-server-launch contract. See
`AUTOFIX_runtime_spec.md` and `P150/tti_smoke/`.

The model was missing API/spec registrations. AutoFix's focused before/after
experiments now select VLLMParamConformanceTest on all three proxy device labels
and legacy chat tests; 29 registration/routing tests pass. This verifies harness
selection, not full API conformance execution. See `AUTOFIX_api_registration.md`.

P150 shared qualitative checks completed with six greedy and six sampled
responses. All six greedy responses, prompts and prompt token IDs exactly match
the prior optimized-vLLM P150 control. Native tokenizer is GemmaTokenizer with a
nonempty chat template. Artifacts and manual review are under `P150/`.

The exact model's built-in TTI recipe contains r1_gpqa_diamond and two agentic
rows, but no mandatory Meta tasks and no same-model canonical thresholds.
AutoFix proves that the official Meta preparer rejects non-Llama datasets.
The exact Meta IFEval and GPQA fixture endpoints return HTTP 401; existing local
HF credentials are unavailable. Meta fixture access is manually gated, and the
original GPQA dataset is also gated. `AUTOFIX_eval_wiring.md` records exact URLs,
revisions, experiments, and the remaining control requirement. No alternate
public task or borrowed Llama/Qwen score was substituted as a passing Meta gate.
The user was asked to authenticate locally, without sending credentials in chat.

Read-only topology inspection finds systemd as PID 1 on the current experiment
host. Normal `docker ps` lacks socket permission; `sudo -n docker ps` succeeds
and lists no containers. Docker may therefore support future agentic client
sandboxes. It has not been used to launch an inference server or task sandbox.
No physical-host fallback or recovery was needed.

## Original smoke evidence and verification

| Proxy profile | Chips | Context | TTI smoke | Nonaligned/tool/reasoning | Shared qualitative |
| --- | ---: | ---: | --- | --- | --- |
| P150 | 1 | 50624 | exited 0, 1 completed, 0 failed | pass | pass; 6/6 greedy match |
| P150x2 (TTI p300) | 2 | 262144 | exited 0, 1 completed, 0 failed | pass | pass; 6/6 greedy match |
| P150x4 | 4 | 262144 | exited 0, 1 completed, 0 failed | pass | pass; 6/6 greedy match |

Each TTI smoke completed exactly 8 input and 8 output tokens. Each feature test
used the same valid 29-token nonaligned chat input and exercised both parsers.
All 36 shared qualitative outputs were read. Profile-local reviews describe
truncation and simplified explanations without claiming complete long answers
or scientific accuracy. `smoke_summary.json` re-derives the result from raw JSON.
These are topology smokes with loose targets, not release benchmarks.

Autoport implementation check: all three copied TTI runtime specs in `run_specs/`
match `models/autoports/google_gemma_4_26b_a4b_it`, exact profile context,
`docker_server=false`, `local_server=false`, port 8000, and
`disable_trace_capture=true`. Server logs independently identify the imported
`models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm.Gemma4ForCausalLM`.
No stock implementation was launched or substituted.

P150x2 and P150x4 exact serving/client commands, return codes and process lifetimes
are in their `smoke_manifest.json`; the driver command was:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/run_smoke_profiles.py --profiles P150x2 P150x4
```

The standard eval command now carries the profile context, preserving r1's
32768-token completion budget. Focused task-copy/command tests pass. P150's
mathematically impossible agentic 98304+32768 envelope is corrected in task
metadata to 17856+32768; mini-swe-agent now checks actual server-tokenized chat/tool input plus the
requested completion against the full profile context. It preserves valid
requests, including shorter completion budgets. The pinned 2.2.8 request-path
host tests pass (48 tests), and a P150 live `/tokenize` tool-template probe
returned HTTP 200, count 70, context 50624. See `AUTOFIX_agentic_context.md` and
`P150/no_trace/agentic_tokenize_probe.json`. Full agentic runs remain outstanding;
TerminalBench now uses a supported custom Terminus subclass that defers
context recovery to actual server errors instead of rewriting valid input
using an approximate count and 8000-token reserve. The pinned mini/Harbor plus
adjacent tests pass together (54 tests, no skips); see
`AUTOFIX_terminal_context.md`. Actual agentic workflow execution is pending.

Combined host verification: **125 passed** in
`tti_host_regressions_importlib.log`. The initial combined invocation failed
collection because v1 and v2 both define a `tests` package; rerunning with
`--import-mode=importlib` resolves that invocation collision. It was not a server
or model failure. Exact successful command, from the TTI checkout:

```bash
PYTHONPATH=tt-inference-server-v2 /home/hous/dev/tt-metal/python_env/bin/python -m pytest --import-mode=importlib -q \
  tests/test_gemma4_api_registration.py tests/workflows/test_v2_bridge_llm_routing.py \
  tt-inference-server-v2/tests/workflow_module/test_command_factory.py \
  tt-inference-server-v2/tests/llm_module/test_eval_context.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_agentic_eval_tests.py \
  tt-inference-server-v2/tests/test_module/llm_tests/test_llm_eval_tests.py
```

Python/config/docs changes require no C++ build. Black, py_compile and
`git diff --check` have passed for the respective authored changes. Original
captured output bytes are retained.

## Runtime projection and next release mode

`eval_runtime_projection.json` uses the measured optimized-vLLM rates with the
198-sample r1 GPQA Diamond 32768-token generation budget. At the respective
single-user rates, its full output budget projects to 48.4 / 38.8 / 34.6 hours.
This is a conservative budget projection, not measured typical output length;
separate CI-burst aggregate-rate scenarios are also recorded. It excludes
prefill, Meta gates, and the 89 TerminalBench / 500 SWE-bench tasks.

Plan `--limit-samples-mode ci-nightly` once integration is ready: the existing
r1 recipe explicitly uses 0.2 (about 40 samples), and each agentic recipe selects
five named tasks. The Meta subset is not defined or executed while exact
fixtures are inaccessible. No accuracy numbers or CI-subset readiness are
claimed. Full-set coverage would require a much longer run than these smokes.

## Cleanup and checkpoint state

The original attempt stopped all three servers. Its bounded tt-smi listing returned 0 with all four
chips visible; no EngineCore/VLLM processes remain. No reset or reboot was needed,
no stage tmux session or container was created, and the reservation was not
released. `cleanup_evidence.json` and `device_health_final.log` record cleanup.
The original TTI-generated `.env` was removed without reading or copying it.
The P150 `no_trace` attempt stopped its server after API failure. A bounded
health listing afterward returned all four chips and no server processes
remained. Further experiments are planned; earlier cleanup artifacts describe
the original attempt until final cleanup is refreshed.

No stage checkpoint has been committed or pushed because Stage 11 has not
obtained clean-pass. All stage-owned changes remain isolated from the initially
clean worktrees. TTI is on local branch `hous/gemma4-autoport-tti-release`, based
on v0.18.0 `d5913e816ac5dc33d86c1f3f5960348bc3fa4e2e`; `tti_changes.patch`
preserves the small source/config/test changes. `checkpoint_state.json` records
repo and SHA state. The tt-metal and vLLM starting SHAs above remain unchanged.

No notification was requested or sent. Independent review returned
`more-work-needed` in `STAGE_REVIEW.md`. Harness repairs and actual API workflows
are addressing its findings; required release/quality gates remain outstanding.

## Independent-review report aggregation finding

The independent reviewer reproduced false acceptance when a real raw smoke row
was changed in memory to contain failed requests and severe target misses.
AutoFix repaired target propagation, benchmark recognition, failed/missing row
handling, and explicit zero failure counts. The initial 106 focused host tests and 24 retained-artifact controls passed.
Follow-up review found explicit failure statuses and colliding row keys; both
are fixed, with 77 acceptance tests and 42 expanded retained-artifact controls
passing. Exact-length mismatches caused by legitimate early EOS are labeled
workload incomparability while preserving actual request failure counts. `AUTOFIX_report_blocks.md` records exact checks;
`report_acceptance_regression/` contains reports regenerated from original raw
outputs. Original reports remain historical evidence of the defect.

The same investigation found that v2 ignored `disable_trace_capture` even when
it was true in the runtime spec. The flag now reaches the runner. This separate
execution defect justifies repeating the tiny smoke on all three profiles.
The serialized rerun also executes complete TTI `spec_tests` and legacy `tests`:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/run_smoke_profiles.py --profiles P150 P150x2 P150x4 --attempt no_trace --api-tests
```

Per-profile `no_trace/smoke_manifest.json` records exact server/workflow argv,
cwd, statuses, and lifecycle. Logs are under each `no_trace/` directory;
`no_trace_profiles.log` records orchestration. New TTI cache roots use
`smoke_<profile>_no_trace_fixed` and `api_<profile>_no_trace_<workflow>` below
`/home/hous/dev/tti-release-gemma4`. Embedded specs set the intended workflow,
port 8000, both server launch modes false, and disable trace capture true.
`ONLY_BENCHMARK_TARGETS=1` applies solely to the one-row tiny benchmark and is
removed for API workflows. P150's new tiny benchmark counted one passing benchmark row. Its full v2
API suite completed with 18 passed and three failed in 655.72 seconds:
`test_non_uniform_seeding` produced 13 distinct outputs for 16 distinct seeds
(the same-seed determinism check passed), presence-penalty/repeat-trap and
repetition-penalty/natural-repetition failed diversity assertions. All three
remain blocking; `AUTODEBUG_api_conformance.md` is being prepared before repair.
The driver stopped P150 and halted subsequent profiles; legacy tests were not
reached. Original reports and logs are copied under
`P150/no_trace/tti_spec_tests/`. No failed row is waived.

`synthetic_benchmark_output_audit.json` classifies the tiny random benchmark's
HTML-fragment/whitespace responses. The upstream driver uses random input and
an explicit eight-token truncation for this topology test. These completions are
ungraded synthetic stress output; natural-language quality is assessed only via
the separate native-chat suite and prior-stage controls.

The tt-metal authored-file pre-commit run passed (`precommit.log`). TTI's own
hook-specific `.pre-commit` environment and Ruff are unavailable; its source
changes have the recorded Black, focused pytest, and diff checks. No C++ or
server implementation was changed in this stage.

## Release row inventory

`RELEASE_ROW_TARGET_INVENTORY.md` and `release_row_inventory.json` enumerate
15/19/19 default text rows and six structured-output rows per profile.
The exact TTI model has no numerical performance targets; earlier Stage 10
measurements use different backend, request-count, warmup, and stop policies.
They cannot establish comparable thresholds for this sweep. Mandatory Meta
fixtures and same-task canonical quality references are also missing.
No release spec with invented thresholds was created. Structured-output
execution is undergoing AutoFix because v2 drops those rows and its legacy
execution path was retired. The complete API workflows are independently actionable. The first P150
execution exposed the three failures above; AutoFix is investigating them.

## Corrected topology smoke results

`no_trace_summary.json` validates all three emitted runtime specs against the
autoport path, full profile context, no-Docker mode, port, trace-disable flag,
and correct device label. Each new raw benchmark completed one 8/8 request with
zero failures; each report grades one passing benchmark row.

After P150 API failure, independent P150x2/P150x4 smokes resumed with:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/run_smoke_profiles.py --profiles P150x2 P150x4 --attempt no_trace
```

Both servers stopped normally. All 12 new greedy responses, prompts and token
IDs match their profile-local Stage 10 controls; all 24 new outputs were read.
The unchanged 256-token cap truncates longer examples, so the reviews claim
serving regression scope only. Per-profile `no_trace/qualitative_review.json`
records the comparison. Live agentic `/tokenize` probes on all three profiles
return count 70 and the correct respective context. These are compatibility
probes, not completed agentic evaluations.

The next experiment is an opt-in P150 sampler audit for the unchanged three
failing API cases. It records bounded scalar/hash/token-history evidence and
is excluded from performance measurements. Instrumentation must first pass
CPU checks; actual audited execution is not yet complete.

## API audit and structured-output remediation in progress

`P150/api_audit_server_launch.json` records the exact live server argv, allowed
key environment, and audit source hashes. The opt-in hook passed 14 isolated CPU
tests including bit-identical draw/RNG controls and deliberately corrupted
seed/history/formula/draw controls. Its first live seeded 8-token request passed
all checks. The original two penalty failures reproduced with zero mechanical
failures: baseline/penalized completion counts 415/549 and 86/99. Penalties changed
logits on 548/549 and 99/99 respective steps. This is API-mechanics evidence;
prose quality is not inferred from it. The original concurrent seed case hit
its 30-second diagnostic timeout; only diagnostic wall time is extended to 300
seconds for a repeat, preserving all 32 requests and 50-token budgets.

Structured collection now selects all 6 configured rows on each profile and
passes 74 host tests; see `AUTOFIX_structured_output.md`. A fresh review found
that upstream correct_rate merely regex-extracts parseable JSON and ignores its
schema. Strict full-response JSON/schema validation remains required before
claiming guided-output correctness. No live structured result is yet available,
and no performance targets were invented.

The first corrected replay exposed an audit-accounting issue: async serving
may sample one extra token after the accepted stop boundary, then discard it.
No model or vLLM source changed. Audit v2 records the actual request stop IDs
and accepts only one fully audited lookahead after the first stop-token
boundary, with stop finish reason and no string-stop or ignore-EOS override.
Negative controls reject arbitrary tails and missing coverage. The old server
stopped cleanly; `P150/api_audit_v2_server_launch.json` records the new server.
All three corrected original API cases now pass on audit v2 in 139.90 seconds.
Full P150 spec/legacy workflows remain to be completed.

## Fresh nightly server and structured diagnostics

2026-09-09 00:37 UTC: Final orchestration scoped rereview returned clean-pass. Started `serve_profile.py --profile P150 --attempt nightly_server --sampler-audit` (session33201, launcher1865876). Version2 server manifest binds actual API listener1865895 and immutable launch/audit sources; startup log identifies `models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm`. Context50624, N150 one-chip proxy, external server; no legacy filter comparison enabled. Started `run_structured_smoke.py --profile P150 --server-attempt nightly_server --attempt structured_v1` (session81594). This runs six N1/C1 diagnostics with original OSL128 before full N100/C4 release rows. No full release pass is claimed.

2026-09-09: The six P150 N1 diagnostics completed (client session81594 exit1). Five functional rows pass; constrained `json-unique` is blocked by `incomplete_workload`: valid190-token prompt, original128-token output budget, actual `finish_reason=length`, incomplete string in the optional UUID field. Strict JSON parsing rejects it. No budget, schema, prompt, threshold or acceptance rule changed. Fresh xhigh AutoDebug agent `diagnose_structured_live` investigates before any full release launch. Missing benchmark targets remain separate blockers in all six rows.

Exact replay attempt `structured_exact_v1` completed128/256 requests, but is **invalid as an exact original-schema control**: the diagnostic script serialized the wire body using recursive `sort_keys=True`, changing JSON Schema property order. The response followed the reordered grammar (UUID first), differing from the original name-first response. Its128 output is truncated; its256 output completes168 tokens with the128 replay prefix. These results are retained but cannot establish original-request equivalence. Root identified the ordering defect; AutoFix agent is correcting only diagnostic serialization and adding an order-preservation control before v2 replay. Original workload/server/acceptance unchanged.

Corrected `structured_exact_v2` preserves original schema property order: exact128 replay equals original failing text; diagnostic256 completes165 tokens with the entire original128 prefix, strictvalid, finishstop. The256 probe is not release acceptance. Started `replay_structured_hf_cpu.py --attempt structured_hf_v1 --threads 16 --max-seconds 1800` (session89380), exact cached checkpoint/nativeprompt/schema/greedy128, explicitfloat32CPU, noTTdevicework. P150 owner1865876 receivedSIGTERM, session33201 exited143 with cleanup.complete=true, no survivors/no forcedkill/no triage. `tt-smi -ls --local` afterward listsall4 healthy/available. Started `serve_profile.py --profile P150x2 --attempt api_release --sampler-audit` session66733; preservedcontext262144/N300two-chipconfig. Root collects independent outstanding API evidence whileHFCPUcontrol runs; no benchmark performance comparison made under concurrentCPUload.

P150x2 `api_release` startup failed in launcher verification, not model initialization: vLLM reached `Application startup complete`, but `runner_identity.command_sha256` was the SHA256 of emptybytes (`e3b0c442...`), captured just afterPopen. The later real argv mismatch caused `Readiness launch process identity changed`. Serverowner normal cleanup completed with no survivors, no forcedkill/triage; session66733 exited1. This is a fresh orchestration startup race to verify/fix underAutoFix; no API requests ran in this attempt. No context/topology change or hardware recovery was needed.

HF control completed exit0 in196.37s (summary `P150/structured_hf_v1/hf_cpu_exact_schema/summary.json`): samecachedweights/native190tokenprompt/originalschemaorder/greedy128, float32CPU. It also endswith128tokens/length in the optionalUUIDfield and failsstrictJSON. Textdiffers (JordanSmith vsTTJaneDoe), so this proves the same failureclass exists in HF, not token-equivalence or all-row behavior. No model/harness defect was demonstrated by this specific truncated-answer probe; originalreleasefailure remainsblocking without an applicableissuewaiver. CPUmodelprocess exited. Root will retain this failure while collecting independent remaining release evidence after launch-race repair.

Startup AutoFix: saved-empty-SHA regression failed before repair; all64 host controls passed after the bounded exact-argv acquisition fix (original50 plus14 startup/lifecycle controls). Blackcheck passed; source hash645489100fce662ea140b65f5e9dfcbe2e2b307cd694f38e2f35a2efc8c63813. Root reviewed the exact patch and launched `P150x2/api_release_v2` (session54536, owner1876687); its initial identity has the expected nonempty hash. Attempt to start independent scoped rereview using the existing reviewer was rejected by the collaboration tool: `agent thread limit reached`. Root continues serial review and live verification; this is not an independent stage clean-pass.

The fresh P150x2 server reached verified readiness. The actual full nightly command `run_nightly_profile.py --profile P150x2 --server-attempt api_release_v2 --attempt nightly_v1` exited1 before collection: TTI `handle_secrets` requires HF_TOKEN for release, which is absent. Failure/log retained under `P150x2/nightly_v1`; no token was fabricated or copied. Root extended the existing helper with an explicit `--workflow benchmarks` mode to collect the independent component. The emitted `benchmark_spec.json` differs from the prepared nightly spec only in cli_args workflow and spec paths; context262144, full row geometry, targets, code path, server flags and port are unchanged. Started `run_nightly_profile.py --profile P150x2 --server-attempt api_release_v2 --attempt benchmarks_v1 --workflow benchmarks` (session67318). No task Docker transport is used by the benchmark component. Results cannot establish eval or overall release pass.

Serial AutoFix found a classification defect in TTI handle_secrets: external EVALS and BENCHMARKS were clients but RELEASE was omitted. The isolated original check failed only release (one failure, two passes). Added WorkflowType.RELEASE to the existing client set; all15 secret-handling tests pass, including absent-HF Docker release and absent-JWT protected Docker server negatives. The live `nightly_v2` release now enters evals, records the actual gated GPQA dataset error, and proceeds to benchmarks. No credential or dataset guard was bypassed. Partial standalone component owner1877478 stopped cleanly (session67318 exit143); three completed raw rows were copied explicitly from its reports_output/benchmarks directory. Server metrics confirmed zero running and waiting requests before full retry. The original reviewed full-release helper was restored exactly (SHA342ede4b65b6126629716086f1011e0a4dec8ec2afb3b47087949708addb982c), and `run_nightly_profile.py --profile P150x2 --server-attempt api_release_v2 --attempt nightly_v2` runs in session65197. Full workflow outputs were missing; TTI has no component-resume option, so the incomplete component was replaced by the full original collection. TTI Ruff hook environment is unavailable; Blackcheck identified existing assertion-format differences plus two added test wraps, and only the added wraps were corrected. No build is needed for these Python-only changes.

### P150x2 nightly progress at 2026-09-09 02:18 UTC

The first 11 of 19 text benchmark rows have completed: 794 requests, zero
failures, exact per-request input/output lengths and totals, and finite positive
latency/throughput metrics in the raw JSON. The 8,192/128, concurrency-31,
62-request row remains active. Fresh server log timestamps and increasing
generation counters distinguish slow execution from a device stall; no reset
was attempted. This is collection progress, not numerical target acceptance.
The full release client, P150x2 server, and task Docker transport remain owned
and active.

At 02:33:45 UTC the P150x2 8192/128, C31, N62 row completed: 62 successes,
zero failures, exact lengths, 1531.688 seconds and 5.1812 output tokens/s.
The cache-pressure observation and source calculation are retained in
`AUTODEBUG_release_preemption.md`; this is not a target acceptance or proof of
quality after preemption. Full collection advanced to 16384-token prompts.

### External turn interruption and durable resume, 2026-09-09 03:08 UTC

The external turn boundary sent SIGTERM at03:03:54 UTC. Both the P150x2
server and nightly client recorded exit143 and complete cleanup with no
survivors. No model exception or hardware failure preceded shutdown. The first
18 text rows (900 measured requests) were copied by the finally handler; only
the final131072/128 C1 N1 row was interrupted. Structured/API/agentic/report
collection had not started. Original data and partial manifests remain intact.

`timeout 60 tt-smi -ls --local` returned0 and all four P300C chips; no reset
was needed. The same P150x2 configuration is starting as `api_release_v3` in
dedicated tmux socket `gemma4-tti-release`, session `server-p150x2-v3`. A fresh
task proxy uses session `task-docker-proxy`; Docker SDK baseline is again zero
containers. The default-socket user session `0` predates this stage and is
preserved. Root will remove only stage-owned sessions during final cleanup.

AutoFix reproduced that the current TTI driver would rerun all19 rows. A
resume path is being repaired to import only hash/provenance/shape-validated
completed raw rows into the normal full release Accumulator while executing
missing rows. This avoids repeating valid expensive collection. See
`AUTODEBUG_release_resume.md` and `P150x2/nightly_v2/INTERRUPTION_REASON.json`.
A fresh independent reviewer is checking the earlier host/token-guard fixes.

At03:12 UTC, `STAGE_REVIEW_HOST_FIXES.md` returned independent scoped
clean-pass for the startup identity and external RELEASE token-guard repairs.
The reviewer re-derived saved patches/tests, exact binding and source hashes,
genuine gated-dataset rejection, and clean SIGTERM interruption. This closes
those two review items only; it does not establish whole-stage readiness.

## Host-only release resumption repair

`AUTOFIX_release_resume.md` records the verified missing-resume defect and
kept narrow driver wrapper. Fifty-four host tests pass. A CPU-only control
with the actual eighteen saved raw files preserves all nineteen ordinary
Accumulator sections and invokes only the missing 131072/128 C1 N1 row; that
row deliberately returns a diagnostic failure, so no new measurement or
release pass is created. Core/spec/config/raw hash mismatches fail closed.
`P150x2/resume_ready_v2.json` is prepared for the unchanged api_release_v3
server contract, pending independent review and root-owned actual collection.
Original cache and raw bytes remain untouched.

### P150x2 legacy API and scheduler repair, 2026-09-09 03:35 UTC

The full legacy suite passed22/22 with ten mechanical cases and50 distinct audited requests. Copied report hashes and all50 final audit records revalidated without failures (`P150x2/legacy_v1/durable_audit_check.json`). Returned completion tokens11102; assertion-time draws11111; final durable draws11119 include asynchronous post-assertion lookahead. This is API-only evidence on vLLM7b24, not release readiness. Client exited0; server was deliberately stopped143 with complete cleanup before serving-source integration. Their dead stage tmux sessions were removed; only task proxy remains.

`AUTODEBUG_cache_pressure_independent.md` confirms a futile prefill retry loop, lost first-pass preemption notifications, and misclassified generated-history replay tails using actual-source CPU controls. Isolated Autofix repairs are undergoing tests/review. Serving-source changes will invalidate old benchmark reuse; retain the18rows as pre-repair baseline and rerun affected collection. No benchmark lengths, concurrency, context, or stochastic generation defaults are weakened.

Independent `STAGE_REVIEW_BENCHMARK_RESUME.md` found unbound precision/config dependencies and inherited environment overrides. These are required repair work; the prepared resume manifest is not approved for live reuse.

Root reapplied both exact serving patches in a separate combined worktree for real-import host validation. State-slot and gathered-DP tests passed20/20 (4.93s), in addition to29 actual-source CPU controls rerun by root. `combined_repair_check.json` records command and log hash. No device was opened and the live vLLM checkout remains unchanged pending independent review. Python-only changes require no C++ build.

### Resume provenance review repair

Independent review P1s are fixed in AUTOFIX_release_resume_provenance.md.
The corrected launch captures consumed nonsecret model overrides and the
selected precision/extends plus imported configuration/helper files. The old
api_release_v2/v3 manifests lack historical inherited-environment evidence
and now reject reuse; their eighteen raw measurements stay preserved only
as historical evidence. Sixty-three resume/runner controls and sixty-nine
launch/orchestration controls pass, with actual logs retained. Model/core
source equivalence remains strict. No new live measurement was made.

At03:46UTC independent serving review confirmed an additional async correctness defect on the combined patches: scheduler resumes8192+5 accepted tokens while the worker retains a discarded sixth generated token. Complete production CPU methods mark the final five-token replay intermediate, suppress its output, then assert on the next mixed batch. This is required Autofix work; the29CPU and20import tests did not cover it. No serving patches have been integrated into the live checkout. Idle `timeout60 tt-smi -ls --local` passed with allfour chips (`health_before_serving_repair.log`); no reset.

`STAGE_REVIEW_SERVING_REPAIR.md` returned more-work-needed with two new independently reproduced findings: stale worker output history (P1) and retained seeded RNG advancement from discarded async output (P2). An isolated accepted-history/RNG Autofix starts from both prior patches; neither is applied to main. `STAGE_REVIEW_BENCHMARK_RESUME_REREVIEW.md` closes both original provenance P1s, but finds a safe-failing future-builder hook-normalization P2; a narrow repair is underway. No whole-stage pass.

The live diagnostic prompt is prepared under `P150x2/replay_prompt_v1/`: exactly8192 IDs from the HF chat template with add_generation_prompt=True and enable_thinking=False. Re-encoding its full rendered text reproduces the IDs; no template/header truncation. The initial preparation misread the installed tokenizer default BatchEncoding as a token list; explicit return_dict=False corrected this before any request or saved prompt evidence. This is a matched preemption quality control, not replacement benchmark data.

### Resume builder hook comparison

Rereview P2 is fixed: exact historical collector equality is checked before
the narrowly verified initial-hook normalization. Nine focused builder source
controls pass, with actual logs in resume_cpu_controls/builder_hook_tests.log.
See AUTOFIX_release_resume_hook.md. Historical rows remain ineligible and
all model/core/provenance guards remain unchanged. No live collection ran.

`STAGE_REVIEW_BENCHMARK_RESUME_FINAL.md` returned scoped clean-pass after the builder P2 fix. It independently reproduced9exact-source controls and confirmed all18historicalrawhashes unchanged/ineligible. Updated `tti_changes.patch` contains59stage-owned TTI source files,259307bytes; reverse-apply check passed and previouspatch/sourceledger preserved in tti_patch_history/before_resume_provenance.*. No commit/push.

Optional `serve_profile.py --replay-diagnostic-config <config>` now snapshots diagnostic source/config, sets the effective childPYTHONPATH and chains the unchanged canonical sampler audit hook. Ordinary launches remove inherited diagnostic activation. Full release and focusedbenchmark clients reject fault-injection servermanifests. All65orchestration controls pass, including both actual CPUchild launch/cleanup paths and precise diagnosticenvironmentbinding (`replay_diagnostic/launcher_tests.log`). Independent scoped diagnostic review is underway; no HTTP/device inference performed.

Accepted-history/RNG repair root checks: nine new CPU controls pass. Actual plugin-import host tests passed62 cases and found one outdated state-slot fixture that used None for CachedRequestState. After replacing only that fake state with a real request object, the affected import test passed1/1; assertions preserved. Logs and hashes: `accepted_history_root_check.json`. Independent repair review remains pending; mainvLLM unchanged.

### Replay diagnostic listener binding AutoFix

The reviewed diagnostic-first/audit-second listener rejection was reproduced
before the narrow verifier fix. Explicit source/config/bootstrap/environment
chain validation now permits that declared chain while ordinary launches
retain auditfirst. Twenty new CPU controls include the actual main listener
callback with nonempty fake listeners; all85orchestration controls pass.
See AUTOFIX_replay_binding.md and retained before/after logs. No live device
or HTTP work ran; independent rereview/root-owned live validation remain.

`STAGE_REVIEW_ACCEPTED_HISTORY_RNG.md` returned scoped clean-pass after38CPUcontrols plus288draw comparisons and pruning/rowcompaction/serialization checks. Root verified clean mainvLLMbase7b24 and absentlistener, applied allthree exact reviewedpatches sequentially, checkedgitdiff andfinalsourcehashes againstisolatedrepair. `vllm_serving_repair_applied.json` records allsourcehashes/patches; no commit/push. All63relevant actual-import hosttests thenpassed from maincheckout (`main_vllm_serving_repair_tests.log`,4.88s). These do notreplace livevalidation. Serverlauncher nowverifies currentservingsourcebytes againstthatledger andsnapshots it intoeachnewlaunch.


### Reviewed serving repair: live validation started

The final isolated diagnostic and benchmark-resume reviews returned scoped clean-pass. The vLLM formatting delta passed both Ruff hooks, preserved all 227 regression assertion nodes and normalized production ASTs, and passed 38 CPU controls. Its exact patch and source hashes are recorded in `serving_format_check.json` and the refreshed `vllm_serving_repair_applied.json`; no commit or push. Launched `P150x2/replay_greedy_v1` with the recorded diagnostic configuration and sampler audit in the existing experiment context. Context remains 262144 and autoport implementation selection is unchanged. This isolated fault-injection control cannot be used as release benchmark evidence.


Both live forced-preemption controls completed all requests with 8192 prompt and 128 completion tokens. Greedy first diverges at accepted index 5; seeded also differs while all 128 retained host RNG states match. Exact scheduler/worker replay ranges, accepted history, final masking, and preemption notification pass. These are failed correctness controls, not release passes. See `replay_diagnostic/controls/{p150x2_v1,p150x2_seeded_v1}/comparison.json`. Fresh Autofix source diagnosis is investigating lost absolute cache positions and attention history in the autoport continuation-prefill path. Greedy server cleanup completed; seeded owned shutdown requested.


Continuation history primitive testing rejected Moreh getitem after both row-major and adapted tile-index variants failed exact comparisons on both TP2 chips. Actual input upload and index extraction passed. Bounded 16-page slices plus concat passed two nonidentity page orders using the real adapter allocator/cache specs and 32-slot metadata. See `AUTOFIX_continuation_page_primitive.md`. All probes closed devices cleanly; no reset or profiler. The model repair will use this verified bounded device path with scheduler page metadata.


Additional continuation investigation found a sliding page-table contract mismatch: upstream vLLM keeps absolute logical columns and replaces expired pages with null, while the autoport truncates the table to its first 16 columns and applies modulo 1024. A fresh source/CPU diagnostic is verifying this. The isolated repair now distinguishes external absolute scheduler tables from the standalone generator's bounded circular cache. Increasing page-table metadata width preserves the existing KV allocation and profile context; it does not raise or lower the request budget. No serving or release experiment is running until these linked cache/history defects are repaired.


Integrated weight-free TP2 continuation probe (`continuation_prefill/probe_v1`) reached a C++ host validation failure before attention: absolute sliding table width 4096 exceeds the bounded 256-block test pool. The rule incorrectly equates virtual table columns with resident physical pages; retained scheduler tables contain reclaimed prefix columns. Independent Autofix is checking reader/writer addressing and correcting the validation, with required build before reuse. Existing local CMake/Clang20/Ninja build artifacts are present; no toolchain installation. Isolated Python correction has 68 passing focused CPU controls, but no device pass is claimed.


Applied the independently inspected two-file C++ sparse-page validation patch to main for build/device verification (`sparse_page_validation.patch`, `sparse_page_validation_applied.json`). All actual cache addresses remain indirect table entries; virtual table width does not bound the number of resident physical pages. The required `.github/scripts/copilot-build.sh` was attempted and exited 1 at Docker availability preflight (`kernel_build_wrapper.log`). The existing local Release build and Clang 20.1.2/Ninja are present; proceeding with the narrow `_ttnncpp` target without installing toolchains or dependencies. No build pass is claimed until that command completes.


The narrow C++ rebuild passed with `/usr/bin/ninja -C build -j 2 ttnncpp` (`kernel_build_local_v2.log`); the initial `_ttnncpp` target spelling was rejected and corrected. `cmake --install build --component ttnn-runtime` installed the rebuilt library into the runtime path; both build/installed hashes are recorded in `kernel_build_result.json`. All five actual-device sparse-page regression cases passed with `python_env/bin/python -m pytest -q tests/ttnn/unit_tests/operations/transformers/test_paged_cache_sparse_table.py` (`kernel_sparse_page_tests.log`): single absolute/modulo update, fused tiled/row-major update, and invalid modulo table bound. No geometry, dtype, or context cap changed.


With rebuilt sparse-page validation, integrated `continuation_prefill/probe_v2` passed all three sliding starts 2048/8192/8196 and aligned full starts 2048/8192 for unchanged logical tail length 5, on both chips. Cache/history readbacks were exact and attention PCC against chronological CPU SDPA was at least 0.99966. The final unaligned full case failed at unsupported device-tile front padding, before attention; replacing the internal Q-only alignment with dummy-prefix concatenation is the next focused fix. The logical request remains unchanged. This is a partial failed diagnostic, not release evidence.


Component validation now passes: TP2 unaligned start8196 lengths1/5/2048 full+sliding; TP1 reversed HMA alias, TP4 normal, TP2 reversed alias at8196/5 on matching profile meshes; TP2 circular-cache starts8196 and8228 with2048-token continuation, including the explicit32/32 full-SDPA branch. All logical cache/history readbacks are exact and CPU attention comparisons pass. Artifacts are under `continuation_prefill/{probe_v3,profile_matrix_v1,ring_probe_v1}`. Kernel review returned scoped clean-pass in `STAGE_REVIEW_SPARSE_PAGE_VALIDATION.md`. Model review found same-object stale host page-table metadata for direct generator callers; the isolated correction accepts authoritative adapter metadata percall and refreshes device table metadata for direct continuations. All102CPU/adapter tests now pass; final rereview and live replay remain pending.

### Integrated continuation repair review and live restart

Main-checkout CPU tests passed 102/102 in 3.00 seconds (`continuation_prefill/main_cpu_adapter_tests.log`). Both continuation-prefill and sparse-page validation independent reviews returned scoped clean-pass. Applied source/runtime hashes are in `autoport_continuation_repair_applied.json`; launch validates and snapshots this ledger. Full-model/replay and release acceptance remain pending.

Fresh greedy control `p150x2_fixed_greedy_v1` completed both 128-token outputs, but continuation comparison failed at output ordinal 22. Exact scheduler/worker replay ranges and accepted-history controls passed. The mismatch summary is retained alongside the comparison. A fresh source-only AutoDebug investigation was requested. The owned diagnostic server stopped cleanly before the seeded control.

Fresh seeded control `p150x2_fixed_seeded_v1` completed both 128-token outputs. RNG state matches at all 128 accepted positions; history, ranges, and masking pass, but continuation diverges at ordinal 5. Two ordinary greedy requests on that server match the earlier greedy serial output byte for byte: three requests across two fresh servers. Evidence is in `replay_diagnostic/controls/p150x2_ordinary_greedy_repeat_v1/summary.json`. Both diagnostic servers exited 143 with complete cleanup, and their dead tmux sessions were removed. The next experiment compares actual model boundaries while feeding both paths the same accepted tokens. No tensors, weights, or profiler dumps were retained. All four chips remained visible in the bounded health check (`continuation_prefill/health_after_fixed_live_replay.log`); no reset was needed.

### Fresh full-model replay diagnosis

`AUTODEBUG_replay_after_continuation.md` finds no newly proven semantic defect from source alone. Seeded decode is eager, so retained traces cannot explain both failures. The next hypothesis test compares identical prefix K/V followed by five eager decode steps versus a logical-five continuation prefill, using real weights and the unchanged serving precision, context, page geometry, and HMA allocator. Prefix K/V equality is a required control. Per-layer producer, written cache, hidden-state, and terminal-logit statistics will locate the first divergence. The observed SDPA compute-config contrast remains an untested hypothesis; no fidelity change or numerical waiver is applied.

Raw serving debug logs total over 600 MB and remain at their original local paths. The model `.gitignore` excludes TTI `server.log` files from checkpoint commits; no log is deleted. `raw_server_log_inventory.json` records hashes and sizes of closed historical logs. Compact release reports and diagnostic statistics remain under `doc/tti_release`.

### Actual full-model five-token localization

`continuation_prefill/teacher_forcing_v1/result.json` completed in 78.67 seconds and closed devices cleanly. Prefix K/V and tables match exactly; all producer-to-cache writes match exactly; all measured values are finite across 30 layers and both chips. Cross-phase differences are already present in layer-0 Q/K/V. All five terminal top-1 choices match, but logit distribution TV ranges up to 0.105 (0.04184 at the final shared query). This explains a distribution difference without establishing a numerical waiver. AutoFix is investigating replaying accepted generated history through its original decode execution path.

Fresh ordinary feature checks passed the 29-token valid prompt, tool parser, and reasoning parser. Six greedy qualitative outputs match the previous optimized profile; the independent sampled-output review also passed (`P150x2/ordinary_fixed_v1/qualitative_review.json`). Tiny benchmark attempts `smoke_fixed_v1` and `v2` were stopped with complete owned-client cleanup after unwanted text/structured rows were selected. `AUTOFIX_existing_server_smoke.md` records the environment and CLI-resolution causes and workflow-level CPU controls. No tiny or full release pass is claimed for these failed attempts. The ordinary server also stopped cleanly; only the task Docker transport remains.

Corrected fresh P150x2 tiny smoke passed in `P150x2/smoke_fixed_v3`: exactly one completed request, zero failed, 8 input and 8 output tokens, and report acceptance true with no blockers. The written runtime envelope confirms external autoport code, context 262144, port 8000, both server-launch flags false, `smoke-test`, and disabled trace capture. TTI serializes its service port as a string; numeric value is verified. Feature and qualitative commands passed; all six greedy outputs again match Stage10 and degeneracy checks pass. This closes the tiny harness failure only. Ordinary server and client completed owned cleanup before the next model probe.

### Decode replay allocation control passed

`continuation_prefill/decode_replay_v1/result.json` passed every exactness gate in 78.42 seconds. One five-token allocation followed by five eager decode calls reproduces the ordinary five one-token allocations: identical prefix K/V and page tables, every measured Q/K/V/output/current-cache value across 30 layers and both chips, and all five terminal logits. All values are finite; devices closed cleanly. This verifies the proposed intervention at the execution-phase boundary. Separate isolated AutoFix work now carries authoritative original prompt lengths through the worker and uses them in the autoport to split prompt prefill from accepted generated-history decode. Ordinary valid prompt lengths and context remain unchanged; no intermediate sampling is permitted. Live controls, mixed-row pressure, and full release remain pending.

### Original prompt boundary and adapter replay repair

The vLLM metadata repair received scoped independent clean-pass in `STAGE_REVIEW_REPLAY_PROMPT_METADATA.md`: 49 CPU controls pass and normal/lane/DP transport preserves original prompt lengths. The isolated adapter candidate preserves ordinary prompt prefill and replays accepted generated history through eager unsampled decode. Its 20 focused controls plus 102 existing adapter/continuation controls pass; MAIN remains unchanged pending device proof and independent review. A serialized actual-model B1 candidate probe is running in `continuation_prefill/adapter_replay_v1`, using the recorded candidate adapter source with unchanged MAIN generator/model and rebuilt runtime. Mixed canonical-32 replay, live greedy/seeded controls and full release remain required.

Actual isolated adapter proof `continuation_prefill/adapter_replay_v1` passed in 78.39 seconds. Prefix K/V and tables match, all measured decode/replay layer boundaries and all five logits are bit-exact, cache writes are exact, and the adapter final host logit row equals its final inner decode output. Devices closed cleanly and the dead owned tmux session was removed; this tmux does not expose pane exit status, so no numeric exit code is claimed. No numerical policy, context or request length changed. Mixed canonical-32 and live controls remain required.

Applied the exact independently reviewed `replay_prompt_metadata/original_prompt_lens.patch` to MAIN vLLM after verifying every before/after hash. All 49 MAIN CPU metadata/history/RNG/recompute/scheduler controls pass (`replay_prompt_metadata/main_validation.json`). The refreshed serving ledger preserves its previous version and identifies this incremental patch. Fresh greedy/seeded controls have new UUIDs and the same native 8192/128 request shape; they are prepared but not run.

`STAGE_REVIEW_GENERATED_REPLAY_ADAPTER.md` returned scoped clean-pass. Applied the exact candidate incremental adapter patch and verified its hash; all 122 MAIN continuation/adapter/replay CPU controls pass in 3.14 seconds. The previous source ledger is preserved and the current ledger includes both review and B1 device evidence. Started fresh `P150x2/replay_generated_greedy_v1` for the original live preemption control, not release collection.

Fresh integrated greedy forced-preemption control passes: both 8192/128 requests have identical 128 accepted outputs and HTTP text; exact scheduler/worker replay ranges and history/masking checks pass. See `replay_diagnostic/controls/p150x2_generated_replay_greedy_v1/comparison.json`. The dead diagnostic client session was removed; owned server shutdown requested before seeded validation.

Fresh integrated seeded preemption control also passes: identical 128 accepted outputs and HTTP text, exact scheduler/worker history and replay ranges, and all 128 host RNG states match with no missing/ambiguous/invalid evidence. See `replay_diagnostic/controls/p150x2_generated_replay_seeded_v1/comparison.json`. Both original failing live controls are now fixed; mixed canonical-32 device validation and original cache-pressure/full release remain required. The dead client session was removed and owned seeded server shutdown requested.

Both final B1 replay servers exited143 with complete owned cleanup; both dead server/client tmux sessions were removed. Only the task Docker transport remains. New reviewer spawn and follow-up calls hit the agent thread limit; the already-running independent mixed-probe reviewer was asked to inspect final live controls in the same independent review session. The mixed diagnostic is being strengthened because identical active histories and retained prior-arm cache values could conceal a routing fault; it has not run on hardware.

Fresh ordinary `P150x2/smoke_generated_v1` passes after the generated-history repair: exactly one completed/zero failed 8-in/8-out request, report acceptance true, no blockers, effective external autoport spec/context262144/port8000/disabled trace capture verified. The feature runner reports `passed` for 29-token valid prompt, tool and reasoning parsers. Six greedy outputs and prompt IDs match Stage10 and degeneracy exit0; sampled human-style review pending. All three owned clients exited0 with complete cleanup; smoke session removed and ordinary server shutdown requested before mixed device proof.

`STAGE_REVIEW_LIVE_GENERATED_REPLAY.md` independently returns scoped clean-pass for the original B1 greedy/seeded failures. It rederives native prompt rendering, all accepted tokens versus HTTP output, exact prior serial equivalence, history/masking, RNG and 23 source/runtime hashes per launch. A generic active-trace allocation warning before forced replay is classified with allocator/split-trace source and passing serial controls; general allocation tracking is not claimed. Fresh ordinary server subsequently exited143 with complete cleanup, and its dead session was removed. Revised mixed probe snapshots are prepared at `continuation_prefill/mixed_adapter_replay_v2`; the rejected initial source is preserved and was never executed on hardware.

The strengthened mixed diagnostic received scoped clean-pass in `STAGE_REVIEW_MIXED_REPLAY_PROBE.md` after 22 independent cache-writer controls. It uses distinct active histories and independently verifies per-arm/user finite sentinels at every scheduled K/V location before execution. Started `continuation_prefill/mixed_adapter_replay_v2` with immutable source/helper snapshots and the actual MAIN adapter, preserving context262144, TP2 canonical32 execution and existing precision/HMA pool. No server is running concurrently.

Strengthened mixed actual-model proof passes in 107.27 seconds (`continuation_prefill/mixed_adapter_replay_v2/summary.json`): 330 verified initialization records, 240 active layer-output rows, 8 inner logits, two final logits, chronological K/V, distinct histories, overwritten tails, retained history, inactive guard, stable/restored page tables and final positions all pass. Device clean-close is recorded; dead probe tmux removed. Independent ordinary qualitative review also passes with no new sampled regression. Restarted ordinary `P150x2/ordinary_generated_v2` for the exact original C31/N62 pressure workload and subsequent nightly collection. No production source changed after the fresh tiny smoke.

Started the exact original cache-pressure command from `cache_pressure_retest_plan.json` against verified ordinary `P150x2/ordinary_generated_v2`: C31/N62, 8192 prompt/128 completion, original omitted sampling flags/server defaults, no profiler. Fresh tiny evidence from `smoke_generated_v1` uses identical model/runtime/worker bytes and context; only the ordinary server was restarted after the serialized mixed proof. Focused benchmark output is `P150x2/cache_pressure_fixed_v1`.

The initial pressure launcher failed before any HTTP request: direct execution of the vLLM client shebang script gave a kernel argv with its Python interpreter prepended, which the strict owned-child startup check rejected. `P150x2/cache_pressure_fixed_v1/manifest.json` records the error and complete cleanup; benchmark.log is empty and no raw result exists. Dead client session removed; ordinary server remains healthy. Isolated AutoFix is reproducing the condition and adapting only invocation/attempt output routing, preserving every workload flag and the strict ownership guard.

Pressure launcher AutoFix received scoped clean-pass. It explicitly invokes the client script with its existing absolute Python shebang interpreter; strict OwnedProcess is unchanged. Only a fresh result filename differs in the benchmark arguments. All 11 MAIN CPU controls pass, including actual ownership/cleanup and preservation of v1 evidence. Retried as `P150x2/cache_pressure_fixed_v2` with `--server-attempt ordinary_generated_v2 --attempt cache_pressure_fixed_v2`.

Original pressure row passes in `P150x2/cache_pressure_fixed_v2`: 62 completed, zero failed, every input8192/output128, measured duration742.418 seconds, output10.6894 tokens/s, client exit0 and complete cleanup. The dead client session was removed; ordinary server remains for full release. Its logged peak-concurrent44 is an inclusive one-second activity estimate from client `vllm/benchmarks/serve.py`, while actual dispatch awaits requests under semaphore31. This coarse field is not claimed as simultaneous concurrency. The prior incorrect-mechanism row took1531.688 seconds; that comparison does not attribute performance to an individual repair or establish release thresholds. Final raw/text review and fullnightly collection remain required.


Pressure output review identified repeated brace/HTML/sentence tails and literal thought-label content. `AUTODEBUG_pressure_outputs.md` records exact affected rows and the next matched serial controls. Correction to the earlier sampling description: the installed TTI vLLM 0.13.0 client changes random-dataset requests after its initial Namespace print, forcing `temperature=0.0` and `ignore_eos=true`. These were the effective settings in both pressure runs; omitted CLI flags did not select server defaults. The immutable original plan and measurement files remain unchanged. Forced continuation beyond EOS is a hypothesis awaiting controls. The focused successful row is ineligible for the unchanged schema-2 interrupted-full-release resume path; the full nightly must measure it again.


Matched-pressure control source and prepared prompts were copied from the isolated AutoFix tree with exact hashes. The first root CPU test invocation accidentally inherited repository conftest and imported TTNN while the server was active; seven tests passed and the import-isolation assertion correctly failed. No device test/open ran and that process exited. This invocation violated the import-serialization rule. The corrected `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python_env/bin/python -m pytest --noconftest -c /dev/null -q .../pressure_quality_control/test_serial_control.py` passed all eight controls with no TTNN/vLLM imports. The only warning concerned pytest attempting its cache under `/dev`; no test failure or model change resulted. Evidence: `pressure_quality_control/main_validation.json` and `main_cpu_tests.log`.


Matched serial controls completed with client exit0 and complete cleanup in `P150x2/pressure_serial_forced_v1` and `pressure_serial_eos_v1`; both dead owned tmux sessions were removed. All forced requests preserve8192/128; rows7/13 reproduce pressure text exactly, rows59/15 preserve the semantic response but differ in wording and later repetition. EOS-respecting diagnostics change only ignore_eos and stop rows7/59/15 at35/37/75 tokens before their late tails. Row13 still loops to128. No benchmark is replaced and no overall quality pass is claimed. Next controls: canonical HF row13 at the same effective native8192-token prompt and small paired59/15 canonical32 execution without cache pressure.


Additional correction to historical diagnosis: `AUTODEBUG_cache_pressure_independent.md` discussed stochastic server defaults and epoch-driven trace recapture using MAIN endpoint source. The actual installed TTI random client forced greedy temperature0 and ignored EOS. The historical scheduler/block-ledger reproduction remains valid, but its stochastic trace-recapture explanation is not proven for the retained pressure measurements. That source-only hypothesis is not used as a release conclusion.


Both exact C2 forced outputs59/15 equal the original pressure text byte for byte (`pressure_pair_forced_v1`), with server observations2running/0waiting and7.3% settled KV usage. C2 EOS stops59 at38 as an exact pressure prefix, removing its thought/HTML tail;15 stops89 coherently without its pressure marker/language-switch tail, but changes wording after its companion can leave the batch. One mixed-EOS diagnostic is being prepared to keep59 forced128 while only15 respects EOS. These correctness runs overlap a CPU HF control and establish no benchmark timing claim. All completed paired/serial client processes and dead owned tmux sessions are cleaned.


Mixed EOS control succeeded: row15 stops96 with an exact original-pressure prefix while row59 remains forced128 and finishes afterward. This classifies the original row15 thought/language-switch tail as post-EOS behavior. Canonical HF row13 completed128 tokens in460.54seconds, float32CPU/SDPA, with a coherent word-salad description rather than the TT sentence loop. The actual live `/tokenize` endpoint returns8205 IDs whose last8192 exactly match the HF saved input; offline cross-version rendering also matches. Thus tokenization-version mismatch is refuted and row13 remains an ordinary-prompt AutoFix investigation independent of cache pressure. Both final diagnostic clients and the CPU reference exited0 with complete cleanup; their dead tmux sessions were removed. Ordinary server remains idle.


A same-workload row13 request adding only the API response option `return_token_ids=true` reproduced all128 original TT output text and returned the exact8192 prompt IDs; no logprobs, sampling or length changed. Root command `python_env/bin/python .../pressure_quality_control/capture_row13_ids.py` exited0. The first actual HF/TT output-ID mismatch is zero-based ordinal1, after one shared token (`row13_first_difference.json`). This gives the next probe a narrow original-prefill plus one-decode-step target. Server remains idle and no diagnostic client is active.


Stopped ordinary `P150x2/ordinary_generated_v2` gracefully before actual-model localization: server lifecycle exit143, cleanup.complete=true, no survivors or force-kill, dead owned tmux removed. `timeout 60 tt-smi -ls --local` exited0 and lists all four P300C chips (`pressure_quality_control/health_after_controls.log`); no reset was needed. No inference server, diagnostic client, CPU reference or device probe is active. The owned task Docker transport remains for later agentic collection.


Execution-context correction: focused read-only detection confirms the current experiment context is the physical host, with systemd/init.scope, no container or VM detected, no container markers, and motherboard DMI B850M-C (`pressure_quality_control/context_detection.json`). No inspected artifact establishes a reservation container. Earlier wording that no physical-host fallback was needed does not establish compliance with the user's explicit placement requirement. Prior numerical/source results remain historical direct-host diagnostic evidence; they are not final reservation-container release evidence. The inference server was already stopped. Further serving, tt-smi, and TT device probes are paused pending the requested reservation-container connection or explicit user change to that requirement. CPU HF diagnosis and source preparation continue; no new host device operation follows this finding.


Canonical HF boundary capture passed in139.26seconds (`P150x2/pressure_hf_boundary_v1/hf/result.json`), with both saved canonical greedy IDs reproduced. The original generation config retains max_new_tokens128; a separately marked diagnostic stopping criterion ends after the two relevant decisions. Actual hooks enforce8192 prompt then one shared decode token and logits_to_keep1. All64 terminal hidden rows are finite; the bounded CPU tensor artifact is739335bytes. Only top32/logit summaries are retained, with no full prompt-by-vocabulary dump. Owned client exit0/cleanup complete and dead tmux removed. TT comparison has not executed and awaits resolution of the user-required reservation context.


Final prepared TT diagnostic source SHA25277145af673d1f339beba4c93217f620e91cf3f63eec6e59fef8824de8564e received scoped clean-pass after all three probe findings were fixed. All10 MAIN CPU controls pass (`main_row13_boundary_cpu_tests.log`). `pressure_quality_control/root_tt_prepared_plan.json` pins the exact command and inputs but explicitly requires reservation-context revalidation before execution. AutoFix cannot complete the required TT comparison without that connection; no row13 numerical repair has been attempted or declared failed. All task-owned servers/clients/CPU models/tmux sessions are stopped; Docker transport children, private socket/directory and generated untracked TTI `.env` are removed, with zero running Docker containers observed. Unrelated default tmux0 is preserved. No final-stage clean-pass or checkpoint commit is claimed.


The missing required reservation-container connection persisted across three consecutive goal turns and was revalidated in `context_revalidation_turn3.json`: current context is not a container, no Docker container is running, no task tmux server exists, and TTI `.env` remains absent. Independent source/CPU preparation is complete. The goal is blocked on that external execution-context dependency; row13 AutoFix validation and the full release remain incomplete. No numerical repair failure, readiness pass, final clean-pass, commit or push is claimed. Resume with the preserved reviewed probe after the required connection is provided or the user explicitly changes the placement requirement. Mandatory Meta access and approved numerical references also remain required.


### Effective-goal resumption, 2026-09-09

The latest user goal explicitly authorizes this physical host for serialized TT health checks and autoport serving; no reservation container is required. The previous placement blocker is resolved. Row13 correctness and the outstanding release gates remain work. The canonical HF reference tensor was moved outside the release handoff to readiness_vllm; original metadata is immutable and pressure_quality_control/hf_boundary_relocated.json records the pointer-only relocation with unchanged tensor hash. The reviewed TT probe source is unchanged.


### Completed row13 TT boundary control, 2026-09-09

P150x2/pressure_tt_boundary_v2 completed with owned exit code 0, device_cleanclosed=true, and no surviving task processes. The diagnostic-only Shape API fix was independently reviewed and all 11 isolated CPU tests passed. The serving arm reproduces both captured HTTP decisions; eager prefill and decode full terminal logits are bit-exact to the serving trace and use identical page tables. The one-shot arm executes one original 8192-token prefill and retains the same two TT decisions, while its logits differ from four-chunk serving. This refutes a chunk-only correction for the first differing decision; it does not classify all numerical drift or resolve the sentence loop. AutoFix next uses the saved hidden rows for a bounded canonical final-norm/head CPU oracle. Selected precision and context remain unchanged. Small result/manifest/progress/source snapshots are in P150x2/pressure_tt_boundary_v2; tensor rows remain outside doc/tti_release. Both owned diagnostic tmux sessions have been removed; no server is running.


### Row13 CPU terminal oracle

The hash-bound bounded CPU oracle completed with exit code 0 (0.468 seconds), reading selected checkpoint embedding rows and the final norm vector only. Both-phase HF norm and head/softcap calibration reproduce saved FP32 outputs exactly. Neither replacing the final head/softcap nor replacing final norm plus head/softcap restores the differing decode pair order: chunked HF-minus-TT margin remains -0.072992 / -0.067333, versus actual -0.125; one-shot remains -0.244415 / -0.253403, versus actual -0.375. This refutes a terminal-only correction for this pair and directs AutoFix toward the hidden-state producer. These candidate-only projections do not establish full-vocabulary argmax or repair the 128-token response. No model precision/source change was made. Small result and execution JSON are in P150x2/pressure_cpu_tail_v1; source/plan/tests are in pressure_quality_control.


Fresh AutoDebug invocation: the collaboration spawn tool hit its thread limit. The skill CLI sandbox preflight failed with `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`; no model started in attempt v1. The user-provided environment has permission_profile disabled and operator permissions specify danger-full-access, so v2 uses the supported AUTODEBUG_ALLOW_UNSANDBOXED=1 fallback without changing the inspection-only/no-device/no-network/no-secret-read scope. Both attempt commands/logs/manifests remain on persistent readiness_vllm storage. V1 dead task session removed; v2 independent investigator active, root awaits report.


### Row13 pair-direction saved-row control

The bounded CPU projection completed with exit code 0 (0.474 seconds), using only two checkpoint head rows and gamma. Its terminal sign/margin checks pass against the calibrated oracle. Raw and RMS-normalized hypothetical reference pair scores cover embedding plus 30 layer outputs for both phases, arms, and chips. Several layers contribute to the trajectory difference; this is exploratory attribution, not actual intermediate logits or evidence identifying a faulty kernel. No model/source/precision change was made. Results/execution JSON are in P150x2/pressure_pair_direction_v1, source/tests/command in pressure_quality_control. Fresh independent AutoDebug inspection continues before choosing the next identical-input component control.


### Same-input router CPU control

The bounded CPU oracle completed with exit code0 (0.466s). Canonical normalization/unfolded projection, captured normalization/unfolded or folded FP32 projection, and BF16-logit ranking all reproduce the actual expert set and ordering for both arms. Folded/unfolded calibration is below5e-7. BF16-rounded CPU softmax sums are about1.0000–1.0010, compared with actual0.984375/0.9892578; maximum component errors remain0.01171875/0.009765625. Thus BF16 rounding alone does not explain actual scores. Counterfactual24zero-logit padding predicts roughly0.86 total mass, also failing to explain the observed deficit; source independently shows the selected softmax factory explicitly enables padded-width masking. Actual router FP32 logits were not captured, so the discrepancy is not yet localized to matmul versus softmax. Next exact-shape TTrouter replay must reproduce actual scores before interpreting intermediate differences. No production change or overall row13 fix claimed. SmallJSON/execution saved in P150x2/pressure_cpu_router_v1.


### Fresh AutoDebug completed

The independent CLI investigation completed with exit0; report copied as AUTODEBUG_row13_hidden.md. Its task session is removed. It found no obvious broad decoder architecture omission and prioritizes hidden-producer module/routing controls. Root assessment: the report's terminal-path wording is accepted only as "terminal-only correction is insufficient for this pair", not proof that terminal arithmetic never contributes. Its candidate-only argmax must not be read as full-vocabulary oracle output. Its broad architecture source citations include the TTI client HF installation; the actual canonical numerical controls remain bound to their recorded Transformers5.12.1 source. Newer controls completed during the investigation refine its hypotheses: H1 projection is nonmonotonic/no unique causal cliff; H2 expert-set/order changes are refuted for the captured layer26 row, while routing-score error persists beyond ordinary BF16 rounding. Next actual-router intermediate replay follows H2 without changing production. H3 dense/expert/attention precision sensitivity remains untested, not a fix.

## Standalone router replay completed (physical host, 2026-09-09)

Corrected metadata-only observer source SHA `8ce32bcf62520292acc5c04ef0ad98723a2f8e5931882a7cc1e66162332e078e`; MAIN 3 guarded CPU tests pass. Original source rejected for hidden Tensor repr/str reads was never executed. Physical-host health rc0, no reset. `P150x2/pressure_tt_router_v1/result.json`: rc0, device cleanclosed, owned session removed; 2.994 seconds. Both captured history arms reproduce actual production sparse routing, compact indices and compact scores bitexact in unobserved and observed modes. Actual BF16 cast and top-k handoffs exact. FP32 matmul max absolute CPU error 0.003332/0.003530; softmax evaluated against CPU on the actual TT top-k values has BF16-output maximum errors 0.011719/0.009766 and sums 0.984375/0.989258. This verifies a local softmax discrepancy, not its cause or a full-model repair. Next controlled experiment varies approximate math only. No production changes; whole-stage release remains incomplete.

## Softmax numerical A/B completed (2026-09-09)

Corrected diagnostic source `69127494d942069224a996f70cc2c25a598425c0acf77bf2fb282b60ab26d659` uses exported arch-aware init; MAIN5 CPU tests pass. Both v2/v3 complete rc0, cleanclosed, ownership cleanup complete; task sessions removed. Both implicit/explicit defaults reproduce actual production scores exactly for both history arms/chips. Disabling `math_approx_mode` alone reduces BF16 score max error from 0.011719/0.009766 to 0.000977/0.001953. Further changing only `fp32_dest_acc_en` improves score RMSE for both arms and one-shot max error to 0.000488; sums 0.999512 for both. The flag controls multiple SFPU operations, so this is a math-mode finding, not proof of an exponential-only cause. All shape/dtype/memory and input values are fixed. These controls justify a router-specific full-model intervention using existing correctness compute config (HiFi4, approximateFalse, FP32destTrue, packerFalse). This is not yet evidence of a token or response repair. No context or request changes. Artifacts: `P150x2/pressure_tt_softmax_ab_v2/` and `P150x2/pressure_tt_softmax_ab_v3/`.

## Router candidate integrated serving validation and nightly restart

One router-softmax keyword selects existing correctness compute config; current optimized_decoder SHA `498dca892f5d8adcc12d5633ebc60082cc2d8f12a1565b32e7c6b7b97f7e641a`. MAIN guarded config regression1pass; independent source review passes. `ordinary_router_v1` serves actual P150x2 autoport on physical host with sampler audit (no fault injection), unchanged context262144. `smoke_router_v1` health/OpenAI/tinyTTI1/0 at8/8/disabledtracecapture, tool/reasoning/29nonaligned and shared12outputs pass. All4 original selected8192/128 serial forced and EOS controls completed; row13 mechanical loop absent. Synthetic-input interpretation is not a semantic accuracy score.

Full release command launched via `run_nightly_profile.py --profile P150x2 --server-attempt ordinary_router_v1 --attempt nightly_v3`, using unchanged fulldefault `release_specs/P150x2_nightly_ungraded.json` with embedded externalserverfalse/port8000/workflowrelease/ci-nightly. No benchmark filtering/resume variables inherited. Task-only Docker transport restored after archiving its finished manifest under `docker_transport_history/pre_router_nightly`; Docker daemon29.7.2, inference stays external (no inference Docker image). Meta fixture/auth and approved numerical references remain blocking. Candidate acceptance and complete profile performance/API/evals remain pending.

## Mandatory Meta gate AutoFix investigation

`meta_fixture_audit/AUTOFIX.md` records recoverable Gemma registration/preparation gaps plus missing authoritative inputs. A source/tokenizer-only public541-row IFEval diagnostic preserves native Gemma formatting but cannot substitute for Meta: the gated table defines membership/order/final prompts, publicwis-k and Google differ at one key, and pinned Meta versus installed common JSON checkers disagree on a demonstrated response. GPQA authorized questions/options/labels and exact Meta prompt protocol remain unavailable; existing R1 task uses a different32768-token protocol. No access bypass, fabricated references, Meta pass or active TTI mutation. Small audit reports copied here; public input/source fixture archive remains in readiness_vllm, outside doc handoff.

## Short random-output suffix diagnostic prepared

The first fresh128/128C1 benchmark contains repeated markup/thought suffixes. No EOS position is available in the benchmark artifact. `pressure_quality_control/post_nightly_eos/` now binds exactseed0/N8 installed-client reconstruction and selectedzero-based[0,3,4,5] requests; client3.11 offline preparation passed (SHA91965e2e6a9a82b156780a4c1ec0efabb9e18fa75ddcdd1b2c98848b9a80ab20), root and independent reviewer each passed10 guardedCPU controls. No HTTP control has run. It will reject an active/unclean nightly, reproduce all original-body forced128 texts before any intervention, reproduce response-onlytokenIDs, then check EOS exactprefix. Root must fill final completednightly SHA and launch through `launch_owned.py` after nightly_v3; preserve server until that control finishes.

### 2026-09-09 continued fresh P150x2 nightly

The existing `ordinary_router_v1` physical-host autoport server and `nightly_v3` release continued without restart. The 4096/128/C32/N128 workload completed 128 requests with zero failures in 710.1878 seconds; the full default 8192/128/C31/N62 workload then began. No requests, lengths, context, or benchmark selection were changed.

Independent agentic source preflight is saved in `P150x2/nightly_v3/agentic_source_preflight.json`. It establishes installed client API and localhost wiring compatibility, not successful task execution. Actual task downloads, execution, and owned-container cleanup remain to be checked.

Independent output inspection found unresolved opening repetition in selected completed 128/1024, 1024/128, and 2048/128 synthetic-chat rows. Exact matched controls are being prepared for after the full nightly workflow finishes; these observations are not waived as post-EOS without measurement. The previously repaired original row13 serial loop remains separately supported by its actual controls.

Fresh nightly pressure workload `8192/128/C31/N62` completed **62/0** in **741.2007 seconds**, preserving 507904 input and 7936 output tokens. Raw SHA256 `0c0cb685bbcb7dae966d3587d84c171e8a712114b0f90c81b233771f4e1b4261`. The original row13 mechanical loop is absent from this actual concurrent run; full output review is pending. The workflow continued into 16384-token prompts. This is transport/loop evidence, not semantic accuracy or numerical-target acceptance.

Fresh `P150x2/nightly_v3` completed all **19 default text workloads: 901 requests, zero failures**, including a logical 131072-token input plus 128 outputs. Exact raw hashes, counts, and timings are in `P150x2/nightly_v3/text_sweep_summary.json`. This completes text transport collection for this profile. Output anomalies, numerical targets, mandatory quality gates, structured/API/agentic checks, and whole-stage readiness remain open. The original one-based row 13 (zero-based index 12) opening loop is absent in the fresh 62-request pressure output.

### 2026-09-09 text sweep review and pending controls

Independent reviews inspected the completed 128–131072-token workload outputs and verified all 19 raw files against `text_sweep_summary.json`. Four additional opening or recursive loops and several forced-generation tail classes remain unclassified. `pressure_quality_control/post_nightly_eos/pending_queue.json` binds eight prepared groups containing 21 selected rows, including the 131072-token request. Reconstruction ran offline on the CPU; no diagnostic HTTP request was sent during the nightly. Each diagnostic must reproduce the original forced output before changing EOS handling.

A separate canonical HF CPU runner is integrated in `pressure_quality_control/nightly_hf_controls/`, with independent source review and 11 guarded CPU tests passing. Tests cover the installed HF behavior that restores a default EOS value: forced mode uses an empty EOS-ID list without suppressing EOS logits. The runner preserves exact native input IDs, original budgets, and canonical FP32, eval-mode, greedy checkpoint settings. Root verified that its owned launcher rejects an active nightly. Actual HF model loading and inference remain unexecuted. HF controls are proposed for the four opening-loop cases if the TT EOS checks leave them unresolved.

The workflow entered its six structured configurations at 15:25 UTC. The source-only grading preflight is `P150x2/nightly_v3/structured_grading_preflight.json`. Constrained rows require all 100 outputs to satisfy their schemas; unconstrained rows retain strict scores without a 100-percent requirement. Request, provenance, and termination evidence and missing numerical targets remain separate gates. No runtime, hardware, or serving-source adjustment was made during collection.

At 16:25 UTC the fresh constrained `json-unique` row completed 100 requests with no transport failures, but all 100 ended at the unchanged 128-token limit and failed strict JSON/schema validation. This is an unwaived release blocker. Raw SHA256: `13bc99cf31fb4f6d301dc0e2b618811964c69d55f8f0d2a71a64e813ee12d6d2`. AutoFix is investigating the new exact schemas and outputs; historical P150 controls do not waive this run. The remaining release workloads continue unchanged, and no diagnostic request will overlap them.

### Structured controls integration, 2026-09-09 16:51 UTC

The persistent exact-schema control harness is under `pressure_quality_control/structured_unique_controls`. Reviewed v2 runner bytes are unchanged; fixtures were regenerated with persistent paths. Root ran 12 guarded CPU tests successfully. The launcher refuses an active nightly client, binds its final manifest hash only after cleanup, and requires a matching completed TT reproduction before HF. No diagnostic inference has run. Independent launcher review is pending.

The fresh unconstrained unique-schema row completed 100 requests, all at length 128, with strict JSON/schema score 0%. Independent review accepts its ratio0 execution/evidence accounting only. The constrained row remains blocking.

Independent persistent-harness review completed: `P150x2/nightly_v3/independent_structured_unique_launcher_integration_review.json` records a scoped source pass, 12 guarded CPU tests and formatting checks. It does not accept the failed constrained release row.

### Fresh constrained XGrammar result, 2026-09-09 16:56 UTC

All 100 requests completed; strict JSON/schema validation passed for 88. There were 87 valid stop completions, one valid length-limited completion and 12 invalid length-limited responses. Original output budget 128 and context 262144 were retained. The 12 failures remain release-blocking while independent raw review and AutoFix diagnosis run. The unconstrained counterpart is now active.

### Structured collection complete, 2026-09-09 17:08 UTC

The six structured workloads completed 600 requests with zero transport failures. Constrained strict scores are 100/100 for json, 0/100 for json-unique and 88/100 for XGrammar. All unconstrained strict scores are zero; those scores are retained under the ratio0 evidence contract. The constrained 112 invalid responses remain blocking. All 19 text workloads also completed, totaling 901 successful requests and zero failures. The nightly client is now executing parameter-conformance API tests.

Exact XGrammar controls are integrated under `pressure_quality_control/xgrammar_controls`: primary index28, optional index5. Root ran nine guarded CPU tests successfully. The TT/HF control runners are unchanged from reviewed unique-schema controls; the fixture preserves the dataset schema as its exact original string. No diagnostic inference has run.

### Nightly completion and first exact control, 2026-09-09 17:27 UTC

`P150x2/nightly_v3` finished rc1 with complete client cleanup and no survivors. Its final manifest SHA is `f2ef5136c999940eca2fd86cd47c3d8b75dabeb47dfc54453f5299900d073949`. Parameter-conformance subprocess exited0; spec-test dispatch reported two blocks and zero failures. Terminal Bench task setup requested32 CPUs against the actual physical-host Docker daemon's16 CPUs. A subsequent missing metric caused `NoneType * float` in accuracy checking and prevented SWE Bench execution. AutoFix is investigating both harness errors; no model failure is inferred from task setup. Docker reported zero running containers after client exit. The finished nightly tmux session was removed; the original server and task transport remain for pending work.

Exact unique-schema sample0 reproduced the full current release output byte for byte at the original128-token budget (191input,128output, strictinvalid,8.99s). The owned client finishedrc0/cleanupcomplete. The canonical CPU HF control is now running against the same exact fixture. The first release markdown/report data were copied automatically; report formatting and acceptance are under independent review.

### Exact structured canonical controls, 2026-09-09 17:35 UTC

Unique-schema sample0 and XGrammar sample28 each reproduced the current TT release output byte for byte at128 tokens. Canonical FP32 CPU HF also produced incomplete JSON at128 for both exact prompt/schema fixtures: unique0 took168.50s and XGrammar28 took177.97s. Both HF clients exited0 with complete cleanup. This is same-failure-class evidence for those two samples, not a waiver of their rows or other samples. Small summaries and original artifact hashes are in each control directory.

Post-nightly EOS controls have started with the four selected short-workload rows. The execution plan pins the final nightly manifest and unchanged original server manifest. Full-budget text reproduction precedes token observation and natural-EOS classification.

### EOS controls and harness integration, 2026-09-09 17:44 UTC

The four short controls reproduced their full forced outputs and accepted token prefixes exactly; natural generation stopped at the first EOS, removing the flagged suffixes. The2048-input original index107 differed under serial reproduction, so that group stopped before EOS classification and index45 was not attempted. The128-input/1024-output index118 and1024-input/128-output index54 reproduced exactly but contained noEOS and still reached their budgets. Those opening repetitions remain unresolved; the canonical HF54 comparison is now active. All completed owned clients cleaned up. Small results are in `pressure_quality_control/post_nightly_eos/runtime_summary.json`.

Reviewed TTI harness fixes are applied locally: the Gemma recipe now uses native task CPUs instead of an invalid32-CPU override; missing agentic metrics/references and execution errors produce failed records without crashing before SWE. Root ran11focusedCPUtests successfully. The independently reviewed renderer change projects large evidence payloads only in Markdown; report JSON, scalar metrics and acceptance remain unchanged. Agentic runtime retry and reports-only regeneration are pending.

### New canonical opening-loop comparisons, 2026-09-09 17:55 UTC

HF54 returned a coherent65-token random-input explanation and stopped atEOS; native1024input IDs match the actualTTcontrol. First outputdivergence is secondtoken, TT2430 versus HF3847 after shared818. HF118 similarly returned coherent172-token random-input analysis withEOS, within its original1024budget; TTproduced its exact recursive1024-token loop withoutEOS. These failures require modeldiagnosis and are not cleared by earlier postEOS controls. A fresh inspection-only repoAutoDebug CLI is running under owned session `autodebug-opening54`, using the persisted operator authorization also used by the prior AutoDebug run. No hardware or inference actions are delegated.

The reviewed finalsampler reconciler ran successfully on50pinned records and wrote `P150x2/nightly_v3/final_sampler_accounting_v1`. Original snapshotcounts remain11163sampled/11154completion/9lookahead; finalfiles contain11169sampled/11154inferredprefix/15lookahead, with alloraclechecks complete. The timing explanation is qualified; originalHTTPresponses and observation-time snapshots were not retained.


### Resumed remediation, 2026-09-09 20:39 UTC

Verified current physical host qb2-120-p11t02 and the existing owned ordinary_router_v1 P150x2 server. Stage10 clean-pass remains recorded. Nightly_v3 finished rc1; no passing Stage11 claim. Prior AutoDebug54 CLI ended at its usage limit with no report; a forked AutoFix investigation is recovering source evidence and preparing focused controls. The completed 8192/16384 EOS groups stopped on exact-text mismatch, so they do not classify their tails. Root launched the prepared 32768 group with unchanged request lengths. Independent agentic-retry review and reports-only regeneration are running without device access. Requested approved fixture/reference paths asynchronously; mandatory Meta gates remain unresolved. No reset, lock removal, source precision change, commit, or push in this resumed turn.


Reports-only regeneration completed in P150x2/nightly_v3/report_presentation_v1: Markdown 132,154 bytes, complete JSON unchanged, all 10,625 scalar values and acceptance exports identical. The retained 28 blockers remain failing; no new inference or regrading. Agentic retry source received independent scoped clean-pass (19 guarded CPU tests); expensive runtime is deferred until current model-loop diagnosis resolves because implementation changes would invalidate newly collected results. Prepared index54 boundary diagnostic source is under independent review.


The final 131,072-token EOS control completed rc0 with exact original text/token reproduction and an exact natural prefix through the first EOS; its post-EOS tail is controlled. The 32,768 group stopped at an exact-text mismatch and remains unclassified. All HTTP diagnostic clients cleaned up. The exact owned ordinary_router_v1 launcher received SIGTERM and closed with rc143, no survivors or force. Its dead session and the stopped task Docker proxy session were removed; the proxy socket/private directory were removed and Docker listed zero running containers. The first health command used a nonexistent python_env/bin/tt-smi path (rc127); corrected timeout60 tt-smi -ls --local passed and listed all four chips. No reset was needed. The independently reviewed index54 HF boundary capture is now running in owned session opening54-hf-v1; TT execution will follow its exact reproduction gates.


Current index54 HF and TT boundary captures both completed rc0 with complete process cleanup. HF reproduced its original two decisions. TT reproduced the HTTP pair and passed trace/eager complete-logit and page-table equality gates, then closed devices cleanly. All 30 layer terminal rows are retained outside docs for the next canonical tail/localization control. No model fix is inferred from successful capture. Small results and lifecycle manifests are copied to P150x2/opening54_hf_boundary_v1 and P150x2/opening54_tt_boundary_v1. Both diagnostic sessions finished; no inference server is active.


The canonical FP32 tail oracle completed rc0 in 1.82 seconds, including one bounded full-vocabulary embedding scan. HF final norm/head and full-vocabulary argmax self-calibration passed. Both canonical head-only and canonical norm-plus-head substitutions on current TT rows retain the TT decode argmax rather than the HF argmax. This refutes a tail-only correction for the current first difference; upstream hidden-state production remains under investigation. Small result/execution evidence is in P150x2/opening54_cpu_tail_v1. No model or precision policy changed.


Independent actual-oracle review confirms the current tail-only refutation and all source/result/cleanup bindings (pressure_quality_control/opening54_cpu_tail_review.md). The source audit finds 17x64=1088 sliding-cache and 9x128=1152 full-cache allocation coverage for the fixed 64-token SDPA read ending at1088. HF and TT source masks both exclude position0 at query1024/window1024. Those checks do not prove kernel equivalence, but provide no current case for an over-allocation fix. Matched canonical forced-branch and fresh full-prefix source controls are being prepared; no hardware or model process is active.


The HF one-decision branch control received scoped independent clean-pass with 10 guarded CPU tests and is running in owned session opening54-hf-branch-v1. Only the second decision is forced to the observed TT alternative; all later scores, native input, generation configuration and original budget remain unchanged. This is diagnostic evidence, not a release waiver. An additional fresh reviewer spawn for the 1025-prefix diagnostic was rejected with agent thread limit reached; the existing non-author reviewer is reviewing that source separately. These scoped reviews do not establish whole-stage fresh-review completion.


The canonical HF one-decision branch control completed rc0 with exactly one forced decision and complete cleanup. It used all 128 tokens without EOS and differed from TT at free-decision ordinal4. Its actual text gives an unsupported but coherent quadratic-equation walkthrough; TT instead invents an image and mechanically repeats fractions. The control demonstrates branch sensitivity in topic interpretation but does not reproduce or resolve the TT repetition. This remains required work, with no waiver. Small output and lifecycle evidence is in P150x2/opening54_hf_forced_branch_v1; the finished session was removed.


The fresh 1025-prefix diagnostic completed rc0 with original trace/eager reproduction gates intact, one pure nonaligned prefill, and clean device/process shutdown. It retains the TT global argmax; the HF-minus-TT candidate margin is -0.125 versus cached decode -0.375. Full logits differ between phase paths, but switching to whole-prefix prefill does not restore the HF decision. This is not a cache-only fix or model-quality pass. Small result/lifecycle/log evidence is in P150x2/opening54_fresh_prefix_v1; its finished session was removed. A newly available fresh xhigh AutoDebug subagent is inspecting current evidence, while bounded layer-direction oracle preparation continues. The earlier CLI weekly quota remains a tool-specific limitation; diagnosis is proceeding through the available isolated agent.

## Current hidden-state diagnosis and pair-direction control

The bounded CPU direction oracle completed with exit code 0 and complete owner cleanup; final-margin self-calibration error was below 2.9e-6. Intermediate head-direction changes are distributed and do not identify a causal layer. A fresh isolated source-only AutoDebug report is recorded at `pressure_quality_control/current_hidden_diagnosis.md`. It identifies dense fast-LUT GELU versus checkpoint tanh GELU as a concrete hypothesis for a same-input test, with router, folded weights, SDPA, norm/reduction, and later decode alternatives still open. The original 128-token fraction loop remains unresolved. No production fix or release waiver follows from these diagnostics. All device/server jobs are stopped; root retains serialized hardware ownership.

## Layer-11 activation probe launched

Root launched `opening54_activation_v1` in the owned `gemma4-tti-release` tmux socket using the exact command in `pressure_quality_control/opening54_activation/COMMAND.json`. The source-only review and nine guarded CPU tests passed; plan SHA256 is `3777483a4ba526af4c523fd5e864c3afb65165a1ee9c77755281a22032900181`. Preflight `timeout 60 tt-smi -ls --local` returned 0 with all four chips. Two bounded TTNN binding-only probes returned 0 for tanh construction and native activation metadata. No reset was needed. The runtime preserves the selected policy and native model outputs while comparing layer-11 products and local down projections. Device results remain pending at launch. Source snapshots and small logs are copied; tensor rows remain under readiness_vllm.

## Layer-11 activation result

`opening54_activation_v1` completed with exit code 0 in 72.87 seconds, clean device close, complete owner cleanup, and no survivors. Its task session is removed. Original serving/traced and observed eager full logits remain exact; repeated native product/down computations are exact. On identical decode gate/up values, product relative L2 versus canonical tanh is 0.10243/0.06643 on the two chips for native fast-LUT, versus 0.00228/0.00186 for fused tanh. Local down-output relative L2 versus canonical product projected through the deployed weights improves from 0.06391/0.06392 to 0.02500/0.02154. Prefill products and downstream comparisons also improve. This verifies a local activation discrepancy; no full-model intervention has run and the sustained fraction loop is not fixed by this evidence. A dense-only full-budget serving A/B is being prepared. No precision policy, expert activation, request length, or production source was changed.

## Dense-only original-budget A/B launched

The layer-11 activation evidence received independent scoped clean-pass (`pressure_quality_control/opening54_activation/STAGE_REVIEW.md`). Root launched `opening54_dense_tanh_ab_v1` using the exact command recorded in the corresponding source snapshot. Plan SHA256 is `4cf9a935a3f8f687025e82f64c05ae777befe300ce85738f38066ee30cfbdb1b`; eleven guarded CPU tests and source bindings pass. A fresh serialized `timeout 60 tt-smi -ls --local` returned 0. Both arms retain the exact 1,024-token prompt and forced 128-token completion, actual sampled token feedback, selected policy, context 262144, HMA allocation, and device sampling. Native traces and state are released before tanh capture. The baseline must reproduce all 128 saved tokens; candidate interpretation remains pending. No production source changes.

## Dense-only full-budget result

`opening54_dense_tanh_ab_v1` completed with exit code 0 in 77.07 seconds, clean device close, complete owner cleanup, and removed task session. The unmodified baseline reproduced all 128 saved token IDs exactly, including the sustained fraction loop. The dense-only tanh arm produced a coherent explanation that the supplied text was random words and reached its first checkpoint EOS at zero-based generated index 84. Both arms still generated the original forced 128 tokens; the candidate post-EOS newline continuation is retained and is not used to judge pre-EOS answer quality. Both arms used actual token feedback and independent fresh trace/KV state with identical allocation plans. This is a verified repair candidate for index54, pending independent review, minimal production integration, and nearby/second-case regression checks. It is not a release pass or performance measurement.

## Second repetition case A/B launched

Root launched `opening118_dense_tanh_ab_v1` with the command in the matching source snapshot after `timeout 60 tt-smi -ls --local` returned 0. Fourteen guarded CPU checks and fixture/source bindings pass. Plan SHA256 is `22e93ed696a6e17f8a9fb25817a3b5a47221e58e9a510d8cb954221dc8d98bc6`. This keeps the exact 128-token input and forced 1,024-token output budget, tests full baseline token equality, and changes only dense GELU in the second arm. Context, serving sampler, HMA, precision policy, and expert activation are unchanged. Main source remains unmodified until the current-source baseline is complete.

## Second-case result and production integration

`opening118_dense_tanh_ab_v1` completed with exit code 0 in 121.79 seconds, clean device close and complete cleanup. Its baseline reproduces all 1,024 saved IDs, including the recursive sentence loop. Dense-only tanh produces a coherent clarification and reaches checkpoint EOS at zero-based generated index 54; the original forced 1,024-token budget is preserved. The task session is removed. Independent first-case review is `verified-fix-for-index54`; independent second-case inspection confirms the output and lifecycle, with its written review finishing.

Root integrated the exact three-site dense GELU_TANH patch after checking the original source hash and `git apply --check`. Source SHA256 changed from `498dca892f5d8adcc12d5633ebc60082cc2d8f12a1565b32e7c6b7b97f7e641a` to `29dbdc03e9d25043286809c66c4aa9cc3d610f3ad5cd59b27ff49f4d214a8687`; the previous router repair and expert activation remain intact. Before-source, patch and integration provenance are in `pressure_quality_control/dense_tanh_fix/`. `python_env/bin/python -m black --check --target-version py312 models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py` passed. The default Black check also returned 0 but emitted a target-version warning, so the explicit Python 3.12 check is the recorded validation. This change is Python only; no build is required for these three expressions.

Following another successful serialized health check, root started the generated P150x2 autoport server with `serve_profile.py --profile P150x2 --attempt dense_tanh_v1 --sampler-audit` in the owned tmux socket. Live integrated-source tiny TTI, feature/parser and shared qualitative validation is next. Context and selected precision policy remain unchanged. No release readiness or performance improvement is claimed.

## Integrated smoke interpreter correction

The first integrated smoke launch used host `python3` and failed during TTI selection imports with `ModuleNotFoundError: aiohttp`, before the smoke manifest and first OpenAI request. This repeats the already diagnosed launcher-environment failure; no model output was collected in that attempt. The checkout interpreter has aiohttp, confirmed with `importlib.util.find_spec`. Root relaunched unchanged smoke source as `python_env/bin/python .../run_existing_server_smoke.py --profile P150x2 --server-attempt dense_tanh_v1 --attempt smoke_dense_tanh_v2`. No dependency installation, model/spec alteration, or hardware reset was needed. Original failure log is `pressure_quality_control/dense_tanh_fix/smoke_launch.log`.

## Integrated smoke completion and HTTP regression

All `smoke_dense_tanh_v2` commands returned 0 with complete child cleanup: the first OpenAI request, one TTI 8/8 request, feature/parser/nonaligned checks, and shared qualitative runner. Independent raw/qualitative review is finishing; its evidence distinguishes source integration chronology from an absent direct startup hash attestation for optimized_decoder.py. The smoke client session is removed. Root launched the pinned exact original-case HTTP control `dense_tanh_http_v1` after smoke client quiescence.

The completed old Docker transport manifest/log were archived byte-for-byte with relocation hashes under `docker_transport_history/ordinary_router_v1/`. A fresh task Docker transport is running for the upcoming agentic workflow. It exposes only the existing host Docker CLI to task clients; model serving remains the generated external autoport vLLM process. This is not a Docker model-server fallback.

## Exact integrated HTTP validation and nightly rerun

`dense_tanh_http_v1` completed with exit code 0 and complete owned-client cleanup. Both original requests retain their exact payloads and full budgets. Forced response text and token observations match the successful A/B tanh outputs exactly; natural EOS responses match the same prefixes through the first EOS. The client binds the integrated source SHA before each request and preserves the autoport/context/server identity. Its task session is removed. Independent smoke review is scoped-pass at `P150x2/smoke_dense_tanh_v2/STAGE_REVIEW_DENSE_TANH_SMOKE.md`; its raw manifest remains unchanged and requires that separate review verdict.

Root started `python_env/bin/python .../run_nightly_profile.py --profile P150x2 --server-attempt dense_tanh_v1 --attempt nightly_dense_tanh_v1` in the owned tmux socket. The exact child command is captured in the new release manifest and uses this checkout's `--runtime-model-spec-json`, `--tt-device p300`, `--workflow release`, `--limit-samples-mode ci-nightly`, external server flags false, service port8000 and full context262144. All default benchmark/API/agentic rows remain selected. The verified activation change affects model outputs, so rerunning expensive collection is necessary; historical nightly_v3 results remain preserved. Missing mandatory Meta fixtures and approved target/reference inputs remain blocking. No readiness pass is implied by starting this workflow.

## Early rerun output triage

The first two text workloads completed 8/8 and 256/256 requests with no transport failures. A text-only repetition scan identified a suspicious opening recursive sentence in the new 128-input/128-output/C1/N8 index0. This is EOS-unclassified and lacks a matching canonical HF control, so it is not yet a confirmed model defect or regression. AutoFix is preparing the exact C1/N8 request binding and later matched controls without changing active collection. Other scan hits include coherent clarifications followed by repeated delimiters; no pre-EOS failure is inferred for those suffixes. The existing sampler audit covers marked seeded API requests and does not supply token-level EOS evidence for ordinary concurrent benchmarks. The two repaired original cases remain independently verified.

### Dense tanh nightly pause and matched opening control

Paused `nightly_dense_tanh_v1` after new row0 opening recursion on the exact prior C1/N8 input. The outer tmux launcher disappeared after SIGTERM while its tagged client remained; root recovered the manifest-bound client and all three owned descendants with identity-checked cleanup. `nightly_pause_cleanup.json` records no survivors. Status143 is explicitly an operator interruption, not a collected child exit. Three completed raw rows are preserved; the partial fourth is not a result. HTTP metrics verified zero running/waiting before the new exact forced/token/EOS control. The server stays live only for that control; full HF follows server shutdown. No model-fix acceptance or release pass is claimed.

Matched opening0 HTTP completed with exact forced reproduction, exact128 native input IDs and identical natural128-token repetition with no EOS. This is not a post-EOS suffix. Server owner2184445 received identity-checked SIGTERM and recorded143/cleanup complete; task proxy stopped and removed its socket. Docker ps was empty and the task tmux socket has no server. The bounded canonical HF control is running only after those shutdown checks.

Canonical opening0 HF completed rc0/cleanup complete in109s. Same128input IDs and original128budget produce a nonrecursive random-text explanation/list, noEOS bybudget; it does not reproduce TT recursion. Exact earlier TT input match has coherent17token/EOS16 output, so integrated source has a newly confirmed quality regression. Next isolated hypothesis is packed expert activation, retaining dense/router repairs. Fresh native AutoDebug spawn hit agent thread limit; existing isolated source investigator is refreshing the report as an explicitly context-limited fallback, not a fresh-context claim.

### Packed-expert-only opening0 trial

`opening0_expert_tanh_ab_v1` completed rc0, device cleanly closed and ownership cleanup complete in74.45s. Root independently ran15 guarded CPU tests before serialized execution. Full128 baseline IDs reproduce currentHTTP exactly. The candidate changes only actual packed-expert GELU0 to tanh while preserving dense tanh/router/policy/context and all other callarguments. It changes at generated index1, removes mechanical recursion, and reaches firstEOS atindex125. All30layers have fourprefill/twodecode hostcalls;180candidatecalls changed,90densecalls unchanged. The candidate overstates some input characteristics; perfect semantics or exactHF wording are not claimed. Same-input product/down arithmetic and fullbudget nearby54/118 validation are pending before any production edit.

### Expert nearby checks and local harness retry

Both nearby A/Bs completed with exact full current dense-tanh baseline reproduction, unchanged original budgets and cleanup/cleanclose. Index54 candidate avoids the fractionloop but overstates input repetition and adds unsolicited programming advice, reaching128 withoutEOS; independent review keeps that quality finding active, with exactnative token ngram checks. Index118 candidate is nonrecursive and firstEOSindex73 within the original1024forcedbudget. Neither is a whole-stagepass. Localv1 failed only on CPU UInt16 comparison; its source/failure snapshot was preserved. The helper now losslessly casts a validation-only view toint64, retaining native routes;15actualdtype guards and independent rereview pass. Localv2 is running serially after118cleanclose.

Localv2 failed on TTShape slice-indexing in CPU helper; archivedbeforefix. One-line listmaterialization and strictshape mock produced16passingguards andindependentrereview. Localv3 completed68.62s rc0cleanclosed/cleanupcomplete; both262144-elementtraced/eager vectors andnativeproduct/downreplays exact. Actualexpert111 scores/routes heldfixed. ProducterrorrelativeL2 small/mixed: prefill .003058→.003076 and .002188→.002199; decode .002294→.001985 and .005149→.005148. Downerror remains~.018–.020 withsomeimprovementsandoneslightworsening. Thisdoesnotestablishlargeexpertarithdefect; fullmodelrow0sufficiency remainsseparate. Expertproductionpatchnotapplied; row54qualityfindingawaitsmatchedHFbranchcontrol.

Canonical expert54 single-branch control completed rc0/cleanup complete in134.98s. Exactly one ordinal5 intervention; all other128decisions follow canonicalgreedy. First free TT/HFdifference isordinal7. HFalsohits128withoutEOS, controllinglength/truncation, butdoesnotreproduceTT repeated-many-times claim orprogrammingdetour. Thosecontentfindingsremainactive. Fullboth-phaseexpertpatchremainsunapplied. Nextcontrolledhypothesis separatesprefill-only anddecode-only expertactivation oncurrentrow0 and54, keepingallothermodelpolicy/context/requestbudgetsunchanged.

Both expert-phase A/Bs completed rc0cleanclose/cleanup. Exact128baselinegatepassed for0/54. Row0prefill-onlyEOS117 anddecode-onlyEOS108 bothremove recursion butretainunsupportedcontent; decode-onlyassertssummaryintentabsentfrominput. Row54decode-onlyEOS89isboundedclarification; prefill-onlyhits128noEOSandclaimsspecificcommonwordsrepeatwithoutinputsupport. No phasefamilyisacceptedforproduction. Reviewer's prior-binding concern was from a stale draft and is explicitly retracted: finalunchangedplanscontain andlauncherchecksallthreepriorresultrecords. Rootinitialpytestpathwasmistyped andran0tests; correctedisolatedguardcommandpassed21tests (author/reviewerhadalreadypassed). NootherTTjoboverlapped. Nextcurrentrow0boundarywillseparateproducerfromterminalmath; original100-token accuracyfixturelocatedforbroaderregression.

Current row0 boundary source review and root guard rerun passed (24 tests). Root initially used a nonexistent test filename, exit4/no tests; corrected to test_opening0_boundary.py and obtained24/24. HF-first owned attempt opening0_hf_boundary_v1 started on the authorized physical host; no TT job overlaps. Original128 request budget and context262144 preserved; two decisions are diagnostic capture only. RELEASE_STATUS refreshed with completed expert arithmetic, nearby and phase-control findings; expert production patch remains unapplied.

Current row0 boundary attempts completed: HF rc0/cleanup with exact two decisions and finite33 rows per phase; TT rc0/clean device close/cleanup, both complete serving/eager vectors and tables match current local-v3. These controls permit boundary interpretation, not release acceptance. Source and small result artifacts copied with provenance; tensors remain under readiness. After independent source review and root15/15 guarded tests, root launched original_accuracy_current_v1 using the prepared owned launcher, exact original161 native prompt/100 teacher decisions and context262144. No precision or production activation change made.

Original accuracy current baseline completed rc0, clean device close and cleanup. Exact original100-decision ranked-label metric passes99/100 top1,100/100 top5/top100 (historical selection reported96/100 top1). Context262144 and logical161-token prefill retained. Observed99 trace replays and99 host teacher-token refreshes,0 device feedback reuse; this is teacher-forced accuracy, not a free-generation loop fix or release eval. Small run artifacts copied; source and actual review remain scoped.

Current row0 canonical CPU tail completed rc0/cleanup. Both full-vocabulary HF self-calibrations pass. TT head-only and norm+head substitutions retain the same TT decode choice, with HF-minus-TT pair margins -0.36534/-0.39178 versus canonical HF +0.98287. Tail-only correction is insufficient at the first shared-history divergence; this does not establish the cause of later recursion. Root then launched reviewed opening0_hf_branch_v1, forcing only the first differing TT choice at ordinal1 after canonical calibration and retaining the original128-token natural-EOS budget. No device or server job overlaps. Small tail artifacts/reviews copied; projection tensors and weights not copied.

Single-branch current row0 HF control completed rc0/cleanup in25.25s. Only ordinal1 was forced after checking canonical choice. HF emits17 tokens, EOS16, without recursion; it first differs freely from TT at ordinal6 after six shared choices, where it says empty and TT opens another quote. The empty-request assertion is imperfect and is not a desired output target. The first choice alone does not explain sustained recursion. Next focused preparation is the current matched-history boundary at absolute133; no production change or precision-policy change follows from this control.

Acceptance-scope decision after independent tail/branch reviews: test coherent both-phase expert tanh on original100-decision accuracy and six instruction-bearing shared qualitative prompts before extending random-benchmark branch localization. Checkpoint activation agreement plus removal of all three observed mechanical loops justifies this candidate test, not integration acceptance. Local expert arithmetic remains small/mixed; unsupported grounding descriptions remain recorded, exact clarification wording/EOS is not a golden target. Target133 localization preparation is held. Candidate accuracy keeps existing90/98/100 thresholds, source/policy/context and native161 input. Only after passing will a provisional one-line integration support actual-server validation; substantive regressions require repair or reversion. No expert production change yet.

Expert candidate first launch failed before output-directory/process/device creation: the reviewed launcher SHAaa96f257… disagreed with its plan self-pin60d128fb…. The source itself matches CPU_VALIDATION and the reviewer-supplied final SHA; the stale nested plan pin was missed by preparation checks. AutoFix author will correct the self-pin and extend the actual-main preflight guard, with old plan/failure preserved. This is a harness preflight failure, not model accuracy or hardware failure; no reset needed. Root had passed23 quality guards plus4 expert guards separately before launch.

Corrected candidate plan73df3fb9… changes only the stale nested launcher hash. Root independently compared old/new plans and passed the additional launcher entrypoint guard; reviewer independently passed28 guards and exercised actual main with a pre-process sentinel, accepting corrected/rejecting old plan. Runtime source is unchanged. Root retried the same fresh attempt original_accuracy_expert_tanh_v1 after rereview; it is now running as the sole device job.

Uniform expert-tanh candidate accuracy completed rc0, clean device close/owned cleanup in68.78s. Existing gates pass98/100 top1,100/100 top5/top100. Relative to current baseline, only ordinal74 changes from HF rank1 equations to rank2 the after the common teacher prefix Step1: Set up. This one-point top1 loss remains explicit; joining predictions across teacher-forced decisions is not a natural generated sentence. Original rank2 fixture ordinal1 remains. Observed240 expert host-call substitutions across all30 layers/both phases,90 unchanged dense calls,99 replays/99 teacher refreshes/0feedbackreuse. Actual review and final server preparation review are pending before one-line provisional integration. Small artifacts copied; no tensor/weight cache copies.

After scoped actual accuracy and server-source clearance, root provisionally applied exactly one packed-expert GELU0→GELU_TANH replacement. multichip_decoder.py SHAe64e290a…; dense/router source unchanged. Guarded ledger update preserves the byte-identical prior record and explicit supplement, active ledger SHA62793db1…. Integration record binds candidate accuracy result. Black --check --target-version py312 passes; Python-only change needs no build. Prelaunch binder completed before server start. Root launched python_env/bin/python doc/tti_release/serve_profile.py --profile P150x2 --attempt expert_tanh_v1 --sampler-audit in persistent execution session4169; manifest records runner2239526 and exact command/environment. Physical-host external autoport serving, no Docker transport or reservation. No other device job overlaps; next work is existing-server smoke and original0/54/118 controls, not full nightly yet.

Expert-tanh P150x2 server ready after about90s, context262144, exact autoport entrypoint, parser flags gemma4/gemma4,32 slots and selected policy. Live-source binder passed and wrote readiness_vllm/P150x2/tti_release/expert_tanh_server_validation_v1/plan.json from actual prelaunch bytes and live manifest. Root launched python_env/bin/python doc/tti_release/run_existing_server_smoke.py --profile P150x2 --server-attempt expert_tanh_v1 --attempt smoke_expert_tanh_v1 in client session75121. No Docker/local model-server fallback; server execution session4169 remains live.

Current expert-tanh tiny/features/shared commands all exited0 and cleaned clients; tiny raw completed1/failed0. Written spec uses runtime_model_spec/runtime_config; docker_server=false/local_server=false, service_port string8000, workflow benchmarks, disable_trace_capture=true and autoport impl path verified. Root corrected an ad hoc envelope/port-type assumption while inspecting it; actual spec is valid. Features pass including29-token logical unaligned input and both parsers. All12 shared outputs were read: coherent/on-topic, with some longer answers partial under unchanged256-token client budget; finish/EOS metadata is not retained. Content review remains pending. Original0/54/118 HTTP controls completed rc0/cleanup in68.96s with exact candidate forced/natural behavior. Initial copy stopped at a2.17MB indented HTTP JSON; it was copied with lossless whitespace compaction, dual hashes and JSON-equivalence verification, original retained under readiness. All small artifacts now copied; server session4169 remains live pending review.

Integrated expert-tanh review returned scoped-pass to advance to fresh profile/release verification, retaining accuracy delta/partial answers/random-input grounding findings. Review copied; smoke manifest updated to reviewed pass with its pre-review bytes preserved. Stopped prior Docker transport manifest/log archived under docker_transport_history/dense_tanh_v1; sudo -n docker ps returned no containers. Root started task-only Docker proxy in execution session40624; inference stays external autoport on physical host. Root launched python_env/bin/python doc/tti_release/run_nightly_profile.py --profile P150x2 --server-attempt expert_tanh_v1 --attempt nightly_expert_tanh_v1 in session30648, with fresh cache/results and unchanged context/default benchmark matrix/ci-nightly sampling. Current TTI checkout remains v0.18.0-based with recorded reviewed local harness changes. Mandatory Meta fixtures and approved numerical targets remain blocking, independently of workflow command exit.
Actual TTI argv: ["/home/hous/dev/tt-metal/python_env/bin/python", "run.py", "--dev-mode", "--model", "gemma-4-26B-A4B-it", "--runtime-model-spec-json", "/home/hous/dev/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/release_specs/P150x2_nightly_ungraded.json", "--tt-device", "p300", "--workflow", "release", "--service-port", "8000", "--no-auth", "--skip-system-sw-validation", "--disable-trace-capture", "--limit-samples-mode", "ci-nightly"]
Actual TTI cwd: /home/hous/dev/tti-release-gemma4/tt-inference-server

At 2026-09-10T00:38:07.662226+00:00, fresh expert-tanh nightly collection remained active. The first three completed benchmark points totalled 268 successes and zero transport failures; this is partial collection, not readiness. R1 GPQA exited 1 because Idavidrein/gpqa requires authenticated access; its wrapper continuing does not constitute an accuracy pass. A source-isolated SWE selected-task accounting repair passed 14 CPU controls and independent scoped review: missing submissions/verifier errors retain the intended denominator and block release. Active TTI source remains unchanged until current collection finishes. Evidence: `pressure_quality_control/swe_accounting_fix/` and its sibling review.

At 2026-09-10T01:10:57.380652+00:00, fresh nightly had ten independently reviewed complete text points, 792 successes and zero transport failures, with the8192/C31 point active. Review records strong raw repetition watches without inferring EOS. Reviewed post-nightly exact-payload controls are prepared for128/128 indices1(C1/N8),28(C32/N256), and1024/128 index116(C32/N128); actual tokenizer/native IDs and HTTP/EOS checks remain pending finished client cleanup. Their source folders and independent reviews are under pressure_quality_control/expert_tanh_opening*. Credential-presence metadata again found no current HF environment token or standard user token files; no contents accessed. The structured protocol follow-up found no new justified harness repair; its unposted issue draft is not a waiver.

At 2026-09-10T01:59:44.752070+00:00, all19 current expert-tanh text benchmark artifacts were complete:901 successful requests and0 failures, including131072 input tokens. The source/effective-spec matrix reconciliation confirms the same901-request matrix as historical nightly_v3. These are transport/count results, not numerical performance acceptance or unrestricted quality. Structured-output collection began01:57:51UTC with unchanged context262144 andOSL128/C4/N100. No source changes, diagnostic HTTP, hardware probes or resets overlapped collection. MandatoryGPQA/Meta/reference gaps and raw-output watches remain blocking.

### 2026-09-10 02:51 UTC — current structured comparisons

The current P150x2 `nightly_expert_tanh_v1` text sweep completed all 19 points (901 successful requests, zero failures), including the unchanged 131072-token prompt. Constrained JSON completed 100/100 strict JSON/schema-valid responses. The unconstrained JSON baseline completed 100/100 requests but 0/100 strict whole-response JSON/schema-valid outputs: every raw response was Markdown fenced. The actual ratio-zero acceptance rule permits this measured baseline; it is neither a strict-conformance pass nor an issue waiver. Missing numerical targets remain blocking. Independent evidence is in `P150x2/nightly_expert_tanh_v1/review/STRUCTURED_POINT_1_REVIEW.json` and `STRUCTURED_POINT_2_REVIEW.json`. Constrained JSON-unique is still running with its original 128-token budget and 262144 context. Server, client, and agentic Docker transport remain owned and active; no additional inference or device checks ran during collection.

### 2026-09-10 03:03 UTC — current constrained JSON-unique failure

Current point 3 completed 100 requests with zero transport errors, but 0/100 strict JSON/schema passes. All 100 finished with `length` at the original 128-token budget; independent parsing found unterminated strings. The exact current acceptance checker reports `incomplete_workload`. Rendered inputs span 188–196 tokens at context 262144, and all prompt/schema bindings pass. Shorter name/email objects satisfy the schemas, so these valid requests cannot be rejected or given an automatic budget waiver. The current UUID manifest differs from historical controls. See `P150x2/nightly_expert_tanh_v1/review/STRUCTURED_POINT_3_REVIEW.json`; the failure remains unwaived and blocking. The remaining release collection continues before any diagnostic runtime.

The SWE accounting candidate additionally passed all 31 targeted repository tests (four new) in an isolated source overlay. Actual imported candidate normalizer/parser paths and hashes were checked, and all 47 checked active source files remained unchanged. Evidence: `pressure_quality_control/swe_accounting_application_review/DURABLE_TEST_VALIDATION.json`. Integration remains deferred until collection ends.

### 2026-09-10 03:27 UTC — current xgrammar constrained result

Point 5 completed 100 requests with zero transport errors and 89/100 strict JSON/schema passes. Eleven responses are `incomplete_length`; index 34 is a complete schema-valid object at the 128-token limit and is correctly classified `valid_length_limited`. The exact checker blocks `incomplete_workload` for the eleven invalid members, not for index 34. Upstream substring extraction reports 88%; independent strict whole-response scoring verifies 89%. This remains below the required 100% constrained gate and is unwaived. Current sample 28 input/schema bindings match the historical canonical fixture; complete raw-output association is under review before any reuse claim. See `P150x2/nightly_expert_tanh_v1/review/STRUCTURED_POINT_5_REVIEW.json`. Point 6 is collecting; server and active TTI source remain unchanged.

### 2026-09-10 04:01 UTC — API completion and agentic runtime

The current v2 VLLMParamConformanceTest exited 0 at 03:54:14 UTC (started 03:36:05). The independent completed-sampler audit found 52 marked records and 11,528 final sampled steps with no recorded oracle failures. Actual per-case counts/request-ID and accepted-usage reconciliation await the final persisted report; the wrapper intentionally deletes its temporary output after loading in-memory blocks. The separate legacy tests workflow remains pending. See `P150x2/nightly_expert_tanh_v1/review/API_CONFORMANCE_REVIEW.json`.

Terminal-Bench began at 03:54:14 with the unchanged five named tasks. All five task sandboxes are now running with one CPU each; current labels and task mounts bind their exact identities in `P150x2/nightly_expert_tanh_v1/agentic_ownership/`. The earlier 32-CPU setup defect is avoided. The existing TTI recipe retains its 48-GiB per-task memory override, and Harbor warns that this differs from task defaults; no leaderboard-comparability claim is made. Inference remains external generated autoport vLLM at localhost:8000; Docker is only task transport. No task scores are available yet.

A source audit identified the universal structured-document closure gate at 128 tokens as an unsupported local addition, not the original performance benchmark contract. The narrow isolated fix is reviewed and passes 128 guarded tests plus formatting; actual six-row regrading removes only two incomplete-workload blockers, preserving all strict scores/classifications and six missing structured targets. Evidence is in `pressure_quality_control/structured_gate_origin/` and `structured_budget_gate_fix_review/`. Patch `c59ebddcb2821b4b0671ba5fdf98ca8a3b660d99ad00d2195b4986ee70173492` remains unapplied while collection runs. This is a pending harness correction, not a waiver or quality pass.

### 2026-09-10 05:25 UTC — three agentic results verified

Terminal Bench has three completed trials, zero execution errors, and two still running. HF model inference and portfolio optimization each earned reward 1 with verifier evidence independently checked. Financial document processing earned reward 0: seven tests produced three passes and four assertion failures after the generated classifier misclassified two documents. AutoFix found no concrete API, context, parser, or verifier defect; the unsuccessful solution and its denominator remain unchanged. Task-level results do not establish aggregate release acceptance or a TT-versus-HF regression. Reviews are under `P150x2/nightly_expert_tanh_v1/review/TERMINAL_*_TASK_REVIEW.*` and `pressure_quality_control/financial_task_review/`.

Exact Compose-project container and network queries confirm Harbor cleaned up the three completed tasks. Caffe and password recovery remain active and reached episode 31 by 05:22 UTC. A bounded review found one password-search command was killed for an unrecorded reason; a subsequent search changed method and returned normally. All reviewed responses parsed, and their prompt plus requested completion lengths fit the unchanged context. No server intervention or timeout change is justified. The original 10,800-second task deadlines remain in force. The scoped progress review is `P150x2/nightly_expert_tanh_v1/review/TERMINAL_CAFFE_PASSWORD_PROGRESS_WATCH.json`.

Inference remains the same external autoport server. No additional model requests, TT probes, resets, active source edits, or Docker cleanup mutations overlapped this observation. The existing memory/time overrides limit comparison with native Terminal Bench runs; HF task runtime also exceeded its native 900-second timeout. Final API case reconciliation, SWE, remaining profiles, mandatory Meta gates, and numerical targets are still pending.

At 2026-09-10T05:33:10.617088+00:00, the two reviewed harness fixes passed a combined isolated preflight: 159 targeted tests (31 SWE/agentic, 85 structured, 43 text), with actual candidate imports verified and all 61 active source files unchanged. Only the expected two production and two test files changed in the overlay. Exact patches, before/after hashes, guards, logs, and post-release integration commands are saved in `pressure_quality_control/combined_harness_fix_preflight/`. This is harness validation, not release acceptance; integration remains pending collection completion.

At 2026-09-10T05:52:05.447462+00:00, the prepared offline v2 report helper passed independent scoped review after fixing actual string-port handling, pure structured validator imports, and consistency between derived runtime sections. All 26 CPU tests passed, including completed current structured peers in synthetic SWE/report exports. Strict scores and missing-target/Meta failures remained unchanged. Actual imports and 58 unchanged active source files were independently verified. Evidence: `pressure_quality_control/report_regrade_helper_review/`. This validates the helper only; final inputs and the completed owner must still be pinned and reviewed before any regeneration. No active checkout or runtime changed.

At 2026-09-10T06:09:56.134465+00:00, a bounded refresh of current official public release material found no applicable new numerical targets, exact mandatory Meta protocol/fixtures/reference, or row-specific waiver. Exact-model central target entries remain TODO; the closed reference issue concerns 31B, not this 26B model. The published Google GPQA number was not imported as an incomparable task target. Fetch limitations and source links/hashes are preserved in `pressure_quality_control/public_release_refresh_20260910/REFRESH.md` and `refresh.json`. No credentials, dataset rows or weights were accessed, and no active source/runtime changed.

### 2026-09-10 06:55 UTC — Terminal Bench completed; SWE started

All five Terminal Bench trials finished by 06:55:08 UTC. HF model inference and portfolio optimization earned reward 1; financial document processing earned reward 0. Caffe and password recovery each reached the unchanged 10,800-second agent deadline, then verification returned reward 0. The five-trial mean is 0.4, with two `AgentTimeoutError` records. This is a CI-subset observation, not release acceptance; timeout rows and missing comparable references remain unresolved. AutoFix is reviewing their actual commands, parser records and verifier results without extending deadlines or repairing sandbox solutions.

Passive server metrics show active generation immediately before the deadlines, then zero running/waiting requests and zero KV usage by 06:54:58, followed by successful SWE traffic. The small bound excerpt is `P150x2/nightly_expert_tanh_v1/review/TERMINAL_TIMEOUT_SERVER_WINDOW.json`; it establishes aggregate progress and request cleanup, not per-request cancellation-ID attestation. No TT probe, reset or extra HTTP was performed. Exact-project Docker queries confirm all five completed Terminal Bench task containers and networks are absent.

SWE began at 06:55:09 UTC with five workers and the original five selected instances. Exact full container IDs from this fresh run log, plus observed names, images and creation metadata, are recorded in `P150x2/nightly_expert_tanh_v1/agentic_ownership/swe_agent_initial_inventory.json`. Docker remains task transport only; inference is the same external autoport server. Active source remains unchanged and no final SWE score is available yet.

At 2026-09-10T07:18:27.434061+00:00, independent review verified the two Terminal Bench timeout classifications across 121 bound artifacts and all 111 completed response/debug/trajectory records. Both original 10,800-second timeouts and zero rewards remain, with denominator five and two errors. Caffe verification had three passes/three failures; password verification had two failures due to the absent required output. No demonstrated required harness/API/mechanical-model repair was found. Optional parser recovery robustness does not explain or waive the timeout. Canonical quality and comparative latency attribution remain unresolved. Evidence: `pressure_quality_control/timeout_classification_independent_review/`.

The active SWE configuration uses five workers, the standard 250-step limit per task, and 32,768 completion tokens. Its context-checking wrapper obtains exact rendered counts from the serving tokenizer and enforces prompt plus completion against 262,144. Docker configuration observations found no explicit per-worker CPU or memory caps; these configured resource values and exact task-image associations are recorded in `agentic_ownership/swe_agent_resource_observation.json`, without leaderboard comparability claims. No active source or runtime configuration changed.

At 2026-09-10T07:35:08.802720+00:00, a separate acceptance-contract audit found that the local Terminal Bench parser overgeneralized its earlier setup-failure guard. Harbor deliberately verifies timed-out agents and independently counts scored trials and exceptions: current five scored trials and two timeouts are overlapping counts. The gate is absent from the pinned original TTI parser. Independent actual-function controls confirm a classification defect without changing the measured score. A minimal isolated correction is being tested; it must retain all raw timeout/reward/denominator data and block genuine setup/API/verifier or missing-scoring errors. Missing references remain blocking. See `pressure_quality_control/terminal_timeout_gate_origin/` and `timeout_classification_independent_review/REPORT_PARSER_SCOPE_ADDENDUM.*`. The earlier no-repair conclusion concerns task execution, not this newly identified reporting defect. No active source or runtime changed.

### 2026-09-10T07:43:17.103037+00:00 — scored-timeout reporting candidate

Terminal Bench is complete with five scored trials, two AgentTimeoutError records and raw mean0.4 (report score40.0). The isolated parser candidate passes44 CPU cases and actual old/current raw regrading; current missing-reference acceptance remains FAIL, and old zero-scored setup errors remain blocked. Independent candidate review, combined three-fix preflight and a separate offline two-agentic-block helper extension are running. Active TTI source and collection remain unchanged. SWE still has five active workers and no final trajectory; bounded passive server-log observation confirms continuing requests/generation without observed engine errors.

### 2026-09-10T07:52:30.587491+00:00 — report fixes reviewed together

The Terminal timeout parser v2 passed independent46-case review after adding job-finish timestamp consistency. Three exact isolated patches (SWE denominator accounting, structured budget classification, Terminal scored-timeout classification) passed205 combined CPU tests;61 active source files remained unchanged. The separate two-agentic-block report helper passed34 independent tests and19 pinned import checks. Reviews are under `pressure_quality_control/{terminal_timeout_gate_fix_review,combined_harness_three_fix_preflight_review,report_regrade_helper_terminal_review}/`. No active patch application or final report regeneration has occurred. Final report/owner/SWE inputs must be bound after the live collection finishes. Scores, failed rewards, errors and missing-reference/target blockers remain intact.

### 2026-09-10T08:41:13.023225+00:00 — first SWE submission

Matplotlib25332 submitted a476-byte patch; trajectorySHA`8767a8f5dd5d00ee75463257a72159440d401a4031a90a83e4e068f7c90fcc1c`. It records61attempts and55 retained tool-call responses; six explicit no-tool feedback messages account for the gap. The largest recorded prompt is20717 tokens, safely within262144 with the32768 completion budget. Submitted is not a verifier pass; task review and official verification remain pending. Four other agent tasks continue. The exact logged Matplotlib container remained present in a read-only check after task completion; mini-swe cleanup is object-destruction based, so root will recheck after client ownership ends. No container mutation or server probe occurred.

### 2026-09-10T10:44:27.261406+00:00 — SWE collection remains active

Matplotlib is the only submitted agent task; its prediction entry exactly matches the saved 476-byte patch and identifies `openai/google/gemma-4-26B-A4B-it`. The independent submission review preserves all six no-tool rejections and the untested restored alignment/drawing behavior. No verifier score is available. Four agents remain active. The passive server-log evidence at 10:38 records 310 completed chat requests since SWE startup, without ERROR/Traceback/Watchdog markers. All reviewed report patches and the offline helper remain unapplied while collection runs. No extra HTTP, hardware probe, reset, container mutation or source change occurred.

### 2026-09-10T11:58:51Z — supervisor stopped unusable SWE client after environment loss

This SIGTERM was deliberate supervisor action, not an unexplained runner crash.
Read `P150x2/nightly_expert_tanh_v1/swe_environment_loss.json`: repeated Docker
inspect through the client's actual DOCKER_HOST confirmed all four unfinished
task container IDs absent. The launch log binds them to --rm sleep2h; the client
had run for over5h. The installed environment returns missing-container failures
as observations rather than terminating the agent. Continued inference could
not recover task files or execute changes in these deleted environments.

The supervisor recorded exact PID/start-ticks/command identity and preserved
prediction/trajectory hashes before sending SIGTERM only to SWE client2543696.
Server/engine/watchdog/proxy were not signalled. SWE rc=-15; release workflow
rc=1 and acceptance FAIL30blockers; nightly manifest records complete cleanup.
Missing task results remain execution failures with no inferred score or reduced
denominator. A future fail-fast Docker environment candidate with CPU source-
integration tests is isolated at /home/hous/dev/gemma-4-26B-A4B-it/evaluation_repair;
it has not been installed into TTI or mini-SWE. Do not silently rerun into the
failed output directory or advertise this collection as a complete quality pass.

### 2026-09-10T12:06:55.115785+00:00 — SWE environment loss and recovery

The original release exited1 at11:58:52 and copied its Markdown and data JSON with30 blockers. `swe_environment_loss.json` records a supervisor SIGTERM to the identity-checked mini-extra client after the four unfinished task containers were confirmed absent. The containers were launched with `--rm sleep 2h`; the agent run lasted over five hours. Continued HTTP progress had not proved task-environment health. Matplotlib’s completed trajectory/prediction survives; the four missing tasks remain environment failures with no inferred model score.

All five exact original task container IDs are now absent. Client ownership cleanup completed, and a process scan found no remaining TTI/mini/Harbor clients. The retained server responds200 to health and reports zero running/waiting requests and zero KV usage. No device reset or inference-server signal occurred. Fresh xhigh AutoFix is preparing an isolated task-lifetime repair and a retry limited to missing SWE evidence, preserving task steps, command timeout, context and completion budgets. The report helper remains held because SWE normalization is missing and the actual report uses uppercase `P300` identifiers. Full legacy API validation has now started in owned session17109 on this same server.

### 2026-09-10T12:27:03.139057+00:00 — full legacy API passed

The separate legacy workflow finished at 12:24:57 UTC with 22/22 distinct passing cases, ten mechanical cases and 50 audited requests. Completion tokens 11,507 match accepted prefixes; assertion-time sampled 11,513 includes six lookahead draws. Acceptance scope is tests, with no waivers; this is not release readiness. The final manifest SHA is `13898516481f46433e8bafa0f96e102f3189478551524efe7734e13f8d7eb543`. Process cleanup is complete with no survivors. `P150x2/legacy_expert_tanh_v1/POST_LEGACY_QUIESCENCE.json` records health 200, zero running/waiting requests and KV usage, and no client candidates at 12:26:25. Selected EOS controls are now serialized on the unchanged server before SWE recovery.

### 2026-09-10T12:30:34.938683+00:00 — EOS preparation validator repair

Opening 1 preparation child exited 0 with complete cleanup, but the outer launcher exited 1 before HTTP: validate_prepared called `.encode()` on TTI's native content-block list. The original failed attempt is preserved and is not an EOS pass. AutoFix is correcting the same assumption in opening1/28 and116 helpers without changing payloads, token IDs or budgets. The minimal SWE retry collector separately passed independent 32-case source review after inspect-error, terminal-status and prior-owner-cleanup fixes. Its materialized plan binds the completed legacy owner; runtime waits for corrected EOS controls and outer server/process supervision review.

### 2026-09-10T12:42:32.985189+00:00 — selected EOS controls completed

The corrected v2 helpers passed independent 78-case review. Opening 1 reused its unchanged successful tokenizer child artifact; openings 28 and 116 prepared normally. HTTP controls for 1 and 116 reproduce original forced text exactly, preserve exact observed token prefixes, and still reach 128 tokens with EOS respected and no EOS token observed. Their repetition is not classified as after-EOS behavior. Opening 28 serial output differs from its original concurrent benchmark, so it stopped before token-observation/EOS classification and remains inconclusive. All three clients cleaned up with no survivors. Small summary: `P150x2/nightly_expert_tanh_v1/review/SELECTED_EOS_CONTROLS_V2.json`. AutoFix is preparing canonical HF controls with exactly served native IDs and the original budget before any further long agentic collection.

### 2026-09-10T13:01:45.749567+00:00 — canonical continuation controls and prompt-format qualification

Root reviewed the frozen canonical launcher and both exact native-input contracts, then ran the existing FP32 CPU/SDPA HF implementation serially. Opening 1 completed in 134.7 seconds with 79 tokens and EOS at index 78; it described incoherent input and invited clarification without the TT self-quotation. Opening 116 completed in 238.6 seconds with 128 tokens and no EOS; it assumed an unsupported sentiment-analysis intent and requested input/examples, without the TT image/fraction behavior. Both clients exited 0 and completed cleanup. Sources, full generation policy and first result passed independent review; the second result review is underway.

The original benchmark renders 142/1,038 tokens, then keeps suffixes of 128/1,024, dropping the chat opening. These same-ID contrasts are therefore labeled truncated continuation stress. Under `.agents/skills/qualitative-check/SKILL.md`, they cannot alone establish a pass/fail serving-quality verdict. Earlier references to quality findings mean observed content differences, not a demonstrated mechanical model bug. Full-template TT/HF controls will preserve the same messages and roles with natural EOS and the unchanged 128-token output budget; original benchmark artifacts and geometry remain unchanged. The proposed one-token HF branch experiment is deferred until that prompt-format prerequisite is handled.

### 2026-09-10T13:26:49.940077+00:00 — full-template comparison passed; SWE recovery started

Both same-message TT/HF pairs retain complete 142/1,038-token chat inputs and the original 128-token natural-EOS budget. All four responses describe disconnected multilingual input and offer clarification; earlier stress self-quotation and image/fraction behavior are absent. All four reach the budget without EOS and remain partial, so no perfect-completion claim is made. Independent actual paired review is scoped clean-pass (`a1c6ce948bf95337ab42310cb7ef20ab2360268d4f3a0e3eeec21bf7ab64fcd9`). This satisfies the comparison prerequisite for SWE recovery without waiving original stress evidence, index28 inconclusiveness, accuracy, targets or release gates. Small lossless selected-input/output evidence is `P150x2/FULL_TEMPLATE_QUALITATIVE_EVIDENCE.json`.

At 13:22 UTC, the unchanged server returned health200 with zero running/waiting requests and zero KV usage; no clients remained. Root then launched the reviewed selective SWE recovery in owned session1620. Its outer manifest is `P150x2/swe_recovery_expert_tanh_v1/launch_manifest.json`; collector output is `/home/hous/dev/tti-release-gemma4/recovery_P150x2_swe_expert_tanh_v1`. All four missing task containers were observed running under the exact run label. The actual third config uses `container_timeout: infinity` and preserves `--rm`; task steps, command timeout, context and completion budgets are unchanged. Matplotlib is retained and all five IDs remain required. No official score exists yet.

### Recovery reporting preparation — 2026-09-10 13:54 UTC

The revised offline recovery report helper received a scoped independent clean-pass (`pressure_quality_control/report_regrade_helper_recovery_review/SOURCE_REVIEW_V2.json`, SHA256 `6e7645cbcf6184de978701d490d8506589c8a2c78776622e84a264956d9f908d`). Its 25 guarded CPU tests passed and reproduced the original P300 SWE failure block (subprocess return code -15) exactly. The helper matches the actual official verifier schema and preserves original and selective-recovery timing separately. Final successful collection, verification, source integration and real report materialization remain pending; this is preparation evidence only.

Before any later harness integration, six changed Python before-images were archived (116,094 bytes), together with four absent-new-file assertions, at `pressure_quality_control/pre_harness_integration_source_snapshot_v1/manifest.json` (SHA256 `90084c4c551064326d0cd8a01bd5f4711663edccc5934d2315175008322f7524`). This explicitly preserves the historical source epoch; current report API source checks remain live. No private configuration, cache, weights or raw trajectories were copied.

### Recovery monitoring and handoff preparation — 2026-09-10 15:00 UTC

The four-task recovery remains active, with only the preserved Matplotlib prediction currently finalized. The collector continually checks exact owned task-container identities; the latest observer record reports no collector error and four running model requests. These observations are liveness evidence, not SWE scores or performance acceptance. Compact observations are retained in `P150x2/swe_recovery_expert_tanh_v1/runtime_observations.jsonl`.

An immutable 18,342-byte copy of the generation server manifest is now `P150x2/swe_recovery_expert_tanh_v1/generation_server_snapshot.json` (SHA256 `c371d10938d10cf6e2803649a61d1f91d448936c0d194a98c532bb5042d5e309`); `GENERATION_SNAPSHOT_COPY.json` records copy time, the matching recovery launch ID, and ten checked archived-source bindings. Actual terminal lifecycle remains a separate future verifier input. The frozen reporting helper and its completed independent review have been copied; its actual run plan remains unmaterialized.

`pressure_quality_control/stage_commit_scope/REFRESH_20260910_1407.json` refreshes hashes for the existing 12 tt-metal, 13 vLLM and 61 TTI stage-owned source paths and names newer evidence candidates. All three indexes were empty; this is not a recursive staging allowlist or final source inventory. `BUILD_ENV_RECHECK.json` records named-environment presence only: both Garage credentials remain absent, so the CI wrapper would use a cold cache. No new build or Docker command was run for this check; the prior successful local Ninja build and runtime install remain documented in `kernel_build_result.json`. No runtime source integration, model change, reset, report regeneration or stage commit occurred in this interval.

### Selective recovery survives prior environment lifetime — 2026-09-10 15:26 UTC

The four exact initial recovery containers were still running after at least 7,278.144482 seconds (78.144482 seconds beyond the original native two-hour lifetime). `P150x2/swe_recovery_expert_tanh_v1/TWO_HOUR_LIFETIME_EVIDENCE.json` binds the compact atomic owner snapshot, exact IDs/creation times, fresh collector observation and still-unfinished task set. Independent review `review/TWO_HOUR_LIFETIME_REVIEW.json` (SHA256 `0b7666d52355882c8f978579d561be5542f11b39a0820731f1616e2190605a1f`) returned a scoped clean-pass: reviewed health code refreshes the current inventory and rejects absent or stopped unfinished tasks before publishing the owner record. This verifies selective recovery container survival only. Collection, official scores, final cleanup, future paired native-harness rollout and whole-stage readiness remain pending.

### First completed recovery task — scikit-learn — 2026-09-10 15:50 UTC

`scikit-learn__scikit-learn-14629` reached native `Submitted`; its 1,775-byte patch SHA256 `c0a31b186f1c91d76a9dec8679fa6ef86b0f2891817ce2465d590db5a7ddf7d5` exactly matches its saved prediction and trajectory submission. `P150x2/swe_recovery_expert_tanh_v1/scikit_learn_submission_binding.json` records the trajectory hash and 69 API calls. Independent `review/SCIKIT_LEARN_14629_SUBMISSION_REVIEW.json` (SHA256 `94bcedb4d1ca15098b23b56e52aab253f6f9bec8cca87ba4dc75718403afac01`) confirms provenance/classification only. The maximum saved prompt plus completion budget is 34,850 + 32,768 = 67,618, within 262,144; cumulative input usage counts repeated history and is not a context length. One missing-command formatting error was corrected.

There is no passing test of the actual patch in this trajectory: the successful reproduction used unrelated installed scikit-learn 1.9, the mock only reproduced the reported bug, and the post-patch local run failed during import. The reviewed native non-login shell configuration was preserved; no TTI setup regression is established. Official verification and patch correctness remain pending. The Matplotlib seed is preserved and three recovery tasks remain active. The completed task’s retained container record is historical, not proof that its container still exists.

### Second completed recovery task — SymPy — 2026-09-10 16:28 UTC

`sympy__sympy-13551` reached native `Submitted`; the 905-byte one-file patch SHA256 `2953a0601eea7a829ec329e13beb27db56c4c2ee54f42dfa4da0dc8c64daef26` matches its prediction, trajectory submission and final displayed exit. `sympy_submission_binding.json` and independent `review/SYMPY_13551_SUBMISSION_REVIEW.json` (SHA256 `feb3553783c6907ca6561187c195bf7c8e1b970b98d61d65fa4af5a7bc81a3f5`) are under `P150x2/swe_recovery_expert_tanh_v1/`. The 73 query attempts reconcile as 54 saved responses plus 19 no-tool feedback events; rejected raw responses were not retained, limiting causal diagnosis. Maximum saved prompt plus completion budget is 36,741 + 32,768 = 69,509, within 262,144. Prior Matplotlib and scikit-learn predictions remain preserved.

Patch correctness remains unverified. Its non-polynomial fallback avoids the original reported 9/2 result but stays unevaluated; the custom script’s FAILURE compares the substituted unevaluated expression directly with 15/2 without a subsequent evaluation, so it does not prove an official arithmetic failure. No product tests were run. The agent initially printed the diff before the submit marker, which the native first-line rule correctly treated as ordinary output; it later submitted with the proper marker. These are recorded task/protocol limitations, with no proved serving/parser/context harness defect. Official verification and final cleanup remain pending; astropy and Django continue.

### Terminal reference-comparison exception — 2026-09-10 16:59 UTC

Adopted the independently supported numerical-comparison-only issue waiver for the existing offline corrected P150x2/P300 Terminal result in `P150x2/nightly_expert_tanh_v1/row_waivers/terminal_reference_comparison.json` (SHA256 `c93d409f0099d461134c4590bae2940ba98c5ad6abda1c99093b53f22baf5303`). The result remains 40.0% across five configured tasks, including both verified scored timeout zeros and their error details. The waiver links [TTI issue4176](https://github.com/tenstorrent/tt-inference-server/issues/4176), [PR4331](https://github.com/tenstorrent/tt-inference-server/pull/4331), and [PR4424](https://github.com/tenstorrent/tt-inference-server/pull/4424). It addresses the uncollected exact-model reference and CI-subset/full-reference mismatch, not model quality or execution failures. The original failed report is unchanged; reviewed accounting integration and actual final report binding remain pending.

The fresh review at `pressure_quality_control/waiver_eligibility_review_20260910/` found current TTI references under `reference_config/` at commit `ba636c30d4cdc6a3982af1671ba310d68717867f`. Its published Terminal14.0 fallback is labeled Terminal Bench Hard in the cited technical report, without comparability to this fixed five-task Terminal-Bench2.0 protocol. Published GPQA82.3 does not resolve gated collection or supply mandatory Meta results. No qualifying waiver was found for the71 all-profile performance-target gaps, six mandatory Meta rows, GPQA access failure or missing SWE verification. These are evidence requirements, not an added approval prerequisite. No published score was imported into the frozen configuration, and the stage remains incomplete.

Public-author GPQA source recovery (2026-09-10 17:14 UTC): fresh AutoDebug established byte identity of the authors’ Diamond CSV with recorded Hub revision `633f5ee89ab8ad4522a9f850766b73f62147ffdd`; see `pressure_quality_control/gpqa_author_source_diagnosis/AUTODEBUG.md` and provenance JSON. The native loader overlay now has a separate offline AutoFix worker checking cold/warm cache, choice RNG, prompts, CI row selection and scoring. Active serving and TTI source remain frozen. No R1 evaluation or score has been produced by this recovery, and the mandatory Meta protocols remain separate unresolved gates.

2026-09-10 17:30 UTC dataset recovery update: R1 native author-source loader equivalence passed on CPU; candidate patch remains unapplied and is under independent review (`pressure_quality_control/gpqa_author_loader_autofix`, `gpqa_loader_candidate_review`). A separate official Meta source investigation found versioned model-independent IFEval541 and historical GPQA-main448 input candidates; the latest public GPQA table is five-shot and incompatible with the required strict answer protocol. Fresh scoped review is assessing the explicit versioned route (`meta_authoritative_source_refresh`, `meta_protocol_adoption_review`); old gated Llama-table byte identity and task pass are not claimed. Read-only common-client dependency probe found NLTK3.10.3 available, pkg_resources absent with setuptools84, and punkt_tab resources absent. No dependencies installed or checker imported. Scorer compatibility, native Gemma single-render/BOS behavior, references and actual evaluations remain work.

2026-09-10 17:47 UTC scoped review outcomes: `gpqa_loader_candidate_review/STAGE_REVIEW.md` is clean-pass for the frozen, unapplied author-source loader candidate. `meta_protocol_adoption_review/STAGE_REVIEW.md` is clean-pass-for-specified-route for versioned official public Meta inputs with original cookbook scoring; old gated Llama-table byte identity is not an added goal requirement. Neither verdict passes an eval or Stage11. Parent read both reports and proceeds with their explicit subsequent work: owned R1 native-completions payload/recovery preflight; isolated Meta scorer compatibility and fixture wiring. Meta preparation uses the pinned tokenizer’s native `enable_thinking=False` default, explicitly, on future TT and canonical controls; no paper score is adopted as a threshold. Current SWE recovery still has retained Matplotlib plus completed scikit-learn/SymPy and active Astropy/Django. No active source patch integrated, no official SWE verifier or new evaluation launched.

2026-09-10 18:13 UTC scoped approvals: pinned Meta scorer compatibility and Meta fixture wiring both received independent clean-pass for their frozen CPU/source scope. The wiring review corrects the preparation prose to 63 total EvalConfig entries, 62 wholly unchanged, and original3 Gemma tasks preserved alongside2 added Meta tasks; two unused all-null GPQA metadata columns have Arrow null types without value changes. No actual quality or Stage11 pass is implied. Canonical CPU preflight is being prepared; heavy generation remains serialized with serving on this16-thread/~256GB host. A suspected until=[] stop-string production bug was refuted by the actual HarnessCLI parser; a legacy offline parser caused that probe discrepancy. R1 owned-recovery preflight is also correcting its client-construction/RNG initialization order before finalizing expected payload hashes. No serializer fix or active source integration occurred.

2026-09-10 18:34 UTC: Parent read the independent Meta fixture and canonical CPU preparation reviews. Canonical batch-one runner review JSON SHA256 `148475061ab3cf9bd4ecb5514d2a8bb5fef80db122142cc269ce20e51712b97d` is a scoped clean-pass, with real phase quiescence, complete199-response collection and truthful CPU-reference report integration still required. Earlier HF peaks were152109716/152111352 KiB (about145.1 GiB), not a measured full-budget bound. The R1 owned preflight review found that direct execution of the native shebang script conflicts with exact owned argv validation. Root preservedv1, prepared separate externalv2 using the pinned native interpreter plus unchanged native script/arguments, reproduced the failure and successful repair with two real synthetic CPU children, reaped both, and passed the19 existing guards. No evaluator, model, HTTP request or device operation ran in these controls. Independent rereview is pending. SWE recovery still has three predictions and two active tasks; active client source remains frozen through official verification.

2026-09-10 18:46 UTC: Astropy reached Submitted; `P150x2/swe_recovery_expert_tanh_v1/astropy_submission_binding.json` binds its 770-byte patch to prediction, trajectory and final exit. Django is the sole remaining SWE client; no official verification or patch-correctness pass exists. Native task environment and test availability are under a focused source-only AutoDebug investigation. GPQA owned-launch v2 and combined client/spec v2 each passed fresh scoped review, read by root; all integration remains pending.

A new actual Meta async-client control supersedes the earlier manual payload EOS-string assumption. Both unchanged native commands use `stream=false` and `stop=[]`; the earlier `_create_payload` probe explicitly passed the EOS string that `amodel_call` omits. Original source/fixture command registration remains valid, but the prior canonical CPU runner review cannot be used as runtime approval until its literal-string stop behavior is corrected and rereviewed. Token EOS IDs remain distinct and unchanged. No canonical or TT Meta inference has run. Old reports stay immutable, and the corrected evidence is versioned separately.

2026-09-10 19:04 UTC: Root verified and applied the reviewed combined TTI and outer patches, then refreshed the prepared spec manifest. Both git apply checks/applies exited0; the recorded after-check confirms all13 active source hashes and the manifest match the tested candidate. The earlier source-freeze ordering ended after clean cancellation of SWE generation invalidated by the confirmed task-shell bug; no official verifier will be run to accept that invalid generation. Allfive original SWE tasks require fresh collection under a reviewed activation repair. Original raw submissions and failed/cancelled manifests remain unchanged.

Native R1 GPQA-only recovery is now running through the reviewedv2 owned launcher on the unchanged P150x2 expert_tanh_v1 server. `P150x2/gpqa_recovery_expert_tanh_v1/plan.json` has SHA256 `03025c0a5c0ae105aa4401d31ce9802f3a6940e04a408efbae79123c169c95b8`; prelaunch server identity, zero requests/KV, completed/cancelled client cleanup and23 current source files are pinned. Original40-row selection,32768 completion budget, T1/top_p.95/top_k20/seed42, native completions/SSE route and full context262144 remain unchanged. No score/pass exists yet. Client/model source remains frozen during this collection.

2026-09-10 19:23 UTC: Native R1 GPQA recovery exited1 after the audit hook raised KeyError while attributing a cloned/preempted request generator. The engine and owned server exited1 and cleanup completed; no inference server or evaluator remains. The reported0.225 score is invalid:39/40 samples contain inference errors. The immutable launch snapshot, terminal ownership, compact engine trace and post-exit serialized four-chip tt-smi health pass are bound in `P150x2/gpqa_recovery_expert_tanh_v1/FAILURE_SUMMARY.json`. No reset was needed. A fresh AutoFix reproduces the sampler attribution failure at the actual TT host-sampling boundary and is preparing a repair without changing production sampling or RNG. A new server and fresh native collection remain required.

Root completed eight serial, local-image Docker shell/import controls for Django, Matplotlib, SymPy and scikit-learn; all corrected bash-lc cases use testbed and import the required project dependencies. Together with the prior three Astropy controls, all five images are covered. All eight new containers are independently absent after exact-identity cleanup; the task-only Docker proxy remains live. See `pressure_quality_control/swe_shell_activation_autofix/ROOT_REMAINING_IMAGE_RESULT.json`. These are environment controls, not task correctness scores. Allfive tasks require fresh native generation under the corrected shell with unchanged native sampling policy and no reused predictions.

2026-09-10 19:24 UTC: Root read the fresh CPU-reference v2 clean-pass review and integrated the seven reviewed TTI files; exact after-hashes match. Both reported validation defects are resolved, and no reference score has been inserted. The approved canonical CPUv2 runner now runs in owned session6885 under materialized plan `9c3fa1cafa62eb37baafa6d477d3ea2ee1081a6e82c0cba3e9000f8eaecff42e`. Command: `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_canonical_cpu_preflight_v2/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_canonical_cpu_preflight_v2/plan.materialized_v1.json --plan-sha256 9c3fa1cafa62eb37baafa6d477d3ea2ee1081a6e82c0cba3e9000f8eaecff42e --ack-serialized-cpu-phase`. Actual phase binds cleaned server/SWE/R1 terminal evidence, no listener/client and205.37GiB available memory at admission. Model load completed in10.81seconds; generation started. It uses canonical Gemma4ForConditionalGeneration FP32CPU SDPA,16threads,batch1,original109/90CI rows,1280/2048budgets,native default non-thinking and exact async-compatible token EOS boundary. Keyenv: offlineHF/datasets, OMP/MKL16, tokenizer parallelismfalse. Serving and officialSWEverification remain serialized until this phase finishes. Rawgeneration stays external; no score or runtimepass yet.

2026-09-10 19:32 UTC: Parent read the fresh sampler selectivity clean-pass review;50 supplied and8 additional independent CPU boundary controls pass, original4failures reproduced. Auditpatch plus reviewed hash integration remains deferred until canonicalCPU generation/scoring finish because that phase pins orchestration.py. Root preserved its identical original source externally and prepared a narrow historical-source report candidate with20 before/after controls through actualreport blocks; independent review pending, activeCPU unchanged. CanonicalCPU has completed its firsttwo rows at397/180generated tokens with naturalstops; parent read exact prompts and completions, with no aggregateaccuracyclaim.

2026-09-10 19:38 UTC: Root read the SWE shell/fresh-cohort review and integrated its accepted four-line Docker-only activation fix plus regressiontests. Exact afterhashes match. The review leaves a preparation-only stalecatalogpin finding; versioned refresh also anticipates the reviewed auditregistry literal update to orchestration SHA14fb15ff4745b38a357297340a7e9c0aa78bad1839a4614e510dc33a7c034b8e. GPQAretry has the same futureownerpin requirement. All actualCPU sources remain unchanged; no serving/officialverifier is active. A finalvalidreference orfreshSWE/GPQAscore is not yet available.

### Current recovery preparation — 2026-09-10T20:01:51.730594+00:00

Canonical CPU reference remains active with 10/199 completed rows; no quality scores yet. Scoped fresh R1 v4 and all-five SWE official verifier review is clean (`pressure_quality_control/gpqa_v4_swe_verifier_review/STAGE_REVIEW.md`). Meta collection two-pin refresh also passed (`pressure_quality_control/meta_owned_collection_review/STAGE_REVIEW_V2.md`). These are preparation verdicts only. CPU generation and scoring retain the original pinned owner source before audit integration.

Benchmark source diagnosis refuted the earlier assertion that every default characterization point needs a numerical target. The original spec is EXPERIMENTAL with no configured performance checkpoints; native status policy permits informational performance comparisons. AutoFix is preparing metric/provenance validation that preserves every workload and all configured targets, while reporting unconfigured comparisons honestly. No numerical performance qualification or release pass is claimed.

### Benchmark characterization repair integrated — 2026-09-10T20:26:48.589568+00:00

Independent review `pressure_quality_control/benchmark_characterization_review/STAGE_REVIEW.md` returned scoped clean-pass. Applied the exact seven-path TTI patch `adb3a85be5164361d714c67eecf08d2c20ccd54d1268327a1deff282a3c8ecc7`; every after-image matches its ledger and `git diff --check` passes. Python-only, no build required. Validation includes307 focused/neighbor tests,81 independent negative controls, all71 unchanged native configurations and the original three configured smoke targets. The retained25 benchmark rows pass measurement checks with comparison explicitly not configured; no numerical performance qualification or full release pass is claimed. Existing failed report artifacts are unchanged. CPU plan and all25 pinned paths were verified unchanged. Actual report regeneration still awaits valid R1/SWE/Meta results.

### Native quality-comparison policy diagnosis — 2026-09-10T20:41:45.285231+00:00

`pressure_quality_control/agentic_reference_comparability_v1/AUTODEBUG.md` distinguishes missing actual metrics from originally unconfigured comparisons. Pinned native R1 and SWE have no numerical references. A valid native R1 score therefore remains measured and quality-ungraded (`NA`), not a numerical quality pass or missing-result waiver. Current published82.3 uses thinking-enabled full Diamond and is not adopted for the native nonthinking CI40 recipe. A local agentic parser branch incorrectly converts missing references into failed execution; an isolated AutoFix is preparing its removal with actual-score validity/error/accounting checks preserved. No model-hardcoded NA-failure rule or new waiver is integrated. Mandatory Meta2 canonical-reference comparisons remain unchanged. Actual R1/SWE collection is still pending; old failed attempts remain invalid.

The prepared native official-verifier adapter has an independent review finding: its validator expects a five-row report while the unchanged native command reports the full dataset. The adapter remains isolated for repair/rereview, including retry-caching checks. No TT server or official verifier is running. Canonical CPU reference has completed22/199 rows at this observation.

### 2026-09-10T21:05:09.955694+00:00 — optional-reference parser integrated

Root read the independent scoped clean-pass and integrated the exact two-file v2 patch into the native TTI checkout. All before/after hashes, git apply check, git diff check, and all 25 active canonical CPU plan bindings pass. The reviewed 97 native tests and independent parser controls remain the validation evidence; they were not repeated after exact-byte integration. Actual finite results without configured references remain NA/quality-ungraded; missing or invalid results and required Meta comparisons still block. Integration record: `pressure_quality_control/agentic_optional_reference_autofix_v2/INTEGRATION.json`. No runtime result or final report was regenerated.

CPU progress at this observation: 31/199 complete, 7715 generated tokens; no scores. No TT server/evaluator is live. Aggregation review found native serialized IFEval target and Unicode JSONL collector mismatches; AutoFix is versioning both corrections. Official ownership adapter v2 is in follow-up review. Current release status was rewritten to remove stale active assertions; its original bytes are archived with a hash, not discarded.

### 2026-09-10T21:08:22.688069+00:00 — remaining-profile owner preparation rereview

Root read the full versioned independent review and helper delta. Both concrete native full-dataset report and retry-cache findings are fixed in the isolated v2 package, with 31 supplied controls and three independent controls passing. Actual original report bytes are preserved and selected errors remain visible; independent incremental retry task counts are [4, 2, 0], preserving first-completed origins. Review: `pressure_quality_control/remaining_profile_owner_review_v2/STAGE_REVIEW.md`. No runtime outcome or release pass is implied. Integration remains deferred until canonical CPU scoring and the fresh P150x2 generation/official verification/aggregation complete. Current source allowlists are refreshed in `pressure_quality_control/stage_commit_scope/REFRESH_20260910_2110.json`: 12 tt-metal, 13 vLLM and 72 TTI paths, with unchanged empty indexes and unrelated .agents paths excluded. Final evidence selection remains pending.

### 2026-09-10T21:15:53.488064+00:00 — Meta saved-output boundaries rereviewed

Root read the exact two production deltas, frozen preparation manifests and full follow-up clean-pass. Aggregation v4 requires native IFEval string target "0" exactly; corrected Meta owned collection v3 reads physical file lines. All commands, budgets, counts, source/capture pins and future owner remain unchanged. Author43 focused tests and independent21 native-writer boundary controls pass; incorrect targets, missing rows and null runtime evidence still reject. Review: `pressure_quality_control/release_aggregation_review/STAGE_REVIEW_V2.md`. Use the corrected v3 Meta launcher and v4 aggregation for future materialization, preserving earlier preparations. No active source changed and no actual scoring/collection/report generation occurred.

### 2026-09-10T22:40:21.035878+00:00 — superseding no-CPU continuation

Read `/home/hous/dev/gemma-4-26B-A4B-it/no_cpu_continuation.md` and current completion scope. User prohibits further CPU model evaluation or inference probes. The cancelled canonical run remains cancelled: supervisor3374126 and worker3374127 absent, worker rc143/signal15, cleanup complete with no survivors. Partial artifacts preserved; no score/pass inferred. Copied only small owner and cancellation metadata to `pressure_quality_control/no_cpu_continuation/`. The prior CPU-required plan is superseded. Required TT evaluation will proceed without a new CPU baseline; unavailable comparisons remain explicitly unverified and no quality failure is waived on that basis. Stage11 prerequisites and contexts remain unchanged; final v5.1 publication belongs to the wider delivery scope and is not claimed by Stage11.

Verified branch `hous/gemma-4-26b-a4b-it`, no active endpoint or TT device descriptors. Existing task Docker proxy still matches recorded PID/start/command identity; no new proxy created. Integrated reviewed sampling-audit patch1a83cb80 and owner registry14fb15ff after exact before/after checks, patch check and git diff check. No new CPU model or mock evaluation ran. Four P300C chips visible in bounded tt-smi health check; no reset performed. Restoring external autoport serving next.

### 2026-09-10T22:58:14.083783+00:00 — Actual TT continuation

Recovered P150x2 autoport server `no_cpu_recovery_v1` is serving context 262144 on this physical host. `smoke_no_cpu_recovery_v1` passed: health/OpenAI request, actual TTI 8/8 benchmark (1 completed, 0 failed, trace capture disabled), 29-token non-aligned text, tool/reasoning parsers and 12 qualitative outputs reviewed. Longer qualitative answers are partial at the unchanged 256-token smoke budget; no full-answer accuracy is inferred.

Fresh R1 native 40-row collection is running under `gpqa_owned_recovery_preflight_v4/plan.no_cpu_recovery_v1.json`, SHA `a832072d7b826cb30a08cce2eb257df2d97c62dbb9cd1078493c063844500e87`, output `/home/hous/dev/tti-release-gemma4/recovery_P150x2_gpqa_expert_tanh_v2`. Concurrency 32, max output 32768, temperature 1.0, top-k 20, top-p 0.95, seed 42 and streaming are unchanged. No score/pass yet. CPU model evaluation remains prohibited; cancelled artifacts remain preserved. Meta v4 collection preparation received scoped clean-pass (`STAGE_REVIEW_V3.md`); actual Meta collection follows terminal R1 cleanup.

### 2026-09-10T23:01:34.349084+00:00 — Task Docker transport recovery

Revalidation found previous proxy PID 2243608 terminal (SIGTERM at 22:42:00 UTC), with its socket removed, process absent and all 18038 recorded transport-child cleanups complete. The signal sender is not recorded. Its original terminal manifest and log are preserved externally under `/home/hous/dev/tti-release-gemma4/docker_proxy_history/terminal_20260910T224200Z`; a bounded summary is `P150x2/no_cpu_recovery_v1/terminal_previous_docker_proxy.json`. Started the unchanged `docker_client_proxy.py` as PID 3451840 (session 54491). Actual identity/private socket checks passed; all five SWE local image tags still match their previously validated immutable IDs (`swe_image_identity_check.json`). No task containers or extra inference clients were started. TT serving remains external on the same physical host; no Docker inference image is used.

### 2026-09-10T23:19:52.598384+00:00 — Real R1 client timeout

The native R1 client interrupted one actively generating request at its default 1800-second total timeout (HTTP200, stream incomplete, `TimeoutError`). The evaluator advanced to39/40; final request still generates. Live server counters increased and no preemptions were recorded. This is not accepted accuracy evidence. Preserve current samples when native collection becomes terminal; run independent Meta TT tasks while the timeout repair is prepared. `$autofix` source/artifact diagnosis is active in `gpqa_timeout_autofix_v1`; request budgets, sampling and full context remain unchanged.

### 2026-09-10T23:23:44.589808+00:00 — Meta TT collection started

R1 v2 ended with returncode1, one timed-out partial response and39 normal stops; native partial samples remain external and no accuracy pass is accepted. Owned process cleanup completed with no survivors/errors. Verified zero running/waiting server requests. Started reviewed Meta v4 with actual plan `/home/hous/dev/tti-release-gemma4/meta_owned_collection_preflight_v4/plan.materialized_v1.json`, SHA `3e35920d7f8d0f7f12ce7cccd7560be59128e25a5952b700a7ccdfb5898e85be`, command `python_env/bin/python -B <v4>/run_owned_meta.py --plan <actual plan> --plan-sha256 <above SHA> --execute --other-clients-quiescent`. This runs original109 IFEval/90 Meta GPQA TT requests, budgets1280/2048, greedy/seed42/concurrency32, tokenized native prompts, full context262144; output `/home/hous/dev/tti-release-gemma4/collection_P150x2_meta_expert_tanh_v1`. Final/base spec8d3eaa... embeds release/external8000/autoport and no CPU reference. Quiescence ack preserves user-cancelled CPU and invalid/cancelled SWE history; comparison explicitly unverified, quality/release flagsfalse.

### 2026-09-10T23:29:22.314375+00:00 — No-CPU report/remaining-profile preparation reviews

Aggregation v5 review found a real ordered-ID hash encoding mismatch (spaced JSON versus original compact fixture encoding). V6 changes only that comparison, matches both actual109/90 fixture hashes, and received scoped clean-pass (`release_aggregation_review/STAGE_REVIEW_V5.md`). It can expose actual TT scores while retaining mandatory comparison failures/unverified status; no reference or quality waiver is invented. Remaining-profile no-CPU launch v1 also received scoped clean-pass (`remaining_profile_no_cpu_launch_review_v1/STAGE_REVIEW.md`): one prelaunch reference-load call becomes task validation, preserving0.2/no overrides and all report gates. Neither patch is applied to live sources; integration follows current P150x2 terminal collection/verification/report boundary. Whole-stage clean-pass remains outstanding.

### 2026-09-10T23:39:03.888431+00:00 — Actual IFEval complete; Meta GPQA active

All109 IFEval responses completed, exact expected payload cohort and native sample counts matched, native returncode0, owned cleanup complete. Original four metric mean: 88.74073498159954 percent on this nightly subset. The native result and sample hashes are recorded in `P150x2/no_cpu_recovery_v1/meta_ifeval_completion.json`; comparison remains unverified and no required quality gate/whole-stage pass is inferred. Same owner has started90 Meta GPQA requests with unchanged2048 budget/greedy/seed42/C32. A separate diagnostic may compare only existing49 completed cancelled-reference outputs with matched TT rows; it must not run any CPU model, supply a109reference or waive a failure.


September 11 continuation: native Meta collection completed with exit 0 and clean task ownership: IFEval 109/109 responses (original four-metric mean 88.74073498159954%), Meta GPQA 90/90 responses and 57 exact matches (63.33333333333333%). Both are nightly-subset measurements; reference comparisons and release quality remain unverified. Actual completion hashes are in P150x2/no_cpu_recovery_v1/meta_collection_completion.json. The punctuation-checker diagnosis is isolated; original metrics remain preserved. No CPU model inference resumed.

R1 client-timeout recovery launched using gpqa_owned_timeout_recovery_v1/run_owned_gpqa.py --plan gpqa_owned_timeout_recovery_v1/plan.actual.json --plan-sha256 a0d47bd6675ecd79b517e3766ba4b67bedcb1db4b260429bf710704da011d8d4 --execute --other-clients-quiescent. Session 64815 binds all 124 unchanged runtime sources, the same actual autoport server, and completed Meta/failed-old-R1/cancelled-CPU/SWE ownership. The new native model_args timeout=7200 is the sole operational setting change; all 40 questions, 32768 output budget, 262144 context, sampling and C32 remain unchanged. Original failed R1 outputs are preserved. Preparation fixes before launch: the root observer treats absent /proc entries as terminated and reads JSONL on physical newline boundaries, preserving embedded Unicode line separators; no evaluated source or output changed.

Broader actual Meta GPQA output skim found9 invalid filtered answers among90. Root directly read docs0,3,4,9,17,26,58,71. Doc58 mechanically repeats the same chemical sentence more than20times; docs17/26/71 also revisit reasoning until the native2048 budget. These are open output-quality findings, not waived canonical behavior. No compatible reference exists. A fresh autofix diagnosis and serialized TT-only control are required after the current R1 collection; original outputs and scores stay unchanged. Details: P150x2/no_cpu_recovery_v1/meta_gpqa_output_review.json.

September11 source-mode clarification: the global sample_on_device_mode=all launch setting does not prove per-request device sampling. The installed platform/runner carries autoport supports_device_seeded_sampling=False; seed42 remains set at T0, so the native seeded Meta path is predicted to select compatibility host sampling of TT logits and eager TT model decode. This is a source-backed mode inference; no doc58 per-request sampler audit exists. CPU model inference remains prohibited and has not resumed. Repetition diagnosis: pressure_quality_control/meta_gpqa_repetition_autodebug_v1/AUTODEBUG.md. Root identified a preflight-only payloadhash guard issue in the proposed single-request client; no request was sent and a versioned correction is underway.

Scoped review pressure_quality_control/meta_scoring_artifact_review_v1/STAGE_REVIEW.md (d830dd0d40d5d0b1c2fcdb3c4781c66bf1c6ca75ae38b7d8e17af1c0f9d72788) passes the isolated symbol correction and actualMeta aggregationv7 bindings. V2 (2cfcec19cc162d3f0a58a8ecb637ee16256df6e7c70fd04cc3e3e820e45b24ac) passes aggregationv8 presentation clarification: comparison_unavailable retains required FAIL, does not claim a measured numerical miss or require a CPU producer. Neither review passes wholeStage11 or resolves GPQA quality findings.

R1 timeout7200 rerun completed on September11 at00:57UTC with native exit0,40/40complete streams,0transporterrors and40samples without partial/error markers. Score28/40=70% on nightlyDiamond subset. Cleanupcomplete, nosurvivors/signalingerrors. Native resultSHA8d1ee9595c88450e0ca76ecd4363bbb576c63b732e8b483640bf09550d915fc2; sampleSHA379ad8ff627fb4fef39bffe7dd8d1ebf2fa1df32c37d0f6964ca54c02d0f41df. Prior timing-out chemistryrequest completedafter2603.118seconds withstop/DONE, provingit crossedtheold1800limit; bothlongrequests didso. Actual40-rowdoc/gold/outputreview is pending; processsuccessisnotqualitypass. CompactresultinP150x2/gpqa_timeout7200_v1.

Rootreadandvalidated corrected diagnosticclient77f97ec04ed39c780c453a5084ef72ec03be49b4c630044a01b6b9deaa135d0e; staticbindingsandactualserveridentitypassed. AfterR1reapandobservedzeroactive/waiting, launched oneB1MetaGPQAdoc58 request: python_env/bin/python -I /home/hous/dev/tti-release-gemma4/meta_gpqa_repetition_autodebug_v2/run_owned_b1.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_repetition_autodebug_v2/plan.actual.json --plan-sha256 e7311b9d0c5e41a92d73b4c716865a06615190a7be5d58b326c8fb48ee753b07 --execute --other-clients-quiescent. Session17848; outputexternalmeta_gpqa_doc58_b1_control_v2. Same326promptIDs,2048budget,T0seed42,stop[],nonstreaming,262144context; onlyisolatedadmissionchanges. Originalwirebytesnotrecorded; exactJSONpayload andnewrequestbodyhashes are distinguished. Noscoring/CPUmodelinference.

The single-request doc58 control completed with exit 0 and clean cleanup in 145.363 seconds. It preserved 326 prompt tokens and the full 2048 completion budget, ending with length. Its response differs at character 171; the exact repeated sentence occurs zero times, but the output still revisits inconsistent chemical reasoning and never reaches a final answer. Root read the full response. This is batch/state/history sensitivity, not a quality pass or a replacement 90-row score. Server drained to zero active/waiting with no preemptions. Compact evidence: P150x2/meta_gpqa_doc58_b1_control_v2/. Next is one predeclared two-request control using the exact original doc58 and doc0 payloads. No CPU model inference or generation-policy change.

## Continuation 2026-09-11T01:18:05.152169+00:00

Root revalidated all 124 runtime source bindings and live external autoport server ownership on qb2-120-p11t02. Admission showed zero running/waiting requests and preemptions. CPU canonical_run_v1 remains cancelled; no CPU model evaluation was run. The fixed two-request TT diagnostic launched with `python_env/bin/python -I /home/hous/dev/tti-release-gemma4/meta_gpqa_pair_autofix_v1/run_owned_pair.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_pair_autofix_v1/plan.actual.json --plan-sha256 2ccc54899254e0adcb4b23c33784850fe29c2cf5ed9333f6980405fe7e8d9576 --execute --other-clients-quiescent`. Tool session 54342 owns the control. Both original payloads preserve T0, seed 42, 2048 output allowance and context 262144; running2 was observed. No quality result is inferred while live.

Root read completed R1 OUTPUT_REVIEW.md and TT109 posthoc REPORT.md. R1 validates all 40 rows and retains 28/40, with semantic cycling findings. The single real-output scorer pass gives 89.2525786554709%, exactly the two symbol-check corrections, with original result retained. Its scoped independent review and explicit aggregation binding are underway; unavailable comparisons remain unverified.

Paired TT control completed at 2026-09-11T01:23:40.363549+00:00: rc0, complete cleanup, zero survivors/errors, 2→1→0 observed with zero preemptions. Doc0 exactly matches original464-token stop; doc58 uses2048 tokens,337.933s and differs at character1525 from original and171 from isolated B1. Root read both full responses: a different reaction sentence repeats mechanically, so the predefined original span count zero is not a quality repair. Compact evidence is `P150x2/meta_gpqa_doc58_doc0_pair_control_v1/`; raw outputs remain external. No model source changed. Next is fixed-token-prefix TT-only localization.

Root read independent scorer review V4: scoped clean-pass, SHA25668073968f9518b5f729fa07590df0fe890e37e09bc2f922c8766d22edb9370ea. All109/177 rows and exactly two deltas validated; no whole-stage pass.

At 2026-09-11T01:27:20.520384+00:00, root verified idle admission and sent SIGTERM only to owned launcher3443218. Session14594 reaped143; launcher3443218/runner3443235/API3443236/engine3443509 are absent. Terminal server manifest SHA256e2cd66decee74194100de94df9af1d7ee4776b7134a4483ead739945b381ff51 records complete cleanup with no survivors/errors. Immutable live snapshot0b2cc00afba84a431bc8dd0e4bf23e679d29c1bf75cbce052e3a264e913027d3 remains preserved. Root initially suspected an aggregation lifecycle defect from generic validation code. Actual-plan inspection refuted it: R1 binds r1_timeout_server_snapshot.json and Meta binds server_live_snapshot.json, both immutable. The completed R1/Meta evidence remains valid after shutdown; no lifecycle code fix is warranted. The diagnostic pair alone binds the mutable server manifest, and its original launch snapshot plus terminal shutdown record are retained separately.

Serialized `timeout 60 tt-smi -ls --local` passed, showing all four P300C Blackhole chips. `TT_METAL_DEVICE_PROFILER=0 timeout 90 python_env/bin/python -B` opened/closed `ttnn.MeshShape(2,2)` with trace_region_size0 and exited0. Small logs copied to `P150x2/no_cpu_recovery_v1/`. No reset or CPU model inference occurred.

At 2026-09-11T01:37:31.715530+00:00, root launched the reviewed TT-only fixed-prefix probe with `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/plan.json --plan-sha256 9656cd68d7594d6ffdb6e68cda8c20b691430f08cf77ea2a3060f34c690c4c7f`. Tool session40222, worker3508806; ownership disables force-kill pending root triage if stalled. Probeplan7cd60707fb21f52537f0b072766441f77a479ff904471135598f3adb108de8fc preserves context262144,32slots,4128blocks,currentweights/precision and original326-token prompt. Both arms use the declared common63-token re-encoded prefix, freshmatchingHMAcaches and eagerTT logits. Width32 has oneactive row with inactivepadding and is an arithmetic-shape control, not reconstruction of originalpairhistory. CPUmodel/reference inference remainsprohibited. Onlyhosttokenization/TTlogitstatistics are used.

Fixed-prefix TT probe completed at 2026-09-11T01:40:20.594146+00:00, owned exit0/clean cleanup/deviceclose, total108.512s. Prefill logits are byte-identical. Decode logits first differ at inputposition326/outputindex1; first top1difference is outputindex63/inputposition388. Width1 selects621 and width32selects1170, reproducing the historical candidate rank ordering under this declared prefix. At checkpoint388, candidate logits [1170,621] are [18.5,19.75] forwidth1 and [18.375,18.125] forwidth32. This demonstrates a TT decode shape-dependent rank change before the final compatibility gather/sampler, not its rootcause or a qualityrepair. Next is first-decode TT layer-boundary localization. Raw TTlogit tensors stay external; small result/owner copied to P150x2/fixed_prefix_result_v1. Float32PCC reduction slightly exceeds1 for identicalprefill; byteequality is authoritative and corrected diagnostic statistics may be computed from saved TTvectors without modelinference.

First-decode boundary probe completed with owned exit 0, clean cleanup and device close (78.756s). Root ran `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/first_decode_plan.json --plan-sha256 8ec32b2c2399b83f446c0f29e5fe3c0a082c7562fdad8606ca7bdf37809bd254` in tool session57673. Both instrumented arms exactly reproduced the earlier full prefill and first-decode logit hashes. Embeddings are identical on both chips; layer0 is the first differing boundary (max absolute difference0.375, RMSE0.016000913). This localizes the earliest observed difference to the first decoder block, without yet identifying its cause or proving which numerical path is more accurate. Small evidence: P150x2/first_decode_boundary_result_v1; TT tensors remain external. Next is a layer0 branch split on the same first decode step.

Root read aggregation V5 review, scoped clean-pass SHA256cf4a42210f4400bfbcd12e20032a3940b94c2af60ce2345e32a3be5768a82283. V9 preserves original collection guards and accepts the explicitly versioned corrected IFEval result. It remains unmaterialized pending SWE/final bindings and does not resolve mandatory comparison or model-quality findings.

Layer0 branch probe completed with owned exit 0, clean cleanup and device close (78.897s). Command: `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/layer0_branch_plan.json --plan-sha256 a017f7913b9dada622ab0e6c16106ae9391347bfdc179079102a5ec86587018b`, tool session5429. Both complete prefill/decode logit hashes again match the uninstrumented baseline. Input normalization and attention input are exact; attention output first differs (RMSE0.026506084/max absolute0.25). All dense/router/expert differences follow this earlier drift. The 44-versus-11 expert setting is therefore not the initial producer in this experiment. Small evidence: P150x2/layer0_branch_result_v1. Next is attention QKV/head/cache/SDPA/output-projection localization; no model or precision change has been made.

Layer0 attention probe completed with owned exit 0, clean cleanup and device close (78.804s). Command: `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/layer0_attention_plan.json --plan-sha256 7f07fdc254f6dc7ac483f3013d35e188cd90334ac992e1e7193dbc64156b65f1`, tool session50322. Root inspected every boundary: page table, six owned cache pages before/after update, QKV projection, heads/norm/RoPE and SDPA query are identical across widths. The first difference is SDPA output (per-chip RMSE0.008824401/0.009336869). Both full logits still reproduce the original baseline. The common SDPA config is grid8x4, qchunk32, kchunk64, exact exponent mode, maxcores16; batch shape changes resolved core assignment. Next is a cap1 side call on the same live SDPA operands, returning the original output to preserve baseline checks. This has not identified which reduction path is more accurate or repaired the late repeated generation. Small evidence: P150x2/layer0_attention_result_v1; raw tensors remain external.


### September 11 resumed: completed SDPA core-cap control

Verified `P150x2/sdpa_core_cap_result_v1/result.json` (SHA256 02521c9323bc998d3aeb932673bf1841156a734ed194c94c960352054a2fcda3) and owner rc0/complete cleanup. Both baseline logit hash guards pass. Width1 cap1 equals width32 default exactly on both chips; width32 cap1 is unchanged. This identifies reduction geometry as sufficient for the local difference, not an accuracy winner or a resolution of GPQA repetition. No production policy changed. Prior worker3519209, server3443218 and Docker proxy3451840 are absent. Next is the declared TT-only component numerical control. CPU evaluation remains prohibited and cancelled canonical artifacts remain preserved.


### TT-only SDPA numerical comparison

Owned probe rc0, cleanclose/cleanup, original model logit reproduction guards passed. Independent TT QK/FP32 stable softmax/PV computed from exact real Q/K/V is byte-identical across width1/32. Default fused SDPA RMSE versus this declared TT comparator is 0.03130/0.01366 for width1 and 0.03066/0.01283 for width32. Fused FP32 destination only has a severe width1 anomaly (RMSE81.23/82.29), while width32 is 0.01648/0.01026. This control is not production policy or a canonical oracle; no quality pass is inferred. A cap1+FP32 factorial and source diagnosis follow before selecting any repair. Small result/owner: `P150x2/sdpa_numerical_result_v1/`; raw tensors remain in work root only.


### Verified FP32 SDPA correction-slot defect and build

The cap1+FP32 control exactly matched width32, isolating the catastrophic FP32 width1 failure to tree reduction. Source diagnosis (`pressure_quality_control/meta_gpqa_sdpa_fp32_diagnosis_v1/AUTODEBUG.md`) found five simultaneous full-stride DST slots with only four available under FP32 half-sync; the in-tree LLK test requires full sync for this correction. An otherwise identical full-sync TT sidecall removed all convex-hull violations and reduced RMSE to 0.01553/0.01031; width32 stayed unchanged. The narrow factory repair enables full sync only when FP32 accumulation and multiple cores per head coincide. Production autoport default uses FP32 accumulation false.

`/usr/bin/ninja -C build -j 2 ttnncpp` and `cmake --install build --component ttnn-runtime` passed. `.github/scripts/copilot-build.sh` failed its Docker-availability preflight; no dependencies were installed. `BUILD_RESULT.json` records before/after source and build/installed binary hashes. Actual rebuilt-library verification is next; GPQA repetition and release quality remain unresolved.


Rebuilt-library verification completed with rc0, clean device/process cleanup, all post-repair comparison guards true and original full-model logits unchanged. The process mapped installed library SHA40ea0f275651e6288f428759af851c75d73362ddf1414f2abb9bbd94942b26d9. Result `P150x2/sdpa_numerical_verification_v4/result.json` SHA63774b836c964a4da1c6e469cfa8563f940827b0cba04cf7a385bfebb9caf8b6. Active serving ledger updated to SHA946234be7af48a5415f4d38ce7613bc4b9ec73dee1133f2c3b4713206e80b34f with prior bytes archived and actual verification linked. `pressure_quality_control/meta_gpqa_sdpa_fp32_diagnosis_v1/AUTOFIX.md` records the verified narrow repair. No production model precision change or GPQA quality pass is implied.


Original doc58 quality control launched with the existing owned wrapper: `python_env/bin/python -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_original_quality_control_v1/owner_plan.json --plan-sha256 7c0a0807d9906fe6b47bcd3a3b3df5d5747cfa2d329f59defe1eaff35fb72fcf`. This is full2048-budget autoregressive generation, as recorded by command/driver result; the unchanged generic wrapper has historical fixed-prefix scope wording. All30layers, P150x2/context262144/4128blocks/nativeEOS1,50,106/native detokenizer remain. Default arm must reproduce saved B1 response text SHA417f7a965fbf16072d04809181a28713ae5875a4e54477235165200f13b547b7,2048tokens,length before candidate runs. Candidate changes only FP32 destination accumulation in paged decode SDPA; no productionpolicychange. No CPU model inference/reference math.


## September 11 recovered serving after SDPA verification

Original full doc58 control completed cleanly: baseline text reproduced exactly; repaired FP32 decode candidate still exhausted the unchanged 2048-token budget without a final answer. The precision hypothesis is refuted; production precision remains unchanged. The independently verified FP32 kernel synchronization fix is retained. Actual result: P150x2/original_quality_control_v1/result.json.

Started external autoport server attempt no_cpu_sdpa_recovery_v2 on physical qb2-120-p11t02, P150x2 proxy, full context262144, sampler audit enabled, profiler disabled; health HTTP200. Server launcher session64621, runner3815813; source ledger binds rebuilt installed runtime40ea0f275651e6288f428759af851c75d73362ddf1414f2abb9bbd94942b26d9. Real smoke session86970 is running. Smoke preflight now resolves actual CLI/configuration helpers directly, without stubbing benchmark execution; actual TTI8/8 benchmark follows. No CPU model evaluation was run.

Recovered-server smoke completed with rc0 and complete client cleanup. TTI raw8/8 JSON: completed1/failed0; written runtime_model_spec and runtime_config both no-Docker/no-local, service_port8000, disable_trace_capture=true; autoport implementation/context262144 confirmed. Tool/reasoning parsers and29-token nonaligned request pass. All12 shared-suite continuations exactly match the previous no_cpu_recovery_v1 TT control; root read outputs and retains fixed-suite truncation as bounded smoke scope. Acceptance: P150x2/smoke_no_cpu_sdpa_recovery_v2/SMOKE_REVIEW.json. GPQA quality remains open.

Task Docker transport: archived completed former PID3451840, confirmed absent/cleanupcomplete; new owned PID3835798/session89358, private0600socket/0700directory, sudo docker dial-stdio per connection. All5 reviewed SWEimage IDs match and containers were empty. No Docker inference image used. Replacement observation: /home/hous/dev/tti-release-gemma4/docker_proxy_replacement_20260911.json.

Fresh SWE launch v4 preparation completed (source-only materialize.py prepare rc0),143current source/runtimepins; prepared plan SHAae1bad2ece8891f8ba01c5b7c50d64c5293bc4dc27428b901ecdf46b6625752c. Native five-task commands/settings unchanged. Old recipe config.py pin updated only after exact reconstruction proved its sole delta is benchmark LLMRunConfig.target_source; SWE ServerConnection/DriverContext and builder unaffected. Frozen prior recipe preserved. Root quiescence observed and bound. Fresh xhigh scoped launch review running; collection not yet launched.

Recovered smoke runner formatting check: python_env/bin/python -m black --check models/autoports/google_gemma_4_26b_a4b_it/doc/tti_release/run_existing_server_smoke.py returned0 (unchanged); git diff --check for smoke runner and SDPA factory returned0. No build needed for smoke-runner Python change; prior C++ build/install and TTverification remain recorded above.

Scoped launch review effective-configuration clarification: native SWE run_config retains max_input_tokens98304 and random_delay_multiplier0.3, but the unchanged mini-agent command/config builder does not emit either as a new enforced request limit/delay. Actual mini context admission uses rendered prompt length +32768 <=262144. Do not present stored config fields as measured runtime enforcement. Native task recipe remains unchanged. Reviewer read-only observed both API3815826 and EngineCore3816345 mapping repaired installed runtime SHA40ea0f275651e6288f428759af851c75d73362ddf1414f2abb9bbd94942b26d9.

Fresh independent xhigh launch review clean-pass for native_swe_fresh_five_generation_only (whole_stage_pass=false): swe_shell_activation_launch_refresh_v4/STAGE_REVIEW.json SHAe26e8a500180f9631b4e2a65f36745de6ab1b46e2204dc9f6a2e629e29fe01b4. Root read complete report. Materialize finalize rc0; final plan SHA4d2e6e6c168464086faedbfa128eb3b52400085deab910cd15b891c98160483d. Actual root command: python_env/bin/python -B /home/hous/dev/tti-release-gemma4/swe_shell_activation_autofix_v1/cohort_launch/launch_owned.py --plan /home/hous/dev/tti-release-gemma4/swe_shell_activation_launch_refresh_v4/plan.final.json --plan-sha256 4d2e6e6c168464086faedbfa128eb3b52400085deab910cd15b891c98160483d; session35581, output /home/hous/dev/tti-release-gemma4/recovery_P150x2_swe_shell_activation_v1. Generation/cleanup/verifier results pending. No CPUmodel work.

Official verifier v2 template delta independently reviewed clean-pass, preparation only, whole_stage_pass=false. Review JSON SHA9290337c0fa3be5d237220c6c237d6b8bef4bf33ae871b57094e831d194db8d0. Verifier/materializer bytes unchanged;14staticpins and actual launchedgeneration transport match. Actual completed generation/native+outercleanup mandatory before materialization. Root read report; no verifier run yet.

Ongoing SWE observability source audit: DefaultAgent.output_path defaultsNone, so per-step saves do not write files; process_instance writes trajectories/predictions and exit-status YAML on task termination. Rich progress refreshes are suppressed with both output streams redirected to native_collection.log; the agent debug logger is separate from the minisweagent file handler. At17:10:54UTC fiveownedcontainers/no terminaltasks/clientexception/servererror were observed. HTTPthroughput proves serveractivity only, not useful modeltaskprogress. Do not infer tasksuccess from log silence or HTTP200.

Prepared next GPQA AutoFix control at pressure_quality_control/meta_gpqa_exact_history_prefill_control_v1: exact saved emittedIDs, checkpoint outputindices1/68/151, baseline152token guards and originalfirst2logithashes, instrumented-vs-uninstrumented checks, equalHMAallocation/page-tablehistory, noCPUmodel arithmetic. Root read design and runtime body. PendingactualTTexecution only aftercurrentSWE/servercompletecleanup; no request budget/policy change.

Root bounded passive response-only TCP capture (40seconds,21packets,0kernel drops, timeout exit124 intentional) observed existing ownedservertraffic without newrequests or changing source. Rawpcap retained onlyexternalworkroot. FivecompleteHTTP200responses:2tokenizer metadata,3valid nativebash toolcalls. Response chatcmpl-b685fcadaa7cbafe exposes literal <|channel>thought\n<channel|> in content besidevalidSymPyreproductionaction, reasoningempty. Twoother distinctresponsesrepeat findsrc command; no actionresults captured, so sustainedloop/tasksuccessunverified. Smallresponse-onlyJSON/sourcebindings copied pressure_quality_control/swe_passive_responses_20260911; no HTTPheaders/requestsecrets/largepcap copied. FreshAutoDebug swe_channel_leak_diagnosis investigatesmarkerleak and possiblehistoryfeedback; activecollectionsourceunchanged.

Passive actual-token-history inspection confirms Matplotlib aligned-label pickle task repeatedlycalls find src -name "*.py" | head -n20. Captured5645tokenprompt:47actions/47toolresponses,37identicalcalls with36consecutive tail. Next5698tokenprompt:48/48,38identicalcalls with37consecutive tail; priorIDs exactprefix and53tokensappendoneaction+explicitreturncode0/emptyoutput. Earlierhistoryincludesweakrefpickleerror andlibdirectorylisting. Sustainedloopproven; tokenizer/clientfeedbackomissionrefuted. EngineCoreconsumption/canonicalattributionunverified, no qualitypass orwaiver. Fullhistoriesremainexternal; compactanalysiscopied.

ParserAutoDebug verifies single-block reasoning extraction then untouchedpretoolprefix; exactobservedmarkercontent retained inminihistory but checkpointstrip_thinking macroremovesitnextturn. Retainedmacroreplay rc0 SHA749a7069dd879736057b6d7a2807913c659e079a1f03898e731ee9d865302bfa. Exactfeedbackcorruptionhypothesisrefuted; APIleak stillopen. Parsercandidatepreparedseparatelywhileactivesourcesfrozen; no claimofrawgenerationpreimage ormodelqualityrepair.

Second bounded passive capture (root session8996) reaped rc124 at its intended55-second timeout:33packets,0kernel drops. Six completeHTTP200 responses comprise three valid identical Matplotlib bash calls and three growing tokenizer histories. At capture time17:37:13UTC, history8083 tokens contains93 action/feedback pairs and82 consecutive identical find commands;45 new pairs since first capture are all rc0/empty. Exact token and rendered-byte prefixes hold. Other tasks are absent from this sample. The API-leak candidate remains unapplied; no new model request, source mutation or quality pass follows from this observation. Compact findings are in pressure_quality_control/swe_passive_responses_20260911_v2; full histories andPCAP remain external.

User priority change: focused repetition repair. Root paused exact mini producer3866304 only while preserving all5worktrees as tracked diff plus untracked archives against pinned task images; no weights/caches/Docker layers copied. Sent SIGTERM to existing cohort launcher3866258; native monitor cooperated, removed exact5owned containers, outerrc143 and complete cleanup, no survivors/signal errors. No trajectories had completed; partial native logs and passive histories remain external. Collection is user-interrupted, not quality-pass. Owned serving launcher3815783 then stopped cleanly, session64621 reaped143. Exact-history plan178pins differed only in newly superseding instruction file; old template preserved and instruction pin refreshed. Diagnostic root session87887 launched from owner_plan c75a9fffa8579d4bcc5207cc059a4f8ad95365482a7d250eb7ecb78572f213a9 with no source/runtime/recipe change. No broad evaluation will resume before focused evidence warrants it.

Exact-history TT diagnostic complete: root session87887 reaped0, owner elapsed123.18s/device compute report120.70s, complete cleanup and clean mesh close. All152 baseline predictions match saved actual baseline, all incremental/fresh-prefill instrumentation guards pass. Embeddings and TP replica outputs match exactly. First captured difference is layer0 for output indices1/68/151 (~1.17/1.45/1.27% relative L2). Only index68 changes argmax, and fresh prefill has an exact23.75 tie between two candidates; this is not better-quality evidence. Layer16–18 amplification warrants operation-level matched-input/cache controls. No implementation repair or quality pass is claimed from phase disagreement. Result7bd603e215685a940e1612001b75fd21127b0b9f50b268084176a390872eb3d9 copied to P150x2/exact_history_prefill_control_v1.

Preserved-worktree follow-up verified all5captured GitHEADs equal their exact existing task-image HEADs using read-only Git metadata, no eval/project test. Thus there are no missing in-sandbox commits; unchanged base is retained in existing pinned images, untracked task work and tracked diffs are preserved externally. These image HEADs differ from upstream dataset base commits because the supplied task images include setup commits; no task evidence was relabeled. Read-only metadata containers auto-removed. Docker proxy session89358 reaped0, socket removed,7798transport ownership cleanup records complete with no survivors/signal errors.

### September 11 focused operation/cache TT diagnostic

Root-owned existing launcher completed with rc0, session95763 reaped, worker186546 cleaned, TT mesh cleanly closed. Plan b19e56d303e799e51a20399f7ede734bf6a190bdb7fa386c33ced56f52b96d64; owner plan f01c5e8bb773b48e1dd2838cac063240671bd038ce228eddd4f998623eb2c9e1. Result P150x2/exact_history_operations_v1/result.json SHA5d8798ba288c9078bb49f8596fd19aae0adb3ccf961e449fc821227bec3f5f94. All saved-logit and 33-boundary guards passed at indices1/68; all K/V producer/cache, old/future row and shared FFN-input invariants passed on both chips. Earliest operation difference is input RMS norm from identical layer input: relative L2 0.001972605 / 0.002922915. This localizes phase sensitivity, not a quality fix. Next actual TT control isolates normalization lowering with matched operands. No broad eval or CPU model inference. Raw tensor artifacts remain external.

### September 11 matched-row RMS baseline and isolated candidate

RMS-only TT baseline completed in70.036s, owner rc0 and complete cleanup; session77627 reaped, worker221539 absent. Result P150x2/rms_scalar_control_v1/result.json SHA4b08a190d74c1f7d07a42647b28d7ad06f03d3fdf1a68fd0b00af670610ad71f. Both actual weighted decode and prefill outputs reproduced exactly from saved rows1/68; original-neighbor fallback was unnecessary. Both half-width permutations were exact at BF16 output, and interleaved legacy/default outputs matched. Independent TT arithmetic favored neither path consistently. No repetition cause or improvement established.

A separately identified scalar ABI mismatch passes duplicated BF16 bits to consumers that interpret FP32 scalar values. The isolated factory-only experimental patch preserves BF16 tile rounding, encodes global FP32 scale correctly, and encodes the second-stage identity as exact1.0. It is applied solely for before/after testing. Source SHA151324e00f302a6b096bf05a8109f4f5d37c715a46efaaa70d5694954f1c6cfd. `/usr/bin/ninja -C build -j2 ttnncpp` and `cmake --install build --component ttnn-runtime` both returned0. Candidate installed factory library SHA1f616676d41b7b32aad20a2aca5cb85667bb35ffe9ba8d9ff072eaeb9f20c422; Python extension unchanged52deb65dd2ded7e68753dc8fbec68ae992a72dd85596b7769beda7968fbe0e98. Evidence/build logs external meta_gpqa_rms_scalar_wire_fix_v1. Hardware verification pending; original runtime ledger remains historical until experiment disposition. No broad evaluation, context change or CPU model inference.

RMS scalar candidate verification completed in68.002s, rc0/clean closure, session43320 reaped and worker242045 absent. Result P150x2/rms_scalar_verification_v1/result.json SHAd5fae1e5de0bd7d2b2b88707259b767adefe41f79a11f16c7821bf05b71736d5. Every R22 weighted, unweighted and half-swap output at both rows/chips was exactly unchanged; unaffected controls also matched. The candidate is refuted as a cause at the measured boundary and was reverted, rebuilt and installed (both commands rc0). No same-case generation or broad eval was launched for this unchanged candidate. DISPOSITION.json pins restored source/runtime. Next: actual mean-square and inverse-RMS kernel readbacks, preserving original output hashes.

### Actual RMS kernel statistics and capture recovery

V1 failed before device open because this checkout requires debug RISC selector `TR0`, not `TRISC0`. V2 corrected only that environment/validator spelling and used fresh paths. Four actual TT unweighted norm calls reproduced every original BF16 output exactly; temporary kernel reads occurred only at existing synchronized consumption points. V2 returned1 because in-process log parsing raced DPRINT shutdown: only4records had flushed at parse time. After owned process271786 exited, all16 records were present with correct per-chip order and equal values on bothchips. Originalerrorresult/rc1 preserved; FINAL_CAPTURE.json records post-terminal verification SHAea83ff284492f1446948f089e5ab835b3143cbfa0ba9d599e20b1d1922997419. Ownercleanupcomplete, session69437 reaped. No hardware recovery/reset was needed.

At output1, actual sharded/interleaved mean-square values are2.5026140213/2.5061314106 versus independentTT2.5063283443. At68:1.6840782166/1.6865205765 versus independentTT1.6865346432. This localizes the error before epsilon/rsqrt; independentTT is not a canonical model oracle. The two-kernel DPRINT patch has been removed and build target returned0; originalkernel hashes restored. Next narrow candidate substitutes only residual RMS lowering with the existing interleaved TT path, keeping precision/weights/context and restoring original R22layout; it must pass saved-row and originalfull-budget same-case checks. No qualitypass or broad eval.

### Residual RMS boundary candidate: same-case repair failed

Single candidate actual TT run completed205.345s total/134.053s generation, ownerrc0/completecleanup; session23391 reaped and worker297951 absent. Result P150x2/rms_boundary_control_v1/result.json SHAee15701edd0870b319f0a0b8139a001f4756c4ba33522d19c82db9326a942fd7. Four saved-row weighted/unweighted outputs exactly matched prior interleaved values, preserved inputs and R22layout; original prefill full-logit hash unchanged. Only residual norm lowering changed, with existing HiFi4/FP32 config, weights/epsilon, native326prompt/T0seed42/EOS/2048budget/context/HMA preserved. Candidate still generated2048tokens, finish length, no final answer; complete text reviewed against baseline. The model identified cyclohexane earlier but continued cycling incompatible explanations. No demonstrated quality repair. The candidate was reverted and optimized_decoder restored exactly to29dbdc03e9d25043286809c66c4aa9cc3d610f3ad5cd59b27ff49f4d214a8687. ACTUAL_QUALITY_REVIEW.json and the reverted patch retain evidence. Fresh AutoDebug will choose the next distinct causal test; broad evals remain stopped.

## Focused next hypothesis after RMS rejection — 2026-09-11T20:02:19.976298+00:00

Fresh AutoDebug report: `P150x2/after_norm_diagnosis_v1/AUTODEBUG.md`, SHA256 `3c76bc48abee8e44a724ef4e4246cbd255d4fdfbd7a53bed5505848d6f6e652f`. Indexed MoE ordering is supported by source and earlier controls, so no new global toggle is justified. Next actual TT control reuses saved layer-0 Q/K/V at output68, requires exact native reproduction, then tests masked-future cache invariance and independent explicit-mask TT attention. This is diagnostic preparation, not a defect or quality-pass claim. All RMS experimental changes are reverted; hardware remains quiescent.

Initial masked-cache run: owned launcher `meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_masked_cache_control_v1/owner_plan.json --plan-sha256 285eec866f30a1ef479e554e7d14a57953d9819d7f3fa33a2e92032a29964bff`; session57128 reaped1, worker347142 absent, clean mesh/owner closure. Failed before any SDPA control at unbound memory-config string equality, not a model/mask failure. Original result preserved in `P150x2/masked_cache_control_v1/`. Earlier materialization rejected a superseded template hash before any hardware; current template frozen at afd87dda0bfe5119517c5cf0c282acd4ba31af3b7bb487b3a9dc0ec352c93884. AutoFix hypothesis agent is correcting fixture metadata comparison while retaining exact saved-output admission.

Masked-cache v2 actual TT PASS (diagnostic only): plan d90b0f3315353efd33ecd7bfcc9f4fd94d2866ec7a8fca58c9414100468bda7a, owner plan6db3f92b52fcff5bf285f922ef86af62061d53a335a3fd84d4b38f17ae6af004, result039f3f87b3bee8ed28c4f939ac480e9f7396042bdbb051a367f865a6f811173f. Same owned launcher with v2 owner_plan.json; session68578 reaped0, worker354377 absent, cleanup complete. Actual6.743s: restored query and all7cachepages exactly on bothchips; native SDPA output matches captured bytes; futureV and futureK+V changes invariant; validV response positive; uniform-score zeroK maskedV sentinel yields exactzero. Explicit-mask prefill also passes invariants, but differs numerically from native actual-input SDPA (relativeL2 .017095/.007012); neither output is canonical quality truth. Fixed v1 harness by checking normalized tensor memory metadata after materialization while retaining exactconfig/output guards; no production changes. Small artifacts in P150x2/masked_cache_control_v2; raw TT outputs remain external. Next focused extension covers1024 window boundary and full-attention HMA logical view, not broad evaluation.

Window-cache capture actualTT complete: owned launcher with meta_gpqa_window_cache_control_v1/owner_plan.json SHAa03c03ae00e31d25be35c2e32adabec10939cdb74fee062368cae1a038abf8f2; plan ea58005987dd41f28f1971b688a153f6254f5319eabe439d90890beecee38494; result11f0c280c1c9703d7872e8a1e605008d51d35687e39b65d286d636951dc69127. Session83688 reaped0, worker372796 absent, clean mesh/owner closure. Model initialization69.200s; plain700outputs46.080s, captured700outputs46.173s. Both match all700original IDs and152historical logit hashes; every paired700full-logit hash matches. Five captures are taken inside original SDPA before later HMA writes: L5position393 plus L0/L5positions1023/1024. Actualfull logical[1,128,512] reuses physical[4128,4,64,256] backing. Small metadata under P150x2/window_cache_control_v1; raw physical-page tensors and79MBdebuglog remain external (only attributed progress/closure excerpt copied). No quality or model repair claimed.

Window-cache replay actualTT PASS (diagnostic only): ownerplan963eb82e4aa77964accde6a5a415c98c10ecb16a566b03f3e781792e9e896850, plan2d3db90983ed858c9354ce27d49a4364076011368cc9676551e4f51eb1e54348, result33512b73f3418865ac3721abd8de16d51d0b7166bec3f1d76b9fa36aae2e3ab0. Sameownedlauncherwith meta_gpqa_window_cache_replay_v1/owner_plan.json. Session78944reaped0,worker390072absent,cleanmesh/ownerclosure. Actual21.417s: all5savednativeSDPAcalls reproduce exactlybothchips; all29arms passphysicalpageidentity, actualmasked-rowinvariance andpositive/uniform-scorecontrols. Syntheticrow0V test includesrow0forSWA1023/full1024 andexcludesitforSWA1024; futuremaskedVremainszero underuniformscores. FullattentionHMAviewandmaskboundaries testedcorrectly. NativevsindependentTTprefillrelativeL2 spans .00904–.02531; notcanonicaloracleorqualityevidence. No productionrepair justifiedbythishypothesis. Fresh xhighAutoDebug launchedagainstnewactualevidence; broad evalremainsstopped. SmallJSON/plans/log/sourcecopiedP150x2/window_cache_replay_v1; rawtensorsremainexternal.

Fresh AutoDebug after cache controls: P150x2/after_cache_diagnosis_v1/AUTODEBUG.md SHA7a6757f6eb46394504ef80d5f2aa7db4ca502ae62078435c572aa279eda55c60. No new defect established. Next bounded test compares selected expert gate/up, activation and down rows in actual B1 indexed/scanning paths with identical TT operands and quantized weights, requiring native router/local-output identity first. Existing B32wholelayercontrol and unequalinputprefill/decode comparisons do not resolve this B1contract. BF16finalsumorderdifference alone is not a bug. Hardware remains quiescent during preparation; noCPUmodelwork or broad evaluation.

B1 MoE actualTT row control complete: ownerplan9003af8029d07e4a055d6fa4f88ac5e68962b8e382ad60e27bdc48eff64cddd7, plan9bb8b150fe73fcae615a301a07b0d313eec1a305004b445fc31deb0289b97b69, result7d998485682ecd8208d4fea6bf7464b47c463e04a125977f847f18b7fcca667f. Existingownedlauncherwithmeta_gpqa_moe_row_control_v1/owner_plan.json;session27180reaped0,worker424646absent,cleanclosure;67.690s includingmodelweightinitialization. No full-modelforward/KVallocation. NativeTTrouterinput/logits/topindices/128scoresandcompactscoremappingexact; originallocalMoEoutputexactbothchips. Allselectedexpertgate/up,activationinput/output,downprojectionrowsbitexact betweenindexedandscanwithsameB1programsandweightobjects. Finalweighted sumdiffers8/4elements (max .0009765625/.0000305176,relativeL2 .000162750/.000002479); no wrongexpertidentity demonstrated. Compact-order TT reduction followupwilltestthatremainingdifferencewithoutmodelinitialization. SmallartifactsP150x2/moe_row_control_v1;rawreadbacksremainexternal.

MoE compact-order actualTT reduction complete: ownerpland8eaf76ebaecb2b570c67e7c98f8487e2963a66cf6fd80eb6ac7056d90279039, plan74725affd8a1ac07600fbe55fe82edd82c31a141e9a6b9406e00f45d7a19dc7e, result45440700df73a85f0401c00a4089d2b201fbae5f0c09313ddabd33e9795ca58a. Existingownedlauncherwithmeta_gpqa_moe_compact_sum_v1/owner_plan.json;session44611reaped0,worker438060absent,cleanclosure. Actual3.041s, no modelinitialization/KVallocation. Sameactualselecteddownrows andcompactTTscores reproduceoriginalnativeweightedoutputexactly; scanningrows inidenticalcompactorderproduceidenticalweightedrows/localoutput. Earlier8/4elementdifference isolatedto128-rowlayout/reductionpath, notwrongexpertrows. No repetitionrepair demonstrated. Fresh xhighAutoDebuginvestigatesremainingsharedarithmeticwithoriginalrecipe/noCPUconstraint; no productionchange orbroadeval.

## Focused gate fidelity control — 2026-09-11T21:06:53.782580+00:00

Actual physical-host TP2 run completed in68.816s, owner exit0/clean mesh and no survivors. Session42432 reaped, worker480697 terminal. No server, container or CPU model inference. Command: `/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_moe_gate_fidelity_v1/owner_plan.json --plan-sha256 9265a3f2e421608bb07716a0f53f485fa94ce1b7299b16462c17d00b3edf769f`. Sources311 validated; original serving env, FABRIC2D TP2 IDs[1,0], watcher/profiler off; autoport source and installed runtime unchanged.

Result `P150x2/moe_gate_fidelity_v1/result.json` SHA256d0f08f973afa69a11c5778c593e712c53b6be31d65529e17133b254986406294: all native router/expert/local/CCL/norm guards exact. Gate-only HiFi4 preserves DST=false, packer=true, block44 and native down. Packed relative L2 to independent block1 reference increases from3.39–4.79% to3.97–6.27%; post-MoE-norm error changes1.6059% to1.4068%. However FP32 reference block1/2 differs up to0.9865%, invalidating policy inference. No quality trial admitted and no production policy changed. Source investigation found reuse factory lacks the FP32 partial-CB unpack flag already used by mcast1d to avoid TF32 reload rounding. Next narrow TT control addresses that reference discrepancy; no broad eval.

## FP32 reference admission failure — September11 21:11UTC

V2 command: `/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_moe_gate_fidelity_v2/owner_plan.json --plan-sha256 af8ef190678b5e3a6f3a52286044aba459184491c3225df590f58467d5fc001f`. Root validated336pins. An earlier old-draft-hash validate-only rejected changed probe before imports; finalfrozen31c2f772/template5dea4b validated and reviewed. Actual run failed4.994s before model initialization: both genericreuse and mcastFP32 paths return1.0 for exactzero-tail expectation1.000244140625. Ownerexit1, cleanmesh/cleanup, session17739reaped, worker501020absent. ResultSHA256aa15e22e9c24374074740f653e144c358ffa9dbce7115659f214e61cc59153eb. No gate-fidelity/quality verdict admitted. Preserve failure; next tinyTT localization distinguishes initialcontraction from partialreload. No modelsource/build/runtimechange, CPUmodelwork, server or broadeval.

Tiny no-spill/zero-tail actualTT control: command `/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fp32_reload_localization_v1/owner_plan.json --plan-sha256 a3ec2ec8ca21a7afb9f350784171c4333943cb5f83fb963e27ae51e873b89b5d`. Result9d2f177bda538efc00e6185a51f250e5e599d5fbe46bdb7ade573fdb93b766c1,3.423s, ownerexit0, cleanmesh; session81061reaped, worker517729terminal. Allfour invariant rows fail: K32 andK2816, reuse andmcast, output1.0 instead1.000244140625. Exit0 means diagnostic collection completed, NOT invariant or qualitypass. Exact input readbacks and priorK2816 failurehashes pass. Loss alreadyexists withoutpartialreload. Generated JIT source inspected byagent confirms mcastCB5Float32/genericCB5TF32, bothDSTtrue/fidelity4/packFP32. No persistentJITcachecopied. Next sourceformatcontrol usesidenticalrepresentableBF16/FP32operandvalues with unchangedaccum/output.

Exact source-storage/face control command: `/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_fp32_reload_localization_v2/owner_plan.json --plan-sha256 39b239ca156d44347bd1dc3977f37e57e33cde20c667a63173790fe8664f10b1`. Result b903df0f64311565480224f80135148ac7d45011ccf0ff42440f7807ae9630ef,3.434s, cleanexit0, session86312reaped/worker528890terminal. FP32/BF16 storage bothfail exacttwo-productsum with productsin same16-termface, atK32 andK2816. Moving secondproducttoK16 whilepreservingallvalues/FP32config givesexact1.000244140625bothchips. This localizes fixtureloss to intra-dotgroup path; reference flags alone cannotestablishprecision. No productionchanges orqualitypass. Next actualexpertreference uses separatelyvalidated TT SFPUFP32mul/add tree; no CPUmodelarithmetic.

SFPU reference v1 harness failure preserved at `P150x2/moe_gate_sfpu_reference_v1`; resultSHA25660fcdf0b939d8ab211d740751cdd994d83bd92aa49ba58dd3b8d1fac0c8f4725, elapsed4.002s, worker545551 cleanexit1/meshclosed. Exact owner command and runtime environment in owner.json. Model initialization was not reached. TTShape slicing corrected to list(shape) in v2. No qualitypass, source/modelpolicychange or CPUmodelarithmetic.

SFPU reference v2 harness failure preserved at `P150x2/moe_gate_sfpu_reference_v2`; resultSHA25684681835ed3f185a674a9a67239166209d3e430f41487163a31ae1d7d3721e76, elapsed4.013s, worker549436 cleanexit1/meshclosed. Exact owner command and runtime environment in owner.json. Model initialization was not reached. Explicit fast_and_approximate_mode=False for ADD only supports BF16output; v3 will omit flag for matchedFLOAT32, which still selects SFPU bydtype. This corrects harnessAPIusage, not modelprecision. No qualitypass, source/modelpolicychange or CPUmodelarithmetic.

SFPUreference v3 actual command: `/home/hous/dev/tt-metal/python_env/bin/python -I -B /home/hous/dev/tti-release-gemma4/meta_gpqa_fixed_prefix_owned_v1/launch.py --plan /home/hous/dev/tti-release-gemma4/meta_gpqa_moe_gate_sfpu_reference_v3/owner_plan.json --plan-sha256 26c3072971deb43ced4eacc4254a38ada889becb69a6934e8f4d7dfc7446048f`. Sources433 pinned; sameautoport/physicalhost/TP2IDs[1,0]/runtime. Exact SFPU small-term invariant and residentinputs/weights/treecopy/native/candidate guards pass. ResultSHA49b6abcb44a79303077196d2923657d1f918ad409d38df298905fe493ce50315,68.101s, ownerexit0, session72163reaped/worker553769absent, meshclosed/cleanupcomplete. Nativegate meanrelativeL2 toSFPU2.480%,HiFi4-only3.362%; candidateworsensall16selectedexpert/chiprows. Local/TPerrorsworsen, postnorm improvesmodestly1.3212%to1.1500%. Fidelity-only candidate REFUTED; no productionpolicychanged and no full-case trialadmitted. Reference usesoneFP32SFPUadditiontree onexactresidentquantizedvalues, isnotcanonicalfullmodelreference. Allbroadevalpaused; next freshAutoDebug investigatesremainingaccumulationerror usingthismeasuredreference.


Sparse FP32 admission: source-confirmed intermediate CB5 format/capacity mismatch fixed using selected intermediate format; native BF16 policy unchanged. Wrapper build unavailable (Docker); existing physical-host ninja -C build -j2 ttnncpp and cmake --install build --component ttnn-runtime both exit0. New loaded runtime e475e4435fd49106b75317d0a43ec822cd63b2e1e8bb7234d84071d81b7f8550. Tiny cancellation v1 admitted444pins, hung at first BF16 arm before FP32 candidate. Live tt-triage captured exit0; both active chips wait in input sender multicast. Owner session13491 cleanup requested after capture. No numerical or quality result inferred; source AutoTriage and serialized recovery underway.

Singleton sparse hang captured before termination: owner143, session13491reaped, worker591300absent, cleanupcomplete/no survivors. Triage report proves forbidden zero-recipient multicast; same dense sibling guard applied separately (source c0d66631468a6c112a5a880d51616de0c8a073ca8f2fb973fbcbf758ab149e26). Physical list/reset/list and four-chip mesh open/close all exit0, no second reset needed. Buildsession93947reaped0/install0; runtime4d49d17d5b0bdddfbd95748e9605ce26b8bee6c89e39f09e92a2e2bd19b789bb. Repeating identical tiny fixture next; no numerical/repetition repair inferred. Small evidence P150x2/sparse_fp32_cancellation_v1 and two sparse fix directories.

Singleton repair verified by identical one-core fixture: v2 result43686560dd5ac56fef9b0a1c7d8d7697131877228ced5f90d545ee284f58a6a3,4.147s, ownerexit0/cleanmesh/cleanup, session79665reaped. Alluploadguards pass. BF16two-block returns0; FP32firstblock returns exact1.000244140625; FP32two-block returns0 versus exact2^-12. Thus initialFP32partial passes and crossblockpreservation fails; no modelqualityclaim. Root applied separate directCB5reload patch (b31e732c179f165cda4a87ad4e145fc8b7912236a088519b52d02b0744ba1078), build pending. Next identicalTTfixture plus multicoreneighbor.

Sparse direct-reload repair actualTT verification passed: v3 result0d2b8643d89f52a03451ec245f6f834a73f4bfb6afbf9eb9eae931a9b14b5f9d,4.216s, ownerexit0/session46300reaped/worker617907absent, meshclosed/completecleanup. Same firstpartial1.000244140625 and BF16control0 hashes unchanged; repaired FP32two-block output exact0.000244140625 bothchips. Separate N64/two-core arms match bothduplicatehalves. Build76805reaped0/install0; runtime df48cbdee53a10187e7459d6034c80623db9a9f7b50ec59513c97ce3d2e491d3. Narrow operationrepair proven; no modelpolicychanged or repetition/qualitypass. Next realoperand FP32DST-only LoFi/packertrue gate candidate preserves allothernativefields and independentSFPUreference.

Actual saved-operand FP32DST-only gate control: result4a3601841d95b546e0f854921ada8f25dd04037669bd2cc1690d851d8b6856d1,68.021s, owner0/cleanmesh/cleanup/session96219reaped/worker626678absent. All476pins and exactnative/SFPUreferenceguards pass. LoFi/packertrue/weights/routing/block44 unchanged; nativegate meanrelativeL2.0248043 ->.0177458, improves16/16, but normalized.0132121 ->.0128567. Fails predefined half-gate/20%-normalized improvement admission. No fullcasegeneration or productionmodelpolicychange. Next freshAutoDebug examines fidelity interaction with validatedFP32DST; no broad eval. clang-format identified assignmentwrap only; whitespace-only formatting applied a07409f148550d6750a59d88f569604b61ef22181b9731d451030c8342f3199e, formatting checkpasses; rebuildpending.

Watcher-only sparse verification v1 failed before mesh open: ACTIVE_ETH instrumentedfabricprogram30112B exceeds26624B buffer, no sparseoperationran. Owner642730exit1/cleanupcomplete/session59164reaped; destructorstoppedWatcher andcloseddrivers. Failedresultb9df3b96120bf550e8b4011ee35e6402f9b14c38c40ca34d290fb29064d70b65 preserved P150x2/sparse_fp32_watcher_v1. Retrywilldisablefabric for this local-op-only fixture; noWatcherfeaturedisabled, nofabriccoverageclaim. Selectedpytest materializer separately rejected lexicalsnapshotpath-vs-resolvedpath metadata beforehardware; fixpending.

Durable sparse regression added and actual TT selectednode passed4/4 in2.62s: test_sparse_matmul_indexed.py::test_indexed_sparse_partial_cancellation --device-id 0 -q; ownerexit0,session50046reaped/worker648922absent/cleanupcomplete. Only literal BF16/BFP8 one/two-core cancellation cases collected; no CPUmatmul/referenceinference. Initial static lexicalpath helper rejection fixed in immutableownedv2. Blackcheck0 and C++clang-format/diffcheck0. Shutdown nanobind reference-count diagnostics retained inlog; no test failure. Watcherv2 now running local sparse scope with fabricdisabled to avoidunrelatedACTIVE_ETH instrumentationcapacity.

Local-only Watcher v2 passed5.859s: result43aae8c0457c50933110350dce324bf8ac8a769582638cccb2443a1cd95ff32e, all6outputs exactv3, rootinspected watcherlog anddiagnosticlog noerrorpatterns. Owner651054exit0/session38530reaped/meshclosed/cleanupcomplete. Watcher1000/profiler0/FabricDISABLED; nofabriccoverageclaim. Current3-arm realoperand run nativeBF16LoFi + savedFP32LoFi exactguards + candidateFP32HiFi4 onlyfidelitydelta, freshAutoDebug33a506...; no fullcaseadmittedyet.

Distinct fidelity-under-FP32 control ADMITTED for samecase only: result5ff778940c500edc6a2cc2953b562c2229c2747e4d55764d639eba1c0852accb,68.120s owner655149exit0/cleanup/meshclosed session14318reaped. Allnative/priorFP32LoFi/SFPUreference exact; candidatechangesonlyfidelityvsFP32LoFi. Mean gateerror.024804324 ->.001675159; normalized .013212071 ->.003096675 bothchips; all16rowsimprovevsnativeandFP32LoFi, allpredeclaredthresholdspass. No qualityclaim yet. Next original326prompt/2048budget/T0seed42/B1/context262144 baseline/candidate viaautoportadapter; preservedprefill and otherops. Broad eval remainsstopped.

Local commit bc4752de65dce32b8e68eacd7ebcb40dac276cbd records only sparsefactory threeprovenrepairs plus61-line durabletest. Independent scopedreview no blockingfindings (P150x2/sparse_kernel_retention_review_v1/REVIEW.md), all applicable pre-commit hooks passed. No push. Hooks temporarily stashed/restored unrelateddirtyfiles; root verifiedmultichipe64e290.../optimized29dbdc.../runtime df48... unchanged afterward. Fullstage review andmodelquality pending; originalserver snapshots retainhistoricalcommit, newexecution records currentHEADseparately.


Original-case expert precision candidate failed qualitative acceptance. Result cff57f4cabc71877bfdc227f5a734408ae69cc15f2daaec91589bf241a7ffc2f; owner 672211 exited0 with complete cleanup and clean mesh; session70075 reaped. Baseline reproduced all2048 IDs and full saved text exactly (127.266s). Candidate preserved prefill and all non-gate operations, also generated2048 tokens and finished length (127.833s). Root read both complete outputs: candidate mentions optionD/18 incidentally, then rejects that reasoning and repeatedly considers identical C6H10 alternatives; no final answer. Strong numerical improvement is not a repetition repair. ACTUAL_QUALITY_REVIEW.json records this failed admission; no model policy patch applied. Proven kernel fixes remain committed. Fresh AutoDebug will audit remaining causal hypotheses, including original prompt/template wiring; no broad eval or further precision matrix.

### Focused prefill follow-up after decode-only quality failure

Fresh AutoDebug report `P150x2/after_expert_quality_diagnosis_v1/AUTODEBUG.md` verifies the original326-token Meta input exactly matches native default/explicit-false chat templating. R1 success changes instructions, choices, sampler and budget and cannot serve as a matched control. Earlier saved prefill captures omit the complete routing group. The bounded next TT control captures original layer0 rows320:352, assesses logicalrow325, requires exact native terminal logits and operation replay, and tests the known HiFi4/FP32 gate against the independent TT SFPU reference. No production policy or broad evaluation admitted. Root confirmed all495 inherited quality-plan pins, no TT fd holders, worker672211 absent and port8000 closed.

### Original-prefill gate control: numerical admission passed

Root executed frozen `meta_gpqa_prefill_gate_control_v1/owner_plan.json` (SHA `effb99ae616f5ba09ade73b79105e35c5c6bd66f420e1c6818f747980440cae2`) via existing owned launcher on the physical host, current HEAD `bc4752de65dce32b8e68eacd7ebcb40dac276cbd`, runtime `df48cbdee53a10187e7459d6034c80623db9a9f7b50ec59513c97ce3d2e491d3`. Original326-token prefill and native operation/352-row CCL/norm replay were exact. Matched HiFi4+FP32 gate on identical packed BFP8 weights improved all16 expert/chip rows against TT SFPU: mean relative L2 0.0283409069 to0.00167508293; normalized0.00972598356 to0.00314167429 on both chips. Observed scratch allocation delta62546944 bytes/device is within83886080 bound. Result SHA `34b23070daa5adeb3f6751b5cd76ff51f33805af884f04322f330593a47d9d38`;73.568s, session41470 reaped0, worker740686 absent, owner cleanup complete and device cleanly closed. No generated tokens, production policy or quality pass. Complete raw capture retained outside repository. Source review fixed draft phase/CCL geometry issues before execution. Small evidence is `P150x2/prefill_gate_control_v1/`. This admits one unchanged original-case candidate covering both prefill and decode, compared with saved native and decode-only results; broad evaluation remains paused.

### Combined prefill+decode gate candidate: original quality failure persists

After local prefill admission and independent scoped review, root executed one candidate under `meta_gpqa_expert_both_phase_quality_control_v1/owner_plan.json` SHA `f29b0aed3b13b8e0c09253c6bda4d6a5ec7a1aaad4fe60e9b674cda9d9225ebf`. Baseline/decode-only trajectories were reused at exact current source/runtime, not regenerated. Original326/2048/T0/seed42/context262144/slots32/HMA4128/B1 recipe and native EOS/sampler were preserved. All330 prefill gate calls and2047 decode gate calls per layer passed instrumented config/geometry guards; native down/SDPA remained unchanged. Result SHA `c78f72a0ed0e46d7598d8537bd8c7b70a934a2d644c182c6972e243b12658abf`, text SHA `d52602d3ffd78c01c1f2c724bde96880aeeb4f7876f8c2a3cb21f338e9834326`. Candidate finishes2048/length after repeatedly revisiting rejected chemical interpretations and gives no final answer. Root read complete output: no repetition repair, no policy applied, no checkpoint-limitation assertion or waiver. Total199.716s, generation129.073s; session45093 reaped0, worker756543 absent, owner cleanup complete, mesh cleanly closed. Small evidence and complete text in `P150x2/expert_both_phase_quality_control_v1/`; raw log remains external. Broad evaluations remain paused. Proven sparse operation fixes in local commitbc4752 remain separate from failed model-quality hypotheses.

### Focused AutoFix final disposition

Independent `P150x2/focused_repetition_autofix_v1/AUTOFIX.md` (SHA `de56f34742c29e521397422625bca13be8165218a8b1b3cc638169d3b9b82813`) confirms that this focused repair chain failed to resolve the original repetition/answer-completion error. No distinct remaining implementation cause is supported by current measurements; attribution to checkpoint behavior versus another unlocalized TT difference remains unresolved. No model precision policy, threshold waiver, CPU inference, broad evaluation, release pass or whole-stage clean-pass was introduced. Current model/runtime byte pins remain unchanged. Root verified no TT fd holders, both new workers absent and port8000 closed. Further work needs new discriminating evidence or engineering direction. Frozen diagnostic script copies use `.py.txt` to preserve executed bytes as evidence. The local focused-evidence commit is recorded externally in `/home/hous/dev/tti-release-gemma4/meta_gpqa_focused_repetition_autofix_v1/COMMIT.json`; no push is authorized or performed.

Evidence-commit validation: the first commit attempt stopped because the EOF formatter added a newline to the verbatim generated-text copy. The executed external text remains unchanged. Its complete bytes are now preserved in `expert_gate_both_phase_hifi4_fp32.text.json` with encoding and original SHA; JSON decode exactly reproduces the original no-newline text. This allows all hooks to run without altering or waiving evidence.
