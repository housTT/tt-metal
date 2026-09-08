# vLLM integration work log

## Starting contract

Target `ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, pinned local snapshot
`/home/hous/dev/ornith-1.5-9b/upstream`. Start from clean branch
`hous/ornith-1.5-9b` at `85710be49fbcce5579aeb6d9571cd3bc1829d953`.
Selected skills: vllm-integration, tt-device-usage; AutoFix environment
investigation invoked after prerequisite failures. Independent stage review
and local commits are required after completed serving verification; never push.

Selected policy is read from `../datatype_sweep/selected_precision_config.json`:
BFP4/LoFi body and LM head, BFP8/LoFi decode QKVG, layer31 attention and MLP
BFP8 exceptions, BF16 activation/residual/logits/sampling, native producer CCL,
BFP8 KV, FP32 recurrent state, UINT32 tokens, head C32/K4/two readers.
`OrnithModel` loads that selection by default. Context contract remains262144.
The previous stage validates mesh1x4 on four Blackhole chips/two P300c boards.

## Environment and source checks

- `importlib.util.find_spec` confirms no `vllm`, `vllm_tt_plugin`, or `openai`
  in the active Python environment; `environment_check.json` records details.
- No vLLM/plugin checkout exists in this workspace. AutoDebug investigates
  alternate mounted environments. `/work` aliases this checkout.
- Restored `models/common/readiness_check/run_vllm_server.py` and its existing
  `test_run_vllm_server.py` from local git object
  `70a596f92229ada922fba743cd0cd9d2658a5c1c`; hashes and exact-source equality
  are in `environment_check.json`. No upstream network resolution was used.
- `python -m pytest --noconftest models/common/readiness_check/test_run_vllm_server.py -q`
  fails during collection with UID1002 missing from passwd (`runner_host_tests.log`).
- `USER=hous python -m pytest --noconftest models/common/readiness_check/test_run_vllm_server.py -q`
  resolves the user lookup and fails during collection with missing `openai`
  (`runner_host_tests_user.log`). No test passed or serving workload ran.
- `USER=hous python -m models.common.readiness_check.run_vllm_server --help`
  exits1 with missing `openai` (`runner_help.log`). Both restored Python files
  pass AST parsing. This Python-only source restoration requires no C++ build.
- `python -m black --check models/common/readiness_check/run_vllm_server.py models/common/readiness_check/test_run_vllm_server.py`
  exits0 (`format_check.log`). No formatter dependency was installed.

The supplied AGENTS.md says: "Do not try to install compilers or dependencies".
The user was asked for a provisioned environment/path or explicit authorization
to provision serving dependencies. No installation, device reset, or reboot
was attempted. No TT device was opened by this stage, and no serving job was
launched. Process audit follows the short-lived runner `--help` check.
The help child exited1 and was reaped; final `environment_check.json` reports
no serving processes. No processes were killed and no device locks were cleared.

## Adapter boundary audit for resume

Read the shared vLLM integration guide, `tt_transformers/tt/generator_vllm.py`,
the 9B generator/model, the shared `contract_vllm.py`, and the prior35B adapter.
Do not copy the old35B adapter's quality/performance claims or eager-prefill,
prefix-cache and YaRN limitations into this model.

- Delegate to `OrnithGenerator.prefill_forward`, `decode_forward`, and
  `configure_sampling`. The canonical model and sampling traces already replay
  nonblocking and feed sampled tokens into persistent device inputs.
- Stable asynchronous decode should call the low-level decode with tokens and
  positions `None`; its page-table refresh already compares content before
  upload. Test scheduler transitions, stale inputs, and changed/unchanged pages
  through the actual plugin before enabling async capability.
- Low-level decode device output aliases the feedback tensor; deferred read
  must queue its copy before the next replay overwrites it. The current9B
  generator needs a low-level deferred-read primitive; preserve queue ordering.
- The current `allocate_cache` sizes blocks as batch times per-request context.
  Add a proven caller-owned shared-pool boundary once the plugin allocation
  interface is available. Bind that exact cache before generator trace capture;
  do not allocate a second standalone serving cache.
- Fixed-slot recurrent/conv state is already model-owned. Confirm the plugin's
  slot compaction/remap contract and implement state movement in the generator,
  including sampler histories and seeds, with focused tests.
- `prefill_forward` accepts logical lengths and explicit slots/start positions.
  It validates full fixed-slot page-table coverage. Translate scheduler table
  padding without truncating logical prompts or lowering native context.
- The restored runner only accepts N150/N300/T3K/TG labels. Establish actual
  Blackhole plugin mesh configuration before adapting it; do not label a1x4
  Blackhole workload as T3K or claim a configuration has been served.

No successful server command, TT config, max-num-seqs, sampling result,
qualitative output, benchmark artifact or runtime fallback proof exists yet.
Final target remains full sampling, on-device token-out, decode trace and async
proof, native context, non-aligned prompt coverage, up to32 sequences,
primary128/128/1 and secondary100/100/32 benchmarks, independent clean-pass,
and stage-owned local commits with SHA receipts. No commit has been created.

## AutoFix outcome — 2026-09-06 UTC

[AutoDebug](AUTODEBUG_environment.md) and the follow-up
[AutoFix experiment](AUTOFIX_environment.md) could not resolve the missing
runtime within the current no-install constraint. All three discovered Python
interpreters lack vLLM, TT plugin and OpenAI; changing interpreters or restoring
source does not close the prerequisite. Exact probes are in
`autofix_environment_probes.json`. Final process audit at00:01:09 UTC found
zero serving/EngineCore processes (`autofix_environment_process_audit.json`).
Stage remains incomplete pending a provisioned environment or installation
authorization. No independent stage-review pass or stage commit exists.

The same external dependency was revalidated on three consecutive goal turns;
`blocker_audit_turn2.json` and `blocker_audit_turn3.json` record continuation
checks. Goal status is blocked, not complete, after AutoFix could not close the
prerequisite. Resume this stage when a serving environment or explicit dependency
installation authorization is provided. All remaining stage gates are unchanged.

## Supervisor recovery — 2026-09-08

The dependency-permission blocker is resolved by the existing user authorization, now explicit in root/model AGENTS.md. Read SUPERVISOR_RECOVERY.md and provision the task-local serving environment. Preserve all prior evidence and gates.

## Resumed implementation — 2026-09-08

Fetched `tenstorrent/vllm` to sibling `/home/hous/dev/ornith-1.5-9b/vllm`,
checked out candidate `bf98d556bb46a5cda25fac540629251e7f474200` on local branch
`hous/ornith-1.5-9b-vllm`. This is the prior Ornith port's tested source baseline,
chosen for its compatible plugin layout; this model still needs all serving
gates. Provisioning logs record the isolated environment and preserved base.
Supervisor-owned changes in both AGENTS files are excluded from stage commits.

Initial `tt/generator_vllm.py` delegates to canonical generator methods. It uses
an explicit architecture alias `OrnithForCausalLM` / `TTOrnithForCausalLM` so the
existing Qwen3.5 registration is unchanged. Server launch will pass
`--hf-overrides '{"architectures":["OrnithForCausalLM"]}'` with the pinned local
snapshot. Async capability remains false until adapter tests prove it.

Runner now accepts the plugin's P150/P150x2/P150x4 labels and derives its TT
config CLI flag from installed EngineArgs. Worker configuration accepts an
explicit fabric packet payload size so the prior model's8192-byte ring policy
can be preserved. No serving speed or compatibility result is claimed yet.
AutoFix generator-contract work adds explicit shared-pool cache sizing,
device-only state remapping and selective refresh, deferred output reads, and
an explicit host-logits compatibility switch without replacing split sampling.
All hardware tests remain serialized in the stage owner; agents own host/source
work only. Bounded `timeout 60 tt-smi -ls --local` exits0; see
`device_list_resume.log`. No reset was needed.

## Reduced serving and focused repair — 2026-09-08

The isolated serving runtime is provisioned and host-validated; see
`provision_work_log.md` and `runner_compatibility_report.md`. Eight runner tests
pass, including private process-group cleanup after launcher exit. Actual pinned
ModelConfig resolves both Ornith aliases and accepts native262144 via
`--additional-config`.

`USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/run_server.py --label reduced_v2 --layers 0,3 --max-num-seqs 4`
launched the shared runner on TP4/P300c, selected precision, native262144 and
on-device split sampling. Exact argv/config are in `reduced_v2.command.json`.
A131-token completion request returned200 with8 generated tokens
(`reduced_v2_request131.json`). These reduced-layer outputs are contract-only.
A subsequent four-request smoke crashed during sampler-history slot remapping:
full-row `ttnn.repeat` requested2209024 bytes L1 against1572864 available.
`reduced_v2_multi_request.json` contains error bodies despite HTTP200; these
are failures. Full stack and shutdown are in `reduced_v2.server.log`.

The first focused broadcast candidate probe aborted during mesh initialization,
before exercising the candidate: active Ethernet core29-25 timed out.
`penalty_remap_broadcast.log` preserves this infrastructure failure. The child
exited134 and no task server/probe remained alive. Bounded `timeout 60 tt-smi
-ls --local`, `timeout 180 tt-smi -r`, and `timeout 60 tt-smi -ls --local` each
exited0 (`recovery1_list_before.log`, `recovery1_reset.log`,
`recovery1_list_after.log`). All four chips were visible; no second reset or
lock removal was needed. `USER=hous timeout 120 ../state/serving-env/bin/python
-c 'from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import open_ornith_mesh,
close_ornith_mesh; mesh = open_ornith_mesh(trace_region_size=0);
close_ornith_mesh(mesh); print("MESH_SMOKE_OK")'` exited0; see
`recovery1_mesh_smoke.log`. No profiler/watcher was active.

The exact candidate command, after recovery, was
`USER=hous ../state/serving-env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.penalty_remap_device_probe --model-path /home/hous/dev/ornith-1.5-9b/upstream --method broadcast --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/penalty_remap_broadcast_retry1.json`.
It exited0, proving exact remapping of all four real sampler-history buffers
on every replica with stable addresses. The largest buffer is32x262144 INT32.
`penalty_remap_broadcast_retry1.log` records clean device close. This verifies
the narrow broadcast repair hypothesis; actual implementation and original
multi-request reruns remain required.


`--method current` then exited0 with the same exact remap checks
(`penalty_remap_current.json`/`.log`). The original failing server shape was
rerun via the invocation wrapper with `--label reduced_v3 --layers 0,3
--max-num-seqs 4`; `request_smoke.py --output .../reduced_v3_requests.json`
passed single131 and concurrent131/129/65/131,8 output tokens each. The helper
asserts choices, absence of error bodies, and exact usage, not merely HTTP200.
Runner shutdown exited0 and a /proc audit found no serving/EngineCore process.
This closes the penalty-buffer L1 failure through its original serving path.

The broader generator probe v1 revealed invalid probe geometry: native page
table width4096 requires at least4096 physical cache blocks in fused update,
but the test supplied96. The test now uses4192 shared blocks, still much less
than three standalone native allocations12288. No runtime/kernel change was
made for this harness error. Its retry v2 aborted134 at Ethernet core29-25
startup after reduced_v3 shutdown, before testing. The same bounded second
list/reset/list recovery succeeds; see `recovery2_*`. A dedicated source-only
AutoDebug investigates the serving shutdown hook. Device-driver close messages
alone do not establish that mesh/fabric shutdown ran. Neither timeout is model
correctness evidence.

### 2026-09-08: AutoFix for post-serving device initialization

Fresh source-only diagnosis (`AUTODEBUG_serving_shutdown.md`) verified that pinned vLLM calls `TTWorker.shutdown`, but the plugin inherited a no-op and performed device cleanup only in its destructor. Added explicit idempotent worker shutdown with adapter teardown before mesh/fabric close, plus best-effort destructor fallback. Removed unconditional `ReadDeviceProfiler` from serving close. The host-only regression first failed (3 failed/1 passed, `worker_shutdown_before.log`) and then passed (4 passed, `worker_shutdown_after.log`). Exact test: `USER=hous ../state/serving-env/bin/python -m pytest -q -c /dev/null -p no:cacheprovider ../vllm/plugins/vllm-tt-plugin/tests/test_worker_shutdown.py`. No TTNN import or hardware command was run by this source investigation. Reproducing the original server-stop/next-open sequence remains required to establish the Ethernet recovery effect.


After recovery2 mesh-smoke exit0, the corrected broader generator probe
`...tests.serving_contract_device_probe --model-path /home/hous/dev/ornith-1.5-9b/upstream --output .../serving_contract_device_v3.json`
exited0. It proves B3 shared4192-block/native262144 cache use, exact all-replica
recurrent/conv/token/seed/history remapping including NaN isolation and UINT32
values, selective refresh, deferred two-step outputs equal to synchronous,
and explicit host-logits sampler bypass/resumption. Model replays6 vs sampling
replays5 correspond to exactly one intentional host compatibility step. These
are reduced contract checks, not full-model quality or performance.

The adapter device probe v1 passed synchronous and deferred-stale cases, with
zero token/position/RoPE/page refreshes across two steady replays and two
minimal deferred readbacks. Its changed-page inspection then hit a test-only
Python error (`ttnn.Shape` does not accept slice indexing). The case is not a
pass until the corrected probe completes. No device fault occurred, and mesh
closed normally. Evidence `adapter_device_v1.json` and `.log`.

The shutdown repair now passes its first hardware sequence: `reduced_v4_async.command.json` records native262144, B4, layers[0,3], async scheduling, ring8192, and runner exit0. Five original requests and six sampling transitions passed (parent-run). `reduced_v4_async.server.log:106-108` shows explicit worker mesh/fabric close completion at12:58:42. The immediate direct mesh open/close, without an intervening reset, printed `POST_SERVER_MESH_SMOKE_OK` and exited0 (`reduced_v4_post_server_mesh.log`). This verifies recovery of the original post-server-open sequence for one cycle. A second cycle remains planned at the next all-layer shutdown. See the updated `AUTODEBUG_serving_shutdown.md` for scope and evidence.


## Async proof and all-layer serving

`adapter_device_v2.json` passes every real adapter case: synchronous vs two
queued deferred reads with deliberately stale host tokens/current positions,
unchanged page tables, changed physical page at a64-token boundary, and slot
permutation. Each pair advances positions/RoPE exactly twice. Steady-state
counter deltas are zero token/position/RoPE/page uploads, two model and sampler
replays, and two deferred readbacks. Changed-page/remap cases upload one table,
not one per token. The changed page receives every K/V shard write while the
old page stays unchanged. Stable persistent addresses and cache identity pass.
`supports_async_decode=True` was enabled only after this proof.

With final fresh-slot admission code and async capability, the same probe also
passes the native allocation guard: `USER=hous TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 ../state/serving-env/bin/python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.adapter_serving_device_probe
--model-path /home/hous/dev/ornith-1.5-9b/upstream
--require-trace-allocation-tracking --output .../adapter_device_tracker_v3.json`.
Both JSON/log record exit0 and clean close. Program-cache allocation tracking
was included. No live tracked unsafe allocations survived replay boundaries;
this is the native conservative lifetime check, not a physical overlap metric.
No profiler/watcher or corruptible-allocation acknowledgment was used. The
tracker's GC overhead is diagnostic only; normal serving excludes it.

`run_server.py --label reduced_v4_async --layers 0,3 --max-num-seqs 4
--async-scheduling --allow-host-sampling` then passed the original non-aligned
single/concurrent smoke and all targeted neutral/penalty/host-logprobs/device
transitions. Explicit host compatibility was enabled for tests; ordinary
sampling remained on-device, trace_mode=all. `reduced_v4_transitions_v2.*` is the
passing transition evidence; its earlier harness false failure compared an
entire usage dictionary and rejected an optional null metadata field. Exact
prompt/completion/total counts are now asserted individually.

`run_server.py --label full_b32_v1 --max-num-seqs 32 --async-scheduling
--allow-host-sampling --sampling-profile full` launches all32 layers with the
same selected policy, TP4/P300c and native262144. Server health passed after
approximately80seconds (`full_b32_v1.runner.log`). A shared-runner attached
smoke sampling invocation passed3 tests, skipped1 all-vocabulary logprobs test
by the canonical suite (`full_b32_sampling_smoke.log`). Full73-test sampling is
in progress; the bad_words failure is under AutoDebug and is not waived.

The serving runtime's original chat-template rendering and IDs match all six
pinned HF/selected-policy control prompts. `qualitative_prompt_format.json`
records exact messages/renderings/IDs and both serving generation profiles;
`qualitative_controls.json` contains existing128-token HF and selected TT
controls. Transformers5.12 returns BatchEncoding by default from tokenized
chat rendering; the metadata check explicitly requests return_dict=False.
This is an API return-shape adjustment, not a tokenizer/template change.


2026-09-08 bad-words AutoDebug/AutoFix continuation: the original full sampling
run is preserved in `full_b32_sampling_full_v1.log` (69 passed, 3 failed, 1
skipped). The exact five-request probe reproduced the greeting text failure
with zero banned generated IDs: `bad_words_original_token_ids.json` shows
seed4 output IDs71 (`h`) and4638 (`ello`), not banned singleton14556/23066. The
canonical greeting check now verifies actual generated token subsequences
against `SamplingParams.update_from_tokenizer` variants, with nonempty text
and token-ID assertions. Negative controls reject actual forbidden singleton
and multi-token sequences. No model-specific exception or test waiver was used.

A separate verified TT-plugin bug dropped output history for multi-token bad
words when penalties were neutral. The minimal model_runner condition now
retains that history; the ordinary on-device path still copies no history.
CPU regression before the fix failed1/passed3; after the fix and assertion
negative controls all7 pass. Commands and evidence are in
`AUTODEBUG_bad_words.md`, `AUTODEBUG_bad_words_history.md`,
`bad_words_history_before.log`, `bad_words_history_after.log`, and
`bad_words_host_regressions.log`. Runtime history confirmation and final
canonical rerun remain pending the parent's serialized server work.

The serialized live pre-fix multi-token history probe now confirms the runtime
bug (exit1): unbanned and `bad_words=["New York"]` requests both returned exact
IDs `[3446,4121,1478,4121,1478,4121]` (three New York phrases), while the
independent final-token-only control correctly returned `[4121]`. Artifacts:
`bad_words_history_before_server.json` and `.log`. These requests used neutral
penalties and explicit host controls; post-restart verification is pending.

### 2026-09-08: second shutdown regression cycle verified

All-layer B32 native262144 async serving (`full_b32_v1.command.json`, runner exit0) explicitly closed worker mesh/fabric at13:24:43 (`full_b32_v1.server.log:1807-1808`). With no intervening reset, `presence_sampler_exact.log` shows fresh four-device fabric initialization at13:25:27.124 and close completion at13:25:32.963. The following all-layer standalone run (`standalone_haiku_512_v1.log`) also initialized four-device fabric, loaded through layer31, and closed at13:28:24.990. The shutdown AutoFix now has host regressions plus two successful server-stop/reopen cycles, including all-layer B32. `AUTODEBUG_serving_shutdown.md` records the final fixed verdict and exact evidence. This follow-up edited documentation only; hardware and processes remained under the parent's control.

2026-09-08 final continuation: the selected standalone haiku control completed
with all390 token IDs exactly matching the serving512 completion, through EOS.
The pinned BF16 HF control completed376 tokens with correct5/7/5; selected TT
and serving both produce6/7/5. Stage disposition is a controlled selected
full-model quality limitation, not a claim of correct haiku execution or HF
quality parity. Exact full-model reproduction satisfies the serving regression
control requirement; the selected datatype policy is preserved. The earlier
128-token suite ended before this error. All12 shared texts were read for
coherence, topic, repetition, gibberish, language drift and contamination;
see qualitative_extended_control_report.md and the raw control artifacts.

The adapter host suite was rerun after teardown counter logging:30 passed
(5 adapter,9 serving contracts,16 penalty admission), command:
`USER=hous ../state/serving-env/bin/python -m pytest -q
models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_vllm.py
models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_serving_contract.py
models/autoports/ornith_ai_ornith_1_5_9b/tests/test_prefill_penalty_admission.py`.

Final all32-layer server command: `USER=hous ../state/serving-env/bin/python
models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/run_server.py
--label full_b32_final --max-num-seqs 32 --async-scheduling
--allow-host-sampling --sampling-profile full` (exact manifest
full_b32_final.command.json). It preserves
native262144, TP4, selected precision and split decode traces. Source hashes
and safe environment fields are recorded in the manifest. The explicit host
mode exists for shared host-only tests; ordinary decoding remains on device.

The restarted full server now passes the same multi-token history API probe
(exit0). `bad_words_history_after_server.json`/`.log` preserve the unchanged
unbanned New York control, banned output IDs `[3446,1478,1478,1478,1478,1478]`
with neither forbidden sequence, and legal standalone final token `[4121]`.
The runtime history defect is fixed with CPU and live before/after evidence;
`AUTOFIX_bad_words.md` records the consolidated disposition. Canonical targeted
and full sampling reruns remain parent-owned broader gates.

The first corrected full profile completed72passed/1skipped in329.46s through
the shared runner (exit0). Raw archive: full_b32_sampling_before_penalty_order.log;
runner log: full_b32_sampling_final.runner.log. full_b32_final stopped0 with
explicit mesh/fabric close and no serving processes left. Source audit then
verified a distinct combined-penalty ordering defect in the shared canonical
`models/common/sampling/tt_penalties.py`: repetition must precede frequency and
presence. AutoFix first proved6/13 focused CPU failures and five mismatching
combined configurations against the actual pinned host helper; after the
minimal operation reorder all13 pass and all12 host-reference cases agree.
No dtype, buffer or sampler replacement was introduced.

Serialized exact device command (exit0, clean close13:46:27 UTC):
`USER=hous ../state/serving-env/bin/python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.presence_sampler_device_probe
--case combined --model-path /home/hous/dev/ornith-1.5-9b/upstream
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/combined_penalty_sampler_exact.json`.
The matching .log preserves launch/close. AllTP4 shard scores and selected
IDs match for mixed32 lanes, positive/negative logits, zero crossings, prompt
versus output histories, counts1/7 and two trace replays. Addresses and warmed
program cache stay stable. The common fix and both focused tests pass all
applicable pre-commit hooks (combined_penalty_precommit.log). The vLLM repo
hooks also pass (plugin_precommit_final.log); Ruff only sorted the new test
imports. A fresh full_b32_release server uses the corrected sampler and hashes
it plus plugin model_runner in its command manifest.

Runner-side stage script currently passes0 (stage_check_initial.log): scopeall
degeneracy and nativecontext gates. Its2048 advisory refers to the CPU-only
registry constructor probe, not serving. Interpreter-exit nanobind leak
diagnostics persist in child shutdown logs; explicit device/fabric close,
process exit and subsequent successful device opens show these do not leave
an EngineCore process or mesh reservation alive. No steady-state leak or
absence of all native binding leaks is claimed.

The next final-server sampling run passed71/failed1/skipped1. The failure's
`first tokens` were actually seven leading characters (`' '`), produced by
string slicing in the canonical seeding/variety harness. The original token
IDs were not captured, so their actual variety remains unknown. Fresh
`AUTODEBUG_first_token_variety.md` preceded the minimal assertion repair;
`AUTOFIX_first_token_variety.md` records commands and evidence. All five
conceptual first-token checks now consume actual API token IDs while retaining
full-text variety, thresholds, batch shapes, sampling parameters, and seed
checks. Only response metadata was added. CPU execution of the actual tests
reproduced the bug before the fix (1fail/1pass) and passes11 expanded positive
and negative controls afterward (`first_token_variety_cpu_before.log` and
`first_token_variety_cpu_after.log`). Parent-owned one-shot live diagnostic and
canonical rerun remain pending; no runtime sampling changes were made.

The parent ran the first-token live diagnostic once. Actual first IDs were
`[326,710,318,318,357,15,23]` (pieces ` S`, ` K`, ` (`, ` (`, ` A`, `0`, `8`).
Five leading spaces hid four distinct first tokens, confirming the source
semantic defect on real serving output. Both variety measures passed this
new batch; it did not reproduce the original seven-space shape, and the
original failing run's token IDs remain unknown. No retry-until-pass was
used. Exact raw evidence: `first_token_variety_original_probe.json`/`.log`.
Corrected canonical node/full-suite reruns remain parent-owned.

2026-09-08 full_b32_release follow-up: corrected penalty-order suite yielded
71passed/1failed/1skipped in331.82s; the sole failure compared first text
characters against a claimed first-token variety requirement. Raw failure is
full_b32_release_sampling_character_failure.log. The original response IDs
were not captured, so its exact seven-token variation is unknown. AutoFix
proved the harness error using actual-method CPU controls, corrected all five
conceptual first-token sites to API token IDs, retained thresholds/sampling/
full-text assertions, and added explicit seeded first-token comparisons.
One-shot original-shape live probe first_token_variety_original_probe.json
shows five leading spaces conceal four distinct first token IDs (the full
batch also has0/8 characters, so it does not reproduce the old seven-space
shape). The exact corrected failing node passed1/1 in1.80s, log
first_token_variety_targeted_live.log. This is not a retry-until-pass waiver.

Independent shared-runner `--stages qualitative,benchmark` on full_b32_release
completed0; all12 new texts were read (qualitative_final_review.md/json). All
six greedy texts exactly match the earlier controlled suite. New sampled texts
remain coherent/on-topic without mechanical repetition, gibberish, language
drift or contamination; the sampled haiku recounts and corrects a draft count
but remains incomplete. Both French outputs finish correctly. Long-form story
and executable Fibonacci completion remain unestablished within256tokens.

Pre-host-resume B32 server metrics are archived under
`b32_before_host_resume/`, preserving raw/normalized files and logs. On
128/128/1/concurrency1, server max-num-seqs32, greedy0: TTFT274.420ms P50/P99,
TPOT67.591ms mean/P50/P99, ITL67.567ms P50/68.366ms P99, aggregate14.449tok/s,
TPOT-derived14.795t/s/u. This is a capacity-32 server, not the B1 headline.
CI100/100/32/no explicit concurrency, server max-num-seqs32, greedy0:32/32
completed, TTFT2709.320ms P50/2710.547ms P99, TPOT75.604ms mean/74.987ms P50/
88.615ms P99, ITL67.641ms P50/309.686ms P99, aggregate315.739tok/s; derived
13.227t/s/u is secondary burst context only. Source manifest
full_b32_release.command.json identifies the measured pre-host-resume adapter.
Final B1 headline and final serving rerun remain pending.

full_b32_release stopped0 with explicit mesh/fabric close14:04:48 UTC.
The serialized `USER=hous timeout 600 ../state/serving-env/bin/python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.host_resume_device_probe
--model-path /home/hous/dev/ornith-1.5-9b/upstream
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/host_resume_device_exact.json`
passed0 and closed the mesh14:05:22. It uses real layers0/3/B3/native context
and compares combined-penalty uninterrupted device continuation with an
explicit host-logits step followed by unchanged-key device resumption. Tokens,
positions, prompt/output masks and counts match exactly on every shard;
host step leaves device counts unchanged and resume performs exactly one
restoration. A source-proven adjacent host-first seed-admission defect remains
under AutoFix before the final server restart.

The host-first seed repair is now proven by
host_resume_seed_device_exact.json/log (exit0; mesh close14:14:24 UTC).
It reruns the history detour plus fresh hostseed11→99 admission with and
without slot remap. The intended first device counter is43507, next43508;
all other lanes advance without reset. Actual request seed setup delegates
to the canonical generator; ongoing host/device transitions retain prior
device streams. No uninterrupted stochastic host/device equivalence is
claimed. The transition audit has no further proven defect in its scope.

Final-source adapter host tests pass5/5 (adapter_host_release.log), and the
native allocation-tracked device probe passes every stale-token/position,
changed-page and permutation case (adapter_device_tracker_release.json/log,
exit0, clean close14:16:41). Command matches adapter_device_tracker_v3 with
the new output path; tracking1 and program-cache tracking0-skip are explicit.
Normal serving does not inherit the diagnostic tracking environment. All26
stage Python files pass repository pre-commit hooks (stage_python_precommit_release.log),
and all touched plugin files pass its hooks (plugin_precommit_release.log).
No C++ changed, so no build is needed.

Final launch uses run_server.py --label full_b32_verified --max-num-seqs 32
--async-scheduling --allow-host-sampling --sampling-profile full (exact
argv: full_b32_verified.command.json). vLLM's
TPsize is1 because its single TT worker owns the1x4 mesh; model-internal
TT tensor parallelism is4. This is not four vLLM worker processes.

The next full suite (69pass/3fail/1skip) exposed a regression in the newly
added first-token helper: its nonempty-display assertion rejected valid
immediate EOS (`finish_reason=stop`, token_ids=[248046], text=''). The original
canonical top-k test had no such assertion. The complete failure is archived
as `first_token_eos_sampling_failure.log`; the pinned tokenizer confirms EOS
248046 decodes to empty text when special tokens are skipped
(`first_token_eos_tokenizer_control.json`). The helper now accepts empty
display only with terminal stop plus the actual tokenizer-configured EOS ID.
Response count, string text, nonempty token IDs, and all original variety and
seed assertions remain. It has no hardcoded model token ID. CPU reproduction
before repair:1fail/11pass; expanded controls after:17pass
(`first_token_eos_cpu_before.log`/`first_token_eos_cpu_after.log`). No runtime
or sampling-policy change was made. Parent targeted top-k/full reruns pending.

Final-source B32 qualitative/benchmark attachment completed exit0:
`USER=hous PYTHONUNBUFFERED=1 ../state/serving-env/bin/python -m models.common.readiness_check.run_vllm_server --stages qualitative,benchmark --model-dir models/autoports/ornith_ai_ornith_1_5_9b --hf-model /home/hous/dev/ornith-1.5-9b/upstream --mesh-device P150x4 --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --server-url http://localhost:8000`.
Log: `full_b32_verified_quality_benchmark.runner.log`. All12 texts were read;
`qualitative_final_review.md/json` records the exact current hash and verdict.
All six greedy texts equal the controlled earlier suite. The sampled story
now begins a coherent compass narrative; sampled haiku drafts valid5/7/5
then mislabels a7-syllable refinement as5. This is not a completed-task pass.
No mechanical repetition, gibberish, wrong-language drift or contamination
appears in the normal shared suite. Scope-vLLM degeneracy passes exit0 in
`qualitative_degenerate_final.log`.

Final B32 capacity benchmark, greedy0,100 input/100 output/32 requests,
no explicit concurrency limit, max-num-seqs32, native context262144:
32/32 complete, TTFT P50/P99=2049.029/2050.523ms; TPOT mean/P50/P99=
75.606/74.991/88.572ms; ITL P50/P99=67.627/309.555ms; aggregate337.722tok/s;
TPOT-derived13.227t/s/u is secondary burst context only. Canonical raw and
normalized files are `../../readiness_vllm/vllm_ci_serving_result.json` and
`vllm_ci_serving_benchmark.json`; exact command/config is embedded there.
The same server's128/128/1/concurrency1 primary-shape diagnostic is archived
under `b32_final_primary/`: TTFT271.340ms P50/P99, TPOT67.591ms mean/P50/P99,
ITL67.561/68.382ms P50/P99, output14.454tok/s, derived14.795t/s/u.
Its server capacity is32 and it is not the B1 headline.

The three canonical top-k EOS regression nodes now pass3/3, exit0:
`first_token_eos_targeted_live.log`. The final full sampling rerun uses the
same native B32 server with `--stages sampling --sampling-profile full`;
log `full_b32_verified_sampling_final.runner.log`. Runtime hashes remain
those in `full_b32_verified.command.json`; only the test assertion changed.

The stage review also requires classifying poor outputs in the original
bad-word test: seed2 replies in Chinese and seed3 drifts into netcat. These
remain wrong-language/off-topic observations, not qualitative passes or
proof of request contamination. `AUTODEBUG_bad_words_distribution.md` records
the pending source/CPU-localized investigation. Prepared
`bad_words_distribution_control.py` for parent-serialized same-seed2/3
banned/unbanned host controls, exact first-divergence token prefixes, neutral
raw top20 logprobs, and the actual pinned bad-word mask. It records source
hashes/request IDs and labels removed probability mass a lower bound. A CPU
fake-client dry run passed all6 request branches with no network/TTNN; no live
quality result is inferred from that dry run. Live paired controls and the
independent standalone numerical baseline remain pending.

Final shared-runner sampling exits0: **72 passed,1 canonical skip in331.57s**.
Canonical readiness log and all TT test source hashes are bound by
`sampling_final_manifest.json`. Numerical HTTP repeat/permutation control also
passes0 (`logit_determinism_vllm.json/log`): all9 comparisons of actual IDs,
chosen-token raw logprobs and complete top20 maps are exact. Logical prompt
usage131/65 is exact; continuations are coherent prose (` and smiled.\n`,
` and better able to`). These diagnostics explicitly use optional host
logprobs; device-sampling correctness and benchmark evidence remain separate.

The final B32 server stopped exit0 at14:39:42 UTC, with explicit model/mesh/
fabric closure (`full_b32_verified.server.log`). Process audit14:40:03 finds
no vLLM launcher or EngineCore (`full_b32_verified_process_cleanup.json`).
Lifetime counters are `full_b32_verified_counters.json`:6944 model replays,
5817 canonical sampler/device decodes,1127 explicit host diagnostics and
6944 deferred reads. These span all tests and requests, not isolated benchmark
deltas. The next standalone mesh opened successfully without reset. Native
interpreter-exit binding diagnostics persist; no live serving process remains.

The independent stage reviewer required numerical logprob reproducibility and
controls for the constrained greeting outputs. Both are now complete:
`logit_determinism.md`, `logit_determinism_standalone.json/log` (all9 complete
vocabulary comparisons exact across rows0/1/31; all11 API comparisons exact;
cleanup true), and `AUTODEBUG_bad_words_distribution.md` plus raw/summary JSON.
The numerical control uses distinct meaningful131/65-token prose. It is not
an exact full-logit comparison of the greeting continuations. Both banned
seed2/3 outputs reproduce their original98/100 raw IDs exactly; unbanned
controls greet in English. Raw top20 maps at every prefix step0..7 agree
exactly before the mask removes ID23066 and the branches diverge. At least
97.39995128% of probability mass is removed in the fresh-prefix control.
Chinese/off-topic restricted outputs remain poor task-quality cases. Rare
fresh-prefill/decode logprobs differ by up to0.2500639nats; no crossmode
bit-equality is asserted for that probe.

A first B1 launch (`full_b1_primary.command.json`, hostcompatOFF, native262144,
asyncON) exposed cold startup overhead, so it is not the final headline yet.
Both exact shared-runner128/128/1/concurrency1/temperature0 runs completed0.
`b1_cold_first/`: TTFT241.559ms, TPOT20.846ms, medianITL11.358ms, P99ITL11.804ms,
output44.301tok/s. `b1_warm_repeat/`: TTFT51.064ms, TPOT11.374775ms,
medianITL11.359064ms/P9911.657535ms, output85.565tok/s, derived87.913829t/s/u.
Mean TPOT's cold excess totals1202.902ms over127 intervals; the original raw
JSON has no interval vector, so the outlier index is unknown. The benchmark
performs no initial request or hidden warmup with its current defaults.
Fresh source-only `AUTODEBUG_b1_latency.md` identifies unexercised first-decode
integer merge kernels followed by program-count-triggered trace recapture;
causal timing verification is the next AutoFix experiment. No speculative
runtime edit has been kept. The B1 server stopped0 at14:46:57UTC. Lifetime
counters:254 model/sampler/deferred replays,0 host decodes,6 token/position/
RoPE/page-table uploads over startup and two requests. Thus this is not a
hidden host-sampling or recurring token-feedback fallback.

Startup AutoFix is now proven: `b1_startup_before.json/log` isolates8 new
width-1 integer-merge programs and subsequent capture in the first decode;
`b1_startup_after.json/log` has zero first/repeat request program growth or
capture and exact token parity. First reduced decode submission18.1025→
0.8138ms. Explicit clean-state checks cover tokens/current/RoPE, seed tensor,
empty prompt/collector state, row flags and every recurrent/conv replica.
Both the actual startup native guard (`b1_startup_after_tracker.json/log`)
and final B3 stale/page/remap guard (`adapter_device_tracker_startup_final.json/log`)
pass0 with cleanup. Final actual adapter host tests5/5 pass. The44 source/CPU
controls and before-fix sensitivity are recorded in `AUTOFIX_b1_latency.md`.
`warmup_source_scope.json` proves only warmup_model_prefill plus the ttnn
import changed; all other adapter methods and generator/model remain unchanged.

Final B1 launch: `USER=hous ../state/serving-env/bin/python
models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/run_server.py
--label full_b1_startup_final --max-num-seqs 1 --async-scheduling
--sampling-profile full`. Host compatibility isOFF. Typed shared flags remain
max_model_len262144, meshP150x4, trace100000000, L1small32768, ring8192,
sample_on_device_modeall. The immediate benchmark attachment is:
`USER=hous PYTHONUNBUFFERED=1 ../state/serving-env/bin/python -m models.common.readiness_check.run_vllm_server --stages benchmark --no-benchmark-ci-serving --additional-benchmark-args=--save-detailed --model-dir models/autoports/ornith_ai_ornith_1_5_9b --hf-model /home/hous/dev/ornith-1.5-9b/upstream --mesh-device P150x4 --max-num-seqs 1 --max-model-len 262144 --sampling-profile full --server-url http://localhost:8000`.
No benchmark warmup; --save-detailed only preserves interval metadata.

Primary first-after-readiness128/128/1/concurrency1/greedy0: TTFT62.613ms
P50/P99, TPOT11.401720ms mean/P50/P99, ITL11.363721/11.689564ms P50/P99,
output84.716501tok/s, derived87.706066t/s/u. All127 intervals exist; maximum
19.105876ms at index0. Exact repeat: TTFT50.127480ms, TPOT11.376768ms,
ITL11.363631/11.652903ms P50/P99, output85.604143tok/s, derived87.898422t/s/u;
maximum16.051877ms at index0. Generated texts match exactly. The headline is
the immediate first request, not the faster repeat. Four raw before/after
rows and hashes: `b1_startup_benchmark_comparison.json`; final first/repeat
archives: `b1_final_primary/`, `b1_final_repeat/`. The roughly1.2s cold pause
is removed; reduced timing alone is not claimed to account for every old
millisecond. Final server stopped0 with explicit close15:07:55UTC. Its lifetime
counters (`full_b1_startup_final_counters.json`) show255 model/sampler/deferred
replays =1 startup+254 request decodes,0 host decodes,6 boundary uploads each.

Final B32 startup-fixed server uses `run_server.py --label full_b32_startup_final
--max-num-seqs 32 --async-scheduling --allow-host-sampling --sampling-profile full`.
Ordinary device-sampled131/65-token prose succeeds singly and in [131,65,131]
concurrent slots, exact usage and known four-token prefixes:
`nonaligned_prose_request.py --server-manifest full_b32_startup_final.command.json
--output .../full_b32_startup_final_nonaligned_prose.json` (exit0).
The A continuation is ` and smiled.\nWhat is the main`; B is
` and better able to handle everyday tasks.` These are raw-prose shape probes,
not completed instruction-following tests. Final attached shared stages
sampling,qualitative,benchmark are running with fullprofile; exact log is
`full_b32_startup_final_checks.runner.log`.

## Final serving regression and closure checks — 2026-09-08

The final all32 B32 server uses `full_b32_startup_final.command.json` and the
startup-fixed adapter hash881b38abae6ff278d3f66212e2ab4d8a462ce4ebd42153f65c5497b1d97ad324.
The attached shared runner completed exit0:

```bash
USER=hous PYTHONUNBUFFERED=1 ../state/serving-env/bin/python -m models.common.readiness_check.run_vllm_server \
 --stages sampling,qualitative,benchmark --model-dir models/autoports/ornith_ai_ornith_1_5_9b \
 --hf-model /home/hous/dev/ornith-1.5-9b/upstream --mesh-device P150x4 \
 --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --server-url http://localhost:8000
```

`full_b32_startup_final_checks.runner.log` and canonical `sampling_tests.log`:
72 passed,1 canonical skip,329.38s. `sampling_final_manifest.json` hashes final
source/tests/logs. The final nonaligned device-sampled prose requests131/65 also
passed; `full_b32_startup_final_nonaligned_prose.json` records all responses.
Both supervising agent and independent reviewer read all12 final texts. Greedy
texts match prior controls exactly; sampled haiku miscounts its draft, story
stays in planning, ML wording is awkward, French final is correct despite an
unsupported informality assumption. Thermodynamics reaches all three laws;
Fibonacci remains incomplete. No mechanical corruption or request contamination
was found. This is a bounded coherence/serving-regression verdict, not complete
task accuracy. See updated `qualitative_final_review.md` and its byte-hashed JSON.

Final CI100/100/32, no explicit concurrency limit,max-num-seqs32,temperature0,
ignore EOS,native262144:32/32 complete,3200 output tokens; TTFT P50/P99
2059.653155/2060.785376ms; TPOT mean/P50/P99 75.442401/74.823694/88.482798ms;
ITL P50/P99 67.643981/304.181603ms; output337.942797tokens/s; derived13.255145t/s/u
is secondary only. Raw `readiness_vllm/vllm_ci_serving_result.json` and normalized
`vllm_ci_serving_benchmark.json` retain exact config/command/metrics. Chunked
prefill is disabled; burst admission and overlapping eager prefill affect TPOT.
Final B32single diagnostic128/128/1 lives in `b32_startup_final_primary/`.
Canonical primary artifacts were restored byte-for-byte from `b1_final_primary/`
(raw/log); normalized metadata binds the B1 server and secondary CI separately.
Headline first-request128/128/1,concurrency1,max-num-seqs1: TTFT62.613036ms,
meanTPOT11.401720ms,ITL P50/P99 11.363721/11.689564ms,output84.716501tokens/s,
87.706066t/s/u. Detailed raw result retains all127 intervals.

Server wrapper session4068 and shared runner5743 were reaped exit0. Explicit
mesh close is in `full_b32_startup_final.server.log`; process scan found no
vLLM/EngineCore owners (`full_b32_startup_final_process_cleanup.json`). The first
health invocation used a nonexistent serving-env binary (exit127, no device
operation); corrected `USER=hous timeout 60 tt-smi -ls --local` exited0 and
listed all four chips in `final_device_health.log`. No reset was required.

`MODEL_DIR=models/autoports/ornith_ai_ornith_1_5_9b HF_MODEL=ornith-ai/Ornith-1.5-9B
PATH=/home/hous/dev/ornith-1.5-9b/state/serving-env/bin:$PATH USER=hous bash
.agents/prompts/model_bringup_multigoal/09-vllm.check.sh` exited0
(`stage_check_final.log`); native262144 contract and scope-all degeneracy pass.
Its2048 advisory refers to an earlier CPU registry probe, not served capability.
The explicit `check_degenerate_output.py --hf-model ornith-ai/Ornith-1.5-9B
--missing-artifacts critical --scope vllm` also exited0
(`qualitative_degenerate_final.log`). Final Python hooks pass in
`stage_python_precommit_final.log`; no C++/CMake change, so no build is required.

Byte-sensitive raw logs, the large bad-word distribution JSON, and generated
haiku text are committed in lossless compressed archives with per-member SHA256
in `raw_evidence_manifest.json`. Plain originals remain unchanged locally.
This avoids changing evidence bytes to satisfy text-formatting hooks and the
repository500KiB file-size gate. The separate43.7MB complete-logits tensor stays
at its recorded persistent-state path/SHA (`logit_determinism_standalone.json`).

The staged-file hook audit found16 raw JSON files without final newlines.
Their original bytes were restored from the staged index and included in the
same lossless archive; plain copies are ignored. Normalized benchmark summaries
and review manifests remain directly tracked. No runtime source changed.

Final staged-file hooks pass in both repositories: `vllm_stage_candidate_precommit_final.log`
and `vllm_plugin_candidate_precommit_rerun.log`. The plugin formatter removed
one blank line between imports in a test only; `plugin_final_format_scope.json`
proves AST identity and preserves tested/committed hashes. Runtime bytes remain
exactly those in the final server manifests. Raw archive now has208 members.

## Independent stage review

Fresh xhigh reviewer `/root/vllm_stage_review` returned **clean-pass**, with no
required work remaining, in `stage_review.md` on2026-09-08. The supervisor read
the complete report and verified its cited final artifacts. The reviewer
independently read all outputs, reconciled numerical controls and metrics,
checked source provenance and cleanup, and verified all208 raw archive members.
The following local checkpoints contain stage-owned changes only; the two
user-owned AGENTS.md edits remain excluded. No push is authorized or performed.
