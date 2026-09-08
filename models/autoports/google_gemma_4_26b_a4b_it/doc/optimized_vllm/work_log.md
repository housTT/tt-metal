# Optimized vLLM work log

Stage 10: `google/gemma-4-26B-A4B-it`, optimizing the completed integration in place.
Starting tt-metal commit: `6eb04274233`. The starting worktree and sibling vLLM
worktree were clean. Use the selected `selected_canonical_profile_policy` without
changing weight, activation, residual, cache, CCL, or fidelity selection.

## Contract and measurement plan

Separate P150/P150x2/P150x4 measurements use the existing P300C 1/2/4-chip proxies.
Context remains 50,624 / 262,144 / 262,144 respectively, as supported by the
profile-specific physical capacity evidence in `../context_contract.json`.
All servers use 32 scheduler slots, block size 64, async scheduling, on-device
sampling `all`, and Gemma4 tool/reasoning parsers. Primary measurements are greedy
128 input / 128 output / 1 request / concurrency 1. Secondary capacity evidence
is greedy 100 input / 100 output / 32 requests / unbounded client concurrency.
Use `run_profiles.py` to reproduce commands and leave exact argv, source hashes,
configuration, and exit status in each run manifest. No profiler is used.

Before changing runtime code, measure all three original profiles. Compare the
selected candidate against those same runner workloads, including full sampling,
shared prompt-correct qualitative outputs, nonaligned API input, parser checks,
async overlap, and profile-local standalone logit controls. Final completion
requires independent clean stage review and local-only commits.

## Initial device recovery

The first P150 baseline failed in device initialization before model load:
device 0 active Ethernet core 29-25 did not return to base-firmware heartbeat.
Preserved the server and runner logs under `triage/failed_{server,runner}.log`.
The initial CLI syntax attempt used space-separated stages and was corrected to
the runner's documented comma-separated `serve,benchmark` before any launch.

Focused triage returned exit 1 because no inspector runtime directory existed;
the mesh failed before a running model could create it. The leftover EngineCore
exited before a targeted SIGTERM could reach it; no process was forcibly killed.
`timeout 60 tt-smi -ls --local`, `timeout 180 tt-smi -r`, and a second bounded list
each returned exit 0. All four devices remained visible. The bounded 1x1
`open_mesh_device` / `close_mesh_device` smoke returned exit 0 and
`MESH_SMOKE_OK`. No locks were deleted. No reboot was needed. The unchanged
P150 baseline was relaunched. AutoFix's independent initial diagnosis is in
`AUTOTRIAGE.md`; this is infrastructure evidence, not model performance.

## Operation topology audit

| Boundary | Existing measured path | Candidate / action |
| --- | --- | --- |
| Decoder / terminal | Existing optimized 30-layer generator; selected datatype policy, final norm, vocab-sharded LM head, Sampling1D local top-32 and semantic k1 | Preserve inherited optimized math and coherent layouts; compare serving against selected full-model token-out artifacts, not teacher-forcing throughput |
| Token / position / RoPE | Stable device tensors, sampling `tt_out_tok` feedback, device position increments | Preserve nonblocking split traces; verify scheduler changes and intentionally stale host feedback |
| Request boundary | Adapter releases both traces after every prefill despite canonical request-boundary refresh support | AutoFix A/B probe of same-shape request trace reuse, retaining host-compatibility, physical-shape and remap invalidation |
| Hybrid page tables | Compare 30 layer tables on every submission; upload every layer when any table changes | Investigate duplicate shared-group comparisons and copies of unchanged layer tables |
| Readback | Deferred device-token CPU copy and plugin event, then host formatting | Preserve async boundary and minimal token-only read; no benchmark host sampling |
| Multi-request execution | B1 for one active request; canonical padded lane width otherwise | Preserve the correctness-proven larger-batch graph and admission behavior |

No new decoder topology, matmul geometry, precision, or collective candidate is
being ranked using serving timing. Those selected full-model contracts remain
the baseline; this stage targets adapter orchestration and data movement.

## AutoFix experiments

1. Initial Ethernet timeout: independent `AUTOTRIAGE.md` plus unchanged reset /
   smoke / retry verifies recovery without a model change. P150 and P150x2
   subsequently completed both benchmarks. The same timeout recurred on the
   TP2-to-TP4 topology transition; `triage/tp2_to_tp4` retains the failed run,
   bounded list/reset/list (all exit 0), and successful 1x4 ring mesh smoke.
   The failed engine exited before cleanup. No profiler or watcher was involved.
2. Request trace recapture: `AUTODEBUG.md` independently verifies the adapter
   invalidation and the generator's existing scheduler-boundary refresh contract.
   P150 baseline mean TPOT 28.434 ms versus median ITL 26.826 ms suggests a setup
   gap; aggregate timing alone does not locate it. A reduced real-weight adapter
   A/B/A replay probe and same-runner serving comparison will decide this
   candidate. New allocation scopes or math changes are not assumed necessary.

Baseline primary results collected so far (greedy 128/128/1, concurrency 1,
32 scheduler slots): P150 TTFT 276.775 ms, TPOT 28.434 ms, ITL median 26.826 ms,
aggregate output 32.921 tok/s; P150x2 TTFT 264.736 ms, TPOT 23.385 ms, ITL median
21.491 ms, aggregate output 39.568 tok/s. The distinct CI 100/100/32 bursts both
completed 32 requests at 100.375 and 121.923 aggregate tok/s respectively.
Raw/normalized files under `readiness_vllm/optimized_vllm/before/<profile>` are
authoritative; final tables will retain every requested metric.


## Baseline completion and focused failures

All three unchanged profiles completed both benchmarks. P150x4 primary
128/128/1 measured TTFT 266.008 ms, mean TPOT 21.523 ms, median ITL 19.155 ms,
aggregate output 42.669 tok/s; its CI 100/100/32 burst completed all requests at
128.930 aggregate tok/s. `baseline_summary.json` retains all normalized rows.
The actual artifacts now live under `readiness_vllm/<profile>/optimized_vllm/before`;
relative symlinks preserve the original runner-recorded paths without rewriting
raw evidence. The context checker passes with those profile-scoped paths.

The first reduced request-reuse probe exposed a greedy translation bug:
`format_sampling_params` normalized temperature zero before the adapter passed
parameters to the generator, losing its semantic greedy trace key. Restoring
that sentinel passes 21 host contract tests. With that fix, all B1 A/B/A cases
match forced recapture tokens and logits exactly, and retain trace addresses.
The padded two-to-three-row transition then fails allocation tracking on three
new program-cache buffers (Concat, TilizeWithValPadding, FillPad). The diagnostic
rerun reproduces the same failure and records allocation stacks. Its optional
Python referrer scan itself raises a Flask application-context exception;
operation contexts and allocation stacks remain available. No tracker bypass
or new corruption scopes were introduced. See `AUTODEBUG.md` and
`trace_reuse_probe/diagnostic_tp4.log` for the next focused experiment.

Explicit TTWorker shutdown now uses the existing close helper and propagates
cleanup failure; repeated shutdown and partial initialization are covered.
Normal pytest passes all nine cases (`worker_cleanup_pytest.log`), in addition
to the agent's isolated ownership probe. Real server close followed by immediate
mesh reopen remains required before attributing the Ethernet recovery issue to
worker lifecycle.


The cache-growth guard is verified on P150x4: 18 requests (cold candidates,
warmed candidates, explicit-release controls) pass with allocation tracking;
every candidate/control logit delta is exactly zero. Cold 33-to-47 and
B2-to-B3 compilation retires both traces, while warmed A/B/A groups retain
both IDs with zero recaptures after their initial request. Each request
refreshes token/current-position/RoPE once at its scheduler boundary and uses
device feedback for the remaining steps. No new allocation scope, host read,
or synchronization was added to the serving adapter. The cache-entry query
resolves to the host program-cache count in `MeshDeviceImpl`.
`trace_reuse_probe/cache_guard_tp4.{log,xml}` and the adjacent JSON hold evidence.
The expanded host adapter suite passes 29 cases (`host_adapter_tests.log`).


## Watcher fit and recovery

The three-profile watcher command passed P150 (18 request/control cases, exact
logits) but failed before P150x2 model construction: FABRIC_2D Ethernet program
30,880 bytes exceeds the fixed 26,624-byte configuration buffer. The same
minimal 2x2 mesh with `TT_METAL_WATCHER_NOINLINE=1` reduced it to 26,784 bytes,
still 160 bytes over. Its failed-initialization destructor hit the earlier
Ethernet heartbeat timeout and aborted (exit 134); the bounded timeout did not
kill it. No process or lock needed manual cleanup.

The bounded list/reset/list sequence under `triage/watcher_size` returned zero
throughout and exposed all four P300C chips. AutoFix source diagnosis
`AUTODEBUG_watcher.md` identified the smallest supported next candidate:
`TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1
TT_METAL_WATCHER_DISABLE_WAYPOINT=1`. A 2x2 FABRIC_2D open/close smoke passes
with `MESH_SMOKE_OK`; its log reports only WAYPOINT disabled. Assertions, NoC
sanitization, dispatch and Ethernet checks remain enabled. Multi-chip adapter
watcher evidence will use this configuration; waypoint progress breadcrumbs
are absent because of the measured Ethernet code-size limit. The failed and
successful commands are represented by `trace_reuse_probe/watcher*.log`.


## Selective page-table update experiment

`page_table_audit.md` derives six HMA groups of five layers from the actual
25-sliding/5-full layer sequence and plugin expansion. The source-level CPU
control executed the original adapter methods with mock device operations:
unchanged tables upload zero, but one changed sliding group uploads all 30
layer tables. Only five copies are necessary. The candidate preserves 30
distinct device buffers, skips unchanged layer copies, shares immutable host
snapshots only for actual source aliases, and memoizes current/snapshot pairs.

`page_table_host_after.json` tests the actual patched methods: 30 initial,
zero unchanged, five group-change, and one isolated-layer/alias-merge repair
uploads. Alias splitting/merging, equal distinct objects, in-place mutation,
and shorter-input zero padding pass. In the same CPU process and one-thread
7x250 comparison regime, P150 32x791 unchanged equality falls from 240.10 to
51.92 microseconds; TP2/4 32x4096 falls from 1,239.71 to 258.64 microseconds.
These are CPU measurements, not serving latency or inferred per-profile
serving results. The host adapter suite passes all 30 tests after this change.

The inherited P150x2 matmul warning about computed versus provided shard
configuration also occurs 240 times in the unchanged before-server log. No
matmul configuration is modified in this stage; this is inherited op behavior,
not evidence of a new adapter fallback. The final profile gates still apply.


## Router shutdown assertion investigation

The final same-process watcher run reports all three functional regressions
passed (18 cases/profile, exact candidate/control logits), then aborts during
process teardown on `DebugAssertNCriscNOCPacketTagClearedTripped` in the
subordinate Ethernet RISC. It is not a clean watcher result. Preserved watcher
log, kernel map, and failed triage capture under `triage/watcher_shutdown`;
tt-triage could not load inspector state after the process self-aborted.
After bounded reset/list and successful ring mesh smoke, isolated P150x4
reproduces the shutdown failure: 1 functional test passes, watcher stops,
and the Ethernet heartbeat fails during firmware handoff 20 seconds later
(exit 134). Same-process topology switching is therefore not necessary.

AutoTriage source inspection locates the packet-tag assert after fabric
`kernel_main()` returns. Router teardown drains outstanding work but, unlike
other runtime kernels, does not clear sticky NoC transaction tags. A narrowly
scoped kernel cleanup experiment is required; neither the assert nor NoC
sanitizer is disabled. See `AUTOTRIAGE_watcher_shutdown.md` and
`trace_reuse_probe/isolated_watcher_tp4.log`. The later recovery has separate
logs under `triage/isolated_watcher_tp4`.


Router candidate adds only `noc_clear_packet_tags(noc_index)` after final
barriers and ERISC synchronization, before publishing termination. Existing
Blackhole/Wormhole helpers and the fabric-mux precedent establish the owned-NoC
operation; no assertion or traffic-path arithmetic changes. `.github/scripts/copilot-build.sh`
was attempted and returned 1 because this account cannot access the Docker
daemon. The standard C++ build is therefore unverified, as permitted by the
repository's environment-blocker rule. Kernel pre-commit passes. Runtime JIT
compilation of the changed ring router and mesh open/close already passes;
full isolated watcher process-lifetime validation is running separately.


Router cleanup is now verified: isolated P150x4 passes and exits zero, then
the combined P150/P150x2/P150x4 watcher+allocation run passes all three tests
in 101.48 seconds and exits zero with complete UMD teardown. The subsequent
2x2 FABRIC_2D parent plus 1x2 submesh opens/closes without any reset and prints
`MESH_REOPEN_WITHOUT_RESET_OK`. Final artifacts:
`trace_reuse_probe/final_router_fix.{log,xml}`, per-profile JSON in the matching
directory, `router_fix_reopen.log`, and `trace_contract_summary.json`.
The normal build remains unavailable because of Docker access, while runtime
JIT compiled and exercised the modified Ethernet kernel on these meshes.
P150 full after-serving gates start from this fixed default source.


## Explicit benchmark warmup correction

The first P150 candidate primary measured 28.422 ms TPOT versus initial
28.434 ms. Inspection of `vllm/benchmarks/serve.py` then proved the apparent
"initial single prompt test run" is skipped when ready-check timeout is zero;
`num_warmups` also defaults to zero. Thus **all initial before measurements
and the P150 `after` measurement are cold-request evidence**, and must not be
presented as the required warmed comparison. They remain preserved unchanged.

The shared readiness runner already supports `--additional-benchmark-args`.
The driver now passes `--num-warmups 1` explicitly to both workloads. The
required paired measurements will use `before_warmed` / `after_warmed` output
phases, with the original adapter retrieved exactly from commit
`6eb0427423392d7c6a7f87a511be892b8bf677ae` for the warmed baseline. Both sides
use the same corrected shutdown-only router/worker code, selected precision,
server config, workload shape, generation mode, sampling mode, and warmup
count. Source hashes and exact commands record the comparison. Existing cold
results are diagnostic artifacts, not headline optimized metrics.

## P150 serving correctness and clean shutdown

The full candidate P150 run (`readiness_vllm/P150/optimized_vllm/after`)
passes 72 sampling tests with the one inherited all-vocabulary logprobs skip,
feature/tool/reasoning/nonaligned checks, async overlap, standalone-logit
controls, and the shared qualitative suite. Its cold benchmark is diagnostic.
The worker logs explicit shutdown completion; no EngineCore/API/runner process
remains. Immediate separate 2x2 parent / 1x2 submesh open and close succeeds
without reset. `cleanup_evidence.json` records the commands and inherited
nanobind interpreter-exit warning control, with identical baseline/candidate
leak counts. This is not a new device-lifetime failure.

The benchmark client now fails before measurement when any explicit warmup
request fails. Seven isolated host cases verify success/failure handling;
`AUTOFIX_benchmark_warmup.md` records this harness correction. Both warmed
comparison phases use these identical benchmark bytes and require the
`Successful warmup requests: 1/1` log marker for each workload. The CI warmup
is one single request; the CI burst is capacity evidence, not proof of
prewarmed execution for every intermediate batch shape.

## Warmed baseline complete

`python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_before_warmed.py`
completed all three profiles, each with successful 1/1 primary and CI
warmups, complete requested tokens, and server-runner exit 0. See
`before_warmed_driver.log`, `before_warmed_summary.json`, and each recorded
manifest. The original adapter hash is `80fcc940cffaccdde91bb480318015ca9fd25bfc2a23320b51a668ddd6a77f46`.
The helper restored candidate hash `649cd5ab17cc97af777881e6be63baac96ba27f964be59e753b61d8abb0f6325`
in `finally`; `baseline_source_swap.json` records restoration and exit 0.
No serving process remained. No reset was needed across P150, P150x2, P150x4.

Primary greedy 128/128/1 baseline (32 slots):

| Profile | TTFT ms | TPOT ms | Decode t/s/u | Aggregate output tok/s |
| --- | ---: | ---: | ---: | ---: |
| P150 | 211.357789 | 27.607652 | 36.221842 | 34.428935 |
| P150x2 | 167.288966 | 22.749701 | 43.956622 | 41.873459 |
| P150x4 | 145.735050 | 20.746327 | 48.201303 | 46.029395 |

## Final serving evidence and context report fix

All warmed primary and CI pairs completed on all three profiles. `README.md`
contains the complete same-workload tables and selected full-model comparison;
`perf_summary.json` reconciles exact configurations, source hashes, successful
1/1 warmups, requested token counts and current profile-local quality gates.
Primary 128/128/1 decode is 37.244 / 46.443 / 52.058 t/s/u for P150/P150x2/P150x4,
within 0.3% of each selected full-model token-out control. CI 100/100/32 capacity
is essentially unchanged and remains secondary.

Each profile passes 72 sampling tests with the configured all-vocabulary
logprobs skip, plus API/parser/nonaligned/async/logit gates. All 36 shared-suite
outputs were read; all 18 greedy outputs exactly match prior profile-local
controls. Individual qualitative reviews record cap limitations, TP2 coherent
overlap-tail variation under wall-clock admission, and TP4 sampled thermodynamic
efficiency overgeneralization without claiming complete scientific correctness.
The numeric controls remain identical after normalizing artifact paths.

The final TP4 server closed normally. Immediate separate 2x2 parent / 1x2
submesh reopen returned 0 without reset; bounded tt-smi listing returned 0 and
shows chips 0–3; final pgrep returned 1 with empty output (no serving process).
`cleanup_evidence.json`, `final_reopen.log`, `final_device_list.log` and
`final_process_check.log` retain the results. All warmed servers shut down with
explicit worker and UMD completion, with no resets between them. Large closed
logs are losslessly archived; `artifact_compression.json` records original
byte counts and SHA-256 values.

The independent lifetime audit uses each profile’s actual cache blocks,
2,048-token scheduler chunk, rounded trace reserve and retained tensor envelope.
It preserves positive per-device accounting headroom, without claiming a
measured high-water or unchanged peak. The context contract links that audit.

The first final gate exposed a checker false positive for P150 context inside
aggregate stage reports. New tests failed before the bounded report-root fix
and all 13 pass afterward. The P150 review now labels its cap under an explicit
profiles mapping. No hardware cap changed. `AUTOFIX_context_reports.md` records
the exact fix and commands; the full final degeneracy/context gate returns 0
in `final_stage_check_after.log`. Independent final review remains pending.


## Final-review thermodynamics control

The independent review returned one P2 evidence gap: the TP4 sampled efficiency
claim was overgeneralized, and earlier 64-token HF/control answers did not reach
that sentence. The targeted command was:

```bash
python3 models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_vllm/run_thermodynamics_control.py
```

Original and optimized adapters each ran three serial unseeded continuations
from the actual rendered chat prompt plus retained sampled prefix, at 247 input
/ 256 output tokens, temperature 0.7, top-p 0.9 and top-k 32. TP4 policy, mesh,
262144 context, 32 slots and async device-sampling settings remained unchanged.
All three corresponding texts/finish reasons and usage values match exactly.
Both reproduce the overgeneralized always-wasted-heat claim and the molecular
motion simplification. No output was discarded and no runtime fix was justified.
This establishes inherited TT conditional behavior, not HF attribution or
whole-original-trajectory equivalence. AUTODEBUG_thermodynamics.md and
AUTOFIX_thermodynamics.md retain source reasoning and limits. Both servers exited
0 and the source-swap ledger confirms exact candidate restoration.

After the control, a separate 2x2 parent / 1x2 submesh open-close passed without reset (exit 0); bounded tt-smi listing passed (exit 0); pgrep found no serving process (exit 1, empty output). The appended post_review_control_lifecycle in cleanup_evidence.json links all artifacts.


## Independent clean review and local checkpoints

The fresh xhigh independent reviewer returned `clean-pass`, with no required
work, in [STAGE_REVIEW.md](STAGE_REVIEW.md). The final P2 was closed by the
matched thermodynamics continuation control and refreshed cleanup evidence.
`post_review_stage_check.log` records the final gate exit 0; summary reconciliation
also passes after the control. Runtime implementation bytes remain unchanged.

Source and authored documentation passed pre-commit, including whitespace and
end-of-file checks. Immutable raw benchmark/test/log capture bytes retain their
original whitespace and final-newline state because artifact hashes refer to
those exact bytes. For the checkpoint containing raw captures, only
`trailing-whitespace` and `end-of-file-fixer` are skipped at commit time; all
other applicable commit hooks run. No raw capture is reformatted.

The implementation checkpoint SHAs below identify the reviewed changes. A
subsequent tt-metal documentation-only ledger commit records them; its own SHA
is reported in the final handoff to avoid a self-referential commit hash.

| Repository | Branch | Reviewed implementation checkpoint |
| --- | --- | --- |
| `/home/hous/dev/tt-metal` | `hous/gemma-4-26b-a4b-it` | `5cc0391d415f371a69150da91dc74076a179a791` |
| `/home/hous/dev/vllm` | `dev` | `7b24b0e5904dac2f3859d9ab36577dbdfb5b7d55` |

Both checkpoints were created locally after clean-pass; nothing was pushed. The tt-metal checkpoint hook log is `checkpoint_commit.log`. All applicable hooks passed except the two intentionally skipped raw-capture formatting hooks described above.
