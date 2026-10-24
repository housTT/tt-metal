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

## 2026 Oct 24 resume: prefix caching and batched prefill (P150)

Resumed stage 10 under skills commit `62814d62` (`/home/hous/dev/skills/plugins/tt-model-bringup`),
worktree `/home/hous/dev/tt-metal-wt-prefill` at `067b6fc879`, serving pair
`/home/hous/dev/vllm` at `f96662a7d` (unchanged). Chip 0 serves; chips 1 and 3 run
generator-level checks, one model construction per process, a fresh
`GEMMA4_TENSOR_CACHE_PATH` per process (a reused cache path stalled one load at
100 percent CPU for five minutes and was terminated with SIGTERM).

### Gate state at resume

`10-optimized-vllm.check.sh` exited 2: `readiness/prefill_resume_check.json` missing,
`readiness_vllm/vllm_prefill_burst_benchmark.json` missing. The previous attempt
recorded that the generator-level resume check failed for multi-token continuations
(top-1 0.28 to 0.66 at L=2560) with both negative controls failing as expected.

### Burst serial reference and before metrics (live per-row server, code 4f3c2b13fa)

- `readiness_vllm/vllm_prefill_burst_serial_benchmark.json` (copy of the stage output,
  also under `readiness_vllm/P150_prefill_features/burst_serial/`): 32 x 128/32 at
  concurrency 32, TTFT p50 3645.5 ms, p99 6712.7 ms, mean 4981.8 ms, 32/32 completed.
  Gate for the batched run: p99 <= 4027.6 ms and mean <= 2490.9 ms.
- `readiness_vllm/optimized_vllm_batched_prefill/before/vllm_benchmark.json` (repo runner
  `models.common.readiness_check.run_vllm_server --stages benchmark`, 128/128/1, C1,
  greedy): TTFT 173.23 ms, TPOT 27.945 ms, ITL p50 27.043 / p99 27.887 ms, 34.384 tok/s,
  decode 35.785 t/s/u.
- `.../before/vllm_ci_serving_benchmark.json` (100/100/32): TTFT mean 10277.4 ms,
  p99 14089.3 ms, TPOT mean 114.288 ms, ITL p50 74.357 / p99 121.513 ms, 148.19 tok/s.

### Resume-check root cause: matmul numerics depend on the physical row count

Bisection artifacts are under `doc/vllm_integration/continuation_bisect/`:

1. `bisect_l2560_layer5.json`: layer 5 (full attention) alone, continuation hidden
   state versus fresh prefill, PCC 0.9994 to 0.9998 at s = 64, 128, 260, 2432. Layer 0
   had passed in the previous attempt. Each layer kind looked correct in isolation.
2. `resume_driver_l05_default.json`, `resume_driver_l05_4096.json`: the real
   `prefill_logits_continuation` driver on layers [0, 5] gives PCC min 0.982, top-1
   0.969 (s=64) and 0.953 (s=2432); identical at `max_seq_len` 4096 and 131072.
   `resume_driver_l0to5.json` (6 layers): top-1 0.94/0.96; `resume_driver_l0to11.json`
   (12 layers): top-1 0.78/0.85, PCC min 0.45. Monotone degradation with depth, so a
   per-layer deviation compounds through the MoE stack; not a cliff at one layer.
3. `resume_driver_l0.json` / `resume_driver_l5.json` (single layers through the
   driver): sliding PCC min 0.993, full PCC min 0.958 to 0.962.
4. `resume_driver_l5_fp32sdpa.json`: a symmetric SDPA compute config (HiFi2, FP32
   accumulate, exact exp) for the fresh and chunked paths changed nothing
   (0.9560 / 0.9599 / 0.9614), so accumulation precision in attention was not it.
5. `resume_driver_l5_forced_chunked2.json`: forcing the fresh full-attention path through
   the same paged chunked SDPA kernel (marker `FORCED_FULL_CHUNKED` confirmed) still gave
   PCC min 0.9598, so the attention kernel was not it either.
6. `sdpa_kernels.json` (op level, random Q/K/V at the real head geometry): non-chunked
   versus paged chunked SDPA at start 0 are bit-exact; chunked at 128-aligned starts with
   the 128/128 program config is bit-exact; the 32/32 config used at non-128-aligned
   starts drifts (row PCC min 0.9994); the sliding history construction is bit-exact when
   the history front is padded to a 128-aligned absolute position and drifts otherwise.
7. `resume_driver_l5_kcache.json`: the K cache written by the continuation differs from
   the fresh K cache in 85 percent of elements for the positions computed by forwards of
   physical length 64, 128, 1024 or 1536, and is bit-identical for forwards of length
   2432, 2496 or 2560.
8. `m_dependence_layer5.json` (per op, same input rows at M = 64 ... 2432 versus 2560):
   embedding, both RMS norms, RoPE and the FP32 router are bit-identical at every M;
   the QKV `ttnn.linear` output differs in 83 percent of elements for M <= 1536, and the
   dense MLP differs for M <= 128. The automatic matmul program config changes its
   K-blocking with M, and with BF16 packer accumulation each block boundary rounds.
9. `tests/probe_linear_m_dependence.py` (op level, QKV shape, BFP8 weights): the output
   depends only on `in0_block_w`; the output subblock shape and M do not matter once
   `in0_block_w` is fixed; FP32 destination accumulation makes the automatically
   configured matmul bit-identical for every M. Timing (`/tmp/probe_linear_timing.py`):
   at M=128 FP32 accumulation costs nothing measurable (QKV 0.151 vs 0.152 ms); at
   M=2048 QKV 0.562 to 1.443 ms and o_proj 0.332 to 1.140 ms per layer, dense gate/up
   0.204 to 0.288 ms, down 0.124 to 0.186 ms.

Remedy (model-owned, no plugin change):

- `tt/optimized_decoder.py`: `PREFILL_FP32_ACC` (env `GEMMA4_PREFILL_FP32_ACC`, default
  on) gives the prefill attention linears and a new `prefill_mlp_compute_config` FP32
  destination accumulation; decode configs are untouched (`_dense_mlp_compute_config`
  selects by phase). Expert sparse matmuls already pin `in0_block_w` and were
  M-independent in the probes.
- `tt/multichip_decoder.py`: `_full_chunked_prefill_attention` front-pads Q to the
  configured SDPA q-chunk (128) instead of the tile, so every continuation uses the
  128/128 grid; `_prefill_sliding_history` front-pads the history with zero rows to a
  k-chunk-aligned absolute position (masked out by the window for every real query).
- `tt/functional_decoder.py`: one `prefill_sdpa_compute_config` (HiFi2, exact exp, FP32
  accumulate, env `GEMMA4_OPT_PREFILL_SDPA_FP32_ACC`) for the fresh, chunked, sliding
  continuation and batched attention calls, so the two arms share one configuration.

Result (`resume_driver_l5_fp32acc.json`, `resume_driver_l5_aligned.json`,
`resume_driver_l0_aligned.json`): the K cache is bit-identical between arms at every
start; layer 5 PCC min 0.9996 to 0.9997 and top-1 1.0 at s = 64, 260, 1029, 2432, 2559;
layer 0 PCC min 0.9997 and top-1 1.0 at s = 64, 260, 1024, 1029, 2432, 2559. The
remaining 3e-4 is the expert-chunk membership of window rows (1024-token MoE chunks
start at the continuation offset).

### Resume check runner change during the run

The installed `readiness_check.run_prefill_resume_check` changed to natural text
from `models/tt_transformers/tests/tale-of-two-cities.txt.bz2` (skills commit
`d51f6767`) while the first full-model run was in flight. The coordinator reported
that the previous prompt source repeats text, which makes the negative controls
stop failing; that result is a check defect. The first run's log is kept as
`readiness/prefill_resume_check_superseded_prompt_source.log` and is not evidence.
The gate artifact `readiness/prefill_resume_check.json` comes from the rerun with
the current runner, launched from the worktree root with
`GEMMA4_TENSOR_CACHE_PATH=/tmp/gemma4_resume_cache`.
The superseded first run already shows the model side of the fix on all 30 layers:
all 14 cases (L = 2560 and 16384; s = 64, 128, 260, 1024, 1029, prefix-cache hit,
last token) pass with PCC mean 1.0000, PCC min 0.9998 to 1.0000 and top-1 1.000;
only its negative controls at the short starts did not fail, for the prompt
reason above (`readiness/prefill_resume_check_superseded_prompt_source.json`).

### Batched prefill: implementation, acceptance and calibration

Design 1 from the prefill-features reference (equal padded length): rows that start at
position 0 are grouped by physical bucket (128, 256, 512, 1024 tokens), the row count is
padded to a power of two, and the group runs once as `[1, 1, B * S, H]` through the
dense work (`MultichipDecoder.prefill_forward_batched` -> `_prefill_residual_block`);
attention reshapes to `[B, heads, S, D]` with one batched SDPA, cache fill stays per row
through the existing `_fill_prefill_cache` (logical length per row, no padding written),
and the terminal gathers each row's last real hidden row with an index-tensor
`ttnn.embedding` lookup so no program depends on the logical length
(`Gemma4FullModel.prefill_forward_batched`, `_terminal_rows`). Continuations
(`start_pos > 0`), rows above 1024 tokens and groups under two rows keep the single-row
path. Env: `GEMMA4_BATCHED_PREFILL` (1/0), `GEMMA4_BATCHED_PREFILL_BUCKETS`,
`GEMMA4_BATCHED_PREFILL_MAX_TOKENS` (4096), `GEMMA4_BATCHED_PREFILL_MIN_ROWS` (2).
`Gemma4Generator.prepare_batched_prefill` compiles all 14 (B, S) variants before any
decode trace; the adapter's `warmup_model_prefill` hook calls it and refuses to run
after a trace exists; a request that arrives unprepared falls back to single rows and
logs `batched_prefill_unprepared` (the lazy variant would have written placeholder K/V
through live page-table rows).

Evidence on the full 30-layer model, chip 3, `max_batch_size` 16, `max_seq_len` 32768
(`readiness_vllm/vllm_batched_equals_solo.json`, log
`readiness_vllm/batched_prefill_acceptance_b16.log`): B in {2, 4, 8, 16}, two row
orders, lengths {5, 31, 32, 33, 127, 128, 129, 257, 640, 1000, 1024} plus two chat
prompts per case; every row top-1 equal, PCC min 0.99998, K and V pages bit-identical
at every real position for a sliding and a full layer, 32-token greedy continuations
(decode with one active row) identical through 8 tokens, status `pass`. B = 32 cannot be
allocated in the generator-owned cache layout on one chip: 32 rows x 1024-token sliding
windows x 25 layers need 6.7 GB and the full-attention state another 2.7 GB at 131072
tokens; the allocator reported 27 MB free for a 268 MB buffer
(`readiness_vllm/batched_prefill_acceptance_b32_oom.log`). Serving covers 32 rows
through the shared vLLM pool; the burst exercises 16-row groups.

Calibration (`readiness_vllm/batched_prefill_calibration.json`, same process, warm):
single-row prefill 128/256/512/1024 tokens and each (B, S) variant versus B single rows.
Batched over serial: 0.92 (2x128), 0.89 (4x128), 0.87 (8x128), 0.87 (16x128), 0.93 to
0.96 at S=256, 0.96 to 0.98 at S=512, 0.99 to 1.00 at S=1024. The dense pass removes
the repeated weight traffic, but this model's prefill is dominated by the MoE expert
path, which computes the per-group expert union (for 128 natural-text tokens with
top-8 routing that is nearly all 128 experts, in both arms), so the per-token cost is
about 1.1 ms in either arm. A 2.5x burst gain needs a token-sorted per-expert prefill
that computes only routed experts; gpt-oss's expert prefill also runs the all-expert
sparsity, so there is no reusable path in the tree. That rewrite is the next lever for
both the burst and single-user TTFT and is outside this attempt.

### Server relaunch on the new code

The first relaunch (13:25 UTC, default `GEMMA4_TENSOR_CACHE_PATH=/tmp/gemma4_full_model_cache`)
stalled on the host while loading layer 18's cached expert tensors (last log line
13:26:32, EngineCore at 100 percent CPU with no device work); the repo runner timed out
after 1200 s and exited, but left `VLLM::EngineCore` pid 184393 and the API server alive
on chip 0. Both were terminated with SIGTERM (the runner's timeout path does not reap
its children; recorded as a runner defect). The second relaunch uses a fresh
`GEMMA4_TENSOR_CACHE_PATH` so the tensor cache is regenerated from the HF weights
rather than read from the shared path; this is the only change to the launch command.
The coordinator's diagnosis: the EngineCore was the only process on
`/tmp/gemma4_full_model_cache`, its read bytes were frozen at about 100 MB, so the
directory itself holds a defective entry (likely truncated by a process killed
mid-write). The directory is left in place for diagnosis. Independent loads from
`/tmp/gemma4_resume_cache` completed. The stalled EngineCore ignored SIGTERM for 100 s
and was ended with SIGKILL at 13:48:47 UTC, after which `/dev/tenstorrent/0` was
released at once and the relaunch (runner pid 254707, cache path
`/tmp/gemma4_serving_cache_<timestamp>`) started at 13:48:47 UTC.

### Expert-sorted MoE prefill

The batched burst on the new server (`readiness_vllm/vllm_prefill_burst_benchmark.json`,
first batched run) gave TTFT p50 2463 ms, p99 4686 ms, mean 3434 ms: ratios 0.698 (p99)
and 0.689 (mean) against the serial reference, above the 0.6 / 0.5 gate, as the
calibration predicted. The scheduler formed 16-row and 15-row batched groups, so the
dense-once path worked; the ceiling was the MoE expert path.

Op-level probe (`tests/probe_sorted_moe.py`, chip 2, random weights at the real shapes,
BFP8 experts, LoFi): with tokens sorted by expert and `sparse_matmul` given a one-hot
expert per group, `nnz = groups`, and a compact `optional_output_tensor`, the gate/up
projection at M=2048 takes 5.83 ms per layer with 128-row groups and 8 groups per call
(24 calls) versus 61.3 ms for the union call (10.5x); 64-row groups 7.26 ms (8.5x);
32-row groups 11.5 ms (5.3x). Wider per-core N with fewer cores per group is slower
(32 groups per call, `per_core_n` 22: 1.3x) and one or two cores per group with
`in0_block_w` 44 overflow L1 (circular buffers 2.44 MB against 1.5 MB). PCC against a
torch reference 0.9996 for every variant. At M=128 the sorted call is not faster
(0.84x to 1.28x depending on geometry), so the union path stays for short prefills.

Implementation (`tt/optimized_decoder.py`): `_moe_prefill_sorted` runs when the physical
row count is at least `GEMMA4_SORTED_MOE_MIN_ROWS` (512) and
`GEMMA4_SORTED_MOE_PREFILL` is on (default). It reads the routing weights back once per
layer, builds the expert-sorted row layout on the host with vectorized torch ops
(`sorted_moe_layout`; a fixed group budget per sequence length,
`sorted_moe_group_budget`, keeps every device shape static for a given length),
gathers the sorted rows on device with one `ttnn.embedding` lookup, runs the gate/up and
down `sparse_matmul`s per call with the union path's program configs (128-row groups,
`in0_block_w` 44 and 11, `per_core_n` 4), and combines by gathering each token's eight
expert outputs in ascending expert order, multiplying by the routing weights and
reducing with `fast_reduce_nc`, the union path's reduction with the zero terms removed.

Layer-level check on real weights (`tests/probe_sorted_moe_layer.py`, layer 5,
`doc/vllm_integration/continuation_bisect/sorted_moe_layer5.json`): the sorted output is
bit-identical to the union output (max abs 0.0, zero differing elements) at S = 512,
1024 and 2048; per-layer MoE time 25.2 to 29.6 ms at 512 (0.85x), 49.9 to 35.4 ms at 1024
(1.41x), 99.9 to 47.7 ms at 2048 (2.1x). The gap to the op-level 10x is the per-layer
host synchronization (routing readback, layout, three uploads) and per-call dispatch
(24 calls x 9 ops at 2048 rows). Since the outputs are bit-identical, the resume and
batched-equals-solo evidence holds by construction; both were rerun regardless.

Serving smoke of the batched path before the MoE change (server relaunched 13:48 UTC on
`09de79d395`): repo runner `--stages sampling --sampling-profile full` passed 72 tests
with the configured all-vocabulary logprobs skip
(`readiness_vllm/P150_prefill_features/sampling_tests.log`, 7 min); the `prefix_cache`
stage passed (`readiness_vllm/prefix_cache_check.json`: TTFT ratio 0.022, long resume
checked with `VLLM_SERVER_DEV_MODE=1`) and its cold greedy text equals the per-row
server's recorded text for the same 2305-token prompt; the qualitative stage completed.
The installed runner's `sampling` stage rejects this serving pair's nested plugin layout
(`Legacy bundled TT plugin detected`), so sampling runs through the repo copy of the
runner. After-benchmarks on that server: primary 128/128/1 TTFT 173.88 ms, TPOT 27.904 ms
(before 173.23 / 27.945); CI burst 100/100/32 TTFT mean 5290 ms, TPOT mean 103.2 ms,
206.4 tok/s (before 10277 ms, 114.3 ms, 148.2 tok/s). These are superseded by the final
measurements after the MoE change below.

Stage profile of the sorted path (`GEMMA4_SORTED_MOE_PROFILE=1`, layer 5, 2048 rows, 24
calls, quiet host): readback 0.26 ms, layout 2.6, gather 1.1, gate/up calls 7.1,
activation 1.2, down calls 3.9, concat and untilize 1.5, combine 2.5; total 20.2 ms
against 99.8 ms for the union path (4.95x), 18.6 ms at 1024 rows (2.69x). Two changes
got it there from the first 61 ms: the per-call output buffers are now allocated with
`ttnn.empty` instead of `ttnn.zeros` (each fill was a device kernel over a 2.9 MB or
5.8 MB buffer, 48 per layer, and dominated both call loops), and the call count comes
from the actual routing rounded up to a multiple of 8 calls instead of the worst-case
budget (24 instead of 32 calls at 2048 rows). Sequence chunks of
`GEMMA4_SORTED_MOE_CHUNK_ROWS` (2048) bound the temporaries; the first full-model resume
run at 16384 rows had exhausted DRAM without that chunking
(`readiness/prefill_resume_check_oom_unchunked.log`). The sorted output stays
bit-identical to the union output at every length measured.

### Final server and serving evidence (code 36dddb9fa6)

Relaunch 14:38 UTC (runner pid 424864, fresh `GEMMA4_TENSOR_CACHE_PATH`, `VLLM_SERVER_DEV_MODE=1`,
all other arguments as in the server launch above); healthy at 14:41:53; batched warmup compiled
the 14 (B, S) variants at startup. Stages ran in sequence on that server after two single and one
paired chat warm request; logs `readiness_vllm/P150_prefill_features/*_runner_final.log`.

- `prefill_burst` (32 x 128/32, concurrency 32, greedy; `readiness_vllm/vllm_prefill_burst_benchmark.json`):
  TTFT p50 1061.6 ms, p99 1907.9 ms, mean 1431.4 ms; serial reference p50 3645.5 / p99 6712.7 /
  mean 4981.8 ms; ratios p99 0.284, mean 0.287, p50 0.291; gate p99 <= 0.6 and mean <= 0.5: pass.
  32 of 32 completed. The first batched run before the MoE change had ratios 0.698 / 0.689 (fail).
- Primary 128/128/1 (`readiness_vllm/optimized_vllm_batched_prefill/after/vllm_benchmark.json`):
  TTFT 173.48 ms (before 173.23, +0.14 percent), TPOT 27.870 ms (before 27.945, -0.27 percent),
  ITL p50 27.025 / p99 27.883 ms, 34.471 tok/s, decode 35.88 t/s/u (before 35.79). Within 5 percent.
- CI burst 100/100/32 (`.../after/vllm_ci_serving_benchmark.json`): TTFT mean 2345.9 / p99 3145.0 ms
  (before 10277.4 / 14089.3), TPOT mean 87.80 / p99 103.80 ms (before 114.29 / 174.93), ITL p50
  78.35 / p99 126.31 ms, 289.85 tok/s (before 148.19). 32 of 32 completed, no missing tokens.
- Qualitative: 6 chat prompts, greedy and sampled, coherent; `check_degenerate_output` (scope vllm)
  reports no degenerate output. The greedy haiku equals the earlier P150x2 greedy output.
- `prefix_cache`: pass; cold 1446 ms, warm 137 to 151 ms (ratio 0.095), warm hit 0.9996, long
  257-token resume checked (first token and 16 greedy tokens identical, hit 0.8993), shared-prefix
  hit rate 0.824. The cold greedy text equals the per-row server's recorded text.
- Sampling (repo runner, full profile): 72 passed, 1 skipped (all-vocabulary logprobs cap), 5 min.
- Mixed-shape trace reuse (`tests/run_vllm_mixed_shape_check.py`,
  `readiness_vllm/trace_reuse/mixed_shape_check.json`): 23 requests completed at lengths 33, 64,
  129, 130, 131, 257, 300, 2200, 3100, 4500 (chunked over the 2048-token budget while short requests
  decoded), one batched prefill group formed, status fail: 13 decode trace captures and 12 releases
  during the workload (reasons: `prefill_program_cache_growth` x 9 for first-seen single-row lengths
  and chunk continuations, `slot_remap` x 2, `batch_shape_change` x 1). TTFT p50 1414 ms, p95 48.6 s,
  p99 66.9 s; the 2200/3100/4500-token prompts and the short requests queued behind them paid
  first-encounter compilation of new single-row chunk programs plus trace recapture. This is the
  trace-lifetime open item the task excluded from this resume; nothing in the batched or sorted paths
  releases traces (their variants are warmed), the releases come from the per-length single-row
  programs and the adapter's width/remap rules recorded in the previous attempt.

Anomaly ledger for the mixed-shape flags:

- Observed anomaly: the greedy 129-token raw prompt produced two texts across batch compositions.
  Evidence: `unseen_129`, `repeat_129_solo`, `repeat_129_solo_again` (solo, batch-1 execution) are
  identical to each other; `repeat_mixed_order_1:r129` and `repeat_mixed_order_2:r129` (three
  concurrent requests, 32-slot lane) are identical to each other; the two texts share the first
  token and diverge at the second. Affected path: decode. Control: the prefill in both cases was the
  single-row path (the three concurrent prompts fell into three different buckets with one row each,
  `prefill_groups` events), so the first-token logits came from the same path; the sampling suite's
  isolation and determinism tests pass within a width. Likely subsystem: the batch-1 versus
  32-slot decode program configs (DRAM-sharded `decode_dram_configs` versus
  `decode_dram_batch32_configs`), which round differently. Investigation: event and text comparison
  above; the decode path was not changed in this resume and the same width rule is documented in the
  previous attempt. Resolution: controlled (pre-existing, width-consistent), recorded as a
  limitation; a width-independent decode config is future work.
- Observed anomaly: `chunked_while_decoding:decode_c` tripped the check's word-repetition heuristic.
  Evidence: the prompt is the check's own corpus paragraph repeated to 129 tokens; the greedy raw
  continuation repeats "the history of human civilization". Control: the chat-formatted qualitative
  suite and `check_degenerate_output` are clean on the same server, and the same prompt's greedy
  text in the solo runs is a different, non-repetitive continuation. Resolution: controlled (raw
  continuation of a repetitive input; not mechanical token doubling).

Trace-reuse adapter probe (`tests/test_vllm_trace_reuse.py`, chip 2, `TT_METAL_TRACE_ALLOC_TRACKING=1`):
the first run reached the assertions with no allocation-tracker error and failed the device
page-table equality check on its very first single-row request (before any batched prefill); two
reruns with the diagnostic print stalled after mesh creation (0 percent CPU for 15 minutes) and were
stopped with SIGTERM. Not reproduced in this attempt; recorded as an open check
(`doc/optimized_vllm/trace_reuse_probe/batched_prefill/`).
