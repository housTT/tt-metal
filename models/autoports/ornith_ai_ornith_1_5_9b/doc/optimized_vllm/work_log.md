# Optimized vLLM work log

## Contract and starting state — 2026-09-08

Stage 10 optimizes the completed real TT-plugin serving path in place. Starting
tt-metal commit: `539734fbb6`; completed serving implementation checkpoint:
`68e3006409c4682ea817f1ee017a77a82e5e3a93`. vLLM checkpoint:
`e0d01006121f319e94e3978a00ffabb967eebb11`. The two dirty AGENTS.md files are
user-owned and excluded from stage changes.

Hardware: four Blackhole chips on two P300c boards, TP4 mesh 1x4, software
profile P150x4. Watchdog owns the serialized hardware lane. Initial bounded
`USER=hous timeout 60 tt-smi -ls --local` passed and found all four chips.
No reset was needed. No profiler collection is authorized or attempted.

Preserve selected `head4_lofi_last8_c32_k4_r2`, pinned HF snapshot revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, native context 262144, device split
sampling, async scheduling, and non-aligned logical prompt support. Existing
standalone post-selection generated-token control is 87.971 t/s/u and 29.002 ms TTFT
at prompt128/output128/B1/native262144 on the same TP4 hardware; see
`../datatype_sweep/post_selection_tokenout_v2.json`.

## Measurement plan and baseline

`run_server.py` retains the completed integration launch configuration and
records source hashes and exact expanded commands. Primary uses max-num-seqs1;
secondary capacity server uses max-num-seqs32. Each before/after pair retains
its own identical configuration. Both use sample_on_device_mode=all, greedy
temperature0, async scheduling and no host-sampling compatibility. Context is
262144, trace region100000000, L1 small32768, ring fabric with8192-byte payload.

`benchmark_repeats.py` invokes the shared run_vllm_server benchmark stage once
for warmup, then three times for measured repeats. Primary is128/128/1 with
concurrency1; CI burst is100/100/32 with no explicit concurrency cap. Archive
every raw result, normalized summary, log and command before the next run.
Do not select best-repeat metrics: use median-TTFT repeat and all its metrics,
and retain the other repeats as stability evidence.

The previous readiness directory is preserved in
`integration_readiness_snapshot/` before any new runner writes.

## Initial operation audit

| Boundary | Existing sequence and constraint | Stage action |
| --- | --- | --- |
| Prefill | External KV cache excludes generator-owned prefill trace; eager model plus split first-token sampling | AutoFix source diagnosis and focused external-cache trace experiment |
| Decode | Persistent token embedding, selected32-layer stack, final norm, vocab-sharded LMHead1D | Preserve selected datatype and measured predecessor topology |
| Sampling | Local power-of-two65536 logits, invalid-ID mask, physical top32 per shard,128 gathered candidates, semantic greedy sampler trace | Reuse unchanged; no force-argmax or full-logits benchmark path |
| Feedback | Device token output, device position/RoPE advance, changed-only page upload | Rerun stale-input and async contract tests |
| Readback | Nonblocking CQ0 token shard copy, event; plugin waits and formats afterward | Preserve minimal plugin output contract |
| Batch | Fixed recurrent slots, vLLM physical attention cache, slot remapping | Preserve B32 and non-aligned serving checks |

Detailed decoder geometry/precision/collective candidates remain in completed
optimized-multichip, optimized-full-model and datatype-sweep evidence. This
serving pass does not recollect device-op profiles or change decoder math.

## AutoFix

Fresh source-only xhigh investigator `/root/prefill_diagnosis` is checking the
external-cache prefill trace contract while the unchanged baseline server runs.
Implementation changes require a verified hypothesis and focused correctness
evidence before acceptance.

### Untouched baseline results

Commands (task-local serving interpreter; USER=hous):

```bash
../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/run_server.py --label before_b1 --max-num-seqs 1 --async-scheduling
../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/benchmark_repeats.py --label before_b1_metrics --max-num-seqs 1
../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/run_server.py --label before_b32 --max-num-seqs 32 --async-scheduling
../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/benchmark_repeats.py --label before_b32_metrics --max-num-seqs 32
```

Each server ran alone and was terminated through its shared runner after all
four benchmark invocations. Both runner return codes are0. No EngineCore/server
remained. Exact commands and source hashes are in the per-server command JSON;
`before_sources.json.gz` preserves source bytes. Actual B1 pool is4098 blocks,
not a standalone per-user cache allocation.

| Baseline workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output t/s | Decode t/s/u |
| --- | --- | --- | --- | --- | --- |
| Primary128/128/1, concurrency1, max-num-seqs1 |49.956/49.956 |11.379/11.379 |11.360/11.656 |85.598 |87.884 |
| CI100/100/32, unlimited admission, max-num-seqs32 |2038.402/2039.860 |75.466/88.498 |67.641/305.210 |338.607 |13.251 (secondary) |

All rows use native262144, TP4/P300c, selected datatype, greedy0 and device
sampling. Primary warmed TTFT repeats49.879/49.956/50.635ms; corresponding
decode87.686/87.884/87.713t/s/u. Headline selects median TTFT repeat2.
Every primary and CI benchmark completed the requested tokens.

### Watcher infrastructure qualification

The unchanged reduced adapter contract command with
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_WATCHER=10` failed before model loading
while opening mesh: ACTIVE_ETH program28720 bytes exceeds kernel buffer26624.
The process exited139 during failed-open cleanup. This matches the Optimize
skill's documented Ethernet watcher size failure, not model correctness.
Evidence: `before_async_contract.log` and `.json`.

No live process remained. Bounded `USER=hous timeout 60 tt-smi -ls --local`
returned0 and all four chips. A bounded `timeout 180 tt-smi -r` was then started
before the prescribed retry with `TT_METAL_WATCHER_DISABLE_ETH=1`; recovery
results will be appended. No profiler was enabled.


Recovery completed: reset/list/TP4 ring mesh open-close each exited0; all four
chips returned, MESH_SMOKE_OK. No lock removal or reboot. The resumed unchanged
adapter probe passed all synchronous/deferred-stale/changed-page/remap cases
with worker Watcher10 and native trace-allocation tracking, program caches
included. Exact command adds TT_METAL_WATCHER_DISABLE_ETH=1 to the previous
command and writes before_async_worker_contract.json/.log. Ethernet watcher
instrumentation alone is excluded because its firmware program does not fit;
worker assertions remain enabled. This is baseline qualification, not final
candidate evidence.

### H1 external prefill trace qualification

The external-cache guard prevented the existing prefill trace from serving B1.
The candidate preselects the startup128 family before empty-pool capture and
reuses it with the caller-owned cache. Other valid logical lengths retain eager
model prefill; a live/external shape miss does not replace the resident family
or snapshot the complete caller pool. No tensor math or datatype changed.

`h1_reduced_b1` real vLLM server passed first131,65,and repeated131 requests and
four128/128/1 benchmark runs. These two-layer results are correctness/inner-loop
evidence only. Teardown counters show five prefill replays, three eager shape
misses,530 model/sampler decode replays,530 async reads,and zero host decodes.

Both `h1_external_reduced` and `h1_external_full32` probes passed all18 exact
comparisons: complete logits, recurrent/conv state on all ranks, next decode
logits/tokens/state, changed/repeated token and page inputs, first nonaligned131,
resident128/131, continuation and live shape misses. The native4098-block
external pool and an unreferenced nonzero physical-page sentinel were preserved;
default repeated capture restored caller state and teardown left zero TRACE
bytes. Worker Watcher10/native allocation tracking included program caches.
Full logits are retained as hashed task-state .pt files, outside repository
artifacts. JSON/report receipts and logs record exact source and command.

### H1 isolated serving result and H2 sampler qualification

H1-only full32 B1 same-harness primary128/128/1/native262144 median-TTFT
repeat measured35.041794ms TTFT and87.679935t/s/u versus baseline49.956104ms
and87.884485t/s/u. One warmup plus three repeats all passed. Exact expanded
commands/source hashes are h1_b1.command.json and h1_b1_metrics/*/manifest.json;
all normalized metrics are h1_comparison.json. Server shutdown exited0.

H2 replaces eager serving first-token sampling with the canonical sampling
trace. Persistent logits staging retires eager prefill's transient output
before replay, including recapture that replaces canonical logits. Token/seed
and penalty-history backups predate capture; exact INT32 merges preserve
untouched physical lanes, including full UINT32 seeds. Masks refresh only on
changed admission membership, including across recapture. A late program miss
raises rather than silently falling back to eager sampling.

Reduced real-weight B1 andB32 native-context sampler probes passed all8 cases
each: real public prefill in greedy,seeded,penalties,greedy-again modes plus
exact production-shape eager-control tokens/history/seeds on every rank,
partial/full32 admission, transient/canonical populated logits and forced
recapture. Synthetic logits isolate sampler numerics and are not qualitative
or performance evidence. Native allocation tracking includes program cache;
worker Watcher10 passes; zero TRACE bytes remain after teardown. See
h2_sampling_b1.json and h2_sampling_b32.json, their execution receipts/logs,
and AUTOFIX_sampling_candidate.md. Host regression suite after changed-only
mask patch:83 passed (h2_mask_host.log).

### Reduced serving smoke invocation correction

`h2_reduced_b4` passed nonaligned131/65/repeated/concurrentA-B-A requests and
the mixed-device-parameters/greedy-top1 tests. I mistakenly attached the shared
four-test smoke profile to the no-host benchmark configuration. That profile
also includes host-only min_p: the preexisting explicit compatibility guard
raised ValueError (no fallback was taken), causing EngineCore exit1; logprobs
was skipped by the API capability check. This is a mismatched invocation,
not evidence that min_p is supported by device sampling. The exact first
result1failed/2passed/1skip is h2_reduced_sampling.log and the server error is
h2_reduced_b4.server.log. Runner/worker exited and closed the mesh; no reset.
The corrected reduced smoke uses --allow-host-sampling; final benchmarks keep
that flag disabled. Corrected result to follow.

Corrected reduced shared smoke `h2_reduced_b4_compat_v2` passed3tests/1skip
and shut down0. The all-vocabulary chat-logprob skip is the preexisting API
capability limit, subsequently checked by the final full profile. The first
compat wrapper attempt omitted serve in --stages and exited2 at argument
validation, without launching hardware; v2 uses --stages serve,sampling.
Python hooks found the new CPU-only pytest.raises missing the repository's
explicit allow annotation; added the same documented CPU-only annotation as
neighboring mocked-TTNN tests and reran hooks.

Native-capacity probe first attempt reached successful all32 startup, then its
metadata walker called buffer_address on a TTNN host staging tensor. The
source/API assertion `StorageType::HOST does not support buffer_address`
preceded long-prefill execution. Fixed the probe walker to report device
buffers only; no runtime/model change. Cleanup closed the mesh, exit1; no
watcher hardware fault or reset. Preserve after_native_capacity.json/log;
corrected run uses after_native_capacity_v2.

Source audit found a stale reporting-only field in the inherited adapter probe:
`supports_async_decode_promoted=False` was hardcoded while the actual adapter
capability is True. It did not disable runtime async scheduling. Corrected the
probe to record the actual capability and explicitly state that its direct
adapter invocation does not exercise the vLLM scheduler. Final real-server
--async-scheduling manifests/counters establish plugin use; focused probe rerun
will record the corrected metadata. Runtime source is unchanged.

Native external serving capacity passed full32/B1 at logical262143 and262144,
with the resident128 prefill family and canonical first-token sampling. The
last decode advanced all four position/RoPE tensors from262143 to262144.
No cache identity/address changed and teardown TRACE usage is zero. Evidence:
after_native_capacity_v2.json/.log and execution receipt. New persistent sampler
element storage is 54534400bytes/device; actual startupDRAM is
5151802368bytes/device. context_contract.json records complete DRAM/L1/TRACE views and
unchanged native capability. These instrumented long-window timings are not
comparable to uninstrumented serving benchmarks or prior native timing rows.

The wrapper initially copied readiness/server.log even after argument-validation
failure, so h2_reduced_b4_compat had a duplicate of the preceding server's log.
Removed that misattributed copy (the original h2_reduced_b4 log is retained)
and restricted wrapper archival to runner logs that actually launched a server.
The failed command/runner log remain. Performance invocation semantics unchanged.

Full32 admission follow-up: after canonical-logit staging/recapture, exactly
all32 unique physical rows now replay the sampler and return, skipping unused
backups, masks and integer restoration. Partial/B1 admission is unchanged;
32 duplicate entries do not enter the shortcut. CPU controls cover penalties,
all-rank sampler state, partial/full/partial mask caching and no backup/merge
work at full32. Existing B32 native-tracker device probe is rerun as
h2_sampling_b32_final before final serving measurements. This three-line branch
adds no allocation and is unreachable in B1 native-capacity and full-model
.generate paths; their preceding exact-source receipts remain applicable to
those unchanged paths.

Final full32 admission device probe passed all8 exact cases with worker watcher
and native allocation tracking, including all32 sampler admission. No runtime
fallback or trace leaks. Final focused host regressions:87passed; all Python
hooks pass (final_host.log, python_hooks_final.log). Runtime is selected for
final same-harness B1 andB32 serving measurements.

### Final primary measurement

| Primary workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output t/s | Decode t/s/user |
| --- | --- | --- | --- | --- | --- |
| Before: 128 input /128 output /1 request, concurrency1, max-num-seqs1 | 49.956/49.956 | 11.379/11.379 | 11.360/11.656 | 85.598 | 87.884 |
| After: 128 input /128 output /1 request, concurrency1, max-num-seqs1 | 35.249/35.249 | 11.403/11.403 | 11.365/11.659 | 86.273 | 87.700 |

Exact comparison: primary_comparison.json. Before selects repeat2; after
selects repeat3 by median TTFT, keeping every metric from that request.
Current standalone final_full_model_tokenout.json uses the same selected policy,
full32 layers,128/128/B1/native262144; generated-output decode87.958991t/s/user
and TTFT29.013056ms belong to the same selected run. Its separate plain
nonblocking token-out loop median88.036210t/s is corroboration only.
The final serving decode87.699652t/s/user is0.295% below the generated-output
control. H2 final latency35.248746ms versus H1-only35.041794ms is within
repeat variation; only the overall baseline-to-final TTFT improvement is claimed.

Final B1 shared qualitative stage exited0; all12 texts directly reviewed in
qualitative_review.md/.json. All6greedy texts exactly match the integration
snapshot. Current tokenizer/rendered prompts/tokenIDs match the pinned controls;
initial comparison incorrectly treated Transformers BatchEncoding as a plain
ID list, corrected by encoding rendered text without extra special tokens.
Final vLLM-scope degeneracy check exited0. B1 normal device requests131/65/A-B-A
pass exact previous token prefixes and usage. B1 server closed mesh and exited0.

Final B1 real-server teardown confirms3585 device decode/model trace replays,
3585 plugin async reads,20 prefill calls and20 first-token sampler replays;
zero host decodes and zero generator explicit synchronizations. Scope covers
startup +4benchmark128/128/1 requests +12chat outputs capped256 +3raw131/65
requests capped8, all max-num-seqs1/native262144. Sampling replay total3605
is exactly model replay3585 plus prefill20. This links the traced/device path
to the measured server; it is not a separate timing benchmark. Variable
page-table refreshes reflect real scheduler page admission, not per-token
uploads; focused stale/changed-page tests distinguish the cases.

### Final CI capacity measurement

| CI capacity workload | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Aggregate output t/s | Secondary1000/TPOT t/s/user |
| --- | --- | --- | --- | --- | --- |
| Before:100 input/100 output/32 requests, unlimited admission, max-num-seqs32 | 2038.402/2039.860 | 75.466/88.498 | 67.641/305.210 | 338.607 | 13.251 |
| After:100 input/100 output/32 requests, unlimited admission, max-num-seqs32 | 2012.919/2014.100 | 75.366/88.242 | 67.666/302.043 | 339.863 | 13.269 |

ci_comparison.json selects the median-TTFT repeat independently for before
and after; every metric in each row comes from that repeat. Exact server
argv and runtime environment compare equal within both B1 and B32 pairs.
B32 nonaligned131/65 repeated/concurrentA-B-A requests passed exact prior
token prefixes and usage; server mesh closed and runner exited0.

### Final anomaly classification

Observed: performance/server logs emit allocator.cpp:130's generic warning,
"Allocating device buffers is potentially unsafe due to the existence of an
active trace ... buffers may be corrupted once a trace is executed ... Use the
trace allocation tracker to verify." This warns on allocation, before it knows
whether the temporary is retired before replay. It is not an observed
corruption assertion. Affected boundary is eager shape-miss prefill/sampler
configuration followed by reuse/recapture of persistent traces.

Control/investigation: h1_external_full32.json validates actual32-layer external
pool, changed tokens/pages and logits/state/next-decode equality under native
allocation tracking including program caches. h2_sampling_b1.json and
h2_sampling_b32_final.json validate canonical/transient logits, forced
recapture, partial/full sampler lanes and persistent buffers with the same
tracker. after_native_capacity_v2.json executes full32 native262143/262144
prefill and last decode with resident traces; after_async_worker_contract_v2.json
validates deferred copies and persistent changed/unchanged scheduler inputs.
Every probe closes with no trace-lifetime assertion and zero TRACE allocation
where measured; final serving token/quality controls also pass. Resolution:
controlled generic warning, not a waived observed corruption. Performance
servers intentionally omit tracker instrumentation and all profiling.

The stage checker exits0. Its2048 advisories recurse into prior integration
CPU-registry/compatibility-probe logs; those are not server settings. All current
server manifests/native probes use262144, and context_contract.json records
no capability reduction.

### Final full profile and cleanup

`USER=hous ../state/serving-env/bin/python -m
models.common.readiness_check.run_vllm_server --stages sampling --server-url
http://localhost:8000 --model-dir models/autoports/ornith_ai_ornith_1_5_9b
--hf-model /home/hous/dev/ornith-1.5-9b/upstream --mesh-device P150x4
--max-num-seqs 32 --max-model-len 262144 --sampling-profile full` exited0:
72passed/1expected skip in328.23s (final_sampling.log). The canonical skipped
all-vocabulary chat-logprobs test detects the framework API capability; it is
not a skipped TT sampling correctness failure. final_b32_compat.command.json
records explicit host compatibility; benchmark server flags remain disabled.

Final compatibility runner received SIGTERM after tests, exited0, and closed
its mesh. final_process_cleanup.json finds no vLLM/EngineCore owners. Bounded
`USER=hous timeout 60 tt-smi -ls --local` exited0 and listed all four chips;
no final reset. The stage check exited0 and all134 stage-owned file hooks pass
(stage_precommit.log). Changes are Python/docs only; no C++ build required.

Baseline and final runner processes emit the same nanobind shutdown warning:
984 types/4503 functions remain registered at interpreter exit. Counts match
before_b1/before_b32/after_b1/after_b32/final_b32_compat runner logs, all clean
exit0. This is inherited Python binding teardown, not a new device allocation
leak: actual worker meshes close, native TRACE allocation reaches0 and the
OS owner scan is empty. Resolution: controlled unchanged binding warning.

Byte-sensitive core evidence is losslessly archived in four448KiB-or-smaller
parts; final sampling/cleanup/readiness outputs are in raw_evidence_final.tar.gz.
Both manifests record every member SHA256; part reconstruction and all archived
member hashes were verified. Originals remain locally and are gitignored to
preserve their exact bytes through formatting hooks. README records restoration.
Final stage_check_final.log exits0 with unchanged native262144 capability.

### Independent stage review and checkpoint validation

The fresh xhigh stage-review agent independently inspected the original goal,
applicable skills, implementation, raw benchmark repeats, generated text,
trace/cache/sampling evidence and all 165 archived member hashes. Its final
[STAGE_REVIEW.md](STAGE_REVIEW.md) verdict is **clean-pass**, with no required
work. The review retains the controlled model-quality, instrumentation and
binding-teardown limitations; it makes no broader release-accuracy claim.

Final checkpoint changes are Python, tests and evidence/docs only. No C++ build
is required. The stage-owned file list excludes both user-owned AGENTS.md edits.
The serving dependency checkout has no stage-owned changes. No push is performed.
