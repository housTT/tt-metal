# Stage 9 (vLLM integration) resumed on the p150 profile, 2026 Oct 27

Times are Eastern Time. The autoport passed stages 1 to 10 on the four-chip
p150x4 profile under an older skill version; this run brings stage 9 to
`clean-pass` under the current skill on one Blackhole chip (readiness mesh
label `N150`, model card profile `p150`). Serving on chip 2, generator-level
checks on chip 3. Checkout `/home/hous/dev/tt-metal-wt-ornith`, branch
`hous/ornith-prefill-features`, start HEAD `c0976f241f`.

## Starting state (16:10)

`09-vllm.check.sh` exited 2: `readiness_vllm/` held only the two committed
benchmark summaries (the rest is gitignored and absent in this worktree),
`doc/full_model/serving_contract.json` did not exist, and there was no
prefix-caching or batched-prefill evidence. All existing stage 9 evidence
(`doc/vllm_integration/*`, `readiness_vllm/*.json`) describes p150x4.

## Serving stack and registration (16:20)

Pair: upstream vLLM `0.26.0` wheel in
`/home/hous/dev/ornith-1.5-9b/tt-metal/python_env` plus the standalone
`tenstorrent/vllm-tt-plugin` editable checkout at
`/home/hous/dev/ornith-1.5-9b/vllm-tt-plugin`, local HEAD
`9bab7d527d4adbece91de7338302c99e652fd642` (upstream base `f1593cbb`). The
local plugin commit is recorded in `../plugin_changes.md`; its host suite
passed 474/474 (`plugin_host_tests.log`). Registration goes through the
plugin's `EXTRA_MODELS_DIR` bundle (`vllm_bundle/ornith-1-5-9b/vllm_metadata.json`,
arch `OrnithForCausalLM`, registered as `TTOrnithForCausalLM`) with
`TT_VLLM_BUILTIN_MODELS=0`; `registration_probe.json` shows the resolved class
file is this worktree's `tt/generator_vllm.py`, and every engine subprocess
logs the same `Registered TT model` line in `readiness_vllm/server.log`.
The pinned plugin has no `docs/MODEL_INTERFACE.md`; the capability contract
lives in its `AGENTS.md` section 6, `docs/SCHEDULING.md` and
`docs/DECODE_RELOAD_CONTRACT.md` (all read, listed in the serving contract).

Environment file: `p150_env.sh` (p150 precision policy
`doc/datatype_sweep/p150_head8_hifi2_kv8_policy.json`, pool 262208 tokens,
model-owned prefix cache off, host sampling off by default, `active8` mode
requested, p150 mesh graph descriptor). Launch wrapper: `serve.sh`.

## Serving contract (16:40)

`doc/full_model/serving_contract.json` written for p150 (28 rows). Prefix
caching is `not_supported` with the documented `recurrent_state` exclusion:
24 of 32 layers are Gated DeltaNet linear-attention layers whose recurrent and
convolution state vLLM-owned KV blocks cannot rebuild at `start_pos`. The
server runs with `--no-enable-prefix-caching`. Undocumented key
`supports_batched_prefill` removed from the adapter; every remaining key is in
the plugin's capability table.

## Per-length prefill programs and trace recapture (16:50 to 17:40)

`tests/p150_prefill_shape_probe.py` (chip 3, all 32 layers, B32 pool of 4097
blocks) measured the adapter path: every unseen logical prompt length compiled
new programs (50 for the first length of a new 128-token bucket, 8 for every
further length inside a warmed bucket) and `_ensure_replay_safe` then released
and recaptured the decode and sampler traces. TTFT: 7.7 s (129), 1.0 to 1.5 s
(130, 131, 64) versus 0.09 to 0.10 s for a repeated length
(`prefill_shape_probe_v1.json/log`). That violates the trace lifetime rule
applied by the mixed-shape check.

`tests/p150_prefill_program_attribution_probe.py` (layers 0 and 3) attributed
the eight per-length programs: `ttnn.embedding` and `clone` on `[1, L]` ids,
`ttnn.pad` of the residual to the bucket, the GDN conv-tail `ttnn.slice` at
`[L-3, L)`, and the model's last-row `ttnn.slice` at `[L-1, L)`. A second
chunk at offset 2048 also compiled the RoPE table `slice` and the page-table
`slice` at the block offset (`prefill_program_attribution_v1.json/log`).

Fix (model-owned, `tt/model.py`, `tt/optimized_decoder.py`,
`tt/fused_decoder.py`, `tt/multichip_decoder.py`, `tt/functional_decoder.py`):
the padded-residual chunk path that already existed for TP4/B32 now runs on
every profile; token ids are padded on the host to the 128-token bucket before
upload; a `PrefillPlan` of runtime tensors replaces every offset-keyed slice:
RoPE rows and the conv tail and the last hidden row are `ttnn.embedding` row
gathers, the fill page table is a fixed-width 32-block host upload, and
chunked SDPA takes `chunk_start_idx_tensor`. Lengths 1 and 2 keep the legacy
conv-tail path and are warmed. The adapter warms every bucket (1, 2, 127, 255,
..., 2047, 2175) before the first trace capture (`_warm_prefill_buckets`,
logged as `Ornith prefill bucket warmup`). The generator logs every trace
capture and release with trace ids and program counts.

Results: `prefill_program_attribution_v2.json` shows zero new programs and
zero captures for 128, 129, 130, 131, 200, 2305, 2306, 4353, 5, 2048; lengths
1 and 2 compile 3 and 2 programs once (now warmed). `prefill_shape_probe_v2`
(all 32 layers) reproduces the v1 first and decode tokens exactly for every
length (128: 3/4276, 129: 271/248068, 130: 2113/678, 131: 92/198, 64:
279/10482, repeat 128: 198/9764, repeat 129: 4088/13) with zero new programs
after the first request. `prefill_shape_probe_v3` adds an in-process parity
run of the pre-change path for every length including 2305, 2306 and 4353.

Host tests: the same 21 adapter/prefix-cache host tests fail at HEAD and after
the change (`NameError: name 'os'` and a stub `forward` without the
`reuse_padded_residual` keyword in fixtures that drifted before this stage);
the failing set is identical before and after (`/tmp/head_failed.txt` versus
`/tmp/now_failed.txt`). 46 pass in both.

## Tracked server, first mixed-shape run, and two more recapture sources (18:40 to 19:00)

First tracked B32 server attempt failed at engine start: the runner exports
`HF_MODEL=<--hf-model>` and `reference/hf_reference.py::resolve_model_path`
accepts only the HF id there. The offline cache holds the pinned revision
(`~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-9B/snapshots/489cb979...`,
byte-identical config, index and tokenizer to `/home/hous/dev/ornith-1.5-9b/upstream`),
so the server is launched with `--hf-model ornith-ai/Ornith-1.5-9B`, the model
card's own form (`tracked_b32/runner_attempt1_hf_path_rejected.log`).

Second attempt (`TT_METAL_TRACE_ALLOC_TRACKING=1`, B32, host-sampling
compatibility on) served the whole mixed-shape workload (41 requests, all
complete, greedy repeats identical), but the server log showed two recapture
sources that the generator-level probes had not exercised
(`tracked_b32/server_attempt2_value_keyed_sampler.log`):

1. the first step with seven concurrent rows compiled 49 programs and
   recaptured: `OrnithModel._transfer_slot` writes a freshly prefilled row into
   the batched recurrent state with `slice_write` at the slot index, one
   program set per fixed slot, and warmup had admitted slot 0 only;
2. `OrnithGenerator._sampling_trace_key` was the tuple of every per-row
   sampling value, so the first sampled request released the model and
   sampler traces and recaptured, and the return to greedy did it again
   (captures 4, 5, 6 at 20:46:05 to 20:46:08 UTC). The common sampler's graph
   depends only on whether penalties are active, the logprob configuration
   and `force_argmax` (off for this model); k, p, temperature and seeds are
   device tensors.

Fixes: the adapter warmup admits a one-token prompt into every fixed slot
(`_warm_prefill_buckets`, `slots: 32`); the sampling trace key is the graph
key `(penalties_requested, enable_log_probs, num_logprobs)`; `_capture`
records the sampler and history traces for both penalty variants into
`_sampling_traces` and a live `configure_sampling` refreshes tensors and
selects the prepared pair (`_select_sampling_traces`), releasing only for an
unprepared raw-logprob configuration; `ensure_traces` warms both variants and
the penalized prompt preparation before the first capture.
`prefill_shape_probe_v4` found one remaining program at the first penalized
request (history mask multiply); `prefill_shape_probe_v5` shows zero new
programs and zero captures across greedy, sampled, penalized, greedy, sampled,
greedy with identical greedy outputs. The mixed-shape client's log parser was
also fixed (it had matched nothing because server lines carry an
`(EngineCore pid=...)` prefix; the attempt-2 "pass" is void and kept only as
history).

Known residual: a prompt resumed at a start position that is not a multiple of
128 (vLLM preemption resume only; chunked prefill and prefix caching are off)
runs the leading tokens through decode-mode blocks with per-position slices
and compiles once per position, then recaptures. No served workload in this
stage preempts; the server logs are checked for preemption lines.

## Third recapture source: scheduler slot compaction (19:05 to 19:25)

With the sampler and slot-admission fixes in place, the tracked B32 server
still recaptured once in the mixed-shape workload (41 requests complete,
greedy repeats identical, `tracked_b32/mixed_shape_check_attempt5_one_recapture.json`).
The full-model generator replay of the same phase-1 sequence with 16 decode
steps per request (`prefill_shape_probe_v6`) showed zero growth, so the path
was serving-only. The generator now logs every eager prefill and every program
growth with the calling chain; the relaunched diagnosis server
(`tracked_b32/server_attempt6_diagnosis.log`) shows the growth
`947 -> 996` inside `submit_async_decode/decode_forward` at `slot_remaps: 1`:
the plugin's first slot compaction. `OrnithGenerator._remap_serving_rows`
snapshots each moved row with `ttnn.slice` at the source slot, one program per
source slot and tensor shape. Fix: the adapter warmup applies two full
rotations through `remap_serving_slots` before the first capture, so every
source-slot slice program exists at capture time (`remap_warmed: true` in the
warmup log). The startup sequence keeps one deferred growth of five programs
in the first warmup decode (`942 -> 947`, inside `warmup_model_prefill`), a
startup capture, not a request-time one.

## B32 compatibility server: sampling, qualitative, capacity, burst (17:30 to 18:00)

Server `b32_compat` (untracked, `ORNITH_VLLM_ALLOW_HOST_SAMPLING=1`, B32,
async scheduling, prefix caching off). Shared runner stages attached with
`--server-url`:

- `sampling` (`--sampling-profile full`, `--max-num-seqs 32`):
  `readiness_vllm/sampling_tests.log`, 73 passed, 1 failed, 4 skipped in
  693 s. Skips: the three `test_chunked_prefill` needles (chunked prefill is
  not declared, budget 0) and `test_chat_logprobs_all_vocab` (all-vocabulary
  logprobs unsupported). Failure: `test_tt_penalties::test_frequency_penalty_changes_output`
  asserts that `frequency_penalty=2.0` changes the greedy output of all eight
  prompts; seven changed, `After a while, she` did not.
  Control (`b32_compat/penalty_margin_probe.json`): the device output under the
  penalty equals the penalized argmax of the raw top-20 logits at every step
  (raw logits from a host-compatibility prefill of prompt plus generated prefix).
  The repeated tokens are ` to` at steps 9 and 12 (counts 1 and 2, penalized by
  2.0 and 4.0 nats, still 3.25 nats ahead of the runner-up) and ` that` at step
  23 (count 1, 7.6 nats ahead); step 13 is an exact raw tie (` get` and ` do` at
  -3.242) decided by tie-breaking. The penalty reaches the device sampler and
  the output is legitimately unchanged for this prompt under the p150 policy.
  `frequency_penalty` above 2.0 is rejected by the API, so no stronger control
  exists. The test's premise, that an unbounded `count * F` must move every
  prompt, does not hold here; the plugin test was not modified.
- `qualitative`: `readiness_vllm/vllm_qualitative_outputs.json`, six chat
  prompts (HF chat template, thinking on by default), greedy and sampled, 256
  tokens each. All twelve texts are coherent, on-topic English reasoning in the
  checkpoint's thinking format; no mechanical repetition (most frequent bigram
  "Let me" at most 6 times in 144 words), no gibberish, no wrong-language
  drift (0 CJK characters; the French prompt produces French only inside the
  translation), no cross-request contamination. Thinking consumes most of the
  256-token budget: four greedy and four sampled texts end inside `<think>`;
  the story, French (sampled) and Fibonacci (greedy) reach the answer. The
  haiku reasoning miscounts syllables, as the p150x4 HF control
  (`../qualitative_extended_control_report.md`) also records for this
  checkpoint; the prompts and rendered template match
  `../qualitative_prompt_format.json` and `../qualitative_controls.json`.
- `benchmark` (CI serving burst 100/100/32, greedy, ignore EOS, no explicit
  concurrency): 32/32 complete, TTFT p50/p99 3775/3776 ms, TPOT mean/p99
  171.3/178.3 ms, ITL p50/p99 168.5/245.5 ms, output throughput 154.6 tok/s;
  TPOT-derived 5.84 t/s/u (secondary, 32 rows decoding). The same server's
  128/128/1 diagnostic: TTFT 99.3 ms, TPOT 168.2 ms (the 32-row decode graph
  costs about 4.4 ms per row on one chip; the FINAL_REPORT decode-graph study
  and the 2026 Sep 26 p150 sweep measured the same 168 ms). The headline comes
  from the B1 server below.
- `prefill_burst` (32 x 128, concurrency 32, 32 output tokens): 32/32
  complete, TTFT p50/p99/mean 3762.5/3763.7/3729.8 ms. Prefill is one row per
  pass on this profile, so the serial reference is the same measurement
  (`vllm_prefill_burst_serial_benchmark.json`, `config_matches: true`, ratios
  1.002); the ratio gate is advisory in stage 9.
- untracked mixed-shape run (`b32_compat/mixed_shape_check.json`): 41/41,
  zero captures, TTFT p50/p95/p99 0.381/1.083/1.084 s, ITL max 0.45 s.

One in-workload program growth remained during the sampling suite
(`1121 -> 1122` in a decode after a nine-row admission): the reduced probe
`penalty_history_attribution_v2` attributes it to `ttnn.mul([32, 262144], 0)`
inside `tt_penalties.reset_output_tokens`, the first host-resumed penalized
row rebuilding its output counts. It compiles once; the sibling warm now calls
`sampling.reset_output_state` so the program exists before capture.

## Hardware note: one unintended device-open attempt (18:00)

The device-free host test sweep (`pytest` over the autoport `tests/test_*.py`
files filtered by name) included `tests/test_optimization_experiments.py`,
whose `mesh_device` fixture opens a device. The sweep ran without
`TT_VISIBLE_DEVICES`; the two cases errored at device open with the UMD
message "Expected sysmem to be mapped at NOC address ... another process is
already holding the sysmem NOC address space" (21:58:59 and 22:00:44 UTC), so
no chip was acquired and no running process (chip 2 server, chip 3 probe, or
the other job's chips 0 and 1) was disturbed; the 125 other tests passed. The
attempt itself was a mistake against the device-usage rules for this run and
is recorded here; later host runs select files explicitly.

## Hardware note: a double B1 launch on chip 2 (18:03)

A relaunch chain whose readiness wait matched its own command line reported
nothing, and a second B1 launch was started on top of it: two runners, two API
servers and two `EngineCore` processes existed on chip 2 for about two minutes
and both wrote to `readiness_vllm/server.log` (kept as
`b1_primary/server_attempt2_double_launch.log`). Both were stopped through
SIGTERM to the runners, the remaining API server and engine processes were
terminated, `pgrep` showed no vLLM process, and one B1 server was relaunched.
No reset was needed; chips 0, 1 and 3 were not involved.

## Final evidence, gate and checkpoint (18:30 to 19:10)

B1 primary server (`b1_primary/`, host sampling off, tracking off): the shared
runner's primary 128/128/1 benchmark, first request after readiness:
TTFT 83.30 ms, TPOT 34.09 ms, ITL p50/p99 34.08/34.89 ms, output 29.01 tok/s,
29.34 t/s/u (`readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json`
with 127 intervals); exact repeat 77.38 / 34.10 ms (`b1_primary/repeat/`).
No capture or program growth after startup (`b1_primary/server.log`); the
non-aligned request script also passes on the B1 server
(`b1_primary/nonaligned_requests.json`).

Final B32 compatibility server (`b32_final/`, final sources): sampling 73
passed, 1 controlled failure, 4 skipped (662 s); qualitative outputs
rewritten and reread (same verdict); CI burst 100/100/32: 32/32, TTFT p50/p99
3778.4/3779.8 ms, TPOT mean/p99 171.26/178.26 ms, ITL p50/p99 168.59/245.33 ms,
154.57 tok/s, 5.84 t/s/u (secondary); prefill burst 32x128: 32/32, TTFT p50
3758.7 ms, serial reference ratios 0.998; untracked mixed-shape: 41/41, zero
captures, TTFT p50/p95/p99 0.381/1.080/1.082 s; non-aligned requests pass.
`b32_final/trace_reuse_summary.json`: 294 eager prefills over the server
lifetime, two startup captures, zero workload captures or releases. The
tracked-server evidence of the same workload is `tracked_b32/`.

Stage gate `09-vllm.check.sh`: exit 1 (advisory only): the prefill-burst
TTFT ratios 0.998 exceed the stage 10 thresholds (advisory in stage 9, serial
prefill path) and `runner_compatibility_report.md` from the p150x4 stage
mentions a 2048-token CPU registry probe; degeneracy, context contract and
serving contract pass.

Artifacts: `readiness_vllm/{vllm_benchmark.json, vllm_result.json,
vllm_benchmark.log, vllm_ci_serving_benchmark.json, vllm_ci_serving_result.json,
vllm_prefill_burst_benchmark.json, vllm_prefill_burst_serial_benchmark.json,
vllm_qualitative_outputs.json, sampling_tests.log, vllm_run_summary.json,
server.log}` (gitignored except the normalized summaries) and
`doc/vllm_integration/p150/` (environment, launch wrapper, probes, tracked
and untracked server runs, logit determinism, device tracker probe, trace
reuse summaries, this log and README).
