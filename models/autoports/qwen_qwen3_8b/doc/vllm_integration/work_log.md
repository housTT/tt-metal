# Qwen3-8B vLLM integration: work log

Stage 9 of the model bringup for `Qwen/Qwen3-8B` on one Blackhole P300 chip
(1x1 mesh, p150 mesh graph descriptor). Times are Eastern Time (ET) unless
marked UTC; the stage started at 13:26 ET on Oct 30 (cap 23:30 ET).
Worktree `/home/hous/dev/tt-metal-wt-qwen3`, branch `hous/qwen3-8b-autoport`,
start commit `6b86a2cb15`. Every command runs through
`doc/full_model/logs/stage_env.sh <chip> <command>` (the operator's
environment) unless shown in full.

Abbreviations: APC (automatic prefix caching), HF (Hugging Face), KV
(key/value), LM (language model), PCC (Pearson correlation coefficient),
DRAM (device dynamic RAM), TTFT (time to first token), TPOT (time per
output token), ITL (inter-token latency), t/s/u (tokens per second per
user), BF16 (bfloat16), BFP4/BFP8 (block floating point), LoFi (low
fidelity), CQ (command queue).

## 1. Startup (13:26 to 13:40 ET)

- Skills read from disk (`tt-model-bringup` 0.1.21 at
  `/home/hous/dev/skills/plugins/tt-model-bringup`): `vllm-integration` and
  its three references, `tt-device-usage`, `tt-enable-tracing`,
  `qualitative-check`, `stage-review`, `model-bringup` (startup section).
- `TT_AUTODEBUG_ROOT=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6`
  (enabled installation in `~/.claude/plugins/installed_plugins.json`);
  `scripts/environment.py` exits 0 and exports `TT_MODEL_BRINGUP_ROOT`,
  `TT_AUTODEBUG_ROOT` and the runtime `PYTHONPATH` entry.
- Stage 8 documents and work logs read; carried items: a tracked-allocation
  run on the selected policy, the `kv_cache_dtype` argument of the
  `max_tokens_all_users` hook, the `check_propagation` dtype assertions.
- Chip health (13:39 ET): a 32x32 add on chips 0, 1 and 2 completes
  (`logs/mesh_smoke_chip{0,1,2}.log`, `MESH_SMOKE_OK`). Chip 3 is the
  operator's (port 20000) and is never touched.

## 2. Serving stack (recorded deviation)

The operator recorded the serving stack as the deprecated vLLM fork
`/home/hous/dev/vllm` at `f96662a7d239c077e1435b658c0e7d316f9c4799` (branch
`dev`, clean tree) with its nested plugin `plugins/vllm-tt-plugin`, both
installed editable in `/home/hous/dev/tt-metal/python_env`
(`vllm 0.1.dev14202+gf96662a7d.empty`, `vllm_tt_plugin 0.0.0`, both resolved
from `python_env/lib/python3.12/site-packages`). `serving-stack.md` says
not to use this lineage; it is used unchanged on the operator's instruction
and recorded in `doc/full_model/serving_contract.json` (`plugin.note`) and
here. Nothing was installed or checked out. `pip check` could not run (no
`pip` module in the environment; `uv`-managed). Verification
(`logs/serving_stack_check.log`, `logs/registration_check.log`):

- entry points `vllm.general_plugins:tt_model_registry` and
  `vllm.platform_plugins:tt` resolve to `vllm_tt_plugin.entrypoints`;
  `platform_plugin()` returns `vllm_tt_plugin.platform.TTPlatform`;
- the plugin test suite `plugins/vllm-tt-plugin/tests/tt` collects 73 tests
  with the server interpreter;
- the packaged readiness runner rejects this layout for its `serve` and
  `sampling` stages (`_reject_legacy_plugin_source`: the plugin's
  grandparent holds a `vllm` directory), so the server is launched with the
  in-tree runner `models.common.readiness_check.run_vllm_server` (identical
  copy in the worktree and in `/home/hous/dev/tt-metal`), the sampling
  pytest is run by hand with the packaged runner's exact command, and the
  packaged runner's `qualitative`, `benchmark`, `prefix_cache` and
  `prefill_burst` stages attach with `--server-url` and `--server-log`.

## 3. Registration

The pinned plugin has no `TT_MODEL_CLASS_OVERRIDES`; its runtime
registration mechanism is `EXTRA_MODELS_DIR` (`platform.py`
`_iter_extra_model_bundles`), read before the built-in map, which maps
`TTQwen3ForCausalLM` to `models.tt_transformers.tt.generator_vllm:QwenForCausalLM`.
The bundle `models/autoports/qwen_qwen3_8b/vllm/qwen3_8b/vllm_metadata.json`
(`arch: Qwen3ForCausalLM`, `main_class:
models.autoports.qwen_qwen3_8b.tt.generator_vllm:Qwen3ForCausalLM`) is
selected with
`EXTRA_MODELS_DIR=/home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/vllm`
in the server environment, which every engine subprocess inherits. Host
check (`logs/registration_check.log`): `ModelRegistry.models["TTQwen3ForCausalLM"].load_model_cls()`
returns `models.autoports.qwen_qwen3_8b.tt.generator_vllm.Qwen3ForCausalLM`
from the worktree file; the server log shows the same `Registered TT model`
line in the API server process and in `EngineCore_DP0`. No plugin or vLLM
file was changed.

## 4. Adapter design (`tt/generator_vllm.py`, generator changes)

Declared capabilities (every key read by `platform.py` at this pin):
`supports_prefix_caching True`, `supports_async_decode True`,
`supports_sample_on_device True`, `max_device_top_k 32`,
`supports_device_penalties False`, `supports_device_seeded_sampling True`.
`supports_chunked_prefill` is not read by this pin (chunked prefill is
disabled for `model_type qwen3` by `_apply_chunked_prefill_policy`, which
also raises `max_num_batched_tokens` to `max_model_len`) and is not declared.

Delegation: `prefill_forward` and `decode_forward` pass the plugin's
keywords straight to `Qwen3Generator.prefill_forward` / `decode_forward`
with `scratch_block=<num_blocks>` (the extra block the model allocates);
`read_decode_output` / `process_decode_output_host` delegate to the new
generator methods of the same name; `warmup_model_prefill` /
`warmup_model_decode` call `prepare()` and, for `enable_trace=True`, the
generator's `capture_decode_traces`; `get_max_tokens_all_users` calls
`Qwen3Generator.max_tokens_all_users` with the selected config's KV dtype
(stage-8 carry-over) and the live mesh when the model is loaded;
`allocate_kv_cache` calls `Qwen3Model.allocate_kv_cache(num_blocks, dtype)`.
The only adapter-side formatting is `unsqueeze(1)` of host-mode decode
logits (`[B, vocab]` -> `[B, 1, vocab]`, which `_get_output_tokens` indexes
as `tt_out[rows, -1, :]`) and an out-of-vocabulary scan of sampled ids.

Generator changes (`tt/generator.py`):

- `decode_forward(read_from_device=False)` returns a `DecodeOutput`
  (bucket state, mode, device tensor) instead of reading; `read_decode_output`
  issues `tensor.cpu(blocking=False)` plus `ttnn.record_event` on CQ 0 (the
  same queue as the replay, so a later replay that overwrites the persistent
  token tensor is ordered after the read); `process_decode_output_host`
  formats the host tensor and counts the readback.
- `adopt_cache(kv_cache, scratch_block)`: the block pool maps onto the first
  1280 blocks of a caller-owned cache so `prepare()` warms every prefill and
  decode program over the tensors that serve requests (paged-attention
  program keys include the cache shape, see 5.1); `release_owned_cache`
  detaches without freeing a cache it does not own.
- `capture_decode_traces(kv_cache)`: phase 2 of the warmup contract; every
  decode bucket's device-mode decode and sampling traces are captured over
  the scratch block at position 0 and executed once; host-mode traces stay
  deferred to first use (documented deferred first capture).
- `_require_prepared()` at the start of `prefill_forward` and
  `decode_forward`: generator-owned warmup enforcement (vLLM callbacks are
  not the only entrypoint).
- `_redirect_table` vectorized (same result, no per-row Python loop on the
  decode path).
- A per-call `qwen3-vllm prefill ...` line (rows, starts, ends, mode, ms,
  program-cache entries, DRAM allocated) and a per-step
  `qwen3-vllm decode ...` line (rows, bucket, mode, traced, reset_batch,
  deferred read, counter deltas, program-cache entries) are written by the
  adapter through the `vllm` logger.

## 5. Device log

### 5.1 Reduced inner-loop target (chip 0, 13:40 ET)

Launch (`logs/serve.sh 0 8000 7200 QWEN3_VLLM_NUM_LAYERS=2`): the in-tree
runner, `--max-num-seqs 32 --max-model-len 40960 --block-size 32`,
`--tt-config '{"trace_region_size": 536870912}'`,
`--additional-server-args "--enable-prefix-caching --async-scheduling"`,
`VLLM_SERVER_DEV_MODE=1`. Registration resolved to the autoport adapter in
`EngineCore_DP0`; `Automatic prefix caching is enabled`; the propagation
check passed; the KV cache was allocated. `warmup_model_decode(enable_trace=True)`
then died at the first capture: `TT_FATAL mesh_workload.cpp:160:
!is_capturing_trace` inside `_attention_decode` (program compile during
capture). Cause: `prepare()` had compiled the decode programs over the
generator's own small pool (193 blocks) and the capture ran over the vLLM
cache (110704 blocks); the paged-attention program hash includes the cache
tensor shape, so the capture hit a program-cache miss. Fix: `adopt_cache`
(section 4) so warmup and capture use the vLLM cache; the adapter now builds
the generator with `prepare=False` and a one-block pool, adopts the vLLM
cache in `allocate_kv_cache`, warms in `warmup_model_prefill`, captures in
`warmup_model_decode(enable_trace=True)` and detaches the pool afterwards.
The failed server log is `logs/reduced/server_attempt1.log`.

The engine subprocess (PID 1296616) of that attempt did not exit: its last
line is `TTWorker mesh shutdown starting` and it spins one thread (one CPU
minute per minute) with the others in `futex` waits, holding
`/dev/tenstorrent/0`. It is treated as a hung job of this stage: left alone
until 30 minutes after its crash (14:11 ET), then killed, recorded, and
chip 0 verified with a trivial op before any further use (section 5.3).

### 5.2 Generator-level checks (chip 1, 13:41 ET)

`run_prefill_resume_check --lengths 2560,16384 --block-sizes 32` on the
selected policy at default thresholds with `--output-on-pass-only`: status
fail, one case, the known 16352 hit window (top-1 31 of 32 = 0.969, PCC
mean 0.9998, min 0.9995; stage 7's margin probe
`doc/full_model/resume_margin_probe_L16384_s16352_fp32.json` shows the flip
is a near tie where the resumed path agrees with HF); both controls fail at
every window with at least 64 positions; all 16 PCC gates pass. Result kept
as `readiness/prefill_resume_check_stage9_default_thresholds.json`
(`logs/resume_check_default_chip1.log`, exit 1). The lowered-threshold run
(0.90 / 0.93, as stages 6 to 8) to the canonical
`readiness/prefill_resume_check.json` follows (`logs/resume_check_lowered_chip1.log`).

### 5.3 Reduced inner loop, attempt 2 (chip 1, 13:53 ET)

Chip 0 stayed held by the orphan, so the reduced server was relaunched on
chip 1 (`logs/serve.sh 1 8000 7200 QWEN3_VLLM_NUM_LAYERS=2`,
`logs/reduced/serve_runner_attempt2.log`, server log copied to
`logs/reduced/server_attempt2.log`). Healthy after 30 s: `async_scheduling=True`
(`Scheduler class: TTScheduler, async_scheduling=True`), `Automatic prefix
caching is enabled`, `Default vLLM sampling parameters have been overridden
by the model's generation_config.json: temperature 0.6, top_k 20, top_p
0.95` (top_k 20 is inside the 32-candidate device sampler, so the served
default recipe stays on the device sampler; no override is needed),
`prepare()` over the adopted vLLM cache, then
`warmup_model_decode(enable_trace=True): captured {1/device: MeshTraceId(0),
8/device: 2, 16/device: 4, 24/device: 6, 32/device: 8}`.

Checks on this server (mechanics only; a 2-layer model emits gibberish):

- one direct 5-token completion (200 OK);
- `logs/mixed_shape_check.py` (`logs/reduced/mixed_shape_trace_reuse_reduced.json`):
  47 requests, all completed; greedy repeats identical before and after
  sampled and seeded requests; same seed reproduces; decode buckets 1, 8 and
  16 used; 905 device-sampled and 23 host-sampled decode steps; 0 decode or
  sampling trace captures during the workload (the three host-mode decode
  traces were captured on first use in the first pass, a documented deferred
  first capture); DRAM allocated 31600 MB before and after; program-cache
  entries 803 -> 829 (continuation prefill shapes, see below);
- the smoke sampling profile by hand (`logs/run_sampling.sh smoke 8000 32`,
  `logs/reduced/sampling_smoke_reduced.log`): 3 passed, 1 skipped
  (`test_chat_logprobs_all_vocab` skips on one chip).

Stopped at idle with SIGINT to the runner: the engine and API server exited
within 2 s and chip 1 was released (the attempt-1 orphan hung only because
it crashed inside a trace capture).

### 5.4 Tracked reduced server (chip 1, 13:58 ET): the continuation compile finding

`logs/serve.sh 1 8001 3600 QWEN3_VLLM_NUM_LAYERS=2 TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` (`logs/reduced/serve_runner_tracked.log`,
server log `logs/reduced/server_tracked.log`). Warmup and the five captures
passed the tracker. The first request (129 tokens, start 0) ran; its repeat
was a prefix-cache hit (`prefill call=2 rows=1 starts=[128] ends=[129]`,
program cache 667 -> 677) and the next decode replay raised
`RuntimeError: Found 10 device buffer(s) still alive before trace replay`.
The ten buffers are the program-cache entries compiled for that continuation
shape: `EmbeddingsDeviceOperation` (exact-length ids), `TilizeWithValPadding
[1, 32, 4096]`, `FillPad`, `Slice [0,0,128,0]-[1,1,160,128]` (RoPE rows at
the chunk start), `Slice [0,4]-[1,1280]` (the fill page table shifted to the
first block), `PagedFillCache`, `SDPAOperation chunk_start_idx=128` (scalar
chunk start, the kernel also offers `chunk_start_idx_tensor`), `Slice
[0,0,0,0]-[1,1,32,4096]`, `UntilizeCodegen`, `Embeddings`. The engine died;
the runner exited 1 and chip 1 was released.

Classification: a continuation prefill (prefix-cache hit, chunked
continuation or preemption resume) at a start offset or length that
`prepare()` did not warm compiles programs after the decode traces are
captured, and those program-cache allocations are live during the next
replay. The stage-6 serving contract recorded this (`prefill_forward` row:
"continuation rows run eagerly with per-offset programs"; runtime-indexed
prefill is the stage-10 item in `prefill-features.md`). The keys are:
`compute_len = seq_len` for `start_pos > 0` (`Qwen3Model.prefill_hidden`),
the RoPE table slices at `phys_start`, `_pad_sequence` by `lead`, the
per-token lead-row slices and pads, the `fill_table` slice by `first_block`,
`k_fill` slice by `row0`, the scalar `chunk_start_idx` and the SDPA chunk
size chosen from the start. The fix (bucketed continuation lengths, RoPE
rows by runtime gather, host-built shifted fill tables and update-index
tensors, `chunk_start_idx_tensor`, a fixed 32-user lead tensor) touches the
decoder's prefill path for every start and needs the resume, multi-row,
tracked and serving chains again; it is not attempted inside this stage's
cap before the required serving evidence exists (section 9 records the
outcome).

### 5.5 Stage-8 carry-over on chip 2 and chip 0 recovery (13:41 to 14:03 ET)

`QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=stage9_tracked
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
pytest tests/test_full_model.py -k 'split_sampling_trace or host_work_counters
or tracked_replay_workload or batch_decode' --timeout 1800` on the selected
policy (chip 2, pre-fix generator): 7 passed in 1325 s, program cache 666
before and after the post-capture workload, 6 decode and 5 sampling
captures, no tracker error (`logs/pytest_full_model_36_stage9_tracked_prefix_chip2.log`,
artifacts copied to `logs/prefix_code_tracked/`, sidecar
`readiness/readiness_stage9_chip2.json`). Positive control on chip 1
(`logs/positive_control_chip1.log`): raised as expected.

The chip-0 orphan (section 5.1) was killed with SIGKILL at 14:03 ET, 22 min
39 s after its crash, earlier than the 30-minute rule set in 5.1 (its state
had not changed: one spinning thread, `TTWorker mesh shutdown starting` as
the last line). Chip 0 released and a 32x32 add completed
(`logs/mesh_smoke_chip0_after_kill.log`). No reset was needed.

### 5.6 Full-model server, pre-fix code (chip 1, 14:03 ET)

`logs/serve.sh 1 8000 21600` (chip 1 instead of the brief's chip 0, which
was still held at launch time). Healthy after 120 s; `prepare()` over the
adopted vLLM cache; `max_tokens_all_users=113664` from the live free DRAM
(model-computed 114432, stage-6 constants 112320); vLLM allocated 3584
blocks of 32 tokens (114688 tokens with the plugin's 1024-token padding)
plus the scratch block; five device decode and sampling traces captured
before the first request.

Served chain (`logs/evidence_chain.sh 8000`, `logs/evidence_chain_chip1.status`,
14:05 to 14:11 ET; artifacts archived in `prefix_code_served/`):

| Stage | Result |
| --- | --- |
| sampling smoke (hand-run, packaged command) | 3 passed, 1 skipped |
| sampling full (`--tt-max-num-seqs 32`) | 72 passed, 1 skipped (`test_chat_logprobs_all_vocab`, single chip) in 160 s |
| qualitative (chat mode, 6 prompts, greedy + sampled, 256 tokens) | written; degenerate check: no degenerate output |
| benchmark 128/128/1 greedy | TTFT p50 47.8 ms, TPOT mean 20.7 ms, ITL p50 20.7 ms, 48.3 t/s/u, 47.8 tok/s |
| benchmark CI burst 100/100/32 greedy | TTFT p50 1702 ms, TPOT mean 25.0 ms, 771.7 tok/s aggregate, 39.9 t/s/u |
| prefix_cache (2305 tokens) | pass: cold TTFT 354 ms, warm 1018 / 63 / 63 ms (the first warm pays the continuation compile), hit rate 1.00 warm, 0.00 cold; long resume hit rate 0.90 (1560 ms, compile) against cold 353 ms; shared-prefix bench run |
| prefill_burst serial (copied) then with reference | 32/32 completed; TTFT p50 1206 ms, mean 1170 ms; ratios 1.000 (one-row-at-a-time prefill is its own serial reference; advisory in stage 9) |
| mixed-shape check (`readiness_vllm/mixed_shape_trace_reuse.json`) | 47/47 completed; greedy repeats identical across sampled and seeded requests; 0 trace captures; program cache 737 -> 862 (continuation compiles, the 5.4 finding at full scale); TTFT p50 88 ms, p99 1398 ms (first encounters) |

### 5.7 Offset-independent continuation prefill (14:05 to 14:14 ET)

Implemented while the chain ran (files: `tt/model.py`, `tt/functional_decoder.py`,
`tt/fused_decoder.py`, `tt/optimized_decoder.py`, `tt/generator.py`):

- `Qwen3Model.prefill_hidden` pads `start_pos % 32` lead rows of a pad token
  in front of the prompt, starts the physical window at the tile boundary
  below `start_pos`, rounds the window to a prefill bucket for every start
  (continuations were computed at their exact length before) and chunks
  windows longer than 4096 for any start; the decoder receives the aligned
  start plus `valid_starts` / `valid_ends` so only the prompt's rows are
  written; `last_row_index` gives the last token's row for the gather;
- RoPE rows by `ttnn.embedding` over the row-major tables with a host index
  tensor (`_rope_rows`) instead of a slice at the chunk start;
- `chunk_start_idx_tensor` (runtime int32 tensor) instead of the scalar
  `chunk_start_idx`;
- lead rows written through a fixed 32-user pseudo-user tensor gathered by
  runtime index (`_lead_users`) with a host-built one-hot index vector per
  token, instead of per-offset slices and pads;
- the fill page table rolled on the host to the first block at full width
  (`host_page_table` threaded from the generator) instead of a device slice;
- warmup: continuation starts 32, 64, 128, 256 (the SDPA chunk-size
  variants) and 33 (lead rows, fill at row 32) for every bucket.

Probe (`logs/continuation_probe.py`, 2 layers, chip 2, tracker on,
`logs/continuation_probe_L2_chip2.json`): continuation logits against a
fresh 600-token prefill at starts 32, 37, 68, 100, 288, 303, 544, 599: PCC
mean 1.0000, min >= 0.9999, top-1 0.981 to 1.000 over 160-position windows;
after a decode capture, served-entry continuations at starts 45, 77, 150,
289, 992, 1000, 100 (4100 tokens), 4095, 31, 4192 (4200 tokens), 8200 (5000
tokens) beside a fresh row: program cache 475 before and after, greedy
output identical before and after, no tracker error, prepare 32.6 s on 2
layers. An earlier probe pass through the readiness all-rows path
(`prefill_logits_continuation` with every position) still compiled a
concat, an untilize and an embedding for a 4128-row window; serving does
not use that path (it gathers one row of the last piece).

### 5.8 Full-model server, fixed code (chip 1, 14:14 to 14:25 ET)

The pre-fix server was stopped at idle (SIGINT, 2 s) and relaunched with the
fixed generator (`logs/serve_runner_full_chip1_fixed.log`; weights on
device 67.2 s, `prepare()` 56.6 s, healthy after 150 s). Served chain
(`logs/evidence_chain_chip1.status`, 14:17:34 to 14:23:36 ET): see README
sections 1 and 4 for every number. Sampling full: 71 passed, 1 failed
(`test_mixed_params_batch`), 1 skipped. Prefix cache: warm TTFT 64.8 / 63.9
/ 64.9 ms (no first-encounter compile any more), long resume 110.8 ms
against cold 353.5 ms, status pass. Mixed-shape check: program cache 461
before and after, 0 captures, DRAM flat. Benchmark 128/128/1: TTFT p50 49.6
ms, TPOT mean 20.7 ms, 48.3 t/s/u (pre-fix 47.8 ms, 20.7 ms, 48.3). The
final `readiness_vllm/` is archived in `served_final/`.

Resume check with the fixed path (chip 2, 14:14 to 14:20 ET,
`logs/resume_check_{default,lowered}_fixed_chip2.log`): identical numbers
to section 5.2 at every start (PCC 0.9970 / 0.9886 minima, top-1 0.992 /
0.988, the 16352 window 0.969); the canonical
`readiness/prefill_resume_check.json` is the lowered-threshold pass on the
fixed code, `readiness/prefill_resume_check_stage9_default_thresholds.json`
the default-threshold run.

### 5.9 Tracked serving (chip 1, 14:26 to 14:30 ET) and the chip-0 tracked tests

`logs/serve.sh 1 8000 7200 TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` (healthy after 130 s). Workload
(`logs/tracked_served/status`): the smoke sampling subset (2 passed, 1
skipped, `test_mixed_params_batch` failed as in 5.10), the prefix-cache
stage without the shared-prefix bench (pass, warm 64.2 ms, resume 110.5 ms;
`logs/tracked_served/prefix_cache_check.json`) and the full mixed-shape
workload (`logs/tracked_served/mixed_shape_trace_reuse_tracked.json`: 47/47,
program cache 461 -> 461, 0 captures, DRAM 31728 MB flat). No `still
alive` line in `logs/tracked_served/server.log`. The same launch wrapper
and environment raised on the pre-fix code (5.4), so the gate was active in
the engine process; the positive control is `logs/positive_control_chip1.log`.
Host-mode decode traces were captured by the smoke suite's `test_min_p`
before the workload (deferred first capture). Stopped at idle; the
untracked evidence was restored to `readiness_vllm/` from `served_final/`
(md5 identical).

The fixed-code full-model tracked tests on chip 0 (`logs/tracked_generator_chip0.status`,
`logs/pytest_full_model_36_stage9_tracked_chip0.log`, artifacts
`doc/full_model/*.stage9_tracked.json`) are recorded in section 7 when they
finish; the pre-fix artifacts are in `logs/prefix_code_tracked/`.

### 5.10 `test_mixed_params_batch` (14:25 to 14:30 ET)

Failed in the fixed-code full run (`Request 0 ... '3, ' != '135'`), in 2 of
3 standalone reruns (`logs/mixed_params_rerun_{1,2,3}.log`: request 3
`temp=0.01 seed=42` `'100'` vs `'198'`; request 5 `temp=2.0 seed=42`
`'135'` vs `'3, '`) and in the tracked smoke (`'198'` vs `'100'`); passed in
the pre-fix full run and the pre-fix smoke. Directed experiment on the
tracked server: `Letter: ` with `temperature 0.01, seed 42` returns `'198'`
solo three times, at every batch position beside 1, 2, 3 and 5 greedy
fillers (seeded first or last), beside a second request with the same
seed, and solo again afterwards. The test's batch also holds requests with
presence, frequency and repetition penalties; `supports_device_penalties`
is `False`, so every decode step that still contains one of them runs
through vLLM's host sampler for the whole batch. A seeded request then
draws some tokens from the device sampler (position-keyed stream) and some
from its host generator (per-request state), and which stream produced its
first tokens depends on the admission order and on when the penalized
requests finish; the two observed outputs per request are the two streams.
Classification: reproducibility-only (the skill names this test), with
correctness, logprobs, crash-free serving and qualitative output passing;
carried to stage 10 (device penalties would remove the host steps).

## 6. Serving contract and gate (14:27 to 14:31 ET)

`logs/update_serving_contract.py` rewrote 14 rows (README section 4). The
0.1.21 `check_serving_contract.py` treats `evidence` as a path under the
model directory and crashed (`OSError: File name too long`) on the stage-6
`tt_supported_decode_batch_sizes` prose; every supported row now carries
one existing artifact path in `evidence` and its description in `notes`
(`evidence_convention` field). Checker: `Serving contract OK ... 37 rows`,
exit 0.

Gate (`logs/gate_09_vllm.log`): `MODEL_DIR=models/autoports/qwen_qwen3_8b
bash .../09-vllm.check.sh` exit 1: no degenerate output; context contract
OK (40960); serving contract OK; prefill features: advisory for the lowered
resume thresholds (default-threshold run kept beside it) and the burst
ratio 0.999 (stage 10), no critical.

## 7. Commits

| Commit | Content |
| --- | --- |
| `6f4d004043` | `tt/generator_vllm.py`, `vllm/qwen3_8b/vllm_metadata.json`, generator (deferred read, adopt_cache, capture_decode_traces, _require_prepared, continuation warmups), model and decoder continuation changes |

Server logs larger than the repository's 500 KB pre-commit limit
(`readiness_vllm/server.log`, `served_final/server.log`,
`prefix_code_served/server.log`, `logs/reduced/server_attempt2.log`,
`logs/tracked_served/server.log`) are committed as `.log.gz`; the plain
files stay in the working tree, untracked.
| `0ea5639d50` | served evidence (`readiness_vllm/`, `served_final/`, `prefix_code_served/`, tracked serving), resume checks, serving contract rows, README, work log, scripts |
| `0abf55e1e1` | stage `.log` files under 400 KB (the repository ignores `*.log`; force-added as evidence) |

### 7.1 Chip-0 tracked full-model tests on the fixed code (14:14 to 14:37 ET)

`logs/pytest_full_model_36_stage9_tracked_chip0.log`: 8 passed in 1379 s
(split sampling, multi-row prefill equals solo for rows 5, 31, 33, 127,
129, 257, 640 and 1000 with top-1 equal in every case and the rerun and
reordered arms identical, host-work counters, the representative tracked
workload, the four batch buckets), program cache 460 before and after the
post-capture workload, 6 decode and 5 sampling captures, 0 eager decode
steps, no tracker error (`doc/full_model/*.stage9_tracked.json`, sidecar
`readiness/readiness_stage9_chip0.json`). The pre-fix run on chip 2
(section 5.5) is kept in `logs/prefix_code_tracked/`.

## Appendix A. Skill gaps met in this stage

- `vllm-integration/references/serving-stack.md`, "Register the new adapter":
  names `TT_MODEL_CLASS_OVERRIDES=...` as a current mechanism. The pinned
  plugin (`f96662a7d`, the operator's deprecated lineage) has no such
  variable; only `EXTRA_MODELS_DIR` bundles exist (`platform.py`
  `_iter_extra_model_bundles`). The reference should say the mechanism is
  revision-dependent and point at `platform.py` `register_tt_models`.
- `vllm-integration/SKILL.md`, "vLLM Server Integration Test", "To attach
  checks to an existing server": the packaged runner's `sampling` stage
  also calls `_find_plugin_tests_dir()` -> `_reject_legacy_plugin_source`,
  so attaching to an external server still fails on the nested plugin
  layout; the skill says only the local launch rejects legacy sources. The
  sampling pytest had to be run by hand with the runner's exact command.
- `vllm-integration/SKILL.md`, "vLLM Adapter" and `tt-enable-tracing`,
  "Program-Cache Warmup": neither says that paged-attention program keys
  include the KV-cache tensor shape, so a generator that warms over its own
  pool and captures over the vLLM-owned cache (the two cache-ownership modes
  the skill describes) hits a program-cache miss inside the first capture
  (`TT_FATAL !is_capturing_trace`). The warmup must run over the served
  cache; the adapter now adopts the vLLM cache for `prepare()`.
- `vllm-integration/references/prefill-features.md`, "Stage split" against
  "Batched prefill" rules and `tt-enable-tracing` "Trace Allocation Safety
  Gate": the stage split says stage 9 measures the burst "on whatever
  prefill path exists" and leaves runtime-indexed prefill to stage 10, but
  the allocation gate that stage 9 must pass fails on the first
  prefix-cache hit when continuation programs key on the start offset
  (section 5.4). The program-key rules belong to the prefix-caching section
  as a stage-9 requirement.
- `vllm-integration/SKILL.md`, async decode: `supports_async_decode` gates
  overlap, but the plugin only overlaps when vLLM's `--async-scheduling`
  is set (`scheduler_config.async_scheduling`, default `None` on this
  fork); neither the skill nor the packaged runner's launch passes it. The
  flag was added through `--additional-server-args`; without it the async
  split is declared but never exercised.
- `vllm-integration/SKILL.md`: the plugin's `tt_supported_decode_batch_sizes`
  attribute (decode padded to the smallest captured bucket, otherwise to
  `max_num_seqs`) is undocumented in the skills; found in
  `model_runner.py:1085`. Without it every decode step would run at batch 32.
- `scripts/check_serving_contract.py` (0.1.21): `missing_evidence` treats
  the `evidence` string as a path and crashes with `OSError: File name too
  long` above 255 bytes (exit 1 looks like an advisory); the stage-6
  contract written to the 0.1.19 convention (prose evidence) hit it. The
  full-model skill should define the row format (`evidence` path, `notes`
  prose) or the checker should tolerate prose.
- `tt-device-usage/SKILL.md`, "Hangs And tt-triage": an engine subprocess
  orphaned by a crash inside a trace capture never exits (one thread spins
  in mesh shutdown) and has no timeout of its own; the rules (no SIGTERM
  with outstanding device work, no reset) give no step for it. This stage
  waited 22 minutes, killed it with SIGKILL and verified the chip with a
  trivial op, which worked; the skill should say so and give a bound.
- `vllm-integration/SKILL.md`, "Mixed-shape trace reuse check": the runner
  gives no counters, so the adapter had to log per-call and per-step lines
  and a client script parses them; the packaged runner's attached stages
  record `server_command: None` in `vllm_run_summary.json`, so the launch
  line is recorded by hand.
- `qualitative-check/SKILL.md`: the packaged qualitative stage generates
  256 tokens (greedy and sampled T 0.7 / p 0.9); the stage-7 HF control is
  128 greedy tokens on the same prompts, so the divergence comparison
  covers the first 128 tokens only. The skill could name the control
  length the vLLM stage needs.
- Information beyond the skills and the brief: the ttnn SDPA
  `chunk_start_idx_tensor` calling convention (op docstring), `ttnn.embedding`
  as a runtime gather for RoPE and K/V rows, `ttnn.Tensor.cpu(blocking=False)`
  plus `ttnn.record_event` for the async read (from `tt_transformers`), the
  repository's `*.log` ignore rule (evidence logs need `git add -f`) and
  its 500 KB file limit (server logs committed gzipped), and the in-tree
  runner's lack of `--enable-prefix-caching` (passed through
  `--additional-server-args`).
