# Qwen3-8B vLLM integration

Status: `Qwen/Qwen3-8B` serves through the TT vLLM plugin on one Blackhole
P300 chip (1x1 mesh, p150 mesh graph descriptor) from the autoport adapter
`tt/generator_vllm.py`, with the selected stage-8 precision config, prefix
caching on, async decode on and on-device split sampling. The served
greedy path is token-identical to the generator path (128-token controls
on six prompts, three runs each, under async overlap). The stage gate
exits 1 (advisory only: lowered resumed-prefill thresholds with the
default-threshold run beside them, and the batched-prefill TTFT ratio that
stage 10 owns). Stage reviews: section 9. Stages 4 and 5 (multichip) were
skipped by the operator; one chip is the target topology.

Abbreviations: APC (automatic prefix caching), HF (Hugging Face), KV
(key/value), LM (language model), PCC (Pearson correlation coefficient),
DRAM (device dynamic RAM), TTFT (time to first token), TPOT (time per
output token), ITL (inter-token latency), e2e (end to end), t/s/u (tokens
per second per user), BF16 (bfloat16), BFP4/BFP8 (block floating point),
LoFi (low fidelity), CQ (command queue), ET (Eastern Time).

## 1. Headline serving numbers (chip 1, final code, 15:46 to 15:52 ET Oct 30)

Primary single-user profile, `vllm bench serve` 128-token random prompt,
128 output tokens, 1 request, `--max-concurrency 1`, greedy
(`--temperature 0.0`, `generation_config_mode explicit`), `ignore_eos`;
server `--max-num-seqs 32 --max-model-len 40960 --block-size 32`,
`--enable-prefix-caching --async-scheduling`, TT config
`{"sample_on_device_mode": "all", "trace_region_size": 536870912}`
(`readiness_vllm/vllm_benchmark.json`, raw `vllm_result.json`):

| Metric | Value |
| --- | ---: |
| TTFT p50 / p99 | 49.3 ms / 49.3 ms |
| TPOT mean / p99 | 20.7 ms / 20.7 ms |
| ITL p50 / p99 | 20.7 ms / 21.1 ms |
| Decode t/s/u (1000 / mean TPOT) | 48.3 |
| Aggregate output throughput | 47.8 tok/s |
| e2e latency | 2.68 s |

The pre-fix server (same launch, generator before the continuation changes,
`prefix_code_served/vllm_benchmark.json`) measured TTFT p50 47.8 ms, TPOT
mean 20.7 ms, 48.3 t/s/u: single-user latency is unchanged within 5
percent (TTFT +1.5 ms on one request).

Generator-level lower bound (stage 8, teacher forcing, traced decode and
sampler, batch 1): 47.24 t/s/u, TTFT 48.05 ms (158-token prompt). Serving
decode is 2 percent faster than teacher forcing (the served loop has no
per-token host writes), so no avoidable serving overhead is left in the
measured path: every decode step replays the decode and sampling traces
with no token, position or page-table refresh (section 4).

Secondary CI serving-burst profile (vLLM-nightly shape: 100-token prompts,
100 output tokens, 32 requests, no concurrency cap, greedy, `ignore_eos`;
`readiness_vllm/vllm_ci_serving_benchmark.json`): TTFT p50 1222 ms / p99
1223 ms, TPOT mean 25.1 ms / p99 32.8 ms, 871.9 tok/s aggregate, 39.9
t/s/u from mean TPOT, 32/32 completed. Not a headline decode number: burst admission serializes 32
prefills (one row at a time at this stage) and the decode runs at batch 32.

## 2. What was built

- `tt/generator_vllm.py`: `Qwen3ForCausalLM`, the plugin-facing adapter.
  Declares `supports_prefix_caching True`, `supports_async_decode True`,
  `supports_sample_on_device True`, `max_device_top_k 32`,
  `supports_device_penalties False`, `supports_device_seeded_sampling True`
  (every key the pinned plugin reads). Delegates `prefill_forward`,
  `decode_forward`, `read_decode_output`, `process_decode_output_host`,
  `warmup_model_prefill` / `warmup_model_decode`, `allocate_kv_cache` and
  `get_max_tokens_all_users` to `Qwen3Generator` / `Qwen3Model`; the only
  formatting it owns is the `[B, vocab] -> [B, 1, vocab]` unsqueeze of
  host-mode decode logits and an out-of-vocabulary scan of sampled ids. It
  logs one `qwen3-vllm` line per prefill call and per decode step (mode,
  bucket, counters, program-cache entries, DRAM allocation).
- Registration: `vllm/qwen3_8b/vllm_metadata.json` bundle selected with
  `EXTRA_MODELS_DIR` (the pinned plugin's runtime registration mechanism,
  read before its built-in map, which would otherwise route
  `TTQwen3ForCausalLM` to `tt_transformers`); resolved in the API server
  and the engine subprocess (`readiness_vllm/server.log`
  `Registered TT model TTQwen3ForCausalLM -> models.autoports.qwen_qwen3_8b.tt.generator_vllm:Qwen3ForCausalLM`).
- Cache ownership: vLLM owns the paged KV cache (`allocate_kv_cache` ->
  `Qwen3Model.allocate_kv_cache(num_blocks, dtype)` plus the scratch block
  that page-table columns past a request name); the generator maps its
  block pool onto that cache for the two-phase warmup (`adopt_cache`),
  because paged-attention program keys include the cache shape, and
  detaches afterwards. 3584 vLLM blocks of 32 tokens (114688 tokens) from
  the live budget `max_tokens_all_users = 113664` (free DRAM after the
  weights, computed with the selected KV dtype).
- Warmup and traces: phase 1 `prepare()` over the vLLM cache (prefill
  buckets, chunk pieces, continuation starts, multi-row prefill, decode
  buckets in both output modes, sampler precompile; 56.6 s on 36 layers);
  phase 2 `capture_decode_traces` captures the five device-mode decode and
  sampling traces (buckets 1, 8, 16, 24, 32) before the first request;
  host-mode decode traces are captured on first use (deferred first
  capture, three in the sampling suite). No trace is released during
  serving.
- Async decode: `decode_forward(read_from_device=False)` returns the
  device handles, `read_decode_output(async_read=True)` enqueues a
  non-blocking read plus an event on CQ 0, `process_decode_output_host`
  formats; served with `--async-scheduling` (`async_scheduling=True` in the
  engine log, every decode line `deferred_read=True`). Device token and
  position state is authoritative across steady steps; host reloads happen
  on `reset_batch`, the first decode after a prefill and mode changes
  (the plugin drains pending async steps before every reload). The
  page-table redirect uses a host mirror of the device position (loaded on
  a reload, advanced once per step), not the plugin's `start_pos`, which
  can lag the device by one step under overlap (section 5).
- Offset-independent continuation prefill (`tt/model.py`, the three
  decoders, `tt/generator.py`): a prefix-cache hit, chunked continuation or
  preemption resume at any start offset and any length, up to the context
  end, runs through programs that `prepare()` compiled (section 5), which
  the trace allocation gate requires.

## 3. Serving stack and registration record

Operator-recorded deviation: the deprecated vLLM fork `/home/hous/dev/vllm`
at `f96662a7d239c077e1435b658c0e7d316f9c4799` (branch `dev`, clean) with its
nested `plugins/vllm-tt-plugin`, both installed editable in
`/home/hous/dev/tt-metal/python_env` (`vllm 0.1.dev14202+gf96662a7d.empty`,
`vllm_tt_plugin 0.0.0`); `serving-stack.md` names the standalone
`tenstorrent/vllm-tt-plugin` plus upstream vLLM as the default and this
lineage as not to be used. Nothing was installed, checked out or changed in
either checkout (`git status` clean). The packaged readiness runner rejects
this layout for its `serve` and `sampling` stages, so the server runs
through the in-tree `models.common.readiness_check.run_vllm_server --stages
serve` (`logs/serve.sh`), the sampling suite runs by hand with the packaged
runner's exact pytest command (`logs/run_sampling.sh`), and the packaged
`qualitative`, `benchmark`, `prefix_cache` and `prefill_burst` stages
attach with `--server-url` and `--server-log` (`logs/run_checks.sh`).

Server launch line (from `logs/serve_runner_full_chip1_fixed.log`), run
with the stage environment plus `VLLM_SERVER_DEV_MODE=1`,
`EXTRA_MODELS_DIR=.../models/autoports/qwen_qwen3_8b/vllm`,
`MESH_DEVICE=N150`, `TT_VISIBLE_DEVICES=1`:

```text
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --block_size 32 --max_num_seqs 32 --port 8000 --max_model_len 40960 --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 536870912}}' --enable-prefix-caching --async-scheduling
```

Served `max_model_len` 40960 equals `doc/context_contract.json`
(`current_supported_context` 40960, the HF context; no reduction). The
checkpoint's `generation_config.json` (temperature 0.6, top_k 20, top_p
0.95) is applied by vLLM to requests without explicit parameters; top_k 20
is inside the 32-candidate device sampler, so the default recipe stays on
the device sampler (no `--override-generation-config`). The adapter's
out-of-vocabulary scan found 0 invalid ids in every served run.

## 4. Serving evidence (final server, chip 1, 15:46 to 15:52 ET; `logs/evidence_chain_chip1.status`)

| Check | Result | Artifact |
| --- | --- | --- |
| Sampling suite, full profile, `--tt-max-num-seqs 32` (73 tests) | 71 passed, 1 failed, 1 skipped in 163 s. Failed: `test_tt_penalties.py::test_presence_penalty_mixed_batch` (seeded requests split between the device and host samplers, section 6; `test_mixed_params_batch` passed in this run and fails 2 of 3 standalone for the same reason); skipped: `test_chat_logprobs_all_vocab` (single chip). Earlier chains on this stage's code: 72 passed, 1 skipped twice (`prefix_code_served/`, `logs/async_fix_chain1/`) | `readiness_vllm/sampling_tests.log`, `logs/final_server/rerun_*` |
| Qualitative (6 prompts, chat template, greedy and sampled T 0.7 / p 0.9, 256 tokens) | served greedy text token-identical to the same-regime generator control (stage-9 code, chip 1, 128 greedy tokens) on every prompt (divergence index 128 = control length); the generator controls on chips 0 and 1 are identical to each other and to the stage-8 control; first divergence from the HF greedy control at tokens 86, 37, 32, 39, 33, 22 (the generator's own numbers since stage 7); every output coherent, on topic, English, HF-like `<think>` structure, adjacent duplication 0.0 to 0.0048, no control-token leakage, no request contamination; degenerate-output check: none | `readiness_vllm/vllm_qualitative_outputs.json`, `qualitative_verdict.json`, `qualitative_control/chip{0,1}/`, `logs/stage_degenerate.log` |
| Primary benchmark 128/128/1 | section 1 | `readiness_vllm/vllm_benchmark.json` |
| CI serving burst 100/100/32 | section 1 | `readiness_vllm/vllm_ci_serving_benchmark.json` |
| Prefix cache (2305-token prompt, APC counters from `/metrics`) | pass: cold TTFT 356 / 354 ms with 0 hits of 2305 queries; warm 64.6 / 65.3 / 63.5 ms with 2304 hits (rate 1.00); warm/cold TTFT ratio 0.182; first token and logprob identical, greedy identical through 16 tokens; long resume (256 new tokens after the 2305 prefix, `/reset_prefix_cache` available): resumed 110.4 ms with hit rate 0.90 against cold 354.6 ms; shared-prefix bench (4 x 2048 prefixes, 128-token suffixes, 32 requests at concurrency 8): hit rate 0.823, 32/32 completed | `readiness_vllm/prefix_cache_check.json`, `vllm_shared_prefix_benchmark.json`, `vllm_run_summary.json` (`apc_log_enabled true`) |
| Prefill burst (32 x 128-token prompts at concurrency 32, 32 output tokens) | 32/32; TTFT p50 1230 ms, p99 1231 ms, mean 1194 ms; serial reference (same workload, same server, prefix cache reset between): ratios 1.000 (`config_matches true`). Prefill is one row at a time in this stage, so the measurement is its own serial reference; the ratio gate is advisory here and stage 10 owns batched prefill | `readiness_vllm/vllm_prefill_burst_benchmark.json`, `vllm_prefill_burst_serial_benchmark.json` |
| Mixed-shape trace reuse and request lifecycle (`logs/mixed_shape_check.py`) | 47/47 requests: unseen lengths 129, 130, 131, 5, 33, 257, 1025 solo then repeated (all seven repeats identical, the last one a one-token prefix-cache continuation at start 1024); a concurrent burst of 31 to 4097 tokens; prefix-cache resumes at start 992 (200, 1 and 0 new tokens) and a 5000-token prompt while a 400-token decode ran; greedy -> sampled -> greedy -> seeded -> seeded -> greedy on one prompt (greedy identical, same seed identical); the burst reordered with new lengths; logprobs and penalty requests beside a device-sampled one. 32 prefill calls, 30 rows with nonzero starts, 928 decode steps on buckets 1, 8, 16 (905 device-sampled, 23 host-sampled), 34 `reset_batch` steps, 0 decode or sampling trace captures, program-cache entries 444 before and after, DRAM allocated 31747 MB before and after, 0 out-of-vocabulary ids, 0 trace releases; TTFT p50 84 ms, p99 1367 ms (the burst's last rows wait for the serial prefill of the earlier rows), longest decode pause 1354 ms (same cause) | `readiness_vllm/mixed_shape_trace_reuse.json` |
| Non-aligned prompt lengths through serving | 129, 130, 131, 5, 33, 257, 1025, 2049, 4097 and 5000-token completions above; the 2305-token prefix-cache prompt; chat prompts of 15 to 25 tokens; near the context end: prompts of 40000, 40956, 40900, 36869, 38000, 40933 and 39956 tokens (section 5) | same, `logs/tracked_served4/served_experiments.json` |
| Served greedy determinism under async overlap | the rendered prompt ids through `/v1/completions` and the chat path: 128 of 128 tokens identical to the generator control on every prompt, three runs each (`logs/final_server/served_determinism.log`); the same with async scheduling off (`logs/noasync/served_determinism.log`); logprob vectors (top 20) identical run to run and across decode buckets 1, 8, 16, 24 (`logs/tracked_served2/logit_repro_check.json`, maximum difference 0) | those files |
| Tracked serving (`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` in the server environment, same launch, final code) | smoke sampling subset, the prefix-cache stage, the full mixed-shape workload (47/47, program cache 444 before and after, 0 captures, all repeats identical) and the near-context hits with decode replays, with no tracker error (`logs/tracked_served_final/`, work log 9.2); the same environment raised on earlier code three times (`logs/reduced/server_tracked.log`, `logs/tracked_served2/`, `logs/tracked_served3/`), which proves the gate is active in the engine process; positive control `logs/positive_control_chip1.log` raised | `logs/tracked_served_final/` |
| Generator-level tracked gate on the selected policy (stage-8 carry-over) | `test_full_model.py` split-sampling, multi-row, host-work, representative-workload and batch-bucket tests with program-cache allocations included: work log section 5.5 (pre-fix code, chip 2, 7 passed, program cache 666), 7.1 and 9 (fixed code, chip 0, 8 passed including multi-row equals solo, program cache flat, 6 decode and 5 sampling captures, no tracker error) | `logs/pytest_full_model_36_stage9_tracked_*`, `doc/full_model/*.stage9_tracked.json` |
| Resumed prefill on the final code (chip 2) | default thresholds: all 16 PCC gates pass (min 0.9886), both controls fail at every window, top-1 fails only the 16352 hit window (31 of 32, the stage-7 near tie with its margin probe); lowered 0.90 / 0.93: pass at all 8 starts; every number identical to the pre-fix run | `readiness/prefill_resume_check.json`, `prefill_resume_check_stage9_default_thresholds.json` |
| Continuation at the context end (chip 2, full model, tracker on) | starts 40000, 32, 40928, 36869 and 40959 against ends 40959 to 40960: PCC min 0.9975 to 0.9997 against a fresh prefill over the last 64 positions, final top-1 equal everywhere, top-1 1.0 except 0.984 (63 of 64) and 0.968 (30 of 31); ten served-entry continuations after a decode capture (five more unseen shift amounts) at program cache 461 before and after; served on the tracked server with replays: hits at 40000, 40928, 40896, 36864, 37984 and 32 with no tracker error and program cache flat | `logs/context_end_probe_L36_chip2_v4.json`, `logs/tracked_served4/served_experiments.json` |
| Serving contract | 37 rows, 14 updated (async split, prefix caching served, warmup, budget hook, batch 32, non-aligned lengths, context); `evidence` is now one artifact path per row with the prose in `notes` (the 0.1.21 checker treats it as a path) | `doc/full_model/serving_contract.json` |
| Gate `09-vllm.check.sh` | exit 1: degenerate check clean, context and serving contracts OK, advisories only (lowered resume thresholds, burst ratio) | `logs/gate_09_vllm.log` |

Process audit: every server was stopped at idle with SIGINT to the runner
and released its chip within 2 s; the one orphaned `EngineCore` of the
first (crashed) launch was killed after 22 minutes and chip 0 verified with
a trivial op; no vLLM process holds a device at the end of the stage.

## 5. Offset-independent continuation prefill and the async page-table fix

Finding (work log 5.4): the first prefix-cache hit on a tracked reduced
server compiled ten programs after the decode traces were captured and the
next replay check raised `Found 10 device buffer(s) still alive before
trace replay`. Continuation prefill computed at its exact length with
programs keyed on the start offset (RoPE slices, per-token lead-row slices
and pads, the fill page-table slice, the scalar SDPA chunk start). Fix
(work log 5.7, 8.1, 8.7): the window starts at the tile boundary below the
start with pad-token lead rows, rounds to a prefill bucket for every start,
moves its start further down when the bucket would pass the 40960-token
context, and chunks any start; RoPE rows and lead rows are gathered by
runtime index; the fill runs over the whole chunk with a host-built page
table whose leading tiles name the scratch block (passed as an extra
trailing column); SDPA takes `chunk_start_idx_tensor`; `prepare()` warms
continuation starts 32, 64, 128, 256 and 33 for every bucket. Evidence:
the reduced-model probe (2 layers, tracker on): continuation PCC min >=
0.9999 and top-1 >= 0.98 against a fresh prefill at eight starts, program
cache 458 before and after served-entry continuations at eleven
start/length pairs including 4100-, 4200- and 5000-token windows, greedy
identical before and after; the full-model resume check reproduces the
pre-fix numbers to the digit; the context-end probe and the served
near-context hits (section 4); the served mixed-shape workload and the
tracked server show zero program-cache growth. Cost: prepare 47 -> 57 s;
prefix-cache warm TTFT 1018 ms (first encounter) -> 65 ms.

Second finding (work log 8.6): under async overlap the served greedy text
left the generator path at absolute positions 32 and 64 and differed run to
run, while it was token-identical with async scheduling off. The decode
page-table redirect derived the request's valid block count from the
plugin's host position, which can lag the device by one step under
overlap, so at a block boundary the new block's column pointed at the
scratch block for one step. Fix (work log 8.10): the generator keeps a host
mirror of the device position (loaded on a reload, advanced once per
step) and derives the redirect from it; the column count stays exact,
because a plugin row can carry stale block ids past the request's blocks
(an interim position-independent variant made a one-token continuation
diverge, work log 8.10). Evidence: section 4, served greedy determinism.

## 6. Limitations and open items

- `test_presence_penalty_mixed_batch` and `test_mixed_params_batch`
  (seeded requests in batches that also hold penalty requests) fail
  intermittently (2 of 3 standalone reruns each on the final server; the
  full suite failed one and passed the other). Server-log windows show the
  mechanism (work log 9.1): when the batch is admitted as 1 + 31 rows the
  lone first request's prefill is device-sampled (no penalty request in
  that step) and all later steps are host-sampled, so that seeded request
  takes its first draw from the device stream and the rest from its host
  generator, one draw offset from the fifteen requests the host sampler
  served from the start; admitted as 3 + 29 rows every step is host-sampled
  and all sixteen agree. Logits through vLLM are bit-reproducible across
  runs and batch shapes (`logs/tracked_served2/logit_repro_check.json`),
  the plain-client replication of the batch is deterministic, and
  correctness, logprobs, crash-free serving and the qualitative suite pass
  on the same server, so this is the reproducibility-only class the skill
  names. Device penalties (`supports_device_penalties True`) keep every
  seeded request on one stream; carried to stage 10.
- Batched prefill is not implemented (one row per decoder pass); the burst
  TTFT ratio is 0.999 against its own serial reference. Stage 10 owns the
  packed-row path and the ratio gate; the continuation path above keeps
  program keys independent of row position and logical length as that work
  requires.
- Prefill is eager (no prefill trace); `prefill_forward` accepts and
  ignores `enable_trace`.
- The 1D multicast O and down prefill kernels (stage-7 open P2) are still
  not integrated.
- The chunked-prefill scheduler path is disabled by the pinned plugin for
  `model_type qwen3` (`max_num_batched_tokens` raised to 40960); nonzero
  starts in this stage come from prefix-cache hits and the model's own
  4096-row chunking of long windows.
- Reproducibility of logits through vLLM: greedy repeats are identical
  across sampled, seeded and host-sampled interleavings (mixed-shape check)
  and the sampling suite's determinism tests pass; a batch-position logit
  comparison through vLLM was not run separately (the generator-level
  `multi_row_prefill` and `batch_decode` tests cover it).

## 7. Reproduction

From the worktree root:

```bash
D=models/autoports/qwen_qwen3_8b/doc/vllm_integration/logs
$D/serve.sh 1 8000 21600                      # full server on chip 1, port 8000 (background)
$D/evidence_chain.sh 8000 $D/evidence_chain_chip1.status   # sampling, qualitative, benchmark, prefix_cache, prefill_burst x2, mixed-shape, degenerate
$D/serve.sh 1 8000 7200 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0   # tracked server
$D/run_sampling.sh smoke 8000 32 <log>; $D/run_checks.sh 8000 prefix_cache --no-prefix-cache-shared-bench; python $D/mixed_shape_check.py --server-url http://localhost:8000 --tokenizer <snapshot> --server-log readiness_vllm/server.log --output <json>
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh; PY=/home/hous/dev/tt-metal/python_env/bin/python
$S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir models/autoports/qwen_qwen3_8b --mesh-device N150 --lengths 2560,16384 --block-sizes 32 --output models/autoports/qwen_qwen3_8b/readiness/prefill_resume_check.json --output-on-pass-only
$S 2 env QWEN3_FULL_MODEL_LAYERS=2 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY $D/continuation_probe.py $D/continuation_probe_L2_chip2.json
$S 0 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=stage9_tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k 'test_split_sampling_trace or test_multi_row_prefill_equals_solo or test_host_work_counters or test_tracked_replay_workload or test_batch_decode' --timeout 1800 -p no:cacheprovider
MODEL_DIR=models/autoports/qwen_qwen3_8b bash /home/hous/dev/skills/plugins/tt-model-bringup/prompts/model_bringup_multigoal/09-vllm.check.sh
```

## 8. Artifact index

- `readiness_vllm/`: the final served evidence (fixed-code server):
  `server.log`, `sampling_tests.log`, `vllm_qualitative_outputs.json`,
  `vllm_benchmark.json` + `vllm_result.json`, `vllm_ci_serving_*`,
  `prefix_cache_check.json`, `vllm_shared_prefix_*`,
  `vllm_prefill_burst_*` (+ `_serial_*`), `mixed_shape_trace_reuse.json`,
  `vllm_run_summary.json`. Copy in `served_final/`.
- `prefix_code_served/`: the same chain on the pre-fix generator (before
  and after single-user comparison).
- `logs/tracked_served/`: the tracked server's log, smoke sampling,
  prefix-cache stage and mixed-shape workload.
- `logs/reduced/`: the 2-layer inner-loop servers (attempt 1 crash,
  attempt 2 checks, tracked attempt with the finding).
- `logs/continuation_probe_L2_chip2.{json,log}`, `logs/resume_check_*`,
  `logs/pytest_full_model_36_stage9_tracked_*`, `logs/positive_control_chip1.log`,
  `logs/registration_check.log`, `logs/serving_stack_check.log`,
  `logs/mesh_smoke_*`, `logs/gate_09_vllm.log`, `logs/mixed_params_rerun_*.log`.
- `qualitative_verdict.json`; `readiness/readiness_stage9_chip{0,1,2}.json`
  provenance sidecars; `doc/full_model/*.stage9_tracked.json`.
- Scripts: `logs/serve.sh`, `run_sampling.sh`, `run_checks.sh`,
  `evidence_chain.sh`, `mixed_shape_check.py`, `continuation_probe.py`,
  `update_serving_contract.py`, `write_sidecar.py`.

## 9. Stage review

Recorded in `work_log.md` section 8 when the review returns.
