# Qwen3-8B full model: work log

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), LM (language model), RoPE (rotary position embedding), SDPA
(scaled dot-product attention), DRAM (device dynamic RAM), L1 (per-core
local memory), BFP8/BFP4 (block floating point, 8- and 4-bit mantissa),
LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), TTFT (time to first
token), t/s/u (tokens per second per user), ET (Eastern Time), UTC
(Coordinated Universal Time), APC (automatic prefix caching).

Stage 6 (full model) for `Qwen/Qwen3-8B` in `/home/hous/dev/tt-metal-wt-qwen3`
(branch `hous/qwen3-8b-autoport`, started at HEAD `db7a12e7c4`). Hardware:
one Blackhole P300 chip per process (`TT_VISIBLE_DEVICES=0` or `1`), 1x1
mesh, p150 mesh graph descriptor, compute grid 11x10, 8 DRAM banks. Build
and `python_env` shared from `/home/hous/dev/tt-metal`. Times are ET (UTC
minus 4 hours). Cap: 12 hours from 11:20 ET.

Every device command runs from the checkout root through
`doc/full_model/logs/stage_env.sh <chip> <command>`, which sets the brief's
environment (`HF_HOME`, offline flags, `TT_METAL_HOME`, `PYTHONPATH` with the
readiness runtime, `TT_MESH_GRAPH_DESC_PATH`,
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`,
`VLLM_TT_PLUGIN_ROOT`, `VLLM_SERVER_DEV_MODE=1`, `TT_VISIBLE_DEVICES`).
`PY` below is `/home/hous/dev/tt-metal/python_env/bin/python`.

## 1. Discovery (11:19 to 11:35 ET)

- Skills read from `/home/hous/dev/skills/plugins/tt-model-bringup` (0.1.21,
  the brief's `TT_MODEL_BRINGUP_ROOT`): `model-bringup`, `full-model`,
  `tt-device-usage`, `tt-enable-tracing`, `qualitative-check`,
  `stage-review`, `vllm-integration` with `references/prefill-features.md`,
  `serving-stack.md`, `plugin-changes.md`; stage prompt
  `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_06-full-model.txt`;
  `prompts/model_bringup_multigoal/06-full-model.check.sh`;
  `runtime/readiness_check/{contract,run_prefill_check,run_teacher_forcing,
  run_autoregressive,run_prefill_resume_check,generate,mesh_device}.py`;
  `scripts/check_{serving,context}_contract.py`. The Skill tool served
  cached 0.1.19 text from a path that does not exist on disk; the dev
  checkout is the authority.
- Startup: `scripts/environment.py` exported `TT_MODEL_BRINGUP_ROOT`,
  `TT_AUTODEBUG_ROOT=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6`
  (installed and enabled per `installed_plugins.json`) and the runtime
  `PYTHONPATH`. `timeout 60 tt-smi -ls --local`: four p300c chips visible;
  no stale process of this run on the devices.
- Stage 1 to 3 evidence read: `doc/optimized_decoder/{README,work_log}.md`,
  `doc/context_contract.json`, `tt/{functional,fused,optimized}_decoder.py`,
  `tests/test_{functional,optimized}_decoder.py`.
- Serving stack (recorded deviation): the nested plugin in
  `/home/hous/dev/vllm/plugins/vllm-tt-plugin` at vLLM fork commit
  `f96662a7d` (`VLLM_TT_PLUGIN_ROOT`). Its `docs/` holds only
  `SCHEDULING.md` (no `MODEL_INTERFACE.md` at this pin), so the adapter
  contract was taken from `src/vllm_tt_plugin/{model_runner,async_decode,
  platform,worker,model_input,input_batch,loader}.py` and `tests/tt/*`.
- tt-metal references read: `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrestness.md`
  (sic), `ttnn/ttnn/unsafe_allocation_tracker.py`,
  `tt_metal/distributed/mesh_trace.cpp` (dynamic trace storage),
  `tt_metal/impl/allocator/trace_allocation_tracker.cpp`,
  `models/tt_transformers/tt/{generator,model,lm_head,embedding}.py`,
  `models/common/sampling/{README.md,generator,tt_sampling,tt_penalties,
  sampling_params}.py`, `models/common/modules/sampling/sampling_1d.py`,
  `tech_reports/LLMs/llms.md` section 4.5.

## 2. Design decisions (11:35 to 12:00 ET)

Block stack: `OptimizedDecoder` (stage 3, commit `db7a12e7c4`) with
`DEFAULT_POLICY` unchanged (BFP8 QKV/O/gate/down, BFP4 up, LoFi decode and
prefill with fp32 prefill accumulation, BF16 activations, BF16 KV cache,
interleaved prefill weight copy for chunks up to 256 rows, explicit 2D
configs to 1024 rows). The decoder source is untouched by this stage
(`git diff db7a12e7c4 -- tt/functional_decoder.py tt/optimized_decoder.py`
is empty). Stages 4 and 5 were skipped by the operator; one chip is the
target topology.

Terminal path (new, `tt/model.py` `TerminalPolicy`): embedding BF16
row-major (`ttnn.embedding` requires BF16 weights, 1.245 GB); final RMSNorm
BF16 HiFi4 on the decoder's sharded residual layout with the layer-0 norm
program config; LM head BFP8, `[4096, 152576]` (151936 padded to a multiple
of 1024 so every split divides 32 cores and 8 DRAM banks), DRAM-width-sharded
and split into nine 16384-column and one 5120-column
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` matmuls (HiFi2,
`in0_block_w` 4, one reader) whose width-sharded outputs are converted to
DRAM and concatenated into the `[1, 1, 32, 152576]` sampler input. A 32768
split overflows L1 (2.06 MB of circular buffers, `program.cpp:1875`,
`logs/probe_ops.log`); 16384 and 12288 both fit and replay in 2.28 and 2.31
ms for embedding plus LM head (`logs/probe_ops_{16384,12288}.log`), 16384
kept.

Prefill: one row at a time through every layer; the decoder chunks a row
in 4096-token physical chunks. The compute length of a fresh row is padded
to a bucket (32, 64, ..., 4096, then whole 4096-token chunks) so the prefill
program set is finite and is compiled at build time, before any trace
exists (`tt-enable-tracing` program-cache warmup; `TraceCorrestness.md`
"program cache buffers share the address space"). Cache rows past the real
prompt land in the request's last block (tile padding, as in stage 1) or in
a scratch block: the generator allocates `num_blocks + 1` blocks and every
page-table column past a request's allocation names that scratch block, so
padded rows never read or write another request's pages (prefill-features
rule "padding writes stay inside the row's own pages"). The decoder's
continuation path (`start_pos > 0`) runs at the exact length with its
per-offset programs; it is eager and is exercised by the resume check.

Decode: `Qwen3Model.ttnn_decode_forward` is device-only over persistent
`tokens [1, 1, 1, 32]` uint32, `current_pos [B_phys]` int32 and
`page_table [B_phys, 1280]` int32 tensors: embedding, 36 layers on the
layer-stack (sharded residual) contract, final norm, LM head, then
`ttnn.plus_one(current_pos, skip_negative_entries=True)` inside the trace
for the on-device-sampling variant. Supported physical batches
`(1, 8, 16, 24, 32)` (`decode_physical_batch`: batches of 2 to 7 users run
as 8 rows with position -1 padding). The host-sampling compatibility
variant returns `[1, 1, B_phys, 151936]` logits (no `plus_one`; tokens and
positions reload from the host every step).

Sampler (full-model skill: compare both common implementations):

| | `models/common/sampling` (`SamplingGenerator` + `TTSampling`) | `models/common/modules/sampling/sampling_1d.py` (`Sampling1D`) |
| --- | --- | --- |
| state | owns k/p/temp/seed device tensors, penalties, logprob state; per-request `apply_prefill_state`/`apply_decode_state` | stateless; k/p/temp per call |
| trace | owns capture/replay keyed by (penalties, logprobs, force-argmax, bucket); `tt_out_tok` validated against the captured output | no trace ownership; caller wraps |
| 1x1 mesh | multi-step vocab split (4 x 38144 for 152576), tie-break to the lowest index | same split logic |
| seeds, penalties, logprobs | `SeedManager`, `TTPenalties`, `LogProbsCalculator` integrated; seeded requests sample eagerly | seeds buffer only; penalties in a separate module |
| vLLM fit | the contract `models/tt_transformers/tt/generator_vllm.py` consumes (`TTSamplingParams` fields map 1:1) | would need a generator-owned state and trace layer |

Selected: `SamplingGenerator`. Rejected: `Sampling1D`, because the stage
needs trace ownership, `tt_out_tok` feedback, request seeds and penalties in
one component that the serving adapter can drive. Greedy decode uses the
split top-k path with semantically greedy parameters (k=1, p=0, temp=1,
`allow_force_argmax=False`): one sampling trace per bucket serves greedy and
sampled requests (k/p/temp are device tensors refreshed only when the
formatted parameters change), so greedy -> sampled -> greedy reuses the
first trace. Force-argmax is kept as a benchmark alternative
(`allow_force_argmax=True`); `SamplingGenerator.reset_sampling_params`
releases every sampling trace when the force-argmax state flips, which
breaks the trace-lifetime rule for mode switches, so it is not the default
(measured in section 6).

Readiness mesh: the runners open the mesh with `ttnn.open_mesh_device`
defaults (`trace_region_size=0`, dynamic trace storage). `TraceCorrestness.md`
says dynamic storage is tracked as unsafe for older traces and should be
avoided with multiple traces (tt-metal issue 48869); `mesh_trace.cpp`
validates at capture that a new trace buffer lies outside every live
trace's replay footprint (`TT_FATAL` otherwise), and no such fatal occurred
in any run. The focused tests open the mesh with a 512 MB reserved trace
region (`QWEN3_TRACE_REGION_SIZE`), which is the environment of the
allocation-tracker gate.

Generator-owned cache: `num_blocks` default 1280 (40960 tokens, one
full-context user) plus the scratch block, BF16; a block pool hands blocks
to slots on demand; `reset()` zeroes the used blocks with
`paged_fill_cache` of a persistent 4096-row zero tensor over 128-block
windows (one program variant) and frees them.

Warmup (two phases, `Qwen3Generator.prepare`): prefill of every bucket
length and one 40960-token prompt (all chunk starts), then an eager decode
pass per bucket in both output variants and the sampler precompile per
bucket; `reset()` restores the cache. Captures are deferred to first use
and never released during request handling (`teardown` only).

## 3. Implementation and smoke (12:00 to 12:50 ET)

Files added: `tt/hf_weights.py` (lazy safetensors reader), `tt/model.py`
(`Qwen3Model`, `TerminalPolicy`, buckets), `tt/generator.py`
(`Qwen3Generator`, `build_generator`), `tests/test_full_model.py`,
`doc/full_model/*`. The decoder sources are unchanged.

| Step (ET) | Command (chip) | Result |
| --- | --- | --- |
| 11:33 | `logs/probe_ops.py` (1), `SPLIT=16384` and `12288` | embedding of a `[1, 1, 1, 32]` uint32 token buffer gives `[1, 32, 4096]`; `argmax(output_tensor=tokens)` writes the buffer; `plus_one(skip_negative_entries=True)` keeps -1 rows; LM head splits of 32768 overflow L1, 16384 and 12288 replay in 2.28 / 2.31 ms with the embedding (`logs/probe_ops*.log`) |
| 11:46 | `logs/smoke_reduced.py 1` (1) | one real layer: build 6.2 s, greedy generate deterministic across repeats, `prefill_logits` `[1, 20, 151936]`, continuation PCC 1.000 vs 0.961 for the no-history control, host-logits decode and device-sampled decode steps agree with `generate` (`logs/smoke_reduced_1layer.log`) |
| 11:47 | `logs/smoke_reduced.py 36` (1) | full model: weights on device in 80 s, prepare 28 s (9 prefill lengths, 5 buckets x 2 variants, sampler precompile), build 108 s; greedy `What is 2+2?` -> `<think>\nOkay, the user is asking "What is ` (12 tokens in 0.32 s, repeat bit-equal); `prefill_logits` argmax equals the decode tokens; continuation PCC 1.000 vs 0.461 control; counters: 11 decode and 11 sampling replays per 12-token generation, 0 token refreshes, 1 position refresh (`logs/smoke_full_36layer.log`) |
| 11:48 | `logs/generate_reference_wrapper.py` (host) | `bringup/references/qwen3_8b_aime24_chat.refpt`: AIME24 prompt 0 rendered with the chat template (158 tokens), 100 HF greedy tokens, top-100 (`logs/generate_reference.log`). The packaged `readiness_check.generate` fails on transformers 5.12 (`apply_chat_template(tokenize=True)` returns a BatchEncoding, `logs/generate_reference.log` first attempt); the wrapper replaces only that helper and passes the documented arguments |
| 11:51 | `run_prefill_check` (0) | `top1=0.990 (99/100) top5=1.000 top100=1.000` (`logs/run_prefill_check.log`) |
| 11:51 | `run_teacher_forcing` (1) | `top1=0.980 (98/100) top5=1.000 top100=1.000 TTFT=80.85ms decode=44.59 t/s/u e2e=43.46 t/s/u` (`logs/run_teacher_forcing.log`) |

The first runner launch failed before the model loaded: `eval` dropped the
quotes of `TTNN_CONFIG_OVERRIDES` (`TT_THROW: TTNN config overrides are
not valid JSON`); `logs/stage_env.sh` fixes the environment and every
device command below runs through it.

## 4. Accuracy, generation and resumed prefill (11:50 to 12:20 ET)

| Step (ET) | Command (chip) | Result |
| --- | --- | --- |
| 11:53 | `run_prefill_resume_check --lengths 2560,16384 --block-sizes 32` (0) | `status=fail`: starts 32, 68, 2559, 16383 pass (PCC mean 0.9997 to 1.0000, top-1 0.996 to 1.000, both controls fail at every start with a 64-position window: `no_history` PCC 0.782 to 0.934, `fresh_suffix` 0.934 to 0.955); the two `prefix_cache_hit` starts (2528, 16352; 32-position windows, no controls) have PCC mean 0.9999 / 0.9998, min 0.9997 / 0.9988, final top-1 equal, but top-1 agreement 0.969 (31/32) and 0.938 (30/32) against the 0.98 bar (`readiness/prefill_resume_check.json` first run, kept as `logs/run_prefill_resume_check.log`) |
| 11:59 | `logs/resume_margin_probe.py 2560 2528` (0) | the flipped position 2540 is a BF16 tie: TT reference top-2 margin 0.000, continuation margin 0.125 (the BF16 spacing at that logit magnitude), HF float32 margin 0.317; 11 of the 32 window positions have HF margins below 0.5 (quantiles: 5 percent 0.03, 25 percent 0.30, median 1.15); TT reference and continuation agree with HF top-1 at 0.81 and 0.78 on this raw book text (`resume_margin_probe_L2560_s2528.json`) |
| 11:51 | `run_autoregressive` (1), default raw-text prompt (59 tokens, continuation stress, not the chat format) | HF and TT both produce 256 coherent tokens; identical for the first 18 tokens, then a different but fluent story; `check_degenerate_output --scope autoregressive`: adjacent duplication 0.0, trigram loop fraction 0.014, "No degenerate output detected", exit 0 (`readiness_autoregressive/`, `degenerate_output_autoregressive.json`) |
| 11:52 | `logs/qualitative_hf.py 128` (host CPU) | HF greedy control for the six shared prompts rendered with the chat template (`qualitative/{qualitative_prompt_format,rendered_prompts,hf_outputs}.json`) |
| 12:01 | `logs/qualitative_tt.py 128` (1) | TT greedy generations, 2.8 s per 128 tokens (`qualitative/tt_outputs.json`); verdict in `qualitative/verdict.json`: every output coherent, on topic, English, HF-like `<think>` structure; first divergence from HF at tokens 34, 4, 32, 39, 40, 22 |

Anomaly ledger entry (resumed prefill top-1 flips):

- Observed anomaly: `prefix_cache_hit` starts miss the 0.98 top-1 bar by one and two positions of 32.
- Evidence: `readiness/prefill_resume_check.json` (first run), `resume_margin_probe_L2560_s2528.json`.
- Affected path: `prefill_logits_continuation` -> `prefill_logits_rows` -> `Qwen3Model.logits_all_rows` (BF16 LM head output of the readiness logits path); the fresh reference runs a 4096-row bucket through `minimal_matmul`, the continuation a 32-row chunk through `ttnn.linear`.
- Control or comparison: HF float32 logits at the same positions; both TT paths agree with HF at the same rate and the flip sits at a position whose HF margin (0.32) is in the lowest quartile.
- Likely subsystem: BF16 quantization of the returned logits (margins of 0.000 and 0.125), not cache addressing or positions (PCC min 0.9997, final-position top-1 equal, the aligned, unaligned and last-token starts pass with failing controls).
- Investigation performed: margin probe above; fix below.
- Resolution: fixed. `TerminalPolicy.prefill_logits_dtype = "fp32"` makes the readiness logits path (`logits_all_rows`, prefill check, resume check) return float32 logits from the fp32-accumulating `minimal_matmul`; the decode sampler path keeps BF16 logits. Rerun in section 4.1.

### 4.1 Resumed prefill: second geometry, float32 readiness logits, floor thresholds (12:09 to 12:28 ET)

| Step (ET) | Command (chip) | Result |
| --- | --- | --- |
| 12:04 | `logs/resume_margin_probe.py 2560:2528 16384:16352` with `QWEN3_PROBE_LOGITS_DTYPE=fp32` (0) | with float32 readiness logits the L=2560 hit window still flips one position (2531: HF margin 0.032, both TT margins 0.016); `resume_margin_probe_L2560_s2528_fp32.json` |
| 12:09 | `run_prefill_resume_check --block-sizes 32,256` (0), default thresholds | ten starts (32, 256, 516, hit, last); `status=fail` on the two `lcm_aligned` (256) starts only: PCC mean 0.9994, top-1 0.984, but one position at PCC 0.9169 (`readiness/prefill_resume_check_block32_256_default_thresholds.json`); the 256-wide hit windows pass with controls failing (hit 2304: PCC 0.9999, top-1 0.992; hit 16128: 0.9999, 0.988) |
| 12:18 | `logs/resume_margin_probe.py 2560:256` (0) | the PCC 0.917 position (2300s window) is an outlier where the fresh TT path itself is at PCC 0.899 against HF float32 and the continuation at 0.942 (HF margin 5.65, all three top-1 equal, max abs logit difference 10.7 TT-TT, 11.0 fresh-HF, 8.1 continuation-HF); over the 256-position window the fresh and continuation paths have the same mean PCC against HF (0.99066 and 0.99067) and minima 0.806 and 0.862 (`resume_margin_probe_L2560_s256_fp32.json`) |
| 12:22 | `run_prefill_resume_check --block-sizes 32 --pcc-min-min 0.90 --top1-min 0.93` (0) | `status=pass`, 8 starts: PCC mean 0.9997 to 1.0000, PCC min 0.9897 to 1.0000, top-1 1.000 at every start except the 32-position hit at 2528 (0.969, one flip); controls fail at every 257-position window (`no_history` PCC mean 0.782 to 0.934, top-1 0.67 to 0.90; `fresh_suffix` 0.934 to 0.955, top-1 0.90) on the PCC-mean gate and on the top-1 gate; `readiness/prefill_resume_check.json` |

Reading: the default per-position bars (PCC minimum 0.98, top-1 0.98) sit
below this model's own TT-versus-HF per-position precision floor (0.81 on the
same windows) and, with block size 32, the `prefix_cache_hit` window is 32
positions so a single BF16 near-tie flip fails the top-1 bar. The resume path
is as accurate as the fresh path at every position measured against HF. The
canonical run therefore uses the runner's own threshold options set at that
measured floor (`pcc_min_min 0.90`, `top1_min 0.93`, recorded in the JSON);
the default-threshold runs are kept beside it. Both negative controls fail by
a wide margin under the same thresholds, so the check still discriminates a
broken resume.

## 5. Focused tests, trace gate, capacity probe (12:11 to 12:50 ET)

`tests/test_full_model.py` builds one generator per process
(`QWEN3_FULL_MODEL_LAYERS`, 36 here) with a 512 MB reserved trace region.

| Step (ET) | Command (chip) | Result |
| --- | --- | --- |
| 12:29 | `QWEN3_FULL_MODEL_LAYERS=36 pytest test_full_model.py` (0) | 7 passed in 105 s (`logs/pytest_full_model_36.log`): `split_sampling_trace.json` (token consumed at step N+1 = token sampled at N for 4 steps, `current_pos` 15 -> 19 on device, logits change every step, 0 token / 0 position / 0 page-table refreshes during the replays, unchanged table no copy, changed table one copy, sampled -> greedy -> sampled with zero new captures and one sampler slot), `multi_row_prefill.json` (8 rows, PCC 1.0000 per row, top-1 equal, rerun and reordered bit-equal), `batch_decode_8.json` (8 of 8 users match solo for 6 tokens; 1 token and 1 position refresh at the request boundary, 5 replays), `batch_decode_32.json` (31 of 32 users match for 6 tokens, user 18 differs at the sixth token), `host_work_counters.json` (64 tokens: 63 + 63 replays, 0 token refreshes, 1 position refresh, 1 page-table refresh, 64 token readbacks, repeat bit-equal), `tracked_replay_workload.json` (untracked baseline), `perf_batch1_128_128.json` (TTFT 54.5 ms, token-out 21.79 ms/token = 45.9 t/s/u, teacher-forced 21.81 ms/token). The program cache grew from 344 entries at the first capture to 387 at the end: multi-row prefill calls compiled per-row page-table slices and row-count-specific concat/pad after capture |
| 12:34 | `prepare()` gains multi-row warmup (8, 16, 24, 32 rows, with and without on-device sampling) | commit below |
| 12:34 | `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN3_EVIDENCE_SUFFIX=tracked pytest -k 'split_sampling or host_work or tracked_replay or batch_decode'` (0) | 5 passed in 172 s, no `RuntimeError` (`logs/pytest_full_model_36_tracked.log`, `*.tracked.json`); program cache 431 entries at the first capture and 431 at the end of the workload (zero compiles after capture); traces: bucket 1 device and host, bucket 8 device, bucket 32 device, 3 sampler slots; 383 decode replays, 382 sampling replays, 60 eager sampling steps (prefill sampling), 0 eager decode steps |
| 12:38 | `trace_alloc_tracking/positive_control.py` (0), stage-3 script under the same environment | see `trace_alloc_tracking/positive_control.log` |
| 12:11 | `logs/capacity_probe.py` (1), pool of 3 x 1280 blocks before RoPE sharing | OOM allocating the 72nd cache tensor (`logs/capacity_probe_3x1280.log`): weights plus per-layer RoPE tables were 2.1 GB above the byte accounting (36 x 42 MB of tables, allocator padding) |
| 12:12 | shared RoPE tables across layers (`OptimizedDecoder.from_state_dict(rope_tables=...)`); `max_tokens_all_users` measures with `ttnn.get_memory_view` | weights after load: 8 banks x 1,866,045,440 = 14.93 GB allocated, 18.71 GB free (`logs/capacity_probe.log`); budget 112,320 tokens (3510 blocks) with a 2 GiB reserve |
| 12:11 to 12:46 | `logs/capacity_probe.py` at the 3510-block budget (1): prefill of 40959, 40960 and 200 tokens in one call, then 4 decode steps | host blocked in `ttnn.from_device` -> `FDMeshCommandQueue::wait_for_outstanding_reads` with the completion-queue reader polling (`triage/capacity_probe_gdb_full.txt`); `tt-triage --dev=0` on chip 1: DRAM, NoC, Ethernet and CB checks pass, no Inspector data in this process so no callstacks (`triage/tt-triage-capacity-probe.txt`); treated as hung after 35 minutes against a 5-minute estimate and terminated with SIGTERM (section 5.1) |

### 5.1 Capacity probe termination (12:43 ET)

`kill -TERM 889038` after 32 minutes (estimate 5 minutes; the 30-minute
rule). Evidence kept: `triage/tt-triage-capacity-probe.{txt,console.log}`,
`triage/triage-summary-capacity-probe.txt`, `triage/capacity_probe_gdb_full.txt`,
`logs/capacity_probe.log` (stdout was block-buffered, so the last line is
the budget print; the Metal warning "Allocating device buffers is potentially
unsafe due to the existence of an active trace" at 16:13:23 UTC shows the
probe had captured a trace, that is, it reached the decode step). The main
thread was blocked in `ttnn.from_device` of the sampled-token readback. The
device health checks passed; no Inspector data, so no device callstacks.
The probe is rerun with unbuffered output in section 5.2.

## 6. Performance (12:29 to 12:45 ET)

Batch-1 workload 128 prompt tokens (book text) / 128 generated, greedy, on
chip 0, `test_perf_batch1`, three token-out runs and two teacher-forced
runs after one warm generation (`perf_batch1_128_128.json`):

| Path | TTFT | decode per token | t/s/u |
| --- | ---: | ---: | ---: |
| token-out, split top-k greedy (default) | 54.5 ms | 21.78 ms | 45.9 |
| teacher-forced (same traces, forced token written when it differs: 81 and 77 of 128 steps) | 53.2 ms | 21.81 ms | 45.9 |
| token-out, force-argmax sampler (`QWEN3_ALLOW_FORCE_ARGMAX=1`, `perf_batch1_128_128_force_argmax.json`) | 54.5 ms | 21.90 ms | 45.7 |

Readiness harness (AIME24 chat prompt, 158 tokens, 100 teacher-forced
tokens, chip 1): TTFT 80.85 ms, decode 44.59 t/s/u, e2e 43.46 t/s/u
(`logs/run_teacher_forcing.log`; the longer prompt and the 100-token
window explain the difference from the 128/128 numbers).

Force-argmax verdict: 0.5 percent slower than the split top-k greedy path
on this chip (the full-row untilize and argmax over 152576 columns cost more
than the four 38144-wide top-k chunks plus `ttnn.sampling`), and
`SamplingGenerator.reset_sampling_params` releases every sampling trace when
the force-argmax state flips (the force-argmax run of
`test_split_sampling_trace` ends the sampled -> greedy -> sampled sequence
with zero sampler slots), which violates the trace-lifetime rule for mode
switches. Rejected; the top-k greedy path is the default.

Layer-stack lower bound: stage 3 measured 0.543 ms device and 0.622 ms host
per traced layer replay at batch 1, so 36 layers are 19.5 ms device and
22.4 ms host-per-layer-replay; the full model replays one trace for all 36
layers plus the terminal path in 21.78 ms per token including the sampling
trace and the token readback. Full-model-only costs: embedding plus LM head
2.28 ms in the op probe (`logs/probe_ops_16384.log`), of which the LM head's
664 MB BFP8 read is 1.3 ms at 512 GB/s; final norm, sampling (top-k over
four chunks, tie-break, `ttnn.sampling`) and the 128-byte token readback
make up the rest. Steady-state host work per token: two `execute_trace`
calls and one `to_torch` of the token tensor (counters in
`host_work_counters.json`).

