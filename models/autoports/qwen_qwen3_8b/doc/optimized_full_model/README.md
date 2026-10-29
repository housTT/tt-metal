# Qwen3-8B optimized full model

Status: optimized on one Blackhole P300 chip (chip 2, 1x1 mesh, p150 mesh
graph descriptor); every accuracy, generation, trace, watcher and gate check
reran on the final code (commit `4c9a98d810` plus this evidence commit); the
stage gate script exits 0. Stages 4 and 5 (multichip) were skipped by the
operator; one chip is the target topology. Stage review: see `work_log.md`
section 6.

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), LM (language model), RoPE (rotary position embedding), SDPA
(scaled dot-product attention), DRAM (device dynamic RAM), L1 (per-core
local memory), BFP8/BFP4 (block floating point, 8- and 4-bit mantissa),
LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), TTFT (time to first
token), t/s/u (tokens per second per user), ET (Eastern Time).

## Headline numbers (batch 1, real weights, chip 2, before = stage 6 at `f967722683`, after = this stage)

| Metric | Before | After | Workload | Artifact |
| --- | ---: | ---: | --- | --- |
| Token-out decode (decode trace + sampling trace + one token readback per step) | 21.86 ms/token = 45.7 t/s/u | 21.01 ms/token = 47.6 t/s/u (three runs 21.011 / 21.02 / 21.03) | 128-token book prompt, 128 greedy tokens, `test_perf_batch1` | `../full_model/perf_batch1_128_128.json` (provenance `checkout_git_sha`, `visible_devices 2`) |
| Teacher-forced decode (same traces, forced token written when it differs) | 21.9 ms/token | 21.06 ms/token | same | same |
| Token-out TTFT, 128-token prompt | 54.5 ms | 50.7 ms (`test_perf_batch1`), 51.0 ms (sweep) | prefill + first sampled token readback | same; `ttft_sweep_{before,after}.json` |
| TTFT at every bucket boundary | see table | 129 tokens 82.5 -> 57.1 ms, 1025 tokens 229 -> 167 ms, 2049 tokens 452 -> 327 ms, 4097 tokens 1090 -> 524 ms, 8193 tokens 1889 -> 1171 ms | `logs/ttft_sweep.py`, 25 lengths | `ttft_sweep_before.json`, `ttft_sweep_after.json` |
| Readiness teacher forcing (AIME24 chat, 158-token prompt, 100 tokens, traced decode and sampler) | TTFT 80.5 ms, 44.9 t/s/u, e2e 43.8; top-1 0.980, top-5 1.000, top-100 1.000 | TTFT 55.2 ms, 46.7 t/s/u, e2e 46.0; top-1 0.980, top-5 1.000, top-100 1.000 | `run_teacher_forcing` | `logs/run_teacher_forcing.log`, `../../readiness/readiness_stage7_chip2.json` |
| Readiness prefill check | top-1 0.990, top-5 1.000, top-100 1.000 | top-1 0.990, top-5 1.000, top-100 1.000 | `run_prefill_check` | `logs/run_prefill_check.log` |
| `prepare()` (two-phase warmup, 36 layers) | 64 s (17 prefill lengths) | 47 s with the kernel cache warm, 89 s on the first build of a new program set (10 buckets, 90 chunk pieces) | build time, not request time | `logs/pytest_full_model_36.log`, `logs/variant_sharded_norm/pytest_full_model_36.log` |

Teacher-forcing and token-out decode share the same traces in this
generator (the teacher-forced loop adds a 128-byte token write when the
forced token differs), so the two decode numbers stay within 0.2 percent.

## What changed (and what did not)

Preserved unchanged: the stage-3 decoder policy (`DEFAULT_POLICY`: BFP8
QKV/O/gate/down, BFP4 up, LoFi, BF16 activations and residual, BF16 KV
cache, 32-core L1 width-sharded decode residual, rejection ledger in
`../optimized_decoder/README.md`), the decode op graph of every layer, the
split-sampling contract (model trace to sampler-ready logits, common
`SamplingGenerator` trace with `tt_out_tok` feedback, on-device `plus_one`,
changed-only page tables, greedy = top-k path with k=1/p=0/temp=1,
force-argmax a benchmark alternative only), the serving-ready generator
API, the context contract (40960 retained) and the decode buckets 1, 8,
16, 24, 32.

Changed, with the measured reason (`work_log.md` section 2 has every
candidate table):

| Change | Measured reason | Where |
| --- | --- | --- |
| LM head decode fidelity HiFi2 -> LoFi (BFP8 weights, same DRAM-sharded geometry) | 10 x `32 x 4096 x N` matmuls 2.26 -> 1.50 ms per token (the HiFi2 row was FLOP-bound at 84 percent); readiness top-1/top-5/top-100 unchanged | `tt/model.py` `TerminalPolicy.lm_head_decode_fidelity`; `QWEN3_LM_HEAD_DECODE_FIDELITY=hifi2` restores |
| Padded-vocab mask moved from the sampler into the LM head: a persistent `[32, 5120]` bf16-min additive mask on the last split's sharded output (2 us); the sampler is built with `vocab_size = padded_vocab_size` | the common sampler's tail-mask path (two slices, add, concat) cost 106 us per step: greedy sampler trace 0.976 -> 0.869 ms | `tt/model.py` `lm_head_tail_mask`, `tt/generator.py` `_sampling_args` |
| Prefill projection kernel per role and chunk size (`PREFILL_KERNEL_TABLES["stage7"]`): 2D `MatmulMultiCoreReuseMultiCastProgramConfig` for O and down from 64 rows, QKV from 128, gate and up from 256 (gate `minimal_matmul` at 512); auto config kept where it won | per-role probe at 64 to 512 rows: the 256-row chunk goes from 1.65 to 0.97 ms of matmul per layer, the 128-row chunk saves 80 us per layer | `tt/optimized_decoder.py` `OptimizedPolicy.prefill_kernel`; `QWEN3_PREFILL_KERNEL_TABLE=stage3` restores |
| Prompts longer than one chunk run chunk by chunk through the whole stack with a bucketed tail (4097 tokens compute 4096 + 32 rows instead of 8192); buckets 1536 and 3072 added inside the first chunk | TTFT 4097 tokens 1090 -> 517 ms, 8193 tokens 1889 -> 1173 ms, 1025 tokens 229 -> 166 ms, 2049 tokens 452 -> 328 ms | `tt/model.py` `prefill_pieces`, `prefill_hidden(keep_all_rows)`, `prefill_row_offset`; `tt/generator.py` `prepare` warms every (chunk start, bucket) piece |

Rejected or blocked, with evidence (`work_log.md` 2.1, 2.2):

- LM head with 2 DRAM readers (LoFi 2.29 ms, HiFi2 2.30 ms), 16-core
  activation shard with `in0_block_w` 8 (2.36 ms or L1 overflow),
  `packer_l1_acc=False` (1.84 ms).
- Sampler topk: the common sampler's full-row `topk_large_indices` route
  (0.735 ms, flat in k) beats every power-of-two chunking the stock
  multi-core bitonic can take (0.87 to 1.73 ms) and the sampler's own
  4-chunk path (0.83 ms); a power-of-two padded vocab (262144) would still
  route to the composite (k = 128 > 64) and read 72 percent more LM-head
  weight. No generator-owned wrapper.
- DRAM-sharded projection weights for prefill (`per_core_M = rows / 32`):
  `matmul_device_operation.cpp:1347: M == 1` for every role and size (exact
  op-contract blocker); the DRAM-sharded copy serves decode only.
- Prefill head ops: slice+reshape+transpose (318 us) and transpose+reshape
  (194 us) lose to `nlp_create_qkv_heads` (89 us) and `nlp_concat_heads` (64
  us) at 128 rows; the sharded-input concat-heads candidate hung the chip
  (killed by its own timeout) and the sharded-input head split trips the
  watcher (stage 3). Both ops stay on their per-tile-row core split (a
  tt-metal kernel item).
- Prefill RMSNorm block-sharded over 8 core columns for 64 to 1024-row
  chunks: 95 us -> 25 to 88 us per norm (PCC 1.0000 against the interleaved
  kernel) and about 4.8 ms of TTFT at 128 tokens, but the resume check then
  fails at the default and the lowered thresholds (`pcc_min` 0.8756 at
  start 68 against 0.9949 without it): the prefix of a continuation (32 or
  128 rows, sharded norm) and the fresh reference (1536 rows and up,
  interleaved norm) disagree at one position per window. Attributed by
  four controls (work log 4.2); rejected, kept as the
  `QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS=<rows>` experiment knob. At 4096 rows
  it overflows L1 anyway (`dataflow_buffer.cpp:2604`).

## Evidence

### Accuracy and generation (full model, real weights, chip 2, this stage's code)

| Gate | Result | Artifact |
| --- | --- | --- |
| `run_prefill_check` (AIME24 chat reference, 100 tokens, top-100) | top-1 0.990, top-5 1.000, top-100 1.000 | `logs/run_prefill_check.log`, `../../readiness/readiness_stage7_chip2.json` |
| `run_teacher_forcing` (traced decode) | top-1 0.980, top-5 1.000, top-100 1.000; TTFT 55.17 ms; 46.74 t/s/u | `logs/run_teacher_forcing.log` |
| `run_autoregressive` (raw 59-token prompt, continuation stress) and `check_degenerate_output --scope autoregressive` | HF and TT 256 coherent tokens, identical first 31 (stage 6: 18); adjacent duplication 0.0, trigram loop 0.027, no degenerate output, exit 0 | `../../readiness_autoregressive/`, `degenerate_output_autoregressive.json` |
| Shared qualitative suite, chat template, HF control, stage-6 TT control | six prompts, 128 greedy tokens each: every output coherent, on topic, English, HF-like `<think>` structure, 0.0 adjacent duplication, no control tokens; first divergence from HF at tokens 34, 37, 32, 35, 40, 22 (stage 6: 34, 4, 32, 39, 40, 22; prompts 2 and 5 are token-identical to the stage-6 TT output over all 128 tokens) | `qualitative/{qualitative_prompt_format,rendered_prompts,hf_outputs,tt_outputs,tt_counters,verdict}.json` |
| Multi-row prefill equals solo (rows 5, 31, 33, 127, 129, 257, 640, 1000) | PCC 1.0000 per row, top-1 equal, rerun and reordered bit-equal | `../full_model/multi_row_prefill.json` |
| Batched decode equals single-user (32 greedy tokens, same-bucket solo arm) | buckets 8, 16, 24, 32: every user bit-equal, 31 decode and 31 sampling replays, 0 captures | `../full_model/batch_decode_{8,16,24,32}.json` |
| Determinism | repeated 64-token `generate` bit-equal; repeated prefill bit-equal | `../full_model/host_work_counters.json`, `multi_row_prefill.json` |
| Resumed prefill (`run_prefill_resume_check`, lengths 1500, 2560, 5000, 16384, block 32, natural text) | default thresholds: every PCC gate passes at all 16 starts (PCC mean 0.9997 to 1.0000, min 0.9897 to 1.0000), both controls fail at every 257-position window, top-1 1.000 everywhere except the 32-position cache-hit window at 2528 (31 of 32, the stage-6 BF16 near-tie flip at position 2531: HF margin 0.032, TT margins 0.016) which makes `status=fail`; lowered thresholds (0.90 / 0.93, stage 6): `status=pass` at all 16 starts, same numbers | `../../readiness/prefill_resume_check_default_thresholds.json`, `prefill_resume_check.json`, `../full_model/resume_margin_probe_L2560_s2528_fp32.json`, `resume_margin_probe_L*_fp32.json` |

### Split-sampling, trace and host-work evidence (this stage's code)

- `../full_model/split_sampling_trace.json`: four replays after a prefill;
  the token consumed at step N+1 equals the token sampled at N on device,
  `current_pos` advances on device, the padded vocab columns of the
  sampler-ready logits are below -1e38 and the sampled id is a real token;
  0 token / 0 position / 0 page-table refreshes during the replays; an
  unchanged page table costs no copy, a changed one costs one copy;
  sampled -> greedy -> sampled = 3 sampling-trace replays, 0 new captures.
- `../full_model/host_work_counters.json`: 64 tokens = 1 prefill, 63 decode
  replays, 63 sampling replays, 0 token refreshes, 1 position refresh, 1
  page-table refresh, 64 token readbacks, 0 eager decode or seeded steps.
- `../full_model/tracked_replay_workload.json`: program cache 651 before and
  after a 3-row prefill, a 32-row prefill and 8192-, 5000- and 12345-token
  generations issued after the first capture (two full chunks; 4096 + a
  1024-row tail piece; three full chunks + a 64-row tail piece).
- Trace allocation gate (`TT_METAL_TRACE_ALLOC_TRACKING=1
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, final code): the split-sampling
  test, the host-work test, the representative workload and the four batch
  buckets, 7 passed in 1261 s with program-cache allocations included and
  no tracker error (`logs/pytest_full_model_36_tracked.log`,
  `../full_model/*.tracked.json`: program cache 639 before and after the
  post-capture workload, 6 decode and 5 sampling captures for the five
  buckets, 0 eager decode steps, 0 seeded eager steps); the stage-3
  positive control raised in the same environment
  (`logs/positive_control.log`: "Found 1 device buffer(s) still alive before
  trace replay"). Acknowledged buffers are unchanged from stage 6: each
  decode trace output (`ttnn.mark_corruptible`, consumed only by the same
  bucket's sampling replay or host read immediately after its replay) and
  the sampler's input and token tensor through `set_trace_bucket`; the
  LM-head tail mask is a persistent tensor allocated before capture; no
  `corruptible_allocation_scope`.
- Watcher (`TT_METAL_WATCHER=10`, reduced model with two real layers, final
  code): split-sampling, host-work, multi-row prefill and the representative
  workload, 4 passed, 6 deselected, 2 warnings in 17.79s =================; 4 watcher dumps, 0 fatal-pattern matches
  (`watcher/summary.json`, `watcher/generated/watcher/watcher.log`,
  `../full_model/*.watcher.json`). Watcher and profiler ran in separate
  processes.

### Performance accounting (token-out decode, batch 1, `perf_summary.json`)

| Term | Value | Source |
| --- | ---: | --- |
| Roofline: 6.47 GB of decoder weights + 0.66 GB of LM-head weights + 24 MB of KV reads per token at 512 GB/s | 13.99 ms | `../context_contract.json` byte accounting |
| Device time per token: 36 x 513 us decoder layer (reduced profile at position about 130) + terminal 2.35 ms | 20.83 ms | `profile/decode.txt` (one blocking replay of the decode and sampling traces, 80 device ops, 2,862 us for one layer plus terminal) |
| End to end (host wall time per token, nonblocking replay plus one 128-byte token readback) | 21.01 ms | `../full_model/perf_batch1_128_128.json` |
| Host gap | 0.18 ms | difference |

Terminal path per token: embedding 23 us (1 core), final RMSNorm 8 us (sharded, 33
cores), LM head 10 x `32 x 4096 x N` 1,400 us (`LoFi BF16 x BFP8`, 87 percent of
DRAM bandwidth, DRAM-bound; 2,149 us FLOP-bound at HiFi2 before), 10 layout
conversions and the concat 94 us, sampler 824 us (`TopkRoutePrep` 50,
`TopkLargeIndices` 648, `TopkRouteFinish` 21, tie-break 40, `ManualSeed` 17,
`Sampling` 28; the stage-6 mask slices and concat, 106 us, are gone),
`plus_one` 1 us. Device time is 67 percent of the roofline; the decoder stack
is 89 percent of the device time and carries the stage-3 policy, so the
remaining gap is the decoder's non-matmul ops (about 150 us per layer: SDPA,
SiLU-mul, head split, norms, RoPE gather, cache update) and the sampler's
candidate stage, both named in `perf_summary.json`. The layer-stack lower
bound (36 x 513 us = 18.5 ms) plus the terminal path (2.35 ms) is 20.8 ms;
token-out decode at 21.0 ms is within 1 percent of it.

## Checklist (optimize skill, single chip)

| Item | Status and evidence |
| --- | --- |
| Decoder path fully traced, no host fallbacks | done: unchanged decode graph; `host_work_counters.json`; `split_sampling_trace.json` |
| Decode activations width-sharded in L1 across norm, attention, residual, MLP, output projection | done, stage 3 layout unchanged; LM head input is the sharded residual layout |
| Prefill activations DRAM interleaved, program configs for large prefill matmuls | done: per-role kernel table (section 2.2 of the work log), `minimal_matmul` from 1536 rows |
| Operation-topology audit | done: work log 1.1 (decode and prefill op sequences, material matmuls, layout conversions, candidates, action) |
| Multi-device families, lower-movement residual, fused CCL, persistent CCL buffers | not applicable: one chip, no collectives (stages 4 and 5 skipped by the operator) |
| Best-candidate comparison and final default reproduces it | done: every kept candidate's probe number is reproduced in the final `test_perf_batch1` and TTFT sweep runs on the default code path |
| Dtype/fidelity policy visible in measured rows | done: `profile/decode.txt` rows 745 to 763 `LoFi BF16 x BFP8` for the LM head, the layer rows unchanged from stage 3 (`LoFi BF16 x BFP8` / `BFP4`); `profile/prefill_128.txt` shows the 44-core 2D QKV, O and down kernels |
| SDPA and composite ops | unchanged from stage 3 (explicit SDPA configs) |
| Fused or packed same-input projections | unchanged from stage 3 (QKV packed; gate/up separate for their different dtypes) |
| Explicit memory, program and compute-kernel configs for important ops | done: LM head (DRAM-sharded config, LoFi compute config, sharded output configs, mask), prefill norms (sharded program config), prefill matmuls (table) |
| Dominant matmul program-config sweep | done for the LM head (readers, `in0_block_w`/cores, fidelity, `packer_l1_acc`; work log 2.1) and the prefill projections (four kernels x four sizes x five roles; 2.2); decoder decode matmuls carry the stage-3 sweep |
| Compute fidelity swept per group | done for the LM head (LoFi kept); stage 3 for the decoder |
| Attention / MLP weight dtype trials, BFP4/LoFi | stage 3 (real weights, real activations; BFP4 attention and BFP4 gate/down rejected on layer-17 delta PCC); not repeated here, `$datatype-sweep` owns the frontier |
| Clean shard specs and core grids | done: 32 x 4 tiles LM head input, 8 x 16-tile norm blocks, 2D matmul grids 11 x 10 |
| DRAM-sharded decode matmuls with reader sweep | done: LM head 1 reader kept against 2 (work log 2.1); decoder roles stage 3 |
| Collective topology, fused matmul-CCL, persistent CCL buffers | not applicable (one chip) |
| MoE | not applicable |
| LM head and sampling: terminal path in the token-out trace, padded vocab masked before TopK, TopK width, no avoidable argmax / full-vocab gather / host readback | done: mask in the LM head output, split top-k greedy through the common sampler's full-row route (measured against every chunking), `tt_out_tok` feedback, one 128-byte token readback per step; terminal costs profiled separately (`profile/decode.txt`, performance accounting above) |
| LM head DRAM-sharded | done (stage 6 geometry, LoFi) |
| Reduced-precision experiments appropriate to this stage | LM head LoFi (kept, readiness unchanged); decoder policy unchanged by design |
| Performance accounting reconciled, `perf_summary.json` | done: roofline 13.99 ms, device 20.83 ms, end to end 21.01 ms from the same code (`perf_summary.json`) |
| Batch capability preserved | done: batch-1 primary; buckets 8, 16, 24, 32 bit-equal to solo on this code |
| Batched prefill (dense work once for all short rows), prefill burst, resume check | batched prefill is implemented by stage 10 per `prefill-features.md` ("Stage 10 implements batched prefill"); this stage leaves the calibration it asks for (single-row prefill cost per bucket: `ttft_sweep_after.json`) and keeps the per-row loop (`multi_row_prefill.json`). Resume check: passes every PCC gate at the default thresholds, fails top-1 only on the 32-position hit window at 2528 (BF16 near tie, margin probes); lowered-threshold run kept beside it (work log 5.1, 5.2) |

## Limitations and open items

- Decode is device-bound: 36 decoder layers are 84 percent of the step and
  carry the stage-3 policy; the next decode lever is the datatype frontier
  (`$datatype-sweep`), not generator orchestration.
- `prepare()` grew from 64 to 89 s (90 chunk-piece warms); it runs once per
  process.
- The prefill norms stay on the interleaved kernel (2 to 32 cores for 64
  to 1024 rows, 95 us each); a sharded norm used on every row count,
  including a two-stage variant for 2048 and 4096 rows, would remove the
  cross-path disagreement that rejected the partial version and is the next
  prefill lever.
- Continuation prefill at an arbitrary `start_pos` (prefix-cache hits,
  chunked continuations) is still eager with per-offset programs (stage 9/10
  item, unchanged from stage 6).
- Head ops in prefill stay on their per-tile-row core split (tt-metal
  kernel item); the sharded-input variants hang or trip the watcher.
- The Tracy post-processing (`process_ops_logs.py`) asserts when a one-core
  op loses its device markers; the profiled process therefore skips the
  32-row multi-row warmup (`warmup_multi_row=False`) and flushes the device
  profiler after `prepare()`. The measured windows are unaffected.
- `test_batch_decode[32]` under the trace allocation tracker needs
  `--timeout 1800` (as in stage 6); the tracked gate takes 21 minutes.

## Reproduction

From the checkout root, every device command through
`doc/full_model/logs/stage_env.sh 2 <command>` under `timeout`
(`PY` = `/home/hous/dev/tt-metal/python_env/bin/python`, `M` =
`models/autoports/qwen_qwen3_8b`, `L` = `$M/doc/optimized_full_model/logs`):

```bash
$S 2 env PROBE=lmhead,topk,sampler $PY $L/probe_candidates.py
$S 2 env PROBE=norm,prefill_mm $PY $L/probe_candidates.py
$S 2 $PY $L/ttft_sweep.py $M/doc/optimized_full_model/ttft_sweep_after.json
bash $L/evidence_chain_chip2.sh      # 36-layer tests, readiness runners, TTFT sweep, resume checks, qualitative, autoregressive
bash $L/attribution_chain_chip2.sh   # margin probes, resume-check controls, then logs/gate_chain_chip2.sh (tracker, watcher, profile)
bash $L/perf_report.sh               # tt-perf-report tables from the reduced-variant profile
$PY $L/qualitative_verdict.py
MODEL_DIR=$M bash /home/hous/dev/skills/plugins/tt-model-bringup/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh
```

## Artifact index

- `probe_candidates.{lmhead,topk,sampler,norm,heads,prefill_mm}.json`: op-level candidate tables (random weights, trace-replay timing).
- `ttft_sweep_before.json`, `ttft_sweep_after.json`: TTFT at 25 prompt lengths, before (`f967722683`) and after.
- `../full_model/*.json`: the focused-test evidence regenerated on this stage's code (provenance `checkout_git_sha`, `visible_devices`); `*.tracked.json` and `*.watcher.json` twins from the gate chain.
- `../../readiness/readiness_stage7_chip2.json`: readiness runner provenance and headline numbers; `prefill_resume_check*.json`: resume checks and controls.
- `qualitative/`: prompt format, rendered prompts, HF control, TT outputs, verdict.
- `profile/`: tt-perf-report tables of the reduced variant (one real layer, real terminal path).
- `perf_summary.json`: the performance accounting.
- `logs/`: every command's console log, the chain scripts and probes; `watcher/`: watcher log and summary.
