# Qwen3-8B optimized full model: work log

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), LM (language model), RoPE (rotary position embedding), SDPA
(scaled dot-product attention), DRAM (device dynamic RAM), L1 (per-core
local memory), BFP8/BFP4 (block floating point, 8- and 4-bit mantissa),
LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), TTFT (time to first
token), t/s/u (tokens per second per user), ET (Eastern Time).

Stage 7 (optimized full model) for `Qwen/Qwen3-8B` in
`/home/hous/dev/tt-metal-wt-qwen3` (branch `hous/qwen3-8b-autoport`, started
at HEAD `f967722683`). Hardware: one Blackhole P300 chip, `TT_VISIBLE_DEVICES=2`
only (chips 0 and 1 wedged pending an operator reset, chip 3 held by another
process), 1x1 mesh, p150 mesh graph descriptor, compute grid 11x10, 8 DRAM
banks. Build and `python_env` shared from `/home/hous/dev/tt-metal`. Cap: 8
hours from 15:50 ET (ends 23:50 ET). Stages 4 and 5 were skipped by the
operator; one chip is the target topology.

Every device command runs from the checkout root through
`doc/full_model/logs/stage_env.sh 2 <command>` (the brief's environment:
`HF_HOME`, offline flags, `TT_METAL_HOME`, `PYTHONPATH` with the readiness
runtime, `TT_MESH_GRAPH_DESC_PATH`, `TTNN_CONFIG_OVERRIDES` with
`throw_exception_on_fallback`, `VLLM_TT_PLUGIN_ROOT`, `VLLM_SERVER_DEV_MODE`,
`TT_VISIBLE_DEVICES=2`), under `timeout`. `PY` below is
`/home/hous/dev/tt-metal/python_env/bin/python`; `L` is
`models/autoports/qwen_qwen3_8b/doc/optimized_full_model/logs`.

## 1. Discovery and startup (15:45 to 16:05 ET)

- Skills read from disk (`/home/hous/dev/skills/plugins/tt-model-bringup`,
  0.1.21): `optimize`, `full-model`, `tt-device-usage`, `tt-enable-tracing`,
  `qualitative-check`, `stage-review`, `model-bringup` (startup),
  `vllm-integration/references/prefill-features.md`; the stage prompt
  `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_07-optimized-full-model.txt`;
  the gate `prompts/model_bringup_multigoal/07-optimized-full-model.check.sh`.
- Startup: `scripts/environment.py` exit 0 (`TT_MODEL_BRINGUP_ROOT`,
  `TT_AUTODEBUG_ROOT=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6`,
  enabled per `installed_plugins.json`); `timeout 60 tt-smi -ls --local` lists
  four p300c chips; no process of this run on any chip.
- Stage 6 evidence read: `doc/full_model/{README,work_log}.md`, the reduced
  profile `doc/full_model/profile/{decode,prefill_128}.txt`,
  `perf_batch1_128_128.json`, `host_work_counters.json`,
  `serving_contract.json`, `doc/context_contract.json`, stage 3
  `doc/optimized_decoder/README.md`; sources `tt/{model,generator,
  optimized_decoder,fused_decoder,functional_decoder}.py`,
  `tests/test_full_model.py`, `models/common/sampling/tt_sampling.py`,
  `ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp` (routing rules),
  `.../matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`
  (`per_core_M > 1` rule), `.../layernorm_device_operation.cpp` and
  `sharded_layernorm_factory_helpers.cpp` (sharded norm rules).
- 15:46 mesh smoke on chip 2 (`doc/full_model/logs/mesh_smoke.py`, 120 s
  timeout): open, add, 53 MB BFP8 upload, close, exit 0
  (`logs/mesh_smoke_chip2.log`).

### 1.1 Where the time goes (stage 6 reduced profile, chip 0, one real layer)

Decode, one token-out replay (21.86 ms per token end to end at 36 layers):

| Term | Device time | Share of the 36-layer step |
| --- | ---: | ---: |
| decoder layer (36 x 512 us at position 128) | 18.4 ms | 84 % |
| LM head, 9 x `32 x 4096 x 16384` + `32 x 4096 x 5120`, BF16 x BFP8 HiFi2, FLOP-bound | 2.15 ms + 10 x 5 us `sharded_to_interleaved` + 54 us concat | 10 % |
| sampler: tail-mask slice 53 + slice 1 + add 3 + concat 53, `TopkRoutePrep` 51, `TopkLargeIndices` 648, `TopkRouteFinish` 22, tie-break about 40, `ManualSeed` 17, `Sampling` 28 | 0.93 ms | 4 % |
| embedding 20 + reshard 2, final norm 8, `plus_one` 1 | 31 us | |

Prefill at 128 rows, one layer 1.29 ms device (36 layers: 46 ms of the 54.5
ms TTFT; the prefill is device-bound, the host runs ahead):

| Term | Device time per layer | Note |
| --- | ---: | --- |
| five projections (`ttnn.linear` auto config, interleaved BFP8/BFP4 copy, LoFi) | 119 + 102 + 180 + 181 + 289 = 871 us | 32 to 59 percent of DRAM bandwidth, marked `SLOW` |
| input and post-attention RMSNorm | 85 + 84 us | 4 cores (one per tile row) |
| `nlp_create_qkv_heads` / `nlp_concat_heads` | 80 / 54 us | 4 cores |
| SDPA, RoPE, cache fill, Q/K norms, SiLU-mul, adds | 58 + 39 + 10 + 19 + 28 + 20 us | |

Levers taken into the candidate tables below: LM head fidelity (carried
item b), the sampler's tail mask and topk route, prefill norms and head ops
on 64 to 1024-row chunks (carried item a), DRAM-sharded prefill projections
(`per_core_M > 1` is legal when `K / in0_block_w` equals the storage core
count), the chunk-boundary TTFT step (carried item c) and the resumed
prefill check at the default thresholds (carried item d).

## 2. Op-level candidate probes on chip 2 (16:06 ET onward)

`logs/probe_candidates.py` (random weights, timing = mean of 10 blocking
replays of a captured trace after 3 warm replays; one JSON line per
candidate, collected in `probe_candidates.<section>.json`).

### 2.1 Decode: LM head, topk, sampler (`PROBE=lmhead,topk,sampler`, `logs/probe_decode.log`, 16:06 to 16:11 ET)

LM head (embedding-free: 10 DRAM-sharded matmuls on a width-sharded
`[1, 1, 32, 4096]` input, each output converted to DRAM and concatenated):

| Candidate | ms per replay | Decision |
| --- | ---: | --- |
| HiFi2, 1 reader, `in0_block_w` 4, 32 cores (stage 6 default) | 2.258 | before |
| LoFi, 1 reader, `in0_block_w` 4, 32 cores | 1.503 | kept (accuracy gate in section 4) |
| LoFi, 2 readers | 2.290 | rejected (slower, as stage 3 found for BFP8 rows) |
| HiFi2, 2 readers | 2.296 | rejected |
| LoFi, 1 reader, tail mask add on the last split | 1.505 | kept (2 us for the mask) |
| LoFi, 1 reader, `packer_l1_acc=False` | 1.841 | rejected |
| LoFi, 2 readers, `in0_block_w` 8, 16 cores | 2.362 | rejected |
| LoFi, 1 reader, `in0_block_w` 8, 16 cores | L1 overflow (`program.cpp:1875`) | blocker recorded |

`in0_block_w` is bounded by the 4 K-tiles per core of the 32-core residual
shard (K = 4096); 16 cores unlock 8 but lose. LoFi versus HiFi2 output PCC
on random data 1.0002 (bf16 noise); the real-weight gate is the readiness
runners.

Topk on `[1, 1, 32, W]` BF16 (the sampler's candidate stage):

| Candidate | ms per replay |
| --- | ---: |
| full row W = 152576, k = 128 (the common sampler's route: `topk_large_indices` composite) | 0.735 |
| full row, k = 64 / k = 32 | 0.735 / 0.734 |
| 4 chunks x 38144, k = 32 (the sampler's chunked path; non-power-of-two chunks route to the composite too) | 0.828 (0.880 with the `ttnn.split`) |
| 5 x 32768 / 10 x 16384 / 3 x 65536 / 19 x 8192, k = 32 (power-of-two chunks, stock multi-core bitonic) | 0.873 / 1.336 / 0.997 / 1.728 |
| full row W = 163840, k = 128 | 0.788 |
| concat of 10 x 16384 pieces | 0.070 |

The composite's cost is flat in `k` and the power-of-two chunkings are all
slower, so the common sampler's full-row route is the fastest available
candidate stage and no generator-owned topk wrapper is written. (A
power-of-two padded vocab, 262144, would also route to the composite because
`k = 128 > 64`, and would read 72 percent more LM-head weight.)

Sampler (`SamplingGenerator` greedy trace, k = 1 / p = 0 / temp = 1, mean
of 20 steps, `tt_out_tok` into a token tensor):

| Candidate | ms per step | Greedy token equals torch argmax |
| --- | ---: | --- |
| `vocab_size = 151936` (sampler masks the 640-column tail: two slices, add, concat) | 0.976 | yes |
| `vocab_size = padded_vocab_size = 152576` (no sampler mask; the tail is masked in the LM head) | 0.869 | yes |

Kept: the LM head adds a persistent `[32, 5120]` additive mask (bf16 min at
the 640 padded columns) to its last split on the sharded output, so the
logits the sampler receives are already masked and the sampler is built
with `vocab_size = padded_vocab_size`. This is the skill's "mask padded
vocab IDs inside the logits before TopK" done at the producer; the generator
keeps the real `vocab_size` for the host-logits slice.

### 2.2 Prefill: norms, head ops, projections (`logs/probe_prefill.log` 16:12 to 16:37 ET, `logs/probe_prefill2.log` 16:23 to 16:27 ET)

The first prefill probe run (`PROBE=norm,heads,prefill_mm`) stopped at the
`nlp_concat_heads` candidate with a height-sharded input (`[rows x heads /
cores, 128]` over 8 cores): no error, no progress for 25 minutes against a
10-minute estimate, killed by its own `timeout 1500` (exit 124, no SIGTERM
from this stage). The chip completed the TTFT sweep that followed, so it was
not wedged; the sharded-input concat-heads candidate is recorded as a hang
and dropped (stage 3 found both sharded-input readers of the head split
trip the watcher too). The norm section of that run had used `subblock_w =
8`, which the fp32-accumulation kernel caps at 4
(`sharded_layernorm_factory_helpers.cpp:85`); the second run
(`PROBE=norm,prefill_mm`) uses 4.

RMSNorm over `[1, 1, rows, 4096]` (HiFi4, fp32 accumulation; sharded =
block-sharded over 8 core columns x `rows / 32 / block_h` core rows, block
`block_h x 16` tiles, subblock 4, including `interleaved_to_sharded` and
`sharded_to_interleaved`; PCC against the interleaved kernel 1.0000 at every
size):

| rows | interleaved (cores) | sharded round trip (cores) | sharded, output left sharded |
| ---: | ---: | ---: | ---: |
| 64 | 95 us (2) | 25 us (16) | 22 us |
| 128 | 96 us (4) | 28 us (32) | 25 us |
| 256 | 96 us (8) | 41 us (64) | 32 us |
| 512 | 99 us (16) | 55 us (64) | 39 us |
| 1024 | 112 us (32) | 88 us (64) | 58 us |
| 4096 | 360 us (110) | L1 overflow, block 13 x 16 tiles (`dataflow_buffer.cpp:2604`) | same |

Kept: the sharded round trip for 64 to 1024 rows
(`OptimizedPolicy.prefill_sharded_norm_max_rows = 1024`,
`OptimizedDecoder._rms_norm` override for hidden-width inputs); the
sharded-output variant needs the consuming matmul to read a block-sharded
input on an 8-column grid and is left as a lever (the probe's 2D matmul
configs use the 11x10 grid). 2048 rows were not probed (block 7 x 16 tiles;
the interleaved kernel costs about 0.2 ms there, so at most 0.1 ms per layer
is at stake).

Head ops at 128 rows: `nlp_create_qkv_heads` 89 us (4 cores) against
slice+reshape+transpose 318 us; `nlp_concat_heads` 64 us (4 cores) against
transpose+reshape 194 us; sharded-input concat heads hung (above), the
sharded-input head split trips the watcher (stage 3). The interleaved head
ops stay; their per-tile-row core split is a tt-metal kernel item.

Projections (interleaved BFP8/BFP4 weight copy, LoFi, fp32 accumulation; ms
per replay; `2D16`/`2D8` = `MatmulMultiCoreReuseMultiCastProgramConfig` on the
11x10 grid with `in0_block_w` 16 or 8, output subblocks as
`_prefill_2d_program_config` picks them; PCC against torch equal for every
kernel at a given size):

| role, rows | auto | minimal_matmul | 2D16 | 2D8 | kept |
| --- | ---: | ---: | ---: | ---: | --- |
| qkv 64 | 0.119 | 0.174 | 0.135 | 0.131 | auto |
| qkv 128 | 0.138 | 0.174 | 0.138 | 0.135 | 2D8 |
| qkv 256 | 0.200 | 0.174 | 0.141 | 0.143 | 2D16 |
| qkv 512 | 0.374 | 0.200 | 0.156 | 0.180 | 2D16 (stage 3 choice) |
| o 64 | 0.101 | 0.124 | 0.093 | 0.103 | 2D16 |
| o 128 | 0.120 | 0.127 | 0.095 | 0.102 | 2D16 |
| o 256 | 0.194 | 0.126 | 0.102 | 0.104 | 2D16 |
| o 512 | 0.466 | 0.141 | 0.114 | 0.123 | 2D16 (stage 3 choice) |
| gate 64 | 0.177 | 0.312 | L1 overflow | 0.235 | auto |
| gate 128 | 0.204 | 0.312 | L1 overflow | 0.242 | auto |
| gate 256 | 0.379 | 0.310 | L1 overflow | 0.255 | 2D8 |
| gate 512 | 0.721 | 0.336 | L1 overflow | 0.378 | minimal (stage 3 choice) |
| up 64 | 0.124 | 0.270 | 0.207 | 0.235 | auto |
| up 128 | 0.200 | 0.272 | 0.209 | 0.236 | auto |
| up 256 | 0.371 | 0.271 | 0.218 | 0.242 | 2D16 |
| up 512 | 0.712 | 0.296 | 0.279 | 0.357 | 2D16 |
| down 64 | 0.260 | 0.323 | 0.250 | 0.244 | 2D8 |
| down 128 | 0.305 | 0.321 | 0.268 | 0.254 | 2D8 |
| down 256 | 0.517 | 0.325 | 0.264 | 0.267 | 2D16 |
| down 512 | 1.340 | 0.333 | 0.267 | 0.310 | 2D16 (stage 3 choice) |

Stage 3 had measured the auto config only against the DRAM-sharded read at
these sizes and the 2D config only from 257 rows; the 256-row chunk with
the auto config is why a 129-token prompt (82.5 ms) cost more than a
257-token prompt (73.5 ms) in the before sweep. Kept as
`PREFILL_KERNEL_TABLES["stage7"]` (`OptimizedPolicy.prefill_kernel_table`,
`"stage3"` restores the old rule); 1024 rows keep the stage-3 winners,
1536 to 4096 rows run `minimal_matmul` as before.

DRAM-sharded weights for prefill (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
with `per_core_M = rows / 32` on a width-sharded activation; the factory
accepts `per_core_M > 1` when `K / in0_block_w` equals the storage core
count): every candidate at 64 to 512 rows for every role fails validation
with `matmul_device_operation.cpp:1347: M == 1`. Exact op-contract blocker;
the DRAM-sharded copy serves decode only.

### 2.3 TTFT before (chip 2, HEAD `f967722683`, `logs/ttft_sweep_before.log`, `ttft_sweep_before.json`, 16:16 to 16:21 ET)

`logs/ttft_sweep.py`: batch 1, book text, `generate(prompt, 1)` wall time
(input preparation, every chunk, final norm, LM head, on-device greedy
sampling, first-token readback), best of three after one warm run; build
and `prepare()` at `f967722683`: 64.0 s `prepare()` (17 prefill lengths).

| prompt tokens | compute rows | TTFT ms |
| ---: | ---: | ---: |
| 32 / 64 | 32 / 64 | 43.5 / 46.0 |
| 96 / 128 | 128 | 54.6 / 54.6 |
| 129 / 160 / 192 / 256 | 256 | 82.5 / 82.4 / 82.4 / 82.5 |
| 257 / 384 / 512 | 512 | 73.5 / 73.5 / 73.8 |
| 513 / 768 / 1024 | 1024 | 113.7 / 113.5 / 113.7 |
| 1025 / 1536 / 2048 | 2048 | 229.2 / 229.9 / 231.0 |
| 2049 / 3072 / 4096 | 4096 | 452.2 / 454.0 / 460.2 |
| 4097 / 6144 / 8192 | 8192 | 1089.6 / 1090.1 / 1097.5 |
| 8193 | 12288 | 1888.5 |
| 16384 | 16384 | 2887.9 |

Chunk-boundary steps: 128 to 129 tokens +51 percent, 512 to 513 +54, 1024
to 1025 +102, 2048 to 2049 +96, 4096 to 4097 +137 percent (a 4097-token
prompt computed two full 4096-row chunks). Levers taken: the per-role kernel
table (section 2.2) for the 256-row bucket, tail buckets for prompts past
one chunk (a 4097-token prompt computes 4096 + 32 rows), and the buckets
1536 and 3072 inside the first chunk.

## 3. Implementation (16:25 to 16:30 ET)

Decode (`tt/model.py`, `tt/generator.py`):

- `TerminalPolicy.lm_head_decode_fidelity` default `hifi2` -> `lofi`
  (`QWEN3_LM_HEAD_DECODE_FIDELITY=hifi2` restores the stage-6 value).
- `Qwen3Model.lm_head_tail_mask`: a persistent `[1, 1, 32, 5120]` bf16 tensor
  in the last split's width-sharded output layout, bf16 min at the 640 padded
  columns, added to the last LM-head split before its `sharded_to_interleaved`
  (2 us). `lm_head_decode` output is therefore sampler-ready; the prefill
  sampler path (`logits_from_tile`) shares it. `logits_all_rows` (readiness,
  `minimal_matmul`) is untouched and still slices `[:vocab_size]`.
- `Qwen3Generator._sampling_args`: `vocab_size = padded_vocab_size`, so the
  common sampler builds no invalid-vocab mask (its slice-add-concat path, 106
  us); `padded_vocab_size`, `max_top_k`, seeds and penalties unchanged (the
  penalties module already sizes itself from `padded_vocab_size`). The
  generator keeps the real `vocab_size` for the host-logits slice.

Prefill (`tt/optimized_decoder.py`):

- `PREFILL_KERNEL_TABLES["stage7"]` and `OptimizedPolicy.prefill_kernel(role,
  rows)`: the per-role winners of section 2.2 for 64 to 512 rows; 1024 rows
  keep the stage-3 rule, longer chunks `minimal_matmul`; the policy field
  `prefill_kernel_table = "stage3"` restores the old rule.
  `_prefill_2d_program_config` takes the `in0_block_w` of the table entry.
- `OptimizedDecoder._rms_norm` override: hidden-width prefill inputs of 64 to
  1024 rows (`prefill_sharded_norm_max_rows`) run `interleaved_to_sharded`,
  the block-sharded `LayerNormShardedMultiCoreProgramConfig` (8 core columns,
  `block_w` 16, `subblock_w` 4, `block_h` = `ceil(rows / 32 / 10)`) and
  `sharded_to_interleaved`; head norms and other rows keep the stage-1
  kernel.

Prompt length and warmup (`tt/model.py`, `tt/generator.py`):

- `PREFILL_BUCKETS` gains 1536 and 3072.
- `prefill_compute_len`: whole 4096-token chunks plus a bucketed tail
  (4097 tokens compute 4096 + 32 rows, before 8192). `prefill_pieces` lists
  the `(start, length, compute_len)` pieces; `Qwen3Model.prefill_hidden` runs
  a fresh row longer than one chunk piece by piece through the whole stack
  (each piece is a decoder continuation at an aligned start, the same
  `PrefillChunk` the decoder's internal loop produced for that chunk; causal
  attention makes the two orders identical) and, with `keep_all_rows=False`,
  returns only the last piece (`prefill_row_offset` gives its first row's
  absolute index; `prefill_forward` gathers `len - 1 - offset`). The readiness
  paths (`return_all_logits`, `prefill_logits_rows`) keep every row
  (concatenated pieces).
- `Qwen3Generator.prepare`: the start-0 buckets as before (10 prefills), then
  every `(chunk start, bucket)` piece for starts 4096 to 36864
  (`DEFAULT_WARMUP_CHUNK_STARTS`, 9 x 10 pieces over a scratch-only page
  table) instead of the 8192 to 40960 full prefills, then the 32-row call and
  the decode buckets. A 40960-token prompt is the ten full pieces, so every
  program of every supported fresh length is compiled before the first capture.
- `tests/test_full_model.py`: the split-sampling test asserts the padded
  columns of the sampler-ready logits are masked and the sampled token is a
  real id; `test_tracked_replay_workload` generates from 8192, 5000 and
  12345-token prompts after the first capture (tail pieces 1024 at 4096 and
  64 at 12288) and still asserts zero program-cache growth; the reduced probe
  skips decode buckets it does not allocate (stage-6 fixture gap, the 16 and
  24 cases raised `KeyError`).

Reduced smoke (2 layers, chip 2, 16:28 ET, `logs/pytest_reduced_smoke.log`):
8 passed, 2 skipped-equivalent failures (the fixture gap above, fixed after
this run), `prepare()` 41.6 s including the 9 x 4 chunk pieces.

## 4. Full-model evidence on chip 2, uncommitted tree over `f967722683` (16:30 ET onward)

`logs/evidence_chain_chip2.sh` (status in `logs/evidence_chain_chip2.status`,
every step under its own `timeout`, readiness provenance in
`../../readiness/readiness_stage7_chip2.json`).

| Step (ET) | Command (chip 2) | Result |
| --- | --- | --- |
| 16:30 | `QWEN3_FULL_MODEL_LAYERS=36 pytest test_full_model.py` | 10 passed in 238 s (`logs/pytest_full_model_36.log`); weights 50 s, `prepare()` 88.6 s (10 buckets, 90 chunk pieces, 32-row call, 5 buckets x 2 variants; stage 6: 64 s). `perf_batch1_128_128.json`: token-out 21.01 ms/token = 47.6 t/s/u (three runs 21.009 / 21.013 / 21.028), TTFT 46.0 ms, teacher-forced 21.04 ms/token, `sampler_traced true`, `lm_head_decode_fidelity lofi`; `host_work_counters.json` 63 + 63 replays, 0 token / 1 position / 1 page-table refresh, repeat bit-equal; `tracked_replay_workload.json` program cache 651 before and after a 3-row prefill, a 32-row prefill and 8192-, 5000- and 12345-token generations after the first capture (tail pieces 1024 at 4096 and 64 at 12288); `multi_row_prefill.json` PCC 1.0000 per row, top-1 equal; `batch_decode_{8,16,24,32}.json` every user bit-equal to the same-bucket solo arm over 32 tokens; `split_sampling_trace.json` passes with the new tail-mask assertions; `seeded_sampling.json` unchanged |
| 16:34 | `run_prefill_check` (AIME24 chat reference) | exit 0: top-1 0.990, top-5 1.000, top-100 1.000 (`logs/run_prefill_check.log`); unchanged from stage 6 with the LoFi LM head and the 256-row kernel set |
| 16:36 | `run_teacher_forcing` (traced decode, traced sampler) | exit 0: top-1 0.980, top-5 1.000, top-100 1.000; TTFT 51.23 ms (stage 6: 80.51), decode 46.81 t/s/u (44.91), e2e 46.17 t/s/u (43.76) (`logs/run_teacher_forcing.log`) |
| 16:37 | `logs/ttft_sweep.py` after | `ttft_sweep_after.json`; table in section 4.1 |
| 16:40 | `run_prefill_resume_check --lengths 1500,2560,5000,16384 --block-sizes 32` at the default thresholds | `status=fail` on `pcc_min` only: 0.9767 at every `block_aligned` start 32 and 0.8756 at every `tile_unaligned` start 68 (PCC mean 0.9996 / 0.9991, top-1 1.000); the `prefix_cache_hit` windows (1472: 28 positions, 2528: 32, 4992: 8, 16352: 32) and the `last_token` starts pass (hit 2528 top-1 0.969, the stage-6 near-tie flip); controls fail at every 257-position window (`../../readiness/prefill_resume_check_default_thresholds.json`). Stage 6 measured 0.9897 and 0.9949 at starts 32 and 68 |
| 16:43 | the same with `--pcc-min-min 0.90 --top1-min 0.93` | `status=fail` (start 68: 0.8756 < 0.90); written to the canonical path by the chain; section 4.2 |

### 4.1 TTFT before and after, sharded-norm variant (chip 2, batch 1, `generate(prompt, 1)` wall time, best of three; superseded by the final table in section 5)

| prompt tokens | rows before -> after | TTFT before -> after (ms) |
| ---: | --- | --- |
| 32 / 64 | 32 / 64 | 43.5 -> 37.3 / 46.0 -> 39.1 |
| 96 / 128 | 128 | 54.6 -> 46.0 / 54.5 -> 46.0 |
| 129 / 160 / 192 / 256 | 256 | 82.5 -> 53.0 (all four) |
| 257 / 384 / 512 | 512 | 73.5 -> 69.1 |
| 513 / 768 / 1024 | 1024 | 113.7 -> 110.2 (1024: 110.6) |
| 1025 / 1536 | 2048 -> 1536 | 229.2 -> 166.1 / 229.9 -> 167.3 |
| 2048 | 2048 | 231.0 -> 228.4 |
| 2049 / 3072 | 4096 -> 3072 | 452.2 -> 328.3 / 454.0 -> 328.8 |
| 4096 | 4096 | 460.2 -> 462.5 |
| 4097 | 8192 -> 4128 | 1089.5 -> 516.7 |
| 6144 | 8192 -> 6144 | 1090.1 -> 790.1 |
| 8192 | 8192 | 1097.5 -> 1084.5 |
| 8193 | 12288 -> 8224 | 1888.5 -> 1172.7 |
| 16384 | 16384 | 2887.9 -> 2856.1 |

The first token is the same before and after at 22 of the 25 lengths; at
256, 257 and 1024 tokens it differs (a different matmul kernel for the
chunk: 2D configs at 256 rows, the per-role table at 512, and the
`in0_block_w`-16 2D config for QKV at 1024 is unchanged but the sharded
norm is new there). Those are TT-versus-TT kernel differences on book text;
the HF-referenced checks above (158-token prompt at 256 rows: top-1 0.99)
are unchanged. Chunk-boundary steps after: 128 to 129 tokens +15 percent
(was +51), 512 to 513 +59 (was +54), 1024 to 1025 +50 (was +102), 2048 to
2049 +44 (was +96), 4096 to 4097 +12 percent (was +137).

### 4.2 Resume-check regression: attribution and decision (16:45 to 17:05 ET)

Anomaly ledger entry:

- Observed anomaly: at the default thresholds the resume check fails on
  `pcc_min` at the two first-window starts of every length (0.9767 at start
  32, 0.8756 at start 68; top-1 1.000, PCC mean 0.9996 / 0.9991); stage 6
  measured 0.9897 and 0.9949 at the same starts.
- Evidence: `../../readiness/prefill_resume_check_default_thresholds_sharded_norm.json`
  (the run on the tree with the sharded prefill norm), the lowered-threshold
  twin `prefill_resume_check_lowered_sharded_norm.json` (fails too: 0.8756 <
  0.90).
- Affected path: `prefill_logits_continuation(ids, start)` prefills the
  prefix (`ids[:start]`, 32 or 68 tokens) as a fresh row at 32 or 128
  compute rows, then the continuation at its exact length; the reference
  runs the whole prompt at 1536 rows and more. The two paths therefore use
  different prefill kernels for the prefix positions, and the window attends
  to the prefix's K/V.
- Controls (`logs/attribution_chain_chip2.sh`, chip 2, L = 1500, block 32,
  default thresholds, `../../readiness/prefill_resume_check_L1500_control_*.json`):

| Control | start 32 `pcc_min` | start 68 `pcc_min` | status |
| --- | ---: | ---: | --- |
| stage-7 tree (sharded norm on, stage-7 kernel table) | 0.9767 | 0.8756 | fail |
| `QWEN3_PREFILL_KERNEL_TABLE=stage3` (sharded norm on) | 0.9767 | 0.8756 | fail |
| `QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS=0` (stage-7 kernel table) | 0.9897 | 0.9949 | pass |
| both restored | 0.9897 | 0.9949 | pass |

- Likely subsystem: the block-sharded prefill RMSNorm (8 core columns,
  partial-sum reduction across cores) on the 32- and 128-row prefix rows
  against the interleaved kernel on the long rows; a bf16-level
  difference in the normalised prefix K/V that one quasi-degenerate window
  position amplifies. The kernel table is bit-neutral here (identical
  numbers with either table).
- Investigation performed: the four controls above; the margin probe
  (per-position PCC of both TT paths against HF float32) crashed on a probe
  harness gap (`decode_buckets=(1,)` without bucket 32 for the 32-row
  warmup call, `logs/resume_margin_probe_1500.log`), fixed, rerun in the
  final chain.
- Resolution: fixed by rejection. `OptimizedPolicy.prefill_sharded_norm_max_rows`
  defaults to 0 (interleaved prefill norms, as in stage 6); the sharded
  path stays as the `QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS=<rows>` experiment
  knob. The TTFT it bought (2 x 67 us per layer at 128 rows, about 4.8 ms
  of the 8.5 ms gain at 128 tokens; 4 ms at 256; 1.7 ms at 1024) is given
  back; the HF-referenced checks on the sharded-norm tree (prefill check
  0.990 / 1.000 / 1.000, teacher forcing 0.980 / 1.000 / 1.000, qualitative
  divergence indices 34, 37, 32, 35, 33, 22) never moved, so this is a
  cross-path consistency rejection by the resume gate, not an HF-level
  accuracy loss. The sharded-norm tree's evidence is kept under
  `logs/variant_sharded_norm/`, `ttft_sweep_after_sharded_norm.json`,
  `perf_batch1_128_128_sharded_norm.json`, `qualitative_sharded_norm/` and
  the `*_sharded_norm.json` readiness files; the final-code evidence below
  supersedes it.

The gate chain that was already running on the sharded-norm tree
(`logs/variant_sharded_norm/gate/`) was left to finish under its own
timeouts (no process with outstanding device work was signalled); the
final chain (`logs/final_chain_chip2.sh`) reruns the evidence chain, the
margin probe and the gate chain on the final code.

## Appendix A. Skill gaps met in this stage

- `optimize/SKILL.md`, "Full-Model Decode Closure" and the checklist item on
  batched prefill: the skill says batched prefill "is part of this target
  for complete model and serving stages" and the checklist requires the
  `prefill_burst` ratio gate, while
  `vllm-integration/references/prefill-features.md` ("Stage split") assigns
  the implementation to stage 10 and the burst measurement to a served
  benchmark that needs the stage-9 server. The two documents disagree on
  which stage owns batched prefill; this stage followed `prefill-features.md`
  (calibration left, implementation deferred) and says so in the README.
- `optimize/SKILL.md`, "LM Head And Sampling": "pad each local logits shard
  to a power-of-two width before TopK when needed" presumes the stock
  multi-core topk is the fast path. On Blackhole with `k = num_splits x
  max_top_k = 128 > 64` the common sampler always takes the
  `topk_large_indices` composite, which the probe shows is faster than every
  power-of-two chunking; the advice would have cost time. The skill should
  name the Blackhole routing rule (`topk.cpp`: multi-core bitonic only for
  `k <= 64`, power-of-two width below 65536) and say that the full-row
  composite is the reference candidate there.
- `optimize/SKILL.md`, "Prefill Latency And TTFT": the DRAM-sharded weight
  advice for "small-M prefill cases" sends the reader to a program config
  whose validation rejects any `M != 1` tile
  (`matmul_device_operation.cpp:1347`). The skill should say the DRAM-sharded
  matmul is decode-only (M = one tile) on this checkout.
- `optimize/SKILL.md`, "Evidence To Leave", `perf_summary.json`: the shape
  lists `decode_ms_per_token_device` but not how to obtain it for a full
  model when the profile must be a reduced variant; this stage scales the
  one-layer window by the layer count and says so. A sentence on the
  accepted derivation would remove the ambiguity.
- `full-model/SKILL.md` and `optimize/SKILL.md` give no guidance on the
  interaction between prefill kernel selection and the resumed-prefill gate:
  a kernel that is accurate against HF can still fail
  `run_prefill_resume_check` because the gate compares two TT paths that use
  different kernels for the prefix (this stage's sharded-norm rejection).
  The resume check in `prefill-features.md` says "a failure here is a cache
  address, position, or history bug, not drift"; a per-row-count kernel
  difference is a third cause the text does not name.
- `tt-device-usage/SKILL.md`: no rule for a device job hung inside an op
  probe with no error (the sharded-input `nlp_concat_heads` candidate). The
  brief's rule (run under `timeout`, never SIGTERM outstanding device work)
  was followed; the skill only covers tt-triage capture before killing,
  which is contradictory with the brief's "let the timeout kill it".
- `qualitative-check/SKILL.md`: the HF control's reuse rule is not stated
  (this stage reused the stage-6 HF outputs because model id, tokenizer,
  prompts and rendering are unchanged; the verdict records it).
- The readiness runners write no provenance (commit, chip); the sidecar
  `readiness_stage7_chip2.json` is this stage's own convention, as in stage
  6. `run_prefill_resume_check` overwrote the canonical artifact with a
  failing run because the chain wrote to that path; the runner has no
  "write only on pass" option.
- Information beyond the skills and the brief: the P300 DRAM bandwidth used
  for the roofline (512 GB/s, carried from stage 3), the `subblock_w <= 4`
  rule of the sharded layer norm under fp32 accumulation
  (`sharded_layernorm_factory_helpers.cpp:85`), the `M == 1` rule of the
  DRAM-sharded matmul, and the `decode_buckets=(1,)` harness gap of the
  stage-6 probe scripts (the 32-row warmup call needs bucket 32).

## 5. Final-code evidence on chip 2, commit `4c9a98d810` (17:20 ET onward)

`logs/final_chain_chip2.sh`: the evidence chain again, the margin probe,
then the gate chain, every step on the committed code (sharded prefill
norm off, LoFi LM head, LM-head mask, stage-7 kernel table, chunk pieces).

| Step (ET) | Command (chip 2) | Result |
| --- | --- | --- |
| 17:20 | `QWEN3_FULL_MODEL_LAYERS=36 pytest test_full_model.py` | 10 passed in 198 s (`logs/pytest_full_model_36.log`); `prepare()` 46.9 s (the 88.6 s of the first build included the compilation of the new program set; the kernel cache serves it now); `perf_batch1_128_128.json`: token-out 21.011 ms/token = 47.6 t/s/u, TTFT 50.68 ms, teacher-forced 21.06 ms/token, `sampler_traced true`, `prefill_sharded_norm_max_rows 0`, `lm_head_decode_fidelity lofi`, `checkout_git_sha 4c9a98d810`; `tracked_replay_workload.json` program cache 639 before and after the post-capture workload (3-row, 32-row, 8192-, 5000- and 12345-token prompts); `host_work_counters.json`, `split_sampling_trace.json`, `multi_row_prefill.json`, `batch_decode_{8,16,24,32}.json`, `seeded_sampling.json` pass as in section 4 |
| 17:24 | `run_prefill_check` | exit 0: top-1 0.990, top-5 1.000, top-100 1.000 |
| 17:25 | `run_teacher_forcing` | exit 0: top-1 0.980, top-5 1.000, top-100 1.000; TTFT 55.17 ms (stage 6: 80.51), decode 46.74 t/s/u (44.91), e2e 46.01 (43.76) |
| 17:27 | `logs/ttft_sweep.py` after | `ttft_sweep_after.json`, table below |

TTFT before (`f967722683`) and after (`4c9a98d810`), batch 1, best of three
(the sharded-norm variant of section 4.1 in brackets):

| prompt tokens | rows before -> after | TTFT before -> after (ms) |
| ---: | --- | --- |
| 32 / 64 | same | 43.5 -> 42.6 (37.3) / 46.0 -> 44.3 (39.1) |
| 96 / 128 | 128 | 54.6 -> 51.0 (46.0) / 54.5 -> 51.0 (46.0) |
| 129 / 160 / 192 / 256 | 256 | 82.5 -> 57.1 (53.0), all four |
| 257 / 384 / 512 | 512 | 73.5 -> 72.0 (69.1) |
| 513 / 768 / 1024 | 1024 | 113.7 -> 112.5 (110.2) |
| 1025 / 1536 | 2048 -> 1536 | 229.2 -> 166.6 / 229.9 -> 167.0 |
| 2048 | 2048 | 231.0 -> 231.3 |
| 2049 / 3072 | 4096 -> 3072 | 452.2 -> 326.7 / 454.0 -> 329.0 |
| 4096 | 4096 | 460.2 -> 463.0 |
| 4097 / 6144 | 8192 -> 4128 / 6144 | 1089.5 -> 524.3 / 1090.1 -> 788.1 |
| 8192 | 8192 | 1097.5 -> 1086.6 |
| 8193 | 12288 -> 8224 | 1888.5 -> 1171.2 |
| 16384 | 16384 | 2887.9 -> 2859.2 |

First token identical before and after at 23 of 25 lengths. At 256 and
1024 tokens it differs; the kernel-table controls of section 4.2 are
bit-neutral and the 1024-row chunk kernels are unchanged, so the only
changed op on those first-token paths is the LoFi LM head
(`logits_from_tile`): near-tie flips of the lower-fidelity logits, TT
against TT on book text, controlled by the unchanged HF-referenced checks
(prefill top-1 0.990, teacher forcing 0.980, qualitative divergence indices
equal or later). Chunk-boundary steps after:
128 to 129 tokens +12 percent (was +51), 512 to 513 +56 (was +54), 1024 to
1025 +48 (was +102), 2048 to 2049 +41 (was +96), 4096 to 4097 +13 percent
(was +137). The remaining steps at 512 to 513 and 1024 to 1025 are the
bucket sizes themselves (a finer bucket such as 768 costs one more warm
prefill and about 20 programs each and was not added).

### 5.1 Resumed prefill at the default thresholds, final code (17:29 ET, `logs/run_prefill_resume_check_default.log`)

`run_prefill_resume_check --lengths 1500,2560,5000,16384 --block-sizes 32`
(PCC mean >= 0.99, PCC min >= 0.98, top-1 >= 0.98, natural text), output
`../../readiness/prefill_resume_check_default_thresholds.json`:

| L | start (kind) | positions | PCC mean | PCC min | top-1 | last top-1 equal | pass | controls (`no_history`, `fresh_suffix`: PCC mean, top-1) |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1500 | 32 (block_aligned) | 257 | 0.9997 | 0.9897 | 1.000 | yes | pass | 0.934 / 0.899, 0.934 / 0.903 |
| 1500 | 68 (tile_unaligned) | 257 | 0.9997 | 0.9949 | 1.000 | yes | pass | 0.782 / 0.673, 0.955 / 0.907 |
| 1500 | 1472 (prefix_cache_hit) | 28 | 0.9999 | 0.9998 | 1.000 | yes | pass | |
| 1500 | 1499 (last_token) | 1 | 0.9999 | 0.9999 | 1.000 | yes | pass | |
| 2560 | 32 / 68 | 257 | 0.9997 | 0.9897 / 0.9949 | 1.000 | yes | pass | as above |
| 2560 | 2528 (prefix_cache_hit) | 32 | 0.9999 | 0.9997 | 0.969 | yes | fail (top-1 31/32) | |
| 2560 | 2559 (last_token) | 1 | 1.0000 | 1.0000 | 1.000 | yes | pass | |
| 5000 | 32 / 68 | 257 | 0.9997 | 0.9897 / 0.9949 | 1.000 | yes | pass | 0.934 / 0.899, 0.934 / 0.903; 0.782 / 0.669, 0.955 / 0.903 |
| 5000 | 4992 (prefix_cache_hit) | 8 | 0.9999 | 0.9999 | 1.000 | yes | pass | |
| 5000 | 4999 (last_token) | 1 | 0.9999 | 0.9999 | 1.000 | yes | pass | |
| 16384 | 32 / 68 | 257 | 0.9997 | 0.9897 / 0.9949 | 1.000 | yes | pass | as above |
| 16384 | 16352 (prefix_cache_hit) | 32 | 0.9998 | 0.9989 | 1.000 | yes | pass | |
| 16384 | 16383 (last_token) | 1 | 0.9999 | 0.9999 | 1.000 | yes | pass | |

`status=fail` on one item: `L=2560 s=2528 (prefix_cache_hit): top1_agreement
0.9688 < 0.98`, the 32-position cache-hit window with the one BF16 near-tie
flip that stage 6 located (position 2531: HF float32 top-2 margin 0.032, both
TT paths 0.016, `../full_model/resume_margin_probe_L2560_s2528_fp32.json`);
every PCC gate passes, including the two starts that the sharded-norm tree
failed (0.9897 and 0.9949 again, the stage-6 values). The new lengths cover
the 1536 bucket (L = 1500: the fresh reference at 1536 rows against
continuations at exact lengths) and the chunk-piece path (L = 5000: fresh
reference as 4096 + a 1024-row tail piece against a 4932-token continuation
at start 68 that the decoder chunks internally) at every start. This is the
failure mode the operator's instruction names; the lowered-threshold run
(`--pcc-min-min 0.90 --top1-min 0.93`, the stage-6 thresholds) is kept
beside it as the canonical artifact (section 5.2) and the margin probe
reruns 2560:2528 on the final code.

### 5.2 Resumed prefill, lowered thresholds (17:32 ET, `logs/run_prefill_resume_check_lowered.log`)

The same lengths with `--pcc-min-min 0.90 --top1-min 0.93` (the stage-6
thresholds set at this model's measured TT-versus-HF per-position floor):
`status=pass` at all 16 starts with the numbers of section 5.1 (the only
difference is the 2528 hit window's top-1 0.969 passing the 0.93 bar);
`../../readiness/prefill_resume_check.json` (canonical, final code,
`4c9a98d810`).

### 5.3 Margin probes on the final code (17:40 ET, `logs/resume_margin_probe_final.log`)

`logs/resume_margin_probe.py 1500:68 1500:32 2560:2528` (the whole window
`[start, L)` rather than the runner's 257 positions; fresh reference, resumed
candidate and HF float32 logits per position):

| L:start | window | top-1 ref=cand | top-1 ref=HF / cand=HF | PCC mean ref/HF, cand/HF | PCC min ref/cand | worst ref/cand positions (ref/HF, cand/HF, HF margin) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1500:68 | 1432 | 0.992 | 0.920 / 0.920 | 0.9867 / 0.9867 | 0.9795 | 558 (0.983, 0.951, 13.3), 482 (0.899, 0.927, 5.65), 605 (0.987, 0.978, 10.3) |
| 1500:32 | 1468 | 0.989 | 0.922 / 0.921 | 0.9866 / 0.9864 | 0.8666 | 482 (0.899, 0.902, 5.65), 293 (0.938, 0.842, 22.6), 558 (0.983, 0.941, 13.3) |
| 2560:2528 | 32 | 0.969 | 0.781 / 0.750 | 0.9959 / 0.9960 | 0.9997 | one flip at 2531: HF margin 0.032, both TT margins 0.016 |

Reading: the resumed path is as close to HF as the fresh path at every
statistic (same top-1 rate, same mean PCC, the same positions worst for
both), the worst TT-against-TT positions are positions where each TT path
is itself 0.84 to 0.94 against HF (quasi-degenerate positions of this book
text, as stage 6 found), and the hit-window flip is a near tie far below
the bf16 logit spacing. Artifacts
`resume_margin_probe_L1500_s68_fp32.json`, `resume_margin_probe_L1500_s32_fp32.json`,
`resume_margin_probe_L2560_s2528_fp32.json`.

### 5.4 Gate chain on the final code (17:43 to 18:07 ET, `logs/gate_chain_chip2.status`)

| Step (ET) | Command (chip 2) | Result |
| --- | --- | --- |
| 17:43 | `QWEN3_EVIDENCE_SUFFIX=tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 pytest -k 'test_split_sampling_trace or test_host_work_counters or test_tracked_replay_workload or test_batch_decode' --timeout 1800` | 7 passed in 1261 s, no tracker error (`logs/pytest_full_model_36_tracked.log`); `*.tracked.json` at `4c9a98d810`: program cache 639 before and after the post-capture workload, 6 decode and 5 sampling captures, 0 eager decode steps, every batch bucket bit-equal to solo, 63 + 63 replays for the 64-token generation |
| 18:04 | stage-3 positive control under the same environment | raised as expected ("Found 1 device buffer(s) still alive before trace replay"), `logs/positive_control.log` |
| 18:04 | `TT_METAL_WATCHER=10` on the 2-layer reduced model: split-sampling, host-work, multi-row prefill, representative workload | 4 passed in 17.8 s (kernel cache warm; the first run on the sharded-norm tree took 82 s); 4 watcher dumps, 0 fatal-pattern matches (`watcher/summary.json`, `watcher/generated/watcher/watcher.log`, `*.watcher.json`) |
| 18:05, 18:06 | `python -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_stage7_reduced logs/profile_reduced.py` | the first two attempts captured (`PROFILE_OK`) but `process_ops_logs.py` asserted "Device data missing: Op N not present in cpp_device_perf_report.csv" for an `EmbeddingsDeviceOperation` (a one-core op whose markers the full profiler buffer of core 0 dropped after the 32-row warmup's 96 one-core ops); the profiled process now skips that warmup (`warmup_multi_row=False`, a new generator knob, default `True`) and calls `ttnn.ReadDeviceProfiler` after `prepare()`; third attempt exit 0 (`logs/profile_reduced.log`), tables in `profile/` (section 5.5) |

### 5.5 Reduced-variant profile, final code (18:07 ET, chip 2)

`profile/decode.txt` (one blocking replay of the decode trace plus the
sampling trace, one real layer, 80 device ops, 2,862 us; stage 6: 83 ops,
3,720 us) and `profile/prefill_128.txt` (warmed 128-token prompt through
embedding, the layer, gather, final norm, LM head, eager greedy sampling).

Decode, terminal path (layer rows as in stage 3: QKV 58, O 41, gate 113, up
70, down 110 us, SDPA 18 us at position about 130):

| Op | Stage 6 | Now |
| --- | ---: | ---: |
| LM head 9 x `32 x 4096 x 16384` + `32 x 4096 x 5120` | 9 x 230 + 79 = 2,149 us, `HiFi2`, FLOP-bound (84 percent of HiFi2 FLOPs, 57 percent of DRAM) | 9 x 150 + 49 = 1,400 us, `LoFi BF16 x BFP8`, DRAM-bound at 87 percent |
| tail mask | sampler: slice 53 + slice 1 + add 3 + concat 53 | LM head: one sharded add (2 us, inside the last split's row) |
| `TopkRoutePrep` / `TopkLargeIndices` / `TopkRouteFinish` | 51 / 648 / 22 | 50 / 648 / 21 |
| concat of the 10 pieces, 10 x `sharded_to_interleaved` | 54, 10 x 5 | 52, 10 x 4 |
| total window | 3,720 us | 2,862 us |

Prefill at 128 rows, one layer (rows that changed): QKV `128 x 4096 x 6144`
119 us on 96 cores (auto) -> 118 us on 44 cores (2D `in0_block_w` 8); O
`128 x 4096 x 4096` 102 -> 78 us (2D 16); down `128 x 12288 x 4096` 289 ->
236 us (2D 8); gate and up unchanged (auto, 182 and 181 us); norms 85 and 84
us on 4 cores (interleaved, by the rejection in 4.2); head split 80 and
concat heads 54 us on 4 cores. Per-layer device time 1.29 -> 1.21 ms; the
256-row chunk's larger saving is in the TTFT table of section 5.

Performance accounting (`perf_summary.json`, `logs/write_perf_summary.py`):
roofline 13.99 ms per token (6.47 GB decoder weights + 0.66 GB LM head +
24 MB KV at 512 GB/s), device 20.83 ms (36 x 513 us + 2.35 ms terminal),
end to end 21.01 ms, host gap 0.18 ms. The gate script
(`07-optimized-full-model.check.sh`) exits 0 on this tree
(`logs/gate_07_optimized_full_model.log`).

## 6. Stage review 1 and fixes (18:09 to 18:30 ET)

Commits before the review: `4c9a98d810` (sources, tests, probes, before
sweep), `411ff9e9d8` (final-code evidence, documents, chain scripts,
`warmup_multi_row` generator knob with default `True`; the tracked, watcher
and readiness runs executed on `4c9a98d810`, only the third profile attempt
on the knob code, whose default path is unchanged), `e78f68b42c`
(`doc/context_contract.json` `optimized_full_model` section, validated by
`scripts/check_context_contract.py`: 40960 retained, no capability change,
the 320 KB LM-head mask the only new persistent allocation; committed during
the review, documentation only).

Review 1 (`logs/stage_review_1.md`, fresh read-only xhigh subagent, 16
minutes): `more-work-needed`, no correctness defect; three P2 items.

- P2-1 readiness sidecar `checkout_git_sha` was the stage-6 hash (created by
  the pre-commit chain, appended by the final chain) and the mesh label said
  N150: fixed in `../../readiness/readiness_stage7_chip2.json` (hash
  `4c9a98d810`, label names the P300 chip and the runner argument) and
  `logs/write_sidecar.py` now refreshes both on every append. The watcher
  summary's model string ("sharded prefill norms") was stale too; fixed.
- P2-2 gate and up prefill rows on the auto config at `in0_block_w` 2 (182
  and 181 us at 128 rows, 30 percent of the layer) with no 1D multicast
  candidate: probed in section 6.1.
- P2-3 sharded prefill norm rejected without the uniform-kernel adaptation:
  probed in section 6.1.

Other concerns adopted: the first-token flip explanation (section 5,
corrected above); the review's note on the 257-position window versus the
whole-window margin probe is in section 5.3; the eager first sampler step
after prefill (about 23 ops inside TTFT) is recorded as a lever below.

### 6.1 Review-1 probes (18:27 to 18:35 ET, `logs/probe_review1.log`, `probe_candidates.{norm2,prefill_mm2}.json`)

Sharded RMSNorm at long and non-factoring row counts (8 core columns, block
`block_h x 16`, subblock 4, `interleaved_to_sharded` and back included;
`exact` = the core rows cover the tile rows exactly, otherwise the last
core row holds a partial shard):

| rows | interleaved | block_h 5 | block_h 6 | block_h 7 and up |
| ---: | ---: | ---: | ---: | --- |
| 1440 (45 tile rows) | 136 us | 115 us (9 rows, exact) | 118 us (8 rows, partial) | L1 overflow `dataflow_buffer.cpp:2613` |
| 1472 (46) | 137 us | 117 us (10 rows, partial), PCC 1.0000 | 120 us | overflow |
| 1536 (48) | 142 us | 120 us (10 rows, partial) | 122 us (8 rows, exact) | overflow |
| 2048 (64) | 182 us | not legal (needs 13 rows) | not legal | overflow (7 and 8), `layernorm` validation (9) |
| 4096 (128) | 360 us | | | overflow for every block_h 13 to 15; two 2048-row pieces with block 8 overflow too |

So the kernel is legal and faster for any row count up to `6 x 10 x 32 =
1920` rows (partial last shards are fine and bit-identical: the per-row
reduction runs across the eight columns regardless of `block_h` or the row
count). Above 1920 rows the only sharded form is pieces of at most 1920
rows (slice, norm, concat). Adopted: `OptimizedDecoder._rms_norm` normalises
every hidden-width prefill input with the sharded kernel, directly up to
1920 rows and in 1920-row pieces above (`prefill_sharded_norm_max_rows`
= 1920; 0 restores the interleaved kernel). This makes the prefix of a
continuation, the continuation itself and the fresh reference use one
kernel at every length, which is what the resume gate measures; section 6.2
reruns it.

1D multicast projections (`MatmulMultiCoreReuseMultiCast1DProgramConfig`,
`mcast_in0`, full 11x10 grid, `per_core_N = ceil(Nt / cores)`; best of
`in0_block_w` 16 / 8 / 4 and the largest legal output subblock; PCC against
torch equal to the other kernels):

| role, rows | stage-7 table so far | best 1D (cores, `in0_block_w`, subblock) | kept |
| --- | ---: | ---: | --- |
| qkv 64 | auto 0.119 | 0.099 (110, 4, 2x2) | 1D |
| qkv 128 | 2D8 0.135 | 0.102 (110, 4, 2x2) | 1D |
| qkv 256 | 2D16 0.141 | 0.121 (110, 8, 2x2) | 1D |
| gate 64 | auto 0.177 | 0.173 (64, 16, 2x2) | 1D |
| gate 128 | auto 0.204 | 0.183 (64, 4, 2x2) | 1D |
| gate 256 | 2D8 0.255 | 0.233 (110, 16, 1x4) | 1D |
| up 64 | auto 0.124 | 0.126 (110, 8, 1x4) | auto |
| up 128 | auto 0.200 | 0.137 (64, 8, 2x2) | 1D |
| up 256 | 2D16 0.218 | 0.177 (110, 8, 1x4) | 1D |

Every 1D candidate at `in0_block_w` 16, 8 and 4 was measured for each
(role, rows, cores); the full list is in `probe_candidates.prefill_mm2.json`.
The rows the review named (gate and up at 128 rows, `in0_block_w` 2 on the
auto config) now run 1D configs with `in0_block_w` 4 and 8 on 64 cores.
O and down keep the 2D winners (not re-probed: their 2D rows were already at
or above the 1D results of the same shape class). At 128 rows the table
change saves 117 us per layer (4.2 ms of TTFT), at 256 rows 84 us per layer.

### 6.2 Reruns on the review-1 code (uniform sharded norm, 1D kernels; 18:29 ET onward, `logs/review1_chain_chip2.sh`)

The round-1 logs and artifacts are kept under `logs/round1/`,
`ttft_sweep_after_round1.json`, `perf_batch1_128_128_round1.json` and
`../../readiness/prefill_resume_check_default_thresholds_round1.json`.

| Step (ET) | Command (chip 2) | Result |
| --- | --- | --- |
| 18:29 | `QWEN3_FULL_MODEL_LAYERS=36 pytest test_full_model.py` | 10 passed in 198 s; `prepare()` 46.7 s; `perf_batch1_128_128.json`: token-out 21.00 ms/token = 47.6 t/s/u, TTFT 42.1 ms, teacher-forced 21.04 ms/token; program cache 666 flat across the post-capture workload; multi-row and batch buckets as before |
| 18:32 | `run_prefill_check` | top-1 0.990, top-5 1.000, top-100 1.000 |
| 18:34 | `run_teacher_forcing` | top-1 0.980, top-5 1.000, top-100 1.000; TTFT 48.24 ms, decode 46.76 t/s/u, e2e 46.18 |
| 18:36 | TTFT sweep | table below |

TTFT (ms), before `f967722683` -> round 1 (`4c9a98d810`, interleaved norms,
2D/auto table) -> round 2 (uniform sharded norm, 1D table):

| prompt tokens | rows | before -> round 1 -> round 2 |
| ---: | ---: | --- |
| 32 / 64 | 32 / 64 | 43.5 -> 42.6 -> 37.2 / 46.0 -> 44.3 -> 38.4 |
| 96 / 128 | 128 | 54.6 -> 51.0 -> 42.0 / 54.5 -> 51.0 -> 41.9 |
| 129 to 256 | 256 | 82.5 -> 57.1 -> 50.1 |
| 257 to 512 | 512 | 73.5 -> 72.0 -> 69.1 |
| 513 to 1024 | 1024 | 113.7 -> 112.5 -> 109.9 |
| 1025 / 1536 | 1536 | 229.2 -> 166.6 -> 166.3 / 229.9 -> 167.0 -> 166.6 |
| 2048 | 2048 | 231.0 -> 231.3 -> 240.3 |
| 2049 / 3072 | 3072 | 452.2 -> 326.7 -> 342.3 / 454.0 -> 329.0 -> 344.1 |
| 4096 | 4096 | 460.2 -> 463.0 -> 476.4 |
| 4097 / 6144 | 4128 / 6144 | 1089.5 -> 524.3 -> 533.4 / 1090.1 -> 788.1 -> 816.2 |
| 8192 / 8193 | 8192 / 8224 | 1097.5 -> 1086.6 -> 1116.0 / 1888.5 -> 1171.2 -> 1193.5 |
| 16384 | 16384 | 2887.9 -> 2859.2 -> 2909.3 |

Reading: every prompt up to 1920 rows gains (the primary 128-token target
54.5 -> 41.9 ms, 23 percent; the 256-row band 82.5 -> 50.1, 39 percent);
chunks longer than 1920 rows pay the piece-wise norm (slice, three sharded
norms, concat instead of one interleaved kernel): +2 to +5 percent at 2048
to 16384 tokens (4096 rows: 463 -> 476 ms). First token identical to the
before run at 22 of 25 lengths (256, 257, 1024 differ: LoFi LM-head
near-ties as in section 5, with the 257 case new in this round; HF-referenced
checks unchanged). Kept because the batch-1 short-prompt TTFT is the
stage's primary target and the long-prompt cost is bounded and explained;
the lever to remove it is a sharded norm kernel that holds more than six
tile rows per core (an `inplace` program config, or a two-stage reduction)
rather than more pieces.

Resumed prefill on the round-2 code (18:38 and 18:41 ET,
`logs/run_prefill_resume_check_default.log`, `logs/run_prefill_resume_check_lowered.log`):

| L | start (kind) | positions | PCC mean | PCC min | top-1 | pass (default thresholds) | controls (`no_history`, `fresh_suffix`: PCC mean / top-1) |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1500 / 2560 / 5000 / 16384 | 32 (block_aligned) | 257 | 0.9997 | 0.9970 | 0.996 | pass | 0.934 / 0.895, 0.934 / 0.899 |
| 1500 / 2560 / 5000 / 16384 | 68 (tile_unaligned) | 257 | 0.9996 | 0.9886 | 0.996 | pass | 0.786 / 0.681, 0.955 / 0.914 (0.911 at L = 5000) |
| 1500 | 1472 (prefix_cache_hit) | 28 | 0.9999 | 0.9998 | 1.000 | pass | |
| 2560 | 2528 (prefix_cache_hit) | 32 | 0.9999 | 0.9998 | 1.000 | pass | |
| 5000 | 4992 (prefix_cache_hit) | 8 | 0.9999 | 0.9998 | 1.000 | pass | |
| 16384 | 16352 (prefix_cache_hit) | 32 | 0.9998 | 0.9995 | 0.969 | fail (top-1 31/32) | |
| every L | last_token | 1 | 0.9999 to 1.0000 | same | 1.000 | pass | |

`status=fail` on the one item `L=16384 s=16352 (prefix_cache_hit):
top1_agreement 0.9688 < 0.98`: the same failure mode as round 1 (one
near-tie flip in a 32-position hit window), at a different window (2528
now agrees at 32 of 32, 16352 at 31 of 32); every PCC gate passes at all 16
starts, so the uniform sharded norm meets the keep condition of review 1.
The lowered-threshold run (`--pcc-min-min 0.90 --top1-min 0.93`) passes at
all 16 starts and is the canonical `../../readiness/prefill_resume_check.json`.
The margin probe of the 16352 window runs after the gate chain (section 6.3).

### 6.3 Margin probes on the round-2 code (18:49 ET, `logs/resume_margin_probe_final.log`)

| L:start | window | top-1 ref=cand | top-1 ref=HF / cand=HF | PCC mean ref/HF, cand/HF | PCC min ref/cand (round 1) | worst ref/cand positions (ref/HF, cand/HF, HF margin) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1500:68 | 1432 | 0.986 | 0.918 / 0.918 | 0.9867 / 0.9867 | 0.9886 (0.9795) | 293 (0.914, 0.931, 22.6), 301 (0.952, 0.931, 3.5), 245 (0.979, 0.965, 14.8) |
| 1500:32 | 1468 | 0.986 | 0.919 / 0.922 | 0.9866 / 0.9865 | 0.9558 (0.8666) | 293 (0.914, 0.836, 22.6), 534 (0.996, 0.988, 14.3), 350 (0.857, 0.828, 15.0) |
| 2560:2528 | 32 | 1.000 | 0.750 / 0.750 | 0.9959 / 0.9959 | 0.9998 (0.9997) | no flip (round 1: one at 2531) |

With the uniform sharded norm the resumed and fresh paths agree more closely in PCC (minimum 0.9886 and 0.9558 against 0.9795 and 0.8666) and slightly less in whole-window top-1 (0.986 against 0.992 and 0.989), while both stay at the same distance from HF; the flipped positions are near ties (HF margins 0.07 to 0.9 at TT
margins 0.02 to 0.2). The 16352 window that fails top-1 at 31 of 32 is
probed in section 6.4 (an HF forward over 16384 tokens on the CPU).

### 6.4 Gate chain, margin probe and profile on the round-2 code (18:52 to 19:21 ET; read from disk at 09:35 ET Oct 30 after the cap)

The stage cap (23:50 ET) passed while this chain ran unattended; every step
finished on its own under its timeout, nothing was signalled. Steps after
the cap: none on the device; the documents, perf tables, summaries and
commits below were produced from the on-disk results at 09:35 to 10:20 ET
on Oct 30 (the coordinator's 90-minute wrap-up).

| Step (ET) | Command (chip 2) | Result |
| --- | --- | --- |
| 18:52 | tracked gate (`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, split-sampling, host-work, representative workload, four batch buckets, `--timeout 1800`) | 7 passed in 1208 s, no tracker error (`logs/pytest_full_model_36_tracked.log`); `*.tracked.json`: program cache 666 before and after the post-capture workload, 6 decode and 5 sampling captures, 0 eager decode steps, every batch bucket bit-equal to the same-bucket solo arm |
| 19:13 | stage-3 positive control | raised as expected (`logs/positive_control.log`) |
| 19:13 | `TT_METAL_WATCHER=10`, 2-layer reduced model, four tests | 4 passed in 21 s, 6 watcher dumps, 0 fatal-pattern matches (`watcher/summary.json`, `watcher/generated/watcher/watcher.log`) |
| 19:13 | `python -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_stage7_reduced logs/profile_reduced.py` | exit 0 first time (`logs/profile_reduced.log`); tables regenerated in `profile/` (below) |
| 19:14 | `logs/resume_margin_probe.py 16384:16352` (HF float32 forward over 16384 tokens on the CPU) | `resume_margin_probe_L16384_s16352_fp32.json`: 32 positions, top-1 ref = cand 0.969 (one flip at 16353: HF margin 0.195, TT margins 0.047 and 0.094; the resumed path agrees with HF there and the fresh reference does not), ref = HF 0.844, cand = HF 0.875, PCC min ref/cand 0.9995 |

Round-2 profile (`profile/decode.txt`, `profile/prefill_128.txt`, one real
layer): decode window unchanged from section 5.5 (LM head 10 x `LoFi BF16 x
BFP8` 1,398 us, sampler 878 us, 36 x 511 us layers, device 20.73 ms per
token against 21.00 ms end to end, host gap 0.27 ms, roofline 13.99 ms;
`perf_summary.json` regenerated). Prefill at 128 rows: both RMSNorms 8 and
9 us on 32 cores (85 and 84 us before), QKV `128 x 4096 x 6144` 84 us on 96
cores (1D, 65 percent of DRAM bandwidth; 118 us before), O 78 us (2D), gate
`128 x 4096 x 12288` 167 us on 64 cores (1D; 182), up 123 us on 64 cores
(1D; 181), down 236 us (2D; 289 in stage 6). Per-layer device time 1.29 ms
(stage 6) -> 1.21 (round 1) -> 0.93 ms.

Commit of this round: see section 7.

## 7. Commits and stage review 2 (09:35 ET Oct 30 onward, after the cap)

Local commits on `hous/qwen3-8b-autoport` (never pushed):

| Commit | Content |
| --- | --- |
| `4c9a98d810` | sources and tests: LoFi LM head, LM-head vocab mask, sampler with padded vocab, per-role prefill kernel table, chunk-piece prefill with tail buckets, chunk-piece warmup, reduced-fixture skip; candidate probes and the before sweep |
| `411ff9e9d8` | round-1 evidence on chip 2, sharded-norm attribution and rejection, profile, `perf_summary.json`, documents, chain scripts, `warmup_multi_row` generator knob |
| `e78f68b42c` | `doc/context_contract.json` `optimized_full_model` section |
| `22d4eaee7e` | review-1 fixes: uniform block-sharded prefill norm (1920-row pieces), 1D multicast prefill kernels, sidecar provenance; round-2 evidence on chip 2 (sections 6.1 to 6.4), documents |

Stage gate on `22d4eaee7e`: `MODEL_DIR=models/autoports/qwen_qwen3_8b bash
.../07-optimized-full-model.check.sh` exit 0
(`logs/gate_07_optimized_full_model.log`: no degenerate output).

Stage review 2: launched at 09:37 ET Oct 30 (read-only, fresh xhigh
subagent, `logs/stage_review_2.md`); verdict recorded below when it returns.

Stage review 2 (`logs/stage_review_2.md`, 09:37 to 09:47 ET Oct 30, HEAD
`22d4eaee7e`): `more-work-needed`, one P2 (the O and down prefill rows were
never measured with the 1D multicast kernel) and document nits (sidecar
note, watcher summary string and README watcher values, stale
teacher-forcing and accounting numbers, context-contract placeholder, the
6.3 sentence), all fixed in the commit below.

### 7.1 Review-2 P2: 1D multicast candidates for O and down (09:46 to 09:50 ET Oct 30, `logs/probe_review2.log`, `probe_candidates.review2_prefill_mm2.json`)

| role, rows | current table (ms) | best 1D (ms; cores, `in0_block_w`, subblock) |
| --- | ---: | ---: |
| o 64 | 2D16 0.093 | 0.068 (110, 4, 2x2) |
| o 128 | 2D16 0.095 | 0.072 (96, 8, 2x2) |
| o 256 | 2D16 0.102 | 0.100 (64, 16, 2x2) |
| down 64 | 2D8 0.244 | 0.161 (96, 4, 2x2) |
| down 128 | 2D8 0.254 | 0.167 (64, 8, 2x2) |
| down 256 | 2D16 0.264 | 0.241 (110, 16, 2x2) |

The reviewer was right: the 1D kernel wins every O and down row at 64 to
256 rows (about 110 us per layer at 128 rows, roughly 4 ms of the 42 ms
TTFT; 48 us per layer at 256 rows). Not integrated: a kernel change
alters prefill numerics and needs the evidence and gate chains again
(about 45 minutes of chip time), which does not fit the post-cap wrap-up
window. Open item for the next stage or a follow-up commit: set
`PREFILL_KERNEL_TABLES["stage7"]["o_proj"]` to `{64: ("1d", (110, 4, 2, 2)),
128: ("1d", (96, 8, 2, 2)), 256: ("1d", (64, 16, 2, 2))}` and
`["down_proj"]` to `{64: ("1d", (96, 4, 2, 2)), 128: ("1d", (64, 8, 2, 2)),
256: ("1d", (110, 16, 2, 2))}`, then rerun `logs/review1_chain_chip2.sh`
(the full evidence, margin-probe and gate chains). The review-2 verdict
stands as `more-work-needed` with this one open P2; the brief's rule for a
P2 after the last review applies (recorded, verdict unchanged).

Hours used: 15:45 ET Oct 29 to 19:21 ET Oct 29 attended and unattended
device work (3 h 36 min; the gate chain, margin probe and profile finished
unattended before 19:22 ET), then 09:35 to 10:00 ET Oct 30 for the wrap-up
(25 min). No device work after the cap except the 4-minute O/down probe at
09:46 ET Oct 30.
