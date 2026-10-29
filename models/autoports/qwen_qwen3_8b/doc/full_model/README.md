# Qwen3-8B full model

Status: accuracy, qualitative, resumed-prefill and serving-contract gates
pass on one Blackhole P300 chip (1x1 mesh, p150 mesh graph descriptor); the
stage gate script exits 0. Stage review 1 returned `more-work-needed`
(`work_log.md` section 8); the host-side findings are fixed in commit
history below, and the device-side follow-ups (listed under "Open items")
wait for an operator reset of chips 0 and 1. Stages 4 and 5 (multichip)
were skipped by the operator; one chip is the target topology. This stage
claims nothing about vLLM serving (stage 9) beyond `serving_contract.json`.

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient), KV
(key/value), LM (language model), RoPE (rotary position embedding), SDPA
(scaled dot-product attention), DRAM (device dynamic RAM), L1 (per-core
local memory), BFP8/BFP4 (block floating point, 8- and 4-bit mantissa),
LoFi/HiFi2/HiFi4 (tt-metal math fidelity levels), TTFT (time to first
token), t/s/u (tokens per second per user), APC (automatic prefix caching),
ET (Eastern Time).

## Headline numbers (batch 1, real weights, chip 0, commit in `work_log.md`)

| Metric | Value | Workload | Artifact |
| --- | ---: | --- | --- |
| Readiness teacher forcing, traced decode and traced sampler (AIME24 chat, 158-token prompt, 100 tokens) | TTFT 80.4 ms, decode 45.2 t/s/u (22.1 ms/token), top-1 0.980, top-5 1.000, top-100 1.000 | `run_teacher_forcing`, commit `2eaaa0b3f9` (later commits change seed registration, warmup and ownership code paths that this run does not exercise; rerun pending the device reset) | `logs/run_teacher_forcing_final.log` |
| Readiness prefill check | top-1 0.990, top-5 1.000, top-100 1.000 | `run_prefill_check`, commit `2eaaa0b3f9` | `logs/run_prefill_check_final.log` |
| Token-out TTFT, 128-token prompt | 54.5 ms | prefill + first sampled token readback | `perf_batch1_128_128.json` |
| Token-out decode, 128/128 | 21.79 ms/token = 45.9 t/s/u, superseded: measured with the sampler on its eager path (review 1 P1, `work_log.md` 8.1); to be re-measured on the traced sampler after the device reset | 128 generated tokens | `perf_batch1_128_128.json` |

Teacher-forcing and token-out decode do not differ in this generator: both
replay the decode trace and the sampling trace and read one token back; the
teacher-forced loop only adds a 128-byte token write when the forced token
differs from the prediction. The readiness teacher-forcing run never sets a
request seed, so its sampler ran through the trace; the 128/128 pytest
numbers were taken after a seeded test in the same process and are not
trace-verified for the sampler.

## What the full model is

`tt/model.py` (`Qwen3Model`) wraps the stage-3 `OptimizedDecoder` stack
(36 layers, `DEFAULT_POLICY`, unchanged source) with:

- token embedding: BF16 row-major `[151936, 4096]` (`ttnn.embedding` requires
  BF16 weights), 1.245 GB;
- final RMSNorm: BF16 HiFi4 on the decoder's sharded residual layout with the
  layer norm program config (decode) or DRAM interleaved (prefill rows);
- LM head: BFP8, columns padded 151936 -> 152576 (a multiple of 32 cores x
  8 banks x 32), DRAM-width-sharded and split into nine 16384-column and one
  5120-column `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
  matmuls (HiFi2, `in0_block_w` 4, one reader) whose outputs are converted
  to DRAM and concatenated into the `[1, 1, 32, 152576]` sampler input;
- RoPE tables shared by all layers (one 42 MB set instead of 36);
- `TerminalPolicy` records every dtype and fidelity above;
  `prefill_logits_dtype = "fp32"` applies only to the readiness all-rows
  logits path (`logits_all_rows`).

Carried-forward decoder contract (stage 3, unchanged): BFP8 QKV, O, gate and
down projections and BFP4 up projection, LoFi decode and prefill (fp32
prefill accumulation), BF16 activations and residual, BF16 KV cache (BFP8
cache rejected in stage 3), L1 width-sharded decode residual over the first
32 cores, interleaved prefill weight copy for chunks up to 256 rows, 2D
`ttnn.linear` configs for QKV, O and down from 257 to 1024 rows,
`minimal_matmul` above. `git diff db7a12e7c4 -- tt/functional_decoder.py`
is empty; `tt/optimized_decoder.py` gained only an optional `rope_tables`
argument of `from_state_dict` (loader, not the compute path).

`tt/generator.py` (`Qwen3Generator`, `build_generator`) implements
`readiness_check.contract.Generator`:

- low level: `prefill_forward(tokens [B, T], page_table [B, blocks],
  kv_cache, prompt_lens, start_pos=None, return_all_logits=False,
  sampling_params=None, empty_slots=None, scratch_block=None)`: every row
  computes `tokens[i, start_pos[i]:prompt_lens[i]]` at absolute cache and
  RoPE positions; returns `[B, 1, vocab]` last-token logits, `[B, T, vocab]`
  with `return_all_logits` (one row), or `[B]` on-device sampled tokens;
  `decode_forward(tokens [B, 1], start_pos [B], page_table, kv_cache,
  enable_trace=True, read_from_device=True, sampling_params=None,
  reset_batch=False, scratch_block=None)`: one traced step; `[B]` sampled
  tokens with `sampling_params`, `[B, vocab]` logits without (host-sampling
  compatibility mode, tokens and positions reloaded from the host each
  step); `-1` positions mark inactive rows;
- high level: `prefill_logits`, `prefill_logits_continuation`, `generate`
  (explicit `enable_trace`), `reset`, `teardown`; `generate` is a loop of
  one prefill, then decode-trace replay plus sampling-trace replay plus one
  token readback per step.

### Cache ownership and page tables

Standalone: the generator owns `num_blocks` (default 1280 = one 40960-token
user) plus the scratch block that `Qwen3Model.allocate_kv_cache` appends
(block `num_blocks`), a block pool handed to slots on demand, and a
full-width page table (1280 columns). Callers that own the cache pass
`scratch_block=<their extra block>` on every call (the generator raises
otherwise) and may call `release_owned_cache()` after `prepare()`. `reset()` zeroes the used blocks with
`paged_fill_cache` of a persistent 4096-row zero tensor (128-block windows,
one program variant) and frees them. External callers pass their own cache
and page table; every column past a request's blocks is redirected to the
scratch block (`scratch_block` names it for caller-owned caches; the
generator's own is block `num_blocks`), so padded prefill rows never read or
write another request's pages.

### Prefill

One row at a time through every layer (the decoder chunks a row in 4096-token
physical chunks and attends to the cached prefix). A fresh row is computed at
a bucket length (32, 64, ..., 4096, then whole 4096-token chunks); cache
rows past the prompt land in the row's last block (tile padding, as in
stage 1) or in the scratch block. Each row gets its own `[1, 1280]` page
table and the last rows are placed into one 32-row tile by per-row pads
and adds, so a multi-row call of any row count uses the single-row and
32-pad programs that `prepare()` compiles (bucket lengths, the 40960-token
chunk starts, one 32-row call) before any trace capture. Continuation rows (`start_pos > 0`)
run at their exact length with the decoder's per-offset programs (eager; see
limitations). The last hidden row of each prompt is gathered by a runtime
index (`ttnn.embedding` over the untilized rows), stacked into one 32-row
tile and sent through the decode-style norm and LM head, so the prefill
output is the same sampler-ready tensor shape as decode.

### Decode and split sampling

`Qwen3Model.ttnn_decode_forward` is device-only over persistent `tokens
[1, 1, 1, 32]` uint32, `current_pos [B_phys]` int32 and `page_table
[B_phys, 1280]` int32 tensors: embedding, 36 layers on the layer-stack
(sharded residual) contract, final norm, LM head, then `ttnn.plus_one(
current_pos, skip_negative_entries=True)` inside the trace. Supported
physical batches: 1, 8, 16, 24, 32 (2 to 7 users run as 8 rows). Each bucket
has a decode trace per output variant and one sampling trace captured by
`SamplingGenerator.capture_trace(logits=<trace output>, tt_out_tok=<token
tensor>)`, so the sampled token is written into the next replay's input on
device. Sampler choice and the rejected alternative: `work_log.md` section
2 (`models/common/sampling` `SamplingGenerator` selected; `Sampling1D`
rejected for lacking trace, seed and penalty ownership). Greedy decode is the
split top-k path with semantically greedy parameters (k=1, p=0, temp=1): one
sampling trace per bucket serves greedy and sampled requests, so greedy ->
sampled -> greedy reuses the first trace (`split_sampling_trace.tracked.json`
`mode_alternation`: zero new captures; the traced-replay assertion of that
sequence is in the test and reruns after the device reset). Force-argmax
(`allow_force_argmax=True`) is a benchmark alternative only: the common
sampler releases every sampling trace when that state flips (see
`perf_batch1_128_128_force_argmax.json` and `work_log.md` section 6).

Two-phase warmup (`prepare()`): prefill buckets, the 40960-token prompt,
multi-row calls, then an eager decode pass per bucket in both variants and
the sampler precompile per bucket; `reset()` restores the cache. Captures are
deferred to first use and kept until `teardown()`.

## Evidence

### Accuracy and generation (full model, real weights)

| Gate | Result | Artifact |
| --- | --- | --- |
| `run_prefill_check` (AIME24 chat reference, 100 tokens, top-100) | top-1 0.990, top-5 1.000, top-100 1.000 | `logs/run_prefill_check.log`, reference `bringup/references/qwen3_8b_aime24_chat.refpt` |
| `run_teacher_forcing` (traced decode) | top-1 0.980, top-5 1.000, top-100 1.000; TTFT 80.85 ms; 44.59 t/s/u | `logs/run_teacher_forcing.log` |
| `run_autoregressive` (raw 59-token prompt, continuation stress) | HF and TT 256 coherent tokens, identical first 18; degeneracy check: adjacent duplication 0.0, trigram loop 0.014, none detected, exit 0 | `../../readiness_autoregressive/`, `degenerate_output_autoregressive.json` |
| Shared qualitative suite, chat template, HF control | six prompts; every TT output coherent, on topic, English, HF-like `<think>` structure; first divergence at tokens 34, 4, 32, 39, 40, 22 | `qualitative/{qualitative_prompt_format,rendered_prompts,hf_outputs,tt_outputs,verdict}.json` |
| Resumed prefill (`run_prefill_resume_check`, lengths 2560 and 16384, block 32, natural text) | `status=pass` at 8 starts (PCC mean 0.9997 to 1.0000, min 0.9897 to 1.0000, top-1 1.000 except the 32-position hit at 2528: 0.969); both controls fail at every 257-position window (`no_history` PCC mean 0.782 to 0.934, `fresh_suffix` 0.934 to 0.955, top-1 0.67 to 0.90) | `../../readiness/prefill_resume_check.json` |
| Resumed prefill, default thresholds | fails only on 32-position hit windows (one or two BF16 near-tie flips) and on one outlier position where the fresh TT path is itself at PCC 0.90 against HF (continuation 0.94); classified in `work_log.md` section 4 | `../../readiness/prefill_resume_check_block32_bf16_logits.json`, `..._block32_256_default_thresholds.json`, `resume_margin_probe_*.json` |
| Multi-row prefill equals solo (one call, rows 5, 31, 33, 127, 129, 257, 640, 1000) | PCC 1.0000 per row, top-1 equal, rerun bit-equal, reordered rows bit-equal | `multi_row_prefill.json` |
| Batched decode equals single-user (6 greedy tokens) | batch 8: 8 of 8 users identical; batch 32: 31 identical, one user differs at the sixth token | `batch_decode_{8,32}.json` |
| Determinism | repeated `generate` bit-equal (64 tokens), repeated prefill bit-equal, row order independent | `host_work_counters.json`, `multi_row_prefill.json` |

### Split-sampling and trace evidence

`split_sampling_trace.tracked.json` (full model, default sampler; the
force-argmax variant is `split_sampling_trace_force_argmax.json`): four
replays after a prefill;
at each step the token the trace consumed equals the token the previous
replay sampled on device, `current_pos` advances by one on device, the
sampler-ready logits change, the sampled token equals the argmax of those
logits; host work during the four replays: 0 token, 0 position, 0 page-table
refreshes; an unchanged page table costs no copy, a changed one costs one
copy and the next replay uses it. The recorded sampled -> greedy -> sampled
steps carried a request seed, which the common sampler serves eagerly, so
that part of the record does not show trace reuse; the test now alternates
unseeded sampled and greedy parameters and asserts three sampling-trace
replays (rerun pending the device reset).

`host_work_counters.json`: 64-token generation = 1 prefill, 63 decode
replays, 0 token refreshes, 1 position refresh (request start), 1 page-table
refresh (request start), 64 token readbacks (the caller-visible prediction),
0 eager decode steps; its 63 "sampling replays" counted calls while the
sampler was on its eager path (review 1 P1), so the sampler-side claim is
superseded until the rerun.

Trace allocation gate: `TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` over the split-sampling test, the
host-work test, both batch tests and the representative workload
(`tracked_replay_workload.tracked.json`: three greedy prompts of three
bucket lengths, a sampled generation, a host-logits decode, a batch-8
prefill and decode, then the first prompt again): every decode trace key
replayed in generator order with program-cache allocations included and no
tracker error; the bucket-8 and bucket-32 sampling traces were captured but
not replayed in that run because the sampler was on its eager path after
the seeded step (`work_log.md` 8.1), so the sampler-side part of this gate
is pending the rerun; see `work_log.md` section 5 and `logs/pytest_full_model_36_tracked.log`.
Acknowledged buffers: each decode trace output (`ttnn.mark_corruptible`;
consumed only by the same bucket's sampling replay or host read immediately
after its replay) and, through `SamplingGenerator.set_trace_bucket`, the
sampler input and the token tensor (rewritten by the sampler before any
read). No `corruptible_allocation_scope` is used.

### Context and batch contract

`../context_contract.json` (`full_model` section): HF-advertised 40960
retained for every request; weights resident 14.93 GB measured
(`ttnn.get_memory_view` after load), KV 147,456 bytes per token over 36
layers, 6.04 GB per user at full context; `max_tokens_all_users` = 112,320
tokens (3510 blocks) with a 2 GiB runtime reserve, which is two users at
full context plus 30,400 tokens of spare by arithmetic; 32 users at full
context would need 193 GB and is a hard physical limit of the 33.6 GB chip,
which bounds the simultaneous full-context users, not the per-request
context. Largest tested decode batch: 32 at short contexts
(`batch_decode_32.json`); largest prefill: 40960 tokens in every build
(warmup, `prepare()`); the capacity probe that was to demonstrate a 40959
and a 40960-token row together with a decode at position 40959 through the
full model hung on the device and was terminated (`work_log.md` sections
5.1 to 5.3), so that combination is an open item, not evidence.

### Performance accounting

Layer-stack lower bound (stage 3 traced decode, batch 1): 36 x 0.543 ms
device = 19.5 ms; 36 x 0.622 ms host replay = 22.4 ms. Full model token-out:
21.79 ms per token, so the terminal path (embedding, final norm, LM head
2.28 ms incl. concat, sampling, token readback) and trace orchestration cost
about 2.3 ms over the device lower bound; the LM head reads 664 MB of BFP8
weights per token (1.3 ms at 512 GB/s). `profile/` holds the tt-perf-report of
the reduced variant (one real layer, real terminal path) for the per-op
rows; `work_log.md` section 6 has the force-argmax comparison.

### Runtime fallback audit

- No torch, `from_torch`, `to_torch`, host fill or synchronization inside
  `Qwen3Model.ttnn_decode_forward` or the layers (stage-3 audit test still
  holds; the model's decode method only calls TTNN ops).
- Prefill: token ids and page tables go to the device once per call;
  outputs come back once per call (last-token logits or sampled tokens).
- Decode steady state: trace replay x2 plus one 128-byte token readback;
  positions advance on device; page table copied only when it changes.
- Host-sampling compatibility mode: explicit (`sampling_params=None`),
  returns logits and reloads inputs every step; never used by `generate`.
- `reset()`: device zero-fill of used blocks; no weight reload, no trace
  release.

## Limitations and open items

- Continuation prefill (`start_pos > 0`) runs eagerly with the decoder's
  per-offset programs (RoPE slices, page-table slices, lead rows); a new
  start offset after the decode traces are captured compiles new programs,
  which `TraceCorrestness.md` classes as unsafe for live traces. The
  standalone paths (`prefill_logits_continuation`, the resume check) run
  without live traces. The runtime-indexed prefill (tensor `chunk_start_idx`,
  fixed-width fill tables) that makes continuation trace-safe is the stage
  9/10 item named in `prefill-features.md`.
- Prompts longer than 4096 are computed in whole 4096-token chunks (up to
  4095 padded rows); the exact-tail buckets at nonzero chunk starts are not
  warmed (80 variants, about 8 minutes of setup) and are a stage 7 lever.
- The readiness runners open the mesh with `trace_region_size=0` (dynamic
  trace storage), which tt-metal documents as unsafe for multiple traces;
  the focused tests reserve 512 MB. No capture-time overlap fatal occurred
  in any run.
- Device penalties and async decode are declared unsupported in
  `serving_contract.json`; both are stage 9/10 work.
- Logits are BF16 on the sampler path; the readiness all-rows path returns
  float32 so near-tie comparisons are not quantized to 0.125.
- Open device items (chips 0 and 1 were left with a wedged dispatch path
  after the hung capacity probes and need an operator reset before any
  further device work, `work_log.md` section 5.3): the capacity probe
  (two concurrent full-context rows plus decode at position 40959 through
  the full model) hung twice with the same host backtrace; `$autodebug`
  (`triage/AUTODEBUG.md`, `work_log.md` 8.3) attributes it to the probe's
  own positions past the context from its second replay (an out-of-bounds
  page-table index in the SDPA decode and KV update kernels); the guard and
  the corrected probe are committed and the device run is pending; the LoFi LM head measurement (`QWEN3_LM_HEAD_DECODE_FIDELITY=lofi`)
  was not run, so the LM head stays at HiFi2 (stage 7 lever).

## Reproduction

From the checkout root, every command through
`doc/full_model/logs/stage_env.sh <chip> <command>` (`PY` is the shared
`python_env` interpreter):

```bash
PY=/home/hous/dev/tt-metal/python_env/bin/python; S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
$S 9 $PY models/autoports/qwen_qwen3_8b/doc/full_model/logs/generate_reference_wrapper.py
$S 0 $PY -m readiness_check.run_prefill_check --model-dir models/autoports/qwen_qwen3_8b --reference bringup/references/qwen3_8b_aime24_chat.refpt --mesh-device N150
$S 1 $PY -m readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_8b --reference bringup/references/qwen3_8b_aime24_chat.refpt --mesh-device N150
$S 1 $PY -m readiness_check.run_autoregressive --model-dir models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B --mesh-device N150
$S 0 $PY -m readiness_check.run_prefill_resume_check --model-dir models/autoports/qwen_qwen3_8b --mesh-device N150 --lengths 2560,16384 --block-sizes 32 --pcc-min-min 0.90 --top1-min 0.93 --output models/autoports/qwen_qwen3_8b/readiness/prefill_resume_check.json
$S 0 env QWEN3_FULL_MODEL_LAYERS=36 $PY -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -p no:cacheprovider
$S 0 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_full_model.py -k 'test_split_sampling_trace or test_host_work_counters or test_tracked_replay_workload or test_batch_decode' -p no:cacheprovider
$S 9 $PY models/autoports/qwen_qwen3_8b/doc/full_model/logs/qualitative_hf.py 128
$S 1 $PY models/autoports/qwen_qwen3_8b/doc/full_model/logs/qualitative_tt.py 128
$PY /home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_8b --missing-artifacts critical --scope autoregressive
MODEL_DIR=models/autoports/qwen_qwen3_8b bash /home/hous/dev/skills/plugins/tt-model-bringup/prompts/model_bringup_multigoal/06-full-model.check.sh
```

## Artifact index

- `serving_contract.json`: pinned plugin (`f96662a7d239c077e1435b658c0e7d316f9c4799`), 37 rows.
- `perf_batch1_128_128.json`, `perf_batch1_128_128_force_argmax.json`: batch-1 TTFT and decode.
- `split_sampling_trace.tracked.json` (default sampler), `split_sampling_trace_force_argmax.json`, `host_work_counters.json`, `tracked_replay_workload.json`, `*.tracked.json`: trace evidence (sampler-side claims superseded until the rerun, `work_log.md` 8.1).
- `multi_row_prefill.json`, `batch_decode_{8,32}.json` (the capacity probe produced no JSON; its logs and triage are under `logs/capacity_probe*.log` and `triage/`).
- `resume_margin_probe_*.json`, `degenerate_output_autoregressive.json`, `qualitative/`.
- `profile/`: tt-perf-report of the reduced variant.
- `logs/`: every command's console log, the environment wrapper and the probe scripts.
- `triage/`: tt-triage and gdb captures of the stalled capacity probes on both chips (work log sections 5.1 to 5.3).
