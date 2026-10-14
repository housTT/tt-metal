# Qwen3.5-35B-A3B full model: stage 06

Abbreviations: TTFT (time to first token), t/s/u (tokens per second per user), ISL/OSL (input/output sequence
length, in tokens), HF (Hugging Face CPU reference), TT (Tenstorrent TTNN implementation), KV (key/value cache),
GDN (Gated DeltaNet linear-attention layer), LM head (final vocabulary projection), BF16 (16-bit brain float),
BFP8 (8-bit block float), DRAM (device memory), L1 (core SRAM), PCC (Pearson correlation coefficient), KL
(Kullback-Leibler divergence), TV (total variation distance), PRNG (pseudo-random number generator), ULP (unit in
the last place), CPU (host processor), API (application programming interface).

## Headline (batch 1, traced decode, ISL 128 / OSL 128, `evidence5/m{4,2,1}/full_evidence.json`)

| Profile | TTFT, ISL 128 | Token-out decode, ISL 128 / OSL 128 | Logits-only decode |
|---|---:|---:|---:|
| p150x4 | 98.9 ms | 9.19 ms per token, **108.9 t/s/u** | 8.59 ms |
| p150x2 | 114.1 ms | 11.36 ms per token, 88.0 t/s/u | 10.48 ms |
| p150 | 183.4 ms | 15.85 ms per token, 63.1 t/s/u | 14.51 ms |

Teacher forcing timing (readiness runner, `evidence5/m*/teacher_forcing_*.log`):

| Profile | AIME24 chat, ISL 161 / OSL 100 | Book, ISL 8193 / OSL 64 |
|---|---:|---:|
| p150x4 | TTFT 125.0 ms, 99.8 t/s/u | TTFT 908.6 ms, 92.3 t/s/u |
| p150x2 | TTFT 136.0 ms, 83.1 t/s/u | TTFT 1214.6 ms, 77.8 t/s/u |
| p150 | TTFT 208.5 ms, 60.9 t/s/u | TTFT 1708.5 ms, 57.7 t/s/u |

Measurement settings: every number above ran with `TT_METAL_TRACE_ALLOC_TRACKING=1
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN35_STRICT_CACHE=1`. The tracker runs `verify_before_replay` on each
trace replay (two per token-out step). Tracker-off control (`evidence5/m*/batch_perf.json`, B1, same prompt and
lengths): TTFT 97.8, 114.0 and 183.4 ms; token-out 9.18, 11.35 and 15.87 ms per token (p150x4, p150x2, p150).
This is within 1.1% of the tracker-on numbers, so the tracker cost is small. Chains 2 to 5 agree within 2% (TTFT)
and 0.3% (decode).

Definitions:
- Prompt for the headline: first 128 tokens of the book text, greedy, one user, one request slot.
- TTFT: wall time from the `generate()` call to the first token on the host. It includes prefill of 128 tokens
  and the first device sample.
- Token-out decode: the remaining wall time divided by 127. Each step is one traced decode replay plus one traced
  device sample; the host reads only the token. Median of 3 runs; all 3 are in `perf_128_128.runs_*`.
- t/s/u: 1000 divided by the token-out milliseconds per token.
- Logits-only decode: one blocking replay of the decode trace without the sampler, median of 64, at context
  positions after the 128/128 run (about 256 to 320; the exact position is not recorded).
- Teacher forcing: readiness `run_teacher_forcing` (traced decode, forced HF tokens). Its `decode t/s/u`
  includes the per-step token write that forcing needs, so it is lower than token-out.
- Greedy decode only. In sampled mode, each step also writes 32 fresh seeds from the host (one
  `copy_host_to_device_tensor` of 128 bytes). That write is not in the headline numbers and is not timed.
- This is not the Stage 13 serving sweep. ISL/OSL 131072/1024 at client concurrency 1 to 32 is measured through
  the packaged endpoint in Stage 13.

### Batch token-out (`evidence5/m*/batch_perf.json`)

`tests/run_batch_perf.py`: tracker off (`QWEN35_STRICT_CACHE=1` only), greedy, ISL 128 per row (distinct book
slices), 127 decode steps, median of 3 runs. B4 and B32 time the traced loop (`_replay` plus token read) after
one `decode_forward`.

| Profile | B1 ms per step | B4 ms per step (aggregate t/s) | B32 ms per step (aggregate t/s, t/s/u) |
|---|---:|---:|---:|
| p150x4 | 9.18 | 35.87 (111.5) | 282.31 (113.3, 3.5) |
| p150x2 | 11.35 | 43.95 (91.0) | 343.90 (93.0, 2.9) |
| p150 | 15.87 | 59.64 (67.1) | 466.10 (68.7, 2.1) |

- Aggregate t/s: batch times 1000 divided by ms per step.
- Step time grows almost linearly with batch, so aggregate throughput stays flat. Cause: B4 and B32 decode run
  the whole decoder layer once per row (see Open items). Stage 7 owns this; it limits the Stage 13 sweep at
  concurrency above 1.

## Accuracy against HF (readiness runners, `evidence5/m*/`)

Top-k accuracy: the share of positions where the TT predicted token is inside the HF top k
(`readiness_check/teacher_forcing.py` `compute_accuracy`). Bar: top-5 >= 98%, top-100 = 100%.

| Check (ISL / scored positions) | p150x4 top-1 / 5 / 100 | p150x2 | p150 |
|---|---|---|---|
| Teacher forcing, AIME24 chat (161 / 100) | 99.0 / 100 / 100 | 96.0 / 100 / 100 | 98.0 / 100 / 100 |
| Teacher forcing, book (8193 / 64) | 92.2 / 100 / 100 | 90.6 / 98.4 / 100 | 89.1 / 98.4 / 100 |
| Prefill check, AIME24 chat (161 / 100) | 99.0 / 100 / 100 | 97.0 / 100 / 100 | 99.0 / 100 / 100 |

- Chain 5 ran these on the final source. Every value equals chain 2. The autoregressive and qualitative
  outputs are byte-identical to chain 2 on all profiles.
- p150 device: the readiness runners (`--mesh-device N150`) open physical device 0. The model scripts open
  physical device 3 (`QDEV=3`). So p150 evidence covers two physical chips of the same type.
- References: fresh, CPU, plain BF16 load of the pinned checkpoint.
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/readiness_aime24_chat.refpt`: prompt `[1,161]`,
    100 tokens, top-100.
  - `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references/readiness_book_8193.refpt`: prompt `[1,8193]`,
    64 tokens.
- Book top-5 on p150x2 and p150 is 63 of 64 positions. It passes the bar with one position of margin.
- The HF reference is itself noisy. Its incremental `generated_tokens` and batched `topk_tokens[:, 0]` disagree
  at 12 of 64 book positions and 1 of 100 AIME24 positions (stage review 1 analysis). Top-1 on the book text is
  therefore a weak discriminator.
- Autoregressive (`run_autoregressive`, story prompt, no chat template, 256 tokens): all three profiles match HF
  for 10 tokens. Then TT writes "move faster than" where HF writes "stretch and shrink". The TT text is coherent
  and on topic (`evidence5/m*/autoregressive/`). Control: an HF batched forward over the same prefix ranks
  " move" first, probability 0.264 (`review2_fixes/calibration/hf_story_m2_teacher_forced.json`, step 10). Only
  the HF incremental reference picks " stretch". So the divergence is HF reference noise at a near-tie.
- p150x2 story content error: the elder says "Not quite" to the correct "seven times", then accepts
  "Seventy-two" hours per week as "Correct!". Control: HF teacher-forced on the TT completion
  (`review2_fixes/calibration/hf_story_m2_teacher_forced.json`) ranks every token of the error first ("Se" 0.29,
  "-two" 0.98, "Correct" 0.25, "Not" 0.43). Over the completion, 243 of 256 tokens are HF top-1, max rank 2.
  Classified as model behavior, not a TT defect.
- Degenerate-output check: no findings (`evidence5/degenerate_check.json`).

### Calibration against HF (`evidence5/calibration/compare.json`)

`tests/run_calibration_probe.py` saves TT logits over the AIME24 chat prompt plus its 100 reference tokens, by
prefill (`prefill_logits`) and by decode (`decode_forward`, forced tokens). One HF CPU forward over the same
tokens is the control (`review2_fixes/calibration/hf_calibration.py`). 100 positions; 40 are unsaturated (HF top-1
probability below 0.99).

| Profile, path | KL(HF, TT) mean | KL max | TT minus HF top-1 logit, mean | HF top-1 prob, unsaturated | TT prob of HF top-1, unsaturated | Top-100 logit slope, median | Argmax agrees |
|---|---:|---:|---:|---:|---:|---:|---:|
| p150x4 prefill | 0.0072 | 0.122 | -0.13 | 0.746 | 0.742 | 1.00 | 99 |
| p150x4 decode | 0.0070 | 0.155 | -0.19 | 0.746 | 0.749 | 1.00 | 99 |
| p150x2 prefill | 0.0063 | 0.134 | -0.16 | 0.746 | 0.740 | 1.00 | 97 |
| p150x2 decode | 0.0074 | 0.186 | -0.18 | 0.746 | 0.742 | 1.00 | 96 |
| p150 prefill | 0.0059 | 0.103 | -0.16 | 0.746 | 0.736 | 1.00 | 99 |
| p150 decode | 0.0055 | 0.051 | -0.16 | 0.746 | 0.748 | 1.00 | 98 |

- No systematic over-confidence or logit scale error on any profile. TT top-1 logits are slightly below HF on
  average, and TT probabilities on unsaturated positions match HF within 0.01.
- Chain 5 logits are bitwise equal to chain 4 on all profiles. The p150x4 and p150x2 chain 5 logits are bitwise
  equal to the review 2 fix runs
  (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/calibration/tt_m{4,2}.pt`).
- So the 428-token continuation over-confidence (see Continuation prefill) is an isolated position.

### Qualitative suite (`evidence5/qualitative/`)

- Prompts: the six shared prompts from the plugin `vllm_prompts.txt`.
- Format: chat template applied, because this is an instruct model (`qualitative_prompt_format.json`,
  `rendered_prompts.json`).
- Decoding: greedy, 128 tokens.
- HF control: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/qualitative/hf_outputs.json`.
- Tokens that match HF before the first difference:

| Prompt | p150x4 | p150x2 | p150 |
|---|---:|---:|---:|
| p0 haiku about AI | 66 | 74 | 39 |
| p1 supervised vs unsupervised | 48 | 90 | 90 |
| p2 story continuation | 119 | 69 | 62 |
| p3 laws of thermodynamics | 19 | 0 | 19 |
| p4 English to French | 44 | 44 | 44 |
| p5 Fibonacci in Python | 29 | 12 | 12 |

- Verdict: pass. All 18 outputs are coherent, on task and in the HF style.
- At 128 tokens, p0 to p4 end inside the thinking section on every profile. The thinking content is correct:
  - the haiku plan counts syllables;
  - the laws of thermodynamics are listed correctly;
  - the translation notes "Comment allez-vous" as formal.
- p5 is the only output that reaches the answer. p150x4 and p150x2 end at the first heading or code fence. p150
  ends at "def fibonacci(n".
- p150x2 p3 starts "Here's a thinking process" where HF starts "Thinking Process:". HF uses the same opening for
  p2 and for the AIME24 reference, so both openings are native to the model.

## Sampling

### Sampling defects and fix

Chain 2 showed that sampled mode did not sample. Rows 1 to 3 of the sampled B4 audit equaled greedy for 48 tokens
on every mesh. HF gives that outcome about 3% probability
(`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/sampling_hf/hf_sampling_margin.json`). There were two defects:

| Defect | Cause | Evidence | Fix in `tt/generator.py` |
|---|---|---|---|
| Seed never advanced | The generator never wrote the device seed tensor. The pipeline runs `ttnn.manual_seed` before every draw, so each draw restarted the PRNG from the same seed | op probe and micro test, frozen-seed control | `_sample` writes 32 fresh seeds before each sampled draw; greedy skips the write |
| Rows 1 to 31 greedy | `format_sampling_params` treats a scalar temperature as one active lane and pads lanes 1 to 31 with greedy defaults (`top_k` 1) | `tests/run_sampler_op_probe.py` captured k = [32, 1, 1, ...], p = [1.0, 0, 0, ...] | `_set_sampling` broadcasts a scalar temperature to all 32 lanes |

- Unprepared sampler variants raise `ValueError` in `_set_sampling`:
  - explicit request seeds (they would run the sampler eagerly after trace capture);
  - non-default presence, frequency or repetition penalty (penalty programs and token state are not prepared);
  - any log-probability request (`enable_log_probs`, `num_logprobs`); unsupported on 1, 2 and 4 devices.
- The refresh key covers every `SamplingParams` field, with `None` normalized to the field default. A change in
  any field refreshes the sampler parameters; the same values do not.
- CPU tests: `tests/test_generator_sampling.py` (rejection per field, refresh per field, inactive-row and slot
  handling). Chain 5 CPU run: 1218 passed (`evidence5/cpu_tests.log`).
- Failed attempts are preserved in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/seed_fix/failed_skip_seed/`
  and `seed_fix/failed_scalar_params/`. Each attempt fixed only one defect and still gave 31 constant rows.
- `ttnn.sampling` itself is correct on Blackhole: the op distribution test passes
  (`seed_fix/op_distribution_test.log`).
- Chains 1 and 2 sampled evidence is invalid and superseded. Greedy evidence is unaffected (see Accuracy).

### Sampled-mode evidence (`evidence5/m*/`)

`tests/run_sampler_micro.py`: synthetic logits (64 candidates, Zipf-like), 32 rows, 64 draws per row, per
parameter set and mode.

| Profile | Fresh seeds and SKIP, eager and traced: constant rows / out of support / device copies agree | Frozen seeds: constant rows |
|---|---|---|
| p150x4 | 0 / 0 / yes; pooled TV 0.020 to 0.049 | 32; pooled TV 0.149 to 0.366 |
| p150x2 | 0 / 0 / yes; pooled TV 0.009 to 0.058 | 32; pooled TV 0.068 to 0.354 |
| p150 | 0 / 0 / yes; pooled TV 0.016 to 0.050 | 32; pooled TV 0.153 to 0.344 |

`tests/run_sampling_audit.py`: full 40-layer model, tracker on, `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`.

| Check | p150x4 | p150x2 | p150 |
|---|---|---|---|
| Greedy B1 equals `generate`; rank 0 at every step | yes | yes | yes |
| Sampled B1 and B4 (T 1.0, top-k 32, top-p 0.95, 48 steps) and two sampled `generate()` runs: sequences equal to greedy, of 7 | 3 | 2 | 0 |
| Worst rank of a sampled token in the TT logits (must be < 32) | 2 | 4 | 1 |
| Every device copy of `tt_tokens` agrees after every step | yes | yes | yes (one device) |
| Greedy B32 rows equal B4; greedy B1 after all buckets identical | yes; yes | yes; yes | yes; yes |
| Greedy `generate()` after the sampled runs identical | yes | yes | yes |
| Distribution: 2048 draws (64 replays x 32 rows), TV against host softmax of top 32 (must be < 0.1) | 0.019 | 0.015 | 0.013 |
| Distribution: constant rows | 0 | 0 | 0 |
| Distribution: draws outside the host top 32 have a logit equal to the 32nd (BF16 tie) | 2 draws; logit 13.125 = 32nd, 3 tokens tied | none outside | 1 draw; logit 13.125 = 32nd, 5 tokens tied |
| Sampler capture scope: buffers that survive, buckets 1, 4, 32 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |

- Check `sampled_not_all_greedy`: at least one of the 7 sampled sequences differs from greedy. Single-sequence
  checks are too weak: TT gives a sampled row a 0.13 to 0.46 probability of reproducing greedy over 48 steps
  (`evidence3/sampling_repeat/m2.json`). All 7 sequences reproduce greedy with probability about 7e-6.
- Rerun 1 of the audit (`evidence3/failed_audit_rerun1/`) used per-run checks. On p150x2 all 4 sampled B4 rows
  equaled greedy (TT probability 0.0034). `tests/run_sampling_repeat.py` on p150x2 then ran sampled B4 8 times:
  per-row equal-to-greedy rates 0.25, 0.375, 0.25, 0.125 against TT probabilities 0.127, 0.334, 0.458, 0.176.
  Sampling works; the failure was a chance event in a weak check.
- Distribution prompt: "My favourite number between 1 and 100 is" (plain text). Fixed decode logits; the sampler
  trace for bucket 32 replays 64 times with T 1.0, top-k 32, top-p 1.0.
- Frozen-seed control (2-layer model, p150x4, `evidence5/control_frozen_seeds_l2_m4.json`): rows vary false, TV
  0.38. The checks detect the original defect.
- Sampled draws use fresh seeds, so these counts change from run to run. Chain 4 gave 3, 1 and 1 of 7.
- p150x4 has 3 of 7 sampled sequences equal to greedy (B4 row 1 and both `generate()` runs). On p150x2 the
  repeat test measured TT probability 0.33 for B4 row 1. The `generate()` prompt has no repeat test. The
  check passes because 4 of 7 differ from greedy.
- Split-sampling trace sampled leg (`split_sampling_trace.json`, T 0.8, top-k 32, top-p 0.95): the 7 sampled
  tokens equal greedy on all profiles. HF says this is expected: the 0.95 nucleus holds one token at each of the
  7 steps, so the probability that sampling reproduces greedy is 1.0
  (`evidence3/hf_trace_prompt_margin.json`). Sampling variation is covered by the audit above.

### Sampler choice

| Candidate | Result |
|---|---|
| `models/common/sampling` `SamplingGenerator` | Selected. It owns trace capture and replay per batch bucket and parameter state across calls, writes `tt_out_tok` in place, refreshes parameters without recapture, and runs local top-k per device then an all-gather of candidates only |
| `models/common/modules/sampling/sampling_1d.py` `Sampling1D` | Rejected. It accepts `tt_out_tok` and per-call k, p and temperature, but it owns no trace or parameter state. The generator would have to capture and manage those itself |

## Correctness of the paths

### Non-aligned lengths, batch, determinism, context (`evidence5/m*/full_evidence.json`)

| Check | p150x4 | p150x2 | p150 |
|---|---|---|---|
| Prompt lengths 5, 127, 129, 300, 1000, 4095, 4097, 8191, 8193, 12000 | run | run | run |
| Batch rows equal to batch 1 (every row) | B4, B32 | B4, B32 | B4, B32 |
| Greedy repeat identical; prefill logits bitwise | yes; yes | yes; yes | yes; yes |
| 262100-token prompt, prefill plus 16 tokens | 62.3 s | 91.0 s | 140.4 s |
| 262143-token prompt: requested 16, returned | 2 | 2 | 2 |
| `pad_fallbacks` | 0 | 0 | 0 |

- 262100 tokens: all three continue the repeated story with " Elena. Every morning, she would climb to the top
  of the eastern hill and", the correct next text. This text is periodic, so it shows that the path runs at the
  native context. It does not test long-range attention accuracy.
- 262143 tokens: `generate` stops at the native 262144-token context. The prompt plus the first token fill it;
  the second token is the last prediction. Context is not reduced on any profile.
- Batch rows mix prompt lengths 5 to 12000 in one batch, each in its own request slot.
- All 46 boolean checks pass on every profile. Every token result equals chain 4; only wall times differ.

### Inactive rows and fixed slots (`evidence5/m*/inactive_rows.json`)

Contract (`tt/generator.py`):
- Decode row r uses request slot r. `request_slots` or `slot_remap` other than row order raises `ValueError`.
- `prefill_forward(empty_slots=...)` (the vLLM TT plugin name) selects the slot of each prefill row, the same as
  `request_slots`. Slots must be distinct and inside the configured request slots.
- `prefill_forward` and `decode_forward` take no catch-all keyword arguments. An unknown argument raises
  `TypeError`. A decode batch that `prepare()` did not warm raises `ValueError`.
- A negative position marks an inactive row. The generator writes position 0 and an all-scratch page-table row
  for it, so its KV writes go to the scratch block, never to a live block.
- An inactive row still updates its own GDN slot state. So the generator tracks live slots: prefill and
  `generate` mark a slot live; `release_slots(slots)` and `reset()` clear it. A decode call that marks a live
  slot inactive raises `ValueError` before any device work. Prefill at start 0 resets the slot.

`tests/run_inactive_rows.py`, 16 decode steps, tracker and strict cache on. Every case passes on every profile:

| Case | Path | Active rows |
|---|---|---|
| B4, rows 1 and 3 inactive | `decode_forward`; traced loop | 0, 2 |
| B32, 28 rows inactive | `decode_forward`; traced loop | 0, 2, 17, 31 |
| B4, later prefill into previously inactive slot 1 | traced loop | 1 |
| B32, request only in slot 5 | `decode_forward` | 5 |
| B32, slots 5 and 1 | traced loop | 1, 5 |
| B4, slot 2 live; at step 5 a decode call marks it inactive | traced loop; the call raises `ValueError`, then decode continues | 2 |

- Every active row equals its B1 tokens. `hidden_live_slot_rejected` is true: the hidden-slot call raises, and
  slot 2 still equals its B1 tokens after it. Greedy B1 after all cases is identical. 3 decode captures (B1, B4,
  B32), so no recapture.

### Continuation prefill (`evidence5/m*/full_evidence.json` `continuation`; `evidence2/continuation/`)

- Contract: a continuation start must be a multiple of 128 tokens. `Qwen35Model.prefill` raises `ValueError`
  for any other start. Starts 32 and 8224 are rejected in the evidence on every profile.
- `tests/test_generator_warmup.py`: every aligned start in the test set (128 to 16256) and every length 1 to 8192
  maps to a warmed program signature.
- Whole vs split prefill (5 cases, starts 128 to 12288, lengths 1 to 4097): same argmax in every case on every
  profile. Full-vocabulary PCC is 0.83 to 1.0.
- HF probe on p150x4 (`tests/continuation_hf_reference.py`, `tests/run_continuation_probe.py`,
  `evidence2/continuation/m4_with_kl.json`): whole, split and decode-fed logits against HF last-token logits.

| Total tokens | KL(HF, whole) | KL(HF, split) | KL(HF, decode-fed) |
|---:|---:|---:|---:|
| 4301, 8193, 12417 | <= 0.0002 | <= 0.0002 | <= 0.0002 |
| 428 | 0.50 | 0.20 | 0.42 |

- The low PCC at 4301 to 12417 tokens is noise in the low-probability tail; the distributions match HF.
- At 428 tokens HF gives the top token 0.926; every TT path is sharper (about 1.0; max logit 25.4 vs HF 20.5).
  All three TT paths agree, so it is not a continuation defect. The calibration probe (above) finds no
  systematic over-confidence over 100 non-periodic positions on any profile, so this position is isolated. Its
  cause is not isolated. It is recorded for the Stage 8 datatype sweep.

### Split-sampling trace (`evidence5/m*/split_sampling_trace.json`, full 40 layers)

All checks pass on all three profiles:
- positions advance on device;
- no host refresh in the loop (token, position and page-table refreshes 0);
- stepwise equals `generate`;
- an unchanged table needs no refresh;
- a changed table is written once and gives the same tokens;
- sampled tokens are valid;
- greedy after sampling is identical;
- decode trace IDs are stable;
- one decode capture and one sampler trace key.

Implementation:
- Decode sampling is on device: `SamplingGenerator`, traced. Output goes to the persistent `tt_out_tok` tensor,
  which is also the next decode input.
- `ttnn.plus_one(pos, skip_negative_entries=True)` inside the trace advances positions; RoPE reads the device
  position tensor.
- Page tables are persistent device tensors. The generator writes them only on change. Unused entries point at
  the scratch block (`kv_pool.scratch_block`), never at block 0.

### Trace allocation safety and lifetimes

- Every evidence run except `batch_perf` uses `TT_METAL_TRACE_ALLOC_TRACKING=1
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN35_STRICT_CACHE=1`. The tracker checks every replay. Strict cache
  forbids program-cache misses after `prepare()`. All runs passed.
- `batch_perf` runs with the tracker off, only to time it. Its logs show the Metal warning "Allocating device
  buffers is potentially unsafe due to the existence of an active trace". The tracker does not run there, so
  that run gives no safety evidence. The same call order (prefill, `decode_forward`, traced loop at B4 and B32)
  passes with the tracker on in `run_inactive_rows` and `run_full_evidence`. `batch_perf`
  tokens are not used as correctness evidence.
- Live traces: decode traces for B1, B4 and B32, and one sampler trace per bucket (1, 4, 32). Six in total.
- Acknowledged tensors and scopes:

| Tensor or scope | Owner | Written by | Read by | Lifetime rule |
|---|---|---|---|---|
| `logits_b` (decode trace output, per bucket b) | decode trace b | decode trace b | sampler trace b; host readback in `decode_forward(sampling_params=None)` | Valid from replay of decode trace b until the next replay of any trace that reuses its memory. All work is on command queue 0 in order. The generator always runs sampler trace b (or the host readback) right after decode trace b, before any other trace |
| `tt_tokens` (`tt_out_tok`) | generator, allocated before any capture | sampler trace b; host `_write_tokens` | decode trace b (next step); host `read_tokens` | Persistent buffer, never freed. Sampler trace b writes it in place; the next decode replay reads it. Host reads happen after the sampler replay on the same queue |
| Sampler capture scope (`corruptible_allocation_scope`, `models/common/sampling/generator.py`) | `SamplingGenerator` | sampler capture | n/a | Recorded with tracebacks per bucket: 0 buffers survive the scope for buckets 1, 4 and 32 on all profiles (`evidence5/m*/sampling_audit.json` `sampler_scope_allocations`) |
| Seed tensor `seeds_tt_tensor` | sampler, allocated before any capture | host `copy_host_to_device_tensor` before each sampled draw | `manual_seed` inside sampler trace b | Persistent; the copy writes in place and allocates no device buffer |

- Replay order per step: decode trace b, then (sampled mode) seed write, then sampler trace b, then token read.
- `decode_forward(sampling_params=None)` replays decode trace b only and reads `logits_b` to the host before
  returning. No other trace runs in between.
- The watcher and profiler runs do not set the tracker. There, Metal prints once "Allocating device buffers is
  potentially unsafe due to the existence of an active trace". In those scripts the next device work after the
  first capture is the next request's prefill. The same multi-request sequences pass with the tracker on.

### Runtime fallback audit

| Path | Evidence |
|---|---|
| Prefill: embedding and all 40 layers | `device_only()` guard (`tt/model.py` `_guard`, `tests/runtime_audit.py`: patches `from_torch`, `to_torch`, `as_tensor`, blocks torch ops) in `run_full_evidence` (`audit=True`) on all profiles |
| Decode: model decode, position advance, sampler input pad | captured under `device_only()` |
| Sampling | traced `SamplingGenerator`; `sampling_replays` equals `decode_replays` in token-out runs; no host argmax in token-out mode |
| Host-logit boundaries | prefill last-row logits (first token in host-sampling mode), `prefill_logits` for readiness, `decode_forward(sampling_params=None)`, `host_sampling=True`. Each one counts in `logit_readbacks` |
| Cache ownership | one KV pool owned by the generator; `_check_kv` rejects any other cache; allocated before the layers |
| Reset | `reset()`: `ttnn.mul_` by 0 on every KV tensor, GDN state reset on device (`ttnn.copy` of zero templates into each slot's state), host table caches cleared. Reset runs between evidence cases (29 resets) and outputs match |
| Host inputs by design | prompt token IDs per prefill chunk, last-row index, page tables and positions when they change, 32 seeds per sampled step |

Host-sampling compatibility mode: `generate(..., host_sampling=True)` reads logits and takes the host argmax. It
is for tests only; the measured token-out path never uses it.

### 2-layer probe and watcher (`evidence2/m*/watcher_probe.json`, `probe_control/`)

- Chain 2 source. Between chains 2 and 5 only `tt/generator.py` changed in `tt/` (`evidence2/source_manifest_end.txt`
  against `evidence5/source_manifest_end.txt`). The changes are sampling parameters for non-greedy lanes, the
  seed write, parameter validation, inactive-row host handling and slot validation. These runs are greedy, so their device work
  is unchanged. Watcher and profiler are not rerun.

- Watcher (`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`) on a 2-layer model (layers 0 and 3, real
  embedding, norm, LM head): `watcher.log` has no error or assert on any profile. The 40-layer stack has no
  watcher run.
- Tokens match HF greedy on p150x4. On p150x2 they differ at step 2 and on p150 at step 8.
- Controls:
  - without the watcher, the tokens are the same (`probe_control/gen_m{1,2}.json`);
  - `tests/run_decode_paths_probe.py` (`probe_control/paths_m{1,2}.json`): logits are bitwise equal for eager
    before warmup, traced, eager after warmup and traced again.
- These are not ties in HF. The HF top-two margins at those steps are 0.625 (p150x2) and 0.4375 (p150). The TT
  margins are 0.0625 and 0.125, so the near-tie is on the TT side.
- The eager probe (`tests/run_model_probe.py`, a different build: default request slots, KV pool allocated after
  the layers, arange page table) matches HF top-1 at every step. Which build setting changes the rounding is not
  isolated. Residual risk: decode numerics of the truncated 2-layer model depend on the build.
- Eager PCC against HF with forced HF tokens (`probe_control/model_m{1,2}.json`): min 0.9872 (p150x2), 0.9876
  (p150); 0.9889 on p150x4 (`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/probe/m4_l0_3.json`).

## Implementation

- `tt/model.py` `Qwen35Model`: embedding (BF16, row-major, replicated), 40 `MultichipDecoder` layers with the
  Stage 5 default policies, final RMSNorm, LM head (BFP8, sharded on the vocabulary over the mesh, HiFi2).
- Stage 5 decoder strategy, dtype and fidelity policy, KV dtype split (BFP8 paged KV, BF16 fill input), FP32 GDN
  state, CCL policy and residual layout are unchanged (`../optimized_multichip_decoder/README.md`). The only
  decoder change is the opt-in prefill plan below. The profile rows confirm `BF16 x BFP8` HiFi2 for the LM head.
- `tt/generator.py` `Qwen35Generator`: readiness `Generator` contract (`prefill_forward`, `decode_forward`,
  `prefill_logits`, `generate`, `reset`), `build_generator` entry point.

### Shape-stable prefill

- Model-level chunks of 8192 tokens; each chunk runs at a power-of-two physical length (128 to 8192).
- Decoder policy `prefill_bucket_plan` (`tt/optimized_decoder.py`, default False, set by the model): 4096-row
  pieces plus one masked chunk that reaches the bucket.
- `prepare()` warms 15 first-chunk and 15 continuation lengths, all-logits per bucket, and decode at B1, B4 and
  B32 with greedy and top-k sampling. After `prepare()`, strict mode forbids program-cache misses.

### Memory and context (`../context_contract.json` `full_model_validation`, `evidence5/m*/full_evidence.json`)

| Per device | p150x4 | p150x2 | p150 |
|---|---:|---:|---:|
| DRAM allocated / free after build | 21.10 / 13.08 GB | 31.19 / 2.99 GB | 32.56 / 1.62 GB |
| Largest free DRAM block per bank | 1492 MB | 287 MB | 179 MB |
| KV pool tokens (full-context sequences) | 1,050,624 (4.0) | 526,336 (2.0) | 264,192 (1.0) |
| GDN request slots (state bytes) | 32 (515 MB) | 32 (1.03 GB) | 32 (2.06 GB) |
| Decode and sampler traces for B1, B4, B32, six traces (DRAM added by capture) | 250 MB | 275 MB | 298 MB |
| DRAM free after all captures; largest free block per bank | 12.83 GB; 1462 MB | 2.72 GB; 256 MB | 1.32 GB; 147 MB |

- All profiles have 32 request slots. Context is not reduced.
- Trace rows: `evidence5/m*/batch_perf.json` `memory_after_build` and `memory_after_all_captures`. The mesh
  opens with `trace_region_size=0`, so trace buffers are ordinary DRAM allocations.
- p150 fit needed three changes (measured OOMs in `work_log.md`):
  - per-layer constants and scratch stored once and shared by all 40 layers;
  - the KV pool allocated before the layers;
  - 128-row blocked all-position logits.
- p150 32 slots: slots 1 to 31 are allocated after all layer weights (`tt/model.py` `late_slots`). With the slots
  allocated before or between the layers, a 570 MB load-time typecast failed on fragmented DRAM (71 MB needed
  per bank, largest free block 59 MB). The failed runs are in `evidence2/slots/failed_typecast_frag/` and
  `evidence2/slots/failed_typecast_skip/`.

### Memory census and handoff recheck (`../memory_handoff_recheck.json`)

`tests/model_census.py` groups live device buffers by owner after build and `prepare()` (MiB per device):

| Owner | p150x4 | p150x2 | p150 |
|---|---:|---:|---:|
| Layer-private (weights and per-layer scratch) | 12703 | 24359 | 24425 |
| KV cache | 5451 | 2731 | 2742 |
| GDN request state | 491 | 983 | 1965 |
| Embedding | 970 | 970 | 970 |
| LM head | 129 | 258 | 515 |
| Sampler | 83 | 106 | 152 |
| Shared across layers | 23 | 23 | 93 |

- Allocator total minus the census (allocations not reachable from the generator object graph): 271 MiB
  (p150x4), 313 MiB (p150x2), 187 MiB (p150). This remainder is not itemized.
- No layer holds a private copy of a shared constant: the census finds every shared constant once, under
  `shared_across_layers` (23, 23 and 93 MiB). An unreachable leftover copy would count in the remainder above.
- Known duplicate: on p150x2 and p150x4, routed experts exist twice. One copy is the expert-parallel prefill
  weights (`experts.dispatch.*`); the other is the tensor-parallel decode weights (`experts.weights.*`). Both
  are consumed. Stage 5 assigned their deduplication to Stage 7. Impact: p150x2 layer-private DRAM (24359 MiB)
  is about equal to p150 (24425 MiB), and p150x2 has 2.99 GB free, which caps its KV pool at 2.0 full-context
  sequences.
- Not measured in Stage 6: the itemized remainder (by buffer size), the allocator after each layer load, and
  construction peaks (host RSS and device). p150 has 1.62 GB free with a 179 MB largest block per bank, so
  Stage 7 must measure these before it changes memory.
- The census and allocator totals in chain 5 are equal to chain 4.

## Profile (reduced model: layers 0 and 3, real embedding, norm, LM head, sampler; `evidence2/m*/profile/`)

`tests/run_model_profile.py` under Tracy; `tt-perf-report` over signposts `PERF_PREFILL` (ISL 128) and
`PERF_DECODE` (4 traced token-out steps, greedy). Per decode step:

| Op | p150x4 | p150x2 | p150 |
|---|---:|---:|---:|
| LM head matmul `32 x 2048 x V/tp`, BF16 x BFP8, HiFi2, DRAM-bound | 388 us (66% DRAM) | 736 us (70% DRAM) | 1479 us (69% DRAM) |
| Sampler `TopkLargeIndices` | 285 us | 567 us | 1070 us |
| Token-out minus logits-only (full model) | 0.60 ms | 0.88 ms | 1.34 ms |

- The perf-report CSVs mix rows from several devices. The p150x2 LM head rows are on device 1; the p150 rows are
  on device 3 (physical device 3).
- LM head plus top-k share of token-out: 7.3% (p150x4), 11.5% (p150x2), 16.1% (p150). They do not dominate.
- Stage 7 targets: top-k on the full local vocabulary (measure a force-argmax path for greedy against the current
  top-k path), and the LM head. A BFP4 LM head belongs to the Stage 8 datatype sweep.
- The 40-layer stack: Stage 5 single-layer traced host times sum to 9.39 ms (p150x4), 10.92 ms (p150x2) and
  14.42 ms (p150) for 30 linear plus 10 full layers. Full-model logits-only decode is 8.59, 10.48 and 14.51 ms,
  which includes the LM head. One trace for the whole stack removes per-layer host overhead.

## Stage 9 contract notes

- Log-probabilities: `LogProbsCalculator` supports only 8 or 32 devices
  (`models/common/sampling/tt_log_probs.py`). They are not available on 1, 2 or 4 devices, and the generator
  rejects a log-probability request. The Stage 13 latency methodology needs `/v1/completions` with `logprobs=0`
  (`/home/hous/dev/qwen3.5-35b-a3b/evaluation/latency-methodology.json`). Stage 9 must choose a path for this; it
  blocks the packaged sweep until solved.
- Explicit request seeds raise `ValueError`. The server must send `seed=None`.
- Non-default presence, frequency and repetition penalties raise `ValueError`. Supporting them needs warmed
  penalty programs and per-request token state (`reset_prompt_tokens`, `reset_output_state`).
- Fixed slots: decode row r is request slot r. The adapter must prefill each request into the slot of its decode
  row (`empty_slots` or `request_slots`), and mark idle rows with a negative position. `slot_remap` must be the
  identity.
- Hidden requests: the pinned plugin can leave a running request unscheduled for a step and expects its state to
  survive (`/home/hous/dev/qwen3.5-35b-a3b/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`, unscheduled
  requests). Here an inactive row overwrites its slot's GDN state, so the generator raises `ValueError` for a live
  slot. Decoding the hidden row anyway would also advance its state. So Stage 9 must prevent hidden live requests
  or add device-side state preservation for inactive rows. Finished or preempted requests must call
  `release_slots`.
- `build_generator` warms only decode batch 1 by default. Serving must pass `decode_batches=(1, 4, 32)`. The
  plugin pads plain decode to its capacity, so a single request can land on the B32 trace unless the adapter
  selects a smaller bucket.
- The plugin uses `num_logprobs = -2` to mean no log-probabilities. `_set_sampling` rejects any nonzero
  `num_logprobs`, so the adapter must map -2 to 0.
- `_check_kv` rejects a caller-allocated KV cache. The vLLM adapter must return `generator.kv_pool.caches`.
- Chunked prefill: a request whose prefill is split over several steps holds a live slot after its first chunk.
  A decode step that marks its row inactive then raises `ValueError`. Stage 9 must schedule such requests so
  their row is not in a decode batch until prefill ends, or add device-side state preservation.
- An active decode row on a slot that no prefill marked live is accepted without error. Stage 9 should add this
  check; the live-slot set makes it cheap.
- Continuation starts must be multiples of 128 tokens. A 32-token KV block allows other starts, so chunked
  prefill in Stage 9 must align to 128.
- Prefix caching is unsupported. GDN state belongs to a slot, not to a prefix. A cached-prefix request that
  starts at `start_pos > 0` in a new slot would run from another request's state or from zeros.
- `decode_forward` builds a host page table of shape [batch, 8192] and compares it on every call. This is
  per-step host work on the serving path.

## Open items for later stages

| Item | Evidence | Owner |
|---|---|---|
| Sampled mode writes 32 seeds from the host per step. The common sampler's SKIP protocol avoids this write. SKIP passes the micro test (`traced_skip`, 0 constant rows) only with sampler-only replays in between. SKIP across real decode replays and sampled token-out speed are not measured | `tt/generator.py` `_sample`; `evidence5/m*/sampler_micro.json` | Stage 7 |
| Routed experts stored twice on p150x2 and p150x4 | census above | Stage 7 |
| DRAM remainder not itemized; construction peaks not measured | census above | Stage 7 |
| LM head DRAM-interleaved, 66 to 70% DRAM use; full-vocabulary top-k 285 to 1070 us per step; force-argmax not measured | Profile above | Stage 7 |
| B4 and B32 decode run the whole decoder layer once per row: attention or GDN, MoE and both all-reduces (`tt/functional_decoder.py` `decode_forward`) | Batch token-out above | Stage 7, Stage 9 |
| 428-token continuation over-confidence | Continuation prefill above | Stage 8 |
| Log-probabilities on 1, 2 and 4 devices | Stage 9 notes | Stage 9 |

## Limitations

- Model load takes 406 to 638 s (BF16 safetensors read, conversion, upload; no tensor cache).
- No logprobs on these meshes (see above).
- `readiness_check.generate` fails with transformers 5.12 chat templates; `tests/hf_offload.py` shims it
  (`_token_list`). Plugin fix candidate.
- HF disk offload gives garbage output; references use a plain memory-mapped load.
- Accuracy beyond 8193 tokens uses periodic text only (12417-token continuation KL, 262100-token run). Native
  context accuracy rests on the layer-level evidence of earlier stages.
- The watcher ran only on the 2-layer model.

## Evidence chains and artifacts

| Chain | Script and log | Time (Eastern) | Result | Used for |
|---|---|---|---|---|
| 5 | `evidence5/evidence_chain5.sh`, `.log` | 2026 Oct 14 06:42 to 12:37 | 39 steps, all exit code 0, `SOURCE_STABLE` | everything except watcher, profiles, continuation KL probe and p150 slot build |
| 4 | `evidence4/` | 2026 Oct 14 00:51 to 06:11 | superseded (`evidence4/SUPERSEDED.md`); outputs equal chain 5 | none |
| 3, 3b | `evidence3/` | 2026 Oct 13 20:41 to 23:53 | superseded (`evidence3/SUPERSEDED.md`) | diagnostics only: `hf_trace_prompt_margin.json`, `sampling_repeat/` |
| 2 | `evidence2/evidence_chain2.sh`, `.log` | 2026 Oct 13 15:34 to 20:01 | 32 steps, all exit code 0, `SOURCE_STABLE`; partly superseded (`evidence2/SUPERSEDED.md`) | watcher, profiles, continuation KL probe, p150 slot build |
| 1 | `evidence1/` | 2026 Oct 13 10:39 to 14:32 | superseded (`evidence1/SUPERSEDED.md`) | none |

- Source identity: `evidence5/source_manifest_start.txt` and `_end.txt` are equal (hashes of `tt/` and
  `tests/`, plus the `ttnn` diff stat). `evidence5/source_provenance.txt` records HEAD
  `fb2e3b5a604e064033aba073d015be1a1869d282` and zero uncommitted entries in the reused modules
  (`models/common/sampling`, `models/common/modules`, `models/demos/blackhole/qwen36`,
  `models/experimental/gated_attention_gated_deltanet`, `models/tt_transformers`, `ttnn`, `tt_metal`).
- Chain 5 source equals the final source: `tt/generator.py` sha256 `4140f40c...` in the manifest and on disk.
- Calibration tensors (about 199 MB each) stay outside the repository in
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence5_work/calibration_m{4,2,1}.pt`.
- Mesh labels: p150 `N150` (physical device 0 for the readiness runners; physical device 3 via `QDEV=3` for the
  model scripts), p150x2 `N300` with `FABRIC_1D` and `TT_VISIBLE_DEVICES=2,3`, p150x4 `P300x2` with
  `FABRIC_1D_RING`.
- Earlier runs, failures and fixes: `work_log.md`. Raw runs are under
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/`; failed attempts are in `failed_*` directories.

| Path | Content |
|---|---|
| `evidence5/m*/full_evidence.json` | perf, determinism, lengths, batches, continuation, long context, memory, census, counters |
| `evidence5/m*/teacher_forcing_*.log`, `prefill_check_aime24.log` | readiness accuracy and timing |
| `evidence5/m*/autoregressive/`, `evidence5/degenerate_check.json` | HF and TT completions, meta, degenerate check |
| `evidence5/qualitative/` | prompt format, rendered prompts, TT outputs |
| `evidence5/m*/sampler_micro.json`, `sampling_audit.json`, `evidence5/control_frozen_seeds_l2_m4.json` | sampled-mode evidence, saved draws, scope accounting, frozen-seed control |
| `evidence5/m*/split_sampling_trace.json` | split-sampling trace checks |
| `evidence5/m*/inactive_rows.json` | inactive rows and fixed slots |
| `evidence5/m*/batch_perf.json` | tracker-off B1, B4 and B32 token-out; DRAM after build and after all captures |
| `evidence5/m*/calibration_probe.json`, `evidence5/calibration/` | calibration probe and HF comparison |
| `evidence5/update_contract5.py`, `memory_handoff_recheck5.py` | regenerate `../context_contract.json` and `memory_handoff_recheck.json` |
| `review2_fixes/calibration/` | HF calibration script and logits source, p150x2 story control |
| `evidence3/hf_trace_prompt_margin.json`, `evidence3/sampling_repeat/` | HF nucleus check; p150x2 sampling repeat test |
| `evidence2/m*/watcher_probe.*`, `probe_control/` | watcher runs and controls |
| `evidence2/m*/profile/` | perf reports and summaries |
| `evidence2/continuation/` | HF KL probe for continuation |
| `evidence2/slots/` | p150 32-slot build and failed attempts |
| `stage_review1.md`, `stage_review2.md`, `stage_review3.md` | independent stage reviews |

## Reproduction

```bash
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal && source python_env/bin/activate
export PYTHONPATH=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PYTHONPATH
bash /home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/evidence_chain5.sh
python /home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/calibration/compare.py 4 2 1
bash /home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence2/evidence_chain2.sh
```
