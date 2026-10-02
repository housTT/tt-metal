# kev-9b stage 1 (functional): device engine

Date: 2026 Oct 01. Box: p300c, 4 Blackhole chips, device 0 used. tt-metal worktree `/home/hous/dev/kev/tt-metal` at 7eac776e9 (branch `hous/kev-9b-bringup`). Full chronology with per-run tables: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/functional/work_log.md`, section `## device`.

## What was built

- `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/engine.py`: `KevEngine`, a composition wrapper around the unmodified `Qwen36Model` (`/home/hous/dev/kev/tt-metal/models/demos/blackhole/qwen36/tt/model.py`). No file under `models/demos/blackhole/qwen36/` was changed.
  - `KevEngine(device, args_cls=None, max_state_len=8192, n_layers=None, chunk_size=2048, snapshot_slots=8)`. `args_cls` defaults to `KevModelArgs` from `tt/loader.py` (LoRA-merged weights) and falls back to `Qwen36ModelArgs`. KV cache and page table cover `max_state_len + 2048` tokens. Every GDN (Gated DeltaNet) layer runs in in-place chunk-state mode; `snapshot_slots` persistent snapshot buffer sets (recurrent plus fused conv state per GDN layer) are allocated up front.
  - `prefill_hidden(token_ids [1,T], positions) -> float32 [len(positions), 4096]`: eager reference path. Segments of 2048 tokens go through the upstream masked fixed-bucket forward (`_forward_prefill_chunk_masked`) with `chunk_start` carried; requested rows are selected by a one-hot matmul (HiFi4, fp32 accumulate), passed through the final RMSNorm, and read back.
  - `prefill_state(state_ids [1,S], slot=0) -> StateHandle(S, S0, suffix_ids, slot)`: `S0 = floor(S/128)*128`; runs `[0, S0)` and copies the GDN state into snapshot `slot`. KV blocks below `S0 // 64` stay filled.
  - `question_hidden(handle, question_ids [1,Q], positions_in_question) -> [n, 4096]`: restores the snapshot, runs `state[S0:S] ++ question` at `chunk_start = S0` in its natural bucket, reads rows `(S - S0) + p`. Repeatable: question 3 equals question 1 bit for bit in every tested case.
- `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tests/test_engine.py`: `test_hidden_vs_hf`, `test_readout_matches_upstream_logits`, `test_tail_matches_full_row`, `test_reference_records`. `n_layers` is parametrized as `l4` (smoke) and `l32` (full; carries the registered `slow` marker, so `-m "not slow"` deselects it). The 4-layer bar (0.97) is a smoke bar for compile and wiring checks, not a correctness bar; the acceptance evidence is the 32-layer runs.
- `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/scripts/reference_control.py`: the base-weights control. It runs the 29 reference rows through `prefill_hidden` twice, once with `KevModelArgs` (merged LoRA) and once with the unmerged `Qwen36ModelArgs`, and writes `/home/hous/dev/kev/reports/stage1_control.json` (`--mode merged`, `--mode base`, then `--mode summarize`).

## How to run

Every device command goes through the lock wrapper. Environment used for all runs:

```
HF_MODEL=/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404
KEV_RUN=/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0
MESH_DEVICE=P150
TT_CACHE_PATH=/home/hous/dev/kev/tt_cache
```

```
cd /home/hous/dev/kev/tt-metal
/home/hous/dev/kev/bin/devrun timeout 1800 pytest models/demos/blackhole/qwen36/tests/test_prefill.py -k "test_masked_bucket_matches_reference and (len50_b128 or len1500_b2048)" --timeout=1800
/home/hous/dev/kev/bin/devrun timeout 1800 pytest models/autoports/jaredpalmer_kev_9b/tests/test_engine.py -k l4 --timeout=1800
/home/hous/dev/kev/bin/devrun timeout 7200 pytest models/autoports/jaredpalmer_kev_9b/tests/test_engine.py -k l32 --timeout=1800
/home/hous/dev/kev/bin/devrun timeout 1800 pytest models/autoports/jaredpalmer_kev_9b/tests/test_engine.py -k test_reference_records --timeout=1800
TT_METAL_WATCHER=10 /home/hous/dev/kev/bin/devrun timeout 1800 pytest models/autoports/jaredpalmer_kev_9b/tests/test_engine.py -k "hidden_vs_hf and l4 and 300" --timeout=1800
```

Device params: `{"l1_small_size": 24576, "num_command_queues": 2, "trace_region_size": 0}`. Stage 1 captures no trace, so the trace region is left at 0 (dynamic). The upstream chunk trace (stage 3) will need an explicit size; not measured here. `pytest.ini` sets a 300 s default timeout, so `--timeout=1800` (or the per-test `timeout` marker already on each test) is required.

Weight caches (`du -sh` and `find -type f | wc -l` on 2026 Oct 01, 20:30 UTC, host clock): base `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8` 8.3 GB, 403 files; merged kev `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf` 8.3 GB, 403 files. Model load from a warm cache: 11 to 12 s for 32 layers.

Logs: `/home/hous/dev/kev/logs/stage1_sanity.log`, `stage1_engine_l4.log`, `stage1_engine_l32.log`, `stage1_engine_round3.log`, `stage1_reference_records.log`, `stage1_watcher_run.log`, `stage1_watcher.log` (copy of `generated/watcher/watcher.log`), `stage1_final.log`, `stage1_hf_bf16_vs_fp32.log`, `stage1_control.log` (base-weights control), `stage1_head_cpu_test.log`, `stage1_loader_cpu_test.log`, `stage1_loader_cpu_test_hostrun.log`. The table of which test ran in which log is in the work log, section `## Stage 1 test ledger`.

## Results

Upstream sanity (`test_masked_bucket_matches_reference`, 4 layers): len50_b128 logit PCC 1.000001, len1500_b2048 logit PCC 0.999954, argmax equal, GDN state PCC 1.0001 / 1.0000. The backbone works on this box.

Hidden rows vs HF (`Qwen3_5ForCausalLM`, bf16 CPU, same unmerged base on both sides, 8 sampled positions from 1 to T-1 including T-1):

| layers | T | min PCC over positions | bar | result | TT time | HF time |
|---|---|---|---|---|---|---|
| 32 | 300 | 0.9945 (positions from 0) | 0.99 | pass | 1.1 s | 3.2 s |
| 32 | 1500 | 0.9930 (0.9772 at position 0 when included) | 0.99 | pass | 3.5 s | 19.3 s |
| 32 | 2300 | 0.9921 | 0.99 | pass | 3.3 s | 31.0 s |
| 4 | 300 | 0.9788 | 0.97 | pass | 0.4 to 0.7 s warm | 0.2 s |
| 4 | 1500 | 0.9835 | 0.97 | pass | 1.5 s | 2.7 s |
| 4 | 2300 | 0.9813 | 0.97 | pass | 1.8 s | 4.1 s |

Readout isolation (`test_readout_matches_upstream_logits`: engine row times the device `lm_head` weights vs upstream `prefill_masked_bucket` logits): PCC 0.999939 (4 layers, T=300), 0.999935 (4 layers, T=1500, top-2 flip 66 vs 67), 0.999902 (32 layers, T=1500, argmax equal). The wrapper reproduces the upstream path; the gap to HF is backbone precision (bf8 weights, LoFi GDN), consistent with upstream's own full-model bar of 0.91 in `qwen36/tests/test_model.py`.

Continuation (`test_tail_matches_full_row`, 32 layers, 4 question positions including 0 and Q-1, three questions back to back):

| S | S0 | Q | state prefill | question (first / steady) | min row PCC vs full row | q3 == q1 |
|---|---|---|---|---|---|---|
| 2048 | 2048 | 50 | 2.36 s | 0.66 / 0.15 s | 1.000000 | yes |
| 2175 | 2048 | 50 | 2.56 s | 0.76 / 0.20 s | 1.000000 | yes |
| 2200 | 2176 | 50 | 2.84 s | 0.19 / 0.25 s | 0.999878 | yes |
| 2048 | 2048 | 400 | 2.71 s | 0.91 / 0.50 s | 1.000000 | yes |
| 2175 | 2048 | 400 | 1.91 s | 1.18 / 0.67 s | 1.000000 | yes |
| 2200 | 2176 | 400 | 2.40 s | 0.72 / 0.48 s | 0.999872 | yes |

The 4-layer runs give the same picture (1.000000 or 0.9998+). S=2200 is not bit-identical to the full row because the tail runs at chunk_start 2176 in bucket 128 while the full row runs its second segment at chunk_start 2048 in bucket 256.

Reference records (`test_reference_records`, merged weights, 32 layers, 29 rows from `/home/hous/dev/kev/reports/reference/rows.json`, HF fp32 reference, `/home/hous/dev/kev/logs/stage1_final.log`): argmax agreement 29/29, max |dp| 0.1024 (row 14:return_reason), mean |dp| 0.0366 (mean over the 29 questions of each question's max |dp|, kev's `agreement()` definition). The hidden-state PCC at the readout positions is not above 0.99 for every real row: 11 of 29 rows are below 0.99, min 0.9768 (0:choice), median 0.9923, max 0.9972. The rows below 0.99 are 0:choice 0.9768, 0:meets 0.9808, 1:affected 0.9824, 2:choice 0.9843, 2:meets 0.9836, 3:decision 0.9787, 4:action 0.9899, 8:product 0.9859, 9:product 0.9894, 10:product 0.9861, 11:product 0.9866 (the hard-v1 and cfpb records); the synthetic and devtools rows are above 0.99. The test floor is 0.97 with 29/29 argmax agreement required; the classification of the gap is in the next section. Per-row table in the work log. Full-row eager times: 0.12 s at T=35 to 1.7 s at T=2265 (2.56 s at T=1585 includes a bucket compile).

Through the HTTP server (stage 2, state prefix pass plus GDN snapshot plus one tail per question, `/home/hous/dev/kev/reports/stage2_parity.json`, details in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/server/work_log.md`): the same 29 questions give max |dp| 0.0947 and mean |dp| 0.0359 against fp32, 0.0992 and 0.0366 against the CPU bf16 reference, 0 argmax flips. For scale, the CPU bf16 kev reference against fp32 is max 0.0189, mean 0.0034, 0 flips.

### Base-weights control and the precision mechanism

Control run (`scripts/reference_control.py`, 2026 Oct 01, 20:32 to 20:33 UTC, device 0, 32 layers, `prefill_hidden` on each full row, same pointer head, log `/home/hous/dev/kev/logs/stage1_control.log`, report `/home/hous/dev/kev/reports/stage1_control.json`). The merged run reproduces `test_reference_records` exactly; the base run sends the unmerged Qwen3.5-9B-Base through the same device path, so it measures how much of the adapter the merged device path carries.

| weights | args class | weight cache | min row PCC | median row PCC | rows < 0.99 | argmax agree vs fp32 | max dp | mean dp (row max) |
|---|---|---|---|---|---|---|---|---|
| merged LoRA | `KevModelArgs` | `tensor_cache_bfp8_kev_2b2a70cf` | 0.9768 | 0.9927 | 11/29 | 29/29 | 0.1024 | 0.0366 |
| base, unmerged | `Qwen36ModelArgs` | `tensor_cache_bfp8` | 0.4554 | 0.5973 | 29/29 | 9/29 | 0.9471 | 0.4295 |

Per-position hidden PCC for the five worst merged rows (options in order, then the decide position):

| row | T | per-position PCC |
|---|---|---|
| 0:choice | 337 | opt0 0.9887, opt1 0.9768, opt2 0.9872, opt3 0.9872, opt4 0.9912, opt5 0.9931, decide 0.9946 |
| 3:decision | 181 | opt0 0.9916, opt1 0.9856, opt2 0.9787, decide 0.9877 |
| 0:meets | 311 | opt0 0.9808, opt1 0.9907, decide 0.9910 |
| 1:affected | 173 | opt0 0.9870, opt1 0.9940, opt2 0.9919, opt3 0.9824, decide 0.9910 |
| 2:meets | 312 | opt0 0.9901, opt1 0.9836, decide 0.9865 |

On 28 of 29 merged rows the worst position is an option `<|box_end|>` position (on 4:action it is the decide position, 0.9899). The decide position is 0.9865 or better on every row and below 0.99 on three rows (2:meets 0.9865, 3:decision 0.9877, 4:action 0.9899). The five worst base rows sit at 0.4554 to 0.5254 (0:choice, 2:choice, 5:change_type, 7:message_match, 7:change_type), and 20 of 29 base rows flip the argmax. The per-row table for both runs is in `/home/hous/dev/kev/reports/stage1_control.json` (`per_row`).

Mechanism. The backbone port stores MLP gate and up projections in bfp4 (block floating point, 4 bits per element with a shared block exponent) and down, attention and GDN projections in bfp8 (8 bits per element) (`/home/hous/dev/kev/tt-metal/models/demos/blackhole/qwen36/tt/mlp.py:154-158`, `attention/weights.py:22`, `gdn/weights.py:83,117`); the KV cache is bf16 (`engine.py`, `allocate_kv_caches`). The merged LoRA delta is 0.5 to 2.6 percent of each weight's norm (reviewer's CPU measurement). The reviewer's CPU block-float emulation (16-element blocks, shared exponent; the bfp8 emulation matched the ttnn conversion at 0.0076 relative error; `/home/hous/dev/kev/reports/review_stage1.md`, P1 mechanism evidence) gives bfp8 rounding noise of 0.0075 of |W| and cos(Q(W+d) - Q(W), d) of 0.52 to 0.85, and bfp4 noise of 0.115 of |W| with cos 0.19 to 0.24 on gate/up. So the fine-tune signal is of the same order as bfp8 rounding and well under bfp4 rounding on gate/up. The control run shows the merged device path still carries most of the adapter (0.9768 to 0.9972 and 29/29 against 0.4554 to 0.6654 and 9/29 for base), so the gap between the real-row PCC and the 0.99 contract bar is a precision loss of the adapter in the weight formats, not a wiring fault. Stage 4 (datatype sweep) owns the remedy; the first candidate is bfp8 for gate/up on the merged weights, measured against these 29 rows with the same `reference_control.py` run.

Watcher (`TT_METAL_WATCHER=10`, two 4-layer `test_hidden_vs_hf` cases): both passed with identical PCCs; `/home/hous/dev/kev/logs/stage1_watcher.log` has 4844 lines and no error, assert, overflow or hang entries.

## Unverified or open

- T=8200 `prefill_hidden` vs HF was not run (time); the segment loop is the same code the 2300 case exercises across two segments.
- Resolved after stage 1. At stage 1 the snapshot slots shared one KV cache and one page table (`engine.py`), so a `question_hidden` on a cached handle after another state's `prefill_state` read the other state's K/V for the 8 full-attention layers (review P1, `/home/hous/dev/kev/reports/review_stage1.md`). Stage 3 gave every slot its own range of paged KV blocks and its own page table (`doc/optimized/README.md`, `doc/optimized/work_log.md`). `tests/test_engine.py::test_slots_interleaved` (two states in two slots; question on A, on B, on A again bit-identical; each against `prefill_hidden` of the concatenated row) passes at 4 and 32 layers in eager, traced and policy modes: `/home/hous/dev/kev/logs/stage3_engine_l32.log` (eager and traced, 32 layers) and `/home/hous/dev/kev/logs/stage4r_engine_l32_policy.log` (eager, traced and policy, 32 layers, selected precision; min PCC against the full row 0.99984 for A and 0.99989 for B).
- `trace_region_size` for the stage 3 chunk trace is not measured.
- The 32-layer hidden PCC above 0.99 at every non-zero position holds for the random-token rows with base weights (`test_hidden_vs_hf`). It does not hold for the real kev rows with merged weights: 11 of 29 rows sit between 0.9768 and 0.99 (see Results). The 4-layer bar of 0.97 is a smoke bar. The reference-records floor of 0.97 is kept with the classification above attached (adapter precision loss through bfp4 gate/up and bfp8 projections, not a wiring fault) and with 29/29 argmax agreement required. Whether max |dp| 0.10 and mean |dp| 0.037 are acceptable for the product, and whether bfp8 gate/up recovers the gap, is the stage 4 (datatype sweep) question.
- HF reference for `test_hidden_vs_hf` is bf16; bf16 vs fp32 HF agree at PCC 0.99994 or better on the 4-layer T=300 input, so this does not move the numbers.
- Tile-boundary lengths (1, 31, 32, 33) were not run; the shortest row exercised on device is T=35 (12:urgent). The engine pads every segment to a 128-multiple bucket, so these lengths share the T=35 path, but they are not covered.
- The watcher run covered `prefill_hidden` only, at 4 layers with base weights. `prefill_state` and `question_hidden` (the `ttnn.copy` snapshot path) and the merged weights were not run under the watcher.
