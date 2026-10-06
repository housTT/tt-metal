# Second checkpoint `convaiinnovations/laya-typed-decisions` on p150: 1024-token bucket, gates, served check

PLAN.md section 5 row R2 and user decision 3 (Track T4). The sibling checkpoint (Hub revision
`e929ae5cf69bc34259cd2f95c9e91145b818b1f0`, snapshot `/home/hous/dev/laya/state/laya_models/laya-typed-decisions`) has the
same architecture and tokenizer as the English checkpoint, `rl_agent_config.json` with `max_len` 1024 and `head_max_len` 256,
the same `temperature_by_options` table, per-type temperatures [1.0148, 1.0374, 1.0575], and float16 for all 206 tensors.
This stage adds the 1024-token seq bucket, builds the sibling gate corpus, runs the stage 6 gates and the tracked replay on
the sibling, serves it on the host and runs E1 to E5, and lists the values the orchestrator must put into
`tt-model-typed-decisions.yaml`. PCC = Pearson correlation coefficient. SDPA = scaled dot-product attention. GeGLU = the
gated GELU MLP of ModernBERT. Every timing carries the host 1-minute load average (amendment A5: under 8).

## Files

| file | role |
|---|---|
| `tt/model_config.py` | `ROW_BUCKETS_AT_1024 = (1, 2, 4, 5, 8, 10, 16)`, `SEQ_BUCKETS_SIBLING = (128, 256, 512, 1024)`, `ROW_BUCKETS_BY_SEQ_SIBLING`, `rows_for_seq`, `select_bucket(..., row_buckets_by_seq)`, `deployment_buckets(..., row_buckets_by_seq)`, `sibling_deployment_buckets`, `parse_buckets('sibling')` and `'sibling1024'`; the English constants and the 30-bucket set are unchanged |
| `tt/laya_model.py` | `TtnnLayaModel(..., row_buckets_by_seq)`, `rows_for_seq`, `max_rows_for_seq`, `deployment_buckets`, `bucket_for` picks the seq bucket first and the row bucket from that seq's list |
| `tt/runner.py` | the default bucket list comes from the model's per-seq lists |
| `tt/engine.py` | `LAYA_ROW_BUCKETS_<seq>` (for example `LAYA_ROW_BUCKETS_1024`), `row_buckets_by_seq` in the constructor and in `shapes()` (`row_buckets_by_seq`, `max_rows_by_seq`), calls above the largest row bucket of their seq bucket run as consecutive replays of that bucket (`_run_device`, `last_buckets`) |
| `common.py` | `checkpoint_pins(model_dir)` reads the Hub repo and revision from the snapshot path |
| `reference/corpus.py` | `--tag td` writes `parity_corpus_td.npz`, `parity_corpus_td_index.json`, `typed_decisions_cpu_td/`; the index records the checkpoint's own pins; the English names and defaults are unchanged |
| `tests/run_fidelity.py`, `tests/decision_agreement.py` | `--corpus`, `--index`, `--model-dir`, `--row-buckets-by-seq`; `--largest-rows` (defaults unchanged: the English protocol) |
| `tests/bench_served_long.py` | served latency with a long state (every row in the 1024 bucket), bench_latency.py shapes |
| `tests/test_sibling_config.py` | no device: constants, per-seq selection, env parsing, the seven 1024 plans, the engine's chunking, pins, corpus names, the sibling corpus against the English gate subset |
| `tests/test_sibling_release.py` | gates over the evidence files of this directory |
| `served_token_lengths_td.json` | E1, E2 and E3 row lengths at the sibling budget through the server's request-to-rows path |
| `sanity_reference_typed_decisions.json` | the STATE_EN / Q_CHOICE answer of the sibling from the server's CPU backend (for the server's startup check) |
| `ab/bench_*.json` | the 1024 A/B matrix (one process per variant, 20 replays after 3 warm) |
| `bench_sibling_final.json` | all 37 sibling buckets on the final configuration |
| `replay_trace_check_sibling.json` | the tracked replay over the 37 buckets in one process |
| `fidelity_bf8w_hifi3_erf_td_natural.json`, `fidelity_bf8w_hifi3_erf_td_seq1024.json` | stage 6 fidelity gates on the sibling, natural buckets and every call forced into the 1024 bucket |
| `decision_agreement_bf8w_hifi3_erf_td_natural.json`, `decision_agreement_bf8w_hifi3_erf_td_seq1024.json` | alone versus in batch, natural buckets (B 64 at 512) and forced 1024 (largest bucket 16x1024) |
| `served_health_td.json`, `served_startup_lines_td.txt`, `served_long_state_td.json` | the host-served sibling: health, startup lines (sanity check), long-state latency at 1024 |
| `work_log.md` | timeline and decisions |

Served evaluation results: `/home/hous/dev/laya/evals/results/host_tt_td-p150_b1_<stamp>/SUMMARY.md` (path in the work log).
Reference corpus: `/home/hous/dev/laya/reference/parity_corpus_td.npz`, `parity_corpus_td_index.json`,
`typed_decisions_cpu_td/`. The English corpus files were not touched (md5 and timestamps checked).

## Bucket set and design

The sibling deployment set is the English set (seq 128, 256, 512 by rows 1, 2, 4, 5, 8, 10, 16, 32, 50, 64) plus seq 1024 by
rows 1, 2, 4, 5, 8, 10, 16: 37 traces. The plan said rows 1, 2, 4, 8, 16 at 1024; 5 and 10 are added for the five-question
protocol (one typed-decisions case is five rows, two cases are ten). Selection rule unchanged: the smallest seq bucket that
fits the longest row of the call, then the smallest row bucket of that seq bucket that fits the rows.

Why the engine needs a per-seq row list and a serving cap. `server/engine.py` keeps one flat row list for every seq bucket
(`Buckets`), plans its chunks with `rows_for(n) * seq_for(longest) <= LAYA_MAX_BATCH_TOKENS` and writes `X-Laya-Batch`
from that plan; this track does not edit `server/`. The engine therefore:

1. reads `LAYA_ROW_BUCKETS_1024=1,2,4,5,8,10,16` (any `LAYA_ROW_BUCKETS_<seq>`) and captures only those buckets at 1024;
2. reports `row_buckets` as the union of all lists (1 to 64), so the server's 413 limits and its chunk plans at 128, 256 and
   512 are bit-identical to the English bundle;
3. runs a call with more rows than the largest row bucket of its seq bucket as consecutive replays of that bucket and
   concatenates the outputs (`tt/engine.py: _run_device`; `last_buckets` lists them; `device_ms` is the sum). This is the
   safety net for `/v1/forward` callers and for a server without the cap below.

For the served package the manifest adds `LAYA_MAX_BATCH_TOKENS=16384` (16 rows x 1024 tokens). The server's own plan then
never exceeds a captured 1024 bucket and the `X-Laya-Batch` header stays exact. The cap also limits a chunk at 512 tokens to
32 rows (the 50x512 and 64x512 traces stay captured for direct callers): a batch of 8 states x 5 questions with rows
over 256 tokens runs as 32x512 plus 8x512 (267 + 65 ms in the stage 7 table) instead of one 50x512 call (428 ms), and 64
states x 5 questions as ten 32x512 calls (2.67 s) instead of five 64x512 calls (2.62 s). At 128 and 256 tokens nothing
changes (64 x 256 = 16384).

## Per-bucket plans at 1024 (`DEFAULT_PORT`, policy `bf8w_hifi3_erf`)

The stage 3 and 7 levers are keyed by rows = B x 1024, so the 1024 buckets inherit them: the L1 attention chain to 4096 rows
(SDPA chunk 128 there), DRAM above; `minimal_matmul` on 11x10 for Wqkv everywhere and for Wo from 4096 rows; SDPA on 8x8 with
chunk 256 from 768 rows (1024 is in `SDPA_MEASURED_SEQ_LENS`); GeGLU padded to 2816, block-sharded at 1024 and 2048 rows,
interleaved with the stage 7 grid rule above; rotary never sharded. New in this stage: `PortConfig.sdpa_full_grid_buckets =
((5, 1024), (10, 1024))` puts SDPA on the full 11x10 grid at exactly those two buckets (the A/B below); the override is keyed
by (rows, seq), so no English bucket changes. The static per-bucket mask tensors (`band` and `zeros`, `2 x B x S x S x 2`
bytes) take 184 MiB of DRAM for the seven 1024 buckets (English set: 252 MiB).

| rows x seq | rows | attention chain | GeGLU plan (grid) | Wqkv | Wo | SDPA chunk (grid) | static masks (MiB) |
|---|---|---|---|---|---|---|---|
| 1x1024 | 1024 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 (8x8) | 4 |
| 2x1024 | 2048 | L1 | block-sharded 2816 (8x8) | minimal 11x10 | core grid 8x8 | 256 (8x8) | 8 |
| 4x1024 | 4096 | L1 | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 128 (8x8) | 16 |
| 5x1024 | 5120 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 256 (11x10) | 20 |
| 8x1024 | 8192 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 (8x8) | 32 |
| 10x1024 | 10240 | DRAM | interleaved 2816 (11x10) | minimal 11x10 | minimal 11x10 | 256 (11x10) | 40 |
| 16x1024 | 16384 | DRAM | interleaved 2816 (11x8) | minimal 11x10 | minimal 11x10 | 256 (8x8) | 64 |

## A/B at the 1024 buckets (`ab/bench_*.json`; traced p50 ms of 20 replays after 3 warm, one process per variant)

Method as in stage 7 (`tests/bench_buckets.py --buckets sibling1024`, sibling weights, real typed-decisions rows, bit identity
of traced against eager checked per cell, load per cell; a load gate under 8, in the second queue under 4, before every
variant). Scatter: four runs of the default configuration (`default_a` to `default_d`, loads 0.5 to 1.9) agree within 0.4
percent at every bucket, so differences above about 2 percent are readable. Jobs `ab_1024` and `ab2_1024`, logs
`/home/hous/dev/laya/logs/p6_ab_1024_20261006T004955Z.log` and `p6_ab2_1024_20261006T011452Z.log`.

| variant (`PortConfig` overrides) | 1x1024 | 2x1024 | 4x1024 | 5x1024 | 8x1024 | 10x1024 | 16x1024 | outcome |
|---|---|---|---|---|---|---|---|---|
| default (`default_a`, load 1.5) | 22.00 | 39.79 | 74.62 | 104.97 | 153.59 | 202.68 | 301.80 | the stage 7 rules |
| `default_b`, `default_c`, `default_d` (loads 0.5 to 1.9) | -0.1 to +0.1 | -0.4 to -0.2 | -0.1 to +0.1 | -0.2 | -0.1 to 0.0 | -0.1 to 0.0 | 0.0 to +0.1 | scatter |
| `sdpa_128` (`sdpa_large_chunk_rows` 1000000) | +3.2 | +3.9 | +0.3 | +9.9 | +7.1 | +9.8 | +7.1 | loses; 256 kept |
| `sdpa_512` (`sdpa_q_chunk`, `sdpa_k_chunk` 512) | TT_THROW | | | | | | | illegal: static circular buffers on the 8x8 grid grow to 2,966,528 B against the 1,572,864 B L1 |
| `sdpa_full_grid` (11x10 everywhere; first run load 5.6 to 7.0, second run `_b` load 1.2 to 1.3) | +12.9 / +12.6 | +4.0 / +3.8 | +5.3 / +4.7 | -8.2 / -8.7 | +2.3 / +2.1 | -11.0 / -11.2 | -0.5 / -0.4 | wins only at 5 and 10 rows |
| `sdpa_full_5_10` (`sdpa_full_grid_buckets` ((5, 1024), (10, 1024)); load 1.1 to 1.2) | -0.2 | -0.3 | +0.2 | -8.6 | 0.0 | -11.2 | 0.0 | adopted |
| `chain_dram` (`l1_attention_max_rows` 0) | +10.5 | +11.2 | +8.5 | -0.3 | -0.1 | -0.1 | 0.0 | L1 chain to 4096 rows kept |
| `geglu_interleaved` | -0.5 | -1.5 | -0.1 | -0.2 | -0.1 | -0.1 | +0.1 | within scatter at 1 and 2 rows; the block-sharded plan kept (stage 6 and 7 numerics) |
| `wo_minimal_all` (`wo_minimal_min_rows` 0) | +0.9 | 0.0 | +0.1 | 0.0 | -0.1 | -0.1 | 0.0 | within scatter; threshold 4096 kept |

Reading of the SDPA grid result: at seq 1024 with 256-token chunks SDPA has B x 16 x 4 work units; the 8x8 grid runs 5 and
10 rounds at 5 and 10 rows where the 11x10 grid needs 3 and 6, and at 1, 2, 4, 8 and 16 rows the larger grid is slower
(per-core work too small at 1 to 4 rows, no round saved at 8 and 16). The first eight marker logits of every cell are
identical between the two grids and across the default runs, so the grid is a timing choice only; the forced-1024 fidelity
and invariance below were re-run on the adopted configuration regardless. The rule is not keyed by rows alone because
5120 and 10240 rows are also the English 10x512 bucket (and a 20x512 call), which stage 7 did not measure with the full grid.

## Sibling gate corpus (`/home/hous/dev/laya/reference/parity_corpus_td.npz`)

Built by `python -m models.autoports.convaiinnovations_laya.reference.corpus --model-dir <sibling> --max-len 1024
--head-max-len 256 --tag td` with the vendored code, fp32, eager attention, 6 threads (job `corpus_td`, 179 s: typed 92 s,
parity_fast 82 s). The same 200-decision gate subset as the English corpus (10 cases per workflow, `RandomState(13)`,
case ids identical) plus the 288 parity_fast questions: 488 items, kmax 6; probabilities under the pip clamp rule
(`probs`) and the raw Hub rule (`probs_hub`); the index records `hf_model convaiinnovations/laya-typed-decisions`,
the revision, the safetensors sha256, max_len 1024, head_max_len 256.

Finding: every row has the same token length as in the English corpus (typed 130 to 510 tokens, parity_fast 62 to 424; no
row above 512), because no gate or parity_fast state was truncated at 512 / 192. The gate protocol therefore never reaches
the 1024 bucket on its own, so the stage 6 gates are run twice on the sibling: at the natural buckets and with every call
forced into the 1024 bucket (`--seq-buckets 1024`). The sibling's confident decisions (margin >= 0.10) are 152 of 200
(English 149); its argmax equals the gold label on 158 of 200 (0.79; published accuracy 0.766).

## Served token lengths at the sibling budget (`served_token_lengths_td.json`)

Rows of the exact E2 and E3 requests through `server/engine.py: encode_state` at max_len 1024 / head_max_len 256:

| source | rows | min | p50 | mean | p95 | max | rows in 128 / 256 / 512 / 1024 | calls by longest row |
|---|---|---|---|---|---|---|---|---|
| E2 typed-decisions (400 cases x 5) | 2000 | 124 | 308 | 291.2 | 428 | 597 | 17 / 630 / 1319 / 34 | 256: 110, 512: 283, 1024: 7 |
| E2 at 512 / 192 (the English budget) | 2000 | 124 | 308 | 290.5 | 428 | 512 | 17 / 630 / 1353 / 0 | 256: 110, 512: 290 |
| E3 AG News (400 x 1) | 400 | 68 | 103 | 104.8 | 139 | 232 | 365 / 35 / 0 / 0 | unchanged from 512 / 192 |
| E3 DAIR Emotion (400 x 1) | 400 | 38 | 51 | 54.2 | 75 | 97 | 400 / 0 / 0 / 0 | unchanged from 512 / 192 |
| E1 corpus, typed gate rows | 200 | 130 | | 300.9 | | 510 | 0 / 59 / 141 / 0 | |
| E1 corpus, parity_fast rows | 288 | 62 | | 152.4 | | 424 | 172 / 68 / 48 / 0 | |

The seven E2 cases above 512 tokens (`customer_service_000019, 000020, 000027, 000044, 000050, 000051, 000083`, longest row
567 to 597) are the seven cases that `usage.truncated` flagged at 512 / 192; at the sibling budget they run at 5x1024. The
speed table's STATE_EN rows (194 to 205 tokens) stay in the 256 bucket, so the 1024 cells of the card come from the long-state
run of `tests/bench_served_long.py` (the STATE_EN ticket with its message repeated 30 times, rows of about 860 tokens).

## Stage 6 gates on the sibling (policy `bf8w_hifi3_erf`, 488 items, chip 0)

`tests/run_fidelity.py --policy bf8w_hifi3_erf --items all --hidden-cases 40 --corpus parity_corpus_td.npz --model-dir
<sibling>` twice: natural buckets (`--seq-buckets 128,256,512,1024 --row-buckets 1,2,4,5,8,10,16`; the 100 calls ran at
5x128 (24), 5x256 (26), 5x512 (38), 4x256 (10), 4x512 (2); load 2.8 to 3.0) and every call forced into the 1024 bucket
(`--seq-buckets 1024`; 88 calls at 5x1024, 12 at 4x1024; load 5.1 to 5.6). Confident = the CPU fp32 top-1 minus top-2
probability >= 0.10 (152 of the 200 gate decisions for this checkpoint). Job `gates_td`, log
`/home/hous/dev/laya/logs/p6_gates_td_20261006T005409Z.log`.

| gate (200 gate decisions) | natural buckets | forced 1024 bucket | threshold | result |
|---|---|---|---|---|
| confident argmax agreement | 152 of 152 | 152 of 152 | >= 98 percent | pass |
| median over decisions of max abs delta p | 0.00312 | 0.00313 | <= 0.02 | pass |
| scorer-logit PCC over the gathered markers | 0.99984 | 0.99985 | >= 0.99 | pass |
| hidden-state PCC encoder output, 40 cases pooled (worst call) | 0.99849 (0.99689) | 0.99850 (0.99695) | >= 0.99 | pass |
| hidden-state PCC head output, pooled (worst call) | 0.99990 (0.99977) | 0.99990 (0.99977) | >= 0.99 | pass |
| NaN | 0 | 0 | 0 | pass |

Reported, not gated: plain argmax agreement 199 of 200 in both runs (natural: `security_incidents_000068/urgency`, reference
margin 0.0027; forced 1024: `agent_trace_observability_000031/needs_review`, margin 0.0089); act argmax 200 of 200 (act-logit
PCC 0.99999); p95 of max abs delta p 0.0094 / 0.0100, max 0.0136 / 0.0143, mean 0.0037 / 0.0039; max abs logit delta 0.12 /
0.11. By type (natural run): choice 60 of 60 argmax, 43 of 43 confident, median 0.0033, PCC 0.9999; score 79 of 80, 63 of 63,
0.0034, 0.9998; noul 60 of 60, 46 of 46, 0.0024, 0.9998. All 488 items: argmax 485 of 488, confident 383 of 383, median
0.0039 / 0.0040, p95 0.0153 / 0.0159, max 0.0402 / 0.0345, scorer PCC 0.99891 / 0.99897; parity_fast alone 286 of 288, 231 of
231 confident, median 0.0050, PCC 0.9985.

The sibling's device deltas are about three times smaller than the English checkpoint's (stage 6: median 0.0108, PCC 0.9962,
encoder hidden PCC 0.9955). The fine-tuned weights produce scorer logits in a narrower range (-3.0 to 14.2 against -5.5 to
20.7) and the per-type temperatures are about 1.0 instead of 1.6 to 2.0, so the same bf16 noise moves the probabilities
less; the encoder PCC also rises (0.9985 against 0.9955, worst call 0.9969 against 0.9912). The shipped policy therefore
passes every gate on the first run and `bf16_hifi4` was not run (the plan runs it only after a failed gate).

The forced-1024 run is the gate on the 1024 plans: padding every row of the corpus to 1024 tokens (the 5x1024 bucket for
88 calls, 4x1024 for 12) gives the same quality as the natural buckets, so the 1024 masks, the 1024 rotary caches, the
256-token SDPA chunks and the 11x8 and 11x10 GeGLU configs at 4096 to 5120 rows are numerically sound.

## Alone versus in batch on the sibling (`decision_agreement_bf8w_hifi3_erf_td_*.json`)

16 gate questions (every 12.5th gate row: 5 choice, 5 score, 6 noul) alone, in B 2, B 4, a mixed B 8 (seed 13) and the
largest bucket. Natural buckets (the English protocol, 37 buckets captured, load 1.1): alone at 1x256 (5) and 1x512 (11), B 2
at 2x256 (1) and 2x512 (7), B 4 at 4x512, mixed B 8 at 8x512, the largest bucket 64x512 (16 questions plus 48 filler rows).
Forced 1024 (`--seq-buckets 1024 --largest-rows 16`, load 2.2): alone at 1x1024, B 2 at 2x1024, B 4 at 4x1024, mixed B 8 at
8x1024, the largest 1024 bucket 16x1024 (the 16 questions alone in one call).

| placement against alone | natural: same argmax / max abs dp / median / max abs dlogit / PCC | forced 1024: same argmax / max abs dp / median / max abs dlogit / PCC |
|---|---|---|
| B 2 | 16 of 16 / 0.0028 / 0.0017 / 0.026 / 0.99997 | 16 of 16 / 0.0 / 0.0 / 0.0 / 1.0 |
| B 4 | 16 of 16 / 0.0028 / 0.0017 / 0.026 / 0.99997 | 16 of 16 / 0.0064 / 0.0010 / 0.036 / 0.99997 |
| mixed B 8 | 16 of 16 / 0.0046 / 0.0014 / 0.022 / 0.99998 | 16 of 16 / 0.0067 / 0.0012 / 0.042 / 0.99996 |
| largest bucket (64x512 / 16x1024) | 16 of 16 / 0.0056 / 0.0016 / 0.039 / 0.99996 | 16 of 16 / 0.0067 / 0.0012 / 0.042 / 0.99996 |

Gates: the same argmax in every placement on 16 of 16 questions, max abs delta p between alone and any in-batch placement
0.0056 (natural) and 0.0067 (forced 1024) against the threshold 0.01: pass (English stage 7: 0.0090). The largest
difference between any two placements is 0.0067 in both runs; every placement agrees with the CPU fp32 argmax on 16 of 16
with max abs delta p 0.0082 to 0.0149. At 1024 the 1x1024 and 2x1024 buckets give bit-identical logits (both run the same
8x8 block-sharded GeGLU plan, as 2x512 and 4x512 did in stage 6), and the mixed B 8 and B 16 placements are identical to
each other (both DRAM-chain plans with the same per-core blocking at 8192 and 16384 rows).

## Tracked replay over the 37 sibling buckets in one process (`replay_trace_check_sibling.json`)

`TT_METAL_TRACE_ALLOC_TRACKING=1 tests/replay_trace_check.py --buckets sibling --rounds 3 --repeats 5 --eager-repeats 2`
(sibling weights, load 2.0 to 2.2): the 37 buckets built, run eagerly and captured in one process, three rounds in forward
and reverse order with two inputs per bucket, 10 replays per bucket, 370 replays. `pass` true, tracker error none, traced
equals eager bit for bit at every bucket (logits and CLS), repeated replays identical, a changed input moves the logits by
1.48 to 5.71, no NaN. Trace bytes 176.8 MiB (185,401,344 bytes) of the 512 MiB region (34.5 percent; the English 30 buckets
141.5 MiB plus 35.3 MiB for the seven 1024 buckets: 4.62, 4.81, 5.12, 5.25, 5.12, 5.25, 5.12 MiB); the region's allocated
bytes after the captures equal the sum. Warmup 3.69 s (37 eager passes) plus 0.75 s (37 captures). The timings in that file
are not usable (the tracker runs `gc.collect()` before every replay). Nothing had to be dropped: the trace region, DRAM
(weights plus 436 MiB of static mask tensors over the 37 buckets plus at most 64 MiB of per-call masks inside a trace) and L1
all hold the set.

## Engine chunking above the largest 1024 row bucket (`engine_chunking_check_td.json`)

`tests/check_engine_chunking.py`: 40 gate rows in one `LayaEngine.forward_detailed` call with only the 1024 buckets captured
ran as 16x1024, 16x1024 and 8x1024 (`buckets` in the result; `last_buckets` on the engine) and are bit-identical, logits and
act logits, to the same rows sent as three explicit calls; the device time of the chunked call (755.1 ms) equals the sum of
the three (755.0 ms). This path is the safety net for `/v1/forward` callers; with `LAYA_MAX_BATCH_TOKENS=16384` the server's
own plan never needs it.

## Server changes the orchestrator must apply (this track does not edit `server/` or the manifests)

1. Sanity reference per checkpoint. `server/app.py` compares the STATE_EN / Q_CHOICE answer with the fixed file
   `server/sanity_reference.json`, recorded from the English checkpoint (billing 0.8899, technical 0.0417, sales 0.0685).
   The sibling answers billing 0.6115, technical 0.1441, sales 0.2444 (`sanity_reference_typed_decisions.json` in this
   directory, produced with the server's own CPU backend, usage identical: 194 input tokens). The argmax agrees, so the host
   server logged `sanity check ok: STATE_EN routing -> billing (max |dp| 0.2877 vs stored CPU value)` and started; the check
   is weak for the sibling because it compares against the wrong checkpoint. Needed change: make the file selectable, for
   example `SANITY_FILE = os.environ.get("LAYA_SANITY_REFERENCE") or <the current default>`, copy
   `sanity_reference_typed_decisions.json` to `server/sanity_reference_typed_decisions.json`, and set
   `LAYA_SANITY_REFERENCE: /opt/tt-metal/models/autoports/convaiinnovations_laya/server/sanity_reference_typed_decisions.json`
   in the sibling manifest's `serve.env`. Alternative without a code change: `LAYA_SANITY_CHECK: "0"` in the sibling manifest
   (no startup check at all), which is worse.
2. Nothing else in `server/` needs to change: the per-seq row list and the oversize-call chunking live in `tt/engine.py`,
   the server's `Buckets` sees `row_buckets` 1 to 64 and `seq_buckets` 128 to 1024 as before, and `LAYA_MAX_BATCH_TOKENS` is
   an existing knob.

## What proved wrong or incomplete in the plan

- Appendix A.5 and the sibling manifest draft give row buckets 1, 2, 4, 8, 16 at 1024 and a separate `LAYA_ROW_BUCKETS_1024`
  list; the list is 1, 2, 4, 5, 8, 10, 16 (the five-question protocol) and the engine now reads `LAYA_ROW_BUCKETS_<seq>`
  for any seq bucket. `doc/context_contract.json` still lists the sibling's seq buckets as [512, 1024] and the 1024 rows as
  [1, 2, 4, 8, 16]; the orchestrator should update it to [128, 256, 512, 1024] and [1, 2, 4, 5, 8, 10, 16].
- The plan expected the 1024 context to be exercised by the gate corpus. It is not: no gate or parity_fast sequence exceeds
  512 tokens at the 1024 / 256 budget, and only 7 of the 400 E2 cases do. The 1024 plans are therefore gated by forcing the
  corpus into the 1024 bucket, and the 1024 speed cells come from a long-state run, not from the model card's STATE_EN rows.
- The sibling's device deltas are three times smaller than the English checkpoint's; the plan's gate margins were sized on
  the English numbers.
- The server's single flat row list and plan-derived `X-Laya-Batch` header were not anticipated for a per-seq row set; the
  resolution is the engine-side per-seq list plus `LAYA_MAX_BATCH_TOKENS=16384` in the manifest.
- The row-keyed plan tables of stages 3 and 7 are right at 1024 except the SDPA grid at 5 and 10 rows (8.6 and 11.2 percent).
- The sibling checkpoint is float16 for all 206 tensors; the English checkpoint keeps its temperature buffer in float32.

## How to run

```
source /home/hous/dev/laya/bin/ttenv.sh; cd $TT_METAL_HOME; A=models/autoports/convaiinnovations_laya; DL=/home/hous/dev/laya/bin/devlock
MD=/home/hous/dev/laya/state/laya_models/laya-typed-decisions; TD=/home/hous/dev/laya/reference/parity_corpus_td.npz; OUT=$A/doc/release_typed_decisions
LAYA_CPU_THREADS=6 python -m models.autoports.convaiinnovations_laya.reference.corpus --model-dir $MD --max-len 1024 --head-max-len 256 --tag td --out-dir /home/hous/dev/laya/reference
export LAYA_MODEL_DIR=$MD TT_METAL_VISIBLE_DEVICES=0
$DL python $A/tests/bench_buckets.py --buckets sibling1024 --variants '{"sdpa_full_5_10": {"sdpa_full_grid_buckets": [[5, 1024], [10, 1024]]}}' --out $OUT/ab/bench_sdpa_full_5_10.json
$DL python $A/tests/bench_buckets.py --buckets sibling --variants '{"default": {}}' --out $OUT/bench_sibling_final.json
TT_METAL_TRACE_ALLOC_TRACKING=1 $DL python $A/tests/replay_trace_check.py --buckets sibling --rounds 3 --repeats 5 --eager-repeats 2 --out $OUT/replay_trace_check_sibling.json
$DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --items all --hidden-cases 40 --corpus $TD --model-dir $MD --seq-buckets 128,256,512,1024 --row-buckets 1,2,4,5,8,10,16 --hidden-cache /home/hous/dev/laya/state/tt_cache/fidelity_hidden_ref_gate40_td.pt --out $OUT/fidelity_bf8w_hifi3_erf_td_natural.json
$DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --items all --hidden-cases 40 --corpus $TD --model-dir $MD --seq-buckets 1024 --row-buckets 1,2,4,5,8,10,16 --hidden-cache /home/hous/dev/laya/state/tt_cache/fidelity_hidden_ref_gate40_td.pt --out $OUT/fidelity_bf8w_hifi3_erf_td_seq1024.json
$DL python $A/tests/decision_agreement.py --policy bf8w_hifi3_erf --corpus $TD --model-dir $MD --seq-buckets 128,256,512,1024 --row-buckets 1,2,4,5,8,10,16,32,50,64 --row-buckets-by-seq '{"1024": "1,2,4,5,8,10,16"}' --out $OUT/decision_agreement_bf8w_hifi3_erf_td_natural.json
$DL python $A/tests/decision_agreement.py --policy bf8w_hifi3_erf --corpus $TD --model-dir $MD --seq-buckets 1024 --row-buckets 1,2,4,5,8,10,16 --largest-rows 16 --out $OUT/decision_agreement_bf8w_hifi3_erf_td_seq1024.json
$DL python $A/tests/check_engine_chunking.py --out $OUT/engine_chunking_check_td.json
LAYA_MODEL_DIR=$MD LAYA_REVISION=e929ae5cf69bc34259cd2f95c9e91145b818b1f0 LAYA_SEQ_BUCKETS=128,256,512,1024 LAYA_ROW_BUCKETS=1,2,4,5,8,10,16,32,50,64 LAYA_ROW_BUCKETS_1024=1,2,4,5,8,10,16 LAYA_MAX_BATCH_TOKENS=16384 LAYA_PRECISION=bf8w_hifi3_erf LAYA_RAW_FORWARD=1 $DL bash /home/hous/dev/laya/bin/serve-tt.sh
TARGET=host_tt PROFILE=td-p150 BUILD=1 BASE_URL=http://127.0.0.1:8710 CHECKPOINT=laya-typed-decisions PARITY_CORPUS=$TD RAW_FORWARD=1 SKIP_E3=1 bash /home/hous/dev/laya/bin/run-evals.sh
E=/home/hous/dev/laya/evals; RES=<the results dir>; $E/.venv/bin/python $E/apps/run.py --base-url http://127.0.0.1:8710 --out $RES/apps --max-len 1024 --head-max-len 256; $E/.venv/bin/python $E/apps/score.py --results $RES/apps --name "laya-typed-decisions host_tt td-p150 b1"
$E/.venv/bin/python $E/typed_decisions/score.py --results $RES/typed_decisions --checkpoint laya-typed-decisions --name "laya-typed-decisions host_tt td-p150 b1" --reference $E/results/cpu_reference_td-cpu_b0_20261005T222846Z/typed_decisions/decisions.jsonl
$E/.venv/bin/python $E/summarize.py --results $RES --target host_tt --profile td-p150 --build 1
$E/.venv/bin/python $A/tests/bench_served_long.py --base-url http://127.0.0.1:8710 --out $OUT/served_long_state_td.json
python -m pytest $A/tests/test_model_config.py $A/tests/test_host_engine.py $A/tests/test_sibling_config.py $A/tests/test_sibling_release.py -q -p no:cacheprovider -o addopts=""
```

## Final configuration: all 37 sibling buckets (`bench_sibling_final.json`, sibling weights, load 1.3 to 1.65)

`tests/bench_buckets.py --buckets sibling` on the adopted port after the SDPA decision (eager p50 of 5, one trace per bucket,
p50 of 20 replays after 3 warm, bit identity of traced against eager per cell). Warmup of the 37 buckets in this process:
3.67 s eager plus 0.67 s capture; trace bytes 177.3 MiB. The 30 English buckets reproduce the stage 7 table
within scatter (the same code paths; the weights change nothing in time), so the table lists the seven 1024 buckets and the
512 buckets of the same row counts for comparison; every one of the 37 cells is bit-identical to eager.

| rows x seq | traced p50 ms | min | p95 | eager p50 | rows per s | padded tokens per s | trace MiB | traced == eager | load | first default run (before the SDPA decision) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1x1024 | 22.00 | 21.91 | 22.11 | 22.25 | 45.5 | 46.5 k | 4.6 | yes | 1.35 | 22.00 |
| 2x1024 | 39.68 | 39.57 | 39.88 | 39.87 | 50.4 | 51.6 k | 4.8 | yes | 1.35 | 39.79 |
| 4x1024 | 74.59 | 74.39 | 74.75 | 74.64 | 53.6 | 54.9 k | 5.1 | yes | 1.35 | 74.62 |
| 5x1024 | 95.66 | 95.28 | 95.97 | 95.51 | 52.3 | 53.5 k | 5.5 | yes | 1.32 | 104.97 |
| 8x1024 | 153.02 | 152.64 | 153.40 | 153.58 | 52.3 | 53.5 k | 5.1 | yes | 1.32 | 153.59 |
| 10x1024 | 179.44 | 179.07 | 179.63 | 179.25 | 55.7 | 57.1 k | 5.5 | yes | 1.3 | 202.68 |
| 16x1024 | 301.24 | 300.94 | 302.17 | 301.07 | 53.1 | 54.4 k | 5.1 | yes | 1.65 | 301.80 |
| 1x512 | 12.52 | 12.50 | 12.55 | 12.84 | 79.8 | 40.9 k | 4.1 | yes | 1.46 |  |
| 2x512 | 20.37 | 20.26 | 20.54 | 20.65 | 98.2 | 50.3 k | 4.6 | yes | 1.46 |  |
| 4x512 | 36.08 | 35.96 | 36.16 | 36.19 | 110.9 | 56.8 k | 4.8 | yes | 1.46 |  |
| 5x512 | 45.05 | 44.98 | 45.37 | 45.21 | 111.0 | 56.8 k | 4.6 | yes | 1.42 |  |
| 8x512 | 65.06 | 64.88 | 65.44 | 65.28 | 123.0 | 63.0 k | 5.1 | yes | 1.42 |  |
| 10x512 | 88.52 | 88.40 | 88.79 | 88.69 | 113.0 | 57.8 k | 5.2 | yes | 1.42 |  |
| 16x512 | 136.75 | 136.47 | 137.04 | 137.23 | 117.0 | 59.9 k | 5.1 | yes | 1.39 |  |

A 1024-token row costs about twice a 512-token row at the same row count at 1 to 4 rows (22.0 against 12.5 ms at one row,
74.6 against 36.1 at four) and 2.1 to 2.3 times from 5 rows (95.7 against 45.1, 301.2 against 136.7): the padded token rate
falls from 57 to 63 k per second at 512 to 54 to 57 k at 1024 because SDPA grows with the square of the sequence. The 37
traces take 177.3 MiB (185,925,632 bytes) of the 512 MiB region in the tracked replay and 177.3 MiB in the server (37 captures,
warmup 4.65 s plus 0.67 s, engine load 8.19 s, "model loaded in 10.3 s"); `LAYA_TRACE_REGION_SIZE` stays at 536870912.

The tracked replay (`replay_trace_check_sibling.json`) and the forced-1024 fidelity and invariance files were re-run on the
adopted port: pass, 37 buckets bit-identical to eager over 370 replays; fidelity forced 1024 confident 152 of 152, median
0.00313, scorer PCC 0.99985, encoder hidden PCC 0.99850, calls at {'4x1024': 12, '5x1024': 88}; invariance forced 1024 16 of 16, max
abs delta p 0.0067 (the same values as before the SDPA grid change, as the identical marker logits predicted).


## Served check: the sibling on the host server (`/home/hous/dev/laya/evals/results/host_tt_td-p150_b1_20261006T012706Z/SUMMARY.md`)

`/home/hous/dev/laya/bin/serve-tt.sh` under one devlock hold with `LAYA_MODEL_DIR=<sibling>`, `LAYA_REVISION=e929ae5c...`,
`LAYA_SEQ_BUCKETS=128,256,512,1024`, `LAYA_ROW_BUCKETS=1,2,4,5,8,10,16,32,50,64`, `LAYA_ROW_BUCKETS_1024=1,2,4,5,8,10,16`,
`LAYA_MAX_BATCH_TOKENS=16384`, `LAYA_PRECISION=bf8w_hifi3_erf`, `LAYA_RAW_FORWARD=1` (job `served_td`, server log
`/home/hous/dev/laya/logs/p6_serve_td_20261006T012706Z.log`). Healthy 20 s after start: "laya model loaded in 10.3 s",
"sanity check ok: STATE_EN routing -> billing (max |dp| 0.2877 vs stored CPU value)" (the English reference file; against the
sibling's own CPU answer the served probabilities billing 0.6022, technical 0.1473, sales 0.2504 are within 0.0093), ready with
`warm=37`. `/v1/health` reports `max_len` 1024, `head_max_len` 256, `row_buckets_by_seq` {"1024": [1, 2, 4, 5, 8, 10, 16]},
`max_rows_by_seq` {"1024": 16}, `limits.max_batch_tokens` 16384; its `model` field reads `convaiinnovations/laya` because
`bin/ttenv.sh` exports `HF_MODEL` for the English checkpoint (the package launcher sets it from the manifest). Then
`TARGET=host_tt PROFILE=td-p150 BUILD=1 CHECKPOINT=laya-typed-decisions PARITY_CORPUS=parity_corpus_td.npz RAW_FORWARD=1
SKIP_E3=1 bash /home/hous/dev/laya/bin/run-evals.sh`, E3 by hand at 1024 / 256 (`run-evals.sh` passes no budget to
`apps/run.py`), E2 re-scored with the sibling CPU reference (`run-evals.sh` attaches a reference only for `CHECKPOINT=laya`),
`summarize.py`, the long-state run, server stopped by PID with a clean shutdown. Load 1.0 to 1.9 throughout; the server's
counters at the end: 1 requests, 1 rows, histogram {'1x256': 1}.

### E1 parity against the sibling CPU fp32 corpus (488 decisions, 100 calls, max_len 1024 / head_max_len 256)

| path, type | n | max abs dp | mean abs dp | argmax agree | agree, margin >= 0.10 | PCC scorer logits | PCC act logits |
|---|---|---|---|---|---|---|---|
| wire, choice | 108 | 0.0318 | 0.0070 | 106 of 108 (0.9815) | 72 of 72 | 0.999607 (probs) | n/a |
| wire, noul | 240 | 0.0402 | 0.0051 | 240 of 240 (1.0000) | 213 of 213 | 0.999645 (probs) | n/a |
| wire, score | 140 | 0.0247 | 0.0052 | 139 of 140 (0.9929) | 98 of 98 | 0.999739 (probs) | n/a |
| wire, overall | 488 | 0.0402 | 0.0056 | 485 of 488 (0.9939) | 383 of 383 | 0.999744 (probs) | n/a |
| tensor, choice | 108 | 0.0319 | 0.0070 | 106 of 108 (0.9815) | 72 of 72 | 0.999692 | 0.999995 |
| tensor, noul | 240 | 0.0402 | 0.0051 | 240 of 240 (1.0000) | 213 of 213 | 0.997774 | 0.999985 |
| tensor, score | 140 | 0.0246 | 0.0052 | 139 of 140 (0.9929) | 98 of 98 | 0.999701 | 0.999987 |
| tensor, overall | 488 | 0.0402 | 0.0056 | 485 of 488 (0.9939) | 383 of 383 | 0.998914 | 0.999988 |

Wire path: median of the per-decision max abs dp 0.00386, p95 0.01533, NaN rows 0; tensor path: median 0.00388, p95 0.01534,
max abs logit delta 2.286 (a saturated noul marker, as in stage 6). These are the in-process numbers of
`fidelity_bf8w_hifi3_erf_td_natural.json` (485 of 488, 383 of 383, median 0.0039, scorer PCC 0.9989): the server's padding
path changes nothing. The three flips have reference margins under 0.01. E1 calls ran at 5x128 (24), 5x256 (26), 5x512 (38),
4x256 (10) and 4x512 (2) on both paths.

### E2 typed-decisions (400 cases, 2,000 decisions, one call per case at 1024 / 256)

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya-typed-decisions` (published, authors' run) | 0.766 | 0.471 | 0.062 | 0.213 | 0.242 |
| `laya-typed-decisions` CPU fp32 reference, this host | 0.766 | 0.471 | 0.061 | 0.213 | 0.242 |
| `laya-typed-decisions` on Blackhole p150, host server, `bf8w_hifi3_erf` | 0.764 | 0.469 | 0.062 | 0.214 | 0.244 |
| `laya` English checkpoint on the same host server (stage 6, for scale) | 0.359 | 0.331 | 0.311 | 0.171 | 0.689 |

Served: 400 of 400 cases, 2000 decisions, 0 NaN, 0 failed requests. Agreement with the sibling CPU reference decisions: argmax
0.9935 of 2,000 (1987 agree, 13 flips, all with a reference margin under 0.10), confident (margin >= 0.10, n 1492) 1.0, max abs dp
0.03062, mean 0.00431, median 0.00356. Per 5-question case client p50 48.858 ms (p95 49.467), server 47.14 ms, device
45.04 ms; `X-Laya-Batch` {'5x1024': 7, '5x256': 110, '5x512': 283} (the seven long cases at 5x1024); input tokens per case p50 1519, max
2831; truncated cases 0 (seven at the English budget). By workflow: agent_trace_observability 0.730, customer_service 0.762, invoice_processing 0.798, security_incidents 0.768; by type: choice 0.727, noul 0.860, score 0.721.
Deltas of the served row against the sibling CPU reference: -0.002 accuracy, -0.002 soft accuracy, +0.001 Brier, +0.001 ECE, +0.002 score MAE
(the amendment A11 confirmation bounds are 0.010 and 0.015).

### E3 application suites at 1024 / 256 (N 400 each, seed 13; no agreement column: the only CPU reference row is the English checkpoint's)

| task | n | Jev (published) | laya (published, authors' CPU run) | laya CPU fp32, this host | laya-typed-decisions on p150, host server | ECE p150 | macro F1 | ms per case (client, server, device) |
|---|---|---|---|---|---|---|---|---|
| AG News (4 labels) | 400 | 0.910 | 0.950 | 0.950 (ECE 0.032) | 0.953 | 0.156 | 0.947 | 9.314, 8.57, 7.89 |
| DAIR Emotion (6 labels) | 400 | 0.480 | 0.595 | 0.595 (ECE 0.306) | 0.595 | 0.201 | 0.470 | 9.265, 8.52, 7.89 |

The sibling matches the English checkpoint's served accuracy on both suites (0.953 against 0.953 and 0.595 against 0.593)
with a different calibration: ECE 0.156 and 0.201 against the English 0.037 and 0.312, the fine-tuned per-type temperatures
(about 1.0) applied to out-of-domain questions. Every AG News call ran at 1x128 (365) or 1x256 (35), every Emotion call at
1x128: the 1024 budget changes nothing for these rows.

### E5 speed (model card protocol, STATE_EN with qs(n), `bench_latency.timed(warmup=3, reps=15)`)

| questions per call | `laya` (Tesla T4, published) | sibling client p50 | server | device forward | bucket | English host server, build 1 (client) |
|---|---|---|---|---|---|---|
| 1 | 39.5 ms | 10.7 ms (10.7 ms per question) | 10.0 ms | 9.2 ms | 1x256 | 11.0 ms |
| 5 | 84.5 ms | 24.5 ms (4.9 ms per question) | 23.6 ms | 22.7 ms | 5x256 | 26.5 ms |
| 10 | 158.6 ms (15.9 ms per question) | 42.5 ms (4.3 ms per question) | 41.6 ms | 40.3 ms | 10x256 | 46.0 ms |
| 50 | 771 ms | 197.5 ms (4.0 ms per question) | 195.6 ms | 191.8 ms | 50x256 | 212.9 ms |

Batched throughput (`/v1/systemone/batch`, warm 3, reps 10): 202 to 249 questions per second on one p150 (English build 1 host
server: 187 to 231; T4 published 103 to 332): 1x5 202.9 (5x256 x1), 1x10 234.5 (10x256 x1), 8x5 202.2 (50x256 x1), 8x10 247.9 (16x256 x1 plus 64x256 x1), 32x5 246.7 (32x256 x1 plus 64x256 x2), 32x10 248.4 (64x256 x5), 64x5 248.4 (64x256 x5), 64x10 248.8 (64x256 x10).
`X-Laya-Batch` histogram over the whole E5 run: {'10x256': 25, '16x256': 10, '1x256': 15, '32x256': 10, '50x256': 25, '5x256': 25, '64x256': 230}. The speed-table rows are 194 to 205 tokens and
land in the 256 buckets, so the sibling's cells equal the English ones within scatter; with `LAYA_MAX_BATCH_TOKENS=16384` the
8 x 5 cell still runs as one 50x256 call (12,800 padded tokens) and the 32 x 5 cell as 64x256 plus 64x256 plus 32x256, the
English plan. Load 1.6 to 1.9 (`timing.contended` false).

### Long-state cells at 1024 tokens (`served_long_state_td.json`, `tests/bench_served_long.py`)

The STATE_EN ticket with its message repeated 30 times (rows of 746 to 752 tokens, `truncated` false), Q_NOUL and Q_CHOICE
alternating, warm 3, p50 of 15 (batch: p50 of 10), load 1.66 to 2.25:

| cell | rows | client p50 | server | device | `X-Laya-Batch` | input tokens | questions per second |
|---|---|---|---|---|---|---|---|
| 1 question | 1 | 23.9 ms | 23.2 ms | 21.9 ms | 1x1024 x1 | 746 | 41.8 |
| 5 questions | 5 | 99.0 ms | 97.6 ms | 95.3 ms | 5x1024 x1 | 3752 | 50.5 |
| 10 questions | 10 | 183.2 ms | 181.6 ms | 178.8 ms | 10x1024 x1 | 7515 | 54.6 |
| 16 questions | 16 | 305.6 ms | 303.9 ms | 300.1 ms | 16x1024 x1 | 12024 | 52.4 |
| 50 questions | 50 | 951.2 ms | 949.3 ms | 939.6 ms | 16x1024 x3, 2x1024 x1 | 37575 | 52.6 |
| batch 8 states x 5 questions | 40 | 763.4 ms | 761.3 ms | 752.9 ms | 16x1024 x2, 8x1024 x1 | | 52.4 |
| batch 16 states x 5 questions | 80 | 1519.3 ms | 1517.1 ms | 1500.7 ms | 16x1024 x5 | | 52.7 |

The served device time equals the in-process traced p50 of the same bucket within 0.6 ms (1x1024 21.9 against 22.00 ms, 5x1024
95.3 against 95.66, 10x1024 178.8 against 179.44, 16x1024 300.1 against 301.24); the 5 and 10 question cells carry the adopted
11x10 SDPA grid. 50 questions at 1024 tokens run as three 16x1024 calls plus one 2x1024 call under the 16384 cap (951 ms,
52.6 questions per second); the batch cells reach 52.4 to 52.7 questions per second at about 750 real tokens per row (about 39 k real
tokens per second), the bandwidth-bound regime of the 1024 bucket.

### Demo feed (30 s at 4 cases per second, concurrency 1)

121 cases, 605 decisions, 20.148 decisions per second, client p50 48.489 ms (p95 49.5), server 46.88 ms, device
44.95 ms, agreement with gold 0.7421 over 605 decisions (English checkpoint: 0.378 to 0.380), 0 errors, batch shapes {'5x256': 31, '5x512': 90}.

## Values for `tt-model-typed-decisions.yaml` (for the orchestrator; this track does not edit the manifest)

`serve.env` (the other lines stay as drafted):

| key | value | source |
|---|---|---|
| `LAYA_PRECISION` | `bf8w_hifi3_erf` | every stage 6 gate passes on the sibling in both protocols; `bf16_hifi4` not needed |
| `LAYA_SEQ_BUCKETS` | `128,256,512,1024` | this README, bucket set |
| `LAYA_ROW_BUCKETS` | `1,2,4,5,8,10,16,32,50,64` | unchanged from the English bundle |
| `LAYA_ROW_BUCKETS_1024` | `1,2,4,5,8,10,16` | read by `tt/engine.py` (`LAYA_ROW_BUCKETS_<seq>`) |
| `LAYA_MAX_BATCH_TOKENS` | `16384` | new line; keeps the server's plan inside the captured 1024 buckets and the `X-Laya-Batch` header exact |
| `LAYA_TRACE_REGION_SIZE` | `536870912` | 37 traces take 176.8 to 177.3 MiB |
| `LAYA_SANITY_REFERENCE` | `/opt/tt-metal/models/autoports/convaiinnovations_laya/server/sanity_reference_typed_decisions.json` | after the server change above; otherwise the check compares against the English answer |
| `LAYA_MAX_ROWS`, `LAYA_MAX_BATCH_STATES`, `LAYA_TRACE`, `LAYA_L1_SMALL_SIZE`, `LAYA_RAW_FORWARD`, `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES` | `64`, `64`, `1`, `79104`, `1`, `0` | unchanged |

`serve_profiles[p150].description`: "One Blackhole p150 chip; bf8w_hifi3_erf; 37 traces captured before the server reports
ready: 1, 2, 4, 5, 8, 10, 16, 32, 50 and 64 rows at 128, 256 and 512 tokens and 1, 2, 4, 5, 8, 10 and 16 rows at 1024 tokens;
at most 16384 padded tokens per forward (16 rows at 1024 tokens, 32 at 512, 64 at 256 and 128)."

`serve_profiles[p150x4].description`: "1x4 mesh, data parallel over rows, no fabric; declared, not validated with this
checkpoint (the English bundle's mesh was bit-identical to one chip at equal per-chip bucket and 3.74x in throughput; not run
with laya-typed-decisions in this track, which used chip 0 only)."

`card.performance` (host-served build 1 numbers; the plan quotes the packaged run's `SUMMARY.md`, so replace these with the
`package_td-p150_b<N>` values once the sibling container has run the same evaluations): "typed-decisions (400 cases, 2,000
decisions, max_len 1024 / head_max_len 256): accuracy 0.764, soft accuracy 0.469, Brier 0.062, ECE 0.214, score MAE 0.244 on p150
(published 0.766 / 0.471 / 0.062 / 0.213 / 0.242; CPU fp32 on this host 0.766 / 0.471 / 0.061 / 0.213 / 0.242); argmax agreement
with the CPU fp32 decisions 99.35 percent, 100 percent on confident decisions (margin >= 0.10). Speed (model card protocol, rows of
194 to 205 tokens): 10.7 / 24.5 / 42.5 / 197.5 ms client p50 for 1 / 5 / 10 / 50 questions (T4 39.5 / 84.5 / 158.6 / 771), 202 to 249 questions per second
batched (T4 103 to 332); long states of about 750 tokens per row: 23.9 / 99.0 / 183.2 / 305.6 ms for 1 / 5 / 10 / 16 questions and
52 questions per second batched. AG News 0.953 and DAIR Emotion 0.595 (published laya 0.950 / 0.595). Parity against CPU fp32 on 488 decisions:
485 agree, 383 of 383 confident, median max |dp| 0.0039, scorer-logit PCC 0.9989."

`card.limitations`: "Fine-tuned for the four typed-decisions workflows; outside them use the base checkpoint. max_len 1024,
head_max_len 256; requests are padded to 128, 256, 512 or 1024 tokens with 1 to 64 rows at 128 to 512 tokens and 1 to 16 rows
at 1024 tokens (a longer 1024-token batch is split into chunks of 16 rows; at most 16384 padded tokens per forward).
Temperatures are clamped to [0.5, 5.0] as pip laya 0.3.27 does. action.act_probability carries no usable signal (authors'
issue #185). 13 of 2,000 reference decisions flip on the device, all within margin 0.10 (max |dp| 0.03062); 3 of 488 parity
decisions flip, all within margin 0.01. One device process serializes requests. p150x4: declared, not validated with this
checkpoint. Not reproduced: MASSIVE, XNLI, 45 of 51 languages, post-temperature ECE."

`card.risks`: "bf8w_hifi3_erf numerics move probabilities by up to 0.040 (parity corpus) and 0.031 (typed-decisions) against
the fp32 reference; gate on confidence, not act_probability. Long option texts are truncated to 48 tokens silently."
