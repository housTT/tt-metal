# Stage 6: full model (device encoder, head and scorer plus the host tail) versus CPU fp32

Plugin stage "full-model" mapped to Laya (PLAN.md section 5, row 6, Appendix A.4 and A.7 gate). This stage adds the
backend `tt/engine.py` that the server loads, runs the complete `DecisionModel` (device part through the stage 2
trace runner, host tail in fp32) on the 200-decision gate subset against the CPU fp32 reference, checks the
alone-versus-in-batch invariance, and reports the first served numbers of the real server with the TT (Tenstorrent)
backend. PCC = Pearson correlation coefficient. CLS = the first token position, whose hidden state feeds the act head.

## Files

| file | role |
|---|---|
| `tt/engine.py` | `LayaEngine`: `from_env()`, `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits float32 [B, kmax] with -1e4 where marker_mask is False, act_logits float32 [B, 2])`, `forward_detailed`, `shapes()`, `last_device_ms`, `close()`; `host_tail` (the host part of `DecisionModel.forward` after the scorer); `build_act_head` (the `act_head` module rebuilt from the checkpoint in fp32); env parsing (`parse_mesh_shape`, `parse_warmup_shapes`) |
| `tests/run_fidelity.py` | the 488-item parity corpus (200 gate decisions plus 288 parity_fast questions) through the engine, metrics per type and temperature bucket, the stage 6 gates, hidden-state PCC against `LayaReference.forward_with_hidden`, `--chunk N` for the B 2 and B 4 rows; JSON per policy |
| `tests/decision_agreement.py` | 16 questions alone, in B 2, B 4, a mixed B 8 batch and the B 64 bucket; same argmax and max abs delta p gates; `decision_agreement.json` |
| `tests/test_full_model.py`, `tests/test_invariance.py` | pytest gates over the saved JSON (`LAYA_FIDELITY_LIVE=1` and `LAYA_INVARIANCE_LIVE=1` rerun them on the device) |
| `fidelity_bf8w_hifi3_erf_1x1.json` | the gate run (all 488 items, 40 hidden-state cases) |
| `fidelity_bf8w_hifi3_erf_1x1_b2.json`, `fidelity_bf8w_hifi3_erf_1x1_b4.json` | the 200 gate decisions re-run in calls of 2 and 4 rows (review R1 request) |
| `decision_agreement.json` | alone versus in-batch |
| `work_log.md` | timeline and decisions |

## The engine (`tt/engine.py`)

- `LayaEngine.from_env()` reads `LAYA_MODEL_DIR` (checkpoint directory; `encoder/config.json`, `rl_agent_config.json`,
  `model.safetensors`), `LAYA_MESH_SHAPE` (`1x1` default, `1x4` for the data-parallel mesh of stage 4), `LAYA_PRECISION`
  (policy name from `tt/model_config.py: POLICIES`, default `bf8w_hifi3_erf`), `LAYA_SEQ_BUCKETS` (`512`),
  `LAYA_ROW_BUCKETS` (`1,2,4,8,16,32,64`, per device), `LAYA_TRACE` (`1`), `LAYA_TRACE_REGION_SIZE` (bytes, 512 MiB),
  `LAYA_L1_SMALL_SIZE` (79104), `LAYA_WARMUP_SHAPES` (`all`, `none` or `8x512,64x512` in per-device rows x seq),
  `LAYA_DEVICE_ID` (0), `LAYA_PORT_OVERRIDES` (JSON of `PortConfig` fields, for experiments), `LAYA_CPU_THREADS`.
  `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` is set with `setdefault` at import.
- Construction: open the device (`open_device(device_id)`) or the mesh (`open_mesh`), upload the weights
  (`TtnnLayaModel`), rebuild `act_head` on the host in fp32 from the checkpoint tensors, then the stage 2 two-phase
  warmup over the warm shapes (`LayaTraceRunner.warmup`: every bucket once eagerly, then one trace per bucket). With
  `LAYA_TRACE=0` the buckets are built and run once eagerly instead. Measured on chip 0: device open plus config and
  safetensors load 2.2 s, weight upload 1.7 s, warmup 0.2 s for two buckets (4.1 s in all); the server reported "model loaded in 6.6 s"
  with all seven buckets.
- `forward`: pads the request to the bucket (`TtnnLayaModel.host_inputs`: pad id, all-zero attention row, qtype 0 on
  pad rows; the row bucket is `ceil(rows / devices)` per device), replays the trace, reads back the all-position scorer
  logits `(n, L)` fp32 and the CLS rows `(n, 1024)`, then runs `host_tail`: gather at `marker_pos`, `masked_fill(~marker_mask,
  -1e4)`, softmax, the four features `[top1, top1 - top2, normalised entropy, k / 255]`, `act_head` on `[cls, features]`.
  This is `vendor/rl_common.py: DecisionModel.forward` from the `torch.gather` on; a CPU check with random inputs
  against `reference/laya_reference.py: _tail` gave bit-identical logits and act logits. `last_device_ms` is the wall
  time of the input write, the replay and the two readbacks. A `threading.Lock` serialises device use (the server also
  holds one `asyncio.Lock`).
- `shapes()`: `backend`, `device`, `arch`, `mesh_shape`, `num_devices`, `precision` (policy name) plus the full policy
  and port description, `seq_buckets`, `row_buckets` (total rows per call, so 4 x the per-device buckets on the mesh),
  `row_buckets_per_device`, `max_rows`, `warm_shapes` (total rows x seq), `trace`, `trace_region_size`, `l1_small_size`,
  `pad_rows_keep_one_token: False` (the device pads rows with an all-zero attention row; T1's stage 1 masks are -1e30,
  not -inf, so SDPA stays finite and no token has to be kept), `max_len`, `head_max_len`, load and warmup seconds.
- `close()`: release every trace, free the weights and buckets, close the device or the mesh.

The server pads to the smallest row bucket >= N and the smallest seq bucket >= the longest row before it calls
`forward` (`server/engine.py: pad_batch`), so the engine's own padding only triggers for the direct callers (tests).

## Gate run: 200 decisions, policy `bf8w_hifi3_erf`, chip 0, traced bucket 8x512 (`fidelity_bf8w_hifi3_erf_1x1.json`)

Log `/home/hous/dev/laya/logs/p3_s6_fidelity_erf_20261005T225618Z.log`. Protocol: one engine call per case (the 5
questions of a typed-decisions case; 4 or 5 questions per parity_fast state), rows trimmed to the longest row of the
call, padded by the engine to the 8x512 bucket; probabilities with the corpus temperatures (`temperature_by_options`
bucket, clamped rule, identical to the raw rule on every corpus item); "confident" means the CPU reference top-1 minus
top-2 probability >= 0.10 (149 of the 200 gate decisions). Host 1-minute load 1.6 to 1.7 during the device pass.

| gate | measured | threshold | result |
|---|---|---|---|
| confident argmax agreement | 149 of 149 (100 percent) | >= 98 percent | pass |
| median over decisions of max_k abs delta p_k | 0.0108 | <= 0.02 | pass |
| scorer-logit PCC over the gathered markers (710 values) | 0.9962 | >= 0.99 | pass |
| hidden-state PCC, encoder output after the final norm, real positions, 40 cases pooled | 0.9955 (worst case 0.9912, worst row 0.9814) | >= 0.99 | pass |
| hidden-state PCC, head output (input of the scorer), real positions, 40 cases pooled | 0.9996 (worst case 0.9972, worst row 0.9882) | >= 0.99 | pass |
| NaN | 0 rows | 0 | pass |
| alone versus in-batch (B 2, B 4, mixed B 8, B 64; section below) | 16 of 16 same argmax, max abs delta p 0.0093 | same argmax, <= 0.01 | pass |

Reported, not gated: plain argmax agreement 194 of 200 (97.0 percent); act argmax agreement 200 of 200 (act-logit PCC
0.99993; the act head is saturated, amendment A7); p95 of max abs delta p 0.030; max 0.089; mean 0.0133; max abs logit
delta 1.52; CLS PCC worst case 0.9981; the eager device path and the traced path agree with the CPU to the same
1.52 (the trace is bit identical to eager, stage 2).

Per type and temperature bucket (gate subset):

| type (temperature bucket) | n | argmax agree | confident agree | median max abs delta p | p95 | max | scorer PCC |
|---|---|---|---|---|---|---|---|
| choice (choice:3-5) | 60 | 57 (95.0 percent) | 42 of 42 | 0.0110 | 0.0269 | 0.0776 | 0.9980 |
| score (score:3-5) | 80 | 77 (96.3 percent) | 65 of 65 | 0.0127 | 0.0302 | 0.0486 | 0.9976 |
| noul (noul:2) | 60 | 60 (100 percent) | 42 of 42 | 0.0065 | 0.0309 | 0.0889 | 0.9879 |

The six flipped decisions all have a reference margin under 0.018 (0.0042, 0.0019, 0.0073, 0.0051, 0.0055, 0.0177:
`agent_trace_observability_000044/outcome`, `customer_service_000023/churn_risk`, `customer_service_000064/action`,
`invoice_processing_000030/discrepancy_severity`, `security_incidents_000070/disposition`,
`security_incidents_000088/urgency`); the device moves their probabilities by 0.01 to 0.02, the same as any other
decision. The noul scorer PCC of 0.988 is a scale effect: noul logits of the reference reach 17 to 19 on saturated
questions, and the device places those 2 to 7 logits lower (`guard/6 jailbreak` 2.29 -> 9.28, `triage/3 refund_requested`
17.56 -> 15.51) while the probability moves by at most 0.024 (the softmax is flat there); 40 corpus items have a
reference logit above 8 and their max abs delta p is 0.017. The largest probability moves are on mid-margin decisions
(`customer_service_000080/needs_human` 0.089 at margin 0.32, `customer_service_000080/category` 0.078 at margin 0.31).

All 488 corpus items (the 288 parity_fast questions added): argmax 475 of 488 (97.3 percent), confident 403 of 403,
median max abs delta p 0.0079, p95 0.0298, max 0.111 (`parity_fast`), scorer PCC 0.9956, act argmax 488 of 488, no NaN.
parity_fast alone: 281 of 288, 254 of 254 confident, median 0.0062, PCC 0.9956; by type choice 46 of 48, score 57 of
60, noul 178 of 180. Note from review R1: the corpus is the eager fp32 model while pip laya runs SDPA; the two differ
by up to 1.3e-4 in scorer logits, which is 1000 times below the device deltas measured here.

Hidden states (40 gate cases, 200 rows, real positions): encoder output PCC per case 0.9912 to 0.9992 (median 0.9961),
head output 0.9972 to 0.9999 (median 0.9997), CLS 0.9981 or better. The encoder loses most on the long
`customer_service` cases (0.9912 for `customer_service_000064`), consistent with T1's per-row dips after layer 19; the
two head layers and the scorer's LayerNorm recover the correlation (0.9996 pooled) because the outlier channels that
dominate the encoder residual are normalised away. CPU cost of the check: 97.9 s for the 40 reference forwards at 6
threads.

Device time per 5-row call during the pass: p50 64.6 ms, max 65.1 ms (bucket 8x512, load 1.6); stage 3 measured
65.8 ms for the same bucket.

### Policy decision

The shipped policy `bf8w_hifi3_erf` passes every stage 6 gate on the first run, so the fallback `bf16_hifi4` was not
run (the plan runs it only after a failed gate). The margin on the median gate is 0.0108 against 0.02, on the
confident-agreement gate 100 percent against 98, on the scorer PCC 0.9962 against 0.99 and on the encoder hidden-state
PCC 0.9955 against 0.99 (the smallest margin; the stage 8 sweep should keep this row in view).

## B 2 and B 4 end-to-end rows (review R1 request)

Review R1 (Hard-Check Gaps) asked for end-to-end rows at the block-sharded GeGLU buckets 2x512 and 4x512, where T1's
encoder test had its lowest PCC (0.9955 after the final norm on a 2-row batch with a 512-token fill row).
`tests/run_fidelity.py --items gate --chunk 2` and `--chunk 4` re-run the 200 gate decisions in calls of 2 and 4 rows;
the fifth question of every case becomes a 1-row call at bucket 1x512 in both runs, so the table splits the rows by the
bucket they ran at. `--hidden-cases 10` compares the hidden states of the first ten calls. Files
`fidelity_bf8w_hifi3_erf_1x1_b2.json` and `fidelity_bf8w_hifi3_erf_1x1_b4.json`, log
`/home/hous/dev/laya/logs/p3_post_evals_20261005T230054Z.log` (steps `s6_fidelity_erf_b2`, `s6_fidelity_erf_b4`), load
1.7 to 1.9, device p50 per call 20.2 ms (B 2 run) and 24.2 ms (B 4 run).

| rows | bucket (stage 3 plan) | n | argmax agree | confident agree | median max abs delta p | p95 | max | scorer PCC | max abs logit delta | encoder hidden PCC at this bucket | head hidden PCC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| questions 1 to 4 of each case | 2x512 (sharded 2816, L1 chain) | 160 | 154 | 110 of 110 | 0.0099 | 0.0300 | 0.0785 | 0.9963 | 1.36 | 7 calls: min 0.9975, median 0.9992 | min 0.9997 |
| the same rows | 4x512 (sharded 2816, L1 chain) | 160 | 154 | 110 of 110 | 0.0099 | 0.0300 | 0.0785 | 0.9963 | 1.36 | 5 calls: min 0.9970, median 0.9984 | min 0.9996 |
| the same rows | 8x512 (interleaved 2816, L1 chain; the gate run above) | 160 | 155 | 110 of 110 | 0.0105 | 0.0326 | 0.0889 | 0.9957 | 1.52 | | |
| question 5 of each case | 1x512 (interleaved 2816, L1 chain) | 40 | 39 | 39 of 39 | 0.0125 | 0.0269 | 0.0340 | 0.9981 | 0.27 | 1-row calls: min 0.9949 | |
| the same rows | 8x512 (the gate run) | 40 | 39 | 39 of 39 | 0.0119 | 0.0259 | 0.0294 | 0.9980 | 0.26 | | |

Whole-run aggregates over the 200 decisions (two buckets mixed): the B 2 run and the B 4 run both give 193 of 200
argmax agreements, 149 of 149 confident, median 0.0102, scorer PCC 0.9966, encoder hidden PCC pooled 0.9986 (B 2) and
0.9982 (B 4), head 0.9998 and 0.9997; every stage 6 gate passes in both files.

Findings: the 2x512 and 4x512 buckets produce bit-identical scorer logits on all 160 rows (the same 8x8 block-sharded
plan with the same per-core blocking), and the 1-row calls are identical across the two runs. End to end the sharded
buckets are not worse than bucket 8 on the same rows (PCC 0.9963 against 0.9957, median 0.0099 against 0.0105, one flip
fewer). The encoder dip that R1 flagged does not appear on real 2-row calls (worst call 0.9975) and does not reach the
decisions. Bucket 2 and bucket 8 differ by at most 0.25 logits on the same rows, the same bucket-to-bucket placement
noise as the invariance test below measures.

## Alone versus in-batch (`decision_agreement.json`)

`tests/decision_agreement.py` (`decision_agreement.json`, log `/home/hous/dev/laya/logs/p3_post_evals_20261005T230054Z.log`
step `s6_invariance_erf5`, load 1.9): 16 gate questions (every 12.5th gate row: 5 choice, 5 score, 6 noul from 16
cases) in five placements: alone (16 calls at bucket 1x512), B 2 (8 calls of 2 rows at 2x512), B 4 (4 calls of 4 rows
at 4x512), a mixed B 8 batch (2 calls of 8 rows at 8x512, order shuffled with seed 13) and the B 64 bucket (one call
with the 16 rows plus 48 other gate rows, 64 real rows at 64x512). Probabilities with the corpus temperatures.

| placement against alone | same argmax | max abs delta p | median | max abs logit delta | PCC of the marker logits |
|---|---|---|---|---|---|
| B 2 | 16 of 16 | 0.0087 | 0.0039 | 0.098 | 0.99978 |
| B 4 | 16 of 16 | 0.0087 | 0.0039 | 0.098 | 0.99978 |
| mixed B 8 | 16 of 16 | 0.0089 | 0.0043 | 0.082 | 0.99978 |
| B 64 | 16 of 16 | 0.0093 | 0.0026 | 0.125 | 0.99965 |

Gates: the same argmax in all five placements on 16 of 16 questions; max abs delta p between alone and any in-batch
placement 0.0093 (threshold 0.01): pass, with a small margin (the 0.0093 is `customer_service_000023/churn_risk` in the
B 64 call; `security_incidents_000004/credential_compromise` reaches 0.0090 there). Not gated: mixed B 8 against B 64
0.0112; the largest difference between any two placements 0.0116. Against CPU fp32 every placement agrees on 15 of 16
(the one flip is `customer_service_000023/churn_risk`, reference margin 0.0019, in all five placements) with max abs
delta p 0.049 to 0.051. B 2 and B 4 are again bit identical to each other.

## First served numbers: the real server with the TT backend on the host

Server: `/home/hous/dev/laya/bin/serve-tt.sh` under devlock (uvicorn, `LAYA_BACKEND=tt`, `LAYA_MESH_SHAPE=1x1`,
`LAYA_TRACE=1`, buckets 1,2,4,8,16,32,64 x 512, `LAYA_RAW_FORWARD=1`), log
`/home/hous/dev/laya/logs/p3_serve_tt_20261005T225920Z.log`: model loaded in 6.6 s, sanity check ok (STATE_EN routing
-> billing, max abs delta p 0.0014 versus the stored CPU value), seven traces warm before "Application startup
complete". Smoke `/home/hous/dev/laya/evals/results/smoke_20261005T225920Z/`: STATE_EN with Q_CHOICE and Q_NOUL 200,
client 26.3 ms, server 23.7 ms, device 20.7 ms, batch 2x512; two-state batch 200, 40.2 / 38.6 / 36.0 ms, batch 4x512.
Evals: `TARGET=host_tt PROFILE=p150 BUILD=0 RAW_FORWARD=1 bash /home/hous/dev/laya/bin/run-evals.sh` ->
`/home/hous/dev/laya/evals/results/host_tt_p150_b0_20261005T225931Z/SUMMARY.md` (log
`/home/hous/dev/laya/logs/p3_served_evals_20261005T225920Z.log`).

The served numbers are the first measured for this port and are those the plan's stage 10 and 11 tables quote.

### E1 parity against CPU fp32 (488 decisions, 100 calls; tensor path `POST /v1/forward` with the exact corpus tensors)

| type | n | max abs dp | mean abs dp | argmax agree | agree, margin >= 0.10 | PCC scorer logits | PCC act logits |
|---|---|---|---|---|---|---|---|
| choice | 108 | 0.1108 | 0.0150 | 103 of 108 (0.954) | 82 of 82 | 0.99902 | 0.99997 |
| noul | 240 | 0.0889 | 0.0077 | 238 of 240 (0.992) | 217 of 217 | 0.99445 | 0.99985 |
| score | 140 | 0.0486 | 0.0135 | 134 of 140 (0.957) | 104 of 104 | 0.99839 | 0.99992 |
| overall | 488 | 0.1108 | 0.0110 | 475 of 488 (0.973) | 403 of 403 | 0.99562 | 0.99990 |

Median of the per-decision max abs dp 0.00793, p95 0.02977, no NaN, max abs logit delta 6.995 (the saturated noul
case discussed above). These are the in-process numbers of the gate run to the last digit: the server's padding path
(`pad_batch` to 8x512, pad rows with no live token) changes nothing. Wire path (`POST /v1/systemone`, probabilities
rounded to 4 decimals): overall 475 of 488, 403 of 403, max abs dp 0.1109, median 0.00791, p95 0.02979, PCC of the
probabilities 0.99920; by type choice 103 of 108, noul 238 of 240, score 134 of 140. (Review R1 note: the corpus is the
eager fp32 model; the SDPA model that pip laya runs differs from it by up to 1.3e-4 in logits, below everything here.)

### E2 typed-decisions (400 cases, 2,000 decisions, one call per case, 512 / 192)

| model | accuracy | soft acc | Brier | ECE | score MAE |
|---|---|---|---|---|---|
| `laya` (published, authors' run) | 0.362 | 0.332 | 0.316 | 0.175 | 0.694 |
| `laya` CPU fp32 reference, this host | 0.3615 | 0.3315 | 0.3155 | 0.1747 | 0.6937 |
| `laya` on Blackhole p150, host server, `bf8w_hifi3_erf` | 0.359 | 0.331 | 0.311 | 0.171 | 0.689 |

Served: 400 of 400 cases, 0 NaN, 0 failed; agreement with the CPU reference decisions: argmax 0.974 of 2,000, confident
(margin >= 0.10, n 1,496) 1.000, max abs dp 0.095, median 0.0106; per 5-question case client p50 69.8 ms (p95 71.3),
server 68.6 ms, device 64.7 ms; every call ran at 8x512; input tokens per case p50 1,519, max 2,560; load 2.1 to 2.5.

### E3 application suites (N 400 each, seed 13)

| task | n | Jev (published) | laya (published, authors' CPU run) | laya CPU fp32, this host | laya on p150, host server | ECE p150 | ms per case p150 (client, server, device) |
|---|---|---|---|---|---|---|---|
| AG News (4 labels) | 400 | 0.910 | 0.950 | 0.950 (ECE 0.032) | 0.953 | 0.037 | 14.4, 13.6, 12.5 |
| DAIR Emotion (6 labels) | 400 | 0.480 | 0.595 | 0.595 (ECE 0.306) | 0.593 | 0.312 | 14.2, 13.5, 12.5 |

Agreement with the CPU reference decisions: AG News 0.9975 (399 of 400), Emotion 0.985 (394 of 400); every call at
1x512; macro F1 0.947 and 0.471; load 2.0.

### E5 speed (model card protocol, `bench_latency.timed(warmup=3, reps=15)`, STATE_EN with `qs(n)`)

| questions per call | `laya` (Tesla T4, published) | p150 client p50 | p150 server | p150 device forward | bucket | p150x4 |
|---|---|---|---|---|---|---|
| 1 | 39.5 ms | 14.5 ms | 13.7 ms | 12.5 ms | 1x512 | not served in this run (stage 4 measured the mesh in process) |
| 5 | 84.5 ms | 68.9 ms (13.8 ms per question) | 67.6 ms | 64.7 ms | 8x512 | |
| 10 | 158.6 ms (15.9 ms per question) | 142.9 ms (14.3 ms per question) | 141.2 ms | 136.0 ms | 16x512 | |
| 50 | 771 ms | 545.1 ms (10.9 ms per question) | 543.1 ms | 522.8 ms | 64x512 | |

Batched throughput (`/v1/systemone/batch`, warm 3, reps 10): 70 to 117 questions per second on one p150 against the
published 103 to 332 on a T4: 1x5 72.6 (8x512), 1x10 69.9 (16x512), 8x5 73.7 (one 64x512 call for 40 rows), 8x10 115.3
(16x512 plus 64x512), 32x5 115.9, 32x10 116.7, 64x5 116.5, 64x10 116.6 questions per second (64x512 calls). Bucket
histogram over the whole E5 run: 64x512 255, 16x512 35, 8x512 25, 1x512 15, 32x512 10. Host load 1.6 to 2.0 during E5.
The device bound is the 64x512 trace at 522.8 ms (122 rows per second): every STATE_EN row is 194 to 205 tokens and is
padded to 512, so a 256-token bucket would halve the padded tokens of every speed-table row, and 40-row and 10-row
requests pad to 64 and 16 rows. Both are stage 7 inputs (seq bucket 256, exact row buckets 5, 10 and 50).

Demo feed (30 s at 4 cases per second, concurrency 1): 121 cases, 605 decisions, 20.1 decisions per second, client p50
71.0 ms, server 69.2 ms, device 64.8 ms, agreement with gold 0.380, 0 errors, all calls 8x512.

### Served token-length histogram (stage 7 seq-bucket input; `served_token_lengths.json`)

Row lengths of the exact served requests through the vendored `build_sequence` at 512 / 192 (the server tokenizes the
same way), plus the E1 corpus rows:

| source | rows | min | p50 | mean | p95 | max | <= 128 | <= 256 | <= 384 | <= 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| E2 typed-decisions (400 cases x 5 questions) | 2000 | 124 | 308 | 290 | 428 | 512 | 0.9 percent | 32.4 percent | 90.7 percent | 100 percent |
| E3 AG News (400 cases x 1 question) | 400 | 68 | 103 | 105 | 139 | 232 | 91.2 percent | 100 percent | 100 percent | 100 percent |
| E3 DAIR Emotion (400 cases x 1 question) | 400 | 38 | 51 | 54 | 75 | 97 | 100 percent | 100 percent | 100 percent | 100 percent |
| E5 STATE_EN x qs(1, 5, 10, 50) | 1 to 50 | 194 | 194 to 199 | 194 to 200 | 194 to 205 | 205 | 0 percent | 100 percent | 100 percent | 100 percent |
| E1 parity corpus | 488 | 62 | 173 | 213 | 407 | 510 | 35.2 percent | 61.3 percent | 91.8 percent | 100 percent |

All 2,800 served E2 and E3 rows by bin: 1 to 64: 324, 65 to 128: 458, 129 to 192: 512, 193 to 256: 153, 257 to 320:
481, 321 to 384: 686, 385 to 448: 115, 449 to 512: 71 (27.9 percent at or under 128, 51.7 percent at or under 256, 93.4
percent at or under 384). Reading for stage 7: a 256 bucket covers the whole speed table, both application suites and
half of typed-decisions; a 128 bucket covers Emotion and 91 percent of AG News; typed-decisions needs 384 for 91
percent and 512 for the rest. The row budget per request (a 5-question case at 8x512 costs 65 ms while its rows are
124 to 512 tokens) is the other half of that decision.

## How to run

```
source /home/hous/dev/laya/bin/ttenv.sh; cd $TT_METAL_HOME; A=models/autoports/convaiinnovations_laya; DL=/home/hous/dev/laya/bin/devlock
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --mesh 1x1 --items all --hidden-cases 40 --threads 6
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --items gate --chunk 2 --hidden-cases 10
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/run_fidelity.py --policy bf8w_hifi3_erf --items gate --chunk 4 --hidden-cases 10
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/decision_agreement.py --policy bf8w_hifi3_erf
python -m pytest $A/tests/test_full_model.py $A/tests/test_invariance.py -q -p no:cacheprovider -o addopts=""     # gates over the JSON, no device
$DL bash /home/hous/dev/laya/bin/serve-tt.sh                                                                     # the server, port 8710
```
