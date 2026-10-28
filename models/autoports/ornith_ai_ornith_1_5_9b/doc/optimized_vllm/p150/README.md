# Ornith-1.5-9B optimized vLLM serving on p150 (one Blackhole chip)

Primary single-user **128 input / 128 output / 1 request / concurrency 1 /
max-num-seqs 1**, greedy temperature 0, ignore EOS, native 262144 context,
first request after readiness, batched prefill on: **TTFT 85.7 ms; decode
29.48 tokens/s/user** from mean TPOT 33.92 ms; ITL median 33.8 ms, P99
37.4 ms; aggregate output 29.13 tokens/s (`b1_final_v3/first/vllm_benchmark.json`;
same-workload repeat 76.4 ms / 33.89 ms, 29.51 t/s/u, in `b1_final_v3/repeat/`). Before
this stage (stage 9, same runner, workload, mesh, sampling mode and TT config):
TTFT 83.3 ms, TPOT 34.09 ms, 29.34 t/s/u (`../../vllm_integration/p150/b1_primary/first/`).
Single-user TTFT changed +2.9 percent (repeat -1.3 percent against the stage 9
repeat) and TPOT -0.5 percent (gate: within 5 percent).

Batched prefill (the stage 10 requirement): the **32 x 128-token burst at
concurrency 32** serves with TTFT p50/p99 **931.5 / 932.9 ms** against the
serial reference 2422.2 / 2423.5 ms measured on the same code with
`ORNITH_BATCHED_PREFILL=0`: TTFT ratios **p99 0.385, mean 0.384** (gate p99
<= 0.6, mean <= 0.5; `../../../readiness_vllm/vllm_prefill_burst_benchmark.json`,
serial reference `serial_b32_v4/vllm_prefill_burst_serial_benchmark.json`,
`config_matches: true`). Stage 9 measured the same burst at 3758.7 ms on the
per-row path. Caveat: the serial reference ran on chip 3 and the batched
server on chip 2 (one server per chip; the same-code one-row 128/128/1
diagnostics on the two B32 servers differ by about 10 ms, 98.0 against
88.5 ms TTFT), a variance the ratio gate margin absorbs.

Sampling: 73 passed, 1 controlled failure, 4 skips (full profile, B32,
unchanged from stage 9). Qualitative: the six greedy completions are
byte-identical to the stage 9 outputs. Mixed-shape trace reuse with multi-row
bursts: zero captures on the tracked and the final server, one capture over
the final server's whole life. Stage gate `10-optimized-vllm.check.sh`: exit 0
(`stage_check.log`; degeneracy, context contract and prefill features pass;
the context script prints the stage 9 advisory line about a 2048-token
mention in the p150x4 `runner_compatibility_report.md`). Independent review: see the work log.

## Serving configuration

All 32 layers on one Blackhole chip (readiness mesh label `N150`, model card
profile `p150`), precision policy `../../datatype_sweep/p150_head8_hifi2_kv8_policy.json`
(the stage 9 selection, logged as `Serving resolved precision` at startup),
served `max_model_len=262144` (`../../context_contract.json`), vLLM-owned pool
of 4129 blocks of 64 tokens for the B32 servers, `sample_on_device_mode=all`,
trace region 100 MB, L1 small 32 KiB, `--async-scheduling`, prefix caching
off (`recurrent_state` exclusion, unchanged). Stack: upstream vLLM 0.26.0 plus
`tenstorrent/vllm-tt-plugin` at local HEAD `9bab7d527d4a` (unchanged in this
stage; the one local plugin commit and the proposed plugin test change are in
`../../vllm_integration/plugin_changes.md`). Registration through the
`EXTRA_MODELS_DIR` bundle, as in stage 9.

Launch wrappers: `serve.sh <label> <max_num_seqs> <chip>` (the stage 9 runner
launch with `BATCHED`, `BATCHED_MAX_LEN`, `HOST_SAMPLING` and `TRACKED`
switches) and `serve_hand.sh <label> <max_num_seqs> <chip> <port>` (the
runner's exact server command by hand, used for the second server on chip 3
while the runner-launched server occupied chip 2; every launch records
`launch_env.txt` with the environment, git head and `sha256` of `tt/*.py`).
`attach.sh` attaches the runner stages and the stage clients to a live server.

Environment of the batched path (model side, read at model construction):

| Variable | Default | Meaning |
| --- | --- | --- |
| `ORNITH_BATCHED_PREFILL` | `1` | `0` keeps every row on the single-row path (A/B override) |
| `ORNITH_BATCHED_PREFILL_MAX_LEN` | `512` | largest padded 128-token bucket that batches |
| `ORNITH_BATCHED_PREFILL_ROWS` | `2,4,8,16,32` | warmed row buckets; a group is padded with dummy rows to the next bucket |
| `ORNITH_BATCHED_PREFILL_MAX_TOKENS` | `4096` | cap on `rows x seq` per group |

## What changed (commits `c4e2ee8efd`, `5469aae4b3`, `e70875637c`, `5cff1036cc`, `9d4de01a6d`)

- `tt/model.py`: `_prefill_validated` groups fresh rows (`start_pos == 0`,
  logical length >= 3) by their padded 128-token bucket up to the configured
  maximum, pads each group to a warmed row bucket and runs
  `_prefill_batched_group`: one embedding, every dense op on `[B, S, H]`,
  per-row residual keep masks, the last hidden row of each real row gathered
  with a runtime index. Rows outside the rules keep the single-row path in
  the same call; output order is preserved. A dummy row repeats the last real
  row of the group (same tokens, same page-table row) so it writes the same
  bytes into the same pages and never depends on an unowned block; it has no
  slot and receives no state. `allocate_cache` allocates the
  batched-prefill constants (zero conv history, zero initial states for
  sub-batches of 1 to 3 rows, per-slot masks) before any capture.
- `tt/optimized_decoder.py`: `_gdn_prefill_batched` runs the KDA causal
  convolution per row from the shared zero history (the op accepts one row
  per launch), gathers per-row inputs with `ttnn.embedding` indices from the
  `[B * S, width]` row-major stream, computes the gates once per group with a
  `[B, S, 1]` keep mask, runs `chunk_gated_delta_rule` on fixed three-row
  sub-batches from the shared zero state (the op serves `cores / 32` rows per
  launch), writes every final state row into its decode slot with the
  sharded `slice_write` (`scatter_recurrent_row`) and the conv tails with one
  gather per tap (`scatter_conv_tails`), and concatenates the head-major
  cores for one gated norm and one output projection. Attention layers use
  the batched `paged_fill_cache` (`batch_idx_tensor` of B entries) and
  chunked SDPA over the group's page-table rows; this path already existed
  for `[B, S]` inputs.
- Slot transfer and remap: `scatter_recurrent_row` replaces the TP1
  `where` broadcast over the whole `[32, 32, 128, 128]` FP32 state in
  `_transfer_slot` (single-row path) and `remap_recurrent_rows` plus
  `remap_conv_rows` replace `_remap_serving_rows` for the decode layers'
  recurrent and conv state at every plugin slot remap (cycle-wise moves, two
  ops per moved row, one gather per conv buffer; the cycle snapshot goes
  through the same warmed sharded slice, so no remap compiles a program
  after warmup). The 31-row admission boundary decode went from 1.047 s to
  58 to 60 ms. `tests/p150_remap_probe.py` checks six permutations (rotation,
  swap, reverse, three cycles, random, identity) for exact row movement with
  zero program growth (`remap_probe_reduced_v1.json`, `remap_probe_full_v1.json`).
- `tt/generator_vllm.py`: `_warm_prefill_buckets` warms every `(rows, seq)`
  bucket after the single-row buckets and slots, before the first capture, and
  logs them (`Ornith prefill bucket warmup: batched_buckets ...`); the adapter
  logs the host time of every prefill call and every boundary decode step.
- Program keys depend only on `(rows, seq)`: gather indices, keep masks and
  slot masks are runtime tensors; sub-batch offsets are fixed per row bucket;
  per-slot `slice_write` programs are warmed by the per-slot admission warm
  and the two remap rotations.

## Batched prefill evidence

Batched equals solo (`batched_probe_full_v3_final.json`, chip 3, all 32
layers, commit `e70875637c`, `tests/p150_batched_prefill_probe.py`): groups of
2, 4, 8 and 32 rows in two orders over lengths 5, 31, 32, 33, 127, 128, 129,
257, 640, 1000, 1024 (random tokens, book text and three chat prompts; rows
above 512 tokens stay on the single-row path inside the same call). For every
row the batched result equals the single-row result: top-1 equal, last-token
logits PCC >= 0.9999999999996, the decode layer's recurrent and conv state
rows and the attention layer's K/V pages at the real positions bit-identical,
and the first token plus 32 greedy continuation tokens identical. The program
cache stays at 1793 entries from warmup to the end of the run (one capture).
`batched_probe_full_v4_dummy_rows.json` (commit `b6b1fdb29a`) repeats the
checks for groups of 3, 5 and 31 rows, which pad to the 4-, 8- and 32-row
buckets with dummy rows. Watcher (`TT_METAL_WATCHER=10`, `watcher/`) ran the
same probe over groups 2, 3, 5, 8, 31 and 32 and the remap probe on the
reduced two-layer model with the identical op configurations: no detected
fault in 85 and 3 watcher dumps.

Calibration (same probe, batched / serial time, minimum of two repeats):
32x128 0.291 (0.774 s against 2.663 s), 16x256 0.486, 8x512 0.748, 4x1024
0.967; the first implementation (`batched_calibration_full_v1.json`) also
measured 2x128 0.669, 4x128 0.510, 8x128 0.445, 16x128 0.418, 2x256 0.716,
4x256 0.644, 8x256 0.601, 2x512 0.877, 4x512 0.845, 2x1024 0.936. Buckets up
to 512 tokens are served; 640 to 1024-token rows stay single-row (open item).

Warmup (`b32_final/server.log`): buckets `[2, 128] ... [8, 512]` (fifteen),
single-row lengths 1, 2, 127, ..., 2047, 2175, 32 slots, two remap rotations;
programs 45 -> 1684 before capture, 1793 at capture, no growth afterwards.
Warmup takes about 13 s on the full model.

Burst TTFT (`readiness_vllm/vllm_prefill_burst_benchmark.json`, 32 x 128
input / 32 output, concurrency 32, greedy, B32 server, commit `5cff1036cc`):
32/32, TTFT p50/p99/mean 931.5 / 932.9 / 929.7 ms; serial reference (same
code, `ORNITH_BATCHED_PREFILL=0`, `serial_b32_v4/`): 2422.2 / 2423.5 / 2420.4 ms;
ratios p99 0.385, mean 0.384.
The client timeline (`b32_final/burst_timeline.json`, `burst_timeline.py`)
shows the scheduler admitting the first request alone and the other 31 in
one 31-row group (bucket 32, one dummy row): the batched call takes 0.749 s
and the boundary decode 0.058 s. The earlier attempts are kept under
`b32_final_attempt3_recaptures/`: attempt 1 (1928.8 ms, ratio 0.559 against
the 3423.7 ms reference before the remap fixes), attempt 2 (1352.3 ms, ratio
0.471 after the first remap fix) and attempt 3 (927.8 ms, ratio 0.381, but
with 19 recaptures over the server life from the per-cycle snapshot slice;
see the work log).

## Mixed-shape trace reuse with multi-row bursts

`mixed_shape_check.py` (stage 9 workload plus phase 7: an eight-row burst of
mixed buckets, three rows admitted while a request decodes, a 32-row burst of
100 to 128-token prompts, repeats afterwards; 88 requests). Tracked servers
(`TT_METAL_TRACE_ALLOC_TRACKING=1` with program-cache allocations):
`tracked_b32_final/` on the final code (sources equal to commit
`b6b1fdb29a`): 88/88 complete, zero captures and zero releases, no greedy
repeat mismatch, no unsafe-allocation report or tracker error, one capture
and zero program growth over the server life including the 31-, 7-, 6-, 5-,
3- and 2-row groups and their slot remaps (`tracked_b32_final/trace_reuse_summary.json`);
`tracked_b32/` is the same workload on the first batched commit `c4e2ee8efd`
with the same result. Final server
(`b32_final/mixed_shape_check.json`, commit `5cff1036cc`, untracked): 88/88
complete, zero captures and zero releases, no greedy repeat mismatch, TTFT
p50/p95/p99 0.750 / 0.899 / 0.901 s, ITL max 0.398 s. Over that server's
whole life (`b32_final/trace_reuse_summary.json`: burst, sampling suite,
qualitative, CI burst, mixed-shape, non-aligned; 302 eager prefill calls)
there is exactly one capture (startup) and no program growth.

## Sampling, qualitative, non-aligned lengths

- `b32_final/sampling_tests.log` (`--sampling-profile full`, B32 compatibility
  server with host sampling allowed): 73 passed, 4 skipped (three chunked
  prefill needles, all-vocabulary logprobs), 1 failed:
  `test_frequency_penalty_changes_output`, the controlled failure analysed in
  stage 9 (`../../vllm_integration/plugin_changes.md` entry 2, `proposed`).
- `b32_final/vllm_qualitative_outputs.json`: six chat prompts, greedy and
  sampled, 256 tokens; every greedy completion is byte-identical to the stage 9
  output reviewed in `../../vllm_integration/p150/README.md`; the vLLM-scope
  degenerate-output check passes.
- `b32_final/nonaligned_requests.json` and `b1_final_v3/nonaligned_requests.json`:
  valid non-aligned lengths (131, 65, chat prompt of 25 tokens), repeats and
  concurrent requests pass with exact prompt usage.

## Secondary CI serving burst (100 / 100 / 32, no concurrency limit, max-num-seqs 32)

| Server | TTFT P50 / P99 ms | TPOT mean / P99 ms | ITL P50 / P99 ms | Output tok/s | 1000 / TPOT (secondary) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 9 before (`../../vllm_integration/p150/b32_final/`) | 3778.4 / 3779.8 | 171.26 / 178.26 | 168.59 / 245.33 | 154.57 | 5.84 |
| Serial reference, final code, batching off (`serial_b32_v4/`) | 2447.1 / 2448.2 | 169.5 / 169.9 | 167.7 / 224.6 | 166.4 | 5.90 |
| After, batching on (`b32_final/`) | 903.8 / 905.0 | 170.4 / 170.8 | 168.5 / 226.5 | 180.0 | 5.87 |

TPOT is the 32-row decode graph (about 168 ms per step on one chip) in every
row; the burst admission and the batched prefill change TTFT only. These
numbers are capacity and nightly-parity context, not headline decode.

## Full-model comparison and accounting

The one-chip traced teacher-forcing bound of stage 9
(`../../vllm_integration/p150/teacher_forcing_p150.json`, median of three
repeats) is 29.18 t/s/u and 81.2 ms TTFT; serving measures 29.48 t/s/u and
85.7 ms (repeat 29.51 t/s/u, 76.4 ms). `perf_summary.json` records the before/after pairs
with artifact hashes, the burst gate, the batched-equals-solo rows, the
device-time fields as `null` (no profiler in a serving stage) and the named
limitations: the batched 32x128 pass is host-dispatch bound (about 7 000
small ops in 0.75 s), a multi-row admission costs one 60 ms boundary step for
the plugin slot remap, the scheduler admits the first burst request alone,
and rows above 512 tokens stay single-row.

## Rejected and open items

- Exact row counts instead of row buckets: would need one program set per
  `(B, S)` for B in 2..32 (about 100 programs each); bucketing with dummy rows
  keeps warmup at fifteen passes and bounds the program cache.
- Packed rows of different lengths in one fixed bucket (design 2): the GDN
  convolution and recurrence run per row anyway, so packing would only
  change the dense shapes; equal buckets give the same dense saving with a
  simpler state scatter.
- Buckets above 512 tokens: measured wins of 3 to 25 percent; left on the
  single-row path to bound warmup time and program count.
- The plugin's one-row first admission in a burst and its slot remap at the
  boundary are plugin scheduling behaviour; the adapter makes both cheap but
  does not change them.
- Prefix caching remains excluded (`recurrent_state`); the generator-level
  resume check does not apply.

## Artifact provenance under readiness_vllm/

Commits: the served performance, sampling and qualitative evidence comes
from `5cff1036cc` (`b32_final/`, `serial_b32_v4/`) and `9d4de01a6d`
(`b1_final_v3/`); the tracked final-code server, the watcher runs and the
dummy-row probe ran the committed `b6b1fdb29a` sources, which differ from
`5cff1036cc` only by the dummy-row contents (a repeated real row instead of
zeros) and two `getattr` guards, with the same op sequence and the same
1793-program cache.

`vllm_prefill_burst_benchmark.json`, `vllm_prefill_burst_result.json`,
`sampling_tests.log`, `vllm_qualitative_outputs.json`,
`vllm_ci_serving_benchmark.json`, `vllm_ci_serving_result.json`,
`vllm_run_summary.json` and `server.log` are the final B32 batched server
(`b32_final/`, chip 2, commit `5cff1036cc`); `vllm_benchmark.json`,
`vllm_result.json` and `vllm_benchmark.log` are copies of the primary
single-user first request from `b1_final_v3/first/` (the B32 server's own
128/128/1 diagnostic, 88.5 ms / 168.3 ms, is in `b32_final/runner_benchmark.log`);
`vllm_prefill_burst_serial_benchmark.json` is the copy of
`serial_b32_v4/vllm_prefill_burst_serial_benchmark.json` that the burst file's
`serial_reference.path` names (the stage 9 copy was replaced after the review). The stage 9 copies are in
`readiness_vllm_stage9_snapshot/`.
