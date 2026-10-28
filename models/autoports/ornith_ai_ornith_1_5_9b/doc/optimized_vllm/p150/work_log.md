# Stage 10 (optimized vLLM) on the p150 profile, 2026 Oct 27 to 28

Times are Eastern Time. This run starts from the stage 9 p150 result (commit
`b71e53cf26`, evidence under `../../vllm_integration/p150/` and
`../../../readiness_vllm/`) and the selected datatype-sweep policy
`../../datatype_sweep/p150_head8_hifi2_kv8_policy.json`. Serving on chip 2,
generator-level probes on chip 3. Checkout `/home/hous/dev/tt-metal-wt-ornith`,
branch `hous/ornith-prefill-features`.

## Starting state (20:30, Oct 27)

Stage 9 serving evidence meets every stage 10 requirement except batched
prefill: `readiness_vllm/vllm_prefill_burst_benchmark.json` records the 32 x
128 burst at TTFT p50 3758.7 ms against a serial reference of the same path
(ratios 0.998), because `OrnithModel._prefill_validated` ran one row per pass.
The stage 10 gate requires p99 ratio <= 0.6 and mean ratio <= 0.5 with the
primary single-user TTFT/TPOT within 5 percent. Prefix caching stays excluded
(`recurrent_state`, serving contract row).

Serial per-row cost on the B32 pool (3758 ms / 32 = 117 ms per 128-token row,
against 83 ms single-user TTFT): every row ran the 32-layer dense work on one
128-token chunk, and `_transfer_slot` wrote each fresh row into the batched
recurrent state of 24 Gated DeltaNet (GDN) layers with a `ttnn.where`
broadcast over the whole `[32, 32, 128, 128]` FP32 buffer (256 MiB read and
written per layer per row) because the `slice_write` staging path was keyed to
the TP4 `(1, 8, 128, 128)` shape only.

## Design (20:50)

Equal-padded-length batching (design 1 of `prefill-features.md`) restricted to
fresh rows (`start_pos == 0`, logical length >= 3) whose 128-token bucket is at
or below `ORNITH_BATCHED_PREFILL_MAX_LEN`, grouped by bucket, padded with dummy
rows to a warmed row bucket (`ORNITH_BATCHED_PREFILL_ROWS`, default
`2,4,8,16,32`), capped by `ORNITH_BATCHED_PREFILL_MAX_TOKENS` (4096). Rows
outside these rules keep the single-row path inside the same call; output row
order is preserved. `ORNITH_BATCHED_PREFILL=0` is the A/B override.

Per group: ids `[B, S]`, one embedding, every dense op (norms, packed
projections, MLP, residual adds) on `[B, S, H]`; attention layers fill the
vLLM pages with the batched `paged_fill_cache` (`batch_idx_tensor` of B
entries, a `[B, blocks]` fill table) and run chunked SDPA over the group's
page-table rows; GDN layers run the KDA causal convolution per row from a
shared zero history, gather per-row inputs with runtime `ttnn.embedding`
indices from the `[B * S, width]` row-major stream, run
`chunk_gated_delta_rule` on fixed sub-batches of three rows from a shared zero
initial state, write each final state row straight into the decode slot with
the sharded `slice_write` (`scatter_recurrent_row`) and each conv tail with a
masked `where` (`scatter_conv_tail`), and concatenate the head-major cores for
one gated norm and one output projection. Per-row gate masks are runtime
tensors; padded residual rows are zeroed between layers as on the single-row
path. The last hidden row of each real row is a runtime gather; dummy rows
have no slot and are never written back. Program keys therefore depend only on
`(rows, seq)`; the adapter warms every `(rows, seq)` bucket plus the per-slot
`slice_write` programs before the first trace capture.

`scatter_recurrent_row` also replaces the TP1 `where` broadcast on the
single-row path, so the serial reference measured in this stage already
includes that fix.

## Generator-level development on chip 3 (21:15 to 21:50)

`tests/p150_batched_prefill_probe.py` builds the serving adapter at B32 (pool
of 4097 blocks, every warm bucket), runs each group once per row (`batched=False`)
and once as a group (`batched=True`) on a cleared cache, and compares per row:
terminal logits (top-1, PCC), the decode layer's recurrent and conv state rows
(exact), the attention layer's K/V pages at the real positions (exact), and a
32-token greedy continuation through the traced decode (same rows decoding
together in both arms, host argmax over returned logits). It also times
`(rows, seq)` buckets with batching off and on.

- `batched_probe_reduced_v{1,2,3}.json` (layers 0 and 3): first two attempts
  fixed a `ttnn.Shape` slicing error in `scatter_recurrent_row` and a chat
  prompt tokenization bug in the probe; v2/v3 pass every comparison exactly.
- `batched_calibration_full_v1.json` (all 32 layers, `ORNITH_BATCHED_PREFILL_MAX_LEN=1024`,
  first implementation): batched / serial time, two repeats each, minimum:
  32x128 0.405, 16x128 0.418, 8x128 0.445, 4x128 0.510, 2x128 0.669,
  16x256 0.584, 8x256 0.601, 4x256 0.644, 2x256 0.716, 8x512 0.824,
  4x512 0.845, 2x512 0.877, 4x1024 0.946, 2x1024 0.936. Every bucket wins;
  the gain falls with sequence length because the group pays the same dense
  cost as one 4096-token chunk while the per-row GDN work stays per row.
  Warmup of 23 batched buckets: 57 s, 2147 programs after warmup (stage 9:
  1122).
- The 32x128 group dispatched about 17 000 small ops (per-row gate gathers,
  nine ops per conv tail row). Second implementation: gates once per group
  from the dense `a`/`b` fields with a `[B, S, 1]` FP32 keep mask and two
  slices per three-row sub-batch (offsets fixed per row bucket), and conv
  tails scattered with one `[1, batch]` gather per tap plus one masked `where`.
  Reduced-model 32x128 went from 60 ms to 48 ms (`batched_probe_reduced_v3.json`).
- `batched_probe_full_v1.json` (all 32 layers, max length 512, second
  implementation): every group (2, 4, 8, 32 rows, two orders, random tokens,
  book text and three chat prompts, lengths 5, 31, 32, 33, 127, 128, 129,
  257, 640, 1000, 1024) has top-1 equal on every row, last-token logits PCC
  >= 0.99999, recurrent state, conv state and K/V pages bit-identical, and the
  33 greedy tokens identical; zero programs and zero captures inside every
  prefill call. Calibration after the op reduction: 32x128 0.295 (0.775 s
  against 2.625 s), 16x256 0.493, 8x512 0.758. Warmup with 15 batched buckets:
  14.7 s, 1824 programs. The eight `program growth ... continuation/decode_forward`
  events in that log are the probe's own per-slot and per-block read slices
  compiled before its decode replays; v2 of the probe reads whole tensors and
  reruns the same cases.

Selected policy: `ORNITH_BATCHED_PREFILL=1`, `ORNITH_BATCHED_PREFILL_MAX_LEN=512`
(buckets 128, 256, 384, 512), row buckets 2, 4, 8, 16, 32, token cap 4096.
Rows above 512 tokens batch at 0.82 to 0.95 of serial and were left on the
single-row path to bound warmup; this is recorded as an open item.

## Tracked B32 server on chip 2 (21:25 to 22:10)

`serve.sh tracked_b32 32 2` with `TRACKED=1` (`TT_METAL_TRACE_ALLOC_TRACKING=1`,
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`), `HOST_SAMPLING=1`, `BATCHED=1`,
`BATCHED_MAX_LEN=512`, sources of commit `c4e2ee8efd` before formatting
(`tracked_b32/launch_env.txt` records the `sha256` of `tt/*.py`; black only
rewrapped lines). Startup: bucket warmup lists the fifteen batched buckets,
one trace capture at 1824 programs, no second capture and no program growth
after `Application startup complete` (the stage 9 startup recapture is gone).

`mixed_shape_check.py` (stage 10 version, the stage 9 workload plus phase 7:
an eight-row burst of mixed buckets, three rows admitted while a request
decodes, a 32-row burst of 100 to 128-token prompts, repeats afterwards).
Attempt 1 sent the wrong model name (every request 404, kept as
`mixed_shape_check_attempt1_wrong_model_name.*`); attempt 2 completed 88/88
with zero captures and releases, but the client keyed its greedy-repeat check
on `(length, label prefix)` and compared different prompts with equal lengths
(`decoding131:b`, `burst32` rows 29 to 31), kept as
`mixed_shape_check_attempt2_client_key_bug.*`; attempt 3 with unique labels:
`mixed_shape_check.json` status pass, 88/88 requests complete, zero captures
and zero releases during the workload, no greedy repeat mismatch, no
unsafe-allocation report from the tracker. The server log shows the
multi-row eager prefill calls the batched path served: 31, 7, 6, 5, 3, 3, 2,
2 rows and the single rows, all without program growth. TTFT under the
allocation tracker (p50 0.747 s, p99 1.92 s) includes the tracker's
per-replay validation and is not a performance number.

## Serial reference server on chip 2 (22:10 to 22:25)

`serve.sh serial_b32 32 2` with `BATCHED=0` (`ORNITH_BATCHED_PREFILL=0`,
every row on the single-row path, which now uses `scatter_recurrent_row`),
`HOST_SAMPLING=1`, otherwise the final B32 configuration. Attached
`benchmark,prefill_burst`:

- prefill burst 32 x 128 / 32 output, concurrency 32: 32/32, TTFT p50/p99/mean
  3423.7 / 3425.0 / 3390.8 ms (stage 9 serial path: 3758.7 / 3760.0 / 3726.0);
  the slot-transfer fix alone removed about 335 ms from the 32-row step.
  Copied to `serial_b32/vllm_prefill_burst_serial_benchmark.json`, the
  `--prefill-burst-serial-result` of the batched run.
- CI serving burst 100/100/32 (batching off): 32/32, TTFT p50/p99 3468.5 /
  3469.6 ms, TPOT mean/p99 171.3 / 178.3 ms, 156.9 tok/s.
- 128/128/1 diagnostic on the B32 server: TTFT 97.5 ms, TPOT 168.3 ms (the
  32-row decode graph; the headline comes from the B1 server).

## Clean batched-equals-solo evidence (22:30)

`batched_probe_full_v2.json` (chip 3, all 32 layers, commit `c4e2ee8efd`,
probe reads whole state tensors to host instead of per-slot device slices):
every group of 2, 4, 8 and 32 rows in both orders has top-1 equal on every
row, last-token logits PCC >= 0.99999 (minimum 0.9999999999999), recurrent
state, conv state and K/V pages bit-identical at the real positions, and the
first token plus 32 greedy continuation tokens identical; the program cache
stays at 1824 entries from warmup to the end of the run and the capture count
stays at 1, so the batched path, the single-row path and the continuation
decode compile nothing after warmup. Calibration in the same process:
32x128 0.294 (0.775 s against 2.639 s), 16x256 0.490, 8x512 0.723.

## Primary single-user server on chip 3 (22:40 to 22:55)

`serve_b1_hand.sh`: the runner's exact server command with `--max_num_seqs 1`
on port 8001 (recorded in `b1_primary/launch_env.txt`), launched by hand so
it could run beside the B32 server on chip 2; `BATCHED=1` (the B1 pool never
forms a group: batching needs two rows), host sampling off, async scheduling
on. The attached runner `benchmark --no-benchmark-ci-serving --server-url
http://localhost:8001 --server-log b1_primary/server.log`:

- first request after readiness (`b1_primary/first/vllm_benchmark.json`):
  TTFT 82.6 ms, TPOT mean 33.9 ms, ITL p50/p99 33.9 / 34.7 ms, 29.2 tok/s,
  29.5 t/s/u; repeat (`b1_primary/repeat/`): TTFT 76.7 ms, TPOT 33.9 ms.
  Stage 9 before: TTFT 83.3 ms, TPOT 34.09 ms (repeat 77.4 / 34.10). Both
  within 1 percent.
- `b1_primary/nonaligned_requests.json`: passed.

## First batched burst and the slot-remap cost (22:30 to 23:20)

`b32_final` attempt 1 (batched on, commit `c4e2ee8efd`): prefill burst 32 x
128 TTFT p50/p99 1928.8 / 1930.4 ms against the serial reference 3423.7 ms:
p99 ratio 0.564 (gate 0.6) but mean ratio 0.559 (gate 0.5). Kept as
`b32_final/vllm_prefill_burst_benchmark_attempt1.json` and
`runner_prefill_burst_attempt1_relative_path.log` (a first attach passed the
serial path relative to the checkout root; the stage rejects a missing file).

`burst_timeline.py` (32 streamed 128-token book prompts, wall-clock send and
first-token times) against the same server (`b32_final/burst_timeline_v1.json`)
and the server log: the scheduler admits the first request alone (one-row
prefill, 46 ms host) and the other 31 in the next step; the 31-row batched
prefill is enqueued in 600 ms host time; request 0's token arrives when that
call returns; the 31 first tokens arrive 1.2 s later. The adapter timing logs
added for the `b32_diag` server (`b32_diag/server.log`,
`burst_timeline_v1.json`) place the gap: the 31-row prefill call takes
0.748 s end to end and the first decode step after the admission takes
1.047 s inside `decode_forward` with `slot_remap` set. The plugin remaps
slots at that boundary and `OrnithGenerator._remap_serving_rows` moved every
row of the 24 `[32, 32, 128, 128]` FP32 recurrent states with a `where`
broadcast over the whole buffer (about 512 MiB of DRAM traffic per row per
layer). Fix (commit `5469aae4b3`): `OptimizedDecoder.remap_recurrent_rows`
snapshots the moved rows and writes them back with the sharded `slice_write`
(`scatter_recurrent_row`); conv rows and sampler histories keep the masked
path. Serial reference remeasured on this code (`serial_b32_v2/`, hand-launched
on chip 3 port 8001 with `serve_hand.sh`, batching off): burst TTFT p50/p99
2855.4 / 2856.8 ms (from 3423.7), CI burst 100/100/32 TTFT p50 2866.3 ms,
TPOT mean 170.0 ms, 162.6 tok/s. The client timeline on that server gives
TTFT median 2.839 s.

## Batched burst after the remap fix, and the B1 server on the same code (23:20 to 23:45)

`b32_final` attempt 2 (commit `5469aae4b3`, runner-launched on chip 2):
prefill burst 32 x 128 TTFT p50/p99 1352.3 / 1353.6 ms against the
`serial_b32_v2` reference 2855.4 ms: p99 ratio 0.474, mean ratio 0.471
(`b32_final/vllm_prefill_burst_benchmark_attempt2_remap_v1.json`). The gate
passes, but the boundary decode after the 31-row admission still costs
0.48 s (`b32_final/server_attempt2_remap_v1.log`): `remap_recurrent_rows`
issues three ops per moved row per layer and `_remap_serving_rows` on the
three conv buffers issues four ops per moved row per buffer, about 11 000
host-dispatched ops. Third version (uncommitted at this point): recurrent
rows move cycle by cycle (one DRAM snapshot per cycle, then a sharded slice
straight into the staging layout plus one `slice_write` per move), conv
buffers are permuted with one `ttnn.embedding` row gather per buffer from a
`[1, batch]` index.

`b1_final/` (chip 3 port 8001, `serve_hand.sh`, commit `5469aae4b3`, batching
on, host sampling off): first request TTFT 83.3 ms, TPOT 33.9 ms; repeat
76.9 ms, 33.9 ms (`b1_final/first/` and `repeat/vllm_benchmark.json`);
non-aligned requests pass.

## Final servers on commit `e70875637c` (23:45 to 00:40, Oct 28)

Sources: `launch_env.txt` of `serial_b32_v3`, `b32_final` and `b1_final_v2`
record the same `sha256` of `tt/*.py` as HEAD `e70875637c`.

- `serial_b32_v3/` (chip 3 port 8001, `serve_hand.sh`, `BATCHED=0`, host
  sampling compatibility on): prefill burst 32 x 128 TTFT p50/p99/mean
  2429.0 / 2430.4 / 2427.1 ms (`vllm_prefill_burst_serial_benchmark.json`);
  CI burst 100/100/32 TTFT p50/p99 2446.4 / 2447.6 ms, TPOT mean/p99 169.5 /
  169.9 ms, ITL p50/p99 167.7 / 225.5 ms, 166.4 tok/s; 128/128/1 diagnostic
  94.9 ms / 167.4 ms; client timeline TTFT 2.36 to 2.42 s.
- `b32_final/` (chip 2, runner-launched `serve.sh b32_final 32 2`, `BATCHED=1`,
  `BATCHED_MAX_LEN=512`, host sampling compatibility on): prefill burst
  32 x 128 TTFT p50/p99 927.8 / 929.0 ms, serial reference ratios p99 0.382,
  mean 0.381 (`readiness_vllm/vllm_prefill_burst_benchmark.json`, gate
  p99 <= 0.6, mean <= 0.5). The adapter log shows the 31-row batched prefill
  call at 0.748 s and the admission boundary decode at 59 to 63 ms (from
  1.047 s before the remap fixes); the client timeline gives TTFT 0.85 to
  0.92 s for the 31 rows and 0.85 s for the first request.
- `b1_final_v2/` (chip 3 port 8001 after the serial server stopped,
  `serve_hand.sh`, `BATCHED=1`, host sampling off): 128/128/1 first request
  TTFT 83.9 ms, TPOT mean 33.9 ms, ITL p50/p99 33.8 / 36.6 ms, 29.17 tok/s,
  29.51 t/s/u; repeat 77.1 ms / 33.9 ms; non-aligned requests pass. Stage 9
  before: 83.3 ms / 34.09 ms (repeat 77.4 / 34.10): TTFT +0.7 percent, TPOT
  -0.6 percent.

## Batched-equals-solo on the final commit (00:45)

`batched_probe_full_v3_final.json` (chip 3, all 32 layers, commit
`e70875637c`, the cycle-wise remap in the warm rotations): every group of 2,
4, 8 and 32 rows in both orders has top-1 equal, last-token logits PCC
>= 0.9999999999996, exact K/V pages and recurrent/conv state, and 33
identical greedy tokens; the program cache stays at 1793 entries from warmup
through the end, one capture. Calibration: 32x128 0.291 (0.774 s against
2.663 s), 16x256 0.486, 8x512 0.748, 4x1024 0.967.

## Final B32 server attempt 3: recaptures found in the server log (01:00, Oct 28)

`b32_final` attempt 3 (commit `e70875637c`) served every stage: prefill burst
927.8 ms (ratios 0.382 / 0.381), sampling 73 passed / 1 controlled failure /
4 skipped, qualitative greedy outputs byte-identical to stage 9, CI burst
TTFT p50 911.3 ms, mixed-shape 88/88 with zero captures in its window,
non-aligned pass. The trace-reuse summary of the whole server life, however,
shows 19 captures after startup: each follows an `Ornith program growth
N -> N+1 before replay` inside `decode_forward` after a slot remap.
`remap_recurrent_rows` took its one DRAM snapshot per permutation cycle at the
cycle's first row, and the warm rotations only start cycles at row 0, so a
cycle starting at another row compiled a new `ttnn.slice` program (keyed by
its start row) and `_ensure_replay_safe` recaptured the decode traces. The
evidence of that server is archived under `b32_final_attempt3_recaptures/`
and does not count. Fix: the snapshot now uses the sharded staging slice,
which the rotations compile for every source row, and is written back with
`slice_write`; no DRAM snapshot program exists. Serial reference, final B32
and B1 servers are rerun on the fixed commit.

## Hardware note: a host test sweep opened all four chips (22:45 to 22:50 ET, Oct 27; 02:45 to 02:50 UTC)

A device-free host test sweep was assembled from an explicit file list that
wrongly included `tests/test_contract_extensions.py`; its `mesh_device`
fixture is requested through a parametrized `device_params` fixture, so the
`mesh_device` text filter used to select host-only files did not catch it. The
sweep ran without `TT_VISIBLE_DEVICES`: at 02:45:44 UTC UMD opened local chip
ids {0, 1, 2, 3}, waited until 02:49:13 for the `CHIP_IN_USE_3` lock held by
the reduced remap probe on chip 3, then created a one-device mesh and ran
`test_trace_changed_page_table` and `test_trace_repeated_identical_state`
until it closed the devices at 02:50:38 (`/tmp/ornith_host_sweep.log`). The
UMD cluster open covered chips 0 and 1, which belong to the other job; no
lock wait on those chips is logged, so no process of that job held them at
that moment, and no reset was needed. The serial v4 server on chip 3 opened
its device after 02:50:38 (model load at 02:51 to 02:53) and the chip 2 server
after that; neither saw an error. This was a violation of the device rules
for this run and is recorded as such; later host-test runs use the files
that the stage 9 sweep used and set `TT_VISIBLE_DEVICES` explicitly.
Host sweep result: 123 passed, 13 failed: the five `test_active8_serving.py`
failures that predate stage 10 (identical set at `b71e53cf26`) and eight
`test_prefix_cache_host.py` cases whose fixtures build `OrnithModel` and its
cache without `__init__` and hit the new batched-prefill policy lookup; the
model now reads the policy and the cache constants with `getattr` guards and
those eight pass (commit `9d4de01a6d`, two guard expressions; the final
servers were launched from `5cff1036cc`, which differs only by those two
lookups). The two `test_contract_extensions.py` device tests passed on the
device they should never have opened.

## Final servers on commits `5cff1036cc` (serial v4, B32) and `9d4de01a6d` (B1 v3) (01:15 to 02:05, Oct 28)

- `serial_b32_v4/` (chip 3 port 8001, batching off, host sampling
  compatibility on): prefill burst TTFT p50/p99/mean 2422.2 / 2423.5 /
  2420.4 ms; CI burst 100/100/32 TTFT p50/p99 2447.1 / 2448.2 ms, TPOT
  mean/p99 169.5 / 169.9 ms, ITL p50/p99 167.7 / 224.6 ms, 166.4 tok/s;
  client timeline TTFT median 2.415 s; one trace event (startup capture).
- `b32_final/` attempt 4 (chip 2, runner-launched, batching on): prefill burst
  TTFT p50/p99/mean 931.5 / 932.9 / 929.7 ms, ratios p99 0.385, mean 0.384;
  adapter log: 31-row batched prefill 0.749 s, admission boundary decode
  58 to 60 ms; client timeline 0.84 to 0.91 s. The remaining stages of this
  server are recorded below.
- `b1_final_v3/` (chip 3 port 8001 after the serial server, batching on, host
  sampling off): first request TTFT 85.7 ms, TPOT mean 33.92 ms, ITL p50/p99
  33.8 / 37.4 ms, 29.13 tok/s, 29.48 t/s/u; repeat 76.4 ms, 33.89 ms,
  29.51 t/s/u; non-aligned requests pass; one trace event (startup capture).
  Against stage 9 (83.3 / 34.09, repeat 77.4 / 34.10): TTFT +2.9 percent
  (repeat -1.3 percent), TPOT -0.5 percent.

## Slot-remap probe (01:20 and 02:10)

`tests/p150_remap_probe.py` builds the adapter at B32, warms it, fills every
GDN layer's recurrent and conv state with random values, and applies six
permutations through `remap_serving_slots`: rotate left, swap rows 3 and
17, reverse, three disjoint cycles starting at rows 5, 20 and 30, a random
permutation, and the identity. Reduced model (`remap_probe_reduced_v1.json`,
layers 0 and 3) and the full 32-layer model (`remap_probe_full_v1.json`,
commit `9d4de01a6d` sources): every case moves every row exactly
(`max_abs_diff 0.0`) with zero program growth and zero captures; the full
32-row rotation takes 0.23 s over 24 layers, the two-row swap 34 ms.

## Final B32 server attempt 4, remaining stages (02:05 to 02:50)

`b32_final/` (commit `5cff1036cc`, chip 2): sampling full profile 73 passed,
4 skipped, 1 failed (`test_frequency_penalty_changes_output`, the controlled
failure of `../../vllm_integration/plugin_changes.md` entry 2); qualitative:
all six greedy completions byte-identical to stage 9, degenerate-output
check passes; CI burst 100/100/32: 32/32, TTFT p50/p99 903.8 / 905.0 ms,
TPOT mean/p99 170.4 / 170.8 ms, ITL p50/p99 168.5 / 226.5 ms, 180.0 tok/s;
128/128/1 diagnostic on the B32 server 88.5 ms / 168.3 ms; mixed-shape
88/88, zero captures, zero releases, no mismatch, TTFT p50/p95/p99 0.750 /
0.899 / 0.901 s; non-aligned pass. `trace_reuse_summary.json`: one startup
capture at 1793 programs, zero captures and zero program growth afterwards
over 302 eager prefill calls. The server was stopped through the runner;
`pgrep -af 'openai.api_server|EngineCore|run_vllm_server'` returned nothing
afterwards. The teardown counters are not logged (the runner's SIGTERM
force-kills `EngineCore`, as in stage 9).

`readiness_vllm/` now holds that server's `server.log`, `vllm_run_summary.json`,
`sampling_tests.log`, `vllm_qualitative_outputs.json`, burst and CI files,
plus the primary single-user `vllm_benchmark.json` / `vllm_result.json` /
`vllm_benchmark.log` copied from `b1_final_v3/first/`.

Stage gate `10-optimized-vllm.check.sh` (`stage_check.log`): exit 0:
degeneracy pass, context contract pass with the stage 9 advisory line about
`doc/vllm_integration/runner_compatibility_report.md` (a 2048-token CPU
registry probe of the p150x4 stage), prefill features pass.

## Independent review 1 and follow-up (23:05 to 23:40 ET, Oct 27)

The first stage review (`stage_review_1.md`) returned more-work-needed with
three P2 items and several concerns:

1. The allocation-tracker evidence (`tracked_b32/`) predates the final remap
   code. Action: a tracked B32 server from the final tree runs the stage 10
   mixed-shape workload again (`tracked_b32_final/`, below).
2. `readiness_vllm/vllm_prefill_burst_serial_benchmark.json` was still the
   stage 9 copy (3765 ms) while the burst file's `serial_reference` named
   `serial_b32_v4` (2422 ms). Action: replaced by the `serial_b32_v4` copy and
   listed in the README provenance.
3. Watcher was not run over the new op configurations. Action: `TT_METAL_WATCHER=10`
   runs of the batched probe (groups 2, 3, 5, 8, 31, 32, including the
   dummy-row buckets) and of the remap probe on the reduced two-layer model
   (`watcher/`), with the watcher logs kept beside them.
4. Dummy rows wrote their padded tokens into block 0 (vLLM's null block) and
   would collide with slot 0 under the generator-owned page table. Action:
   a dummy row now repeats the last real row of the group (same tokens, same
   page-table row): it writes the same bytes into the same pages and has no
   slot. `batched_probe_full_v4_dummy_rows.json` runs groups of 3, 5 and 31
   rows (buckets 4, 8 and 32) on the full model for the equality checks.
5. `ORNITH_BATCHED_PREFILL_MAX_LEN` defaulted to 128 in code while every
   server used 512. Action: the code default is 512.
6. The `b1_final_v3` server ran commit `9d4de01a6d` (two `getattr` guards
   beyond `5cff1036cc`); the work-log heading now says so. The host sweep log
   is kept as `host_sweep_opened_devices.log`.

## Tracked B32 server on the final code (23:40 to 00:05 ET)

`serve.sh tracked_b32_final 32 2` with `TRACKED=1`, `HOST_SAMPLING=1`,
`BATCHED=1`, `BATCHED_MAX_LEN=512`; `launch_env.txt` records `git_head`
`66bd9e836b` with the `sha256` of `tt/model.py` and `tt/optimized_decoder.py`
equal to the committed `b6b1fdb29a` files (the launch preceded that commit
by minutes and the tree was identical). Startup: warm 45 -> 1652 programs,
one capture at 1793. The stage 10 mixed-shape workload: 88/88 complete,
zero captures and zero releases, no greedy repeat mismatch, TTFT p50/p95/p99
0.680 / 0.926 / 0.928 s under the tracker; non-aligned requests pass. Over
the server life (48 eager prefill calls including the 31-, 7-, 6-, 5-, 3-
and 2-row groups and their slot remaps): one capture, zero program growth,
no unsafe-allocation report, no tracker `RuntimeError`
(`tracked_b32_final/server.log`, `trace_reuse_summary.json`). The server was
stopped through the runner; no vLLM process remained.

## Watcher runs and dummy-row equality on the final code (23:35 to 00:05 ET)

`watcher/` (chip 3, `TT_METAL_WATCHER=10`, no profiler, reduced two-layer
model 0 and 3 with the same op configurations as the full stack):
`batched_probe_reduced_watcher.json` runs groups of 2, 3, 5, 8, 31 and 32
rows in both orders (3, 5 and 31 pad to the 4-, 8- and 32-row buckets with
dummy rows) plus the 32x128 calibration; every row top-1 equal, PCC
>= 0.99999, state and K/V pages exact, greedy tokens identical, zero
program growth; `watcher_batched_probe_reduced.log` holds 85 watcher dumps
over 843 s with no detected fault (no NoC, circular-buffer, L1, assert or
stack signature; the only per-core notes are the usual highest-stack-usage
lines). `remap_probe_reduced_watcher.json`: six permutations exact with zero
program growth; `watcher_remap_probe_reduced.log` has 3 dumps over 22 s, no
fault. The reduced model runs the identical kernels and shapes (two layers
instead of 32), so the watcher covered `chunk_gated_delta_rule` on one- to
three-row sub-batches, the sharded `slice_write` state scatter, the batched
`paged_fill_cache`, the row gathers and the cycle-wise remap.

`batched_probe_full_v4_dummy_rows.json` (chip 3, all 32 layers, commit
`b6b1fdb29a`, groups of 3, 5 and 31 rows in both orders, 32 greedy steps):
every comparison exact, program cache constant at 1793, one capture; dummy
rows repeat the last real row of the group.
