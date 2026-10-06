# Stage 7 work log (Track T3, UTC, 2026 Oct 5)

- 23:14 Track T3 started. Read PLAN.md (amendments A5, A9, A10), the stage 2, 3, 4 to 6 and serving READMEs, review R1 and
  its responses, the build 0 served summary. Deliverables and rules recorded in the task; no human present.
- 23:18 Code changes for the bucket set: `tt/model_config.py` `ROW_BUCKETS = (1, 2, 4, 5, 8, 10, 16, 32, 50, 64)`,
  `SEQ_BUCKETS = (128, 256, 512)` (the old tuples kept as `ROW_BUCKETS_POW2` and `SEQ_BUCKETS_STAGE6`),
  `select_bucket`, `deployment_buckets`, `parse_buckets`; 128 added to `SDPA_MEASURED_SEQ_LENS` (without it a 128-token
  bucket would run SDPA on ttnn's default program config). `tt/runner.py` records the trace bytes per capture through
  `ttnn.get_memory_view(device, ttnn.BufferType.TRACE)` (allocated bytes per bank times banks, before and after each
  `end_trace_capture`) and the build, eager and capture seconds per bucket. `tt/engine.py` times the host tail per call
  (`last_host_tail_ms`) and reports the trace bytes in `shapes()`. `tests/laya_inputs.build_inputs` takes `max_len` so
  rows fit short buckets (default `min(512, seq_len)`, head budget halved below 512).
- 23:20 Job `s7_smoke_all` (`/home/hous/dev/laya/logs/p3_s7_smoke_all_20261005T232046Z.log`): `bench_buckets.py --buckets all`,
  all 30 buckets in one process, eager then traced. Every bucket builds, runs and captures; traced equals eager bit for
  bit at all 30; trace bytes 4.1 to 5.1 MiB per bucket, 142.0 MiB for the set in the 512 MiB region; warmup phase 1
  2.85 s, phase 2 0.54 s. Load 0.1 to 3.7. One anomaly: 5x128 (640 rows, 20 tile rows) at 19.9 ms is slower than
  8x128 (18.9 ms) because no interleaved GeGLU program config applies when the tile rows do not divide by 8.
- 23:25 Job `s7_ab_queue` (`/home/hous/dev/laya/logs/p3_s7_ab_queue_20261005T232521Z.log`, JSON under `ab/`): 12 variants,
  one process each, 20 traced replays after 3 warm, against a `default_pairs` arm run with the same protocol in the
  same queue. Run-to-run scatter: `default_pairs` against the smoke run (10 replays) is within 0.9 percent at all 24
  shared buckets. Results in the README A/B table. Decisions: GeGLU interleaved config on an 11x10 grid wherever the
  tile rows divide by 10 (new rule `mlp_grid_y_choices = (10, 8, 5, 4, 2, 1)`), everything else unchanged.
- 23:31 Job `s7_ab_queue2` (`/home/hous/dev/laya/logs/p3_s7_ab_queue2_20261005T233140Z.log`): confirmation of the grid rule
  against the old 8-only rule in the same queue, and the interleaved plan with the new grid at the 1280-row buckets
  (5x256, 10x128) against the block-sharded plan.
- 23:33 Confirmation (`ab/bench_gridrule_*.json`): new grid rule against the 8-only rule in the same queue: 1x128 -5.2
  percent (11x4 grid replaces ttnn's automatic config), 5x128 -31.3, 10x256 -5.3, 5x512 -4.4, 50x128 -5.0, 50x256 -5.6,
  50x512 -5.1; buckets the rule does not touch move by -0.2 to +1.2 (scatter). Interleaved plan with the new grid at
  the 1280-row buckets beats the block-sharded plan: 5x256 22.86 against 23.66 ms (-3.3 percent), 10x128 21.65 against
  22.32 (-3.0); at 1024 and 2048 rows it is a dead heat (4x256 -0.3, 8x256 -1.4, 2x512 -0.3, 4x512 -1.4 percent).
- 23:34 First decision: `geglu_plan = "interleaved"` at every bucket; `s7_measure` queue launched
  (`/home/hous/dev/laya/logs/p3_s7_measure_20261005T233442Z.log`).
- 23:41 `s7_measure` results: all-bucket tracked replay passes (30 buckets, tracker on, no error, bit identical);
  fidelity on the final buckets passes (149 of 149, median 0.01104, scorer PCC 0.9965, hidden 0.9957 / 0.9996; the calls
  ran at 5x256 x 10 and 5x512 x 30); the invariance gate missed by 0.0005 (0.0105 against 0.01, alone against the B 2 and
  B 4 placements at 2x512 and 4x512, which stage 6 measured at 0.0087 with the block-sharded plan; alone against B 8 and
  B 64 0.0089 and 0.0090). The latency ladder and two Tracy profiles failed on `ModuleNotFoundError: laya`: my script
  imported the authors' `bench_latency.py`, which imports pip laya (eval venv only). Fix: the STATE_EN, Q_CHOICE, Q_NOUL
  and qs(n) definitions copied verbatim into `tests/bench_latency.py` (diff against the authors' file: only STATE_HI
  omitted).
- 23:43 Second decision on the GeGLU plan: keep the block-sharded plan where stage 6 validated it (1024 and 2048 rows,
  where the interleaved plan gains nothing) and let it yield to the interleaved 11x10 config where that config applies
  (`shard_yields_to_grid_rows = 10`: 1280 rows, 5x256 and 10x128, the measured 3 percent). `s7_measure2` queue launched
  (`/home/hous/dev/laya/logs/p3_s7_measure2_20261005T234359Z.log`): invariance on the final buckets and on the stage 6
  protocol, final all-bucket bench, tracked replay, latency ladder, throughput, Tracy at 1x256, 50x256, 5x256, fidelity
  rerun with the cached reference hidden states.
- 23:47 Reconciliation of the two profiles that did run (64x256, 64x512; their buckets are above 2048 rows, so the
  plan decision does not touch them): `perf_summary.py reconcile` splits the 456-op pass into masks (2 ops),
  embeddings (2), encoder stack (419 = 28 layers), final norm, type add (2), head layers (25), scorer (4), CLS slice.
  64x256: device kernel sum 245.4 ms against traced 253.0 (gap 3.0 percent); 64x512: 511.5 against 524.4 (2.4 percent).
  Finding: the head layers' ReLU runs as a separate `UnaryDeviceOperation` (2 x 690 us at 64x256), not fused as the
  stage 2 README states; dated note appended there; open item (0.5 percent).
- 23:50 `s7_measure2` done (`/home/hous/dev/laya/logs/p3_s7_measure2_20261005T234359Z.log`): invariance passes on the final
  buckets (16 of 16, 0.0090) and reproduces stage 6 on the 512-only protocol (0.009276, the stage 6 value to the last
  digit, so the final port equals the stage 6 numerics at the 512 buckets); all-bucket bench, tracked replay (pass, 30
  buckets, 141.5 MiB), latency deployment and fresh, throughput, Tracy at 1x256, 5x256, 50x256, fidelity rerun (pass,
  cached reference hidden states, 149 of 149, median 0.01104) all written. One defect: the 64x512 throughput cell ran
  at 64x256 because STATE_EN rows (205 tokens) select the 256 bucket; fixed (typed-decisions rows with one row filled to
  512 tokens) and the cell re-queued behind the sweep (`s7_throughput2`).
- 23:50 Stage 8 sweep launched as `s8_sweep` (`/home/hous/dev/laya/logs/p3_s8_sweep_20261005T235020Z.log`, subprocess log
  `doc/datatype_sweep/sweep_subprocess.log`).
- 23:56 Reconciliation files written for 1x256, 5x256, 50x256, 64x256, 64x512; `perf_summary.json` and
  `latency_table.json` assembled; `tests/test_performant.py` 6 passed, 1 skipped (live test).
- 23:58 Sweep defect: `bf16_hifi4` fails in warmup at the first block-sharded bucket (program 322, "statically allocated
  circular buffers clash with L1 buffers", `modernbert_mlp._sharded` down projection): bf16 weights double the circular
  buffers of the 8x8 block-sharded 2816 plan. Stage 3 measured that policy at 1x512, 8x512 and 64x512 only (no sharded
  bucket). Fix: `mlp_shard_plan` declines when `policy.linear_dtype` is bf16 (those policies run the interleaved config
  at every bucket, as stage 3 measured them); the sweep is re-run with `--skip-existing` after the current pass
  (`s8_sweep2`, chained) to fill the missing fidelity file. The policies already measured use bfp8 weights, so the rule
  does not change their plans.
- 00:00 (Oct 6) `s7_throughput2` done: 64x512 at bucket 64x512 (typed-decisions rows) 528.48 ms end to end, 523.33 ms
  replay only, 121.1 rows per second; `perf_summary.json` and `latency_table.json` regenerated; `test_performant.py`
  6 passed, 1 skipped.
- 00:04 to 00:10 (Oct 6) Served checks with the final buckets (`LAYA_SEQ_BUCKETS=128,256,512`,
  `LAYA_ROW_BUCKETS=1,2,4,5,8,10,16,32,50,64`, `LAYA_RAW_FORWARD=1`): host server loaded in 7.0 s (warmup 3.5 s plus
  0.54 s, 30 traces, 148,373,504 trace bytes) and reported ready with every trace captured; build 1 results
  `/home/hous/dev/laya/evals/results/host_tt_p150_b1_20261006T000400Z` (policy `bf8w_hifi3_erf`); the 1x4 mesh loaded in
  11.6 s (warmup 8.1 s plus 0.7 s) and served the E5 column `/home/hous/dev/laya/evals/results/host_tt_p150x4_b1_20261006T000708Z`,
  merged into the build 1 SUMMARY. The served E5 cells carry 1.8 to 21 ms of server-side work above the device time
  (request tokenization through the vendored builder, two `build_sequence` calls per question, decode); it was 22 ms
  at 50 questions in build 0 as well and is now a larger share (10 percent of the 50-question cell); recorded as an
  open item for the serving track.
