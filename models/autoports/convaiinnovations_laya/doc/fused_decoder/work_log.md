# Work log: stage 2 (graph fusing), Track T1

All times UTC, 2026 Oct 5. Same conventions as `../functional_decoder/work_log.md`.

- 21:20 to 21:30 (while stage 1 device jobs ran): wrote `tt/runner.py` (`LayaTraceRunner`) and
  `tests/replay_trace_check.py` from the local trace API (`begin_trace_capture(mesh_device, cq_id)`,
  `end_trace_capture`, `execute_trace(blocking)`, `release_trace`) and `ttnn/unsafe_allocation_tracker.py`.
- 21:34: `s2_replay_check` with `TT_METAL_TRACE_ALLOC_TRACKING=1` on 1x512 and 8x512
  (`/home/hous/dev/laya/logs/p3_s2_replay_check_20261005T213425Z.log`): pass; traced == eager bit identical, replays
  identical, input change detected, no tracker error. Whole job 24 s including the weight upload and two captures.
- 21:36: the tracked run's traced timings (305 and 348 ms) are dominated by the tracker's per-replay `gc.collect()`;
  an untracked timing run is queued behind the Tracy profile in the quiet-host queue
  (`/home/hous/dev/laya/scratch/t1_quiet_queue.sh`, waits for a 1-minute load under 8, then each step under devlock).
