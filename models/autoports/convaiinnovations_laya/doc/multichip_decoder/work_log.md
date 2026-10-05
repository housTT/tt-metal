# Stages 4 and 5 work log (Track T2, UTC)

- 22:40 Read PLAN.md with amendments A1 to A9, the stage 1 to 3 READMEs, the probe and serving docs, `tt/laya_model.py`,
  `tt/runner.py`, `tt/model_config.py`, `tt/weights.py`, `tt/modernbert_masks.py`, `reference/laya_reference.py`,
  `server/engine.py`, the test harness patterns and the workspace scripts. T1 had already threaded a `mesh_mapper`
  through the weight upload, the rotary caches and the static masks; the per-call inputs, the readback and the bucket
  arithmetic were single-device only.
- 22:46 Mesh smoke `bin/mesh-smoke.py --chips 4` plus an API probe (`/home/hous/dev/laya/scratch/t2_mesh_probe.py`,
  log `p3_mesh_probe_20261005T224656Z.log`): mesh open 1.3 s without a fabric config, device ids [1, 0, 3, 2];
  `ShardTensorToMesh(dim=0)` host tensors copy into `allocate_tensor_on_device` buffers on the mesh and read back
  through `ConcatMeshToTensor` exactly; `buffer_address()` is stable; `from_torch(device=mesh)` without a mapper
  replicates; the broadcast mask add and trace capture, replay and release run on the mesh. Decision: keep
  `allocate_tensor_on_device` for the per-call buffers (same code path as one chip) instead of `host.to(mesh)`.
- 22:48 `tt/laya_model.py`: `open_mesh`, `close_device`, `mesh_size`, mesh detection in `TtnnLayaModel`, sharded
  `host_inputs`, concat `readback`, `rows_per_call`, `row_buckets_total`, `bucket_for` on `ceil(n / devices)`,
  `forward_with_hidden`. `tt/runner.py`: dummy inputs with `rows_per_call` rows, `run_timed` (blocking replay, split
  timing), `num_devices` in `describe()`. The 1x1 call sequence is unchanged.
- 22:50 `tests/test_dp_mesh.py` written (CLI for the two runs, pytest gates over the saved JSON, optional live test).
- 22:51 1x1 reference run on chip 0 (`p3_s4_dp_1x1_20261005T225108Z.log`, 35 s, load 0.8): B 8 65.09 ms, B 16
  137.21 ms, B 64 527.72 ms traced p50, matching stage 3 (65.81, 137.57, 528.23).
- 22:52 1x4 run (`p3_s4_dp_1x4_20261005T225247Z.log`, 17 s, load 0.8 to 1.9): all five gates pass; bit-identical logits
  at equal per-chip bucket; B 16x4 versus 1x1 B 64 PCC 0.99999999997, 53 of 53 confident agreements; 453.6 rows per
  second against 121.3 (3.74x); replay overlap 0.988 and 0.991; host concat cost 0.49 and 1.11 ms; the sharded input
  write costs 1.6 to 2.6 ms against 0.1 ms on one chip.
- 22:58 README written. Decision for stage 5: nothing to optimise on the device side (overlap 0.99); the host overhead
  per call is 2.7 to 3.0 ms (1.9 to 4.6 percent) and is reported, not tuned, because the plan's `p150x4` profile is a
  throughput profile where the 64-row call already reaches 3.74x.
- Deviations from the plan: none in substance. The plan's "static masks built per device for the per-device batch"
  holds because `TtnnMaskBuilder` already received the per-device bucket. A tracked (`TT_METAL_TRACE_ALLOC_TRACKING=1`)
  replay check was not repeated on the mesh (same capture code and address checks as the single chip; recorded as open
  in the README).
- 23:10 Tracked re-run of the 1x4 comparison with `TT_METAL_TRACE_ALLOC_TRACKING=1` launched
  (`s4_dp_1x4_tracked`, outputs under `/home/hous/dev/laya/scratch/`, 5 replays per cell) to close the open item.
- 23:11 Tracked run done: no tracker error, correctness gates identical; timings under the tracker 4x slower (not a
  mesh property) so the throughput gate of that run is disregarded; README updated.
