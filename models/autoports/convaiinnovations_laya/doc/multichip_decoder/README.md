# Stages 4 and 5: data-parallel 1x4 mesh (Appendix A.6) and its efficiency audit

Plugin stages "multichip" and "optimize (multichip)" mapped to Laya (PLAN.md section 5, rows 4 and 5). Four Blackhole
chips (two p300c cards, `/dev/tenstorrent/0..3`) opened as one `ttnn.MeshShape(1, 4)` mesh device without a fabric
config. The workload is data parallel (DP) only: every chip runs the whole encoder plus head on its own rows; there is
no collective, so the CCL (collective communication library) audit of stage 5 is not applicable and this README reports
the per-device replay overlap and the host concat cost per cell instead. PCC = Pearson correlation coefficient.

## What changed in the code (recorded as the plan allows for stage 4)

| file | change |
|---|---|
| `tt/laya_model.py` | `open_mesh(mesh_shape, l1_small_size, trace_region_size)` and `close_device(device)`; `mesh_size(device)`; `TtnnLayaModel` reads `device.get_num_devices()`, builds `ReplicateTensorToMesh` for the weights, the rotary caches and the static masks when the caller passes no mapper, `ShardTensorToMesh(dim=0)` for the three per-call inputs and `ConcatMeshToTensor(dim=0)` for the two readbacks; `rows_per_call(bucket) = bucket x devices`, `row_buckets_total()`, `bucket_for(n_rows)` picks the per-device bucket for `ceil(n_rows / devices)`; `host_inputs` builds `(bucket x devices, S)` host tensors sharded over the mesh; `forward_with_hidden` (eager, used by stage 6 for the hidden-state PCC). On a 1x1 device every call is unchanged (no mapper, no composer) |
| `tt/runner.py` | the dummy warmup inputs carry `rows_per_call` rows; `run_timed` replays with `blocking=True` so the write, the replay and the readback are timed apart; `describe()` reports `num_devices` |
| `tests/test_dp_mesh.py` | the evidence script (`--mesh 1x1` writes the single-chip reference, `--mesh 1x4` compares and benchmarks) and the pytest gates over the saved JSON |

Per-call device buffers are still `ttnn.allocate_tensor_on_device` with the per-device shape; on the mesh that is one
buffer per device at one address (`buffer_address()` is checked before every replay as on one chip). The probe that
established this build's behaviour (`/home/hous/dev/laya/scratch/t2_mesh_probe.py`, log
`/home/hous/dev/laya/logs/p3_mesh_probe_20261005T224656Z.log`): a host tensor built with `ShardTensorToMesh(dim=0)`
copies into such a buffer with `copy_host_to_device_tensor` and reads back through `ConcatMeshToTensor` bit for bit;
`from_torch(device=mesh)` without a mapper replicates (4 shards); the broadcast mask add and trace capture, replay and
release work on the mesh device.

Static masks per device: `TtnnMaskBuilder` receives the per-device bucket, so `band` and `zeros` are `(B, 1, S, S)`
per chip and replicated; the per-call `pad_row` shard is `(B, 1, 1, S)` per chip. Trace capture is one
`begin_trace_capture(mesh)` per bucket; `execute_trace(mesh, ..., blocking=False)` replays on all four chips.

## Inputs and method

`tests/test_dp_mesh.py` takes the first 64 rows of the typed-decisions gate subset from
`/home/hous/dev/laya/reference/parity_corpus.npz` (13 cases, rows 0 to 63: 26 choice, 25 score, 13 noul questions with
their CPU fp32 logits and temperatures), seq bucket 512, shipped policy `bf8w_hifi3_erf` and shipped `PortConfig`.

- Single chip (chip 0, `ref_1x1.json`, log `/home/hous/dev/laya/logs/p3_s4_dp_1x1_20261005T225108Z.log`): per-device
  buckets 8, 16 and 64; every bucket runs all 64 rows (8 calls of 8, 4 of 16, 1 of 64) and records the marker logits;
  timing = traced p50 of 20 replays after 3 warm (`runner.run`, non-blocking replay plus the two readbacks) and a
  blocking split of 10 replays (`run_timed`: write, replay, readback).
- Mesh (`dp_agreement_1x4.json`, `bench_1x4.json`, log `/home/hous/dev/laya/logs/p3_s4_dp_1x4_20261005T225247Z.log`):
  per-device buckets 8 and 16, so 32 and 64 rows per call; the same 64 rows, the same timing protocol.
- Probabilities use the corpus temperatures (clamped rule equals the raw rule on every corpus item). "Confident" means
  the reference top-1 minus top-2 probability is at least 0.10; for the 1x1-versus-1x4 comparisons the reference is the
  1x1 result, for the CPU columns it is the CPU fp32 result.

Host 1-minute load during both runs: 0.75 to 1.94 (16 cores), well under the orchestrator's limit of 8.

## Results

### Correctness gates (stage 4)

| gate | measured | threshold | result |
|---|---|---|---|
| same inputs, 1x1 versus 1x4 at the same per-chip bucket, max abs marker-logit delta | 0.0 at B 8x4 (32 rows) and 0.0 at B 16x4 (64 rows): bit identical | <= 1e-3 | pass |
| same inputs, same per-chip bucket, argmax agreement | 32 of 32 and 64 of 64 | 100 percent | pass |
| 1x1 B 64 versus 1x4 B 16x4, scorer-logit PCC over the 236 valid markers | 0.99999999997 (max abs delta 1.2e-4, max abs delta p 1.5e-5) | >= 0.999 | pass |
| 1x1 B 64 versus 1x4 B 16x4, confident agreement | 53 of 53 (plain argmax 64 of 64) | >= 99 percent | pass |
| throughput, 1x4 B 16x4 versus 1x1 B 64 | 453.6 versus 121.3 rows per second, 3.74x | >= 3x | pass |

The per-chip program is the same object at the same bucket, so the mesh reproduces the single chip exactly. The 64-row
cross check differs only because the 64x512 and 16x512 buckets use different matmul blocking (both DRAM attention
chain, interleaved GeGLU, `minimal_matmul` for Wqkv and Wo); the 1.2e-4 logit delta is the same order as T1's
traced-versus-eager noise floor of zero and far below the gate.

Both placements against CPU fp32 on the same 64 rows (informational here; stage 6 runs the full 200-decision gate):
PCC 0.9980 (B 8 per chip) and 0.9982 (B 16 per chip and 1x1 B 64), argmax 62 of 64, confident 56 of 56, max abs delta p
0.043 (B 8) and 0.035 (B 16 and B 64), median 0.0126 and 0.0124. The two flips are the same two low-margin decisions in
every placement.

### Throughput and latency (`bench_1x4.json`, traced p50 of 20 after 3 warm, load 0.8 to 1.9)

| cell | rows per call | traced p50 | min | p95 | rows per second | 1x1 at the same per-chip bucket (p50, rows per second) | speedup versus 4 single-chip calls | speedup versus 1x1 B 64 |
|---|---|---|---|---|---|---|---|---|
| 1x4, B 8 per chip | 32 | 69.67 ms | 66.82 | 70.46 | 459.3 | 65.09 ms, 122.9 | 3.74x | 3.79x |
| 1x4, B 16 per chip | 64 | 141.10 ms | 139.94 | 143.51 | 453.6 | 137.21 ms, 116.6 | 3.89x | 3.74x |
| 1x1, B 64 | 64 | 527.72 ms | 527.27 | 528.31 | 121.3 | | | 1.0x |

Single-call latency is unchanged within 4.6 ms (B 8) and 3.9 ms (B 16) of the single chip; the mesh pays the host-side
input shard and output concat, not device time. Expected from Appendix A.6: 3.6x to 3.9x. Measured: 3.74x to 3.89x.

### Stage 5: per-device replay overlap and host costs per cell (blocking split, p50 of 10)

| cell | write inputs 1x1 -> 1x4 | device replay 1x1 -> 1x4 | replay overlap (1x1 replay / 1x4 replay) | readback 1x1 -> 1x4 | host concat cost (readback delta) | total host overhead of the mesh per call |
|---|---|---|---|---|---|---|
| B 8 per chip (32 rows) | 0.07 -> 2.57 ms | 64.42 -> 65.17 ms | 0.988 | 0.37 -> 0.87 ms | 0.49 ms | 2.99 ms (4.6 percent of the call) |
| B 16 per chip (64 rows) | 0.11 -> 1.67 ms | 135.85 -> 137.08 ms | 0.991 | 0.69 -> 1.81 ms | 1.11 ms | 2.68 ms (1.9 percent of the call) |

Reading: the four chips replay within 1.2 percent of a single chip (overlap 0.99); the measurable mesh costs are on the
host: the sharded `from_torch` plus four `copy_host_to_device_tensor` writes (1.6 to 2.6 ms against 0.1 ms on one chip)
and the concat readback of the `(64, 512, 1)` fp32 logits and `(64, 32, 1024)` bf16 CLS rows (0.5 to 1.1 ms extra).
Per row the overhead is 0.04 to 0.09 ms. Nothing in the stage 5 plan item remains: there is no collective to audit.

### Mesh bring-up costs

Mesh open 1.14 s (no fabric config); weights uploaded to four chips in 3.16 s (one chip: 1.75 s); warmup phase 1
(eager, two buckets) 0.80 s, phase 2 (capture) 0.04 s. Device ids as the mesh enumerates them: [1, 0, 3, 2].

## Gates not met or open

None. Trace safety on the mesh: the 1x4 comparison was repeated with `TT_METAL_TRACE_ALLOC_TRACKING=1`
(`/home/hous/dev/laya/scratch/dp_agreement_1x4_tracked.json`, log
`/home/hous/dev/laya/logs/p3_s4_dp_1x4_tracked_20261005T230916Z.log`, 5 replays per cell): the tracker raised no
error over the warmup, the captures and 60 replays, and all four correctness gates pass with the same values (logit
delta 0.0, PCC 1.0, 53 of 53). The timings of that run are not usable (replay 262 ms and 317 ms against 65 and 137
without the tracker, because the tracker runs `gc.collect()` before every replay, as stage 2 recorded on one chip), so
its throughput gate reads 1.67x and is not a measurement of the mesh. `LayaEngine` (`tt/engine.py`) exposes the mesh through `LAYA_MESH_SHAPE=1x4`: `shapes()["row_buckets"]` then lists the
total rows per call (4 x the per-device buckets) so the server pads to a multiple of four rows.

## How to reproduce

```
source /home/hous/dev/laya/bin/ttenv.sh; cd $TT_METAL_HOME
A=models/autoports/convaiinnovations_laya
TT_METAL_VISIBLE_DEVICES=0 /home/hous/dev/laya/bin/devlock python $A/tests/test_dp_mesh.py --mesh 1x1
TT_METAL_VISIBLE_DEVICES=0,1,2,3 /home/hous/dev/laya/bin/devlock python $A/tests/test_dp_mesh.py --mesh 1x4
python -m pytest $A/tests/test_dp_mesh.py -q -p no:cacheprovider -o addopts=""        # gates over the saved JSON, no device
```
