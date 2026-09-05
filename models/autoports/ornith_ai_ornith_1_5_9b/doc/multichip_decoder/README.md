# Ornith-1.5-9B multichip decoder

Status: complete. Numerical, native-capacity, worker-watcher and profiler
checks pass; independent [stage review](STAGE_REVIEW.md) returns clean-pass.
Local checkpoint SHAs are recorded in [work_log.md](work_log.md).

The target is all four Blackhole chips on two physical P300c boards, logical
1x4 ring. `p150x4` is the software profile name. The implementation builds on
`tt/optimized_decoder.py` from baseline commit
`d085eb6d1abcc8b25f213fede6b68aa873d6cd6b` and pinned HF revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. This is a dense model: expert routing
and MoE execution do not apply. No full model or serving implementation is part
of this stage.

## Layer and stack contract

- Weights use TP4: column-parallel QKV/Z/MLP gate/up and row-parallel attention,
  DeltaNet and MLP outputs. Each row output is reduced before its residual add.
- Public input and output are replicated BF16 `[B,T,4096]`. Prefill output is
  interleaved DRAM; decode output is width-sharded within each chip. These match
  the next decoder's accepted input, with no per-layer boundary conversion.
- Full attention owns four query heads and one KV head per chip, dimension256.
  K and V each have local shape `[physical_pages,1,64,256]`, BFP8. Page tables
  are replicated int32, padded to32 entries, and use64-token page IDs. Positions
  remain absolute: device `current_pos[B]` int32 and `rot_idxs[1,B]` uint32.
- DeltaNet owns four key heads and eight value heads per chip, dimension128.
  Its FP32 recurrent state is `[B,8,128,128]`. Convolution history follows local
  Q512/K512/V1024 channel ownership. A/B fields are independently padded8→32
  during load; their packed offsets are2048 and2080.
- Logical sequence lengths remain arbitrary within capacity. Tile, page, and
  prefill chunk padding is internal. Native262144 is retained; batches1–32 are
  tested with shorter per-user caches. Aggregate batch/context allocation is
  caller-owned, as in the optimized baseline.

`mesh_plan.md` records the plan made before implementation and subsequent
measured refinements, with all per-device shapes, shard formulas, collective
payloads, padding, and rejected strategies. `memory_capacity_plan.json` and
`../context_contract.json` record the full-stack weight/KV/reserve calculation.
The current estimate is12,915,998,720 bytes per device at native batch1,
including8GiB for activation/trace/allocator peaks. The reservation run passed with the planned persistent bytes plus scratch
held during native full-attention execution. This is not a full-model run.

## Selected candidate and alternatives

The selected path shares interleaved BFP4 projection weights between prefill
and decode, except that full-attention decode uses an additional raw-HF BFP8
QKV/gate copy in width-sharded DRAM. All weight projections use LoFi. Public residuals and CCL use BF16; the
inherited DeltaNet recurrence and its output-projection input remain FP32. Decode QKV uses the dedicated DRAM matmul with32 input-shard
cores and K block4; other projections use an8x4 multicast compute grid.
The complete DRAM-sharded family remains an explicit control
(`MeshConfig(decode_grid=None)`). Projection storage across24 linear and8
full-attention layers is1,064,697,856 bytes per device. FP32 recurrent L1
placement budgets all24 layers together; larger batches use DRAM state.

Replicated residuals currently outperform measured hidden-sharded contracts.
The measurements include directly consumed reduce-scatter output, distributed
RMSNorm, delayed gather, gathered-input column output projections, fused AG-MM,
fused MM-RS, persistent collective buffers, fused norm-input gather/MM, packed
MLP projections, and BFP8 activation/collective boundaries. The test-boundary
gather for sharded candidates is outside layer timings. See
`candidate_measurements.csv` / `.md` and `geometry_search.md` for actual results,
including failed and rejected controls. These are measured choices, not a claim
of global optimality.

The original220-candidate BFP4 DRAM geometry sweep and92-candidate
interleaved block sweep cover both layer kinds. The32-worker interleaved
family beats64/110 workers. BFP4 QKV needs block2 to pass the real-weight
batch32 HF gate; larger blocks, HiFi2 and FP32 accumulation controls fail.
Decode-only BFP8 permits a faster block and passes the same per-user gate.
The precision-locked BFP8 QKV sweep compares DRAM input-core counts and all
fitting blocks against interleaved blocks. The mixed32-core DRAM/block4 path
measures0.2840ms full-attention decode and minimum batch32 HF PCC0.99687457.
Final default reproduction and the compatible topology matrix are recorded
under `logs/finalmixed_*`, `logs/release_l1small_*` and
`logs/review_prefill_stable_b16_*`. The same-policy
64/110-worker QKV controls and corrected DRAM block32 control also pass and
are slower; see `PERF_ANALYSIS.md`.

The fabric packet policy is8192 bytes. Configure it before opening the mesh:

```python
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import (
    MultichipDecoder,
    fabric_router_config,
)

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_1D_RING,
    router_config=fabric_router_config(),
)
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 4), trace_region_size=32 * 1024**2, l1_small_size=24576,
)
# Construct each decoder from its own checkpoint layer state and shared mesh.
# Close the mesh after releasing layer tensors and traces.
```

## Validation evidence

| Check | Final artifact/result |
|---|---|
| Logical/cache/HF/trace/batch/native/fallback contracts | `logs/release_mixed_watcher_contracts.log.gz`:99 passed,815.61s; both kinds, batches1–32, positive host-guard controls |
| Direct linear/full/linear stack | `logs/release_l1small_stack.log.gz`:131-token prefill,8 changed-input/position trace steps; minimum TTNN baseline PCC0.99996213 prefill /0.99992172 decode; exact restored eager/trace; zero boundary conversions |
| Native paged-cache oracle | Same99-case log:position262143, permuted4096pages, BFP8, one distinct KV head per chip; HF PCC0.99836834, exact eager/trace |
| Native full-prefix checks | Same99-case log:262143/262144 both kinds, final-position trace, chunk2048-vs1024 tail PCC0.999972 linear /0.999954 full, HF8001 checks |
| Full-stack memory reservations | Same99-case log:6,473,908,224 additional DRAM bytes/device plus24 recurrent L1 states held during native full-attention execution |
| Worker watcher | Same99-case run exits0; all-fixture appended watcher log and kernel-name map in `release_mixed_watcher_manifest.json`; ETH explicitly disabled |

`validation_summary.json` ties the final gates to the production source hash.
The worker watcher raw log has no assertion/error signatures. Profiler was
unset during that run and watcher was unset during all four final captures.

The native cache oracle uses exactly representable binary-fraction historical
KV fixtures, real checkpoint weights, and a real recorded query input. It does
not claim an HF native-length prefill rollout. Long contract inputs cycle
recorded HF layer-input rows; full-layer HF comparison is at8001 tokens.

## Performance and reproduction

Warmed batch1 results use a2048-token prefill and decode at position2048.
Prefill is the median of15 warmed synchronized API calls; decode is the
median of five windows of32 trace replays. The baseline runs
on an actual1×1 mesh before the1×4 run. Both use32MiB trace and24KiB small-L1
reservations. Exact commands, source and environment are in
`logs/review_prefill_stable_b16_layer{0,3}.provenance.json`.

| Layer | Mode | Single-chip ms | TP4 ms | Speedup | Efficiency |
|---|---|---:|---:|---:|---:|
| Linear attention | Prefill | 6.9491 | 3.3638 | 2.066× | 51.6% |
| Linear attention | Decode | 0.5239 | 0.3709 | 1.412× | 35.3% |
| Full attention | Prefill | 5.4046 | 2.6779 | 2.018× | 50.5% |
| Full attention | Decode | 0.4271 | 0.2840 | 1.504× | 37.6% |

All-rank PCC against optimized TTNN is0.99996308/0.99998799 for linear
prefill/decode and0.99997725/0.99969599 for full attention. Restored eager,
repeated eager and trace replay are bitwise identical on every output rank.
Speedup=single/TP4; efficiency=speedup/4. No full-model speed is inferred.

`PERF_ANALYSIS.md` and `performance_summary.json` reconcile unprofiled
latency with instrumented host/device windows and cover compute, DRAM,
collectives, data movement, and every material advice category. Projection
runtime rows confirm the selected BFP4/BFP8 LoFi policy. Advice-enabled
human/CSV reports for all four devices are under `tracy/<kind>/final/`.
Original merged reports and compressed ops CSVs remain available; per-device
window tables remove only the first pre-window profiler-drain gap. Device
times are never summed across chips.

Reproduce the complete worker-watcher gate, direct-stack comparison and paired
timings with `python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/run_validation.py <new-label>`.
Collect separate profiler reports with the same directory's
`run_profiles.py <new-label>` and watcher unset. These scripts serialize all
hardware commands and preserve source/command provenance.

Commands use the persistent environment:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py <unique-name> \
  timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --length 2048
```

`record_run.py` preserves exact commands, environment, source hashes, source
archives, exit status and compressed logs. `work_log.md` records decisions,
recoveries and review/commit SHAs. Raw Tracy captures and bulky
diagnostic tensors remain in the persistent local workspace; compact CSVs,
reports, source/log archives and tensor checksums are committed.
`diagnostic_tensor_manifest.json` identifies raw tensor files exactly.

## Investigations and limits

- `AUTODEBUG_reader_mesh.md`: reader2/3 DRAM-matmul mesh API limitation; reader1
  and wider interleaved compute are tested alternatives.
- `AUTODEBUG_sharded_trace.md` / `AUTOFIX_sharded_trace.md`: two-link async
  collective corruption controlled by one-link async candidates. Native
  all-reduce has different decomposition/semaphore ownership but shares RS
  kernel code; it must be validated for its actual shapes.
- `AUTODEBUG_batch32.md` / `AUTOFIX_batch32.md`: same-input single-chip control,
  projection localization, exact state/trace checks, and verified block fixes.
- `AUTODEBUG_long_replica.md`, `AUTOTRIAGE_capacity.md`: original replica and
  read-completion failures recovered after reset. Original gates pass without
  a speculative code change; original-order, repeated capacity and scoped
  watcher controls pass (`AUTOFIX_long_capacity.md`). Final mixed-path gates
  are recorded separately.
- Full Ethernet watcher instrumentation exceeds the kernel buffer; the
  no-inline retry completes model checks but aborts during Ethernet teardown.
  Worker watcher with `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`
  passes all99 final contracts and exits cleanly. All-fixture watcher logs are
  archived with append enabled. An all-feature watcher pass is not claimed.

Hardware commands are serialized. Watcher and profiler are separate runs.
This Python-only change does not require a C++ build. Repository checks and
independent review/commit status are recorded in `work_log.md`; never push.
