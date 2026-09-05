# Final performance analysis

All measurements use four Blackhole chips on two P300c boards, logical1×4 ring,8192-byte fabric payloads,32MiB trace region and24KiB small-L1 reservation. Production projection source is the99-case validated mixed path. The optimized baseline uses an actual1×1 mesh. No full-model throughput is inferred.

## End-to-end and profiler reconciliation

Device clocks are independent. Each device is accounted separately; no times are summed across chips. Supplemental rank tables remove only the first gap that starts before the PERF signpost (including the preceding profiler drain). Every kernel and all intra-window gaps are retained. Original merged reports and raw compressed ops CSV remain available.

| Layer | Mode | Unprofiled TP4 ms | Profiled host ms | Device kernel μs range | Intra-window gap μs range | Maximum device window μs | Host minus maximum window μs |
|---|---|---:|---:|---:|---:|---:|---:|
| linear_attention | prefill | 3.3638 | 4.4084 | 3245.7–3266.7 | 770.5–785.0 | 4044.8 | 363.7 |
| linear_attention | decode | 0.3709 | 0.4025 | 300.5–302.2 | 81.0–83.1 | 384.4 | 18.2 |
| full_attention | prefill | 2.6779 | 3.8174 | 2558.7–2571.5 | 958.5–963.1 | 3530.0 | 287.4 |
| full_attention | decode | 0.2840 | 0.3359 | 266.7–268.8 | 50.3–50.6 | 319.1 | 16.8 |

Profiled decode uses four replays, versus32 per window in the unprofiled benchmark. Instrumented kernels, host tracing and the shorter batch of replays increase the profiled times. The remaining host/window difference covers work outside device operation intervals, including enqueue/synchronization; it is not relabeled as CCL or compute. Unprofiled prefill uses15 warmed samples to resolve the close block8/16 comparison. Prefill is an untraced synchronized API call in both regimes, and shows greater host instrumentation overhead. Only unprofiled paired measurements determine speedup/efficiency.

## Optimistic decode bandwidth floor

At512GB/s per device, linear decode reads30,818,304 projection-weight bytes,
giving a60.19μs weights-only lower bound. Full attention reads34,734,080
projection bytes:29,491,200 persistent BFP4 bytes minus5,898,240 bytes of
unused decode BFP4 QKV plus11,141,120 bytes of the BFP8 QKV copy. At2049
logical cache tokens,65 token tiles×8 head-width tiles×1088 bytes×K/V
adds1,131,520 bytes. The resulting35,865,600-byte full-attention lower bound
is70.05μs. Equivalently, aggregate TP4 bytes divided by4×512GB/s gives
the same times; using all-layer persistent copies as one-layer traffic would
be incorrect.

These optimistic lower bounds omit recurrent-state traffic, cache writes,
activation movement, tile/chunk over-read, compute, communication and host
overhead. They compare to unprofiled decode370.9μs linear /284.0μs full,
and profiled device windows approximately384μs /319μs. They are not
achievable end-to-end targets under the current operator contract. The
residual/collective and movement tables below explain a substantial part of
the difference, while narrow batch1 work and recurrent/attention compute
remain after weight parallelism.

## Device work breakdown

| Layer | Mode | Projection/internal matmul μs | Other compute μs | CCL μs | Data movement μs |
|---|---|---:|---:|---:|---:|
| linear_attention | prefill | 780.1–789.8 | 1635.5–1659.4 | 697.4–717.1 | 117.4–118.9 |
| linear_attention | decode | 137.4–143.3 | 76.2–76.8 | 38.8–43.7 | 43.3–43.5 |
| full_attention | prefill | 703.9–706.4 | 994.0–1002.6 | 699.0–704.1 | 157.9–160.3 |
| full_attention | decode | 115.4–126.8 | 67.2–67.5 | 37.3–50.3 | 35.4–35.8 |

The two row-output reductions remain material: approximately0.70ms of prefill device work and39–50μs of decode device work. Each BF16 replicated residual has S=B×Tphysical×4096×2 bytes:16MiB at2048 prefill or256KiB for a batch1 decode tile. Two ring all-reduces imply3S algorithmic bytes sent per device, excluding protocol and padding. Tiny decode messages achieve less effective bandwidth because collective startup and synchronization remain.

Direct reduce-scatter consumption through distributed norm, fused norm/gather/MM, gathered-output column projections, fused MM-RS, persistent buffers, packed MLP and BFP8 CCL/activation boundaries were measured as whole layers. Their source/precision/topology is archived in finalmixed*/selected32* controls. The lower-communication contracts did not reduce whole-layer time on this ring. The sharded test-boundary gather is excluded from layer timing.

Data movement is35–43μs during decode and117–160μs during prefill. Resharding, rotary/head layout conversion, slicing, and cache preparation are visible in per-device tables. Wider standard matmuls and shared weights remove the former DRAM-sharded input/output transformations for most projections. The retained QKV DRAM path beats interleaved alternatives despite its two layout conversions.

## Projection runtime and roofline advice

Runtime rows, not policy labels, establish dtype/fidelity. All prefill projections and all decode projections except full-attention QKV use BFP4/LoFi weights. Decode QKV uses BFP8/LoFi. Public residuals and CCL use BF16; the inherited DeltaNet recurrence and gdn_out input are FP32. Three internal recurrence matmuls remain FP32/HiFi4, as in the optimized baseline. They are not BFP4 weight projections.

| Layer | Mode | Matmul shape | Runtime fidelity/dtypes | Mean kernel μs | Mean DRAM % | Mean FLOPs % | K block | Reported output subblock |
|---|---|---|---|---:|---:|---:|---|---|
| linear_attention | prefill | 2048 x 4096 x 2112 | LoFi BF16 x BFP4 => BF16 | 111.7 | 52.0 | 52.1 | 16 | 1×6 |
| linear_attention | prefill | 2048 x 4096 x 1024 | LoFi BF16 x BFP4 => BF16 | 96.9 | 46.5 | 29.2 | 16 | 1×3 |
| linear_attention | prefill | 2048 x 1024 x 4096 | LoFi FP32 x BFP4 => BF16 | 101.8 | 52.3 | 27.8 | 16 | 1×6 |
| linear_attention | prefill | 2048 x 4096 x 3072 | LoFi BF16 x BFP4 => BF16 | 155.6 | 44.8 | 54.5 | 16 | 1×3 |
| linear_attention | prefill | 2048 x 3072 x 4096 | LoFi BF16 x BFP4 => BF16 | 163.1 | 42.7 | 52.0 | 16 | 1×6 |
| linear_attention | decode | 32 x 4096 x 2112 | LoFi BF16 x BFP4 => BF16 | 19.4 | 43.5 | 23.4 | 32 | 1×3 |
| linear_attention | decode | 32 x 4096 x 1024 | LoFi BF16 x BFP4 => BF16 | 12.5 | 32.8 | 12.1 | 32 | 1×1 |
| linear_attention | decode | b={8} x 32 x 128 x 128 | HiFi4 FP32 x FP32 => FP32 | 6.5 | — | 14.4 | 1 | 1×4 |
| linear_attention | decode | 32 x 1024 x 4096 | LoFi FP32 x BFP4 => BF16 | 9.7 | 42.2 | 15.6 | 8 | 1×4 |
| linear_attention | decode | 32 x 4096 x 3072 | LoFi BF16 x BFP4 => BF16 | 26.4 | 46.7 | 17.3 | 8 | 1×3 |
| linear_attention | decode | 32 x 3072 x 4096 | LoFi BF16 x BFP4 => BF16 | 25.4 | 48.4 | 17.9 | 6 | 1×4 |
| full_attention | prefill | 2048 x 4096 x 2560 | LoFi BF16 x BFP4 => BF16 | 137.1 | 46.3 | 56.6 | 16 | 1×4 |
| full_attention | prefill | 2048 x 1024 x 4096 | LoFi BF16 x BFP4 => BF16 | 93.2 | 48.3 | 30.3 | 16 | 1×6 |
| full_attention | prefill | 2048 x 4096 x 3072 | LoFi BF16 x BFP4 => BF16 | 155.8 | 44.7 | 54.4 | 16 | 1×3 |
| full_attention | prefill | 2048 x 3072 x 4096 | LoFi BF16 x BFP4 => BF16 | 163.6 | 42.6 | 51.8 | 16 | 1×6 |
| full_attention | decode | 32 x 4096 x 2560 | LoFi BF16 x BFP8 => BF16 | 30.5 | 67.5 | 50.0 | 4 | factory-derived |
| full_attention | decode | 32 x 1024 x 4096 | LoFi BF16 x BFP4 => BF16 | 9.8 | 41.9 | 15.5 | 8 | 1×4 |
| full_attention | decode | 32 x 4096 x 3072 | LoFi BF16 x BFP4 => BF16 | 26.2 | 47.0 | 17.4 | 8 | 1×3 |
| full_attention | decode | 32 x 3072 x 4096 | LoFi BF16 x BFP4 => BF16 | 25.7 | 47.9 | 17.8 | 6 | 1×4 |

`tt-perf-report` uses its Blackhole estimate of512GB/s DRAM and per-core compute ceilings. These percentages are shape/dtype roofline estimates, not measured hardware counters; tiled batch1 work includes the physical32-row tile. BFP exponent headers are included in our capacity accounting, but the tool’s nominal operand byte estimate does not include that storage overhead.

Decode QKV reaches about70% of the tool’s DRAM ceiling. Its public DRAM-sharded program exposes no output-subblock fields, which explains blank CSV cells. The factory derives8 reader/compute workers from8 banks×reader1, N80tiles→10tiles/worker, and output subblock1×5. The public per_core_N3 instead describes storage distribution across32 input/storage cores. Source: `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:159–193` and `device/config/matmul_program_config.cpp:16–24,974–1011`. This is source-derived interpretation, not a field invented in the raw profiler CSV. Reader2/3 are blocked by the precisely reproduced mesh helper API mismatch in AUTODEBUG_reader_mesh.md; source and binary identity are recorded there.

Other decode projections remain SLOW in the advice tables (roughly33–50% DRAM and12–24% padded compute estimates). Their chosen BFP4/LoFi geometry was swept across DRAM input-core counts and all legal K blocks, then across the32-worker interleaved grid. Wider64/110-worker BFP4 whole-layer controls lose. The final BFP8 QKV additionally has same-policy32/64/110-grid and DRAM-core sweeps; corrected block32 fits but loses. Whole-layer QKV-only controls confirm the component ranking, including consistent24KiB small-L1 setup. No larger-core or small-block rejection rests only on another precision policy.

Z has per_core_N1 on the32-worker grid, so its legal output subblock is1×1; a wider subblock cannot consume more useful output tiles. Its smaller-compute DRAM family and all legal K divisors were measured. Other interleaved decode subblocks are1×3 or1×4 and are present in raw attributes/CSV. Prefill uses110 workers, K block16 and explicit1×3/4/6 subblocks at2048. Lower-fidelity advice is already applied to projections; higher-fidelity controls were measured, and BF16/FP32 state precision follows the optimized baseline.

No claim is made that the roofline is saturated or that arbitrary untested kernel algorithms cannot improve this result. Acceptance is based on the tested local geometry, precision, topology, capacity and correctness contract.

## Same-policy prefill geometry controls

All controls retain the final BFP4/LoFi prefill policy and24KiB small-L1 setup.
Only the prefill grid/K block changes. Times are warmed whole-layer medians.

| Grid / K block | Linear prefill ms | Full prefill ms | Evidence label |
|---|---:|---:|---|
|8×8 /8|3.6901|2.9323|review_prefill_g8x8_b8_layer0/3|
|8×8 /16, linear gdn_out8|3.6067|2.9241|review_prefill_v2_g8x8_b16_layer0/3|
|11×10 /8,15 samples|3.4480|2.8013|review_prefill_stable_b8_layer0/3|
|11×10 /16,15 samples, selected|3.3638|2.6779|review_prefill_stable_b16_layer0/3|

The unadapted8×8/block16 linear control fails specifically at FP32 gdn_out:
static circular buffers require1,717,248 bytes against physical L1 of1,572,864.
The adapted control keeps only that projection at block8; full attention needs
no adaptation. This exact op-capacity limit differs from the earlier QKV
probe's diagnostic retained-input pressure. Reset/list/mesh recovery passed.

Initial three-sample11×10/block8 results were close enough to require a longer
comparison. Fifteen measured calls after one warmup favor block16 for both
kinds; these paired runs also supply the final reported baseline and TP4
timings. No production change follows. Exact commands, source snapshots and
outputs are in logs/<evidence-label>.{provenance.json,sources.json.gz,log.gz}.

## Exact artifacts

- `performance_summary.json`: paired timings and this profile accounting.
- `candidate_measurements.csv` / `.md`, `geometry_search.md` and geometry JSON: alternatives with source/command provenance.
- `tracy/{linear_attention,full_attention}/final/{prefill,decode}_device{0,1,2,3}_report.{txt,csv}`: advice-enabled human and CSV reports.
- Matching `*_window.csv`, `*_rank_accounting.json`, original `*_perf_report.{txt,csv}`, `*_ops.csv.gz`, `*_provenance.json`: preserved/reconciled data.
- `logs/profile_final_*.{log.gz,provenance.json,sources.json.gz}`: exact captures, HF checks and source identity.
- Reproduction: `python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/run_profiles.py <new-label>` with watcher unset.
