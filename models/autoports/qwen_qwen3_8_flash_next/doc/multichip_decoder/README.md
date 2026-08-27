# Qwen3.8-Flash-Next multichip decoder

This stage is complete for the fixed `1x2` Blackhole P300 mesh. It starts
from the completed `OptimizedDecoder`, keeps a persistent fractured residual
inside a layer stack, distributes QSA and the shared expert, and uses exact,
bounded host backing for routed experts and the PLE table that cannot fit in
device DRAM. Prefill, first-token decode, progressing segmented traces,
two-live-layer workspace reuse, paged KV, advertised-context geometry,
allocation tracking, latency, device profiling, and watcher are validated.
Full-model and vLLM work are outside this stage.

## Fixed target and delivered parallel plan

- Hardware: P300 dies 0 and 1, mesh `1x2`, `FABRIC_1D`, linear topology,
  one link, `11x10` worker grid, 34,225,520,640 bytes DRAM/die.
- Public ABI: replicated BF16
  `[1,1,logical_sequence_or_batch,10240]`. Stack-internal residuals are
  fractured as `[1,1,4*logical_rows,1280]` and are gathered only at an
  explicit stack/test exit.
- Hyperconnection: distributed RMSNorm and local projections operate on the
  persistent fractured residual.
- GDN: the 16-key/48-value-head recurrence and state are replicated. Its
  output projection is column-parallel into the local residual shard. A
  measured 24-value-head split was rejected for PCC loss.
- QSA: 12 query heads and one main KV head per die; Q/K/V widths are
  6144/256/256 per die. The indexer is replicated and the output is reduced
  into the local residual shard.
- MoE: the shared expert remains TP2 at intermediate width 320/die. Routed
  experts use deterministic EP2 ownership `expert_id % 2`: the owner executes
  the checkpoint-identical full K=640 projection and the other rank executes
  an exact-zero, full-shaped slot. The MoE collective sums the one routed
  owner plus both shared-expert partials into the local residual shard.
- Sparse execution: all 512 routed experts remain addressable, but only the
  gate-selected top 10 execute. Decode uses ten fixed slots; prefill processes
  larger route unions in bounded ten-expert waves. Dense all-expert execution
  is never used.
- KV cache: each QSA layer/die owns one BFP8 K and V head
  `[max_blocks,1,64,256]`; the BFP8 raw index cache
  `[max_blocks,1,64,128]` and BF16 compressed index cache are replicated.
  Page tables, rotary tensors, and INT32 positions are replicated.

The complete shapes, placement, communication, and rejected alternatives are
in `mesh_plan.md`.

## Why routed EP2 is required

The original resident/host-backed TP2 experiment split the routed expert's
down projection at K=640 into two K=320 sparse matmuls and reduced the
partials. A fresh AutoDebug run localized the first material error to this
boundary: reconstructed weights and gate/up were exact, but the down partial
sum was about 0.926 PCC and the routed output about 0.938 PCC. HiFi2, FP32
down output, fused-precision changes, and baseline routing substitution did
not repair it.

The isolated AutoFix retained only two changes:

1. place each selected routed expert on `expert_id % 2` with full K=640 and
   exact zeros on the peer;
2. select the full-width sparse program `expert_bfp4_lofi_g40b16_d40b5` for
   host-backed routed experts.

Full K with the old g20 program still failed layer-0 prefill at 0.99064916.
Full K plus g40 passed at 0.99942303. `FINAL_SOURCE_AUTODEBUG.md` and
`FRACTURED_EXPERT_AUTOFIX.md` preserve the diagnosis and isolated A/B evidence.
The known-invalid resident split-K path is retained only as negative evidence;
it is not an acceptance oracle for the delivered host-backed configuration.

## Exact bounded host capacity

Ordinary resident TP2 BFP4 routed experts alone require 33,973,862,400
bytes/die before maximum-context caches, non-expert graph weights, endpoints,
state, and runtime reserve. The accepted path mmap-loads exact checkpoint
weights only on a miss and keeps ten fixed BFP4 slots plus one upload staging
expert per layer/rank. Layer 1 uses the real 95.37 GiB PLE table through
EOS-aware n-gram hashing, exact selected-row mmap lookup, and stable BF16 TT
staging. Compact route IDs, sparse host lookup, and bounded expert/PLE H2D are
the only declared host boundaries; decoder, expert, PLE, and collective math
remain on TT.

| Per-device resource | Bytes |
| --- | ---: |
| Maximum-context QSA caches, 12 layers | 2,340,421,632 |
| Decoder non-expert weights | 3,479,858,176 |
| Natural TP2 embedding/final-mixer/LM-head endpoints | 1,279,016,960 |
| 48-layer routed-expert slots and upload staging | 1,459,814,400 |
| PLE prefill/decode staging | 819,200 |
| Canonical batch-one GDN/PLE decode state | 260,702,208 |
| Additional live prefill/user state | 208,928,768 |
| Runtime, activation, allocator, and trace reserve | 1,073,741,824 |
| Planned total | **10,103,303,168** |
| Headroom from physical DRAM | **24,122,217,472** |

Each expert/rank slot is 2,764,800 bytes: gate/up
`[1,1,2560,1280]` plus down `[1,1,640,2560]` in BFP4. A cold top-10 decode
transfers 55,296,000 bytes across both ranks; a hit transfers zero. The
checkpoint read remains 9,830,400 BF16 bytes per missed expert.

Canonical batch-one recurrence and taps live in DRAM. The 36 GDN layers share
one fixed L1 workspace sequentially: 65,536 bytes/worker for GDN plus 55,296
bytes/worker for PLE, or 120,832 bytes/worker at the layer-1 peak. The full
accounting is machine-checked by `test_host_weight_cache.py` and recorded in
`../host_weight_contract.json` and `../context_contract.json`.

## Correctness, paging, and trace evidence

The acceptance oracle is the unchanged TTNN `OptimizedDecoder` on the same
real checkpoint and inputs. The minimum gate is PCC 0.995.

| Representative layer | Kind | Seq-33 prefill PCC | First-token decode PCC |
| --- | --- | ---: | ---: |
| 0 | GDN | 0.99942303 | 0.99996978 |
| 1 | PLE + GDN | 0.99949104 | 0.99984211 |
| 3 | QSA | 0.99972457 | 0.99988294 |

The hardware contract also checks deterministic full-K owner/exact-zero peer
placement, packed-slot PCC against the BF16 source, exact PLE rows, shuffled
paged prefill/decode, rank-local K/V heads, replicated index state, device
INT32 positions, non-aligned logical length 33, batch 32 for all layer kinds,
and stack ingress/egress layout.

The allocation-tracked current-source acceptance matrix passed **12 tests in
224.56 s** (`expert_ep2_final_correctness_alloc.xml`). The complete CPU,
capacity, fallback, and non-aligned matrix passed **32 tests in 12.42 s**
(`expert_ep2_final_static_contracts.xml`). Maximum-context QSA capture/replay
passed at context 262,144 and current position 262,143 with cache shapes
`[4096,1,64,256]` and `[4096,1,64,128]`
(`expert_ep2_advertised_context_trace.xml`). No advertised capability was
reduced.

### Progressing segmented trace

Host-backed batch-one decode uses two TT traces around the declared expert
service boundary:

```text
stable inputs -> front trace (PLE/attention/router/state)
              -> compact route D2H + exact expert cache service
              -> back trace (selected routed/shared experts/collective/output)
```

GDN state is canonical in DRAM and hydrates/commits through the fixed shared
L1 workspace. Trace registration warms every stack capture signature before
freezing program-cache misses. Two live layer traces retain fixed workspace
addresses and independent state.

`expert_ep2_trace_stress100.xml` passed both GDN kinds and paged QSA for 100
changing tokens in 90.26 s. Each token checks output, routes, GDN/PLE state,
or QSA KV/index cache against eager TTNN.

## Warmed latency

`expert_ep2_perf_count7.xml` contains seven independent samples per
representative layer with seq-33 prefill and 100 decode replays/sample. The
path includes PLE, compact route D2H, exact expert lookup/packing/H2D, both TT
traces, state hydrate/commit, and collectives. Values below are medians across
the seven samples.

| Layer | Baseline prefill ms | Host EP2 prefill ms | Baseline trace decode ms | Host segmented decode ms | Speedup | 2-device efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 GDN | 18.693426 | 3,583.877886 | 1.106864 | 3.766486 | 0.293795x | 14.690% |
| 1 PLE+GDN | 20.052107 | 1,541.191421 | 1.484079 | 4.184993 | 0.354558x | 17.728% |
| 3 QSA | 43.731704 | 2,064.964676 | 2.861884 | 3.050174 | 0.938269x | 46.914% |

All 21 performance cases passed in 322.92 s. Demand-loaded prefill is much
slower because it services a large routed-expert union in bounded waves.
Decode also does not beat the single-chip graph baseline: exact host service,
canonical-state DRAM movement, gathers, and fixed collectives dominate the
token-sized TP savings. These are measured limitations, not omitted work.

## Device profiler and communication evidence

Watcher was off for profiling. Three fresh one-layer Tracy processes used a
2,000-program support buffer and the final full-width g40 expert program. All
three pytest processes and postprocessors passed, and each
`tracy_ops_data.csv` contains zero `Profiler DRAM buffers were full` messages.
`tt-perf-report` accepted every prefill/decode signpost range.

| Layer/window | Modeled DRAM roofline |
| --- | ---: |
| L0 prefill | 119 GB/s, 23.2% |
| L0 decode | 77 GB/s, 15.0% |
| L1 prefill | 94 GB/s, 18.3% |
| L1 decode | 75 GB/s, 14.7% |
| L3 prefill | 64 GB/s, 12.5% |
| L3 decode | 34 GB/s, 6.6% |

The oversized raw capture/ops CSVs were inspected locally and remain
gitignored. Compact detailed report CSVs, summary CSV/PNGs, and human tables
are retained under `tracy_host_ep2/*_capacity2000`. Custom-op categorization
warnings remain visible; they are not profiler-buffer overflow or dropped
evidence.

## Fallback, stress, and watcher

Static and runtime guards prohibit Torch conversions in attention, GDN,
selected-expert math, collectives, state workspace, and trace-open regions.
Allowed host work is limited to compact route IDs, exact expert/PLE mmap
lookup, and stable staging uploads.

The separate final watcher suite passed **4 tests in 32.88 s** for GDN,
PLE+GDN, paged QSA, and the two-live trace stack. The 556-line archived log has
SHA-256 `50e398a486e7b8290e31e8837284c1e2bcf7b2d4127e855c6eeb76be5b101830`
and no watcher error, assertion, panic, hang, timeout, NoC, or RISC failure
signature. `TT_METAL_WATCHER_DISABLE_ETH=1` retains Tensix, dispatch, NoC/CB,
stack, and waypoint checks while avoiding the known fabric-ERISC teardown
instrumentation issue. Watcher and profiler were never enabled together.

## Reproduction and retained evidence

Every hardware command used:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

Exact commands are in `work_log.md`. Primary retained artifacts are:

- `expert_ep2_host_cpu_reconciled.xml`, `expert_ep2_final_static_contracts.xml`
- `expert_ep2_first_decode_pcc.xml`, `expert_ep2_hardware_contracts.xml`
- `expert_ep2_final_correctness_alloc.xml`, `expert_ep2_trace_stress100.xml`
- `expert_ep2_advertised_context_trace.xml`, `expert_ep2_watcher.xml`
- `expert_ep2_perf_count7.xml`, `tracy_host_ep2_layer{0,1,3}.xml`
- `tracy_host_ep2/*_capacity2000`
- `FINAL_SOURCE_AUTODEBUG.md`, `FRACTURED_EXPERT_AUTOFIX.md`
- `SEGMENTED_TRACE_AUTOFIX.md`, `STAGE_REVIEW.md`
- `evidence_manifest.sha256`

No remote operation or push was performed.
