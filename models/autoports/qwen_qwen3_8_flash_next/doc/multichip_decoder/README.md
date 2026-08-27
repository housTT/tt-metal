# Qwen3.8-Flash-Next multichip decoder

This decoder-layer stage targets the fixed `1x2` Blackhole P300 mesh and
subclasses the completed single-chip `OptimizedDecoder`.  It implements real
TP2 QSA and gate-selected sparse MoE execution, replicated GDN, rank-local
paged KV caches, BF16 fabric reductions, and warmed decode trace replay.  No
full-model or vLLM code was started.

## Status

The per-layer implementation and hardware gates pass, but this stage is **not
pipeline-complete**.  The physical full-stack residency gate remains blocked:
ordinary TP2 BFP4 experts alone require 31.640625 GiB/die, leaving no room for
replicated/sharded non-expert weights, the maximum-context cache, or trace and
activation allocations.  All tested mixed BFP2/BFP4 and residual-compression
candidates that fit missed the decoder correctness threshold.  See
`mesh_plan.md` and `AUTOFIX.md` for exact arithmetic and rejected alternatives.
Even the no-padding mixed-precision bound permits only 53.081868% BFP4; the
available compressed kernel's 640-to-768 bank padding lowers the practical
limit to 32.131060%, requiring at least 67.868940% BFP2/zero.
This limitation prevents a clean stage-review verdict and prevents advertising
this layer as the full-model stack baseline.
The independent rereview is retained in `STAGE_REVIEW.md`; its final verdict is
`more-work-needed` solely for this physical capacity constraint.

## Delivered mesh path

- P300 dies 0 and 1, mesh `1x2`, `FABRIC_1D`, linear topology, one link.
- Replicated public BF16 hidden state `[1,1,logical_rows,10240]`.
- Replicated hyperconnection, PLE, router, indexer, and full 16-key/48-value
  head GDN.
- QSA: 12 query heads and one main KV head per die; replicated indexer.
- MoE: local intermediate 320, all 512 experts retained, exactly top-10
  gate-selected execution per logical token.
- One BF16 all-reduce after MoE; QSA layers have a second one after attention.
- BFP8 local QSA K/V and replicated raw-index caches; BF16 compressed-index
  cache.  Maximum-context allocation is 2.1796875 GiB/die across 12 QSA layers.
- Setup-only rank patching; no host conversion in prefill, decode, CCL, or
  trace replay.

`mesh_plan.md` records every global/per-die weight, activation, cache, padding,
collective, and expert shape, plus alternatives evaluated before and during
implementation.

## Correctness and contract evidence

All PCC values below compare the multichip layer directly with the single-chip
TTNN optimized baseline using real checkpoint weights.  The hardware suite ran
with trace-allocation tracking and passed 28 non-long tests.  The separate
maximum-context trace test passed, and `static_contracts.xml` records 19
CPU/static checks including the added 262,143/262,144 chunk-plan cases.

| Representative layer | Kind | Prefill PCC, seq 33 | Decode PCC | Warm vs replay PCC |
| --- | --- | ---: | ---: | ---: |
| 0 | GDN | 0.99892092 | 0.99999958 | 1.00000000 |
| 1 | PLE + GDN | 0.99943250 | 0.99999976 | 1.00000048 |
| 3 | QSA | 0.99885875 | 0.99932384 | 0.99999964 |

QSA BFP8-cache PCC against the BF16 optimized control was
0.99997681/0.99997419/0.99997586 for prefill K/V/index and
0.99997675/0.99997461/0.99997586 after decode.  Both ranks return bit-identical
public output.  Coverage also includes:

- shuffled QSA page tables and reconstructed local KV-head caches;
- GDN recurrent and convolution state transfer;
- distinct batch-32 QSA page tables and positions 33 through 64;
- direct stacked layout `layer0 -> layer1 -> layer3`;
- five deterministic trace replays for every meaningful layer kind;
- maximum context 262,144, position 262,143, page 4095, and cache shapes
  `[4096,1,64,256]` / `[4096,1,64,128]` under trace;
- logical non-aligned prefill chunk plans through 262,143 without changing the
  public length;
- source and runtime guards against Torch/host fallback.

Primary artifacts are `final_correctness.xml`,
`advertised_context_trace.xml`, and `final_watcher.xml`.

## Warmed latency

Medians are seven independent real-checkpoint samples.  Prefill uses logical
sequence 128; decode is the mean of 100 warmed trace replays.  Efficiency is
`speedup / 2` for the two-die target.

| Layer | Single prefill ms | Multi prefill ms | Speedup | Efficiency | Single decode ms | Multi decode ms | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 GDN | 22.461979 | 19.057457 | 1.178645x | 58.932% | 1.090756 | 1.099877 | 0.991707x | 49.585% |
| 1 PLE+GDN | 25.621410 | 22.127126 | 1.157919x | 57.896% | 1.455516 | 1.478821 | 0.984241x | 49.212% |
| 3 QSA | 48.602010 | 32.853951 | 1.479335x | 73.967% | 2.867152 | 2.880215 | 0.995465x | 49.773% |

Decode does not speed up: token-sized TP compute savings are offset by one or
two fixed fabric reductions.  This is reported as a measured limitation, not
hidden by an untraced or host-timed path.  The raw benchmark artifacts are
`singlechip_perf_count7.xml` and `multichip_perf_count7.xml`.

## Profiler findings

Profiler and watcher runs were separate.  Direct-decode profiling executes the
same graph without trace capture solely to avoid Tracy's multi-device trace-ID
correlation limitation; reported decode latency remains real trace replay.
Each layer was captured in its own process with explicit device-profiler
checkpoints so no markers were dropped.

| Layer/window | Compute | Tensor manipulation | Data movement | Other | Modeled DRAM roofline |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 prefill | 74.54% | 22.92% | 0.40% | 2.14% | 86 GB/s (16.8%) |
| L0 decode | 58.05% | 29.74% | 4.11% | 8.10% | 86 GB/s (16.8%) |
| L1 prefill | 68.23% | 29.61% | 0.33% | 1.82% | 92 GB/s (18.0%) |
| L1 decode | 60.28% | 29.43% | 4.21% | 6.06% | 89 GB/s (17.4%) |
| L3 prefill | 76.74% | 18.09% | 0.93% | 4.23% | 55 GB/s (10.7%) |
| L3 decode | 33.46% | 17.55% | 1.95% | 47.05% | 34 GB/s (6.6%) |

GDN prefill is dominated by two active sparse-MoE projections (47.45% of
device time in L0 and 42.31% in L1).  QSA prefill is dominated by SDPA (27.44%)
and sparse MoE (26.43%).  QSA decode is dominated by index/cache gather work
(40.57%); sparse MoE is only 3.09%.  Fabric rows total 8.59/9.16 us per GDN
decode and 18.42 us per QSA decode.  These results reject further decode TP on
the fixed two-die boundary unless a future graph fuses or removes collectives
and cache/index gathers.

Human tables, detailed CSVs, summary CSVs/PNGs, and raw provenance CSVs live
under `tracy/layer0_gdn`, `tracy/layer1_ple_gdn`, and `tracy/layer3_qsa`.

## Reproduction

Every hardware command used the repository-local runtime:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

Correctness and maximum context:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole \
  -m 'not long_context' \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py

TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole --long-context \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_qsa_trace_at_advertised_context

pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py \
  --collect-only
```

Latency:

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 pytest -q --tt-arch blackhole --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py
```

Profiler attribution uses `QWEN38_MC_PROFILE_DIRECT_DECODE=1`, one
`QWEN38_MC_PERF_LAYERS` value per process, `python -m tracy -r`, and
`tt-perf-report` windows `MC_PERF_PREFILL_Lx..._END` and
`MC_PERF_DECODE_Lx..._END`.  The exact commands and hashes are retained in
`work_log.md` and `evidence_manifest.sha256`.
