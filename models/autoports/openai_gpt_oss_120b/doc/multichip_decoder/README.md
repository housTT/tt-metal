# GPT-OSS 120B multichip decoder

Status: complete; independent stage review returned `clean-pass`.

This stage adds the P150-family decoder-layer baseline in
`tt/multichip_decoder.py`. P150 `(1,1)` delegates exactly to the completed
`OptimizedDecoder`. P150x2 `(1,2)` and P150x4 `(1,4)` use 1D tensor parallelism
on mesh axis 1, a replicated stack boundary, local-head paged attention, and
gate-selected top-4 sparse experts. It does not add a full model or vLLM path.

## Final implementation

- QKV is column-parallel; each rank owns `64/TP` Q heads and `8/TP` KV heads.
- The paged BFP8 K/V cache is local-head sharded with shape
  `[2048 * max_batch, 8/TP, 64, 64]` per K or V tensor.
- Attention O and expert down projections are row-parallel. Their reduced
  outputs restore the replicated logical `[1,1,S,2880]` stack contract.
- TP4 attention retains its natural zero-padded 2944 physical output through
  the ring reduction, selecting a 736-column-per-rank RS+AG, then slices back
  to 2880. TP2 has no hidden-width padding and uses canonical attention.
- The replicated BF16 router stays L1-resident. Decode executes only the four
  gate-selected experts. Gate/up and down weights are rank-sharded BFP4; TP4's
  intermediate channels are padded from 720 to 736 at load time.
- Expert outputs deliberately reduce at logical width 2880. Runtime padding to
  2944 was 1.64x slower in the warmed topology probe.
- Public sequence lengths remain logical and non-aligned. Batch-1 acceptance
  uses prefill length 127 and decode positions 127, 128, and 129 across page
  and sliding-window boundaries. Batch-2 acceptance uses prefill length 33,
  exact maximum decode position 131071, three randomized high-context logical
  blocks per layer, and two randomized page-table mappings.

The calculated tensor/shard plan, memory accounting, topology selection, and
rejected alternatives are in `mesh_plan.md`.

## Correctness and performance

All PCC values compare the real-checkpoint multichip layer with a separate
single-chip TTNN `OptimizedDecoder` artifact. Decode numbers are medians of
five warmed samples, each containing 100 trace replays. Efficiency is
`single-chip latency / (multichip latency * TP)`; values below 1 mean the
multichip layer does not provide latency scaling.

| Target | Layer | Prefill ms | Prefill PCC | Decode ms | Decode PCC | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 | sliding | 36.580892 | baseline | 0.530752270 | baseline | 1.000000 | 1.000000 |
| P150 | full | 32.245878 | baseline | 0.528672960 | baseline | 1.000000 | 1.000000 |
| P150x2 | sliding | 47.729884 | 0.991984 | 0.752627020 | 0.983523 | 0.705200 | 0.352600 |
| P150x2 | full | 47.502560 | 0.991052 | 0.752160320 | 0.960387 | 0.702873 | 0.351436 |
| P150x4 | sliding | 25.874462 | 0.991819 | 0.652859620 | 0.984313 | 0.812965 | 0.203241 |
| P150x4 | full | 25.605209 | 0.991578 | 0.652924140 | 0.962575 | 0.809700 | 0.202425 |

P150x4 improves the measured length-127 prefill wall time, but neither
multichip target beats P150 for warmed batch-1 decode. The decoder is retained
as the full-stack capacity baseline: 36 layers of decoder weights plus a
batch-1 131072-token cache are physically feasible only on P150x4 under the
selected TT formats. No decode speedup claim is made.

The two refreshed trace outputs per multichip layer passed PCC 0.974073--
0.998842. Reconstructed local-head K/V blocks for randomized page tables
passed PCC 0.999703--0.999996. Every returned rank was bitwise identical and
had the exact logical output shape. `performance_summary.csv` contains the
machine-readable result table.

Stage-review AutoFix added an independent batch-2/high-context gate for both
layer kinds and all three target sizes. It constructs P150 with the same
`DEFAULT_OPTIMIZED_POLICY` BFP8 attention policy used by TP2/TP4, so the
comparison isolates tensor parallelism rather than a capacity-selected BFP4
single-chip policy. The P150 artifact and TP2/TP4 tests all passed:

| Target | Layer | Non-aligned prefill PCC | Decode PCCs, initial/refresh/new page table | Cache PCC min |
| --- | --- | ---: | --- | ---: |
| P150 | sliding | baseline | baseline | baseline |
| P150 | full | baseline | baseline | baseline |
| P150x2 | sliding | 0.992263 | 0.998757 / 0.998116 / 0.998116 | 0.999827 |
| P150x2 | full | 0.989610 | 0.998312 / 0.997288 / 0.962366 | 0.999797 |
| P150x4 | sliding | 0.992361 | 0.982412 / 0.998227 / 0.998227 | 0.999827 |
| P150x4 | full | 0.990435 | 0.997775 / 0.997178 / 0.961553 | 0.999797 |

The first position set includes 131071 and a randomized upper-half context
position for each layer. The second set changes hidden values and positions;
the third keeps them fixed and changes only the page table. Replaying the
initial inputs and the changed-page-table inputs was bitwise deterministic.
The constructor retains logical batch 1--32, while hardware correctness is
qualified through batch 2; batch 3--32 remains capacity-only and is not yet a
full-model correctness claim.

## Topology and profiler evidence

The warmed 200-repeat TP4 boundary probe measured:

| Candidate | Time (ms) | Ratio to current | PCC to current | Decision |
| --- | ---: | ---: | ---: | --- |
| slice/copy/all-reduce logical 2880 | 0.193193 | 1.000 | reference | rejected |
| all-reduce physical 2944, then slice | 0.110467 | 0.572 | 0.999937 | selected for attention |
| reduce-scatter, distributed norm, fused AG+QKV | 0.129149 | 0.668 | 0.999750 | rejected; slower and broader stack contract |
| expert logical 2880 reduction | 0.070270 | 1.000 | reference | selected for experts |
| expert runtime pad/reduce 2944/slice | 0.115074 | 1.638 | >0.999 | rejected |

The final P150x4 decode `tt-perf-report` contains 76 device operations and no
host operations. Its attention communication is a 13 us
`ReduceScatterMinimalDirect` plus a 10 us `AllGather`, followed by the logical
slice. The prior path used a 35 us attention `AllBroadcast` plus pre-CCL data
movement. The unchanged expert collective remains a 34 us `AllBroadcast`.
Final device-op totals are 581 us (sliding) and 582 us (full), with 33 GB/s
reported aggregate DRAM traffic (6.5% of peak). Important remaining kernels
are gate/up sparse matmul 94 us, down sparse matmul 22 us, QKV 44 us, O
projection 15 us, and router projection 53--54 us. Prefill totals are
24.710 ms and 24.690 ms; its 128-expert packed work dominates, while both row
boundaries already use RS+AG. Human tables, sliced CSVs, raw provenance CSVs,
and plots are under `artifacts/20260828_final_v2/tracy*`; raw provenance CSVs
and profile logs are stored losslessly as `.gz`. `artifacts.sha256` covers the
committed evidence set.

## Stress, watcher, and runtime audit

- Both layer kinds on TP2 and TP4 passed warmed trace replay, two refreshed
  inputs/positions, randomized page-table reconstruction, and exact rank
  replication.
- P150, TP2, and TP4 batch-1 trace gates were rerun with
  `TT_METAL_TRACE_ALLOC_TRACKING=1`; no allocation occurred while a captured
  trace was live. Cache inspection now follows `release_trace`, and trace
  refresh uses only preallocated capture inputs. These tracking runs add about
  190 ms of instrumentation overhead and are not latency evidence.
- The batch-2/high-context gate also passed under allocation tracking. A
  follow-up TP4 full-attention run passed under Tensix watcher at positions
  131071/91904 and changed-page-table positions 117841/71839.
- The final changed TP4 path passed three samples of 1,000 trace replays per
  layer under the supported Tensix watcher configuration. The earlier TP2/TP4
  path passed the same stress. Raw watcher logs contain no NoC, CB, assert, or
  device error.
- Full watcher debug does not fit with this model's trace allocation. The
  supported evidence uses `TT_METAL_WATCHER_DISABLE_DISPATCH=1` and
  `TT_METAL_WATCHER_DISABLE_ETH=1`; active-fabric ordinary-dispatch stress was
  also clean. This is Tensix watcher evidence, not a claim that Ethernet
  watcher instrumentation ran.
- Post-run `tt-smi` showed four healthy devices, DRAM status true, and zero
  correctable or uncorrectable GDDR errors.
- `runtime_fallback_audit.md` records the clean source audit. There are no
  host/CPU execution fallbacks in forward or trace replay, and decode retains
  sparse top-4 expert selection.

One fused topology probe left fabric state that caused a reproducible rank-2
PCC divergence in unchanged canonical attention. A bounded all-device warm
reset restored the unchanged baseline, after which the final implementation,
profiler, and watcher runs passed. The pre/reset/post health logs are preserved
under `artifacts/20260828_final_v2/recovery_after_topology_probe/`; topology
experiments must run in an isolated process followed by a device-health check.

## Reproduction

Set `SNAPSHOT` to the checkpoint snapshot ending in
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, and set `ARTIFACT_DIR` to
`doc/multichip_decoder/artifacts/20260828_final_v2`. Hardware commands must be
serialized through the safe wrapper.

```bash
env GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
  GPT_OSS_120B_SNAPSHOT="$SNAPSHOT" \
  GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR="$ARTIFACT_DIR" \
  GPT_OSS_120B_MULTICHIP_RUN_ID=20260828_final_v2 \
  GPT_OSS_120B_MULTICHIP_TRACE_REPEATS=100 \
  GPT_OSS_120B_MULTICHIP_TRACE_SAMPLES=5 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_write_real_weight_optimized_baseline_artifact -q -s

env GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
  GPT_OSS_120B_SNAPSHOT="$SNAPSHOT" \
  GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR="$ARTIFACT_DIR" \
  GPT_OSS_120B_MULTICHIP_RUN_ID=20260828_final_v2 \
  GPT_OSS_120B_MULTICHIP_TRACE_REPEATS=100 \
  GPT_OSS_120B_MULTICHIP_TRACE_SAMPLES=5 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_real_weight_multichip_against_baseline_artifact -q -s -k tp2

# Repeat the preceding target command with: -k tp4

env TT_METAL_TRACE_ALLOC_TRACKING=1 \
  GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
  GPT_OSS_120B_SNAPSHOT="$SNAPSHOT" \
  GPT_OSS_120B_MULTICHIP_ARTIFACT_DIR="$AUTOFIX_ARTIFACT_DIR" \
  GPT_OSS_120B_MULTICHIP_RUN_ID=20260828_batch2_highctx_bfp8_v3 \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_write_real_weight_batch2_high_position_baseline_artifact -q -s

# Run the corresponding test_real_weight_multichip_batch2_high_position_against_baseline_artifact
# target in separate processes with -k tp2 and -k tp4.

env GPT_OSS_120B_MULTICHIP_TOPOLOGY_PROBE=1 \
  GPT_OSS_120B_MULTICHIP_TOPOLOGY_REPEATS=200 \
  scripts/run_safe_pytest.sh \
  'models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_residual_sharded_distributed_norm_fused_qkv_probe[blackhole-p150x4-parent-tp4]' -s
```

For profiling, add `--profile` to an individual target node and generate the
human table from its raw CSV with:

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report \
  --no-summary --signpost PERF_DECODE --csv raw_ops.csv
```

For the final watcher stress, set trace repeats to 1000, samples to 3, and add
`TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_DISPATCH=1
TT_METAL_WATCHER_DISABLE_ETH=1`. Exact expanded commands, remediation history,
and artifact names are recorded in `work_log.md` and in each log header.

## Limitations

- This is a real multichip decoder-layer baseline, not evidence that a full
  36-layer model, tokenizer/generator, or serving path has been assembled.
- P150 and P150x2 cannot hold the estimated 36-layer weights plus a batch-1
  full-context cache in 32 GiB per device. Their decoder APIs retain 131072 for
  compatibility and layer testing; P150x4 is the feasible resident-stack
  target. The future full-model stage must measure all non-decoder allocations.
- Decode batch values greater than one use a static per-user active-expert loop
  because the reusable sparse kernel's decode representation is batch one.
- Batch 2 is qualified for both layer kinds, non-aligned prefill, exact-limit
  decode, trace refresh, page-table remapping, cache reconstruction, and rank
  replication. The configured batch 3--32 range is not yet hardware-qualified.
- The TP4 attention decode specialization mirrors the canonical attention
  prefix because that implementation has no collective-tail hook. Its
  correctness and trace tests guard this model-local fork against drift.
