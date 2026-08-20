# Multichip decoder work log

## 2026-08-20: inventory and topology selection

- Baseline commit: `c7000d2596b` (`Record Qwen optimized decoder checkpoint`).
- Baseline implementation: `tt/optimized_decoder.py`; representative layer
  kinds are linear attention (layer 0) and full attention (layer 3).
- Unrelated dirty dependency entries present at start and excluded from stage
  work: `tt_metal/third_party/tracy`, `tt_metal/third_party/umd`, and
  `tt_metal/third_party/tt-cluster-descriptors/`.
- `timeout 60 tt-smi -ls --local`: four Blackhole P300c devices visible.
- A first mesh smoke run from the checkout mixed the installed runtime with
  checkout dispatch sources and failed compilation on `init_telemetry`.  It
  closed all devices; a following bounded `tt-smi` list was healthy.  Repeating
  from `/tmp` with the installed runtime root passed.  This is environment
  isolation evidence, not a model or hardware failure.
- Isolated mesh smoke: `MeshShape([1, 4])`, four devices, per-device compute
  grid `11 x 10`, passed open and close.
- Fabric smoke: `FABRIC_1D_RING`, `Topology.Ring`, one-link BF16 all-gather of
  global `[1,1,32,128]` sharded on width passed bitwise on all four replicas.
- Selected target: one 1x4 Blackhole ring, TP=4.  The complete tensor/state,
  per-device shape, padding, collective, context, and rejected-alternative plan
  is in `README.md` and was recorded before final-path coding.

Hardware commands run from `/tmp` with:

```bash
TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal
```

All hardware-facing commands are serialized.  Watcher and profiler runs will
remain separate.

## 2026-08-20: implementation and correctness

- Added `tt/multichip_decoder.py`, inheriting `OptimizedDecoder`.  Q/K/V,
  attention output, linear-attention projections/state, and MLP tensors use
  TP-local head/feature ownership.  The residual stays replicated across the
  1x4 ring and is reduced at attention/output and MLP/down boundaries.
- Initial full decode exposed a Blackhole L1/static-CB overlap.  `$autofix`
  produced `autofix/full_decode_l1/AUTODEBUG.md` and `AUTOFIX.md`.  A gate DRAM
  spill verified the cause but moved the clash to down; every legal smaller
  `in0_block_w` was refuted with identical addresses.  Per-device full-MLP
  padding 4,352 -> 4,608 enabled 16 cores and passed (prefill 0.998384, decode
  0.999116, two consecutive decode calls).  Linear MLP stays 4,352 / 8 cores.
- `tests/test_multichip_decoder.py` covers local tensor/cache/state contracts,
  both real layer kinds, lengths 33/65, paged trace, forced chunking, batch 32,
  runtime fallback audit, performance, and advertised context.
- Main correctness log: `correctness/pytest_acceptance.log` (four core tests
  passed; its two initial batch failures were a test-composer 4x replicated
  output issue).  The corrected batch-only rerun is
  `correctness/pytest_batch32.log`: 2 passed.
- Native-context evidence: `context/full_native_decode.log`,
  `context/full_context_prefill.log`, and `context/linear_native_context.log`.
  Full 32,769/262,144 prefills pass; full last-position trace PCC is 0.999836
  with repeat 1.0; linear 262,144 prefill and final traced decode pass in
  709.79 s.  Advertised capability is unchanged.

## 2026-08-20: performance and profiler

Commands used the installed runtime from `/tmp`; performance set
`QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=20`.

Initial compatibility-path benchmark (superseded by the final real-weight
topology-selection measurement below):

| Layer | Single-chip optimized | Initial TP=4 | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Linear | 1456.214 us | 776.948 us | 1.874x | 46.9% |
| Full | 1133.241 us | 558.465 us | 2.029x | 50.7% |

Raw logs are `perf_singlechip.log` and `perf_multichip.log`.  Tracy used the
profiler runtime and one traced replay per representative layer.  The first
linear post-process hit a pandas inferred-string regression; raw capture was
preserved and successfully reprocessed with `PANDAS_FUTURE_INFER_STRING=0`.
Both `profiler/{linear,full}_decode/` directories contain `decode_ops.csv`,
advice-enabled `report.csv`, human-readable `report.txt`, summary CSV/PNG, and
generation provenance.  Full: 479 us kernels + 88 us gaps.  Linear: 588 us
kernels; the raw trace includes a host gap outside meaningful device work.
Reports show ring all-reduce lowering to reduce-scatter plus all-gather and the
selected BFP4/BFP8 policies on material rows.

## 2026-08-20: watcher

- Initial watcher run failed before model execution because ACTIVE_ETH watcher
  instrumentation was 27,968 bytes for a 25,600-byte config buffer, followed by
  a mesh-reopen segfault.  Devices remained visible and healthy.
- Prescribed retry used `TT_METAL_WATCHER_DISABLE_ETH=1`.  Four representative
  tests passed in 120.62 s.  `watcher/no_eth/.../watcher.log` has 2,247 lines,
  no Watcher/NoC/assert/hang/timeout match, and normal detach for devices 0--3.
- After adding the fractured stack graph, a focused watcher rerun of both real
  layer kinds passed in 23.95 s.  `watcher/fractured_no_eth/watcher.log` has
  10,304 lines, no error/assert/NoC/hang/timeout match, and clean detach on all
  four devices.  Ethernet instrumentation remains disabled for the same hard
  ACTIVE_ETH config-buffer limit; CCL correctness/stress is covered by the
  topology probe, all-replica gates, traces, and consolidated acceptance run.

## 2026-08-20: review remediation and artifact ledger

The first independent stage review returned `more-work-needed`.  Remediation
added direct optimized-baseline PCC, physical all-replica boundary assertions,
a shape-faithful topology comparison, a fractured stacked-decode candidate,
and Blackhole-correct profiler metadata.  The first stacked API run failed in
the MLP DRAM-sharded matmul with `bad optional access`; the isolated cause was
the gathered normalized tensor remaining DRAM-interleaved.  Converting it to
the optimized decoder's 8/16-core L1 norm contract fixed both layer kinds.

All commands below ran from `/tmp` with the three environment variables shown
at the top of this log and exited zero unless a failure is explicitly stated.
`TEST` below is the exact absolute path
`/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py`;
`OPT_TEST` replaces the basename with `test_optimized_decoder.py`.

| Artifact | Exact command (after common environment) | Result |
|---|---|---|
| `correctness/direct_optimized/baseline.log` | `QWEN36_DIRECT_BASELINE_MODE=write QWEN36_DIRECT_MESH_SIZE=1 QWEN36_DIRECT_BASELINE_DIR=/tmp/qwen36_direct pytest -q -s $TEST::test_direct_optimized_baseline_and_all_replicas` | 3 passed |
| `correctness/direct_optimized/tp4_compare.log` | `QWEN36_DIRECT_BASELINE_MODE=compare QWEN36_DIRECT_MESH_SIZE=4 QWEN36_DIRECT_BASELINE_DIR=/tmp/qwen36_direct pytest -q -s $TEST::test_direct_optimized_baseline_and_all_replicas` | 3 passed |
| `topology_probe.log` | `QWEN36_RUN_TOPOLOGY_PROBE=1 pytest -q -s $TEST::test_replicated_vs_fractured_stack_boundary` | 2 passed |
| `fused_ccl_probe.log` | `QWEN36_RUN_FUSED_CCL_PROBE=1 pytest -q -s $TEST::test_installed_fused_ccl_contracts` | fused RS PCC >=0.9999869 and fused AG-matmul PCC 0.9999915 on all devices; 1 passed |
| `correctness/fractured_stacked_decode.log` | `pytest -q -s $TEST::test_fractured_stacked_decode_trace` | first run: 2 failed as documented; fixed rerun: 2 passed |
| `perf_singlechip.log` | `QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=20 pytest -q -s $OPT_TEST::test_optimized_decoder_perf` | 2 passed |
| `perf_multichip.log` | `QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=20 pytest -q -s $TEST::test_multichip_decoder_perf` | 2 passed |
| watcher control | `TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 pytest -q -s $TEST -k 'local_tensor_and_state_contracts or linear_non_aligned or full_non_aligned or full_paged_decode_trace'` | 4 passed |
| fractured watcher | `TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DUMP_ALL=1 pytest -q -s $TEST::test_fractured_stacked_decode_trace` | 2 passed |
| linear profiler capture | `QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1 tracy -r -p pytest -q -s $TEST::test_multichip_decoder_perf -k linear_attention` | passed |
| full profiler capture | `QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1 tracy -r -p pytest -q -s $TEST::test_multichip_decoder_perf -k full_attention` | passed |
| corrected full report | `PANDAS_FUTURE_INFER_STRING=0 tt-perf-report profiler/full_decode/decode_ops_blackhole.csv --start-signpost FULL_ATTENTION_DECODE_TRACE_START --end-signpost FULL_ATTENTION_DECODE_TRACE_END --csv profiler/full_decode/report_blackhole.csv --summary-file profiler/full_decode/report_blackhole_summary --no-color` | Blackhole, 110 workers |
| corrected linear report | `PANDAS_FUTURE_INFER_STRING=0 tt-perf-report profiler/linear_decode/decode_ops_blackhole.csv --start-signpost LINEAR_ATTENTION_DECODE_TRACE_START --end-signpost LINEAR_ATTENTION_DECODE_TRACE_END --csv profiler/linear_decode/report_blackhole.csv --summary-file profiler/linear_decode/report_blackhole_summary --no-color` | Blackhole, 110 workers |

The profiler runtime/build provenance is TT-Metal runtime root
`/home/ttuser/.local/lib/model-bringup/tt-metal`, Blackhole firmware bundle
19.11.0, KMD 2.10.0, 1.35 GHz device clock, four P300c devices, and the repo
source baseline `c7000d2596b`.  Corrected CSVs differ from raw capture only by
filling the previously blank `DEVICE ARCH` with `blackhole` and
`AVAILABLE WORKER CORE COUNT` with `110`.

The fused AG-matmul preflight first refuted one-link/four-worker geometry (the
8-core sender axis requires eight workers total), then refuted one-link/eight
workers because the fabric-mux memory map exceeded Blackhole L1.  The legal
two-link/four-worker contract passed and is retained in `fused_ccl_probe.log`;
the final test encodes only that passing geometry.

## 2026-08-20: second-review autofix and final topology selection

The second review required a real-layer A/B for the fractured graph and an
exhaustive linear BFP4 geometry disposition. Both findings were worked through
the `$autofix` isolation loop.

- `performance/fractured_selected/real_weight_topology_selection.log` runs
  independent real recurrent state/KV cache, excludes the one-time fracture and
  validation gather from both traces, checks all four output devices, and uses
  50 warmed replays. Linear is 772.178 us replicated versus 871.299 us
  fractured; full is 551.352 us versus 652.115 us. PCC is 0.999917 on every
  linear device and 1.0 on every full device. Replicated is the production
  layer-stack boundary; the fractured method remains a correctness-valid
  rejected alternative.
- `autofix/bfp4_geometry/sweep.log` covers every exact-shape block geometry at
  BFP4/LoFi. Gate/up block 10 wins; down block 17 wins. The only useful padded
  alternative, 4,352 -> 4,608 over 16 workers, passed real-layer PCC but its
  Blackhole profile regressed the rows from about 43 us to 45.7--46.3 us.
  `autofix/bfp4_geometry/AUTOFIX.md` records the full divisor/core/padding
  proof. Production uses exact 4,352 over 8 workers with blocks 10/10/17.
- The selected exact graph was freshly captured with the Tracy-enabled runtime.
  `profiler/linear_decode/selected_bw10_ops_blackhole.csv`,
  `selected_bw10_report_blackhole.csv`, and
  `selected_bw10_report_blackhole.txt` are Blackhole/110-worker raw, CSV, and
  human-readable artifacts. The 4,608/16-worker rejection is retained as
  `selected_padded16_*`. The final report still labels the exact BFP4 rows
  `SLOW`, but the exhaustive sweep proves there is no faster legal block/core
  or useful padding candidate; this is an advice-model limitation.
- `autofix/fused_decoder/AUTOFIX.md` records the exact decoder-shape audit.
  Fused RS/AGMM dimensions are algebraically legal, but the installed kernels
  require MinimalMatmul programs, dedicated CCL cores, 2/3 global semaphores,
  and persistent buffers not owned by the selected DRAM-sharded graph. The
  fused residual add also rejects BF16 residual with BFP4/BFP8 weights. This is
  a graph-fusing implementation boundary, not a drop-in multichip candidate.

Additional exact commands after the common environment:

| Artifact | Command | Result |
|---|---|---|
| topology selection | `QWEN36_RUN_STACKED_PERF=1 QWEN36_STACKED_PERF_REPLAYS=50 pytest -q -s $TEST::test_real_weight_replicated_vs_fractured_decode_perf` | 2 passed; replicated selected |
| BFP4 geometry | `QWEN36_RUN_BFP4_GEOMETRY_SWEEP=1 pytest -q -s $TEST::test_linear_attention_bfp4_mlp_geometry_sweep` | 1 passed; all 4 replicas/candidate |
| selected linear profiler | `TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal TT_METAL_PROFILER_DIR=/tmp/qwen36_linear_profile QWEN36_PERF_PHASE=decode QWEN36_DECODE_REPLAYS=1 python -m tracy -r -p -m pytest -q -s $TEST::test_multichip_decoder_perf -k linear_attention` | 1 passed; Blackhole profiler capture |
| selected report | `PANDAS_FUTURE_INFER_STRING=0 tt-perf-report $DOC/profiler/linear_decode/selected_bw10_ops_blackhole.csv --start-signpost LINEAR_ATTENTION_DECODE_TRACE_START --end-signpost LINEAR_ATTENTION_DECODE_TRACE_END --csv $DOC/profiler/linear_decode/selected_bw10_report_blackhole.csv --summary-file $DOC/profiler/linear_decode/selected_bw10_report_blackhole_summary --no-color` | Blackhole, 110 workers, 22.0% modeled DRAM roofline |
| final acceptance | `pytest -q -s $TEST -k 'runtime_has_no_host or local_tensor_and_state or linear_non_aligned or full_non_aligned or full_paged_decode_trace or full_forced_chunked or real_weight_batch32 or fractured_stacked_decode_trace'` | 10 passed; `correctness/pytest_final_acceptance.log` |
| final selected watcher | `TT_METAL_WATCHER=1 TT_METAL_WATCHER_DISABLE_ETH=1 TT_METAL_WATCHER_DUMP_ALL=1 pytest -q -s $TEST -k 'linear_non_aligned or full_paged_decode_trace'` | 2 passed; clean detach on all devices; `watcher/final_selected_no_eth/{pytest,watcher}.log` |

`DOC` is
`/home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder`.

## Local checkpoints

- Stage implementation, tests, documentation, and evidence:
  `65ac60e460fa3c9db18c58c879af1fc3b39f1843`.
- Independent post-remediation `$stage-review`: `clean-pass` before the
  implementation checkpoint. No commit was pushed.
