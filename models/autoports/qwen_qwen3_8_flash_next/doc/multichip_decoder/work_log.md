# Multichip decoder work log

Stage input commits: optimized implementation `d7baac46495` and optimized
decoder signoff `bed265a9c76`.

## 2026-08-27: environment and target lock

- Confirmed devices 0 and 1 are the two Blackhole dies of one P300 board.
- Confirmed the fixed `1x2` mesh, `FABRIC_1D`, 11x10 compute grid, eight DRAM
  channels, and 31.875 GiB nominal DRAM per die.
- The first unsourced probe mixed an installed TTNN with checkout JIT sources.
  `$autodebug` identified the provenance skew; `AUTODEBUG.md` and `AUTOFIX.md`
  retain the proof and corrected repo-local invocation.
- Locked TP2 QSA/MoE with a replicated 10240-wide public boundary.  The tensor,
  activation, cache, state, CCL, MoE, padding, and capacity plan was recorded in
  `mesh_plan.md` before the final implementation.

Common environment for every retained hardware command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

## Implementation and isolated repair

- Added `tt/multichip_decoder.py`, subclassing `OptimizedDecoder`.
- Added setup-only rank-local checkpoint slicing and rank-1 device-to-device
  patching; no runtime Torch conversion.
- Sharded QSA to 12 query/one KV head per die and MoE to local intermediate
  width 320.  Router/indexer stay replicated; all 512 experts remain available
  and execution stays gate-selected top-10.
- Initial 24-head-per-rank GDN decode missed PCC at about 0.992.  State copies,
  ordering, FP32 projection, and FP32 CCL were refuted.  Replicating full GDN
  restored exact optimized kernel geometry and final decode PCC above 0.999999.
- Rejected an L1 all-reduce output candidate after it produced rank divergence
  in combined correctness/trace coverage.
- Selected BF16 row partials/reductions in DRAM and BFP8 QSA caches.  BFP8 cache
  saves 1.7578125 GiB/die at maximum context and passes direct BF16-control PCC.
- Added structural, real-weight PCC, paged cache, state-copy, non-aligned,
  batch-32, stacked-layout, fallback, maximum-context, and trace determinism
  tests plus the real-checkpoint performance/profiler workload.

## Final correctness and trace gates

Command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole \
  --capture=tee-sys -o junit_logging=all -m 'not long_context' \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/final_correctness.xml
```

Result: `28 passed, 1 deselected` in 209.79 seconds.  A later CPU-only
`static_contracts.xml` run covers 19 static/structural checks, including the
extra 262,143/262,144 chunk-plan cases now present in the source.
Representative hardware evidence:

| Layer | Prefill PCC | Decode PCC | Warm/replay PCC |
| --- | ---: | ---: | ---: |
| 0 GDN | 0.99892092 | 0.99999958 | 1.00000000 |
| 1 PLE+GDN | 0.99943250 | 0.99999976 | 1.00000048 |
| 3 QSA | 0.99885875 | 0.99932384 | 0.99999964 |

Each trace test replayed five times and required bitwise-identical repeated
outputs.  Batch-32 passed for all three layer kinds; the stacked layer-0 ->
layer-1 -> layer-3 contract passed; the runtime fallback guard was active.

Maximum-context command:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 pytest -q --tt-arch blackhole --long-context \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_qsa_trace_at_advertised_context \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/advertised_context_trace.xml
```

Result: one pass in 11.73 seconds at context 262,144/current position 262,143,
with last-page shuffled addressing and trace replay.

## Latency

Single-chip baseline artifact: `singlechip_perf_count7.xml`.
Multichip artifact: `multichip_perf_count7.xml`.

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 pytest -q --tt-arch blackhole --count=7 \
  --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/multichip_perf_count7.xml
```

Result: 21 passes.  Seven-sample medians:

| Layer | Single prefill | Multi prefill | Speedup / efficiency | Single trace decode | Multi trace decode | Speedup / efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 22.461979 ms | 19.057457 ms | 1.178645x / 58.932% | 1.090756 ms | 1.099877 ms | 0.991707x / 49.585% |
| 1 | 25.621410 ms | 22.127126 ms | 1.157919x / 57.896% | 1.455516 ms | 1.478821 ms | 0.984241x / 49.212% |
| 3 | 48.602010 ms | 32.853951 ms | 1.479335x / 73.967% | 2.867152 ms | 2.880215 ms | 0.995465x / 49.773% |

## Tracy and `tt-perf-report`

Watcher was off.  An initial combined traced capture passed its workload but
post-processing could not correlate one device-1 trace ID.  A combined direct
capture then overflowed profiler marker buffers.  The retained solution uses
one direct-decode process per layer and explicit `ttnn.ReadDeviceProfiler`
checkpoints outside signposted windows:

```bash
QWEN38_MC_PROFILE_DIRECT_DECODE=1 \
QWEN38_MC_PERF_DECODE_REPLAYS=1 \
QWEN38_MC_PERF_LAYERS=<0|1|3> \
python -m tracy -r --check-exit-code -o <layer-output> \
  -m pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py
```

All three profiled tests and post-processors passed.  Each raw
`ops_perf_results_*.csv` was analyzed twice:

```bash
tt-perf-report <ops.csv> \
  --start-signpost MC_PERF_<PREFILL|DECODE>_Lx \
  --end-signpost MC_PERF_<PREFILL|DECODE>_Lx_END \
  --no-color --no-host-ops --active-experts <routing-union|10> \
  --csv <detailed.csv> --summary-file <summary.csv>
```

The reports are under `tracy/layer0_gdn`, `tracy/layer1_ple_gdn`, and
`tracy/layer3_qsa`.  Major findings are recorded in `README.md`: sparse MoE and
reshape traffic dominate GDN prefill; SDPA+MoE dominate QSA prefill; cache/index
gathers dominate QSA decode; one/two fixed fabric rows explain flat decode
speedup.  Detailed reports preserve custom-op warnings where the report tool
cannot categorize Qwen GDN/index kernels.

Raw provenance SHA-256:

```text
ce82fe9035b8298fde583534872f672bcc7b652d06ff4dc32858f3638a0a6525  layer0 ops CSV
6e1d82074c780e095ff2bcbaa0dbd8591d134474a5d2e9498a7638c8dab91025  layer1 ops CSV
2a6bda51846b97483031d5ab0b0fb5b0ed6470fb5bd3f2a1ab147c74e6758e3a  layer3 ops CSV
203887c817b31fc392907d288111b3c63d9fe25fa66d7533fdf4212f50d574a6  single-chip count7 XML
0be77da82910d4a4d1c68466ae5557d77aaa8801b5e6c820fdaf48353c50e129  multichip count7 XML
```

## Watcher and recovery

The first all-tests watcher run made fabric ERISC `(29,25)` fail to return to
base firmware at fixture teardown; subsequent hardware setups failed before
model execution.  A board reset (`tt-smi -r 0 1`) restored healthy DRAM,
firmware heartbeat, and a clean bounded 1x2 mesh open/close.  A single-fixture
stacked-layer watcher run passed the workload and all watcher polls, but with
Ethernet watcher instrumentation enabled the same ERISC teardown failed after
pytest had written a passing XML.

The supported retained mode sets `TT_METAL_WATCHER_DISABLE_ETH=1`: Tensix,
dispatch, NoC/CB sanitization, asserts, stack usage, and waypoints remain
watched; fabric ERISC is not instrumented.  Fabric is independently covered by
the correctness, trace, and profiler runs.  Failed recovery artifacts are
`watcher_pre_reset_failure.xml` and
`watcher_eth_enabled_teardown_failure.xml`; they are retained deliberately.
The clean artifact is `final_watcher.xml`, with raw log under `watcher_final`.

## Full-stack capacity and stage outcome

Standard TP2 BFP4 experts require 31.640625 GiB/die.  With max-context caches,
non-expert weights (including replicated GDN), and a 1 GiB reserve, the
optimistic packing limit is only 53.081868% BFP4 tiles.  Uniform/heavy BFP2,
low-error mixed precision, residual quantization, and low-rank residual tests
all missed PCC.  EP2, pipeline placement, smaller context, and host streaming
do not meet the resident full-stack contract.  Exact trials are in
`AUTOFIX.md` and `mesh_plan.md`; raw probe outputs and command-execution
provenance are retained in `capacity_candidate_probes.log`.

The final independent capacity audit also charged the compressed kernel's
640-to-768 gate/up bank padding.  That raises the physical routed-expert count
to 66,846,720 tiles/die and lowers the practical BFP4 ceiling to 32.131060%;
at least 67.868940% of the physical tiles must be BFP2/zero.

The per-layer path is correct and measured, but the full-stack residency gate
is **autofix-failed** on this fixed board.  Therefore this work must not be
marked pipeline-complete, and a clean stage-review cannot be claimed for the
original goal.  A local commit may record this blocked evidence state, but it is
not a completion signoff.  No remote operation or push was performed.

The first independent stage review found the physical blocker plus one
repairable evidence-provenance gap.  `capacity_candidate_probes.log` fixed the
gap and was added to the SHA-256 manifest.  Rereview found no remaining
repairable gap and returned `more-work-needed` solely for the resident-stack
capacity impossibility; `STAGE_REVIEW.md` records that final verdict.

## Local commits

- `07a1a588e27`: blocked multichip implementation, tests, documentation, and
  retained correctness/performance/watcher/capacity evidence.

This SHA records a blocked stage, not a pipeline-complete signoff.  No push or
other remote operation was performed.
