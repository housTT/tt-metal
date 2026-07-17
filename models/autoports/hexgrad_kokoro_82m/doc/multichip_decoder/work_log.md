# Kokoro-82M — Multichip Decoder work log (stage 03)

Branch `agentic-research/hous/kokoro-82m-p150`. Runtime env: `TT_METAL_HOME` +
`PYTHONPATH` to the dev checkout (see project memory `tt-metal-dev-env`).

## 0. Baseline + hardware

- Single-chip baseline = stage-02 `tt/optimized_decoder.py` (`OptimizedDecoder`,
  2.95 ms traced decode @T=512, PCC ≥ 0.995). `PrecisionPolicy` + `_dtype` /
  `_fidelity` / `_round_up` / `TILE` are imported from it — no numeric drift.
- Hardware: `tt-smi -ls` → 4× Blackhole p300c. `SystemMeshDescriptor().shape()` =
  `(2,2)`, `cluster.get_cluster_type()` = `P300_X2`.

## 1. Feasibility (mesh + CCL)

- Opened `(1,4)` submesh with `FABRIC_1D`; `all_gather` OK; `all_reduce` needs
  `cluster_axis` kwarg (Sum-only). Fabric adjacency log: intra-mesh degree {2:4}
  → physical **4-ring**. `default_topology` returns Linear for non-T3K 4-chip.
- Collective micro-bench (untraced, `/tmp/ccl_bench`): `all_reduce[1,512,768]`
  172 µs (composite, slow); `reduce_scatter` 67 µs; `all_gather` 67 µs → favour a
  reduce_scatter/all_gather (sequence-parallel) residual over composite all_reduce.
- SP probe (`/tmp/sp_probe`): seq-dim (dim=2) `reduce_scatter_minimal_async` +
  `all_gather_async` via `models/common/modules/tt_ccl.TT_CCL` semaphores →
  AG(RS(x)) == device-sum, PCC 0.99999; num_links=2.

## 2. Strategy (see README table)

Head-parallel TP attention (fracture qkv/dense, 3 heads/device) + sequence-parallel
FFN (replicate FFN weights) + sequence-sharded residual `[b,1,S/4,H]`. Only the
attention block needs the full sequence → 1 `all_gather` (residual) + 1
`reduce_scatter` (WO) per layer; everything else local on the 1/TP shard. QKV
column-reorder to head-group-interleave so a contiguous shard gives each device its
local Q/K/V. WO bias added once after RS. Seq padded to mult. of TP·TILE=128.

## 3. Implementation

`tt/multichip_decoder.py::MultichipDecoder` (same public API as OptimizedDecoder;
sequence-sharded I/O). Uses `TT_CCL` + `all_gather_async` / `reduce_scatter_minimal_async`
(trace-safe persistent semaphores). Topology auto-selected from `ttnn.get_fabric_config()`
(Ring when `FABRIC_1D_RING`).

First-run correctness (`/tmp/mc_smoke*`): prefill/decode PCC vs HF 0.9975–0.9993,
vs single-chip TTNN 0.9989–0.9997, across aligned/non-aligned/batched/masked.

## 4. Tests — `tests/test_multichip_decoder.py` (32 passed)

`test_uses_multichip_tp_path` (attention fractured, FFN replicated, per-device
shapes, precision preserved), `test_only_two_collectives_per_layer` (guards 1 AG +
1 RS via source inspection), prefill/decode PCC vs single-chip (≥0.998) + vs HF
(≥0.995) over T∈{8,31,32,96,128,256,500,511,512}, batch {2,4,8,32}, IPA, masked
variable-length, `test_component_pcc_vs_single_chip` (embedding + one AlbertLayer),
determinism, stateless, repeated-replay stress.

```
python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_multichip_decoder.py -q  # 27 passed
```

## 5. Perf / topology / geometry

- Topology (`sweeps/probe_topology.py` → `sweeps/topology.log`, final FF1 config):
  Linear **2.71 ms** vs **Ring 2.56 ms** @T=512 (Ring uses all 4 ring links);
  @T=128 ~tied (Ring 1.95 / Linear 1.93) → adopted `FABRIC_1D_RING` +
  `Topology.Ring` for the max-context win. (An earlier pre-FF1 probe read Linear
  2.84 / Ring 2.68; superseded by the final-config numbers above.)
- Rejected DP-KV-gather (`sweeps/probe_dp_kv_gather.py` → `sweeps/dp_kv_gather.log`,
  replicate all weights, 1 KV all_gather): **4.05 ms** @512 — slower ([b,24,S/4,64]
  gather = tiny per-head chunks; 12-head SDPA/device). Confirms head-TP+SP is
  better. Collective micro-bench in `sweeps/collectives.log` (all_reduce 172 µs vs
  RS 67 + AG 67 µs).
- FF1 (`128×768×2048`) was the dominant SLOW matmul (in0_block_w=1, 10 % DRAM).
  Sweep (`sweeps/probe_ffn_in_matmul.py` → `sweeps/ffn_in_matmul.log`) over
  grid/in0_block_w (8/12/24; 8 fastest) → explicit
  `MatmulMultiCoreReuseMultiCast` g=8×(m_tiles) in0_block_w=8 fused-gelu: 43→33 µs,
  util 10→14 %; Matmul total 1927→1680 µs. Added `_ffn_in_program_config`.
- Final latency (`gen_evidence.py`, warmed, 50 iters): traced decode T=512
  single 2.95 / **multi 2.56 ms = 1.15× (29 % eff)**; T=128 2.46 / **1.95 ms =
  1.26× (31 %)**.
- `tt-perf-report` (traced decode T=512, `tracy/decode_perf_report.txt`): CCL 24 %
  (RS 14 % + AG 10 %), LayerNorm 19 %, Matmul 33 %, SDPA 13 %; DRAM roofline 6.9 %.
  Modest speedup is inherent (tiny launch/movement-bound encoder → sub-linear op
  scaling + fixed collective overhead); alternatives measured slower. Full analysis
  in README.
- Watcher (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`): 7 tests passed,
  `watcher/watcher.log` 0 fault/assert/tripped/overflow markers, min stack 1348 B
  free. ETH excluded — fabric ETH kernel + watcher overflows the ACTIVE_ETH config
  buffer (27920 > 25600); infra limit, not model code.
- Fallback audit (`fallback_audit.txt`): forward/trace path pure TTNN.

## 6. Context contract

`../context_contract.json → multichip_decoder`: mesh (1,4), TP=4, per-device weight
≈4.3 MB (attn fractured 1/4, FFN replicated), KV cache 0 (N/A). Advertised =
supported = 512, `capability_reduction=false` (weights fit with huge margin; TP is
a latency play, not memory-forced).

## 7. Stage review + remediation

- `$stage-review` (fresh xhigh subagent): **more-work-needed** → 3 P2 + concerns,
  all fixable without design changes. Remediation:
  - P2-1 (topology numbers contradictory / not like-for-like): re-measured Linear
    vs Ring under the **final** FF1 config (`sweeps/topology.log`): Linear 2.71 /
    Ring 2.56 ms @512. Fixed README, work_log, and the code comment to one sourced
    value; labeled the earlier pre-FF1 probe numbers as superseded.
  - P2-2 (gated coverage regressed at tight lengths): added T=16 and T=33 to
    `test_prefill_pcc_vs_single_chip` and `test_decode_traced_pcc` (32 tests); set
    `PCC_BAR_SC=0.997` (documented short-seq bf16 floor; worst 0.99803 @T=16) and
    added a hard assertion on worst vs-single-chip / vs-HF PCC in `gen_evidence.py`.
  - P2-3 (sweep evidence not preserved): probe scripts preserved as
    `sweeps/probe_*.py`; `sweeps/RESULTS.md` consolidates all measured numbers with
    provenance; `sweeps/topology.log` + `sweeps/ffn_in_matmul.log` captured fresh.
  - Concerns: removed the stale failed `tracy/decode_perf_report.console.log`;
    captured the ETH+watcher config-buffer overflow to
    `watcher/eth_watcher_overflow.log`; documented the profiler "input0 in L1"
    advice disposition in the README (launch-bound, not DRAM-bound; not adopted).
- Environment note (infra, not model code): mid-stage, a standalone probe
  core-dumped and I cleared a stale 23 GB JIT cache; this exposed a host quirk
  where an editable-install finder (`__editable__.ttnn` + `ttnn-custom.pth`) can
  make `import ttnn` resolve to a *second* installed tt-metal tree, so dev kernel
  source JIT-compiles against installed headers → `init_telemetry` build error.
  Recovery: warmed the dev-tree JIT cache with the dev `ttnn` (mesh smoke OK,
  devices healthy via `tt-smi -ls`); the documented dev recipe
  (`TT_METAL_HOME`+`PYTHONPATH` to the dev checkout) resolves dev `ttnn` under
  pytest and the full suite passes. See project `tt-metal-dev-env` note.

## 8. Commit

- `$stage-review` final re-review verdict: **clean-pass** (fresh xhigh subagent;
  two remediation rounds; verified topology.log, ffn_in_matmul.log +
  tracy/decode_perf_report.csv corroboration, T=16/T=33 gates, ETH-overflow log,
  RESULTS.md provenance). No remaining required work.
- tt-metal branch `agentic-research/hous/kokoro-82m-p150`.
- Stage commit SHA `8c4d5efe7eb60c43f3a5cae701ab5a418eee853e` (stage-owned files
  only: tt/multichip_decoder.py, tests/test_multichip_decoder.py,
  tests/test_perf_multichip.py, doc/multichip_decoder/, doc/context_contract.json;
  pre-existing unrelated .agents/* excluded). Not pushed.
- Follow-up commit records this SHA + a stale-docstring fix (Ring, not Linear).
