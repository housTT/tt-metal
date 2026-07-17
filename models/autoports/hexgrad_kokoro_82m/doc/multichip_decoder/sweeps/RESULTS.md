# Multichip decoder — measurement sweeps / rejected alternatives

All numbers are warmed device measurements on the 4× p300c `(1,4)` mesh (or single
device for the FF1 matmul sweep). Reproduce with the dev recipe
(`TT_METAL_HOME`+`PYTHONPATH` to the dev checkout — see `tt-metal-dev-env` note);
the standalone probe scripts here must resolve the **dev** `ttnn` (matching
`TT_METAL_HOME`), else the dev kernel source is JIT-compiled against installed
headers and fails to build.

## Topology: Ring vs Linear — `probe_topology.py` → `topology.log` (FRESH, final config)

| topology | fabric | decode T=512 | decode T=128 |
|---|---|---|---|
| Linear | FABRIC_1D | 2.71 ms | 1.93 ms |
| **Ring** (selected) | FABRIC_1D_RING | **2.56 ms** | 1.95 ms |

Ring uses all 4 physical ring links; adopted for the max-context win. (An earlier
pre-FF1-config probe read Linear 2.84 / Ring 2.68; superseded.)

## FF1 (up) matmul geometry — `probe_ffn_in_matmul.py` → `ffn_in_matmul.log` (FRESH)

FF1 = `[M,768]×[768,2048]` fused-gelu on the local sequence shard. Default
`core_grid` heuristic picked `in0_block_w=1` (SLOW, ~10% DRAM). Sweep (wall-clock,
single device; device kernel ≈ wall for this op):

| M | core_grid default | g=8×(m_tiles) in0_block_w=8 (selected) | ibw=12 | ibw=24 |
|---|---|---|---|---|
| 128 | 43.0 µs | **33.2 µs** | 33.7 µs | 35.5 µs |
| 64 | 34.3 µs | **32.5 µs** | 32.8 µs | 34.6 µs |
| 32 | 30.5 µs | **28.9 µs** | 29.4 µs | 31.5 µs |

`in0_block_w=8` fastest among legal K-divisors (K=24 tiles); adopted via
`_ffn_in_program_config`. **Corroborated by the real profiled decode**
(`../tracy/decode_perf_report.csv`, row `128 x 768 x 2048`): 32.6 µs, `in0_block_w=8`,
subblock `1x2`, profiler advice "in0_block_w=8 ... look good 🤷" — i.e. the selected
FF1 geometry is validated on the actual measured path, not just this microbench.

## Rejected: data-parallel + KV all-gather — `probe_dp_kv_gather.py` (earlier design-time run; see provenance)

Replicate ALL weights, seq-shard activations, 1 KV `all_gather`/layer (vs selected
1 AG + 1 RS). Measured traced decode: **T=512 = 4.05 ms, T=128 = 3.42 ms** — much
slower than selected (2.56 / 1.95 ms). Cause: the `[b, 2·heads, S/TP, 64]` KV
gather has tiny per-head chunks (head_dim 64) and each device runs 12-head SDPA
instead of 3. Confirms head-TP+SP is the better design.

## Collective micro-bench — `probe_collectives.py` (earlier design-time run; see provenance)

Untraced wall-clock, [1,512,768] bf16 on the (1,4) mesh:
`all_reduce` (composite) = **172 µs**; `reduce_scatter` = **67 µs**;
`all_gather` = **67 µs**. → a reduce_scatter/all_gather (sequence-parallel)
residual is cheaper than the composite all_reduce, and shards the norms too;
motivated the sequence-sharded residual contract.

## Sequence-parallel collective correctness — `probe_seq_parallel.py` (earlier run)

Seq-dim (dim=2) `reduce_scatter_minimal_async` then `all_gather_async` via the
`TT_CCL` semaphores: `all_gather(reduce_scatter(x))` == device-sum, PCC 0.99999;
num_links=2. Validated the collective primitives before wiring the decoder.

## Provenance note (precise)

- **Fresh, valid committed logs:** `topology.log` (Ring-vs-Linear, the topology
  decision) and `ffn_in_matmul.log` (FF1 geometry) — both captured in the final
  config. The FF1 decision is *additionally* corroborated by the real profiled
  decode (`../tracy/decode_perf_report.csv`).
- **Earlier design-time runs, not re-captured as committed logs:** the DP-KV-gather
  (4.05 / 3.42 ms), collective micro-bench (all_reduce 172 / RS 67 / AG 67 µs) and
  seq-parallel-correctness (PCC 0.99999) numbers were measured during design when
  the probe processes resolved the **dev** `ttnn`. Their standalone re-capture on
  this host currently fails: an editable-install finder makes `import ttnn`
  resolve the *installed* tt-metal tree (dev kernel source then JIT-compiles against
  installed headers → `init_telemetry` build error; or top-level `ttnn.MeshDevice`
  / `set_fabric_config` are missing on partial lazy init). This is the documented
  `tt-metal-dev-env` host quirk, not a defect in these probes; the scripts here
  carry a dev-`ttnn` preamble and reproduce the numbers when `import ttnn` binds to
  the dev checkout. The crashed re-capture logs were removed rather than committed
  as if they were measurements.
- **None of these earlier-run numbers gate the shipped design:** the selected path
  (head-TP + seq-parallel FFN, Ring) is validated by `topology.log`,
  `ffn_in_matmul.log`, the real profiled decode, and the 32-test PCC suite; the
  DP-KV/collective numbers only *motivate* rejected alternatives.
