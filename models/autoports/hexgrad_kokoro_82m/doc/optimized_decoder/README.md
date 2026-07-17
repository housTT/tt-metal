# Kokoro-82M — Optimized Decoder (TTNN)

Stage 02 (optimized-decoder) for `hexgrad/Kokoro-82M` on Tenstorrent Blackhole
(p300c, single 1×1 device). Implements
`models/autoports/hexgrad_kokoro_82m/tt/optimized_decoder.py`, the optimized
counterpart of the stage-01 `tt/functional_decoder.py`.

The target is the same component as stage 01: Kokoro's only attention-transformer,
`plbert` = HF `AlbertModel` (12 weight-tied `AlbertLayer`s; hidden 768, 12 heads,
head_dim 64, intermediate 2048, embedding 128→768, vocab 178, max ctx 512). Kokoro
is **non-autoregressive / stateless** (no causal decoder, no KV cache); see the
functional-decoder README for the full architecture/contract rationale.

## Result summary

Warmed, real weights, single p300c device. Baseline = stage-01 functional decoder
(fp32 activations, bf16 weights, HiFi4, DRAM, separate QKV, hand-built attention).

| Metric (T=512) | Functional (baseline) | Optimized | Speedup |
|---|---|---|---|
| Warmed prefill (eager) | 21.01 ms | **2.95 ms** | **7.1×** |
| Traced warmed decode | 20.99 ms | **2.95 ms** | **7.1×** |
| Traced warmed decode (T=128) | 7.06 ms | **2.46 ms** | **2.9×** |
| Worst-case prefill/decode PCC vs HF | 0.9958¹ | 0.99674 | — |

¹ Functional worst-case over its 16-seed sweep (fp32 activations). The optimized
worst over the delivered test lengths is **0.99674** — above the 0.995 bar and
above the functional floor, i.e. correctness is preserved, not regressed. All 32
optimized tests pass. Per-length numbers in `pcc_results.json`.

## What changed (operation-topology audit → actions)

The stage-01 decode window was dominated **not** by matmul FLOPs but by fp32
data-movement: per layer the head-split `reshape`/`permute`/`transpose` chain
(~650 µs), the broadcast additive-mask `BinaryNg` (~322 µs), and `scale`/`softmax`
(~140 µs) together dwarfed the matmuls (~530 µs). Modeled DRAM roofline was 7.3%.

| Baseline op(s) (per layer) | Cost | Optimized replacement | Skill |
|---|---|---|---|
| 3× separate Q/K/V matmul (same input) | 96 µs | **1 packed QKV matmul** (weights concatenated at load) | OPT-001 |
| reshape×3 + permute + transpose×2 (head split, fp32) | ~470 µs | **`ttnn.experimental.nlp_create_qkv_heads`** (fused) | — |
| scores matmul + `*scale` + mask `add` (322 µs) + softmax + context matmul | ~470 µs | **`ttnn.transformer.scaled_dot_product_attention`** (FlashAttention-2, additive mask, explicit `SDPAProgramConfig`) | OPT-002 |
| transpose + reshape (concat heads, fp32) | ~180 µs | **`ttnn.experimental.nlp_concat_heads`** (fused) | — |
| fp32 activations, bf16 weights, HiFi4 | — | **bf16 act, BFP8 weights, HiFi2, fp32 dest-acc** | precision |
| separate `gelu` unary (23 µs + DRAM round-trip) | 23 µs | **fused `activation="gelu"` in FF1 matmul** (accurate erf; bit-identical) | — |
| default `ttnn.linear` core selection (~24 cores, all `SLOW`) | — | **explicit 2D core grid** (`CoreGrid(y=8,x=10)`); ffn_output gets a shape-adaptive `MatmulMultiCoreReuseMultiCastProgramConfig` | OPT-004/matmul |

### Precision policy (real-weight sweep, `sweeps/precision.json`, `sweeps/final.json`)

Selected: **bf16 activations · BFP8 attention+MLP+map weights · bf16 norms/embeddings
· HiFi2 matmul & SDPA fidelity · fp32 dest accumulation.**

- **bf16 activations are mandatory** — ttnn SDPA rejects fp32 inputs, so the
  functional stage's fp32-activation choice cannot be carried into the fused path.
- **fp32 dest accumulation is required** for the reduced-precision weights: without
  it BFP8 drops T=16 (seed 16) prefill PCC to **0.9909 < 0.995**; with it the same
  policy passes every tested length (worst 0.99674). It is **not** a latency cost
  here — traced decode is 2.95 ms with fp32-dest-acc vs 2.96 ms without (neutral /
  within noise), so it is kept for free. (The 0.9909 failure reproduces only at the
  test seed; the seed=5000+L sweep in `precision.json`/`final.json` does not surface
  it, which is why the authoritative gate uses the test-seed `gen_evidence.py`.)
- **BFP8 weights selected over bf16**: BFP8 passes all real-weight PCC, is ~4%
  faster on traced decode (2.95 vs 3.07 ms), and halves weight memory.
- **BFP4 weights and LoFi rejected** on real-weight correctness **and** perf: BFP4
  MLP → PCC ~0.95, LoFi → ~0.991, with *no* latency benefit (matmuls are not the
  bottleneck; latency is dtype-invariant until the core-grid fix). This is a
  perf-and-correctness rejection, not a synthetic-PCC veto.
- **HiFi4 rejected**: same-dtype (bf16-weight) fidelity comparison — HiFi4 3.35 ms
  vs HiFi2 3.06 ms traced decode (+9%) for only +0.0008 worst-case PCC.
- Dtype policy verified in the measured rows: every dominant matmul reports
  `HiFi2 BF16 x BFP8 => BF16` in `tracy/albert_layer/decode_perf_report.txt`
  (OPT-013).

### Matmul geometry (`sweeps/matmul_grid.json`)

Default `ttnn.linear` assigned ~24 cores (all `SLOW`). `core_grid=CoreGrid(y=8,x=10)`
(80 cores) is near-optimal and robust for qkv/dense/ffn/map (e.g. ffn 51→24 µs,
dense 42→14 µs). The **ffn_output** (512×2048×768, the largest single matmul) uses
an explicit shape-adaptive `MatmulMultiCoreReuseMultiCastProgramConfig`
(`in0_block_w=8`, subblock `1×3`, grid rows chosen to divide M-tiles): 98→22 µs, and
it avoids the heuristic's small-M cliff (M=32/64 were ~53 µs on core_grid vs ~13-17 µs
here) so it is faster at every valid sequence length.

## Workload-shape mapping (important)

Kokoro "decode" is **not** autoregressive M=1 decode — it is the full-sequence
encode captured into a TTNN trace and replayed for a fixed `(batch, padded_seq_len)`.
The activation M dimension is the whole tile-padded sequence (up to 512), so this
workload is **prefill-shaped**, not small-M. Per the `$optimize` skill's
"map each requirement to the nearest equivalent" rule:

- "Decode activations width-sharded in L1" / "DRAM-sharded decode matmul" (the
  small-M patterns) **do not apply**. DRAM-interleaved activations with 2D-grid
  compute-bound matmul program configs is the skill-correct prefill choice, used for
  both prefill and decode here.
- The `tt-perf-report` "place matmul input0 in L1" advice was **tried and rejected
  with evidence** (`sweeps/advice_disposition.json`): M=512 is only 16 M-tiles, which
  cannot tile-align an L1 height-shard across >16 cores (ttnn errors), and the
  matmuls already run on 64-80 cores at 30-42% DRAM util. LayerNorm hits the same
  16-tile constraint, so it stays on the default 16-core row-parallel path.

## Performance accounting (T=512 traced decode, same run)

- **Device-time decode:** 2.93 ms/replay (`tt-perf-report` window 5866 µs / 2 replays).
- **End-to-end decode:** 2.95 ms/replay warmed wall-clock.
- **Gap:** ~0.02 ms → the path is fully traced with no host/dispatch overhead
  (`execute_trace`+`synchronize` only; fallback audit clean).
- **Modeled DRAM roofline:** 12.8% (66 GB/s). The low fraction is expected and
  explained: this is a small (~30M-param encoder) module built from many small ops
  (fused head TMs, SDPA, 2 LayerNorms/layer, small projections) whose per-op
  launch/movement cost dominates over raw DRAM bandwidth — it is compute/movement
  bound, not DRAM-bound, so it legitimately sits low on the bandwidth roofline.
- Op-type share (decode): Matmul 44.6%, LayerNorm 16.8%, NlpCreateHeads 13.2%,
  SDPA 12.1%, BinaryNg 7.6%, NlpConcatHeads 4.7%, Embeddings 0.5%.

## Correctness / capability (all preserved)

- **PCC ≥ 0.995** on every meaningful length: prefill T∈{8,16,31,32,33,64,128,256,
  500,511,512}, traced decode T∈{32,64,128,500,511,512} (incl. non-tile-aligned),
  batch {2,4,8,32}, real IPA sentences, masked variable-length batches. Worst 0.99674.
- **Non-aligned logical seq lengths supported** — public API takes any length 1..512;
  `prepare_inputs` owns tile padding + masking; the adaptive ffn_output config is
  valid at every tile-padded M. No `seq_len % chunk == 0` restriction.
- **Prefill/decode semantics, determinism, stateless (`decode == prefill`)** preserved
  (`test_optimized_decoder.py`).
- **Batch 1..32** preserved (batch-1 is the primary latency target).
- **Stress / repeated-replay** coverage: `test_decode_stress_repeated_replay`
  (revisited shapes, 3× replays each, bit-identical) + `test_prefill_stress_batched`.
- **Watcher clean** (`TT_METAL_WATCHER=10`): 0 fatal/fault/assert/tripped/overflow
  markers (`watcher/generated/generated/watcher/watcher.log`).
- **Fallback audit clean** (`fallback_audit.txt`): no torch/from_torch/to_torch/
  tilize/untilize/reshard/host-fallback in the measured path.
- **Context contract unchanged**: advertised = supported = 512; both dtype changes
  reduce memory, so no capability reduction (`../context_contract.json`).

## Reproduce

```bash
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"

# Correctness suite (32 tests):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_optimized_decoder.py -v

# PCC + warmed latency evidence -> pcc_results.json, perf_summary.json:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/optimized_decoder/gen_evidence.py

# Precision / geometry sweeps -> sweeps/*.json:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/optimized_decoder/sweep.py precision
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/optimized_decoder/sweep.py final

# Profiling (one window/run; KOKORO_PERF_ITERS=2 keeps the marker buffer bounded):
env $ENV KOKORO_PERF_ITERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf_optimized.py -k prefill   # or decode
# then tt-perf-report with --start-signpost PERF_PREFILL/PERF_DECODE (see work_log.md).

# Watcher-clean run (separate from profiler):
env $ENV TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_optimized_decoder.py -k "512 or determinism or stress"
```

## Artifacts

- `pcc_results.json`, `perf_summary.json`
- `sweeps/precision.json`, `sweeps/final.json`, `sweeps/matmul_grid.json`,
  `sweeps/advice_disposition.json` (advice tried/applied/rejected with evidence)
- `tracy/albert_layer/{prefill,decode}_perf_report.txt` (human tables),
  `*_perf_report.csv` (filtered), `*_perf_report.console.log`,
  `decode_perf_report_advice.txt` (advice-enabled). Raw `*_ops.csv` kept on disk as
  provenance (excluded from git by the 500 KB large-file gate; filtered CSVs are the
  committed evidence).
- `logs/` (gen_evidence, watcher run), `fallback_audit.txt`,
  `watcher/generated/.../watcher.log` (clean)

## Limitations / deviations

- KV cache / paged cache / current-position / autoregressive decode: **N/A**
  (non-autoregressive stateless model), same as stage 01.
- LM head / sampling / MoE / multi-device: out of scope for this decoder stage
  (Kokoro's plbert has none; downstream Kokoro modules are later stages).
- LayerNorm (16.8%) and the fused head TMs (NlpCreateHeads/NlpConcatHeads, 17.9%)
  are bounded by the 16-M-tile shape to ~16-core ops; L1/width-sharding them errors
  on tile-alignment for this shape and is rejected with evidence
  (`sweeps/advice_disposition.json`). They are the practical floor for this shape.
