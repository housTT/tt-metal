# Work Log — Kokoro-82M Optimized Decoder (Stage 02)

Date: 2026-07-17
Branch: `agentic-research/hous/kokoro-82m-p150`
Hardware: Blackhole p300c, single device (device_id=0), compute grid 11×10 (110 cores).
Runtime env (dev-checkout ttnn build; avoids JIT header/kernel mismatch):
`TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal`

## Summary

Optimized the stage-01 functional plbert/ALBERT encoder. Traced warmed decode
T=512 went **20.99 ms → 2.95 ms (7.1×)**; warmed prefill T=512 **21.01 → 2.95 ms
(7.1×)**. PCC preserved ≥ 0.995 on all tested lengths (worst 0.99674, above the
functional 0.9958 floor). 32/32 optimized tests pass; watcher clean; fallback
audit clean. Skill mapping recorded: this TTS "decode" is a full-sequence traced
replay (prefill-shaped M=seq_len), not autoregressive M=1 decode.

## Device health (start)

`tt-smi -ls --local`: all 4 p300c chips visible. `open_device(0)` + close =
SMOKE_OK, grid 11×10. No reset needed this session.

## 1. Baseline re-confirm

- `pytest tests/test_functional_decoder.py` → **29 passed** (~13 s).
- Baseline decode `tt-perf-report` (stage 01) read: per-layer dominated by fp32
  head reshape/permute/transpose (~650 µs), broadcast mask-add BinaryNg (322 µs),
  scale/softmax (~140 µs); matmuls only ~530 µs; modeled DRAM roofline 7.3%.
  Conclusion: movement/fp32-bound, not matmul-bound.

## 2. Operation-topology audit → optimized_decoder.py

Implemented `tt/optimized_decoder.py` (`OptimizedDecoder`, configurable
`PrecisionPolicy`). Same public contract as functional (prepare_inputs / prefill /
decode / from_state_dict / release_traces). Topology changes (full table in README):
- Packed QKV (weights concatenated at load) → `nlp_create_qkv_heads` (OPT-001).
- `scaled_dot_product_attention` (FlashAttention-2, additive `[b,1,s,s]` mask built
  only when there is tile/user padding; `None` for the aligned full-context path)
  with explicit `SDPAProgramConfig(q=k=128)` (OPT-002).
- `nlp_concat_heads`; two cheap unit-dim reshape adaptors (no layout conversion).
- Fused `activation="gelu"` in FF1 (verified bit-identical to separate accurate-erf
  gelu, both PCC 0.99986 vs torch gelu_new — dropped the separate 23 µs unary).
- Explicit 2D matmul core grid + adaptive ffn_output program config.

Derisk probe (`/tmp/probe_sdpa.py`): fused path bf16/bf16/HiFi2 T=128 PCC 0.9969.

## 3. Precision sweep (real weights) — `sweeps/precision.json`, `sweeps/final.json`

Latency was **dtype-invariant** before the core-grid fix (all policies ~4.75 ms),
proving matmuls were not the bottleneck. PCC (real weights):

| policy | worst PCC | note |
|---|---|---|
| bf16 act / bf16 w / HiFi2 | 0.99833 | pass |
| bf16 act / BFP8 w / HiFi2 | 0.9975 | pass |
| bf16 act / BFP8 w / LoFi | 0.99129 | FAIL |
| bf16 act / BFP8 attn+BFP4 mlp / LoFi | 0.954 | FAIL |
| bf16 act / BFP4 w / LoFi | 0.938 | FAIL |

- fp32 activations impossible (ttnn SDPA rejects fp32 inputs — TT_FATAL).
- BFP4/LoFi rejected on real-weight correctness AND zero latency benefit.

At the **exact test seeds** (prefill seed=seq_len) BFP8 **without** fp32 dest-acc
fails T=16 (0.9909). With `fp32_dest_acc=True` it passes (T=16 0.99697; worst over
all lengths 0.99674) at **no latency cost** (2.95 ms with acc vs 2.96 ms without —
neutral). bf16-weight control worst 0.99887 but ~4% slower (3.07 ms) and 2× weight
memory. **Selected: bf16 act / BFP8 weights / HiFi2 / fp32_dest_acc.**
(`/tmp/eval_policy.py`, `/tmp/eval_fid.py`, `/tmp/eval_bfp8acc.py`; `sweeps/final.json`
re-run with the shipped fp32_dest_acc=True default.)

## 4. Matmul geometry — `sweeps/matmul_grid.json`

Default `ttnn.linear` used ~24 cores (all `SLOW`). `core_grid=(y=8,x=10)` (80 cores):
ffn_out 98→27, dense 42→14, ffn 51→24, qkv 52→26, map 24→9 µs. ffn_output further
improved with an explicit shape-adaptive `MatmulMultiCoreReuseMultiCastProgramConfig`
(in0_block_w=8, subblock 1×3): 27→21 µs, and avoids the small-M core_grid cliff
(M=32/64 were ~53 µs → ~13-17 µs). Adaptive config validated correct at every test
length (all PCCs present, no errors). qkv/dense/ffn kept on core_grid (explicit
variants measured slower). `/tmp/probe_mm.py`, `/tmp/probe_mmcfg.py`,
`/tmp/probe_ffnout.py`.

## 5. Remaining-advice dispositions — `sweeps/advice_disposition.json`

- SDPA chunk sizes swept: q=k=128 near-optimal (256/512 worse). Kept.
- "Place matmul input0 in L1": **rejected with evidence** — M=512 = 16 M-tiles
  cannot tile-align an L1 height-shard over >16 cores (ttnn errors), matmuls already
  on 64-80 cores at 30-42% DRAM util; block-shard reshard would exceed benefit for
  this prefill-shaped workload (`/tmp/probe_l1.py`).
- "FF1 output subblock 1×1 small": **rejected with evidence** — explicit subblock
  2×2 (55.9 µs) / 1×4 (54.8 µs) slower than core_grid 48.5 µs (`/tmp/probe_ff1sub.py`).
- Sharded LayerNorm: **rejected** — same 16-tile constraint; generic height-shard
  errors on shard alignment (`/tmp/probe_sdpa_ln.py`). Default 16-core row-parallel kept.
- "Use HiFi4": rejected (+9% latency, +0.0008 PCC).
- Dtype policy confirmed in measured rows: `HiFi2 BF16 x BFP8 => BF16` on every
  dominant matmul (`tracy/albert_layer/decode_perf_report.txt`) — OPT-013.

## 6. Final perf / evidence

- `gen_evidence.py` → `pcc_results.json` (worst 0.99674), `perf_summary.json`
  (T=512 prefill 2.95 / decode 2.95 ms; T=128 prefill 2.74 / decode 2.46 ms).
- Tracy (KOKORO_PERF_ITERS=2, one window/run) for decode and prefill →
  `tracy/albert_layer/{decode,prefill}_ops.csv`; tt-perf-report 1.2.8 →
  `*_perf_report.{txt,csv,console.log}` + `decode_perf_report_advice.txt`.
  Device-time decode 2.93 ms/replay ≈ e2e 2.95 ms (no host gap; fully traced).
  Modeled DRAM roofline 12.8% — small many-small-op module, movement/compute-bound
  (explained, not DRAM-bound).
- Fallback audit (`fallback_audit.txt`): 0 to_torch; from_torch/torch.* only in
  from_state_dict + prepare_inputs; forward/trace path pure TTNN.
- Watcher (`TT_METAL_WATCHER=10`, `-k "512 or determinism or padding or stateless
  or real_ipa or stress or optimized_path"`): 10 passed;
  `watcher/generated/generated/watcher/watcher.log` — 0 fatal/fault/assert/tripped/
  overflow/sanitize markers.

## 7. Tests — `tests/test_optimized_decoder.py` (32 passed)

Mirrors the functional suite (same lengths/seeds) + `test_uses_optimized_path`
(asserts packed QKV, no separate q/k/v, BFP8 dtype, fp32_dest_acc reached the build)
+ stress (`test_decode_stress_repeated_replay`, `test_prefill_stress_batched`).
`tests/test_perf_optimized.py` for signposted profiling.

## Optimization checklist (from $optimize)

- [x] Decode path fully traced, no host fallbacks (fallback_audit.txt).
- [x] Activation layout: DRAM-interleaved is the skill-correct choice for this
      prefill-shaped (large-M) workload; L1 width-sharding rejected with evidence.
- [x] Prefill: DRAM-interleaved + 2D matmul program configs.
- [x] Operation-topology audit recorded (README table + this log).
- [x] Best-candidate comparison: final beats functional baseline (7.1×) and the
      bf16-weight candidate (4%); BFP8 selected over bf16 on perf+memory, passing PCC.
- [x] Final dtype/fidelity verified in measured rows (HiFi2 BF16 x BFP8).
- [x] SDPA used (not hand-built attention); explicit program config; chunks swept.
- [x] Packed same-input projection (QKV) — wins vs separate (baseline had 3 matmuls
      + head reshapes; packed + fused heads removes both).
- [x] Explicit memory/program/compute-kernel configs for important ops.
- [x] Dominant matmul program-config sweep with before/after table (matmul_grid.json).
- [x] Decode compute fidelity swept (LoFi/HiFi2/HiFi4) — HiFi2+fp32acc kept.
- [x] Attention vs MLP weight dtype swept separately (BFP8 both; BFP4 tried/rejected).
- [x] MLP BFP4/LoFi trial done (rejected on real-weight PCC + no perf win).
- [x] DRAM-sharded decode matmul: N/A mapping recorded (prefill-shaped).
- [x] Collectives / MoE / LM-head / sampling: N/A (single device, encoder).
- [x] Performance accounting reconciled (device 2.93 ≈ e2e 2.95 ms; roofline 12.8% explained).
- [x] Batch capability preserved (batch-1 primary; 2/4/8/32 tested).
- [x] Stress + watcher-clean optimized run exist.

## Commit

- Local checkpoint commit of stage-owned files under
  `models/autoports/hexgrad_kokoro_82m/` after `$stage-review` clean-pass. Never pushed.
  - Repo: `/home/ttuser/dev/tt-metal`, branch `agentic-research/hous/kokoro-82m-p150`
  - SHA: `2a9991dc0a9` (36 files). Pre-commit hooks (autoflake/isort/black/whitespace)
    reformatted the .py/.json; post-reformat suite re-run = 32 passed.
  - Raw Tracy `*_ops.csv` (1.3M/795K) kept on disk as provenance but excluded by the
    500 KB large-file gate + `.gitignore` (`*.csv`); the filtered `*_perf_report.csv`
    (force-added) are the committed CSV evidence, matching stage 01.
  - Excluded (unrelated dirty runner files, NOT stage-owned): `.agents/README.md`,
    `.agents/requirements.txt`, `.agents/scripts/multigoal`.
- Follow-up commit records this SHA in the work log (mirrors stage 01).
