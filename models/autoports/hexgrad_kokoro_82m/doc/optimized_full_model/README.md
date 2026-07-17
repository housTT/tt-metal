# Kokoro-82M — Optimized Full Model (TTNN, TP=4)

Stage 06 (optimized-full-model) for `hexgrad/Kokoro-82M` on **4× Blackhole p300c**
(`ClusterType.P300_X2`, physical 4-ring exposed as a `(1, 4)` mesh). Starts from
the completed stage-05 full model and optimizes the **complete model/generator
token-out path** across the mesh. The stage-04 optimized multichip decoder and the
stage-05 generator orchestration are preserved **verbatim**; the optimization work
is in the **terminal path** (reconstruction readout / LM-head + on-device greedy
sampling), which was the largest avoidable full-model-only cost.

## Headline performance — before (stage-05) → after (stage-06)

Warmed, real weights, batch-1, `(1,4)` ring mesh, min-of-3 traced replay
(`perf_full_model.py`, same mesh session for the decoder-layer-stack lower bound
and the full token-out path).

| Metric | Workload | **Before** | **After** |
|---|---|---|---|
| **Token-out decode** (traced, greedy, on-device argmax) | T=128 | 1.906 ms · 525 t/s/u | **1.796 ms · 557 t/s/u** |
| **Token-out decode** (traced, greedy, on-device argmax) | T=512 | 2.869 ms · 349 t/s/u | **2.571 ms · 389 t/s/u** |
| **TTFT** (eager prefill @128, host-visible first output) | T=128 | 13.74 ms | **9.09 ms** |
| **Traced teacher-forcing decode** (`run_teacher_forcing`, growing prefix) | — | 314 t/s/u | **330 t/s/u** |
| Decoder-layer-stack lower bound (traced 12-layer decode) | T=128 / T=512 | 1.754 / 2.412 ms | 1.754 / 2.412 ms (unchanged) |
| **Terminal overhead** over the layer-stack lower bound | T=128 | **7.8 %** | **2.4 %** |
| **Terminal overhead** over the layer-stack lower bound | T=512 | **18.3 %** (> target) | **6.6 %** (within target) |

Teacher-forcing and token-out decode are reported separately: teacher forcing
re-encodes a **growing** prefix each step plus generator-loop overhead, while
token-out is a fixed-`T` traced replay that includes the readout + on-device
argmax. Both improved.

## What was optimized (and why it was the right target)

Kokoro has **one** layer kind (weight-tied `AlbertLayer` ×12), so the stage-04
optimized multichip decoder's traced 12-layer decode **is** the layer-stack lower
bound directly. Profiling the stage-05 token-out path
(BEFORE report: `../full_model/tracy/tokenout_perf_report.txt`; AFTER report:
`tracy/tokenout_perf_report.txt`) showed the terminal overhead was dominated by
two avoidable costs:

1. **On-device greedy argmax ran single-core.** `ttnn.argmax(dim=-1)` on a **TILE**
   input runs **single-core** (112 µs @T=128 / ~454 µs @T=512 device — 25 % of the
   reduced 2-layer probe). On a **ROW_MAJOR** input it runs **multi-core**. Fix:
   untilize the tiny replicated-vocab logits shard to ROW_MAJOR before argmax
   (`model.py::_argmax_seq`). Result: **6 cores, 36 µs @T=128** (bit-identical
   greedy token; 9.94 % of the reduced probe — no longer a dominant cost).
2. **The readout (LM-head) matmul used the decoder's auto core-grid** and was left
   `SLOW` at `in0_block_w=1` (17 µs). Fix: an explicit
   `MatmulMultiCoreReuseMultiCastProgramConfig` (`gx` tiling the vocab N dim,
   `in0_block_w=8` dividing K=24) → **4–5 µs** (`model.py::_readout_program_config`).

Both changes are **pure on-device TTNN**, trace-safe, and add **no host boundary**
(`fallback_audit.txt`). The measured token-out path stays fully traced with split
sampling: encode → readout → ROW_MAJOR untilize → `ttnn.argmax`, one captured
graph, `host_argmax=0` / `logits_readbacks=0` on the greedy path.

### Operation-topology audit (measured token-out path, per sequence shard)

| Op (role) | Before | After | Action / rationale |
|---|---|---|---|
| 12× `AlbertLayer` (decoder stack) | stage-04 optimized (TP=4 attn, seq-parallel FFN, block-sharded L1 norm, persistent CCL, 1 AG + 1 RS/layer, BFP8/HiFi2) | **unchanged (verbatim import)** | Preserve dtype/fidelity/CCL policy + rejection ledger + inter-layer sharded residual. Not re-tuned here (stage-04 owns it). |
| readout matmul `[b,1,S/TP,768]@[768,192]` (LM-head analog) | auto core_grid, `in0_block_w=1`, SLOW, 17 µs | explicit 2D PC `in0_block_w=8`, **4–5 µs** | LM-head is a real terminal matmul; explicit K-dividing block width restores utilization. |
| greedy sampler | `ttnn.argmax` on TILE, **1 core, 112 µs** | `to_layout(ROW_MAJOR)` + `ttnn.argmax`, **6 cores, 36 µs** | ttnn argmax is single-core on TILE, multi-core on ROW_MAJOR; output UINT32/ROW_MAJOR unchanged so `gather_tokens` contract is preserved. |
| terminal gather | tiny `[b,S]` token tensor only (logits gather only on `want_logits`) | unchanged | No full-vocab all-gather; vocab is tiny + REPLICATED (no vocab-split — see below). |

### LM-head / sampling contract — decisions preserved

- **No vocab-split / no full-vocab all-gather.** Kokoro's reconstruction readout is
  a tiny (178, padded 192) **replicated** vocab computed per-token on the sequence
  shard. Splitting a 178-wide vocab across 4 devices would give <2-tile shards and
  force a cross-device gather before argmax — strictly worse. The sequence-sharded
  replicated-vocab readout keeps all sampler work local; only the tiny token tensor
  gathers. (Recorded in `../full_model/sampler_comparison.json`.)
- **Greedy stays on device** via `ttnn.argmax` (semantically greedy, tile-shaped
  logits). `Sampling1D`/TTTv2 and the `models/common/sampling` TTTv1 path both
  target a large **sharded** vocab with a cross-device all-gather and remain the
  recorded rejected alternatives for this tiny replicated vocab; TTTv2 is retained
  for a top-k/top-p path if sampled TTS-token selection is ever needed.
- After the fix, **no sampler op dominates** token-out: Matmul is the clear
  dominant op (32.5 % of the reduced probe — the decoder's stage-04-selected
  BFP8/HiFi2 FFN/attention matmuls), argmax is 9.9 %.

## Accuracy — unchanged on the optimized path (`results.json`)

The argmax result is **bit-identical** (same on-device bf16 logits, just multi-core),
so accuracy is preserved exactly:

- **`last_hidden_state` PCC vs HF ≥ 0.995** at every tested length incl. non-aligned
  (worst **0.996999** @T=128; 8/31/33/64/127/200/256/511/512), batch-4 0.998881.
- **Prefill reconstruction: top-1 = top-5 = top-100 = 1.000** (222 positions,
  `run_prefill_check`).
- **Teacher-forcing: top-1 0.9865, top-5 0.9955, top-100 1.000** (222 positions,
  `run_teacher_forcing`). Bars: top-5 ≥ 98 % ✓, top-100 = 100 % ✓.
- **Free-running / autoregressive analog:** HF-vs-TT token agreement 1.000,
  adjacent-dup 0.000; runner degenerate-output gate **rc=0** (adjacent_dup 0.0,
  trigram-loop 0.1429).
- **Split-sampling:** outputs differ across contexts, deterministic replay,
  on-device argmax == host argmax; `host_argmax=0`, `logits_readbacks=0` on greedy.
- `AIME24` chat-template is **N/A** (Kokoro is a TTS model with no chat template /
  causal LM); replaced by real IPA phoneme references (carried from stage 05,
  `../full_model/make_references.py`).

## Context / batch / non-aligned contract — preserved

Advertised = supported = **512**, **no reduction** (`../context_contract.json →
optimized_full_model`). No new weights or material persistent buffers (the
ROW_MAJOR argmax intermediate is <0.05 MB/device, freed by the trace lifetime).
Public API still accepts any logical length 1..512 incl. non-aligned (31/33/127/
200/511 validated at PCC ≥ 0.995). Batch-1 primary; batch dims not hard-coded
(batch-4 PCC 0.998881).

## Performance accounting (`perf_summary.json`)

- **Roofline (modeled):** ~47 MB/device BFP8 linear-weight reads per 12-layer
  encode / ~512 GB/s per-device DRAM BW → ~0.092 ms — **not the binding constraint.**
- **Binding constraint = op dispatch / launch.** `tt-perf-report` shows only **5.3 %**
  aggregate DRAM utilization and ~1 µs op-to-op gaps across ~200 device ops; this
  82M encoder is built from many small ops (12 tied layers, tiny M), so it
  legitimately sits far below the DRAM roofline. The gap is launch-bound, not an
  untraced-loop / host-readback / input-refresh defect (those are all zero on the
  measured path).
- **Device time == e2e** for traced nonblocking replay (one `execute_trace` +
  one `synchronize`; no per-op host dispatch, no per-step token/position/page-table
  refresh — the bidirectional encoder has no AR feedback loop).

## Watcher / fallback

- Watcher clean: 0 fault/assert/overflow markers (`watcher/watcher.log`, ETH
  excluded — infra limit, same as stages 03/04/05).
- Fallback audit clean (`fallback_audit.txt`): no host fallback in the measured
  token-out path; the stage-06 terminal fixes are pure on-device TTNN.

## Reproduce

```bash
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"
D=models/autoports/hexgrad_kokoro_82m/doc/optimized_full_model

# before/after perf (min-of-3 traced replay; run --tag before with model.py stashed):
env $ENV python $D/perf_full_model.py --tag after
# full accuracy + split-sampling + autoregressive evidence on the optimized path:
env $ENV python $D/gen_evidence.py
# tests (19):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_full_model.py -v
# argmax / readout isolation sweeps:
env $ENV python $D/argmax_iso.py
env $ENV python $D/readout_iso.py
# reduced-layer token-out tt-perf-report (tracy; separate from watcher):
env $ENV KOKORO_PERF_LAYERS=2 KOKORO_PERF_ITERS=8 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf_full_model.py -k tokenout
# runner gate:
env $ENV python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/hexgrad_kokoro_82m --missing-artifacts critical --scope autoregressive
# watcher (separate from profiler; ETH excluded):
env $ENV TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_full_model.py \
  -k "pcc_vs_hf or split_sampling or decode_on_device or free_running"
```

## Artifacts

- `perf_before.json`, `perf_after.json`, `perf_summary.json`, `results.json`,
  `split_sampling.json`
- `tracy/{tokenout_perf_report.txt,tokenout_perf_report.csv,tokenout_perf_report_stacked.csv,tokenout_perf_report.console.log}`
  (the 3.4 MB raw `tokenout_ops_perf.csv` is left on disk only — gitignored per the
  skill's raw-Tracy-CSV exclusion — and is reproducible via the tracy command below)
- `watcher/watcher.log`, `fallback_audit.txt`, `logs/`
- `perf_full_model.py`, `gen_evidence.py`, `argmax_iso.py`, `readout_iso.py`, `work_log.md`
- `../context_contract.json → optimized_full_model`; HF references reused from
  `../full_model/` (`readiness_recon_prefill.refpt`, `readiness_recon_tf.refpt`,
  `make_references.py`, `sampler_comparison.json`)
- `../../readiness_autoregressive/{autoregressive_meta.json,hf_completion.txt,tt_completion.txt}` (refreshed)
- tests: `tests/test_full_model.py`, `tests/test_perf_full_model.py`

## Limitations / deviations

- The "full model" is the plbert encoder + reconstruction readout (LM-head analog),
  not the full phonemes→audio TTS pipeline (out of scope; also non-autoregressive).
  Carried from stage 05.
- Autoregressive/KV/paged-cache/current-position/token-feedback contract items are
  N/A (bidirectional stateless encoder); model-specific equivalents implemented +
  validated.
- The multi-core argmax still reduces vocab for **every** sequence position (needed
  for the per-position reconstruction / prefill-check / free-running paths); a
  last-position-only argmax does not apply to this all-positions contract.
- Decoder matmul rows remain tagged `SLOW`/DRAM-bound by tt-perf-report — that is
  the stage-04 selected geometry/dtype policy (with its rejection ledger),
  preserved verbatim; the decoder is not re-tuned in this stage.
