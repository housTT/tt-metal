# Kokoro-82M optimized-full-model — work log (stage 06)

Target: `hexgrad/Kokoro-82M`, branch `agentic-research/hous/kokoro-82m-p150`,
4× Blackhole p300c (`ClusterType.P300_X2`, `(1,4)` ring mesh, FABRIC_1D_RING).
Env: `TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=.../ttnn:...` (dev checkout).
Skills: $multichip, $optimize, $tt-device-usage (+ $full-model / $tt-enable-tracing
contracts carried from stage 05). Device health verified (`tt-smi -ls --local`:
4× p300c visible) before any run.

## Approach
Kokoro has ONE layer kind (weight-tied AlbertLayer ×12), so the stage-04 optimized
multichip decoder's traced 12-layer decode IS the layer-stack lower bound. Stage 06
optimizes the FULL token-out path around that decoder (which is preserved verbatim):
embeddings/norms are inside the decoder; the new full-model-only work is the terminal
reconstruction readout (LM-head analog) + on-device greedy sampling. Profiled the
stage-05 token-out path first, found the terminal overhead was dominated by two
avoidable costs, fixed both, and remeasured before/after in the same regime.

## Diagnosis (stage-05 token-out profile)
- BEFORE report `../full_model/tracy/tokenout_perf_report.txt` (reduced 2-layer
  probe): ArgMaxDeviceOperation = 25.06% (112 µs/op, **1 core**) — the dominant
  avoidable terminal cost. (The stage-06 AFTER report is `tracy/tokenout_perf_report.txt`.)
- Readout matmul `32 x 768 x 192` = SLOW, `in0_block_w=1`, 17 µs.
- Before baseline (min-of-3, `perf_full_model.py`): terminal overhead over the
  decoder-layer-stack lower bound = 7.8% @T128 / **18.3% @T512** (above the
  10-15% target → must close).

## Isolation experiments
- `argmax_iso.py`: `ttnn.argmax(dim=-1)` on TILE input is single-core; on ROW_MAJOR
  it is multi-core. ROW_MAJOR (incl. the untilize) = 43 µs vs 118 µs @T128 (2.7x),
  148 µs vs 454 µs @T512 (3x); bit-identical greedy token (device-vs-device agree).
  The earlier fp32-vs-bf16 "mismatch" was a synthetic-reference artifact, not a bug.
- `readout_iso.py`: explicit `MatmulMultiCoreReuseMultiCastProgramConfig`
  (gx=6 tiling vocab N, gy dividing M, `in0_block_w=8` dividing K=24) beats the auto
  core_grid: 11.8 vs 21.6 µs @T128; ~neutral @T512. Selected in0_block_w=8.

## Changes (tt/model.py only; decoder + generator orchestration untouched)
1. `_argmax_seq`: untilize logits shard to ROW_MAJOR before `ttnn.argmax` → multi-core.
   `logits_s` left in TILE so the want_logits host cross-check is unaffected; argmax
   output is UINT32/ROW_MAJOR regardless of input layout, so `gather_tokens` unchanged.
2. `_readout` + new `_readout_program_config(m_tiles)`: explicit 2D program config
   (cached per M-tile count; valid at every tile-aligned local length).

## Results (real weights, warmed, batch-1, min-of-3)
- Token-out T128: 1.906 → **1.796 ms** (525 → 557 t/s/u); overhead 7.8% → **2.4%**.
- Token-out T512: 2.869 → **2.571 ms** (349 → 389 t/s/u); overhead 18.3% → **6.6%**
  (was above the target, now within it).
- TTFT eager @128: 13.74 → **9.09 ms**. Teacher-forcing decode: 314 → **330 t/s/u**.
- Decoder-only lower bound unchanged (1.754/2.412 ms) — confirms decoder preserved.
- tt-perf-report (reduced 2-layer, `tracy/`): ArgMax 25.06% → **9.94%** (112 µs/1-core
  → 36 µs/6-core); readout matmul 17 → 4.5 µs; Matmul now cleanly dominant (32.52%,
  the stage-04 BFP8/HiFi2 FFN/attention, `HiFi2 BF16 x BFP8 => BF16` in the rows).

## Accuracy (optimized path, `results.json` via `gen_evidence.py`)
- PCC vs HF ≥ 0.995 all lengths (worst 0.996999 @128), batch-4 0.998881 (identical to
  stage 05 — argmax bit-identical).
- Prefill top-1/5/100 = 1.000. Teacher-forcing top-1 0.9865 / top-5 0.9955 / top-100
  1.000 (bars: top-5 ≥ 98% ✓, top-100 = 100% ✓).
- Autoregressive: HF-TT agreement 1.0, adjacent-dup 0.0. Split-sampling: outputs
  differ, deterministic, on-device argmax == host, host_argmax=0/logits_readbacks=0.
- 19/19 `tests/test_full_model.py` pass (incl. on-device-argmax==host, split-sampling
  trace feedback, non-aligned PCC 31/33/127/200/511, batch).

## Gates
- `check_degenerate_output.py --scope autoregressive --missing-artifacts critical` → rc=0.
- `check_context_contract.py --stage optimized-full-model --require-contract` → rc=0.
- Watcher (`TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1`): 0 fault markers.
- Fallback audit: clean (no host boundary on the measured token-out path).

## Performance accounting (`perf_summary.json`)
- Modeled weight-read roofline ~0.092 ms/encode; NOT binding. Binding constraint =
  op dispatch/launch (5.3% DRAM util, ~1 µs op-to-op gaps over ~200 small ops on a
  tied-layer 82M encoder). Device time == e2e for traced nonblocking replay
  (0 per-step host refreshes; no AR feedback loop by construction).

## $multichip / $optimize checklist coverage (full-model terminal path)
- Decoder path fully traced, no host fallback (fallback_audit.txt). Terminal readout
  + argmax fully traced in the token-out graph.
- Operation-topology audit recorded (README table). Terminal matmul program config
  swept (readout_iso). Sampler op no longer dominant (argmax 9.94%, Matmul 32.52%).
- LM-head: vocab-split rejected with reason (tiny replicated 178 vocab; would force
  <2-tile shards + cross-device gather). No full-vocab all-gather; pad columns masked
  to -inf; greedy on-device semantically greedy.
- Persistent CCL/buffers + BFP8/HiFi2 decoder policy preserved verbatim (rows show it).
- Datatype frontier NOT swept here ($datatype-sweep owns Pareto selection).

## Stage review
- Independent $stage-review (fresh general-purpose subagent, read-only, no device):
  **clean-pass**, no Required Work. Re-derived every headline claim from JSON / perf
  tables / git / code: confirmed only tt/model.py changed (decoder + generator
  byte-identical to HEAD via `git diff --stat HEAD -- .../tt/`), argmax 1-core/112µs
  → 6-core/36µs and readout 17→5µs in the measured rows, decoder BFP8/HiFi2 policy
  present in the matmul rows (preserved), T512 terminal overhead 18.31% → 6.59%,
  greedy on-device (host_argmax=0), non-aligned PCC (31/33/127/200/511) ≥ 0.995,
  prefill top-1/5/100 = 1.0, TF top-5 0.9955 / top-100 1.0, watcher 0 markers,
  context 512 preserved. Only P3/cosmetic concerns raised.
- P3 fixes applied post-review: (1) corrected the BEFORE tracy citation in
  README/work_log to point at `../full_model/tracy/tokenout_perf_report.txt`
  (the in-doc `tracy/tokenout_perf_report.txt` is the AFTER report); (2) retained
  fresh run logs in `logs/` (test_full_model.log 19/19, degenerate_gate.log rc=0,
  context_contract.log rc=0) to close the thin-logs hard-check gap. The two
  anomalies (logits_readbacks=1 = the single host cross-check in gen_evidence, not
  the greedy path; readout 6-core in both reports = the in0_block_w 1→8 / DRAM%
  4→15 win, not a core-count change) were reviewed and classified expected.

## Commit
- (SHA recorded in a follow-up note after the checkpoint commit)
