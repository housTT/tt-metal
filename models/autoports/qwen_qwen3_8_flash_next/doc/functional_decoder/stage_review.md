# Stage Review

Verdict: more-work-needed

## Required Work

- P1: QSA advertised-context public prefill is not evidenced by the delivered tests.
  Evidence:
  `doc/context_contract.json` advertises `functional_decoder_supported_context_tokens: 262144` with `capability_reduction: null`, but the layer-3 QSA entry records `largest_public_prefill_tokens_tested: 2049` and says the complete public QSA prefill at 262144 was not executed because the graph has quadratic prompt work. `doc/functional_decoder/README.md` repeats that full public QSA prefill at 262144 was not run. The long-context QSA test in `tests/test_functional_decoder.py` does not call public `prefill_forward` on `[1, 1, 262144, 10240]`; it calls private `_qsa_prefill` on a terminal `[1, 1, 128, 2560]` mixed-attention chunk at offset 262016, then runs public decode. `long_context_qsa_final.log` shows only this reduced layer-3 test passing in 10.92 s. The functional-decoder skill requires full supported prefill unless a measured L1/DRAM capacity limit forces a reduction, and explicitly says tractability/runtime is not sequence-capacity evidence. The skill does allow reduced layer-level coverage for slow long tests, but that language cannot replace the separate full-supported-prefill gate while the context contract still claims no reduction.
  Why this matters:
  Downstream stages will treat the public decoder boundary as supporting the full HF-advertised context. The reduced QSA harness exercises useful geometry, but it bypasses the 10240-wide public prefill path, hyperconnection/MoE work, the full 2048-chunk loop, and end-to-end cache/page-table progression across the advertised prompt.
  Required next step:
  Either run and record an end-to-end public QSA `prefill_forward` at 262144 tokens, or record hard physical capacity evidence and update the context contract/largest-feasible supported value accordingly. If only a time-saving reduced harness is retained, label it as partial geometry evidence rather than satisfying the full public prefill gate.

- P1: The QSA underfilled fixed-512 top-k semantic discrepancy is still open.
  Evidence:
  The HF source at `/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py` selects `scores.topk(min(self.block_topk, num_complete_blocks), ...)` and appends the incomplete tail exactly once. The TTNN implementation in `tt/functional_decoder.py` always executes `ttnn.topk(scores, k=QSA_BLOCK_TOPK, ...)` after masking invisible blocks, expands those 512 blocks, appends the tail, and later marks validity only with `selected <= pos`. `QSA_AUTODEBUG.md` identifies this as a real source-level discrepancy, notes that a masked filler block equal to the incomplete tail block would duplicate valid tail tokens, and records an actual-weight CPU control where deliberate duplication drops complete layer-3 PCC to about 0.992716. `AUTOFIX.md` demotes but does not fix the issue, and `README.md` lists it as a known limitation because the repaired single real-weight device test still passes.
  Why this matters:
  This is stage-critical QSA selection semantics, not a cosmetic limitation. The current passing PCC at one 33-token real-weight case does not prove equivalence for all underfilled/top-k-tie cases, and the implementation appears to rely on TT top-k tie ordering for masked filler blocks.
  Required next step:
  Fix the TTNN selection so underfilled complete blocks match HF semantics, for example by propagating complete-block eligibility through the expanded lanes, using a dynamic/min effective top-k, or deduplicating the incomplete tail. If a model-specific waiver is intended instead, add an exact selected-token multiplicity regression that proves each query's valid token multiset matches HF for underfilled positions and relevant page/tile boundaries, plus PCC evidence for the affected cases.

- P2: The context contract overstates long non-aligned prefill execution.
  Evidence:
  `doc/context_contract.json` lists `262143` in `logical_prefill_lengths_tested`, but the current test code only checks `prefill_chunk_plan(HF_ADVERTISED_CONTEXT - 1)` for that length. The device-executed long GDN/PLE public prefill logs cover aligned 262144, while QSA public prefill is device-executed only through 2049. The skill asks for a long non-divisible length near the supported context, with a reduced harness allowed when the long run is too slow; planner-only coverage is not the same as public or reduced-layer execution.
  Why this matters:
  The stage specifically claims public padding/chunking and output slicing for arbitrary logical lengths through the advertised context. The near-max non-aligned path is exactly where final-chunk masking, page slicing, and output concatenation can diverge from the aligned 262144 run.
  Required next step:
  Add executable evidence for a near-supported-context non-aligned length such as 262143 for each meaningful layer kind, or revise `context_contract.json` to distinguish planner-only checks from executed TTNN prefill evidence and record the remaining gap.

## Other Concerns

- `targeted_trace_pcc.log` appears stale: it reports `test_linear_kinds_batched_decode_accept_distinct_current_positions`, which is not present in the current `tests/test_functional_decoder.py`. I did not treat this as a blocker because the final README/work_log rely on `correctness.log`, but stale extra evidence should be removed or clearly marked.
- The QSA Tracy decode provenance reports `traced_decode_ms=271.951550` for a one-replay profiler run, while `perf_host_timing.log` reports the final non-profiled ten-replay host timing as `11.247692` ms/token. The README separates host timing from profiler timing, so this is controlled, but future summaries should avoid using the profiler-intrusive host number as a latency metric.
- `static_checks.log` is compact and does not preserve the full command transcript for every work-log claim, such as compileall. This is not a stage blocker given the stronger code/log evidence above, but the next finalization pass should make static-check provenance less ambiguous.

## Hard-Check Gaps

- Full-context QSA decode is executed eagerly in the long-context test, and traced replay/PCC is evidenced at smaller QSA context (`max_seq_len=4096`, `current_pos=33` for real PCC and position 0 for determinism). The submitted evidence does not combine trace replay with the advertised 262144-token QSA page-table/RoPE geometry. This is secondary to the P1 prefill gap, but should be closed or explicitly justified when reworking context evidence.
- Batch coverage reaches `max_batch=2` and records that value in `context_contract.json`. The functional-decoder skill asks to test up to batch 32 when hardware/harness allows, or record the hard limit. No hard batch limit evidence is recorded. I did not promote this above because the original stage contract emphasized layer-kind/context/paging coverage rather than batch-32 capability.

## Anomaly Ledger

- Observed anomaly:
  Full public QSA prefill at 262144 was not run even though the context contract advertises 262144 support with no reduction.
  Evidence:
  `context_contract.json` records QSA `largest_public_prefill_tokens_tested: 2049` and the remaining-risk text says the complete public QSA prefill was not executed due quadratic prompt work. `test_full_advertised_context[3]` calls `_qsa_prefill` on the terminal 128-token attention chunk instead of public `prefill_forward`.
  Affected path:
  QSA public prefill, advertised-context capability, page-table/cache progression over the full prompt.
  Control or comparison:
  GDN and GDN+PLE have separate public 262144-token prefill pass logs (`long_context_linear0_full.log`, `long_context_linear1_full.log`); QSA only has reduced terminal-chunk geometry evidence.
  Likely subsystem:
  QSA long-context test design / capability contract.
  Investigation performed:
  Compared README/work_log/context JSON against test code and long-context logs.
  Resolution:
  more-work-needed

- Observed anomaly:
  QSA underfilled top-k can differ from HF by selecting masked filler blocks before appending the incomplete tail.
  Evidence:
  HF uses `min(self.block_topk, num_complete_blocks)` and a boolean selected-token mask; TT always requests 512 blocks and then validates by `selected <= pos`. `QSA_AUTODEBUG.md` documents the discrepancy and an actual-weight CPU control showing duplicated-tail semantics can fall below the 0.995 PCC bar.
  Affected path:
  QSA prefill/decode selected-token mask under short or underfilled contexts.
  Control or comparison:
  The repaired device case at length 33 passes PCC 0.99679226, but no exact selected-token multiplicity control is present.
  Likely subsystem:
  QSA indexer/top-k/tail mask construction.
  Investigation performed:
  Compared HF source, TTNN source, `QSA_AUTODEBUG.md`, `AUTOFIX.md`, and final PCC logs.
  Resolution:
  more-work-needed

- Observed anomaly:
  Earlier watcher run found a tiled-index embedding NOC/CB overflow.
  Evidence:
  `watcher_embedding_failure_20260826_1244/pytest.log` reports a 4096-byte NOC read into a 512-byte scratch circular buffer in `embedding_ind_tilized.cpp`; `AUTOTRIAGE.md` provides the source/CB ledger.
  Affected path:
  QSA RoPE/cache embedding index lookups.
  Control or comparison:
  Current `_embedding_tiled_output` converts indices to row-major before `ttnn.embedding`; `watcher_qsa_fix_20260826_1251` and `watcher_final` pass. Grepping final/focused watcher artifacts found no `NOC transaction`, `overflows`, `TT_THROW`, `fatal`, `sanitize`, or `embedding_ind_tilized`.
  Likely subsystem:
  TTNN embedding tilized-index kernel selection, with a model-scope row-major workaround.
  Investigation performed:
  Read AUTOTRIAGE/AUTOFIX and audited final watcher logs/kernel names.
  Resolution:
  fixed / controlled

- Observed anomaly:
  QSA Tracy one-replay decode host timing is much slower than the final host-timing log.
  Evidence:
  `tracy/layer3_qsa/decode_provenance.log` prints `traced_decode_ms=271.951550 decode_replays=1`; `perf_host_timing.log` prints `traced_decode_ms=11.247692 decode_replays=10`.
  Affected path:
  Performance reporting.
  Control or comparison:
  The profiler run uses profiler flush/readback around a one-replay signposted device window; the README reports host timing from a separate non-profiled ten-replay run and device-op time from signpost-filtered `tt-perf-report` CSV.
  Likely subsystem:
  Profiling overhead / provenance interpretation.
  Investigation performed:
  Compared provenance logs, host timing log, and `tt-perf-report` CSV/text summaries.
  Resolution:
  controlled

## Scope Inspected

- Goal/skill paths:
  `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/stage-review/SKILL.md`; `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/functional-decoder/SKILL.md`; `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths:
  `doc/functional_decoder/README.md`; `doc/functional_decoder/work_log.md`; `doc/context_contract.json`; `doc/functional_decoder/correctness.log`; `long_context*.log`; `perf_host_timing.log`; `targeted_trace_pcc.log`; `two_user_prefill_decode.log`; `static_checks.log`; `AUTODEBUG.md`; `AUTOFIX.md`; `QSA_AUTODEBUG.md`; `AUTOTRIAGE.md`; `watcher_*` pytest/watcher/kernel-name artifacts; all `doc/functional_decoder/tracy/*/*_perf_report.{txt,csv}`, `*_provenance.log`, and `*_ops.csv` by listing and summary aggregation.
- Code paths:
  `tt/functional_decoder.py`; `tt/model_config.py`; `tests/test_functional_decoder.py`; `tests/test_functional_decoder_perf.py`; `tests/harness.py`; `tests/conftest.py`; HF reference source `transformers/models/qwen4_exp/modeling_qwen4_exp.py` and local checkpoint `config.json`.
- Commands run:
  Read the three skill files with `cat`; inspected repo scope with `git status --short --branch` and `git rev-parse HEAD`; listed artifacts with `find`; read docs/code/logs with `nl`, `sed`, `head`, `tail`, and `cat`; searched code/evidence with `rg`; used small read-only Python scripts to parse local HF config, current test names, PCC evidence, and `tt-perf-report` CSV device-time totals. I did not run pytest, Tracy, tt-smi, hardware reset/list, TT device open, servers, or any device-facing command.

## Residual Risk

- This was an artifact/code review only, by instruction. I did not rerun hardware correctness, watcher, or profiler commands.
- I aggregated the `tt-perf-report` CSVs and checked human-readable summaries, but did not manually inspect every row of every large ops CSV.
- The PLE n-gram table is excluded from the device decoder boundary with a recorded physical-size reason. That boundary appears reasonable for this functional layer stage, but downstream full-model work must preserve and revalidate the caller-prepared PLE embedding contract.
