# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The first review's material-candidate provenance requirement remains incomplete.
  Evidence: The first review explicitly required every material candidate/rejection to be tied to a source/test hash or checkpoint SHA and an exact command (`stage_review.md:12-15`) and separately identified decode-QKV/reshard, conv1d, activation-fold, mixed-tap, SDPA chunk-size, repeated-KV, and intermediate-GQA evidence as missing (`stage_review.md:30-33`). Those candidates still appear only as prose rows in `work_log.md:73-89`; repository-wide searches find their PCC and latency values only in `work_log.md`, `graph_inventory.md`, and the first review. No retained candidate runner transcript, source snapshot/hash, checkpoint, minimal repro, or profiler artifact establishes the claimed results. `provenance_manifest.md:59-69` ties the final group-major result to current artifacts and cites the first review's independent sums for the immediately preceding per-group graph, but it explicitly leaves all earlier exploratory rewrites documented only in the work log. The final-source evidence is strong and current, but that closes only the final-gate/stale-artifact portion of the original provenance finding.
  Why this matters: Exhausting applicable graph-fusing patterns and proving that rejected rewrites were actually tried are original stage gates. The source and final profiles prove that the selected graph is correct and faster, but they cannot independently establish the claimed PCC failure, NaN, op-contract blocker, or slower adapted result for the material alternatives. This is the same prose-only evidence gap the first review required the stage to remediate, not a newly invented evidence format.
  Required next step: Recover and retain the original candidate stdout/stderr, profiler output, and corresponding source snapshots/hashes if they still exist; otherwise rerun the material candidates from reversible source patches. Record the exact command, source/test SHA (or checkpoint SHA), correctness result, and warmed comparable latency or exact adapted op-contract failure for the GDN activation/softplus and mixed-tap candidates, specialized decode-QKV plus required reshard, native-conv1d adapted attempt, SDPA chunk sweep, and repeated/native-GQA progression. Preserve a source-linked artifact for the immediately preceding per-group expert-down candidate as well, rather than relying only on numbers copied into a review report. Then update the candidate table to link each retained artifact.

- P2: The profiler provenance manifest claims report-generation evidence that the retained transcripts do not contain.
  Evidence: `provenance_manifest.md:53-57` says every `*_tracy.log` records report generation. Each of the six logs records the `python -m tracy` command, signposts, selected pytest node, passing result, and generated raw `ops_perf_results` path, but ends after Tracy raw-report generation; none contains the `tt-perf-report` command, the raw-CSV copy command, or hashes of the filtered/stacked outputs. The generic report command is documented in `work_log.md:157-166`, but not captured in the claimed transcript. Independent CSV analysis does confirm that every filtered row exactly matches the raw row between its start/end signposts and that all six stacked totals reconcile within rounding, so this is a provenance/documentation defect rather than evidence that the reported totals are wrong.
  Why this matters: Complete, accurate provenance was part of the first-review remediation. A manifest must not claim primary command evidence that is absent, especially when the retained Tracy scratch directories were removed.
  Required next step: Re-run only the offline `tt-perf-report` transformation over the retained raw op CSVs, capture the six exact commands and exit statuses in a report-generation transcript, and add hashes for the filtered and stacked CSV/PNG outputs; or correct the manifest to state precisely which steps are evidenced and provide an equivalent reproducible mapping. No TT hardware rerun is needed for this item.

## Other Concerns

- The two other first-review findings are genuinely closed. `FusedDecoder._routed_experts` has one packed gate/up call and exactly one group-major A-sparse/B-dense down call (`tt/fused_decoder.py:252-327`) with `is_input_a_sparse=True` and `is_input_b_sparse=False`. The binding contract independently confirms output `[A,E,M,N]` for this mode. `graph_inventory.md:49-58` and `work_log.md:73-89` assess indexed mode; the TTNN binding independently confirms that `indices` requires B-sparse mode, so it cannot apply to the final A-sparse down call.
- The current source/test/context hashes exactly match `provenance_manifest.md`. Source mtime is 16:24:44; the final raw runs span 16:28:00 through 16:46:54; the hash capture is 16:48:18. This closes the stale-final-source concern for the retained final gates.
- The retained final logs directly show 35 non-long tests with trace-allocation tracking, seven advertised-context tests, nine repeated trace/determinism cases, and 35 watcher-enabled tests, all with exit code 0. Final real-weight prefill/traced-decode PCC is 0.99871039/0.99997759 (layer 0), 0.99909765/0.99989104 (layer 1), and 0.99656421/0.99974990 (layer 3), all above 0.995.
- Like-for-like host timing is now valid. Independent median recomputation from three 10-replay samples gives functional to fused prefill/decode in milliseconds: layer 0, 24.117971/5.732531 to 19.635788/4.258239; layer 1, 27.575097/6.699843 to 23.124805/5.201074; layer 3, 408.279154/11.339128 to 45.512445/6.417564. The reductions are 18.584%/25.718%, 16.139%/22.370%, and 88.853%/43.403%.
- All six filtered reports reconcile row-for-row with their raw signpost windows. Independently recomputed functional to fused device sums in microseconds are layer 0, 23790.432/5395.860 to 19341.630/4223.077; layer 1, 27269.015/5922.392 to 22769.297/4726.134; layer 3, 408094.485/11080.636 to 45277.332/6252.131. Against the immediately preceding values recorded by the first independent review, the final graph is lower by 3207.550/252.371 us, 3206.737/250.709 us, and 3208.588/262.463 us respectively.
- Source inspection and measured-call guards show no runtime `torch`, `ttnn.from_torch`, `ttnn.to_torch`, `ttnn.as_tensor`, or explicit `ttnn.reshard` fallback. The raw signpost windows contain only device-op rows. The final QSA prefill profile contains one 17.331 ms SDPA row and none of the old 38.7 ms reshape or 15-17 ms permute rows.
- Watcher and Tracy were captured in separate commands, satisfying `tt-device-usage`. The watcher pytest transcript passes all 35 selected nodes; the retained 288-line watcher log has no assert/fatal/hang/stuck/timeout/NoC/kernel/sanitizer signature and ends with an all-idle dump and clean device-0 detach.

## Hard-Check Gaps

- Stage evidence with `.log`, `.csv`, or `watcher_final_v2/generated` paths is ignored by repository rules. A later checkpoint must force-add or otherwise explicitly track the required evidence; ordinary `git add` will omit it. The checkpoint remains correctly deferred until after a clean review.
- The post-run `tt-smi -ls --local` result is present only in `reliability_summary.log`, not as raw command output. No recovery occurred and the watcher/device runs closed cleanly, so this is not by itself a stage blocker.
- The batch-32 gate proves execution and output shape for all representative layers; exact per-user page/cache-slot assertions are exercised at batch 2, not batch 32.

## Anomaly Ledger

- Observed anomaly: Material candidate PCC, failure, and latency claims remain prose-only.
  Evidence: `work_log.md:73-89`; `stage_review.md:12-15,30-33`; `provenance_manifest.md:59-69`; no matching candidate artifacts elsewhere in the stage root.
  Affected path: Graph-exhaustion and best-correct-candidate provenance.
  Control or comparison: Current final-source gates and six final profiles are primary and source-linked; the first review independently measured the immediately preceding graph but did not preserve its source hash/raw CSVs.
  Likely subsystem: Experiment artifact retention.
  Investigation performed: Searched all stage Markdown/log/CSV artifacts for the candidate metrics and inspected manifest hashes, mtimes, commands, and the first review's required next step.
  Resolution: more-work-needed.

- Observed anomaly: The manifest says Tracy transcripts contain report-generation commands, but they do not.
  Evidence: `provenance_manifest.md:53-57`; all six `tracy/*/*_tracy.log` files end after raw Tracy report generation.
  Affected path: Six filtered and stacked profiler report provenance.
  Control or comparison: Independent raw-window parsing reproduced every filtered row and total exactly; stacked totals reconcile within 0.037 us from CSV rounding.
  Likely subsystem: Documentation/provenance assembly.
  Investigation performed: Inspected all six transcripts, raw op CSVs, filtered reports, stacked CSVs/PNGs, and generic command documentation.
  Resolution: more-work-needed.

- Observed anomaly: Watcher setup emits an active-trace allocation warning.
  Evidence: `final_watcher_pytest.log` records the exact allocator warning during the layer-0 real-weight trace path.
  Affected path: Trace buffer lifecycle.
  Control or comparison: The final-source aggregate run with `TT_METAL_TRACE_ALLOC_TRACKING=1` passes all 35 selected tests, and repeated trace replay plus watcher runs complete cleanly.
  Likely subsystem: Generic host allocator warning while a trace object exists.
  Investigation performed: Compared the warning against the trace-allocation transcript, stress run, watcher output, and trace release points in tests/source.
  Resolution: controlled for this stage.

- Observed anomaly: Successful runs emit nanobind leak diagnostics and single-visible-chip/custom-board warnings.
  Evidence: Final raw runner and profiler transcripts; `ttenv.sh` pins `TT_VISIBLE_DEVICES=0` and the 1x1 context contract targets chip 0.
  Affected path: Process teardown and device discovery, not decoder output or measured windows.
  Control or comparison: The completed functional baseline emits the same nanobind class; every final command exits 0, closes the device, and the watcher detaches cleanly.
  Likely subsystem: Python binding reference accounting and intentional single-device visibility.
  Investigation performed: Compared final and functional transcripts, environment pinning, watcher teardown, and context target.
  Resolution: controlled for this stage.

## Scope Inspected

- Goal/skill paths: supplied fused-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: all files under `models/autoports/qwen_qwen3_8_flash_next/doc/fused_decoder`; `doc/context_contract.json`; functional-decoder source, tests, host timing logs, and six baseline raw/filtered Tracy CSVs.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`, `tests/test_fused_decoder.py`, `tests/test_fused_decoder_perf.py`, `tests/test_functional_decoder.py`, `tests/test_functional_decoder_perf.py`, `tests/harness.py`; TTNN sparse-matmul binding and A-sparse/indexed tests; representative GPT-OSS sparse-matmul usage.
- Commands run: read-only `git status/branch/rev-parse/check-ignore`, `find`, `stat`, `sha256sum`, `wc`, `sed`, `tail`, `head`, `rg`, `diff`, and small read-only Python CSV/statistics analyses. No pytest, Tracy capture, watcher, TT device, `tt-smi`, reset, reservation, server, vLLM, or later-stage command was run.

## Residual Risk

- Performance uses zero inputs and sparse-matmul reports `active=?`; the evidence proves latency/topology for this harness but not real routed-token expert utilization.
- Long-context tests prove full public execution and shape/capacity semantics, while real-weight numerical PCC is measured at a shorter non-aligned prompt plus traced decode.
- The full PLE n-gram table remains the accepted caller-preprocessing boundary from the functional stage; the fused measured path still includes PLE projection, convolution, gate, and state.
- This review is inspection-only. The candidate-provenance finding requires artifact recovery or owner-run hardware experiments, followed by another fresh independent review.
