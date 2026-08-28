# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The 3.878x full-model decode gap is measured but not closed or proven physically unavoidable.
  Evidence: The retained layer medians rederive to `35*2.082220 + 1*2.613159 + 12*3.027236 = 111.817691 ms/token`; subtracting those medians from the three reduced full-path windows gives the reported 2.015389 ms endpoint estimate and a 113.833080 ms lower bound. The two final full runs rederive to a 441.444494 ms/token midpoint, or 3.878x that bound. `full_model_performance.json` attributes only the *last* replay to 226.302 ms cache-control/DMA submission, 95.224 ms route-read/device stall, 13.664 ms PLE service, and 0.547 ms trace submission; this totals 336.834 ms and is neither an average over the measured 126 tokens nor a complete reconciliation to the 451.733 ms/token end-to-end result. `final_cold_warm_chunked_prefill.xml` records bytes and source-pack time but no H2D time/bandwidth, overlap, exposed stall, or synchronization split. No current owner-only/deferred-DMA candidate tests batched/coalesced miss service, compatible per-rank parallel service, deeper stable staging, queue/event overlap, or supplies an exact runtime/API blocker. The inherited `threaded_rank_h2d` result used the superseded both-rank, per-miss-fenced path and does not reject those candidates for the selected path.
  Why this matters: The goal explicitly requires closing a gap above 10–15% and requires cold/warm store/cache/lookup, host/H2D, overlap/stall, TTFT, and decode accounting. Calling 327.611 ms/token of excess latency “physically required” from one last-token timing sample is an unsupported dismissal of the dominant stage cost.
  Required next step: Record per-token or aggregate decode-only cache regimes (cold/warmed-realistic/hit-heavy/adverse) with misses, host bytes/time, owner H2D bytes/time/effective bandwidth, D2D time, route/device completion, overlap, exposed stall, and synchronization boundaries. Establish a physical/runtime lower bound, then test exact compatible batching/coalescing, stable staging/parallel transfer, and overlap candidates, or retain minimal repro/blocker evidence for each material unavailable mechanism. Re-run full 128+128 token-out and accuracy/state gates for the selected result.

- P1: The terminal LM head is a material untuned BF16 interleaved-DRAM path.
  Evidence: All three `tracy_reduced_decode/*/token_out_report.csv` files show four vocabulary-split `MatmulDeviceOperation` rows of shapes `32 x 2560 x 32768` (three) and `32 x 2560 x 25856` (one), with BF16 weights, HiFi2, `DRAM Sharded=False`, interleaved DRAM input, and `Inner Dim Block Size=2`. Their summed device times are about 1.618/1.607/1.611 ms, respectively 40.35%, 36.50%, and 30.07% of the corresponding 4.009/4.402/5.358 ms reduced token-out windows. `tt/model.py::_load_endpoints` hard-wires BF16 `LazyWeight`, BF16 `LMHead1DConfig`, HiFi2, and interleaved DRAM. `optimization_matrix.csv` has no LM-head dtype/fidelity, DRAM-sharded layout, split-width, grid, block-geometry, or output-layout candidate.
  Why this matters: Embedding/norm/LM-head/logits/sampling optimization is binding stage scope. The optimize contract specifically requires DRAM-sharded LM-head work when an LM head is present and requires a precision-locked geometry search for a material `in0_block_w<=2` matmul or a precise blocker.
  Required next step: Run a real-weight, same-trace LM-head matrix covering at least the intended reduced weight dtype/fidelity, DRAM-sharded/vocab-sharded weight layouts, split width/core grid, larger legal K blocks/output subblocks, and sampler-facing output layout. Report row and full token-out latency plus PCC/top-k/readiness results, and retain exact L1/divisibility/op-contract failures for rejected candidates.

- P1: Profiler and performance accounting do not meet the required optimized-full-model evidence contract.
  Evidence: Every retained `tt-perf-report` command in `profiler_provenance.txt` uses `--no-advice`. Each `*_table.txt` is CSV-mode status/warning chatter (“Writing CSV output...”), not a rendered operation table. There is no warmed-prefill profiler capture, even though `test_reduced_full_model_profile_window` implements `QWEN38_FULL_MODEL_PROFILE_MODE=prefill`. There is no `doc/optimized_full_model/perf_summary.json`, no bytes/bandwidth theoretical roofline, and no same-run reconciliation of theoretical, device, and end-to-end decode. The 113.833 ms bound combines prior standalone layer medians with endpoint residuals from separate profiler runs; it is useful diagnostic arithmetic but is not the required same-run roofline/device/e2e accounting. The report also leaves actionable classifications hidden by `--no-advice`; the CSVs already expose the material non-sharded BF16 LM-head rows.
  Why this matters: Advice-enabled human-readable prefill/decode reports and `perf_summary.json` are mandatory for a non-vLLM complete-model optimization stage. Without them the claim that applicable optimization advice is exhausted cannot be reviewed.
  Required next step: Reprocess or recapture representative warmed prefill and decode with advice enabled, preserve a real rendered table separately from CSV console output, classify and test applicable advice, and write `perf_summary.json` with same-workload theoretical roofline, device time, end-to-end time, and named limitations.

- P2: A faster real-weight GDN precision candidate remains rejected without an accepted blocker.
  Evidence: `../optimized_multichip_decoder/optimization_matrix.csv` records layer-0 `gdn_bfp8_lofi` at real-checkpoint PCC 0.99977404 prefill/0.99986905 decode and 157.846770 ms prefill/1.987174 ms decode. The retained default is 158.877355 ms/2.082220 ms, so the candidate wins both target metrics and decode is 4.56% faster. Its only layer-0 disposition is `rejected_baseline_pcc_change`; both PCCs exceed the 0.995 real-weight bar. The layer-1 form is slower and need not be selected. The current stage merely inherits the ledger and states that no rejected policy was introduced; it does not run a targeted 35-GDN-layer full-model candidate or show a model-visible failure.
  Why this matters: The goal says no rejected faster policy, and the optimize/stage-review contracts do not allow a slower higher-fidelity default when a faster lower-fidelity candidate passes real target-model evidence and has no trace/runtime, full-model latency, accuracy, or exact op-contract blocker.
  Required next step: Treat this as a targeted layer-kind exception rather than a broad datatype sweep. Run the layer-0/GDN policy through the traced full-model token-out, required accuracy/qualitative/state gates, and final default reproduction; keep it if the full path wins, or record the concrete model-visible correctness/runtime/latency blocker that earns rejection.

- P2: Final artifacts are not bound to the current dirty source snapshot.
  Evidence: `tt/model.py` has mtime `2026-08-28 02:06:07 -0400`, later than every final JUnit/JSON artifact (latest `final_static_contracts.xml` at 02:00:10), the profiler captures, and even README/work-log generation at 02:05:31. `tests/test_full_model.py` is also newer than the 01:31:29 prefill artifact. Provenance records only base commit `a38187012fbc55714487c3d53a78a848809c8cf1`, not hashes of the dirty source used for each run. The current JSON does contain several current-diff metric fields, so this may be a late no-op or evidence-only save, but the retained artifacts cannot prove that.
  Why this matters: The review contract rejects stale evidence, and this live worktree has no commit identity that ties the final results to the implementation being proposed.
  Required next step: Record hashes of the stage-owned dirty source/tests/contracts and rerun the affected final static, correctness, performance, watcher, and profiler gates after the last semantic source change; alternatively provide a content-hash audit proving the late touches were non-semantic and bind every retained run to the matching source snapshot.

## Other Concerns

- The headline arithmetic is reproducible: baseline 644.562 ms/token versus the two-run mean 441.444 ms/token is a 31.51% reduction, and the reported capacity sums are exact (`10,303,149,144` bytes/device at batch-1 max context and `23,691,657,304` at batch-32/context-4096). The two optimized token-out runs span 431.156–451.733 ms/token, a 4.66% spread relative to their mean; calling the two-point arithmetic mean a “conservative midpoint” is imprecise, and more warmed repetitions should support a stable headline once remediation is complete.
- The before/after warm-TTFT baseline is imported from `doc/full_model/cold_warm_chunked_prefill_final.xml`, whereas the stage-local baseline covers the combined 128+128 workload. The harness appears like-for-like, but dirty-source hashes are needed to make that comparison auditable.
- Direct inspection of the generated text supports the qualitative review: AIME and the shared outputs are coherent/non-mechanical; the explanation and coding samples, like their HF controls, stop during visible reasoning at the fixed 128-token limit. This is an honest limitation, but these two prompts do not demonstrate final-answer helpfulness.
- The selected full-vocabulary greedy path is semantically valid and on device. Its measured 0.535–0.537 ms sampling window is not dominant in a 431–452 ms/token path, and the 0.645 ms versus 0.905 ms A/B compares two correct greedy implementations. I found no malformed force-argmax result.
- Source inspection supports CQ0 exactness: front readback is a completion boundary, owner H2D/owner D2D/peer-zero D2D/index publication/back traces remain ordered on CQ0, and the final compact token read completes the tail. The final accuracy and repeated generation runs are consistent with this. The report should nevertheless distinguish enqueue time from completed transfer time throughout; `h2d_seconds` explicitly measures enqueue only.

## Hard-Check Gaps

- The scripted gate did not require advice-enabled `tt-perf-report`, a rendered report table, warmed-prefill profiling, theoretical roofline reconciliation, or `perf_summary.json`; all are mandatory skill artifacts and are missing or malformed here.
- The stage-local watcher evidence is only JUnit XML with no retained console/stderr log. It proves the pytest node passed but does not independently prove the claimed absence or classification of warnings. The known generic active-trace allocation warning was controlled in the prior full-model stage with `TT_METAL_TRACE_ALLOC_TRACKING=1`, but no stage-local tracker rerun binds that control to the current owner-only/deferred-DMA source.
- Host-cache hard checks accept aggregate bytes and a single last-token timing sample; they do not require decode-only average/p50/p95 host/H2D/overlap/stall accounting or an adverse locality regime.
- No gate checks that final evidence timestamps/content hashes postdate the current dirty implementation.

## Anomaly Ledger

- Observed anomaly: The full token-out path is 3.878x its reported stack-plus-terminal lower bound, yet the remainder is labeled physically required.
  Evidence: 441.444494 ms/token measured mean versus 113.833080 ms/token derived lower bound; only last-token 226.302 ms cache submit and 95.224 ms route/device stall are retained.
  Affected path: Full 48-layer traced batch-1 token-out decode and exact expert/PLE host boundary.
  Control or comparison: Nonblocking trace submission, owner-only H2D, and deferred CQ0 completion improve 644.562 to 431.156/451.733 ms/token, but no selected-path transfer/overlap family or physical bandwidth lower bound closes the remainder.
  Likely subsystem: Exact expert cache lookup/packing/staging/DMA orchestration and mandatory route completion.
  Investigation performed: Recomputed the layer bound and final mean; inspected timing JSON, cold/warm XML, current cache/trace source, and inherited candidate matrix.
  Resolution: more-work-needed

- Observed anomaly: The largest endpoint matmul group is BF16, non-DRAM-sharded, and fixed at `in0_block_w=2` with no candidate ledger.
  Evidence: Four LM-head rows total about 1.61 ms and 30–40% of each reduced profile window; `tt/model.py` hard-wires this policy.
  Affected path: Terminal vocabulary projection for every prefill/decode/token-out invocation.
  Control or comparison: No stage-local dtype/fidelity/layout/geometry A/B exists.
  Likely subsystem: `LMHead1D` configuration and endpoint layout/program policy.
  Investigation performed: Parsed all three filtered CSVs and inspected `_load_endpoints` plus `optimization_matrix.csv`.
  Resolution: more-work-needed

- Observed anomaly: A faster layer-0 GDN BFP8/LoFi candidate passed real-weight PCC but was rejected for changing baseline PCC.
  Evidence: 1.987174 versus 2.082220 ms decode and PCC 0.99986905 in `candidate_gdn_bfp8_lofi.xml`/the inherited matrix.
  Affected path: The 35 ordinary GDN layers if applied as a layer-kind exception.
  Control or comparison: Layer 1 is slower under the same broad policy; no layer-0-only full-model accuracy/token-out control exists.
  Likely subsystem: GDN projection fidelity selection.
  Investigation performed: Read the candidate XML, confirmed `weights=real-checkpoint`, and compared the final repeated default rows.
  Resolution: more-work-needed

- Observed anomaly: Prior watcher runs emitted the runtime's generic active-trace allocation warning.
  Evidence: `../full_model/AUTODEBUG_TRACE_ALLOC.md`, `reduced_split_trace_alloc_tracker.xml`, and `full48_tokenout_trace_alloc_tracker.xml` document and test it.
  Affected path: Multi-live segmented decode trace allocation/lifetime.
  Control or comparison: Both inherited tracker runs passed; source inspection shows the new immutable-zero buffers are constructed before model trace capture, while replay adds no allocations. The current stage's watcher JUnit passes, but its console and a current-source tracker run are absent.
  Likely subsystem: TT trace allocator/lifetime tracking.
  Investigation performed: Read the prior AutoDebug/review evidence and current cache/trace construction/replay paths; searched current artifacts for warning text.
  Resolution: controlled for the inherited path; current-source evidence gap remains

- Observed anomaly: Two shared qualitative samples end before a final response.
  Evidence: Direct `qualitative_shared_suite_final.json` inspection shows explanation and coding stop in reasoning at 128 tokens.
  Affected path: Fixed-length qualitative demonstration only.
  Control or comparison: HF controls stop at the same point; TT summarization completes correctly and all TT outputs remain coherent/non-degenerate.
  Likely subsystem: Generation-length budget, not TT decode correctness.
  Investigation performed: Read rendered metadata and HF/TT completions directly and compared the manual review.
  Resolution: controlled

- Observed anomaly: Current source timestamps postdate all final evidence.
  Evidence: `tt/model.py` mtime 02:06:07 versus latest final JUnit 02:00:10 and report files 02:05:31; no dirty-source hash manifest exists.
  Affected path: All claimed current-source final gates.
  Control or comparison: The retained performance JSON includes current metric fields, suggesting at least part of the diff was present by 01:51, but it cannot identify the later edit.
  Likely subsystem: Artifact provenance/stage handoff.
  Investigation performed: Compared `stat` timestamps, git diff/HEAD, artifact metadata, and JSON fields.
  Resolution: more-work-needed

## Scope Inspected

- Goal/skill paths: the supplied stage-7 contract; `.agents/skills/{stage-review,multichip,optimize,host-weight-cache,tt-device-usage,full-model,tt-enable-tracing,qualitative-check}/SKILL.md`.
- Artifact paths: all requested files under `doc/optimized_full_model/`, including README/work log, both CSV summaries, qualitative review and generated HF/TT completions, final accuracy/state/capacity/watcher JUnit XMLs, `full_model_performance.json`, profiler provenance, all three raw/filtered Tracy report families, and compressed raw CSVs; `doc/context_contract.json`; `doc/host_weight_contract.json`; the prior full-model trace-allocation investigation; and `doc/optimized_multichip_decoder/{README.md,optimization_matrix.csv,candidate_gdn_bfp8_lofi.xml,final_default_perf_count7.xml}`.
- Code paths: complete git diff for `tt/host_weight_cache.py`, `tt/multichip_decoder.py`, `tt/model.py`, and touched tests; focused inspection of cache load/validation/close, segmented trace capture/replay, full-model trace orchestration, endpoint loading, fallback metrics, performance harness, and qualitative harness.
- Commands run: read-only `git status/diff/rev-parse`, `rg`, `find`, `stat`, `sed`, `jq`, `sha256sum`, `xz -t`, and small local Python XML/CSV/JSON arithmetic parsers. The published profiler hashes match all raw and `.xz` files and all compressed files pass integrity testing. No server, TT device, reservation, reset, pytest hardware run, watcher, Tracy capture, or vLLM experiment was started.

## Residual Risk

- I found no direct evidence of wrong cache ownership, stale token/position feedback, full-logits host fallback, malformed greedy sampling, context reduction, memory-accounting error, or qualitative degeneration. AIME prefill is 100% top-5/top-100; 99-row teacher-forcing is 100% top-5/top-100; the split/mixed-state suite is 5/5; batch-32/context construction is 2/2; and the final static suite is 34/34.
- Because this review is deliberately hardware-read-only, CQ0 completion behavior and immutable-zero device contents were assessed from source plus retained end-to-end results. A current-source trace-allocation-tracked run and retained watcher console remain the appropriate controls after remediation.
- Advertised context 262,144 is preserved and the capacity arithmetic checks, but the stage evidence is construction/static-plan coverage plus inherited near-max execution rather than a fresh full advertised-context run; this is acceptable only while the inherited source contract remains unchanged and is hash-bound.
- The final speedup is large enough that the two-run variance does not erase it, but stable p50/p95 claims should wait for repeated current-source measurements after the required performance changes.
