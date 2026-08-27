# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The shared MoE down path contains an unassessed, material structural-fusion candidate.
  Evidence: `tt/fused_decoder.py:300-334` loops over each 32-token group, slices the hidden tensor and sparsity mask, launches one `sparse_matmul` per group, and concatenates the results. The TTNN `sparse_matmul` contract in `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1105-1111` explicitly supports the adapted A-sparse/B-dense shapes `[A,E,M,K]`, `[1,E,K,N]`, sparsity `[1,1,A,E]`, and output `[A,E,M,N]`. Therefore the current 128-token tensors can plausibly be transformed to group-major `[groups,E,32,K]` and evaluated in one call with the existing `[1,1,groups,E]` mask before restoring token order. The retained prefill CSVs make this material: the current four-call down region is 5923.132 us (26.27% of layer 0), 5922.204 us (22.80% of layer 1), and 5916.734 us (12.20% of layer 3). In each case it consists of about 416 us of slices, 1743-1751 us of per-call `nnz=None` unary work, 1999-2004 us across the four sparse matmuls, and 1751-1759 us for the final concat. Neither `graph_inventory.md` nor the candidate table in `work_log.md:65-79` assesses this family.
  Why this matters: The original contract requires every applicable dedicated/structural/adjacent fusion to be tried and retained only when faster, and requires no unnecessary measured-path movement. A documented operator mode that may eliminate three sparse dispatches, repeated runtime sparsity work, slices, and a 1.75 ms concat cannot be left untried while claiming the graph is exhausted or the default is fastest.
  Required next step: Implement the adapted group-major candidate in isolation (including the correct `is_input_b_sparse=False` mode), prove exact token/expert ordering and real-weight PCC for representative layers 0/1/3, and collect warmed prefill profiles. Keep it if correct and faster. If it fails, adapt legal shape/layout/program configuration and retry as required by `graph-fusing`; retain a minimal repro or measured slower result if rejected. Also assess whether the documented indexed/gather sparse mode can profitably avoid dense expert-group scans before declaring the MoE region exhausted.

- P1: Core correctness, context, trace-allocation, stress, watcher-suite, and candidate claims are not backed by retained primary runner output tied to the reviewed source.
  Evidence: The only fused correctness/context/reliability artifacts outside the device watcher dump are the 0.4-1.0 KiB `*_summary.log` files. They are manually condensed `Command:/Result:` statements, all have the same 16:15:44 timestamp, and contain none of pytest's test-node output, `PCCEVIDENCE` lines, warnings, environment, or source identity. No fused pytest/provenance/raw logs exist under the evidence root. The current `generated/test_reports/most_recent_tests.xml` records only one layer-3 perf test, not the claimed 35-test correctness, long-context, stress, or watcher runs. The retained watcher log is directly clean, but it only shows a watcher attach, one all-idle/blank-kernel dump, and detach; it cannot by itself establish that 35 fused tests passed. Likewise, all per-rewrite PCC/latency rejections in `work_log.md:65-79` are prose without retained runner output. Stale-path risk is real: final `fused_decoder.py` is timestamped 15:56:43 while layer-0/1 profiler tables are timestamped 15:48:26-27, and the current perf harness is timestamped 16:10:00 while five of six fused profiler tables predate it. The reports contain no source hash or runner provenance. The work log's statement that the later source change was QSA-only is a claim, not artifact provenance.
  Why this matters: PCC >= 0.995, deterministic trace replay, full/non-aligned advertised context, trace allocation, watcher cleanliness, and faster/correct rewrite selection are stage-critical gates. The review can verify that the test code is capable of exercising these paths, but it cannot verify that the reviewed final source produced the reported pass values. The stage-review standard explicitly treats prose-only evidence and stale artifacts as required work when they cover core metrics.
  Required next step: Retain unedited stdout/stderr (or equivalent structured result artifacts) for the final-source real-weight PCC suite, trace-allocation aggregate, repeated replay stress, watcher pytest run, and final long-context gates. Preserve the exact warning text and test node list. Tie every final profiler table and material candidate/rejection to the source/test hashes or a checkpoint SHA and exact command. Recollect stale profiler rows after the MoE disposition; retain the six signpost-filtered tables and enough provenance to verify their signpost windows.

- P1: The advertised warmed traced-decode host comparison is not like-for-like.
  Evidence: The committed functional baseline runner defaults to 10 replays (`tests/test_functional_decoder_perf.py:20`), and its retained `functional_decoder/perf_host_timing.log` explicitly prints `decode_replays=10` for 5.729530/6.700191/11.338447 ms. The fused result was run with `QWEN38_FUSED_PERF_DECODE_REPLAYS=100` (`work_log.md:122-127`) to produce 4.518391/5.454957/6.671424 ms. Both runners enqueue nonblocking traces and synchronize once after the loop, so changing 10 to 100 changes synchronization/queue amortization. `README.md:18-30` nevertheless describes the comparison as 100-replay traced decode. The direct one-replay device-op tables are comparable and their sums do verify 17.1%, 16.0%, and 41.2% device-time reductions, but they do not validate the claimed host reductions.
  Why this matters: Like-for-like warmed prefill and traced warmed decode before/after is an explicit acceptance requirement. The mismatch can inflate the reported host improvement, especially for the smaller GDN deltas.
  Required next step: Run functional and final fused paths with the same replay count, warmup, synchronization placement, inputs, max sequence length, and environment; retain the raw output and preferably repeated samples. Replace the host before/after table and all derived speedup claims with those comparable results. Keep the already verified device-op comparison separate.

## Other Concerns

- The filtered device tables themselves are internally consistent: independently summing `Device Time` reproduces fused prefill/decode totals of 22549.180/4475.448 us (layer 0), 25976.034/4976.843 us (layer 1), and 48485.920/6514.594 us (layer 3), all below the retained functional tables. This supports a real whole-graph device improvement but does not close the missing-candidate or provenance findings.
- Source inspection found no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, `ttnn.as_tensor`, or explicit `ttnn.reshard` call in `FusedDecoder`. The correctness wrappers replace the functional test module's constructor symbol, `from_state_dict` constructs via `cls`, and inherited public methods dynamically dispatch the fused overrides. I found no functional fallback switch.
- All current stage evidence with `.log`, `.csv`, or `watcher_final/generated` paths is ignored by repository rules. Before the later checkpoint commit, the owner must explicitly include required retained evidence (or adopt tracked extensions/paths); ordinary `git add` will omit it.

## Hard-Check Gaps

- There is no retained raw fused host-performance log comparable to `functional_decoder/perf_host_timing.log`.
- There is no retained pytest output for the 35-test trace-allocation/watcher claims, the 9-run stress claim, or the final three QSA advertised-context gates.
- The exact two warnings from the trace-allocation summary and the claimed active-trace watcher warning are absent, so their category and lifecycle impact cannot be audited.
- Candidate measurements/rejections other than the final QSA CSVs have no retained source-linked artifacts. In particular, decode-QKV/reshard, conv1d, activation-fold, mixed-tap, SDPA chunk-size, repeated-KV, and intermediate GQA decisions are prose only.
- The worktree is otherwise clean relative to the supplied stage scope: HEAD is `85b1099e34b` on `hous/qwen3.8-flash-next`, and only fused-decoder implementation/tests/docs are stage-owned uncommitted state. The required checkpoint commit is correctly deferred until after a clean review.

## Anomaly Ledger

- Observed anomaly: A generic active-trace warning is acknowledged but not quoted or retained.
  Evidence: `README.md:136-138`; `correctness_summary.log` reports two warnings without their text.
  Affected path: Trace capture/release and watcher/trace-allocation reliability evidence.
  Control or comparison: The README says a separate `TT_METAL_TRACE_ALLOC_TRACKING=1` run passed, but only its handwritten summary remains.
  Likely subsystem: Host trace allocator/lifecycle diagnostics.
  Investigation performed: Inspected all fused evidence files and searched for warning/active-trace text; no primary warning output exists.
  Resolution: more-work-needed.

- Observed anomaly: Five profiler tables predate the current performance harness, and layer-0/1 tables predate the final fused source file.
  Evidence: filesystem timestamps described in Required Work; no hash/provenance metadata accompanies the filtered reports.
  Affected path: Final-default performance and movement claims.
  Control or comparison: The QSA-only nature of the source edit is stated in the work log but cannot be independently derived from retained artifacts.
  Likely subsystem: Artifact provenance/stale report retention.
  Investigation performed: Compared source/test/report timestamps, hashes, worktree state, and retained report contents.
  Resolution: more-work-needed.

- Observed anomaly: Nanobind teardown leak diagnostics are disclosed.
  Evidence: `README.md:192-194`; the functional baseline's retained raw host timing log contains the same class of nanobind teardown diagnostics after all tests pass.
  Affected path: Process teardown, not measured decoder execution.
  Control or comparison: Completed functional baseline in the same environment.
  Likely subsystem: Python binding reference accounting.
  Investigation performed: Compared the disclosure against the functional baseline raw log and searched fused artifacts for runtime corruption signatures.
  Resolution: controlled for this stage; retain the fused raw log so the exact signature remains comparable.

## Scope Inspected

- Goal/skill paths: supplied fused-decoder contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: all files under `doc/fused_decoder`, including all six filtered and stacked Tracy CSVs and watcher output; `doc/context_contract.json`; functional decoder README/work log, host timing log, and six baseline filtered CSVs.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`, `tests/test_fused_decoder.py`, `tests/test_fused_decoder_perf.py`, `tests/test_functional_decoder.py`, `tests/test_functional_decoder_perf.py`; TTNN sparse-matmul binding contract and representative sparse-matmul tests/model uses.
- Commands run: read-only `git status/branch/rev-parse/log/check-ignore`, `find`, `stat`, `sha256sum`, `wc`, `sed`, `rg`, and small read-only Python `csv` analyses that recomputed report sums and the expert-down region. No server, TT device, pytest, Tracy, watcher, reset, reservation, or vLLM command was run.

## Residual Risk

- The batch-32 gate validates successful execution and output shape, while exact per-user page/cache-slot assertions are exercised at batch 2. This is reasonable structural coverage but does not provide a batch-32 numerical oracle.
- Performance uses zero inputs and reports `active=?` for sparse matmul. The reports are valid latency/topology evidence for that harness, but routed-expert utilization and gains may differ with real activation routing.
- This review is inspection-only. Hardware confirmation required by the findings must be performed by the stage owner, followed by a fresh independent stage review.
