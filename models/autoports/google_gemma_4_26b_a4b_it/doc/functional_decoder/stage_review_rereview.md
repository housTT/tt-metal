# Stage Review

verdict: clean-pass

## Required Work

- None.

The three required findings in `stage_review.md` are fixed, and inspection of
the rest of the functional-decoder contract found no new stage-blocking issue.

## Original-Finding Closure

- P1, host-backed non-aligned tail construction: fixed. The current decoder is
  SHA-256 `1de6d3e39b2ba645f1d9f0de032178b12faa733b3d51ae4735e5ee33953cdf23`.
  `_fill_prefill_cache` now constructs each tail position with rank-2 INT32,
  row-major `ttnn.moreh_full` and passes it to both paged updates
  (`tt/functional_decoder.py:685-742`). The TTNN source independently confirms
  that `ttnn.full` builds a host vector and transfers it
  (`ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73`), whereas
  `ttnn.moreh_full` launches `ttnn::prim::full`, allocates a device tensor, and
  fills it with a device data-movement kernel
  (`ttnn/cpp/ttnn/operations/full/full.cpp:13-20`,
  `full/device/full_device_operation.cpp:24-48,65-68`, and
  `full/device/kernels/writer_full.cpp:13-68`). The rank guard requires more
  than one dimension, so `(1, 1)` is intentional. The runtime audit now also
  forbids literal `ttnn.full(` over the complete local hot-path method list
  (`tests/test_functional_decoder.py:1801-1823`). Direct inspection of the
  delegated expert prefill call graph found no runtime torch conversion or
  host-backed creation in the invoked functions. The work log records the
  focused bounded-tail pass and the current-source default suite also includes
  that test.

- P1, bounded sliding decode was eager rather than traced: fixed. The current
  regression captures the complete `decode_forward` with
  `cache_position_modulo=1024`, updates stable hidden/RoPE/current-position
  buffers, and executes 1,104 trace replays
  (`tests/test_functional_decoder.py:1312-1454`).
  `bounded_modulo_decode_across_wrap.json` is bound to the current
  decoder and test hashes and records PCC 1.0 against the eager unbounded
  control at positions 1023, 1024, 1025, and 1103. The complementary
  batch-32 A/B/A test captures `decode_forward` once and overwrites hidden,
  RoPE, current positions, independently permuted per-user page tables, and
  both nonzero K/V caches between replays
  (`tests/test_trace_mutable_buffers.py:168-263`). Both retained artifacts
  report eager-control PCC `[1.0, 1.0,
  1.0]`, A-repeat PCC 1.0, visibly different A/B outputs, private page pools,
  and every page-table row changing between A and B. Their mtimes are after the
  final source/test edits, and the work log records the producing run as two
  passing cases.

- P2, placeholder reproduction commands: fixed. The two v3 host-timing JSON
  files, `perf/README.md`, `work_log.md`, and `watcher/summary.json` contain
  executable pytest node IDs for the exact layer kind and batch. No placeholder
  remains in current reproduction/provenance fields; matches under the old
  `stage_review.md` are historical findings rather than active instructions.

## Full Contract Check

| Contract item | Re-derived evidence | Result |
| --- | --- | --- |
| HF architecture and meaningful modes | The checked-in HF config advertises 262,144 positions, hidden size 2,816, 30 layers, a 1,024-token sliding window, 25 sliding layers and 5 full layers. `num_kv_shared_layers=0`, so layers 0 and 5 cover the two meaningful attention computations. The implementation validates the real config and handles the full-attention K-as-V rule. | pass |
| Real weight boundary | `from_state_dict` consumes canonical HF keys and performs setup-time conversion. `real_weight_stats.json` inventories all 22 layer-0 and 21 layer-5 tensors at real shapes from checkpoint snapshot `4d7ae498...`. The current-source default run rewrote all three real-weight PCC artifacts. | pass |
| HF-vs-TTNN PCC >= 0.995 | Current-hash real-weight prefill/decode PCC is 0.999163/0.999739 for sliding, and 0.998457/0.999860 for both natural and HMA-shared full cache views. Batch-2 prefill is 0.998311/0.998659. Batch-32 traced decode is 0.999455/0.999860. Every boundary and long-attention row also clears 0.995. | pass |
| Paged cache semantics | Prefill and decode use paged fill/update/read APIs. Permuted page tables cover page/tile/window boundaries; full attention passes one atomic `PagedCacheGeometryOverride(128, 2)` to paged SDPA while fill/update use their required view keywords. The current host regression verifies the atomic override. Bounded prefill preserves slots 1..1023 while replacing wrapped slot 0. | pass |
| Non-aligned logical lengths | Full-layer boundary tests cover 1/31/32/33, 127/128/129, and 1023/1024/1025; sliding covers 1/31/32/33, 63/64/65, and 1023/1024/1025. Returned logical shapes and PCC pass. The 262143 capacity probes cover the long non-divisible case, and the bounded 1025 tail exercises the repaired device-only update path. | pass |
| Full 262144 context | `doc/context_contract.json` retains target and supported context 262144 with no reduction. Real-weight full-layer prefill completed at 262143 and 262144 for both kinds. Traced decode ran at current position 262143 for both kinds with rolled page tables, device-initialized 1 GiB full / 2 GiB sliding K+V histories, sentinel readback/preservation, finite output, and repeat PCC 1.0. The strict contract checker was rerun read-only and passed. | pass |
| Traced decode and determinism | Batch 1 and 32 replay artifacts exist for both kinds; HF PCC exceeds 0.995 and eager/repeat PCC is 1.0. The bounded-wrap trace and mutable-buffer A/B/A evidence close the stronger cache-mutation cases. Advertised-context repeats are also 1.0. | pass |
| No runtime host fallback | Runtime source and delegated hot helpers contain no torch, `from_torch`, `to_torch`, or `ttnn.full` call. The only newly needed index tensor is produced by a verified device operation. Test-side staging and PCC readback remain outside the measured decoder pass. | pass |
| Warmed performance evidence | The v3 current-source/current-test runs use exact batch-1 seq-1024 node IDs. Recomputed CSV totals are sliding 1242.489 ms prefill over 557 device ops and 3.0125 ms traced decode over 74 ops; full is 1243.618 ms over 557 ops and 3.206965 ms over 76 ops. All 1,264 derived global-call-count rows occur in the corresponding v3 raw reports. Raw host/device audits show 1294/1294 and 1298/1298 IDs with zero missing or extra IDs and three complete 74/76-op trace sessions. Host timings are 1243.350/3.112 ms and 1244.512/3.304 ms. These are correctly described as functional baselines, not improvement claims. | pass |
| Watcher | The retained raw watcher log hashes to `c34b2745cbf790028a4a9b17b26d0eb3ae99ba8009bcbdb450d66b610e4ead42`, matching the summary. Repeating the recorded fatal/error/assert/hang/deadlock scan found no match; the log shows clean attach, one completed dump, and detach. Its mtime is after the final source change, and the summary records two exact passing batch-1 trace node IDs. | pass |
| Default and static gates | The work log records 28 passed and 10 intentional opt-in skips in the current-source default suite, plus 14 passing focused host/audit checks. The skip count is exactly the two context, two capacity, two long-attention, and four perf cases with separately retained opt-in evidence. The stage owner records all applicable pre-commit hooks passing. This reviewer additionally reran the read-only strict context check, `git diff --check`, and JSON parsing; all passed. No C++/CMake file changed, so `AGENTS.md` does not require a build. | pass |
| Scope | Current git status changes only this autoport's functional decoder, functional test, and documentation tree. No optimized, multichip, full-model, vLLM, C++, CMake, or external-dependency change is part of this stage. | pass |

## Other Concerns

- The four required `perf/*/{prefill,decode}.csv` files are ignored by the
  repository-wide `*.csv` rule, and the watcher raw tree is ignored by the
  `generated` rule. They exist and were inspected, but the checkpoint must
  force-add them as the main-agent handoff already anticipates.

## Hard-Check Gaps

- Several compact matrix artifacts (`prefill_boundaries_*`, `trace_*`,
  `trace_mutable_buffers_*`, and `long_prefill_attention_*`) do not embed
  provenance objects. This is not required work here: the current default-run
  artifacts have post-edit mtimes and contents exactly matching their current
  producing tests, while the long-context artifacts are tied to the documented
  capture sequence and were not invalidated by the sole later source change.
  Embedding provenance in every small artifact would be useful but would not
  expose a concrete correctness gap.

- The default-suite and pre-commit terminal output is summarized in the work
  log rather than retained as a raw log. Current hashes in the rewritten PCC,
  bounded-wrap, and perf artifacts plus the static checks above corroborate the
  final source state. This review did not rerun pre-commit because hooks may
  rewrite files and reviewer mode is read-only.

## Anomaly Ledger

- Observed anomaly: A host-backed `ttnn.full` remained in the runtime bounded
  prefill tail after it had already been diagnosed in the context-test setup.
  Evidence: original `stage_review.md`; current source lines 685-742; TTNN full
  and moreh-full C++ implementations; current fallback audit.
  Affected path: Non-aligned prefill with `cache_position_modulo`.
  Control or comparison: In-memory reconstruction of current source with only
  this constructor reverted produces the exact old artifact hash
  `8a60c572...`; current source is `1de6d3e...`.
  Likely subsystem: Tail position-tensor construction and static fallback audit.
  Investigation performed: Inspected both runtime/API implementations, rank and
  dtype checks, paged-update use, test coverage, hashes, and the recorded
  focused/default results.
  Resolution: fixed.

- Observed anomaly: The README previously called an eager bounded-wrap loop
  traced.
  Evidence: original `stage_review.md`; current test lines 1312-1454; current
  `bounded_modulo_decode_across_wrap.json`; both mutable-buffer artifacts.
  Affected path: Sustained bounded sliding-cache decode and stable trace inputs.
  Control or comparison: Full traced bounded decode versus full eager unbounded
  decode, plus independent eager controls for A/B/A payloads.
  Likely subsystem: Trace/cache test coverage and evidence reporting.
  Investigation performed: Followed capture, buffer-copy, replay, output, and
  cache arguments in source and checked every recorded PCC and page-table row.
  Resolution: fixed.

- Observed anomaly: Perf and watcher reproduction commands previously contained
  placeholders.
  Evidence: old report versus current host-timing JSON, `perf/README.md`, work
  log, watcher summary, and matching pytest parametrization IDs.
  Affected path: Evidence reproducibility.
  Control or comparison: Exact current commands select the intended kind and
  batch; the corresponding v3/watcher artifacts exist.
  Likely subsystem: Provenance documentation.
  Investigation performed: Searched current code/docs for placeholders and
  cross-checked node IDs against the test parameters and artifact kind.
  Resolution: fixed.

- Observed anomaly: The first 262144-context initializer hung with five missing
  NoC read responses, while the triage summary mislabeled the NoC check as
  passing.
  Evidence: `AUTOTRIAGE.md` and both raw triage captures.
  Affected path: The retired test-only multi-gigabyte host upload, before
  decoder construction.
  Control or comparison: Device-side `moreh_full`, sentinel readback, and later
  passing advertised-context artifacts for both kinds.
  Likely subsystem: Large host-to-device fast-dispatch transfer, not decoder
  math/cache/trace.
  Investigation performed: Rechecked the preserved diagnosis, current test
  initializer, C++ implementations, and successful artifacts.
  Resolution: fixed for the test path; the lower-level retired upload failure is
  controlled and is not a functional-decoder failure.

- Observed anomaly: The first Tracy capture dropped 51 device op IDs.
  Evidence: `AUTOFIX_TRACY_POSTPROCESS.md`, current profiler drains, and v3 raw
  reports.
  Affected path: Profiler collection and derived performance totals.
  Control or comparison: Current v3 host/device ID sets are exactly 1294/1294
  and 1298/1298 with complete three-session trace rows; every derived row maps
  back to raw data.
  Likely subsystem: Finite per-RISC profiler buffers.
  Investigation performed: Recomputed host/device ID set differences, trace
  session counts, derived row membership, operation counts, and time totals.
  Resolution: fixed by supported profiler drains.

- Observed anomaly: Full-context/capacity artifacts carry decoder hash
  `8a60c572...` rather than the final `1de6d3e...`.
  Evidence: their provenance fields and a byte-for-byte in-memory hash
  reconstruction.
  Affected path: Stale-artifact assessment.
  Control or comparison: Reverting only the bounded non-aligned tail constructor
  from `moreh_full` to the historical `ttnn.full` reproduces `8a60c572...`
  exactly. Those context/capacity commands do not pass
  `cache_position_modulo`; current-hash real-weight, boundary, bounded-tail,
  trace, perf, and watcher evidence covers the changed and adjacent paths.
  Likely subsystem: Evidence capture sequencing, not decoder behavior.
  Investigation performed: Reconstructed and hashed the exact previous source
  in memory and traced which branch each older test invokes.
  Resolution: controlled; no rerun is required.

## Scope Inspected

- Goal/skill paths: supplied full functional-decoder contract;
  `.agents/skills/stage-review/SKILL.md` in full;
  `.agents/skills/functional-decoder/SKILL.md` in full; repository `AGENTS.md`.
- Branch/snapshot: live worktree on `hous/gemma-4-26b-a4b-it`, base HEAD
  `e983152d5760c80e794280e8f1862169ec36a7e0`.
- Artifact paths: `doc/context_contract.json`; every functional-decoder JSON and
  markdown artifact; four perf CSV/text reports; raw v3 Tracy report/log CSVs;
  watcher summary/raw log; triage and AutoFix reports.
- Code/test paths: `tt/functional_decoder.py`, `tests/test_functional_decoder.py`,
  `tests/test_trace_mutable_buffers.py`, synthetic-weight helpers/tests, the
  delegated Gemma4 sparse-prefill functions, HF Gemma4 config/modeling code,
  and TTNN full/SDPA source.
- Commands run: read-only `git status`/branch/HEAD/diff/diff-check and ignore
  inspection; `find`/`rg`/`sed`/`nl` source and artifact inspection; SHA-256,
  JSON, timestamp, PCC/page-table, CSV time/count, signpost, raw global-call-ID,
  profiler-session, and watcher-log analyses; the read-only strict context
  checker. No pytest workload, TT device command, profiler capture, watcher
  workload, reset, server, or code/test edit was run by this reviewer.

## Residual Risk

- Hardware behavior was assessed from retained current artifacts, raw reports,
  hashes, source, and test controls as required; this reviewer did not rerun TT
  hardware.
- The 262143/262144 prefill capacity probes establish real-weight execution,
  exact logical shape, finite last-token output, and cache/page-table capacity,
  not full-sequence HF PCC. Real-weight full-layer, boundary, and 32800-token
  sampled comparisons establish the separate PCC contract. This disclosed split
  is proportionate and does not reduce advertised capability.
- Performance is a correctness-first one-device baseline with no target or
  improvement claim. Optimization, multichip, full-model generation, and
  serving remain later-stage responsibilities.
- Clean pass assumes the ignored CSV and watcher artifacts are force-added to
  the stage checkpoint and unrelated dirty files are excluded, per the
  stage-review checkpoint rule.
