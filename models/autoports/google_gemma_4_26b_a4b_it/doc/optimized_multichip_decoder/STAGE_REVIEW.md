# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- Baseline command provenance is weaker than final provenance. The immutable
  baseline timing JSONs contain a stale delegated functional-decoder command,
  and `baseline/provenance.json` correctly labels its replacement commands as
  reconstructed rather than verbatim shell history. The measured decoder
  SHA, test SHA, device IDs, target mesh, fallback policy, warmup/replay
  regime, timing files, and JUnit hashes remain recorded, so this does not
  invalidate the comparison, but the exact interim test source identified by
  SHA `22561a48...` is not retained as a standalone source file.
- `tt-perf-report` device time is explicitly an operation-duration sum inside
  signposted ranges, not an end-to-end device critical path. The reports show
  large cross-device/inter-replay gaps which are not added to that sum. The
  separately retained same-run host wall times bound this limitation, and
  `perf_summary.json` also states that modeled roofline bytes omit non-matmul
  and CCL traffic. The published labels are therefore accurate, but those
  device/roofline values should not be read as serving latency.

## Hard-Check Gaps

- I did not run Tenstorrent hardware during this independent review, as
  required. Hardware conclusions are based on retained logs, JUnit, JSON,
  Tracy CSV, watcher, and health artifacts.
- The release profiles were exercised on the accepted one-, two-, and
  four-device P300C QB2 proxy meshes, not production P150 systems. Device IDs
  are distinct and consistent across retained artifacts: `[1]`, `[1,0]`, and
  `[1,0,3,2]`.
- The tight P150 50,624-token context boundary was physically rerun on the
  final source for sliding and full attention. The unchanged 262,144-token
  TP2/TP4 limits are supported by corrected full-stack projections and
  inherited boundary evidence rather than a new full-length physical run in
  this stage.
- Active-Ethernet watcher instrumentation cannot fit this fabric program:
  the retained failure records a 30,064-byte program against a 26,624-byte
  watcher buffer. The final 12-case watcher run therefore disables only ETH
  instrumentation while keeping worker and idle-Ethernet checking enabled.
- Full-model, generator, LM-head, sampling, and vLLM validation were not run;
  they are explicitly outside this decoder-only stage.

## Anomaly Ledger

- The requested initial final SHA `fe7cf075...` was audited and rejected
  during review, not silently lost. That revision briefly allocated separate
  expert gate/up device tensors before constructing their packed replacement,
  invalidating constructor-peak capacity. Direct host-packed upload removed
  that peak. Subsequent advice closure selected TP4 sliding QKV DRAM sharding
  and then fixed an erroneous full-attention QKV allocation/guard. All
  canonical final evidence was regenerated after those fixes. The actual
  frozen final pair is decoder
  `9c6735c56ff48309215845e3f73aa640c54fdb7e5ca0a5ff3b36155b409473ba`
  and multichip test
  `54a87ce28b179efc740bd5ea1078340348a2fd7dee92f01379e3c814916f8071`.
- The earlier TP4 B32 score of 0.994989 against a BFP8-dense single-chip
  oracle was reproducible but policy-mismatched. Fresh prompt-derived,
  policy-matched references are hash-manifested under
  `final/tp4_full_b32_matched_reference/`; TP4 then reaches
  0.999764/0.999833 and TP2 reaches 0.999854/0.999895.
- Earlier failed or stale artifacts carrying names such as `final` were
  separated from the canonical result. Failing XMLs are retained under
  `candidates/superseded_final_failures/`; canonical final XML contains no
  failures or errors. README/work-log references now point to
  `final/current_pcc_p150`, `final/current_pcc_p150x2`,
  `final/current_correctness`, and `final/watcher_current` rather than the
  superseded precision directories.
- The corrected capacity model includes BF16 full-attention dense gate, up,
  and down across exactly five full-attention layers at indices
  `[5,11,17,23,29]`, plus the TP4 QKV copy across 25 sliding layers. Its
  conservative per-device headroom is 268,670,464 / 861,289,472 /
  5,026,182,144 bytes for TP1/TP2/TP4; exact-physical headroom is higher.
- The last open material profiler recommendation, TP4 sliding QKV DRAM
  sharding, was not dismissed as noise: alternating controls measured
  0.584680/0.585390 ms versus 0.594262/0.594501 ms and the candidate was
  selected. TP1 down's 0.23-0.48% nominal gain was rejected as sub-noise with
  a documented 59.5 MB retained-copy and complexity cost. Other grid, L1,
  reader, dtype/fidelity, collective, and residual alternatives have retained
  accept/reject evidence.

## Scope Inspected

- Reviewed the complete working-tree diff and confirmed the implementation
  remains within the decoder stage: `multichip_decoder.py`, its inherited
  optimized-decoder support, decoder tests, context contract, and the new
  optimized-stage documentation/artifacts. No full-model or vLLM source was
  added, and prior-stage tracked artifacts are unchanged.
- Read `README.md`, `work_log.md`, `AUTOFIX.md`, `operation_topology.json`,
  `perf_summary.json`, `capacity_projection.json`, baseline provenance,
  candidate evidence, all canonical final PCC/timing/capacity artifacts,
  matched-reference provenance, watcher provenance/log/JUnit, profiler
  provenance, all six raw Tracy captures, generated tables/CSVs, and final
  health data.
- Recomputed hashes of the frozen source set. Besides the final pair above,
  the current optimized decoder is `f6a09543...`, the functional decoder is
  `1de6d3e3...`, and the functional test is `aafb5d0f...`. Canonical host,
  PCC, capacity, and profiler timing artifacts carry the exact final decoder
  and multichip-test hashes; the matched-reference and watcher manifests also
  carry the exact optimized-decoder hash.
- Verified all 303 stage JSON files parse, all six compressed Tracy inputs
  decompress, all manifest hashes match retained files, and all canonical
  final JUnit XML is passing. Independently recalculating profiler CSV sums
  reproduces the six provenance totals, including division by three traced
  decode replays. Raw capture hashes are distinct across all six cases.
- Verified the six final warmed timing pairs are final-default measurements
  and all beat baseline: P150 80.051155/0.759652 ms sliding and
  78.798871/0.885008 ms full; TP2 52.653659/0.646949 and
  57.376768/0.722745; TP4 63.521466/0.585234 and
  70.092498/0.914335. Each uses one untimed prefill warmup, five requested
  decode warmups plus trace validation, and 30 traced replays, with fallback
  throwing.
- Verified prompt-derived final PCC for both attention kinds on every profile;
  the minimum reported value is 0.999510, above the accepted 0.995 gate.
  Retained B32, S=33 nonaligned, TP2/TP4 stacked-trace, replay bit-exactness,
  cache-consumption, and 12/12 clean watcher evidence cover the required
  stress cases.
- Inspected source and static assertions for true target-mesh enforcement,
  replicated inter-layer output contract, profile-specific defaults,
  runtime-indexed top-8 sparse execution, packed-weight lifetime, QKV
  layer-count capacity, context checks, and isolation of watcher/candidate
  artifacts. No measured TP2/TP4 path is a replicated single-chip fallback.
- Rechecked 45 explicit host/static cases, Python compilation, JSON parsing,
  and `git diff --check`; all pass. No C++ build is required for this
  Python/JSON/Markdown-only change.

## Residual Risk

- Performance and fabric behavior may differ on actual P150 hardware from the
  accepted P300C proxy, especially for the narrow TP4 full decode improvement
  and multi-device timing variance.
- The decoder's replicated inter-layer contract and capacity projection are
  ready for a later full-model stage, but only that future integration can
  expose stack-wide allocator fragmentation or generator-level scheduling
  effects. Those risks do not justify expanding this stage into full-model or
  vLLM work.
