# Stage Review

verdict: more-work-needed

## Required Work

- P1: The non-aligned bounded-prefill hot path contains a proven host fallback.
  Evidence: `tt/functional_decoder.py:649-722` enters the tail-update path when
  `cache_position_modulo` is set and the logical length is not tile aligned,
  then constructs every tail position with `ttnn.full` at lines 715-722.
  `ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73,164-202` shows that
  `ttnn.full` allocates and fills a host `std::vector`, creates a host tensor,
  and transfers it to the device. The stage's own diagnosis reaches the same
  conclusion in `doc/functional_decoder/AUTOTRIAGE.md:5-8,30-43`. In contrast,
  the module contract says all host conversion is confined to setup
  (`tt/functional_decoder.py:9-14`), the README calls the runtime audit clean
  (`doc/functional_decoder/README.md:71-76`), and the audit only searches for
  literal `torch`, `from_torch`, and `to_torch` tokens
  (`tests/test_functional_decoder.py:1729-1751`), so it cannot catch this known
  host-backed TTNN API.
  Why this matters: The user contract and functional-decoder skill require no
  torch or host fallback inside a prefill/decode pass, including hidden helper
  fallbacks. This path is part of the advertised valid non-aligned and bounded
  cache contract, and it performs up to 31 host allocations/transfers during
  one forward pass.
  Required next step: Replace the runtime `ttnn.full` construction with a
  genuinely device-side or setup/preallocated position-tensor mechanism,
  strengthen the fallback audit to reject known host-backed creation APIs in
  the runtime call graph, and rerun the affected non-aligned bounded-prefill
  cache-integrity and audit gates. Update the README/work log after the new
  evidence exists.

- P1: The claimed traced bounded sliding-cache decode path has only eager
  evidence.
  Evidence: `doc/functional_decoder/README.md:45-50` says the bounded sliding
  cache "was also traced" across positions 1023, 1024, 1025, and 1103. The
  cited behavior comes from
  `tests/test_functional_decoder.py:1315-1412`, but its two controls call
  `_attention_decode` eagerly in a Python loop at lines 1347-1383; there is no
  `begin_trace_capture`, `execute_trace`, stable-buffer update, or full
  `decode_forward` invocation in that test. Its JSON artifact likewise records
  bounded-vs-unbounded PCC only. The existing stronger A/B/A mutable-buffer
  test at `tests/test_trace_mutable_buffers.py:168-263` would exercise distinct
  positions and page tables under trace for the two layer kinds, but no
  `trace_mutable_buffers_*.json` artifacts exist and it is absent from the
  logged functional-stage commands; it also does not exercise
  `cache_position_modulo` across the sliding-window wrap.
  Why this matters: `cache_position_modulo` affects both paged updates and the
  paged-SDPA read view, so trace capture/replay is a stage-critical cache
  contract rather than an interchangeable eager check. The current evidence
  cannot support the README statement or demonstrate that the complete bounded
  decode mode is fully traced.
  Required next step: Trace/replay the full `decode_forward` bounded-cache path
  with `cache_position_modulo=1024`, update stable hidden/RoPE/current-position
  buffers across the wrap, compare against an eager/unbounded or HF control,
  and leave the resulting artifact. Run and retain the existing mutable-buffer
  coverage as well, or explicitly incorporate its distinct current-position
  and per-user page-table cases into the bounded trace test.

- P2: Performance and watcher reproduction fields use placeholders rather than
  exact commands.
  Evidence: Both host-timing JSON files label a command as `exact_command` but
  end it with the literal `exact-node-id`
  (`layer0_sliding_attention_seq1024_batch1_host_timings.json:17-23` and
  `layer5_full_attention_seq1024_batch1_host_timings.json:17-23`). The perf
  provenance and work log retain `<exact perf-profile node id>` or
  `<exact batch1 perf node id>` placeholders
  (`perf/README.md:9-19`; `work_log.md:139-149`). The watcher command similarly
  contains `<batch1 traced-decode node ids for sliding and full attention>` in
  `watcher/summary.json:5` and `work_log.md:162-170`.
  Why this matters: The stage explicitly requires performance provenance and a
  watcher-clean command. The raw perf reports, signposts, hashes, and watcher
  log are present, but fields represented as exact/reproduction commands are
  not directly executable as written.
  Required next step: Replace each placeholder with the actual collected pytest
  node ID or an exact executable `-k` selector, and ensure the recorded command
  identifies the relevant layer kind and batch without relying on unstated
  substitution.

## Other Concerns

- The four required `perf/*/{prefill,decode}.csv` files exist and their device
  totals reproduce the README values, but repository `.gitignore:8` ignores all
  CSV files. The checkpoint step must explicitly include these required derived
  CSV artifacts; otherwise the committed stage will contain only the rendered
  text tables.
- The raw watcher tree is also ignored by `.gitignore:73`. The current raw log
  exists, its SHA-256 matches `watcher/summary.json`, and a representative fatal
  pattern scan is empty. If the raw tree is intentionally not retained, the
  stage owner should ensure the durable summary/provenance policy is sufficient
  before checkpointing.

## Hard-Check Gaps

- The default functional test command covers 38 collected cases (28 passing and
  10 opt-in skips in the recorded run), but the opt-in full-context and long
  attention artifacts do not all carry the same test-file hash because the test
  harness changed between captures. Their measured decoder SHA-256 is
  consistently the current `8a60c572...`, and the later harness edits are
  localized, so this is not independently a correctness failure; a final
  evidence manifest should make the intentional capture sequence explicit.
- The 262143/262144 prefill probes intentionally validate execution, exact
  returned logical shape, and finite last-token output rather than full-length
  HF PCC. This limitation is disclosed in `README.md:129-135`; correctness of
  the long attention implementations is separately sampled at the 32768
  chunking cliff. No stronger full-context HF materialization was required by
  this review.

## Anomaly Ledger

- Observed anomaly: Host-backed `ttnn.full` remains in a runtime prefill helper
  even though the same API was diagnosed and removed from the context test.
  Evidence: `tt/functional_decoder.py:715-722`,
  `ttnn/cpp/ttnn/operations/creation/creation.cpp:51-73`, and
  `AUTOTRIAGE.md:5-8,42-43`.
  Affected path: Non-aligned prefill with `cache_position_modulo`.
  Control or comparison: `ttnn.moreh_full` is documented and source-verified as
  the on-device alternative in `AUTOTRIAGE.md:43,64-77`.
  Likely subsystem: Runtime cache-tail position-tensor construction and static
  fallback audit.
  Investigation performed: Inspected the Python call path, the concrete C++
  implementation, the stage's triage report, and reran the host-only audit
  selector.
  Resolution: more-work-needed.

- Observed anomaly: README says bounded sliding decode was traced, while the
  producing test is eager.
  Evidence: `README.md:49-50` versus
  `tests/test_functional_decoder.py:1347-1383`.
  Affected path: Bounded sliding-cache decode across wrap.
  Control or comparison: The eager bounded/unbounded comparison reports PCC
  1.0 at 1023/1024/1025/1103; unrun
  `tests/test_trace_mutable_buffers.py:168-263` demonstrates the intended
  stable-buffer trace methodology for other cache shapes.
  Likely subsystem: Trace/cache test coverage and evidence reporting.
  Investigation performed: Inspected every trace call in the functional test,
  the bounded artifact, the README claim, and the separate mutable-buffer test.
  Resolution: more-work-needed.

- Observed anomaly: The first advertised-context cache initialization hung with
  five outstanding NoC reads, while `triage-summary.txt` incorrectly labeled
  `check_noc_status.py` as passing.
  Evidence: `AUTOTRIAGE.md:18-28` and raw `triage/tt-triage*.txt`.
  Affected path: The old test-only, host-backed multi-gigabyte `ttnn.full`
  initializer, before decoder construction.
  Control or comparison: Both repeated triage captures agree; later
  `ttnn.moreh_full` runs passed both advertised-context parameters after bounded
  device recovery.
  Likely subsystem: Large host-to-device fast-dispatch transfer; not decoder
  compute.
  Investigation performed: Compared detailed triage evidence with the summary,
  source diagnosis, current test initializer, and replacement artifacts.
  Resolution: fixed for the context-test initializer; the separate runtime
  fallback finding above remains open.

- Observed anomaly: The first Tracy capture dropped 51 device op IDs.
  Evidence: `AUTOFIX_TRACY_POSTPROCESS.md:35-99` and the raw profiler captures
  under `generated/profiler/gemma4_functional_*`.
  Affected path: Profiler collection/post-processing, not decoder math.
  Control or comparison: The two v2 captures have zero missing host/device IDs;
  their derived slices map back to raw global call counts.
  Likely subsystem: Finite per-RISC profiler buffers.
  Investigation performed: Recomputed derived CSV row counts/totals and checked
  all derived global-call-count values exist in the corresponding raw report.
  Resolution: controlled/fixed by supported profiler drains.

## Scope Inspected

- Goal/skill paths: the supplied functional-decoder stage contract;
  `.agents/skills/stage-review/SKILL.md`; and
  `.agents/skills/functional-decoder/SKILL.md`.
- Artifact paths: `doc/context_contract.json`; every file directly under
  `doc/functional_decoder`; derived perf CSV/text reports; raw v2 Tracy reports;
  watcher summary and raw watcher log; AutoFix/AutoTriage reports and triage
  captures.
- Code paths: `tt/functional_decoder.py`;
  `tests/test_functional_decoder.py`; the relevant
  `tests/test_trace_mutable_buffers.py`; delegated Gemma4 expert prefill helpers;
  HF Gemma4 config/modeling code; and TTNN `full`/`moreh_full` C++ sources.
- Commands run: `git status --short`, branch/HEAD/diff inspection; `rg`/`find`
  artifact enumeration; numbered source/doc inspection; JSON parsing; SHA-256
  comparison; CSV row-count/device-time/global-call-count analysis; watcher log
  hash and fatal-pattern scans; and the host-only pytest selector
  `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider ... -k
  'host or audit or delegates'` (14 passed, 24 deselected, 2 warnings). No TT
  device command, server, reset, or hardware test was run by this reviewer.
- Positive evidence re-derived: both meaningful HF layer kinds are represented;
  current context remains 262144; real-weight HF-vs-TTNN PCC artifacts exceed
  0.995; traced decode PCC artifacts exceed 0.995; the four derived perf CSVs
  contain 557/557 prefill ops and 74/76 decode ops with totals
  1242.569/1243.470 ms and 3.012/3.206 ms; raw v2 profiler mappings are complete;
  and the watcher log hash matches its summary with no fatal-pattern matches.
- Scope discipline: current git status contains only functional-decoder
  implementation/test edits and the functional-decoder documentation tree; no
  later-stage file has a current diff.

## Residual Risk

- Hardware-dependent claims were assessed from existing artifacts only, as
  required. The reviewer did not rerun device correctness, context, profiler,
  trace, or watcher workloads.
- Only a later review after the runtime fallback is removed, bounded-cache trace
  evidence is produced, and provenance placeholders are corrected can return
  `clean-pass`.
