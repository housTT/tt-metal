# Report-merger raw benchmark AutoFix

Date: 2026-09-02

## Starting evidence

- Source-only diagnosis:
  `evidence/merger_raw_evidence_AUTODEBUG.md`.
- Aggregate benchmark report:
  `tti_cache/workflow_logs/reports_output/benchmarks/data/report_data_openai__gpt-oss-120b_2026-09-02T194657+0000.json`.
- The merger attempted `int(error_request_count)`, but TTI's aggregate parser
  represents raw `failed == 0` as JSON `null`; this caused `int(None)` to fail.
- The aggregate also omits per-request completion, error, and exact-length
  evidence, so merely accepting `null` would have weakened the release gate.

## Hypothesis and experiment

Hypothesis: each aggregate row can be bound uniquely to its raw result by
normalizing aggregate `targets.timestamp` and raw `date`, then checking that
artifact against the exact expected workload tuple.

The current artifacts verified the prediction: all 21 aggregate timestamps
matched one unique raw artifact. Each raw artifact reported the expected
model/backend, concurrency, request count, completed count, zero failures,
only empty error entries, exact per-request input/output lengths, and exact
token totals.

Verdict: **verified**.

## Fix

- Added required `--benchmark-raw-dir` release-merger input.
- Treat aggregate `error_request_count: null` as lossy zero-failure summary
  only after raw validation proves the actual request result.
- Require the exact ordered 21-row matrix and a unique normalized completion
  timestamp binding for every row.
- Validate raw `num_prompts == completed`, `failed == 0`, empty per-request
  errors, exact per-request input/output lengths, exact token totals, expected
  concurrency, model, and backend.
- Record only small structural provenance in merged metadata. Raw artifacts
  remain uncopied.
- The raw loader is a streaming top-level allowlist projection. It scans past
  `generated_texts` without materializing, returning, logging, or copying it.

## Verification

- `python -m py_compile .../regenerate_release_report.py`: passed.
- Current-artifact validation: 21 rows and 21 unique raw artifacts passed.
- Sanitized count tamper (`completed=7`): rejected.
- Sanitized per-request input-length tamper: rejected.
- Duplicate raw completion timestamp: rejected as ambiguous.
- Aggregate workload tamper: differed from the required matrix and was
  rejected by the merger gate.
- Privacy sentinel under `generated_texts`: the allowlist reader did not pass
  the sentinel to `json.loads` and did not return it.

No server request, device operation, or generated model text inspection was
performed.

## Final status

Fixed and host-verified. The final release merge must supply the corrected raw
benchmark directory alongside the aggregate benchmark report.
