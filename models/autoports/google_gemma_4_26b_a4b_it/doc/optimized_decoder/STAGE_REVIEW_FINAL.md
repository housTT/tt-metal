# Final stage review: optimized decoder

Verdict: clean-pass

## Summary

The frozen optimized-decoder implementation satisfies the original stage contract and the optimize-stage checklist. The five findings in the initial `STAGE_REVIEW.md` are closed by the source/test changes and preserved AutoFix evidence: actual cache-dtype propagation rejects BFP8 numerically; final-policy BFP4 expert geometry and packed/separate controls are measured; the repaired R22 loader is verified by runtime counters and a genuine reader matrix; prefill route-union accounting and profiler advice are corrected and closed; and the canonical commands, hashes, health records, and labels are internally consistent. No fallback, correctness, trace, context, topology, precision, performance-evidence, or documentation blocker remains.

## Required work

None.

## Other concerns

None blocking. The v5 context/watcher/allocation/serving/profile evidence has the exact final decoder hash and the documented pre-format test hash; the independently reconstructed file is byte-exact at that hash and AST-identical to the final formatting-only test revision. The v6 complete and performance gates use both exact current hashes. The stage-owned local commit is correctly pending this review. Hardware was not rerun during this independent review.

## Evidence reviewed

- Original optimized-decoder prompt; `stage-review`, `optimize`, and `tt-device-usage` contracts; repository `AGENTS.md`.
- Frozen source and tests at decoder SHA-256 `feebc8cb2f20ad9ba81c7d0f50f8323694d6d91ebb31cb9072e18b0f6b0a9c45` and test SHA-256 `01ed0de36451891c6c41968baeeae34f1cb77acf4e3c6db0c322506fb83bd349`.
- Exact-current v6 complete suite: 44 passed, 15 skipped, 0 failed/error in 121.221 s; exact-current v6 performance suite: 4 passed, 0 failed/error in 48.501 s.
- Preserved real-weight PCC, nonaligned/cache-consuming trace, mutable-buffer, batch-32, wrap-stress, context-capacity, watcher/health, serving-prefill, allocation, candidate-sweep, Tracy, and `tt-perf-report` artifacts, including the replacement `reviewfix2_r22_*` matrix.
- `README.md`, `work_log.md`, `AUTODEBUG.md`, `AUTOFIX.md`, `AUTOTRIAGE.md`, `final_manifest.json`, `profiler_summary.json`, persistent-allocation accounting, and `doc/context_contract.json`.
- Read-only validation: scoped `git diff --check`, Python AST parsing, canonical JSON parsing, XML result recounting, and independent hash verification all passed.
