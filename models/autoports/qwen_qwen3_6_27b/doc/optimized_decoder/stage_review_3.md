# Independent stage review 3

Verdict: **more-work-needed**.

## Required work

1. **Final full-prefill precision and byte-floor mismatch.** The canonical seq-32 prefill has logical `m == 32`, so the shape-based MLP branch uses the full-decode BFP8 copies rather than the documented BFP4/BFP4/BFP8 policy. Final rows 81/82/84 are all BFP8, while the Blackhole floor counts mixed-policy bytes. Selection must become phase-aware; final prefill correctness, profiler, E2E, and roofline must be recollected from the actual policy.
2. **Geometry, projection, precision, and fidelity families stop too early.** The 8/16-core MLP attempts use only their first large block widths; 64 cores stop at one output-padding failure; linear input projection stops at one 4-core/large-N L1 failure. Mixed BFP4/BFP4/BFP8 has not been crossed with legal sharding, and final DRAM-sharded projection rows are LoFi despite documentation claiming HiFi2. Retry smaller blocks, compatible grids/padding, selected-topology mixed precision, and LoFi/HiFi2 with real-weight PCC and whole-layer latency.
3. **Actionable GDN and large-prefill advice remains.** Linear decode's transpose-update matmul remains a 74.653-us `SLOW` row with `in0_block_w=1` and 1x1 subblock; the installed config tunes only recalled/core reads. Linear prefill has analogous generic FP32 rows, and final prefill projection rows retain DRAM-sharding advice. Adapt and measure GDN update/program candidates and add a representative larger-prefill profiler node with explicit phase-appropriate configs.

## Accepted repairs and controlled residuals

- Batch-one linear decode now enters the 32-core sharded path; counter assertions and final BFP4 DRAM-sharded rows prove the repair.
- Correctness, non-aligned lengths, native context, paged cache, determinism, batch 32, Watcher cleanliness, source/artifact integrity, and device/E2E speedups are credible.
- Post-test nanobind diagnostics occur after successful JUnit and normal device closure and are controlled binding-teardown leakage.
- The issue #50475 sequence-kernel experiment is credible: its kernel row is 141.937 us, but the callable adapter loses at 344 ops / 6482.619 us. Lack of a callable recurrent/fused kernel is an upstream residual only after the remaining generic-path opportunities are completed.

Review was read-only; no TT device was opened and no file was modified by the reviewer.
