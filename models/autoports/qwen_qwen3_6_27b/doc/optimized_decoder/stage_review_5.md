# Independent stage review 5

Verdict: **more-work-needed**.

## Required work

1. Both final-v4 decode profiles began with a one-core DRAM-interleaved input RMSNorm around 85.2 us. The inherited decode entry point normalized before the optimized path sharded the residual. A sharded input-norm/residual candidate was required on both layer kinds.
2. Full decode converted Q/K from height-sharded head output to interleaved for RMSNorm/RoPE and then back to height-sharded for cache/SDPA. The review required a no-round-trip candidate or an exact installed-op blocker.
3. The final-v4 `report.txt` files contained CSV-generation chatter rather than human-readable `tt-perf-report` tables. Tables and broader artifact-manifest coverage were required.

## Accepted evidence

The reviewer accepted final-v4 correctness, context, batch-32, paged cache, determinism, Watcher, GDN kernel rejection, inverse optimization, large-prefill experiments, and Blackhole accounting. Review was read-only; no TT device was opened and no file was modified.

## Resolution

AutoFix round 6 selected a 32-core width-sharded decode-entry residual and RMSNorm. The final device rows are 1.4-us interleaved-to-sharded plus 6.7-us 32-core RMSNorm, replacing the 85.2-us one-core norm. A native height-sharded Q/K RMSNorm attempt reached the installed operation and failed with `Height sharded inputs are not supported`; head creation and decode RoPE/cache require height sharding, so the two conversions are an installed-layout boundary. Final-v5 evidence recollects correctness, Watcher, E2E, six Tracy captures, CSV reports, human-readable tables, and accounting.
