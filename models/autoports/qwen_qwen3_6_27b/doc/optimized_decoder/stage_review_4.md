# Independent stage review 4

Verdict: **more-work-needed**.

## Required work

1. The inherited `_linear_chunk_inverse()` still issued ten plain FP32 matmuls. The reviewer measured 905.176 us at sequence 32 and 1814.692 us at sequence 128 and required an optimized override.
2. Large-prefill projection and MLP rows still carried actionable program-config advice. Phase-specific 2D configs and a full-attention large-prefill profile were required.
3. The movement audit needed measured lower-movement attempts or exact installed-runtime blockers for GDN preparation and full Q/K/head/cache formatting.
4. Blackhole byte-floor and host/kernel/gap accounting needed to be regenerated from the final source rather than inherited from v2.

## Accepted evidence

Correctness, context, Watcher, trace determinism, real-weight precision selection, decode sharding, projection fidelity, GDN decode geometry, and the issue #50475 dedicated-kernel experiment were accepted. Review was read-only; the reviewer opened no TT device and modified no file.

## Resolution

AutoFix round 5 routed all inverse matmuls through the legal FP32 reuse config, measured generic 2D large-prefill candidates on both layer kinds, added a warmed full chunked-prefill profiler node, regenerated Blackhole accounting, and documented movement blockers. Exact evidence is under `candidates/autofix5/` and `final_selected_v4/`.
