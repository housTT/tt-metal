# Independent stage review 1

Verdict: **more-work-needed**

The fresh review found that correctness, context preservation, Watcher cleanliness, and the measured speedups were credible, but optimization coverage was not yet sufficient for signoff.

Required remediation:

1. Measure coherent residual/norm/DRAM-sharded decode and precision-locked per-role geometry families, rather than using only an MLP-local sharding candidate with immediate reshards.
2. Retain auditable real-weight evidence for mixed MLP precision and compare packed versus separate gate/up under the same dtype and fidelity.
3. Continue issue #50475 work through AutoFix: test a lower-movement sequence-kernel adapter and establish the recurrent-decode/gated-attention limitation with evidence.
4. Add same-run warmed end-to-end timing and a Blackhole-specific byte roofline for both layer kinds.
5. Repair candidate-ledger field counts, stale profiler output paths, and replay-mean labeling.
6. Expand the static runtime audit beyond optimized overrides to inherited entry points and residual/state paths.

The reviewer also recorded two controlled tool limitations: Tracy CSVs omit architecture metadata and default roofline calculations to Wormhole, and a ten-replay linear capture overflows the device-profiler buffer. Neither invalidates device-duration rows, but both require explicit accounting.

This review was performed before the local stage commit, so no commit SHA was expected.
