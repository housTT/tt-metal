# Selected Ring terminal profile

Date: 2026-08-20 EDT

The profiler-enabled runtime executed one real linear-attention layer, final
norm, TP4 vocabulary LM head, and the selected common Ring force-argmax sampler
between `QWEN36_FULL_MODEL_TOKEN_OUT_START` and
`QWEN36_FULL_MODEL_TOKEN_OUT_END`.

The runtime logged force argmax with `cluster_axis=None`, `num_links=1`, and
`Topology.Ring`; the pytest exited successfully. `tt-perf-report` found four
devices and produced the signpost-bounded `ring_token_out_report.csv` and
`ring_token_out_summary.csv` plus the summary PNG.

- operations: 145 merged rows, 29 operation groups
- summed device time: 3,962.61 us
- argmax: 1,417.43 us (35.77%)
- Ring all-gather: 883.30 us (22.29%)
- 21 width-sharded matmuls: 988.52 us (24.95%)
- top-k: absent

This selected Ring profile improves the corresponding rejected Linear compact
profile from 4,595.74 us to 3,962.61 us, chiefly by reducing all-gather from
1,517.66 us to 883.30 us. The production 64-layer direct split remains the
token-out latency authority: 42.141 ms model, 2.416 ms sampler, 44.556 ms
combined, so sampler work is 5.4% of production token-out.

Raw Tracy/device intermediates were intentionally deleted after compact
distillation; they occupied about 2.5 GiB and are reproducible with the command
recorded in `../../work_log.md`.
