# Stage-review AutoFix record

The first independent stage review returned `more-work-needed` for missing
randomized high-context, batch-greater-than-one, and trace-allocation-lifetime
evidence. Three fresh AutoFix investigations audited those concerns.

## Proven repairs

- The acceptance harness now releases each trace before allocating host/cache
  inspection tensors and uses only preallocated capture inputs during replay.
- P150, TP2, and TP4 passed both meaningful layer kinds with
  `TT_METAL_TRACE_ALLOC_TRACKING=1`; no unsafe live-trace allocation warning
  appears in the accepted logs.
- Batch 2 passed non-aligned prefill S=33, exact maximum position 131071,
  randomized high-context blocks, changed hidden/position inputs, page-table-
  only remapping, exact rank replication, local-head cache reconstruction, and
  bitwise replay determinism.
- TP4 full attention repeated the batch-2/high-context gate under the supported
  Tensix watcher configuration and passed; post-run device health was clean.

## Refuted hypotheses

The automatic batch-2 optimized policy uses BFP4/LoFi QKV, while the selected
TP2/TP4 multichip policy uses BFP8 attention. Comparing those policies produced
TP2 prefill PCC 0.8613. Replacing target prefill with a per-user loop reproduced
the same value, refuting batching as the cause; that experimental code was
fully reverted. Constructing the real P150 `OptimizedDecoder` with the same
`DEFAULT_OPTIMIZED_POLICY` BFP8 policy controls precision and yielded accepted
TP2/TP4 prefill PCC 0.989610--0.992361.

## Accepted artifacts

The accepted evidence is under `artifacts/20260828_stage_review_autofix_v1/`:

- `p150_batch1_trace_tracker_v1.log.gz`;
- `p150x2_x4_batch1_trace_tracker_v1.log.gz`;
- `p150_batch2_highctx_bfp8_v3_trace_tracker.log.gz`;
- `p150x2_batch2_highctx_bfp8_v3_trace_tracker.log.gz`;
- `p150x4_batch2_highctx_bfp8_v3_trace_tracker.log.gz`;
- `p150x4_batch2_full_watcher.log.gz`;
- `post_batch2_watcher_tt_smi.log.gz`.

These tracker/watcher logs are correctness and lifecycle evidence, not latency
evidence. The authoritative warmed latency/profiler artifacts remain under
`artifacts/20260828_final_v2/`.
