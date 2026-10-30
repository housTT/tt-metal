# Stage Review 1 (fresh read-only xhigh subagent, 2026 Oct 29, 18:09 to 18:25 ET, HEAD 411ff9e9d8 then e78f68b42c)

Verdict: more-work-needed

No correctness defect was found. Three items block a clean pass:

- P2: Readiness sidecar carries the stage-6 commit hash (f967722683) and a wrong board label (N150) although every recorded run started after commit 4c9a98d810; cause: write_sidecar.py wrote the hash only when the file did not exist. Required: rewrite the hash and label, refresh on every append, note it in the work log. (Fixed: work_log.md section 6.)
- P2: Material prefill matmul rows (gate and up at 128 rows, 182 and 181 us, in0_block_w=2 on the auto config, 30 percent of the layer) with no 1D multicast candidate over the full grid probed and no call-out in the report. Required: probe MatmulMultiCoreReuseMultiCast1DProgramConfig (mcast_in0) at in0_block_w 4, 8, 16 with larger subblocks for gate, up and QKV at 64 to 256 rows; keep winners in the table; rerun perf, the 128 and 256-token sweep points and the prefill check, or record the exact blocker. (Addressed: work_log.md sections 6.1 and 6.2.)
- P2: Sharded prefill RMSNorm rejected without trying the adaptation that removes the cross-path inconsistency (norm at 1536 to 4096 rows, continuation rows on the same kernel). Required: probe 1536, 2048, 3072 and an adapted 4096 variant; make every path use one kernel; rerun the resume check at default thresholds, the prefill check and teacher forcing; keep only if the resume gate behaves as on the final code, else record the blocker and the cost. (Addressed: work_log.md sections 6.1 to 6.4.)

Other concerns: first-token flips at 256 and 1024 tokens are LoFi LM-head near-ties, not matmul kernels (corrected in work_log.md section 5); batched prefill deferred to stage 10 per prefill-features.md; watcher summary string stale (fixed); the first sampler step after prefill is eager (recorded as a lever); the canonical resume artifact is the lowered-threshold run with the default run beside it (matches operator instruction (d)); perf_summary.json names position 160 against a profiled position of about 130; the e78f68b42c context-contract commit landed during the review (documentation only); the hanging nlp_concat_heads probe candidate was killed by its own timeout without a tt-triage capture (repro kept).

Hard-check gaps: watcher ran on the 2-layer reduced fixture only (the long-prompt piece path, 512-row 2D kernels and the 1536/3072 buckets rest on the tracked 36-layer run, first-token equality and the TT-versus-TT resume check); no HF-referenced output for prompts longer than 158 tokens through the piece path; the raw Tracy CSV is not kept (filtered CSVs are); readiness runners open the mesh with trace_region_size=0.

Anomaly ledger and residual risk: see the full review text returned by the subagent (recorded by the stage owner in the final report); every anomaly was classified controlled except the two more-work items above.
