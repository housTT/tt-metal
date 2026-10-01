# Stage 7: optimized full model (traced batched encoder)

## What changed from stage 6

- Traced prefill is called directly (`TtQwen3Encoder._traced_prefill` -> `Generator._easy_trace_prefill`) for
  nine variants: padded lengths 128 / 1024 / 2048 x batch 1 / 4 / 8. A request's texts are grouped by bucket and
  padded up to the next captured batch size.
- Two-phase warmup at startup (`TtQwen3Encoder.warmup`): every variant is prepared first
  (`_prepare_trace_prefill`: persistent device inputs, compile pass), then all nine traces are captured back to
  back (`_record_trace_prefill`), and each trace output is marked corruptible
  (`UnsafeAllocationTracker.mark_corruptible`). No device allocation happens after the first capture; the
  outputs are consumed by host readback before any other trace replays, under the encoder lock.
- The post-trace tail (device slice of the last tile, final RMSNorm, layout change) is gone: the whole
  `[batch x padded, 4096]` pre-norm residual is read back and the last real token of each sequence is
  normalized on the host in fp32 with the HF `model.norm.weight` (eps 1e-6). This removed the per-tile-offset
  program compiles (a 313 ms first `rank` request became 145 ms) and the tracker's 43 flagged buffers.
- Multi-device outputs are concatenated across the mesh (`ConcatMeshToTensor(dim=-1)`).

## Trace allocation safety gate

`tests/replay_trace_check.py`, fresh process, `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`,
bfp8_attn policy, 3 rounds alternating forward and reverse variant order, two different inputs per variant:
`replay_trace_check_bfp8_attn.json`, pass = true, no unsafe buffers.

| variant (padded, batch) | replays | min cosine, repeated replay | max cosine between different inputs |
|---|---|---|---|
| 128, 1 / 4 / 8 | 6 each | 1.0000 / 0.9999999 / 0.9999998 | 0.769 / 0.931 / 0.852 |
| 1024, 1 / 4 / 8 | 6 each | 0.9999999 / 0.9999999 / 0.9999999 | 0.976 / 0.977 / 0.933 |
| 2048, 1 / 4 / 8 | 6 each | 0.9999999 / 0.9999997 / 0.9999999 | 0.858 / 0.987 / 0.958 |

Earlier runs of the same check are kept in `/home/hous/dev/clm-v0.1-8B/logs/replay_trace_check*.log`: 43 flagged
buffers (eager tail), 176 (per-variant prepare+capture), 8 (trace outputs), then clean.

## Performance (`perf_summary.json`, selected policy bfp8_attn, p150, warm traces, p50 of 10)

| padded length | real tokens | batch 1 | batch 4 | batch 8 | tokens/s at batch 8 |
|---|---|---|---|---|---|
| 128 | 32 or 128 | 54.0 ms | 84.4 ms | 142.4 ms | 7,191 |
| 1024 | 512 or 1024 | 148.1 ms | 570.9 ms | 1,140 ms | 7,184 |
| 2048 | 2048 | 286.2 ms | 1,149 ms | 2,290 ms | 7,154 |

Accuracy policy for comparison (`perf_summary_accuracy.json`): 57.6 / 170.5 / 322.8 ms at batch 1.
Model load with a warm weight cache: 6.0 s; warmup (nine traces): 6.2 s.

## Lower-bound reconciliation

Per-layer device time from the Tracy profile (accuracy policy, layer 0): 1.609 ms at 128 tokens, 4.692 ms at
1024 tokens (24 ops per pass). 36 layers: 57.9 ms and 168.9 ms. Measured end to end (accuracy): 57.6 ms and
170.5 ms, within 1 percent, so the terminal work (embedding, readback of up to 8 MB, host norm) and dispatch are
hidden. The bfp8_attn rows sit 7 to 14 percent below that bound because the policy moves fewer weight bytes; a
policy-matched profile was not captured. Eager and traced execution are at parity (`../fused_decoder/README.md`),
so no "token-out slower than layer stack" gap exists to close; the plugin's 10 to 15 percent rule is satisfied.

## Head placement

Heads stay on the host (torch fp32): two 4096x1536 + 1536x1536 + 1536x512 MLPs over a handful of vectors per
request cost well under a millisecond, versus a device round trip per request. Measured in the server:
warm cache-hit answers take 0.1 ms server-side.

## Throughput ceiling and what is left

Prefill throughput saturates near 7.2k tokens/s at bfp8_attn (6.5k at accuracy) from 1024 tokens up; batching
short texts amortizes little because the per-token kernel cost, not weight streaming, dominates (DRAM roofline
23.6 percent). The remaining levers are the matmul core grid (64 of 110 cores) and attention math fidelity; see
`../optimized_decoder/README.md`.
