# Stage 2: graph fusing (encoder plus head as one trace per bucket)

Plugin stage "graph-fusing" mapped to Laya (PLAN.md section 5, row 2): record the fusions the executed graph carries,
prove the traced path equals the eager path bit for bit at B 1 and B 8, capture the encoder plus head as one Metal
trace per (rows, seq) bucket with the two-phase warmup, run the allocation tracker, and run the drift test.

## Fusions in the executed graph

| fusion | where | how |
|---|---|---|
| Q, K, V projections as one matmul | `tt/modernbert_attention.py` | `Wqkv` kept fused `(1024, 3072)`; `split_query_key_value_and_split_heads` splits on device (the 3 is the outer dim of the HF reshape) |
| attention scale folded into Wq | `tt/weights.py: fold_q_scale`, `fold_q_scale_bias` | the Q third of `Wqkv`, of the head `in_proj_weight` and of the head `in_proj_bias` is multiplied by 1/8 at load (exact, power of two); SDPA runs with `scale=1.0`, so sdpa.cpp does not rescale the mask on every call |
| GELU in the Wi_act matmul | `tt/modernbert_mlp.py` | sharded plan: `fused_activation=UnaryWithParam(GELU, 1)` in the program config; interleaved path: `activation=` on `ttnn.linear`; the gate half carries no activation (GeGLU order, negative control 0.803) |
| ReLU in the head `linear1` | `tt/laya_head.py: TtnnHeadLayer` | `activation=UnaryWithParam(RELU)` on the biased `ttnn.linear` |
| erf GELU in the scorer dense | `tt/laya_head.py: TtnnLayaHead.scorer` | `activation=UnaryWithParam(GELU, 0)` on the biased `ttnn.linear`; the last linear runs in fp32 after a typecast |
| biases inside the matmuls | head layers and scorer | `bias=` row vectors `(1, N)` on `ttnn.linear` (in_proj, out_proj, linear1, linear2, dense, out) |
| sliding band and padding in one SDPA mask | `tt/modernbert_masks.py` | static `band` and `zeros` per bucket plus the per-call `pad_row`, combined by two `ttnn.add` broadcasts inside the trace; SDPA receives one `(B,1,S,S)` DRAM mask and never `sliding_window_size` (rejected together with a mask in this build) |
| type embedding as a broadcast add | `TtnnLayaHead.add_type_embedding` | `ttnn.embedding` on `(B,1)` ids then one `ttnn.add` over `(B,S,1024)`; no host tensor per row |
| residual stream resident in L1 | `tt/modernbert_layer.py: _resident` | when the GeGLU plan is sharded (B 2 and 4 at S 512, B 1 and 2 at S 1024) both norms, both residual adds and the three GeGLU matmuls run on one 8x8 block-sharded grid; two reshards per layer (into and out of the attention chain) |
| attention chain in L1 | `PortConfig.l1_attention_max_rows` | Wqkv output, Q/K/V, rotary, SDPA and Wo stay L1 interleaved up to 2048 rows |
| whole forward in one trace | `tt/runner.py: LayaTraceRunner` | per bucket: masks, embeddings, 28 layers, final norm, type add, two head layers, scorer, CLS slice; per call the host writes three preallocated inputs (`input_ids`, `pad_row`, `qtype`) and reads two outputs |

Not fused and why: rotary is two calls per layer (q and k) because `rotary_embedding_llama_fused_qk` is decode-only
(upstream note); the scorer's final linear is kept separate in fp32 by design (Appendix A.4); head splitting and
concatenation are the stock `split_query_key_value_and_split_heads` and `concatenate_heads` ops.

## Trace design (`tt/runner.py`)

`LayaTraceRunner(model, buckets)`:
1. `warmup()` phase 1 runs every bucket once eagerly on dummy inputs (program cache warm, intermediates freed), phase 2
   captures every bucket back to back: write dummy inputs, `begin_trace_capture`, `model.device_forward(bucket)`,
   `end_trace_capture`; the two outputs are kept and marked corruptible (`UnsafeAllocationTracker.mark_corruptible`); the
   three input addresses are asserted unchanged across the capture and again before every replay.
2. `run(input_ids, attention_mask, qtype)` picks the bucket, builds the padded host tensors, `copy_host_to_device_tensor`
   into the three preallocated inputs, `execute_trace(blocking=False)`, reads the logits `(B,S,1)` fp32 and the CLS rows
   `(B,32,1024)` with `ttnn.to_torch`, slices to the real rows and length.
3. `release()` releases every trace. Device opened with `trace_region_size` 512 MiB and `l1_small_size` 79104.

Per-call host input bytes at S 512: `input_ids` 2 KiB x B, `pad_row` 32 KiB x B (tile padded), `qtype` 4 B x B.

## Results

`tests/replay_trace_check.py --buckets 1x512,8x512 --rounds 3 --repeats 10` with `TT_METAL_TRACE_ALLOC_TRACKING=1`
(`/home/hous/dev/laya/logs/p3_s2_replay_check_20261005T213425Z.log`, JSON `replay_trace_check_1x512_8x512.json`,
`TT_METAL_TRACE_ALLOC_TRACKING` recorded as "1", `unsafe_allocation_error` null, `pass` true). Three rounds, forward
then reverse bucket order, two different inputs per bucket, the first input replayed 10 extra times in round 0.

| bucket | traced == eager (max abs delta, logits and CLS) | repeated replay identical | input change moves output (max abs delta) | NaN | replays |
|---|---|---|---|---|---|
| 1x512 | True (0.0, 0.0) | True | True (3.332) | False | 15 |
| 8x512 | True (0.0, 0.0) | True | True (3.479) | False | 15 |

Warmup for the two buckets: phase 1 (eager) 0.11 s, phase 2 (capture) 0.09 s.
Marker logits of the first question, eager and traced: [1.8701171875, -1.8837890625, 0.36669921875, -0.364013671875] and [1.8701171875, -1.8837890625, 0.36669921875, -0.364013671875].

### Gates

| gate | result |
|---|---|
| traced == eager, bit identical, B 1 and B 8 | pass (max abs delta 0.0 on the scorer logits and on the CLS rows at both buckets) |
| `TT_METAL_TRACE_ALLOC_TRACKING=1` clean | pass (no `RuntimeError` from the tracker over 30 replays; outputs are marked corruptible, inputs preallocated, static masks and weights allocated before the first capture) |
| drift test | pass: repeated replays of the same input are identical; a different input moves the logits by up to 3.33 (B 1) and 3.48 (B 8) |

### Timing

The timing columns of the tracked run are not usable: with the tracker on, every `execute_trace` first runs
`UnsafeAllocationTracker.verify_before_replay`, which calls `gc.collect()` and walks referrers, so the traced path
measured 305 ms (B 1) and 348 ms (B 8) against eager
16.7 ms and 89.9 ms; the host 1-minute load was 13 to 25 during the run.
Untracked run on a quiet host (`replay_trace_check_untracked_timing.json`, 1-minute load 7 to 9 during the run,
`tests/replay_trace_check.py --buckets 1x512,8x512 --rounds 3 --repeats 20`), p50 of the host wall time around
write inputs, forward or replay, and the two readbacks:

| bucket | eager p50 (min) | traced p50 (min) | traced / eager | bit identical |
|---|---|---|---|---|
| 1x512 | 14.72 ms (14.09) | 13.75 ms (13.59) | 0.934 | True |
| 8x512 | 85.91 ms (85.63) | 85.50 ms (85.13) | 0.995 | True |

Trace removes about 1 ms of host dispatch at B 1 (7 percent) and nothing measurable at B 8, where the device work
(85 ms) hides the dispatch of the roughly 450 ops; the same pattern as the CLM port on this box. The traced path is
kept because it is deterministic, keeps Python out of the request path and is what the allocation tracker was run
against. `bench_default.json` in `../optimized_decoder/` repeats the measurement with 20 replays per bucket
(13.92 and 85.52 ms) and adds 64x512 (646 ms).

## Re-validation on the stage 3 shipped configuration

`tests/replay_trace_check.py --buckets 1x512,8x512,64x512 --rounds 3 --repeats 5` with `TT_METAL_TRACE_ALLOC_TRACKING=1`
on the shipped `PortConfig` and policy (`minimal_matmul`, 11x8 GeGLU, SDPA 8x8, erf GELU;
`../optimized_decoder/replay_trace_check_shipped_tracked.json`, log `/home/hous/dev/laya/logs/p3_final_queue_20261005T222233Z.log`):
pass, no tracker error, warmup phase 1 0.57 s and phase 2 0.09 s for three buckets.

| bucket | traced == eager (max abs delta logits, CLS) | repeated replay identical | input change moves output (max abs delta) | NaN | replays |
|---|---|---|---|---|---|
| 1x512 | True (0.0, 0.0) | True | True (3.358) | False | 10 |
| 8x512 | True (0.0, 0.0) | True | True (3.477) | False | 10 |
| 64x512 | True (0.0, 0.0) | True | True (3.763) | False | 10 |
