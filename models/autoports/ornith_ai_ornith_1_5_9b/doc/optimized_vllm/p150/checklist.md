# Serving optimization checklist, p150 profile

Scope: stage 10 on one Blackhole chip, all 32 layers, the selected
`p150_head8_hifi2_kv8_policy.json`, commit `e70875637c`. Only the `$optimize`
items that the vLLM serving path owns are listed; decoder-level items were
closed by the p150 decoder, full-model and datatype-sweep stages and are not
re-measured here (no profiler collection in a serving stage).

| Optimize requirement | Serving-path evidence and action |
| --- | --- |
| Operation-topology audit of the measured path | Work log "Design": the single-row prefill ran the 32-layer dense work once per row and transferred each row's recurrent state with a 256 MiB `where` broadcast per layer; the admission boundary remapped slots with the same broadcast. Actions: one dense pass per equal-bucket group, `slice_write` scatter for state rows, gathered conv rows, cycle-wise remap. |
| Decoder fully traced, device token feedback, async split | Unchanged from stage 9: model trace, sampler trace with `tt_out_tok`, `execute_trace(..., blocking=False)`, `read_decode_output(async_read=True)`, no host argmax or full-logits readback; `b32_final/server.log` shows one startup capture and zero releases. |
| Prefill and TTFT, batched prefill | `readiness_vllm/vllm_prefill_burst_benchmark.json`: 32 x 128 burst TTFT p50 927.8 ms against the serial reference 2429.0 ms (`serial_b32_v3/`), ratios p99 0.382 and mean 0.381; batched equals solo in `batched_probe_full_v3_final.json`; warm bucket list and program counts in `b32_final/server.log` (`Ornith prefill bucket warmup`). |
| Bucket calibration | `batched_calibration_full_v1.json` and `batched_probe_full_v3_final.json` calibration: every bucket wins; buckets above 512 tokens left on the single-row path (0.75 at 8x512, 0.97 at 4x1024). |
| Program keys independent of row position, warmup before capture | Gather indices, keep masks and slot masks are runtime tensors; row buckets 2/4/8/16/32 with dummy rows; the adapter warms every `(rows, seq)` bucket, every slot and two remap rotations before the first capture; tracked and untracked mixed-shape checks show zero captures. |
| Trace allocation safety | `tracked_b32_final/` (final code, `TT_METAL_TRACE_ALLOC_TRACKING=1`, program-cache allocations included): the stage 10 mixed-shape workload with multi-row groups and their slot remaps, 88/88 requests, zero captures, zero releases, no program growth, no unsafe-allocation report or tracker error over the server life; `tracked_b32/` is the same run on the first batched commit. |
| Non-aligned prompt lengths and context | `b32_final/nonaligned_requests.json`, `b1_final_v2/nonaligned_requests.json`; served `max_model_len` 262144 unchanged (`doc/context_contract.json`). |
| Batch capability up to 32 | B32 servers for sampling, qualitative, CI burst, prefill burst and mixed-shape; 31-row and 32-row groups exercised. |
| On-device sampling, no host fallback in the measured path | Primary servers run with `ORNITH_VLLM_ALLOW_HOST_SAMPLING=0`; the compatibility server is used only for the canonical sampling suite. |
| Before/after on the same harness | Primary 128/128/1: before `../../vllm_integration/p150/b1_primary/first/` 83.3 ms / 34.09 ms, after `b1_final_v2/first/` 83.9 ms / 33.9 ms (repeat 77.1 / 33.9). CI burst 100/100/32: before 3778.4 ms TTFT / 171.26 ms TPOT, after 911.3 ms / 170.4 ms. |
| Full-model comparison | One-chip traced teacher-forcing bound 29.18 t/s/u (`../../vllm_integration/p150/teacher_forcing_p150.json`); serving 29.51 t/s/u from TPOT. |
| Performance accounting | `perf_summary.json`: device-time fields null with the no-profiler reason; named limitations list the host-dispatch-bound batched pass, the admission boundary step and the scheduler's one-row first step. |
| Qualitative | `b32_final/vllm_qualitative_outputs.json`: the six greedy completions are byte-identical to the stage 9 outputs reviewed in `../../vllm_integration/p150/README.md`; degenerate-output check passes. |
| Sampling | `b32_final/sampling_tests.log`: 73 passed, 4 skipped, 1 failed (`test_frequency_penalty_changes_output`, the controlled failure of `../../vllm_integration/plugin_changes.md` entry 2). |
| Prefix caching | Excluded (`recurrent_state`), unchanged; `supports_prefix_caching=False`, `--no-enable-prefix-caching`. |
| Profiler | None collected in this stage (skill rule). |
| Watcher | `watcher/` (`TT_METAL_WATCHER=10`, chip 3, reduced two-layer model with the same op configurations): the batched probe over groups 2, 3, 5, 8, 31 and 32 rows and the 32x128 calibration, and the remap probe over six permutations; watcher logs kept beside the probe outputs. See the work log for the result. |
| Commits | `c4e2ee8efd`, `5469aae4b3`, `e70875637c`, `5cff1036cc`, `9d4de01a6d`, `66bd9e836b`, `b6b1fdb29a` and the final evidence commit recorded in the work log; never pushed. |
