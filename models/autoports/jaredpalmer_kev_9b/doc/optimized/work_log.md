# Stage 3 (optimize): work log

Date: 2026 Oct 01, 20:30 ET onward. One chip (chip 0 through `ttnn.open_device(device_id=0)`), `l1_small_size=24576`, `num_command_queues=2`, `trace_region_size=1 GiB`. All paths are absolute. Logs under `/home/hous/dev/kev/logs`, compact evidence under `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/optimized`.

Acronyms: GDN (Gated DeltaNet), KV (key/value), SDPA (scaled dot-product attention), PCC (Pearson correlation coefficient), CB (circular buffer), MLP (multi-layer perceptron), LoRA (low-rank adaptation), bfp4 / bfp8 (block floating point, 4 or 8 bits).

## Environment

- Interpreter `/home/hous/dev/kev/tt-metal/python_env/bin/python` through `/home/hous/dev/kev/bin/devrun` (device, exclusive lock) and `/home/hous/dev/kev/bin/hostrun`.
- `HF_MODEL`, `KEV_RUN`, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache`, `HF_HUB_OFFLINE=1`, `KEV_MESH_SHAPE=1x1`, `KEV_DEVICE_ID=0` as in the stage 2 work log (`/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/server/work_log.md`).
- Profiler: the tt-metal build has `ENABLE_TRACY=ON` and `build/tools/profiler/bin/tracy-capture`; `tt-perf-report` 1.4.0 at `/home/hous/.tenstorrent-venv/bin/tt-perf-report`.
- Skill mapping (prefill-only model, no decode): "traced decode" maps to the traced question tail (one bucketed segment per question) and the traced 2048-token state chunk; "TTFT" maps to the server `latency_ms` of one request (state prefill or cache hit, every question tail, head on CPU); decode items of the optimize skill (sampling, LM head, token feedback, DRAM-sharded decode matmuls) do not apply and are recorded as not applicable in README.md.

## Item 1: where the eager time goes (before)

Script `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/scripts/perf_probe.py` (stage 2 engine code, eager, 32 layers, merged weights), `--reps 5`, log `/home/hous/dev/kev/logs/stage3_probe_eager.log`, JSON `doc/optimized/perf_probe_eager.json`. Split per question through `ttnn.synchronize_device` after each step (restore of the GDN snapshot, forward, readout).

| path | tokens | restore ms | forward ms | readout ms | total ms |
|---|---|---|---|---|---|
| tail bucket 128 | 50 | 0.55 | 128.3 | 0.64 | 129.5 |
| tail bucket 256 | 200 | 0.50 | 194.0 | 0.68 | 195.1 |
| tail bucket 512 | 450 | 0.53 | 482.6 | 0.60 | 483.7 |
| tail bucket 1024 | 1000 | 0.56 | 669.5 | 0.79 | 670.8 |
| tail bucket 2048 | 2000 | 0.56 | 1575.4 | 0.84 | 1576.8 |
| state 2048 (one masked bucket, valid_len 2048) | 2048 | | | | 1507.7 |
| state 2392 (S0 2304: 2048 + masked 256) | 2392 | | | | 1692.0 |
| card short (S 31, 6 questions of 42/73/47/22/16/22 tokens) | 89 | | | | 748.7 new / 746.5 cached |
| card long (S 2192, 5 questions) | 2392 | | | | 2269.9 new (state 1594.6 + questions 675.3) / 650.0 cached |

The card rows reproduce the stage 2 server numbers (753 / 751 and 2271 / 648 ms). Host work outside the forward (snapshot restore, one-hot upload, readback, norm) is under 1.5 ms per question; the forward is the whole cost.

Device profiler attempt on the full 32-layer eager engine (`TT_METAL_DEVICE_PROFILER=1`, log `/home/hous/dev/kev/logs/stage3_probe_eager_profile.log`): the run completed but `generated/profiler/.logs/profile_log_device.csv` reached 58 GB and `process_ops_logs.py` found no op-level logs (the device profiler alone does not write `tracy_ops_data.csv`). The dump was deleted. Per the optimize skill, the op-level profile was taken on a reduced 4-layer variant (3 GDN layers + 1 full-attention layer, the real repeating pattern of the 32-layer stack) under Tracy instead; see item 5.

## Items 2 and 3: trace design

`/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/engine.py`, `KevEngine(traced=True)`.

- Persistent device buffers, all allocated before the first capture: per bucket b in {128, 256, 512, 1024, 2048}: token buffer `[1, b]` uint32, chunk page table `[1, b/64]` int32, cos/sin `[1, b, 64]` bf16, one-hot selector `[1, 32, b]` bf16, output hidden `[1, b, 4096]` bf16 and selected rows `[1, 32, 4096]` bf16; shared: `chunk_start_idx` `[1]` int32 and the full page table `[1, blocks_per_slot]` int32. Snapshot slots (GDN recurrent and conv state per layer) are allocated before the traces too.
- Traces (27 at 8 slots, 273 MiB of the 1 GiB trace region at 32 layers): `fwd[b]` = `Qwen36Model._forward_prefill_chunk(tok, cos, sin, csi, pt_full, cpt)` (the same body as `capture_prefill_trace_chunked`, `valid_len=None`, flexible SDPA with the runtime `chunk_start_idx`) followed by `ttnn.copy` into the pre-allocated hidden buffer; `gather[b]` = one-hot matmul (HiFi4, fp32 accumulate) of up to 32 rows, final RMSNorm, copy into the rows buffer; `restore[slot]` and `save[slot]` = 48 `ttnn.copy` between the live GDN state and snapshot slot; `zero` = `_reset_dn_state_inplace`.
- Warm order: allocate KV caches and all persistent buffers, warm the copies and every `fwd[b]` and `gather[b]` eagerly, synchronize, `set_program_cache_misses_allowed(False)`, then capture everything. The bucket-sized bodies are captured by the engine (not by `capture_prefill_trace_chunked`) so that every allocation and every compile happens before the first capture; the upstream helper would capture the 2048 trace before the bucket buffers exist. The 2048 chunk trace is the same body with the 2048 buffers.
- Per question: `restore[slot]`, write tokens (padded with zeros to the bucket), `chunk_start_idx = S0`, the slot's page table, the chunk page table slice, cos/sin for positions `[S0, S0 + b)`, `execute_trace(fwd[b], blocking=False)`, write the one-hot rows, `execute_trace(gather[b])`, read back `[32, 4096]` bf16 (256 KB). No device allocation and no compile after setup.
- State: `zero`, then `fwd[2048]` per full chunk, then the aligned remainder (a multiple of 128 under 2048) as exact full buckets in descending order (for example 384 = 256 + 128), each with `valid_len=None` and no padding so the carried GDN state is exact; then `save[slot]`. The state remainder is never padded.
- `prefill_hidden` (reference rows) uses the same traces: 2048 chunks, then the last partial chunk padded to its bucket (post-segment state discarded), rows gathered per segment.
- Program cache misses stay forbidden after setup in traced mode, so an un-warmed op would raise instead of compiling over a parked trace.

### L1 circular-buffer clash at bucket 512

First traced run (`/home/hous/dev/kev/logs/stage3_probe_traced_l4.log`, 20:49 ET): `gated_delta_attn_seq` raised "Statically allocated circular buffers in program ... clash with L1 buffers" while warming bucket 512. Cause: `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_deltanet.py` places the GDN inputs in L1 for `T <= _L1_SEQ_THRESHOLD = 512` when `valid_len is None`; the masked path avoided it by forcing DRAM whenever `valid_len` is set (which is also why upstream's `test_prefill.py` notes that `prefill_paged` throws for lengths in (256, 512]). Fix in the engine only: `_setup_traces` sets `ttnn_gated_deltanet._L1_SEQ_THRESHOLD = 256`. This changes behaviour only for segments in (256, 512] with `valid_len=None`, which previously raised, so existing callers are unaffected. No qwen36 file was edited for it.

### Gates for the traced engine (items 2 and 3)

`tests/test_engine.py` gained `test_slots_interleaved[eager|traced]` (two states in two KV slots, question on A, on B, on A again bit-identical, both against `prefill_hidden` of the concatenated row in a third and fourth slot; the traced run is also compared with the eager run of the same test), `test_tail_buckets[eager|traced]` (S 2048 state, tails of 50 / 200 / 450 / 1000 / 2000 tokens, three back-to-back questions per bucket bit-identical, traced compared with eager per bucket) and `test_long_state` (slow). The eager parameter runs first and stores its outputs; the traced parameter compares against them (PCC bar 0.999).

4 layers (`/home/hous/dev/kev/logs/stage3_engine_l4.log`, 10 passed, 126 s): `tail_matches_full_row` 6/6 (PCC 1.000000 at S 2048 / 2175, 0.99984 and 0.99991 at S 2200 as in stage 1); slots eager A 0.99992+ B 0.99994+, traced A 0.99986+ B 0.99994+, traced vs eager min 0.999893 (A has a 384-token aligned remainder, run as 256 + 128 traced pieces against one masked 512 bucket eagerly) and 1.000000 (B); buckets traced vs eager 1.000000 on all five buckets.

32 layers (`/home/hous/dev/kev/logs/stage3_engine_l32.log`, 14 passed, 414 s): `hidden_vs_hf` T 300 / 1500 / 2300 pass on the traced engine (base weights, `Qwen36ModelArgs` asserted); `tail_matches_full_row` 6/6 with the stage 1 PCCs (1.000000 at S 2048 / 2175, 0.999878 / 0.999872 at S 2200); `test_reference_records` on the traced engine with `KevModelArgs` asserted: 29 rows, min PCC 0.976770, argmax 29/29, max |dp| 0.102370, mean |dp| 0.036647 (stage 1 eager: 0.102370 and 0.0366, unchanged); slots traced vs eager 0.999950 / 1.000000, both against the full row 0.99992+; buckets traced vs eager 0.999999 / 1.000000 x4.

Steady-state question time at 32 layers (third call, ms): eager 128.6 / 194.5 / 483.6 / 670.2 / 1576.4 against traced 106.5 / 166.5 / 467.2 / 643.7 / 1526.3 for tails of 50 / 200 / 450 / 1000 / 2000 tokens.

### Traced engine numbers before device-side tuning (item 1, after tracing)

`perf_probe.py --traced --trace-region 1073741824 --reps 5`, log `/home/hous/dev/kev/logs/stage3_probe_traced.log`, JSON `doc/optimized/perf_probe_traced.json`. Engine build 8 slots (KV per slot 2.06 GiB at `max_state_len` 65536, DRAM free 21.26 GiB after the weights), 27 traces, trace region used 273.1 MiB of 1 GiB.

| path | eager ms | traced ms |
|---|---|---|
| tail bucket 128 (restore / forward / readout) | 0.55 / 128.3 / 0.64 | 0.31 / 106.2 / 0.46 |
| tail bucket 256 | 195.1 | 166.9 |
| tail bucket 512 | 483.7 | 468.1 |
| tail bucket 1024 | 670.8 | 645.6 |
| tail bucket 2048 | 1576.8 | 1528.7 |
| state 2048 | 1507.7 | 1510.7 |
| state 2392 | 1692.0 | 1676.6 |
| card short new / cached | 748.7 / 746.5 | 619.1 / 618.1 |
| card long new / cached | 2269.9 / 650.0 | 2153.0 / 535.9 |

The card probabilities are identical to the eager run to the printed 16 digits for the short case and for the long case. Tracing removed the host dispatch gap (about 20 ms per 128-token tail) and nothing else: the 2048-token chunk is pure device time (the Tracy window below sums to the same 190 ms per 4 layers that the wall clock shows), so the remaining work is on-device op efficiency.

## Item 5: device-side optimization (optimize skill, prefill mapping)

### Op-level profile of the traced engine (reduced 4-layer variant)

`python -m tracy -r -p -v models/autoports/jaredpalmer_kev_9b/scripts/perf_probe.py --profile --layers 4 --slots 4 --traced --trace-region 1073741824` with `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000`, log `/home/hous/dev/kev/logs/stage3_tracy_traced_l4.log`, ops CSV and per-window `tt-perf-report` tables (`--tracing-mode`, advice enabled) under `doc/optimized/tracy/traced/` (`PERF_TAIL_Q{50,200,450,1000,2000}` and `PERF_STATE_S{2048,2392}` windows; the `PERF_TAIL_Q50` window is empty in the ops CSV, the bucket-128 replay rows were attributed before the signpost by the post-processor). The raw `profile_log_device.csv` (5.7 GB) and `tracy_ops_times.csv` (6.7 GB) were deleted after the ops CSV was produced.

One 2048-token forward through 3 GDN layers + 1 full-attention layer (`PERF_STATE_S2048`): 739 ops, device kernel time 188.2 ms, op-to-op gap 0.9 ms, so the chunk is pure device kernel time (x8 = 1.51 s, the measured 32-layer chunk). By op:

| op | ms per 4 layers | share | note |
|---|---|---|---|
| MatmulDeviceOperation | 99.3 | 52 % | every large matmul flagged SLOW by tt-perf-report at 5 to 7.5 % FLOPs utilization: gate/up 2048x4096x12288 bf16 x bfp4 6.66 ms each (x8), down 2048x12288x4096 bfp8 4.50 ms (x4), GDN in-proj 2048x4096x12352 bfp8 4.85 ms (x3), GDN out-proj fp32 x bfp8 1.63 ms (x3), attention q/gate 2048x4096x8192 3.02 ms, o-proj 1.53 ms |
| ReshapeViewDeviceOperation | 31.2 | 16 % | head split and merge of q/k/v/gate/o in `ttnn_gated_deltanet.py` ([1,2048,4096] tile <-> [1,2048,32,128]), 1.6 to 2.9 ms each |
| BinaryNgDeviceOperation | 22.0 | 12 % | GDN chunk preprocessing in fp32 (WY inverse, masks), DRAM-bound at 420 GB/s |
| Unary, Slice, Tilize, Permute, Transpose, Untilize, Concat, Ternary, LayerNorm | 25.5 | 13 % | GDN relayouts and norms |
| GatedDeltaAttnSeqDeviceOperation | 2.9 | 1.5 % | the GDN scan kernel itself |
| SDPAOperation | 1.2 | 0.7 % | |

Bucket 512 (`PERF_TAIL_Q450`): 58.1 ms per 4 layers of which the MLP down projection 512x12288x4096 in L1 costs 5.4 ms each (x4 = 21.6 ms, 7.3 TFLOPs); the same matmul in DRAM takes 1.4 ms with the default config and 0.29 ms with a tuned 2D config (sweep below).

### Matmul program-config sweep (OPT prefill matmul tuning)

`scripts/matmul_sweep.py` (random weights at the real shapes and dtypes, in0 bf16 DRAM interleaved, median of 5 after one warm call, PCC against a host fp32 product of the same quantized weights; `doc/optimized/matmul_sweep.json` for M 128 / 512 / 2048 and `matmul_sweep_256_1024.json`; logs `/home/hous/dev/kev/logs/stage3_matmul_sweep.log`, `stage3_matmul_sweep2.log`). Candidates: the current default (no program config, LoFi, fp32 accumulate), the same without fp32 accumulate, the upstream `prefill_progcfg` 2D config (in0_block_w 4, out_subblock_h 1), a 2D `MatmulMultiCoreReuseMultiCastProgramConfig` sweep over grids 11x10 / 11x8 / 8x8 / 10x10 etc., in0_block_w 1..16 and legal subblocks, and `ttnn.experimental.minimal_matmul`. Compute grid is 11x10 (110 cores). Only candidates that keep LoFi with fp32 accumulation were adopted (identical math to the stage 2 path; PCC equals the default's in every row). Dropping fp32 accumulation was 1.3 to 3.7x faster on its own but changes the numerics (PCC 0.9963 to 0.9998 against the default's 0.99991 to 1.0003) and is left to the stage 4 datatype sweep.

| shape (M x K x N, weight) | default ms | prefill_progcfg ms | adopted | adopted ms |
|---|---|---|---|---|
| 2048 x 4096 x 12288 bfp4 (gate/up) | 6.751 | L1 overflow | minimal_matmul | 1.013 |
| 2048 x 12288 x 4096 bfp8 (down) | 4.563 | 1.560 | 2D 11x10 in0 16 sub 1x4 pcM 7 pcN 12 | 0.799 |
| 2048 x 4096 x 12352 bfp8 (GDN in-proj) | 4.919 | L1 overflow | minimal_matmul | 0.999 |
| 2048 x 4096 x 8192 bfp8 (attention q/gate) | 3.089 | 1.113 | minimal_matmul | 0.662 |
| 2048 x 4096 x 4096 bfp8 (o-proj) | 1.636 | 0.579 | 2D 11x10 in0 16 sub 1x4 pcM 7 pcN 12 | 0.324 |
| 1024 x 4096 x 12288 | 1.431 | 1.359 | minimal_matmul | 0.510 |
| 1024 x 12288 x 4096 | 2.670 | 0.914 | 2D 11x8 in0 16 sub 1x4 pcM 4 pcN 12 | 0.465 |
| 1024 x 4096 x 12352 | 1.451 | 0.938 | minimal_matmul | 0.534 |
| 1024 x 4096 x 8192 | 1.789 | 0.648 | minimal_matmul | 0.341 |
| 1024 x 4096 x 4096 | 0.944 | 0.350 | 2D 11x10 in0 16 sub 1x4 pcM 4 pcN 12 | 0.201 |
| 512 x 4096 x 12288 | 0.744 | 0.725 | 2D 11x10 in0 16 sub 2x1 pcM 2 pcN 35 | 0.309 |
| 512 x 12288 x 4096 | 1.402 (7.090 with L1 output) | 0.483 | 2D 11x8 in0 16 sub 1x4 pcM 2 pcN 12 | 0.294 |
| 512 x 4096 x 12352 | 0.750 | 0.502 | minimal_matmul | 0.357 |
| 512 x 4096 x 8192 | 0.569 | 0.355 | 2D 11x10 in0 16 sub 1x4 pcM 2 pcN 24 | 0.227 |
| 512 x 4096 x 4096 | 0.495 (2.404 with L1 output) | 0.200 | 2D 11x8 in0 16 sub 1x4 pcM 2 pcN 12 | 0.135 |
| 256 x 4096 x 12288 | 0.400 | 0.380 | 2D 11x10 in0 16 sub 1x1 pcM 1 pcN 35 | 0.243 |
| 256 x 12288 x 4096 | 0.552 | 0.311 | 2D 11x10 in0 16 sub 1x1 pcM 1 pcN 12 | 0.280 |
| 256 x 4096 x 12352 | 0.405 | 0.301 | 2D 11x8 in0 8 sub 1x2 pcM 1 pcN 36 | 0.280 |
| 256 x 4096 x 8192 | 0.314 | 0.219 | 2D 11x10 in0 16 sub 1x2 pcM 1 pcN 24 | 0.204 |
| 256 x 4096 x 4096 | 0.220 | 0.140 | 2D 11x8 in0 16 sub 1x1 pcM 1 pcN 12 | 0.129 |
| 128 x 12288 x 4096 | 0.340 | 0.310 | 2D 11x10 in0 16 sub 1x4 pcM 1 pcN 12 | 0.270 |
| 128 x 4096 x 4096 | 0.152 | 0.132 | 2D 11x10 in0 16 sub 1x2 pcM 1 pcN 12 | 0.120 |
| 128 x 4096 x 12288 / 12352 / 8192 | 0.230 / 0.229 / 0.179 | slower | default kept | |

Application without editing qwen36: `KevEngine(matmul_policy=True)` (env `KEV_MATMUL_POLICY`) rebinds `ttnn.linear` in the engine's process to `policy_linear` (`tt/engine.py`), which looks up `(M, K, N)` of a rank-3, batch-1 input against a rank-2 weight in `MATMUL_POLICY` and runs the adopted kernel with the caller's compute kernel config; everything else (other shapes, fp32 inputs, biased or 4D calls) passes through unchanged. For policy matmuls at M >= 256 the output goes to DRAM interleaved (the L1 output at M 512 was the 5.4 ms pathology above); at M 128 the caller's memory config is kept. A fused `activation="silu"` (MLP gate) becomes a separate `ttnn.silu` after the policy matmul. Every engine construction sets `ttnn.linear` to match its own flag, so eager and policy engines can alternate in one process.

### Rejected: fused GDN chunk op on the single-device path

`ttnn.transformer.chunk_gated_delta_rule` (the TP path's "fast path" in `tt/gdn/fused_chunk.py`) was tried as a drop-in for `chunk_gated_delta_rule_seq_adapter` through a module-level rebind in the engine. 4 layers, traced: bucket 128 forward 13.5 to 10.0 ms, 2048 chunk 189 to 163 ms (`doc/optimized/perf_probe_traced_l4_gdnfused.json`, log `/home/hous/dev/kev/logs/stage3_probe_traced_l4_gdnfused.log`). Correctness failed: `test_slots_interleaved[l4-fused]` gave hidden PCC 0.916 / -0.032 / -0.036 / 0.994 against the full row for state A (2450 tokens, segments 2048 + 256 + 128 + tail) while state B (2048 + 128 + tail) was 0.99999, and the process ended with a segmentation fault at teardown (`/home/hous/dev/kev/logs/stage3_engine_l4_fused.log`, rc 139). The carried-state contract of the fused op on this path is not established, so the code was removed; the seq adapter stays. The remaining GDN relayout and fp32 elementwise cost (about 30 % of the 2048 chunk) is recorded as an open item for a kernel-level change in qwen36.

### Matmul policy gates and 32-layer numbers

4 layers (`/home/hous/dev/kev/logs/stage3_engine_l4_policy.log`, 6 passed): `policy` mode (traced + policy) against the eager stage 2 path, slots min PCC 0.999893 / 1.000000 (same values as traced without the policy) and buckets 1.000000 on all five buckets, so the adopted kernels reproduce the default kernels' results. Steady-state 4-layer question time traced -> policy: 13.8 -> 13.5 (bucket 128), 21.2 -> 19.3 (256), 58.9 -> 33.5 (512), 81.0 -> 60.1 (1024), 191.6 -> 115 ms (2048); state 2048 189 -> 113 ms.

32 layers, policy on by default (`/home/hous/dev/kev/logs/stage3_engine_l32_policy.log`, 13 passed, 378 s): `tail_matches_full_row` 6/6 unchanged (1.000000 / 0.999878 / 0.999872), `test_reference_records` min PCC 0.976770, 29/29, max |dp| 0.102370, mean |dp| 0.036647 (bit-for-bit the stage 1 and stage 2 numbers), slots and buckets policy vs eager 0.999950 / 1.000000 and 0.999999 / 1.000000 x4.

`perf_probe.py --traced --matmul-policy --reps 5` at 32 layers (`/home/hous/dev/kev/logs/stage3_probe_traced_policy.log`, `doc/optimized/perf_probe_traced_policy.json`): 27 traces, 273.8 MiB trace region, probabilities identical to the eager run.

| path | eager (stage 2) ms | traced ms | traced + policy ms |
|---|---|---|---|
| tail bucket 128 (Q 50) | 129.5 | 107.0 | 104.9 |
| tail bucket 256 (Q 200) | 195.1 | 166.9 | 151.0 |
| tail bucket 512 (Q 450) | 483.7 | 468.1 | 266.4 |
| tail bucket 1024 (Q 1000) | 670.8 | 645.6 | 476.4 |
| tail bucket 2048 (Q 2000) | 1576.8 | 1528.7 | 917.3 |
| state 2048 | 1507.7 | 1510.7 | 898.9 |
| state 2392 (2048 + 256 + suffix 88) | 1692.0 | 1676.6 | 1048.7 |
| card short new / cached | 748.7 / 746.5 | 619.1 / 618.1 | 605.0 / 604.8 |
| card long new / cached | 2269.9 / 650.0 | 2153.0 / 535.9 | 1527.5 / 525.6 |

### Why bucket 128 stays near 100 ms (named limitation)

The `PERF_TAIL_Q200` window (bucket 256, 4 layers) has 745 device ops with a median kernel time of 6.3 us; the kernel sum is 20.2 ms but the firmware-duration sum is 33.7 ms, so about 18 us of launch overhead per op dominates. At 32 layers a bucket-128 tail is about 5900 ops, which at the per-program launch floor explains the 105 ms: the small buckets are op-count bound, not compute bound. Per 4 layers at bucket 256: matmuls 7.3 ms over 122 calls, GDN head reshapes 3.6 ms over 55 calls, GDN fp32 elementwise 2.9 ms over 214 calls, SDPA 1.1 ms. Lowering this needs fewer ops per GDN layer (the fused chunk op or a fused relayout) inside qwen36, which stage 3 does not own; it is the first candidate for a backbone change. Batching the question rows of one request into a single segment was considered and rejected: the rows must not attend to each other and the GDN state must not carry across rows, which the chunked SDPA and GDN kernels used here cannot express.

## Item 4: host overhead

Per question in traced mode the host does: one `copy_host_to_device_tensor` each for the tokens (padded to the bucket), `chunk_start_idx`, the slot page table (pre-built host tensor), the chunk page-table slice, cos and sin (host `prefill_cos_sin_torch` for `[S0, S0 + b)`), two `execute_trace`, one one-hot upload `[1, 32, b]` and one `to_torch` of `[1, 32, 4096]` bf16 (256 KB). Measured restore + readout at 32 layers: 0.31 + 0.46 ms (eager 0.55 + 0.64 ms). The head stays on CPU in fp32 and already batches all options of a question in one `probs` call (`PointerHead.probs(h[-1], h[:-1])`). The remaining host share of a request is under 1 % of `latency_ms`, so nothing else was moved.

## Item 6: stage 1 tests on the traced engine, allocation tracker, watcher, long states

- `hidden_vs_hf` (l32, base weights, T 300 / 1500 / 2300), `tail_matches_full_row` (l32, 6 cases), `test_reference_records`: all pass on the traced engine with the matmul policy; reference records min PCC 0.976770, 29/29, max |dp| 0.102370, mean |dp| 0.036647 (stage 1 and stage 2: 0.102370 / 0.0366). Logs `/home/hous/dev/kev/logs/stage3_engine_l32.log` (traced, before the policy) and `stage3_engine_l32_policy.log` (policy).
- Trace allocation tracker: `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 devrun pytest tests/test_engine.py -k "l4 and (slots or tail_buckets) and policy"` in a fresh process (two states in two slots, two reference rows in two more slots, five buckets with three replays each, all 27 traces captured first). No `RuntimeError` from the tracker, no acknowledgments and no corruptible scopes are used: every persistent buffer is allocated before the first capture and the replay path allocates nothing on the device. Log `/home/hous/dev/kev/logs/stage3_trace_alloc_tracking.log`. The first run of this command ended as `2 skipped` because the cross-mode comparison skipped when no eager result existed in the process (all replays and the bit-identity asserts had run); the comparison now logs and returns instead, and the run was repeated.
- Watcher: `TT_METAL_WATCHER=10` on the same selection, separate run, `/home/hous/dev/kev/logs/stage3_watcher.log` copied from `generated/watcher/watcher.log`: no error, assert, overflow or hang entries. Watcher and profiler were never combined.
- The single Metal warning "Allocating device buffers is potentially unsafe due to the existence of an active trace" appears once per engine build, between the first and the last capture (later captures allocate their intermediates while earlier traces are live); the tracker run confirms no live buffer is unsafe at replay.
- `test_long_state` (`/home/hous/dev/kev/logs/stage3_long_state.log`, 1 passed, 86 s): engine at `max_state_len` 65536 with 2 slots; S 16384 (8 chunks) question against the full row in the other slot PCC 1.000000 / 1.000000 / 1.000000; S 65536 (32 chunks) prefill plus one question 37.65 s wall, finite output, no crash.

## Item 7: server

Driver `/tmp/claude-1002/-home-hous-dev-kev/0a6793e1-6f42-41ef-926e-8b91dbe0b95b/scratchpad/server_stage3.sh` (scratch): `devrun timeout 5400 python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/kev/logs/stage3_server.log` with the stage 2 environment plus the new defaults (`KEV_MAX_STATE` 65536, `KEV_TRACED` 1, `KEV_MATMUL_POLICY` 1). Ready in 44 s (`worker 0 ready`, 8 slots, 27 traces, 273.8 MiB). `/v1/models`: `max_state_tokens 65536`, prefix cache size 8. Quickstart request 202.4 ms.

Parity (`scripts/parity_remote.py --passes 2`, `/home/hous/dev/kev/reports/stage3_parity.json`, log `/home/hous/dev/kev/logs/stage3_parity.log`): max |dp| 0.0947, mean 0.0358, 0 flips vs fp32 (stage 2 0.0947 / 0.0359); every one of the 16 revisits after the other states had run returned the first-pass answers (the prefix cache held 8 of 16 states, 24 misses / 10 hits), which is the server-side check of the per-slot KV fix.

Bench (`serving_bench_remote.py --reps 20 --quick --concurrency 1,8,32,64`, `/home/hous/dev/kev/reports/bench/p150_stage3/report.json`, log `/home/hous/dev/kev/logs/stage3_bench.log`): card row `| P150 (1 chip, traced, stage 3) | 604.7 / 605.2 ms | 1528.5 / 525.5 ms | 1.6 |` (throughput in `--quick` mode, 32 requests per level, as in stage 2). 790 requests, 0 5xx, 0 tracebacks in the server log.

Shutdown at 21:40 ET: SIGTERM to the uvicorn python process, `Application shutdown complete`, `Finished server process`, lock free, device reopened by a fresh process (`/home/hous/dev/kev/logs/stage3_device_release_check.log`). The driver's own `kill -TERM $(pgrep -f ... | head -1)` had hit the `timeout` wrapper and left the server up for one more minute; noted so the next driver kills the python process.

## Files touched in stage 3

`tt/engine.py` (rewritten: per-slot KV, direct `KevModelArgs`, traces, matmul policy), `tt/server.py` (max_state 65536, `KEV_TRACED`, slot return on failure, per-slot `FakeEngine`), `tests/test_engine.py` (new tests and modes, args-class asserts, mean |dp|), `tests/test_server_api.py` (max_state, eviction / revisit and failed-prefill tests), new `scripts/perf_probe.py`, `scripts/matmul_sweep.py`, `scripts/parity_remote.py`, `doc/optimized/*`, `doc/server/README.md` (stage 3 section, review items), `doc/context_contract.json`. No file under `models/demos/blackhole/qwen36` or `models/experimental` was edited by stage 3 (the stage 4 agent's `precision.py` and env knobs there were left untouched). Nothing committed.
