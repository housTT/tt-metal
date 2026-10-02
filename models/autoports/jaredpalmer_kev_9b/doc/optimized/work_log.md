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

Recomputed from `doc/optimized/tracy/traced/ops_perf_results_traced_l4.csv` (rows between the `PERF_*` and `PERF_*_END` signposts, device clock 1.35 GHz from the FW cycle and duration columns; the stage review's recomputation gives the same numbers):

| window (4 layers) | device ops | kernel sum ms | FW sum ms | device span ms | idle ms | median kernel us | (FW - kernel) per op us |
|---|---|---|---|---|---|---|---|
| `PERF_TAIL_Q50` (bucket 128) | no device rows | | | | | | |
| `PERF_TAIL_Q200` (bucket 256) | 745 | 20.17 | 33.71 | 21.42 | 0.74 | 6.3 | 18.2 |
| `PERF_TAIL_Q450` (bucket 512) | 736 | 58.09 | 81.82 | 59.19 | 0.70 | 10.6 | 32.2 |
| `PERF_TAIL_Q1000` (bucket 1024) | 736 | 80.18 | 131.25 | 81.38 | 0.81 | 21.1 | 69.4 |
| `PERF_TAIL_Q2000` (bucket 2048) | 736 | 190.37 | 227.83 | 192.03 | 1.27 | 40.8 | 50.9 |
| `PERF_STATE_S2048` | 739 | 188.23 | 224.45 | 189.07 | 0.45 | 40.5 | 49.0 |

What the data shows: the kernels alone fill 94 % of the bucket-256 span and the device is idle (no FW running) for 0.74 ms of 21.42 ms, so the small buckets are not waiting on launches. The FW sum exceeds the span by 57 %, so FW durations overlap across cores and ops; their excess over the kernel sum grows with the bucket (18 to 69 us per op), which a fixed per-program launch floor cannot produce. The earlier "18 us launch floor" sentence derived from FW sum minus kernel sum is withdrawn.

What is actually known about the small buckets: the bucket-256 forward is many short kernels with low utilization. Top ops by kernel time in `PERF_TAIL_Q200`: MatmulDeviceOperation 122 calls, 7.27 ms (36.0 %, 59.6 us mean, against 814 us mean for the same 122 calls at bucket 2048); ReshapeViewDeviceOperation 55 calls, 3.60 ms (17.9 %); BinaryNgDeviceOperation 214 calls, 2.92 ms (14.5 %, 13.7 us mean); SDPAOperation 1 call, 1.09 ms (5.4 %); TilizeDeviceOperation 33 calls, 0.79 ms; LayerNormDeviceOperation 20 calls, 0.76 ms; UntilizeWithUnpaddingDeviceOperation 24 calls, 0.66 ms; UnaryDeviceOperation 58 calls, 0.51 ms. 537 of the 745 kernels run under 20 us and hold 16.5 % of the kernel time; the other 208 kernels hold 83.5 %. The op count is the same at every bucket (736 to 745 per 4 layers), so a 32-layer forward is about 5900 ops at any bucket; this is an observation about the graph, not the bound. The bucket-128 window has no device rows (the post-processor attributed its replay rows before the signpost), so the 105 ms bucket-128 tail at 32 layers is not decomposed here; a measurement needs a bucket-128 window with a working signpost. Lowering the small-bucket time needs fewer or larger ops per GDN layer inside qwen36 (fused relayout or the fused chunk op), which stage 3 does not own. Batching the question rows of one request into a single segment was considered and rejected: the rows must not attend to each other and the GDN state must not carry across rows, which the chunked SDPA and GDN kernels used here cannot express.

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

## Stage 4 follow-up: stage 3 remediation (2026 Oct 01, 22:35 ET onward)

Agent: stage 3 remediation plus the stage 4 follow-up (review `/home/hous/dev/kev/reports/review_stage3.md`). Chip 0 through `devrun`, the selected precision `mlp_bfp8` from `tt/precision_defaults.py` (bfp8 gate / up, bfp8 elsewhere, LoFi + fp32 accumulate), weight cache `/home/hous/dev/kev/tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8`. Logs `/home/hous/dev/kev/logs/stage4r_*.log`.

### Matmul policy, correct by construction (task 1)

`policy_linear` in `tt/engine.py` now (a) keys `MATMUL_POLICY` on `(M, K, N, in1 dtype, in0 dtype)`, so a bfp4 and a bfp8 weight at the same shape, or an fp32 activation (the GDN out-projection), cannot reuse a config swept for another dtype; (b) passes every caller keyword through unchanged (`compute_kernel_config`, `memory_config`, `dtype`, and anything a future caller adds; `activation="silu"` is still split into a separate `ttnn.silu` after the policy matmul), and applies the policy only when the caller gave no `program_config` and no `bias`; (c) falls back to the plain `ttnn.linear` call for any key not in the table. The MLP down projection reaches the policy because the engine sets `QWEN9B_MLP_DOWN_AUTO=1` (the qwen36 switch that makes `mlp.py` pass `program_config=None`) when the policy is on; with the policy off the upstream `prefill_progcfg` is used as before. For policy matmuls at M >= 256 the output memory config is DRAM interleaved as in stage 3; at M = 128 the caller's is kept.

Sweep with bfp8 gate / up weights: `scripts/matmul_sweep.py --ms 128,256,512,1024,2048 --gate-up-dtype bfp8` (new option; the shapes other than gate / up were bfp8 already), chip 0, `doc/optimized/matmul_sweep_bfp8.json`, log `/home/hous/dev/kev/logs/stage4r_matmul_sweep_bfp8.log` (22:38 to 22:39 ET, 909 timed rows, 376 configs rejected by the runtime, mostly the bfp8 L1 overflow `dataflow_buffer.cpp:2682` that stopped the stage 4 sweep with the stage 3 table). Selection rule per key: among rows with LoFi and fp32 accumulation and DRAM output whose PCC against the host fp32 product of the same quantized weights is within 2e-4 of the default row's, the fastest; adopted only when at least 3 % faster than the default row, else no entry (the plain call is used). Every adopted row has the same PCC as the default row to six digits.

| shape (K x N), bfp8 weight | M | default ms | adopted (LoFi, fp32 acc) | ms | stage 3 best eligible ms (bfp4 gate/up) | configs rejected by the runtime |
|---|---|---|---|---|---|---|
| gate/up 4096 x 12288 | 128 | 0.232 | default kept | 0.232 | 0.23 | 10 |
| gate/up 4096 x 12288 | 256 | 0.405 | 2d grid=11x8 in0=8 sub=1x1 pcM=1 pcN=35 | 0.284 | 0.243 | 10 |
| gate/up 4096 x 12288 | 512 | 0.752 | minimal_matmul | 0.365 | 0.309 | 28 |
| gate/up 4096 x 12288 | 1024 | 1.479 | minimal_matmul | 0.529 | 0.51 | 40 |
| gate/up 4096 x 12288 | 2048 | 6.867 | minimal_matmul | 1.041 | 1.013 | 42 |
| down 12288 x 4096 | 128 | 0.339 | 2d grid=11x10 in0=16 sub=1x4 pcM=1 pcN=12 | 0.267 | 0.27 | 0 |
| down 12288 x 4096 | 256 | 0.551 | 2d grid=11x8 in0=16 sub=1x4 pcM=1 pcN=12 | 0.285 | 0.28 | 0 |
| down 12288 x 4096 | 512 | 1.365 | 2d grid=11x10 in0=16 sub=1x4 pcM=2 pcN=12 | 0.288 | 0.294 | 0 |
| down 12288 x 4096 | 1024 | 2.700 | 2d grid=11x10 in0=16 sub=1x4 pcM=4 pcN=12 | 0.471 | 0.465 | 0 |
| down 12288 x 4096 | 2048 | 4.556 | 2d grid=11x10 in0=16 sub=1x4 pcM=7 pcN=12 | 0.795 | 0.799 | 19 |
| GDN in-proj 4096 x 12352 | 128 | 0.225 | default kept | 0.225 | 0.229 | 10 |
| GDN in-proj 4096 x 12352 | 256 | 0.411 | 2d grid=11x10 in0=8 sub=1x2 pcM=1 pcN=36 | 0.280 | 0.28 | 10 |
| GDN in-proj 4096 x 12352 | 512 | 0.756 | 2d grid=11x10 in0=8 sub=1x4 pcM=2 pcN=36 | 0.357 | 0.357 | 22 |
| GDN in-proj 4096 x 12352 | 1024 | 1.454 | minimal_matmul | 0.517 | 0.534 | 40 |
| GDN in-proj 4096 x 12352 | 2048 | 4.950 | minimal_matmul | 1.034 | 0.999 | 42 |
| attention q/gate 4096 x 8192 | 128 | 0.178 | default kept | 0.178 | 0.179 | 0 |
| attention q/gate 4096 x 8192 | 256 | 0.312 | 2d grid=11x10 in0=8 sub=1x2 pcM=1 pcN=24 | 0.206 | 0.204 | 0 |
| attention q/gate 4096 x 8192 | 512 | 0.575 | 2d grid=11x10 in0=16 sub=1x4 pcM=2 pcN=24 | 0.228 | 0.227 | 10 |
| attention q/gate 4096 x 8192 | 1024 | 1.795 | minimal_matmul | 0.349 | 0.341 | 34 |
| attention q/gate 4096 x 8192 | 2048 | 3.106 | minimal_matmul | 0.647 | 0.662 | 40 |
| o-proj 4096 x 4096 | 128 | 0.151 | 2d grid=11x10 in0=16 sub=1x2 pcM=1 pcN=12 | 0.122 | 0.12 | 0 |
| o-proj 4096 x 4096 | 256 | 0.220 | 2d grid=11x8 in0=8 sub=1x2 pcM=1 pcN=12 | 0.126 | 0.129 | 0 |
| o-proj 4096 x 4096 | 512 | 0.498 | 2d grid=11x10 in0=16 sub=2x2 pcM=2 pcN=12 | 0.138 | 0.135 | 0 |
| o-proj 4096 x 4096 | 1024 | 0.947 | 2d grid=11x8 in0=16 sub=1x4 pcM=4 pcN=12 | 0.198 | 0.201 | 0 |
| o-proj 4096 x 4096 | 2048 | 1.594 | 2d grid=11x10 in0=16 sub=1x4 pcM=7 pcN=12 | 0.312 | 0.324 | 19 |

22 bfp8 entries adopted (the same count as stage 3); the stage 3 gate / up entries for bfp4 weights are kept under their own `(.., ttnn.bfloat4_b, ttnn.bfloat16)` keys for `KEV_PRECISION=baseline`. The stage 3 bfp4 gate / up configs `2d in0 16 pcN 35` at M 256 and 512 are among the rejected rows with bfp8 weights (the in1 block doubles in bytes), which is the production failure the stage 4 sweep hit.

### Stage 5 server patch (task 7)

`patch -p1 --dry-run < doc/multichip/server_patch.diff`: hunks 1 to 8 apply (offset 11 lines from the stage 4 import), hunks 9 to 11 fail because the patch was written against a one-line-per-statement formatting of `submit`, `card` and `lifespan`; they were applied by hand with the same content (dispatch through `plan` / `collect`, `backlog_ms` and `dispatch` in the card, fan-out fields in the start-up log). `hostrun python -m pytest tests/test_dispatch.py tests/test_server_api.py -q -p no:cacheprovider`: 32 passed in 9.4 s.

### Disk (task 6)

`rm -r /home/hous/dev/kev/tt_cache_all_bfp8 /home/hous/dev/kev/tt_cache_mlp_bf16 /home/hous/dev/kev/tt_cache_all_bf16` after checking that no symlink under `tt_cache` or `tt_cache_mlp_bfp8` points into them (the selected cache lives in `tt_cache/P150`, `tt_cache_mlp_bfp8/P150` holds two symlinks to it, `tt_cache_baseline` is a symlink to `tt_cache`). `df -h /`: 328 G available before, 367 G after; `tt_cache` 27 G (base cache, baseline kev cache, selected kev cache).

### Bucket-128 explanation (task 4)

The "18 us launch floor" text in `README.md` and in item 5 above was replaced with the recomputation from `tracy/traced/ops_perf_results_traced_l4.csv` (kernel sum, FW sum, device span, idle time, top ops of the bucket-256 window; the bucket-128 window has no device rows). See the rewritten "Why bucket 128 stays near 100 ms" subsection in item 5.

### Policy gates at 32 layers (task 1, continued)

`devrun timeout 3000 python -m pytest tests/test_engine.py -k "l32 and (slots or tail_buckets)" -s --device-id 0`, log `/home/hous/dev/kev/logs/stage4r_engine_l32_policy.log`, 6 passed in 228 s (22:46 to 22:50 ET). A first attempt failed at the first warm-up because `mlp.py` passes `program_config=None` explicitly once `QWEN9B_MLP_DOWN_AUTO=1` is set and the pass-through then handed `ttnn.linear` two `program_config` keywords; `policy_linear` now drops the None-valued `program_config` and `bias` keys before forwarding (both are None by the guard that selected the policy).

| test (32 layers, selected precision) | eager | traced | traced + policy |
|---|---|---|---|
| `test_slots_interleaved`, A / B against the full row (min) | 0.99984 / 0.99989 | 0.99989 / 0.99989 | 0.99989 / 0.99989 |
| `test_slots_interleaved`, mode vs eager (A, B) | | 0.999891, 1.000000 | 0.999891, 1.000000 |
| `test_tail_buckets`, mode vs eager (5 buckets, min) | | 0.999999 | 0.999999 |
| steady-state question tail 50 / 200 / 450 / 1000 / 2000 tokens, ms | 128.0 / 193.9 / 484.1 / 671.6 / 1585.5 | 106.7 / 166.7 / 467.8 / 644.8 / 1535.3 | 104.6 / 153.1 / 269.2 / 475.7 / 917.0 |

The policy reproduces the eager numerics to the same PCC as the stage 3 bfp4 run (0.999950 there; the 0.999891 here is the bf16 rounding of the composite 256 + 128 remainder against one masked 512 bucket, identical for traced with and without the policy), and the policy timings equal the stage 3 policy-on timings within 2 ms per bucket.

### Reference rows with the policy on, margin rule (tasks 1 and 2)

`KEV_MATMUL_POLICY=1 devrun timeout 1800 python -m pytest tests/test_engine.py -k reference_records -s --device-id 0`, log `/home/hous/dev/kev/logs/stage4r_reference_records.log`, 1 passed in 42 s: 29 rows, min PCC 0.987337, argmax 28/29, max |dp| 0.087636, mean |dp| 0.026433, flips on rows with an fp32 top-2 margin >= 0.05: none, near-tie flips: `0:choice` (margin 0.0315). These are the stage 4 `mlp_bfp8` numbers (0.987337 / 28 / 0.087636 / 0.026433) to six digits, so the re-swept policy changes nothing in the reference rows. `test_reference_records` now applies the orchestrator's margin rule (`FLIP_MARGIN = 0.05`: a flip counts only when the fp32 reference's top-2 margin is at least 0.05; near-tie flips are logged), asserts `max |dp| <= 0.10` and logs the mean |dp|. With the three conditions met (policy equals eager above 0.999 on every bucket, reference rows reproduce within 1e-3, timings improve), `tt/precision_defaults.py` sets `KEV_MATMUL_POLICY=1` in the `selected` profile.

### Traced engine numbers, selected precision, policy off and on (task 1, before / after)

`scripts/perf_probe.py --traced --trace-region 1073741824 --device-id 0` without and with `--matmul-policy`, 32 layers, 8 slots (KV per slot 2.06 GiB, DRAM free 19.76 GiB after the bfp8 weights), 27 traces, 273 to 274 MiB of trace region, median of 5 with `ttnn.synchronize_device` around each step. JSON `doc/optimized/perf_probe_traced_bfp8.json` (off) and `perf_probe_traced_bfp8_policy.json` (on); logs `/home/hous/dev/kev/logs/stage4r_probe_policy_off.log`, `stage4r_probe_policy_on.log` (22:51 to 22:54 ET). The card probabilities are identical to the printed 16 digits between the two runs.

| path | policy off ms | policy on ms | change | stage 3 (bfp4 gate / up, policy on) ms |
|---|---|---|---|---|
| question tail, bucket 128 (Q 50) | 107.3 | 105.1 | -2.0 % | 104.9 |
| question tail, bucket 256 (Q 200) | 167.2 | 153.5 | -8.2 % | 151.0 |
| question tail, bucket 512 (Q 450) | 468.9 | 270.0 | -42.4 % | 266.4 |
| question tail, bucket 1024 (Q 1000) | 646.7 | 477.5 | -26.2 % | 476.4 |
| question tail, bucket 2048 (Q 2000) | 1538.1 | 919.4 | -40.2 % | 917.3 |
| state 2048 | 1520.1 | 901.3 | -40.7 % | 898.9 |
| state 2392 | 1686.2 | 1053.8 | -37.5 % | 1048.7 |
| card short (6 questions) new / cached | 619.7 / 619.4 | 605.4 / 605.4 | -2.3 % | 605.0 / 604.8 |
| card long (2,192-token state, 5 questions) new / cached | 2163.1 / 536.2 | 1531.1 / 525.8 | -29.2 % / -1.9 % | 1527.5 / 525.6 |

The re-swept bfp8 policy recovers the stage 3 policy-on timings within 0.2 to 5 ms per path; the bfp8 gate / up weights cost 2 to 5 ms per path against bfp4 at the same policy (the minimal_matmul rows: 1.041 vs 1.013 ms per gate / up matmul at 2048).

### max_state 65536 in the production layout (task 3, review P2)

`test_long_state` now builds the engine with the defaults the server uses (`max_state_len` 65536, 8 slots, selected precision, policy on) and asserts 8 slots; every timing is wrapped in `ttnn.synchronize_device`; the 65536 state runs in slot 7 (the last KV range) followed by two questions; DRAM free is read before and after; and the eager 2048-token forward body is run once under `ttnn.graph` capture to measure its transient DRAM peak. `KEV_MATMUL_POLICY=1 devrun timeout 3600 python -m pytest tests/test_engine.py -k long_state -s --device-id 0`, log `/home/hous/dev/kev/logs/stage4r_long_state.log`, 1 passed in 93 s (22:55 to 22:56 ET):

- Build: KV per slot 2.06 GiB, DRAM free 19.76 GiB after the bfp8 weights, 8 of 8 slots, 27 traces, 274.0 MiB of trace region; DRAM free after the build 2.78 GiB.
- S 16384 (8 chunks): state 7.69 s; question tail against the full row in another slot PCC 1.000000 / 1.000000 / 1.000000.
- S 65536 (32 chunks) in slot 7: state 37.51 s; questions 226.9 and 226.6 ms (bucket 128 with 65536 keys of SDPA); output finite; DRAM free 2.78 GiB before and 2.78 GiB after (the replay allocates nothing).
- Peak transient DRAM of the 2048-token forward body (graph capture, running sum of DRAM `buffer_allocate` minus `buffer_deallocate`): 0.545 GiB (585,236,480 B).

Derivation of the reserve: persistent allocations after the slot fit are 19.76 - 8 x 2.06 - 2.78 = 0.50 GiB (GDN snapshots for 8 slots, the bucket buffers, page tables), the transient peak of the largest body is 0.545 GiB, so the engine needs 1.05 GiB above the KV slots. `kv_reserve_bytes` is now `2 << 30` (measured need plus about 90 percent headroom for allocator fragmentation) instead of the undocumented 3 GiB; the slot count at 65536 is 8 with either value ((19.76 - 2) // 2.06 = 8.6, (19.76 - 3) // 2.06 = 8.1, 8 requested). `doc/context_contract.json`: `served_context` 67584 (65536 state plus 2048 question), the stale "8192" note replaced, a `stage4_followup` block with the numbers above.

Context-contract checker: `hostrun python /home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/scripts/check_context_contract.py --model-dir models/autoports/jaredpalmer_kev_9b` exits 2 with "supports context 67584, below HF-advertised 262144, without device-DRAM capacity evidence". That is the correct outcome: the served limit is kev's own serving limit (SERVE_MAX_STATE 65536 plus the 2048-token question bucket), not a DRAM limit, and no DRAM evidence is claimed for it.

### Final one-chip server run (task 8)

Driver `/tmp/claude-1002/-home-hous-dev-kev/0a6793e1-6f42-41ef-926e-8b91dbe0b95b/scratchpad/server_stage4r.sh` (scratch; log `/home/hous/dev/kev/logs/stage4r_server_driver.log`): the stage 2 / 3 environment (`HF_MODEL`, `KEV_RUN`, `TT_CACHE_PATH=/home/hous/dev/kev/tt_cache`, `HF_HUB_OFFLINE=1`, `KEV_MESH_SHAPE=1x1`, `KEV_DEVICE_ID=0`) plus `KEV_FANOUT=0` (stage 5 patch applied, fan-out off on one chip), every other knob at its default, so the engine runs the selected precision with the re-swept policy on (`KevEngine ... max_len=67584 traced=True matmul_policy=True`, `KV per slot 2.06 GiB, DRAM free 19.76 GiB, reserve 2.00 GiB: 8 slot(s) of 8 requested`, 27 traces, 274.0 MiB). `devrun timeout 5400 python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/kev/logs/stage4r_server.log`; `starting:` 22:58:05 ET, `worker 0 ready` 22:58:45 (40 s), warm-up 3 questions 303.4 ms, health up after 44 s. `/v1/models`: `max_state_tokens 65536`, prefix cache 8, `dispatch.fanout false`.

Parity (`scripts/parity_remote.py --passes 2`, `/home/hous/dev/kev/reports/stage4r_parity.json`, log `/home/hous/dev/kev/logs/stage4r_parity.log`, 22:58:47 to 22:58:58): 16 records, 29 questions, against fp32 max |dp| 0.0878, mean |dp| 0.0272, 1 argmax flip; the flip is record 0 `choice` (fp32 0.2895 vs 0.3211, top-2 margin 0.0315, served 0.2783 vs 0.2550), the same near-tie row as `0:choice` in `test_reference_records`, so 0 flips at margin >= 0.05. Stage 3 (bfp4 gate / up): 0.0947 / 0.0358 / 0 flips. All 16 revisits after the other states had run returned the first-pass answers (23 misses, 10 hits, 8 slots).

Bench (`serving_bench_remote.py --reps 20 --quick --concurrency 1,8,32,64`, `/home/hous/dev/kev/reports/bench/p150_stage4r/report.json`, log `/home/hous/dev/kev/logs/stage4r_bench.log`, 22:58 to 23:04): card row `| Stage 4 final (1 chip, bfp8 gate/up, policy on) | 606.4 / 606.4 ms | 1531.6 / 525.5 ms | 1.6 |` (stage 3: 604.7 / 605.2, 1528.5 / 525.5, 1.6); 2 questions short 202.2 / 202.0 ms; 5 questions with a 370-token state 878.3 / 688.3 ms; decision-v7 7.5 req/s at 1 client and 7.2 at 64 clients; the 2,200-token case 1.9 req/s. 789 requests, 0 tracebacks, 0 5xx in the server log. Shutdown 23:04:51: SIGTERM to the uvicorn python process (`pgrep -f "^python -m uvicorn ..."`, not the `timeout` wrapper), `Application shutdown complete`, `Finished server process`, no server process left; the device lock passed directly to the queued GDN repro, so the `LOCK_HELD` line in the driver log is that job, not the server. Added to `doc/server/README.md` and `perf_summary.json` (`stage4_final` entries, `kv_reserve_bytes`).

### Fused GDN chunk op repro (task 5, review P2)

`scripts/gdn_fused_repro.py --device-id 0` (4 layers, eager, pieces 2048 then 256, chunk starts 0 and 2048, carried state): first run with the stage 3 calling convention only (`/home/hous/dev/kev/logs/stage4r_gdn_fused_repro.log`, `doc/optimized/gdn_fused_repro_l4_run1.json`, 22:56 ET): the fused op already disagrees on the first 2048 piece with the same zero initial state (layer 0 NaN, layers 1 and 2 o PCC 0.814 and 0.574), so the carried state is not the cause. Second run with three calling conventions (`stage4r_gdn_fused_repro2.log`, `gdn_fused_repro_l4.json`, 23:05 ET): with the host L2-norm of q / k restored, or with flat q / k / v plus `qkv_head_dims` as `gdn/tp.py` passes them, o PCC >= 0.9987 and state PCC >= 0.9982 on every call including the carried-state 256 piece. Named step: `fused_chunk.py` skips the L2-norm when `flat_qkv_enabled()` is True, which is right for flat inputs (in-kernel norm) and wrong for the 4D inputs the single-device `ttnn_gated_deltanet.py:702` call passes. Hand-off written in `README.md` ("Stage 4 follow-up: fused GDN chunk op, hand-off"); no qwen36 file edited; adoption left to a later stage. Device time about 1 minute in total, no segfault in either run (the stage 3 teardown segfault did not reproduce in these eager processes).

### Dispatcher cost-model source after the update (task 7, follow-up)

`perf_summary.json` `engine_ms` now carries the stage 4 follow-up traced numbers (policy on: tails 105.1 / 153.5 / 270.0 / 477.5 / 919.4 ms, states 901.3 / 1053.8 ms), which `CostModel.from_perf_summary` loads (tail 128 = 105.1 ms, 56.44 ms per 128-token block). Re-running `hostrun python -m pytest tests/test_dispatch.py tests/test_server_api.py -q` after the update gives 30 passed, 2 failed: `test_single_short_request_spreads_over_idle_workers` and `test_mid_state_replicates_only_when_the_share_pays_for_it` compare against the literal stage 3 value 104.9 ms (`tests/test_dispatch.py` lines 92,103,104,151: `209.8` and `5 * 104.9`) instead of `model.tail_ms`; `test_cost_model_from_perf_summary`, which reads the file, passes. The 32 passed reported above were measured before the cost-model update. The two literals were then replaced by `model.tail_cost_ms(...)` in `tests/test_dispatch.py` (in commit `0591d956196`); at HEAD the host suite passes (`hostrun python -m pytest tests/test_dispatch.py tests/test_server_api.py tests/test_loader_cpu.py tests/test_head_cpu.py -q`, 34 passed, re-run during the stage 4 review remediation, log `/home/hous/dev/kev/logs/stage4r2_host_suite.log`).

## Stage 4 review remediation (2026 Oct 02, from `/home/hous/dev/kev/reports/review_stage4.md`)

- `tt/engine.py`: the policy no longer writes `QWEN9B_MLP_DOWN_AUTO=1` into the process environment. When `matmul_policy` is on, the engine sets `prefill_progcfg = None` on its own `KevModelArgs` instance after construction; `qwen36/tt/mlp.py` then passes `program_config=None` for the down projection and the policy applies. A policy-off engine built later in the same process keeps the upstream `prefill_progcfg`, which the review showed was not the case for the eager control in `stage4r_engine_l32_policy.log` (second eager engine built after a policy engine). No other single-device reader of `args.prefill_progcfg` exists (`mlp.py:216-219` only; the `tp.py` readers are tensor-parallel paths the kev engine does not take).
- `qwen36/tt/mlp.py`: the two comments that still described fixed bfp4 / bfp8 dtypes were deleted (dtype comes from `precision.py`).
- `doc/datatype_sweep/selected_precision_config.json`: `performance_traced.regime` now says policy off (numbers from `/home/hous/dev/kev/reports/sweep/perf_probe_mlp_bfp8.json`, `matmul_policy: false`); a `performance_traced_policy_on` block carries the default engine's numbers from `doc/optimized/perf_probe_traced_bfp8_policy.json` (tail 105.08 / 153.53 / 919.36 ms, state 2048 901.28 ms, cards 605.4 / 605.4 and 1531.1 / 525.8 ms); `propagation_check` describes the default engine (policy on, traced, `tt_cache/P150/tensor_cache_bfp8_kev_2b2a70cf_gu-bfp8_dn-bfp8_pj-bfp8`, per-tensor cache file dtypes) from `/home/hous/dev/kev/logs/stage4r_reference_records.log`, and the sweep-process read-back is kept as `propagation_check_sweep_process`. `scripts/dtype_sweep_summary.py` builds both from the files (`traced_block`, `default_engine_propagation`) instead of a literal. The rest of the JSON is unchanged.
- `doc/datatype_sweep/README.md`: policy-default, cache-presence and commit-state sentences brought to the post-follow-up state; the untagged symlink in `tt_cache_mlp_bfp8/P150` is documented as a trap and left on disk. `doc/optimized/README.md` policy bullet updated for the args route. The `test_dispatch.py` sentence above ("Dispatcher cost-model source") corrected: the literals were replaced in `0591d956196` and the host suite passes (34 passed, `/home/hous/dev/kev/logs/stage4r2_host_suite.log`).
- Verification: `devrun timeout 1200 python -m pytest tests/test_engine.py -k "tail_buckets and l4" -q` (eager, traced, policy in one process), log `/home/hous/dev/kev/logs/stage4r2_tail_buckets_l4.log`. Ran 2026 Oct 02 00:52 to 00:53 ET on chip 0 (the first attempt at 00:51 failed before opening the model because the shell lacked `HF_MODEL` / `KEV_RUN`, kept as `stage4r2_tail_buckets_l4_noenv_attempt.log`): 3 passed in 66.5 s; engines built eager, traced, policy in that order, all with the tagged cache; `traced vs eager` min PCC 1.000000 and `policy vs eager` min PCC 1.000000 over the five buckets (per-bucket 1.000000 x4, 1.000001); steady-state policy tails at 4 layers 13.5 / 19.5 / 34.0 / 59.8 / 115.2 ms against eager 18.4 / 25.4 / 61.5 / 84.9 / 199.4 ms.
