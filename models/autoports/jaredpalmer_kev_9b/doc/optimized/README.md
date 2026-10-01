# kev-9b stage 3 (optimize): traced prefill engine on one Blackhole P150

Date: 2026 Oct 01. Chip 0 of the p300c box, `ttnn.open_device(device_id=0, l1_small_size=24576, num_command_queues=2, trace_region_size=1073741824)`. Chronology, commands and every intermediate number: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/optimized/work_log.md`. Compact numbers: `perf_summary.json` in this directory.

Acronyms: GDN (Gated DeltaNet), KV (key/value), SDPA (scaled dot-product attention), PCC (Pearson correlation coefficient), MLP (multi-layer perceptron), bfp4 / bfp8 (block floating point, 4 or 8 bits per element).

## What changed

`/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/engine.py` (`KevEngine`):

- Per-slot KV cache (stage 1 and 2 review P1): every snapshot slot owns a disjoint range of paged KV blocks and its own page table; `prefill_state`, `question_hidden` and `prefill_hidden` take the slot. The slot count is `min(requested, (free DRAM - 3 GiB) // KV per slot)`, 8 at the default `max_state_len` 65536 (2.06 GiB per slot).
- `KevModelArgs` is imported directly (review P2); the engine logs the args class, adapter sha8 and weight-cache path at build.
- Traced path (`traced=True`, default): one trace per bucket for the forward (`fwd[b]`, the `_forward_prefill_chunk` body over persistent buffers with the runtime `chunk_start_idx`), one per bucket for the readout (`gather[b]`: one-hot matmul of up to 32 rows, final RMSNorm), one `restore` and one `save` per slot and one `zero` trace for the GDN state. The 2048 trace is the state chunk; aligned state remainders run as exact smaller buckets (384 = 256 + 128); question tails run padded in their bucket. All buffers are allocated and all programs are warmed before the first capture; program-cache misses are forbidden after setup. `traced=False` keeps the stage 2 eager path for comparison.
- GDN sequence kernel L1 threshold lowered to 256 for the traced bodies (bucket 512 clashed its static circular buffers with L1 inputs; only previously failing lengths are affected).
- Matmul policy (`matmul_policy=True`, default; `KEV_MATMUL_POLICY=0` disables): the engine rebinds `ttnn.linear` to `policy_linear`, which runs the measured-best kernel for the 22 prefill matmul keys `(M, K, N, in1 dtype, in0 dtype)` in `MATMUL_POLICY` (a tuned 2D `MatmulMultiCoreReuseMultiCastProgramConfig` or `ttnn.experimental.minimal_matmul`, LoFi with fp32 accumulation as before), forwards every caller keyword, applies only when the caller gave no `program_config` or `bias`, and passes every other call through unchanged (stage 4 follow-up; the stage 3 version keyed on shape only and dropped caller keywords). When the policy is on the engine sets `QWEN9B_MLP_DOWN_AUTO=1` so the MLP down projection reaches it without a caller `program_config`. No file under `models/demos/blackhole/qwen36` or `models/experimental` was edited by stage 3.

`tt/server.py`: `KEV_MAX_STATE` default 65536 (kev's limit), `KEV_TRACED`, slot returned to the free list when `prefill_state` fails, `FakeEngine` keeps per-slot state and raises on a stale slot; `tests/test_server_api.py` adds the eviction / revisit and failed-prefill tests (14 pass on the host). `tests/test_engine.py` adds `test_slots_interleaved`, `test_tail_buckets` (eager, traced and policy modes compared) and `test_long_state`, and asserts the args class. New scripts: `scripts/perf_probe.py`, `scripts/matmul_sweep.py`, `scripts/parity_remote.py`.

## Numbers (32 layers, merged weights, chip 0)

Engine-level, `scripts/perf_probe.py` (median of 5, `ttnn.synchronize_device` around each step):

| path | stage 2 eager ms | traced ms | traced + matmul policy ms | stage 4 follow-up: bfp8 gate / up, policy off ms | bfp8 gate / up, re-swept policy on ms |
|---|---|---|---|---|---|
| question tail, bucket 128 | 129.5 | 107.0 | 104.9 | 107.3 | 105.1 |
| question tail, bucket 256 | 195.1 | 166.9 | 151.0 | 167.2 | 153.5 |
| question tail, bucket 512 | 483.7 | 468.1 | 266.4 | 468.9 | 270.0 |
| question tail, bucket 1024 | 670.8 | 645.6 | 476.4 | 646.7 | 477.5 |
| question tail, bucket 2048 | 1576.8 | 1528.7 | 917.3 | 1538.1 | 919.4 |
| state 2048 tokens | 1507.7 | 1510.7 | 898.9 | 1520.1 | 901.3 |
| state 2392 tokens | 1692.0 | 1676.6 | 1048.7 | 1686.2 | 1053.8 |
| card short (6 questions) new / cached | 748.7 / 746.5 | 619.1 / 618.1 | 605.0 / 604.8 | 619.7 / 619.4 | 605.4 / 605.4 |
| card long (2,192-token state, 5 questions) new / cached | 2269.9 / 650.0 | 2153.0 / 535.9 | 1527.5 / 525.6 | 2163.1 / 536.2 | 1531.1 / 525.8 |

The first three columns are stage 3 with bfp4 gate / up weights; the last two are the stage 4 follow-up with the selected precision `mlp_bfp8` (`perf_probe_traced_bfp8.json`, `perf_probe_traced_bfp8_policy.json`), where `MATMUL_POLICY` is keyed on `(M, K, N, in1 dtype, in0 dtype)` and was re-swept with bfp8 gate / up weights (`matmul_sweep_bfp8.json`). The policy is on by default (`tt/precision_defaults.py`).

Device versus host: in the traced engine a request is device time to within 1 % (per question: 0.3 ms restore, 0.5 ms readout, the rest is the forward replay). The 2048-token chunk is 100 % device kernel time (Tracy window sums equal the wall clock). The small buckets are dominated by many short kernels with low utilization, not by launch gaps: in the bucket-256 window (`PERF_TAIL_Q200`, 4 layers, `tracy/traced/ops_perf_results_traced_l4.csv`) 745 device ops have a kernel sum of 20.17 ms inside a device span of 21.42 ms (first firmware start to last firmware end), so the device is busy 97 % of the time and idle for 0.74 ms; the median kernel is 6.3 us and 537 of the 745 kernels are under 20 us. The firmware-duration sum (33.71 ms) exceeds the span by 57 %, so firmware durations overlap and their excess over the kernel sum is not a serial launch cost. The bucket-128 window has no device rows (`PERF_TAIL_Q50_perf_report.console.log`), so its 105 ms at 32 layers is not decomposed; the op count is the same at every bucket (736 to 745 per 4 layers, about 5900 per 32-layer forward). See `work_log.md`, item 5.

Server-level (card format, `scripts/serving_bench_remote.py`, latency columns full `--reps 20`, throughput `--quick`): see the "Stage 3" row in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/server/README.md` and `perf_summary.json`.

## Gates

- Traced equals eager: `test_tail_buckets` PCC 1.000000 (4 layers) and 0.999999 / 1.000000 (32 layers) on every bucket, three back-to-back questions bit-identical; `test_slots_interleaved` A / B against the full row 0.9999+, traced vs eager 0.999893 (4 layers) / 0.999950 (32 layers) on the state with a composite remainder, 1.000000 on the other.
- Stage 1 tests on the traced engine (32 layers): `hidden_vs_hf` pass at T 300 / 1500 / 2300, `tail_matches_full_row` 6/6 with the stage 1 PCCs, `test_reference_records` min PCC 0.976770, 29/29 argmax, max |dp| 0.102370, mean |dp| 0.036647 (stage 2 parity through HTTP: see the stage 3 parity file).
- Trace allocation tracker (`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`), watcher (`TT_METAL_WATCHER=10`) and the 16384 / 65536-token state runs: results in `work_log.md`, item 6.

## Optimize-skill item mapping (prefill-only model)

| skill item | mapping here | status |
|---|---|---|
| traced decode, no host fallback | traced question tail and state chunk; eager path kept only as `traced=False` | done, 27 traces, 274 MiB |
| prefill / TTFT | server `latency_ms` of a request; chunk 2048, buckets 128 to 2048 | measured before and after |
| operation-topology audit | Tracy ops CSV + tt-perf-report per bucket window | done; matmuls 52 %, GDN relayout 16 %, GDN elementwise 12 % of the 2048 chunk |
| 2D program configs for large prefill matmuls | `MATMUL_POLICY` sweep, 22 shapes | done, 1.5 to 6.7x per matmul |
| decode DRAM-sharded matmuls, LM head, sampling, token feedback, CCLs, multi-device | no decode, no LM head, one chip | not applicable |
| precision / fidelity trials | not changed (LoFi + fp32 accumulate kept); dropping fp32 accumulate is 1.3 to 3.7x faster but changes numerics | deferred to stage 4 with the measured rows in `matmul_sweep*.json` |
| batch capability | `snapshot_slots` and the server prefix cache carry several states; one request per worker at a time | preserved |
| watcher clean, allocation tracker clean | separate runs | see work_log item 6 |

## Open issues

- Bucket 128 and 256 tails are dominated by many short kernels with low utilization (bucket 256: device busy 97 % of the span, median kernel 6.3 us, matmuls 36 % of kernel time at 59.6 us mean, GDN head reshapes 18 %, GDN fp32 elementwise 14.5 %); the bucket-128 window itself was not captured. The remedy is fewer or larger ops per GDN layer inside qwen36 (fused relayout or the fused chunk op; see the stage 4 follow-up below for the fused op).
- The fused GDN chunk op (`ttnn.transformer.chunk_gated_delta_rule`) was 25 % faster at bucket 128 but gave wrong hidden states in stage 3; the stage 4 follow-up repro (below) names the cause: the single-device call passes 4D q/k and `fused_chunk.py` then skips the L2-norm. Not shipped; adoption is a later stage after a one-line fix in qwen36.
- The matmul policy is a process-level rebind of `ttnn.linear`; a per-call-site hook in qwen36 would be the proper home.
- The GDN output projection takes an fp32 activation (1.63 ms at 2048) and is not in the policy; the sweep did not cover fp32 inputs.

## Stage 4 follow-up: fused GDN chunk op, hand-off (review P2)

Bounded repro, `scripts/gdn_fused_repro.py` (4-layer eager engine, selected precision; the GDN layers call the seq adapter as usual, and a wrapper also runs `gdn/fused_chunk.py:chunk_gated_delta_rule_fused_adapter` on the same inputs and the same carried state, in three calling conventions, and compares `o` and the new recurrent state against the seq result per call). Pieces 2048 then 256 at chunk starts 0 and 2048, so the second piece carries the state of the first. Result `doc/optimized/gdn_fused_repro_l4.json`, log `/home/hous/dev/kev/logs/stage4r_gdn_fused_repro2.log` (a first run without the variants is `gdn_fused_repro_l4_run1.json`, `/home/hous/dev/kev/logs/stage4r_gdn_fused_repro.log`); 2 x 30 s of device time, no segfault in either run.

| calling convention | piece 2048, o PCC vs seq (layers 0 / 1 / 2) | state PCC | piece 256 with carried state, o PCC | state PCC |
|---|---|---|---|---|
| `4d`: q / k / v as `[1, T, 32, 128]`, as `ttnn_gated_deltanet.py:702` passes them (what stage 3 did) | NaN / 0.814 / 0.574 (per position: 0.83 at t 0, 0.0 after, layer 0) | NaN / 0.578 / 0.705 | NaN / 0.851 / 0.706 | NaN / 0.871 / 0.859 |
| `4d_l2norm`: the same with `l2_norm_ttnn` on q and k on the host before the op | 1.0020 / 1.0014 / 1.0005 (every sampled position >= 0.9983) | 0.9990 / 0.9986 / 0.9982 | 1.0005 / 0.9998 / 0.9987 | 0.9993 / 0.9990 / 0.9986 |
| `flat`: q / k / v as `[1, T, 32 x 128]` with `qkv_head_dims`, as `gdn/tp.py:590` passes them | 1.0020 / 1.0014 / 1.0005 | 0.9990 / 0.9987 / 0.9983 | 1.0005 / 0.9999 / 0.9987 | 0.9993 / 0.9990 / 0.9986 |

Named failing step: `chunk_gated_delta_rule_fused_adapter` skips the L2-norm of q and k whenever `flat_qkv_enabled()` is True (`fused_chunk.py`, "Host L2-norm q/k; with flat QKV the prep kernel normalizes in-kernel instead"), but the in-kernel norm belongs to the flat input path (`qkv_head_dims` given); with 4D inputs the op receives un-normalized q / k. That is the stage 3 failure (first positions near 1.0, later positions near 0, NaN in layer 0 where the activations are largest), and the carried recurrent state is not involved: with the norm restored, the fused op agrees with the seq adapter on the 2048 piece and on the 256 piece that carries its state. The fix is one condition in `models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py` (normalize on the host when `qkv_head_dims is None`, or pass flat inputs from the single-device path), which this stage does not own. Per-call cost at 4 layers: seq 12.3 ms, fused 4.0 ms (2048 tokens, warm); seq 3.8 ms, fused 1.1 ms (256 tokens). The remaining 0.001 to 0.002 of PCC against the seq adapter (chunk 32 inside the op versus the fp32 chunk-128 seq kernel) needs the model-level gates (`test_tail_buckets`, `test_slots_interleaved`, reference rows) before adoption; left to a later stage.
