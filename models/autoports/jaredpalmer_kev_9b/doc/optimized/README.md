# kev-9b stage 3 (optimize): traced prefill engine on one Blackhole P150

Date: 2026 Oct 01. Chip 0 of the p300c box, `ttnn.open_device(device_id=0, l1_small_size=24576, num_command_queues=2, trace_region_size=1073741824)`. Chronology, commands and every intermediate number: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/optimized/work_log.md`. Compact numbers: `perf_summary.json` in this directory.

Acronyms: GDN (Gated DeltaNet), KV (key/value), SDPA (scaled dot-product attention), PCC (Pearson correlation coefficient), MLP (multi-layer perceptron), bfp4 / bfp8 (block floating point, 4 or 8 bits per element).

## What changed

`/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/engine.py` (`KevEngine`):

- Per-slot KV cache (stage 1 and 2 review P1): every snapshot slot owns a disjoint range of paged KV blocks and its own page table; `prefill_state`, `question_hidden` and `prefill_hidden` take the slot. The slot count is `min(requested, (free DRAM - 3 GiB) // KV per slot)`, 8 at the default `max_state_len` 65536 (2.06 GiB per slot).
- `KevModelArgs` is imported directly (review P2); the engine logs the args class, adapter sha8 and weight-cache path at build.
- Traced path (`traced=True`, default): one trace per bucket for the forward (`fwd[b]`, the `_forward_prefill_chunk` body over persistent buffers with the runtime `chunk_start_idx`), one per bucket for the readout (`gather[b]`: one-hot matmul of up to 32 rows, final RMSNorm), one `restore` and one `save` per slot and one `zero` trace for the GDN state. The 2048 trace is the state chunk; aligned state remainders run as exact smaller buckets (384 = 256 + 128); question tails run padded in their bucket. All buffers are allocated and all programs are warmed before the first capture; program-cache misses are forbidden after setup. `traced=False` keeps the stage 2 eager path for comparison.
- GDN sequence kernel L1 threshold lowered to 256 for the traced bodies (bucket 512 clashed its static circular buffers with L1 inputs; only previously failing lengths are affected).
- Matmul policy (`matmul_policy=True`, default; `KEV_MATMUL_POLICY=0` disables): the engine rebinds `ttnn.linear` to `policy_linear`, which runs the measured-best kernel for the 22 prefill matmul shapes in `MATMUL_POLICY` (a tuned 2D `MatmulMultiCoreReuseMultiCastProgramConfig` or `ttnn.experimental.minimal_matmul`, LoFi with fp32 accumulation as before) and passes every other call through unchanged. No file under `models/demos/blackhole/qwen36` or `models/experimental` was edited by stage 3.

`tt/server.py`: `KEV_MAX_STATE` default 65536 (kev's limit), `KEV_TRACED`, slot returned to the free list when `prefill_state` fails, `FakeEngine` keeps per-slot state and raises on a stale slot; `tests/test_server_api.py` adds the eviction / revisit and failed-prefill tests (14 pass on the host). `tests/test_engine.py` adds `test_slots_interleaved`, `test_tail_buckets` (eager, traced and policy modes compared) and `test_long_state`, and asserts the args class. New scripts: `scripts/perf_probe.py`, `scripts/matmul_sweep.py`, `scripts/parity_remote.py`.

## Numbers (32 layers, merged weights, chip 0)

Engine-level, `scripts/perf_probe.py` (median of 5, `ttnn.synchronize_device` around each step):

| path | stage 2 eager ms | traced ms | traced + matmul policy ms |
|---|---|---|---|
| question tail, bucket 128 | 129.5 | 107.0 | 104.9 |
| question tail, bucket 256 | 195.1 | 166.9 | 151.0 |
| question tail, bucket 512 | 483.7 | 468.1 | 266.4 |
| question tail, bucket 1024 | 670.8 | 645.6 | 476.4 |
| question tail, bucket 2048 | 1576.8 | 1528.7 | 917.3 |
| state 2048 tokens | 1507.7 | 1510.7 | 898.9 |
| state 2392 tokens | 1692.0 | 1676.6 | 1048.7 |
| card short (6 questions) new / cached | 748.7 / 746.5 | 619.1 / 618.1 | 605.0 / 604.8 |
| card long (2,192-token state, 5 questions) new / cached | 2269.9 / 650.0 | 2153.0 / 535.9 | 1527.5 / 525.6 |

Device versus host: in the traced engine a request is device time to within 1 % (per question: 0.3 ms restore, 0.5 ms readout, the rest is the forward replay). The 2048-token chunk is 100 % device kernel time (Tracy window sums equal the wall clock). Bucket 128 is op-count bound: about 5900 device ops per 32-layer forward at a per-program launch floor near 18 us, which is why tracing and the matmul policy move it only from 129 to 105 ms. See `work_log.md`, item 5.

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

- Bucket 128 and 256 tails are launch-overhead bound (about 5900 ops per forward); the remedy is fewer ops per GDN layer inside qwen36 (fused relayout or the fused chunk op, which failed its carried-state check here).
- The fused GDN chunk op (`ttnn.transformer.chunk_gated_delta_rule`) was 25 % faster at bucket 128 but gave wrong hidden states after a multi-segment state and crashed at teardown; recorded, not shipped.
- The matmul policy is a process-level rebind of `ttnn.linear`; a per-call-site hook in qwen36 would be the proper home.
- The GDN output projection takes an fp32 activation (1.63 ms at 2048) and is not in the policy; the sweep did not cover fp32 inputs.
