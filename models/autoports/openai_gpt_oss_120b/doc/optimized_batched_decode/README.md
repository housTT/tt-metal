# Batched decode and indexed prefill for GPT-OSS 120B on P150x4

Date: 2026-09-10. Scope: `tt/multichip_decoder.py`, `tt/model.py`,
`tt/generator_vllm.py`. Hardware: 4x Blackhole (2x p300c), mesh (1, 4), TP=4.

## Problem

The 2026-09-10 latency sweep showed decode TPOT of 15.4 ms at one user and
506 ms at two users (2 tok/s/u, 4 tok/s aggregate). Two causes:

1. The MoE block ran a static per-user loop of the batch-1 expert graph for
   any decode batch above one, and vLLM padded every batch above one to 32
   because the model advertised only decode buckets `(1, 32)`. That is
   32 x 36 = 1152 sequential MoE graphs per token.
2. Prefill computed every 32-token group against the union of its experts
   with an expanded dense output over all 128 experts (zero-filled), so the
   expert matmuls showed `active=512/512` and the elementwise tail ran over
   `tokens x 128 experts`. Prefill ran at about 460 tok/s at every length.

## Changes

### Decode (`_ActiveExpertTPMLP`)

- `_run_indexed_slots_decode` for `1 < B <= 8`: gathers the `top_k * B`
  routed expert slots by index (duplicates allowed, static trace shape),
  applies each slot's routing score to its user's row through a one-hot
  selector before the linear down projection, reduces over slots once.
- `_run_grouped_decode` for `B > 8`: the padded 32-row decode tile is one
  token group; a union expert mask (`ttnn.sum` of dense routing weights over
  rows) drives both sparse matmuls, routing weights are applied before the
  down projection, one reduction over experts.
- Shared: two-op fused SwiGLU (`ttnn.mul` with input activation chains for
  clamp, alpha scale, sigmoid, and `+1`), down bias folded through the dense
  routing weights with one `[rows, 128] x [128, 2880]` matmul, dense routing
  built from the same fused router as batch 1 plus `ttnn.scatter`, per-width
  fused-router bias tensors.
- Weight layout: each rank's expert intermediate slice is padded 720 -> 768
  (cache suffix `_pad768`) so the packed gate/up split is a tile-aligned slice
  and the down contraction is 24 tiles. 46 tiles (736 padding) is rejected by
  the sparse matmul: every core of the rectangular grid must have work, so
  the gate/up grid is (6, 8) = 48 cores at one tile each.
- Decode trace buckets `(1, 4, 8, 16, 32)` (`tt/model.py::decode_trace_buckets`,
  advertised per instance by the vLLM adapter and warmed/captured in order).
- Decode norms (`_DecodeShardedRMSNorm`) run on the ten-way L1 width shard at
  every batch size since 2026-09-16, including the full 32-row tile that
  bring-up had left interleaved after a nondeterministic trace. The batch-32
  layer test replays the traced decode 100 times per layer type with
  bit-identical output (`GPT_OSS_120B_DECODE_TRACE_DETERMINISM=1`), and the
  sharded norms save 66-70 us per layer (sliding 1.328 -> 1.262 ms, full
  attention 1.454 -> 1.384 ms), about 2.5 ms per 32-user step.
  `GPT_OSS_120B_DECODE_NORM_SHARD_FULL_TILE=0` restores the interleaved norms.
  The DRAM-sharded QKV policy (`GPT_OSS_120B_MULTICHIP_CANDIDATE=dram_sharded_qkv`)
  measured 1.299 / 1.430 ms alone and 1.266 / 1.375 ms together with the
  sharded norms, so it is not adopted.

### Prefill

Shipping path (`prefill_indexed_experts=True`, prompts of 512+ tokens):
`_run_indexed_prefill` gathers each expert's routed tokens into per-expert
slabs and runs compact indexed sparse matmuls, replacing the packed
32-token-group path with its dense 128-expert expanded outputs. Design points
that were needed to make it work in serving, not just in a layer test:

- An expert's tokens fill `count // 512` full slabs of 512 rows plus one
  remainder slab whose height is a power-of-two tile multiple (32..256), so
  padding is at most one short slab per expert. Taller slabs overflow L1 in
  the sparse matmul (per_core_M x in0_block_w input tiles); 4096-row slabs
  crashed 16k-token prompts.
- Slabs of equal height form a group; a group is split into power-of-two
  chunks (largest first, at most 128 ids, the indexed mode's limit) with no
  padding slabs, and each chunk is one indexed sparse matmul pair. With
  routing-dependent shapes every prompt recompiled programs (32 s TTFT);
  with these shape classes the set is 5 heights x 8 sizes and stays cached.
  The slab buffer is rounded to 1/8 of its power-of-two octave (<= 12.5%
  waste; the earlier full power-of-two rounding plus padding slabs made the
  16k-token dispatch gather 2.5x larger than the slots).
- `warmup_indexed_prefill_shapes` compiles that set once on layer 0 during
  the adapter's prefill warmup (program caches key on shapes, not weights),
  including the per-group token gather.
- Prefill is untraced and host-synchronised, so the routing runs in torch:
  the router linear runs on device and its bf16 logits are read back (4 MB
  at 16k tokens, the same bytes as the padded index/score tiles); top-4,
  softmax, per-expert counts, slab layout, the stable argsort of slots and
  the gather-back columns are torch and are uploaded as four small tensors.
  A device `ttnn.sort` of the slots alone cost 5-18 ms per layer, and the
  device `ttnn.topk` path was no faster than the readback.
- Token rows are gathered straight into each group's slab stack with one
  tile-layout `ttnn.embedding` per group (no capacity-sized copy). Expert
  rows come back with one `[top_k, rows]` embedding gather, are multiplied
  by the `[1, top_k, rows, 1]` routing weights and reduced over `top_k`
  (`fast_reduce_nc`); the down bias is folded in through the dense routing
  weights. Below 512 tokens the packed path is used, since the per-layer
  sync costs more than it saves there.
- Every device op in the path sees prompt-independent shapes and attributes.
  The first serving build sliced each group's dispatch rows and expert ids
  out of one uploaded vector with `ttnn.slice` at routing-dependent offsets;
  each distinct offset is a separate program-cache entry holding a DRAM
  kernel buffer, so the cache grew by hundreds of entries per prompt for the
  life of the process and, after a few hundred prompts, every later prefill
  ran 30-40x slower (the bench-sweeps harness measured 4k TTFT 38-50 s and
  16k 105 s after its 16-user row; decode, being traced, was unaffected).
  Per-group index vectors are now uploaded from the host layout instead.
  `_stage_mark` also accounts program-cache growth per stage
  (`_prefill_program_growth`) and `test_indexed_prefill_moe` prints it for
  three new prompts; steady-state growth is bounded by the static shape set.
  Two programs per layer per prompt still leaked after that change: the
  `ttnn.concat` of the per-group expert outputs (keyed on the per-prompt list
  of group shapes) and the untilize of the capacity-sized gather-back table.
  Since 2026-09-16 each group's tiled down-projection output is written into
  one tile-layout arena with `paged_fill_cache` (the page table is the slab's
  32-row block ids, so the destination offsets are data, not program
  attributes) and a single untilize produces the gather-back table; the arena
  is sized to a half-octave class of the slab capacity, so the fill and
  untilize programs are keyed on a bounded set as well. Layer outputs are
  identical, timing is unchanged (4k 19.9 ms, 16k 68 ms per layer), and the
  layer test shows no growth in those stages for new prompts.
  `GPT_OSS_120B_LOG_PROGRAM_CACHE=1` makes the vLLM adapter log the program
  cache size after every prefill call as serving-side evidence.
- Prompts longer than `GPT_OSS_120B_INDEXED_PREFILL_CHUNK` tokens (default
  65,536) run the indexed MoE prefill in equal row chunks. The 131,072-token
  bucket (which every prompt above 65,536 tokens pads to) needs a 4.5 GB
  expert-output arena plus a gather-back table of the same size, and the
  serving process has about 5.5 GB of DRAM per chip free beside the weights
  and KV cache, so a 98k prompt killed the engine with a DRAM OOM on
  2026-09-16. Two 64k chunks reuse the footprint the 64k sweep row already
  proved; the layer test shows PCC 0.99975 against the packed path at 131,072
  tokens (533 ms per layer for the MoE block, packed: 7.3 s) and a 15 ms
  chunking cost at 16k when forced into 8k chunks (not used by default).
- Matmul blocking is specific to this path (`_indexed_prefill_matmul_config`):
  gate/up `(in0_block_w, out_block_h, out_subblock_h, out_subblock_w)` =
  (15, 4, 2, 1), down (12, 4, 2, 2). With `out_block_h = 1` (the decode
  configs) the kernel re-reads every weight block once per slab tile row:
  16k-token gate/up went from 27.7 ms to 15.5 ms per layer and down from
  20.8 ms to 6.8 ms with (30, 4, 2, 1) / (24, 8, 4, 2). Those configs use
  up to ~960 KB of circular buffers and clashed with the ~600 KB of resident
  L1 buffers the serving process holds per core (the layer test has no such
  pressure), so the shipped configs keep the CB set near 300 KB at a cost of
  ~2 ms per layer at 16k (16.8 / 7.9 ms). Both matmuls are bound by the
  single in0 multicast sender of the 1D kernel (bf16 slabs: 425 MB per chip
  at 16k tokens); a bfp8 slab saved 3 ms on the matmul but cost 2 ms in the
  cast and lowered the PCC, so it is not used (`indexed_prefill_slab_dtype`).

### Batched prefill (2026-09-16)

New requests that vLLM schedules in the same step and that share a padded
prompt length are prefilled as one forward pass. The shared tt-transformers
generator already had this path (Galaxy 70B); the autoport had opted out.
What was needed:

- **Compact device rows** (opt-in `batched_prefill_compact_rows` on the model
  args, honoured by `Generator._prefill_forward_text_impl`): requests occupy
  device rows `0..B-1` instead of their physical sampler slots, so the device
  batch is the request count rounded up to a supported size (2, 4, 8, 16, 32)
  rather than the highest slot in use. Two new requests at slots 30 and 31 no
  longer cost a 32-row pass. Per-layer page tables (vLLM hybrid KV groups)
  arrive in request order and are padded to the device batch with rows of -1,
  which `paged_fill_cache` skips.
- **Layer-stack entry** (`Model._forward_layers_and_head`): the generator hands
  `[1, 1, B*S, H]` with RoPE for `B*S` positions; the decoder stack takes
  `[1, B, S, H]` with one user's RoPE and concatenates internally (attention
  fills the KV cache per user and runs causal SDPA per user; the MoE sees one
  longer row set). The pass returns un-normed hidden states.
- **Terminal on the gathered rows**: `extract_last_tokens_batched_prefill`
  gathers each user's last hidden row on device (one untilize, one embedding
  gather keyed only on `B*S`) into the sampler's `[1, 1, 32, H]` slot tile;
  `_apply_norm_and_lm_head` runs the final norm and the DRAM-sharded LM head on
  it. The batched shape is part of the prefill-variant key, so first-time
  compilation still releases the decode traces.
- **Cap and warmup**: batching applies to prompts up to 2,048 padded tokens
  (`GPT_OSS_120B_BATCHED_PREFILL_MAX_LEN`); longer prompts are compute bound one
  at a time, and each new (batch, length) pair compiles its own programs (19 s
  for the first 8 x 1k pass). The common pairs (2..32 users x 128 and 1,024
  tokens, `GPT_OSS_120B_BATCHED_PREFILL_WARMUP`) are compiled at server start
  (30 s). `GPT_OSS_120B_BATCHED_PREFILL=0` disables batching.

Correctness: with 8 prompts of 128 tokens and 4 of 1,024 tokens, the first
sampled token of every prompt is identical to a sequential prefill; later
greedy tokens diverge for 7 of 8 random-token prompts, exactly as they do with
sequential prefill followed by concurrent decode (batch-8 vs batch-1 decode
programs), so that is decode numerics, not batching.

Serving (local server, `bench_client.py`, N = 4 x users, zero failures; sweep
v20 of 2026-09-11 in parentheses):

| ISL | OSL | users | TTFT | TPOT | decode tok/s/u | aggregate tok/s | E2EL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 8 | 0.62 s (1.89) | 32.6 ms | 30.7 | 215 (170) | 4.8 s (6.0) |
| 128 | 128 | 32 | 1.02 s (7.49) | 53.6 ms | 18.7 | 524 (286) | 7.8 s (14.3) |
| 1,024 | 256 | 8 | 1.65 s (2.60) | 32.9 ms | 30.4 | 204 (186) | 10.0 s (11.0) |
| 1,024 | 256 | 16 | 2.73 s (5.13) | 40.9 ms (63.0) | 24.4 | 311 (193) | 13.2 s (21.2) |
| 1,024 | 256 | 32 | 5.07 s (9.90) | 55.3 ms | 18.1 | 427 (343) | 19.2 s (23.9) |
| 4,096 | 256 | 8 | 6.0 s (6.0) | 33.4 ms | 29.9 | 141 (141) | 14.5 s (14.5) |
| 4,096 | 256 | 16 | 11.5 s (11.8) | 42.1 ms (64.1) | 23.8 | 184 (145) | 22.3 s (28.2) |

The 16-user rows also carry the new 16-user decode bucket (TPOT 63 -> 41 ms);
4k prompts are above the batching cap and unchanged.

Packed path improvements that remain in use below 512 tokens: 768 layout
(tile-aligned gate/up slices), pre-transposed gate/up bias, down bias folded
through the dense routing weights.

## Measurements

Layer 0 (sliding attention), TP4, traced decode, median of 5 x 20 replays
(`tests/test_multichip_batched_decode_perf.py`):

| batch | before (per-user loop) | after | per user |
|---:|---:|---:|---:|
| 1 | 0.384 ms | 0.383 ms | 0.383 ms |
| 2 | 0.961 ms | 0.411 ms | 0.205 ms |
| 8 | 3.551 ms | 0.863 ms | 0.108 ms |
| 32 | 14.030 ms | 1.326 ms | 0.041 ms |

Every batched row matches the batch-1 indexed path at PCC >= 0.99993.
Batch-1 acceptance against the single-chip baseline artifacts
(`test_real_weight_multichip_against_baseline_artifact`, TP4): prefill PCC
0.993 / 0.990, decode PCC 0.9992 / 0.9978 (thresholds 0.95).

Full layer-0 prefill on real token embeddings, one call with a host sync
(packed group-sparse vs experimental indexed), `test_indexed_prefill_moe`:

| tokens | packed | indexed | speedup | PCC |
|---:|---:|---:|---:|---:|
| 128 | 7.9 ms | 4.5 ms | 1.8x | 0.99984 |
| 1,024 | 53.8 ms | 8.0 ms | 6.7x | 0.99997 |
| 4,096 | 228.8 ms | 20.0 ms | 11.4x | 0.99989 |
| 16,384 | 912.5 ms | 63.6 ms | 14.3x | 0.99989 |

Per-stage times at 16,384 tokens (`GPT_OSS_120B_PREFILL_STAGES=1`, device
sync after every op, so the sum is above the pipelined 63.6 ms): gate/up
matmuls 16.8 ms, down matmuls 7.9 ms, gather-back + weights + reduce 7.3 ms,
token gathers 3.8 ms, all-reduce 3.8 ms, SwiGLU 3.2 ms, logits readback
3.0 ms, uploads 2.8 ms, down untilize 2.8 ms, gate/up slices 2.6 ms, gate/up
bias 4.3 ms, host layout 1.5 ms. The earlier version of this path measured
14.9 / 33 / 174 ms at 1k / 4k / 16k.

The packed numbers already include the 768 layout and bias fold (the 127-token
layer prefill in the acceptance test went from 14.4 ms to 9.2 ms).

Serving, `python -m vllm.entrypoints.openai.api_server` from this tree,
ISL 128 / OSL 128, `bench_client.py`, 4 x users prompts (max 64), zero
failures. "Before" is the 2026-09-10 sweep of the shipped container. v8 =
768 layout, batched decode, buckets (1, 4, 8, 32), packed prefill:

| users | TPOT before | TPOT v8 | decode tok/s/u v8 | decode-only aggregate v8 | TTFT before | TTFT v8 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.4 ms | 15.3 ms | 65 | 65 | 0.28 s | 0.25 s |
| 2 | 506 ms | 21.9 ms | 46 | 91 | 0.60 s | 0.48 s |
| 4 | n/a | 21.9 ms | 46 | 183 | n/a | 0.95 s |
| 8 | ~590 ms | 32.6 ms | 31 | 245 | n/a | 1.9 s |
| 16 | ~590 ms | 61.7 ms | 16 | 259 | n/a | 3.7 s |
| 32 | ~590 ms | 53.2 ms | 18.8 | 602 | n/a | 7.5 s |

TTFT at high concurrency is the serialized queue of 128-token prefills (about
235 ms each); the plugin does not mix prefill and decode steps.

Long prompts, one user, OSL 64-128 ("before" is the 2026-09-10 sweep of the
shipped container; v10 = packed path with the 768 layout; v13 = indexed
prefill, warm shapes; cold = first prompt of that length after server start):

| ISL | TTFT before | TTFT v10 | TTFT v13 warm | TTFT v13 cold | TTFT v15 warm | TTFT v15 cold | TPOT v15 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 2.15 s | 1.73 s | 0.79 s | 4.5 s | 0.62 s | 7.5 s | 15.4 ms |
| 4,096 | 8.86 s | 7.19 s | 2.47 s | 5.5 s | 1.20 s | 3.8 s | 15.6 ms |
| 16,384 | 35.6 s | 28.9 s | 10.8 s | 16.5 s | 2.96 s | 7.3 s | 15.9 ms |
| 32,768 | 71.7 s | 58.3 s | 23.0 s | 31 s | 5.95 s | 9.1 s | 16.4 ms |

v15 = indexed prefill v5 (this document's design section); its 1k cold number
was the first request the server served. Decode at ISL 128 is unchanged in
v15 (1 / 4 / 8 / 32 users: 15.3 / 21.9 / 32.6 / 53.1 ms TPOT, zero failures).
1024-token prompts at concurrency (v15): 8 users TTFT 4.6 s, TPOT 32.9 ms;
16 users TTFT 8.0 s, TPOT 62.3 ms. 4096-token prompts, 8 users: TTFT 16.9 s
(the serialized prefill queue), TPOT 33.3 ms.

1024-token prompts at concurrency (v13): 8 users TTFT 6.4 s (v8: 13.7 s),
16 users 12.4 s (27.4 s), 32 users 24.7 s (not measured before); decode TPOT
33 / 51 / 53 ms. TTFT at concurrency is still the serialized prefill queue.

### Sweep v22 (2026-09-16 19:57 UTC, bench-sweeps harness, package b03f982b)

Cold boot after `tt-model rm`, random-token prompts, 4 x users prompts per
row (1 or 2 at 16k and above), zero failures. v20 (2026-09-11 package) in
parentheses. Changes since v20: batched prefill for prompts up to 2k tokens,
16-user decode bucket, prompt-independent indexed prefill programs, sharded
32-row decode norms, 64k-row chunked prefill (the 131,072-token bucket serves
at 30.8 s TTFT on the local server; the harness stops at 64k).

| ISL | OSL | users | TTFT | prefill tok/s/u | decode tok/s/u | aggregate tok/s | E2EL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 128 | 1 | 0.26 s (0.25) | 490 | 65.8 | 58 (58) | 2.2 s |
| 128 | 128 | 8 | 0.58 s (1.89) | 221 | 30.6 | 217 (170) | 4.7 s |
| 128 | 128 | 16 | 0.78 s (3.76) | 164 | 23.7 | 334 (175) | 6.1 s |
| 128 | 128 | 32 | 1.14 s (7.49) | 112 | 20.3 | 552 (286) | 7.4 s |
| 1,024 | 256 | 1 | 0.38 s (0.48) | 2,709 | 64.6 | 59 (58) | 4.3 s |
| 1,024 | 256 | 16 | 2.83 s (5.13) | 362 | 23.4 | 298 (193) | 13.7 s |
| 1,024 | 256 | 32 | 5.22 s (9.90) | 196 | 19.5 | 447 (343) | 18.3 s |
| 4,096 | 256 | 1 | 0.78 s (0.98) | 5,276 | 63.8 | 54 (51) | 4.8 s |
| 4,096 | 256 | 16 | 12.26 s (11.82) | 334 | 23.4 | 177 (145) | 23.2 s |
| 4,096 | 256 | 32 | 24.32 s (23.79) | 168 | 19.9 | 220 (217) | 37.2 s |
| 16,384 | 256 | 1 | 2.58 s (2.54) | 6,352 | 62.1 | 38 (39) | 6.7 s |
| 32,768 | 256 | 1 | 5.29 s (5.23) | 6,195 | 59.9 | 27 (27) | 9.5 s |
| 65,536 | 256 | 1 | 12.01 s (12.16) | 5,455 | 56.1 | 15 (15) | 16.6 s |

Batch-32 layer-0 traced decode with the sharded norms: 1.262 ms sliding,
1.384 ms full attention (was 1.326 / 1.454 ms).

Sweep v23 (2026-09-17 01:00 UTC, package bbe275d0, gate/up bias fused into the
sparse expert matmul), v22 in parentheses: 128/128 at 32 users TTFT 0.99 s
(1.14), 558 tok/s (552); 1k at 32 users 5.09 s (5.22), 457 tok/s (447); 4k at
32 users 22.9 s (24.3); 16k 2.51 s (2.58); 32k 5.15 s (5.29); 64k 11.88 s
(12.01); decode rows unchanged; every row measured, zero failures.

### Accuracy re-check: IFEval (2026-09-17 01:20 to 01:42 UTC, package bbe275d0)

None of the changes above had an accuracy re-check, so IFEval (google/IFEval,
541 prompts) was re-run against the local server with the release generation
policy (chat API, `reasoning_effort` medium, greedy, `max_gen_toks` 4096, seed
42, chat template applied) but at 32 concurrent requests, so the batched
prefill, the 32-user decode trace and the indexed MoE prefill with the fused
bias are the paths exercised. Harness: the TT lm-evaluation-harness fork
(`evals-common`) in `.venv_evals`; `lm-eval[api,ifeval]` needs `transformers`
and `tokenizers` added for the Hugging Face tokenizer backend.

| metric | 2026-09-17 (this package) | 2026-09-03 release run | published | release minimum |
|---|---:|---:|---:|---:|
| prompt-level strict | 85.2% (461/541, stderr 1.5) | 85.6% (463/541) | 78.2% | 74.29% |
| instruction-level strict | 89.0% | not recorded | | |
| prompt-level loose | 88.2% | not recorded | | |
| instruction-level loose | 90.9% | not recorded | | |

Unchanged within noise. 14 of 541 answers are empty (the reasoning channel
used the whole 4096-token budget, as under the release policy). Two of 543
chat requests returned HTTP 500 from the vLLM Harmony parser
(`HarmonyError: Unexpected token 392 while expecting start token 200006`: the
model emitted text where the parser expected `<|start|>`, and the
non-streaming chat path has no fallback); the harness retried both
successfully. About 0.4% of requests at 32 users; the random-token sweeps
cannot see it because they do not parse the output. Fix belongs in the vLLM
fork (catch the parser error, return the raw text). Results and samples:
`/home/ttuser/dev/gpt-oss-120b/benchmarks/perf_plan/phase4/ifeval_v23/`,
chain script `benchmarks/perf_plan/phase4/ifeval_chain2.sh`.

### Serving fix: Harmony parser recovery (2026-09-17, vLLM fork 53b3c0128)

The IFEval re-check surfaced 2 HTTP 500s in 543 chat completions at 32
concurrent requests: the model emitted a plain text token where the Harmony
grammar expects `<|start|>` after `<|end|>`, `openai_harmony` raised
`HarmonyError: Unexpected token ... while expecting start token 200006` from
`vllm/parser/harmony.py::process_chunk`, and the non-streaming chat path had
no handler, so the client got an opaque 500 (the streaming path died
mid-stream). The fork already recovered when the stream *ended* in a
non-terminal state (`flush`); commit 53b3c0128 on branch `tt/gpt-oss-serving`
extends that to a token rejected mid-stream: completed messages are kept, the
current message and everything after it are surfaced as final-channel
`content`, a warning names the offending token, and the strict parser is not
fed again for that turn. Unit test:
`benchmarks/perf_plan/phase4/test_harmony_fallback.py` (run with
`PYTHONPATH=/home/ttuser/dev/gpt-oss-120b/vllm tt-metal/python_env/bin/python`;
note the bare `python_env` resolves `vllm` to the gpt-oss-20b mirror
checkout). The bundle ships the fork as a prebuilt wheel
(`runtime.vllm.wheel` in `tt-model.yaml`), rebuilt with
`VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.26.0.dev2+g53b3c0128 uv
build --wheel --python 3.10` from the fork checkout.

### Sparse matmul kernel work (2026-09-16 evening)

- Per-group fused bias: `ttnn.sparse_matmul(..., indices=..., bias=...)` adds group
  `indices[i]`'s `[1, N]` bias inside the matmul (bf16 tile row per expert, new
  `BIAS_PER_GROUP` path in the shared in1 reader and compute kernels, sparse
  factory only). The indexed prefill passes its gate/up bias this way
  (`GPT_OSS_120B_INDEXED_FUSED_BIAS=0` restores the gather + add; the adapter
  falls back automatically on a ttnn without the operand). Fused vs gather+add
  on the same input: PCC 0.99997 / 0.99996 / 0.99996 at 1k / 4k / 16k, MoE
  prefill 8.5 vs 8.8 ms, 19.7 vs 23.8 ms, 62.1 vs 64.4 ms per layer.
- Second in0 multicast sender: `in0_senders=2` lets the first two cores of the
  grid alternate the in0 blocks (the sender's DRAM reads and NoC injection are
  split). Output bit-identical, no gain on this model (8.25 -> 8.04 ms at 1k,
  19.7 -> 19.1 ms at 4k, 62.9 -> 63.5 ms at 16k), so the single sender is not
  what bounds these matmuls: with the shipped (15, 4, 2, 1) blocking a 16k
  layer issues about 3,500 multicast blocks of 120 KB at about 4.8 us each, and
  the per-block chain (multicast, flag, 47 acknowledgements into one core)
  dominates the 1.4 us transfer and the sub-microsecond compute. The next
  lever is fewer synchronisations per byte: larger blocks when the L1 budget
  allows, or one acknowledgement per two blocks with a deeper in0 CB.
  Option kept, default 1 (`GPT_OSS_120B_INDEXED_IN0_SENDERS=2` to enable).
- Two K blocks per in0 synchronisation (`in0_block_pairs=True`, four-block in0
  CB, one acknowledgement and one flag per pair): bit-identical, gate/up stage
  16.8 -> 15.6 ms at 16k (-7%), 5.23 -> 5.08 ms at 4k, down matmul unchanged,
  whole-layer time within noise. So halving the synchronisations is worth about
  1 ms per 16k layer; together with the second-sender result this leaves the
  in1 (weight) reads and the compute pipeline as the remaining candidates, to
  be settled with a device profile (`run_safe_pytest.sh --profile`). Option
  kept, default off (`GPT_OSS_120B_INDEXED_IN0_BLOCK_PAIRS=1`): it doubles the
  in0 L1 footprint to 480 KB, which the serving budget is unlikely to afford.
- Still open from item 6: the gate/up slice copies (2.5 to 3 ms per 16k layer)
  need a two-output writer in the sparse matmul or an in-place SwiGLU.

## Remaining per-layer overheads at batch 32 (device profile, v5)

sparse matmuls 643 us (near DRAM roofline for ~81 experts), two unsharded
layernorms 82 us (sharding at a full tile was disabled during bring-up after a
nondeterministic trace), zero-fill of the expanded sparse outputs 56 us,
gate_up bias add 41 us, QKV matmul 44 us (19 us at batch 1, auto program
config), fused router 40 us.

## Reproduce

```bash
cd /home/ttuser/dev/gpt-oss-120b/tt-metal
SNAP=/home/ttuser/dev/gpt-oss-120b/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
env GPT_OSS_120B_BATCHED_DECODE_PERF=1 GPT_OSS_120B_BATCHED_DECODE_CHECK=1 GPT_OSS_120B_SNAPSHOT=$SNAP \
  scripts/run_safe_pytest.sh --run-all \
  models/autoports/openai_gpt_oss_120b/tests/test_multichip_batched_decode_perf.py -q -s -k sliding
bash /home/ttuser/dev/gpt-oss-120b/tti-release/openai_gpt_oss_120b/launch_autoport_server.sh
/home/ttuser/dev/gpt-oss-120b/benchmarks/perf_plan/run_conc_sweep.sh <out_dir> http://127.0.0.1:8000 128 128 1 2 4 8 16 32
```

Raw logs: `/home/ttuser/dev/gpt-oss-120b/benchmarks/perf_plan/`.
