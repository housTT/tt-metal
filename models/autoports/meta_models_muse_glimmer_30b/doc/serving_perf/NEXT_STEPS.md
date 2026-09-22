# Serving performance — next steps for prefill and scheduling

Muse-Glimmer-30B decodes well under load and prefills badly under load. This
document records the measurement, names the cause in code, and ranks the work.

Measured 2026-08-31 on P300x2 (4 Blackhole dies, 1x4 `FABRIC_1D_RING`) against
tt-metal `0dd37ce6ee3`, through the vLLM OpenAI server at `max_num_seqs=32`,
`max_model_len=131072`. Raw data: `logs/batchsweep/`, summary
`logs/batch_sweep_summary.json`.

## 1. What the sweep found

**Decode batching is close to free.** At ISL 1,024 / OSL 512, concurrency 1 to 32
costs 4% of per-user speed and returns 22.7x aggregate throughput.

| conc | TPOT ms | t/s/u | out tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 23.84 | 41.94 | 41.5 |
| 32 | 24.88 | 40.19 | 942.0 |

**Prefill does not batch at all.** Concurrent requests prefill one after another.
The arithmetic is exact — at ISL 4,096, concurrency 32:

```
predicted  32 x 458.1 ms (batch-1 TTFT)  = 14.66 s
measured   median TTFT                   = 14.87 s   (1.4% error)
```

**The serial queue then corrupts decode.** Decode steps interleave with the
still-draining prefill queue, so TPOT under load is contention, not decode rate:

| ISL | conc | TPOT ms, batch 1 | TPOT ms, at conc | inflation |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 32 | 24.97 | 24.88 | 1.0x |
| 16,384 | 32 | 30.33 | 59.05 | 1.9x |
| 32,768 | 31 | 35.24 | 125.37 | 3.6x |
| 130,560 | 8 | 64.79 | 169.17 | 2.6x |

Aggregate throughput falls from 1,154 tok/s (ISL 128) to 17.7 tok/s (ISL 130,560).
Decode is not the limiter anywhere in that range. Prefill scheduling is.

**Agent loops pay the whole prefill again on every turn.** Prefix caching is
disabled (`enable_prefix_caching=False`), so nothing is reused between turns.
Measured with an agent-shaped transcript — user, assistant `tool_call`, tool
result, assistant text — each shape warmed then timed:

| rounds | conc | prompt tok | TTFT ms | total ms | ms/1k tok |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2,949 | 357.7 | 721.2 | 121.3 |
| 2 | 1 | 5,515 | 682.9 | 756.9 | 123.8 |
| 4 | 1 | 10,647 | 1,416.4 | 1,467.0 | 133.0 |
| 8 | 1 | 20,911 | 2,807.8 | 2,860.9 | 134.3 |
| 16 | 1 | 41,439 | 6,190.3 | 6,248.3 | 149.4 |
| 32 | 1 | 82,495 | 14,020.2 | 14,088.2 | 170.0 |

`total ms` sits on top of `TTFT`, so time to first token is effectively the whole
response time for an agent turn. Cost per 1k tokens rises 121 -> 170 ms, so
prefill is slightly worse than linear as context grows.

A 32-turn task that adds 2,578 tokens per turn therefore spends:

```
re-prefilling every turn   215.0 s
the same history, once      14.0 s
wasted                     201.0 s   (15.3x)
```

This is the dominant cost of running the model as an agent backend. It is larger
than decode and larger than any concurrency effect, and it grows with
conversation length.

## 2. Where it comes from, in code

1. `vllm-tt-plugin/src/vllm_tt_plugin/platform.py:57` —
   `_CHUNKED_PREFILL_MODEL_TYPES = {"gemma4", "gemma4_unified"}`. This is an
   allowlist. `model_type=muse_glimmer` is absent, so line 79 sets
   `enable_chunked_prefill = False`. The restriction is a plugin policy, not a
   property of this model.
2. `platform.py:98` forces `long_prefill_token_threshold = 0`, because the base
   scheduler would otherwise split a prefill the model cannot resume.
3. `tt/generator_vllm.py:808-815` rejects any resumed prefill:
   `NotImplementedError`, "serving prefill starts every request at position 0 ...
   this port does not expose the layer stack's continuation prefill through the
   serving path."
4. The capability exists one level down. `tt/model.py:718` already chunks prefill
   internally at 8,192 tokens, and `generator.prefill_forward` documents
   mixed-length batched prefill into distinct cache slots.
5. The adapter already forwards the plural forms — `prompt_lens=lens` and
   `user_ids=empty_slots` (`generator_vllm.py:820`, `:823`). Nothing in the port
   forces one request per prefill call.

6. Prefix caching is **per-model opt-in**, not a backend limitation.
   `generator_vllm.py:323` declares `"supports_prefix_caching": False`, and
   `platform.py:1191` disables it on that basis.
7. Flipping that flag alone is not enough. `platform.py:1199-1207` then disables
   prefix caching for **any** model whose config reports a sliding window, and
   that check is not model-gated.
8. The root cause of 7 is one line. `get_kv_cache_spec` reads `layer_types`
   (39 `sliding_attention`, 13 `full_attention`), validates each kind, and then
   emits `FullAttentionSpec` for every layer regardless
   (`generator_vllm.py:515-519`). The sliding distinction is parsed and
   discarded, which is also why `_HYBRID_KV_CACHE_GROUPS_ENABLED = False`
   (`generator_vllm.py:331`) is correct as written.

The gap is between the scheduler and the serving adapter, not in the layer stack.

## 3. Ranked work

### P0 — Bounded sliding-window KV cache, which unlocks prefix caching

39 of the 52 layers attend within a 2,048-token window. Every one of them is
allocated the full 131,072-token cache. That single defect causes the KV
over-allocation *and* blocks prefix caching, so one change returns both.

```
current  token-layers/user   6,815,744   52 x 131,072
bounded  token-layers/user   1,783,808   13 x 131,072 + 39 x 2,048
over-allocation                   3.82x
```

Work, in order:

1. Emit `SlidingWindowSpec` for the 39 sliding layers in `get_kv_cache_spec`
   instead of `FullAttentionSpec`, and set
   `_HYBRID_KV_CACHE_GROUPS_ENABLED = True`.
2. Implement a bounded (ring) KV cache for sliding layers in the layer stack,
   with the page-table arithmetic to match.
3. Set `"supports_prefix_caching": True`.
4. Narrow `platform.py:1199-1207` so it refuses prefix caching only when the
   spec is uniform, not whenever a sliding window exists.
5. Implement continuation prefill — shared with P1, below.

Expected, all three from the same change:

| | today | bounded |
| --- | ---: | ---: |
| KV at full context, batch 1 | 1.85 GB/device | 0.49 GB/device |
| concurrency at 131,072 | 8 | 30, then capped at 32 by `DECODE_ROWS` |
| prefill on a 32-turn agent loop | 215.0 s | 14.0 s |

Risk: the same sliding-window correctness trap as P1. A bounded cache that drops
a tail the window still needs returns fluent, wrong text. Gate on PCC.

### P1 — Chunked prefill

Split a long prefill across scheduler steps so other requests decode in between.
This addresses both the TTFT queue and the TPOT inflation.

Step 1 below is the same continuation-prefill work P0 needs. Doing it once
serves both items, so sequence P0 and P1 together rather than independently.

Work:
1. Implement continuation prefill in `generator_vllm.prefill_forward`. Accept
   `start_pos > 0`. Remove the guard at `generator_vllm.py:808-815`.
2. Add `muse_glimmer` to `_CHUNKED_PREFILL_MODEL_TYPES` in the plugin.
3. Restore a nonzero `long_prefill_token_threshold` for this model type.

Risk — this is the hard part, and it is specific to this model:

* 39 of the 52 layers are sliding-window (`sliding_window=2048`, `model.py:358`,
  `:748`). A chunk boundary must carry the correct K/V tail into the next chunk,
  or the window silently truncates. A truncated window does not raise; it returns
  fluent, wrong text. This is the single most likely way to ship a regression here.
* `paged_fill_cache` requires a multi-token prefill to start on a 64-token page
  boundary. Chunk sizes must be multiples of 64.
* Verify with PCC against the reference, not with throughput. A wrong window
  produces plausible text.

Expected: TPOT at ISL 16,384 / conc 32 returns from 59.05 ms toward the batch-1
30.33 ms; TTFT becomes fair-shared instead of FIFO.

### P2 — Batched prefill for short prompts

Prefill several short requests in one call.

The port already supports this (section 2, items 4 and 5). The limiter appears to
be scheduler-side: only one prefill reaches the runner per step.

**Confirm before building.** The present evidence is timing, not instrumentation.
Count the requests per prefill step in `model_runner.py` around line 2068 and
confirm the count is 1. If the scheduler already groups them, the cause is
elsewhere and this item is void.

Expected if confirmed: at ISL 128 / conc 32, TTFT 2.16 s toward roughly one
batched prefill. Low risk, no new numerics.

### P3 — Re-measure prefill tracing

`MUSE_GLIMMER_VLLM_PREFILL_TRACE=0` is the shipped default, chosen in the
optimized-vLLM stage. Re-measure with tracing on now that prefill is known to
dominate serving. Cheap: an env var and one sweep.

### P4 — Scheduler knobs, no code change

Deployments that care about TTFT more than aggregate throughput can lower
`--max-num-seqs`. Table B shows the trade directly. Document per-workload
profiles rather than one default.

## 4. Operating guidance until P0 and P1 land

* Interactive and agentic coding — long ISL, short OSL. Cap concurrency well
  below 32. TTFT is set by the queue ahead of the request.
* Batch and offline — short ISL. Run at 32. Throughput peaks at 1,154 tok/s.
* Do not quote a concurrent TPOT above ISL 16,384 as a decode figure. It is
  contention. Quote the batch-1 column.
* Agent harnesses — keep sessions short, or compact history aggressively.
  Turn latency tracks total conversation length, not the size of the new turn.
