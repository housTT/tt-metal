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

6. Prefix caching was **per-model opt-in**, not a backend limitation
   (`platform.py:1191`), plus a second refusal for any model reporting a sliding
   window (`platform.py:1199-1207`). Both are addressed in P0.
7. The uniform `FullAttentionSpec` (`generator_vllm.py:515-519`) is **not** the
   obstacle it looks like. An earlier revision of this document called it the
   root cause of 6; that was wrong. It is what makes cross-request block reuse
   legal here, because one block id indexes all 52 per-layer cache tensors. What
   it costs is memory, which is P5.
8. What was genuinely missing was a *resumed* prefill. The sliding layers read
   their window from a handed-over tail, which only exists when the same process
   prefilled the prefix; a prefix-cache hit has no such call. That is what
   `FusedDecoder._prefill_sdpa_sliding_paged` now supplies.

The gap was between the scheduler and the serving adapter, not in the layer stack.

## 3. Ranked work

### P0 — Prefix caching (implemented)

Prefix caching does **not** need a bounded sliding KV cache, and the earlier claim
in this document that it did was wrong. The two are independent:

* Prefix caching needs a *resumed* prefill ``[start_pos, prompt_len)`` whose prefix
  K/V is already in the paged cache. The uniform ``FullAttentionSpec`` is what makes
  that work rather than what blocks it: one block id indexes all 52 per-layer cache
  tensors, so a donor request's blocks carry valid K/V for the sliding layers too,
  and vLLM's ``FullAttentionManager`` retains every block, which is conservative for
  layers that read fewer.
* The bounded cache is a *memory* change, and it is the one the absolute-position
  hazard blocks. It is now P5 below.

What it took:

1. **tt-metal host plumbing.** The kernels already composed chunked prefill with a
   sliding mask, but ``sdpa.cpp:125`` passed ``std::nullopt`` for
   ``sliding_window_size`` in the chunked overloads, commented "not supported yet".
   Forward-ported to both overloads plus the nanobind arg.
2. **A paged window read**, ``FusedDecoder._prefill_sdpa_sliding_paged``. A sliding
   layer reads its window out of the cache with a page-table row shifted to
   ``sliding_window_origin(start_pos)``, so ``chunk_start_idx`` is exactly
   ``sliding_window`` at any offset and one SDPA program serves every request.
   Recomputing the window instead does not work: a tail-less chunk runs the square
   SDPA where query row ``j`` sees ``j+1`` keys, and the error compounds with depth
   to roughly ``39 x 2048`` tokens of required recomputation.
3. **Resumed prefill** threaded through ``model.py`` and ``generator.py``, with the
   prefill trace skipped (it bakes ``start_pos=0``) and the logits row made
   chunk-relative.
4. **``prefix_in_cache`` as a required keyword** on ``_prefill_chunk`` /
   ``_prefill_attention``, so an overriding decoder class that misses it raises
   ``TypeError`` instead of silently taking the tail-less path.
5. **A separate plugin capability**, ``supports_prefix_caching_with_sliding_window``.
   The plugin refuses prefix caching for any model reporting a sliding window, and
   several models (``mistral_7b``, ``phi4``) declare ``supports_prefix_caching`` and
   depend on that refusal. Relaxing the generic key would have enabled prefix
   caching for them silently, so the new key is what the plugin now consults.

Evidence:

| check | result |
| --- | --- |
| chunked SDPA vs torch band mask, permuted paged cache | 0.99976 — 0.99982 |
| the same, vs an **unwindowed** reference | 0.9487, i.e. correctly worse |
| paged read vs HF reference (shipping class) | 0.99185 — 0.99338 |
| paged read vs the proven tail hand-off | 0.99976 — 1.0 |
| prefix remapped onto unwritten blocks | 0.59 — 0.66, i.e. the read really goes through the page table |

The last row is the load-bearing one. Every other assertion would still hold if the
window came from somewhere incidental; that one only passes if the read follows the
donor's page table.

### Measured end to end, served

vLLM with `--enable-prefix-caching` on the 4-chip mesh, same agent-shaped transcript
as section 1, warm server:

| rounds | prompt tok | TTFT, APC off | TTFT, APC on | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,949 | 357.7 ms | 85.3 ms | 4.2x |
| 2 | 5,515 | 682.9 ms | 106.8 ms | 6.4x |
| 4 | 10,647 | 1,416.4 ms | 109.2 ms | 13.0x |
| 8 | 20,911 | 2,807.8 ms | 128.0 ms | 21.9x |
| 16 | 41,439 | 6,190.3 ms | 142.1 ms | 43.6x |
| 32 | 82,495 | 14,020.2 ms | 353.6 ms | 39.6x |

Cumulative TTFT across the six shapes: **25.5 s to 0.9 s**. The shape is the point:
TTFT no longer tracks total conversation length.

Two honest qualifications.

The harness sends each shape twice and times the second, so the timed call is a
**100 % prefix hit**. A real agent turn adds new tokens, so its saving sits between
these numbers and the APC-off column — but much nearer this one, because the new
tokens per turn are small against the history.

The *first* pass over a cold server was far slower (2.56 s at 1 round, 1.33 s at 32)
because each distinct resume offset compiles its own SDPA program. That is the
program-cache growth noted under *Known follow-up*; it is one-time per offset, and
the table above is the warm steady state. Worth fixing before this is relied on for
first-request latency.

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

### P5 — Bounded sliding-window KV cache

Still worth doing, and still blocked by what this port already documented twice
(``tt/generator_vllm.py:60-66``, ``doc/vllm_integration/stage_review.md:223-227``):
decode passes absolute positions, while vLLM's ``SlidingWindowSpec`` zero-pads a
sliding group's page table, so positions past the window collapse onto physical
block 0. No ring or modulo cache indexing exists anywhere in the port.

It buys memory and concurrency, not prefill latency:

```
current  token-layers/user   6,815,744   52 x 131,072
bounded  token-layers/user   1,783,808   13 x 131,072 + 39 x 2,048
over-allocation                   3.82x
```

| | today | bounded |
| --- | ---: | ---: |
| KV at full context, batch 1 | 1.85 GB/device | 0.49 GB/device |
| concurrency at 131,072 | 8 | 30, then capped at 32 by ``DECODE_ROWS`` |

``models/demos/gemma4`` is the reference port for the pieces it needs: per-layer
page-table routing, ``allocate_kv_cache_per_layer``, and hybrid KV groups.

## 4. Operating guidance

* Interactive and agentic coding — long ISL, short OSL. Cap concurrency well
  below 32. TTFT is set by the queue ahead of the request.
* Batch and offline — short ISL. Run at 32. Throughput peaks at 1,154 tok/s.
* Do not quote a concurrent TPOT above ISL 16,384 as a decode figure. It is
  contention. Quote the batch-1 column.
* Agent harnesses — keep sessions short, or compact history aggressively.
  Turn latency tracks total conversation length, not the size of the new turn.
