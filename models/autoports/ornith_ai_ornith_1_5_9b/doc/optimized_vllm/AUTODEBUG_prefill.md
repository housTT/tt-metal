# AutoDebug: serving prefill trace reuse

Source-only investigation, 2026-09-08. No implementation edits, TTNN imports,
device access, server requests, or profiler collection. The supervising agent
owns the unchanged serving baseline and subsequent experiments.

## Finding and measurement boundary

**The existing B1 prefill trace is structurally reusable with the exact external
cache already bound by the adapter. The ownership eligibility check prevents
that reuse.** The size of the serving benefit remains unmeasured. Start with
this one hypothesis; keep B2..32, continuation, long/chunked, and all-logits
prefill on their existing eager path.

The completed integration repeat records **50.127 ms serving TTFT**, native
262144 context, capacity B1, prompt128/output128; its fresh-first request is
62.613 ms ([integration latency evidence](../vllm_integration/AUTOFIX_b1_latency.md)).
The selected predecessor's **35.52 ms** is teacher-forcing TTFT for the
161-token AIME prompt with physical cache context2048, not prompt128 token-out.
The matching selected standalone workload reports **29.002 ms** warmed
prompt128/output128 token-out TTFT, native cache context262144
([datatype evidence](../datatype_sweep/README.md)). Neither cross-harness
subtraction establishes removable serving overhead. Preserve the selected
`head4_lofi_last8_c32_k4_r2` policy and use fresh/repeated serving A/B results
for the performance verdict.

## H1: external ownership unnecessarily excludes the proven B1 body

Source facts:

- [Adapter cache construction](../../tt/generator_vllm.py), lines99–124,
  allocates the scheduler-sized physical pool once and constructs
  `OrnithGenerator(kv_cache=cache, page_table=...)`. Both adapter `_generator`
  and generator `_prefill` require that exact Python cache object.
- [Generator selector](../../tt/generator.py), lines603–630, additionally
  requires `owns_cache`, `max_batch_size == 1`, `rows == [0]`, `starts == [0]`,
  no all-logits result, and exact logical length1..prefill_chunk. Serving sets
  `owns_cache=False` at construction, so every adapter prefill is eager.
- [Model cache allocation](../../tt/model.py), lines339–355, uses the same
  layer objects for B1 decode and prefill whether the cache is external or
  internal. The prefill trace body already receives `self.kv_cache`, never
  allocates/rebinds the KV pool, and copies its result into canonical `_logits`.
- `validate_prefill` (model lines487–511) runs before the traced/eager branch,
  checks logical tokens/lengths, fixed slots, native page-table coverage and
  physical block bounds. The trace key binds exact cache identity, logical
  length, start, slot, and table shape; page values are refreshed separately.

A host-only AST execution of the **actual unchanged selector** verified these
resident-key cases, without importing runtime modules:

| Cache | Capacity | Length/start | Selected |
| --- | ---: | --- | --- |
| Owned | 1 | 128/0 and 131/0 | yes |
| External | 1 | 128/0 and 131/0 | no |
| Either | 32 | 128/0 | no |
| Owned | 1 | 128/128 | no |

Verdict: the selection exclusion is verified; safe external-cache replay and
serving speedup remain hypotheses requiring the focused experiment below.

### Smallest candidate and startup ordering

Permit external caches under the other existing B1 eligibility conditions.
Prepare the startup's exact128-token IDs/page-table inputs **before**
`warmup_model_prefill` calls `ensure_traces(preserve_cache=False)`. Keep its
canonical sampling-key priming, real admission/decode warmup, deferred read,
and complete state reset. That prepares the same coordinated four traces over
the known-empty pool and avoids an unnecessary native-cache snapshot.

Simply removing `owns_cache` without moving preparation earlier is incomplete:
startup currently captures decode/sampler first (adapter line345), then its
first prefill selects a new shape. `_prepare_prefill_trace` releases those
traces; `_prefill` calls the default `ensure_traces()`, which clones all external
KV/recurrent/conv buffers before resetting/warming/restoring them
(generator lines638–673). The explicit `preserve_cache=False` optimization
would therefore be lost.

Also test the **first real request having length131 after startup128**. Startup
ends in `gen.reset()` and leaves `_live=False` (adapter line379; generator
lines1051–1054). It is not yet a live shape miss: current selection would evict
128 and take the default external snapshot/warm path. Either prove that path's
native-memory fit and state preservation, or retain the prepared external
shape and fall back on misses once startup has prepared it. Do not claim all
non128 requests already use the live fallback. A first sampled/penalized request
also exercises non-live sampling reconfiguration and should be included in
startup correctness evidence; its cache snapshot behavior predates this change.

Do not set `owns_cache=True` to bypass the gate. That also disables external
cache preservation in `ensure_traces`. Do not call `reset(clear_kv=True)` on a
live serving pool to enable shape eviction. The existing `_live` miss guard
and exact-cache identity errors remain required.

### State, shape and trace invariants

Logical length131 remains131 in the persistent token tensor and last-valid-row
selection (`model.prefill_last_logits`, lines522–531). The existing multichip
decoder pads internally, passes `logical_len` to the block, and trims afterward
(`multichip_decoder.py`, lines512–529). Do not bucket131 into128/160 or sample
the final padded row. Existing all32-layer exactness covers128/131 with owned
caches; reduced edge gates cover1/2048. These are useful controls, not proof
of the new external path ([prior integration](../optimized_full_model/AUTOFIX_prefill_integration.md)).

At B1 the fresh-prefill reset touches that one request's shared hybrid state;
full-attention reset does not clear its KV pool. At B>1, prefill has separate
single-user hybrid scratch and `_prefill_validated` transfers states to/from
the selected decode slot, constructs the complete batch's terminal input,
and chooses that slot's page-table row (model lines551–598). Removing the
B1/slot/start guards would bypass those operations. B32 remains valid through
its existing eager path and needs a serving regression measurement, not a
new prefill-tracing claim.

Keep `_prepare_prompt_sampling` before `_ensure_replay_safe`: seed/penalty
setup may compile new programs, and recapture replaces `_logits`. Keep capture
order decode → sampler → history sampler → prefill; release old traces before
changing persistent inputs. The prefill output remains a temporary copied
into canonical logits and deallocated inside capture. Public
`return_device_logits=True` receives an owned clone; freeing that result must
not free the captured canonical destination. Live sampler reconfiguration must
retain cache, feedback, seed/history, and the resident prefill key while
recapturing the complete family (generator lines491–514,573–601,784–826).

## H2: first-token sampling still submits eagerly after H1

`_prefill` calls `_sample_prefill_device`, which unconditionally calls
`_sample_device` (generator lines516–543,829). Standalone `generate` instead
uses `_sample_first_token`; that helper replays the existing canonical sampler
only when the input is `_logits` and the program-cache count matches the
capture, otherwise safely samples eagerly (lines337–344,978–980).

This applies to eager B32 prefill as well as B1. H2 can therefore be tested
independently of H1: route ordinary prefill logits into the existing canonical
sampler while preserving partial-admission state. Required boundaries are:

- For an eager, independently owned `outputs`, finish seed/history preparation
  and `_ensure_replay_safe()` before copying `outputs` into the current
  `self._logits`. Capture may replace `_logits`, so the copy must follow any
  recapture. Release the owned eager output before replaying the older sampler
  trace. Keep return-all-logits and explicit host-logits ownership separate.
- For already canonical traced-prefill output, finish any recapture before the
  prefill replay, as today. Do not subsequently replace that populated buffer.
  Warm the exact logits-copy signature before capture, and check program counts
  at the actual replay boundary; a newly compiled copy/helper cannot be ignored.
- Preserve the existing exact INT32 mask restoration of unadmitted tokens,
  full UINT32 seeds and penalty histories. **Simply replacing `_sample_device`
  with `_sample_first_token` inside today's helper is unsafe:** its saved clones
  are allocated after the sampler trace and remain live during sampling.
  `tt_metal/impl/allocator/trace_allocation_tracker.cpp`,
  `record_allocation_if_unsafe`/`get_unsafe_tracked_ids`, records these against
  every older trace. The native replay guard will reject that lifetime, and
  replay can overlap those buffers without the guard.

The smallest candidate using the existing sampler trace needs stable backup
buffers allocated before capture, with per-prefill copies into them. Token and
seed backups are small; penalty-history backups must retain the exact target
shapes/dtypes and require native-memory evidence. Allocate any newly required
history backups only at a trace-release/rebuild boundary, never while an older
trace can replay. Masks may remain ephemeral if created after sampler replay
and freed before the next replay. A separate prefill-sampler trace wrapping the
existing common sampling plus temporary save/restore is another feasible
boundary if persistent history backups exceed budget, but adds a fifth trace
and a broader lifecycle change. Neither option is an implementation result.

Prediction: warmed B1 and B32 prefill record canonical sampler replays with
exact eager-control tokens and preserved untouched lanes. Test H2 separately,
including native allocation tracking; no measured sampler speedup is claimed.

## Focused verify/refute plan

1. Extend the existing CPU selector/lifecycle and startup fixtures:
   `doc/optimized_full_model/test_prefill_trace_contract.py`,
   `tests/test_generator_serving_contract.py`, and
   `tests/test_serving_startup_warmup.py`. Replace the old external-cache
   rejection expectation with positive exact-cache B1 cases. Prove startup
   preselection precedes capture; no snapshot occurs for the empty startup
   pool; default `ensure_traces()` still preserves external sentinels; wrong
   cache identity fails; B32/continuation/all-logits remain eager; changed
   logical shape cannot release live traces. Explicitly cover the first131
   request immediately after startup128 and its allocation/snapshot policy.
2. Use the existing `probe_prefill_integration.py` comparison pattern with
   sequential external-cache candidate/eager generators, native context,
   exactly the production physical block count and selected dtypes. Never
   coexist two native pools or trace families. First run layers0/3, then all32.
   Compare complete logical logits, all-rank recurrent/conv state and at least
   the next decode logits/tokens for: token A/B/A at128; page P/Q/P; fresh131
   capture; resident128 → live131 eager miss →128 reuse; same128 while `_live`;
   startup128 → first131; continuation; and owned-result deallocation before
   replay. Put sentinels in unreferenced physical pages and require them
   unchanged. Inspect cache identity and persistent input addresses directly.
3. Alternate greedy → seeded → penalties → greedy using real adapter
   `configure_sampling` paths, including nonneutral first requests. Assert
   seed/history/feedback preservation, correct new-request resets, zero host
   sampling, and exact eager-control output. For H2 additionally prove the
   helper handles program-count changes without losing populated logits and
   retains untouched sampler lanes. Cover B32 partial admissions, eager-output
   deallocation, persistent backup lifetime, and exact logits-copy warming.
   Reuse the existing partial-admission tests.
4. Run the focused correctness probe once under native trace-allocation
   tracking, including program-cache allocations, with Watcher only if required
   by the supervising lane. Require repeated recapture and teardown leave zero
   trace bytes. Keep all such instrumentation out of performance measurements.
5. Benchmark the actual server with the unchanged primary128/128/1 workload:
   one fresh-first request and repeated warmed requests, same server settings,
   native context, sampling policy and selected precision. Compare H1 alone
   first, then H1+H2 if H2 is pursued. Retain raw TTFT/ITL/TPOT/throughput and
   counters proving prefill replay, unchanged token feedback and no host decode.
   Repeat the32-request burst as a regression; B32's counters should still
   show eager prefill. No profiler of any kind is part of these experiments.

The existing `tests/b1_startup_latency_probe.py` provides bounded host spans
and source hashing without profiling, and
`tests/adapter_serving_device_probe.py` provides changed-page, async-read and
slot-remap controls. Neither currently proves external B1 prefill replay;
extend the focused coverage rather than relabeling old passing artifacts.

## Inspected source receipt

Only this report was written. Initial worktree modifications were the two
user-owned `AGENTS.md` files; neither was changed. Source SHA256 at inspection:

```text
generator.py       7a8c046f33768bab079812e6d4c048b60045625fbd2e383b564dbadca3600993
model.py           f9da26100bdf48dd1a2e487a3687895e75b2b385f0a32cf21d10c2786907e615
generator_vllm.py  881b38abae6ff278d3f66212e2ab4d8a462ce4ebd42153f65c5497b1d97ad324
```

Status: source exclusion verified; implementation, external-cache device
correctness, lifecycle qualification and serving performance remain pending.
