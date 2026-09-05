# AutoDebug: traced output history collection

Follow-up: the parent subsequently verified the component probe on hardware and
authorized a generator candidate. See `AUTOFIX_output_collection.md` for the
implemented decode-only window policy, preserved first-token TTFT boundary, and
current validation status. The diagnosis below records the original proposals.

Source-only investigation on 2026-09-05, starting from local HEAD
`2e4b8f828c`. No device access, implementation edits, or performance claim.
The stage harness directory was already untracked. This investigation follows
`autofix`, `autodebug` (already-isolated investigator), and `tt-enable-tracing`.

## Finding and proposed first experiment

The device free-running branch of `tt/generator.py::generate` submits one
`_read_async()` for every decode step and `_finish_read()` waits for each event.
Its asynchronous submission does not meet the current no-per-token-readback
contract. The existing model and sampling traces already own token feedback,
position advance, sampler penalties, and seed advance; collection can observe
the sampled tokens without changing those mechanisms.

The smallest source-supported exact-integer collector is:

```python
updated = ttnn.indexed_fill(cursor, history, tokens, dim=0)
ttnn.copy(updated, history)
ttnn.plus_one(cursor)
ttnn.deallocate(updated)
```

All three operations belong in the same warmed trace. **The copy is required:**
`indexed_fill` allocates an output and does not mutate its history input. Merely
assigning a Python variable to its result during capture would make every replay
read the original history allocation, losing prior rows.

| Tensor | Logical/padded shape | Dtype/layout/memory | Meaning |
| --- | --- | --- | --- |
| `tokens` | `[1,1,1,32]` | UINT32, ROW_MAJOR, DRAM INTERLEAVED, replicated on native TP4 | Existing `gen._inputs[0]`, unchanged |
| `history` | `[G,1,1,32]` | UINT32, ROW_MAJOR, DRAM INTERLEAVED, replicated on TP4 | One complete 32-lane row per generation step |
| `cursor` | `[1]` | INT32, ROW_MAJOR, DRAM INTERLEAVED, replicated on TP4 | Generation-window row index, initially zero |
| `updated` | Same as history | Same as history | Trace temporary, copied into stable history |

Use request-sized `G` (or a small capacity bucket), not native context capacity.
Each history row is 128 bytes, aligned on Blackhole. Logical batch 1..32 is
independent of the 32 physical sampler lanes. After one final history read,
reshape to `[G,32]`, select the requested fixed slots, and transpose to
request-major output. Mixed prompt lengths do not change the cursor: it counts
generation steps, not absolute KV positions. Inactive lanes may be recorded and
discarded by the host's existing slot mapping; the collector never changes their
token, RNG, position, or penalty state.

## Source evidence and caveats

- `tt/generator.py` allocates the token tensor through `model.upload` as
  replicated UINT32 RM DRAM `[1,1,1,32]`. `_sample_device` passes it as
  `tt_out_tok`, then advances `seeds_tt_tensor` exactly once. `tt/model.py`
  advances current positions with `skip_negative_entries=True` and RoPE indices
  separately. A collector must only read that token buffer after sampling.
- `ttnn/cpp/ttnn/operations/data_movement/indexed_fill/indexed_fill.cpp` requires
  matching ranks and matching dimensions except the indexed dimension;
  `input_b.size(dim)` must equal `batch_id.padded_shape()[-1]`. Our rank-four
  tensors and single index satisfy those conditions. RM dim 0 uses the native
  wrapper route without permute; DRAM interleaved takes the generic device
  kernel.
- `indexed_fill/device/indexed_fill_device_operation.cpp` requires rank four,
  equal layouts, and integer RM indices. It allocates a new output with the
  input's dtype. The generic reader/writer copy rows using NoC dataflow rather
  than arithmetic. `indexed_fill_program_factory.cpp` derives row bytes from
  `input_a.element_size()` and uses aligned buffer page sizes, including
  Blackhole's 64-byte DRAM alignment. No BF16 conversion is involved.
- `data_movement/copy/device/copy_device_operation.cpp` explicitly supports
  UINT32 source/output, equal shapes/layouts, and a supplied output allocation.
  `ttnn.copy(src, dst)` supplies that allocation. Same-dtype RM copy performs
  exact data movement.
- `experimental/plusone/device/plusone_device_operation.cpp` supports INT32 and
  UINT32 RM and returns the input allocation. **Keep the cursor in DRAM.**
  `plusone_program_factory.cpp` explicitly documents that interleaved L1 uses
  scratch and never references its input; an L1-interleaved cursor can appear to
  run while remaining unchanged. DRAM and L1-sharded have real backing paths.
- `tests/ttnn/unit_tests/operations/data_movement/test_indexed_fill.py` covers
  rank-four RM/TILE dimensions, duplicate-index semantics, and cache rebinding,
  but its data tensors are BF16. It does not establish exact UINT32 traced TP4
  correctness. `test_plus_one.py` covers width-one integer tensors, negative
  entries, and cache rebinding, but uses PCC and relatively small integers.
  Therefore the exact-shape UINT32 probe is required before production edits.
- `indexed_fill` scans valid output slices for matching indices; it does not
  validate cursor values on the host. Out-of-range values can silently leave
  history unchanged. Bound replay count to the allocated history capacity.
- The collector copies the entire history twice per step (indexed fill and
  copy-back). Storage is `128*G` bytes per rank for each persistent/temporary
  history, and data movement grows with `G` per step. This is a correctness-first
  candidate, not an asserted optimization. Measure it at G=128 before keeping it;
  very long generation windows may justify another mechanism.

## Alternatives inspected

| Operation | Exact source contract | Decision |
| --- | --- | --- |
| `ttnn.scatter` | Equal-rank input/index/source; matching input/source dtypes; integer indices; interleaved only; wrapper converts to RM and moves scatter axis last. Output is newly allocated. | Legal alternative: history `[1,1,32,G]`, token view and cursor `[1,1,32,1]`, dim 3, then copy-back. More shape/index work than indexed fill and still copies full history. No in-place advantage. |
| `ttnn.tosa_scatter` | `tosa_scatter.cpp::pre_tosa_scatter_transform_tensor` calls `.cpu()`, converts indices to UINT16, then `.to_device()`. | Reject for capture: hidden host read/write. |
| `ttnn.experimental.slice_write` | Actual C++ signature takes `SmallVector<uint32_t>` begins/ends/steps and writes a supplied output. Python docs say BF16/RM only, while current validation is broader. | No tensor-valued dynamic offset. One captured invocation always writes the same row. Do not infer integer rejection solely from stale docs. |
| `update_slice` | `rg -n update_slice ttnn tests models/common models/tt_transformers` returned no symbol. | Not an available API in this checkout. |
| `ttnn.index_fill` | Fill value is a host `variant<float,int>`, not a token tensor. | Cannot append dynamic sampled tokens. |
| `ttnn.experimental.paged_update_cache` | In-place; tensor index INT32 RM DRAM; input and cache TILE; input must be height/block sharded with one core per input user. Input dtype FLOAT32/BF16, cache dtype FLOAT32/BF16/BFP8/BFP4. | UINT32 directly is invalid. A separate FLOAT32 history with one pseudo-user, 32 token lanes as head width, one-core `[32,32]` input shard, cache `[1,1,ceil32(G),32]`, cursor `[1]`, and `fp32_dest_acc_en=True` is a later bounded-write candidate. All real token IDs are below 2^24, but conversion/packing needs exact tests. It cannot preserve arbitrary seed-sized UINT32 values and adds more operations/configuration than the first probe. |

## Integration constraints after verification

1. Allocate history/cursor before warming and capture. Warm the exact history
   capacity, memory configs, dtypes, and append op sequence; forbid cache misses
   during capture when supported. Reset the cursor/history after warm execution.
   Capture records work but does not execute the first append.
2. Keep `_sample_device`, seed initialization/advance, penalty updates, selected
   BF16 HiFi4 LM head, decoder configuration, and native 1x4 TP mesh unchanged.
   Do not place an unconditional collector in `_sample_device`: that helper also
   runs during sampler warmup and partial prefill, where it must not advance
   unrelated output history.
3. For fixed-window free-running generation, append prefill's first sampled
   token once, then append once after each sampling trace replay. Initially a
   separately captured collector can prove the boundary without recapturing the
   model; after verification it can be appended to the sampling trace to avoid
   another host trace submission. Keep stable input/output identities through
   recapture and wait for pending work before releasing their traces/buffers.
4. The free-running window needs one final history transfer; teacher forcing and
   explicit host-sampling callbacks still need their existing visible token
   boundary. Existing EOS behavior already runs a fixed window and trims after
   generation, so collection does not require a new EOS policy.
5. Preserve TTFT meaning. Removing the first read also removes its completion
   barrier: a host timestamp immediately after asynchronous sampling is only
   submission time. Use device/event timing or separately measured TTFT; never
   rename submission latency as TTFT.
6. High-level `generate` uses leading active slots. For arbitrary scheduler slot
   IDs, select columns by the existing slot mapping after collection. Do not
   compact lanes, reset inactive sampler state, or derive cursor from per-slot
   position. Low-level APIs that return a token each call need their existing
   output contract unless an explicit collection interface is added.

The parent's proposed integration uses two sampling trace variants: plain
sampling for existing low-level APIs, and sampling followed by history append
for fixed-window output. `_replay(collect_output=False)` chooses the variant;
the model trace plus chosen sampler still total exactly two host submissions
per decode step. This is a smaller hot-loop boundary than a permanent third
collector trace and preserves the plain path's externally visible behavior.
The following conditions make that design coherent:

- Both sampling variants consume the exact persistent `_logits` allocation and
  write the exact `_inputs[0]` allocation. Capture the model first, then both
  sampler variants. Releasing/recapturing the model must recapture both variants
  before replay, because they refer to the previous logits address.
- Warm the append operations before either capture. Warm sampler execution
  mutates tokens, seeds, and penalties: use the existing save/restore mechanism
  when state is live. A collector-only warm need not run the sampler; save/reset
  only history and cursor after this warm. `configure_sampling`, `_capture`,
  `_ensure_replay_safe`, and `teardown` all need consistent variant ownership.
- Keep history, cursor, and the append temporary address valid for every trace
  referencing them. Releasing a tensor object after capture is only safe under
  the same trace-allocation lifecycle already used for other intermediates;
  do not reallocate a same-shaped history while retaining its old sampler trace.
- A fixed capacity of 128 supports one final read for a 128-token request:
  append the prefill token once, then do 127 collection replays. A request longer
  than 128 needs a host read/reset at window boundaries, or a larger history.
  Window reads are not per-token reads, but are also not one final read for the
  entire request. Report this distinction and keep window-boundary counters
  separate from steady-state token counters. Never wrap and overwrite unread
  output. Capacity must be checked before accepting a replay count.

The existing `_ensure_replay_safe` policy notices newly compiled prefill programs
and recaptures to avoid overlap with temporary trace addresses. The collection
variant must participate in that policy. Adding a variant without updating this
lifecycle is a concrete stale-address risk even if the standalone collector
probe passes.

## Verify/refute experiment

`probe_output_history.py` is a source-only prepared component probe, not a
production edit. The parent hardware owner must run it in the serialized lane:

```bash
python models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/probe_output_history.py \
  --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/output_history_probe.json
```

It imports the real TP4 mesh open/close helper, uses the exact tensor contracts
above with capacity 128, warms/reset/captures, and replays nonblocking. A fourth
probe-only `plus_one(tokens)` produces distinct device-owned inputs each step.
History comparison is exact UINT32 on all four ranks, including values above
65535, 2^31, and wraparound near 2^32. One final mesh history transfer per window
is the only read; its four already-host rank shards are checked separately.
Cases exercise one token, sparse fixed slots, all 32 lanes, full capacity,
non-tile window lengths, changed capture inputs, and repeated request reset.
The whole history including unwritten sentinel rows is compared; stable buffer
addresses, program-cache counts, host-work counters and elapsed times are logged.

Prediction: history row `s` equals `(initial_tokens+s) mod 2^32` on every rank,
unwritten rows stay sentinel, and repeated identical resets reproduce the entire
history. Refute this candidate on capture failure, stale rows, integer corruption,
wrong rank output, or measured cost that fails the stage's performance objective.
The probe alone cannot establish unchanged full-model seed/penalty semantics;
after a passing component probe, run identical seeded and greedy requests with
penalties enabled/disabled, mixed lengths/inactive slots, changed pages, and
alternating sampling modes against the prior output path.

Status: **hypothesis supported by source; hardware experiment pending**. No build
is required for this report/probe-only change. Parent owns hardware execution,
full-model verification, timing interpretation, and final stage status.

Source-only validation completed: `python3 -m py_compile` for the probe and
`pre-commit run --files` for both report/probe passed. No hardware-facing Python
import, listing, open, or test was run by the investigator.


## Final stage closure

The earlier investigation status above is preserved as historical evidence.
The selected implementation and completed final gates are recorded in the
[stage report](README.md), [runtime audit](runtime_audit.md),
[final full32 watcher control](prefill_integration_full32_v2/summary.json),
[long exact replay controls](prefill_integration_long_v2.json), and
[final profiling report](tracy/README.md). Earlier pending experiments are not
claims that these final gates remain unrun; rejected hypotheses and failed
receipts remain preserved. Independent stage review owns the final verdict.
