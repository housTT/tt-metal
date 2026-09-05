# AutoDebug: batch-32 fixed-slot trace disagreement

Inspection: 2026-09-05. This is a source-only investigation under AutoFix. No
TTNN import, device listing/open/reset, model execution, or implementation edit
was performed. Only this report was written. Line references below refer to the
immutable failure source archive, not subsequent generator edits.

## Evidence and conclusion

The original failing command, recorded in
`logs/trace_b32_watcher_fixed.provenance.json`, was:

```bash
timeout 240 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.trace_contract --batch 32 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/trace_b32_watcher_fixed.json
```

Its environment includes `OMP_NUM_THREADS=8`, `HF_HUB_OFFLINE=1`,
`TT_METAL_WATCHER=10`, and `TT_METAL_WATCHER_DISABLE_ETH=1`. Trace allocation
tracking was **not enabled**. The log exits at the Python page-permutation
assertion; worker watcher did not assert and the devices closed normally.
`AUTOFIX_watcher.md` separately verifies the prior split-reader NoC-accounting
repair with bitwise terminal outputs. This report does not reopen that bug.

Source snapshot: `logs/trace_b32_watcher_fixed.sources.json.gz`, SHA-256
`b8e19b583b91c73a8a3b44059c5eb09fe7b43a91d76911f03edc1e1188ad4991`.
The inspected model and four decoder files still matched that snapshot.

**The symptom is broader than page-table permutation.** Parsing the actual
assertion payload on the host found:

- Original versus permuted outputs match all 31 slots at token indices 0, 1,
  2, 5, 6, and 7. Only slot 30 differs at indices 3 and 4.
- Within the original run, all eleven copies of the 127-token prompt start
  `[12, 13]`, then slots 0/3/6 produce `97743/96/46550` at token index 2.
  Every duplicate of slot 0 diverges at that index. All duplicates of the
  131-token prompt likewise diverge from slot 1 at index 2. Short-prompt copies
  diverge at indices 2 or 3.
- Slot 30 is `[12,13,220,220,220,220,220,220]` originally and
  `[12,13,220,12,13,220,220,220]` after permutation. Slot 31 is inactive.

Thus there is already an equal-prompt/fixed-slot problem before using physical
page permutation as the distinguishing variable. This is an exact functional
invariant even for the reduced layers `[0,3]`; language quality of that
untrained composition is irrelevant. Existing B4 prefill equality and isolated
B32 terminal/sampler successes do not establish equal-prompt B32 decode.
No particular cache, kernel, or numerical cause is proven by source inspection.

## Relevant implementation boundaries

| Boundary | Failure-snapshot source | What the probe must distinguish |
| --- | --- | --- |
| Sequential ragged prefill | `tt/model.py:406-448` | A batch-one layer object is reused per prompt; only hybrid state is copied into the fixed batch. Full-attention objects share the allocated K/V pair. |
| Hybrid state handoff | `tt/model.py:359-380` | `repeat` plus in-place `where` writes one batch row; earlier rows must remain exact after each later prompt. B4 slots 0/3 passing does not cover B32 masks/state geometry. |
| Prefill hidden assembly | `tt/model.py:440-455` | Last logical token is sliced/cloned, then slot rows are concatenated. Capture individual rows and the combined terminal input. |
| Decode input/output packing | `tt/model.py:313-319,457-477` | Token slicing/reshape, embedding gather/reshape, and `[B,1,D]` to `[1,B,D]` terminal reshape introduce boundaries absent from isolated decoder tests. Compare native shapes before flattening on the host. |
| Inactive hybrid rows | `tt/model.py:464-472`; `tt/generator.py:134-149` | Model clones each state and restores inactive rows using persistent masks. Inspect masks and active/idle rows; do not infer correctness merely from `current_pos == -1`. |
| Trace lifetime | `tt/generator.py:198-242,413-415` | Traces are warmed before prefill and recaptured only when the program-cache entry count changes. That count is a proxy, not an allocation-lifetime proof. |
| Token feedback/readback | `tt/generator.py:163-168,244-269,422-448` | Model and sampler are separate nonblocking traces; sampling writes the persistent 32-token tensor. Async CPU copies must match synchronous token-buffer snapshots. |
| Greedy tie resolution | `models/common/sampling/tt_sampling.py:663-778,1093-1122` | Lowest-index tie adjustment is already present. Its documented limit is more than 32 tied maxima in one shard. Compare actual gathered candidates with the full-vocabulary CPU argmax. |

The exact failed cache geometry is available without guessing: page size 64,
table `[32,32]`, 1024 physical blocks, and local TP4 K/V shape
`[1024,1,64,256]` in BFP8_B. Slot 30 originally owns pages 960..991; reversing
columns maps its logical pages 0/1/2 to 991/990/989 instead of 960/961/962.
These are in range, unique, and do not touch slot 31's 992..1023 pages.
The first equal-prompt divergence consumes positions 128/132/4 for the three
prompt groups. Only the first group crosses a 64-token page boundary then.
SDPA's selected 256-token chunk and the 2048-token backing per slot make a
simple end-of-allocation cliff unlikely; still log the rounded read interval.
Prefill physically writes 128/256/128 tokens for logical lengths 127/131/3.
Logical-tail and padded-tail cache checks must therefore be reported separately.

## Minimum diagnostic, before changing implementation

Use the reduced `[0,3]` model, B32 allocation, 31 original prompts, current
selected precision/geometry, and one serialized hardware lane. Preserve the
original free-run reproducer. A diagnostic may add blocking reads; it is not
performance evidence. Run with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_TRACEBACKS=1` before importing TTNN. If tracking stops the
run, save the traceback and resolve that exact allocation before interpreting
later numerical experiments. Do not disable program-cache accounting.

1. **Separate repetition from permutation.** On the same generator, run three
   reset/prefill cases: A = original table, A2 = original table again, B =
   column-reversed table. Optionally A3 follows B if only B differs. At each
   boundary below, compare A/A2, A/B, and identical-prompt slots within a case.
   The present test changes request reuse and page mapping together.

2. **Capture prefill before any sampled feedback.** Use
   `prefill_forward(..., return_device_logits=True)` and read exact all-rank
   terminal logits. Record last-token hidden states before slot concatenation
   and the assembled terminal hidden tensor. A temporary diagnostic wrapper
   around `model.terminal` can copy its input into a buffer allocated before
   traces; read it outside capture. Do not put host reads in captured methods.
   Record state immediately after the first slot handoff and after all 31
   handoffs, especially slots 0/3/27/30/31. Compare full recurrent and all three
   convolution states per rank, not just their norms or first-device outputs.
   Check earlier rows remain unchanged when later slots are populated.

3. **Compare cache in logical coordinates.** Read each rank's K/V and reorder
   pages on the host using the case's table. For cache tensor `C` and slot `u`,
   use `C[table[u].long()].permute(1,0,2,3).reshape(local_kv_heads,-1,head_dim)`.
   Compare equal-prompt slots, A/A2, and A/B exactly over the logical prefix;
   separately compare padded prefill rows through the physical prefill end.
   Within one mapping also compare untouched physical pages against a snapshot.
   Confirm decode/prefill full-attention objects refer to the intended same
   cache buffer on every rank. Raw physical A/B tensor equality is not expected.

4. **Force the first three decode inputs.** After prefill and
   `_ensure_replay_safe()`, set the same complete 32-slot token/position/RoPE
   arrays for each case and refresh its page table. For decode steps 1/2/3 use
   the common per-prompt baseline token sequence (anchor slots 0/1/2), even if
   another slot predicts differently. The first two consumed-token sets are
   `[12,1076,220]` and `[13,1076,220]` by prompt group. Use explicit `-1` for
   slot 31's current position. First execute only `_model_trace` with a blocking
   correctness read, and collect logits, current positions, RoPE indices,
   masks, and post-update states. Then execute the sampling trace and compare
   every device's persistent token tensor against CPU argmax of those *same*
   logits. This localizes a wrong model output versus a wrong sampler result
   before different feedback tokens contaminate subsequent states.

5. **Stop at the first differing boundary.** Report exact equality, differing
   elements, max absolute error, finite/NaN/Inf counts, first differing slot,
   layer, rank, token index, absolute position, page and offset. For logits,
   retain the top 10 values/IDs, top-two gap, total tied-max count and tied-max
   count per vocabulary shard. If model logits differ, add persistent diagnostic
   copies at embedding output, layer 0 output, layer 3 output and terminal
   input. Allocate these before capture and copy inside capture; never create
   long-lived diagnostic device clones between capture and replay. Keep native
   `[B,1,D]` shapes in the first three copies, then reshape on the host to
   distinguish packing from decoder changes. Snapshot the original tensors'
   logical/padded shapes, dtype, memory config and per-rank addresses too.

This ladder needs no all-layer accuracy rerun until the first source of
inequality has been found. It also preserves the exact cache dtype, mapper,
page width, local head count and fixed-slot allocation required by AutoFix.

## Hypotheses and decisive follow-ups

| Hypothesis | Prediction / smallest verify-or-refute control |
| --- | --- |
| H1: hybrid handoff or fixed-slot packing corrupts otherwise equal prompts | Prefill last-token hidden states may match while batched states or assembled rows differ; or identical forced tokens produce different embedding rows. Check handoff preservation first. If needed prefill slots in reverse order with the same slot IDs; an error following visit order implicates transfer/lifetime, while an error following slot ID implicates row geometry. |
| H2: an allocation created after capture overlaps trace intermediates, or the recapture lifecycle is wrong | Allocation tracker gives a concrete offending live buffer, or exact restored eager and traced forwards disagree with identical input/state. Save trace IDs, program counts, input/cache addresses before and after `_ensure_replay_safe()`. Capture after all exact prefill variants are warmed as a single diagnostic control; do not keep unconditional recapture as a speculative fix. |
| H3: incorrect physical cache write/read or page-row indexing | Logical K/V differs across A/B before sampling. If writes differ, intercept the actual K/V producers and compare fused update against separate updates into scratch caches allocated before capture, checking every untouched row. If cache bytes and Q match but attention differs, compare SDPA output against CPU attention over the actual dequantized local-head cache. Adapt the existing exact-row diagnostic from `tests/test_cache_precision_regression.py`; its single-chip hard-coded four heads are not this TP4 shape. |
| H4: inactive-row restoration or tile padding spills into adjacent slot 30 | The mismatch moves when the inactive slot moves, or vanishes with all 32 active while positions/tokens of the compared active slots stay identical. Perform this only after boundary localization; record states for both active neighbors and the idle row. Filling 31 versus 32 rows also changes prefill allocation history, so disappearance alone is not proof of masking. |
| H5: common sampler placement, candidate ties, or asynchronous readback | Exact logits have the same CPU argmax but sampled device tokens differ. First compare all four token replicas; then run the same frozen logits through eager common sampling and its trace, capturing top-k values/indices and the adjusted winner. More than 32 maxima tied in one shard specifically supports the documented candidate limit. If persistent token buffers are correct but returned tokens differ, repeat generation with synchronous token readback; do not change model/cache precision. |
| H6: normal reduction/precision drift flips a narrow rank tie | Prefill/state/embedding checks pass, divergence starts at a specific collective or terminal boundary, and measured logits show a sufficiently narrow gap. Compare per-rank hidden states and use CPU final norm/head on the same accelerator hidden states. Only then run a same-cache, higher-precision control at that boundary. A passing higher-precision free run alone does not diagnose low-precision cache instability. |

The default formatted greedy parameters are semantically greedy for every lane:
scalar `temperature=0` and padding defaults both map to `k=1,p=0,temp=1`.
The common scalar-parameter formatting issue for *sampled* batches cannot by
itself explain this greedy failure. Parameter and seed tensors should still be
included in the frozen-logit sampler probe.

## Completion criteria

No fix is proposed as proven. First preserve the diagnostic's first mismatching
boundary and verify one hypothesis in isolation. After a verified repair, rerun
the original B32 trace contract with allocation tracking, its worker-watcher
variant, and equal-prompt B32 checks with all slots active and one inactive.
The existing `full_batch_contract.py` already checks all-layer identical-prompt
token equality; keep that as a later full-model gate. Retain B4 cache ownership,
continuation, changed-page and repeated sampled/greedy checks. Do not weaken
exact deterministic slot/page requirements or infer a quality/performance claim
from the reduced structural model.
