# AutoDebug: batch-32 full-attention trace PCC

Source-only investigation on 2026-09-05. No TTNN imports, device commands, or
implementation edits were performed. Root retains the hardware lane. All paths
below are relative to this autoport unless qualified otherwise.

## Evidence and verdict

`logs/contracts_initial_short.log` reports 53 passes, then failure in
`tests/test_functional_decoder.py::test_traced_decode_pcc`, layer 3, batch 32:
one user's HF/output PCC is `0.9948051904765759`, below the unchanged `0.995` bar.
The failure precedes the first `step=0` aggregate-PCC log, so it occurs at the
first replay, position 63. The user index and aggregate PCC were not logged.
All four residual replicas had already passed the adapter's exact-equality
check. That rules out unequal returned replicas, not identically wrong results.

The passing batch-32 eager test is **not an equivalent eager control**: it uses
prefix length 96 and seeds 39/131. The failing trace uses prefix length 63,
prefill seed 31, and decode seeds 3100/3101/3102. No current artifact establishes
whether the failing input also fails in eager mode.

The completed single-chip optimized stage ran the same recorded-input selection
and the same per-user trace assertions successfully. Its
`../optimized_decoder/logs/final_release_v4_short.log.gz` records batch-32 full
attention aggregate PCC `0.998479`, `0.998674`, `0.998607` at steps 0–2; the
corresponding watcher log repeats those values. The logs do not report the
single-chip minimum per-user PCC. A pre-existing unavoidable baseline precision
limit is therefore **not established**; the matching baseline has passing
evidence. The multichip regression is not yet localized to trace, cache, local
compute geometry, or collective rounding. Do not change precision or waive the
gate from this evidence alone.

## Source findings

- The input adapter in `tests/test_multichip_decoder.py::multichip_contract`
  duplicates the optimized stage's recorded-row selection formula. It retains
  real checkpoint weights and the per-user HF threshold.
- It changes snapshot semantics: baseline `_snapshot_state` uses host copies
  and explicitly warns about trace intermediates overwriting device snapshots;
  the multichip adapter keeps live `ttnn.clone` tensors and restores with
  `ttnn.copy`. Keeping live references makes ordinary allocation reuse unlikely,
  but snapshot immutability and exact restoration across capture have not been
  proven at this shape. The baseline comment is a hypothesis lead, not proof of
  a current allocator bug. The current adapter correctly avoids broadcasting
  rank-zero KV heads to every rank.
- Full-attention TP packing is structurally consistent: 4 local Q heads,
  1 local KV head, head dimension 256; packed projection width 2560 = Q1024 +
  K256 + V256 + gate1024. The HF Q/gate head-paired rows are split at whole-head
  boundaries and then unpacked locally. All projection split boundaries are
  tile-aligned. No obvious rank/head ordering bug was found.
- The inherited `_attention_decode` retains a 32-head physical height per user
  for head-sharded Q/K/V. With batch32, Q's 8×4 RoPE rectangle matches the first
  32 cores of the 8×8 SDPA grid. K uses the disjoint next 32 cores of the device
  grid; V stays on the first 32. The existing `_sdpa_query` correction remains
  active. The old arbitrary-batch RoPE/SDPA grid bug is not an obvious fit here.
- Caches are `[1024, 1, 64, 256]` per rank: 32 rounded blocks/user × 32 users.
  Page-table row `u` maps to physical blocks `32*u ... 32*u+31`; current
  position63 maps to physical block `32*u`, row63. Positions64/65 map to
  block `32*u+1`, rows0/1. The 256-token SDPA chunk has ample allocated pages
  (2048 physical tokens/user despite logical max_context1024). An allocation
  coverage shortage is not supported by this calculation.
- Numerical differences are nevertheless expected at specific boundaries:
  TP row projections compute four BF16 partial outputs followed by native
  all-reduce, instead of one single-chip reduction over K. Local projection
  geometry also changes from tuned per-role cores/readers to 8 cores,
  K-block4, reader1. Which boundary affects the failing user is unmeasured.
  Native two-link all-reduce is a different operation from the previously
  diagnosed two-link async all-gather/reduce-scatter corruption; do not infer
  that the earlier bug also explains this failure.

## Focused verify/refute sequence

Run experiments serially in the assigned hardware lane. Preserve each run's
source archive and policy ledger with `record_run.py`; only a verified cause
justifies a production change.

1. **Classify the failure using identical inputs, unchanged policy.** Build a
   diagnostic test beside the multichip tests using precisely the failing
   prefix, decode inputs, weights, batch, table, cache allocation, and position
   helpers. Before assertions, save per-user HF PCC and maximum error for every
   step/rank, minimum user's index, finite counts, and tensor hashes. Compare:
   fresh eager decode; eager decode after exact state restore; capture output;
   and first replay after exact state restore. Collect all four local KV caches
   individually outside forward. Record snapshot hashes immediately after
   cloning, warm-up, first restore, capture, second restore, and replay.
   A snapshot mutation or failed exact restoration verifies a harness/state
   issue; eager/trace identity with the same HF miss refutes a trace-only cause.
   The first numerical miss must log `(user, step, position, physical_block,
   row_in_block, floor(position/256)*256)`.

2. **Only if replay differs, isolate snapshot and trace state.** As a single
   changed variable, snapshot all four rank-local caches to the host and
   restore their own values using mesh-sharded host uploads into the existing
   buffers (never rank-zero broadcast and never new persistent cache addresses).
   Compare restored caches exactly, retaining the BFP8 dtype. If this repairs
   replay and immutable host snapshots reproduce eager output, keep the
   smallest adapter fix and rerun the original test. If it does not, discard
   the proposed fix and compare lowered input/position/table contents plus
   cache write results before pursuing precision.

3. **Run a genuine 1×1 baseline for the same diagnostic case.** Close the TP4
   mesh before opening baseline hardware; baseline reader2/3 cannot execute on
   a non-unit mesh. Save all per-user outputs, not just aggregate PCC. Compare
   TP4 against baseline for eager and trace. This measures baseline margin and
   establishes whether the currently observed HF regression is TP-specific.
   A failing baseline rerun still needs reconciliation with final-release-v4
   source/provenance; it is not permission to lower the threshold.

4. **If eager also misses, localize before tuning.** Capture these boundaries
   outside the measured path: input norm, packed QKVG, normalized/rotated Q/K,
   cache update, per-head SDPA output, gated attention output, reduced o_proj,
   first residual, FF norm, gate/up product, reduced down_proj, final residual.
   Compare TP local heads/columns to the corresponding single-chip slices and
   reconstruct global row-reduced outputs. Use HF/reference substitution at
   the earliest divergent boundary. If collective output is the first excess
   error, compare native all-reduce with existing one-link async RS+AG on the
   **same local partial tensors**, and with a host FP32 sum of those tensors.
   This distinguishes communication corruption, BF16 reduction order, and
   upstream projection error without changing multiple policies at once.

5. **If cache/SDPA is implicated, require exact-shape controls.** Use the model's
   local 1-KV-head layout, 64-token pages, allocation helper, 32-user table,
   mapper and update op. Seed distinct rank/user/head/page values, execute at
   positions62/63/64/65, and verify updated logical rows plus untouched pages.
   Test the same BFP8 cache with higher-precision SDPA or reference attention
   before changing cache dtype. A BF16-cache pass alone only localizes a
   sensitive boundary. If a boundary cliff appears, run an over-allocation
   control even though the present allocation calculation predicts coverage.

6. **Only if localized to numerics, test one policy boundary.** Examples are
   one row projection's accumulation/fidelity, the CCL sum, or decode SDPA.
   Keep weight groups, activations, other fidelities, cache dtype, CCL payload,
   page size/layout, table policy, and update/read ops fixed and logged. A
   higher-precision win must improve the localized intermediate and original
   per-user HF gate; otherwise revert it. Recheck both layer kinds and warmed
   trace latency for any kept production change.

The original narrow reproduction command is:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py batch32_original_repro timeout 180 python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -k 'test_traced_decode_pcc and 32 and full_attention' -x -q -s
```

That command is proposed, not executed by this investigator. The diagnostic
test in steps 1–5 must be authored by the repair agent; no hypothetical test
filename or result is presented as an existing artifact. After a proven fix,
rerun the original narrow test, the remaining short contract suite, and the
appropriate separate watcher check. Final status: **unresolved, actionable
localization plan; no speculative implementation patch**.
