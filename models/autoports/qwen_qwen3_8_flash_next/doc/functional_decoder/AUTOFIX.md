# Functional-decoder AutoFix report

Date: 2026-08-26

Scope was restricted to the functional decoder for
`Qwen/Qwen3.8-Flash-Next`. AutoFix used independent, fresh-context diagnosis
for the environment, QSA correctness, and watcher failure. No optimized,
multichip, full-model, or vLLM work was started.

## Starting failures

- The initial hardware environment mixed editable TTNN 0.65.1 native libraries
  from a sibling checkout with the current checkout's 0.75 JIT sources. The
  resulting five compile diagnostics were exact July-to-August API mismatches.
- Real layer-3 QSA prefill PCC was `0.27623364`; representative GDN layers were
  already above `0.998`.
- After correctness repair, watcher found a real 4096-byte NOC read into a
  512-byte embedding index scratch circular buffer.

## Hypotheses tested

| Hypothesis | Result | Evidence |
| --- | --- | --- |
| BF16/compute precision caused QSA PCC collapse | Refuted; the failure was too large and isolated to page/gather geometry | `QSA_AUTODEBUG.md` |
| Int32 `floor_div` generated exact page quotients | Refuted on device; it returned the wrong quotient for the relevant operands | `QSA_AUTODEBUG.md` |
| Power-of-two shift/mask page decomposition repairs aliasing | Verified and retained for token pages and compressed-block pages | `QSA_AUTODEBUG.md`; final real PCC |
| Multi-row tiled embedding indices preserve logical row order | Refuted; flattening indices to one row before gather repaired the ordering | `QSA_AUTODEBUG.md`; final real PCC |
| Four QSA index heads can rely on implicit batch broadcast | Refuted for batch greater than one; explicit replication retained | batch-2 paging/current-position tests |
| Fixed 512-block top-k explains the catastrophic result | Refuted as the cause of PCC 0.276, but independently verified as an underfilled-context semantic bug during stage-review remediation; fixed and regression-tested | `QSA_AUTODEBUG.md`; `topk_multiset.log` |
| Watcher failure was a model arithmetic fault | Refuted; source ledger proves the tiled-index embedding factory reads a 4096-byte UINT32 tile into a 512-byte DFB entry | `AUTOTRIAGE.md`; failing watcher artifact |
| Row-major device indices avoid the unsafe embedding kernel without host fallback | Verified and retained for every QSA embedding lookup | `watcher_qsa_fix_20260826_1251/pytest.log`; final watcher artifact |

## Retained repair

`functional_decoder.py` now:

- uses bit shifts and masks for page decomposition;
- flattens multi-row QSA embedding indices and reshapes gathered results;
- explicitly replicates batched index keys across all four indexer heads; and
- converts embedding indices to TTNN row-major layout before device embedding,
  requesting tiled output. This avoids `embedding_ind_tilized` while remaining
  fully on-device and trace-safe; and
- masks fixed-shape top-k filler lanes separately from the incomplete tail, so
  every HF-visible underfilled-context token is gathered exactly once.

The underlying TTNN tiled-index embedding scratch-size mismatch remains outside
this model directory; this functional decoder no longer selects that kernel.

## Final acceptance evidence

Real checkpoint results after the retained fixes:

| Layer kind | Prefill PCC | Traced replay decode PCC |
| --- | ---: | ---: |
| GDN, layer 0 | 0.99871051 | 0.99997765 |
| GDN + PLE, layer 1 | 0.99910986 | 0.99991739 |
| QSA, layer 3 | 0.99679226 | 0.99988198 |

All exceed the unmodified `0.995` functional-decoder bar. The dynamic fallback
guard is active during these passes. Focused QSA watcher rerun passed with the
same PCC and no sanitizer fault. The aggregate final watcher run and profiler
artifacts are referenced from `README.md` and `work_log.md`.

## Diagnostic artifacts

- `AUTODEBUG.md`: native/JIT environment split-brain diagnosis.
- `QSA_AUTODEBUG.md`: QSA page/gather diagnosis and hypothesis ranking.
- `AUTOTRIAGE.md`: exact watcher CB/NOC source ledger.
- `watcher_embedding_failure_20260826_1244/pytest.log`: preserved failing run.
- `watcher_qsa_fix_20260826_1251/pytest.log`: focused repaired run.
- `topk_multiset.log`: exact HF token-multiset regression for the final
  underfilled top-k repair.
- `watcher_final_20260826_1419/pytest.log`: post-repair aggregate watcher run.
