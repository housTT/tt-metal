# Watcher log classification — optimized decoder

The committed `watcher_log.txt` is the log of a single watcher-enabled run of the optimized
decoder's state-, trace- and optimization-critical test subset. It is classified here so "the log is
clean" is a checkable claim rather than an impression: every line falls into exactly one bucket, the
buckets sum to the file's line count, and a fatal-class regex over the whole file returns zero
matches.

`census.py` reproduces both the census and the grep into `census_summary.txt`.

## Command

The watcher run is the **last step** of [`../logs/run_evidence.sh`](../logs/run_evidence.sh), which is the
one place the test filter is defined. Run that rather than retyping it; the shape is:

```bash
rm -f generated/watcher/watcher.log
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
python -m pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_optimized_decoder.py \
  -v -p no:randomly -k "<filter, see the watcher step of run_evidence.sh>" \
  > doc/optimized_decoder/logs/watcher_pytest.txt
cp generated/watcher/watcher.log doc/optimized_decoder/watcher/watcher_log.txt
python doc/optimized_decoder/watcher/census.py > doc/optimized_decoder/watcher/census_summary.txt
```

Three things about it are load-bearing:

* **`TT_METAL_WATCHER=10`, no disabled features.** Watcher's asserts are left on; nothing is
  skipped. `TT_METAL_WATCHER_DISABLE_ETH` is *not* set — this is a single-chip run with no Ethernet
  traffic, so the ACTIVE_ETH kernel-config overflow that would justify it does not arise.
* **`TT_METAL_WATCHER_APPEND=1`.** Watcher truncates its log on each device open, and this subset
  opens the device once per test, so without it the committed log would hold only the last test's
  session.
* **No profiler.** Watcher and Tracy/device-profiler collection are separate runs, in separate
  steps of `run_evidence.sh`, and never share a process; the profiler step runs immediately before it.

## Subset

The filter selects the tests that exercise something this stage changed or that writes device state
in place, which is what watcher can actually catch:

* every in-place cache/state writer — paged fill and update, the DeltaNet recurrent and conv state,
  the poisoned-free-pool and determinism cases;
* trace capture and replay (`traced_decode`);
* the batch fallbacks (`above_head_split_limit`, `batch_smaller_than_allocated_state`, `ragged`);
* and this stage's own contracts — `padded_rows` (the routing-sparsity change),
  `tuned_program_configs` (the decode program configs and the L1 expert intermediates),
  `precision_policy` (the dtype policy), `layout_churn` (the sharded-norm boundary), and
  `optimized_matches_fused`.

Tests that drive prefill or decode purely to check numerics — the PCC ladders, the weight-source
cases, the full-context cases — are deliberately out: they exercise no memory pattern the tests
above do not, and they are the slowest cases in the suite.

## Fatal-class grep

`census.py` greps the whole log for watcher errors/asserts, out-of-bounds accesses, invalid NOC
coordinates or addresses, L1/stack overflow, sanitizer reports, corruption, and hang/deadlock
markers. The result is reproduced into `census_summary.txt` as `fatal-class matches: N`; the
committed run has **0**, and there are no suspected false positives to explain.

## Stack watermarks

Watcher prints a stack watermark only for dumps where the firmware happened to record one, so
whether a given log carries stack-headroom evidence varies between runs of the same subset.
`census.py` reports an absence explicitly rather than letting it read as "no overflow". Either way
the conclusion is the same: a stack overflow would also surface through the fatal-class grep, which
is clean.
