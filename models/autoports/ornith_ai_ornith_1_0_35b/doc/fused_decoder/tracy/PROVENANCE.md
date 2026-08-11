# Performance artifact provenance — fused decoder

Device: 1×1 Blackhole mesh (`p300c`), 11×10 worker grid (as auto-detected by `tt-perf-report`).
Weights: real `ornith-ai/Ornith-1.0-35B` checkpoint, `bfloat16`.
`tt-perf-report` version: 1.2.8 (installed into the active tt-metal venv).

These files are the **after** side of the fusing stage. The **before** side is the functional
stage's committed capture set, `../../functional_decoder/tracy/`, produced by the same script with
the same signposts, the same iteration counts and the same weights — so the two are directly
comparable, window by window. `../README.md` §5 tabulates them side by side.

## How these files were produced

`run_profiling.sh` (committed next to this file) drove four separate captures — one per
(layer kind × phase) — so each Tracy capture holds exactly one signposted window and one device
session:

```bash
python -m tracy -r -p -v --op-support-count 50000 -o <out> -m pytest <test>::<node-id>
```

Two things about that command line are load-bearing:

* **Node ids, not `-k "a and b"`.** `python -m tracy -m pytest` re-splits its argv, so a quoted
  `-k` expression containing spaces arrives at pytest as separate arguments and fails with
  `ERROR: file or directory not found: and`.
* **`--op-support-count 50000`.** The profiler's default program budget is 1000
  (`tools/tracy/common.py: PROFILER_DEFAULT_OP_SUPPORT_COUNT`). A warmed 2048-token prefill (three
  passes: two warmup + one measured) and 32 traced decode replays both exceed it, after which
  post-processing asserts
  `Device data missing: Op <id> not present in cpp_device_perf_report.csv`.

Reports were then rendered from the copied ops CSV:

```bash
tt-perf-report <phase>_ops.csv --start-signpost PERF_<PHASE> --end-signpost PERF_<PHASE>_END \
  --csv <phase>_perf_report.csv --no-advice > <phase>_perf_report.console.txt
tt-perf-report <phase>_ops.csv --start-signpost PERF_<PHASE> --end-signpost PERF_<PHASE>_END \
  --no-summary --no-advice > <phase>_perf_report.txt
```

`--csv` mode prints command/status chatter rather than the rendered table, so it goes to
`*_perf_report.console.txt`; the human-readable table is the separate `*_perf_report.txt`.

## Files per layer kind

| File | Contents |
| --- | --- |
| `<phase>_ops.csv.gz` | raw post-processed Tracy ops CSV for the whole run (`gunzip -k` to re-run `tt-perf-report` on it) |
| `<phase>_perf_report.txt` | human-readable `tt-perf-report` table for the signposted window |
| `<phase>_perf_report.csv` | same window, one row per op, machine-readable |
| `<phase>_perf_report.console.txt` | `--csv` run chatter (roofline summary, warnings) |
| `<phase>_perf_report_stacked.{csv,png}` | `tt-perf-report`'s stacked-category report |
| `<phase>_tracy_run.txt` | the profiled pytest console output with kernel-compilation spam stripped; retains the `PERF_*` signposts, the logged wall clocks and the pass/fail summary |

The raw Tracy capture directories were deleted after the CSVs and reports were extracted; re-create
them by re-running `run_profiling.sh`.

## Latency column and unit

Latency is taken from the filtered `tt-perf-report --csv` output, column **`Device Time`**, which
this version reports in **microseconds**. `summarise_perf.py` sums that column over the signposted
window and divides by the iteration count inside the window: **1** for prefill, **32** for decode
(32 `execute_trace` replays). `perf_summary.txt` is its output and is the only place the headline
device times are computed.

`summarise_slow_ops.py` reduces the same reports a second way, into `slow_ops_summary.txt`: every
`SLOW`-flagged row per window (not a top-N), grouped by geometry, math fidelity **and** core count,
with launch counts and DRAM/FLOP utilization ranges.

## Known report warnings (expected, not defects)

* `SparseMatmulDeviceOperation rows without numeric nnz were found. Their DRAM/FLOP utilization is
  omitted; pass --active-experts K …` — the MoE deliberately passes `nnz=None` so the op infers the
  non-zero count at runtime (a static `nnz` that disagrees with `count_nonzero(sparsity)` deadlocks
  the in0-mcast receivers). `--active-experts K` is **not** passed because there is no single
  correct `K`: the sparsity mask is the union of the experts chosen by the 32 tokens sharing a
  batch entry, which is ~8·batch in decode and about 64 % of 256 in prefill (`1 - (1 - 8/256)**32`). Only the DRAM/FLOP
  utilisation columns for those rows are affected; the device times are unaffected.
* `Unclassified operation 'TopKDeviceOperation' / 'ScatterDeviceOperation' /
  'DeepseekMoEFastReduceNCDeviceOperation' / 'RotaryEmbeddingHfDeviceOperation' /
  'NLPCreateQKVHeads*DeviceOperation' / 'NLPConcatHeadsDeviceOperation' /
  'PagedFusedUpdateCacheDeviceOperation'` — these ops are simply missing from `tt-perf-report`'s
  category table. Their rows are still timed and included in the sums.
