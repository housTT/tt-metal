# Performance artifact provenance

Device: 1×1 Blackhole mesh (`p300c`), 11×10 worker grid (as auto-detected by `tt-perf-report`).
Weights: real `ornith-ai/Ornith-1.0-35B` checkpoint, `bfloat16`.
`tt-perf-report` version: 1.2.8 (`python -m pip install tt-perf-report`, installed into the active
tt-metal venv).

## How these files were produced

`run_profiling.sh` (committed next to this file) drove four separate captures — one per
(layer kind × phase) — so each Tracy capture holds exactly one signposted window and one device
session, as the functional-decoder skill requires for "warmed prefill and warmed decode captured
separately":

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
| `<phase>_ops.csv.gz` | raw post-processed Tracy ops CSV for the whole run (copied from `<phase>/…/ops_perf_results_<timestamp>.csv`, then gzipped — `gunzip -k` to re-run `tt-perf-report` on it) |
| `<phase>_perf_report.txt` | human-readable `tt-perf-report` table for the signposted window |
| `<phase>_perf_report.csv` | same window, one row per op, machine-readable |
| `<phase>_perf_report.console.txt` | `--csv` run chatter (roofline summary, warnings) |
| `<phase>_perf_report_stacked.{csv,png}` | `tt-perf-report`'s stacked-category report |
| `<phase>_tracy_run.txt` | the profiled pytest console output, with kernel-compilation spam stripped; retains the `PERF_*` signpost markers, the logged wall-clock numbers and the pass/fail summary |

The raw Tracy capture directories (`<phase>/`, 4.9 GB in total: `.tracy` captures, per-core device
logs, wasm trace copies) were deleted after the CSVs and reports were extracted. Re-create them by
re-running `run_profiling.sh`.

## Latency column and unit

Latency is taken from the filtered `tt-perf-report --csv` output, column **`Device Time`**, which
this version reports in **microseconds** (the rendered table prints it as `… μs`). `summarise_perf.py`
sums that column over the signposted window and divides by the number of iterations inside the
window: **1** for prefill, **32** for decode (32 `execute_trace` replays).

```
kind              phase       ops  iters  device ms/iter  time column
linear_attention  prefill     369      1         337.759  Device Time
linear_attention  decode     3712     32           2.533  Device Time
full_attention    prefill     335      1         316.373  Device Time
full_attention    decode     3424     32           2.340  Device Time
```

The same tests also log a wall clock. Outside the profiler (`../logs/pcc_summary.txt`) that is
338.22 / 317.04 ms for prefill and 2.614 / 2.393 ms per traced decode step; inside the capture
(`<phase>_tracy_run.txt`) it is 338.94 / 316.98 ms and 2.676 / 2.451 ms. So the device sums track the
unprofiled wall clock to 0.14 % / 0.21 % on prefill and 3.2 % / 2.3 % on traced decode — the measured
windows are device-bound, with the decode residue being per-replay host dispatch. README §6 tabulates
all three columns side by side and is the place to read them.

`summarise_slow_ops.py` reduces the same reports a second way, into `slow_ops_summary.txt`: the
`SLOW`-flagged rows per window (17 / 352 for `linear_attention` prefill / decode, 16 / 256 for
`full_attention`) grouped by geometry, fidelity and core count — every group, not a top-N — with
launch counts and DRAM/FLOP utilization ranges. Those are
the un-tuned dense matmuls this stage deliberately leaves to the optimization stage; README §7 item 5
carries the table and what each group's shape implies.

## Known report warnings (expected, not defects)

* `SparseMatmulDeviceOperation rows without numeric nnz were found. Their DRAM/FLOP utilization is
  omitted; pass --active-experts K …` — the MoE deliberately passes `nnz=None` so the op infers the
  non-zero count at runtime (a static `nnz` that disagrees with `count_nonzero(sparsity)` deadlocks
  the in0-mcast receivers). `--active-experts K` was **not** passed because there is no single
  correct `K` here: the sparsity mask is the union of the experts chosen by the 32 tokens sharing a
  batch entry, which is ~8·batch in decode but close to all 256 in prefill. Only the DRAM/FLOP
  utilisation columns for those rows are affected; the device times are unaffected.
* `Unclassified operation 'TopKDeviceOperation' / 'ScatterDeviceOperation'` — the router's on-device
  top-k and scatter are simply missing from `tt-perf-report`'s category table; their rows are still
  timed and included.
