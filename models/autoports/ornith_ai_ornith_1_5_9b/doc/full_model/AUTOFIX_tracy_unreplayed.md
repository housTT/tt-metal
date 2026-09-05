# AutoFix: unreplayed Tracy capture metadata

The original `tracy_decode_final_v1` device collection completed and closed all
four devices successfully, then Python report generation exited 1 at
`_enrich_ops_from_perf_csv`: Op 256001, device 1, trace 0 lacked a device row.
The original command status, raw files and immutable provenance remain intact.

## Diagnosis and focused verification

Raw `tracy_ops_data.csv` explicitly contains BEGIN/END/RELEASE for traces 0 and
1 on every device, but no REPLAY for either. Op 256001 is a SliceDeviceOperation
captured in trace 0. The generator's initial warmup captures these traces, then
new prefill program compilation requires recapture before their first replay.
Traces 2 and 3 each replay six times/device: two warm iterations and four
measured iterations between the start/stop signposts. Only actual execution
produces device timing rows, so absent timings for traces 0/1 are expected.

The raw device report has exactly 5772 measurements: per device, 537 nontrace,
714 trace-2 (119 operations × 6 replays), and 192 trace-3 (32 × 6). This consistent
identity/replay coverage refutes stale output-directory mixing as the cause.
The assertion demanded timing for definitions that never ran.

A new CPU regression with an unused trace-0 definition followed by two real
trace-2 executions fails against the original parser. Five accompanying
negative controls already pass. `logs/tracy_unreplayed_original_tests.log`
preserves that before-fix result.

## Minimal repair

When a host operation has no matching device row, a non-null captured trace ID,
and available host replay metadata showing no replay of that trace, skip device
enrichment for that unused definition. Preserve its original host metadata in
`ops`. Every existing device row is retained. Missing data for a replayed trace,
nontrace operation, or unavailable replay metadata still raises the original
assertion; no durations are invented and no executed measurement is discarded.

The 11-line change is in `tools/tracy/process_ops_logs.py`. Six CPU regression
cases in `tests/ttnn/tracy/test_process_ops_logs.py` cover unused captures,
multiple real replays, missing executed/nontrace/unknown-metadata rows, and
retaining device measurements even when host replay metadata is absent.

`python_env/bin/python -m pytest --noconftest -q
 tests/ttnn/tracy/test_process_ops_logs.py` passes **10 tests**, with one
pre-existing skipped test unchanged. Black formatting and `git diff --check`
pass. These are Python-only changes, requiring no C++ build. This investigation
never opened hardware and did not modify model/generator/collection behavior.

## Recovery of the original capture

`reprocess_tracy.py` invokes the repaired canonical parser against the unchanged
capture and verifies exact raw-to-report execution identities plus firmware/kernel durations and firmware start/end cycles. It hashes every raw input before processing
and verifies those hashes afterward. It also preserves both signposts and counts
the measured interval's per-device/trace/session rows.

The initial successful CPU reprocess is `logs/tracy_decode_reprocess_fixed.log`.
The final immutable recovery command/source/provenance is
`logs/tracy_decode_reprocess_verified_v2.provenance.json`; the final coverage
verification is `logs/tracy_decode_coverage_verified_v3.provenance.json`, its
artifact is `tracy_decode_coverage_final.json`, and its report is
`tracy/raw/decode/reports/decode_final_verified/ops_perf_results_decode_final_verified.csv`.
`record_run.py` now includes the changed parser and regression source in future
collection/reprocessing snapshots. Final verification proves **5772/5772** exact execution identities and all
exported firmware/kernel duration fields plus firmware start/end cycles.
Between signposts, exactly **2416** device measurements remain: four replays of
119 model operations plus 32 sampling operations on each of four devices.
Raw hashes are unchanged. No new device collection was needed.

Two verifier-only corrections remain visible in their immutable logs: its first
version assumed kernel start/end columns that the canonical output does not
export; the second correctly verified execution coverage but used HOST START TS
for interval filtering. Trace replay rows retain capture-time HOST START TS,
whereas canonical row order is sorted by replay timestamps. The final verifier
uses that signpost row-order contract, matching `tt-perf-report --tracing`, and
requires a nonempty measured interval. These corrections never change raw
measurements or the successfully generated report.

## Roofline display classification

`tracy/head_roofline_classification.json` explains the installed report's
117.96% FLOPs display for a 658.481-us BF16/HiFi4 head chunk. The external report
hardcodes eight Blackhole DRAM-sharded compute workers, but the recorded config
has two workers per each of eight banks. Native factory runtime arguments enable
compute on those 16 workers and return immediately on nonworkers. The 32 input
storage cores and the raw enclosing-grid count 110 are different quantities.
The actual measured 13.045 TF/s divided by the correct modeled 16-worker HiFi4
peak is 58.98%, not physically above 100%. Raw timings and rendered values remain
unchanged; this is a reporting-model limitation, not a runtime defect.

The chunk reads 268435456 weight bytes in 658.481 us: 407.659 GB/s, or 79.62% of
the tool's 512-GB/s Blackhole memory roofline. Its ideal weight-read floor is
524.288 us versus a 388.361-us modeled HiFi4 compute floor. This supports a
memory-bound classification and does not mandate lowering fidelity. BF16 LoFi
and HiFi2 were not measured; no actual speed/quality equivalence or inability to
improve is claimed. The selected HiFi4 policy has direct qualitative evidence.

Final status: parser defect fixed with regression and complete raw measurement
coverage. Parent retains hardware ownership and runs final report rendering and
prefill collection.
