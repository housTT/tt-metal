# AutoFix: reader confirmation profiler

**Final status: fixed and verified.** The complete reader comparison contains 66 profiled and 66 unprofiled real-input cases, all passing PCC and exact eager/trace equality. Every production reader choice wins both device matmul time and unprofiled traced time at its common legal comparison geometry. No production runtime change was needed.

## Starting evidence

Independent diagnosis: `AUTODEBUG_reader_profiler.md`. The original `logs/profile_readers_linear_attention.log` reports 400 dropped-marker warnings, then a missing-device-operation assertion. Its 36 JSON rows contain 32 passes and four separate reader-3 factory errors. Original logs, source archives and failed rows remain intact. Compressed host metadata and device-perf CSVs for that failure are in `tracy/linear_attention/reader_original_failure/` with a hash manifest; full raw captures remain in the original ignored `raw/reader_confirmation` folder.

The hardware lane was explicitly granted after the continuation verifier closed all device processes. All following device commands ran serially on one Blackhole chip on physical P300c boards. No reset, hang, Watcher/profiler overlap, or device recovery occurred. The lane was explicitly released after the final unprofiled pair completed.

## Hypothesis experiments

### Profiler capacity: verified

The same microbenchmark workload, geometries and reader order were run with capacity 8000 and default capacity 1000. The larger capacity completed with zero dropped-marker warnings; the default reproduced 400 warnings and `Device data missing: Op 205825`. The test source hashes match. Runtime source changes between those two captures consist only of import ordering and formatting, as confirmed from their archived source diff.

Commands, from the checkout root (`D=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_decoder`):

```bash
python "$D/run_reader_profiles.py" --label reader_capacity8000 --kind linear_attention --op-support-count 8000 --no-account
python "$D/run_reader_profiles.py" --label reader_capacity1000_control --kind linear_attention --no-account
```

Exact argv, environment, source archives, start/end time and return status are in `logs/profile_reader_capacity{8000,1000_control}_linear_attention.provenance.json`; the companion `collector.json` and compressed collector source archives record the effective child profiler flags and collector implementation. The capacity8000 accounting deliberately has status `incomplete_diagnostic`: its original four unsupported geometry rows are retained rather than counted as valid comparisons.

### Periodic drains at default capacity: verified and kept

The independent drain-only control retained default capacity 1000, unchanged geometry and all replay windows:

```bash
python "$D/run_reader_profiles.py" --label reader_drain_control --kind linear_attention --drain --no-account
```

It completed with zero dropped-marker warnings and accounted all 32 supported rows. The same four reader-3 factory errors remained, separating collection correctness from geometry legality. The kept test-only repair calls `ttnn.ReadDeviceProfiler(mesh_device)` after synchronized windows, after the ending signpost, and at setup/candidate boundaries. Drains are outside timed intervals and trace capture. Normal unprofiled measurements do not call the drain API.

### Reader-3 output storage geometry: verified adaptation, performance rejection

The factory failure predicted by AutoDebug is real at cores8/block8/per_core_N16 for N4096 with three readers. Holding cores8/block8 fixed and using per_core_N18 for *all* reader counts makes the physical output storage legal:

```bash
python "$D/run_reader_profiles.py" --label reader_n18_probe --kind linear_attention --roles gdn_z_epilogue,gdn_out --per-core-n 18 --drain
```

All 12 cases passed, with exact eager/trace equality and minimum PCC0.9999113301. Device medians, microseconds:

| Role | Original N16 reader2 | Common N18 reader2 | Adapted N18 reader3 |
| --- | ---: | ---: | ---: |
| gdn_z_epilogue | 33.907 | 34.107 | 41.710 |
| gdn_out | 26.170 | 26.319 | 36.377 |

Thus the first API error was not used to dismiss reader3: a supported exact-shape adaptation was tested and lost. Production keeps reader2/per_core_N16. The full-attention o_proj shape uses the same common-N18 comparison and likewise confirms reader2. No writer assertion was removed and no C++ source was changed.

## Complete confirmation and accounting

```bash
python "$D/run_reader_profiles.py" --label reader_legal_confirmation --per-core-n '{"gdn_z_epilogue":18,"gdn_out":18,"o_proj":18}' --drain
python "$D/run_candidates.py" "$D/reader_legal_host_plan.json"
python "$D/summarize_readers.py"
```

The profiler command runs both kinds serially. The unprofiled plan has explicit policy/config/geometry controls; each candidate uses seven windows of 64 replays and reports the median of the final five. Reader order is 1/2/3/3/2/1, with two independently allocated/captured measurements per reader. The final test writes evidence before asserting all requested roles and all six rows per role passed. This prevents exploratory runtime errors from producing a false confirmation pass.

| Kind | Profiled cases | Unprofiled cases | Minimum real-input PCC | Exact eager/trace |
| --- | ---: | ---: | ---: | --- |
| Linear attention | 36 | 36 | 0.9997713770 | All cases |
| Full attention | 30 | 30 | 0.9998026584 | All cases |

All runtime weight rows are BFLOAT4_B/LoFi. The final reader winners are packedGDN/QKVG/gate/up=3 and Z/GDNout/o_proj/down=2. `reader_comparison_summary.md`, `.csv`, and `.json` show all 33 role/reader combinations, device matmul time, separately collected unprofiled host time, BFP4 physical bytes including tile headers and reader padding, logical bytes and percentage of the declared 512 GB/s single-chip bandwidth model. These are measured microbenchmark times, not whole-layer latency claims.

Exact accounting artifacts for each kind are under `tracy/<kind>/reader_legal_confirmation/`:

- `reader_accounting.json` and `.csv`: runtime dtype, fidelity, full attributes, logical shapes, actual bank/tile bytes, shard/program geometry, source hash, device/global-op/trace/replay identity and kernel source/hash.
- `reader_ops.csv.gz`: byte-preserving compressed joined operation CSV, including setup and each signposted replay.
- `cpp_device_perf_report.csv.gz` and `tracy_ops_data.csv.gz`: compressed device executions and host operation metadata.
- Full raw markers, host timing CSV and `.tracy` remain in `tracy/<kind>/raw/reader_legal_confirmation/`. Large raw captures are intentionally excluded from Git; compact derived evidence and the above compressed CSVs are included.

Accounting requires unique start/end signposts, one replay identity and one matmul per interval, exact row coverage, passing statuses, and replay core counts matching the maximum observed for that operation. `reader_capture_validation.json` confirms zero dropped-marker warnings and unique device execution keys: 2664/2664 linear and 2091/2091 full. It records raw-device CSV hashes. Failed capacity-control raw host/device metadata is separately archived under `tracy/linear_attention/reader_capacity1000_control/`.

The diagnostic `--retain-failed` accounting option labels its result `incomplete_diagnostic`, records every failed row and never counts it toward complete reader coverage. Final accounting uses the strict default.

## Changes and final checks

Only reader tests, reader tooling and documentation were changed by this repair. `test_projection_geometry.py` now supports the scoped per-core-N and drain controls, records actual physical geometry, avoids synthesizing a removed `gate_up` role, and enforces complete confirmation rows. `run_reader_profiles.py` uses fresh explicit labels and preserves collector provenance; `account_readers.py` validates coverage and execution identity. `summarize_readers.py` independently asserts that the production choices win device and unprofiled medians.

Host `py_compile`, Black formatting, accounting regeneration and summary assertions passed. A host-only invocation against the existing final capture confirmed the runner refuses to overwrite either raw captures or measurement JSON before any hardware access (`EXISTING_CAPTURE_REJECTED_BEFORE_HARDWARE`). No build was required for these Python/docs changes. The initial runner-only `Path.with_suffix` typo was corrected before any hardware process started; it produced no device observation. There are no unresolved collector or reader-selection findings.
