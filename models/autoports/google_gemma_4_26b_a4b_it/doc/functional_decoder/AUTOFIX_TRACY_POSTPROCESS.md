# AutoFix Report: Tracy post-processing

## Starting evidence

- The functional-decoder perf pytest passed, but Tracy post-processing failed in
  `tools/tracy/process_ops_logs.py::_enrich_ops_from_perf_csv` with:
  `Device data missing: Op 1145857 not present in cpp_device_perf_report.csv for device 1 (trace_id=None)`.
- Failed-capture inputs are under
  `generated/profiler/gemma4_functional_sliding/.logs/`.
- This investigation was read-only with respect to model/profiler code and did
  not run hardware.

## Hypothesis experiments

### Hypothesis: the C++/Python merge lost a valid device row

Experiment:

```bash
rg -n -m 10 '^1145857,' \
  generated/profiler/gemma4_functional_sliding/.logs/cpp_device_perf_report.csv
rg -n -m 10 ',1145857,' \
  generated/profiler/gemma4_functional_sliding/.logs/profile_log_device.csv
rg -n -m 10 '1145857' \
  generated/profiler/gemma4_functional_sliding/.logs/tracy_ops_data.csv
```

Result: op 1145857 exists only in the host Tracy metadata. It is absent from
both the C++ device summary and the raw device marker log. Legacy Python device
post-processing therefore cannot recover it.

Verdict: refuted. The join is not discarding a valid row, and relaxing its
assertion would silently undercount the measured layer.

### Hypothesis: finite per-RISC profiler buffers dropped device markers

Experiment: compare unique host op IDs with the C++ device report for the failed
capture and the subsequent captures under
`generated/profiler/gemma4_functional_{sliding_v2,full_v2}`.

```bash
python - <<'PY'
import csv
import re
from pathlib import Path

for name in ("gemma4_functional_sliding", "gemma4_functional_sliding_v2", "gemma4_functional_full_v2"):
    logs = Path("generated/profiler") / name / ".logs"
    host = {}
    with (logs / "tracy_ops_data.csv").open() as stream:
        for row in csv.DictReader(stream, delimiter=";", quotechar="`"):
            text = row["MessageName"]
            if "TT_DNN_DEVICE_OP" not in text and "TT_METAL_DEVICE_OP" not in text:
                continue
            match = re.search(r",\s*(\d+)\s*$", text) or re.search(
                r"global_call_count[\"']?\s*:\s*(\d+)", text
            )
            if match:
                host[int(match.group(1))] = text
    with (logs / "cpp_device_perf_report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    device_ids = {int(row["GLOBAL CALL COUNT"]) for row in rows}
    sessions = {}
    for row in rows:
        key = (row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])
        sessions[key] = sessions.get(key, 0) + 1
    print(name, len(host), len(device_ids), len(set(host) - device_ids), sessions)
PY
```

Result:

| Capture | Host op IDs | Device unique IDs | Missing IDs | Trace rows by replay session |
| --- | ---: | ---: | ---: | --- |
| `sliding` (failed merge) | 1294 | 1243 | 51 | 37, 31, 31, 31, 31, 30, 30, 28 |
| `sliding_v2` | 1294 | 1294 | 0 | 74, 74, 74 |
| `full_v2` | 1298 | 1298 | 0 | 76, 76, 76 |

The failed capture is missing 14 non-trace ops and 37 distinct trace ops. The
first missing IDs are sparse matmuls at 1145857 and 1148929; later, more op
kinds disappear, and the number of retained traced programs declines on each
replay. There are no device-only extra IDs. This is the signature of individual
RISC buffers filling at different times, not an ID offset or trace-ID mismatch.

The runtime's dropped-zone diagnostic in `tt_metal/impl/profiler/profiler.cpp`
explicitly says to decrease the number of profiled ops or call the device
profiler reader more often. `DeviceProfiler::readResults` reads the DRAM buffer
and resets its control buffers. The functional perf harness now calls
`ttnn.ReadDeviceProfiler(mesh_device)` between warmup, measured prefill, initial
trace replay, each trace warmup, and each measured replay. The complete v2
captures verify that intervention boundary. The prefill workload is unchanged
between the failed and v2 sliding captures, so its 14-to-0 missing-op change is
specifically evidence for the prefill drain.

Verdict: verified.

Fix: retain the functional perf harness drains. They are outside the timed
interval used for `decode_trace_host_ms`; device report signposts still bracket
only the decoder work.

Verification artifacts:

- `generated/profiler/gemma4_functional_sliding_v2/reports/2026_09_05_03_12_03/ops_perf_results_2026_09_05_03_12_03.csv`
- `generated/profiler/gemma4_functional_full_v2/reports/2026_09_05_03_12_37/ops_perf_results_2026_09_05_03_12_37.csv`

## Final status

Fixed by rerunning the profiler workload with periodic supported
`ttnn.ReadDeviceProfiler` drains. The original raw capture is not recoverable
because the missing device markers were never written to either raw device CSV.
No profiler merge source change is appropriate.

For a run that cannot add explicit drains, Tracy also exposes the supported
`--op-support-count` option (backed by
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`). A conservative alternative for this
1294-op layer plus repeated trace executions is `--op-support-count 2500`, but
that alternative was not hardware-verified in this investigation. The verified
path is the harness drain strategy represented by the v2 reports above.
