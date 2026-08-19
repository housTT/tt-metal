# AutoDebug: Tracy launcher cannot find tools from `/tmp`

Date: 2026-08-19

## Headline finding

The reported error has a two-part configuration cause, not a TT-device cause.

1. `tracy.common` sets `TT_METAL_HOME` to the current working directory when the
   environment variable is absent, then derives the CLI directory as
   `$TT_METAL_HOME/build/tools/profiler/bin`. Launching from `/tmp` therefore
   probes `/tmp/build/tools/profiler/bin` and emits `Tracy tools were not found`.
2. The ordinary installed runtime at
   `/home/ttuser/.local/lib/model-bringup/tt-metal` contains `tracy-capture` and
   `tracy-csvexport`, but both of its CMake caches record
   `ENABLE_TRACY:BOOL=OFF`. Pointing only at those CLI tools fixes discovery but
   cannot produce operation profiling data from that `_ttnn.so`.

There is an already-installed profiler runtime at
`/home/ttuser/.local/lib/model-bringup/tt-metal-profiler`. Its CMake caches
record `ENABLE_TRACY:BOOL=ON`, it contains the required CLI tools, and a
non-device report-generation smoke produced a valid Tracy capture and raw CSVs.

## Source-level causal chain

- `tools/tracy/common.py:13` uses
  `Path(os.environ.get("TT_METAL_HOME", Path.cwd()))`.
- `tools/tracy/common.py:26` derives `PROFILER_BIN_DIR` beneath that root.
- `tools/tracy/common.py:45-50` accepts a tool only when the exact file exists.
- `tools/tracy/__init__.py:90-120` requires both `tracy-capture` and
  `tracy-csvexport`; otherwise it emits the reported message and exits before
  starting pytest.
- `tools/tracy/__main__.py:162-164,194-200` exposes
  `--tracy-tools-folder` as an explicit override.
- TTNN op instrumentation is compiled behind `TRACY_ENABLE`; a CLI executable
  alone cannot enable instrumentation in an already-built `_ttnn.so`.

## Direct observations

The following checks were run without opening a TT device:

```text
/home/ttuser/.local/lib/model-bringup/tt-metal/build/CMakeCache.txt:
ENABLE_TRACY:BOOL=OFF

/home/ttuser/.local/lib/model-bringup/tt-metal/build_Release/CMakeCache.txt:
ENABLE_TRACY:BOOL=OFF

/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/CMakeCache.txt:
ENABLE_TRACY:BOOL=ON

/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build_Release/CMakeCache.txt:
ENABLE_TRACY:BOOL=ON
```

Both installed roots contain:

```text
build/tools/profiler/bin/tracy-capture
build/tools/profiler/bin/tracy-capture-daemon
build/tools/profiler/bin/tracy-csvexport
```

With the ordinary `ENABLE_TRACY=OFF` `_ttnn.so`, explicitly selecting its CLI
directory advanced past tool verification, but the launcher exited with:

```text
No profiling data could be captured. Please make sure you are on a Tracy-enabled build (default).
```

With the `ENABLE_TRACY=ON` profiler install, this non-device command completed
with exit code 0:

```bash
cd /tmp
env \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -r --no-device --check-exit-code \
    --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
    -o /tmp/qwen36_tracy_profiler_toolcheck \
    -m pytest --version
```

Verified outputs:

```text
/tmp/qwen36_tracy_profiler_toolcheck/.logs/tracy_profile_log_host.tracy  809187 bytes
/tmp/qwen36_tracy_profiler_toolcheck/.logs/tracy_ops_times.csv           3975908 bytes
/tmp/qwen36_tracy_profiler_toolcheck/.logs/tracy_ops_data.csv           670 bytes
```

The smoke deliberately ran no TTNN device operations, so it did not generate a
device-backed `ops_perf_results_*.csv`. Existing prior artifacts under the same
profiler install demonstrate that its normal device runs produce
`reports/<timestamp>/ops_perf_results_<timestamp>.csv` and
`profile_log_device.csv`.

## Verified command fix for the functional-decoder pytest

Run from `/tmp` so repository sources do not shadow the installed kernel source,
but explicitly select the profiler-enabled TTNN and Tracy installation before
the checkout on `PYTHONPATH`:

```bash
cd /tmp
env \
  TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
  TT_METAL_CACHE=/tmp/qwen36_functional_tracy_cache \
  PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
  python -m tracy -p -r -v --check-exit-code --device-trace-profiler \
    --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
    -o /tmp/qwen36_functional_tracy \
    -m pytest \
    /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_functional_decoder.py \
    -k '<performance test selector>' -s
```

The explicit `--tracy-tools-folder` is redundant when `TT_METAL_HOME` is set to
the profiler root, but retaining it makes tool provenance unambiguous in the
captured command. The profiler install and checkout are different Git revisions
(`162f0add...` and `d58cb341...`, respectively), so the actual target test must
still validate API/runtime compatibility and correctness. This investigation
did not use TT hardware and does not claim that device-side collection itself
was exercised.

## AutoDebug workflow note

The repo-local AutoDebug runner was invoked from this triage directory as
required. Its fresh Codex sandbox could not start shell commands or write its
own `AUTODEBUG.md`; its independent static conclusion correctly identified the
`TT_METAL_HOME`/working-directory discovery failure. The checks above then
adjudicated that partial finding and found the additional compile-time
instrumentation requirement before recommending a command.
