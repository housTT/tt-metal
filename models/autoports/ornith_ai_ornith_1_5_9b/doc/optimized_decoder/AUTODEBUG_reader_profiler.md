# AutoDebug: reader profiler capture

Date: 2026-09-05. Investigation was source-only in an isolated AutoDebug agent;
no device was listed, opened, reset, or used. No implementation or test was
changed. `R` below means `models/autoports/ornith_ai_ornith_1_5_9b` and `D` means
`R/doc/optimized_decoder`.

## Verdict

The failed capture contains proven device-marker loss. The missing operation
287745 is a successful-path sharded-to-interleaved conversion, and has no
markers in the raw device CSV. It is not a trace-ID join collision or one of
the caught reader-3 matmul failures. A capture with periodic profiler drains
is the smallest durable remedy to verify.

Separately, the selected 8-core geometry fails for reader 3 in `gdn_z_epilogue`
and `gdn_out`. The test records these runtime errors and still passes; therefore
neither pytest PASS nor successful report generation establishes a complete
reader comparison. The errors must remain visible and require an exact-shape
adaptation experiment before claiming reader-1/2/3 coverage.

## Starting evidence and reproduction

- Failing log: `D/logs/profile_readers_linear_attention.log`.
- Exact invocation, environment, source hashes and return code:
  `D/logs/profile_readers_linear_attention.provenance.json`.
- The source archive is
  `D/logs/profile_readers_linear_attention.sources.json.gz`; its archived
  `tests/test_projection_geometry.py` is the source of the failing run.
- Raw data directory:
  `D/tracy/linear_attention/raw/reader_confirmation/.logs`.
- Measurement rows: `D/readers_profile_linear_attention.json`.
- Source HEAD recorded by the run: `bc8f514f3000da7b24c4d2b289b0ea507674e999`.
- Run interval: 2026-09-05 02:10:00.339326 through 02:10:35.201391 UTC.
- Hardware recorded by this stage: one Blackhole chip on physical P300c
  boards. Device ID in this capture is 1; device frequency header is 1350 MHz.

The recorded command is, with `$PY` standing for the recorded absolute
`python_env/bin/python` and `$R` expanded to the absolute model path:

```bash
$PY -m tracy -r -p -v --check-exit-code --web-app-port 18940 \
  -o "$R/doc/optimized_decoder/tracy/linear_attention/raw/reader_confirmation" \
  -m pytest "$R/tests/test_projection_geometry.py" -k linear_attention -v -s
```

The recorded policy is BFLOAT4_B weights and LoFi attention/MLP. The run used
`ORNITH_READER_CONFIRM=1`, `ORNITH_READER_PROFILE=1`, and explicit six-role
selection. See the provenance JSON for its complete serialized config.

The current test/candidate files changed during the parent agent's independent
runtime promotion and formatting. Their current bytes differ from the archive;
this report does not attribute those changes to the failed run. The log and
source-archive hashes both match the original provenance record.

## Finding 1: device profiler capacity exhausted

**Evidence.** Log lines 6218–6617 contain 400 warnings explicitly stating that
profiler DRAM buffers were full and markers were dropped, across 80 worker
cores and five RISC processors. Every warning reports `bufferEndIndex = 12000`.
At 02:10:34 the Python report join fails at
`tools/tracy/process_ops_logs.py:683`:

```text
Device data missing: Op 287745 not present in cpp_device_perf_report.csv
for device 1 (trace_id=None)
```

Host event 287745 is `ShardedToInterleavedDeviceOperation` at
21,599,149,950 ns. It follows matmul 286721 with
`num_workers_per_dram_bank=1`, after signpost
`READER_L0_gdn_z_epilogue_R2_I4_END` and before trace 9 begins. It belongs to
the eager correctness path of `gdn_z_epilogue` reader 1, iteration 5.

A direct CSV audit found:

| Check | Observed |
| --- | ---: |
| Host device-op IDs | 390 |
| Device perf rows | 1,929 |
| Unique device `(op ID, trace ID, replay session)` keys | 1,929 |
| Unique device op IDs | 347 |
| Host op IDs absent from device perf | 43 |
| Host trace IDs | 0–31 |
| Host replay events per trace | 29 |
| Device trace IDs | 0–30 |
| Raw markers for missing ops 287745, 288769, 290817, 397313, 399361 | 0 |
| Raw markers for neighboring surviving matmul 289793 | 41,760 |

The missing operations initially cluster on layout conversions; the final
reader-1 `down_proj` eager matmul and entire trace 31 are also missing. Several
earlier matmuls survive while their conversion operations do not. Per-core
buffer exhaustion accounts for this selective loss: different programs use
different cores, which reach their limits at different points.

**Source mechanism.** The archived test executes seven windows of four replays
and one signposted replay for each successful candidate, but never calls
`ReadDeviceProfiler`. It also profiles decoder setup, prefill, activation
collection, reference calculations, eager calls, and trace captures. There
are 32 successful traces × 29 replays = 928 replays, with multiple operations
inside most traces. A four-replay window is not a four-operation capture.

`tt_metal/impl/profiler/profiler_state_manager.cpp:21` defaults to 1,000 programs
of storage per RISC. Its lines 42–82 size the backing allocation from
`profiler_program_support_count`. The warning at
`tt_metal/impl/profiler/profiler.cpp:1684` tests the firmware `DROPPED_ZONES` bit;
this is an explicit loss signal, not a timing heuristic.

**Focused experiment, preferred durable change.** Add conditional profiler
drains only when `ORNITH_READER_PROFILE=1`: after prefill/setup and activation
collection, at candidate boundaries, and after each synchronized replay
window. Take the host timing sample before the drain. Drain before the
signpost start; execute its single blocking replay; emit its end signpost;
then drain. Do not drain during trace capture or inside a measured interval.
This changes collection cadence while retaining reader order, geometry,
inputs, correctness checks and replay counts.

The exact Python API is `ttnn.ReadDeviceProfiler(mesh_device)`:

- `ttnn/ttnn/device.py:140` forwards to the binding.
- `ttnn/cpp/ttnn-nanobind/device.cpp:629` accepts `MeshDevice*` and invokes
  `ReadMeshDeviceProfilerResults(..., ProfilerReadState::NORMAL, ...)`.
- `tt_metal/impl/profiler/tt_metal_profiler.cpp:1180` implements that function;
  it finishes each command queue before reading device results, processes
  them and waits for the processing thread pool.

**Independent capacity control.** An unchanged-test rerun with the single
additional Tracy argument `--op-support-count 8000` is a useful verify/refute
control. `tools/tracy/__main__.py:358` maps it to
`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT`, parsed at
`tt_metal/llrt/rtoptions.cpp:1103`. It increases profiler DRAM allocation, so
record that change and do not use its profiled host times as production timing.
Do not combine this change with drains in the first experiment: one variable
at a time makes the result interpretable. Capacity is not a universal guarantee
for future larger captures.

**Verification required.** No dropped-marker warnings; no host/device join
assertion; complete signpost replay tuples; no duplicate device keys; all
expected operations and full runtime core counts in each measured replay.
Compare observed per-replay core counts against that operation's eager/capture
metadata, since partial-core loss can yield a duration row without a complete
measurement. After a valid small capture, run both layer kinds.

## Finding 2: reader 3 hits a real output-storage assignment failure

The JSON contains 36 rows: 32 passes and four `runtime_error` rows. Reader 3
fails twice each for `gdn_z_epilogue` and `gdn_out`, both with
`[M,K,N]=[1,4096,4096]`, cores 8, block width 8. Each error says:

```text
matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:814:
curr_storage_core_idx < num_cores_written_back
Worker 6-3 has no storage area assigned
```

This is independent of profiler collection. The same geometry failures are
recorded in `D/geometry_layer0.json`. The failing factory constructs reader
assignments before enqueuing a matmul, and its padding calculations explain
the exact passing/failing boundary:

| Quantity | Reader 1 | Reader 2 | Reader 3 |
| --- | ---: | ---: | ---: |
| Logical output tiles | 128 | 128 | 128 |
| Tiles per DRAM bank | 16 | 16 | 18 |
| Total physical output-width tiles | 128 | 128 | 144 |
| Workers across eight banks | 8 | 16 | 24 |
| Tiles per worker | 16 | 8 | 6 |
| Output storage tiles per core | 16 | 16 | 16 |
| Output storage cores | 8 | 8 | 8 |
| Output capacity tiles | 128 | 128 | 128 |

The factory's lines 126–144 remove only wholly padded *banks*. Reader 3 has
16 padding tiles, less than its 18-tile bank, so none are removed. Lines
159–160 derive six tiles per worker. Lines 761–762 size storage from logical
N. In the branch `per_core_N_in1_sender < per_core_N_storage`, lines 813–848
advance storage for every reader, including readers whose entire chunk is
padding. At zero-based worker 22, prior work accounts for 132 tiles, making
`curr_storage_core_idx=8`, outside the eight output storage cores. Reader 1/2
do not exceed 128 tiles. This arithmetic was checked in a host-only Python
simulation of these exact index updates.

**Focused adaptation experiment A.** Preserve cores 8 and block width 8, but
allow output `per_core_N=18` for all three readers. This creates eight storage
cores with capacity 144 and removes the exact bound violation in the source
simulation. Hold this output geometry fixed across the entire 1/2/3/3/2/1
comparison. This is a hypothesis, not a verified fix: device validation must
check logical/padded output shape, output allocation, PCC versus the same
reference, exact eager/trace agreement, and padding-neutral consumption by
the subsequent operation. The current DRAM-sharded validator does not require
`per_core_N=ceil(N/input_cores)`, but that alone does not prove correctness.

**Existing supported-geometry control B.** Use cores 32/block width 4/output
`per_core_N=4` for all readers of these two roles. The writer then takes the
other branch, whose lines 852 onward explicitly guard exhausted output
storage. `geometry_layer0.json` already records reader-3 passes at that common
geometry with exact eager/trace equality and PCC 0.9999137028
(`gdn_z_epilogue`) / 0.9999970036 (`gdn_out`). These are prior non-profiled
controls; rerun and profile them with current sources. Do not mix a 32-core
reader-3 result with 8-core reader-1/2 results and describe it as a fixed
geometry reader comparison.

Do not remove the factory assertion or silently skip these reader-3 rows. A
C++ writer fix would require its own exact-shape correctness tests and build;
it is outside this source-only task and unnecessary for first testing the
model-local adaptations above.

## Finding 3: accounting and provenance must reject incomplete evidence

- The archived `test_projection_geometry.py:125` computes row status and
  catches runtime errors at line 126, then writes rows without asserting all
  requested rows passed. This behavior is useful for exploratory sweeps but
  insufficient for a confirmation gate. A confirmation must assert the
  expected roles/order, successful row counts and row statuses after saving
  evidence, or explicitly fail with the unsupported configurations named.
- `account_readers.py` unconditionally reads `measured["profile_signpost"]`.
  The four runtime-error rows have no such field, so fixing collection alone
  reveals a second accounting failure. Validate row statuses and coverage
  before indexing signposts; retain failed rows separately with their exact
  errors and do not count them toward successful comparison coverage.
- Before computing metrics require one start and end per label, exactly one
  replay event inside, exactly one matching matmul execution and complete
  device/core coverage. Record device ID, original global op ID, trace ID,
  replay session ID and kernel paths/hashes alongside runtime attributes and
  dtypes. Sum only the operations belonging to that precise replay.
- The account script hard-codes eight banks and 576 bytes per BF4 tile. Those
  match this capture's policy/hardware, but record actual bank count and tile
  bytes in measurements and verify them before calculating physical bytes.
- `record_run.py` archives model `tt/reference/tests` Python sources but not
  the `doc` runner/accounting scripts or Tracy/profiler source, and omits the
  program-support-count environment setting. Preserve hashes/snapshots of
  those scripts and effective profiler flags for the repaired runs. The
  failing provenance has `TT_METAL_DEVICE_PROFILER=null` because it snapshots
  the wrapper's environment; Tracy sets it to `1` for its child at
  `tools/tracy/__main__.py:422`. Record both wrapper and effective settings.
- Use a fresh record name, output JSON and Tracy output directory for every
  retry. `record_run.py` correctly rejects an existing record name, but
  `tools/tracy/__init__.py:101` deletes the selected `.logs` directory before
  capture. Reusing the original `-o` destroys raw provenance. Archive/compress
  raw host/device CSVs and the `.tracy` file even on failure. Most raw CSVs
  are git-ignored; use `rg --files --hidden --no-ignore` to audit retention.

## Refuted or secondary hypotheses

- **Trace ID/session collision as cause of this assertion: refuted.** Perf
  keys are unique; the failing non-trace operation has no raw device markers.
  `_enrich_ops_from_perf_csv` also falls back to lookup by op ID across trace
  IDs before asserting. A different trace label cannot recover absent data.
- **A failed reader-3 matmul emitted op 287745: refuted.** It is the successful
  reader-1 output conversion identified above. The API failures are real but
  separate.
- **Legacy parser fallback would recover this capture: refuted for the
  specified missing operations.** Their raw device markers are absent too.
  Do not bypass the join assertion to manufacture a full report.
- **Tracy web UI copy warning causes the device gap: unsupported.** The copy
  targets the default generated path while this run uses custom `-o`; host
  trace and CSV exports exist in that output directory. This ancillary warning
  does not explain the missing device markers. A retry should avoid leaving
  another task-owned GUI server running, following the existing hardware lane
  and process ownership rules.

## Checks performed and retained hashes

Read the raw CSVs with Python standard-library `csv`/`json`, reconstructed host
op IDs from cached/uncached metadata, counted unique device execution keys and
trace/replay events, streamed the device CSV for exact `run host ID` matches,
checked source-archive/log hashes, and simulated the writer index arithmetic.
These are host-only checks. No build was needed for this documentation-only
deliverable. No repaired hardware run is claimed.

| Artifact relative to D unless stated | SHA-256 |
| --- | --- |
| `logs/profile_readers_linear_attention.log` | `f0dbd6366372bc416997d4319247ee767928dd744ecbb5a61f29fe9b52b3e042` |
| `logs/profile_readers_linear_attention.sources.json.gz` | `45de0e388e580365a961705a98cb89f830dbbbff075026581628cf72abae0194` |
| `readers_profile_linear_attention.json` | `8ac3965d887b101248305c8c9ecad6ebb3ec012912ffc272caee0553f8bebac5` |
| `geometry_layer0.json` | `54e4f7d63f8b95705e805c9c5f5320769772425f9370685ef3d67fa5fc599484` |
| Raw `cpp_device_perf_report.csv` | `c1b38783099eb496da0ad80c4e0f7ed0428ffff4bac8d108e87cf1bd60a1b97c` |
| Raw `tracy_ops_data.csv` | `c7790ea228512a6175c99dcf5c9a36c83aaf92761914eaf3abb94d2e7a565993` |
| Raw `profile_log_device.csv` | `554ec8ef386aec287a50e142c242959e9a81c7656ed0cf4b107f00c6cb671f52` |
| Raw `tracy_ops_times.csv` | `74d821fe458929b1c9eabc1bf92698340afb144aa1a60987b462aa4a862d2459` |
| Raw `tracy_profile_log_host.tracy` | `e282d8322a19ff01fbb151b7271f9b0a9cfa77fc031a3743ff844fc3b36834b9` |
| Repo `tools/tracy/process_ops_logs.py` | `8f644fd6bc33a04ef9046d57aafadb46f0c5b2d93950c3d8285e62324991eb0d` |

Final status: diagnosis complete; collector repair, reader-3 adaptation,
strict accounting and actual-device confirmation remain to be verified by
the parent agent in the serialized hardware lane.
