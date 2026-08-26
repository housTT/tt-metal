# AutoDebug Report: `open_mesh_device` dispatch JIT compile failure

Date: 2026-08-26 (functional-decoder environment investigation)

## Verdict

The leading cause is verified TT-Metal provenance skew, not model code and not a
dispatch implementation bug in this checkout.

`/home/ttuser/.tenstorrent-venv/bin/python` imports editable TTNN 0.65.1 and
loads `_ttnn.so`, `_ttnncpp.so`, and `libtt_metal.so` from the July checkout at
`/home/ttuser/.local/lib/model-bringup/tt-metal`. Because the command is run
from `/home/ttuser/dev/qwen3.8-flash-next/tt-metal`, runtime kernel resolution
selects dispatch JIT source from this August checkout (TTNN 0.75.0). The stale
host-side dispatch builder and the current device source/header contract are
incompatible.

This explains why the failure occurs during `open_mesh_device`, before any
model code. No implementation edit is indicated. The minimal recovery is to
run with this checkout's existing `python_env` and build artifacts, with the
runtime root and Python path pinned to this checkout. A hardware run is still
needed to prove recovery; AutoDebug remained source-only.

## Direct observations

- The current checkout was clean before this report was created:
  `hous/qwen3.8-flash-next`, commit
  `20e418fb9277d9921f1ca33577e01c2800b3d3ce`.
- The stale checkout is commit
  `559921b40a8b7b21c807d5592323c2fa5e8c7ecb`. Its local modifications do not
  include the dispatch files examined here.
- The active venv contains an explicit stale editable install:
  - `ttnn-custom.pth` adds
    `/home/ttuser/.local/lib/model-bringup/tt-metal`, its `ttnn` directory, and
    its `tools` directory.
  - `ttnn-0.65.1rc17.dev7396.dist-info/direct_url.json` points to that checkout.
  - `__editable___ttnn_0_65_1rc17_dev7396_finder.py` maps `ttnn` to that
    checkout.
- Importing with the active venv loads, as confirmed through `/proc/self/maps`:
  - `/home/ttuser/.local/lib/model-bringup/tt-metal/ttnn/ttnn/_ttnn.so`
  - `/home/ttuser/.local/lib/model-bringup/tt-metal/build_Release/ttnn/_ttnncpp.so`
  - `/home/ttuser/.local/lib/model-bringup/tt-metal/build_Release/tt_metal/libtt_metal.so`
- `PYTHONPATH=$PWD $PWD/python_env/bin/python` instead loads all three from this
  checkout and reports editable TTNN `0.75.0rc10.dev880`.
- The native artifacts are not interchangeable:
  - stale `_ttnncpp.so`: 37,720,272 bytes, SHA-256
    `07b878cc47d913a4f071462a222b209d8c8f832f5d52ac365745f316ba9d6737`
  - current `_ttnncpp.so`: 53,369,984 bytes, SHA-256
    `91f3e4bda35173ff48b87439bda34935069fe530e96101b189994487de5638c4`
  - stale `_ttnn.so`: 18,440,984 bytes
  - current `_ttnn.so`: 19,183,576 bytes
- This checkout already has `build_Release`, `python_env`, its local TTNN native
  extension, and the requested p150 mesh descriptor. Rebuilding is not the
  first recovery step.
- `RunTimeOptions` in `tt_metal/llrt/rtoptions.cpp` uses
  `TT_METAL_RUNTIME_ROOT` when provided, otherwise accepts the current working
  directory when it contains `tt_metal/`. `tt_metal/impl/kernels/kernel.cpp`
  resolves relative kernel sources against the current working directory
  before other roots. This accounts for the observed August JIT source with a
  July loaded runtime.

## Compile-error contract match

The July and August dispatch trees differ in precisely the contracts named in
the failure:

| Reported error | July contract | August contract |
| --- | --- | --- |
| missing `COMPLETION_COUNTER_OFFSET` | `cq_dispatch.cpp` does not consume it | current source consumes it; current `kernel_config/dispatch*.cpp` supplies it |
| missing `dispatch_telemetry_types` | telemetry types are directly under `tt::tt_metal` | current source uses nested `tt::tt_metal::dispatch_telemetry_types` |
| missing `programmable_core_type` | `cq_common.hpp` defines `fd_core_type` from `FD_CORE_TYPE` | current header defines `programmable_core_type` from `PROGRAMMABLE_CORE_TYPE` |
| missing `CQ_DISPATCH_CMD_RT_PROFILER_FLUSH` | absent from July `cq_commands.hpp` | present in current `cq_commands.hpp` |
| `cq_noc_async_write_init_state` arity | July helper has four runtime parameters | current helper adds `ndests`, and current dispatch has five-argument calls |

This is not a generic ABI suspicion: current dispatch source combined with old
host-generated defines or old headers predicts the exact diagnostic cluster.

## Ranked hypotheses

### 1. Verified: split-brain editable TTNN/runtime and JIT source

Evidence: explicit stale `.pth`/editable metadata, mapped stale shared objects,
different TTNN versions and native hashes, current-working-directory kernel
resolution, and the exact dispatch contract diff above.

Prediction: switching as one unit to this checkout's `python_env`, native
libraries, source tree, and runtime root eliminates all five compile errors.

### 2. Secondary only: an old JIT artifact remains after provenance is aligned

This is not needed to explain the initial failure. If a correctly aligned
process still emits the old diagnostic, use a new `TT_METAL_CACHE` directory as
a controlled test. Do not delete shared caches first; an isolated cache gives a
cleaner verify/refute experiment.

### 3. Low confidence after alignment: a real current-checkout dispatch build bug

The current source and its current host builder contain matching sides of the
new contracts. This hypothesis becomes credible only if the repo-local process
fails in a fresh cache and its `/proc/self/maps` contains no `model-bringup`
paths. At that point capture the full compiler command and first diagnostics.

### Refuted for the present symptom: model code

`open_mesh_device` initializes and JIT-compiles dispatch infrastructure before
model construction. Model logic cannot account for these missing dispatch
identifiers.

## Minimal recovery

Use the already-built, repo-local environment. The sibling-stage launch recipe
is the correct leading fix. `TT_METAL_HOME` and `PYTHONPATH` preserve the model
and import convention; adding `TT_METAL_RUNTIME_ROOT` explicitly pins the JIT
root instead of relying on the current-working-directory fallback.

```bash
TT_REPO=/home/ttuser/dev/qwen3.8-flash-next/tt-metal
cd "$TT_REPO"
source "$TT_REPO/python_env/bin/activate"

export TT_METAL_HOME="$TT_REPO"
export TT_METAL_RUNTIME_ROOT="$TT_REPO"
export PYTHONPATH="$TT_REPO"
export TT_VISIBLE_DEVICES=0
export TT_MESH_GRAPH_DESC_PATH="$TT_REPO/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"
unset TT_METAL_KERNEL_PATH
hash -r
```

`TT_VISIBLE_DEVICES` and the p150 descriptor retain the known-good target
topology. They are not the explanation for the compiler errors.

Before touching hardware, prove process provenance:

```bash
python - <<'PY'
import importlib.metadata
import pathlib
import sys
import ttnn

repo = pathlib.Path('/home/ttuser/dev/qwen3.8-flash-next/tt-metal')
print('python:', sys.executable)
print('ttnn version:', importlib.metadata.version('ttnn'))
print('ttnn:', ttnn.__file__)

mapped = sorted({
    line.rstrip().split()[-1]
    for line in open('/proc/self/maps')
    if '_ttnn' in line or 'libtt_metal' in line
})
print('\n'.join(mapped))

assert pathlib.Path(sys.executable).is_relative_to(repo)
assert pathlib.Path(ttnn.__file__).is_relative_to(repo)
assert mapped and all(str(repo) in path for path in mapped), mapped
assert not any('model-bringup' in path for path in mapped), mapped
PY
```

Then rerun only the original open/close boundary (not the model):

```bash
timeout 60 python - <<'PY'
import ttnn

mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, 1),
    trace_region_size=0,
    physical_device_ids=[0],
)
ttnn.close_mesh_device(mesh)
print('open/close passed')
PY
```

Expected result: no ncrisc dispatch compiler errors and `open/close passed`.

## Focused verify/refute controls

### Confirm the original contamination without opening a device

```bash
/home/ttuser/.tenstorrent-venv/bin/python - <<'PY'
import importlib.metadata
import sys
import ttnn

print(importlib.metadata.version('ttnn'))
print(ttnn.__file__)
for line in open('/proc/self/maps'):
    if '_ttnn' in line or 'libtt_metal' in line:
        print(line.rstrip().split()[-1])
PY
```

Expected: version `0.65.1rc17.dev7396` and `model-bringup` paths.

### Isolate JIT cache only if the aligned run still fails

```bash
TT_JIT_PROBE=$(mktemp -d /tmp/tt-metal-qwen-jit.XXXXXX)
export TT_METAL_CACHE="$TT_JIT_PROBE"
echo "isolated JIT cache: $TT_JIT_PROBE"
```

Rerun the same open/close command. A pass only with this setting verifies a
secondary stale-cache issue. The temporary directory is intentionally retained
for inspection; no existing cache is removed.

### Escalation boundary

If the aligned, isolated-cache run still fails, capture:

```bash
python - <<'PY'
import sys
import ttnn
print(sys.executable)
print(ttnn.__file__)
for line in open('/proc/self/maps'):
    if '_ttnn' in line or 'libtt_metal' in line:
        print(line.rstrip().split()[-1])
PY
env | sort | rg '^(PYTHONPATH|LD_LIBRARY_PATH|TT_METAL|TT_VISIBLE|TT_MESH)='
git status --short --branch
git rev-parse HEAD
```

Also retain the full JIT compiler invocation and its first error. Only then
consider rebuilding this checkout or investigating current dispatch source.

## Investigation note

The required fresh-context `.agents/scripts/autodebug.sh` run was attempted.
Its isolated Codex process could not start local shell commands or write the
report because bubblewrap failed with
`loopback: Failed RTM_NEWADDR: Operation not permitted`. It returned the same
provenance-skew ranking based on independent dispatch inspection. All headline
claims above were then checked directly against the local checkout, installed
metadata, mapped libraries, git state, and July/August source diffs before this
report was written. No implementation or environment changes were made.
