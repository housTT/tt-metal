# AutoDebug: mesh-device environment provenance failure

Date: 2026-08-27

## Verdict

The primary failure is environment provenance skew, not model code and not a
confirmed hardware fault. The failing plain `python` process resolves `ttnn` to
the installed/model-bringup tree at:

```text
/home/ttuser/.local/lib/model-bringup/tt-metal/ttnn/ttnn/__init__.py
```

while the intended smoke was meant to use this checkout:

```text
/home/ttuser/dev/qwen3.8-flash-next/tt-metal
```

That split explains a JIT compile failure before model code: the loaded native
TTNN/tt-metal libraries and the dispatch JIT source/header tree are from
different TT-Metal revisions. The safest minimal fix is to source the repo-local
environment script, prove `ttnn` provenance without opening a device, then retry
the intended bounded mesh smoke only from that same shell.

## Direct observations

- Current unsourced `python` in this workspace is
  `/home/ttuser/.tenstorrent-venv/bin/python`.
- With no TT environment variables set, `importlib.util.find_spec("ttnn")`
  resolves to `/home/ttuser/.local/lib/model-bringup/tt-metal/ttnn/ttnn`.
- The active `.tenstorrent-venv` contains stale TTNN path injectors:
  - `__editable__.ttnn-0.65.1rc17.dev7396.pth`
  - `ttnn-custom.pth`, adding `/home/ttuser/.local/lib/model-bringup/tt-metal`,
    its `ttnn`, and its `tools` directories.
- Sourcing
  `models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh`
  switches Python to
  `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/python_env/bin/python` and makes
  `ttnn` resolve to
  `/home/ttuser/dev/qwen3.8-flash-next/tt-metal/ttnn/ttnn/__init__.py`.
- `ttenv.sh` also sets `TT_METAL_HOME`, `TT_METAL_RUNTIME_ROOT`,
  `PYTHONPATH`, `TT_VISIBLE_DEVICES=0` by default, and the p150 mesh descriptor;
  it unsets `TT_METAL_KERNEL_PATH`.
- Current `RunTimeOptions` uses `TT_METAL_RUNTIME_ROOT` as the explicit runtime
  root override, with current-working-directory fallback only when the current
  directory contains `tt_metal/`.
- Current JIT build setup derives include roots from `rtoptions.get_root_dir()`,
  including `root_/ttnn`, `root_/ttnn/cpp`, `root_/tt_metal`,
  `root_/tt_metal/hw/inc`, and `root_/tt_metal/hostdevcommon/api`.
- Current dispatch source/header/config are internally consistent:
  - `cq_dispatch_subordinate.cpp` consumes `COMPLETION_COUNTER_OFFSET`.
  - `dispatch_s.cpp` supplies `COMPLETION_COUNTER_OFFSET` as a JIT define.
  - current `cq_common.hpp` declares `cq_noc_async_write_init_state(...,
    ndests, noc)`, matching current five-argument dispatch calls.
- The optimized-decoder work log already labels a prior "global-environment"
  launch as a mixed installed-TTNN/checkout failure. The exact compile-skew
  diagnostics are visible in
  `doc/optimized_decoder/autofix_dram_sharded_exact_final_hifi2_qsa_input_attn_out_run1.xml`.
- The named
  `doc/optimized_decoder/autofix_dram_sharded_exact_final_hifi2_infra_mesh_failure.xml`
  is a distinct open-mesh topology/descriptor failure under repo-local paths
  (`Physical chip id 0 not found...`), so it should not be conflated with the
  dispatch compile-skew error.

## Ranked hypotheses

### 1. Provenance-skewed Python/native/JIT environment

This is the headline hypothesis. It predicts simultaneous missing or mismatched
dispatch contracts, such as undeclared `COMPLETION_COUNTER_OFFSET`, missing
`dispatch_telemetry_types`, and stale `cq_noc_async_write_init_state` arity.
Those are exactly the reported errors and exactly the kinds of errors expected
when old installed headers/libraries compile current checkout dispatch sources.

### 2. Secondary stale JIT cache after provenance is fixed

This is plausible only after the process provenance is clean. A fresh
`TT_METAL_CACHE` should be used as a non-destructive control before deleting or
rebuilding anything.

### 3. Mesh descriptor/device mapping issue

This is real in the prior `infra_mesh_failure.xml`, but it predicts
`Physical chip id 0 not found...`, not C++ compile errors. It becomes relevant
after the import/native/JIT provenance check passes and the command reaches
actual mesh initialization.

### 4. Current checkout dispatch source bug

Low confidence. Current source, headers, and kernel config match each other for
the reported identifiers. Promote this only if a repo-local process with fresh
cache still emits the same compile errors and `/proc/self/maps` contains no
`model-bringup` paths.

## Focused verify/refute commands

### A. No-hardware provenance probe

Run this first. It imports no TTNN package code and opens no device.

```bash
python - <<'PY'
import importlib.util, os, site, sys
print("python =", sys.executable)
for key in ["TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "PYTHONPATH", "TT_METAL_KERNEL_PATH", "TT_MESH_GRAPH_DESC_PATH", "TT_VISIBLE_DEVICES"]:
    print(f"{key}={os.environ.get(key)}")
spec = importlib.util.find_spec("ttnn")
print("ttnn origin =", None if spec is None else spec.origin)
print("ttnn locations =", None if spec is None else list(spec.submodule_search_locations or []))
print("site packages =", site.getsitepackages())
print("sys.path[:12] =", sys.path[:12])
PY
```

Refutes the headline if `ttnn origin` is already under the qwen checkout in the
failing shell. Confirms it if `ttnn origin` is under
`/home/ttuser/.local/lib/model-bringup/tt-metal`.

### B. Source the intended repo-local environment and re-probe

```bash
cd /home/ttuser/dev/qwen3.8-flash-next/tt-metal
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
python - <<'PY'
import importlib.util, os, sys
print("python =", sys.executable)
for key in ["TT_METAL_HOME", "TT_METAL_RUNTIME_ROOT", "PYTHONPATH", "TT_METAL_KERNEL_PATH", "TT_MESH_GRAPH_DESC_PATH", "TT_VISIBLE_DEVICES"]:
    print(f"{key}={os.environ.get(key)}")
spec = importlib.util.find_spec("ttnn")
print("ttnn origin =", None if spec is None else spec.origin)
print("ttnn locations =", None if spec is None else list(spec.submodule_search_locations or []))
print("sys.path[:12] =", sys.path[:12])
PY
```

Expected: Python is `.../tt-metal/python_env/bin/python`, `TT_METAL_HOME` and
`TT_METAL_RUNTIME_ROOT` both equal the qwen checkout, `TT_METAL_KERNEL_PATH` is
unset, and `ttnn origin` is
`.../tt-metal/ttnn/ttnn/__init__.py`.

### C. Import-level native provenance probe

This imports `ttnn` but still does not open a device.

```bash
python - <<'PY'
import pathlib, sys
import ttnn
repo = pathlib.Path("/home/ttuser/dev/qwen3.8-flash-next/tt-metal").resolve()
print("python =", sys.executable)
print("ttnn =", ttnn.__file__)
mapped = sorted({
    line.rstrip().split()[-1]
    for line in open("/proc/self/maps")
    if "_ttnn" in line or "libtt_metal" in line
})
for path in mapped:
    print(path)
bad = [path for path in mapped + [ttnn.__file__] if "model-bringup" in str(path)]
assert not bad, bad
assert pathlib.Path(ttnn.__file__).resolve().is_relative_to(repo), ttnn.__file__
PY
```

Expected: no mapped `_ttnn*` or `libtt_metal` path contains `model-bringup`.

### D. Optional cache isolation before hardware retry

Use this only if A-C pass. It is non-destructive and avoids deleting shared
caches.

```bash
export TT_METAL_CACHE="$(mktemp -d /tmp/tt-metal-qwen-mesh-env.XXXXXX)"
printenv TT_METAL_CACHE
```

### E. Bounded hardware smoke, after A-C pass

This is the first command in this plan that touches TT hardware. It should be
run only by the main agent/operator after the no-hardware provenance checks
pass.

```bash
timeout 60 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("open_mesh_device 1x2 OK")
PY
```

If this still fails with compile errors, keep the full compiler command and
diagnostics. If it fails with `Physical chip id ... not found`, switch the
investigation to mesh graph descriptor / visible-device topology instead of
dispatch compilation.

## Safest minimal fix

Do not edit implementation code. Do not rebuild first. Use the repo-local
environment as a unit:

```bash
cd /home/ttuser/dev/qwen3.8-flash-next/tt-metal
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
```

Then run probes A-C from that same shell. If they pass, run the bounded hardware
smoke with an isolated `TT_METAL_CACHE`. Avoid plain
`/home/ttuser/.tenstorrent-venv/bin/python` for this stage unless its stale
TTNN editable install and `ttnn-custom.pth` are intentionally removed or
overridden and the probes prove no `model-bringup` paths remain.

## Remaining uncertainty

This AutoDebug pass did not open TT hardware. The environment fix is strongly
supported by import-path and source/header evidence, but actual 1x2 mesh
availability still depends on the machine's visible-device mapping and mesh
graph descriptor. A later topology error would be a different failure class than
the reported dispatch JIT compile skew.

## Method note

The repo-local `.agents/scripts/autodebug.sh` runner was attempted first, as
required by the autodebug workflow. Its nested Codex session could inspect via
indexed search but could not run local shell commands or write files because
its sandbox rejected both shell and `apply_patch` operations. This report was
therefore written by the outer fresh autodebug agent after direct local,
inspection-only verification. No TT hardware was opened and no implementation
code was edited.
