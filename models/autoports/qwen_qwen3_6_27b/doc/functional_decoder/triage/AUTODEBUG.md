# AutoDebug Report: device-open JIT source/package mismatch

## Scope and constraints

- Checkout: `/home/ttuser/dev/qwen-perf/tt-metal`
- HEAD: `d58cb341c703310cf41b5d88baafc0790ec0270b`
- Installed extension: `/home/ttuser/.local/lib/model-bringup/tt-metal/build_Release/ttnn/_ttnncpp.so`
- Investigation was source-only. No device open, reset, `tt-smi`, or hardware-facing test was run by this investigation.
- Pre-existing worktree state was left untouched: modified submodules `tt_metal/third_party/tracy` and `tt_metal/third_party/umd`, plus untracked `tt_metal/third_party/tt-cluster-descriptors/`.

## Direct observations

1. Relative kernel paths are deliberately resolved in this order by `tt_metal/impl/kernels/kernel.cpp`: current working directory, `TT_METAL_KERNEL_PATH`, system kernel directory, then runtime root. `KernelSource` stores the first resolved path. Therefore, when the process CWD is this checkout, `tt_metal/impl/dispatch/kernels/cq_dispatch.cpp` and `cq_prefetch.cpp` resolve to the checkout even when `TT_METAL_RUNTIME_ROOT` names the installed package.
2. `tt_metal/jit_build/genfiles.cpp` writes the resolved `KernelSource::path_` as an absolute include in generated `kernel_includes.hpp`.
3. The failed fresh-cache artifact confirms that mechanism:
   - `/tmp/qwen36_tt_cache_probe_20260819/tt-metal-cache5822202293905037534/kernels/cq_dispatch/6216444706482813536/kernel_includes.hpp` includes the checkout's absolute `cq_dispatch.cpp`.
   - The corresponding `brisc/brisck.o.log` reports `init_telemetry` undeclared at checkout `cq_dispatch.cpp:1486`.
   - Its dependency file shows the firmware and ordinary header dependencies coming from `/home/ttuser/.local/lib/model-bringup/tt-metal`, while the top-level dispatch source comes from the checkout. This is a mixed source graph.
4. Checkout commit `67833ca351d` added both calls to `init_telemetry` and the implementation in `telemetry.hpp`. At current HEAD:
   - checkout `cq_dispatch.cpp:21` and `cq_prefetch.cpp:25` include `tt_metal/impl/dispatch/kernels/telemetry.hpp`;
   - checkout `telemetry.hpp:21` defines `init_telemetry`;
   - the installed source and `build_Release/libexec` copies of `telemetry.hpp` have SHA-256 `12646ed2...` and do **not** define `init_telemetry`;
   - checkout `telemetry.hpp` has SHA-256 `e7c8f684...` and does define it.
5. The verified control supplied after the original failure is decisive: with CWD `/tmp`, `TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal`, and a fresh `TT_METAL_CACHE`, the same 1x1 probe succeeded (`DEVICE_IDS [3]`, `MESH_1X1_SMOKE_OK`). The failed repo-CWD control also used a fresh cache.

## Ranked hypotheses

### H1 — Verified: CWD-first kernel resolution mixes newer checkout dispatch sources with older installed headers/runtime

This is the direct cause of the compile error. The installed `_ttnncpp.so` asks for relative dispatch kernel paths. From the repository root, `resolve_path()` chooses checkout files before consulting the installed runtime root. Generated `kernel_includes.hpp` consequently names the newer checkout dispatch source, but the JIT toolchain's firmware and include graph is installed-package-owned. The installed `telemetry.hpp` predates the `init_telemetry` helper used by the checkout dispatch source, so compilation fails exactly at that name.

Prediction: moving CWD outside a tree containing `tt_metal/` makes resolution fall through to the installed runtime and yields a self-consistent installed source graph. The `/tmp` control passed and its generated headers include installed absolute dispatch paths, satisfying this prediction.

### H2 — Verified contributing mechanism: `TT_METAL_RUNTIME_ROOT` is not an override for a kernel file found beneath CWD

`RunTimeOptions` accepts the environment runtime root, but `KernelSource::resolve_path()` checks CWD before the runtime root. Thus setting `TT_METAL_RUNTIME_ROOT` cannot prevent checkout kernel shadowing while running from the repository root. This explains why the initial runtime-root control reproduced rather than disproving a package mismatch.

### H3 — Refuted: stale JIT cache is the primary cause

Both failing and passing controls used fresh cache roots and generated different absolute source includes based on CWD. Cache deletion alone cannot repair the mixed graph; it merely regenerates the same CWD-selected path.

### H4 — Refuted: P300 device health or an on-device kernel failure caused the symptom

The failure is a host-side C++ compile error before device open completes. The same installed runtime opened and closed the device from `/tmp`. No device reset is indicated by this evidence.

### H5 — Unlikely and not needed to explain the failure: checkout `telemetry.hpp` itself omits the declaration

The checkout header contains the template definition at line 21. The error arises because the mixed installed include graph selects the older installed header, not because HEAD lacks the helper.

## Focused verify/refute experiments for AutoFix

Run hardware-facing experiments serially under `$tt-device-usage`; the first two controls are already recorded and need not be repeated unless validating a repair.

1. **Static resolution proof (no hardware):** inspect the failed and passing generated headers and dependency files.
   ```bash
   sed -n '1,5p' /tmp/qwen36_tt_cache_probe_20260819/tt-metal-cache5822202293905037534/kernels/cq_dispatch/6216444706482813536/kernel_includes.hpp
   sed -n '1,5p' /tmp/qwen36_tt_cache_probe_from_tmp_20260819/tt-metal-cache5822202293905037534/kernels/cq_dispatch/3078774857514829284/kernel_includes.hpp
   rg -n 'cq_dispatch.cpp|telemetry.hpp' /tmp/qwen36_tt_cache_probe_20260819/tt-metal-cache5822202293905037534/kernels/cq_dispatch/6216444706482813536/brisc/*.d
   ```
   Expected: failed artifact uses checkout top-level source plus installed dependencies; passing artifact uses installed top-level source and dependencies.

2. **Header API proof (no hardware):**
   ```bash
   rg -n 'init_telemetry' tt_metal/impl/dispatch/kernels/telemetry.hpp
   rg -n 'init_telemetry' /home/ttuser/.local/lib/model-bringup/tt-metal/tt_metal/impl/dispatch/kernels/telemetry.hpp
   sha256sum tt_metal/impl/dispatch/kernels/telemetry.hpp /home/ttuser/.local/lib/model-bringup/tt-metal/tt_metal/impl/dispatch/kernels/telemetry.hpp
   ```
   Expected: helper only in checkout header and hashes differ.

3. **Operational workaround verification (hardware; already passed):** run the installed extension from `/tmp` with installed `TT_METAL_RUNTIME_ROOT` and a fresh cache. Expected: generated includes stay entirely under the installed tree and 1x1 open/close succeeds.

4. **Durable environment repair verification (hardware):** use a `_ttnncpp.so` built from this exact checkout (or install a source/runtime package matching the extension), run from repo CWD with a fresh cache, then inspect the dependency file before accepting the repair. Expected: extension/JIT runtime/kernel source/header APIs are from one revision and open/close succeeds.

5. **Negative control for root precedence (hardware only if still useful):** keep the older installed extension and repo CWD while changing only `TT_METAL_RUNTIME_ROOT`. Expected: it still selects checkout dispatch sources because CWD has higher priority. This was already observed; repeating it adds little value.

## Fix boundary

No implementation change is justified by this report for the functional decoder. The immediate safe workaround is to invoke the installed TTNN environment outside the checkout root so that its packaged kernels are selected. The durable fix for development in this checkout is a checkout-matched TTNN build/runtime package. If project policy expects an installed extension to be runnable from an arbitrary checkout CWD without source shadowing, that is a separate runtime path-resolution/product issue; changing global precedence should be handled with dedicated compatibility tests because current CWD-first behavior is explicit and may support development workflows.

## Final status

Root cause is verified: a version-skewed, CWD-dependent mixed JIT source graph. It is recoverable without device reset. AutoFix should validate either the `/tmp` operational workaround for the current stage or a checkout-matched TTNN build, then resume functional-decoder work.
