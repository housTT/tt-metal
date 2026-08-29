# AutoDebug: Blackhole ACTIVE_ETH watcher kernel-config overflow

> Superseded final disposition: `$autofix` subsequently retained normal router
> inlining, used the supported `TT_METAL_FABRIC_OPT_LEVEL=Os` watcher override,
> and fixed a separately proven sticky packet-tag teardown defect. The fully
> enabled TP2/TP4 matrix now passes and closes cleanly. See
> `AUTOFIX_watcher_active_eth.md`.

## Outcome

The original failure is a deterministic host-side admission failure in the
Blackhole two-ERISC fabric-router program, before any GPT-OSS model operation
runs. With all watcher features enabled and normal forced inlining, the router
program requires 29,072 bytes in an ACTIVE_ETH kernel-config buffer whose HAL
capacity is 26,624 bytes. The overage is 2,448 bytes (9.2%).

`TT_METAL_WATCHER_NOINLINE=1` is a useful size experiment, not a safe final
resolution. It admits the fully watched program and a simple four-device mesh
smoke closes successfully. TP2 sliding, TP2 full, and TP4 sliding also complete
their real model/PCC assertions. However, the isolated TP2-full and TP4-sliding
processes then abort from MetalContext teardown with a frozen ACTIVE_ETH
heartbeat after the watcher thread stops. Their safe-pytest exit is 134.

The safer implementation candidate is to keep normal forced inlining and give
ACTIVE_ETH, specifically, a 30-KiB kernel-config reservation:

```c
#define MEM_ERISC_KERNEL_CONFIG_SIZE (26 * 1024)   // idle ERISC
#define MEM_AERISC_KERNEL_CONFIG_SIZE (30 * 1024)  // active ERISC
```

The observed 29,072-byte program then has 1,648 bytes of headroom without
changing watcher code generation or disabling any feature. The candidate in
the working tree consistently updates the active HAL size/base and device
address map while leaving idle ERISC at 26 KiB. It passed a local incremental
`tt_metal` target build, and the required CI wrapper configured successfully in
a clean build directory. A full wrapper build and hardware validation remain
unverified; the wrapper warned that Garage credentials were absent and the
initial default-directory invocation also encountered an existing CMake-cache
path mismatch.

The final TP2/TP4 acceptance matrix is therefore blocked on building and
running the active-only 30-KiB candidate with normal inlining. There is no
source-backed reason to weaken watcher coverage with
`TT_METAL_WATCHER_DISABLE_ETH=1`.

## Scope and evidence

This investigation was source-only: this agent did not run hardware or edit
implementation code. It inspected the supplied logs, generated/cache ELFs, and
the current source tree. The repository AutoDebug wrapper was invoked as
required, but its nested agent could not read the checkout because the host's
`bwrap` sandbox could not create its namespace/loopback setup; it was stopped
after the failure proved systemic. The findings below were independently
checked against source and ELF program headers.

Primary logs:

- `artifacts/20260829_watcher/default_release_watcher10_all.log.gz`
- `artifacts/20260829_watcher/default_release_watcher10_noeth_all.log.gz`
- `artifacts/20260829_watcher/autofix_watcher_noinline_mesh_smoke.log.gz`
- `artifacts/20260829_watcher/autofix_watcher_noinline_all.log.gz`
- `artifacts/20260829_watcher/autofix_watcher_noinline_tp2_full_separate.log.gz`
- `artifacts/20260829_watcher/autofix_watcher_noinline_tp4_sliding_after_reset.log.gz`
- `artifacts/20260829_watcher/autofix_noinline_post_reset_mesh_smoke.log.gz`
- `artifacts/20260829_watcher/autofix_aerisc_30k_local_build.log.gz`
- `artifacts/20260829_watcher/autofix_aerisc_30k_wrapper_configure.log.gz`

## Observations versus interpretation

| Evidence | Observation | Interpretation |
| --- | --- | --- |
| Full watcher log, lines 45-51 | `disabled features: None`; failure immediately after `Initializing Fabric`: `Program size (29072) ... (26624) on ACTIVE_ETH` | Failure belongs to fabric compilation/finalization, not decoder execution. |
| Full watcher stack, lines 218-228 | `ProgramImpl::finalize_program_offsets` is reached through `Device::configure_fabric()` | Host rejects the router image before launch. No device-side watcher report is involved. |
| Ethernet-disabled log, lines 45-51 and 469-472 | `disabled features: ETH`; fabric initializes and all TP2/TP4 variants pass | Removing watcher code from Ethernet kernels avoids the overflow, but weakens the requested semantics. |
| No-inline mesh smoke, lines 1 and 46-55 | `TT_METAL_WATCHER_NOINLINE=1`, `disabled features: None`, four-device fabric initializes, test passes | No-inline is sufficient to admit and launch the fully watched fabric router. |
| No-inline decoder log, lines 45-152 | `disabled features: None`; fabric initializes; TP2 sliding completes and passes | No-inline remains viable through real model prefill/decode, not only mesh setup. |
| Isolated TP2-full log, lines 45-169 | Model/PCC passes; after watcher stops, core 29-25 heartbeat remains `0xdcbada00` for 20 seconds; process aborts | Fresh-process isolation does not cure no-inline teardown. |
| Post-reset isolated TP4-sliding log, lines 45-235 | Model/PCC passes; after watcher stops, core 29-25 heartbeat remains `0xdcba9540`; process aborts | Failure reproduces from a reset board on a different device, after valid model output. |
| Post-reset no-inline smoke, lines 22-30 | Fully enabled watcher initializes fabric and prints `MESH_SMOKE_OK` after close | The teardown failure needs sustained/model traffic or state; it is not guaranteed by mesh open/close alone. |

## Exact causal chain

1. `TT_METAL_WATCHER=10` enables watcher; the numeric value becomes a
   10,000-ms polling interval. It does not select a reduced feature set
   (`tt_metal/llrt/rtoptions.cpp:1172-1184`). With no disable variables, all
   watcher features remain enabled.

2. JIT builds receive `-DWATCHER_ENABLED`. The no-inline variable separately
   adds `-DWATCHER_NOINLINE`, while feature-disable variables become
   `-DWATCHER_DISABLE_<feature>` (`tt_metal/jit_build/build.cpp:242-253`).
   Device debug print is independently gated by `DEBUG_PRINT_ENABLED`
   (`tt_metal/jit_build/build.cpp:255-257`), so debug print was not implicitly
   enabled by this watcher command.

3. Blackhole defaults to two-ERISC mode
   (`tt_metal/llrt/rtoptions.hpp:315-320`). The fabric router builder emits one
   Ethernet kernel per configured RISC, targeting RISCV_0 and RISCV_1
   (`tt_metal/fabric/compute_mesh_router_builder.cpp:959-995`).

4. ACTIVE_ETH stores kernel text in the program config buffer. Program
   finalization iterates every kernel in the group, adds each binary's packed
   size, and aligns after each binary
   (`tt_metal/impl/program/dispatch.cpp:479-540`). Thus the active and
   subordinate ERISC router images share and add against one ACTIVE_ETH
   capacity; they do not each receive 26 KiB.

5. The Blackhole memory map declares a shared
   `MEM_ERISC_KERNEL_CONFIG_SIZE` of `26 * 1024`, with a comment that it was
   chosen to fit the 25,680-byte non-watcher 2D-torus router
   (`tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:231-233`). The
   active-Ethernet HAL places it at `MEM_AERISC_MAP_END` and reports that exact
   size (`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_active_eth.cpp:93-95` and
   `:138-150`). The source itself marks the shared sizing as needing an
   Ethernet-specific value.

6. `get_ringbuffer_size` uses the HAL's KERNEL_CONFIG size for non-Tensix
   programmable cores (`tt_metal/impl/program/program.cpp:108-115`). After RT
   args, semaphores, CB/DFB state, and both binaries have been assigned,
   `finalize_program_offsets` checks the total against the HAL capacity and
   raises the exact fatal seen here (`tt_metal/impl/program/program.cpp:2932-2999`).

## ELF accounting

Packed sizes below are the sum of `PT_LOAD` `FileSiz` values from the cached
XIP ELF program headers. That matches the binary payload counted by
`finalize_kernel_bins`; its alignment is applied between payloads.

| Build | Largest active ERISC payload | Largest subordinate payload | Conservative pair | Result |
| --- | ---: | ---: | ---: | --- |
| Full watcher, normal inlining | 14,772 B | 13,960 B | 28,732 B | Exact failing pair plus 340 B metadata = 29,072 B |
| Watcher disabled on Ethernet | 9,504 B | 7,996 B | 17,500 B | Fits, but Ethernet watcher semantics are removed |
| Full watcher, no-inline | 13,364 B | 11,052 B | 24,416 B | Worst observed independent maxima plus 340 B = 24,756 B, leaving about 1,868 B |

The full-watcher equality is especially strong evidence: `14,772 + 13,960 +
340 = 29,072`, exactly the host-reported program size. No tensor layout,
decoder datatype, TP partition, trace buffer, model cache, or prompt can affect
this pre-execution value.

The no-inline figure uses the independent maximum for each processor and is
therefore conservative even if those maxima do not belong to the same router
configuration. It fits the current 26-KiB region without disabling any watcher
feature.

It also rules out a static kernel-config overlap as the cause of the no-inline
teardown failure. The Blackhole active config interval starts at
`MEM_AERISC_MAP_END = 62,256 (0xf330)` and ends before 88,880 (`0x15b30`);
UNRESERVED starts aligned at 88,896 (`0x15b40`). The conservative no-inline
program total is only about 24,756 bytes. Both ERISC images and program metadata
remain inside the declared config interval with roughly 1,868 bytes remaining.
The later heartbeat failure must therefore be sought in execution/termination,
not static active/subordinate image overlap.

## Ranked findings and hypotheses

### 1. Proven: forced watcher inlining overfills the shared two-ERISC config buffer

Confidence: very high.

The log location, source allocation chain, exact ELF arithmetic, and
Ethernet-disable comparison all agree. The overflow is caused by full watcher
instrumentation being duplicated/inlined into two fabric-router images that
are packed into a single 26-KiB ACTIVE_ETH program region.

### 2. Proven: watcher no-inline fixes admission but is not teardown-safe here

Confidence: very high for the observed 1D-ring cases.

`rtoptions.cpp:1199-1203` describes the variable as disabling watcher-function
inlining to reduce binary size. `risc_attribs.h:43-49` implements it by changing
`FORCE_INLINE` from `inline __attribute__((always_inline))` to an empty
attribute only when both watcher and watcher-noinline are defined. It does not
enter any watcher-disable branch. The logs independently confirm `disabled
features: None`.

The important caveat is that this is not scoped to watcher helpers. The macro
is global: setting watcher no-inline removes `always_inline` from every
`FORCE_INLINE` use in the translation unit, including the fabric termination
poll in `tt_metal/fabric/hw/inc/tt_fabric_utils.h:23-30` and many router/NoC
helpers. The cached representative ERISC0 ELF still fits and its `kernel_main`
frame is 768 bytes, so there is no proof of stack overflow. But altered call
layout/call depth is now the leading no-inline-specific risk, and the repeated
atexit aborts make this knob unsuitable for final promotion.

### 3. Safest candidate: give ACTIVE_ETH a 30-KiB distinct config region

Confidence: high for resolving the proven admission failure; device teardown
and the full TP2/TP4 matrix are not yet verified.

There is ample numerical L1 space for the current router. With the current map,
`MEM_AERISC_MAP_END` evaluates to 62,256 and the aligned unreserved base after
26 KiB is 88,896. The candidate 30-KiB active config moves the aligned
unreserved base to 92,992, still far below `MEM_ERISC_MAX_SIZE` (455,264), and
leaves 1,648 B above the observed 29,072-byte program. This costs 4 KiB of the
366-KiB active unreserved span.

However, changing the shared macro blindly is unsafe. The idle-Ethernet HAL
starts KERNEL_CONFIG at `MEM_IERISC_MAP_END` but computes UNRESERVED from
`MEM_AERISC_MAP_END` (`tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_idle_eth.cpp:56-58`).
At the current values, the declared idle config end is already 2,544 bytes past
the declared unreserved start. Both active and idle HALs also carry the TODO
that the config size needs an Ethernet-specific value (`bh_hal_active_eth.cpp:147-150`,
`bh_hal_idle_eth.cpp:77-80`).

A safe memory-layout change therefore does all of the following:

- Introduces distinct active- and idle-ERISC config-size constants.
- Derives each UNRESERVED base from that core type's own map end and size.
- Updates `internal/tt-1xx/blackhole/eth_l1_address_map.h:32-36`
  consistently.
- Adds compile-time non-overlap/order assertions for both maps.
- Sweeps 1D and 2D fabric router variants before selecting a capacity.

The existing comment says the non-watcher 2D router already needs 25,680 B, so
30 KiB must not be generalized to all watcher/topology combinations without
ELF evidence.

### 4. Proven blocker: no-inline model teardown can strand active ERISC

Confidence: very high that it is distinct from sizing; the precise device-side
divergence point requires a watcher/triage capture during teardown.

The isolated TP2-full and post-reset TP4-sliding runs both complete their model
checks and then abort at atexit. The throwing loop waits for base-firmware
heartbeat progress and repeatedly clears `aerisc_run_flag` until timeout
(`tt_metal/llrt/llrt.cpp:557-597`). The frozen `0xdcba....` value is the
software-active-ERISC heartbeat form (`internal/tt-1xx/blackhole/eth_fw_api.h:355-369`),
so base-firmware service never resumes.

The source has an exact deadlock surface. Host fabric teardown writes
`IMMEDIATELY_TERMINATE` only to each device's master router and does not wait for
the router's `TERMINATED` acknowledgement before clearing its state
(`fabric_firmware_initializer.cpp:405-419`). Each ERISC polls the signal, exits
the main loop, and then enters router `teardown`. That path contains three
two-ERISC scratch-register barriers; its comment explicitly says there is no
termination escape, and either ERISC missing a barrier leaves the other
spinning forever (`fabric_erisc_router.cpp:2853-2957`). The first barrier is
also before the master propagates termination to subordinate routers. Once the
active application kernel returns, ERISC0 firmware waits for subordinate ERISC
completion before marking the launch done (`active_erisc.cc:324-350`).

This chain explains why a router can stop advancing the active heartbeat yet
never return to base firmware. It does not establish which local ERISC or which
barrier diverges. A teardown-time waypoint/tt-triage capture is the minimal
remaining evidence needed. Avoiding global no-inline via the active-only 30-KiB
reservation is safer than weakening watcher or masking the atexit failure.

## Watcher feature-size audit

- NoC sanitization is compiled for ERISC whenever watcher is enabled and
  neither `WATCHER_DISABLE_NOC_SANITIZE` nor `FORCE_WATCHER_OFF` is present
  (`tt_metal/hw/inc/internal/debug/sanitize.h:21-35`). Its target/address/range
  checks are substantial, including `debug_sanitize_noc_addr`
  (`sanitize.h:435-535`), making it the most plausible large component of the
  watcher delta. Disabling it would weaken the required semantics and is only a
  useful diagnostic compile point.
- Circular-buffer sanitization is explicitly excluded from ERISC builds
  (`sanitize.h:304-305`), so it is not the ACTIVE_ETH growth source.
- Linked-transaction validation is compiled only under the extra opt-in
  `WATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION` (`sanitize.h:502-506`); it
  was not requested by these commands.
- Debug print has its own runtime feature gate and define
  (`tt_metal/jit_build/build.cpp:255-257`). The watcher command alone does not
  enable it, and the inspected router ELFs had no non-empty device-print-string
  payload. Dprint is not the cause.
- `TT_METAL_WATCHER_DISABLE_ETH=1` is not an equivalent sizing toggle. Kernel
  creation injects `FORCE_WATCHER_OFF=1` into Ethernet kernels
  (`tt_metal/impl/host_api/tt_metal.cpp:1492-1496`), so the passing adapted log
  deliberately removes all Ethernet watcher instrumentation.

## Minimal reproducers and focused experiments

### A. Original admission failure

The smallest device repro is fabric open; no model weights or decoder are
required:

```bash
TT_METAL_WATCHER=10 python_env/bin/python -c '
import ttnn
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
m = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
'
```

Expected baseline result: host fatal `29072 > 26624` while initializing fabric.
No-inline makes this minimal open/close pass, but that smoke is insufficient:
real model processes later reproduce the atexit heartbeat abort.

### B. Compile-only size matrix

Use the Blackhole mock fabric compile path
(`fabric_firmware_initializer.cpp:296-303,458-473`) to generate router ELFs
without device I/O. Mock program finalization intentionally skips real binaries
(`tt_metal/impl/program/dispatch.cpp:488-493`), so acceptance must explicitly
sum/alignment-model the generated active and subordinate XIP ELF `PT_LOAD`
payloads rather than expect the 26-KiB fatal.

Compile these points:

1. full watcher, current inlining;
2. full watcher plus `TT_METAL_WATCHER_NOINLINE=1`;
3. full watcher, current inlining, active-only 30-KiB config;
4. full watcher plus `TT_METAL_FABRIC_OPT_LEVEL=Os`, as a configuration-only
   size alternative that retains forced inlining;
5. one watcher feature disabled at a time, for attribution only; and
6. debug print independently enabled, to prove it is orthogonal.

For every generated 1D/2D router configuration, compute the exact pair plus
program metadata. Promotion criterion is `<= 30720` with margin for the
active-only candidate; feature-disable points are not promotion candidates.
`TT_METAL_FABRIC_OPT_LEVEL` is source-supported
(`erisc_datamover_builder.cpp:1781-1790`, `rtoptions.cpp:1660-1672`) and is safer
than global watcher no-inline if it independently produces a fitting image,
but it needs an ELF-size check and device performance/correctness evidence.

### C. Final model and teardown acceptance

After a full wrapper build, run TP2 sliding, TP2 full, TP4 sliding, and TP4 full
with normal inlining and the active-only 30-KiB build:

```bash
TT_METAL_WATCHER=10
```

Require all of the following for every case:

- watcher reports `disabled features: None`;
- fabric initializes without a program-size fatal;
- model correctness checks pass;
- safe-pytest teardown is clean; and
- no watcher errors or Ethernet heartbeat stalls occur.

Then run a dedicated two-cycle fabric open/close/open/close test. Also capture
watcher waypoints or tt-triage before the watcher server stops if termination
hangs, to identify which ERISC missed which teardown barrier.

### D. Verification status for the active-only candidate

Current evidence:

1. `cmake --build build_Release --target tt_metal -j8` passed locally.
2. `.github/scripts/copilot-build.sh --build-dir build_copilot_aerisc_30k
   --configure-only` passed in the CI image.
3. The initial default wrapper build was not valid verification: it warned of a
   cold cache due missing Garage credentials, then failed because an existing
   `build_Release/CMakeCache.txt` referred to the host path rather than `/work`.
4. No full clean wrapper compilation has completed.
5. No hardware run of the 30-KiB candidate has been performed.

Before promotion, complete the full wrapper build, compile-sweep 1D and 2D
router variants, prove every pair fits, and run initialization, model,
termination, and repeated-open checks on Blackhole hardware.

## Rejected explanations

- **Decoder/model bug:** rejected for the original failure because it occurs in
  fixture fabric setup before model construction or execution.
- **TP2/TP4 sharding choice:** rejected as the size trigger because the first
  parametrized case never reaches its model body; fabric topology and watcher
  JIT defines determine the router image.
- **Trace-region allocation:** rejected because the fatal concerns the fixed
  ACTIVE_ETH kernel-config region, not general device tensor L1.
- **Watcher polling interval `10`:** rejected as a code-size selector. Source
  parses it only into milliseconds after enabling watcher.
- **Debug print:** rejected because it is separately gated and not enabled in
  the failing command.

## Recommended disposition

Do not promote global `TT_METAL_WATCHER_NOINLINE=1`: it fixes admission and
model correctness but reproducibly aborts during real-model atexit. Do not
accept the Ethernet-disabled run as final watcher evidence.

The preferred next step is the distinct 30-KiB ACTIVE_ETH config region with
normal forced inlining. It exactly addresses the proven capacity defect,
preserves full watcher semantics, leaves idle ERISC unchanged, and avoids the
no-inline call-layout change. It remains a candidate until the required full
wrapper build and Blackhole TP2/TP4 plus teardown matrix pass. If it still
hangs, the exact blocker is the unacknowledged two-ERISC teardown protocol
described above; capture both ERISC waypoints at the three local barriers rather
than disabling watcher.
