# AutoDebug: replicated gather-projection corruption

2026-09-05. Fresh isolated AutoDebug/AutoFix investigation. The investigator
read source and existing artifacts, imported no TTNN, accessed no devices, and
ran no build. After the coordinator requested a focused diagnostic, the only
code addition was `tests/multichip_ag_diagnostic.py`; implementation is unchanged.

**First bad boundary verified: the BF16 3072-wide gather before the MLP down
projection corrupts rank 3, already during eager execution.** The post-projection
gathers correctly broadcast their inputs, including the later corrupt result.
The new packed GDN/shared MLP composition and trace capture are unnecessary.
**Concrete source defect:** the larger gather automatically places workers
outside the 8x8 grid on which its global semaphores are allocated/initialized.
Both independent hardware coverage controls pass. The coordinator integrated
model-local `MeshCCLManager` with the full actual device grid; the original
replicated gather failures and cumulative fused-norm failure now pass. See
[AUTOFIX_ag_trace.md](AUTOFIX_ag_trace.md) for commands and results. No
production code was edited by this investigator.

## Direct observations

Hardware evidence belongs to the coordinator's serialized lane: four Blackhole
chips on physical P300c boards, logical 1x4 ring, packet payload 8192 bytes,
BF16 residual/MLP/CCL outputs, FP32 GDN state, BFP4/LoFi projections, layer 0,
real pinned weights, 2048 recorded prefill tokens, one decode at position 2048.

Original command:

```bash
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --length 2048 --variant cumulative_ag_mm
```

| Artifact under `logs/` | Observation |
| --- | --- |
| `family_ag_mm_replicated_layer0` | All four replicas have trace PCC -0.24263827179624342 and max difference 2.7234680864278366e28; subsequent restored eager assertion fails. |
| `family_ag_mm_replicated_retry_layer0` | Identical metrics after coordinator reset/list/mesh-smoke recovery and native reader rebuild. Devices close normally. |
| `family_original_ag_mm_replicated_layer0` | Original `--variant ag_mm`, with separate QKVAB/Z and original MLP, fails with exactly the same trace metrics and restored eager assertion. |
| `family_original_ag_mm_sharded_layer0` | Original `ag_mm --residual sharded` passes exact restored eager/trace. Prefill PCC 0.9999220235537754, decode PCC 0.9999741188852158. |
| `family_ag_mm_sharded_layer0` | Cumulative `ag_mm --residual sharded` also passes exact restored eager/trace with the same output PCCs. |
| `family_default_replicated_layer0` | Packed GDN/shared MLP with native row-output reductions passes exact eager/trace; prefill/decode PCC 0.9999630806578761 / 0.9999879890284807. |

Logs and `.provenance.json`/`.sources.json.gz` artifacts contain the exact
coordinator commands and frozen sources. The historical
`../multichip_decoder/logs/selected32_ag_mm_layer0` command explicitly included
`--residual sharded`; it is not a previous passing replicated result.
Its archived `GatherOutputProjection` class matches the failing class.

The failed probe reads E1, captures, restores, replays, reads T1, restores,
then reads E2. It has **no E1/E2 comparison before capture**. It asserts restored
eager equality before saving its trace-mismatch artifact and before the final
baseline PCC comparisons. These failures therefore establish neither finite
E1 output nor prefill/baseline correctness nor that trace first corrupts state.
The diagnostic below closes the first two localization gaps.

### First-boundary experiment, run by the coordinator

`logs/ag_boundary_layer0` runs the new diagnostic, exits 1 on
`synchronized eager model gather differs from host concat`, and closes devices.

| Boundary | Result |
| --- | --- |
| Prefill output | All replicas finite, range [-0.474609375, 8.125]. |
| E1/E2 AG0, FP32 GDN input1024 | Exact host concat on every rank. |
| E1/E2 AG1, BF16 GDN output1024 | Exact host concat on every rank. |
| E1/E2 AG2 source, BF16 MLP input3072 | All ranks finite, total range [-0.6328125, 0.337890625]. |
| E1 AG2 output12288 | Ranks0/1/2 exact; rank3 max difference 1.5845632502852868e29. |
| E2 AG2 output12288 | Ranks0/1/2 exact; rank3 max difference 0.4281005859375. |
| E1/E2 AG3, BF16 down output1024 | Every rank exactly matches concatenation of the already-corrupt local down results. |
| Restored E1/E2 before trace | Different layer outputs; max difference 9.903520314283042e27. |
| Isolated four-gather sequence E1 | Reuploaded finite actual AG2 sources, persistent inputs and retained outputs: AG2 rank3 still corrupt, max difference3.770263671875. Other gathers exact. |
| Isolated E2 and T1/T2/T3 | All gathers exact. |

Thus a model source-buffer deallocation and trace capture are not necessary to
reproduce the corruption. Tile layout or allocator effects are not completely
excluded, but are not established by these observations. First-use failure
followed by warmed success is consistent with uninitialized semaphore memory
being reset by the first executed kernels. It is not evidence for padding as
the cause.

## Headline source defect: worker/semaphore grid mismatch

`models/demos/gpt_oss/tt/ccl.py:25-69` hard-codes `ccl_cores` to `(0,0)..(7,7)`
and initializes all AG/RS/barrier semaphores on those cores only. It creates a
Python `SubDevice` object but does not load a subdevice manager or constrain the
device worker set. The model's `_gather` supplies neither `sub_core_grid` nor an
explicit worker count.

The larger MLP gather has 786432 output bytes, so per-link/per-direction data is
`786432 * 3 / 4 / 2 = 294912 > 262144` bytes. The default factory therefore
chooses four workers per direction, plus one mux per direction: ten cores.
`ccl_common.cpp:454-532` chooses row-major from the entire Blackhole worker grid,
starting `(0,0)..(9,0)`. Muxes are `(0,0)` and `(5,0)`; direction-1 workers include
`(8,0)` and `(9,0)`, outside the semaphore allocation. The smaller gathers
choose two workers plus mux per direction, six cores `(0,0)..(5,0)`, all covered.

`GlobalSemaphoreImpl::setup_buffer/reset_semaphore_value`
(`tt_metal/impl/buffers/global_semaphore.cpp:59-104`) allocates a sharded uint32
buffer on the explicitly supplied cores and writes initial zero only there.
The kernel receives the same scalar semaphore address on all selected workers;
the CCL factory does not intersect worker selection with semaphore coverage.
Both ready and barrier addresses are uninitialized on those two workers.
Readers use `noc_semaphore_wait_min`, then reset ready memory to zero at exit;
premature reads from output buffers explain arbitrary finite magnitudes without
a hang, and subsequent initialization can explain the isolated warm pass.

The sharded family changes the MLP gather from AG pair0/barrier0 to pair1/barrier1
and changes preceding CCL history. Its pass does not make uncovered semaphore
addresses valid; it can incidentally initialize or encounter different memory.
This is a stronger causal story than blaming the added output gather itself,
which the measured boundary directly refutes.

One candidate intervention: pass `sub_core_grids=self.ccl.ccl_cores` at the
model `_gather` caller, keeping the automatic four-worker algorithm but placing
its ten cores within the allocated semaphore grid. An independent control
allocates semaphores on the full device grid and leaves the original worker
geometry unchanged. Both independently pass all diagnostic checks. The
coordinator selected full-device semaphore coverage so explicit-offset fused
CCL workers are covered as well; original uninstrumented checks pass.

## Effective path and source adjudication

`tests/optimized_multichip_candidates.py` creates
`(GatherOutputProjection, PackedGDNSharedMLP)`. Its C3 order is
`GatherOutputProjection -> PackedGDNSharedMLP -> PackedGDN -> SharedMLPInput ->
MultichipDecoder -> OptimizedDecoder ...`. Cooperative construction reaches
both weight preparations. GatherOutputProjection owns row-output `_linear`;
PackedGDN owns `_gdn_project`; SharedMLPInput owns `_activate_mlp`. There is no
skipped initializer or wrong dispatch established by source, and the original
class failure refutes composition as a necessary cause.

`tests/multichip_topology_candidates.py:80-141` gathers each local activation,
multiplies by weights sharded along output channels, then, **only for replicated
residuals**, gathers the 1024-wide projected output again at line 140.
The matmul has grid `(8,1)`, K block 4 tiles, M=1 tile, N=4 tiles/core, subblock
`(1,4)`, BFP4 weights, LoFi, BF16 output. GDN global K is 4096; down global K is
12288. Each rank produces its own 1024 output channels. Column weight ordering
and local GDN head ordering agree; no missing TP sum is indicated.

`tt/optimized_decoder.py:393-426` then converts that gathered update into
32-core width-sharded L1, performs the residual add, and runs the next norm.
The sharded-residual family uses a different `_block` and omits both output
gathers, so its pass does not by itself select between gather, conversion/add,
or changed allocation/timing.

Decode gather ledger, assuming initial indices zero:

| Replicated call | Source local shape, padded height 32 | Dtype | AG pair / barrier |
| --- | --- | --- | --- |
| Before GDN output matmul | `[1,1,1,1024]` | FP32 | 0 / 0 |
| After GDN output matmul | `[1,1,1,1024]` | BF16 | 1 / 1 |
| Before MLP down matmul | `[1,1,1,3072]` | BF16 | 0 / 0 |
| After MLP down matmul | `[1,1,1,1024]` | BF16 | 1 / 1 |

Sharded decode instead runs stats32 AG0, norm1024 AG1, GDN1024 AG0, stats32 AG1,
norm1024 AG0, MLP3072 AG1. Both sequences return the Python AG and barrier
indices to their initial parity. A missing Python counter reset alone cannot
explain the observed contrast. Kernel semaphore reuse may still need testing.

`tt/multichip_decoder.py:286-296` supplies `async_links=1`, two semaphores, a
barrier, dim3, interleaved L1, no persistent output buffer. `links=2` applies
to native prefill all-reduce, not these decode gathers. The older
`../multichip_decoder/AUTOFIX_sharded_trace.md` verified a two-link async
workaround; the current path already uses its one-link setting. The older
single-chip mixed-dtype sharded-add report also does not establish this cause:
the current projected update is explicitly BF16, matching the residual.

Lowering: `all_gather_async.cpp` selects the persistent-buffer overload with
an absent buffer. Widths 1024/3072 are tile aligned, so
`composite_common.cpp:285-307` rejects the composite fallback. Interleaved input
rejects the Llama sharded factory. `all_gather_async_device_operation.cpp:43-75`
selects MINIMAL_DEFAULT and constructs a same-dtype output with width times 4.
The default factory emits `minimal_default_reader.cpp` and
`minimal_default_writer.cpp` with ring size 4, gather dim3 and one link.

Concrete default-factory ledger for payload 8192:

| Source | Input pages | Page bytes | Output bytes | Workers/direction | Worker page ranges | Pages/packet | Chunks/sync |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| FP32 width1024 | 32 | 4096 | 524288 | 2 | `[0,16)`, `[16,32)` | 2 | 8 |
| BF16 width1024 | 32 | 2048 | 262144 | 2 | `[0,16)`, `[16,32)` | 4 | 4 |
| BF16 width3072 | 96 | 2048 | 786432 | 4 | `[0,24)`, `[24,48)`, `[48,72)`, `[72,96)` | 4 | 6 |

The four-device even-ring split forwards 8/8 or 12/12 pages per worker for
the opposite rank; these ranges and packet counts divide evenly. The same
BF16 width1024 factory geometry is exercised by the passing sharded norm
gathers. No unique illegal shape was found in the replicated output gather.
Source: `all_gather_async_default_program_factory.cpp:156-202,242-280,403-427,679-716`.

The cache hash deliberately omits semaphore addresses, but the override callback
updates each mesh program's input/output addresses, ready semaphore, and barrier
(`all_gather_async_default_program_factory.cpp:112-141,828-875`). Thus the
apparent repeated-same-shape cache risk is **not** a proven missing-address
update. Retaining no persistent output buffer is allowed by this API; it is not
alone a trace ownership violation. The parent native reader patch changes the
DRAM-sharded matmul factory; these gather-output projections use the interleaved
multicast path and fail identically before/after that patch.

## Minimal controlled experiments

Run serially through the stage recorder under the coordinator hardware lane.
Do not change precision, multiple lifetimes, or several collectives together.

1. **Original diagnostic, completed as `ag_boundary_layer0`.**

   ```bash
   python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --length 2048
   ```

   The added diagnostic checks actual prefill output, every E1/E2 gather input
   and output against exact host concatenation, and restored E1/E2 equality
   before trace. Then it uploads those exact local sources, retaining their
   dtype/logical shape, and runs the four-gather sequence twice eagerly and
   three times from one trace. No host action occurs inside that trace.
   Inline model-boundary reads deliberately synchronize and may suppress an
   asynchronous failure. The isolated sequence keeps all outputs live and is
   therefore a data-path control, not proof of the original lifetime pattern.

   Result: first corruption is AG2 rank3, including the isolated eager control,
   as recorded above. No trace/lifetime/padding explanation is established.

2. **Keep automatic worker count; constrain it to allocated semaphore cores.**

   ```bash
   python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --length 2048 --gather-core-grid
   ```

   Predict every gather and E1/E2 become exact, without dtype/layout changes.
   First attempt failed at Python argument binding: the diagnostic originally
   passed singular `sub_core_grid`, while nanobind requires `sub_core_grids`
   (`all_gather_async_nanobind.cpp:259,280,297`). Corrected in the diagnostic;
   that API failure has no numerical interpretation. Corrected
   `ag_boundary_core_grid_layer0_v2` exits0: all eight model gather checks,
   restored E1/E2 and isolated eager/three-replay checks are exact.

3. **Independent control: expand semaphore coverage, keep worker placement.**

   ```bash
   python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --length 2048 --semaphore-full-grid
   ```

   Allocates/zeros semaphores on the full device compute grid before forward.
   No AG worker/grid override. A pass supports missing coverage independently
   of control2's placement change. `ag_boundary_full_grid_layer0` exits0 with
   all model and isolated checks exact; device/semaphore grid is11x10.
   `--gather-workers 2` is also available as a
   third control; it uses six covered cores, but changes work partitioning and
   is less diagnostic than preserving four workers.

These controls were collected before the coordinator integrated the fix;
their source archives preserve the failing8x8 manager and isolated interventions.
Current default construction uses full coverage. The coordinator reran both
original replicated failures successfully; verification is in AutoFix. Keep
exact restored eager/trace and >=0.995 PCC gates. Worker/semaphore watcher
verification belongs on the separate serialized lane, not with profiling.

## Related fused-norm failure and cache-lifetime adjudication

`family_fused_norm_ag_mm_sharded_layer0` failed initial restored trace equality
(PCC0.9999518537282491, max difference0.0625), then restored eager equality.
Source reveals the **same coverage defect**: `FusedNormGatherProjection`
passes `all_gather_core_grid_offset=(0,8)`
(`tests/multichip_topology_candidates.py:509-522`). The fused factory forwards
that unchanged to the same default AG builder
(`all_gather_matmul_async_program_factory.cpp:128-150`). BF16 local norm1024
gathered to4096 selects two workers and one mux per direction, now on
`(0,8)..(5,8)`; every worker is outside the old8x8 semaphore grid.
`cclfixed_cumulative_fused_norm_ag_mm_sharded_layer0` passes without candidate
changes after full-grid semaphore coverage: eager/trace max difference0,
decode PCC0.9999655849013239, state checks pass.

The `_gathered_norm` attribute is reset at each attention/FF norm boundary;
it only shares the gathered input between projections of one normalized value.
During capture, the attention gathered buffer is produced, consumed and freed
at the FF norm boundary; the FF gathered buffer is produced/consumed and retained.
Replay records both producers and consumers at fixed addresses. A subsequent
eager decode frees the capture-created FF buffer, then creates and retains its
own final FF buffer. This alone is not stale-input reuse: the next eager norm
discards the retained attribute before any projection can consume it, and each
replay produces its own scratch contents before reading them.

Allocator protection is not automatic reservation of all captured scratch:
`allocator.hpp:174-176` explicitly notes trace-used memory is not tracked after
capture; `allocator.cpp:119-134` warns new live allocations may be overwritten.
The new eager cached buffer may therefore be overwritten by replay, but its
contents are semantically dead and are never read by the following eager epoch.
No independent lifetime bug is established by the inspected path or the passing
full-grid result. Do not rewrite cache ownership to explain a fixed symptom.

There is a focused harness gap: `multichip_probe.py:206-256` checks E1/T1/E2,
then lines258-264 time160 later replays without reading their output. The
`trace_values` comparison after E2 still uses the earlier T1 host tensor.
A restore→T2→read/compare after E2, with state validation, is the minimal
correctness check for this concern. Executing timed later traces alone does
not establish their output correctness.

## Investigator status

First bad eager CCL boundary and worker/semaphore coverage cause are verified
by the coordinator's two independent controls and original-command reruns.
Model-local full-grid manager is integrated. New diagnostic
passes Python syntax and Black (`--target-version py310 --check`); no build
required for this Python-only addition.
The related fused-norm failure is also fixed by the same coverage correction;
its retained cache exposes a later-replay test gap, not a verified second bug.
