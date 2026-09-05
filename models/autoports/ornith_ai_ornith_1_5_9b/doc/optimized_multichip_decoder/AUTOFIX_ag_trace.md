# AutoFix: CCL semaphore coverage

## Starting evidence

Fresh investigation: [AUTODEBUG_ag_trace.md](AUTODEBUG_ag_trace.md).
Original failure was `multichip_probe --layer 0 --length 2048 --variant
cumulative_ag_mm` with replicated residuals. Original/cumulative gather-projection
variants failed with trace PCC-0.24263827179624342 and max difference2.72e28,
then restored eager disagreement. Reset and native reader rebuild did not
change the failure. Packed GDN/shared MLP without gather projections passed.

All hardware experiments below were run by the coordinator on its serialized
lane: four Blackhole chips on physical P300c boards, logical1x4 ring, real
pinned weights, length2048, BF16 activations/CCL, FP32 GDN state, BFP4/LoFi
projections. This isolated investigator used source/artifacts only and added
the diagnostic; it did not import TTNN, access hardware, or edit production.

## Verified cause and controlled experiments

The borrowed GPT-OSS CCL manager allocated and initialized ready/barrier
semaphores only on logical8x8 cores. The Blackhole worker planner uses the
actual11x10 device grid. A BF16 local3072→12288 gather moves294912 bytes per
direction, crossing the256KiB heuristic: four workers plus one mux per
direction occupy ten row-major cores, including `(8,0)` and `(9,0)` outside
semaphore coverage. Kernel waits there can consume uninitialized values and
read the gathered output before its remote writes complete.

| Artifact name under `logs/` | Intervention and observed result | Verdict |
| --- | --- | --- |
| `ag_boundary_layer0` | Exact real-source model/isolated gather checks. First corrupt boundary is pre-down MLP gather on rank3: E1 max difference1.58456e29; E2 max difference0.4281005859375. Smaller gathers exact. Isolated first eager also fails; second eager and three traces pass. Exit1. | Eager CCL corruption localized. Trace and original producer lifetime are unnecessary. |
| `ag_boundary_core_grid_layer0` | Intended core-grid restriction initially used singular Python keyword `sub_core_grid`; nanobind requires plural `sub_core_grids`. Binding error, exit1; typo corrected. | No numerical conclusion. |
| `ag_boundary_core_grid_layer0_v2` | Restrict automatic AG placement to original initialized8x8 cores; keep four workers, same shapes/dtypes/links. Every model gather, E1/E2, isolated eager and three replay checks exact. Exit0. | Predicted coverage correction verified through worker placement. |
| `ag_boundary_full_grid_layer0` | Expand semaphore initialization to actual11x10; retain original worker placement/count. Every model gather, E1/E2, isolated eager and three replay checks exact. Exit0. | Independent coverage correction verified through allocation. |

The post-projection output gather was not the corrupting operation: it exactly
broadcast already-corrupt local down results, explaining why final replicas all
matched despite an initial rank3 error. No precision change, tile-padding
adaptation, retained source-buffer workaround, or semaphore reset between
replays was required.

Exact diagnostic commands from provenance:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --length 2048
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --gather-core-grid
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_ag_diagnostic --layer 0 --semaphore-full-grid
```

The latter commands use default length2048. Each named artifact's
`.provenance.json` and `.sources.json.gz` preserve exact source/intervention,
command, environment, runtime hashes, return code and console archive. These
controls predate integration; current default construction already covers the
full device grid.

## Fix and original-command verification

The coordinator added model-local `MeshCCLManager(CCLManager)` in
`tt/multichip_decoder.py`. Its `_init_subdevice` derives `ccl_cores` from
`mesh_device.compute_with_storage_grid_size()` before inherited semaphore
initialization. `MultichipDecoder.from_state_dict` constructs this manager.
The generic GPT-OSS manager and all candidate implementations remain unchanged.
Full coverage also supports explicit fused-gather core offsets without forcing
different worker algorithms.

Completed uninstrumented reruns below all exit0, with exact E1/T1/E2 and passing
local-state checks. Each runs `timeout 240 python_env/bin/python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048`
plus the listed arguments; exact absolute interpreter paths are in provenance.

| Artifact | Remaining arguments | Prefill PCC | Decode PCC |
| --- | --- | ---: | ---: |
| `cclfixed_ag_mm_replicated_layer0` | `--layer 0 --variant ag_mm --residual replicated` | 0.9999630806578761 | 0.9999798967823771 |
| `cclfixed_cumulative_ag_mm_replicated_layer0` | `--layer 0 --variant cumulative_ag_mm --residual replicated` | 0.9999630806578761 | 0.9999798967823771 |
| `cclfixed_ag_mm_replicated_layer3` | `--layer 3 --variant ag_mm --residual replicated` | 0.9999772466069937 | 0.9996948101965183 |
| `cclfixed_cumulative_fused_ag_mm_replicated_layer0` | `--layer 0 --variant cumulative_fused_ag_mm --residual replicated` | 0.9999630806578761 | 0.9999776459276569 |
| `cclfixed_cumulative_fused_norm_ag_mm_sharded_layer0` | `--layer 0 --variant cumulative_fused_norm_ag_mm --residual sharded` | 0.9999220235537754 | 0.9999655849013239 |

The separate fused-norm failure had trace max difference0.0625 with the old
manager. Its fused AG-MM explicitly places CCL workers at offset`(0,8)`, outside
the old8x8 semaphore grid. It now passes unchanged. Cached `_gathered_norm`
lifetime was not changed or needed to fix that failure.

Additional completed `cclfixed_async{1,2}_{replicated,sharded}_layer0` controls
run packed GDN/shared MLP with `--collective async --async-links {1,2}`. All four
exit0 with exact E1/T1/E2 and valid state. Replicated decode PCC is
0.9999873674853811; sharded decode PCC is0.9999760449677115 for either link count.
The older two-link corruption workaround described experiments with incomplete
semaphore coverage; these new passes should not be described as proof of a
remaining inherent two-link kernel failure. Link-policy performance selection
belongs to the coordinator's stage measurements.

## Final status and limits

The originally reported corruption is fixed with two independent focused
controls and original uninstrumented failing-command verification. Related
fused-norm failure also passes after the same fix. No performance improvement
is claimed in this repair report.

The diagnostic passes `python -m py_compile` and Black with
`--target-version py310 --check`. This repair is Python-only and requires no
build; unrelated native reader build evidence belongs to its separate report.
Coordinator watcher verification and any broader matrix continuation remain
separate evidence, not implied by these tests.

Source review found a focused test gap for retained fused-norm scratch:
the current probe checks E1/T1/E2, then times160 later replays without reading
their output. It does not prove correctness of a replay after E2. `_gathered_norm`
is discarded before the next eager projection, and replay records its own
producer/consumer sequence, so no separate stale-cache bug is established.
The minimal follow-up is restore→T2→read/state comparison after E2; see AutoDebug
for the allocator and cache-lifetime analysis. Preserve all existing gates.
