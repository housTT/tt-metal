# AutoFix: batch-32 full-attention accuracy

## Starting evidence

`AUTODEBUG_batch32.md` investigated the original short-contract failure after
53 passes: layer 3, batch 32, first trace replay, HF per-user PCC
`0.9948051904765759 < 0.995`. The input was 63 recorded activation rows selected
with seed31; decode inputs use seeds3100–3102 and positions63–65. No threshold,
activation selection, reference, cache dtype, or test fixture was changed.

The repair agent held the exclusive hardware lane. Every listed hardware
command finished with exit0 and closed its mesh before the next command.
No reset, process kill, triage, or hardware recovery was needed in this repair.

## Hypothesis experiments

1. **Trace or device-clone snapshots corrupt local KV state: refuted.**
   `logs/batch32_localize_v1.log.gz` records an actual 1×1 optimized decoder,
   then the 1×4 multichip decoder, on exactly the failing input. Every rank-local
   K/V snapshot hash remained unchanged after eager, capture, and all three
   replays. Restored cache hashes exactly matched the snapshots before capture
   and replay. First eager, repeated eager, and first trace outputs were bitwise
   identical on each rank. All four output replicas were identical and finite.
   The failing coordinate is user31, step0, position63, physical block992,
   row63, SDPA chunk start0. Steps1/2 already pass. There is no trace-only bug.

2. **Unavoidable baseline precision failure: refuted.**
   The genuine single-chip control reproduced prior release aggregate PCC
   `0.9984789759395264`; its worst user31 clears the gate at
   `0.9956847306477445`. The multichip control gives aggregate
   `0.9984563599639582` and user31 `0.9948051904765759`.

3. **QKVG decode accumulation geometry introduces the first excess error:
   verified, fixed.** Input to QKVG is bitwise equal to baseline. With multichip
   K-block4, reconstructed packed QKVG differs (PCC0.9999749615293192,
   max absolute difference0.1875). Changing only QKVG's decode `in0_block_w`
   from4 to2 makes its global reconstructed output **bitwise identical** to the
   baseline. The baseline also uses block2, although its output worker and
   reader counts differ. With block2, user31's final HF PCC improves to
   `0.995282624343465`; all three steps clear the unchanged per-user gate:

   | Step | Position | Minimum per-user HF PCC | Worst user |
   | --- | --- | --- | --- |
   | 0 | 63 | 0.995282624343465 | 31 |
   | 1 | 64 | 0.9977741891703988 | 5 |
   | 2 | 65 | 0.9963819471520426 | 10 |

   `logs/batch32_qkvg_block2.log.gz` preserves the single-variable experiment.
   `batch32_boundary_comparison.json` compares every captured projection input
   and output for both configurations. `batch32_diagnostic.pt` and
   `batch32_qkvg_block2.pt` contain references, all-rank outputs, and boundary
   tensors; the first artifact is the historical block4 control. Source archives
   in each run's provenance preserve the exact pre-fix diagnostic implementation.

The production change is only `MeshConfig.local.role_configs['qkvg']`:
`cores=8, block_w=2, readers=1`. Other roles retain their prior defaults.
No precision change was required. The policy ledger is unchanged:
BFP4 projection weights, LoFi projection math, BF16 activations and CCL payload,
BFP8 K/V cache, one local KV head with head dimension256, page size64,
32 rounded physical pages/user, identity disjoint 32-user page table,
`paged_fused_update_cache`, and paged SDPA decode. Native all-reduce retains
its independently validated two-link ring implementation. Async RS/AG remains
one-link. No claim of a cache kernel bug or communication corruption is made.

## Verification and performance

All commands below use:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 \
python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py NAME timeout SECONDS COMMAND
```

Exact command arrays, source hashes, compressed sources, environment, return
codes, and console logs are in the corresponding `logs/NAME.*` provenance.

| Run name | Command after timeout | Result |
| --- | --- | --- |
| `batch32_localize_v1` | `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batch32_diagnostic` (300s; historical default name) | Exit0; baseline passes, original TP4 numerical miss reproduced, exact eager/trace and state checks |
| `batch32_qkvg_block2` | `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batch32_diagnostic --only tp4 --qkvg-block 2 --name batch32_qkvg_block2` (180s) | Exit0; localized QKVG becomes baseline-exact; all per-user PCCs pass |
| `batch32_fixed_original` | `python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -k 'test_traced_decode_pcc and 32 and full_attention' -x -q -s` (180s) | 1 passed, 96 deselected; unchanged original test |
| `batch32_fixed_watcher` | Same pytest command (180s), with `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` | 1 passed, exit0; watcher checked all four devices, no reported violation |
| `batch32_fixed_full_latency` | `python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --length 2048` (180s) | Exact eager/replay; prefill/decode and local cache PCC pass |
| `batch32_block4_full_latency_control` | Same probe plus `--local-config '{"role_configs":{"qkvg":{"cores":8,"block_w":4,"readers":1}}}'` (180s) | Batch1 control passes; remains rejected by batch32 evidence |

Watcher evidence is scoped: ETH instrumentation was disabled using the existing
stage workaround; this is not an all-features watcher claim. Raw watcher,
kernel-name, and kernel-ELF-path files are preserved as
`logs/batch32_fixed_watcher_{watcher.log,kernel_names.txt,kernel_elf_paths.txt}.gz`.
No profiler was enabled in the watcher run.

The nearby warmed batch1/2048 full-attention probe measured five windows of
32 nonblocking trace replays each, reporting median per-replay time:

| Configuration | Decode ms | Prefill ms |
| --- | --- | --- |
| Fixed TP4 QKVG block2 | 0.3335812835 | 2.772639040 |
| Genuine single-chip paired with fixed TP4 | 0.4266325341 | 5.413125036 |
| TP4 block4 control | 0.3144631555 | 2.907635993 |
| Genuine single-chip paired with block4 control | 0.4271172511 | 5.441008019 |

Block2 costs approximately6.1% decode latency relative to the invalid block4
configuration in these adjacent runs. Fixed TP4 still gives approximately1.279×
decode speedup versus its paired baseline. Prefill does not use this decode
role geometry; its timing variation is not attributed to the fix. Fixed prefill
PCC is0.9999772466, decode PCC0.9999933759 against the single-chip baseline;
local reconstructed K/V PCC is approximately1.0. These are narrow repair
measurements, not the final stage performance report.

## Final status

**Fixed with unchanged precision and acceptance thresholds.** Original narrow
check and separately instrumented watcher check pass. Durable diagnostic
`tests/multichip_batch32_diagnostic.py` now requires a unique `--name`, refuses
artifact overwrite, defaults to the fixed block2 geometry, and supports an
explicit `--qkvg-block 4` control. For numerical A/B runs it records failing PCCs
without asserting the HF gate, so exit0 alone is not an accuracy pass; inspect
scores or use the original pytest gate.

Root must rerun the remaining short/native/stack gates and final stage review
on the final tuned implementation. Other role geometry is outside this narrow
repair. Future QKVG tuning must retain this batch32 per-user gate; a faster
batch1 result alone cannot establish an acceptable configuration.

## Followup: retest after geometry sweep

The parent applied the isolated geometry sweep's fastest candidate per role.
`logs/contracts_geometry_v1.log.gz` then failed the same batch32 trace gate
(after 53 passes), at `0.9948033411587486`. This section supersedes the preceding
final configuration description while preserving its historical evidence.

Before this followup, recovery ran serially and waited for actual completion:
`timeout 180 tt-smi -r` exited0 and reset PCI devices0–3;
`timeout 60 tt-smi -ls --local` exited0 and listed all four P300c Blackhole chips;
`logs/batch32_geometry_recovery_mesh.log.gz` records a successful ring 1×4
open/close, exit0, `MESH_SMOKE_OK`. No second reset, lock cleanup, process kill,
or triage was needed. The repair agent then held the exclusive hardware lane;
all diagnostic and verification commands below exited0 and closed devices.
Diagnostic exit0 still means the experiment completed, not that HF PCC passed.

The tuned starting geometry was QKVG32/B2, o_proj4/B8, gate8/B16, up8/B16,
down8/B12 (notation: activation cores/K-block width; all use reader1).
The diagnostic's new `--role-configs` JSON option merges only the specified
roles into the current defaults, preserving all other policies and geometry.
All controls below use the identical recorded inputs and unchanged precision
ledger from the first repair. Each run records three traced steps, per-user
scores, exact eager/replay comparisons, and immutable/exactly restored cache
snapshots.

| Run suffix after `batch32_geometry_` | Single-variable or related-group change from tuned starting geometry | Step0 minimum HF PCC | Verdict |
| --- | --- | --- | --- |
| `localize` | None | 0.9948033411587486 | Reproduces regression |
| `o4_control` | o_proj8/B4 | 0.9947158806094987 | Not sufficient; discarded |
| `down4_control` | down8/B4 | 0.9949262228959062 | Not sufficient; discarded |
| `gateup4_control` | gate8/B4 and up8/B4 | 0.9952901028547718 | Pass, but two changes unnecessary |
| `gate4_control` | gate8/B4 only | 0.9951210523574053 | Pass; retained minimal repair |
| `up4_control` | up8/B4 only | 0.9950633371685013 | Pass; alternative, not retained |
| `gate8_control` | gate8/B8 only | 0.9949539202626506 | Faster intermediate block still fails; discarded |

QKVG input/output and o_proj input in the tuned failing run are bitwise equal to
the earlier passing run. The first newly differing boundary is o_proj output,
but the o_proj-only rollback experiment proves that first difference alone is
not sufficient to explain or repair the final threshold miss. The gate-only
experiment localizes a sufficient intervention to MLP gate accumulation
geometry. This is a numerical envelope issue, not a trace, cache, or CCL fault.
No claim is made that every isolated output difference is itself incorrect.

After retaining gate8/B4, the requested larger legal QKVG controls were also
run, each changing only QKVG from the accepted candidate:

| Run suffix | QKVG geometry | Step0 minimum HF PCC | Verdict |
| --- | --- | --- | --- |
| `q8_gate4` | 4 cores / block8 | 0.9943592957784508 | Fail |
| `q16_gate4` | 4 cores / block16 | 0.9941451599948757 | Fail |
| `q32_gate4` | 4 cores / block32 | 0.9944324137249841 | Fail |

These controls were finite and exact between eager/replay. They reject the
larger-block candidates with actual batch32 HF evidence; rejection is not
inferred from block4's earlier failure or from non-bitwise isolated outputs.

The retained production change from the tuned setup is **only gate_proj
block16→4**, keeping its8 activation cores and reader1. QKVG remains32/B2,
o_proj4/B8, up8/B16, and down8/B12. No batch-dependent dispatch logic or
precision change was added. The parent sweep measured isolated gate projection
at37.64µs for block4 versus32.05µs for block16; this is approximately5.6µs
isolated cost, not a measured whole-decoder latency delta. Gate-only and up-only
repairs have similar isolated costs; gate-only retained the larger measured
HF margin of those single-role alternatives. Full-decoder measurements remain
with the parent's final performance pass.

Verification ran the original failing test together with its nearby cases:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 \
python models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder/record_run.py \
  batch32_geometry_fixed_nearby timeout 240 python -m pytest \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py \
  -k 'test_traced_decode_pcc or test_batched_prefill_decode_pcc' -x -q -s
```

Result: **10 passed, 87 deselected**, both layer kinds and all existing traced
and batched-eager parametrizations, including batch32 full attention. The
parent retains responsibility for remaining contracts and final watcher/profile
runs on the final path. The earlier watcher evidence covers the previous
configuration; it is not presented as a watcher run for this followup change.

Every diagnostic used `record_run.py NAME timeout 180 python -m
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batch32_diagnostic --only
tp4 --name NAME`, plus the exact `--role-configs` argument stored in its
`logs/NAME.provenance.json`. Corresponding `NAME.pt` files retain the boundary
and output tensors. `batch32_geometry_comparison.json` summarizes all ten
controls' per-step minimum HF PCC and projection differences against the
previous passing run. Source archives preserve the defaults used by each
experiment. No experimental configuration other than the proven gate-only
repair was kept in production.
