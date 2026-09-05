# AutoFix: batch-32 convolution state corruption

Status: **original fixed-slot/page-permutation failure fixed**. No decoder
strategy, state/cache/activation/weight dtype or fidelity was changed.

Follow-up: the small cross-slot logit discrepancy described below was later
localized and reproduced as the existing native Q/K RMSNorm cyclic partial
reduction order. See `AUTOFIX_norm_row_order.md` for the persistent traced
boundaries, frozen same-op controls and CPU oracle; its evidence supersedes
this report's initial localization limitation.

Starting evidence is `AUTODEBUG_batch32.md` and the immutable
`logs/trace_b32_watcher_fixed.*` failure artifacts.

## Verified hypothesis and repair

The first diagnostic localized disagreement before decode: equal prompts had
exact terminal inputs, prefill logits and FP32 recurrent states, but writing
later slots changed the earlier slot's BF16 convolution history. All three
convolution buffers were affected on all four ranks.

`ttnn.where` selects its SFPU data path using the condition dtype (see
`ttnn/cpp/ttnn/operations/eltwise/ternary/device/ternary_op_utils.cpp`). The
model used an FP32 condition with BF16 branch/output buffers both during
prefill handoff and inactive-row restoration. A focused control used the real
allocated convolution shape `[32,1,2048]`, exact selected TP4 mesh and
in-place output contract. With FP32 condition, 65,352 of 65,536 elements were
wrong on ranks 0–2 and 65,412 on rank 3, with enormous finite values. With a
BF16 condition, every element was bitwise equal to the CPU `where` result on
every rank. This confirms a dtype contract violation, not precision drift.

The retained repair is:

- `_transfer_slot` uploads the condition using `dst.dtype`.
- `active_conv` is a BF16 condition; `active_recurrent` remains FP32.
- Generator `_write_positions` uploads each condition using its target dtype
  (coordinated with the readiness repair agent).

FP32 recurrent state, BF16 convolution history, BFP8 paged KV and BF16 cache
update payload remain unchanged. There is no host cache/state fallback.

## Experiments and evidence

All commands use this environment from the repository root:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
```

The `record_run` wrapper records exact commands, environment, log hashes and
immutable source archives in `logs/<name>.*`.

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run batch32_localize_v1 \
timeout 300 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch32 \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/batch32_localize_v1.json

python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run batch32_where_control \
timeout 120 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch32 \
--where-only --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/batch32_where_control.json

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run batch32_localize_fixed_v2 \
timeout 180 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch32 \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/batch32_localize_fixed_v2.json

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run trace_b32_masks_fixed \
timeout 240 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.trace_contract \
--batch 32 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/trace_b32_masks_fixed.json
```

The first diagnostic exposed an additional **probe-owned** lifetime error:
it held returned prefill logits across the next model trace replay. The
allocation tracker identified precisely that live tensor on the second case;
the probe now explicitly frees it after its blocking snapshot. No corruption
waiver was added. The first fixed retry (`batch32_localize_fixed`) stopped
before inference because it preceded the matching generator host-upload dtype
edit. Both failed logs remain intact. Neither failure required a device reset.

`batch32_localize_fixed_v2.json` and compact
`batch32_localize_summary.json` establish:

- Earlier slot 0 remains exact after handoffs into slots 3, 6 and 30.
- Duplicate slots 0/3/6/27/30 have exact recurrent and all three convolution
  histories on all four ranks after prefill and two forced decode steps.
- Original A, repeated A2 and column-permuted B have bitwise identical
  selected-slot logits and hybrid states at every recorded boundary.
- Both forced sampling traces exactly match CPU argmax of those same full
  vocabulary logits on all four token replicas.
- Duplicate-slot decode logits can differ by at most 0.0625 between different
  slots, with identical greedy winners; the same slot is exact across A/A2/B.
  This residual small cross-slot difference was not further localized and is
  not claimed as bitwise logit equivalence or an accuracy result.

`batch32_logit_rounding_observations.json` quantifies the retained observations:
slot 0 top logits are 4.75 and 4.59375 at the two decode steps, with BF16
spacing 0.03125 at both values. The largest cross-slot discrepancy is 0.0625,
twice that spacing; top-two gaps are 0.15625 and 0.5. This scale comparison
does not prove rounding is the cause. Full logits were not persisted, so PCC
and maximum absolute logit magnitude cannot be reconstructed from this run.
The probe now records both for any future invocation. No diagnostic copies of
intermediate decoder outputs were recorded: the remaining possible interval
is from linear-decoder output through full attention and terminal projection.
The exact hybrid states and A/A2/B results rule out the original handoff
corruption and mapping-dependent behavior, but do not identify an individual
operation behind the small cross-slot difference.

`trace_b32_masks_fixed.json` has `pass: true`: 31 active mixed-length prompts
plus one inactive row, all 32 active slots, exact duplicate-prompt token
sequences, arbitrary physical page permutation, changed/unchanged page-table
copy behavior, persistent device feedback and positions, inactive hybrid
state preservation, changed token/position logits, seeded rank equality and
seeded/greedy reuse all pass. Worker watcher and allocation tracking were
enabled together; Ethernet watcher remains explicitly disabled under the
existing hardware contract. No profiler was active.

## Remaining stage work

These reduced `[0,3]` layers prove the repaired state/trace contract. They do
not substitute for the parent stage's full-stack accuracy, batch and
qualitative checks. The parent must rerun applicable B4 and full-model batch
checks using these condition fixes. Hardware ownership was returned after
the passing worker-watcher run closed all devices normally.
