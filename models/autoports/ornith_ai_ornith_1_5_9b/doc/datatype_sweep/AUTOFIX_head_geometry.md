# AutoFix: selected-precision head geometry

## Starting evidence

Independent datatype-sweep review requested geometry controls at selected
BFP4/LoFi because the earlier C64/K2/R2 L1 rejection used BF16/HiFi4.
`AUTODEBUG_head_geometry.md` verifies the changed weight tile size and derives
the bounded existing geometry family before adapting the experiment.

## Hypothesis experiments

Hypothesis: the smaller BFP4 weight CB reopens K2 and other material choices.

Source experiment: inspect native validation, exact tile size, DRAM alignment,
CB allocation, reader placement, and the common two-chunk LM-head lifecycle.
Result: selected C64/K2/R2 static end is 734208, versus 1299456 at BF16.
R1/K1 also fits the necessary source bound. C16/C32 support larger K blocks
than C64. `head_geometry_plan.json` records 27 legal points, 20 source-feasible
including the baseline; source-refuted points are documented in AutoDebug.
Verdict: old BF16 allocation explanation is inapplicable to the selected
BFP4 policy. Device correctness, dynamic allocation fit, and speed remain
separate hypotheses.

Experiment artifact: `probe_head_geometry.py`, an adaptation of the prior
optimized-full-model real-hidden harness. Baseline is fixed C64/K1/R2;
precision is locked to selected BFP4/LoFi with BF16 input/output and FP32
destination accumulation. Both geometries reserve 221952 persistent L1
bytes/bank and use the same 8x4 final norm. Three readers pad each chunk to 33024
then independently trim to 32768. Receipts contain actual dtypes, geometry,
source hashes, memory, all-logit errors, top-k, boundaries, eager repeat and
trace exactness, paired head/terminal times, and score hashes. Score tensors
are not written. `--serial-traces` preserves the corrected watcher lifecycle.

Host verification commands:

```bash
python3 -m py_compile models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/probe_head_geometry.py
python_env/bin/python -m black --check --target-version py310 models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/probe_head_geometry.py
python3 models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/probe_head_geometry.py --plan --cores 64 --in0-block-w 2 --readers 2
```

All pass. A pure-Python import-and-enumeration check verifies27 legal points,
20 source-feasible points, predicted C64/K2/R2 end 734208 and C16/K8/R3
end 1009152, rejection of illegal C64/K4/K8 and C32/K8, and absence of torch
or TTNN imports. It produces `head_geometry_plan.json`.

## Final status

The focused selected-head-precision geometry investigation is complete:
**19 device passes, one controlled allocation rejection, and seven source
exclusions** resolve all 27 legal points. Parent-owned device execution produces
the measurements below; this report-writing agent uses source and receipts
only. No production edit is made by this agent, and Python/docs-only changes
need no C++ build. Retaining a complete model configuration still requires the
parent's full-model precision/geometry, qualitative, watcher/trace, default,
native-context, and end-to-end performance gates.

## C16/K1/R1 allocation control

The parent-run `geometry_c16_k1_r1_v1` fails at `head_candidate_eager`, exit1.
The factory's static end is exactly the predicted **1123328**, but the live
allocation frontier is **1023232**, a 100096-byte collision. Its baseline
eager/repeat/trace checks pass. Normal device close completes; this is a host
allocation rejection, not a hang.

First hypothesis: keeping the baseline trace resident interferes with candidate
allocation. Focused control: repeat the same selected precision, input, weights,
norm and reservation with `--serial-traces`. The parent-run
`geometry_c16_k1_r1_serial_v2` releases the baseline trace and output before
candidate eager execution, and fails at the **same 1023232 frontier**, exit1,
with normal close. Verdict: **retained-baseline-trace interference refuted**.
Both receipts also record identical 230144 bytes/bank allocated before baseline
and candidate projection; the trace does not add visible live L1 in this check.

The required common-head temporaries explain the exact frontier:

```text
frozen normalized tensor address         1301760
minus C16 input shard [32,256] BF16        16384
minus two output shards [32,2048] BF16   262144
observed dynamic frontier                1023232
```

`LMHead1D.forward` assigns `output = ttnn.linear(...)` for each chunk. Python
retains the preceding sharded `output` until the next linear call returns, so
both 131072-byte C16 output shards coexist. The failure's program-cache-hit
stack is consistent with the second same-shape chunk. The successful C64/R1/K1
control has smaller output shards and does not refute the C16 lifetime bound.
Removing that temporary would change the common module, which is outside this
fixed implementation/geometry comparison; no such change is made here.

Verdict: **C16/K1/R1 is rejected for the exact selected-precision common-head
resident/norm contract**, after the independent lifecycle control. This is
not a universal rejection of one-reader matmul or another head implementation.
`geometry_rejections.json` records both run IDs, accepted exit1 classification,
artifact/provenance/log hashes, reciprocal control evidence, source hashes,
exact frontier arithmetic, and normal-close evidence. Failed receipts remain
immutable. The remaining matrix points have since completed as recorded below.

## Completed geometry matrix and focused handoff

[All 27 point results](head_geometry_README.md) and
`head_geometry_results.json` record both head and terminal medians, numerical
groups, eager/trace checks, source exclusions, and every device receipt/hash.
All 20 source-feasible points were attempted; the C16/K1/R1 serial retry brings
the device-run count to 21. The seven excluded points are C16/R1/K2/K4/K8,
C32/R1/K2/K4, C64/R1/K2, and C16/R2/K8. Their exact violated source bounds
remain listed in the complete table. There are no untested points within the
existing C16/C32/C64 x legal K x R1/R2/R3 family.

**Focused winner: C32/K4/R2**, two logical 32768-column chunks with no added
reader padding. The paired baseline is C64/K1/R2 at the same BFP4/LoFi head
precision, fixed 8x4 normalization and 221952 persistent L1 bytes/bank.

| Paired measurement | C64/K1/R2 baseline ms | C32/K4/R2 candidate ms | Latency reduction |
| --- | ---: | ---: | ---: |
| Head including reshard/trim | 0.814679 | 0.436340 | 46.44% |
| Final norm plus head | 0.827854 | 0.449184 | 45.74% |

These medians use three alternating rounds of 64 trace replays on TP4,
four Blackhole chips on physical P300c boards. They are component timings,
not full-model throughput improvements. The nearest measured terminal
alternative is C64/K2/R3 at 0.460149 ms; C64/K2/R2 is 0.472656 ms.

All 19 successful points have finite logits, exact eager repetition, exact
trace versus eager output, exact output after every timed replay batch, and
exact head-versus-terminal output. Every candidate keeps baseline token 39102
as top-1 on this frozen row and therefore in top-5. All 248320-logit hashes are
identical within each K group across cores and readers, including independently
trimmed R3 weights: eight K1 points are bit-identical to baseline; six K2,
four K4 and one K8 points each have a shared group hash. K2/K4/K8 all differ
from K1 with max absolute error 0.125 and also differ from each other. These
component checks do not establish full-model accuracy or qualitative quality.

Verdict: **the stale BF16 geometry exclusion is refuted by device execution;
C32/K4/R2 wins the complete focused BFP4/LoFi geometry family**. The parent
now compares complete precision policies and geometries with full 32 teacher
forcing, then owns final qualitative/default/native/performance verification.
This component result alone does not select the final full-model policy or
waive any remaining stage gate.
