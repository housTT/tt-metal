# AutoDebug: TP4 sharded decode corruption

Inspection and host-artifact analysis, 2026-09-05. Applied AutoFix, AutoDebug, and
TT Enable Tracing. This investigator did not import TTNN, access devices, run a
reproduction, or edit implementation code. **The eager output is already severely
corrupted; no implementation root cause has been verified.**

## Evidence

Original command, from [provenance](logs/sharded_linear_v1.provenance.json):

```bash
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --residual sharded --collective async
```

The [original log](logs/sharded_linear_v1.log.gz) reports passing single-chip exact
eager/restored-trace equality, then completes TP4 prefill and fails the same
equality assertion after TP4 replay. It does **not** establish TP4 prefill
correctness: baseline/TP4 PCC checks are after that assertion and never execute.
Hardware provenance is the parent's four Blackhole chips on P300c boards,
1x4 mesh, ring fabric. The test defaults to batch 1, 128 prefill tokens, and
decode position 128.

The [diagnostic log](logs/sharded_linear_diagnostic.log.gz) adds restored eager
repeat and trace comparisons. Both TP4 PCC/max-difference pairs are NaN; the
single-chip comparisons remain exact. The investigator originally ran host-only `torch.load(...,
map_location="cpu", weights_only=True)` on the then-current `trace_mismatch.pt`,
recording the counts below. That original tensor was subsequently overwritten
by the probe and no longer exists; these original counts survive only in this
report and the investigator command output. The later tensor is now preserved
as [sharded_link1_gather_control_trace_mismatch.pt](sharded_link1_gather_control_trace_mismatch.pt),
and **does not support the original counts below**:

| Output, BF16 [1,1,4096] | Finite | +Inf | -Inf | NaN | Values with absolute magnitude > 1e10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Eager | 3349 | 384 | 363 | 0 | 4095 |
| Restored trace | 3247 | 419 | 430 | 0 | 4095 |
| Restored repeated eager | 3195 | 451 | 450 | 0 | 4094 |

Finite extrema reach +/-3.3895313892515355e38, the BF16 maximum. Eager infinities
by 1024-channel rank partition are `[190,226,164,167]`; corruption is widespread.
The NaN *metrics* do not mean the saved tensors contain NaNs. This is not a
quantization-sized disagreement or evidence of a trace-only fault.

Both original and diagnostic archived source hashes and archive-file hashes
match their provenance. At inspection, current `tt/multichip_decoder.py` matches
the original archive (`eeabf4941b011ff9f26938d1c31d00f3fc10ba1034ec8fc01c7d6ba36a72e1a7`).
The probe has diagnostic edits relative to that archive. Use the frozen sources
when reconstructing either run.

A later [replicated TP4 control](logs/replicated_linear_trace_v2.log.gz), supplied
by the parent and read by this investigator, passes exact eager/repeat/trace
equality on all four replicas at length2048. Its prefill PCC is
0.9999630806578761 and decode PCC is 0.9999896361495468 against single-chip.
This strongly prioritizes the new distributed norm/gather and residual path
over a universal local TP4 GDN defect. It is not a matched length128 control;
use the latter if a length-dependent ambiguity remains.

## First experiment: find the first corrupted boundary

Before changing precision, collect per-rank finite counts and absolute maxima
for the actual prefill output, recurrent state, and all convolution buffers.
The decode snapshot may faithfully preserve an already-corrupted prefill state.
Also check the recorded input and uploaded A_neg/dt_bias/gamma values against
their host partitions. Then instrument one eager decode in execution order:

1. First residual norm: input, local stats, gathered stats, normalized local
   shard, gathered normalized activation.
2. GDN: packed QKV/A/B, Z, convolution output, beta/g, recurrent state before
   and after update, local `gdn_out` projection **before** reduce-scatter.
3. Reduced mixer output, first residual sum, second distributed norm, MLP
   gate/up/activation, local down projection before reduce-scatter, reduced down
   projection, final residual sum.

Read each intermediate before its explicit deallocation for eager localization.
For traced instrumentation, preallocate persistent destinations **before**
capture and copy into them with device operations; read only after replay.
Retaining a Python reference does not protect explicitly force-deallocated
buffers. Instrumentation can alter allocation/timing, so any eventual fix must
pass the uninstrumented original check.

If prefill state is corrupt, localize prefill first. A zero-state decode control
can separate that problem from the decode path. Once eager is finite and
correct, compare state-restored E1/E2/T1 plus post-decode state, then repeated
restored replays. Preserve exact equality and the >=0.995 baseline PCC gate.

## Focused hypotheses and source adjudication

### H1. Distributed norm or its gathers first corrupt activations

This is a newly added boundary, **not a proven faulty op**. `_residual_norm`
(`tt/multichip_decoder.py:198-213`) runs pre-norm statistics, async gather,
post-norm with local gamma, then async activation gather. Decode local input
is logical `[1,1,1,1024]`, padded `[1,1,32,1024]`, BF16 L1 interleaved. Statistics
are local width 32, gathered width 128; normalized shards gather to width 4096.
The `weight + 1` transformation and local gamma width 1024 match the inherited
Qwen-style norm convention.

Lowering selects the **interleaved** distributed norm factories, despite the
model's mesh-sharded residual. The post factory derives four devices from
128/32 statistics columns and uses reduction factor `logical_W*num_devices`
=4096 (`layernorm_distributed/device/layernorm_post_all_gather_program_factory.cpp:81-97,269`).
It selects `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather_metal2.cpp`,
not the neighboring layernorm kernel. No obvious width-factor or gamma-shape
contradiction was found. Do not infer a bad reduction just because it differs
from single-chip RMSNorm.

**Experiment:** Run the exact norm chain on the recorded finite token with
actual per-rank gamma. Compare local statistics with host sums of squares,
gathered statistics with the concatenated per-device source tiles, post-norm
with a host global RMSNorm, and final gathered output with concatenated local
normalized shards. If necessary, substitute only the gathered statistics or
only the full normalized activation with a host-derived device input. These
controls distinguish computation from communication and downstream consumers.

### H2. Local GDN at TP4 geometry corrupts state or output

**Demoted by the passing replicated TP4 control.** The single-chip baseline uses
different local shapes, but the replicated TP4 control exercises the same local
partitioning. TP4 has four key heads,
eight value heads, head width 128, convolution width 2048, Z width 1024, and
packed projection width 2112: QKV2048 + A32 + B32. The actual A/B fields are
eight values each, with independent tile-aligned starts 2048 and 2080
(`multichip_decoder.py:64-73,156-165`). The inherited FP32 recurrence and
BF16 convolution histories remain mutable local state.

**Experiment:** On the first bad GDN boundary, compare a single rank's exact
partitioned local module with the inherited optimized module at the same local
configuration, input, weights, and restored state. In particular inspect A_neg
(must be negative), beta, g, and state before assuming a CCL fault. Large
positive g can overflow the exponential state update, but no such values have
yet been observed. Do not promote this mechanism without those measurements.

### H3. Async CCL corrupts data or resource reuse is unsafe

`_gather` supplies two global semaphores and a barrier; reduce-scatter receives
three and a barrier. `CCLManager` creates these before capture. One sharded
decode consumes AG0, AG1, RS0, AG0, AG1, RS1; the respective barrier indices are
0,1,0,1,0,1. All three Python indices return to their starting parity. Thus
simply failing to reset the manager's Python indices does **not** explain an
eager/capture sequence mismatch.

The gathers use TILE inputs with gather widths 32 or 1024, so they avoid the
composite fallback selected for padding on the gathered dimension
(`experimental/ccl/composite_common.cpp:285-310`). L1 inputs are interleaved,
so the special Llama sharded factory is not selected. Reduce-scatter is ring,
dim3, with a width4096 input and width1024 output. The public API explicitly
allows omitted persistent buffers and allocates staging/output tensors; their
absence alone is not proof of a trace lifetime violation.

**Experiment:** If a collective is the earliest bad boundary, feed the exact
finite captured producer output to that collective alone. For gathers, compare
all output replicas byte-for-byte with host concatenation; for reduce-scatter,
compare each rank with the corresponding slice of the host sum. Test eager
repeat and restored trace, then the six-op resource-reuse sequence. A dedicated
semaphore set per call site is a focused reuse control, not an accepted fix
without an A/B result. A watcher run belongs after localization and must use
the parent's serialized hardware lane.

The 35B reference's `doc/optimized_multichip_decoder/work_log.md:740-826`
attributes its measured divergence to **deprecated `ttnn.all_gather` plus local
sum**. Its async gather with barrier and native all-reduce controls were clean.
Our path already supplies the async/barrier resources; that historical failure
does not establish the current mechanism.

### H4. Residual layout or ownership assumptions changed

The new `_block` leaves its first residual `x` interleaved and asks the sum for
32-core width-sharded output (`multichip_decoder.py:219-226`). The inherited
optimized block converts operands before summing. This deserves an exact-shape
add control if its inputs are finite and its output is the first bad boundary.
However, the proposed **mixed BF16/FP32 first-add** explanation is not supported
by the intended GDN dtype: `OptimizedDecoder._linear:255-257` explicitly forces
`gdn_out` to BF16. Verify actual lowered dtypes before applying the older mixed
sharded-add workaround. The inherited `_residual_sum:312-321` already handles
mixed dtypes when the residual itself is sharded.

No force-deallocation of the borrowed original residual is visible in
`_residual_norm`. Its local conversion/reshape views are not explicitly
deallocated, and post-norm allocates a distinct result. The A_neg/dt_bias slice
from width32 to logical width8 is not the full-logical-extent slice fast path:
`slice.cpp:129-136,398-411` routes it through an allocated primitive output.
Therefore neither is currently a demonstrated use-after-free. Validate buffer
identity and immutable-input contents if localization implicates ownership.

## Disposition

The next forked experiment should own one serialized hardware lane and perform
the finite-value localization above. State snapshots cover recurrent and all
convolution buffers, and restore copies into stable allocations. Missing token
refresh, capture-position advancement, page-table boundaries, and sampling
trace keys cannot explain this fixed-input linear-attention eager corruption.
No performance conclusion or speculative implementation fix follows from this
report. Docs-only addition; no build required.
