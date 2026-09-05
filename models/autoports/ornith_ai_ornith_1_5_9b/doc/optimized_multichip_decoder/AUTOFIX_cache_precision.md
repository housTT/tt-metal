# AutoFix: TP4 cache precision adjudication

Date: 2026-09-05. Scope: the optimized multichip decoder only. Hardware
commands below were executed serially by the parent agent on four Blackhole
chips on physical P300c boards, using the actual 1x4 TP ring. This investigator
performed source analysis, wrote the diagnostic, and reviewed saved evidence;
it did not import TTNN or access devices.

## Finding

**BFP4 cache storage fails both the unchanged 0.99 state gate and, with the
selected BFP4 QKV projection, real-user HF output accuracy. Keep K8/V8.**

The failure is isolated from page mapping, cache update packing and head
partitioning. Matched TP4 K4/V4 versus K8/V8 executions have bitwise-identical
prefill and decode K/V producers on all four ranks. Both cache dtypes pass
bitwise physical-row checks and CPU attention over the actual dequantized
cache. Their different stored values therefore demonstrate quantization
loss at the existing state bar, not an error fixed by changing the harness
reference or relaxing a threshold.

The final BFP4-QKV B32 control adds model-visible evidence: K4/V4 fails
decode users 8 and 26, on every rank, at PCC **0.993657745** and
**0.994208792**, respectively. The same producer geometry with K8/V8 passes
all users, minimum **0.996814237**, against the unchanged 0.995 gate.
Earlier BFP8-QKV batch successes do not certify this different cumulative
precision policy. No batch-dependent cache fallback is proposed.

## Controlled results

All values below are fresh TP4 measurements from this optimization pass.
These diagnostic forwards deliberately read tensors to the host and do not
provide latency evidence.

| QKV weight / geometry | Batch / prefix / mapping | K4 vs K8 state PCC | V4 vs V8 state PCC | K4/V4 minimum HF prefill / decode PCC | K8/V8 minimum HF prefill / decode PCC |
| --- | --- | ---: | ---: | --- | --- |
| BFP8, DRAM32 cores/block4/readers2 | B1 / 2048 / identity | 0.982177633 | 0.983067251 | 0.997841821 / 0.998896646 | 0.998628343 / 0.999366455 |
| BFP4, DRAM8 cores/block16/readers2 | B1 / 2048 / identity | 0.982178184 | 0.983067508 | 0.997841821 / 0.998610981 | 0.998628343 / 0.999027635 |
| BFP4, DRAM8 cores/block16/readers2 | B1 / 2048 / permuted | 0.982178184 | 0.983067508 | 0.997841821 / 0.998610981 | 0.998628343 / 0.999027635 |
| BFP4, DRAM8 cores/block16/readers2 | B32 / 96 / identity | 0.981854700 | 0.983021483 | 0.997106086 / **0.993657745** | 0.998359447 / **0.996814237** |
| BFP4, DRAM8 cores/block16/readers2 | B32 / 96 / permuted | 0.981854700 | 0.983021483 | 0.997106086 / **0.993657745** | 0.998359447 / **0.996814237** |

All five controls completed and saved their results. The final B32/permuted
run reproduces both failed users and all exact cache/SDPA checks; changing
physical page assignment does not change the numerical outcome.

The rest of the policy is held fixed: raw pinned HF weights, LoFi
projections, BFP4 MLP with DRAM8-core blocks8/8/6 and two readers, native
TP4 collectives, BF16 replicated residual, one local KV head per rank,
head256, tile/interleaved DRAM cache with page64. Explicit cache allocation
dtype is the only difference within each matched pair. The existing
`ProductionCache4` and `ProductionQkv4Cache4` setup classes exercise the
production implementation; the diagnostic adds no production fallback.

## Verify/refute results

1. **Different input projection or TP partition:** refuted for the matched
   comparisons. Every rank's raw K/V producer digest matches exactly
   between cache policies, during both prefill and decode.
2. **Wrong cache fill/page mapping:** refuted for all five controls.
   Reconstructing every physical page from the already device-quantized
   fill input agrees bit-for-bit with all four actual caches. The B1
   and B32 identity/permuted mappings produce identical model outputs.
3. **Fused update packing or writes to other users/pages:** refuted.
   Independent unfused updates into cloned original caches equal fused
   updates bit-for-bit. Independently reconstructing only the mapped
   decode row also agrees exactly, proving untouched rows remain intact.
4. **SDPA consumes the wrong cache or user:** refuted for the measured
   target shapes at the existing diagnostic bar. CPU attention over each
   actual rank's dequantized cache and captured query agrees for every
   user at PCC >= 0.999. B32 minimums are 0.999551172 for K4/V4 and
   0.999232139 for K8/V8.
5. **BFP8 versus BFP4 decode QKV would remove the cache-state failure:**
   refuted. Both geometries have K and V state PCC around 0.982–0.983,
   below 0.99. Logical-used-row scores also fail; unused zero cache pages
   do not explain the result.
6. **A mixed cache retains an eligible reduced payload:** ruled out at the
   unchanged state bar. K4 and V4 each independently fail with identical
   producers, so neither K4/V8 nor K8/V4 can satisfy both state gates by
   changing the other cache dtype. Do not invoke mixed dtypes through
   `paged_fused_update_cache`.

## Harness interpretation and preserved failures

The initial ordinary probe's error label, `head-local state/cache partition
mismatch`, was too specific: it compared TP4 BFP4 state with a single-chip
BFP8 reference. The matched TP4 diagnostic now proves actual precision
loss without changing that original threshold or relabeling its failure.

The first diagnostic's `cache_implementation_checks_passed` aggregate also
included HF output checks. Its B32 value is false solely because the
BFP4 output accuracy failed; all exact cache and SDPA checks passed.
The archived JSON and source manifests remain unchanged. A label-only
correction for future runs separates `cache_implementation_checks_passed`
and `output_accuracy_gate_passed`. Overall status still requires both,
plus `state_precision_gate_passed`; no assertion or threshold is removed.

Every run exits 1 intentionally after saving its complete
observations because BFP4 fails acceptance. These are measured rejection
artifacts, not successful optimization runs or first-error dismissals.
No device reset, build or recovery is part of this investigation.

## Commands and provenance

Initial source investigation and proposed commands are recorded in
[AUTODEBUG_cache_precision.md](AUTODEBUG_cache_precision.md). The parent
executed the commands captured in these immutable manifests:

- [BFP8 QKV, B1](logs/cache_precision_qkv8_b1.provenance.json)
- [BFP4 QKV, B1 identity](logs/cache_precision_qkv4_b1.provenance.json)
- [BFP4 QKV, B1 permuted](logs/cache_precision_qkv4_b1_permuted.provenance.json)
- [BFP4 QKV, B32 identity](logs/cache_precision_qkv4_b32.provenance.json)
- [BFP4 QKV, B32 permuted](logs/cache_precision_qkv4_b32_permuted.provenance.json)

Each has corresponding `.log`, `.log.gz` and `.sources.json.gz` files,
including native-source and installed-runtime provenance. The main JSON
artifacts have the same run stems in this directory. They retain every
per-rank/per-user score, exact-check count and producer digest.
The compact [JSON summary](cache_precision_summary.json) and
[CSV table](cache_precision_summary.csv) cover all five runs and ten
cache-policy executions. The summary recomputes implementation status from
the exact/SDPA leaf checks and preserves the earlier aggregate field under
an explicitly historical key; it does not edit original results.

The initial whole-layer `production_cache4_layer3` measurement was
0.272232312 ms versus the same family's K8/V8 0.272172 ms; thus that run
did not establish a decode speed benefit either. These are candidate
numbers, not final optimized-stage numbers. Keeping K8/V8 preserves the
cache layout and existing native-context allocation contract. Final
default performance, watcher, context and stage-review gates remain the
parent’s responsibility.
