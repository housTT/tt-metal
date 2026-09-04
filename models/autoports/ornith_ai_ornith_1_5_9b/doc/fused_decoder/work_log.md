# Fused decoder work log

Stage: fused-decoder, in progress. Model ornith-ai/Ornith-1.5-9B, pinned revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53.
Initial clean branch hous/ornith-1.5-9b at ae18ba18bd. Hardware: single Blackhole chip on P300c boards (p150 profile); four chips visible in bounded tt-smi listing. Hardware commands serialized under supervisor lock. No later stages started.

Read graph-fusing, tt-device-usage, stage-review, autofix and autoport AGENTS; inspected prior 35B functional decoder, RoPE, reference and tests. Preserve native context 262144, BF16 paged cache, FP32 recurrent state, arbitrary logical lengths and batch 1–32.

Functional committed profiler baseline: prefill linear/full 40.075804/35.104062 ms kernel sums; traced decode 1.60966675/1.42231950 ms. Fresh comparable wall latency and correctness controls follow. Prior claims are not new fused evidence.

## Initial full-attention experiments

`record_run.py` preserves exact argv/env, initial source hashes, exit status and log hashes. Initial wrapper root path was corrected before the baseline launched. All device tests run with `ORNITH_WEIGHTS=real`, `OMP_NUM_THREADS=8`, persistent `TORCHINDUCTOR_CACHE_DIR`.

| Evidence log | Result / interpretation |
| --- | --- |
| baseline | 6 passing functional real-weight/perf cases; linear wall prefill40.63ms, trace1.624ms; full values in raw log |
| dedicated_v1 | Linear cases pass; partial RoPE decode fails concat because rotary op returns padded logical rows |
| dedicated_v2 | Trim rotary output to logical shape; 12 passing real-weight/perf/traced-batch cases |
| dedicated_pair | Five measured samples: linear40.635→40.246ms prefill,1.624→1.620ms trace; full35.428→34.892ms prefill,1.512→1.460ms trace. PCC effectively1.0 linear; full>.99996 |
| packed_attention_v1 | Full Q/K/V/gate host packing + dedicated head splits;35.444→32.230ms prefill,1.510→1.315ms trace; PCC>.99996 |
| packed_matrix | Harness assertion missed tests constructing via class factory directly; implementation cases passed before teardown error |
| packed_matrix_v2 | Fixture now instruments factory and bans functional `_block`;69 selected contract tests pass |
| native_attention_v1 | Test candidate incorrectly sliced `ttnn.Shape`; corrected to list(shape) |
| native_attention_v2 | Native decode head layout + batch-as-sequence HF rotary: trace1.512→1.288ms; output/cache equivalence passes |
| full_width_attention_v1 | Missing helper import in copied probe; corrected |
| full_width_attention_v2 | Full-width rotary with setup Q/K/norm permutation passes (K cache inverse-permuted for comparison), but trace1.314ms loses to native partial1.288ms. Rejected, no permutation in final cache |
| prefill_attention_v1 | Batched paged fills + tiled setup prefill tables + dedicated head concatenation: prefill31.503ms, trace1.288ms; equivalence passes |
| cache_attention_v1 | Keep V head-split shard, move K to disjoint cores, joint K/V update: trace1.279ms, equivalence passes |
| native_cache_matrix | Accepted attention path39 selected nonlong contract cases pass, including ragged B4/13/32, trace and changed-page-table, continuation, host-guard and determinism |

Paired timing uses 2048-token prompt, decode position2048, seven windows of32 replays (first two discarded); prefill seven runs (first two discarded). Standard inherited perf uses128-token prompt/decode position128, so those timing regimes are not mixed. Layer0 state advances on replay; timing setup restores state outside measured windows. Paired tests compare prefill, decode, replay, and final32-replay outputs, plus recurrent/conv or paged state after one replay. Eager/replay outputs are bit-identical from restored state.

## AutoFix linear diagnosis

`AUTODEBUG_linear.md` is a fresh xhigh inspection-only report. Hypotheses are not accepted fixes. Experimental classes are isolated in `tests/linear_fusion_candidates.py`, adapted from the pinned35B mechanisms with9B weights/config. Main owns every hardware run.

`flat_gdn_v1`: raw flat Q/K/V + internal L2/GVA + head-major output passes real-weight paired PCC (.999973 prefill,.999829 decode,.999880 final32-replay output) and state PCC; prefill40.624→36.534ms. Decode unchanged at1.620ms. Dedicated pre-norm core amplitude probe and broader HF coverage remain required before final promotion.

### Remaining attention and MLP candidates

`native_sharded_rope_v1` used partial64 HF native decode shards and achieved
1.266286 ms traced decode; `native_sharded_rope_matrix` passed the selected
real/batched/traced full-attention matrix. `llama_qk_rope_v1` passed the packed
interleaved-basis adaptation, including inverse-permuted K-state comparison,
but measured 1.267136 ms. `joint_rope_v1` passed but cost1.289323 ms.
`concat_decode_v1` and `concat_decode_sharded_v1` passed at1.279936/1.308239 ms,
so dedicated concat plus extra movement loses to the native output reshape.

`packed_mlp_v1` passed both kinds and cut prefill about8 ms (linear32.206560,
full23.473765 ms), but trace cost rose roughly3 us versus separate linears.
A mode-specific combination is required. `matmul_silu_v1` passed, but the default
matmul activation argument lowers to a separate unary; an explicit epilogue
adaptation remains under investigation.

`mixed_silu_fp32_v1` and its operand-A control both passed after explicitly
casting Z toFP32; decode1.505951/1.506642 ms is slower than separate BF16 SiLU
plus mixed multiply (Arithmetic1.503353). Both direct mixed-input attempts had
failed, including NaN for activated operand A. The successful cast adaptation
establishes a supported implementation and a measured rejection.

`softplus_localization_v1` isolated the gate delta: real prefill A+bias has5272
values below-5; standalone FP32 Softplus preserves all positive tails, while
BinaryNG fused Softplus zeros all5272. FP32 standalone relative L2 error against
Torch is5.71e-7 versus1.57e-4 fused. Boundary and one-token controls agree.
Source inspection identifies the missing INP_FLOAT32 define in fused unary
codegen. Retain standalone FP32 Softplus to preserve recurrent decay; the
whole-layer PCC above threshold alone would hide changed long-tail semantics.
Full source snapshots and exact argv/environment are attached to each newer
run's provenance JSON; earlier runs retain source hashes and candidate code.

## Later AutoFix resolutions and coherent selection (2026-09-04)

The following entries resolve the earlier open hypotheses without changing their
historical observations. The complete pattern-by-pattern inventory, source
contracts, isolated measurements and rejection/adaptation evidence are in
[patterns.md](patterns.md). All hardware measurements were run by the coordinator;
this evidence consolidation is docs-only. Scope remains fused decoder, with no
datatype sweep, general performance tuning or later-stage work.

The initially promising flat DeltaNet path failed the per-user B32 HF traced
check: [FlatGDN](logs/flat_gdn_batch32_trace.log.gz) PCC0.99420974. The supported
hybrid retains functional rank4 Q/K normalization while using flat V and
head-major output; [HybridNorm](logs/hybrid_norm_batch32_trace.log.gz) passes the
same case. [HybridArithmetic](logs/hybrid_arithmetic_batch32_trace.log.gz) and
[HybridKdaNorm](logs/hybrid_kda_norm_batch32_trace.log.gz) also pass. Thus the early
B1 FlatGDN result above did not become a blanket accuracy claim.

KDA conv+SiLU first needed a program-config namespace repair. Its valid
[B1 retry](logs/kda_conv_gdn_v2.log.gz) achieved34.097302 ms prefill, but the
[broader matrix](logs/kda_conv_matrix.log.gz) failed B32 trace PCC0.99315313 after22
passes. [HybridConvOnly](logs/hybrid_conv_only_batch32_trace.log.gz) still failed
0.99257131, isolating conv independently of flat Q/K normalization. The corrected
[conv localization](logs/conv_localization_v2.log.gz) compares identical real BF16
projected QKV/taps/history: relative L2 versus functional0.00481486, max error
0.0625. Disabling FP32 destination accumulation worsens relative L2 to~0.00510.
KDA applies SiLU before its BF16 output pack, unlike the functional FIR sequence.
Its legal padded-T32 [single-token adaptation](logs/kda_conv_decode_v1.log.gz)
also passes B1 but loses decode speed at1.606864 ms. KDA conv is rejected after
these supported adaptations and localization, not after its first error.

Ordinary grouped Conv1d provides a separate supported replacement. The initial
2048-channel [attempt](logs/conv1d_gdn_v1.log.gz) could not fit L1 even after width
and height fallback. Narrowing each independent group to512 channels
[passes](logs/conv1d_narrow_v1.log.gz) at35.632926 ms prefill and passes the
[B32 trace](logs/conv1d_narrow_batch32_trace.log.gz). The1024-channel
[adaptation](logs/conv1d_1024_v1.log.gz) passes at35.472440 ms with minimum paired
PCC0.99990836 and becomes the coherent choice. Prepared weights are setup-only;
real logical row-major tails update persistent conv state. Standalone BF16 SiLU
remains because this depthwise conv path does not execute the generic activation
macro. The optional256-channel class is implemented but unmeasured;512/1024
already establish a valid replacement for this pattern.

The remaining successful linear folds combine packed QKV/A/B with separate Z,
decode Z's matmul SiLU epilogue, joint Q/K norm before repeated heads, FP32 EXP
inside state multiply, direct FP32 A+bias, and a sigmoid chain with explicit BF16
RNE before FP32 output. Dedicated KDA sigmoid-gated RMSNorm is selected for
prefill only. Independent controls retained in the ledger include all-four and
split projection packing, separate prefill projections, shared FIR rows,
row-major state tails, joint prefill normalization, scalar norm weights,
rank-one broadcast addcmul, and BF16/FP32 matmul-bias folds. Valid rank-one
addcmul1.542487 ms and norm-weight1.575975 ms lose to Arithmetic1.503353 ms.
BF16/FP32 bias folds1.501722/1.518878 ms move the bias rounding boundary and
lose to the final explicit-FP32-bias combination. The previously recorded
softplus and mixed-SiLU numerical failures remain rejected.

Full-attention selection keeps native partial HF RoPE, native head/cache layouts,
batched prefill fills and fused K/V decode updates, with the direct slice-to-L1
adapter measured at1.262636 ms in [its pair](logs/slice_l1_attention_v1.log.gz).
The full-width, fused-Llama, joint-Q/K, direct-K and dedicated decode-concat
alternatives all received legal adaptations and passed but were slower.
MLP gate/up packing is selected in prefill; decode retains separate linears and
fused binary SiLU. The initial default matmul-activation investigation is resolved:
true [explicit epilogues](logs/mlp_epilogue_v1.log.gz) pass but cost linear/full
1.622200/1.281628 ms versus the [matched control](logs/mlp_epilogue_control_v1.log.gz)
1.619908/1.279136 ms. The valid adaptation loses, so it is not retained.

The pre-cast [ConvFinalCombination](logs/conv_final_combination_v1.log.gz) passes8
paired/raw-core cases: linear40.603138→26.194263 ms prefill and
1.623650→1.492912 ms traced decode; full35.390193→23.509240 and
1.511645→1.263116 ms. The [selected matrix](logs/selected_matrix.log.gz) then
passes69 nonlong cases at22:40:29Z. These candidate results preceded the next
cast and L1 handoff changes and are not the final integrated measurements.

## Final short-prefill handoff and cast merges

The earlier NormDRAM2048-token measurement did not exercise the changed small
input path. The matched [B1/T128 handoff control](logs/short_norm_handoff_v2.log.gz)
at22:46:28Z measures25 samples after5 warm iterations: original L1→DRAM2.493768 ms,
direct DRAM2.499128 ms, and direct L1 reader input2.465545 ms. All outputs are
bit-identical. Existing TensorAccessor support permits Q/K to remain in L1.
The integrated rule chooses L1 only when `batch * physical_seq <= 512`, otherwise
DRAM, so the removed copy does not cause large-batch L1 growth. Direct DRAM is
superseded; the isolated short-prefill timing question is resolved.

[QueryCast](logs/query_cast_v1.log.gz) emits FP32 directly from BF16 Q times exact
1/128 and passes at1.484959 ms; [ValueCast](logs/value_cast_v1.log.gz) subtracts the
FP32 state read directly from BF16 V and passes at1.483691 ms. Their
[combination](logs/combined_cast_v1.log.gz) passes at1.472689 ms. The
[KeyCastChain](logs/key_cast_chain_v1.log.gz) passes at1.482004 ms: its single unary
chain multiplies by181/2048, explicitly rounds the product FP32→BF16 with RNE,
then emits FP32.181/2048 preserves the original BF16 scalar coefficient; passing
the unrounded1/sqrt128 to unary codegen would change the original arithmetic.
[AllCast](logs/all_cast_v1.log.gz) passes at1.461613 ms, minimum paired output
PCC0.99989795. These isolated candidates in
[decode_cast_candidates.py](../../tests/decode_cast_candidates.py) preserve the
fixed BF16/FP32 precision policy and caller-owned state/tensors. All three merges
were integrated only after these measured controls.

## Integrated runtime: short and native-context checks

[final_short](logs/final_short.log.gz) finishes at2026-09-04T22:55:49Z with84 passed,
9 deselected,2 warnings. [final_long](logs/final_long.log.gz) finishes at22:59:19Z
with9 passed,73 deselected,2 warnings. Both run the default integrated runtime,
with `ORNITH_FUSION_CANDIDATE` unset. Their
[short provenance](logs/final_short.provenance.json) and
[long provenance](logs/final_long.provenance.json) preserve exact commands,
environment, exit0, matching actual log hashes, and identical source maps.
Runtime SHA256 is
`3cf319b63a169574a20d318b7cdc78ecff03aba5498e07de862ec1de5fb84083`;
both corresponding source snapshots are archived beside the logs.

| Final_short pair | Functional→fused prefill ms | Functional→fused traced decode ms | Minimum output PCC |
|---|---|---|---|
| Linear0 |40.674152→26.229826|1.624296→1.461873|0.99989795|
| Full3 |35.464214→23.489323|1.510768→1.263002|0.99996715|

These comparable2048-token paired medians reduce prefill35.51%/33.77% and decode
10.00%/16.40% for linear/full respectively. The six final raw-core probes at
lengths1/2/3/127/2047/2049 each include32 distinct decode inputs. Minimum core
PCC is0.99994460, norm ratio0.99783322–1.00369780, and minimum recurrent-state
PCC0.99998375. The repeated T128 handoff control remains bit-identical and measures
L1→DRAM2.492780, direct DRAM2.495246, direct L1 input2.455730 ms.

The nine long cases cover both layer kinds at262143/262144 tokens, finite and
non-degenerate last64-token outputs, and last-position eager/replay decode
PCC1.00000000 after the262143 prompt. Independent262144 prefills with2048/1024
chunks agree on the final128 tokens at logged PCC1.000000 for both layers.
The8001-token HF prefill/decode comparisons pass linear0.999426/0.999315 and
full0.999528/0.999745. The exact-cache full-attention HF/traced oracle at
position262143 passes0.99965014; historical KV is a deterministic BF16 fixture
under a permuted page table, while layer/query weights are real. These tests
do not claim a full262144-token HF prefill oracle.

## Final graph audit remains open

After those runs, all four final profiler reports were collected in
[performance.json](performance.json). They report filtered device-time sums,
not the paired wall medians: linear/full prefill25.554401/22.995263 ms and
decode1.439590/1.208572 ms at the profiler's recorded context129. The coordinator
reports23 passing Watcher cases, recorded in [watcher_audit.json](watcher_audit.json)
and [Watcher provenance](logs/watcher_final.provenance.json).

The final profiler audit found a material remaining lowering issue:
`transpose_a=True` still inserts a separate outer-product transpose under
`MatmulMultiCoreProgramConfig`. Earlier correctness and latency observations
for that spelling remain valid, but did not prove dispatch fusion. A supported
explicit reuse-program adaptation is now being tested by the coordinator's
experiment agent. Its result and any required final reruns will be appended by
the coordinator. No stage-closure or exhaustive-final-graph claim is made here.


## Final audit repairs and native transpose selection

The final op-by-op rescan is in [final_graph_audit.md](final_graph_audit.md).
It found two additional concrete adaptations and exhausted both:

- Ordinary FP32 GDN decode `nlp_concat_heads` passes the real pair with unchanged
  PCC but takes 1.509660502 ms traced versus approximately 1.462 ms for the
  existing permutation. [gdn_concat_v1](logs/gdn_concat_v1.log.gz) earns its
  rejection on this 9B hardware; the 35B reference's timing was not reused.
- The outer `transpose_a=True` flag previously materialized a transpose under
  the default multicore matmul. Explicit reuse with one M strip per work block
  fails both native/manual controls at PCC0.23859875 because reader loops advance
  whole matrices. The adapted whole-head config (M4/N4/K1, subblock1×4, HiFi4,
  FP32) passes seven native pair/core tests; the manual-transpose matched-program
  control passes its one pair. [Candidate profile](reuse_outer_profile.json)
  proves 49 rather than 50 decode operations, with essentially equal kernel sum.

The first simultaneous-trace timing harness was invalid: it allocated model B's
persistent tensors after A's capture. [AutoDebug](AUTODEBUG_transpose_trace.md)
and [AutoFix](AUTOFIX_transpose_trace.md) record the source finding and verified
control. In [trace_allocation_repro_v1](logs/trace_allocation_repro_v1.log.gz), two
identical runtime models agree eagerly; one A replay changes twelve B buffers,
including 425981 recurrent entries and small weights, while B never replays and
A remains equal to its eager output. The intentionally unsafe diagnostic is
opt-in (`ORNITH_TRACE_ALLOCATION_REPRO=1`), not a normal model acceptance gate.
The fix prepares both models and every persistent input before either capture,
and checks own-eager equality before timing and again afterward. This changes
only the benchmark harness. Both v2 (15×64) and v3 (31×256) matched tests pass.

[reuse_outer_matched_v3](logs/reuse_outer_matched_v3.log.gz) shows a repeatable
small native-transpose win: 25/31 paired windows faster, mean -0.248252 µs,
standard error0.040963 µs, paired median -0.250328 µs. Eager and 256-step stress
PCC are effectively1. Selection therefore rests on measured traced performance,
not the operation count. The selected runtime SHA256 is
`18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d`.
The previous default matmul graph remains a test-only control (`DefaultOuterGDN`).

[final_short_v2](logs/final_short_v2.log.gz) passes85 tests in150.23s, including
all batch32/HF cases, all six raw-core probes, and the final runtime's matched
31×256 comparison. That independent repetition wins29/31 paired windows,
paired median -0.252512 µs. Final functional→fused paired wall medians (B1,
prefill2048/decodeposition2048, five measured windows,32 replays/window):
linear40.662267→26.180386 ms prefill and1.623890→1.462064 ms decode;
full35.449361→23.498724 ms prefill and1.512215→1.263523 ms decode.
PCCs and rounding behavior are unchanged from the preceding accepted graph.
Do not compare submicrosecond fluctuations across these32-replay windows with
the separately amortized256-replay matched experiment.

Observed anomaly: the corrected simultaneous-trace test still emits the
allocator's conservative active-trace allocation warning during its second
capture. Evidence: final_short_v2 at23:28:12 and the AutoFix report.
Affected path: experimental two-trace timing only. Control: all persistent
consumers are allocated before either capture; later second-trace scratch/output
is overwritten by its own replay before consumption; own-eager equality before
and after timing, stress output and persistent state all pass. Resolution:
controlled temporal warning, not observed final-path corruption. Do not hide
this diagnostic behind a blanket claim that no corruption-related warning exists.
The ordinary one-trace correctness and profiler runs use their established setup
order. There were no hangs or resets.

The original validated graph's reports are preserved as `v1_*` files and
[performance_v1.json](performance_v1.json). Final native-transpose long-context,
Watcher and all four profiler reruns are being recorded with `_v2` run names;
the next entry will record their completed outcomes and independent review.


## Completed final-source validation

The serialized final queue completed with runtime SHA256 `18d59502…f9b11d`:
[final_short_v2](logs/final_short_v2.log.gz)85passed/150.23s;
[final_long_v2](logs/final_long_v2.log.gz)9passed/169.40s;
[watcher_final_v2](logs/watcher_final_v2.log.gz)23passed/44.57s.
The latter run has profiler unset; [watcher_audit.json](watcher_audit.json)
records device checks, no error/corruption signatures and compressed dump hashes.
All preserved native-context and non-aligned shape capabilities pass, including
real HF8001-token comparisons and the position262143 exact-cache HF oracle.
[context_contract_check_v2](logs/context_contract_check_v2.log.gz) passes the
strict full-HF-capacity contract check at262144 without a capability reduction.

All four `profile_final_{linear_attention,full_attention}_{prefill,decode}_v2`
commands pass their measured-output HF tests. Exact argv/environment/source
snapshots are in their same-stem provenance files. Prefill uses2048 tokens;
decode uses position128 and four measured replays. `render_perf.py <kind> <mode>
--capture <mode>_v2 --iterations <1-or-4>` generated each text/CSV report.
`summarize_perf.py --run-suffix _v2` validates complete replay-ID coverage,
no dropped markers, all kernel durations, raw-ns/CSV-µs sums and report hashes.
[performance.json](performance.json) reports final kernel sums:
linear25.647288ms prefill/1.4412645ms decode (120/49ops),
full23.002021ms/1.20938125ms (33/36ops). These beat the matching functional
40.075804/1.60966675 and35.104062/1.42231950ms sums. Final decoded outputs retain
PCC0.99976799 linear and0.99889471 full under this four-replay profile regime.
The selected graph's further submicrosecond win over the preceding candidate
is supported by the matched wall experiment, not different-run kernel sums.

`summarize_validation.py --suffix _v2` produces [validation.json](validation.json),
checking that final short/native/Watcher records all match the current runtime.
[artifact_audit.json](artifact_audit.json) verifies121 run/log/source records,
including all16 failed experiments retained with their subsequent localization
and supported adaptation evidence. `summarize_candidates.py` now records82
real-weight paired comparisons. No rejected result is silently deleted.
The bounded final `timeout 60 tt-smi -ls --local` succeeds with all four P300c
Blackhole chips visible; [log](logs/final_device_list.log.gz). Hardware jobs
were serialized and no reset/recovery was necessary.

Only Python, tests and documentation changed; a C++ build is not required by
AGENTS.md. `pre-commit run --files <stage Python files>` passed in
[precommit_final_sources_v3.log.gz](logs/precommit_final_sources_v3.log.gz).
Final all-artifact host checks and independent review are recorded below before
the local checkpoint commits. No later model pipeline stage was started.


Final host checks: [host_checks.json](host_checks.json) records the exact staged
file list and `pre-commit run --files ...` argv (return0), and
`git diff --cached --check` (return0). Generated filtered CSV line endings and
text trailing whitespace were normalized; raw ops archives remain byte-exact
and performance report hashes were regenerated without metric changes.
All stage-owned files are isolated in the requested model runtime/tests/docs.
The independent reviewer verified all121 run records and final source maps.


## Independent stage review

Fresh xhigh reviewer `/root/fused_stage_review` returns **clean-pass** with no
required work in [STAGE_REVIEW.md](STAGE_REVIEW.md). It independently rederived
final measurements and source/archive hashes, checked all applicable fusion
closures, and classified every material anomaly. Required work identified during
the audit (actual outer transpose fusion, GDN concat trial, trace benchmark
allocation ordering, and stale metadata) is fixed or measured/rejected and
reviewed. The runtime remains exactly the tested `18d59502…f9b11d` source.
The stage is ready for its local checkpoint. No push is authorized or performed.
