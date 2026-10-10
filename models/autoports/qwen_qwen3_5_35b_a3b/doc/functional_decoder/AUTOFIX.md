# AutoFix — active repair

Contract: exact target-layer semantics, PCC ≥0.995 in prefill and each decode
step, paged cache/request ownership, and device-only traced execution. Initial
adapter source diagnosis is in `AUTODEBUG.md`; its captured adapter predates
the repairs, so each finding is checked against current source before editing.

## Supported repairs and tests

1. Rank3 attention to rank4 MoE mismatch: local reshape bridge. Both real-weight
   layer kinds execute; initial 128/129 and full-attention4096 logs show PCC pass.
2. Runtime host masks and public alignment rejection: bounded exact-length
   chunks plus recurrent/paged single-token tails. Both 129-token paths pass
   tracking-enabled repeated decode. Runtime torch/conversion guard passes in
   both 4096/128 paths (accuracy is separately checked).
3. Fixed B=1 attention state and bad batched full-attention axes: execute rows
   separately with setup-allocated per-request GDN state. Full-attention B32,
   S33/page128 passes with permuted pages and traced decode; `full_batch32.json`.

## Numerical hypotheses

The aggregate GDN4096/128 PCC barely passed, so the harness now gates every
decode token separately. The original trace has 38 failures, minimum .9701314.
This is required work, not an accepted accuracy exception.

| Candidate | Prefill PCC | Aggregate decode PCC | Minimum step PCC | Failed steps |
|---|---:|---:|---:|---:|
| Inherited alpha0.25 | .997615516 | .995002866 | .970131397 | 38/128 |
| Exact alpha0 | .998961210 | .997550130 | .986624837 | 16/128 |
| Alpha0 + BF16 experts | .998966515 | .997561336 | .986684322 | 16/128 |

Evidence: `linear_4096_128_per_token.json`, `linear_alpha0.json`,
`linear_alpha0_bf16experts.json`, corresponding console logs. Commands use
`run_layer --layer 0 --length 4096 --steps 128 --trace --audit`; controls set
`QWEN_GDN_DIAG_ALPHA=0` and the final row adds `--expert-precision bf16`.
Tracking is enabled, including program-cache allocations, in all three.

The diagonal changes HF mathematics and is not an acceptable default. Expert
weight precision does not explain the remaining failures. A focused isolated
hypothesis agent is preparing an autoport-local GDN using the native exact
chunk recurrence and FP32 recurrent kernel/state, with BF16 projections. The
main agent owns hardware validation. No shared source or installed plugin edits.

Status: repair in progress. No AutoFix failure or stage completion declared.

## Current numerical result

The native-GDN change alone left two of128 decode tokens below .995
(`linear_native_4096_128.json`). BF16 expert control did not fix them
(`linear_native_bf16experts.json`). Matching HF FP32 zero-centered normalization
resolved both (`linear_native_fp32norm.json`, minimum .995265305); this is the
current implementation. The component diagnostic runner is available for any
new failure; its host reads are deliberately outside runtime/performance claims.
Native-context linear attention also passes (`linear_context262143.json`).
AutoFix remains active until the remaining stage gates and review pass.

## Trace reuse repair

Independent review identified an unseen logical-length hole. The expanded
`run_reuse --layer3` reproduced a forbidden SliceDeviceOperation cache miss at
logical36 inside the already warmed physical128 bucket; `full_reuse_unseen.log`.
The older test warmed every length and therefore did not expose this defect.

Current prefill uses resident offset vectors plus runtime scalar add, embedding
reads for token chunks, device page gathers, tensor-valued chunk starts, and
fixed-shape output slice writes. Static full-attention chunks are now512/128 at
128-aligned positions, with exact recurrent/paged single-token unaligned tails.
The output retains its physical bucket as padding with the requested logical
length. No runtime host selector is created. The initial repaired reuse matrix
passes including withheld36 (`full_reuse_bucket.json`). It is being expanded
with a withheld continuation start36 after a warmed start35 using the same
physical input buckets. All current numerical/capacity/performance gates will
use this repaired implementation.

## Verified single-token recurrence precision repair

The heterogeneous B32 test exposed two additional GDN failures: row 10,
prefix 255, decode 0 PCC .99385923; row 18, prefix 465, decode 3 PCC .98969662
(`positions_linear_b32.json`). Router FP32 softmax/top-k/renormalization,
HF-style BF16 Q/K normalization, and more accurate activation evaluation
individually did not resolve row 18. Running native recurrent execution for
the entire prefill worsened its minimum decode PCC to .97152418
(`components_positions18_recurrent_prefill.json`).

Source inspection found a concrete precision mismatch, without finding a
GQA, token/state indexing, or recurrence-order mismatch. The native wrapper
repeats adjacent Q/K heads and the kernel performs decay, memory read, delta,
state update, and post-update output read in HF order. Its buffers are FP32,
but `fused_recurrent_gated_delta_rule_program_factory.cpp:136` does not request
`UnpackToDestFp32`. The JIT selects TF32 source format for this configuration
(`tt_metal/jit_build/genfiles.cpp:978-980`); the full-FP32 exception requires an
explicit unpack-to-destination mode (`data_format.cpp:215-221`). In the native
compute file `fused_recurrent_gated_delta_rule.cpp`, state decay at line138,
memory matmul at line196, residual subtraction at line113, state seed copy at
line77, and output matmul at line222 consequently cross reduced-precision
source registers. FP32 destination accumulation does not restore lost operand
bits. This repeatedly quantizes carried state and introduces cancellation
error in the delta calculation; it is not just one output matmul rounding.

The outer planner uses 128-token GDN buckets (`functional_decoder.py:351-356`),
so prefix 465 has 81 single-token tail updates and prefix 255 has 127. The repair
therefore changes prefill tails as well as decode. Accumulation over these
tails is consistent with the observed approximately 5% native GDN relative-L2
error. Attribution to one instruction or a quantified amplification mechanism
has **not** been isolated; the verified result localizes the problem to the
native single-token recurrence path, with source-supported repeated TF32
rounding as the numerical explanation.

`FunctionalGatedDeltaNet._recurrent_step_fp32` now follows HF's explicit
FP32 multiply/subtract/add/reduce boundaries. FP32 binary operations request
full-FP32 unpacking (`binary_ng_program_factory.cpp:1258-1265`), and the two
key-dimension sums use the accurate SFPU path with FP32 accumulation
(`reduction/generic/device/reduce_op.cpp:198-215`). Q/K heads expand 16→32;
persistent state remains [1,32,128,128]. Aligned prefill retains native chunk
execution. No shared implementation was changed.

The parent agent ran the following real-weight checks; the investigating
agent performed source inspection and local implementation only.

| Evidence artifact | Prefill PCC minimum | Decode PCC minimum | Scope |
|---|---:|---:|---|
| `components_positions18_sfpu_recurrence.json` | .999584079 | .998062253 | Isolated row 18, 8 steps; diagnostic host reads |
| `positions_linear_sfpu.json` | .999298871 | .998062253 | B32 heterogeneous prefixes, 8 traced steps per row |
| `linear_accept_4096_128.json` | .999701798 | .997864842 | B1, 4096-token prefill, 128 traced decode steps |

In the isolated control, minimum GDN PCC improved from .998921275 to
.999951303; decode 3 GDN relative L2 improved from .05019733 to .00846517
(`components_positions18_fp32sfpu.json` versus the repaired component artifact).
B32 trace evidence records exact repeated-input and warm-versus-replay outputs,
runtime audit enabled, allocation tracking enabled, and no source change
during the run. These checks establish correctness evidence, not performance.
The pre-repair snapshot is `functional_gdn_before_sfpu_recurrence.py`.

Retain BF16 Q/K normalization and FP32 router selection as explicit HF dtype
contracts; their earlier failures only show they were insufficient alone.
FP32 activation evaluation followed by BF16 rounding is an additional numerical
choice, not an isolated necessary fix. Any simplification should be controlled
independently and revalidated; none of these changes is assumed harmless around
MoE routing cutoffs. Remaining full-attention, capacity, performance, and review
gates remain separate work; this repair does not declare stage completion.

## Small KV pool repair

The follow-up review identified two capacity defects. Padding a logical table to
8 columns for chunk SDPA made paged cache updates reject smaller physical pools.
`run_layer --layer 3 --length 65 --steps 31 --batch 2 --minimum-cache-pages 0
--cache-guard-pages 0 --trace --audit` reproduces the8-vs6 validation failure;
`small_pool_before.log`. The generic decode helper also rounds reads to64 tokens:
with an odd number of32-token pages it can read an uninitialized extra table entry
before masking. The reviewer records the exact native reader/slice source chain.

The local decoder now keeps original tables for updates and gathers and pads only
chunk-attention read tables. `FunctionalGatedAttention` delegates common paths to
the existing module and uses an equivalent local decode with k_chunk_size32 when
a row's logical capacity is not divisible64. Weight/norm/RoPE/gate/projection math
and fidelity are unchanged. This only affects odd tile capacities; native-context
and headline tables have even tile capacities. The new source remains local.

Initial adapter integration exposed unsupported TTNN Shape slicing; replacing
`x.shape[:2]` with indexed dimensions fixes that host API error. The failing log is
`small_pool_adapter_shape_failure.log`. The repaired B2 test has prefill PCC
.998859763 and minimum31-step decode PCC .996751010, exact replay, runtime audit,
and watcher enabled (`small_pool3_b2.json`, `watcher_pool3_b2_fixed/`). Additional
B1/B2 odd-pool watcher cases and current-source reuse/profiles are recorded in
`completion_commands.json`. No reset or device process termination was needed.

Before final profiling, Black formatted production/test source; AST
comparison proved every formatted file unchanged (`format_provenance.json`).
Historical diagnostic snapshots retain their original bytes. This formatting
changes source hashes, not the tested calculations.

The attention Shape indexing repair described above occurred after that formatting
record, and the profile harness later gained only between-pass profiler drains.
Current source hashes are recorded by the final profile and edge-case JSONs; the
formatting record is a historical transformation, not a final-source manifest.
