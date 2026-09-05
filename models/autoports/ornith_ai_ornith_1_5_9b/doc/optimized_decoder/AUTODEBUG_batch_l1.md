# AutoDebug: batch-32 L1 capacity failures

Date: 2026-09-05. Inspection-only investigation for the optimized-decoder stage of
`ornith-ai/Ornith-1.5-9B`. No implementation edits, TTNN imports, device access, or
hardware reproductions were performed. Existing sharded-trace reports are preserved.

## Findings

**Two independent resource failures are supported by source and exact allocation
arithmetic.** The base/KDA cases fail during decode's first residual addition;
the combined candidate fails earlier, during prefill's first RMSNorm. Removing the
redundant FP32 cast alone cannot repair the base case. Changing projection geometry
or the GDN output dtype cannot explain or repair the combined prefill failure.

The repairs must retain B1–32, all logical lengths, context 262144, real-weight HF
PCC >= 0.995, immutable public inputs, and exact output/state equality after restore
and trace replay. This report does not judge stage completion or claim a measured
improvement.

## Direct observations and reproduction boundary

The exact commands and environment are in the adjacent
`logs/{sharded_real_batch_contract,kda_real_batch32_contract,combined_decode_batch_contract}.provenance.json`.
All use real weights, recorded layer activations, BFP4/LoFi projection policies,
the selected per-role DRAM-sharded geometry, and `residual_cores=32`.
`run_candidates.py` supplies the campaign defaults; `batch_screen_plan.json`
records variant-specific changes.

| Run | Observed failure | Passing contrast in that run |
| --- | --- | --- |
| `sharded_real_batch_contract` | B32 linear decode, `optimized_decoder.py:161`, FP32-to-FP32 `typecast`; request 16,777,216 B over 32 banks, 524,288 B/bank; allocated 1,310,720, free 125,952, largest free block 72,704; bank size 1,436,672 | B32 linear prefill completed HF PCC checks; B4 linear/full prefill+decode, B1 linear/full traces, full-attention ragged B4/B13 passed |
| `kda_real_batch32_contract` | Same B32 linear decode operation and identical byte counts despite replacing the prefill convolution | B32 linear prefill completed HF PCC checks |
| `combined_decode_batch_contract` | B32 linear prefill, first norm at `optimization_extra_candidates.py:474`; input `[32,128,4096]` BF16; static buffers end at 992,256, occupied L1 starts at 937,984, core range `[0,0]–[10,9]` | B4 linear/full prefill+decode, B1 linear/full traces, full-attention ragged B4/B13 passed |

The shared batch test uses logical prefill length 96, physically padded to 128,
followed by one decode at position 96. The combined error precedes every mixer,
projection, residual, and recurrent-step operation in that forward call.
`--maxfail=1` stops these campaigns at their first failure: subsequent B32 full
attention and remaining traced cases are unverified by these logs. Ragged B13
passing is a **full-attention** observation, not evidence that B13 linear state
placement is safe.

These are host-side allocation/region-validation exceptions, not observed device
hangs. Successful HF comparisons require tensor readback; trace tests explicitly
synchronize after replay. No numerical or trace-corruption claim follows from
the failed allocations.

## H1: expanded residual tile padding exhausts L1

**High confidence for the base and KDA failures.** In `optimized_decoder.py`,
`_block:167–171` and `_norm:115–125` compute shard rows from the product of the
*padded* dimensions. Decode public shape `[B,1,4096]` has physical shape
`[B,32,4096]`. Width sharding over 32 cores gives `[32B,128]` per core.
That calculation accurately describes the expanded tensor, but keeps 31 padding
rows for every user in the residual path.

`_linear:135–150` already demonstrates the compact alternative: it reshapes to
`[1,B,K]` before projection, fitting B<=32 into one physical tile row, then expands
back to `[B,1,N]` for mixer consumers. `_block:183` subsequently converts the
expanded FP32 mixer output into the large residual shards.

At entry to the failing cast the live tensor ledger is:

| Tensor | B1 bytes/core | B4 bytes/core | B32 bytes/core |
| --- | ---: | ---: | ---: |
| Original BF16 residual `x` | 8,192 | 32,768 | 262,144 |
| FP32 mixer output `mixed` | 16,384 | 65,536 | 524,288 |
| FP32 promoted residual | 16,384 | 65,536 | 524,288 |
| **Total before redundant cast** | **40,960** | **163,840** | **1,310,720** |
| Additional FP32 cast output requested | 16,384 | 65,536 | 524,288 |

The B32 total exactly matches the allocator's `allocated` value. The request is
32 * 524,288 = 16,777,216 bytes. This needs no stale allocation or allocator bug
to explain the failure. Fragmentation is secondary: even total free bytes are
far smaller than the request.

The same-dtype cast is a real allocation, not a no-op. C++
`operations/copy/typecast/typecast.cpp:43–54` inherits the input memory config;
`device/typecast_device_op.cpp:137–161` constructs a new output tensor and only
skips launch for zero-volume outputs. Here the optimized sharded factory is
selected because FP32 tile sizes match and input/output are L1-sharded. The
failure occurs while allocating its output, before the kernel executes.

**Refutation of the one-line-fix hypothesis:** skipping the second cast leaves
1,310,720 B/core live, but the following out-of-place homogeneous FP32 add needs
another 524,288. Its 1,835,008-byte lower bound still exceeds the bank capacity.
The final BF16 cast can raise the no-copy peak to 2,097,152 B/core. `mixed` remains
referenced by the caller throughout the helper. Even reusing the promoted buffer
for the sum, without another lifetime/layout change, leaves the final BF16 output
allocation at 1,572,864 B/core, still too large. Explicitly deleting public inputs
or mutating their buffers is not an acceptable memory repair.

**Repair boundary:** retain the existing homogeneous FP32 sum and its single
BF16 rounding point while reducing the physical residual footprint or placing
large-batch residual work in DRAM/interleaved memory. Folded residual shape
`[1,B,4096]` needs only `[32,128]` shards for every B<=32, reducing this B32
footprint by 32x. A setup-selected larger-batch interleaved/DRAM residual policy
is a smaller initial control that can preserve the current B1 path.

Folding requires actual `ttnn.reshape`, not shape metadata edits: `[B,1,H]`
contains per-user physical padding. `reshape_view/reshape.cpp:613–619` excludes
this from the ordinary view path; the tiled reshape machinery materializes the
mapping and recomputes width-shard shapes (`:177–185`). Preserve `[B,1,H]` at
mixer/public boundaries. Existing `_linear` dispatch uses `x.shape[1]==1`, so
blindly feeding `[1,B,H]` through the whole block changes decode dispatch. A
compact implementation must explicitly manage those boundaries or confine
folding to residual/norm helpers. Avoid premature explicit deallocation of
aliases; dropping an owned local reference is different from freeing a borrowed
public buffer. The original `x` also remains live after the first residual sum,
so lifetime cleanup may reduce later peaks, but cannot explain away the first
failure.

## H2: persistent L1 state collides with prefill RMSNorm buffers

**High confidence for the combined failure.** `RecurrentConfigCandidate.allocate_state`
(`optimization_extra_candidates.py:27–32`) unconditionally converts the full
recurrent state to interleaved L1 when `ORNITH_STATE_MEMORY=l1`. This occurs at
setup, before prefill. The base allocator creates FP32 state
`[B,32,128,128]` (`functional_decoder.py:345–351`), as required by the pinned
checkpoint. At B32 that is 64 MiB / 16,384 FP32 tiles.

The error reports an 11-by-10 norm core range. Using 110 L1 banks and 4,096-byte
FP32 tiles, state requires `ceil(16384/110)*4096 = 610304` bytes/bank. The
test reserves 24,576 bytes of small L1 (`test_functional_decoder.py:65`), leaving
the allocation top at `1572864-24576 = 1548288`. State therefore starts at
`1548288-610304 = 937984`: **exactly the conflicting logged address**.

The first norm inherits DRAM output from its BF16 DRAM input, uses tiled BF16
gamma, no bias/residual fusion, and the default RMSNorm compute configuration
(HiFi4, approximate math, FP32 destination accumulation disabled). The selected
factory is `LayerNormMultiCoreProgramFactory`; its ordinary reader, blocked
writer and `kernels/compute/layernorm.cpp` handle this shape.

For width 4096, Wt=128 and block size=8. In
`layernorm_op_multi_core.cpp:304–326,495–570`, the actual static allocation is
430 BF16 tiles: input 128, output 16, scaler 2, epsilon 2, variance 2,
squares 128, variance-plus-epsilon 8, gamma intermediate 16, gamma 128.
Thus `430*2048 = 880640` bytes. Its reported end 992,256 implies static base
111,616, consistent with the logged general bank size:
`1548288-111616 = 1436672`. The overlap is `992256-937984 = 54272` bytes.
B4 state needs only 77,824 bytes/bank, explaining its passing contrast.

The norm planner's `buffers_can_fit_in_L1` uses total device L1 capacity
(`layernorm_op_multi_core.cpp:37–76,356–372`), rather than remaining space below
live tensor allocations. It selects the normal kernel, while
`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2588–2622` later checks the
actual lowest occupied address and correctly rejects the collision. The earliest
model-owned intervention is the candidate's unconditional state placement;
rewriting the allocator/validator is unnecessary for this repair.

**Repair boundary:** select recurrent-state placement at setup from batch and
validated working-set requirements, keeping the proven B1 L1 path and using DRAM
for larger batches that cannot coexist with prefill kernels. `ORNITH_STATE_INTERMEDIATES=l1`
is a separate choice: it allocates an additional state-sized `outer` during decode
(`:52–63`). Moving only persistent state may expose later transient pressure, so
test both controls independently. Do not relocate persistent state inside capture
or replay; stable addresses and in-place state updates remain required.

Changing prefill token chunk size does not reduce this width-dependent first
norm's static buffer demand or the persistent state size. Changing its projection
program cannot affect a failure before projections. BF16 GDN projection output
may be a valid independent optimization, but it does not establish either the
FP32 residual repair or this state-placement repair.

## Focused verify/refute experiments

Run only through the coordinator's exclusive hardware lane, saving commands,
environment, source archives, log hashes, and actual effective memory configs.
Do not run the whole campaign concurrently with these controls.

1. **H1 negative control:** base variant, original FP32 GDN output, B32 linear
   batch contract; skip only already-FP32 cast. Prediction: failure moves to the
   FP32 add allocation. If it passes, capture the live shape/dtype/shard ledger:
   another effective lifetime/config change must have invalidated the ledger.
2. **H1 repair:** original FP32 projection with only compact residual geometry or
   larger-batch interleaved/DRAM policy. Check B32 first, then B1/B4/B13 and
   boundary batches through 32. Retain the exact `(a.float()+b).bfloat16()`
   component oracle from the prior regression, repeated eager output equality,
   unchanged operands, restored trace output/state equality, and per-user HF
   prefill/decode PCC. Prove the FP32 update case before adopting a BF16 projection
   optimization. Record B1 warmed latency under the unchanged policy.
3. **H2 independent control:** combined variant, original residual code, set only
   `ORNITH_STATE_MEMORY=dram`; retain `ORNITH_STATE_INTERMEDIATES=l1`. Re-run the
   original B32 linear test. Prediction: first norm succeeds; it may fail later
   at H1 or another intermediate allocation. Reaching a later failure validates
   the first-norm intervention but is not an integrated pass.
4. **H2 transient control:** combine the separately proven H1 repair with DRAM
   state; compare intermediate placement L1 versus DRAM. Verify decode/trace at
   B32, recording state address stability and the `outer` allocation. This
   distinguishes persistent-state repair from later transient pressure.
5. **Integration:** both proven repairs together; re-run all selected `batched or
   traced_decode` cases without stopping after a repaired subset. Add exact
   restored-state/eager/trace assertions for B>1: the shared B1/4/32 traced HF
   test checks PCC, whereas the exact restored-output regression currently uses
   B1. Include immutable input checks for both DRAM and sharded public inputs.
   Retain nonaligned lengths, native-context gates, and full-attention coverage;
   no context limit or accuracy threshold change is justified by these failures.

## Evidence hashes and limitations

SHA-256 values were checked locally against the preserved files. Runtime
`tt/optimized_decoder.py`, `tt/functional_decoder.py`, and `tt/fused_decoder.py`
match all three failure provenances. Candidate source changed between the base
and KDA runs; each archive is retained and must be used with its own provenance.
Lowered C++ was inspected from the current checkout, not recovered from a
build-time binary manifest; the exact agreement with error arithmetic supports
the reconstruction but is not proof of binary/source identity.

| Evidence | SHA-256 |
| --- | --- |
| Base log | `f6915f5df75f9de6d586ce5d81c5e44315580c27e11c40d2bf6b8b410564e6bd` |
| KDA log | `b1d24312e7e62ea18bf5c65b6644d4681995d0b69f654124f9c4194b479fcf4c` |
| Combined log | `605b417e343958f7e4091c8ed41af19fdd23d735a112aaaeaf90ffcebbcf49de` |
| Base source archive | `1534ae65631da51f81d16e5394fb39f630c9cf9e705fb66327dcaada255c70fd` |
| KDA/combined source archive | `608f057b141c173a56bcc733677a282a983e3f5f53cd3f165d4b0bb6dec6bbe0` |
| Optimized decoder | `90dfde96ea2357af9da3c5475ad3075557b6c314e6afced7b1e360dc23345250` |
| Candidate source, base run | `c78b77cf06e661e507f6f99d5b50cb4c1666410805ae61b562a42b86548c2f77` |
| Candidate source, KDA/combined run | `dff1093d81e059ad73772fce4041d7aedf39b86a27acb3c18a5cc3bf58281272` |
| Functional decoder | `bbc44e62d6045a15fb0c56060fffe91e264983874731f96ff1b187cce826bd30` |
| Fused decoder | `18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d` |
| Typecast wrapper C++ | `469360dc3997e6ebf9ff3c52df04562070f8716fae2e94c1cd37e7cdcd4d481c` |
| Typecast device operation C++ | `25c534b7f88b9a52f329624ddf07143a32b766165f7c6c8e07ff9e046380f14b` |
| Interleaved norm factory C++ | `a83e9164af2c59671e6bd50c9c6a770bb45a07ccaeecddefe22d39439decf80d` |
| Reshape C++ | `83686700d5ccb42545b73e0fd233d63a24e26c285a6216f32e033b24848cf975` |
| Dataflow-buffer validator C++ | `7d03aba8e41093d26af40971441a75ed1dead9ace40640a8e2dc7b6a79cb4660` |
| Preserved `AUTOFIX_sharded_trace.md` | `76459414741c142560ed0c2f9d18060edd3de6dba136447ea93493a2b77b2fe6` |
| Preserved `AUTODEBUG_sharded_trace.md` | `38e957fbf1426ad69ef9db5d1e10f5f41984add554a5a91f4134fca0afc924f6` |

The earlier mixed BF16/FP32 sharded-add nondeterminism remains a separate,
experimentally established issue. Reverting its homogeneous-FP32 adaptation or
relaxing equality to resolve resource pressure would discard that evidence.
No hardware repair is marked verified in this source-only report. Docs-only
change: no C++ build is required.
