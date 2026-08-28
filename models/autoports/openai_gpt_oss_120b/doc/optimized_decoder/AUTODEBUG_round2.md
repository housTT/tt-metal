# AutoDebug round 2: optimized batch-32 trace nondeterminism

## Headline

The supplied A/Bs refute the L1 router, sharded post-attention norm/split
fallback, and optimized QKV projection as *individually necessary* causes: all
three variants still fail replay-1 versus replay-2 `torch.equal`. The canonical
post-norm control also removes both split warnings without restoring
determinism. The same batch-32 fused decoder passes sliding and full attention,
so the test/trace/cache contract remains a strong control.

No hardware experiment was run in this source-only pass. The highest-priority
untested runtime delta is the optimized **input** RMSNorm. The only other direct
decode-op call delta left at batch 32 is the output projection's explicit
compute-kernel argument.

## Exact remaining deltas

At `max_batch_size=32`, `OptimizedDecoder.from_state_dict` automatically selects
`ATTENTION_BF16_CONTROL` (`optimized_decoder.py:427-429`). Therefore BFP8/BFP4
attention weights, output-activation typecast, and DRAM-sharded QKV are not in
the failing graph. QKV/o-proj weights have the same BF16 dtype as the passing
fused control.

1. **Input norm (highest priority).** Optimized decode converts the cloned DRAM
   residual to a fixed ten-core L1 width shard, runs sharded RMSNorm, and returns
   that L1 tensor (`optimized_decoder.py:335-371,559-565`). Canonical `RMSNorm`
   invokes plain `ttnn.rms_norm` without that conversion/program config
   (`models/demos/gpt_oss/tt/rms_norm.py:83-90`). This boundary precedes QKV,
   RoPE, and the mutating KV-cache writes, and has not been A/B tested. (The
   optimized norm docstring says nine-way, but its grid is correctly ten cores:
   `10 * 288 = 2880`; that wording mismatch is not itself a root cause.)
2. **Output projection call.** Optimized o-proj passes
   `decode_projection_compute_kernel_config` and `program_config=None`
   (`optimized_decoder.py:189-196`); canonical o-proj omits both arguments
   (`attention/decode.py:166-168`). H3 removed the explicit config only from
   QKV, not from o-proj, so the work-log statement must not be generalized to
   the whole attention projection. Source inspection makes this lower priority:
   for BF16 inputs with no program/core config, matmul normalization selects
   HiFi2, `math_approx_mode=false`, FP32 accumulation off, and L1 accumulation
   on (`matmul_device_operation.cpp:2795-2805,2843-2850`), exactly matching the
   optimized explicit configuration (`optimized_decoder.py:488-494`). A direct
   A/B is still decisive against any optional-argument/cache-path distinction.
3. **Interactions/constructor effects.** The three negative one-delta removals
   do not rule out an interaction between optimized components. Setup-only
   differences still include `_OptimizedAttention`, `_OptimizedMLP`, and the
   optimized norm objects. Policy metadata and the optimized `_forward` method's
   `decode_mode` assignments do not emit device operations. With no tensor cache
   path supplied by this synthetic capacity test, the alternate attention cache
   subdirectory is also inert.

## Decisive bisection

Use the exact existing sliding batch-32 seed/node and retain its full
131072-token-per-user allocation.

1. **Positive control first:** construct an `OptimizedDecoder` through its local
   constructor, but give it the exact canonical BF16 components: `Attention`,
   `RMSNorm` for both norms, and `_FusedMLP`. Assert those concrete types and the
   BF16 weight dtype. This “all-canonical optimized-constructor” control should
   pass like `FusedDecoder`; if it does not, stop interpreting component A/Bs
   and diff tensor addresses/op traces against the fused control.
2. **Canonical input norm only:** on the current optimized graph, replace only
   `input_layernorm.forward` with `RMSNorm.forward`; assert post norm remains
   `_DecodeShardedRMSNorm`, attention remains `_OptimizedAttention`, and the L1
   router remains selected. A pass isolates the input-norm/L1-lifetime boundary.
3. **Canonical o-proj only:** leave optimized QKV unchanged, but make the o-proj
   call byte-for-byte canonical by omitting both `compute_kernel_config` and
   `program_config`. A pass isolates the remaining attention call delta.
4. If both single removals fail, run them together. If that also fails, start
   from the passing all-canonical constructor control and add back one optimized
   component at a time (input norm, complete optimized attention, post norm,
   router). This direction detects interactions that removal from the fully
   optimized graph can mask.

For every run, record differing user rows/count, maximum absolute delta, and
PCC in addition to bitwise equality. Once a component flips the result, trace
that component or the shortest prefix ending at it and compare its output after
each replay. Confirm any passing variant three times, then run both sliding and
full attention before retaining a production change.

## Evidence summary

- Baseline optimized sliding: fails after two complete replays;
  `/tmp/optimized_final_batch32.log`.
- Canonical fused sliding and full: both pass;
  `/tmp/optimized_autofix_fused_batch32_control.log`.
- DRAM router: still fails; `/tmp/optimized_autofix_h1_dram_router_sliding.log`.
- Canonical post norm: warnings disappear, replay still fails;
  `/tmp/optimized_autofix_h2_canonical_postnorm_sliding.log`.
- Canonical QKV: replay still fails;
  `/tmp/optimized_autofix_h3_canonical_qkv_sliding.log`.
