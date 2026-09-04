# AutoDebug: continuation prefill frees its persistent position ramp

Date: 2026-09-04. Scope: isolated, source-only investigation for AutoFix;
no implementation edits, tests, device listing, or hardware access performed.

## Starting evidence

- Command supplied by coordinator:
  `TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests -k 'continuation or contract_extensions or decode_pcc' -x -v -s`
- Log: `logs/trace_continuation.log`. Six selected cases pass before
  `test_prefill_continuation[blackhole-63-linear_attention-mesh_device0-device_params0]`
  fails during its second `prefill_forward` call. The actual failing operand is
  `self.w["pos_ramp"]`, FLOAT32/TILE, logical shape `[1,128,1]`, printed as
  `<buffer is not allocated>`. The stack reaches `_gdn_gates` line 644 and the
  generic device-operation input-allocation assertion.
- Read current work log, local AGENTS contract, tests, current implementation,
  relevant pinned 35B implementation, TTNN slice/memory conversion/deallocation
  source. Git status shows the new autoport implementation, reference, tests and
  documentation are untracked; no tracked changes were present at inspection.

## Finding H1: full-extent ramp slice is borrowed but force-deallocated

**Confidence: strong source-level causal evidence; focused runtime verification
is still required.** The log proves the ramp is dead; the following source chain
identifies the operation that kills it.

1. `tt/functional_decoder.py:280` uploads `pos_ramp` with logical shape
   `[1,prefill_chunk,1]`. This is persistent decoder-owned state.
2. `_gdn_gates`, lines 643–646, masks padded tokens by slicing this ramp to
   `[1,t,1]`, creating `keep`, then unconditionally calling
   `ttnn.deallocate(ramp)`.
3. `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:199` detects a
   zero-start, unit-step, full-extent slice. Its adjustment path preserves the
   requested layout and memory config. The identical-memory-config path in
   `ttnn/cpp/ttnn/operations/core/to_memory_config/to_memory_config_op.cpp:256`
   returns the original tensor; no fresh buffer is allocated.
4. Python `ttnn.deallocate` defaults `force=True`
   (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:185`).
   `ttnn/core/tensor/tensor.cpp:124` permits freeing shared storage under that
   flag. The result's Python object identity cannot establish ownership.
5. With `prefill_chunk=128`, the first 63-token call is physically padded to
   128 (`prefill_forward`, lines 927–933). Its mask slices the complete
   `[1,128,1]` ramp, then frees the persistent allocation. The second 321-token
   call runs two full 128-token chunks without needing the ramp; its final
   65-token chunk is padded to 128 and attempts to read the now-dead ramp.
   This predicts the exact logged operand and failure location.

This ownership mistake also exists in the pinned 35B reference at
`tt/functional_decoder.py:629–632`. It is inherited behavior, not evidence that
the 9B dense MLP adaptation caused the failure. The current module already has
the appropriate local ownership convention in `_slice_owned` (line 100), which
explicitly treats full-extent slices as borrowed.

### Parameter predictions

For the test's total length 384 and ramp/chunk length 128:

| Linear-attention split | First-call logical blocks | Second-call logical blocks | H1 prediction |
| --- | --- | --- | --- |
| 63 | 63 | 128,128,65 | First mask frees ramp; final second-call mask fails. |
| 65 | 65 | 128,128,63 | Same lifetime failure. |
| 128 | 128 | 128,128 | No mask, so H1 does not fire. |
| 129 | 128,1 | 128,127 | First-call tail frees ramp; second-call tail fails. |

The full-attention variants do not use this ramp; H1 makes no claim about their
correctness. The `-x` log establishes only the first continuation failure, not
the outcome of later cases. Previously passing decode/prefill cases do not
refute H1: a larger ramp makes a 128-token prefix a proper subrange, and a
single masked call need not read the ramp again before test completion.

## Focused verify/refute experiments for the hardware owner

1. **Verify alias lifetime with the exact persistent ramp.** In a temporary
   diagnostic around `_gdn_gates`, record `t`, `logical_len`, the original ramp's
   allocation status and buffer address, and the sliced ramp's address before
   line 646; record original allocation status after it. Run only:

   `TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 ORNITH_WEIGHTS=real pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py -k 'prefill_continuation and 63 and linear_attention' -x -v -s`

   Prediction: first call has `logical_len=63`, `t=128`, equal buffer addresses,
   and original ramp changes from allocated to unallocated immediately after
   deallocation. If it stays allocated or buffers differ, refute H1 and locate
   the actual earlier deallocation before changing code.

2. **Minimal intervention after verification.** Replace this one direct slice
   with `ramp, ramp_owned = _slice_owned(...)`; deallocate `ramp` only when
   `ramp_owned`. Do not change mask values, precision, recurrence, or chunking.
   Rerun the narrow failing check and assert the persistent ramp remains
   allocated. Then run every continuation split/layer kind and the original
   broad command. Existing real-weight HF PCC assertions remain the numerical
   acceptance gate; merely avoiding the exception is insufficient.

3. **Ownership control if needed.** Repeat with a decoder ramp length 256 and
   per-call `chunk_size=128`. A proper 128-row ramp slice should own its buffer
   and leave the persistent ramp alive after deallocation. This isolates
   full-extent borrowing from recurrence or offset semantics. Treat this as a
   diagnostic control, not the fix.

No independent second root cause is supported by the supplied failure. Kernel,
cache precision, trace replay, and recurrent numerical changes are unnecessary
to explain an explicitly deallocated host-visible input. Additional failures
after H1's focused fix require their own evidence.
