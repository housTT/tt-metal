# AutoDebug: KDA sigmoid-gated RMSNorm candidate

Date: 2026-08-26

## Scope

Investigate why the previously reconstructed prefill-only
`ttnn.experimental.kda.sigmoid_gated_rms_norm` candidate was reported to pass
layer-1 prefill PCC but reduce the following traced-decode PCC to about 0.86.
No optimized-decoder or later-stage path is in scope.

## Fresh-context result and correction

The repository AutoDebug launcher was run from this directory against
`../../tt/fused_decoder.py`. Its isolated shell could not start because its
bubblewrap network namespace failed with `RTM_NEWADDR: Operation not
permitted`; its two internal subagents saw the same failure and it could not
write this report. It recovered the KDA binding and unit-test contract through
indexed source, but proposed a Qwen SiLU-versus-sigmoid mismatch as the likely
cause.

Direct inspection of the checkout disproves that hypothesis:

- `tt/functional_decoder.py::_gdn_epilogue` computes per-head RMSNorm followed
  by `sigmoid(z)` and multiplication.
- The KDA binding computes the same `RMSNorm(input) * weight * sigmoid(gate)`
  and converts `[B*H,T,V]` to `[B,T,H*V]`.
- `linear_attn.norm.weight` is uploaded as one shared 128-element value-head
  weight, exactly the KDA op's rank-one weight contract.

Thus the dedicated op is mathematically compatible for prefill chunks
`T=128`, `H=48`, `V=128`. Decode `T=1` is binding-illegal and must retain the
primitive path.

## State-flow proof

The prior claimed failure mechanism is impossible for an epilogue-only change:

1. `_gdn_prefill` updates `user_conv_state` before the recurrence.
2. It assigns `user_recurrent_state` before calling the epilogue.
3. Layer-1 PLE state is written earlier by `_ple_prefill`.
4. `prepare_decode_state()` copies only those three persistent tensors.
5. `_gdn_epilogue` writes none of them.

The previous hand summary retained neither its source nor raw output, so its
actual branch scope and head-major reshapes cannot be audited. The reported
decode collapse therefore cannot be accepted as evidence against the op.

## Repair hypothesis and discriminating experiment

Reconstruct the candidate with a separate `_gdn_prefill_epilogue` method and
call it only from `_gdn_prefill`. Request `[B*H,T,V]` directly from
`chunk_gated_delta_rule_fused_adapter(return_o_bh=True)` and pass that tensor
without a token-major reinterpretation. Keep `_gdn_decode` and the inherited
primitive `_gdn_epilogue` unchanged.

For real layer 1, run the same layer and input once with the primitive prefill
epilogue and once with the dedicated prefill epilogue. Snapshot and compare
`user_recurrent_state`, `user_conv_state`, and `user_ple_conv_state` before
`prepare_decode_state()`. Any difference proves an unintended branch/layout or
buffer-lifetime defect; bit-identical states isolate numerical comparison to
the epilogue output. Then run the existing real-weight prefill/traced-decode PCC
gate and like-for-like warmed timing. Promote only if every layer kind remains
above the functional PCC bar and performance improves.

## AutoFix outcome

The exact original candidate was recovered from the Codex session journal and
rerun. It changes only prefill: the decode core remains rank four and never
enters the KDA branch. The three saved user states and their prepared device
copies are bit-identical to the primitive control. A source-level kernel audit
also bounds every input/gate/output access to pages 0..767 and the weight to
pages 0..3; only the newly allocated output is writable.

Fresh-process tests then exposed a supported-lifetime discriminator. If the
temporary decode input objects are discarded inline, eager/replayed layer-1
decode passes at PCC 0.99988902. If the caller retains those valid input
objects—the behavior used by the public correctness gate—first eager decode
fails at 0.86030912 and traced decode fails at 0.86216938. Moving the KDA
output to L1, deferring core/gate lifetimes, and clearing the program cache did
not repair it. Therefore the earlier state-perturbation explanation was wrong,
but the candidate still cannot be delivered: composing the op changes
allocator/address behavior and makes a legal caller lifetime numerically
incorrect. The exact patch, state probes, failed/passing XMLs, and seven-run
timing transcript are retained in `candidates/`; the candidate log is the
authoritative disposition.

## Primary source contracts

- `tt/functional_decoder.py::_gdn_epilogue`
- `tt/functional_decoder.py::prepare_decode_state`
- `models/demos/blackhole/qwen36/tt/gdn/fused_chunk.py::chunk_gated_delta_rule_fused_adapter`
- `ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/`
- `tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_sigmoid_gated_rms_norm.py`
