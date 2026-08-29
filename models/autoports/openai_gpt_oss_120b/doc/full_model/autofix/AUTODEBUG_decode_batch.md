# AutoDebug: mixed-slot decode batch mismatch

Date: 2026-08-29
Mode: source-only; the hardware repro was not run.

The repo AutoDebug runner was invoked with Codex 5.5/xhigh, but its fresh worker
could not initialize the bubblewrap read sandbox. The findings below were then
checked directly against the local source.

## Headline finding

The failure is deterministic batch-metadata drift at the maintained GPT-OSS
model boundary, not a decoder shape or inactive-row bug.

`models/demos/gpt_oss/tt/model.py::ttnn_decode_forward` computes
`actual_batch = current_pos.shape[-1]` and uses it to trim the padded token
buffer. That produces the reported hidden tensor `[1, 1, 2, 2880]`. It then
calls `_forward_layers_and_head` without `batch_size`, so that method retains
its default `batch_size=1` and passes 1 to every layer. The autoport
`_LayerAdapter` forwards that stale value unchanged. The optimized decoder's
explicit invariant—`hidden_states.shape[-2] == batch_size`—correctly rejects
the contradictory pair `(2, 1)`.

The legacy GPT-OSS decoder did not expose this omission because its decode
attention derives batch directly from `hidden_states.shape[2]`; its explicit
`batch_size` parameter is only used on prefill. The autoport decoder makes the
previously latent caller bug visible.

## Smallest correct fix

At the shared source boundary, pass `batch_size=actual_batch` in the
`ttnn_decode_forward` call to `_forward_layers_and_head`.

If the full-model stage must remain autoport-local, the equivalent smallest
fix is to make `_LayerAdapter` replace the inherited decode value with
`int(hidden_states.shape[-2])` before calling `decoder.decode_forward` (and
optionally assert that `position_idx.shape[-1]` covers the same span). Do not
derive batch from the number of non-negative positions.

For the repro, both `hidden_states.shape[-2]` and
`current_position.shape[-1]` describe the two fixed slots. The `-1` position
marks row 1 inactive; it does not remove that row from the token, page-table,
RoPE, sampler, or trace buffers. Therefore the effective decode batch must stay
2, preserving the inactive slot through trace capture and replay.

## Ranked hypotheses

1. **Confirmed:** omitted `batch_size=actual_batch` leaves the inherited
   default at 1; this accounts exactly for the reported shape/value pair and
   failure site.
2. **Equivalent local manifestation:** `_LayerAdapter` trusts metadata that is
   stale specifically on the mature decode call path. Deriving the physical
   fixed-slot span from the decode hidden tensor repairs the local contract.
3. **Rejected:** counting only active positions would yield 1 here, reproduce
   the mismatch, and violate fixed-slot/inactive-row semantics. Changing the
   optimized decoder validator or collapsing the batch would merely hide the
   caller error.

## Minimal verification after the fix

Run the supplied two-layer mixed-prompt test. It should compile and capture
with decode batch 2, return two sampled slots, and preserve persistent
positions `[8, -1]`. A small host unit should also assert that a mocked adapter
receives `batch_size=2` for hidden shape `[1, 1, 2, 2880]` even when the
inherited argument is 1.
