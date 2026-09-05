# AutoDebug: eager prefill gaps

Source-only diagnosis before the experiment, 2026-09-05. The parent authorized
exclusive serialized hardware for the subsequent bounded test. No runtime model
or generator edits are part of this investigation.

## Findings and hypotheses

The reduced real-layer0/3 profile records rank1 kernel time3.043997ms and
interior gaps2.071031ms, totaling5.115028ms. The same signposted host window is
5.263950ms. The144 rows' native `HOST DURATION [ns]` sum is only176.7us; this
field cannot account for Python validation, uploads, object lifetime, dispatch
submission, or profiler reporting. Gaps before embedding139.271us, first head
76.784us, ChunkGdnPrep70.846us, embedding gather60.226us, and SDPA53.152us
therefore cannot be labeled mandatory device synchronization from this CSV.

Hypothesis A: profiler instrumentation increases the gaps. The smallest control
is exactly the `profile_reduced.py` prefill-plus-first-sample window, including
the public generator preparation, using the same native cache and128 tokens,
with neither Tracy nor device profiling enabled. Compare synchronized warmed
repeats to the recorded instrumented host5.263950ms; do not invent an
uninstrumented device-kernel split from a host timer.

Hypothesis B: fixed logical-shape prefill has removable eager dispatch overhead.
`model.py:prefill_forward` validates the request and uploads the per-slot page
table and IDs, then executes embedding, the decoder stack, final-row slice,
clone and terminal. `multichip_decoder.py:465` is device-only: continuation uses
preallocated device positions/rotary indices; nonaligned lengths retain existing
internal padding/trimming. `optimized_decoder.py:714` reads RoPE and page-table
slices on device and fills caller-owned paged KV. DeltaNet state updates preserve
addresses (`functional_decoder.py:361`, `fused_decoder.py:477`). This permits a
trace keyed by cache identity, logical length, start/slot and execution mode.
Host token/page updates and request state preparation remain outside capture.

An exact fixed-shape fast lane is therefore worth testing. Dynamic lengths do
not justify rejecting capture globally. A bounded cache may retain a recently
used trace; unsupported/new shapes can use the existing eager method with all
validation, mixed-slot, explicit-state and continuation semantics preserved.
This probe tests B1/start0 lengths128 and131, changed token IDs and permuted page
tables. It does not claim that every public case is captured or that an
unbounded trace cache fits the existing100MB region.

## Operation topology audit

| Boundary | Existing behavior | Contained experiment |
| --- | --- | --- |
| Validation / token and page upload | Host validation and new device tensors each request | Keep public-control timing; stable tensors refreshed outside a prepared-body comparison |
| Embedding and gather | Real vocab/hidden shapes, temporary prefill gather | Same ops in eager prepared body versus trace |
| Linear-attention layer | Packed projections, native conv/chunk GDN and persistent state updates | Same selected precision/config and native ops |
| Full-attention layer | Device RoPE/table slices, paged fill and SDPA | Same ops; changed pages and next decode verify state |
| Final slice / norm / head | Existing logical last row and selected C64/K1/R2 BF16 head | Same slice/clone/head, fixed norm |
| Request sampler preparation / sample | Seed/history preparation on host, chosen common device sampler | Preparation outside capture, identical device sampler inside |

## Verification and ownership plan

`probe_prefill_gaps.py` will first measure the unchanged public prefill/sample
window. Then, on the same reduced model and native cache, it will compare a
prepared eager body with the identical traced body, recording request refresh,
reset/preparation and synchronized total separately. Capture only after warmup;
release the generator's earlier decode traces before creating the probe trace.
Use one live prefill trace at a time, retain its output while replaying, and
release it before trying a new shape. No tracker suppression is permitted.

Require exact logical prefill logits, exact recurrent/conv state, and exact next
decode logits under original and changed token/page inputs. Record first-build
and capture cost separately from reuse. If a material exact gain exists, return
the helper/trace-key/lifetime requirements to the parent for integration and
full-model validation. A reduced-layer speedup is not a full-model TTFT claim.
Watcher and profiling stay separate. A capture/runtime exception is preserved;
a live hang follows tt-device-usage before any kill/reset.
