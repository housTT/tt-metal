# Host-backed segmented trace AutoFix

Date: 2026-08-27

## Final status

Progressing batch-one GDN trace replay is enabled on the target 1x2 P300 mesh.
The kept repair leaves the optimized GDN decode graph unchanged: it computes
the exact current-token output and performs its ordinary newest-tap state copy.
Afterward, one extra optimized packed projection slices the same mixed row
directly into the original persistent FP32 L1 tap as the final corrective
writer.

The final implementation retains the common two-segment topology:

1. a TT front trace through PLE/GDN or QSA and routing;
2. declared host route service and gate-selected expert staging; and
3. a TT back trace through active experts, shared expert, hyperconnection, and
   reduction.

No relocation guard, alternate state bank, or PLE-only trace remains in the
delivered path.

## Observed failure boundary

The initial layer-1 comparison used changing hidden inputs and token ids that
included EOS transitions. The first Python-visible divergence was repeatedly
localized to `fused_conv_state[-1]` after a front replay. At that boundary the
recurrent state, both older GDN taps, all nine PLE taps, and the compact staged
projection row were exact. The later route/output divergence followed the bad
history value. Host expert service and the back trace occur after that boundary.

Those observations ruled out PLE history, expert staging, and the back trace as
the first producer. They did **not** prove a particular allocator collision,
runtime-argument bug, or low-level L1 fault. Python tensor addresses cannot see
every captured temporary range, and no trace-allocation dump identified a
specific conflicting range.

## Kept corrective-writer repair

For batch-one decode, `_gdn_decode` first calls the unchanged optimized-decoder
implementation. That graph computes and returns the current-token output,
updates recurrent state, shifts the convolution history, and performs its
ordinary transient-to-newest-tap copy. Its output numerics and program ordering
therefore remain the optimized single-chip baseline.

Only after that output and state transition are complete,
`_commit_newest_gdn_state_direct` repeats the optimized packed FP32
`gdn_qkv_b_a` projection through `_linear_impl`, including its bias, datatype,
compute-fidelity, and program-selection policy. It slices the mixed prefix
directly into `fused_conv_state[-1]` with
`ttnn.slice(..., output_tensor=...)`, making that slice the final writer, then
deallocates the extra packed projection. The corrective projection does not
participate in the already-computed token output.

On the host-backed path, the direct slice targets the stable newest FP32 L1
tap in the shared trace workspace, after which the ordinary workspace commit
copies all state to the layer-owned canonical DRAM tensors.  Both the L1 trace
target and the canonical DRAM tap preserve their shape, padded shape, datatype,
tiled layout, address, and unique ID across capture/replay.  On the resident
batch-one path the same direct slice targets the layer's own L1 tap.  This also
avoids relying on a manually reconstructed GDN graph for output numerics.
Decode with `max_batch != 1` returns to the unchanged optimized-decoder
implementation because the fixed batch-one output tensor is not a valid
batch-32 destination.

Capture uses temporary replicated DRAM snapshots and device-to-device copies
to restore progressing GDN and PLE state after warm compilation and after front
capture. The first logical token is an ordinary replay from the restored
caller state. Snapshots are freed after capture; they are not a steady-state
shadow. Captured input tensors are held strongly until consumer-first trace
release, trace release remains retryable after a partial failure, and host
backing cannot close while a segmented trace is active.

## Eight-token validation

Command:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
pytest -sv models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py \
  -k host_backed_gdn_segmented_trace_progression
```

Result:

```text
PASSED ...progression[blackhole-True-0-device_params0]
PASSED ...progression[blackhole-True-1-device_params0]
2 passed, 35 deselected in 9.04s
```

The two cases cover plain GDN layer 0 and PLE+GDN layer 1. Each begins from an
already-progressed eager prefix and executes eight logical tokens
`(23, 91, 248044, 7, 248044, 19, 31, 248044)` with changing hidden values and
positions. On every token:

- recurrent state and all three GDN taps are bit-exact against eager TP2;
- layer 1's nine PLE taps are also bit-exact, including repeated EOS/history
  transitions;
- route IDs are exact;
- decoder output clears PCC `>= 0.995`; and
- the canonical DRAM newest tap and shared L1 workspace target retain their
  addresses, unique IDs, tensor configurations, and allocations through trace
  release.

This is evidence for the graph-level repair on the target runtime. It is not a
claim that the underlying TTNN/firmware mechanism has been identified.

## 100-token all-kind stress

The same final implementation was stressed for 100 progressing tokens on plain
GDN layer 0, PLE+GDN layer 1, and QSA layer 3:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
export QWEN38_MC_TRACE_STRESS_STEPS=100
timeout 7200 pytest -q -s \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/host_backed_trace_stress100.xml \
  --disable-warnings --tb=short
```

Result:

```text
3 passed in 60.67s
```

The GDN cases retain exact route IDs and exact recurrent/GDN/PLE state at every
step while clearing output PCC `>= 0.995`. The QSA case retains exact route IDs,
clears output PCC on every step, and finishes with exact local KV/index caches.
The durable JUnit artifact is `host_backed_trace_stress100.xml`.

## Two-live-trace allocation AutoFix

A later allocation-tracked stack test retained layer-0 and layer-1 traces
against one shared L1 state workspace.  Its first run found seven live younger
allocations in an older trace's captured ranges.  Six were the complete
`HostDecodeFront` crossing set; these are valid targeted corruptible buffers
because their own front trace regenerates every field before host service or
the back trace reads it.  The seventh was a lazy BF16 DRAM Clone program-cache
buffer.  It remains live for future launches and cannot be marked corruptible.

The accepted protocol prewarms every distinct stack capture signature before
any trace is registered, then passes `programs_prepared=True` to every capture.
This freezes program-cache misses before snapshots or warm execution.  Only
the six front crossings, PLE staging, and the back output are marked
corruptible.  Snapshot tensors are ordinary D2D DRAM clones, are restored and
freed before older replay, and are never marked.

`shared_trace_alloc_autofix.xml` and the broader
`final_current_correctness_alloc.xml` pass with two simultaneous layer traces,
stable workspace addresses/IDs, exact independent GDN/PLE state, exact routes,
and output PCC `>=0.995`.  Capture, replay, and back-before-front release are
serialized by the shared workspace.

## Refuted variants

Every isolated candidate below was removed from the delivered source.

| Variant | Observation | Conclusion |
| --- | --- | --- |
| Fixed pre-capture DRAM source for the newest row | Same progressing-replay failure | Transient source lifetime alone was not causal |
| `add(..., output_tensor=L1)` instead of `copy` | Same failure | A `copy`-opcode-only explanation was refuted |
| Newest or all convolution taps resident in DRAM | Changed the optimized numerical/kernel contract; PCC was about 0.8 | Not a baseline-preserving fix |
| Attention/router trace split | Later replay still failed | A monolithic front lifetime was not sufficient to explain the failure |
| Tiny captured state commit | Same later state failure | Merely moving the commit to another captured trace did not repair it |
| State commit after back, or warmed eager D2D commit | Same failure | Commit phase and captured-versus-eager dispatch were refuted |
| Canonical DRAM shadow restored into the same L1 buffers | Same failure | Cross-replay storage alone did not repair the writer graph |
| Relocate newest tap and retain the old allocation as a guard | Plain layer 0 passed, but layer 1 failed with only the newest tap wrong | Not a valid all-GDN solution; original-address theory refuted |
| Reorder nine PLE L1 allocations around the GDN tap | Destination moved as predicted, but layer 1 failed at the same token | Simple persistent-range placement refuted |
| Two alternating newest-tap front traces | Distinct stable banks; phase AB still failed on its second deployed use | Single-address and immediate read/write-cycle explanations refuted |
| Separate PLE trace feeding a stable GDN/router crossing | All PLE taps stayed exact; newest GDN tap still failed | PLE scratch in the same trace was not causal |
| Manually reordered direct-producer GDN graph with the packed tensor retained | Layer-0 optimized-baseline PCC was `0.99200940` | Trace progression alone was insufficient; the graph missed the required optimized-baseline PCC |
| Same manual graph with explicit mixed-view deallocation | Layer-0 optimized-baseline PCC fell to `0.94199741` | Explicit deallocation did not recover baseline numerics and was substantially worse |

Single-layer allocation tracking became clean after the explicitly disposable
final output was marked corruptible.  The later two-live-layer failure showed
why that was not a sufficient stack proof; the focused two-phase program warm
and targeted crossing policy above is the accepted full-stack allocation
contract.  Full-state ping-pong was analyzed but not run; it is not listed as
refuted evidence and was unnecessary after the final corrective writer passed.

## AutoDebug limitation

The required fresh `autodebug.sh` process was launched from
`/tmp/qwen-segmented-autodebug-run`. Its nested bubblewrap sandbox could not
start local shell commands or write its requested report. Consequently, no
nested result was accepted as executable proof. The outer investigation wrote
the source-only provisional report at
`/tmp/qwen-segmented-trace-AUTODEBUG.md` after independently checking the
checkout.

That provisional report correctly summarized the then-known boundary and
ranked experiments, but its strongest destination-address hypothesis was later
refuted: the final corrective slice passes while retaining the original tap
address. It is therefore historical diagnostic input, not the final root cause.
The existing repo-local `AUTODEBUG.md` documents a separate environment
provenance investigation and was intentionally left unchanged.
