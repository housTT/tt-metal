# Explicit-upsample full-stack stall recurrence

Date: 2026-08-28 (America/New_York)

## Status

The full 48-layer server still stalls on request 4, the second consecutive
logical-length-63 prefill, after replacing the embedding expansion with the
explicit row-major `unsqueeze_to_4D -> upsample(scale=(1,4)) -> TILE` path.
This refutes the claim that avoiding the generic `repeat_interleave`
composite is sufficient to fix the serving failure. It does not identify a
new first non-completing device operation because the live Inspector capture
could not read any NoC endpoint on devices 0 or 1.

Server evidence:

- EngineCore PID: `918531`
- API server PID: `918467`
- Server log: `final_virtual_b2_upsample/server.log`
- Host stack: `autofix_full_output_stall_upsample_pyspy_3.txt`
- Bounded focused `dump_running_operations` output:
  `autofix_explicit_upsample_stall_running_ops.txt` (empty because the
  remote capture timed out)
- Current `model.py` SHA-256:
  `0064c64a5f2ae9e59dec744259c4f21ad3c2858095c2331c22cc9b326e0d25d0`

The server returned HTTP 200 for requests 1, 2, and 3. Its final model marker
is `event=prefill_start, completed_requests=3`; there is no fourth response.
The API metrics endpoint remained responsive and vLLM printed
`Running: 0, Waiting: 0`, while EngineCore was still blocked in the
synchronous worker call. Root preserved the evidence, stopped the process,
and owns the post-stop process/device audit and any recovery; those results
are not inferred from this live capture.

## Inspector runtime ledger

Inspector RPC reported devices in use `[0,1]` and the expected build
environment, but both devices returned all-NoC read timeouts when the live
capture attempted to inspect cores. Therefore there is no defensible
on-core first-stuck-op result for this recurrence.

The host-side mesh-workload runtime ledger was still available. For request
3, the first successful logical-length-63 request, it records this embedding
endpoint sequence:

| Runtime ID | Operation | Input logical shape/layout |
| --- | --- | --- |
| 91445 | Embeddings | `[1,1,1,63]` row-major |
| 91446 | TilizeWithValPadding | `[1,63,1280]` row-major |
| 91447 | UntilizeWithUnpadding | `[1,63,1280]` tiled |
| 91448 | UpsampleOperation | `[1,1,63,1280]` row-major |
| 91449 | TilizeWithValPadding | `[1,1,252,1280]` row-major |
| 91450-91451 | Copy | endpoint copies |
| 91452 | FillPad | `[1,252,1280]` |
| 91453 | Pad | `[1,1,252,1280]` |
| 91454 | LayerNormPreAllGather | layer 0, `[1,1,512,1280]` |
| 91455 | AllGather | layer 0, `[1,1,512,32]` |
| 91456 | LayerNormPostAllGather | layer 0, `[1,1,512,1280]` |

The ledger continues through the full request and final sampling path to
runtime ID 91840. Request 3 returned successfully. Runtime ID 91449 in this
run is consequently **not** the earlier failing rank-three
`ReshapeViewDeviceOperation`; it is the new post-upsample tilize. A cached
request-4 invocation reuses registered workloads and does not create a new
runtime ID, so the ledger alone cannot tell whether the second invocation of
91448/91449 or a later cached program stopped physically.

## Host synchronization boundary

Repeated `py-spy` placed the EngineCore main thread at:

```text
FDMeshCommandQueue::wait_for_outstanding_reads
  ttnn.to_torch
  _read_compact_route_ids (multichip_decoder.py:1434)
  _routed_experts (multichip_decoder.py:1700)
  _moe
  prefill_forward_host_backed_fractured (multichip_decoder.py:2225)
  Qwen38FullModel.prefill_forward (model.py:1880)
  generator._prefill_forward_virtual
  generator_vllm.prefill_forward
  vllm_tt_plugin.model_runner.submit_prefill
```

`Qwen38FullModel.prefill_forward` selects
`prefill_forward_host_backed_fractured` only for a PLE layer. In the canonical
stack, the first PLE layer is zero-based layer 1. The blocked host read thus
proves that request 4 passed the Python embedding call and entered layer 1's
MoE router after layer 0 had been submitted. `_routing_from_logits` has
already enqueued `topk`, `softmax`, row-major conversion of compact route
IDs, and `scatter` before `_read_compact_route_ids` calls `to_torch`.

This read is a synchronization victim: it waits for all earlier queue work,
so it does not prove that routing or layer 1 is the first device-side failure.
It does rule out a lost final async decode output, release hook, or client-only
stall: request 4 is still inside synchronous prefill.

## AutoFix hypothesis ledger

### Explicit upsample removes the cached-program stall

- Prediction: request 4 completes after removing tiled reshape and generic
  repeat-interleave lowering from multi-token embedding expansion.
- Result: request 4 stalls at the same request boundary.
- Verdict: **refuted as a sufficient fix**.

### Endpoint upsample/tilize itself is the new failing cached program

- Evidence for: the problem still occurs only on a repeated length-63 full
  prefill, and request 4 reuses the endpoint's shape-specific workloads.
- Evidence against: the exact endpoint sequence completed on request 3, and
  the physical first-stuck operation could not be read on request 4.
- Verdict: **still uncertain**, not proven.

### Same-shape cached full-stack prefill reuse fails downstream

- Evidence for: two different replacement endpoint lowerings pass focused
  reduced-stack tests but the full 48-layer second length-63 serving request
  stalls; the visible host wait is in layer 1 after endpoint and layer 0
  submission.
- Prediction: diagnostic synchronization markers will advance past embedding
  on request 4 and stop at one later boundary only for repeated length 63.
- Verdict: **leading independent hypothesis, unverified**.

## Smallest next experiment

Run separate all-48 serving probes with four logical-length-1 requests and
four logical-length-63 requests. Add diagnostic-only device synchronization
and progress markers at these boundaries:

1. immediately after `embed_tokens`;
2. after layer 0 returns;
3. in layer 1 after `_routing_from_logits` has produced the row-major compact
   route-ID tensor;
4. immediately before and after `_read_compact_route_ids`.

The first marker that does not complete on request 4 establishes a much
smaller queue interval without changing production operation lowering. If
embedding synchronization fails, isolate repeated 91448 upsample versus
91449 tilize directly. If embedding succeeds but layer 0 synchronization
fails, bisect layer 0. If layer 0 succeeds, bisect layer 1 before the compact
route read. Preserve the server, marker, Inspector, cleanup, and device-health
logs for the next AutoTriage pass.

## Read-only request-lifecycle audit

No host-expert or compact-route resource has a semantic threshold at three
requests:

- `_routing_from_logits` assigns one layer-local `_host_route_ids`; the
  following `_read_compact_route_ids` reads it before `ensure_wave`, and
  `_routed_experts` deallocates and clears it after all waves. Both
  `_host_route_rows` and `_host_logical_route_rows` are cleared in `finally`
  blocks.
- `ExpertSlotDirectory` has ten entries, an unbounded LRU clock and per-slot
  generations. Every wave validates the requested expert set against current
  valid generations. There is no third-request branch.
- Expert upload staging uses depth one per physical owner. The owner staging
  index is monotonically incremented modulo that depth, so there is no count-3
  transition. H2D, rank-local D2D, the expert computation and collectives are
  intentionally submitted to CQ0 without a per-wave completion fence.
- `PLEDeviceStaging` retains exactly one host source until the next PLE call.
  Its stated completion edge is the intervening route-ID read; it has no
  request-count capacity.

The one cumulative resource at exactly three is the virtual decode-state bank
reset count. Every finished request calls `release_virtual_slot`, which calls
`reset_slot` unconditionally. For sequential physical-B1 service the captured
metrics show three resets but zero bank commits and restores: each reset
therefore queues a full 260,718,612-logical-byte zero-copy tree for a slot
whose `_virtual_slot_banked` flag is already false. This is avoidable work and
deserves a separate cleanup, but it is unlikely to be the direct stopped
operation here. Request 4 reached layer 1; layer 0's compact route-ID read had
already drained the reset tree, embedding and the layer-0 router.

The remaining queue interval at the observed host wait is precise:

1. layer 0 after its route-ID read: host expert miss H2D, fixed-slot D2D,
   wave sparse computations, accumulation and reduce-scatter;
2. layer 1 before its route-ID read: state resets, PLE H2D/projection, GDN,
   hyperconnection mixing, router top-k/softmax/row-major conversion/scatter.

The narrow queue-order A/B is a diagnostic completion fence after every layer
0 `ensure_wave` upload/D2D submission and before `_routed_expert_wave` consumes
the fixed slots, followed by the same four length-63 requests. Completion only
with that fence verifies deferred expert staging/CQ ordering. A hang with the
fence refutes expert upload ordering and leaves layer-0 expert compute versus
the layer-1 front graph for the progress-marker bisection.

## Instrumented all-48 localization and expert-wave fence

The next all-48 progress run separated the shapes and produced a tighter
boundary:

- five consecutive length-1 requests passed;
- the first length-63 request passed;
- on the second length-63 request, `embedding:sync_done` and layer 0
  `route_sync_done` printed;
- the process then printed model `layer=0:sync_start` and wedged, with both
  devices again unavailable to NoC reads.

Thus the endpoint, layer-0 front graph and compact router read completed. The
non-completing interval is layer 0 after its router read: expert slot service,
wave compute/accumulation, shared-expert combination, reduce-scatter and final
hyperconnection output.

AutoFix added a diagnostic-only A/B at the cache/consumer boundary. Setting
`QWEN38_HOST_EXPERT_WAVE_FENCE=1` now logs an `upload_sync_start` marker,
synchronizes the mesh after `ensure_wave` and `validate`, logs
`upload_sync_done`, and only then calls `_routed_expert_wave`. The default path
does not synchronize. A host-only fake-cache test verifies both default-off
and enabled ordering:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_expert_wave_fence_is_opt_in_and_orders_upload_before_compute \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_expert_wave_fence_static.xml
```

Result: two parameterized cases passed in 0.32 seconds. The next server A/B is
the identical all-48 repeated-length-63 progress run with the fence variable
enabled. If it completes, deferred slot service versus consumption is
verified. If it wedges at a wave's `upload_sync_start`, the interval is that
wave's uploads/D2D plus any preceding wave compute; if every layer-0
`upload_sync_done` prints and the layer-0 model sync still wedges, the upload
ordering hypothesis is refuted and the sparse wave/shared/reduce/output graph
must be bisected.

The stronger fixed-slot lifetime hypothesis needs a distinct fence. A wave's
TT operations are only enqueued when `_routed_expert_wave` returns; without a
completion edge, the next `ensure_wave` immediately overwrites the same fixed
expert slots. Slow first-shape compilation can hide this lifetime violation,
whereas a cached second invocation submits the consumer and overwrite quickly.

`QWEN38_HOST_EXPERT_WAVE_REUSE_FENCE=1` therefore synchronizes immediately
after every non-final `_routed_expert_wave` and before the next
`ensure_wave`. It logs `compute_sync_start` and `compute_sync_done`. It is
separate from the pre-compute upload fence and deliberately emits no fence
after the final wave. A two-wave host test proves the enabled order
`compute(wave0) -> reuse fence -> ensure(wave1)` and that only one fence is
issued; its disabled control proves the unmodified `compute(wave0) ->
ensure(wave1)` order. The combined fence suite passed four parameterized cases
in 0.66 seconds; JUnit is
`readiness_vllm/autofix_expert_wave_fences_static.xml`.

The highest-causal cold A/B enables only the reuse fence plus prefill progress,
then repeats five length-1 and two length-63 requests. Completion verifies the
fixed-slot overwrite race. A wedge at `compute_sync_start` means the current
wave itself is non-completing before any following overwrite.

## Exact command-queue audit

The source does not support a separate physical-device versus parent-mesh
queue explanation. `ttnn.get_device_tensors` constructs coordinate-restricted
views that share the original `MeshTensor` holder
(`ttnn/core/distributed/api.cpp:68-84` and
`ttnn/core/tensor/storage.cpp:134-165`). `Tensor.device()` returns that
holder's parent `MeshDevice` (`ttnn/core/tensor/tensor.cpp:507-512`). Thus both
entries constructed as `slot_devices` in `host_weight_cache.py:652` resolve to
the same parent mesh; they are not independent physical `MeshDevice` command
queues.

The exact submissions are all parent CQ0:

- `copy_host_to_device_tensor` selects
  `device_tensor.device()->mesh_command_queue(raw_optional(cq_id))`; no CQ is
  supplied by the cache (`ttnn/core/tensor/tensor_ops.cpp:173-184` and
  `host_weight_cache.py:815-828`).
- `ttnn.copy` launches a normal device workload, and the device-operation
  dispatcher enqueues it through
  `mesh_device->mesh_command_queue()` (`ttnn/api/ttnn/device_operation.hpp:207-211`;
  cache D2D calls are at `host_weight_cache.py:830-841`).
- The two first fixed-slot consumers are the gate/up and down concats at
  `multichip_decoder.py:1639-1641`; their subsequent sparse operations use the
  concat bank tensors and follow through the same default mesh queue.

Consequently a physical-to-mesh event bridge is neither exposed by this path
nor justified by the source. `wait_for_event` queues the wait through the
event's own mesh device (`ttnn/core/events.cpp:25-27`). Bare
`synchronize_device(mesh)` drains every CQ; specifying `cq_id=0` drains only
CQ0 (`ttnn/cpp/ttnn-nanobind/device.cpp:607-624`). A parent-CQ0 event followed
by `event_synchronize` is the narrowest verified host completion edge.

If the full reuse-fence A/B passes, the smallest follow-up is a slot-release
event recorded immediately after both concats. The rest of the wave can then
be enqueued, and the host waits on that event only before the following
`ensure_wave` overwrite. This completes uploads, D2D and the last slot readers
without waiting for the downstream sparse compute. Every non-final wave is a
full capacity-ten chunk, so neither concat can use the documented single-input
alias case; the final wave needs no release wait.

This event A/B is needed before calling the fence a production fix: same-CQ
FIFO should already order these commands, so the broad post-compute sync also
changes queue depth and all outstanding work and cannot by itself prove a
cross-domain overwrite race. A second ten-slot bank is not a lower-overhead
answer: it adds `48 * 10 * 2,764,800 = 1,327,104,000` bytes per rank for the
full stack and still needs a completion edge before a third wave reuses the
first bank.

## Reuse-fence refutation and routed-stage bisector

The all-48 reuse-only A/B did not complete. On the failing second length-63
request, layer 0 wave 0 printed both `compute_sync_start` and
`compute_sync_done`; the next visible marker was the model's layer-0
`sync_start`, where the devices wedged. The wave-0 consumer therefore did
complete before wave 1's slot service. Because wave 1 is final and has no
following overwrite, the fixed-slot overwrite theory is refuted as a
sufficient explanation. The remaining interval contains final-wave routed
work, wave accumulation, shared expert, the MoE collective, and the final
hyperconnection injection.

`QWEN38_HOST_EXPERT_STAGE_FENCE=1` now supplies diagnostic-only, layer/wave
qualified completion boundaries at these exact points:

1. bank concat plus route/sparsity preparation;
2. gate/up sparse matmul;
3. gate/up activation and token weighting;
4. down sparse matmul;
5. fast reduce and output reshape;
6. final routed-wave accumulation;
7. shared expert;
8. MoE reduce-scatter;
9. post-injection and logical trim layer tail.

Every boundary logs `QWEN38_EXPERT_STAGE_FENCE layer=<L> wave=<W>
stage=<name>:sync_start` and `:sync_done`; the environment variable defaults
off. Both ordinary fractured prefill and the layer-1 host-PLE prefill wrapper
include the layer-tail boundary. The next cold A/B should enable this variable
with both earlier wave-fence variables unset, then run the same five length-1
and two length-63 requests. The first `sync_start` without a `sync_done` names
the exact preceding subgraph; if all nine layer-0 boundaries finish, the model
layer sync failure lies outside MoE in the residual-boundary handoff.

Host/static verification:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_expert_wave_fence_is_opt_in_and_orders_upload_before_compute \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_expert_wave_reuse_fence_precedes_next_slot_overwrite \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_expert_stage_fence_is_opt_in_and_source_ordered \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_expert_stage_fences_static.xml
```

The current combined rerun passed all six parameterized cases in 1.38 seconds.
JUnit `readiness_vllm/autofix_expert_stage_fences_static.xml` has SHA-256
`459c480b4c9ec3abab4866687a9959ebae289bd35ae5ae50df4d1bca6f68fe67`.
The test proves the new gate is default-off, checks layer/wave marker context,
and statically asserts the internal wave stages plus
final/shared/collective/tail order.

## Final-wave sparse-matmul source audit

The length-63 prefill is padded to 128 routed rows, hence four 32-row groups.
The layer-0 log proves exactly two cache waves; with ten resident slots, the
final wave contains between one and ten unique experts. For a final-wave size
`E`, the host-backed sparse operands are:

- gate/up: A `[1, 4, 32, 2560]`, B `[1, E, 2560, 1280]`, sparsity
  `[1, 1, 4, E]`;
- down: A `[4, E, 32, 640]`, B `[1, E, 640, 2560]`, with the same sparsity.

The exact value of `E` was not preserved by the existing Inspector artifacts,
so claiming it more narrowly would be speculation. Existing sparse-matmul
unit tests do not cover this BFP4, four-group, `E=1..10`, Qwen-width, repeated
variable-bank program-reuse combination.

One directly applicable source defect is present at current commit
`3b928912d26`: upstream commit `1f4441216b3` (`fix(ttnn): size sparse matmul
FP32 intermediate buffers correctly`) is not an ancestor. The current sparse
factory sizes `interm0_single_tile_size` from BF16 `output_data_format` at
`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:119-120`, but then
selects a Float32 intermediate at lines 191-194. Qwen's selected expert policy
uses BFP4 LoFi with both `fp32_dest_acc_en=True` and `packer_l1_acc=True`.
Gate/up has `Kt=80`, `in0_block_w=16` (five blocks); down has `Kt=20`,
`in0_block_w=5` (four blocks). Both therefore enable L1 spill accumulation and
allocate a 2048-byte BF16-sized page for each 4096-byte FP32 tile: a verified
twofold circular-buffer under-allocation.

Upstream fixes this by sizing the intermediate page from
`interm0_data_format`, and its Blackhole regression proves result corruption.
It does not prove a device hang, so this remains a source-exact candidate, not
the causal verdict. The stage-fence experiment is the discriminator: an
unfinished `gate_up_sparse_matmul` or `down_sparse_matmul` synchronization
justifies applying the upstream fix in isolation and running a repeated
small-`E` Qwen-shape regression. Completion of both sparse boundaries refutes
this defect as the cause of the serving stall, although it remains an
independent correctness issue.

## Final-wave preparation split

The stage-fence server completed the first length-63 request through all 48
layers. On the second length-63 request, layer 0 wave 0 completed all five
internal wave boundaries. Wave 1 then printed
`stage=bank_and_sparsity:sync_start` without a matching completion. No wave-1
sparse matmul had been launched. This refutes the missing FP32 intermediate-CB
fix as the trigger for this hang while retaining it as an independent
correctness repair.

The same opt-in `QWEN38_HOST_EXPERT_STAGE_FENCE=1` now splits that interval:

1. `slot_upload`, immediately after `ensure_wave` and directory validation;
2. `bank_concat`, immediately after gate/up and down bank construction;
3. `route_sparsity`, after route selection, grouped reshapes, max reduction,
   sparsity reshape, and row-major conversion.

An adjacent `QWEN38_EXPERT_STAGE_PLAN` line records exact expert, hit, miss,
and eviction identities for each wave. The first unmatched split marker in the
next identical cold run will distinguish cache H2D/D2D service, expert-bank
concat, and route/sparsity preparation without changing default behavior.

Host verification ran
`test_host_expert_stage_fence_is_opt_in_and_source_ordered` in both disabled
and enabled modes. Both cases passed in 0.57 seconds. The test asserts
`ensure -> validate -> plan log -> slot_upload -> wave compute`, then
`bank_concat -> route_sparsity -> gate/up -> elementwise -> down -> reduce`.
JUnit: `readiness_vllm/autofix_expert_stage_split_static.xml`, SHA-256
`ec056bb58274e35b4a1bf5e7711539f89ad307005e349106f0500cd3ac3d1608`.

The cache directory also proves the layer-0 wave-1 service composition without
recovering the missing expert identities. `waves` de-duplicates route IDs and
splits them into disjoint capacity-ten chunks. A wave 1 therefore implies a
full ten-expert wave 0; after wave-0 `ensure`, all ten slots contain exactly
that protected wave. Every disjoint wave-1 expert is consequently a miss:
wave 1 has zero hits and `E` misses, where `E` remains between one and ten.

No stale program-cache address was found in that service path. Host-to-device
copy directly enqueues the live destination tensor and has no cached device
program. D2D copy registers both source and destination buffers and the
descriptor adapter rebinds them on a cache hit. Concat similarly registers all
slot inputs plus its output; different `E` values are separately keyed by the
input-vector length and TensorSpecs, while `E=1` returns the persistent slot
alias without launching concat. The same process completed layer-0 wave 1 for
the first length-63 request (`server.log:6764-6798`) before the later identical
logical length failed there (`server.log:11090-11109`), further weakening a
deterministic copy/concat cache-reuse theory.

One separate source issue was found: the tensors described as rank-local
upload staging are allocated through coordinate-filtered tensor views whose
`.device()` remains the parent mesh. The subsequent `from_torch` therefore
creates replicated staging, and each purported owner-only H2D broadcasts to
both ranks. Coordinate-restricted D2D still selects the correct target shard,
so this is not stale-address evidence for the hang; it likely doubles physical
H2D traffic and means the current H2D-byte metric undercounts actual traffic.

## Route-sparsity reshape split

The next cold run narrowed the repeated length-63 failure further. The first
length-63 request and the failing second request selected the identical layer-0
final wave: seven experts `(238, 425, 489, 359, 85, 283, 52)`, all misses, with
the identical seven evictions. On the failing request, `slot_upload`,
`bank_concat`, `route_index_upload`, `route_gather`, `route_group_reshapes`,
and `route_sparsity_max` all synchronized successfully. The first unmatched
marker was `route_sparsity_to_layout:sync_start` at
`readiness_vllm/autofix_route_sparsity_split/server.log:18935`; the exact plan
is at line 18922 and the successful max boundary is at lines 18933-18934.

That marker followed two operations, not one: a tiled reshape and row-major
layout conversion. The max output has logical shape `[1, 4, 1, 7]` and padded
shape `[1, 4, 32, 32]`; the requested reshape is logical `[1, 1, 4, 7]` with
padded shape `[1, 1, 32, 32]`. Its physical volume changes from 4096 to 1024,
so this is not safely attributable to metadata-only reshape. The subsequent
`to_layout` uses the unpadding path. Program-cache stale bindings were not
found for gather, max, or layout conversion: each registers input/output
buffers and the generic mesh cache-hit path patches current addresses, while
their keys include the exact tensor specifications and layout parameters.
Immediate index deallocation is also outside the failing interval because the
post-gather synchronization completed.

`QWEN38_HOST_EXPERT_STAGE_FENCE=1` now emits a new
`route_sparsity_reshape` synchronization immediately after the reshape and
before `to_layout`. This makes the next identical cold run decisive: a hang at
`route_sparsity_reshape:sync_start` assigns the failure to the reshape; a
completed reshape followed by a hang at `route_sparsity_to_layout:sync_start`
assigns it to layout conversion. No workaround or default-path behavior was
introduced.

The focused host/static gate verifies the exact
`max call -> max marker -> reshape call -> reshape marker -> to_layout call ->
to_layout marker` order in both opt-in and disabled modes: 2 passed, 49
deselected in 0.26 seconds. JUnit:
`readiness_vllm/autofix_route_sparsity_reshape_split_static.xml`, SHA-256
`bc91f764f0565b8b8c78bb287cc8f32f39cf02b0d7bad428c6707cc31404cb34`.

The final discriminator run proved the operation. Five length-1 requests and
the first length-63 request completed. On the second length-63 request, the
identical layer-0 final wave again selected seven experts, all misses. Max
completed at
`readiness_vllm/autofix_route_sparsity_reshape_split/server.log:20135-20136`;
the following `route_sparsity_reshape:sync_start` at line 20137 never
completed, and no `to_layout` was submitted. This assigns the device deadlock
to the exact tiled reshape from logical/padded `[1,4,1,7]/[1,4,32,32]` to
`[1,1,4,7]/[1,1,32,32]`. Internally it becomes a four-input-page to
one-output-page tiled mapping.

The isolated repair changes only the order of the two structural operations:
max is first converted to row-major in its native `[1,4,1,7]` shape, then
reshaped to `[1,1,4,7]`. Both row-major shapes have the same physical
`[groups,E]` footprint and last dimension, so `ttnn.reshape` selects the
metadata-only view path; sparse matmul still receives a row-major tensor with
the same 28-element logical volume. The stage markers now record
`max -> to_layout -> reshape` in that actual order.

The updated host/static gate passed both opt-in modes: 2 passed, with the new
hardware regression collected but skipped unless
`RUN_QWEN38_ROUTE_SPARSITY_TT=1`. JUnit:
`readiness_vllm/autofix_route_sparsity_rm_first_static.xml`, SHA-256
`4cc338ae2a8bc6f44eb713fae4c553643872884150f8bc48c3041ae7bf0b7470`.
The opt-in TT test repeats four fresh `[1,4,32,7]` inputs, checks exact max and
reshape output, proves the row-major reshape preserves physical addresses,
and asserts the program-cache count is stable after the first iteration.

The focused TT regression subsequently passed: 1 test, suite time 1.530
seconds and test time 1.495 seconds. Program-cache counts remained
`[4, 4, 4, 4]`; input/output row-major reshape address pairs were
`[(5848704,5848704), (5850816,5850816), (5848704,5848704),
(5848768,5848768)]`, directly proving that each reshape was a metadata-only
view. JUnit: `readiness_vllm/autofix_route_sparsity_rm_first_tt.xml`, SHA-256
`51f82149c97ee79c81cdd9836418d293668f3143f5c0e3d05e1249e1ca63cba4`.

The independent sparse-matmul FP32 intermediate-CB correction was rebuilt and
its exact Blackhole regression passed: 1 test, suite time 1.379 seconds and
test time 1.351 seconds. JUnit:
`readiness_vllm/autofix_sparse_matmul_fp32_cb.xml`, SHA-256
`3bef465d183d651cc36e67c1a72cd7338e6dfe9f8d9cd83fa51bcfb3ae6b6cec`.

## Diagnostic cleanup

After the causal reshape fix passed on device, all refuted experiment-only
serving scaffolding was removed from production: the upload fence, inter-wave
reuse fence, per-stage fences and plans, and prefill-progress synchronization
and prints. The production source contains none of
`QWEN38_HOST_EXPERT_WAVE_FENCE`,
`QWEN38_HOST_EXPERT_WAVE_REUSE_FENCE`,
`QWEN38_HOST_EXPERT_STAGE_FENCE`, or
`QWEN38_VLLM_DEBUG_PREFILL_PROGRESS`.

The currently unused `_routed_expert_slot` helper was retained because it
predates this diagnosis and its ownership is not established, but its
sparsity preparation now uses the same safe `max -> row-major -> reshape`
order. This is the smallest behavior-preserving change and prevents the
hazardous tiled reshape from returning if that helper is reactivated. The
static regression inspects both routed-expert helpers, requires row-major
conversion before reshape and sparse matmul, verifies cache service precedes
wave compute without synchronization hooks, and rejects all removed debug
symbols in production methods.

The final cleanup gate passed: 2 tests passed, the opt-in TT test was skipped,
45 tests were deselected, and the suite completed in 0.25 seconds.
`py_compile` and `git diff --check` also passed. JUnit:
`readiness_vllm/autofix_diagnostic_scaffolding_cleanup_static.xml`, SHA-256
`9e07e5d6b24d4e609d9add2ff8ddbd2aab4df54af2577d3fba0fa066da1f7878`.

## Final verdict

The explicit-upsample change was refuted as a serving fix. The full-stack
failure was assigned exactly to the layer-0 final-wave tiled sparsity reshape
on the second length-63 request. The row-major-first reorder removes that
device program while preserving layout, volume, and downstream sparse-matmul
contracts. Host/static verification and the focused repeated-shape TT
regression pass. An identical full-server repeat remains required before the
serving repair is proven end to end.
