# AutoDebug: request-boundary decode trace reuse

## Verdict

**Verified from source:** the serving adapter discards the canonical model and
sampling traces at every prefill-to-decode transition, including a new greedy
request with the same execution shape, cache, and device page-table tensors.
The generator explicitly supports retaining those traces across prefill and
refreshing request inputs before replay. This is a concrete avoidable capture
candidate. **Still uncertain:** how much of the measured TPOT overhead it
causes, and whether retained traces pass the adapter's complete buffer-lifetime
and serving correctness contracts. No performance improvement is claimed.

This is the isolated, source-only initial AutoDebug pass for AutoFix at
tt-metal baseline `6eb04274233`. Only this report was written. No model source,
prior-stage artifact, hardware state, server, or profiler was changed or run.
The coordinating agent owns the unchanged three-profile serving baselines.

## Observed evidence

### AutoFix follow-up: greedy-key assumption refuted

The first TP4 reduced-adapter hardware probe after removing request-only
releases still recaptured on request B: model/sampling IDs changed from 0/1 to
2/3. It reached the reuse assertion without an allocation-tracker failure
(`trace_reuse_probe/tp4.log`, `tp4.xml`). This refutes the original report's
assumption below that an adapter greedy request reaches `_sampling_spec` with
temperature zero. The adapter calls `format_sampling_params`, which converts
zero temperature into k=1, p=0, temperature=1. The canonical generator therefore
builds a non-greedy semantic key containing the request's changing seed.

A standard-library AST-isolated execution of the actual formatter and adapter
confirmed temperature 1 and seeds 104729/209458 across epochs 1/2 before the
follow-up change. The minimal adapter correction restores temperature zero
only when all original request temperatures are zero. The generator already
maps this sentinel to its existing greedy device parameters and seed-independent
key. Positive-temperature and mixed requests keep their prior translation.
`trace_reuse_probe/greedy_translation.json` records the corrected host proof;
this is not hardware validation of the correction. The reduced probe now logs
requested/cached keys, capture/release caller locations, and per-request state
before its reuse assertions, retaining partial evidence on failure. Parent-owned
hardware reruns and serving measurements remain required.

`../../readiness_vllm/optimized_vllm/before/P150/vllm_benchmark.json` records the
greedy primary workload: 128 input tokens, 128 output tokens, one request,
concurrency one, 32 scheduler slots. Mean TPOT is **28.433678 ms**, median ITL
**26.826434 ms**, P99 ITL **27.167832 ms**, TTFT **276.775386 ms**, and
TPOT-derived decode throughput **35.169562 tokens/s/user**. The corresponding
`vllm_result.json` reports ITL standard deviation **18.095431 ms**.

For 127 decode intervals, total excess over median ITL is approximately
204.120 ms. A deliberately simplified model with 126 equal intervals and one
outlier fits the mean/standard deviation with 26.821613 ms ordinary intervals
and one 231.554393 ms interval. This supports investigating one setup delay;
it does **not** establish that the first decode caused it. The saved raw JSON
contains aggregate metrics only, with no per-token `itls` array or capture
timing. Source predicts that capture is charged after the first prefill token,
so it can affect TPOT without appearing in TTFT.

The baseline server log has an allocator warning at 19:30:18.062. It identifies
neither a particular allocation nor corruption. `allocator.cpp:123` emits this
warning at most once per host thread for the process lifetime; warning counts
cannot count unsafe boundaries or certify a candidate.

## Source findings

Paths in this table are relative to the model directory, except explicitly
named runtime or sibling-vLLM sources. Line numbers refer to the inspected
baseline.

| Finding | Evidence and implication |
| --- | --- |
| Every prefill forces the next decode to recapture. | `tt/generator_vllm.py:381` sets `_decode_ready=False`. Lines 461–464 release both traces when `reset_batch`, remapping, `not _decode_ready`, or execution-shape change is true. `_release_decode_traces` at line 302 releases model and sampling trace IDs and clears the generator cache. Even unchanged page tables cannot avoid this release after prefill. |
| Changed prefill page tables cause an additional early release. | Adapter lines 341–346 release before refreshing tables. The comment attributes this to staging allocations. But `_refresh_page_tables` at lines 273–290 allocates **host** staging tensors: `ttnn.from_torch` has no `device=` argument, followed by a copy into an existing device target. The explanation does not describe the current implementation. |
| Page-table contents can change without trace identity changes. | Device tables are allocated once at adapter lines 219–238 and copied in place at line 288. Decode already uses this policy at lines 420–425. Generator cache keys use state/table identities, not table contents (`tt/generator.py:662` and 750). Its explicit-state path receives those same target objects and skips extra adoption/copy (`generator.py:193–215`). |
| A new request does not need an identical old active mask to trigger refresh. | Generator prefill sets `_request_boundary=True`, clears seeded slots and the sampling key, and initializes host positions (`generator.py:476–490`). Decode treats that flag as a scheduler boundary (`633–646`), copies the new host token to the persistent trace token tensor (`696–700`), refreshes current position and RoPE (`701–708`), and arms seeds (`709–712`). This happens even when batch one remains in row zero. |
| An explicit scheduler reset can refresh inputs while keeping the same graph. | Adapter line 483 passes `active_mask` whenever `reset_batch` or remapping is true. Any supplied mask makes the canonical generator take the scheduler-boundary path. Sibling `../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:1158–1161` derives reset from decode layout changes; lines 1390–1401 drain pending async steps before host inputs are rebuilt for such changes. Preserve that refresh/drain contract. |
| Greedy request seed changes do not force a canonical cache miss. | Adapter increments `_unseeded_epoch` on prefill. Generator `_sampling_spec` immediately returns `SamplingSpec(greedy=True)` when all temperatures are zero (`generator.py:327–329`), so the epoch/seed values are absent from the greedy key. Stochastic specs retain parameters and seeds in their key and may correctly recapture. |
| A cache miss performs substantial extra setup. | `generator.py:750–770` returns immediately for a hit, otherwise releases the old trace set. Lines 773–879 allocate stable inputs/parameters, warm model and sampler, synchronize, capture both graphs, and restore positions/seeds. This runs before the first actual model/sampler replay. It is absent on a cache hit. |
| Steady replay already advances device state. | Model trace increments current/RoPE positions (`generator.py:852–853`). Sampling writes directly into `tt_out_tok=token_input` (`861–864`). Both replay nonblocking (`713–715`). Removing request-level invalidation must retain the existing boundary-only host refresh; passing a mask on every token would overwrite intentionally stale host feedback. |

The host-staging interpretation is also established by
`tests/test_vllm_adapter_contract.py:259–285`, which asserts that `from_torch`
receives no device argument and the copy destination is the original tensor.
Runtime `ttnn/ttnn/operations/core.py:268,370–379` passes its default
`device=None` through to construction; the nanobind copy binding at
`ttnn/cpp/ttnn-nanobind/operations/core.cpp:329–335` calls `copy_to_device` with
the supplied destination.

## Allocation and sampling lifetime

Retaining traces is safe only if new allocations that may overlap captured
scratch are finished and no longer needed before replay. Stable request state,
KV, page tables, sampling inputs, and collective resources must retain their
correct allocations. Updating existing page-table contents is not new device
storage and needs no blanket allocation suppression.

Canonical `generator.prefill_forward` deliberately scopes model-prefill
intermediates as corruptible (`generator.py:499–506`). Its standalone
`generate` also scopes first-token parameters, padded logits, and sampled
output, reading the first token before replay (`943–965`). The adapter instead
calls `sample_device_logits`, which allocates four per-call parameter tensors
and possibly padding/output (`394–422`), then immediately reads the result and
returns a host token (`generator_vllm.py:384–397`). Its local TT tensors normally
fall out of scope before the later decode call. Consequently, the absence of
the standalone scope is a **lifetime-check gap, not proof that another scope is
required**. Standalone `generate` retains its locals in the generation frame;
the adapter has a different Python lifetime.

Sampling1D's lazy index/seed buffers are materialized at generator construction
(`generator.py:123–126`); sampler parameters are per-call arguments, not new
module fields (`models/common/modules/sampling/sampling_1d.py:55–57,184–223`).
The reduced experiment must still inspect actual survivors, including any
sampler/collective resources, rather than assuming every allocation is local.

Runtime `tt_metal/impl/allocator/trace_allocation_tracker.cpp:117–173` tracks
new allocations made while traces are active and retires deallocated buffers.
`ttnn/ttnn/unsafe_allocation_tracker.py:67–115` runs garbage collection and
rejects live tracked buffers before replay. A corruptible scope skips this
accounting; it does not reserve memory or make a persistent tensor safe. Use
`TT_METAL_TRACE_ALLOC_TRACKING=1` and, when needed,
`TT_METAL_TRACE_ALLOC_TRACEBACKS=1` before importing TTNN for a **functional**
probe. These are allocator diagnostics, not profiling; keep their timing out
of serving performance comparisons.

Do not add a scope preemptively. If diagnostics prove that only consumed
first-token sampling temporaries survive, prefer freeing them before replay.
If an explicit lifetime annotation is necessary, keep it in a narrowly
documented generator-owned prefill-sampling helper, with host token consumption
completed before any replay. Do not wrap the whole adapter, mark new persistent
state corruptible, or disable the tracker to obtain a pass.

## Existing coverage and limits

- `tests/test_full_model_contract.py:1602–1785` contains a reduced two-layer TP4
  probe for both attention kinds. It checks inactive capture leaves KV intact,
  stale host positions are ignored during steady decode, device feedback aliases
  the persistent token tensor, changed tables keep addresses/trace IDs, and
  active-row changes refresh positions. It also retains a **teacher** trace
  across reset/prefill with changed prompt lengths.
- `../optimized_full_model/final/mixed_state_tp4.json` records a prior pass:
  equal token/output addresses, stable table addresses, changed table contents,
  and a retained teacher trace restarting positions at `[36,46,-1]`. This is
  inherited standalone evidence, not a fresh optimized-vLLM result.
- Adapter tests at `test_vllm_adapter_contract.py:179–255` use a fake generator
  to check boundary masks, stale host arguments, and changed table refreshes.
  They do not exercise a retained trace across actual adapter prefill,
  first-token sampling, and the next request. Most neighboring contract tests
  inspect source. Those tests do not prove device buffer lifetime or output
  correctness for the proposed change.
- Host compatibility is a separate lifetime: canonical host decode is eager
  (`generator.py:614–618`) and the adapter gathers logits. Retain explicit
  release before host-mode work and verify returning to device mode refreshes
  scheduler token/position inputs. Do not extend the optimization to retained
  eager-host traces incidentally.

## Smallest verify/refute experiment

1. **Bound the candidate to adapter orchestration.** In device-sampling prefill,
   keep existing traces while copying changed page-table contents into their
   stable targets. In decode, retain releases for host compatibility,
   nonidentity remapping, and physical execution-batch changes. Treat a
   same-shape reset/new-prefill boundary as an input refresh rather than a
   graph invalidation: preserve the existing `active_mask` decision and the
   generator request-boundary flag. Remove `_decode_ready`'s unconditional
   invalidation role; do not remove required boundary refresh. Let the
   generator's semantic key continue handling sampling-mode/parameter changes.
   Do not change precision, decoder math, padded multi-request width, overlap
   capability, page-table copy policy, or sampler strategy in this experiment.
2. **First verify the mechanism on a reduced real adapter.** Build one layer
   of each attention kind with serving-owned cache and the real terminal and
   sampler. Warm request A; complete its async token read. Then run request B
   with different token IDs and nonaligned prompt length but the same physical
   decode batch. Cover unchanged and changed legal scheduler mappings. Record
   model/sampling trace IDs, capture/release counts, input/table addresses,
   token/position/RoPE refresh counters, and allocator-check outcome. Capture
   must remain unchanged for B, with one boundary refresh and no steady-step
   host feedback copies. Use wrapper counters or host timestamps, not a device
   profiler.
3. **Check actual outputs and adversarial state.** Compare B with an explicit
   release/recapture control initialized with identical prompt/cache contents.
   Compare greedy token sequences and selected logits; inspect persistent
   inputs before replay to prove B consumes B's first token and absolute
   position. Then supply intentionally stale host tokens/positions on steady
   steps and prove feedback still advances once. Exercise a padded multi-row
   same-shape reset, inactive-row change, changed/unchanged tables, B1-to-padded
   transition, nonidentity remap, and a host-compatibility request. Keep
   stochastic/seed behavior under the existing capability contract; parameter
   changes may still require recapture.
4. **Only after functional success, measure the original workload.** Reuse
   the exact `run_profiles.py` runner configuration and compare primary
   128/128/1 P150 before/after; record first-decode interval and capture count
   in a separate diagnostic run if the harness still saves aggregate ITL only.
   Prediction: a warmed same-shape request uses no capture and reduces a setup
   interval while median steady ITL stays similar. A stable trace ID alone is
   not proof of a speedup. Repeat the same profile comparisons on P150x2/x4,
   and retain separate 100/100/32 burst evidence and existing full serving gates.

The first candidate should add **no new allocation scope**. A tracker failure
is evidence for a focused lifetime investigation, not permission to suppress
all allocations. A key mismatch, required shape transition, stale-output
failure, or unchanged measured setup delay refutes the corresponding broader
claim and should narrow or reject the candidate before integration.

## Checks performed

Read the adapter, canonical generator, full-model prefill/decode, Sampling1D,
relevant runtime allocator/copy sources, sibling plugin reset ordering, tests,
and existing artifacts. A local Python AST check confirmed the adapter release
sites and host-only staging keywords. A local standard-library calculation
produced the explicitly hypothetical ITL outlier model above. No pytest,
device command, server request, benchmark, or profiler was run by this
investigator. This documentation-only output needs no C++ build.

## Follow-up: cold prefill program allocations during retained replay

This appendix diagnoses the coordinating agent's subsequent reduced TP4
experiment after its greedy sampling-key correction. It supersedes any reading
of the initial candidate as saying that unchanged **decode** shape alone is
sufficient for safe retention. The investigator again read source/logs only;
no TTNN import, hardware action, or implementation edit was performed.

### New observation and ownership

`trace_reuse_probe/tp4.log` fails before executing retained model trace
`MeshTraceId(8)`, in `UnsafeAllocationTracker.verify_before_replay`. Its JSON,
`trace_reuse_probe/P150x4/trace_reuse_tp4.json`, records successful B1 A/B/A
retention and explicit-release controls: unchanged trace IDs, exact token
agreement, and maximum compared logit difference zero. The failing request is
the transition from prefill lengths `[33,47]` to `[35,45,39]`. Both requests
decode with physical batch 32, but prefill's concatenated terminal output
changes from two to three logical rows. The second request finishes prefill
and its first-token host read before the first retained decode replay is
rejected.

The coordinating agent's `trace_reuse_probe/diagnostic_tp4.log` repeats the
same three live allocation IDs and gives these allocation call stacks:

| Buffer | Allocation context | Python origin |
| --- | --- | --- |
| 57180 | `program_cache: ConcatDeviceOperation`, dim 2, interleaved DRAM | `generator.prefill_forward:543`, `ttnn.concat(outputs, dim=2)` |
| 57195 | `program_cache: TilizeWithValPaddingDeviceOperation`, padded shape `[1,1,32,65536]`, `enough_space_height=false` | The same concat call, through concat's row-major fallback and retilization |
| 57209 | `program_cache: FillPadDeviceOperation`, zero fill | `sample_device_logits:412` → `_pad_sampling_logits:428` → `ttnn.pad` |

**These contexts identify cache-setup allocations, not the corresponding
ordinary returned tensors.** TTNN creates output tensors at
`ttnn/api/ttnn/device_operation.hpp:516`, before entering the operation launch
at line 547. Only cache creation/enqueue enters the `program_cache:` context
at lines 375–389. Cache storage owns the workload at lines 334–355.
`tt_metal/distributed/mesh_workload.cpp:131–206` lazily allocates and retains
`kernel_bin_buf_` on first enqueue, preserving that operation-specific context.
The field is a C++ `shared_ptr<MeshBuffer>` in `mesh_workload_impl.hpp:62`;
cached mesh workloads own it across Python returns. This is the source-backed
kernel-binary ownership explanation for the three survivors. Their exact
device address ranges were not included in the diagnostic output, so actual
overlap/corruption is not established by the log.

Several independent details corroborate that diagnosis:

- Concat output allocation is in `concat_device_operation.cpp:204–207`, outside
  the cache context. `concat/concat.cpp:93–138` implements the padded-row
  fallback by untilizing inputs, concatenating, and retilizing. It accounts for
  the two different operation contexts with the same Python concat origin.
- The logged tilize attributes select the interleaved block factory through
  `tilize_with_val_padding_device_operation.cpp:51–52`; its output tensor is
  created separately at lines 217–219. The selected factory builds kernels
  and circular buffers using the supplied input/output tensors.
- FillPad cannot account for an independent leaked output: its
  `create_output_tensors` returns the input tensor at
  `fill_pad_device_operation.cpp:39–42`. Tile pad first calls
  `fill_implicit_tile_padding` and may then return a view
  (`pad/pad.cpp:478–480`). Its distinct surviving allocation is consequently
  setup storage, not a newly returned FillPad result.
- The tracker already performs garbage collection before checking live IDs.
  Clearing Python prefill locals cannot free a buffer retained by a cached
  C++ workload. The diagnostic's Python referrer scan itself failed with
  Flask's “Working outside of application context”; that failure neither
  proves nor disproves a Python owner. Do not modify the generic tracker to
  resolve this model-stage issue.

The prefill concat is outside the generator's existing model-prefill
corruptible scope, and sampler padding is also outside that scope. Expanding
either scope would hide persistent program allocations alongside temporary
outputs. It would not establish their safety. The existing model-prefill
scope can already exclude new binaries created deeper inside the model from
tracker accounting, so a passing tracker check alone is insufficient evidence
that a cold prefill did not grow the program cache. Do not set
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` or mark these buffers corruptible.

### Minimal safe-boundary experiment

**Recommended candidate: invalidate retained traces when the program cache
grows during the complete prefill plus first-token sampling/read path.**
Snapshot the mesh's `num_program_cache_entries()` before prefill; compare it
after first-token sampling has completed and its host result is available.
If the count grew, release every retained model/sampling trace before returning
to decode. The next decode then captures with the new program resources
already allocated. Apply the comparison to all successful prefill return
paths, including per-row sampling and host compatibility; preserve existing
host-mode, remap, and physical decode-shape releases. Add no allocation scope
and no page/token-content heuristic.

This boundary is supported by the source lifetime:

1. Prefill and `sample_device_logits` enqueue eager work; neither calls
   `execute_trace`. The adapter completes `_read_tokens` before returning a
   CPU token. The plugin drains earlier async work at scheduler boundaries;
   the reduced test also consumes previous decode reads before beginning a
   request. Thus the old trace does not replay between the cache-growth
   snapshot and the release decision.
2. `_trace_cache` keeps the persistent token, current-position, RoPE,
   parameters, and terminal-output tensors allocated. The ordinary allocator
   cannot reuse their live storage for prefill outputs or new binaries. KV
   and page-table updates intentionally mutate their stable caller-owned
   tensors; prefill does not replace those allocations. Temporary allocations
   can reuse formerly freed trace scratch, which is exactly why a later old
   replay must be prevented.
3. Releasing the old traces before replay eliminates their recorded writes
   into that scratch. Newly created cached workloads and their binary buffers
   remain allocator-owned. The next capture allocates around those live
   buffers, warms the exact decode path, and refreshes the new request's token,
   position, RoPE, and seeds through the existing scheduler-boundary logic.
4. Counting cache entries captures newly cached model operations even when
   an existing corruptible scope suppressed their allocation diagnostics.
   It also catches the outer concat and first-token sampler padding; comparing
   immediately after model prefill would miss the observed FillPad allocation.

This is a conservative invalidation test, not a general allocator proof.
`MeshWorkloadImpl::load_binaries` explicitly allows deferred initialization of
an already-cached workload; cache-count equality alone cannot detect that rare
case. Normal successful eager calls enqueue and load their workloads, so the
observed cold three-row path is covered. Keep allocation tracking enabled in
functional checks and retain the unsupported-case failure rather than
suppressing it. Any identified same-count persistent allocation requires its
own lifetime fix or explicit invalidation boundary.

An exact-shape warmup registry is a larger, less reliable first intervention.
Logical length matters separately from padded length (`model.py:602–614` and
last-token slice at 631–632); ordered prefill row count/user IDs affect concat,
cache-row slicing, and sampler padding. Mesh, precision, cache/table geometry,
sampling mode, and effective input specifications also matter. Furthermore,
concat's program hash includes a factory choice based on live L1 occupancy
(`concat_device_operation.cpp:78–89`), and tilize computes free-space flags.
Identical host shape labels do not prove identical runtime program signatures.
Cache-growth invalidation avoids having the adapter duplicate that logic.

For verification, update the reduced test to distinguish **cold** and **warm**
prefill transitions: a cold B2→B3 path may release/recapture, while repeating a
fully warmed B3 path must preserve IDs and pass token/logit controls. Keep B1
identical-length requests with changed tokens and changed/unchanged tables as
the headline reuse test. Previously successful B1 33→47 retention is useful
output evidence but need not remain a zero-recapture requirement if unseen
logical-length programs were created under the existing model scope. Record
cache counts alongside trace events, then run the unchanged full-model
128/128/1 benchmark only after the functional candidate passes. No measured
performance result is established by this appendix.
