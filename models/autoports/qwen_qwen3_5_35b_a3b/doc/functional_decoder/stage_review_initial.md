# Stage Review

Verdict: more-work-needed

Latest follow-up: the original unseen-length failure is reproduced and repaired
for full attention, including an unseen continuation start. A second source
inspection found repeated whole-input/RoPE layout conversion; that is now
repaired in source and awaiting the owner's final validation runs. See the
follow-up section below. This remains an initial review, not a stage pass.

Independent initial inspection, 2026-10-10 UTC. Stage 01 functional decoder for
`Qwen/Qwen3.5-35B-A3B`, pinned checkpoint
`59d61f3ce65a6d9863b86d2e96597125219dc754`. Live untracked autoport on
`feat/qwen3-5-35b-a3b`, checkout base
`98dc4ae62ccfc32836ddcfad50f9fd97863612d4`.

This review intentionally precedes several queued/running hardware gates. It is
not a final review of artifacts that do not exist yet. No devices were opened,
TTNN imported, experiments launched, or implementation files changed by the
reviewer.

## Required Work

- **P1: Unseen logical lengths still introduce new program signatures after
  decode capture.**
  Evidence: `tt/functional_decoder.py:328-349` uses Python-valued input slice
  offsets, absolute position slices, chunk page-table slices, and variable
  output concatenation. `tests/run_reuse.py:100-110` warms every listed request
  except the padded logical-length-36 case, then forbids cache misses before
  capture. The logical-35 and logical-36 inputs both have physical length 128,
  but the latter executes the additional input slice `[35:36]`.
  `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:235-237,343-347`
  routes its unaligned begin through row-major layout; lines 398-405 pass
  static start/end attributes. `device/slice_device_operation.cpp:355-377`
  hashes those attributes and the source logical/padded shapes. A pure-Python
  enumeration of the current warmup planner confirms that
  `(source_length=128, begin=35, end=36)` is absent for both layer kinds.
  Why this matters: an already prepared physical bucket does not currently
  imply safe execution of a new logical length. Misses-forbidden should expose
  the new signature; allowing the miss can introduce persistent program-cache
  allocations unsafe for the live decode trace. Variable concat and absolute
  continuation slices are additional signatures to account for.
  Required next step: run the deliberately withheld length-36 case with
  allocation tracking and program-cache checking retained, then implement a
  bounded preparation/execution contract that covers new lengths and offsets
  without request-time trace release or recapture. Validate an unseen logical
  length and an unseen continuation start within prepared shapes. Merely adding
  each new request to a request-specific warmup list does not establish the
  advertised reuse contract.

- **P2: Per-row current positions are not yet exercised.**
  Evidence: `tests/run_layer.py:95-96,121-122` constructs one shared position
  for every batch row. `tests/run_reuse.py:85-86` does the same. Page IDs are
  permuted and nonzero request slots are selected, but current positions do not
  differ among simultaneous rows. The public decode path slices positions
  independently at `tt/functional_decoder.py:373-375`.
  Why this matters: equal-position batches cannot expose position/row broadcast
  errors or interactions between unequal live prefix lengths and changed page
  ownership.
  Required next step: add a focused real-weight paged decode case with distinct
  valid prefix lengths/positions per row, changed row ordering/page mapping,
  and per-row HF comparisons from the same retained trace. The stage owner has
  acknowledged this gap and is adding coverage.

- **P2: Completion evidence remains pending.**
  Evidence: `doc/context_contract.json` still has null supported/tested-context
  fields and status `in_progress_unvalidated`; README/work log explicitly call
  the stage incomplete. Current visible artifacts do not finish the native full
  attention, exact-full-length prefill, updated reuse, current batch-32,
  synthetic regression, watcher-10, warmed profiler, human perf table, and
  whole-layer roofline gates.
  Why this matters: the currently passing subset cannot support stage closure.
  Required next step: finish the already authorized gates, classify warnings,
  reconcile the final README/work log/context contract to exact artifacts and
  final source, retain the local checkpoint commit, and request final review.
  This is a known in-progress set, not a new demand to run unrelated stages.

## Other Concerns

- `tests/summarize_perf.py` initially appended a literal backslash plus `n` to
  JSON. This was reported immediately and corrected by the owner during review;
  a subsequent AST check found a real newline. No remaining action for that
  defect. Review the resulting profiler artifacts when they exist.
- Its prefill span includes gaps between the first and last device firmware
  timestamps of eager layer dispatch. Preserve that definition in the final
  report; do not present it as the sum of device kernel busy times. Decode
  per-step signposts exclude host input copies and readback.
- `linear_native_fp32norm.json` records only the decoder adapter hash and an
  obsolete inherited `gdn_diag_alpha` field. The current GDN source does not
  consume that environment variable. This is explained by the hypothesis
  record; the pending current profiler/correctness run can provide the final
  source identity without relabeling the older artifact.

## Hard-Check Gaps

- The report should record tracking enabled and program-cache exclusions disabled
  for the final reuse/representative replay commands; performance commands must
  separately record diagnostics/tracking disabled. Current work-log commands
  assert tracking, but the old result JSONs do not capture that environment.
- Final `context_contract.json` must distinguish native capacity 262144 from
  the already completed prefill length 262143 and decode position 262143.
- Current batch-32 evidence must use the final FP32 norm/GDN implementation and
  check each row/token; aggregate PCC alone cannot close that gate.
- Same-instance reuse must retain the original trace and include correct
  continuation output plus decode after the cache/state update. Existing
  harness structure checks this, but its updated run is still pending.

## Anomaly Ledger

- Observed anomaly: historical GDN aggregate PCC concealed failing individual
  decode tokens.
  Evidence: `linear_4096_128_per_token.json`,
  `linear_native_4096_128.json`, `linear_native_fp32norm.json`.
  Affected path: recurrent decode following long prefill.
  Control/comparison: exact native recurrence, BF16-expert diagnostic, and
  FP32 zero-centered norm repair.
  Investigation: inspected current GDN/norm source and independently counted
  final recorded per-token PCCs. All 128 recorded tokens pass; minimum
  0.995265305. Full-attention final 4096/128 also has 128 passing tokens;
  minimum 0.995969355.
  Resolution: fixed for the recorded 4096/128 workload; remaining workload
  gates still apply.

- Observed anomaly: a new valid length in a warmed physical bucket needs a new
  slice signature.
  Evidence: static slice/hash inspection and warmup-planner enumeration above.
  Affected path: eager prefill between decode trace replays, both layer kinds.
  Control/comparison: padded logical35 versus withheld padded logical36.
  Likely subsystem: preparation/program-cache contract.
  Investigation: source proof supplied to owner; hardware reproducer pending.
  Resolution: more-work-needed.

## Scope Inspected

- Goal/skills: parent review contract, `../AGENTS.md`; installed
  `stage-review`, `functional-decoder`, `tt-enable-tracing`, and model-bringup
  startup instructions. Enabled AutoDebug inventory checked.
- Artifacts: README, work log, contract, context contract, AutoDebug/AutoFix and
  component/GDN diagnosis records, current 4096/128 and native-GDN JSON/logs,
  historical PCC results, sparse-HF control and real-weight statistics.
- Code: complete `tt/functional_decoder.py`, `tt/functional_gdn.py`;
  `tests/run_layer.py`, `run_reuse.py`, `hf_reference.py`, `runtime_audit.py`,
  weight loader/statistics generator, CPU/device regression definitions and
  `summarize_perf.py`; reached Qwen36 decoder/attention/MoE helpers; installed
  HF GDN recurrence; TTNN slice program hashing and trace-correctness guide.
- Commands: read-only `rg`, `cat`, `sed`, numbered source reads, `git status`,
  and small Python standard-library JSON/hash/AST/planner analyses. The sole
  write by this reviewer is this report.

## Residual Risk

The current layer has real target dimensions and all routed/shared experts;
the public runtime path is device-only by inspection and the recorded audit.
Output-only corruptible acknowledgment is narrow: the output is consumed before
another replay, while model/cache/state/input allocations precede capture.
No whole-forward exemptions were found.

The completed native GDN artifact tests every TT token at length262143 and the
last native decode position, with selected-output HF probes retaining all
recurrent contributions. Its source hashes match the inspected implementation.
The reduced HF reference has a 1153-token dense control with PCC0.999992788 for
GDN and 1.0 for full attention. Those facts do not establish pending full
attention capacity or cross-request trace safety.

Implementation snapshot hashes at initial inspection:
`functional_decoder.py=fedad930000e59ff07cb577d19c9965fe68cd45572cb745b699debbeccabafd8`,
`functional_gdn.py=57855156c611bcdee557a2a98ce82048dc4629ad8c78979b2b5ccccf17d47bdd`.
This live worktree may change after the report; affected checks must follow the
final implementation. No full-model, generation, serving, or multichip pass is
claimed or required by this stage review.

## Follow-up: bucket preparation repair

Verdict remains **more-work-needed** because the current implementation and
remaining stage gates still require final validation. No new unresolved source
correctness defect was identified in the repaired bucket mechanism.

### Original failure reproduced, then repaired

`full_reuse_unseen.log:41` records `SliceDeviceOperation: program cache miss
occurred, but cache misses are forbidden` at the withheld logical36 case. This
confirms the original P1 finding rather than leaving it as a theoretical risk.

The repaired source uses resident FP32 index vectors plus runtime scalar add,
embedding reads from a fixed physical input, page-table gather, tensor-valued
chunk positions, and fixed output storage filled by row-major `slice_write`.
Full-attention chunks are now512/128 and begin at positions divisible by both
page size and128; individual real tokens advance an unaligned prefix/tail.

Reviewed the implementation and the underlying operation contracts:

- `binary_ng_device_operation.hpp:76-128` excludes scalar values from the
  operation attributes. `binary_ng_utils.cpp:735-741` packs the FLOAT32 scalar
  as FP32, avoiding BF16 index rounding. For batch32 and physical lengths up to
  the native262144, flattened row offsets remain exactly representable.
- `experimental/slice_write/device/slice_write_device_operation.cpp:86-96`
  excludes outer offsets for row-major interleaved input; its factory's cache
  hit callback rebuilds runtime offsets and buffer addresses. The decoder uses
  that path and keeps the innermost offset zero.
- The final reshape changes logical volume while retaining the physical padded
  shape. `data_movement/reshape_view/reshape.cpp:724-733` permits this as a
  metadata view; no logical-tail data-movement signature is introduced.
- Page gathers contain exactly the real chunk's pages, and tensor chunk starts
  select the absolute causal position. Alignment by128 satisfies the flexible
  SDPA's default q/k chunk sizes. Only real tokens update recurrent/KV state.
- Remaining static row slices depend on prepared batch and metadata shapes,
  not the changing token values or request lengths within an input bucket.
  Physical buffer shapes and batch variants still require preparation before
  capture, as documented by the existing caller contract.

`full_reuse_bucket.json` matches adapter hash
`a854888fae05035856e5e33331d0a4e6b6469d9c6d6e4fb5264592b7b96a8f38`.
The withheld logical36 has per-row minimum PCC0.998941720 for prefill and
0.998453021 for decode. `full_reuse_final.json` extends the same source with a
withheld continuation cut36 after warming cut35 using the same physical128/256
input buckets. Its continuation has minimum PCC0.998735726/0.997589588. Both
artifacts retain `MeshTraceId(0)`, one capture, zero request releases, one
teardown release, and clean guard pages. The runner keeps program-cache misses
forbidden throughout the live-trace workload. The owner reports these commands
ran with allocation tracking and program-cache allocations included.

The original P1 is therefore **fixed for the recorded full-attention bucket and
continuation workloads**. The GDN counterpart and current-source repeats remain
part of final validation; do not reuse the old long-context evidence as proof
of the changed input/output/SDPA path.

### Avoidable data movement found and corrected in source

During review, the repaired `flat_input` remained tiled while each chunk used
it as embedding weights. `operations/embedding/embedding.cpp:28-32` converts
the entire tiled weight tensor to row-major on every call. At native GDN
length262144, this meant2048 conversions of the same1GiB input. The preexisting
RoPE tables were also tiled, so each lookup reconverted both full32MiB tables.

The owner immediately moved the hidden-state conversion before flattening,
outside the chunk loop, and constructs RoPE tables row-major at setup. This
removes the repeated conversion mechanism without changing the selected token
indices or mathematical operations. The reviewed follow-up adapter hash is
`f23a72f61721135886733fdfc955288eb191b5dc0004279119944cf54df1d8be`.
Resolution: **fixed in source; current numerical/capacity/performance reruns
pending**. No speedup is claimed from source inspection alone.

### Position coverage and remaining evidence

Inspected the new `tests/run_positions.py`. It constructs separate HF caches
for each distinct prompt length, independent page ownership, reversed request
slots, per-row device positions, and eight decode inputs per row. It captures
once, checks eager/replay identity and repeated replay, compares every row and
token, and verifies sentinel guard pages after teardown. This directly addresses
the equal-position blind spot in the initial review.

Independently read `positions_full_b32.json`: status pass, minimum prefill
PCC0.998529851 and minimum decode PCC0.996284366 over32 rows and8 tokens per row;
repeated and eager/replay identity checks pass, guard pages are intact, one
capture and10 replays are recorded, and the source was unchanged during the run.
Its implementation hash is the pre-layout-change `a854888f...` above.

`positions_linear_b32.json` records `failed_gate` under the same adapter hash.
Prefill minimum is0.998860121, but decode row10 (initial position255) reaches
PCC0.993859231 and row18 (initial position465) reaches0.989696622. Determinism
and eager/replay identity pass. These numerical outliers remain required work
and are already under investigation by the owner. The passing full-attention
row is not evidence for GDN correctness. Final review must inspect completed
artifacts against the final implementation rather than treating this follow-up
as clearance for stage closure.

## Read-only follow-up: SFPU recurrence and arbitrary page capacities

Verdict: **more-work-needed** for the two concrete page-capacity defects below.
This is an interim source review, not the final stage gate. No hardware was
used and no implementation or test files were edited by this reviewer.

Inspected implementation hashes:

- `tt/functional_decoder.py`:
  `9f3e67c25ed34b9b8f2720eddf5e39bb5dfee77709e47757454d61b0b3438837`.
- `tt/functional_gdn.py`:
  `b370d6e353c1e2de50e721b6e515151463b3927e36df10ded9c57b54a06ff337`.

### P1: attention-only table padding rejects valid small physical pools

`functional_decoder.py:337-340` pads every full-attention prefill page table to
a multiple of eight entries. The token-tail branch at line363 then passes that
padded table to the inherited attention decode path, including
`paged_update_cache`. However,
`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_device_operation.cpp:189-194`
requires the page table's padded width to be no greater than the physical
cache's page count. This validation is independent of which positions are
actually updated.

A valid B1 example is cache shape `[3,2,32,256]`, table shape `[1,3]` containing
the three owned pages, and prefill length65. The public capacity check succeeds
because96 >=65. Prefill then pads the table to width8; its first token update
fails because8 >3. A nine-page pool with length257 similarly becomes width16
and fails. These examples require no invalid page indices or context overrun.

The original table must remain available for cache writes. The 32-byte stick
padding is a read-side requirement of chunk SDPA, not extra physical ownership.
`run_reuse.py` allocates64 pages per row at its default page size; and
`run_positions.py` forces at least32 pages per row. Neither covers these valid
small pools. Resolution at review time: **confirmed by source; fix assigned**.

### P1: fixed64 token SDPA reads beyond an odd final table entry

Public decode at `functional_decoder.py:398` uses the original page-table row.
The inherited token-attention implementation selects fixed `k_chunk_size=64`
in
`models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py:141-152`.
The following source chain proves that the last odd32-token page is unsafe:

- `sdpa_decode/device/kernels/rt_args_common.hpp:62` rounds `cur_pos+1` up to
  the configured64-token chunk size.
- `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:752-770` and
  `:833-846` read every32-token tile in that chunk, including the tile beyond
  the final real page. There is no logical-capacity check in these loops.
- `sdpa/device/kernels/dataflow/dataflow_common.hpp:64-66` indexes
  `page_table_ptr[virtual_block]` without a bounds check. The bounded-pool
  modulo feature is not configured by this decoder.
- The row-major slice writer
  `data_movement/slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp:85-86`
  copies only the logical stick bytes. It does not initialize the remaining
  aligned bytes in the row allocation. Those bytes are nevertheless loaded by
  the SDPA page-table reader through the aligned stick size.

A current-source example independent of the first defect is B2, page size32,
table `[2,9]` assigning18 unique physical pages, cache with18 physical pages,
prefill257 followed by decode at position257. Prefill padding to16 entries
passes16 <=18. Decode slices each original nine-entry row, initializes only36
bytes, and the last64-token chunk reads virtual pages8 and9. Entry9 is outside
the nine real entries and is taken from unwritten allocation padding. Depending
on its value, this can issue an invalid KV read; causal masking does not prevent
the read itself. A passing PCC with accidentally zero padding is insufficient
to establish correctness.

The local remedy must ensure safe reads as well as preserve original ownership
for updates: for example, use a32-token SDPA chunk when logical table capacity
is not divisible64, or explicitly supply a separate initialized read table.
Resolution at review time: **confirmed by source; local attention fallback and
B1/B2 exact3/9-page pool watcher regressions assigned**. Those tests should cover
positions64 through95 and256 through287 so they exercise the final odd page.

### Repaired recurrence and trace-path findings

The SFPU single-token recurrence preserves the explicit HF order: BF16 Q/K
normalization, FP32 query scaling and decay, state decay, key-state reduction,
beta-weighted delta, state update, and query-state reduction. Adjacent value
heads share each normalized key head. Convolution retains only the last three
real projected tokens. The outer planner advances unaligned prefixes one real
token at a time, passes aligned real-token chunks to the native chunk op, and
writes only logical outputs. I found no additional concrete recurrence or
prefill-tail defect in this inspection.

Independently read `positions_linear_sfpu.json`: status pass,32 rows, minimum
prefill PCC0.999298871 and minimum per-token decode PCC0.998062253, source
unchanged during the run. It uses the current GDN hash and adapter hash
`a2a96c28a659e9d96499062958ea792dfeec770af058942d98a6363fbc379491`, before the
full-attention-only padding change. This resolves the previously recorded GDN
heterogeneous-position numerical failures for that workload.

`linear_accept_4096_128.json` records minimum prefill/decode PCC
0.999701798/0.997864842 with the same GDN and pre-padding adapter hashes.
`full_accept_4096_128.json` records0.998596847/0.995969236 with both currently
inspected hashes. Both record128 traced decode tokens and runtime audit.
These results do not exercise the small and odd-capacity cases above.

The inspected `run_reuse.py`, `run_positions.py`, and `runtime_audit.py` still
prepare host inputs outside `device_only`, forbid program-cache misses before
capture, preserve persistent state/input/output allocations, and acknowledge
only the final trace output. Fixed chunk shapes and resident index vectors
avoid a new scalar-offset-dependent program signature. The supplied runtime
guard blocks torch tensor operations and `from_torch`, `to_torch`, and
`as_tensor` inside public passes. I found no further concrete trace or runtime
host-compute violation in these paths. Final review must evaluate the two
capacity fixes and their completed evidence; this follow-up supplies no final
stage clearance.
