# Scoped prefill-gate diagnostic review

**No blocking findings remain in the frozen diagnostic.** This is a source-only
review of a prepared component experiment, not a hardware result, quality pass,
production-policy approval or Stage11 review.

Reviewed directory:
`/home/hous/dev/tti-release-gemma4/meta_gpqa_prefill_gate_control_v1`.

| Artifact | SHA256 |
| --- | --- |
| `probe_tt.py` | `5922679ca3355980ab6c412e505ce11066e3cde7c0d63438aab86fa95f3e81d2` |
| `plan.template.json` | `e6c9cf5a9e5c5b9fde14eda8c240d634bba0d3ddaa0bb720fb8c4dc75e5d6490` |
| `materialize_plan.py` | `961c0a91da4d33bc984d025ec3a6a0aea0850eb5102b6e56254dd27342e26cec` |

## Findings resolved before freeze

1. Original prefill returns the layer to `idle`. The draft component replay
   incorrectly expected it to remain in `prefill`. Each replay now temporarily
   sets the prefill phase and restores the previous phase in `finally`.
2. The draft reduced only the final 32 rows before restoring the 352-row shape
   for normalization. `_all_reduce_hidden` can choose a different persistent
   collective for 32 rows. The frozen version captures the complete native
   local 352 rows, substitutes the replayed final group, and preserves the
   original 352-row all-reduce and `_rms_norm` path.
3. The reference inverse reshape/transpose now has an exact comparison at the
   activation consumer against the entire substituted canonical tensor. Its
   verification covers the selected row and all untouched rows/experts, beyond
   a shape-only check.

## Reviewed causal controls

The probe calls model prefill exactly once using the original 326 IDs, native
configuration, fresh HMA pages, context262144 and 32 slots. It generates no
continuation. It requires the original terminal-logit SHA256
`2f404628b93a6c59f01d8ae2d8f3f26dcd396f965e106ae9164265604a132c93`
and exactly eleven layer0 gate calls before admitting component work.

The final physical group is positions320:352. Input, routing and native gate
captures retain all 32 rows, including 26 padded rows; the measured logical row
is325, group offset5. Runtime assertions bind DRAM operands/output, resident
BFP8_B `[1,128,2816,704]` weights, grid11×2, block44, per-core M/N1,
subblock1×1, `fuse_batch=False`, no indices and `nnz=None`. Replays verify the
complete original routing union and input exactly. Native gate, local group,
full local tensor, full reduced tensor and normalized group must match the
original capture exactly.

The sole candidate uses the previously validated HiFi4+FP32 destination gate
configuration. Other compute fields, resident weights, routing, geometry and
downstream operators remain unchanged. Candidate computation covers the native
32-row group; reported expert/downstream comparisons select row325.

The independent reference performs only TT FP32 elementwise multiplication and
pairwise addition. It retains the exact small-term invariant, checks decoded
resident BFP8 values and exact operand upload, and processes the selected eight
experts serially on both chips. Host operations copy/index data or calculate
statistics on TT outputs. There is no host model arithmetic or MVMUL oracle.
BF16 narrowing happens on TT. Only the eight selected experts at row325 are
substituted into the saved native canonical gate tensor; explicit checks preserve
all other rows and experts. The inverse reshape/transpose restores the native
sparse-output representation, and its consumer equality guard verifies actual
movement before using the reference downstream output.

The original admission thresholds remain: all 16 expert/chip comparisons
improve, mean gate relative L2 halves, and normalized error improves by20% on
both chips. These are component criteria only. A pass would support root review
of the omitted prefill phase together with the already tested decode path;
another decode-only generation would not isolate a new cause.

## Capacity, artifacts and restoration

The reference allocates one expert at a time. Each padded FP32 operand/product
is11MiB per chip; explicit simultaneous operands, product, half-tree copies,
addition output and decoded expert fit within the declared conservative80MiB
incremental budget. The probe checks live free and contiguous allocator capacity
before reference allocation, records product/add/release snapshots, and asserts
the largest observed incremental allocation stays within that budget. These
snapshots explicitly do not claim an allocator high-watermark. Actual capacity
and numerical behavior remain unverified until root executes the probe.

Complete native capture readbacks are saved and hashed before component arms;
target-row comparison readbacks are saved separately. Hooks and gate settings
restore through `finally`; each replay restores the prior phase. The outer
cleanup releases traces, closes the parent mesh and records clean closure only
after that call returns. Source/runtime checks precede device opening, and the
template is still `prepared_not_runtime_admitted` with no quiescence assertion.
Root retains materialization, ownership, launch and recovery responsibility.

## Verification performed

I read the no-CPU continuation first, inspected the complete probe and relevant
model/sparse/collective/allocator source, checked the three frozen file hashes,
and parsed the probe with Python's standard-library AST without importing or
executing it. AST inspection finds one `prefill_forward` call and no
`decode_forward` or `generate` call. The preparer separately reported 504 static
pins passing; I did not rerun that validator. No TT/model import, probe launch,
hardware action or production edit was performed by this reviewer.
