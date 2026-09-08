# Adapter page-table refresh audit

**Recommendation:** make selective layer uploads a separate candidate. The
current adapter avoids uploads when every table is unchanged, but uploads all
30 layers when any one changes. A host-only execution of the actual methods
verified 30 uploads after changing only one five-layer hybrid group; only
those five uploads are necessary. No TTNN import, hardware command, pytest,
implementation edit, or profiler was used for this investigation. The trace
allocation diagnosis in `AUTODEBUG.md` was left unchanged.

## Actual sharing topology

The checkpoint's text config has 25 sliding and five full-attention layers.
Upstream `../vllm/vllm/v1/core/kv_cache_utils.py:1018–1066` groups layers by
cache specification, selects minimum group size five, and splits each type
using strided layer lists. With this configuration it produces six groups:

| Group | Layer indices | Stable device table shape |
| --- | --- | --- |
| Sliding 0 | 0, 6, 12, 18, 24 | `[32,16]` |
| Sliding 1 | 1, 7, 13, 19, 25 | `[32,16]` |
| Sliding 2 | 2, 8, 14, 20, 26 | `[32,16]` |
| Sliding 3 | 3, 9, 15, 21, 27 | `[32,16]` |
| Sliding 4 | 4, 10, 16, 22, 28 | `[32,16]` |
| Full | 5, 11, 17, 23, 29 | `[32,396]` P150; `[32,2048]` P150x2/x4 |

This is a source-derived topology, not a captured live group dump. The actual
plugin mapping comes from `KVCacheGroupSpec.layer_names`, cached in
`model_runner.py:367–394`; an optimization must consume the supplied tables
rather than hard-code two types or six groups.

Plugin `input_batch.py:648–667` creates one host table per group for the
requested rows. `model_runner.py:1179–1186` pads decode group tables to the
execution lane width. `_block_tables_per_layer` at lines 468–508 expands them
by reference into the layer list. When a group table already has the target
`[32,W]` shape, its five layer entries reference the same tensor. If further
padding is needed, that method allocates a separate tensor per layer. This
often happens for prefill with fewer than 32 rows, so source aliases are not
guaranteed on every call. Effective host widths for the configured profiles
are 791 and 4096; the adapter's narrower device tables retain their existing
per-layer geometry and receive cropped/padded contents.

The adapter allocates **30 distinct device page-table tensors**. Sharing host
group data does not justify uploading once and ignoring four distinct device
destinations. Each changed group still needs five writes under the current
allocation contract. KV-buffer sharing is a separate mapping and must not be
used to infer page-table ownership.

## Verified copy counts

`page_table_host_copy_counts.json` records a CPU control that extracts the
actual `_page_tables_changed` and `_refresh_page_tables` methods from the
adapter AST and executes them with a mocked TTNN staging/copy interface.
`torch` is CPU-only; `ttnn` is absent from `sys.modules`. The fake `from_torch`
asserts no `device=` argument, and each copy records the actual padded staging
tensor and destination layer.

| Case | Existing uploads | Necessary uploads |
| --- | ---: | ---: |
| Initial table binding | 30 | 30 |
| All 30 tables unchanged | 0 | 0 |
| One sliding group changes at row 0, column 0 | 30 | 5 |

For the one-group change, the required layers are `[0,6,12,18,24]`. The other
25 staging tensors exactly match their previous device contents. Byte counts
per chip, excluding transport overhead, are:

| Profile | Current upload bytes | Required bytes | Avoidable bytes |
| --- | ---: | ---: | ---: |
| P150 | 304,640 | 10,240 | 294,400 |
| P150x2/x4 | 1,361,920 | 10,240 | 1,351,680 |

If all five sliding groups change but the full group does not, the necessary
payload is 51,200 bytes per chip instead of the current totals above. Different
block sizes (64 sliding, 128 full) allow such updates; their actual frequency
in the benchmark has not been measured. These are exact copy/byte counts from
the source and host control, not measured PCIe bandwidth or serving gains.

## CPU comparison cost

`page_table_host_audit.json` records seven samples of 250 iterations after
30 warmups, `torch==2.11.0+cpu`, one Torch thread, using `perf_counter_ns`.
The current unchanged-table check uses 30 comparisons against independent
snapshots. A host-only alternative uses six immutable shared snapshots and
per-call `(id(current), id(previous_snapshot))` memoization.

| Host table shape | Current 30-comparison median | Shared six-comparison median |
| --- | ---: | ---: |
| `[32,791]` | 231.137 µs | 48.622 µs |
| `[32,4096]` | 1,200.263 µs | 237.481 µs |

The alternative produced the same changed-layer bitmap after one shared
source mutated, and remained correct when sources were separate objects.
These timings exclude normalization, host cloning/staging, TTNN writes, and
server work. Server thread settings and live contention were not reproduced;
do not report their difference as measured serving speedup. The source hash
and full timing ranges are in the JSON.

## Smallest implementation and verification boundary

First return or compute a per-layer change bitmap and upload only changed
destinations. Preserve host-only staging, stable device identities, full
target-shape zero padding, cropping, and the existing single refresh-event
counter semantics. Initial binding must still initialize every destination.
Unchanged layers can retain their existing immutable snapshots. A changed
host object must be compared by **value**, since the scheduler may mutate it
in place or replace it with an equal-valued tensor. Do not interpret a
page-table-only update as a token/position reset.

If comparison deduplication is included, share one cloned snapshot among
layer entries that share the same current source object; keep it independent
of the mutable caller. Use pair-of-object-identity memoization only within
one comparison call. A current source shared by two layers can have different
previous snapshots after alias topology changes; memoizing only the current
source would miss one layer's update. Equal but distinct source tensors may
fall back to independent comparisons. Do not cache raw Python IDs across
calls, or assume aliases from layer type or KV tensor index. The six-comparison
case requires shared snapshots: the current per-layer cloning loses that
sharing even when the plugin supplies aliased sources.

The current adapter's host test
`test_page_table_refresh_uses_host_staging_and_existing_device_storage`
checks only one layer. Its fake-generator steady-decode test likewise has one
layer, and the real reduced trace-reuse probe changes both attention tables
together. Add a focused multi-layer regression covering initial/unchanged,
one changed layer, a shared changed group, separate equal-valued objects,
in-place mutation, alias split/merge, and source shape shrink with zeroed
trailing destinations. Assert exact destination IDs and upload counts, not
just the aggregate refresh counter. Retain device stale-feedback coverage and
add a real reduced replay where only one attention table changes and the
other remains untouched.

Comparing only cropped effective source contents could avoid further copies
when changes lie outside a device table's extent, but it requires deliberate
snapshot/shape semantics. Leave that as a separate follow-up rather than
combining it with the proven all-layers-on-any-change issue. Aliasing device
page tables across layers is also outside this candidate.

## Candidate prepared for parent-owned verification

The adapter now computes a layer change bitmap using per-call
`(id(current), id(previous_snapshot))` memoization. Refresh copies only changed
destinations, shares independent snapshots for shared current source objects,
and retains valid prior snapshots for unchanged sources. It keeps raw source
value comparisons and the existing device-table geometry/cropping/zero-padding.
No device page tables are shared by the change.

The new host regression uses six host groups and 30 distinct device targets.
Its required results are 30 initial writes, zero unchanged writes, six equality
checks on an unchanged shared batch, and five writes for one shared-group
mutation. It also checks alias splitting, merging one current object against
different old snapshots, equal distinct tensors, later in-place mutation, and
shorter source zero-padding. These are test assertions awaiting the parent's
test run, not newly measured results.

The real reduced probe now changes only the sliding layer's table. It records
and asserts exact page-upload layer indices at prefill and after steady decode,
checks both device tables against scheduler contents, and retains the existing
trace-identity, stale-feedback, cold-program, and explicit-release controls.
Its report records watcher/noinline/disabled-feature environment values. Black
formatting and diff checks passed during preparation; hardware and serving
measurements remain parent-owned and are required before performance claims.
