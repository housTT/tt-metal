# AUTODEBUG — GPT-OSS-120B trace-allocation warning

Date: 2026-08-31
Scope: `models/autoports/openai_gpt_oss_120b` plus the Metal trace-allocation machinery
Mode: investigation only. This investigation did not edit implementation/test files,
run TT hardware, or run a profiler.

The repo-local AutoDebug runner was invoked with a fresh Codex 5.5/xhigh
context. Its worker could not initialize its bubblewrap network namespace
(`RTM_NEWADDR: Operation not permitted`), so its provisional report could not
inspect this checkout or replace this file. The findings below were checked
directly against the checkout and the supplied full-36-layer tracking evidence.

## Headline finding

There are two distinct facts which the one-shot normal-mode warning conflates:

1. The visible warning in all four optimized-vLLM artifact logs is emitted
   during the first B1 split decode-trace capture, before B32 is captured. It is
   very likely the reserved `BufferType::TRACE` storage for the sampling trace,
   allocated while the already-finished B1 model trace is live. The tracker
   explicitly excludes reserved TRACE buffers, but the generic warning path
   does not. This is a Metal warning-path inconsistency, not evidence that B32
   was already live.
2. The decisive full-36-layer tracker run found a separate, real safety failure
   on the first production B1 replay: two persistent program-cache buffers had
   been allocated during the preceding real prefill while the prepared B1 and
   B32 traces were live. One came from an accidental device `slice` used only
   to discover KV block size; the other came from a previously unwarmed
   `paged_fill_cache` program shape for explicit per-layer vLLM page-table rows.

The Python `corruptible_allocation_scope` around coexisting trace capture is not
the cause of either real-prefill survivor. Real prefill is intentionally inside
`transient_allocation_scope`, which suppresses the noisy generic warning but
continues tracker accounting. The tracker therefore caught exactly the
persistent objects that must not be marked corruptible.

## Direct evidence

### Normal-mode artifact logs

The warning occurs exactly once in each of:

- `doc/optimized_vllm/artifacts/before_reproduced/server.log:1179`
- `doc/optimized_vllm/artifacts/after_minimal_token_read/server.log:1180`
- `doc/optimized_vllm/artifacts/after_full_validation/server.log.gz:1179`
- `doc/optimized_vllm/artifacts/after_final_clean/server.log.gz:1179`

In each log it is immediately before the first
`Done Capturing Decode Trace`. The following warmup reports token shape `[1,1]`.
The second capture completes about four seconds later and is followed by token
shape `[32,1]`. The adapter's source agrees: `widths` is `[1, max_batch_size]`
and it records the prepared buckets in that order. Therefore the warning is in
B1 capture while the B1 model trace is live and its sampling trace is being
created, not B1 capture after B32 is live.

`AllocatorImpl::verify_safe_allocation` emits at most one warning per host
thread for the entire process (`thread_local static bool warning_generated`). A
count of one consequently gives neither an allocation count nor a reliable
location for later unsafe allocations.

### Why the capture-time warning is probably reserved trace storage

`MeshDeviceImpl::end_mesh_trace` populates the new trace buffer and only then
registers that new trace on scope exit. During the B1 sampling-trace end, the
B1 model trace is already registered. With the configured nonzero
`trace_region_size`, `MeshTrace::populate_mesh_buffer` allocates the new storage
as `BufferType::TRACE`.

`AllocatorImpl::record_allocation_if_unsafe` explicitly ignores
`BufferType::TRACE`, and `mesh_trace.cpp` documents the reserved trace region as
excluded from unsafe tracking. In contrast,
`AllocatorImpl::verify_safe_allocation()` has no buffer/type argument and warns
before it can make the same exclusion. This matches the exact log timing. The
existing corruptible scope should also suppress the warning, so a thread-local
scope propagation issue remains a secondary possibility; the type-level TRACE
exclusion is nevertheless the correct invariant. Dynamic trace storage uses
`BufferType::DRAM` when `trace_region_size == 0` and must remain tracked.

### Full-36-layer tracking result

The full tracker plus Python traceback run was decisive and failed the first
primary B1 replay with:

```text
Found 2 device buffer(s) still alive before trace replay
```

Both buffers had a `program_cache:` allocation context and no Python tensor
referrer, which is expected for buffers owned by cached C++ programs:

1. `program_cache: SliceDeviceOperation`. Its traceback reaches generic
   `_prefill_forward_text_impl`, where each `layer_cache` (`[K,V]`) was passed
   to `get_block_size`. The old helper expression
   `kv_cache[0][0].shape[2]` indexes K once more and materializes a device slice.
   The block size is metadata and should be read directly from
   `layer_cache[0].shape[2]` (or the helper must distinguish a layer pair from a
   full `[[K,V], ...]` cache using only Python container types).
2. `program_cache: PagedFillCacheDeviceOperation`. Its traceback reaches
   `models/demos/gpt_oss/tt/attention/prefill.py` during real vLLM prefill.
   Warmup uses routed persistent max-width/uniform page tables. Real sequential
   vLLM prefill supplies explicit per-layer scheduler rows, slices them to
   `[1, ceil(valid_seq_len/block_size)]`, and materializes transient TT tables.
   That produces a different program hash/shape and the first cache miss occurs
   only after decode traces are live.

The current prefill-variant identity contains prompt padding, page-rounded
length, last-token tile, and host/device-sampling path. It does not encode the
explicit per-layer page-table representation or program-key geometry. It can
therefore classify the real 128-token prefill as warmed even though its
`paged_fill_cache` program is not present, bypassing the existing safe
release/compile/recapture lifecycle.

Program-cache eviction is a weaker explanation: `ProgramCache` is an
`unordered_map` with insert/get/clear and no capacity or eviction policy. A
missing production variant is more likely than spontaneous eviction. The
2-layer result predates and does not reproduce the full 36-layer program set.

### Why the reduced tracker pass was not dispositive

`trace_allocation_autofix/summary.json` used
`GPT_OSS_120B_VLLM_NUM_LAYERS=2`. Its zero survivor result only covers that
reduced program-cache population and workload.

Also, zero *normal warnings* under tracking is expected by construction:
`register_active_trace` creates per-trace tracking maps instead of setting
`allocations_unsafe_`, while the generic warning consults
`allocations_unsafe_`. Only `verify_before_replay` survivor results are decisive
in tracking mode.

## Ranked root causes and fix paths

### 1. Confirmed source bug: KV block-size metadata performs a device slice

Fix the generic call site to use `int(layer_cache[0].shape[2])`, or make
`get_block_size` accept both `[K,V]` and `[[K,V], ...]` by inspecting only
`list`/`tuple` nesting before reading `shape[2]`. Audit the analogous autoport
call site that iterates layer cache pairs. Do not probe TT tensors with
`__getitem__` merely to determine their shape.

Regression test: use a fake/spy TT tensor whose `__getitem__` raises, verify
both accepted cache container shapes return the block size, and verify the
prefill page-table slicing path performs no tensor slice.

### 2. Very high confidence: warmup does not compile the production hybrid page-table program

The smallest targeted repair for this workload is to run one exact explicit
per-layer 128-token sequential prefill before either decode trace is captured,
using a `[1, 2]` page-table row for block size 64. That should populate the
observed `PagedFillCacheDeviceOperation` program-cache entry.

The more general repair is to include explicit per-layer page-table/KV geometry
in `_prefill_program_signature` before any device op. When a real signature is
unseen, use the already-implemented lifecycle: release all decode traces,
compile the prefill variant, then rebuild B1 and B32 traces. This protects prompt
lengths and scheduler geometries beyond the fixed 128-token benchmark.

Regression tests should distinguish warmup's persistent max-width table from a
real explicit `[1, ceil(seq_len/block_size)]` per-layer list and assert that the
latter either is precompiled before trace capture or triggers release before
compile and recapture afterward.

### 3. High confidence for the visible warning: generic warning must skip reserved TRACE buffers

Change `verify_safe_allocation` to receive the buffer or `BufferType` and return
early for `BufferType::TRACE`, matching `record_allocation_if_unsafe` and the
documented reserved-trace-region contract. Do not skip `BufferType::DRAM`, which
is used for dynamic trace storage when no reserved trace region exists.

If the normal warning remains after that change, instrument the warning with
buffer type and `current_allocation_context` and audit propagation of
`corruptible_allocation_scope` across the allocation thread. Do not broaden the
Python corruptible scope as a first response.

### 4. Fallback only: audit any remaining tracked program-cache misses

If the two fixes remove the known survivors but a full run reports new ones,
classify each traceback by production input geometry and extend warmup/signature
coverage. Do not use `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1`, a broad
`corruptible_allocation_scope`, or `mark_corruptible` as a pass: program-cache
buffers are persistent and must not be overwritten by replay.

## Decisive reproduction and verification

Yes: a full 36-layer tracker run is the correct decisive test. The environment
variables are read during TTNN/Metal initialization, so set them before Python
imports. Use only the official workspace roots established by the checked-in
environment script and explicitly remove the reduced-layer override:

```bash
cd /home/ttuser/dev/gpt-oss-20b/tt-metal
source .agents/scripts/gpt_oss_workspace_env.sh
unset GPT_OSS_120B_VLLM_NUM_LAYERS
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_TRACEBACKS=1
unset TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE

VLLM_SYSTEM_START_DATE=2026-08-31 \
python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b \
  --mesh-device P150x4 \
  --max-num-seqs 32 \
  --max-model-len 131072 \
  --block-size 64 \
  --server-timeout 2400 \
  --tt-config '{"trace_region_size":750000000,"fabric_config":"FABRIC_1D_RING"}' \
  --additional-server-args "--async-scheduling --disable-log-stats --structured-outputs-config '{\"reasoning_parser\":\"openai_gptoss\",\"enable_in_reasoning\":false}'" \
  --benchmark-prompt-len 128 \
  --benchmark-output-len 128 \
  --benchmark-num-requests 1 \
  --benchmark-concurrency 1 \
  --benchmark-temperature 0 \
  --ci-benchmark-prompt-len 100 \
  --ci-benchmark-output-len 100 \
  --ci-benchmark-num-requests 32
```

The primary one-request workload is already sufficient: a 128-token prompt,
128-token greedy output, concurrency one. The known failure occurs on its first
B1 decode replay after prefill, before the CI burst. Keep the CI burst in the
final gate to cover later B32/B1 reuse.

Pass criteria:

- all 36 layers load; no `GPT_OSS_120B_VLLM_NUM_LAYERS` override;
- tracking and tracebacks are enabled, with no program-cache skip;
- primary request and 32-request burst complete over HTTP;
- no `Found ... device buffer(s) still alive before trace replay` at any B1 or
  B32 replay;
- rerun once without tracking and classify any remaining generic warning
  against the full-depth tracker result. The final normal run retains one
  warning for reserved trace storage; the exact tracked run suppresses that
  generic path and proves zero unsafe program-cache survivors across every
  B1/B32 replay.

No profiler is needed. The decisive artifact is the tracker survivor check at
trace replay plus its allocation contexts and tracebacks.
