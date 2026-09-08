# Adapter async probe: authoring and supervising validation

Date: 2026-09-08 UTC. Added
`tests/adapter_serving_device_probe.py`; no adapter, generator, model, plugin, or
runner implementation changed in this pass. The authoring subagent did not
import TTNN, import the probe, open hardware, or execute inference.

AST parsing and all applicable pre-commit hooks passed. Evidence:
`adapter_probe_static_checks.log`. No C++ or CMake changed, so no build was
needed. Authoring was source-only; the supervising device result is recorded
separately below.

The supervising agent's first device run passed the synchronous and deferred
stale-input cases, then found a probe-helper `TypeError` because TTNN `Shape`
does not support Python slices. The test-only `page_snapshot` helper now uses
`list(tensor.shape)[1:]`. AST parsing and all applicable pre-commit hooks passed
again (`adapter_probe_static_checks_v2.log`). The first device log/JSON remains
the supervising agent's evidence; the authoring subagent did not rerun hardware.

## Supervising v2 result

The supervising agent ran the corrected probe and reported **exit 0**.
`adapter_device_v2.json` records `status="passed"` and
`cleanup_completed=true` for all four cases, using real layers 0/3, batch three,
native logical context 262144, and 4192 shared physical blocks on the TP4 P300c
mesh. The v1 harness correction was solely
`tensor.shape[1:]` → `list(tensor.shape)[1:]` in the page snapshot helper.

| Case | Two sampled token vectors | Final positions |
| --- | --- | --- |
| Synchronous | `[220,220,220]`, `[220,12,220]` | `[64,65,66]` |
| Deferred stale inputs | Exact synchronous match | `[64,65,66]` |
| Changed page | Exact synchronous match | `[64,65,66]` |
| Slot permutation `[2,0,1]` | `[220,220,220]`, `[220,220,12]` | `[66,64,65]` |

Every pair executed exactly two model and sampling replays with zero host
token/current/RoPE refreshes. The deferred cases each made two asynchronous
readbacks. The page case performed one table refresh, left physical page 34
unchanged, and wrote page 97 on every K/V shard. The permutation case recorded
one slot remap and one table refresh. Successive control vectors differ, so the
deferred-copy discriminator was satisfied. The parent subsequently promoted
`supports_async_decode=True` in the adapter; this probe itself does not modify
capabilities. These remain reduced adapter contract results, not a full serving
stage pass.

## Supervising lane command

Run only when the supervising watchdog owns an idle hardware lane, with its
established runtime environment and bounded-run policy:

```bash
USER=hous ../state/serving-env/bin/python -m \
    models.autoports.ornith_ai_ornith_1_5_9b.tests.adapter_serving_device_probe \
    --model-path ../upstream --batch 3 \
    --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/adapter_serving_device_probe.json
```

`--batch 4` is also supported. This script opens the existing TP4 mesh helper:
four Blackhole chips on two P300c boards. It does not profile or start a server.

## Actual execution boundary

The probe loads the real pinned HF configuration locally, calls
`TTOrnithForCausalLM.initialize_vllm_model`, and selects actual checkpoint layers
0 and 3 through `ORNITH_VLLM_LAYER_INDICES`. It allocates the caller-visible
cache through `adapter.allocate_kv_cache` and verifies the generator uses that
exact object with `owns_cache=False`. Every prefill, decode, async read, and
host-output formatting call goes through the actual adapter API.

The logical context is 262144, the native page-table width is 4096, and the
physical pool is `width + batch * num_blocks_for_context(256)`: 4192 blocks for
batch three, 4224 for batch four. This follows the parent's hardware finding
that `paged_fused_update_cache` requires physical blocks at least equal to the
page-table width. It remains a shared pool smaller than one complete native
allocation per request. Each request receives 32 mapped blocks; unused table
columns address reserved block zero, and a spare physical block supports the
scheduler-boundary test.

Sampling remains on device throughout. There is no host argmax or full-logits
fallback. Each case resets the A/B fixture through `generator.reset` and the
adapter's bookkeeping flags, then repeats actual adapter prefill and one normal
decode to establish live device-owned token and position state. Those explicit
fixture resets are outside the tested pair; this is not a public request-reset
API test.

## Assertions

- **Synchronous control:** ragged real-token prompts of lengths 61/62/63
  (plus 64 at batch four), then two synchronous adapter decodes.
- **Deferred stale-input pair:** submit decode N and its async output copy,
  then decode N+1 and its copy before waiting on either event. Supply valid but
  deliberately wrong host token IDs and current positions. Both outputs must
  exactly match the synchronous control, and their host copies must be distinct
  from the persistent device output and each other.
- **Changed scheduler page:** row one's second tested decode is exactly
  position 64, the first live token of logical page one. Change that mapping to
  a spare block immediately before this decode. Both outputs must still match
  the control; the old page must remain byte-equivalent on every K/V shard,
  while every new K/V shard must be written. The persistent device page table
  must equal the caller's changed table on every replica.
- **Slot permutation:** pass a full cyclic `slot_remap` plus the correspondingly
  permuted page table through the adapter. Continue supplying stale host
  values. Both outputs and final current/RoPE positions must equal the
  correspondingly permuted synchronous result.
- **State and work accounting:** every replica's current position and RoPE
  index must advance exactly twice across the tested pair, persistent input
  buffer addresses must remain stable, and the final feedback token must equal
  the final output. Each pair must execute exactly two model and sampling
  replays, perform no host token/current/RoPE refreshes, and use exactly one
  page-table refresh only for the page-change/permutation cases. Adapter
  counters must show two device decodes and zero host decodes.

The final gate requires that the two synchronous token vectors differ. If the
fixture samples identical vectors, the report explicitly fails as inconclusive
for deferred-copy token isolation; position/counter equality alone is not
silently treated as sufficient proof.

## Evidence and limits

The script emits `ADAPTER_PROBE_BEGIN/PASS` markers and writes JSON after each
case. It records source SHA256 hashes, actual precision/cache geometry,
capabilities at entry, sampled IDs, positions, counter deltas, mapping details,
and any exception. Cleanup independently attempts adapter teardown and mesh
close, recording failures rather than leaving a passed report after failed
cleanup.

The probe never changes `supports_async_decode`. A passing supervising run can
support a later promotion decision alongside real serving evidence. It covers
full slot permutations, not duplicate-source slot condensation, new-request
insertion, cancellation, or scheduler/API concurrency. Those remain separate
integration gates, as do full-model accuracy, all mesh profiles, parser API
behavior, and performance.

## Optional native allocation guard

After the v2 pass, a source-only audit added
`--require-trace-allocation-tracking`. With that flag the probe fails before
mesh opening unless TTNN imported with native allocation tracking enabled and
program-cache allocation tracking remains included. The JSON records the
tracking mode. All applicable pre-commit hooks and AST parsing passed again;
see `adapter_probe_tracker_static_checks.log`.

For the supervising lane's separate diagnostic run, add:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
USER=hous ../state/serving-env/bin/python -m \
    models.autoports.ornith_ai_ornith_1_5_9b.tests.adapter_serving_device_probe \
    --model-path ../upstream --batch 3 --require-trace-allocation-tracking \
    --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/adapter_device_tracker.json
```

The supervising lane subsequently ran the guarded v3 probe and reported exit 0.
`adapter_device_tracker_v3.json` and `adapter_device_tracker_v3.log` record all
four cases passed, cleanup completed, `enabled_at_ttnn_import=true`, and
`skip_program_cache="0"`. No tracked live unsafe allocation survived the
exercised model/sampler replay boundaries. The allocator's general advisory
alone does not establish corruption; this guarded reduced-adapter result
resolves that concern for the tested boundaries without suppressing tracking.
It is neither a physical-address-overlap measurement nor a full-server or
performance result. The detailed API audit is in `trace_allocation_audit.md`.
