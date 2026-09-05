# AutoDebug: full-attention optimized decode passes cache-write kwargs to paged SDPA

## Headline finding

**High confidence: `OptimizedDecoder._attention_decode` reuses the flat cache-write
kwargs for an SDPA API that now requires `PagedCacheGeometryOverride`.**

The optimized path builds `cache_view` with
`_cache_view_kwargs(prefill=False, ...)` and correctly passes it to the two
`paged_update_cache` calls. It then incorrectly passes that same dictionary to
`paged_scaled_dot_product_attention_decode` at
`tt/optimized_decoder.py:1136-1146`.

For a full-attention layer, the dictionary is:

```python
{"block_size": 128, "num_kv_heads": 2}
```

Those remain valid keywords for `ttnn.experimental.paged_update_cache`, whose
binding declares both fields. They are not keywords of the current paged decode
SDPA binding. The SDPA binding declares the single atomic keyword
`paged_cache_geometry`, plus the independent `cache_position_modulo` keyword.
Nanobind therefore raises `TypeError: incompatible function arguments` before
the SDPA operation reaches validation or a device kernel.

`FunctionalDecoder` already contains the intended split:

- `_cache_view_kwargs(...)` supplies flat `block_size` / `num_kv_heads` to
  paged cache fill/update operations.
- `_sdpa_cache_view_kwargs(...)` supplies
  `PagedCacheGeometryOverride(block_size=128, num_kv_heads=2)` to paged SDPA.
- Its decode call uses the latter helper at `tt/functional_decoder.py:927-937`.

`OptimizedDecoder` inherits `_sdpa_cache_view_kwargs` unchanged, so the repair
does not require a new helper or any policy-specific logic.

## Why this matches the entire reported matrix

The optimized perf profile parametrizes layer 0 sliding attention and layer 5
full attention, each at batch 1 and batch 32
(`tests/test_optimized_decoder.py:565-572`). The layer kind alone controls the
bad kwargs:

| Case | `_cache_view_kwargs(prefill=False)` | Binding result |
| --- | --- | --- |
| sliding, batch 1 or 32 | `{}` | No obsolete keyword is passed; this bug is absent |
| full, batch 1 or 32 | `block_size=128, num_kv_heads=2` | Current SDPA binding rejects the call |

This predicts both reported full-attention failures and both passing sliding
measurements. Batch size, trace iteration count, math fidelity, and sharding do
not affect keyword parsing. The profile harness performs an eager decode during
its compile/warm-up sequence before trace capture, so this mismatch can fail the
run immediately and is not a trace replay failure.

The profile's full-attention case uses a natural physical cache
(`shared_physical=False`). That does not invalidate the diagnosis: the
functional implementation deliberately supplies the atomic geometry for every
full-attention SDPA call, and the C++ validation accepts a no-op geometry view.
For the natural cache, both sides contain `2 * 128 * 512 = 131072` elements per
block. A shared sliding-shaped physical block likewise contains
`8 * 64 * 256 = 131072`, so the same full view is also the intended HMA
reinterpretation.

## Causal history

Git history makes this a missed caller migration rather than an ambiguous
runtime regression:

1. Commit `b6b3119fdd9` replaced paged SDPA's flat `block_size` /
   `num_kv_heads` Python arguments with `paged_cache_geometry`.
2. Commit `3f1f3a9de59` migrated `FunctionalDecoder._attention_decode`, added
   `_sdpa_cache_view_kwargs`, and added a host contract test.
3. `OptimizedDecoder._attention_decode` retained its earlier copied call and
   still forwards `**cache_view`.

The fused baseline does not reproduce the stale call because it inherits the
corrected functional attention implementation; the optimized class overrides
that material path.

## Focused source/host experiments

No TT device was opened and no hardware test was run.

1. **Binding contract check.** Reading
   `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp:78-106`
   shows `paged_cache_geometry` and `cache_position_modulo`, with no flat
   geometry keywords. A host-only call with `None` tensor placeholders printed
   the installed binding's available signature; it also lists
   `paged_cache_geometry` and rejects the legacy call containing
   `block_size=128, num_kv_heads=2`.
2. **Write contract check.** Reading
   `ttnn/cpp/ttnn/operations/experimental/paged_cache/paged_cache_nanobind.cpp:54-70`
   confirms `paged_update_cache` still accepts `block_size`, `num_kv_heads`,
   and `cache_position_modulo`. Therefore the existing `cache_view` must remain
   on the two writes.
3. **Inherited helper check.** A host-only `object.__new__(OptimizedDecoder)`
   probe produced `{}` for sliding SDPA and
   `PagedCacheGeometryOverride(128, 2)` for full SDPA. The existing host test
   `test_sdpa_cache_view_uses_atomic_geometry_override_host` passed (`1 passed`)
   without opening a device.
4. **Lowered geometry check.** C++ validation computes the effective SDPA view
   from `PagedCacheGeometryOverride` and Q's head dimension, checks that the
   physical and logical per-block element counts agree, and then threads the
   override into the program factory. This is the contract the functional
   helper satisfies.

## Hypotheses considered

### H1 — stale caller API (confirmed, primary)

The caller provides two keyword names absent from the binding, only for full
layers. This is sufficient to cause the exact error and explains every
passing/failing case named in the report.

### H2 — invalid full-attention cache geometry (refuted for the observed error)

Both natural and shared physical layouts preserve 131072 elements per block,
and the requested `(2, 128, 512)` view satisfies current validation. Geometry
could only be evaluated after binding, whereas the observed legacy keywords are
rejected at binding.

### H3 — batch-specific SDPA grid, trace, or numerical policy issue (not causal here)

The same source mismatch applies at batch 1 and 32 and is resolved before the
program config or tensor contents are consumed. Such issues may still require
normal post-fix hardware validation, but they cannot produce this keyword
`TypeError`.

### H4 — stale installed extension rather than stale model caller (refuted)

The installed host binding and checked-in nanobind source expose the same
current signature. The divergence is between `OptimizedDecoder` and that
contract.

## Recommended minimal fix

Keep `cache_view` exactly as-is for both `paged_update_cache` writes. Change only
the final argument expansion of optimized paged SDPA:

```diff
-            **cache_view,
+            **self._sdpa_cache_view_kwargs(cache_position_modulo=cache_position_modulo),
```

This is the same call boundary already used by `FunctionalDecoder`; it preserves
the optional modulo field and emits no geometry override for sliding attention.
It is a one-line Python-only repair and requires no TTNN, C++, cache allocation,
or test-harness change.

Do not replace the write kwargs with `_sdpa_cache_view_kwargs`: cache update and
SDPA intentionally have different public representations of the same geometry.

## Regression checks after the fix

1. Add or extend a cheap host audit in `test_optimized_decoder.py` that verifies
   the optimized paged SDPA call expands `_sdpa_cache_view_kwargs`, not the
   write-form `cache_view`. An AST/`inspect` audit is sufficient to catch this
   caller drift without a device; also instantiate `OptimizedDecoder` without
   `__init__` and verify full/sliding plus modulo helper outputs.
2. Run the existing optimized real-weight correctness cases for full attention
   with both natural and shared physical cache views. These validate that the
   atomic override is not merely accepted but reads the same view written by
   `paged_update_cache`.
3. Rerun the reported optimized perf profile and require all four cases:
   sliding/full times batch 1/32. The full cases are the direct regression gate;
   the sliding cases ensure the empty-override path stays unchanged.
4. Preserve `throw_exception_on_fallback=true` and the trace path in the final
   run, then check the full-attention artifacts were freshly written. The source
   diagnosis cannot establish device correctness or performance.

Because this is Python-only, repository policy does not require a C++ build.
Hardware correctness and performance still require the normal device runs.

## Nearby issue outside this optimized-decoder repair

`tt/multichip_decoder.py:873-892` contains the same stale pattern: it correctly
uses its flat local cache view for updates and then passes `**cache_view` to
paged SDPA. Full-attention multichip decode will therefore encounter the same
binding mismatch when exercised. This is not on the current single-device
optimized repro path and should not broaden the minimal fix.

If repaired separately, multichip must not blindly use the inherited functional
SDPA helper: each rank owns one local full-attention KV head, while that helper
uses the global count of two. It needs a multichip SDPA helper/override with
`block_size=128` and `num_kv_heads=1`, plus a local contract test.
