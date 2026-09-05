# AutoDebug: BFP4 KV cache and batch-32 real-input accuracy

Date: 2026-09-05. Fresh source-only investigation under AutoFix. Only this
report was written; no implementation edits, TTNN imports, hardware commands,
tests, resets, or profiling were performed. The root agent owns hardware and
verification. All proposed causes below remain hypotheses until measured.

## Finding

**The final BFP4-cache default fails an existing real-input per-user gate. The
available evidence does not establish a batch-specific bug or an inherent BFP4
cache limit.** The leading hypothesis is cumulative projection/cache error on
particular recorded inputs. The decisive first experiment is to replay each
actual failing batch-32 user as batch one, preserving its exact prefix and token.
A different passing batch-one prompt does not justify a batch-aware fallback.

No concrete page-allocation defect was found for this failure. Position 96 is
page 1, offset 32: a tile boundary inside an allocated page, not a page or
256-token SDPA chunk boundary. Both cache format handling and the tile update
still require an exact-shape probe before assigning the cause to precision.

## Starting evidence

The original command in [provenance](logs/final_release_short.provenance.json) is:

```text
/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python -m pytest /home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_decoder.py -m "not long" -x -v -s
```

It uses `ORNITH_WEIGHTS=real`, default candidate/policy/config, and recorded
inputs. [The log](logs/final_release_short.log.gz) fails
`test_batched_prefill_decode_pcc[blackhole-32-full_attention-mesh_device0-device_params0]`
at `test_functional_decoder.py:388`: one user's decode PCC is
**0.9936757125704948**, below the unchanged 0.995 threshold. Every prefill user
passes; aggregate prefill PCC is 0.997484. The failed user index and aggregate
decode PCC are not logged because assertion stops the loop. Do not infer that
only one user fails.

The failing executed optimized source hash is
`3a1962225668b2451d188caa5f2ff14c95d525f28e31f8759561b2b04bca0bf9`.
The inspected working source hash is
`712ba019737906f9e65c44c0dc5d578d52f5eb5208085960f0fe8388a788bc6f`;
comparison with [the archive](logs/final_release_short.sources.json.gz) shows
only a multiline conditional-expression formatting change. Both relevant test
files still match the failing manifest exactly.

| Recorded run | Cache | Relevant result |
| --- | --- | --- |
| `final_cache_bfp4` | K4/V4 | B1 HF prefill/decode 0.9978195529861253 / 0.9986027463924887; exact restored replay; 6.2267739559 / 0.4240502785 ms |
| `final_cache_bfp4_attention_bfp8_hifi2_control` | K4/V4 | B1 attention-weight BF8/HiFi2 control: 0.9985089665663278 / 0.9992500071409154; 6.9396150066 / 0.4961429986 ms |
| `autofix_batch_combined_final` | K8/V8 | Previous candidate B32 HF aggregate prefill/decode 0.998627 / 0.998686, with per-user assertions passing; B32 traced decode also passes |
| `autofix_batch_all_full_exact` | K8/V8 | Previous candidate all B1–32 per-user HF and restored-state replay passes; minimum decode PCC across cases 0.9962758864936756 |
| `final_release_short` | K4/V4 | Current default B32 per-user decode fails as above |

These are historical measurements, not new tests performed by this investigator.
The earlier BF8 batch controls precede final integration and use their recorded
candidate classes. They motivate a current-source cache-only A/B; they do not
replace that A/B. The B1 same-cache higher-precision control does not test the
new failing inputs.

## Effective precision and geometry ledger

Sources: `tt/optimized_decoder.py:23–29,88–94,176–183,505–553,622–623,705–751`,
`tt/functional_decoder.py:112–118,150–164,301–329`,
`tests/test_functional_decoder.py:150–186,350–392`, and
`tests/test_optimized_decoder.py:78–105`.

| Boundary | Failing default / cache-only BF8 control | Same-cache attention control |
| --- | --- | --- |
| Input/ordinary projection activation and output | BF16 | Same |
| Attention Q/K/V/gate and output weights | Original-host-packed BF4 | Original-host-packed BF8 |
| Attention projection compute | LoFi, approximate false, FP32 destination false, packer L1 accumulation true | HiFi2; other flags unchanged |
| MLP gate/up/down weights and compute | BF4/LoFi | Same |
| Norm/RoPE compute | Inherited HiFi4, approximate false, FP32 destination true | Same |
| Prefill SDPA | HiFi2, approximate false, FP32 destination true | Same |
| Decode SDPA | HiFi2, approximate true, FP32 destination false, packer L1 accumulation false; exp approximate false | Same |
| K/V cache | Failing K4/V4; A/B K8/V8 | K4/V4 |
| CCL payload | None: one-device mesh | Same |
| Cache/page policy | TILE interleaved DRAM, replicated mapper; page 64; 4 local KV heads, width 256; disjoint identity physical spans per user | Same |
| Fill/update/read | Cast each prefill K/V to its cache dtype, `paged_fill_cache`; BF16 decode K/V to `paged_fused_update_cache`; chunked prefill / paged decode SDPA | Same |

For B32/max-context1024, `num_blocks_for_context` rounds 16 needed pages up
to **32 pages per user**. Cache shape is **[1024,4,64,256]**, page table
**[32,32]**, each user has 2048 physically backed positions. Logical prefill96
is padded to128 and filled through127. Decode consumes Q `[1,32,16,256]`
(padded heads32); update rows K/V are logically `[1,32,4,256]`, one padded
`[32,256]` shard per user, on disjoint K/V core sets. SDPA uses grid8x8,
Qchunk32, **fixed Kchunk256**. Its first read window is [0,256), covered by
four valid pages per user. This is not a dynamic-chunk selection failure:
fixed Kchunk256 was supplied. A physical coverage issue at position96 is not
supported by these numbers.

The earlier B12 query-grid bug is already addressed by `_sdpa_query`, which
canonicalizes sharded Q onto the consumer's first-B core sequence. B32's 8x4
RoPE rectangle already agrees with that sequence. Cache dtype alone does not
change this coordinate mapping. Reopen that hypothesis only if Q readback,
B1/B32 comparison, or exact restored replay supplies contrary evidence.

## Minimal verify/refute sequence

1. **Identify users and hold inputs fixed.** Run the exact B32 test with a
   temporary optimized-test diagnostic that collects all per-user PCC values
   before asserting the same 0.995 gate. Record first failing user, every failed
   index, positions, max-absolute error, and all lowered shapes/dtypes/layouts.
   The fixture uses recorded rows, not random inputs: for user `u`, prefix rows
   are `(39 + 137*u + arange(96)) % N`, and the decode row is
   `(131 + 137*u) % N`, where `N` is the recorded layer-3 corpus length.
   Record/slice these actual tensors; do not recreate them with the ordinary B1
   test's seeds. Input rows are real captured activations, but this assembled
   prefix is not claimed as a contiguous HF generation rollout.
2. **Extract each failing user to B1.** Use exactly `prefix[u:u+1]`,
   `token[u:u+1]`, position96 and an independently allocated B1 cache. Compare
   B1 output both to HF and to that user's B32 output. If BF4 also fails B1,
   reject a B>1-only fallback: the failure follows input content. If B1 passes
   and B32 does not, repeat the same user in different B32 slots and permute
   page mappings to separate batch/slot dependence from input variation.
3. **Current-source cache-only control.** Repeat the exact B32 and extracted
   B1 cases with explicit BF8 allocation, keeping weights, compute settings,
   input tensors, page widths, mapper and update/read ops fixed. On this default
   class, `ORNITH_KV_DTYPE` is **not read**; use an explicit allocator argument
   in an optimized test helper/subclass and log the actual K/V dtype. Do not
   silently select an old experimental class with different attention code.
4. **Same-cache higher-precision control on the failing inputs.** Keep K4/V4
   and set only attention weight policy to BF8/HiFi2 (MLP remains BF4/LoFi).
   The existing candidate selector accepts
   `ORNITH_OPT_POLICY='{"attention":"bfloat8_b","attention_fidelity":"HiFi2"}'`.
   A pass establishes sensitivity to attention projection precision, not that
   BF4 cache is intrinsically defective. If helpful after localization, separate
   weight dtype from fidelity, or test only decode SDPA FP32 accumulation with
   Q/K/V frozen; each is an independent A/B, not a bundled proposed fix.
5. **Exact-shape fill/update/read probe.** Import the model's config, page size,
   allocation helper, actual mapper, and update shard contract. Use B32,
   `[1024,4,64,256]` cache and `[32,32]` tables. Capture real projected Q/K/V,
   fill the prefix, snapshot physical and logically unmapped cache entries,
   update position96, then compare K and V before/after separately. Verify the
   correct user/head/page/row changes and all untouched valid history. Run
   identity and permuted disjoint page tables. Compare paged SDPA to a CPU
   attention oracle consuming the **read-back dequantized cache and same Q**;
   separately compare to unquantized HF Q/K/V. Thus write/address error,
   quantization error and attention-kernel error are distinguishable. Exact
   format controls may use representable sentinel values for address diagnosis;
   synthetic accuracy must not veto a correct measured real-input policy.

For a position sweep, keep the same recorded-data policy and check
63/64/65, 95/96/97, 127/128/129 and 255/256/257. Log first failing position,
page, tile, chunk and all user PCC values. Only if a cliff or dependence on
unused pages appears, run a same-policy over-allocation/poisoned-unused-page
control before numerical conclusions. Existing failure at96 alone establishes
neither a cliff nor an insufficient allocation.

## Q/K/V localization and mixed-cache constraint

A compact localization ladder captures prefill and decode Q/K/V before cache
quantization, K/V after fill and after update, raw SDPA heads, output projection,
and final block output. CPU SDPA on dequantized cached K/V with actual Q isolates
the reader/math boundary; replacing one of Q, K, or V with its reference version
then separates projection drift, key/softmax sensitivity and value error.
Raw cache PCC is diagnostic; the unchanged final real-input PCC gate decides
acceptance.

**Mixed K8/V4 or K4/V8 must not use the present fused update as-is.** Decode SDPA
validates each tensor dtype independently and its program factory derives
separate K and V formats (`sdpa_decode_device_operation.cpp:37–44`,
`sdpa_decode_program_factory.cpp:429–431`). In contrast,
`paged_tiled_fused_update_cache_program_factory.cpp:94–103` derives one cache
format/tile size from `cache_tensor1` and applies it to both cache/output CBs
across both core sets (`:194–202,248–256`). The validator checks the two cache
dtypes independently without enforcing equality. Thus apparent API acceptance
is not evidence that mixed fused updates are supported. This is a separate
source hazard, not the cause of the equal-BF4 failing run.

A safe localization experiment can use frozen, already-updated mixed caches
with decode SDPA, or two independent `paged_update_cache` calls, each supplied
its own cache dtype and the model's same row/page contract. Keep this diagnostic
out of the production path unless correctness and performance justify it.
No C++ change is needed for the present equal-dtype investigation.

## Repair decision and required verification

Retain only an isolated, measured repair. BF8 cache is a plausible narrow
fallback if it fixes the actual failure and the exact-shape probe excludes an
address/update error. A same-cache attention improvement or mixed-cache policy
is also only a candidate until tested and timed. B1's old measured BF4 win must
be preserved only while it remains correct for actual B1 input diversity.
A BF8-for-B>1 default requires both failing-user B1 controls and full batch
verification; otherwise it hides an input-sensitive B1 failure. Also,
`build_decoder` currently allocates KV **before** `allocate_state(batch)`, so
`self.batch_size` is unset at allocation time. Do not infer batch from total
page count or add host-dependent changes inside decode/trace to work around
that ordering.

After a proven change, rerun the original short suite, all B1–32 real per-user
and exact restored trace contracts, and required native262144 and continuation
coverage in the final dtype. Run any required watcher check separately from
profiling. Preserve all PCC thresholds, logical batches, native capacity and
functional/fused source boundaries. Measure the final real-input B1 pair and
relevant changed paths before calling any alternative fastest.

**Status:** source diagnosis complete; the original real-input failure remains
unresolved. The minimal controls above are ready for the sole hardware owner.
