# AutoDebug: batch-12 attention query handoff

Date: 2026-09-05. Fresh source-only investigation for the optimized decoder of
`ornith-ai/Ornith-1.5-9B`. Only this report was written. No TTNN imports, tests,
device access, reset, profiler, or implementation changes were performed.
Hardware experiments belong exclusively to `/root/fix_batch_l1`.

## Finding

**The native RoPE output grid violates the downstream SDPA sharded-query address
contract at B12.** RoPE places one user on each core of a 6-by-2 rectangle. SDPA,
configured for an 8-by-8 grid, reads query shards from the first twelve cores of
that grid instead. Its reader does not use the actual query shard coordinates.
Two requested source cores contain no query shard; four others contain a
different user's query. This is a concrete source-level defect and the leading
explanation for the restored eager/replay failure. Hardware confirmation remains
pending; no successful repair is claimed here.

The narrow intervention is the optimized query handoff after native RoPE:
interleave Q, or reshard Q into the consumer's actual first-B core layout, whenever
its existing shards do not meet SDPA's contract. Preserve native RoPE, all batch
sizes, the chosen precision policy, cache behavior, and exact equality gates.

## Evidence and reproduction

The original run is
[autofix_batch_final_resources.log](logs/autofix_batch_final_resources.log), with
the exact argv/environment in
[its provenance](logs/autofix_batch_final_resources.provenance.json) and the
executed Python sources in
[its source archive](logs/autofix_batch_final_resources.sources.json.gz).
The run ended at `2026-09-05T01:17:26.582174+00:00`, return code 1: 34 passed,
4 failed. The other two failures are B17/B31 grouped-RoPE L1 allocation errors,
already assigned to the resource-repair agent.

The recorded command was:

```text
/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_batch_resources.py -v -s -k ""
```

I checked the archive and log hashes against provenance and every archived
source hash against the manifest; all agree. Relevant executed source hashes:

| Source | SHA-256 |
| --- | --- |
| `tt/optimized_decoder.py` | `e23d4ee39345344e291132e8d9e5885c8d45bce3a4166759dfaa6e1b2c5711b1` |
| `tt/fused_decoder.py` | `18d59502e7e168e584e58762396b9cc61eae6de304045542d15bbcc070f9b11d` |
| `tests/optimization_extra_candidates.py` | `a8096c2371299553469ef68b46062f7fd561070a66e175c7608cc945f095a86b` |

The archived optimized class has no `_decode_rotary` override. The current
resource-repair override handles `users_per_core > 1` and still delegates B12 to
the same fused implementation. This investigation uses the archive for the
failing Python path, and the unchanged checkout SDPA sources at recorded Git
HEAD `bc8f514f3000da7b24c4d2b289b0ea507674e999` for lowering. The Python archive
does not independently fingerprint the loaded compiled library.

At log lines 145 and 336, full-attention B12 with DRAM and borrowed sharded public
input has identical failure statistics:

| Comparison | Maximum absolute difference | PCC |
| --- | ---: | ---: |
| First eager vs restored second eager | 0.0625 | 0.9999919921214356 |
| First eager vs restored trace 0 | 0.0625 | 0.9999922010329753 |
| First eager vs restored trace 1 | 0.0625 | 0.9999921800277936 |
| First eager vs restored trace 2 | 0.0625 | 0.9999921585373568 |
| First eager vs restored trace 3 | 0.0625 | 0.9999923060031702 |
| First eager vs restored post-stress trace | 0.0625 | 0.9999921277982187 |

Every corresponding K/V-state and immutable token/prefix comparison is exact.
B4/8/13/16/32 full-attention output/state comparisons pass for both public memory
variants. All linear-attention resource-suite cases pass, including B12.

`test_batch_restored_trace` prefills 63 tokens, decodes at position 63, snapshots
host state, restores into the original state allocations, checks restoration and
addresses, performs four blocking restored replays, and synchronizes after 32
nonblocking replays before the post-stress restored replay. Tensor readback also
waits for the relevant results. The eager failure precedes trace capture, so
trace capture alone cannot explain it.

## Actual operation and precision path

The selected class is `CombinedDecodeCandidate`: `_attention_decode` resolves to
`ShardedHeadNormCandidate` (`optimization_extra_candidates.py:381–428`). It uses
QKVG projection, decode head split, separate sharded Q/K RMSNorm, native partial
RoPE, fused paged K/V update, SDPA, output gate/projection, and the optimized
residual/MLP path. There is no functional block fallback.

The model config gives Q heads 16, KV heads 4, head width 256 and RoPE width 64.
Before SDPA, logical Q is `[1,B,16,256]`, physically `[1,B,32,256]`, BF16 tiled.
For one user per RoPE core its shard shape is `[32,256]`, row major in L1.
Projection weights are original-host-packed BFP4/LoFi; projection and public
activation policy remains BF16. Residual sums retain the existing homogeneous
FP32 adaptation followed by BF16 rounding. RoPE uses the inherited HiFi4,
FP32-destination compute config. KV is BFP8 tiled DRAM, page block 64, four KV
heads, distinct physical block spans per user. Updates use
`paged_fused_update_cache` and reads use
`paged_scaled_dot_product_attention_decode` with device INT32 positions and page
table. There is no CCL in this per-device failure.

The candidate explicitly supplies SDPA grid `(8,8)`, Q chunk 32, K chunk 256,
and `exp_approx_mode=False`. It does **not** pass the inherited
`sdpa_compute_kernel_config`; the public decode wrapper supplies its default
HiFi2 config (`sdpa_decode.cpp:154–155`). Keep this effective setting fixed in
the first control instead of silently substituting the inherited prefill config.

## H1 — incompatible producer/consumer query coordinates

**Rank 1: concrete structural bug; high confidence as the observed cause.**

1. `fused_decoder.py:35–46` chooses the largest rectangular core count dividing
   B, breaking ties toward wider rectangles. On the reported 11-by-10 device
   grid, B12 selects 6-by-2, one user/core. `_decode_rotary:350–372` slices,
   rotates, and concatenates Q and K in that rectangle. Its handoff at
   `:374–377` interleaves Q only for grouped users. B12 therefore stays sharded.
2. `optimization_extra_candidates.py:403–412` explicitly relocates K for the
   fused cache update. V retains its original split-head shards. Neither uses Q;
   exact K/V state is compatible with a bad Q-only SDPA read.
3. `sdpa_decode.cpp:157–178` forwards the supplied Q to the device operation
   without resharding it. GQA is ordinary attention here, not the MLA replicated
   Q special case.
4. `sdpa_decode_program_factory.cpp:301–314` places batch output cores at
   `(batch_index % grid_size.x, batch_index / grid_size.x)` whenever Q is
   sharded and no subcore grid is supplied. `:352–365` constructs the physical
   output-core coordinate arrays from this assignment. Query shard coordinates
   do not participate. `:939–940` passes those arrays to the reader.
5. `kernels/dataflow/reader_decode_all.cpp:186–218` selects the output core for
   the current batch and calls `read_q`. In
   `kernels/dataflow/dataflow_common.hpp:524–560`, the sharded branch reads
   `q_addr` from that core's L1. It does not consult the shard grid or apply the
   DRAM branch's batch offset. An output core reads its own corresponding L1
   address. For this non-MLA case, this contract requires one padded user per
   output core with the exact same row-major ordering.

Concrete B12 mapping:

| User | Correct RoPE Q core | SDPA source core | Q actually resident there |
| --- | --- | --- | --- |
| 0–5 | `(0–5,0)` | `(0–5,0)` | Correct user |
| 6 | `(0,1)` | `(6,0)` | No Q shard |
| 7 | `(1,1)` | `(7,0)` | No Q shard |
| 8 | `(2,1)` | `(0,1)` | User 6 |
| 9 | `(3,1)` | `(1,1)` | User 7 |
| 10 | `(4,1)` | `(2,1)` | User 8 |
| 11 | `(5,1)` | `(3,1)` | User 9 |

The two absent locations can expose unrelated/reused L1 contents, providing a
mechanism for differing repeated outputs despite stable logical Q and KV.
Reading the other four wrong users can produce a repeatable correctness error
that an own-eager equality test cannot detect. The final error magnitude and
the exact contents at the absent locations cannot be deduced from source.

B4 uses 4-by-1, B8 8-by-1, B16 8-by-2, and B32 8-by-4: their Q locations exactly
match SDPA's first-B coordinates. B13 uses grouped 1-by-1 then interleaved Q,
which activates the reader's proper batch-offset path. This explains every
reported passing contrast without changing precision or restoration behavior.

The validator (`sdpa_decode_device_operation.cpp:82–101`) checks sharded Q's
memory-layout category, but does not enforce this coordinate contract. Merely
having `B` height shards is insufficient.

Static enumeration of the existing selector, with reported device grid 11-by-10
and SDPA grid 8-by-8, predicts incompatible still-sharded Q at
**B9/10/11/12/14/15/18/20/21/22/25/27/28/30**. These are source predictions, not
additional measured failures. B26 uses two cores with 13 users/core and takes
the interleaved path. B1–8 and B16/24/32 have compatible coordinates.

## Focused verify/refute experiments

Run each experiment serially in the existing hardware lane with the original
provenance environment and new source/log provenance. The minimal existing test
selection is:

```text
python_env/bin/python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_optimized_batch_resources.py -v -s -k 'test_batch_restored_trace and full_attention and 12'
```

This selects both public input memory variants. Confirm collected node IDs.
Do not drop the exact output/state/address/immutability assertions.

1. **Q-only interleaved control.** For B12 only, immediately after native RoPE,
   convert returned Q to DRAM; preserve K and all other settings. Prediction:
   restored eager and all trace/stress output comparisons become exact, with
   unchanged exact KV. If not, localize before broad changes. This control also
   changes SDPA's core-group arrangement, so a pass establishes Q-handoff
   sensitivity; it is not by itself proof of the precise L1 contents read.
2. **Same-sharded-branch control.** Reorder Q into `[32,256]` height shards on
   `(0..7,0)` plus `(0..3,1)` before the same SDPA call. Retain sharded-Q dispatch,
   the existing SDPA core arrangement and arithmetic configuration. Prediction:
   exact output/state restoration, and agreement with the Q-DRAM control within
   the numerical expectations of the two SDPA layouts. Exactness is required
   within each implementation's own eager/replay path; HF still supplies the
   independent accuracy gate. This control more directly tests the coordinates.
3. **Boundary localization if needed.** In a focused component run with fixed
   K/V, retain/read the logical Q before SDPA and SDPA output before the gate.
   Compare repeated runs by user; also log physical/padded shapes, dtype, shard
   grid/orientation/shape, Q address and effective SDPA grid. H1 predicts stable
   logical Q for all users, with the first unstable SDPA results concentrated in
   users 6/7. Users 8–11 can be consistently wrong. Compare those against the
   corrected mapping, or a host oracle using the same BFP8 cache values, rather
   than assuming repeated equality proves correctness. Readbacks perturb
   scheduling, so the original uninstrumented test remains the final check.

A geometry-only control using SDPA width 6 and enough rows to cover the batch
should also avoid B12's coordinate mismatch, but changes work allocation and
offers weaker isolation than the canonical sharded-Q control.

## Lower-ranked alternatives and refutation criteria

| Hypothesis | Existing evidence and rank | Focused prediction / next action |
| --- | --- | --- |
| Q normalization, partial-RoPE padding, or Q producer lifetime causes unequal Q values before SDPA | Rank 2, unproven. Stable BFP8 K state does not strictly prove every pre-quantization Q/K value is identical. No producer-boundary readback exists in the failing artifact. It does not explain the independently established coordinate violation. | Compare logical Q before SDPA across restore/replay. If Q already differs, localize norm → rotated part → concatenation with unchanged precision. If Q is exact and canonical mapping repairs SDPA, this is unnecessary to explain the failure. |
| An independent SDPA reduction/CB or later residual/MLP defect | Rank 3, currently unsupported. Eager/replay dependence can occur below these ops, but neighboring batches and the exact coordinate mismatch favor H1. The earlier mixed-dtype residual bug has a separate proven adaptation in the failing source. | If canonical Q plus identical KV still yields unequal SDPA output, isolate that component at exact B12 geometry; if SDPA output is exact, continue through gate, output projection and first residual. Do not change weights or fidelity before finding the first divergence. |
| Missing persistent-state restore, public-input mutation, or trace-only binding | Strongly disfavored for this symptom. The first repeated eager call already fails, host state restoration/readback and state addresses are checked, and immutable token/prefix checks pass. | Only revive with evidence of an additional mutable input/state field or unequal actual SDPA inputs. |

There is no evidence here for inherent low-precision instability. A precision
change is neither necessary to repair an invalid query address nor a useful
first isolation step. If later cache-precision experiments become necessary,
preserve the exact page64/four-KV-head/batch allocation and run the same-cache
high-precision control required by AutoFix.

## Likely narrow fix and required follow-through

Keep the repair in optimized orchestration: after native RoPE, ensure sharded Q
has one user/core and exactly the consumer's first-B core sequence, including
orientation. Interleave Q on mismatch, or explicitly reshard it to that sequence.
Do not special-case B12 alone. **`_batch_grid(device,B)` is not the correct
canonicalizer for this candidate:** it uses the device width 11 while SDPA uses
width 8. Derive the sequence from the actual SDPA program grid/subcore grid, or
centralize the chosen grid in optimized configuration before both consumers use
it. Comparing only rectangle width would needlessly change compatible B1–7;
compare the complete expected core sequence or an equivalent exact predicate.

The ongoing grouped-RoPE temporary-lifetime repair is independent. It can keep
its release order; this report does not propose duplicating or reversing it.
For one-user rectangles the optimized handoff can adapt the inherited outputs
without editing the fused/functional implementations. Deallocate only owned
temporary Q after a real layout conversion; do not touch borrowed public input.

After the focused control verifies H1, retain the smallest proven fix and rerun
the original resource suite, real-weight HF decode coverage for B12 and at least
one additional predicted incompatible rectangle (for example B9 or B14), nearby
compatible B8/B16 cases, and a separate minimal watcher run for this L1 address
boundary. Prefer a source-level geometry regression covering B1–32 plus targeted
device cases to hiding the latent cases with a B12-only policy. Preserve all
existing context, precision, capability, exactness and HF gates. Any performance
effect requires measurement by the owning lane; none is claimed here.

**Status:** source diagnosis complete; H1 hardware verification and implementation
are pending with `/root/fix_batch_l1`. The original failure remains a failure.
