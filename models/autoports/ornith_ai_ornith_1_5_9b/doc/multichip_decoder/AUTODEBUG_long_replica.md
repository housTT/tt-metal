# AutoDebug: 8001-token TP4 residual replica assertion

Inspection date: 2026-09-05. Source-only isolated investigation using the
`autodebug` skill. No hardware commands, TTNN imports, implementation changes,
or baseline/C++ edits were made. Hardware described by the supplied evidence:
four Blackhole chips on two P300c boards, logical 1x4 ring.

## Findings

**The supplied failure does not yet establish a particular corrupting operation,
or even that the ranks contain different finite values.** The adapter checks
`torch.equal`, which is false for matching NaNs. Its assertion records neither
finite counts nor coordinates or magnitudes of differences. First save all four
raw outputs, finite masks, and per-chunk discrepancy statistics without changing
the acceptance gate.

Three concrete source findings materially change the diagnosis:

1. The two passing native-context tests read only their final 64 or 128 tokens.
   The failing test reads all 8001 tokens. Earlier chunks can therefore be wrong
   while both native-context tests pass. These are not full-output controls.
2. The 8001-token case executes three physical 2048-token blocks and one physical
   1920-token block containing 1857 valid tokens. The final block selects a
   different prefill matmul geometry. Trimming it also makes final concatenation
   run untilize/unpad, row-major concat, and tilize over **all four** inputs.
3. Native `ttnn.all_reduce` is not an independent reduce-scatter implementation.
   Its prefill path invokes native reduce-scatter, whose program factory calls
   the same ring artifact builder as `reduce_scatter_minimal_async`, followed by
   the new native all-gather. Native all-gather independently detects links;
   `MeshConfig.links=1` changes reduce-scatter links but does not force native
   all-gather to one link.

No definite Python ownership, weight-partition, scatter-axis, or packet-count
violation was found that explains the reported failure. The controls below are
ranked to locate the first bad boundary before choosing a repair.

## Evidence and provenance

Original command, recorded in [final_long provenance](logs/final_long.provenance.json):

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache OMP_NUM_THREADS=8 \
python -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_multichip_decoder.py -m long -x -q
```

The recorder adds `timeout 2400` and uses the repository Python environment.
[final_long.log](logs/final_long.log) reports eight selected tests, two passes,
then failure in linear-attention `test_long_context_pcc` before its HF PCC or
decode runs. The five remaining selected cases are unexecuted, not passes.

The source archive, compressed log, and plain log hashes all match their
provenance. At inspection, current `tt/multichip_decoder.py`,
`tt/optimized_decoder.py`, and `tests/test_multichip_decoder.py` match the
`final_long.sources.json.gz` contents exactly. Relevant implementation hashes:

| Artifact | SHA256 |
|---|---|
| `tt/multichip_decoder.py` | `7b63bbb22ee64f76025d8363af0c60248296925609fc199f620007d12d8c7c0c` |
| `tt/optimized_decoder.py` | `01a3e3d084f6ca039ab78cc9545607f509afd4aff6852d47cb4f9ca1cc7f1608` |
| Source archive | `8c849eae6081aa7131e9a8c687b6df5faa3a3af88e0db1df3e1bd0f5bc6fcf05` |

Comparing the earlier `contracts_wide32_short.sources.json.gz` structurally
shows the implementation delta is import ordering/formatting and omission of
unused separately uploaded decode weights when `decode_grid` is active. Both
the 8192-byte packet policy and 32-core decode path already existed in that
passing short suite. The optimized baseline is byte-identical. This excludes a
new prefill arithmetic change between those two source snapshots, but does not
exclude allocation-layout sensitivity or incomplete shape coverage.

The archive covers model Python, not the full C++/kernel dependency closure or
binary build identity. Lowered-source conclusions below describe the checked-out
source; this inspection does not independently prove those binaries' provenance.

## Exact execution and shape boundaries

The test adapter uploads recorded real activations, seeded with offset 61,
replicated onto the mesh. It builds real layer-zero weights with context 16384,
batch one, and default prefill chunk 2048. The residual is replicated BF16
`[1,T,4096]`. TP-local linear dimensions are Q512/K512/V1024, eight value heads,
four key heads, head width128, MLP intermediate3072. Head groups and their
corresponding output-weight rows remain paired. A and B each occupy their own
32-column field in packed `[4096,2112]` weights.

| Chunk index | Input offset | Logical rows | Physical rows | Prefill grid / K block | Collective input per rank |
|---|---:|---:|---:|---|---|
| 0 | 0 | 2048 | 2048 | 11x10 / 16 tiles | BF16 `[1,1,2048,4096]`, 16 MiB |
| 1 | 2048 | 2048 | 2048 | 11x10 / 16 tiles | same |
| 2 | 4096 | 2048 | 2048 | 11x10 / 16 tiles | same |
| 3 | 6144 | 1857 | 1920 | 8x8 / 8 tiles | BF16 `[1,1,1920,4096]`, 15 MiB |

For 2048 rows, `_prefill_linear` has 64 M tiles, `per_core_M=7`, and
`out_block_h=7`; for 1920 rows, 60 M tiles give `per_core_M=8` and
`out_block_h=8`. Overcoverage is normal matmul tiling and is not by itself a bug.
The altered geometry is a controlled contrast, not proof of faulty matmul.

Each chunk executes:

1. Replicated RMSNorm → local packed QKV/A/B and Z projections.
2. Local convolution/history → gates and chunk DeltaNet → FP32 recurrent-state
   copy and local convolution-state update.
3. Head-major gated RMSNorm (FP32) and Z multiplication → local `gdn_out`
   projection, explicitly BF16 → all-reduce → first BF16 residual add.
4. Replicated RMSNorm → local gate/up projections and SiLU multiplication →
   local down projection → all-reduce → second BF16 residual add.
5. Retain output; trim final `[1,1920,4096]` to logical `[1,1857,4096]`, padded
   to1888 rows; concatenate retained outputs into logical `[1,8001,4096]`,
   padded to8032 rows.

The final concat sees padding on its sequence axis, so
`concat.cpp:build_untilize_rm_retilize_concat` untilizes/unpads **every input**,
concatenates row-major data, and retilizes. Reading a final64-token slice in the
native test exercises a different host-read shape as well as checking fewer
positions. A readback issue must be separated from device-output corruption.

Full-attention has the same residual/MLP/collective/output-collection boundaries;
its mixer instead performs paired local Q/K/V projections, per-head norms and
RoPE, local-head paged cache writes, SDPA and output projection. No cross-rank
head gather is required for either mixer. Full-attention long HF correctness
remained untested in the supplied failed run.

## Native collective lowering and count audit

Relevant checked-out paths:

- `ttnn/cpp/ttnn/operations/ccl/all_reduce/all_reduce.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/reduce_scatter.cpp`
- `ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_program_factory.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_unicast_factory.cpp`
- `ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_{reader,writer,common}.hpp/.cpp`

`all_reduce` passes absent external semaphores to the mesh overload of
`all_reduce_async`. `finding_scatter_dim` checks dimensions from the right in
tile units. Width4096 is128 tiles, divisible by four, so scatter dimension3 is
selected for both physical lengths. Width1024 after scatter is tile-aligned;
neither composite fallback predicate fires. Inputs are much larger than native
RS's512-KiB direct-path cap. The reduction therefore uses the ring factory with
two links; its output is `[1,1,S,1024]` in DRAM. That ring factory is shared with
the experimental async RS family. Native ownership creates its own semaphores
and performs synchronization when building the workload.

Native all-gather receives scatter dimension3 and DRAM output configuration;
it derives topology/links from the active mesh instead of accepting the
all-reduce link choice. For two detected links, the checked-out Blackhole
heuristics choose unicast and eight workers per direction per link at both
lengths. These are BF16 **2048-byte** tile pages;4096-byte tiles apply to FP32.
The8192-byte packet contains four BF16 pages versus two for4352 bytes.

| AG quantity, assuming two detected links | S=2048 | S=1920 |
|---|---:|---:|
| Input pages per rank | 2048 | 1920 |
| Worker slices, links × workers | 16 | 16 |
| Pages per worker slice | 128 | 120 |
| Antipode half-slice pages | 64 | 60 |
| CB pages for initial full slice, four tiles/CB page | 32 | 30 |
| CB pages for relayed antipode half | 16 | 15 |
| Expected incoming tile count per direction | 192 | 180 |

The reader reserves/pushes one CB page for each batch it reads; the writer
waits/pops one CB page for each corresponding batch. Both full slices and
antipode halves divide four at both lengths. The reader's completion wait is
full-slice plus half-slice. Thus a simple partial-packet or uneven-worker-count
explanation is **not supported** at these shapes. Remaining protocol/timing
issues, native RS work partitioning, and fabric transport need measured boundary
evidence before deeper kernel diagnosis. This is not a complete proof of either
collective kernel.

## Ownership, aliases, and completion

- The active decode grid is never called in linear prefill: physical time is at
  least128, so `_linear` delegates to `_prefill_linear`. Removing unused
  `decode_weights` cannot directly change a selected prefill weight tensor.
  It does change allocation pressure and addresses, which can expose a latent
  stale-address or lifetime issue.
- Projection replacement explicitly deallocates old weights after allocating
  replacements. The packed subfields are proper slices. A/B constant slices
  are also proper subranges. Checked-out `SliceDeviceOperation` allocates fresh
  output unless an output buffer is supplied; this inspection found no input
  alias for those nonidentity slices.
- `_slice_owned` preserves ownership of a whole-input identity slice. This
  four-chunk input uses proper subranges. `_pad_dim` can alias existing tile
  padding, but the caller does not free the pre-pad slice separately. For the
  tail1857→1920, tile padding1888 must actually grow.
- `to_memory_config` and reshape may alias. `_linear` does not explicitly free
  its local/folded aliases before returning the collective result. Retained
  output chunks remain referenced until concat is enqueued. No immediate
  double-free is apparent in the active replicated path.
- C++ all-reduce explicitly deallocates the scattered intermediate after
  enqueuing AG; downstream correctness relies on normal dispatch/completion
  ordering and the AG completion protocol. Merely observing host return from
  `_linear` or `_block` is not a device fence. Native RS cache-miss construction
  adds a synchronization that cache-hit replay does not repeat.
- Readback is the first observable end-to-end boundary in the failing test.
  Internal capture that uses `to_torch` can change scheduling and hide a race.
  Prefer keeping device clones/references until normal end-of-forward, then
  reading them; compare with a synchronized version as a separate control.

## Focused verification and refutation plan

Run only under the root-owned hardware lane and existing recovery rules. Keep
every replica/PCC gate unchanged. Record source and exact effective configs for
each intervention separately.

1. **Quantify the original failure first.** Run only linear
   `test_long_context_pcc` from a fresh mesh. Save all four outputs before
   raising. For each rank record NaN/+Inf/-Inf counts, finite mismatch count,
   maximum absolute/relative finite delta, first/last mismatching token and
   channel, per2048-token block counts, and equality to every other rank.
   Compare raw BF16 bits separately from `torch.equal` to expose equal NaNs.
   Report each rank's HF PCC only when its values are finite. Also compare
   full-output readback with fixed small slices from the same live tensor.

2. **Locate before changing defaults.** Retain device snapshots/references at
   both post-all-reduce boundaries (`gdn_out`, `down_proj`), both residual adds,
   every completed chunk before trim, final trimmed chunk, and final concat.
   Include replicated input slices and first RMSNorm if the first residual
   diverges despite equal mixer reductions. Save the corresponding rank-local
   pre-reduction projections; these should differ between ranks and must be
   checked against their sum, not against one another. First capture only
   completed chunks to limit disturbance, then narrow the failing chunk.

3. **If chunks match and concat does not:** replace final collection in an
   isolated diagnostic with concat of still-physical chunks followed by one
   final slice to8001; all preceding chunks are aligned, so this avoids
   untilize/unpad concat. Check against host concat of the captured chunks,
   all four replicas, HF output, continuation/decode state, and a standalone
   replicated synthetic concat of `[2048,2048,2048,1857]`. This is a concrete
   Python-only repair candidate only if that boundary is proven faulty. It
   must preserve logical continuation behavior and input ownership.

4. **If post-reduction replicas differ:** isolate the two collective halves
   using captured local projections. Compare native RS output to the matching
   quarter of the host sum, then native AG output to host concatenation of
   those RS shards. Use exactly physical S2048 and1920, BF16 tile DRAM, two
   links, and8192 packets. Repeat without intermediate host fences. Test
   `links=1` to alter only native RS and separately change router packet size
   to4352. An explicit experimental one-link RS+AG is a broader contrast,
   because it changes AG, semaphore ownership and completion as well as links.
   Do not attribute a pass to one of those differences without isolating it.

5. **If replicas share nonfinites or local arithmetic fails:** inspect projected
   QKV/Z/A/B, convolution output, DeltaNet core/state, and gated norm output on
   the first failing block. Compare1920-row geometry to the2048-row geometry
   using identical logical1857 rows with neutral padding and restored state.
   Test sequence lengths1857,2048,6144,8000,8001,8064,8192 with the same activation
   source/offset;8000 has a tile-aligned1856-row remainder but still physical1920,
   so it separates unaligned trimming/concat from the1920 compute geometry.
   Use only the smallest informative subset before broadening.

6. **If the isolated case passes or instrumentation heals it:** run the exact
   original three-test order and compare a fresh isolated mesh. Separate
   retained-output lifetime from state reuse by keeping references/clones of
   completed chunks and by adding one synchronization per chunk as distinct
   diagnostics. Restore unused weight allocations without changing decode
   math only as an allocation-layout control; an apparent cure would not
   establish that redundant weights are a valid fix. Matching source-level
   program shapes does not exclude cached address/semaphore reuse failures.

The smallest useful first run is step1; the smallest useful second run records
all completed chunks before trim/concat. Those measurements determine whether
to spend effort on collective kernels, final data movement, nonfinite local
arithmetic, or cross-test state. Do not relax exact replica equality on the
basis of possible rounding: a correctly gathered RS shard should be copied
identically to every rank, and the replicated residual operations should
preserve that identity for finite inputs.

## Review of alternative explanations

- **Wrong TP head or weight order:** would harm the mathematical answer, but
  complete all-reduce would still replicate the resulting wrong sum. No
  unmatched head/weight partition was found. It does not explain finite
  replica differences by itself.
- **BF16 reduction association:** RS computes each output shard once and AG
  distributes it. Different shards may use different association, but all
  replicas of a given shard must receive the same value. This cannot justify
  weakening the equality assertion.
- **Page table / absolute position error:** layer zero is linear-attention;
  its failing prefill does not consume the paged full-attention cache or RoPE.
- **Decode32-core math regression:** the failure precedes decode and never
  enters that branch; only indirect allocation/timing effects remain plausible.
- **Oversized advertised context or recurrent-state capacity:** the failing
  allocation is16384 context/batch1 and much smaller than the preceding native
  context runs. No capacity rejection is reported. Tail-only native checking
  is still insufficient to dismiss earlier-chunk corruption.
- **Profiler sync warning as root cause:** the same device-zero real-time
  profiler timeout also appears during the preceding passing native decode
  test. It does not identify a corrupting operation. Treat it as context for
  order-sensitive controls, not proof of bad hardware or a model bug.

This report is intentionally inconclusive about the repair. It identifies
specific untested boundaries and reproducible controls; source inspection
alone does not warrant changing the baseline, C++ kernels, precision, or gates.
