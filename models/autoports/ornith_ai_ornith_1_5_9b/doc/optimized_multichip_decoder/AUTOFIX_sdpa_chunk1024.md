# AutoFix: SDPA chunk1024 live L1 allocation clash

**Adjudication:** chunk1024 runs correctly after reducing live intermediates
and capping reduction workers at eight, but is slower in the measured
production family. Retain chunk256 and the existing production buffer
lifetimes. The rejection follows an adapted successful run and matched
controls; it is not based on the first allocation error. No native change is
needed.

## Starting evidence

`remaining_op_matrix_v2.json` tested layer3 at position2048 on four Blackhole
chips on physical P300c boards, using a native 1x4 ring. The original command
was:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048 --layer 3 --variant production_candidate --local-config '{"sdpa_chunk":1024}'
```

[Its provenance](logs/sdpa_chunk1024_layer3.provenance.json) reports return code
1. The log reaches `SdpaDecodeDeviceOperation`, then
`ProgramImpl::validate_circular_buffer_region` rejects static circular buffers
ending at **1,526,144** against a live L1 allocation starting at **1,455,104**.
The overlap is **71,040 bytes**. This is a compile/allocation failure before
SDPA dispatch, rather than a device stall or numerical failure. Log SHA256:
`77f5c462ca9a9fdafaf84569e40256a968ecb39ec3532aad78f150f893f7e748`.
Archived source SHA256:
`05c9d262689bba5ee09278d41107279b76fc1873a0eb8f7ddfb3783204c234b2`.

The original run used BFP8 QKVG at 32 input cores/block4/reader2. The requested
retry uses the parent's newer `production_qkv4_c8` family, with BFP4 QKVG at
8 input cores/block16/reader2. Neither projection choice changes the BF16
SDPA query or the BFP8 KV-cache contract. Chunk0/128/512 succeeded in the older
family; their timings are not a matched control for the newer family.

## Hypothesis and source evidence

Hypothesis: live attention intermediates unnecessarily constrain the static
SDPA allocation. Completing their lifetimes before SDPA and retaining Q/gate
in DRAM may leave enough L1 space to measure chunk1024.

- `tt/optimized_decoder.py:509–559` retains the packed QKVG tensor, the QKV
  slice, updated K, and V as local variables through SDPA. K and V are no
  longer needed after `paged_fused_update_cache`; packed/QKV are no longer
  needed after the slices/head split. Borrowed caller input remains live.
- `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/`
  `sdpa_decode_device_operation.cpp:82–105` permits either height-sharded Q or
  interleaved DRAM Q. DRAM output is also supported. The original high-level
  call already defaults its output memory configuration; specifying DRAM
  output alone does not address the retained input allocations.
- `sdpa_decode_program_factory.cpp:421–427` double-buffers K and V with sizes
  proportional to K chunk. With chunk1024/head256, each cache CB has
  `32 * 8 * 2 = 512` BFP8 tiles, or 557,056 bytes. Both cache CBs alone total
  1,114,112 bytes. Changing Q/gate placement does not shrink these CBs.
- Query/global-CB aliasing is only selected for the separate MLA path
  (`sdpa_decode_program_factory.cpp:133–151,540`). This model's ordinary paged
  attention allocates its Q CB for both supported query memory layouts.
- The allocator explicitly checks the lowest live L1 allocation against the
  CB region end (`tt_metal/impl/program/program.cpp:1931–1939`). Thus earlier
  releases target the observed failure boundary directly.

The live allocation address does not identify its owning tensor. Source
establishes releasable allocations and the collision, but cannot prove that
this combination supplies enough headroom under the parent's exact trace
allocation sequence. The focused runtime retry is required.

## Opt-in candidate

[tests/multichip_sdpa_candidate.py](../../tests/multichip_sdpa_candidate.py)
defines `SdpaLowLiveBuffers`, a mixin applied before the selected production
candidate. It copies the existing attention decode sequence and changes only
temporary memory placement/lifetimes:

- Slice gate directly into DRAM and release packed projection after slicing.
- Release the QKV slice after head creation and owned norm inputs after use.
- Preserve existing RoPE and disjoint K/V cache-update shard geometry; release
  K and V immediately after the update has been enqueued.
- Move Q to DRAM before SDPA, release any replaced L1 allocation, and explicitly
  retain SDPA output in DRAM. Already-DRAM Q from prime-batch RoPE retains its
  allocation rather than freeing an alias.
- Leave caller input, cache tensors, page table and position tensors owned by
  their caller. Preserve scale, fidelity, exponential mode, cache update,
  output gating, and collective behavior.

Chunk size and grid remain `self.optimization.sdpa_chunk` and `sdpa_grid`, so
256 and 1024 use the same adapted path. The native default
`max_cores_per_head_batch=16` is explicitly preserved for this first experiment.
No production default or shared registry was edited by this investigator.

Parent integration snippet, after the existing production candidate registry
is constructed:

```python
from .multichip_sdpa_candidate import SdpaLowLiveBuffers

CANDIDATES["production_qkv4_c8_sdpa_dram"] = type(
    "ProductionQkv4C8SdpaDram",
    (SdpaLowLiveBuffers, CANDIDATES["production_qkv4_c8"]),
    {},
)
```

Run under the parent's hardware owner and stage provenance wrapper:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048 --layer 3 --variant production_qkv4_c8_sdpa_dram --local-config '{"sdpa_chunk":1024}'
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048 --layer 3 --variant production_qkv4_c8_sdpa_dram --local-config '{"sdpa_chunk":256}'
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --length 2048 --layer 3 --variant production_qkv4_c8 --local-config '{"sdpa_chunk":256}'
```

The second command isolates chunk choice within the adapted path; the third
measures the adaptation cost against the selected family. Keep the existing
all-rank PCC, eager/trace/repeated-trace and trace-after-eager checks. A passing
batch1 control needs the existing batch4/32, continuation and watcher gates
before any integration; this candidate adds earlier frees around cache
updates and therefore needs trace-reuse evidence.

## Bounded fallback: reduction scratch

If the same live-memory clash remains, the native program already exposes a
second buffer control. `SDPAProgramConfig.max_cores_per_head_batch` defaults
to 16 in `sdpa_config.hpp:20` and its Python binding at
`transformer_nanobind.cpp:42`. For this batch1/local-KV-head1 call, the factory
uses 16 cores per head and allocates reduction CB19 as
`(8 output tiles + 2 statistics tiles) * (16 - 1) * 1024 = 153,600` bytes.
The half-tile statistics format is selected for four logical BF16 query heads.

A cap of eight reduces CB19 to 71,680 bytes, saving **81,920 bytes**. Holding
the other geometry fixed, the static end becomes **1,444,224**, below the
original live-buffer address by 10,880 bytes. This is a source-derived
prediction, not an observed successful allocation. It reduces the reduction
scratch without changing K chunk, cache precision, or logical attention work.

The existing mixin class attribute permits this distinct follow-up:

```python
CANDIDATES["production_qkv4_c8_sdpa_dram8"] = type(
    "ProductionQkv4C8SdpaDram8",
    (SdpaLowLiveBuffers, CANDIDATES["production_qkv4_c8"]),
    {"sdpa_max_cores_per_head_batch": 8},
)
```

Use the same commands with that variant and compare 1024/256 at the same cap.
This is a second isolated experiment if needed, rather than grounds to reject
chunk1024 after its first allocator error.

## Verification status

Python `py_compile` and all applicable repository pre-commit hooks pass.
AST-only checks confirm that packed/QKV/K/V are released before SDPA, borrowed
state is not released, and chunk/reduction parameters remain configurable.
Candidate SHA256:
`a77b26c7448345dbce83334680e4b52356ebd2698cd9cff571e9c8f56fe900c3`.
No TTNN import, device command, hardware test, or build was executed by this
investigator. The parent's runtime adjudication follows. This is a Python-only
candidate; no C++ build is required.

## Parent runtime adjudication

The first lifetime/DRAM-Q adaptation at cap16/chunk1024 progressed beyond the
original failure. Eager decode, trace capture and initial replay completed;
the log reports four-rank `trace_maxdiff=[0,0,0,0]`. It then failed on the
restored **eager decode while the captured trace/output remained allocated**:
the stack identifies `multichip_probe.py:232`, `repeated = decode()`, after
the first replay. This corrects the earlier shorthand description of a
trace-capture failure.

The new lowest live L1 address is **1,514,496**, with the same static end
**1,526,144**. The remaining overlap is **11,648 bytes**, compared with the
original 71,040-byte overlap. The adaptation improved observed headroom but
did not satisfy the complete trace/eager reuse contract at cap16/chunk1024.
[Failure provenance](logs/sdpa_low_live_chunk1024_layer3.provenance.json)
reports return code 1; log SHA256:
`0bc5bac1d3b2a1df37067a04a402c58c5686900d6602843f89f5bb64f9389d97`.

The parent then executed all three commands in
[sdpa_adapted_matrix_v2.json](sdpa_adapted_matrix_v2.json). Cap8/chunk1024,
cap16/chunk256 and cap8/chunk256 all pass. Their actual variant names are
`production_qkv4_c8_sdpa_dram` and `production_qkv4_c8_sdpa_dram8`; the latter
sets the reduction-worker cap to eight. All three results pass all-rank
prefill/decode PCC, repeated eager equality, initial trace equality, and the
second trace replay after the restored eager call. Minimum prefill PCC is
`0.9999772466069937`; minimum decode PCC among the three is
`0.9999779426493981`. The common archived Python-source SHA256 is
`d818ff2c7cb15a343e7a4f5616e34169af71dd877c3af6d789e825012b15f243`.

The following whole-layer warmed decode values were checked against
[candidate_measurements.json](candidate_measurements.json), each run's
provenance and the SHA256 of its decompressed log:

| Production QKV4 family configuration | Decode ms | Difference from selected default | Evidence |
| --- | ---: | ---: | --- |
| Existing lifetimes, cap16, chunk256 | 0.266111874 | baseline | [selected default](logs/finalfamily_default_replicated_layer3.provenance.json) |
| Lower live buffers, DRAM Q/gate, cap16, chunk256 | 0.267339281 | +1.227 µs | [adapted 256 control](logs/sdpa_low_live_cap16_chunk256_layer3.provenance.json) |
| Lower live buffers, DRAM Q/gate, cap8, chunk256 | 0.271272438 | +5.161 µs | [cap8 256 control](logs/sdpa_low_live_cap8_chunk256_layer3.provenance.json) |
| Lower live buffers, DRAM Q/gate, cap8, chunk1024 | 0.276570376 | +10.459 µs | [adapted 1024 result](logs/sdpa_low_live_cap8_chunk1024_layer3.provenance.json) |

At matched cap8 and buffer layout, chunk1024 is 5.298 µs slower than chunk256.
The lifetime/DRAM adaptation at the original cap16/chunk256 also does not beat
the selected default. The source-backed scratch reduction therefore supplies
a valid experiment for chunk1024, while the measured configuration is not
selected for production. Preserve the opt-in candidate and evidence for
reproduction; retain the existing chunk256 production path. No claim is made
that chunk1024 is unsupported or slower for every context, batch, or model.

This adjudication changes only this report. Applicable documentation
pre-commit checks pass; the investigator performed no hardware operation.
