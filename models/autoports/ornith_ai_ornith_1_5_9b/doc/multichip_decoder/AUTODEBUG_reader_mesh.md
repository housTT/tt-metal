# AutoDebug: multi-reader DRAM matmul on the TP4 mesh

Date: 2026-09-05. Inspection only; no device commands, TTNN imports, builds, or
implementation edits were performed. Scope of a subsequent repair is the model's
`tt/multichip_decoder.py` and stage tests/docs. C++ changes are outside this task.

## Finding

The observed exception is explained by the checked-out source and the current
runtime binaries. **There is no evidence that a missing mesh-coordinate overload
or stale binary causes this failure.** The DRAM-sharded matmul factory calls the
four-argument `IDevice*` overload, which deliberately rejects a non-unit mesh.
The five-argument mesh-coordinate overload exists in both source and binary but
is not called by this factory.

`num_workers_per_dram_bank=1` returns before this unsupported call. Counts 2 and
3 cannot be made legal on a non-unit mesh in this factory by changing activation
cores, K block size, output shard size, DRAM shard padding, or replicated versus
TP weights. A model-local reader-1 configuration is the supported direct
alternative. The parent's subsequent real-weight layer-0 and layer-3 smokes
verify that alternative for replicated residuals, 128-token prefill and one
decode step. Those results do not establish the remaining stage gates or a
performance improvement.

## Observations and provenance

- Original command: `timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0`.
- Evidence: `logs/initial_linear_v2.log`, its `.provenance.json`, and
  `.sources.json.gz`. The process exited 1, with a host `TT_FATAL` at
  `/work/tt_metal/impl/device/experimental/device.cpp:20`, after `DECODE baseline`.
- The archived test opened one `MeshShape(1,4)` with `FABRIC_1D_RING` and first
  ran a replicated `OptimizedDecoder`. Thus this failure occurred before
  constructing or executing the TP decoder. Archived source hashes for the
  optimized decoder, multichip decoder, and probe were recomputed and match the
  provenance JSON.
- The prefill output was read with `ttnn.to_torch` before decode. The exception
  occurs during decode program descriptor construction; it is not evidence of
  a stalled device kernel or CCL failure.
- Git HEAD during inspection and in the original provenance:
  `483920f536f64462ba6c3dc785c59b8ff002cdc1`. The optimized decoder has no diff
  against stage baseline `d085eb6d1abcc8b25f213fede6b68aa873d6cd6b`.
- The live probe has subsequently changed: it opens a 1x1 baseline, closes it,
  then opens the 1x4 TP mesh. This is a different test setup from the archived
  failure and preserves the optimized single-chip baseline's tuned readers.
  The parent's subsequent logs were inspected below; this investigator did
  not run the hardware commands.
- Existing working changes were the context contract plus untracked multichip
  implementation, probe, and stage documentation. They were not altered here.

## Causal chain checked against source

1. `tt/optimized_decoder.py:44` sets per-role readers, including
   `gdn_packed: cores=32, block_w=4, readers=3`. Setting only the dataclass's
   global `readers=1` does **not** override these role entries: `_role_config`
   at lines 189–192 gives role values precedence.
2. `OptimizedDecoder._linear`, lines 254–283, folds decode into one tile row,
   width-shards A into L1, supplies DRAM-width-sharded B, and explicitly passes
   `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with the role reader
   count. No automatic matmul choice can replace that explicit config here.
3. `ttnn/cpp/ttnn/operations/matmul/device/factory/`
   `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:937–979`
   obtains `IDevice* device = a.device()`. `Tensor::device()` returns the device
   of the mesh buffer (`ttnn/core/tensor/tensor.cpp:507–511`), so this is the
   1x4 `MeshDevice`, not a physical chip.
4. That factory's helper, lines 115–124, validates the reader count and
   Blackhole/NOC0 restrictions, then calls `get_dram_bank_reader_assignments`
   with this unchanged device pointer.
5. `device/utilities/matmul_utilities.cpp:404–419` has a reader-1 early return
   based on the primary DRAM reader assignment. For readers 2/3 it scans free
   worker candidates and calls
   `get_worker_noc_hop_distance(device, candidate, primary, noc)` at lines
   446–447. This is the four-argument overload.
6. `tt_metal/impl/device/experimental/device.cpp:14–22` rejects
   `mesh->num_devices() != 1` in exactly that overload. The separate overload
   at lines 45–73 requires a `MeshCoordinate` argument and resolves a physical
   device. C++ cannot select it from a four-argument call.

The factory has no mesh-coordinate parameter. Its descriptor is built through
the adapter's no-coordinate path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:607–615`). The otherwise
available `get_device_for_dram_banks(a, coord)` helper is not used by the
DRAM-sharded factory either.

The allocation/shape controls do not evade this chain: padding-only DRAM banks
are removed at factory lines 127–140, **after** reader assignment. If a core
grid excluded every secondary candidate, it would produce “No free DRAM reader”
instead of a usable multi-reader program. The public program config exposes no
reader-coordinate override (`device/config/matmul_program_config_types.hpp:75–80`).

The selected dataflow kernels would be
`reader_bmm_tile_layout_in0_sender_dram_sharded.cpp` and
`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp`, with
`bmm_large_block_zm_fused_bias_activation.cpp` for compute (factory lines
429–495). This exception precedes their descriptor setup and dispatch; changing
their numeric policy is not an intervention for the reported failure.

## Exact failing geometry and clean control

The full replicated GDN projection is `[K,N]=[4096,8256]`: QKV8192, A32, B32.
On the recorded 8-bank Blackhole geometry:

| Parameter | Original first failing call | Reader-1 control |
|---|---:|---:|
| Mesh devices | 4 | 4 |
| Logical A | `[1,1,4096]` | same |
| Padded A | `[1,32,4096]` | same |
| A storage cores / shard | 32 / `[32,128]` | same |
| K tiles / K block | 128 / 4 | same |
| Readers per bank | 3 | 1 |
| B shard `[K,width]` | `[4096,1056]` | same |
| B width tiles per bank | 33 | 33 |
| Output `per_core_N` | 9 | 9 |
| Reader assignment | reaches unsupported hop call | early return |

For R3, the intended 24 readers each receive 11 N tiles; for R1, eight readers
each receive 33 N tiles. There are 32 K blocks, one per A storage core. The
input tile count, exact K sharding, and K block divisibility already satisfy
the checks at `matmul_device_operation.cpp:1340–1365,2518–2527`.

This particular R3-to-R1 A/B can reuse the identical DRAM shard shape, so a pass
cannot be explained by changing weight padding. Other roles/counts can require
repacking; compute width as `ceil(N/(32*banks*readers))*32*readers` at setup.

## Runtime binary adjudication

Read-only `nm`, `readelf`, `objdump`, and SHA256 checks found:

- `build` resolves to `build_Release`; CMake records `Release`, project version
  `0.79.0`, and source root `/work`. The `/work` assertion path is consistent
  with this build root, not proof of loading a different source tree.
- `libtt_metal.so` exports both overloads, at `0xbcba90` (four arguments) and
  `0xbcbce0` (five arguments).
- `_ttnncpp.so` imports only the four-argument hop-distance symbol. Its reader
  assignment function calls that PLT symbol at `0x1093071`; the source and the
  logged stack therefore agree on the relevant executable call boundary.
- `_ttnn.so` has RUNPATH `$ORIGIN/build/lib:$ORIGIN/../../build/lib:$ORIGIN`;
  the failure stack names this checkout's `build/lib/_ttnncpp.so`.
- ELF build IDs: `_ttnncpp.so` `ff9e9a30660d3bd63837c467b1135d1bbe1ecf03`;
  `libtt_metal.so` `7db548842b1b36dd0499e7d6684ddf3cdf07c1a1`.

| Current artifact | SHA256 |
|---|---|
| `build/lib/_ttnncpp.so` | `ced5beb1350f68b201e2f8cfb73f36ec8ef5a09c432e09fc88f8d31fac743507` |
| `build/lib/libtt_metal.so` | `530c6803ca7c3bdb0d3e17e091a50e9bf492c89e55feffc031c45afe41d634a9` |
| `ttnn/ttnn/_ttnn.so` | `5ad3718294e6db542832e5389c6617eb4c64ff5ab020e13eed15601172adb587` |

These hashes identify the inspected binaries. The original provenance did not
capture their hashes, so this is not a claim of byte-for-byte original-run
binary provenance or a complete match to every current source file. No rebuild
is needed to explain the observed overload selection.

## Exact supported decoder-local alternative

Keep the original optimized baseline on a true 1x1 mesh. For the TP4 decoder,
the existing local config is the appropriate starting candidate:

```python
MeshConfig(local=DecoderConfig(
    cores=8, block_w=4, readers=1, role_configs={}, conv_chunk=512,
))
```

Retain the explicit DRAM-sharded matmul family and re-upload each TP-local B
with its reader-1 shard shape. For ordinary 32x32 tiles, a single folded tile
row, eight A storage cores, and K block 4, the local roles lower as follows:

| Local role | K,N | A shard | B shard | `per_core_N` |
|---|---|---|---|---:|
| GDN packed QKV/A/B | 4096,2112 | 32,512 | 4096,288 | 9 |
| GDN Z | 4096,1024 | 32,512 | 4096,128 | 4 |
| GDN out / attention O | 1024,4096 | 32,128 | 1024,512 | 16 |
| MLP gate/up | 4096,3072 | 32,512 | 4096,384 | 12 |
| MLP down | 3072,4096 | 32,384 | 3072,512 | 16 |
| Full-attention Q/K/V/gate | 4096,2560 | 32,512 | 4096,320 | 10 |

Each K shard is an exact divisor of K and has 4, 12, or 16 tiles, divisible by
the selected K block 4. B and output are width sharded; A/B use row-major shard
orientation. Output `per_core_N` describes output storage and is not a way to
request a different reader count.

Counts 2/3 remain usable on actual unit meshes subject to their existing
Blackhole, NOC0, shard, and resource constraints. The nightly reader-count test
uses a `device` fixture, not a four-device mesh
(`tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py:261–306`),
so its coverage does not prove native mesh support.

Do not try to obtain a unit-device execution path by looping over
`get_device_tensors`: this API creates coordinate-restricted views sharing the
same mesh tensor (`ttnn/core/distributed/api.cpp:77–83`,
`ttnn/core/tensor/storage.cpp:134–136`). They retain the 1x4 device pointer.
Creating actual unit submeshes and rebinding/rejoining allocations would be a
different execution design with trace/CCL/lifetime implications, not a legal
layout adjustment to the existing native mesh matmul. No such design is proven
or recommended by this report.

## Hypotheses and focused follow-up experiments

| Hypothesis | Inspection result | Focused runtime prediction/control |
|---|---|---|
| A stale library lacks the mesh overload | Refuted at the relevant symbol/call boundary; both overloads are present | No rebuild experiment is justified by this finding |
| Native non-unit mesh + R2/R3 selects the unit-only API | Verified by source, stack, symbol import, and call instruction | The exact first projection on 1x4 fails for R2/R3; R1 passes this boundary |
| Changing padding/cores/blocks can legalize R2/R3 in this factory | Refuted: reader assignment occurs first and has no bypass config | Do not spend a broad geometry sweep retrying this structural limitation |
| TP-local reader1 removes the original blocker | Verified for both layer-kind short smokes in the parent's logs below | Remaining trace, continuity, and performance stage gates are separate |

### Parent's completed runtime verification

The parent executed the two commands in step 3 below. I inspected their logs,
provenance, and archived sources, recomputing archive, log, and relevant source
hashes. Both returned 0. The archived setup uses a genuine 1x1 optimized
baseline followed by TP4 with `cores=8, block_w=4, readers=1, role_configs={}`
and replicated residuals.

| Evidence | Prefill PCC, every rank | Decode PCC, every rank |
|---|---:|---:|
| `logs/initial_linear_v3.log`, layer 0 | 0.9999763147100362 | 0.9999771173573931 |
| `logs/initial_full.log`, layer 3 | 0.9999829421299804 | 0.9999720257518367 |

Both use source archive SHA256
`b56a43efc19c469e947faf02d1535a1086ab8aae6b85610d08a675a34751751a`.
Log SHA256 values are
`7aee75367b35eb7968bec353cd6b76bd9f6b94f732830b6fc8cb39823ec01341`
and `5880f22b0ed8fc6f7f5326d1c6e51241ac9b84ad9b2ba31cbf199fe544515da9`,
respectively. These validate removal of this blocker under the archived smoke
conditions. They do not validate later sharded-residual experiments; separate
NaNs there need their own diagnosis.

Run hardware experiments only through the parent's serialized device workflow:

1. If a narrow regression or explicit reader-count adjudication is needed,
   add a stage test for the original GDN packed projection, holding
   K4096/N8256, A32-core shard, block4, B `[4096,1056]`, BFP4/LoFi, and input
   fixed. Compare R1 versus R3 on 1x4; read R1 output to host and compare to the
   same uploaded/quantized interleaved-B control. An R3 unit-mesh control
   distinguishes the mesh restriction from reader/resource issues. An R2 run
   uses `[4096,1088]` B shards and should fail at the same hop call on 1x4.
2. When tuning new role geometries, probe the TP-local shapes in the table
   with R1 and per-rank
   weights, preserving dtype/fidelity and fused activation. Compare every
   rank to the corresponding local Torch/interleaved projection. This checks
   actual TP mapping separately from the already diagnosed reader failure.
3. Completed by the parent for the archived smoke source; rerun after relevant
   implementation changes, retaining the true 1x1 baseline:

   ```bash
   timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0
   timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3
   ```

   Archive each source snapshot and log. Both prefill and decode must satisfy
   the existing per-rank PCC checks. A later exception or accuracy failure is
   new evidence; reader1 removes this call boundary, not every possible TP bug.
4. After correctness, use the stage's required watcher, trace, continuity,
   and measured performance checks. Measure native TP4 R1 against the original
   tuned single-chip baseline; do not call reader1 faster or retain a faster
   configuration based only on this inspection.

Final status: the original host failure is diagnosed and the reader-1
alternative passes the parent's two real-weight layer smokes. R2/R3 remain a
native-mesh limitation of this factory under the allowed intervention scope.
No implementation change or hardware command was performed by this
investigator. Broader stage completion and sharded-residual behavior are
outside this finding.
