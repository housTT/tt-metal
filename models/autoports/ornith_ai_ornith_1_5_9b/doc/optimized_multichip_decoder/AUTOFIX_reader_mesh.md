# AutoFix: native multi-reader DRAM matmul on the TP4 mesh

Date: 2026-09-05. Starting HEAD: `65abe7f69dbf128012108ef99262bf337f7cdc70`.
Target: Ornith-1.5-9B, four Blackhole chips on physical P300c boards, 1x4 ring.

## Status and scope

The reported cause is verified independently against source, the archived
exception, and current binary symbols/call instructions. A minimal native C++
fix was prepared in the isolated worktree `/tmp/ornith-reader-mesh-autofix`.
It passes four host C++ syntax checks, including the matmul operation's template
and factory-concept instantiation. The parent subsequently integrated, built,
installed and ran the patch: **reader 2 passes a real-weight TP4 layer-3 QKVG
control with exact eager/trace replay.** Subsequent adapted-output probes pass
all six layer-0 roles with readers 1/2/3, including changed-input and fresh-buffer
checks. Full-layer reader-policy integration, remaining layer-3 roles, batch
and watcher coverage are separate. See the measured follow-up below; no
general whole-layer performance benefit is claimed from projection timing.

Patch: [`reader_mesh.patch`](reader_mesh.patch), five existing files,
42 insertions and 11 deletions. SHA256:
`10b01e5e5aff4c7007c412547c5143ef07a65991d3068ae1db496adb6686a395`.

The parent initially authorized inspection only, then authorized implementing
this native candidate in an isolated worktree, running the required build
wrapper to capture its environment failure, and checking host/source contracts.
The earlier report's “C++ out of scope” statement does not apply to this stage.
This investigation did not import TTNN, open/list/reset devices, run a device
test, compile/link a library, install dependencies, or change a live runtime.

## Starting evidence and verified cause

Starting report: `../multichip_decoder/AUTODEBUG_reader_mesh.md`.
Original command:

```bash
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0
```

The archived `../multichip_decoder/logs/initial_linear_v2.log:26,48–53`
records `get_worker_noc_hop_distance() is only supported on unit MeshDevice`
at `tt_metal/impl/device/experimental/device.cpp:20`. That archived probe first
executed a replicated optimized decoder on a 1x4 mesh. The present probe
instead uses a true unit-mesh baseline and then TP4. This distinction does not
remove the TP4 reader-2/3 limitation in the shared native factory.

Current source path, before the isolated patch:

1. `Tensor::device()` (`ttnn/core/tensor/tensor.cpp:507–511`) returns the
   owning `MeshBuffer`'s device, including a multi-device mesh.
2. The DRAM-sharded factory (`ttnn/cpp/ttnn/operations/matmul/device/factory/`
   `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:937–979`)
   takes `a.device()` and has no mesh-coordinate parameter. Its helper calls
   `get_dram_bank_reader_assignments` at lines 123–124.
3. Reader assignment (`device/utilities/matmul_utilities.cpp:404–447`) returns
   before hop scoring for reader 1. Readers 2/3 call the four-argument
   `IDevice*` overload. Padding-only banks are pruned afterward in the factory,
   so padding/cores/K-block adjustments cannot bypass this boundary.
4. `tt_metal/impl/device/experimental/device.cpp:14–22` explicitly rejects
   a non-unit mesh in that overload. Its separate coordinate overload at
   lines 45–73 resolves the selected device and supports this query.
5. The mesh adapter already supports coordinate-dependent descriptors:
   `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:424–449,607–615` detects
   a fourth `optional<MeshCoordinate>` argument and builds one program per
   coordinate inside the same mesh workload. The current factory does not
   opt into this mechanism.

There is a second placement detail worth fixing with the hop call:
`MeshDevice::get_optimal_dram_bank_to_logical_worker_assignment(noc)` returns
only the reference device's assignment (`tt_metal/distributed/mesh_device.cpp:1299–1300`).
Its coordinate overload at 1302–1326 returns that chip's bank-ID-to-worker map.
The header explicitly warns that the old API can misplace readers on a
heterogeneously harvested mesh. Using a coordinate only for hop scoring would
leave primary and secondary placement based on different chips.

### Binary evidence recomputed in this investigation

Read-only commands included `nm -D`, `readelf -n`, `objdump -d`, and `sha256sum`.

| Artifact | SHA256 |
| --- | --- |
| `build/lib/_ttnncpp.so` | `ced5beb1350f68b201e2f8cfb73f36ec8ef5a09c432e09fc88f8d31fac743507` |
| `build/lib/libtt_metal.so` | `530c6803ca7c3bdb0d3e17e091a50e9bf492c89e55feffc031c45afe41d634a9` |
| `ttnn/ttnn/_ttnn.so` | `5ad3718294e6db542832e5389c6617eb4c64ff5ab020e13eed15601172adb587` |

`libtt_metal.so` exports both hop overloads: four arguments at `0xbcba90`,
coordinate overload at `0xbcbce0`. `_ttnncpp.so` imports only the four-argument
overload. Its reader-assignment function starts at `0x1092ad0` and calls that
PLT entry at `0x1093071`. Build IDs are
`ff9e9a30660d3bd63837c467b1135d1bbe1ecf03` (`_ttnncpp`) and
`7db548842b1b36dd0499e7d6684ddf3cdf07c1a1` (`libtt_metal`). These match the
previous inspection's artifacts; they do not retroactively establish original
failure-run binary provenance.

Verdicts: a missing mesh overload or simple stale-library explanation is
refuted at this call boundary. Native multi-reader assignment selecting the
unit-only overload is verified. Changing model reader geometry alone cannot
fix that call boundary.

## Native candidate and host experiments

The patch:

- Adds `optional<MeshCoordinate>` as the factory's fourth C++ argument and
  passes it through to reader assignment. The existing mesh adapter then
  supplies each coordinate and retains native mesh workload/trace execution.
- Uses that coordinate for both primary-reader placement and NoC hop scoring.
  It converts the unordered primary-reader map in ascending numeric bank-ID
  order, preserving the weight shard ordering and deterministic tie breaking.
- Preserves unit-mesh/direct reader-1 behavior when no coordinate was supplied.
  An explicit descriptor request for readers 2/3 on a non-unit mesh without a
  coordinate fails with a specific contract message.
- Preserves the Python factory's existing fourth `core_range_set` argument
  and appends optional `mesh_dispatch_coordinate` as the fifth argument.
  The C++ factory has one signature: its internal fourth positional argument
  changes type, and all discovered source callers are updated.

No kernels, numerical settings, sharding formulas, CCL policy, or mesh adapter
implementation change. Physical-to-virtual worker coordinate checks elsewhere
in the factory remain intact. Remote-coordinate queries retain the existing
runtime API's best-effort homogeneous-topology fallback; this patch does not
claim general heterogeneous multi-host support.

Host experiments:

| Experiment | Result and interpretation |
| --- | --- |
| `git diff --check` in isolated worktree | Pass |
| `clang-format -i` on the five changed files | Completed; final diff inspected |
| `.github/scripts/copilot-build.sh` in isolated worktree | Exit 1 before building: Docker unavailable; see `reader_host_checks/copilot_build.log` |
| `python /tmp/ornith-reader-mesh-autofix/reader_host_syntax.py` | Final run passes utility, DRAM factory, matmul device operation, and nanobind translation units |

The syntax script reuses the existing build's compiler defines/include search
paths, adds isolated source/header paths first, removes PCH and object/dependency
output flags, and invokes `/usr/bin/clang++-20 -fsyntax-only`. It reads existing
generated/dependency headers but writes no object, library, shared build output,
or PCH. Exact argv and statuses are in [`reader_syntax_results.json`](reader_syntax_results.json);
empty successful compiler logs are under `reader_host_checks/`.

Two errors in initial drafts were caught and corrected before exporting the
patch: an unqualified `CoreCoord`, and an overloaded factory signature rejected
by `HasDescriptorFactory`'s `&T::create_descriptor` requirement
(`ttnn/api/ttnn/operation_concepts.hpp`). The final operation translation-unit
check verifies the single-signature factory satisfies the actual concepts.
These are host compile-contract checks, not the repository-required linked build.

## Build prerequisites and proposed build commands

Observed environment:

- `docker` is absent from PATH and `/usr/bin`/`/usr/local/bin`; no
  `/var/run/docker.sock` exists. Both Garage credential variables and
  `TT_CI_BUILD_IMAGE` are absent. No credential values were printed.
- `/work` is the same checkout, and `/.dockerenv` exists. This is already a
  container-like environment with `/usr/bin/clang++-20`, LLVM 20 archiver/scanner,
  CMake, Ninja, and ccache present. No dependency installation is necessary
  for the completed host syntax checks.
- The existing Release CMake tree uses `/work`, clang++-20, Ninja, and
  `/work/.cpmcache`; the CPM cache occupies 3.7 GB and the agent cache directory
  138 MB. The default `ccache --show-stats` reports an empty effective cache;
  do not claim the wrapper's warm remote cache is available.
- A read-only `ninja -C /work/build_Release -n ttnncpp ttnn` exits 0 but reports
  pending glob checking and CMake regeneration. An incremental library build
  may therefore reconfigure before compiling.
- The isolated worktree has no initialized submodule content or build tree.
  A fully independent linked build there is **not demonstrated**; do not
  present the syntax check as proof that isolated CMake configuration will
  complete offline. Existing source/dependency/generated-header reuse makes
  isolated translation-unit checking possible, as executed above.

Required wrapper command remains:

```bash
.github/scripts/copilot-build.sh
```

It has been attempted and its environment failure recorded. If the parent uses
the toolchain already present after closing its hardware workload, the narrow
existing-tree build and installation commands are:

```bash
git apply --check models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_mesh.patch
git apply models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_mesh.patch
cmake --build build_Release --target ttnncpp ttnn -j 2
cmake --install build_Release --component tar
cmake --install build_Release --component tt_pybinds
```

Those fallback build/install commands were not run by this investigator; the
parent subsequently ran them successfully, as recorded below. They are separate
from the mandated wrapper. The `ttnncpp`/`ttnn` targets build into
`build_Release/ttnn/`; runtime Python uses the installed `build/lib/_ttnncpp.so`
and `ttnn/ttnn/_ttnn.so`, so checking only a newly built target is insufficient.
Recompute installed binary hashes and verify `_ttnncpp.so` imports the
coordinate overload before judging device results. C++ source hashes and patch
must be archived alongside the stage recorder, which currently archives model
Python sources but does not automatically capture this runtime C++ change.

### Parent's completed build and reader-2 control

The following parent-produced provenance and logs were inspected directly:

| Log prefix under `logs/` | Command | Exit |
| --- | --- | --- |
| `reader_copilot_build` | `.github/scripts/copilot-build.sh --build-ttnn-tests` | 1, Docker unavailable |
| `reader_native_build` | `timeout 1200 cmake --build build_Release --target ttnncpp ttnn -j 2` | 0 |
| `reader_install_cpp` | `cmake --install build_Release --component tar` | 0 |
| `reader_install_bindings` | `cmake --install build_Release --component tt_pybinds` | 0 |
| `reader2_qkvg_layer3` | command below | 0 |

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --length 2048 --role-configs '{"qkvg":{"cores":32,"block_w":4,"readers":2}}'
```

The QKVG control uses the normal TP4 decoder and BFP8 decode QKVG, retaining
the other default role settings. Every residual rank matches the unit baseline
with prefill PCC `0.9999772466069937` and decode PCC `0.999695987736785`.
Trace and restored eager outputs are exactly equal; state PCC is
`[0.9999980445697845, 0.9999975275009155]`. Observed whole-layer traced decode
is `0.27809531457023695 ms`. This proves the old multi-reader host exception
is removed for a real TP4 role; it does not validate other reader geometries
or establish a stable performance win.

Control log SHA256:
`5570187477130f0896861c2105aa5c4a26e0c2dd75bdd99faf4749e4244b3c9a`.
Its Python-source archive SHA256:
`734a63283593f05530a9d9b20859d2f2d86d44e2a8c2c0ea0381a84107c7c40b`.
The native build log SHA256 is
`32290d3d9266c50530c3985d3c67b55d6023ec66678c3218d42b6511e13b74e8`.

After install, read-only checks confirmed both hop overloads are imported by
`_ttnncpp.so`. Installed hashes are
`8a5f138ae0c043c6984042290c113964f7b8c864728d10e10de9c56047e92cf7`
for `_ttnncpp.so`, and
`f3e5f0c846100fac5cbf38be7267338de4539e9628ed31dbd034d80208a6abe2`
for `ttnn/ttnn/_ttnn.so`; `libtt_metal.so` remains unchanged. Exact C++ source
and binary metadata is in [`reader_native_artifacts.json`](reader_native_artifacts.json).
All five live C++ files equal the isolated patch byte-for-byte. Their archived
contents are in `reader_native_sources.json.gz`, SHA256
`90c76e316410d2b7ac47319ca138b4d910ed7d5e61d71719f6e802f8ad54095e`.

## Model-local adaptation adjudication

No existing model configuration makes native readers 2/3 legal on TP4.
`get_device_tensors` also fails as a workaround: it restricts coordinates while
sharing the parent mesh buffer (`ttnn/core/distributed/api.cpp:77–83` and
`ttnn/core/tensor/storage.cpp:134–136`).

A model-local descriptor adapter is technically plausible with current Python
APIs, but unverified and less direct than the native patch:

1. The native DRAM factory is Python-bound at `matmul_nanobind.cpp:1308–1324`.
   Create a genuine unit submesh as a setup-only topology context and allocate
   a same-spec placeholder A on it. Call the bound factory with placeholder A,
   the parent-mesh B, and the parent-mesh output; descriptor construction uses
   A's device for topology and does not require B/output device equality in the
   bias-free model path.
2. Replace CB2's placeholder input backing with the real parent-mesh A backing,
   retaining its original core ranges/size. `cb_descriptor_from_sharded_tensor`
   plus `ttnn.experimental.copy_cb_backing` can perform this; the latter copies
   both `Buffer*` and `MeshTensor*` backing, unlike `set_buffer_from_cb`.
   Source inspection finds A embedded only as CB2 backing in this factory;
   B writer bindings and output CB6 can already reference parent tensors.
3. Supply a descriptor per coordinate to `ttnn.generic_op` using
   `MeshProgramDescriptor`. Its factory (`operations/generic/device/`
   `generic_op_program_factory.cpp:12–36`) dispatches per coordinate through
   the same mesh adapter. Matmul execution and trace remain on the parent mesh.

This would require explicit placeholder/descriptor lifetime, allocator, cache
rebinding, and trace experiments. `unit_mesh::disaggregate` in C++ really can
construct unit-mesh views at the same address, but has no discovered Python
binding; it is not equivalent to `get_device_tensors`. Neither submesh execution
loops nor Python monkeypatching the C++ hop function are recommended. The
native coordinate patch avoids the additional placeholder/submesh design.

## Focused hardware verification for the parent

All commands must use the parent's serialized device workflow, after building
and verifying installed artifacts. None were run by this investigator.

The newly authored durable narrow probe is
`tests/multichip_reader_probe.py`. It passed `python -m py_compile` and the
repository's applicable pre-commit hooks without a TTNN import. It captures
actual TP4 role inputs after real-weight prefill at 2048, directly partitions
raw HF weights, uploads each reader-padded matrix independently at the original
dtype, and measures readers 1/2/3 in alternating forward/reverse timing windows.
Every rank is checked against reader 1. It also checks unchanged replay, fresh
input/output allocations on a program-cache hit, and changed-input replay on
the parent mesh. Every window and any first failure leave a partial JSON file.

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 0 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_layer0.json
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 3 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_layer3.json
```

`--roles` narrows scope; aliases `gdn_z`, `gate`, `up`, and `down` are accepted.
`--cores` and `--block` accept an integer or a JSON map using canonical role
names. Default geometry is the current role configuration. `--readers` narrows
candidate counts but always includes reader 1 for comparison. Outputs include
actual program reader count, input/weight/output memory and shard geometry,
padding and active-bank counts, tile-row bytes per bank/reader, actual dtypes,
compute configuration, warmup/measured samples, and runtime/source hashes.
This is batch-1 projection coverage; whole-layer acceptance remains separate.

The existing real-weight probe accepts per-role reader settings and restores
state before comparing eager, repeated eager, and parent-mesh traced outputs.
It also compares TP4 against the genuine unit-mesh baseline. Start with a single
role, retaining the same cores, K block, dtype/fidelity and input for R1/R2/R3:

```bash
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --role-configs '{"qkvg":{"cores":32,"block_w":4,"readers":1}}'
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --role-configs '{"qkvg":{"cores":32,"block_w":4,"readers":2}}'
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 3 --role-configs '{"qkvg":{"cores":32,"block_w":4,"readers":3}}'
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --decode-grid null --role-configs '{"gdn_packed":{"cores":4,"block_w":32,"readers":2}}'
timeout 180 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe --layer 0 --decode-grid null --role-configs '{"gdn_packed":{"cores":4,"block_w":32,"readers":3}}'
```

For the layer-0 family, also record an otherwise identical R1
`--decode-grid null` control before interpreting a full-layer accuracy failure.
DRAM B shard width must be repacked per count; model setup already computes
`ceil(N / (32 * banks * readers)) * 32 * readers`. Verify printed model config
and actual selected program count rather than only the dataclass global default.
In particular, most current default TP4 roles use interleaved decode and do
not exercise readers unless `--decode-grid null` is selected.

Required assertions/evidence:

- Readers 2 and 3 progress beyond descriptor construction on actual TP4.
  Inspect per-coordinate primary and secondary assignments or descriptor
  writer runtime args (bank ID at index 3, reader index at 5) to show real
  multi-reader programs are constructed; no automatic reader-1 fallback.
- Compare every rank's projection to the corresponding TP-local quantized
  reference, then whole-layer real-weight PCC. A pass at the old hop boundary
  does not prove later resource, padding, or numeric behavior.
- Verify cache-hit calls with newly allocated input/output buffers and different
  inputs. Capture on the parent mesh, change input contents, replay, and compare
  to eager for the same state. The existing one-step probe proves restored-state
  equality, not the complete changed-input or batch-32 regression contract.
- Once a candidate is kept, run changed-input/page-table tracing, repeated
  state, batch boundaries including 32, and the normal minimal watcher check.
  The existing unit-device nightly `test_matmul_in1_dram_sharded_worker_counts`
  remains a regression control, not TP4 coverage.
- Record actual warmed role and whole-layer timing with the same policy,
  including layout transitions. More readers are not assumed faster.

Initial native-fix verdict: verified native host API bug; minimal native repair
built and installed by the parent, with one real TP4 reader-2 role passing
whole-layer and exact-trace checks. The following sections record subsequent
writer-layout adjudication and broader projection results.

## Follow-up: padding-only readers outlive output storage

The parent's first `reader_layer0.json` run stops at reader-3 GDN packed
descriptor construction with `curr_storage_core_idx < num_cores_written_back`
at factory line 815, reporting `Worker 6-2 has no storage area assigned`.
Reader-1 and reader-2 records show exact eager/trace output and all-rank PCC
effectively 1 before the stop; their timing and changed-input checks had not
yet run, so those records are correctly still marked `timing`, not `pass`.

Hypothesis: the small-reader-output branch assumes every reader starts within
the output allocation even when the final DRAM-bank readers produce only
padding. Source and a standalone host arithmetic probe reproduce the exact
failure:

- Logical N is 2112 columns, or 66 tiles. Reader 3 has eight active banks,
  three readers per bank, and three tiles per reader: 24 readers cover 72 tiles.
- Default output `per_core_N=17` creates `ceil(66/17)=4` storage shards,
  total capacity 68 tiles (`matmul_device_operation.cpp:2536–2547`).
- The writer branch at factory lines 814–848 advances storage position by
  three tiles per reader. Reader index 23 starts at tile 69, beyond all four
  output shards, and hits the assertion. This is a padding-only tail reader;
  all 66 logical output tiles already have storage.
- The other writer branch already supports a zero-count write-back when no
  storage remains (lines 852–904). The fixed writer-argument padding at
  906–913 explicitly documents this legitimate case. No change to that
  native code was made during this follow-up.

This failure is independent of mesh-coordinate placement. It is exposed only
after the original host API bug is fixed. It is not evidence of a stalled
kernel, CCL failure, missing logical output memory, or numerical instability.

The narrowest model-layout experiment keeps input cores/block and raw weight
padding unchanged but uses `per_core_N=18`, giving four output shards and 72
tiles of capacity. The host calculation predicts every reader starts in a
valid shard. The newly added `--per-core-n` option accepts a scalar or canonical
role map and applies the same output layout to all compared reader counts:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 0 --roles gdn_packed --per-core-n 18 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_gdn_packed_n18.json
```

The arithmetic probe is recorded in `reader_storage_geometry.json`:

| Role | N tiles | Default output tiles/shard | First failing R3 reader | Candidate output tiles/shard |
| --- | ---: | ---: | ---: | ---: |
| GDN packed | 66 | 17 | 23 | 18 |
| GDN Z | 32 | 8 | 16 | 9 |
| GDN out / attention O | 128 | 32 | 22 | 36 |
| MLP down | 128 | 16 | 22 | 18 |
| QKVG | 80 | 3 | none; other writer branch | 3 |

Gate/up N96 tiles already divides its default output and reader grids exactly.
If the narrow GDN packed experiment passes, source-backed full-role candidates
at unchanged input cores and K blocks are:

```bash
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 0 --per-core-n '{"gdn_packed":18,"gdn_z_epilogue":9,"gdn_out":36,"down_proj":18}' --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_layer0_output_padding.json
timeout 240 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 3 --per-core-n '{"o_proj":36,"down_proj":18}' --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_layer3_output_padding.json
```

At the time of this source analysis these were predictions. The parent
subsequently verified the layer-0 candidates, as recorded below.
The CLI option is confined to the new probe. Keeping this output-shard
adaptation in the actual decoder would need explicit role configuration
plumbing and full-layer/trace checks. A larger legal input grid is a second
control: GDN packed cores32/block4 gives `per_core_N=3`, selecting the already
zero-count-safe writer branch, but also changes input layout and K blocking.
The output-only experiment isolates the writer-capacity hypothesis more tightly.

### Corrected output storage verified on the parent

The narrow `reader_gdn_n18` run exited 0, with
`reader_gdn_packed_n18.json` reporting all three reader counts pass at unchanged
input cores4/block32 and output `per_core_N=18`. All-rank reader-1 PCC is
effectively 1, unchanged and changed-input trace replay are exact, and fresh
output addresses are distinct. JSON SHA256:
`ba160bde9ff6a5283fbb1e8b9d8923e90d457573a3d8e96a2ddc2c3c759929a2`.
Log SHA256:
`6a655cfafa4fa6cc1776bf416d77d8131b70ca69a3ac7871e068eff2196c8af2`.
This verifies the output-capacity adaptation and the padding-only reader
hypothesis; no further native writer change was required for this configuration.

The parent then executed:

```bash
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 0 --per-core-n '{"gdn_packed":18,"gdn_z_epilogue":9,"gdn_out":36,"down_proj":18}' --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_adapted_layer0.json
```

`logs/reader_sweep_adapted_layer0.provenance.json` reports exit 0. All 18
role/count cases pass all-rank PCC, unchanged replay, fresh-input/output reuse,
and changed-input replay. Minimum fresh-input PCC is `0.9999999999928121`.
Every fresh output address is distinct from its traced output. JSON SHA256:
`e740c9b91970cd1a3e23890d6287e117a1d13fc3f53c06383bac7e4db4090084`.
Log SHA256: `c5811753aef25686b3ddb04177bdb50072def143a5b287255f32b8235a1357eb`.
Archived Python source SHA256:
`dba18a44a86323f7d48fe83507eca34b293eebb4354b56b9e66c7641d26845df`.

Measured median projection trace times in microseconds, with common output
geometry per role and alternating reader order:

| Role | R1 | R2 | R3 |
| --- | ---: | ---: | ---: |
| GDN packed | 26.282 | 26.328 | 27.198 |
| GDN Z | 20.305 | 20.535 | 22.941 |
| GDN out | 17.425 | 18.618 | 18.469 |
| MLP gate | 32.941 | 27.516 | 32.026 |
| MLP up | 32.879 | 27.487 | 32.029 |
| MLP down | 33.665 | 27.708 | 35.368 |

Reader 2 is faster for these three MLP projections in this experiment. Reader 1
remains faster for these GDN projections. These measurements exclude input
sharding, row collectives, and the rest of the layer; policy integration and
whole-layer benefit require the parent's separate measurements.

## Optional packed-GDN-all reader probe

`tests/multichip_reader_probe.py` now accepts
`--variant packed_gdn_all_modes`. It selects the parent's `PackedGDNAllModes`
class for prefill and decode input capture, then constructs each local
`gdn_all` weight directly from raw HF `gdn_packed` plus raw HF Z columns.
It preserves independent A32/B32 padding and logs fields `[QKV, A_pad, B_pad, Z]`
and widths `[2048, 32, 32, 1024]`. The projection itself has no fused SiLU:
the class applies SiLU to Z in its later epilogue. The default variant remains
the existing `MultichipDecoder`; default baseline capture is unchanged.

For local `[K,N]=[4096,3136]`, N is 98 tiles. With four input cores and K block32,
the default output width25 creates four shards with 100 tiles total. Reader 3
uses 15 tiles per DRAM bank; one padding-only bank is pruned, leaving seven banks
and 21 readers with five tiles each. Its last reader starts at tile100, so the
default output width repeats the previously diagnosed assertion.

Use the same output `per_core_N=27` for every reader count. Four output shards
then cover 108 tiles, enough for all padded reader output in this geometry:

| Reader count | DRAM tiles per bank | Active banks | Active readers | Tiles per reader | Produced tile coverage | Common output capacity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 13 | 8 | 8 | 13 | 104 | 108 |
| 2 | 14 | 7 | 14 | 7 | 98 | 108 |
| 3 | 15 | 7 | 21 | 5 | 105 | 108 |

This geometry recommendation is source-derived and pending the parent's run:

```bash
timeout 300 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_reader_probe --layer 0 --variant packed_gdn_all_modes --roles gdn_all --cores 4 --block 32 --per-core-n 27 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/reader_gdn_all_n27.json
```

The existing per-role `--cores`, `--block`, `--per-core-n` and reader controls
apply unchanged. `py_compile` and applicable repository pre-commit hooks pass.
This investigator did not import TTNN or execute a hardware path.

## Compact reader/core/K-block follow-up matrix

The passing adapted layer-0 sweep establishes correctness and timing at its
particular input cores and K blocks. It does not establish that a reader family
is slower across coherent alternative input layouts. The source-derived plan
in [reader_geometry_matrix.json](reader_geometry_matrix.json) crosses readers
1/2/3 with the current input cores, 32 cores, and 64 cores where exact K
sharding permits. Each core choice uses its largest legal K block and largest
proper divisor. Block widths below are in 32-column tiles:

| Role | Local K × N | Current cores: block pair | 32-core block pair | 64-core block pair | Fixed output `per_core_N` |
| --- | --- | --- | --- | --- | ---: |
| GDN packed | 4096 × 2112 | 4: 32,16 | 4,2 | 2,1 | 18 |
| GDN Z | 4096 × 1024 | 4: 32,16 | 4,2 | 2,1 | 9 |
| GDN out | 1024 × 4096 | 4: 8,4 | 1 | unavailable | 36 |
| MLP gate/up | 4096 × 3072 | 8: 16,8 | 4,2 | 2,1 | 12 |
| MLP down | 3072 × 4096 | 8: 12,6 | 3,1 | unavailable | 18 |
| Full QKVG | 4096 × 2560 | 32: 4,2 | already covered | 2,1 | 6 |
| Full O | 1024 × 4096 | 4: 8,4 | 1 | unavailable | 36 |
| Optional GDN all | 4096 × 3136 | 4: 32,16 | 4,2 | 2,1 | 27 |

The largest block is `K / (32 * input_cores)`. K1024 has only 32 tiles;
K3072 has 96 tiles, which does not divide evenly over 64 cores. Those row
projections therefore omit 64 cores. A one-tile shard has no smaller block.
The JSON records exact configuration/source hashes, per-reader bank geometry,
core/block maps, and executable command arrays for the existing real-input
microprobe. It introduces no change to the probe or production decoder.

Output layout is held fixed per role across every input-core, block and reader
choice. Its capacity covers the maximum reader-produced padding, using the
factory's padding-only-bank pruning and the output allocator's
`ceil(N_tiles / per_core_N)` shard count. This retains the verified GDN packed
`per_core_N=18` adaptation. For QKVG, R3 prunes one of eight banks, leaving
21 readers × 4 tiles = 84 tiles. `per_core_N=6` gives 14 output shards and
84 tiles of capacity. The previously working QKVG width3 uses the other writer
branch; it is not equivalent to this matrix's common output layout, so the
JSON includes a fresh full-attention current-core control at width6.

There are five additional recipes per family: alternate current-core block,
32-core largest block, 32-core smaller block, 64-core largest block, and
64-core smaller block. Five are needed to cover both block endpoints at both
larger core counts after reusing an existing current-core endpoint. Duplicate
role/core/block cases are omitted. The linear family adds 75 reader cases in
five commands; the full-attention family adds 54 in five commands plus its
unmatched current-core control. Optional packed GDN adds 15 cases in five
commands; its pending width27 current-core control can be reused after it
passes. All commands sweep readers 1/2/3 together with identical per-role
dtype, fidelity and output layout. The 15 additional commands cover 144 reader
cases across both layer kinds and the optional packing variant.

The JSON `baseline_controls` entries state when existing evidence can be
reused; its `runs` entries contain the new commands. Run them serially under the
parent's hardware owner. A first failure leaves a partial artifact; preserve
it and resume remaining roles with a different output filename. Before using
timing samples, require every rank's PCC, unchanged and changed-input trace
replay, and fresh-buffer/cache checks to pass. These projection times exclude
input layout conversion and row collectives, so any selected geometry still
needs a whole-layer measurement.

CPU validation passed for all nine role shapes, all 18 command definitions
including reusable controls, and all 60 unique role/core/block points. It
checked exact K sharding, block divisibility, padded bank/reader arithmetic,
output capacity, small-reader writer start indices, shell/structured command
equivalence, complete endpoint coverage and absence of duplicates. Device
reader placement, CB/L1 capacity, correctness, replay and latency remain
runtime gates. No matrix command was executed by this investigator.

### Parent results and fewer-core controls

The parent subsequently ran all 15 matrix commands. Their corresponding
`logs/reader_geometry_*.provenance.json` entries report return code 0, and all
144 reader cases pass. The seven fewer-core follow-up commands also exited 0
and passed all 51 cases; their evidence is `reader_fewer_*.json` and
`logs/reader_fewer_*.provenance.json`. Including `reader_adapted_layer0.json`,
`reader_adapted_layer3.json`, and the now-passing `reader_gdn_all_n27.json`
control gives **231 passing cases across 25 result files**. A CPU audit checked all four ranks, actual
program reader counts, unchanged and changed-input trace replay, fresh output
addresses, two warmup/five measured windows, and median calculation. Minimum
PCC across original and fresh inputs is `0.9999999999927212`; every artifact
records the same installed runtime binary hashes. The child read this evidence
and did not execute hardware.

[reader_geometry_best.csv](reader_geometry_best.csv) records the fastest
observed geometry separately for each layer, role and reader count, including
source JSON/hash, actual dtypes/fidelity, sample range, and reader-1 timing from
the same geometry/run. The table below uses the fixed common output layouts
from the matrix. Thus its 228 eligible cases exclude the three earlier QKVG
width3 controls; those earlier controls remain valid evidence at their own
output geometry. QKVG weights in this microprobe are BFP8; the parent's
separate BFP4 whole-layer candidate is not included in these minima. Times
are median warmed projection trace microseconds;
parentheses show input cores/K block:

| Layer / role | Best R1 µs (cores/block) | Best R2 µs (cores/block) | Best R3 µs (cores/block) |
| --- | ---: | ---: | ---: |
| 0 / GDN packed | 26.052 (4/16) | 25.093 (4/16) | 27.198 (4/32) |
| 0 / GDN Z | 20.305 (4/32) | 20.535 (4/32) | 22.941 (4/32) |
| 0 / GDN out | 17.425 (4/8) | 18.618 (4/8) | 18.469 (4/8) |
| 0 / MLP gate | 31.380 (4/16) | 27.516 (8/8) | 31.717 (8/16) |
| 0 / MLP up | 31.376 (4/16) | 27.487 (8/8) | 31.730 (8/16) |
| 0 / MLP down | 31.883 (4/12) | 27.708 (8/6) | 34.628 (8/12) |
| 0 / optional GDN all | 33.668 (4/16) | 31.236 (4/16) | 37.284 (32/4) |
| 3 / BFP8 QKVG, width6 | 31.225 (8/8) | 36.636 (32/4) | 38.385 (32/4) |
| 3 / attention O | 17.390 (4/8) | 18.429 (4/8) | 18.454 (4/8) |
| 3 / MLP gate | 31.380 (4/16) | 27.501 (8/8) | 31.708 (8/16) |
| 3 / MLP up | 31.431 (4/16) | 27.514 (8/8) | 31.697 (8/16) |
| 3 / MLP down | 31.966 (4/12) | 27.689 (8/6) | 34.605 (8/12) |

The corrected reader-2 path is useful beyond the original MLP result: GDN
packed at four input cores/block16 measures 25.093 µs against matched reader-1
26.052 µs, and packed GDN all measures 31.236 µs against matched reader-1
33.668 µs. The earlier block32 GDN result did not support a family-wide slower
verdict. MLP reader-2 minima remain at eight cores/block8 for gate/up and
eight cores/block6 for down after including the four-core larger-block
controls. Four-core reader-1 MLP minima improve slightly, but do not displace
reader 2's lower times at eight cores.

The compact follow-up
[reader_geometry_fewer_cores.json](reader_geometry_fewer_cores.json) specified seven
commands and 51 reader cases, all with the same common output layouts:

- Layer 0: group the three MLP roles at four input cores, using gate/up
  block32 and down block24, then gate/up block16 and down block12.
- Layer 3: group QKVG and the three MLP roles in equivalent four-core runs;
  QKVG uses block32, then block16.
- Layer 3 QKVG: eight input cores/block16, then eight cores/block8.
- Layer 3 QKVG: the missing 32-core/block4 control with output width6.

Exact K-shard and block divisibility and complete reader-output capacity were
CPU-validated. Existing eight-core MLP endpoints are already covered and are
not repeated. The now-completed QKVG block4/width6 control is material: the older
width3 block4 microprobe measured R1/R2/R3 at 34.656/36.613/38.363 µs, while
the new width6 control measures 34.575/36.636/38.385 µs at the same input
cores/block. Reader 1's 8-core/block8 minimum is 31.225 µs, with matched
reader 2 at 37.467 µs. These results support reader 1 for the measured BFP8
projection class; they do not establish a universal reader preference across
weight precision, output layout or whole-layer configuration.

In the parent's whole-layer selected family with MLP reader 2, BFP8 QKVG at
32 cores/block4 measures
[R1 0.271353 ms](logs/production_packed_dram2_qr1_layer3.provenance.json),
[R2 0.272172 ms](logs/production_packed_dram2_qr2_layer3.provenance.json), and
[R3 0.274966 ms](logs/production_packed_dram2_qr3_layer3.provenance.json).
All three runs exit 0 with exact trace replay. The earlier isolated whole-layer
reader-2 advantage therefore does not persist in this selected family.

A different candidate uses BFP4 QKVG at eight input cores/block16 with
reader 2. Its
[whole-layer run](logs/production_qkv4_c8_layer3.provenance.json) exits 0 at
**0.266119 ms**, with exact trace replay and decode PCC
`0.9999813329826338`. Its
[batch-contract run](logs/production_qkv4_c8_batch_contracts.provenance.json)
passes all four cases: both layer kinds at batch4 and batch32. This is
whole-layer evidence for a different precision/geometry combination and is
excluded from the BFP8 microprobe CSV. The parent is separately comparing
output width12, block8/16, and readers 1/2/3 under BFP4 and BFP8 before final
policy selection. No generic reader default has changed. Whole-layer
correctness, tracing and latency measurements govern integration of any
projection candidate.

Applicable pre-commit checks pass for the matrix/CSV/report artifacts; these
are documentation and CPU arithmetic changes and require no C++ build.
