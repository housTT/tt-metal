# AutoFix: rank-local expert staging

Status: fixed and verified on the target P300 TP2 mesh on 2026-08-28.

## Defect

`QwenDeviceExpertCache` previously obtained shard views with
`ttnn.get_device_tensors()` and then used each view's `device()` as an
allocation target. A shard view has coordinate-limited `DeviceStorage`, but
`device()` is still the parent `MeshDevice`. Each supposed rank-local staging
and zero tensor was consequently replicated over both ranks. The recurring
1x1-host copy also took the generic replication special case. Physical PCIe
traffic and staging DRAM were about 2x the declared owner-only values, while
`h2d_bytes` counted only one payload.

Source audit confirmed the failure boundary:

- `get_device_tensors()` creates views over the existing parent `MeshBuffer`;
  it does not return independent physical-device allocation handles.
- Tensor construction with a coordinate-sparse mesh mapper still allocates a
  replicated parent `MeshBuffer`; it only makes the initial host distribution
  sparse.
- Generalizing `experimental_to_single_device` to interleaved DRAM is unsafe:
  its physical-device allocator is not coordinated with the parent
  `MeshDevice` DRAM allocator and can select an address already live in a
  parent-mesh tensor.
- A sparse generic `copy_host_to_device_tensor` was also rejected for shared
  staging: its non-uniform path replaces the destination `DeviceStorage` and
  topology, so two owner threads writing views of the same holder can race.

## Accepted repair

`ttnn.copy_host_to_device_tensor_at_coordinate` is a narrow nanobind TTNN
primitive that validates host/device storage, allocation, logical shape,
dtype, page config, target coordinate, and exact packed byte count. It builds
one `ShardDataTransfer` and calls `MeshCommandQueue::enqueue_write_shards` on
the existing raw parent `MeshBuffer`. It never replaces `DeviceStorage` or
topology.

The model cache now allocates:

- one normal replicated parent-mesh staging gate/down pair per staging depth;
- one normal replicated immutable zero gate/down pair;
- coordinate-limited views of those shared buffers for the existing D2D path.

The retained packed host tensor is written into only the owner's coordinate of
the full staging base. The owner staging view and non-owner zero view are then
copied D2D into the fixed slot views. All calls use parent CQ0, preserving H2D,
D2D, index publication, and consuming-trace order. `close()` deallocates the
underlying shared bases, never their views.

The completed single-owner and dual-owner H2D benchmark probes use the same
coordinate primitive, so their byte denominators now match physical PCIe
traffic rather than measuring a hidden mesh broadcast.

Coordinates are derived from the allocated base tensor's canonical
`device_coords()`. The first TT gate found that
`MeshCoordinateRange(mesh_device.shape)` duplicated `(0, 0)` after the fixture's
in-place reshape; the failed log is preserved, and the production code was
corrected before the passing rerun.

The physical accounting is now exact for service misses:

- H2D per miss: one owner payload, 2,764,800 bytes;
- zero reset per miss: one rank-local D2D payload, 2,764,800 bytes;
- device bytes per rank: `(capacity + staging_depth + 1) * 2,764,800`, or
  33,177,600 bytes at capacity 10 and depth 1;
- replicated construction-time initialization remains excluded from service
  `h2d_bytes`, as declared by `h2d_bytes_scope`.

## Verification

Build and import:

```text
ninja -C build_Release ttnn/_ttnn.so -j 8
# PASS; generated extension exposes
# ttnn.copy_host_to_device_tensor_at_coordinate
```

Host/source gate:

```text
python_env/bin/python -m pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py \
  -k 'host_expert_staging_uses_one_parent_allocation_and_coordinate_h2d or rank_local_staging_coordinate_write_preserves_parent_topology' \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_rank_local_staging_host.xml

# 1 passed, 1 opt-in TT test skipped, 48 deselected
# XML sha256: ed4f57eee237db6940e00f74810c0852395ec9cc6cfe2409903ae5a9099fc439
```

Authorized target-hardware gate:

```text
TT_VISIBLE_DEVICES=0,1 \
TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto \
RUN_QWEN38_RANK_LOCAL_STAGING_TT=1 \
python_env/bin/python -m pytest -q --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_rank_local_staging_coordinate_write_preserves_parent_topology \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_rank_local_staging_tt.xml

# 1 passed in 1.15s
# log sha256: 28cb6c5442c4fd6831fb691e2ac755091cf97e992a84b866e0028200570bf092
# XML sha256: 09c51ae3caaa52821c40ac51e334620a511af86fe04528a4b5a626ec7e5aae0c
```

The hardware gate writes distinct BF16 sentinels to both owner coordinates
from two Python threads, checks that the full base retains both DeviceStorage
coordinates and both topology coordinates, copies the rank views D2D, and
reads exact rank-specific staging and target values.

The first failed TT evidence is retained as
`autofix_rank_local_staging_tt_failed_coordinate_enumeration.{log,xml}`. Its
log SHA256 is
`81e056fb134e8aaca7798543ed8eeaea9e99b0982e811a881c8afb8c47506bd5` and
its XML SHA256 is
`08de928f32584602ec9a8b043765e781b099730765ad67faf66876293bebf390`.

Preflight and postflight `tt-smi -s` reported healthy P300 dies 0 and 1.
Postflight `fuser` found no `/dev/tenstorrent/{0,1}` holders, and the process
audit found no vLLM, EngineCore, or model pytest processes.

## API limitation

There is still no supported allocator-safe primitive for an independent
interleaved-DRAM allocation on only one physical coordinate of a live parent
mesh. This repair deliberately avoids that unsupported operation. The new
copy primitive writes a full, exact single-shard host payload into an already
allocated parent-mesh tensor; it is not a partial-region or multi-shard API.
Callers must retain the host tensor through submission/completion. Qwen3.8's
bounded packed-host cache retains it for the cache lifetime.
