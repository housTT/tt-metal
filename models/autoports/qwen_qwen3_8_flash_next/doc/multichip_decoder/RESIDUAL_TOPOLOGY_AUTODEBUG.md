# Residual topology AutoDebug

Date: 2026-08-27

Target: P300 dies 0 and 1, `1x2` TP mesh, `FABRIC_1D`, linear CCL

Scope: Qwen layer-3 QSA output through the next real hyperconnection consumer

## Outcome

The stage-review finding was valid: the original multichip decoder reduced every
QSA/MoE row partial immediately with BF16 `ttnn.all_reduce`, so it had not tested
a shape-faithful fractured residual through a real consumer.

The decisive hardware A/B now implements all three relevant chains with the
real layer-3 QSA output partial, represented packed checkpoint weights,
decode-mode matmul configs, actual router projection, and trace capture:

| Result | Replicated baseline | Norm-gather fracture | Fully fractured HC |
| --- | ---: | ---: | ---: |
| collective sequence | AR 2560 | RS 2560 + AG stats + AG normed 4x1280 | RS 2560 + AG stats + AR packed324 + AG mixed1280 |
| mixed PCC | reference | prior probe `0.99999207` | `0.99999219` |
| next-injection PCC | reference | prior probe `0.99999386` | `0.99999654` |
| injected-residual PCC | reference | prior probe `0.99999988` | `0.99999988` |
| actual-router PCC | reference | validated in expanded test | `0.99998784` |
| median traced latency | `0.293827 ms` | `0.296919 ms` | **`0.234947 ms`** |
| contender / baseline | `1.000000` | `1.010523` | **`0.799611`** |

This smoke used seven internal samples of ten warmed replays.  Fully fractured
HC saves `0.058880 ms` (`20.039%`) and is a `1.250610x` speedup over replicated
AR.  It also beats norm-gather by `0.061972 ms`.  The earlier 100-replay
norm-gather result (`0.254016` versus `0.256512 ms`) remains useful diagnostic
evidence but is superseded for selection because it did not include the actual
router or the winning represented half-HC weights.

The selected production direction is therefore **persistent within-stream TP2
residual sharding at stack-internal boundaries**.  The winning trace deliberately
gathers its local residual for PCC only after releasing the trace.  Adding that
AG after every decoder would move another 81,920 B/die at decode and would no
longer implement the measured chain.  Replicated 10240 remains only a one-time
standalone/stack ingress and egress compatibility ABI.

Evidence target:

- activation SHA-256
  `635cca15a36d85c5053c7627812f17d47137691b1d7fdc08ddaf5417387417dc`
- test
  `test_multichip_decoder_perf.py::test_qsa_residual_topology_through_real_hyper_consumer`
- exact full contender:
  `reduce_scatter_2560+all_gather_rms_stats+all_reduce_packed324+all_gather_mixed1280`

## Exact contracts

Model constants are hidden size 2560, four hyperconnection streams, flat
hyperconnection width 10240, and low-rank width 320.  `M` is the logical row
count: decode `M=1`, normal prefill chunk `M=128`.  The probe uses BF16 TILE DRAM
activations and CCL payloads.  Existing optimized weight dtypes and compute
fidelity are unchanged.

The within-stream fracture that aligns a reduced 2560 block with every
hyperconnection stream is:

```text
global semantic residual       [1, 1, M, 4, 2560]
global TT grouped view         [1, 1, 4*M, 2560]
rank-local TP2 view             [1, 1, 4*M, 1280]
```

Flat contiguous sharding of `[M,10240]` would instead assign two whole streams
to each rank and would not align with the `[M,1280]` reduced block.

| Boundary | Current | Selected fully fractured candidate |
| --- | --- | --- |
| real QSA output partial | rank-distinct `[1,1,M,2560]` | same producer tensor |
| reduced block | replicated `[1,1,M,2560]` | local `[1,1,M,1280]` |
| injection scalars | replicated `[1,1,M,4]` | same |
| hyper input | flat replicated `[1,1,M,10240]` | partitioned grouped `[1,1,4*M,1280]` |
| local injected residual | N/A | `[1,1,4*M,1280]` |
| RMS pre-AG stats | N/A | logical `[1,1,4*M,32]` |
| RMS post-AG output | N/A | `[1,1,4*M,1280]` |
| norm gain | `[10240]` | grouped/partitioned `[1,1,4,1280]`, broadcasts over `M` |
| input to packed HC down | `[1,1,M,10240]` | local reshape `[1,1,M,5120]` |
| HC packed down+inject | weight `[10240,324]`; result `[M,324]` | represented local weight `[5120,324]`; partial AR to replicated `[M,324]` |
| HC up | weight `[320,10240]`; result `[M,10240]` | represented local weight `[320,5120]`; result `[M,4,1280]` |
| mixed output before router | replicated `[1,1,M,2560]` | local `[1,1,M,1280]`, then AG to replicated `[1,1,M,2560]` |
| actual router | rank-local weight `[2560,1153]` | identical real op/weight on gathered mixed |
| returned local residual | replicated `[1,1,M,10240]` | `[1,1,4*M,1280]`; no timed gather |

There is no logical alignment restriction: 2560/2=1280 and `4*M` works for any
valid `M`.  Decode rows receive normal TILE height padding to 32.  Physical BF16
sizes for decode are: block input 163,840 B, block shard 81,920 B, local grouped
residual 81,920 B, stats 2,048 B, and packed width 324 padded to 352.

## Exact measured operations and mesh mapping

The test temporarily bypasses `_all_reduce_block` while executing the real
`_decode_attention_host`, preserving its rank-distinct QSA output-projection
partial rather than manufacturing a producer.  It partitions the existing
residual and norm gain within every stream:

```python
hyper_groups = ttnn.reshape(attention.hyper, (1, 1, 4, 2560))
local_hyper_groups = ttnn.mesh_partition(
    hyper_groups,
    dim=3,
    cluster_axis=layer.collective_axis,  # 1 on the 1x2 mesh
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
norm_groups = ttnn.reshape(layer.w["mlp_hc_norm"], (1, 1, 4, 2560))
local_norm_weight = ttnn.mesh_partition(
    norm_groups,
    dim=3,
    cluster_axis=layer.collective_axis,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

After the shared RS/injection/distributed-norm prefix, the selected trace keeps
the normalized tensor local through both represented HC projections:

```python
block_s = ttnn.reduce_scatter(
    qsa_partial,
    dim=3,
    cluster_axis=1,
    num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
gate = ttnn.reshape(attention_injection, (1, 1, 4, 1))
projected_s = ttnn.multiply(
    block_s, gate,
    input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
)
local_updated = ttnn.mac(projected_s, 2.0, local_hyper_groups)

stats_s = ttnn.rms_norm_pre_all_gather(
    local_updated,
    compute_kernel_config=_hifi4(fp32=True),
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
stats = ttnn.all_gather(
    stats_s, dim=3, cluster_axis=1, num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
normed_s = ttnn.rms_norm_post_all_gather(
    local_updated, stats,
    epsilon=layer.shapes.rms_norm_eps,
    compute_kernel_config=_hifi4(fp32=True),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
normed_s = ttnn.multiply(normed_s, local_norm_weight)
local_flat = ttnn.reshape(normed_s, (1, 1, 1, 5120))
packed_partial = layer._linear_impl(
    local_flat, local_down_inject, dtype=ttnn.bfloat16
)
packed = ttnn.all_reduce(
    packed_partial, cluster_axis=1, num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
low = ttnn.silu(layer._slice_last(packed, 0, 320))
next_injection = layer._slice_last(packed, 320, 324)
local_mix = layer._linear_impl(low, local_up, dtype=ttnn.bfloat16)
local_mix = ttnn.reshape(local_mix, (1, 1, 4, 1280))
local_mixed = ttnn.multiply(
    normed_s, local_mix,
    input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
)
local_mixed = ttnn.mean(local_mixed, dim=2, keepdim=True)
mixed = ttnn.all_gather(
    local_mixed, dim=3, cluster_axis=1, num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
mixed = ttnn.reshape(mixed, (1, 1, 1, 2560))
router = layer._linear(mixed, layer.w["moe_input"])
```

After `ttnn.release_trace`, a separate `ttnn.all_gather(local_updated, dim=3,
cluster_axis=1, ...)` reconstructs `[1,1,4,2560]`, then the host reshapes it to
`[1,1,1,10240]` for residual PCC.  This gather is not a layer operation and is
not timed.

## Required topology table

Wire volumes are calculated physical BF16 bytes per die for TP2.  An AR contains
an RS half plus AG half and moves one physical input tensor per die.  The table
distinguishes measured rows from source-feasible but unmeasured alternatives.

| Family | Residual topology through next consumer | Decode CCL B/die | Prefill-128 CCL B/die | Status / decision |
| --- | --- | ---: | ---: | --- |
| all-reduce | QSA partial AR; replicated injection, norm, HC tail and router | 163,840 | 655,360 | Superseded baseline. Expanded trace: `0.293827 ms`. |
| immediate RS + AG | RS `[M,2560] -> [M,1280]`, then AG before any consumer | 81,920 + 81,920 | 327,680 + 327,680 | Reject: reconstructs AR with two launches and proves no useful local lifetime. |
| RS + delayed gather after real norm | RS; local `[4M,1280]` injection; distributed RMS; AG normed `[4M,1280]`; real HC tail/router | 81,920 + 2,048 + 81,920 = 165,888 | 327,680 + 32,768 + 1,310,720 = 1,671,168 | Measured `0.296919 ms`, `1.010523x` baseline; reject for decode. Prefill wire volume is 2.55x current and was not run. |
| fused AG-matmul | Keep normed/mixed local; `all_gather_matmul_async` feeds packed router `[2560,1153]` | included in complete delayed-HC row below | included below | API exists; requires persistent semaphore/barrier and disjoint CCL workers. Unmeasured. |
| fused matmul-RS | Fuse QSA output projection with RS to `[M,1280]` via `matmul_reduce_scatter_async` | same 81,920 payload, overlapped | same 327,680 payload, overlapped | Source-feasible. Sibling Qwen36 reports decode `M=1` may lose and prefill can win. Not needed to answer residual-consumer P1. |
| fused matmul-AR | QSA MMRS followed by immediate AG | 163,840 | 655,360 | No separate fused matmul-AR API was found. Retain only if MMRS+AG beats ordinary MM+AR. |
| residual-sharded / half-HC | Keep `[4M,1280]` through local norm, local HC down/up and local mixed output; packed-324 AR; AG mixed into actual router | 81,920 RS + 2,048 stats AG + 22,528 packed AR + 81,920 router-input AG = 188,416 | 327,680 + 32,768 + 90,112 + 327,680 = 778,240 | **Selected.** `0.234947 ms`, `0.799611x` baseline despite 15.0% more wire bytes; halved HC work wins. |
| residual-sharded stack | Carry `[1,1,4*M,1280]` across layers; QSA/MoE RS; GDN output-N shard; PLE bridge | same per HC plus producer-specific RS | same formula by `M` | **Required ABI.** Avoids an excluded 81,920 B/die residual AG after every decode layer. |

The norm-gather loss is consistent with its payload calculation: it moves
165,888 rather than 163,840 B/die (`+1.25%`) and saves no HC-tail work.  The
fully fractured result demonstrates that halving both HC projections more than
repays its 15% higher wire volume.

## Selected half-HC kernel

The selected topology shards HC weights within all four streams and does not
gather the 10240 normalized tensor.  The measured version gathers the final
1280-wide mixed shard and then runs the real packed router matmul; it does not
depend on unproven AG-matmul fusion:

```text
down `[4,2560,324]` -> local `[4,1280,324]` -> reshape `[5120,324]`
up   `[320,4,2560]` -> local `[320,4,1280]` -> reshape `[320,5120]`
norm gain `[4,2560]` -> local `[4,1280]`
```

After local down, `ttnn.all_reduce` must combine the packed `[M,324]` partial.
Local up produces `[M,4,1280]`; sigmoid multiply and group mean produce
`mixed_s [1,1,M,1280]`.  The measured final calls are:

```python
mixed = ttnn.all_gather(
    mixed_s, dim=3, cluster_axis=1, num_links=1,
    topology=ttnn.Topology.Linear,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
router_packed = layer._linear(mixed, layer.w["moe_input"])
```

`all_gather_matmul_async` remains an optional later fusion, not a production
dependency for accepting this path.  Its likely call is:

```python
tt_ccl = get_tt_ccl(mesh_device)  # allocate before warm/capture
ag_input, router_packed = ttnn.experimental.all_gather_matmul_async(
    mixed_s,
    layer.w["moe_input"],                 # per-rank [2560,1153]
    persistent_output_buffer=None,
    dim=3,
    multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(1),
    all_gather_core_grid_offset=(0, 5),   # disjoint from current 8x5 MM grid
    barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
    num_links=1,
    memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
    topology=ttnn.Topology.Linear,
    memory_config_mm=ttnn.L1_MEMORY_CONFIG,
    program_config=layer._decode_1d_program_config(layer.w["moe_input"], 40),
    compute_kernel_config=layer.projection_compute_cfgs["shared"],
    dtype=ttnn.bfloat16,
    chunks_per_sync=10,
    num_workers_per_link=2,
    num_buffers_per_channel=2,
)
```

The non-fused control is `ttnn.all_gather(mixed_s, dim=3, cluster_axis=1,
...)` followed by the current `_linear(..., layer.w["moe_input"])`.  QSA MMRS
would use the P300/Qwen36 `matmul_reduce_scatter_async` helper with persistent
`[M,2560]` producer and `[M,1280]` output buffers.

## Narrow production integration map

### ABI decision

Do not gather the 10240 residual at each decoder return.  That gather is absent
from the winning trace and costs 81,920 B/die plus one CCL launch for decode.
Use two explicit ABIs instead:

- external compatibility: replicated `[1,1,M,10240]` in/out;
- stack-internal: MeshShard on TP axis 1, local
  `[1,1,4*M,1280]` BF16 TILE DRAM in/out.

`prefill_forward` and `decode_forward` can remain compatibility wrappers that
fracture once on entry, execute the sharded core, and gather once on return.
Add explicit `prefill_forward_sharded` / `decode_forward_sharded` entry points
for a layer stack.  A stack fractures once before layer 0, calls only sharded
entry points, and gathers once after layer 47.  Do not infer layout from a
padded shape; make the method/contract explicit.

The adapters are:

```python
def _fracture_residual(self, full, rows):
    grouped = ttnn.reshape(full, (1, 1, 4 * rows, 2560))
    return ttnn.mesh_partition(
        grouped, dim=3, cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

def _gather_residual(self, local, rows):
    grouped = ttnn.all_gather(
        local, dim=3, cluster_axis=1, num_links=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.reshape(grouped, (1, 1, rows, 10240))
```

### Weight setup

Immediately after the two optimized rank instances have been patched into one
mesh tensor, replace both `attn_hc` and `mlp_hc` weights with within-stream
shards.  Device `mesh_partition` of the already represented tensor is preferred
to checkpoint re-quantization:

```text
{prefix}_norm:
  [10240] -> reshape [1,1,4,2560] -> partition dim 3 -> [1,1,4,1280]
{prefix}_down_inject:
  [1,1,10240,324] -> [1,4,2560,324]
  -> partition dim 2 -> reshape [1,1,5120,324]
{prefix}_up:
  [1,1,320,10240] -> [1,320,4,2560]
  -> partition dim 3 -> reshape [1,1,320,5120]
```

Replace/deallocate the global copies rather than retaining both layouts.  Rebind
each new tensor in `weight_group_by_id` to `shared` and in
`weight_role_by_id` to its original role, so `_linear_impl` selects the tested
decode-mode policy.  Preserve each source tensor's represented dtype.

For GDN layers, leave every recurrence/input tensor replicated but partition
only `w["gdn_out"]` on output N (`[linear_value_width,2560] ->
[linear_value_width,1280]`, mesh partition dim 3).  This makes the existing
`_gdn_epilogue` emit the local block directly without changing the proven
48-head recurrence.  Register the new tensor as group `gdn`, role `gdn_out`.

### Shared fractured hyper primitives

Add `_hyper_mix_fractured(local_hyper, prefix, rows)` by lifting the proven test
helper into the class.  Its contracts are:

```text
input/local hyper       [1,1,4*M,1280]
distributed RMS output [1,1,4*M,1280]
local flat down input   [1,1,M,5120]
packed partial/result   [1,1,M,324] -> AR replicated
local up/mix groups     [M,4,1280]
local mixed             [1,1,M,1280]
AG mixed/output         [1,1,M,2560] replicated
injection               [1,1,M,4] replicated
returned hyper          original local input
```

Use exactly `rms_norm_pre_all_gather`, stats `all_gather`,
`rms_norm_post_all_gather`, local norm multiply, `_linear_impl` down,
`all_reduce` packed324, `_linear_impl` up, sigmoid multiply/group mean, then
ordinary `all_gather` mixed1280.  Keep the measured ordinary router `_linear`;
do not make async AG-matmul part of the first production patch.

Add `_hyper_inject_fractured(local_hyper, local_block, injection, rows)`:
reshape block `[1,1,M,1280] -> [M,1,1280]`, gate `[1,1,M,4] ->
[M,4,1]`, sigmoid multiply, reshape `[1,1,4*M,1280]`, then
`ttnn.mac(projected, 2.0, local_hyper)`.

Add `_reduce_scatter_block` beside `_all_reduce_block`; it performs BF16
`ttnn.reduce_scatter(..., dim=3, cluster_axis=1, num_links=1,
topology=Linear, DRAM)` and retains the current FP32-to-BF16 safeguard.

### QSA and GDN attention

- `_qsa_prefill` and `_qsa_decode`: replace their final `_all_reduce_block`
  wrapper with `_reduce_scatter_block`.  The inherited rank-local attention and
  KV/cache contracts are unchanged; only the real `attn_out` partial changes
  from `[1,1,M,2560]` to local `[1,1,M,1280]` after RS.
- `_gdn_prefill` / `_gdn_decode`: retain all current recurrence, causal-conv,
  workspace and direct-state-commit logic.  The output-N-sharded `gdn_out`
  makes only the returned block `[1,1,M,1280]` local.  If that weight program
  config does not compile, the correctness bridge is
  `ttnn.mesh_partition(full_gdn_block, dim=3, cluster_axis=1)`; do not split
  GDN heads again.
- Both block kinds feed `_hyper_inject_fractured` directly.  No AR/AG belongs
  between the producer and injection.

### MoE and the layer back edge

The gathered mixed activation `[1,1,M,2560]` preserves the current actual
router, top-k=10 active-expert path, shared expert and host-cache contracts.
Resident, host-backed, prefill-wave and batch paths therefore need no input
rewrite.  Their routed+shared result is still a rank-local contribution of
shape `[1,1,padded_M,2560]`.

Change the `_moe` wrapper and `_decode_back_host` from AR to RS.  For the
segmented batch-one path, RS the padded `[1,1,32,2560]`, then slice to
`[1,1,1,1280]`.  Inject that local block into the retained local MLP residual;
the returned decoder residual stays `[1,1,4,1280]`.  For ordinary prefill or
batch decode, slice RS output to logical `[1,1,M,1280]` before local injection.

### PLE layer 1

The narrow safe implementation keeps exact PLE replicated and inserts one
explicit layer-1-only bridge:

```text
local stack residual [4*M,1280]
 -> AG/reshape [M,10240]
 -> existing exact PLE gate, norm, FIR and replicated state
 -> replicated add
 -> group/partition back to [4*M,1280]
 -> fractured attn_hc
```

This does not make the stack ABI replicated; it is one exceptional consumer in
one layer.  It leaves host row lookup, `PLEDeviceStaging`, user state,
`fused_ple_conv_state`, workspace snapshots and non-aligned masks unchanged.
A later all-local PLE would require repacking `ple_key_value` into four local
key slices plus a local 1280 value, distributed key/query norms, AR of four dot
scores, and local state shaped by stream; that is too broad for the narrow
winning-HC production patch.

### Prefill, non-aligned lengths, and batch 32

Implement the existing forward loop in `_prefill_forward_sharded`.  A chunk of
logical/padded lengths `(logical,padded)` uses local shape
`[1,1,4*padded,1280]`.  Sharded slicing uses row interval
`[4*start, 4*(start+logical))`, pads to `4*padded`, trims to `4*logical`, and
concatenates pieces on `-2`.  This retains arbitrary logical sequence lengths;
never require `M` or `4*M` to be tile aligned publicly.

Decode batch `B` uses local residual `[1,1,4*B,1280]` (token-major, then four
streams).  `_hyper_mix_fractured` reshapes it to `[1,1,B,5120]` for down and
`[B,4,1280]` for mixing.  QSA/GDN local blocks are `[1,1,B,1280]`; gates are
`[1,1,B,4]`.  At `B=32`, the exact shapes are residual
`[1,1,128,1280]`, block `[1,1,32,1280]`, packed `[1,1,32,324]`, and mixed
`[1,1,32,2560]`.  Current MoE padding is already 32, so no new batch-32
alignment contract is introduced.

### Segmented decode traces

Update `_decode_attention_host` to accept local `[1,1,4,1280]`; apply the PLE
bridge only on layer 1, run fractured `attn_hc`, and return
`HostDecodeAttention(hyper=[1,1,4,1280], block=[1,1,1,1280],
injection=[1,1,1,4])`.  `_decode_router_host` performs local injection,
fractured `mlp_hc`, and the unchanged actual router/shared projection.  Its
`HostDecodeFront.work` remains replicated padded `[1,1,32,2560]`, while
`hyper` is local `[1,1,4,1280]`.

In `_decode_back_host`, RS routed+shared and locally inject as described above;
the trace output is local `[1,1,4,1280]`.  Keep route-id D2H/expert H2D service,
front corruption marking, two-phase warm/freeze/capture, GDN state snapshots,
and shared workspace serialization unchanged.  Allocate/compile the new
RS/stats-AG/packed-AR/mixed-AG programs during `warm_programs` before any older
stack trace is live.  Captured input/output validators and stack tests must use
the local ABI; comparison gathers occur only after the relevant trace is
released.

## Blockers and rejected alternatives

- The winning smoke covers one QSA-to-MLP-HC/router boundary, not both HC
  prefixes, GDN output-N sharding, MoE RS/back injection, prefill, batch 32 or a
  multi-layer segmented trace.  Each is a mandatory focused gate before final
  acceptance.
- PLE layer 1 keeps flat replicated 10240 key/query/conv state.  The explicit
  one-layer bridge is the narrow correctness choice; its latency must be
  included in layer-1/stack evidence rather than hidden outside timing.
- GDN must retain the proven replicated 48-head recurrence.  Only `gdn_out` N
  may be split; any recurrence/head split repeats an already failed topology.
- The first production path uses ordinary mixed AG plus router matmul.  Async
  AG-matmul semaphores/worker-core placement are optional later optimization,
  not blockers.
- Host-backed MoE should reuse the router's gathered 2560 activation for the ten
  active experts.  Gathering per expert would erase the topology benefit.
- Flat `[10240] -> [5120]` TP gives each rank two complete streams and does not
  align with the local 1280 block injection.
- Group sharding (two whole streams/rank) makes norm local but still needs a
  complete 2560 block and cross-rank group mean.
- Lowering CCL dtype before BF16 topology selection mixes precision with layout
  and is not a valid first discriminator.

## Focused hardware command

The decisive smoke can be reproduced without an outer pytest repeat because it
collects seven internal timing samples:

```bash
cd /home/ttuser/dev/qwen3.8-flash-next/tt-metal
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto"
export TT_METAL_WATCHER=10
export QWEN38_MC_RUN_RESIDUAL_TOPOLOGY=1
export QWEN38_MC_RESIDUAL_TOPOLOGY_REPLAYS=10
timeout 1800 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_qsa_residual_topology_through_real_hyper_consumer \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/residual_topology_probe.xml
```

Watcher and profiler runs must remain separate.  Release traces before any
comparison-only allocation/gather, as the passing probe does.

## Ranked hypotheses after measurement

1. **Confirmed for decode:** fully fractured HC with ordinary mixed AG and the
   actual router is the selected topology: `0.799611x` latency / `1.250610x`
   speedup.  Halving HC projection work dominates its 15% extra CCL payload.
2. **Required to retain the win:** a persistent sharded stack ABI avoids the
   comparison-only residual gather.  Replicating every layer boundary is not
   the measured algorithm and is likely to consume much of the 0.058880 ms win.
3. **Likely safe narrow exception:** one replicated PLE bridge at layer 1 is
   lower risk than rewriting its gate/conv state and is amortized across the
   48-layer stack, but it needs measured stack evidence.
4. **Likely prefill-only opportunity:** fused QSA matmul-RS may hide collective
   time for `M=128`; source precedent warns that under-filled `M=1` decode can
   lose.  The measured decode result must not be generalized to a prefill
   benchmark that was not run.
5. **Very likely losses:** immediate RS+AG, whole-stream sharding, and MMRS+AG
   without a useful local consumer add launches or misalign the residual.
6. **Precision only after topology:** BF8 CCL could reduce bytes but needs a
   separate layer/full-stack PCC sweep after BF16 correctness.

The fresh-context AutoDebug runner could not enter this checkout because its
`bwrap` user namespace failed (`loopback: Failed RTM_NEWADDR`).  The source/API
investigation was therefore reconstructed against the local checkout.  The
hardware numbers above are not inferred: they are copied exactly from the main
stage's passing hardware output.  The checked-in XML still contains the earlier
norm-gather run until the decisive smoke is persisted by the main stage.
