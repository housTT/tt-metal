# AUTODEBUG — GPT-OSS-120B fused decoder: single-device MoE dispatch fails on a 1×1 P150 mesh

Date: 2026-08-28
Focus path: `models/autoports/openai_gpt_oss_120b/tt/fused_decoder.py`
Mode: **source-only AutoDebug**. The AutoDebug run did not use hardware. One focused hardware
contrast supplied by the parent investigation is recorded separately below. No implementation or
test file was modified.
Deliverable: this file only.

---

## 0. Method, and what is proof versus inference

`import ttnn` fails in this checkout (`_ttnn.so: undefined symbol: ...scaled_dot_product_attention...`),
so the AutoDebug agent produced no new runtime result itself. Every source claim below is either

* **(P)** proved by reading source, with `file:line`, or
* **(I)** inferred, and labelled as such.

Two pieces of runtime evidence are available and are kept distinct from the source proof. First,
the failing pytest run is
captured verbatim in `.agents/runs/gpt-oss-120b-p150-family-20260827T214421Z/02-02-fused-decoder.jsonl`.
It contains two distinct TT_FATAL backtraces from the real hardware — the earlier `offset_cumsum`
one and the current `dispatch` one — and both are quoted below. Those are **direct observations**.
Second, the parent investigation performed a focused `FABRIC_1D` contrast on the same class of
host; that result is summarized in §2 and §5.1. Everything derived from these observations is
interpretation and is marked as such.

Line numbers were re-verified directly against the working tree for every claim that is
headlined. Where a supporting deep-dive reported a line number I could not reproduce, I use my own.

---

## 1. TL;DR

**The reported `TT_FATAL: Trying to get un-initialized fabric context` is only the first visible
failure. One decisive, fabric-independent neighbour requirement proves that
`ttnn.experimental.deepseek_prefill.dispatch` and `...combine` cannot run on a 1×1 mesh at all;
several additional unconditional fabric accesses explain why the current fabric-free call fails
even earlier.**

`dispatch` and `combine` are structurally multi-device ops. Their host program factories
unconditionally ask the mesh for the current device's neighbours along the cluster axis and
`TT_FATAL` when there are none. On a `(1, 1)` mesh with `Topology::Linear` there are none, **and
that assertion fires whether fabric is initialised or not**.

So the specific question in the brief — *"is FABRIC_1D on 1×1 a viable focused workaround?"* — has
a clean, source-provable answer:

> **No.** On the actual 1×1 P150-class host, `FABRIC_1D` timed out during router initialization
> before the decoder ran. Even on a hypothetical 1×1 system where fabric initialization succeeds,
> the op then reaches `TT_FATAL(!neighbors.empty(), "No neighbors found")` at
> `ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.cpp:93`, called unconditionally from
> `dispatch_program_factory.cpp:492-493` and `combine_program_factory.cpp:173-174`.

The repair contract forbids C++ changes, so **the fused decoder cannot use `dispatch`/`combine` at
1×1 at all** and the design has to change. Section 5 ranks the options; the leading one is to drop
those two ops and keep the parts that are already fabric-free.

Two further **certain** defects sit behind the fabric one and would each hard-fail the moment it is
bypassed: the fused decoder hands `dispatch` and `post_combine_reduce` TILE-layout routing tensors
where both ops `TT_FATAL` unless they are ROW_MAJOR. The shared DeepSeek caller does the conversion
(`tt_moe.py:698-699`); `fused_decoder.py:287-296` does not.

Good news, also proven: the parts of the pipeline that are *not* `dispatch`/`combine` really are
fabric-free and single-chip-tested — `masked_bincount`, `unified_routed_expert_moe`, and
`post_combine_reduce` all have in-tree single-chip tests that pin `fabric_config=DISABLED`. And the
hand-written local cumsum that replaced `offset_cumsum` computes **exactly** the right thing at
`dispatch_group_size == 1`; that substitution is correct (§3.4).

---

## 2. Direct observations (from the recorded run)

Both of these are quoted from the run log, not reconstructed.

**Observation A — the earlier failure, already repaired.** `TtMoERoutingSetup.forward` →
`offset_cumsum` → `ttnn::all_gather` → `get_tt_fabric_max_payload_size_bytes()` →
`control_plane.cpp:2224`. This is why `offset_cumsum` was replaced with a local cumsum.

**Observation B — the current failure.**

```
models/autoports/openai_gpt_oss_120b/tt/fused_decoder.py:290: in _run_chunk
    dispatched, metadata = dispatch(
models/demos/deepseek_v3_d_p/tt/moe/tt_dispatch.py:273: in forward
    ) = ttnn.experimental.deepseek_prefill.dispatch(
E   RuntimeError: TT_FATAL @ .../tt_metal/fabric/control_plane.cpp:2224: this->fabric_context_ != nullptr
E   info: Trying to get un-initialized fabric context
E   backtrace:
E    --- tt::tt_fabric::get_fabric_topology()
E    --- ttnn::ccl::get_usable_topology(...)
E    --- ttnn::operations::experimental::deepseek_prefill::dispatch::dispatch(...)
```

with `device_params = {'fabric_config': None, 'trace_region_size': 100000000}` and
`mesh_device = MeshDevice(1x1 grid, 1 devices)`, and the call-site kwargs recorded as
`{'cluster_axis': 0, 'dispatch_group_size': 1, ...}`.

The same log records `expert_offsets_tensor` as
`ttnn.Tensor([[0, 32, ..., 4160, 4192]], shape=Shape([1, 128]), dtype=UINT32, layout=ROW_MAJOR)` —
i.e. the local cumsum produced a plausible, 32-aligned, monotone offset vector before the crash.

**Observation C — the focused fabric workaround was tried and failed before model execution.**
The parent investigation set `FabricConfig.FABRIC_1D` before opening `MeshShape(1,1)` on the
four-board P150-class Blackhole host. Fabric initialization timed out after 10 seconds on Device 3
waiting for `LOCAL_HANDSHAKE_COMPLETE`; routers remained `STARTED`, consistent with an absent or
failed remote Ethernet handshake. The decoder never ran. `tt-smi -ls --local` was healthy
afterward. This refutes `FABRIC_1D` as an operational workaround on the target topology; it does
not replace the source proof in H1b, which shows the op would still fail if initialization happened
to succeed elsewhere.

---

## 3. Headline finding

### H1 — `deepseek_prefill.dispatch` / `combine` cannot execute on a 1×1 mesh, with or without fabric

This finding has one decisive fabric-independent blocker (H1b), two ways the fabric-free setup
reaches the same missing-context failure (H1a/H1c), and a corroborating coverage gap (H1d). They
are separated so that the observed first fatal is not mistaken for the whole causal chain.

#### H1a (P) — the reported symptom: `std::optional::value_or` evaluates its argument eagerly

`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:195`

```cpp
tt::tt_fabric::Topology topology_ = topology.value_or(tt::tt_fabric::get_fabric_topology());
```

`value_or`'s argument is an ordinary function-call argument, so `get_fabric_topology()` runs on
**every** call, including when `topology` is engaged. And it always *is* engaged at this call site:
`dispatch.cpp:50` computes `auto topology_ = topology.value_or(tt::tt_fabric::Topology::Linear);`
and passes that concrete value at `dispatch.cpp:57`. `combine.cpp:46` / `:53` mirror it.

`get_fabric_topology()` (`tt_metal/fabric/fabric.cpp:509-512`) dereferences
`control_plane.get_fabric_context()`, which is the `TT_FATAL` at
`tt_metal/fabric/control_plane.cpp:2223-2224`.

The recorded backtrace matches this exactly (`get_fabric_topology()` ← `get_usable_topology` ←
`dispatch::dispatch`), so the mechanism is **confirmed by the hardware trace**, not just by reading.

The tree already knows about this hazard and works around it per-op. `dit_fused_distributed_rmsnorm_device_operation.cpp:397-403`:

> `// get_usable_topology reaches into the fabric context, which is null when the op runs on a`
> `// single device with fabric uninitialized (TP=1, ring_size==1). At num_devices==1 there is no`
> `// ring / all-gather and the topology is never used ... so skip the fabric query and use a`
> `// harmless default.`

`dit_fused_distributed_groupnorm_device_operation.cpp:247-248` has the same guard. The
`deepseek_prefill` ops do not.

**This one line is the whole of the reported symptom, and it is also a genuine upstream C++ bug**:
even a caller who supplies a concrete `topology` pays for a fabric lookup it does not need.

#### H1b (P) — the decisive one: `get_neighbors` fatals on a 1×1 mesh *regardless of fabric*

Both program factories call the shared CCL neighbour helper at top level, before any `num_links`
or remoteness gate:

* `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/dispatch_program_factory.cpp:492-493`
* `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/combine_program_factory.cpp:173-174`

```cpp
const auto [neighbors, directions] =
    ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);
```

Trace it for our case — `mesh_view.shape() == (1,1)`, `mesh_coordinate == (0,0)`,
`topology == Linear`, `axis == 0`:

1. `moe_utils.cpp:23-26` — `get_boundary_mode(Linear)` → `BoundaryMode::NONE`.
2. `moe_utils.cpp:61` — `mesh_coordinate.get_neighbor(shape, ±1, 0, NONE)`.
3. `tt_metal/common/mesh_coord.cpp:178-183` — `case BoundaryMode::NONE: if (new_pos < 0 || new_pos >= boundary) return std::nullopt;`
   With `boundary = shape[0] = 1`, `new_pos ∈ {-1, +1}`, **both are out of range → `nullopt`**.
4. `neighbors` stays empty.
5. `ttnn/cpp/ttnn/operations/ccl/common/host/moe_utils.cpp:93` —
   ```cpp
   TT_FATAL(!neighbors.empty(), "No neighbors found");
   ```

`get_neighbors` reads only the mesh shape and the topology. **It never touches the fabric context.**
Initialising fabric therefore does not help: it converts the reported error into
`TT_FATAL: No neighbors found` from the program factory.

This is the single most important fact in this report, because it invalidates the workaround the
brief asked us to assess.

#### H1c (P) — two more unconditional fabric-context reads inside the program factories

Even with H1a and H1b hypothetically removed:

* `dispatch_program_factory.cpp:511` — `get_tt_fabric_packet_header_size_bytes()`, guarded only by
  `if (operation_attributes.num_links > 0)` at `:509`; and `dispatch.cpp:46-49` requires
  `num_links ∈ [1,4]`, so the guard is always true.
* `dispatch_program_factory.cpp:539` — `get_tt_fabric_max_payload_size_bytes()`, ungated.
* `combine_program_factory.cpp:170` and `:501` — the same two calls.

Both helpers (`tt_metal/fabric/fabric.cpp:67-76`) go through `control_plane.get_fabric_context()`,
i.e. the same `TT_FATAL`. Observation A's backtrace shows `get_tt_fabric_max_payload_size_bytes()`
hitting it for real, via `all_gather`. So this is a confirmed live path, not a theoretical one.

#### H1d (P) — the configuration has zero test coverage anywhere in the tree

`models/demos/deepseek_v3_d_p/tests/pcc/mesh_configs.py:60+` (`ALL_MESH_CONFIGS`, the single source
of truth for every dispatch/combine test) starts at `(2, 1)` and contains **no 1×N and no 1×1
entry**. Every entry uses a `FABRIC_2D*` config. The only in-tree caller of `dispatch_group_size=1`
is `fused_decoder.py` itself (`:133`, `:191`, `:197`, `:211`).

By contrast, the three ops the fused decoder uses that *are* fabric-free all have explicit
single-chip coverage that pins fabric off:

| op | single-chip test | fabric |
| --- | --- | --- |
| `masked_bincount` | `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_masked_bincount.py:51-56` | `DISABLED` |
| `unified_routed_expert_moe` | `tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_swigluoai_routed_expert.py:35-37` | `DISABLED` |
| `post_combine_reduce` | `tests/ttnn/nightly/.../test_deepseek_moe_post_combine_reduce.py` | bare `device` fixture |

**Interpretation.** The 1×1 configuration was never a supported mode of these ops. The token
permutation that `dispatch` performs exists to move tokens *between chips*; at
`dispatch_group_size == 1` there is nothing to move, and the op has no local-only path
(no `local_only` / `ring_size == 1` branch exists anywhere under `deepseek_prefill/`).

---

## 4. Confirmed blockers sitting behind the reported one

These do **not** explain the observed `TT_FATAL` — the wrapper crashes at `dispatch.cpp:57`, before
`ttnn::prim::prefill_dispatch` at `:63` ever runs `validate()`. They are listed because each is a
hard, source-certain `TT_FATAL` that fires the instant H1 is bypassed, and because both are caused
by the same omission: `fused_decoder.py` skips a preparation step the shared caller performs.

### H2 (P) — `dispatch` requires ROW_MAJOR indices; the fused decoder passes TILE

`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/dispatch_device_operation.cpp:20-22`

```cpp
TT_FATAL(
    tensor_args.indices_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
    "Indices tensor must be ROW_MAJOR layout");
```

What the fused decoder passes: `fused_decoder.py:287` calls
`self.router(routed_input, use_throughput_experts=True)`. For every shape this test exercises the
router takes the **non-fused** branch — `models/demos/gpt_oss/tt/topk.py:126` only takes the fused
path when `actual_tokens == 32`, and `_run_chunk` pads to a multiple of 64, giving 192 tokens for
the 129-token prefill and 64 for batch-1 decode. The non-fused branch returns
`ttnn.topk(...)` indices, whose spec is fixed to **TILE** at
`ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:386-388`.

Those TILE indices go straight into `dispatch` at `fused_decoder.py:293`. No `to_layout` exists
anywhere in `_local_routing_setup` (`:233-270`) or `_run_chunk` (`:272-296`).

The reference caller does it explicitly, `models/demos/deepseek_v3_d_p/tt/moe/tt_moe.py:697-705`:

```python
# Ensure ROW_MAJOR layout for dispatch compatibility
indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
scores  = ttnn.to_layout(scores,  ttnn.ROW_MAJOR_LAYOUT)
...
indices = ttnn.reshape(indices, (batch_dim, seq_dim, indices.shape[-1]))
```

Note the reshape to 3-D as well; the fused decoder passes a rank-2 `[192, 4]`.

`masked_bincount` at `fused_decoder.py:244` wants the *opposite* — TILE, uint16, 2-D
(`masked_bincount_device_operation.cpp:18-22`) — so the conversion has to be placed between the two
consumers, not applied globally. `_local_routing_setup` also dropped the reference's defensive
`if layout != TILE: to_layout(TILE)` (`tt_moe_routing_setup.py:214-215`); that is currently harmless
only because `ttnn.topk` happens to emit TILE.

### H3 (P) — `post_combine_reduce` requires ROW_MAJOR weights *and* indices; the fused decoder passes TILE

`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_device_operation.cpp`

```
:87   TT_FATAL(weights.layout() == ttnn::Layout::ROW_MAJOR, "weights must be ROW_MAJOR");
:88   TT_FATAL(weights.dtype()  == DataType::BFLOAT16,      "weights must be bfloat16");
:91   TT_FATAL(indices->layout() == ttnn::Layout::ROW_MAJOR, "indices must be ROW_MAJOR");
```

`fused_decoder.py:320-327` passes `scores` and `indices` through `_as_dram` (`:226-231`), which
calls `ttnn.to_memory_config` — that changes the memory config and **not** the layout. `scores` is
the output of `ttnn.softmax` inside `topk_router` (`topk.py:37`), i.e. TILE bfloat16; `indices` is
still the TILE `ttnn.topk` output.

Same root cause as H2 and the same fix location (`tt_moe.py:699` converts `scores` too).

Secondary consequence, worth noting because it survives a naive fix: `post_combine_reduce` also
requires `weights.padded_shape[dim] == combine_output.padded_shape[dim]` for every non-expert,
non-channel dim (`:59-71`). A TILE `[192, 4]` scores tensor has padded width 32, not 4. Converting
to ROW_MAJOR fixes the padding as well as the layout, so one `to_layout` addresses both.

### The minimal fix for H2 and H3

One insertion, between `fused_decoder.py:288` and `:290` — after `_local_routing_setup`, which
*requires* TILE (`masked_bincount_device_operation.cpp:19`), and before `dispatch`, which requires
ROW_MAJOR. It mirrors `tt_moe.py:698-699` exactly:

```python
indices = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
scores  = ttnn.to_layout(scores,  ttnn.ROW_MAJOR_LAYOUT)
```

This is stated for completeness; it does not make the stage run, because H1 is upstream of it.

---

## 5. Direct answers to the questions in the brief

### 5.1 "Is FABRIC_1D on 1×1 supported/safe as a focused workaround?"

Split into three sub-questions, because they have different answers.

**(a) Will this 1×1 P150-class host open with `FABRIC_1D`? — No. (observed)**

The focused contrast in Observation C resolved the source-level ambiguity: initialization timed
out waiting for `LOCAL_HANDSHAKE_COMPLETE`, with routers stuck `STARTED`, before the decoder ran.
This matches the warning in `models/tt_dit/tests/encoders/qwen3vl/common.py:20` that `FABRIC_1D`
on 1×1 has no Ethernet partner and times out router initialization. The device list was healthy
afterward, so this was a failed configuration attempt rather than lasting device damage.

There is a genuine source/test discrepancy worth preserving for upstream diagnosis:
`ControlPlane::initialize_fabric_context()` (`tt_metal/fabric/control_plane.cpp:737-742`) has no
device-count guard; `FabricContext` merely warns for a zero-hop mesh
(`tt_metal/fabric/fabric_context.cpp:131-140`); a host-only routing-table test constructs the 1×1
P150 descriptor (`tests/tt_metal/tt_fabric/fabric_router/test_routing_tables.cpp:1217-1222`); and a
few Python tests pair `(1,1)` with `FABRIC_1D`. None of those proves router firmware can initialize
on this physical four-board topology. The focused run proves that it does not here.

**(b) Does enabling it make `dispatch`/`combine` work? — No. (P)**

See H1b. `get_neighbors` is fabric-independent and fatals first.

**(c) Is `topology=Ring` a way around H1b? — No, and it is actively dangerous. (P)**

It looks tempting, because on a 1-extent cluster axis `ccl_common.cpp:99-103` returns `WRAP`
(it only special-cases extent **2**), so `get_usable_topology` would return `Ring`, and
`get_neighbor(..., WRAP)` (`mesh_coord.cpp:176`) returns the device itself twice — no empty list,
no fatal.

But then the two `directions` flags are set **true** (`moe_utils.cpp:64-65`) and baked into the
kernel's `DIRECTIONS` define (`dispatch_program_factory.cpp:624-631`), while `dst_nodes` filters
both self-neighbours out (`:1000-1005`) so **zero** connection runtime args are appended (`:1034-1037`).
The kernel then consumes RT args per true direction
(`ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp:249-257`,
`build_from_args(rt_args_idx)` inside `if (directions[i])`) and reads past the end of its argument
array. That is precisely the corruption the factory's own INVARIANT comment warns about
(`dispatch_program_factory.cpp:495-501`). Expect a hang or garbage, not an error.

Incidentally this exposes a second latent `ccl_common` issue for 1×1 meshes: a cluster axis of
extent **1** should be `NONE`, not `WRAP`. See §7.

### 5.2 "Direct `unified_routed_expert_moe` reportedly works with `FabricConfig.DISABLED`" — confirmed, and the reason matters (P)

True, and the explanation is that the op is **fabric-free**, not that `DISABLED` initialises
anything. `DISABLED` leaves `fabric_context_ == nullptr` exactly like `None`
(`control_plane.cpp:737-742`; `fabric_context.cpp:245-247` even fatals if you try to build a context
for `DISABLED`). A scan of every host file under
`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/` for fabric symbols finds hits in only
`dispatch/` and `combine/`; `offset_cumsum` is fabric-bound indirectly through its internal
`ttnn::all_gather` (`offset_cumsum.cpp:10`). `masked_bincount`, `unified_routed_expert_ffn/_moe`,
`post_combine_reduce`, `extract`, `insert`, `routed_expert_ffn` and the rest are clean.

**This is the load-bearing fact for the repair**: the expensive, GPT-OSS-specific kernel already
runs fabric-free on one chip. Only the permutation around it does not.

### 5.3 "Does `dispatch_group_size=1` help?" — No (P)

`grep -c dispatch_group_size dispatch_program_factory.cpp` is **0**. For `dispatch` it is purely a
program-cache key; it does not gate a single decision below the API layer. `combine` uses it only to
derive `num_dispatch_groups` (`combine_program_factory.cpp:586-598`), which validates fine at 1.

### 5.4 The local cumsum that replaced `offset_cumsum` is correct at 1×1 (P) — do not change it

Worth stating plainly, because it is the one piece of new code in this area that is *right*.

`offset_cumsum`'s three outputs are produced by
`.../offset_cumsum/device/kernels/reader_offset_cumsum_interleaved.cpp`. With `H == 1` the only
device has `row_idx = 0`, so the guard at `:99` (`if (h + 1 == row_idx)`) never fires, `local_off[]`
stays all-zero (`:84-87`), and the final add at `:131-133` is a no-op. Therefore
`global_dispatch_offsets == expert_region_offsets` **exactly**. Passing `region_offsets` as
`dispatch`'s `expert_offsets_tensor` (`fused_decoder.py:294`) is correct at one source device.

The alignment arithmetic also matches. The kernel does
`prefix += (val + TILE_HEIGHT - 1) / TILE_HEIGHT * TILE_HEIGHT` (`:122`) with `TILE_HEIGHT == 32`;
`fused_decoder.py:253-256` computes `((c + 31) >> 5) << 5`, and `:257-269` takes an inclusive
`ttnn.cumsum` minus the rounded count to get the exclusive prefix. `ttnn.cumsum` on `int32` is
covered by `tests/ttnn/unit_tests/operations/reduce/test_cumsum.py:57`. The 64-row padding at
`fused_decoder.py:274-276` matches `masked_bincount`'s hard requirement that the token count be a
multiple of its 64-core grid (`masked_bincount_device_operation.cpp:28-34`).

Capacity also checks out. For the 129-token prefill gate the working sequence is 160 tiles-padded
then 192 routing-padded; `compute_constants` (`init_helpers.py:495-505`) then gives
`capacity = 192*4 + 32*(128-1) = 4832`, against a worst-case aligned span of `128*32 = 4096`.
The recorded offset vector ends at 4192, comfortably inside. Decode (64 tokens) gives 4320 against
the same 4096 bound.

---

## 6. Ranked repair options

All options are Python-only, as the contract requires. C++ is called out separately at the end.

### Option 1 — Sort and locally regroup routed slots around `unified_routed_expert_moe` *(only in-scope repair direction; prototype first)*

Keep `masked_bincount` → local cumsum → `unified_routed_expert_moe` → `post_combine_reduce`, all of
which are proven fabric-free and single-chip-tested (§5.2). Replace only the two collective ops
with a static-shape, device-only permutation:

1. Flatten the TILE `UINT16` expert ids from `[M,K]` to `[1,1,1,M*K]` and call `ttnn.sort` on the
   last dimension. This part is **source-feasible**, contrary to the first draft's concern:
   `sort_device_operation.cpp:85-117` accepts rank 4, `UINT16`, and widths divisible by 64. Here
   `M` is padded to a multiple of 64 and `K=4`, so the width is a multiple of 256. The sort wrapper
   returns `UINT32` permutation indices for `UINT16` input (`sort.cpp:252-256`), and exact integer
   index behavior is covered for widths 2080–8192 by
   `tests/ttnn/unit_tests/operations/data_movement/test_sort.py:380-448`. Expert ids 0–127 are exact;
   no BF16 typecast is needed.
2. Convert each sorted slot index to its source token row with a two-bit logical right shift
   (`slot // 4`). `logical_right_shift` explicitly supports `UINT32`
   (`binary_op_dtype_policy.hpp:38`; its nanobind contract states division by `2^shift_amt`). Use
   those row ids with `ttnn.embedding` after converting the hidden-state table to interleaved
   ROW_MAJOR BF16. Embedding requires `UINT32` indices and ROW_MAJOR BF16 weights
   (`embedding_device_operation.cpp:33-47`).
3. Map each sorted slot into its 32-aligned expert region:
   `dest = aligned_offset[e] + sorted_position - unaligned_prefix[e]`. The existing counts provide
   both prefixes; scalar `ttnn.gather` supports `UINT16`/`UINT32` indices
   (`gather_device_operation.cpp:69-73`). Build a small inverse-capacity map with `ttnn.scatter`,
   then use an embedding/gather to materialize the aligned `[capacity,H]` buffer. Gaps may point to
   any valid row because the fused expert kernel reads only `count[e]` rows starting at each
   aligned offset (`unified_routed_expert_ffn_device_operation.cpp:160-180`).
4. After `unified_routed_expert_moe`, gather the same live aligned positions, invert the sort
   permutation, reshape to `[M,K,H]`, and call `post_combine_reduce` with ROW_MAJOR BF16 outputs,
   scores, and indices. The fused expert output is TILE BF8_B
   (`unified_routed_expert_ffn.cpp:74-88`), while `post_combine_reduce` requires ROW_MAJOR BF16
   (`post_combine_reduce_device_operation.cpp:85-97`), so this return path needs an explicit
   typecast/layout conversion or H-wide gather/scatter indices.

**Feasibility verdict (I):** the sort and scalar-index stages satisfy public source contracts and
all shapes are static per traced prefill/decode graph. The complete path is **not proven**:
`ttnn.sort` has no trace-capture regression in-tree, gather/scatter require interleaved staging,
the aligned remap needs careful inverse-map construction, and the BF8/TILE expert output must be
returned to BF16/ROW_MAJOR before reduction. Expect several small index ops plus at least two
H-wide data movements over up to `capacity=4832` rows. This is nevertheless the smallest known
Python-only design that preserves the fused SwiGluOai+bias kernel, exact GPT-OSS numerics, no host
readback/fallback, and fabric-free 1×1 execution.

### Option 2 — Arithmetic rank construction instead of sort *(in scope, fallback prototype)*

If `ttnn.sort` cannot be captured or is too slow, compute each slot's local expert rank as
`sum(one_hot * exclusive_cumsum(one_hot, dim=0), dim=-1)` and use the same aligned-buffer and
inverse-map steps as Option 1. `ttnn.scatter` (already used by the router at `topk.py:44`) can build
the one-hot tensor and `ttnn.cumsum` is already in the fused graph. This stays device-only but uses
more ops and temporary memory than sort, so rank it second rather than assuming it is safer.

### Option 3 — Keep the accepted sparse-expert path for the MoE *(source-proven baseline, out of this stage's no-functional-fallback contract)*

Worth stating because the brief's "no dense host fallback" constraint is **already satisfied by the
accepted stage**: `functional_decoder.py` is not dense. It routes through
`models/demos/gpt_oss/tt/experts/{prefill,decode}.py`, which use **`ttnn.sparse_matmul`**
(`prefill.py:91,120,183`; `decode.py:73,99,131`) with a sparsity mask built on device by
`ttnn.scatter` (`topk.py:44`). That computes only the top-4 experts, is fabric-free, has no host
readback, no arch guard, and is already Blackhole-proven in production.

There is also an **indexed** mode on the same op — `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:1075` — that takes a ROW_MAJOR uint16 list of active group
ids and iterates only those, with a dedicated single-device test
(`tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`). `ttnn.experimental.topk_router_gpt`
already emits exactly that tensor shape (`topk.py:169`).

This abandons `unified_routed_expert_moe` and reuses the functional implementation, so the current
stage contract excludes it even though it is device-only and sparse. Keep it as the correctness and
performance baseline, or use it only if the stage scope is explicitly relaxed.

### Option 4 — Use `ttnn.experimental.moe_compute` FullLocal *(structurally ideal, numerically wrong for GPT-OSS-120B)*

`moe_compute` has an explicit local mode: `moe_compute_device_operation.cpp:465-475`

> `// - FullLocal: compute_only=false, cluster_axis=None, only valid on a 1x1 mesh. No CCL`
> `//   options; combine runs as a local reduction with no fabric.`

It is Python-reachable (`moe_compute_nanobind.cpp:240`, `cluster_axis = None`), has an in-tree
single-card test that never touches fabric
(`tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py`), and on a
1-device mesh the "sparse buffer" input degenerates to the plain token activations — so **no
dispatch step is needed at all**.

**Why it is not recommended (P):** its activation set is `{SILU, SWIGLU, GELU}`
(`moe_compute/device/hostdevcommon/config.hpp:12`). GPT-OSS needs the clamped OAI variant,
`(clamp(up,±L)+1)·clamp(gate,max=L)·σ(α·clamp(gate,max=L))`, which exists only as
`RoutedExpertActivation::SwiGluOai`
(`unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp:42-43`). The sibling
`ttnn.experimental.moe_gpt` *does* implement that activation and is fabric-free in its host code,
but its API carries no bias tensors (`moe_gpt.hpp:15-24`) and GPT-OSS-120B experts have gate/up/down
biases. Both would change numerics, so both would fail the PCC gate.

### Option 5 — Enable `FABRIC_1D` and keep `dispatch`/`combine` — **REFUTED**

Listed only to close it out. On this target it times out before the decoder runs (Observation C).
Even if initialization succeeded on another 1×1 system, it would clear H1a/H1c but not H1b
(§5.1b), so the op would still fail with `No neighbors found`.

### The C++ fix, for escalation only (out of contract)

Three one-liners would make the ops 1×1-clean and are worth filing upstream:

1. `ccl_common.cpp:195` — make the `value_or` lazy so a caller-supplied topology never triggers a
   fabric lookup. This alone fixes H1a for **every** CCL op, and matches the per-op workaround that
   `dit_fused_distributed_rmsnorm` already carries.
2. `dispatch_program_factory.cpp:492` / `combine_program_factory.cpp:173` — skip `get_neighbors` (and
   the fabric-size queries) when `mesh_view.num_devices() == 1`, exactly as
   `dit_fused_distributed_groupnorm_device_operation.cpp:247-248` does.
3. `ccl_common.cpp:99-103` — return `NONE` for a cluster axis of extent 1, not just extent 2.

---

## 7. Other potential issues

Each of these was checked and is **not** capable of producing the reported failure. Kept for the
record, ordered by how likely they are to matter later.

* **`get_boundary_mode` treats a 1-extent cluster axis as WRAP (P).** `ccl_common.cpp:99-103` only
  special-cases `mesh_shape[axis] == 2`. Any op that resolves `topology` to `Ring` on a 1×1 mesh
  therefore gets `WRAP` and a self-neighbour. Latent for now because `dispatch`/`combine` pass
  `Linear`, but it is the mechanism behind the §5.1c trap.
* **`max_dispatched_tokens_per_expert=4096` is 21–64× larger than the reference value — legal
  today, fragile by construction (P).** `fused_decoder.py:306` hardcodes `_EXPERT_CHUNK_SIZE`;
  `tt_moe.py:491` uses `compute_constants`'s `dispatch_group_size * seq_len_per_chip` (192 for the
  prefill gate, 64 for decode). I expected a CB blow-up and there is none: it only sets `m_tiles`,
  which is an *upper cap*. The kernels pick the real chunk at runtime from the device-read
  per-expert count (`adaptive_chunk.hpp:97-110`, `..._reader.cpp:337`, `..._writer.cpp:176`), and
  the CBs are sized from a fixed `kMaxChunkMTiles = 8 * GRID_Y = 64`
  (`..._program_factory.cpp:130`), not from `m_tiles`. Writes are bounded by both `count_tiles` and
  `M_tiles_full` (`..._writer.cpp:174-176, 270-271`), so there is no OOB.
  **But the margin is thin.** Validate requires `m_tiles <= x.padded_shape[-2]/32`
  (`unified_routed_expert_ffn_device_operation.cpp:103-104`). `m_tiles = 4096/32 = 128`; the decode
  buffer is `capacity = 64*4 + 32*127 = 4320` → **135 tiles**. Seven tiles of slack, and only
  because 128 experts happen to contribute a `32*127` alignment reserve. Any future config with
  `capacity/32 < 128` turns this into a `TT_FATAL`. It should be derived from `dispatch_tokens` or
  `capacity`, as the reference does. (The one argument for a constant is program-cache stability
  across token counts, since `m_tiles` is a cache key — but then it should be derived from
  `_EXPERT_CHUNK_SIZE`'s *capacity*, not from the chunk size itself.)
* **Chunk loop frees the wrong list (P, peak-DRAM only).** `fused_decoder.py:356-360`:
  ```python
  chunks  = ttnn.split(hidden_states, _EXPERT_CHUNK_SIZE, dim=2)
  outputs = [self._run_chunk(chunk) for chunk in chunks]
  output  = ttnn.concat(outputs, dim=2)
  for chunk in outputs:          # <-- named `chunk`, iterates `outputs`
      chunk.deallocate(True)
  ```
  The variable name says the intent was to free `chunks`. Freeing `outputs` after the concat is
  itself harmless, but no split chunk is released until the function returns, so peak DRAM holds
  every chunk *and* every output simultaneously. Only reachable above 4096 tokens, i.e. the
  `GPT_OSS_120B_RUN_CHUNK_BOUNDARIES` and `GPT_OSS_120B_RUN_MAX_PREFILL` gates.
* **Expert weights are transposed twice on the host (P, RSS only).** `_expert_torch_lists`
  (`fused_decoder.py:97-99`) materialises `.transpose(0,1).contiguous()` copies of all 128 experts'
  gate/up/down matrices, and `TtRoutedExpert._convert_and_cache_expert_weights`
  (`tt_routed_expert.py:112,116,120`) transposes them straight back with `w.T.contiguous()`. Net
  orientation is correct; the work is pure waste, at 128 × 3 × 2880² elements. The recorded run
  completed this stage ("Expert weights (convert): 100%|██| 128/128"), so it is not fatal on this
  host — setup cost and peak host RSS only.
* **`TtReduceModule(cluster_axis=1)` while dispatch/combine use `cluster_axis=0` (P, benign).**
  `fused_decoder.py:179`. `tt_reduce.py:135` skips `ttnn.reduce_scatter` entirely unless
  `mesh_device.shape[cluster_axis] > 1`; on a 1×1 mesh both axes are 1, so either value is a
  pass-through and `TtReduceModule` is fabric-free. Matches `tt_moe.py:545`.
* **Decode pads 1 token to 64 and routes 63 zero rows (P, wasteful not wrong).**
  `_ROUTING_GRANULARITY = 64` is forced by `masked_bincount`. The padded rows get the router's bias
  argmax, inflate a few expert counts, and are sliced away at `fused_decoder.py:341-346`. Capacity
  still fits (§5.4). `padding_config` is available on `dispatch` and is not used; supplying it would
  bound the token loop (`tt_dispatch.py:210-215`). Perf item only.
* **`ttnn.cumsum(dim=-1)` on a `[1,128]` tensor round-trips through `permute` (P, perf only).**
  `accumulation_common.cpp:24-53` reshapes rank-2 to rank-4 and permutes the cumulative axis to dim
  0, so the op runs over 128 near-empty tiles. Correct, just not cheap.
* **`counts = ttnn.reshape(histograms, ...)` may alias, and `histograms.deallocate(True)` runs at
  `:333` (I, currently safe).** Every use of `counts` (`:301`, `:317`) precedes the deallocate. Worth
  a comment rather than a change.
* **Stale reference doc (P).** `models/demos/deepseek_v3_d_p/tt/moe/README.md:28` still
  documents `metadata_len = 5` (`src_chip, token_idx, topk_idx, expert_id, weight`). The live contract is `metadata_len = 3`
  and a flat 4-D buffer. Do not use that README as a contract.
* **Unvalidated caller obligations in the DeepSeek contract (P, informational).** Nothing validates
  that `expert_region_offsets` is 32-aligned or consistent with `expert_token_counts`;
  `unified_routed_expert_moe` truncates (`..._reader.cpp:350`) and `combine` ignores the offsets
  entirely and **recomputes** region starts from counts (`reader_untilize.cpp:159-170`). Dispatch's
  overflow guard is against the whole buffer, not the per-expert region
  (`reader_worker_dispatch.cpp:380-387`), so a region overrun corrupts the next expert silently.
  The fused decoder currently satisfies all of these, but any future change to the offset math must
  keep counts and offsets in lockstep.

### Checked and cleared — attractive-looking suspects that are not defects

Recorded so nobody re-opens them.

* **`compute_kernel_config = COMPUTE_KERNEL_CONFIG_LOFI` is *not* a precision regression.**
  `fused_decoder.py:307` forwards `TtRoutedExpert`'s default, which is
  `MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True`
  (`tt_routed_expert.py:42-47, 268`). That looks like a downgrade next to the accepted stage, which
  passes no `compute_kernel_config` at all — but the accepted stage *does* pass a `program_config`
  (`models/demos/gpt_oss/tt/experts/decode.py:87`, `prefill.py`), and
  `matmul_device_operation.cpp:2796-2800` sets `increase_fidelity = !has_program_config && ...`,
  so it resolves to **LoFi as well**. Both paths run the expert matmuls at LoFi. Not a delta.
* **`dispatched.deallocate(True)` at `fused_decoder.py:313` cannot alias the expert output.**
  `unified_routed_expert_moe` writes in place (aliasing its input) only on the TILE/bf8 path
  (`unified_routed_expert_ffn.cpp:75-88`), and `dispatch`'s output layout is **unconditionally**
  ROW_MAJOR — `dispatch_device_operation.cpp:169`, `auto layout = Layout::ROW_MAJOR;`, with no
  branch on the input. `fp8_output=True` would change the dtype, not the layout. The deallocate is
  safe for every reachable configuration.
* **`region_offsets` substituted for `global_dispatch_offsets`** — proved equivalent at one source
  device, §5.4.
* **32-vs-64 rounding, cumsum inclusivity, capacity, the `masked_bincount` 64-row rule, expert
  weight orientation, the `SwiGluOai` constants (α=1.702, limit=7.0), `ttnn.pad` on a batch-1
  decode tile, and the `ttnn.split`/`ttnn.slice` signatures** were each checked against the C++
  contract and are correct. Details in §5.4 and §7.

---

## 8. Focused verify/refute experiments

Ordered so that the cheapest decisive one runs first. Items 1–2 need no hardware.

1. **Refute the workaround, statically (no hardware).** Read
   `dispatch_program_factory.cpp:492-493` and `moe_utils.cpp:93` together. If `get_neighbors` is
   unconditional there, no fabric setting can help. *Already done — H1b.*
2. **Confirm H2/H3 statically (no hardware).** Print `indices.layout()` / `scores.layout()` at
   `fused_decoder.py:288` in a scratch copy, or simply compare against `tt_moe.py:698-699`.
   *Already done.*
3. **Refute the fabric workaround on the target (1 hardware run).** Set `FABRIC_1D` before opening
   the 1×1 mesh. *Already done — Observation C.* The mesh timed out in router initialization before
   model execution, resolving the §5.1a ambiguity and closing Option 5. H1b remains the static proof
   that a hypothetical successful initialization elsewhere would still not make the ops 1×1-safe.
4. **Prototype only the index path first (1 hardware run, no model weights).** For `M∈{64,192}` and
   `K=4`, flatten synthetic `UINT16` expert ids, run `ttnn.sort`, derive `slot//4`, aligned
   destinations, and both inverse maps, then capture/replay that fixed-shape graph. Verify every
   permutation against Torch and require no host readback between capture and replay. This cheaply
   refutes Option 1 before any 120B weight setup if sort, integer remapping, scatter, or trace reuse
   is unsupported.
5. **Establish the fabric-free floor (1 hardware run).** Run the existing single-chip tests
   `test_swigluoai_routed_expert.py` and `test_masked_bincount.py` on this P150. They should pass
   with fabric disabled, confirming that everything except `dispatch`/`combine` is viable at 1×1.
6. **Prototype the H-wide path without experts (1 hardware run).** Apply the forward permutation to
   a uniquely tagged `[M,H]` BF16 tensor, build the aligned-capacity buffer, simulate an identity
   expert op, invert the permutation, and require exact recovery in ROW_MAJOR BF16. Capture/replay
   it and measure the two data movements separately; this adjudicates correctness, layout, and
   whether the regrouping cost already exceeds the sparse baseline.
7. **Cost the alternatives before building one.** Run
   `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py -k gpt_oss`
   for a wall-clock reference on the FullLocal path, and the existing functional-decoder perf gate
   for the `sparse_matmul` baseline. Option 1 is only worth its complexity if it beats the
   `sparse_matmul` number.

---

## 9. What this report does not settle

* Why host/control-plane source permits a nominal zero-hop 1×1 fabric context while this physical
  P150-class topology starts routers that wait for a remote handshake. The operational result is
  settled (it times out); the source/runtime discrepancy is not.
* Whether Option 1's sorted permutation, aligned inverse map, H-wide data movement, and full graph
  are correct, trace-safe, and fast enough. The sort/index primitives satisfy their individual
  source contracts; the composed design is a focused experiment, not a validated path.
* Any numerical/PCC question. Once the graph runs, `doc/functional_decoder/` and the prior
  repo-root `AUTODEBUG.md` cover the precision→routing amplifier analysis for this model; none of it
  was re-checked here.

---

## 10. Stage-resolution addendum

The parent fused-decoder investigation subsequently ran the recommended hardware experiments and
closed the open items in §9:

- local masked-bincount/sort/gather/scatter regroup is correct, trace-safe, and fabric-free;
- the dedicated prefill routed-expert path has functional-vs-fused PCC 0.9997700 and reduces the
  standalone M=192 expert time from 205.512 ms to 31.699 ms;
- final whole-layer prefill is 44.336–45.016 ms versus the 137.401–137.418 ms functional baseline;
- DeepSeek dispatch/combine remains rejected for the source and physical reasons proved above.

Correction to an early parent-side hypothesis: the FullLocal `moe_compute` candidate does support
the exact GPT-OSS OAI SwiGLU formula. It was not rejected for ordinary-SwiGLU semantics. The actual
blocker is its BF4-only expert-weight path, which produced Torch PCC 0.980417 against the required
0.995 fusion-equivalence bar. Its 0.384 ms device time therefore cannot be accepted. The retained
BF16 compact indexed decode path reaches PCC 0.9997081765 to Torch and is documented in
`graph_fusion_assessment.md`.
