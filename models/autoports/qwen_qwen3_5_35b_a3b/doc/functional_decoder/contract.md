# Functional layer contract

Target is the pinned text decoder of Qwen/Qwen3.5-35B-A3B, revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`. The exact config is also checked in at
`tests/target_config.json`. There is no image/video path in this layer stage.

## Construction and ownership

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
mesh_device, tensor_cache_path=None, max_batch_size=32,
expert_precision="inherited")` builds one layer on a 1×1 Blackhole mesh.
`state_dict` accepts real HF checkpoint names or their Qwen36-remapped equivalents.
The default precision is validated; `expert_precision="bf16"` exists for the
recorded diagnostic control. Shape reduction is never used.

All40 layers have the same 256-expert top8 MoE with a gated shared expert,
intermediate width512 and hidden width2048. Layers3,7,…,39 have full attention
(Q16/KV2/head256); all other layers have gated delta attention (K16/V32/head128,
causal convolution width4). Both kinds use pre-norm, residual attention and
residual MoE in HF order. Layer0 and layer3 supply the real-weight representatives.

Create inputs, weights, caches, request states and all required compiled
signatures before the first live trace. Slots are fixed address bindings within
a trace. A caller that reorders requests must move their inputs/page tables to
the corresponding captured rows; GDN state must remain with its captured slot.
A decode trace belongs to its batch size, slot tuple, input/cache addresses and
metadata shapes. Reuse that trace across requests by refreshing stable buffers;
do not release or recapture it at request boundaries.

## Prefill

`prefill_forward(hidden_states, *, valid_len=None, page_table=None, start_pos=0,
request_slots=None)` accepts BF16 device activations `[B,T,2048]`, with
`1 <= B <= max_batch_size`. `valid_len` defaults to T, and may select a shorter
positive logical prefix. The returned `[B,valid_len,2048]` contains every real
output row. `start_pos` is the absolute length of an already cached prefix;
`start_pos + valid_len` must not exceed262144. A continuation retains state;
a fresh GDN request requires `reset_request_state(slots)` before its prefill.

The implementation plans bounded chunks: full-attention512/128 at 128-token and
page-aligned starts, GDN128 with a native inner recurrence chunk32. Remaining
real tokens use the exact device decode path. Padding is never advanced into
recurrent state or exposed as K/V. This accepts valid non-aligned logical lengths.
Full attention reads the whole causal prefix while chunk-local page-table slices
control writes at absolute positions. Device RoPE lookup handles the64 rotated
coordinates of each256-wide head, with the checkpoint's theta10000000.

An input physical bucket may be larger than `valid_len`. Returned tensors retain
that physical allocation as padding while exposing only the logical output shape.
To reuse an existing trace across different logical lengths, prepare the physical
input bucket and its eager chunk/tail paths first. Device indices and tensor-valued
attention offsets allow unseen logical lengths and continuation starts in that
bucket without new programs. `physical_prefill_length(n)` rounds to128 for this
caller preparation; unpadded logical inputs also work before trace preparation.

## Paged cache and decode

`bind_paged_kv_cache(key_cache, value_cache)` binds caller-owned matching BF16,
tiled DRAM `[physical_pages,2,page_size,256]` tensors. Page size is a positive
multiple of32. Full-attention calls require an int32 row-major device page table
`[B,logical_pages]`; entries map each logical page to a valid physical page.
The caller owns unique pages for live requests and must provide enough capacity
for every read/write position. Unused pages may contain stale values: current
positions and causal masks define validity, not zero contents.
Chunk attention requires a32-byte page-table stick. Prefill pads table columns
to a multiple of8 on device when necessary, after validating original capacity.
Those extra entries lie beyond the logical causal prefix and do not acquire pages.
Original tables still drive cache updates and chunk page gathers. Decode selects
32-token attention chunks when the logical page capacity is not divisible by64,
so its final read never requires a nonexistent page-table entry.

`decode_forward(hidden_states, *, position_tensor=None, page_table=None,
request_slots=None)` consumes `[B,1,2048]`. Full attention requires int32
row-major device positions `[B]`, each in `[0,262144)`, denoting the token being
written. Each row must already contain all earlier prefix K/V. Metadata contents
are refreshed outside replay; there is no host read of positions/page IDs inside
forward. Values and ownership are caller preconditions. GDN instead consumes
its selected slot's convolution/recurrent history and has no paged K/V or RoPE.
Its sequential recurrence inherently advances one token per decode call.

`request_slots` defaults to `range(B)` and must identify distinct slots within
setup capacity. Full-attention state is externally paged; slots only bind batch
rows. `reset_request_state(request_slots=None)` zeros selected GDN state in place
outside capture and preserves buffer addresses. Explicit prefix continuation
must not reset. Caller-owned full-attention pages do not need zeroing for a fresh
request if its complete new prefix is written before decode.

## Runtime and trace boundary

Forward contains Python metadata/chunk planning and TTNN device operations only.
Torch, `ttnn.from_torch`, `ttnn.as_tensor` and `ttnn.to_torch` belong exclusively
to setup or named test boundaries. `tests/runtime_audit.py` rejects those calls
and any dispatched Torch operation during each measured functional pass.

The tests warm eager and decode signatures, synchronize, capture decode with
program-cache misses forbidden, restore warmed recurrent state, and replay with
changed stable inputs/positions. They include allocation tracking and program
cache allocations. The exact final output allocation alone is acknowledged:
it is read before replay and holds no persistent model/cache/program state.
All input/cache/state buffers predate capture; transient eager outputs are read
and released before replay. The same-instance test prepares its physical paths
before capture and keeps one trace live through changed requests and continuation.

The native-context HF reference computes every K/V or recurrent contribution,
but only selected prefill queries and their MoE outputs. The TT side still runs
the complete layer for every token. Small-context dense-HF controls validate this
reference decomposition; this is layer correctness evidence, not generation or
full-model accuracy.
