# AUTOTRIAGE

## Diagnosis

The optimized R22 path selects TTNN's width-sharded QKV decode splitter, whose
reader unconditionally looks up the next input-core coordinate after consuming
the final V tile. That final lookup is one past the runtime coordinate table and
trips watcher's unique-runtime-argument bounds assertion. The exact cached
kernel constants reproduce the reported assertion arithmetically. This is an
upstream TTNN kernel defect exposed by the optimized layout; the available
evidence does not justify classifying it as stale state on an idle device.

This is an inspection-only diagnosis. No implementation or TTNN C++ was edited,
and this investigator did not run hardware commands. The main agent owns the
serialized verification and recovery workflow.

## Triage Evidence

The first fresh watcher run used:

```bash
TT_METAL_WATCHER=10 GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
python -m pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py \
-k 'real_weights_prefill_decode or (traced_decode_batch_contract and batch32) or trace_mutable_stable_buffers or bounded_modulo_decode_stress'
```

The main agent's captured console reported a tripped watcher assertion on Device
1, worker logical core `(0,0)` / virtual core `(1,2)`: NCRISC accessed a unique
runtime argument index out of bounds. The current kernel was
`reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp`. The process aborted in
the first sliding-attention real-weight case. Ordinary correctness and profiler
runs on this source had passed.

Inspected source hashes at diagnosis time:

- Decoder: `1160aeac946f27a10ad5c8eb5ab3a0d5dd55fb6b3cc1790953147ed1610a34c9`.
- Tests: `a1eca3e86076b110e347cbe5b6e1aa1deb00aaa7a5a08156c0741f7d2170e02f`.

The saved `generated/watcher/watcher.log` from the failing run contained initial
snapshots and attachment of all four devices at 2.803 seconds, but no flushed
assert line. Its initial Device 1 `(0,0)` snapshot had blank kernel IDs. The
kernel map did register the reported QKV kernel as IDs 248/249, with NCRISC ELF
under cache key `2343107275798316172`. There is no live `tt-triage` dump: the job
had already aborted and recovery completed before this inspection. The assertion
signature is therefore console evidence supplied by the main agent, not a claim
that the incomplete watcher file contains it.

The main agent subsequently supplied an isolated residual-disabled watcher
control. `candidate_runs/watcher_nonresidual_qkv_control.xml` records one passing
sliding-attention shared-cache real-weight case, zero failures/errors/skips,
6.724 seconds total, at `2026-09-05T08:31:03.281264+00:00`. With
`GEMMA4_OPT_RESIDUAL_SHARD_CORES=0`, QKV reaches the interleaved splitter. This
passing control supports the factory/layout-specific explanation.

## Source Evidence

Paths below are relative to the repository root.

1. `models/autoports/google_gemma_4_26b_a4b_it/tt/optimized_decoder.py:575`
   configures QKV input/output on a rectangular width-sharded `8x1` grid.
   `_attention_decode` passes this tensor directly to
   `ttnn.experimental.nlp_create_qkv_heads_decode` at line 1340.
2. `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/nlp_create_qkv_heads_decode_device_operation.cpp:12`
   selects `NLPCreateQKVHeadsDecodeShardedProgramFactory` for that input. A
   non-sharded input selects `NLPCreateQKVHeadsDecodeInterleavedProgramFactory`.
3. `.../device/nlp_create_qkv_heads_decode_sharded_program_factory.cpp:247`
   declares one named runtime argument (`index_in_cores`) and exactly
   `num_x + num_y` runtime varargs. Lines 316–326 install x-coordinate values
   followed by y-coordinate values, with no sentinel coordinate appended.
4. `.../device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode.cpp:244`
   advances to the next shard whenever a shard is fully consumed. At lines
   250–251 it always reads the new x and y coordinate, including after the final
   V tile, before the final NoC read barrier. No end-of-input guard exists.
5. `tt_metal/hw/inc/api/dataflow/dataflow_api.h:108` checks
   `arg_idx < rta_count` only when watcher assertions are enabled. A discarded
   one-past coordinate read can therefore leave ordinary output correctness
   unaffected while failing watcher.

The cached NCRISC file at
`/home/hous/.cache/tt-metal-cache/11231292445245388867/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode/2343107275798316172/kernel_args_generated.h`
contains these exact values:

```text
num_q_heads = 16
num_kv_heads = 8
head_size_num_tiles = 8
num_x = 8
num_y = 1
get_vararg(idx) = get_arg_val<uint32_t>(1 + idx)
```

The producer/consumer ledger is:

| Resource or loop | Producer / size | Consumer / progression |
| --- | --- | --- |
| Named RTA | Host: one value | `index_in_cores` selects the batch row |
| NoC coordinate varargs | Host: eight x values plus one y value | Valid vararg indices `0..8`; y always at index 8 while on row 0 |
| Input Q/K/V tiles | 128 Q + 64 K + 64 V = 256 tiles across eight shards | 32 tiles consumed per shard; Q uses shards 0–3, K 4–5, V 6–7 |
| Last V shard boundary | Final tile completes shard 7 | `(x,y)=(7,0)` advances to `(0,1)` |
| Terminal coordinate lookup | No next coordinate exists | `get_vararg(8+1)` reads unique RTA index `1+9=10`; total RTA count is 10 |
| Output dataflow buffers | Borrow Q/K/V output tensor storage | NCRISC/BRISC write disjoint subtile phases; no FIFO producer/consumer wait in this path |

The out-of-bounds lookup is unused by any later payload read: all V tiles have
already been issued. The first causal failure is this lookup, not a CB wait,
cache update, SDPA, dispatch completion, or teardown.

The alternate sharded subcoregrid kernel also unconditionally increments
`cur_core_idx` and fetches the next x/y varargs after its final V tile at
`.../device/kernels/reader_tm_tile_layout_nlp_create_qkv_heads_decode_on_subcoregrids.cpp:230`.
Moving the same input to an offset grid alone is not a credible fix.

Device numbering does not rebut this source explanation.
`conftest.py:662` opens the requested `1x1` mesh without specifying physical
device IDs; the fixture's mesh shape alone does not prove Device 1 is outside
the allocated mesh. `tt_metal/impl/debug/watcher_server.cpp:113` initializes
watcher state on every cluster chip, including clearing assert status at line
398, and `attach_devices` monitors every cluster chip. The actual failing
mesh-to-physical mapping was not preserved in the supplied evidence.

## Downstream Effects

Watcher intentionally halts on the RTA bounds violation. Process abort and any
subsequent host synchronization or teardown failure are downstream effects.
The main agent reported all four devices healthy after the abort, then performed
a bounded all-device warm reset/list and successful `1x1` mesh open/close smoke.
Those checks establish recovery, not correctness of the sharded QKV kernel.

The prior watcher pass used an earlier decoder source. It does not validate
the newly selected R22 sharded QKV path.

## Proposed Fix

Within this goal's Python-only scope, convert the produced QKV tensor to
L1-interleaved immediately before `nlp_create_qkv_heads_decode`. This selects the
interleaved factory while preserving the sharded QKV matmul and its numerical
policy. Release the consumed sharded QKV allocation promptly, preserving the
batch-32 allocation-lifetime fix. The extra conversion is a required workaround
for an unavailable legal sharded splitter in this checkout, and must be measured.

The upstream kernel correction would guard next-coordinate lookup when no input
tiles remain; this must be applied consistently to rectangular and subcoregrid
variants. That C++ change is outside the user-authorized files and is not part of
this report's implementation recommendation.

Focused verification for the main agent:

1. Run the first failing real-weight watcher case with the scoped interleaved
   boundary, assertions and fallback exceptions enabled.
2. Run watcher correctness/stress for both attention kinds and batch 1/32,
   including mutable trace buffers and bounded modulo-cache stress.
3. Recheck ordinary PCC/cache contracts and warmed traced latency. Compare with
   the correct nonresidual control and all retained candidates; do not retain a
   slower final decoder merely to preserve the R22 layout.
4. Regenerate final profiler and provenance artifacts if this path changes.

## Uncertainty

The runtime assertion's exact PC and `rta_count` were not captured by live
triage; the precise final-V instruction is a source-and-cached-kernel diagnosis,
not a recovered device call stack. Nevertheless, its arithmetic exactly matches
the reported RTA bounds signature, and the residual-disabled watcher control
passes. A watcher-clean scoped-workaround run is still required before closing
this failure. If the failure persists, preserve the live stop-site and physical
mesh mapping for `$autofix` rather than attributing it to idle-device state.
