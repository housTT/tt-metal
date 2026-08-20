# AutoDebug: fused and persistent CCL candidates

## Headline findings

1. **Explicit persistent `all_reduce_async` is decoder-applicable and is not
   blocked.**  The selected decode row projections already produce BF16,
   TILE, L1 width-sharded `[1,1,32,5120]` partials on 8 or 16 cores
   (`optimized_decoder.py:481-526,965-980`).  That is precisely the class of
   input accepted by the explicit-buffer overload.  On Blackhole the minimal
   kernel rejects DRAM input, but it accepts this L1 path
   (`all_reduce_async_device_operation.cpp:18-72`).  Its persistent buffer
   must use a superset of the output core grid and have at least four times the
   output shard volume.  Therefore the exact buffers are:

   - 8-core linear-MLP-down partial: output shard `[32,640]`, buffer shard
     `[32,2560]`;
   - 16-core output-projection/full-MLP-down partial: output shard `[32,320]`,
     buffer shard `[32,1280]`;
   - in both cases the tensor-level buffer is `[1,1,32,20480]` BF16/TILE/L1,
     while the returned output remains `[1,1,32,5120]` in the original output
     memory config.

   The ordinary `ttnn.all_reduce` is already asynchronous, but it calls the
   semaphore-managed composite overload (`all_reduce.cpp:16-56`).  For a
   sharded input that implementation explicitly converts to interleaved and
   later converts back (`all_reduce_async.cpp:166-180,422-424`).  The explicit
   persistent overload instead enters `ttnn::prim::all_reduce_async` directly
   (`all_reduce_async.cpp:453-482`) and can remove those material conversions.
   It is therefore a distinct candidate, not redundant API spelling.

   A concurrent exact-shape harness, written outside this inspection-only
   pass, corroborates the source conclusion in `persistent_attempt1.log`: all
   four replicas passed at PCC 0.9999957; 8 cores measured 60.082 us standard
   versus 25.641 us persistent (2.343x), and 16 cores measured 60.422 us versus
   23.943 us (2.524x).  This microbenchmark is actionable evidence, not yet a
   whole-layer result.

2. **`matmul_reduce_scatter_async` has a real program-family incompatibility,
   but not a shape, dtype, or hardware blocker.**  Validation accepts only
   `MatmulMultiCoreReuseMultiCastProgramConfig` (2-D multicast), rejecting the
   decoder's selected `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
   (`matmul_reduce_scatter_async_device_operation.cpp:23-49`).  The API also
   unconditionally requires caller-owned intermediate and output buffers and
   returns a fractured `[1,1,32,1280]` RS result
   (`matmul_reduce_scatter_async.cpp:12-63` and device operation lines 175-206).
   These facts prevent a drop-in substitution, but they do not justify
   rejecting fusion.  The decoder already retains interleaved TP-sharded
   copies of every row weight (`multichip_decoder.py:123-137`), so an exact
   2-D-multicast fused A/B can be built without repacking weights.

3. **The earlier residual-dtype explanation is not a blocker for this API.**
   Plain `matmul_reduce_scatter_async` produces the requested BF16 matmul
   output independent of BFP4/BFP8 weight storage.  Residual addition can
   occur after RS in the fractured family or after a following AG in the
   replicated family.  The optional bias/addcmul restrictions documented for
   the separate *minimal strided* fused API do not apply to a plain fused
   matmul+RS experiment.  Thus a BF16 residual need not be converted to the
   weight dtype.

## Decoder-faithful fused experiment

Test all three material row-projection shapes, not only a generic probe:

| Role | Per-device A | Per-device B | Matmul output | RS output | weight dtype |
|---|---|---|---|---|---|
| linear/full attention output | `[1,1,32,1536]` | `[1,1,1536,5120]` | `[1,1,32,5120]` | `[1,1,32,1280]` | BFP8 |
| linear MLP down | `[1,1,32,4352]` | `[1,1,4352,5120]` | same | same | BFP4 |
| full MLP down | `[1,1,32,4608]` (256 inert columns owned internally) | `[1,1,4608,5120]` | same | same | BFP8 |

Start with interleaved DRAM operands/output so the experiment isolates legal
fusion before attempting layout tuning.  A source-consistent initial program
is an 8x1 2-D multicast grid with `per_core_M=1`, `per_core_N=20`,
`out_subblock_h=1`, `out_subblock_w=1`, `out_block_w=10`, and
`transpose_mcast=False`.  Use role-specific `in0_block_w` values 6, 17, and 9
for K tile counts 48, 136, and 144 respectively.  Place RS workers beginning
at `(0,1)`, use `dim=3`, `num_links=2`, Ring topology, an all-worker
subdevice, three global semaphores, persistent intermediate
`[1,1,32,5120]`, and persistent output `[1,1,32,1280]`.  Preserve the
decoder's LoFi MLP compute policy and current projection policy.

Measure three controls with identical inputs/weights and trace protocol:

1. 2-D multicast `ttnn.linear` + `reduce_scatter_minimal_async` + AG;
2. `matmul_reduce_scatter_async` with the same 2-D matmul config + AG;
3. current DRAM-sharded matmul + `ttnn.all_reduce`.

Control 1 versus 2 isolates fusion.  Control 2 versus 3 includes the cost of
leaving the selected DRAM-sharded local matmul family.  The AG in controls 1
and 2 is necessary for a fair replicated-contract comparison; separately run
the already coherent fractured residual/norm/next-projection family with the
fused RS result and no immediate restore.  A first 2-D-config validation or L1
allocation error is an adaptation result: retry interleaved DRAM, then 8x1
L1/interleaved output, and only then try the 8x2 geometry.  Do not infer that
the fused operation itself is unsupported from rejection of the decoder's
DRAM-sharded program config.

## Alternative fused AG-matmul experiment

If the row-fused local program loses to the DRAM-sharded winner, the installed
`all_gather_matmul_async` is the better second fusion boundary.  It accepts
1-D or 2-D multicast configs, rank-4 input, `dim=3`, and batch dimensions
`[1,1]` (`all_gather_matmul_async_device_operation.cpp:25-80`).  It can consume
the coherent fractured normalized residual directly:

- A is `[1,1,32,1280]` fractured on mesh dim 3;
- persistent gathered A is `[1,1,32,5120]`, width-sharded over exactly four
  cores with shard `[32,1280]` (the persistent-buffer validator requires the
  number of gather-output shards to equal ring size);
- B is the existing interleaved column-parallel weight `[1,1,5120,N_local]`;
- `N_local` is 4352 for the linear packed projection, 3584 for the full packed
  projection, 4352 for linear MLP gate/up, and 4608 for full MLP gate/up.

Use an 8x1 `MatmulMultiCoreReuseMultiCast1DProgramConfig` with
`in0_block_w=20`, `per_core_M=1`, `mcast_in0=True`, and role-specific
`per_core_N` 17/14/17/18.  Start with the established Ring settings
`num_links=2`, four workers per link, two buffers per channel, global AG
semaphores, a barrier semaphore, and CCL core offset `(0,4)`.  For the shared
MLP lhs, fuse AG with gate and feed the returned gathered tensor to the
separate up matmul; gathering independently for gate and up would invalidate
the movement comparison.  Compare this inside the complete fractured
residual + distributed RMSNorm + next-consumer graph, never with an immediate
restore to the replicated residual.

The installed API permits `persistent_output_buffer=None`, and production
Qwen/Llama call sites use that form, so a persistent-buffer allocation failure
is not a fused-AGMM blocker.  Retry with `None` to isolate fusion, then restore
the exact four-shard buffer to measure preallocation separately.

## Persistent all-reduce integration plan

- Load one all-worker subdevice manager before trace compilation.  The minimal
  kernel subtracts the output shard grid from the subdevice and needs one free
  worker core per link (`all_reduce_async_program_factory.cpp:202-230`), so an
  output-only subdevice is invalid.
- Allocate two independent buffer/semaphore slots per decoder instance, one
  for attention output and one for MLP output.  Reusing a slot before its
  dependent residual consumer has executed risks overwriting the globally
  attached reduction CB.  A repeated trace may reuse the same two stable
  addresses after queue ordering is established.
- Keep this path decode-only.  The Blackhole validation explicitly rejects a
  DRAM input; prefill should retain the current composite all-reduce unless its
  row partial is first proven L1-resident without a harmful conversion.
- Run layer correctness and repeat-PCC before timing.  Then compare complete
  traced decode, because the 24--26 us isolated result can be offset by
  subdevice setup, buffer conversions, or changed downstream memory configs.
- Account for 2.5 MiB/device/layer if two BF16 `[32,20480]` buffers are owned
  per layer.  A future stack should cycle shared CCL buffers rather than
  multiplying that reserve by 64; until then, include the physical reserve in
  the context contract.

## What is proved versus still uncertain

Proved from source and the exact-shape microprobe: the explicit persistent
all-reduce is legal for both 8- and 16-core decoder partials, numerically
correct, trace-replayable in isolation, and materially faster than the current
collective micrograph.  It must be integrated and measured in both meaningful
layer kinds.

Proved from source: standard fused matmul+RS cannot reuse the selected
DRAM-sharded matmul program, but every decoder shape is tile/division legal
for an adapted 2-D-multicast experiment; BF16 residual dtype is not a blocker.
Still uncertain until hardware measurement: whether fusion repays the local
matmul/layout regression, whether its persistent buffers remain trace-stable
inside the full layer, and whether fused AGMM can offset the distributed-norm
cost of the coherent fractured family.
