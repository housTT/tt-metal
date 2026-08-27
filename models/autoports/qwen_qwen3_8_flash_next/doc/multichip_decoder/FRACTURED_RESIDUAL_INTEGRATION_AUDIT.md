# Fractured residual production-integration audit

Date: 2026-08-27

Scope: `tt/multichip_decoder.py`, its multichip tests, and the fixed P300
`1x2` TP target.  This audit does not modify implementation source.

## Verdict

The fully fractured hyperconnection is production-worthy, but it must be
landed as a **stack-internal residual ABI**, not as an isolated replacement of
`_hyper_mix`.

The smallest correct design that retains the measured `0.234947 ms` QSA
boundary result is:

```text
stack residual S [1,1,4*M,1280] per rank
  -> distributed HC norm
  -> local [M,5120] down/inject partial
  -> AR [M,324]
  -> local up and four-stream mean
  -> AG mixed [M,2560]
  -> existing GDN/QSA or router/active-expert graph
  -> GDN output-N shard, or QSA/MoE RS, [M,1280]
  -> local four-stream injection
  -> next S boundary
```

Replicated `R=[1,1,M,10240]` remains a compatibility ingress/egress only.
Provide explicit `prefill_forward_sharded` and `decode_forward_sharded`
entry points for stacking.  Existing `prefill_forward` and `decode_forward`
can remain compatibility wrappers which fracture once on entry and gather
once on return.  A 48-layer stack must call only the sharded entry points
between layer 0 and layer 47.

This preserves the existing public API without inserting the excluded residual
all-gather at every actual layer boundary.  Gathering after every layer would
not be the measured algorithm and is particularly expensive for prefill.

## Audit of the current in-progress patch

The shared worktree already contains the correct core primitives and weight
direction, but the patch is not runnable end to end at the time of this audit.
The following are integration blockers, not optional cleanup.

1. `_hyper_mix` now accepts only S (`last_dim=1280`), while inherited
   `FunctionalDecoder.prefill_forward` and `decode_forward` still validate and
   pass R (`last_dim=10240`).  No sharded core forward or compatibility wrapper
   exists yet.  The first ordinary direct test will fail before attention.
2. `prefill_forward_host_backed` and `_decode_attention_host` likewise pass R
   directly to the new `_hyper_mix`.  The layer-1 PLE S-to-R-to-S bridge is not
   implemented.
3. `_hyper_inject_crossing` and `_hyper_inject_preserve` still reshape blocks
   to width 2560 and assume a replicated hyper input.  Their new contracts are
   local block `[M,1280]` and local hyper `[4*M,1280]`.
4. `_decode_back_host` still all-reduces routed+shared to `[32,2560]`, slices a
   2560-wide row, and calls the replicated preserve helper.  It must RS to
   `[32,1280]`, slice `[1,1,1,1280]`, inject locally, and return S.
5. `_qsa_prefill`, `_qsa_decode`, and `_moe` correctly changed from AR to RS.
   The GDN output-only shard is also the correct topology: its replicated
   48-head recurrence must not be reduced or partitioned again.
6. General-M math in the new `_hyper_mix` is structurally sound: repeating the
   four norm rows produces token-major `s0,s1,s2,s3` ordering, flattening gives
   `[M,5120]`, and `mean(dim=1)` on `[M,4,1280]` is the stream mean.  It still
   needs hardware gates at M=32 and M=128; the winning probe covered M=1 only.
7. The decisive topology test currently fractures weights from
   `layer.w["mlp_hc_*"]`.  Once production setup has already replaced those
   tensors with local shapes, the helper cannot reshape them back to global
   `[4,2560,...]`.  The test must either use the installed production weights
   directly or build its contender from a separate unmodified baseline layer.
8. The exact non-expert inventory still describes both HC prefixes and
   `gdn_out` as replicated.  The source constants and inventory test therefore
   overstate the delivered allocation and are no longer an exact inventory.
9. The module header and optimization manifest still say attention/MoE output
   all-reduce and replicated hyperconnection.  They must describe persistent
   S, QSA/MoE RS, GDN output-N sharding, and compatibility-only R adapters.

The source should not be accepted merely because the M=1 primitive test still
passes.  That test does not exercise any of the currently broken forward or
host-trace interfaces.

## Exact layout and operation contracts

`M` is padded token rows inside a kernel: decode M is batch (`1` or `32`) and
normal prefill M is `128`.  Logical prefill length remains arbitrary.

| Tensor / operation | Global semantic shape | Per-rank TT shape | Placement / collective |
| --- | --- | --- | --- |
| stack residual S | `[1,1,M,4,2560]` | `[1,1,4*M,1280]` | persistent TP2 within each stream |
| HC norm gain | `[4,2560]` | `[1,1,4,1280]` | setup-time hidden shard |
| RMS pre-AG stats | one row per local stream row | `[1,1,4*M,32]` | stats AG on dim 3 |
| HC down+inject | `[10240,324]` | `[5120,324]` | K/row parallel; AR result `[M,324]` |
| HC up | `[320,10240]` | `[320,5120]` | output/column parallel |
| local mixed | `[M,2560]` | `[1,1,M,1280]` | AG dim 3 before real consumer |
| QSA output partial | `[M,2560]` sum | `[1,1,M,2560]` before CCL | RS dim 3 -> `[M,1280]` |
| GDN output | `[M,2560]` | `[1,1,M,1280]` | `gdn_out` N shard; no CCL |
| MoE routed+shared partial | `[M,2560]` sum | `[1,1,padded_M,2560]` before CCL | RS, then trim to `[M,1280]` |
| injection | `[M,4]` | replicated `[1,1,M,4]` | obtained from packed-324 AR |

For M=1 each HC moves 2,048 B of stats AG, 22,528 B of packed-324 AR,
and 81,920 B of mixed AG per die.  QSA and MoE each add an 81,920 B RS.
The stack-internal QSA layer therefore moves 376,832 B/die across these
collectives; the replicated baseline moved 327,680 B/die but did twice the HC
projection work.  The hardware probe proves the compute saving wins for the
measured QSA-to-router boundary.

For M=128 each HC moves 32,768 + 90,112 + 327,680 = 450,560 B/die.
A residual compatibility gather would move another 1,310,720 B/die, which is
why it cannot appear at every prefill layer boundary.

## Production code map

### Setup after rank patch

Keep `_install_fractured_residual_weights` immediately after the two optimized
rank graphs have been patched and the temporary graph released.

For both `attn_hc` and `mlp_hc`:

```text
norm [1,1,1,10240]
  -> reshape [1,1,4,2560]
  -> mesh_partition dim=3
  -> local [1,1,4,1280]

down_inject [1,1,10240,324]
  -> reshape [1,4,2560,324]
  -> mesh_partition dim=2
  -> local [1,1,5120,324]

up [1,1,320,10240]
  -> reshape [1,320,4,2560]
  -> mesh_partition dim=3
  -> local [1,1,320,5120]
```

For GDN only, shard `gdn_out [1,1,6144,2560]` on output dim 3 to
`[1,1,6144,1280]`.  Keep qkv/b/a/z, convolution, recurrence, and GDN head
geometry replicated.

Rebind every replacement to its existing `weight_group_by_id` group and
`weight_role_by_id` role, then deallocate the global original.  The current
host read/re-upload is setup-only and allowed, but it creates an avoidable
double representation/quantization question.  Prefer reshape plus
`ttnn.mesh_partition` of the represented BFP8 tensor if that operation passes
a focused exact-shard hardware probe.  If the host round trip is retained,
prove rank-local represented values/PCC and change the docstring from
"preserves exactly" unless bitwise equality is measured.

Track every temporary allocation before reshape.  In the current exception
path, `local_down` or `local_up` can leak if their following reshape throws
before the tensor is inserted in `replacements`.

### Shared helpers

Retain the current `fracture_residual`, `gather_residual`,
`_reduce_scatter_block`, general-M `_hyper_mix`, and local `_hyper_inject`
families, with explicit ownership:

- ingress/egress adapters do not deallocate caller-owned input;
- `_hyper_mix` returns the original S as its hyper residual;
- normal injection consumes local block and injection, not the caller's S;
- host crossing injection retains trace-crossing inputs;
- host back injection consumes only its new RS/slice intermediates and retains
  all `HostDecodeFront` fields until trace release.

Do not infer R versus S from padded physical shape.  The public/sharded method
name is the contract.  Shape assertions inside a method are still required.

### Sharded decode core

Add `decode_forward_sharded(S, ...)` rather than trying to reuse the base
validator.  It validates `[1,1,4*max_batch,1280]`, then executes:

1. layer-1-only PLE bridge if needed;
2. fractured `attn_hc` -> gathered mixed `[B,2560]`;
3. existing GDN or QSA (both return local `[B,1280]`);
4. local injection -> S;
5. fractured `mlp_hc` -> gathered mixed `[B,2560]`;
6. existing router and exactly active experts, with MoE RS result;
7. local injection -> output S.

Set `_decode_active=True` around the entire core exactly as
`OptimizedDecoder.decode_forward` does so the represented local weights retain
their tuned decode program configs.  Restore the flag in `finally`.

The compatibility `decode_forward(R, ...)` validates the old shape, fractures
once, calls the sharded core, gathers once, and frees only its adapter-owned
temporaries.  Batch 32 is S `[1,1,128,1280]`, block `[1,1,32,1280]`, packed
`[1,1,32,324]`, and mixed `[1,1,32,2560]`.

### Sharded prefill core

The base prefill loop cannot be called unchanged because it slices, trims, and
concatenates token rows in R.  Implement `prefill_forward_sharded` using the
same chunk plan and state reset, but multiply every residual row interval by
four:

```text
slice start       4*start
logical rows      4*logical
padded rows       4*padded
trim rows         4*logical
concat axis       -2
```

Mixed attention/MoE inputs remain ordinary `[1,1,padded,2560]`, so page
tables, `chunk_start`, valid masks, GDN logical lengths, router logical rows,
and cache updates retain token (not stream-row) units.  This is essential for
sequence lengths 33, 2049, and 262143.

The compatibility prefill wrapper fractures once and gathers once.  Do not
gather every chunk merely to reuse the base trim code.

### PLE bridge

Only zero-based layer 1 needs the bridge:

```text
S [4*M,1280]
 -> gather/reshape R [M,10240]
 -> unchanged exact PLE gate/norm/FIR/state with embedding [M,2560]
 -> replicated add
 -> fracture back to S
 -> fractured attn_hc
```

Apply the same bridge in direct prefill/decode, exact host PLE prefill/decode,
and segmented trace capture.  Request history, staged rows, logical masks,
canonical DRAM PLE state, and the shared L1 workspace remain unchanged.

### Segmented host trace

Use S as the captured stack input and output:

- `_decode_attention_host`: validate `[1,1,4,1280]`, apply the PLE bridge only
  for layer 1, run fractured `attn_hc`, then GDN/QSA local output.
- `_decode_router_host`: local crossing injection, fractured `mlp_hc`, then
  unchanged router/top-k/shared projection.  `work` stays replicated padded
  `[1,1,32,2560]`; `hyper` is S `[1,1,4,1280]`.
- `_decode_back_host`: routed+shared partial -> RS `[1,1,32,1280]` -> slice
  `[1,1,1,1280]` -> local preserve injection -> output S.

`HostDecodeFront` needs no new fields.  Its generic release/corruption loops
already cover the changed hyper shape.  Keep two-phase `warm_programs`, cache
miss freezing, state snapshots, expert service, and workspace serialization.
The new RS/stats-AG/packed-AR/mixed-AG programs must all compile in
`warm_programs` before any older stack trace becomes live.

For a real traced stack, capture layer N using layer N-1's stable local output
as its input; retain that dependency and release traces in reverse order.  PCC
reconstruction should concatenate the two host-read local shards within every
stream.  Do not allocate a comparison-only device gather beside live traces.

## Test changes and mandatory gates

1. CPU/static: exact setup shapes/roles, source fallback audit, corrected
   non-expert inventory, R/S validation failures, and 4x token-row chunk-plan
   arithmetic for all existing non-aligned lengths.
2. Production primitive: compare installed local norm/down/up and GDN output
   shards to the represented global tensors; no test-only replacement weights.
3. Single-layer PCC: R compatibility and S core, prefill/decode, layers 0, 1,
   and 3; compare every meaningful block after reconstructing local outputs.
4. Producer coverage: GDN output-N shard, QSA RS, resident MoE RS, host indexed
   MoE RS, and host prefill-wave RS.
5. Shapes: M=1, batch32 M=32, padded prefill M=128, and logical prompt 33.
6. Paged QSA: unchanged per-rank KV heads, page table/current-position and
   advertised-context trace gates under S.
7. PLE: exact prefill/decode and request-history behavior through the bridge.
8. Stack ABI: at least GDN -> PLE/GDN -> QSA chained entirely in S, one ingress
   fracture and one final gather; compare to the R compatibility chain.
9. Trace: ordinary warmed S decode plus segmented GDN/PLE/QSA S traces,
   two-live-layer workspace allocation test, and the existing 100-step stress.
10. Performance: whole-layer and at least three-layer chained S timing.  Report
    compatibility-wrapper timing separately; it includes ingress/egress and is
    not the stack latency.
11. Watcher and allocation tracking on the final current source, followed by
    fresh profiler/provenance evidence because collectives and op balance have
    materially changed.

## Capacity correction for the new weights

If both HC prefixes and GDN output are physically replaced as above, the
decoder-only exact inventory changes.  The important corrections are:

```text
96 HC norms:
  old [1,10240] BF16 physical [32,10240] = 62,914,560 B/die
  new [4,1280] BF16 physical [32,1280]   =  7,864,320 B/die

96 HC down+inject: half K       saves 183,828,480 B/die
96 HC up:          half output  saves 167,116,800 B/die
36 GDN out:        half output  saves 300,810,240 B/die
```

Total decoder saving is `706,805,760 B/die`; the delivered decoder becomes
`3,479,858,176 B/die` rather than `4,186,663,936 B/die`.  Keeping the future
full-text endpoint allowance replicated is conservative and does not block
this stage, but documents/tests must stop calling the old decoder number the
exact delivered allocation.

## Remaining uncertainty

The source structure needed for a correct implementation is clear.  Hardware
evidence is still required before accepting it because the winning experiment
covered only M=1 QSA -> MLP-HC -> router.  In particular, no inference can
replace GDN output-shard PCC, M=128 distributed-norm/program legality, the PLE
bridge, MoE back-edge RS, or multi-layer segmented trace replay.  Those are the
focused AutoFix gates for the production patch.
