# AutoDebug: Gemma TP4 exact-shape fused matmul + reduce-scatter

## Scope and verdict

This is a fresh-context, inspection-only investigation at checkout
`7f1f91d3167e6e36e547b095ee5ddce8a7514855`. No TT device was opened and no
implementation or test file was edited.

The four Gemma row-parallel shapes are not intrinsically illegal. With the
matmul output width fixed at `N=2816`, a common Blackhole matmul grid of
`(11, 6)` gives `per_core_N=8` and exactly 11 N blocks for every role. The
earlier dense/expert failure saying that 13 output blocks exceeded 11 cores was
an artifact of deriving N geometry from the role's K width.

The smallest conservative exact-shape configuration is therefore one common
grid with role-specific K blocks, not role-specific grids. It is suitable for
an opt-in diagnostic repro. It is **not safe to integrate into the model path
yet**:

1. the current fused operation derives the reduce-scatter return spec from
   input A rather than the matmul output;
2. none of the current local artifacts reaches and proves fused RS correctness
   for the corrected shape;
3. current Blackhole history contains both a proven nondeterministic race in a
   sibling fused MM+RS implementation (#46181) and a newer, closer failure in
   `matmul_reduce_scatter_async` itself (#55223/#55375); and
4. a 704-wide fractured output is not by itself a stack-compatible replacement
   for the current replicated residual.

A one-shot PCC pass is insufficient. Keep the candidate as an opt-in repro and
reject production integration unless all correctness, determinism, watcher,
trace, and whole-chain gates below pass.

## What the exact operation must compute

For each role, the intended TP=4 contraction is:

```text
global A       [1, 1, 32, K]
global W       [1, 1, K, 2816]
rank-local A   [1, 1, 32, K/4]       (A sharded on dim 3)
rank-local W   [1, 1, K/4, 2816]     (W sharded on dim 2)
rank partial   [1, 1, 32, 2816]
RS output      [1, 1, 32, 704]       (sum partials, scatter dim 3)
```

Thus `rs_input_shape` denotes the tensor entering reduce-scatter, namely the
matmul output. It must be `[1, 1, 32, 2816]`; it must not be
`[1, 1, 32, K]`. The current worktree correction at
`test_multichip_decoder.py:166` is semantically right.

In tiles, all four roles share `Mt=1` and `Nt=88`; only rank-local `Kt`
changes:

| Role | Global K | Rank-local K | Rank-local Kt |
| --- | ---: | ---: | ---: |
| sliding O | 4096 | 1024 | 32 |
| full O | 8192 | 2048 | 64 |
| dense down | 2176 | 544 | 17 |
| one selected expert down | 768 | 192 | 6 |

The fixed-selected-expert rank-4 repro is shape-faithful to one selected expert
contraction, but it is not evidence that dynamic rank-5 `sparse_matmul` can be
replaced by this fused rank-4 op.

## Why the current helper fails

`tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py` has
several assumptions that happen to fit its original test but not Gemma's four
nonsquare shapes:

- line 131 fixes `core_grid=(8, 6)`, while the model repro originally passed
  `(11, 6)`; that keyword and helper support were present together only on the
  unmerged `c78b0d5959a` branch;
- line 132 floors `rank-local Kt / grid_x` without clamping to one or selecting
  a divisor of Kt;
- line 134 derives `per_core_N` from `rs_input_shape[3]`; the old caller put K
  there, even though N is the matmul output width;
- line 142 sets `out_block_w=per_core_N // 2`, which is not necessarily a
  divisor of `per_core_N`; and
- cleanup at lines 299-300 is not in a `finally`, so any earlier exception
  leaves the sub-device manager/stall group installed for the rest of that
  pytest process.

The current sliding case illustrates the output-block error exactly:

```text
grid_x       = 8
Nt           = 2816 / 32 = 88
per_core_N   = ceil(88 / 8) = 11
out_block_w  = floor(11 / 2) = 5
11 % 5       != 0
```

This is rejected by `validate_matmul_block_and_subblock_configuration` in
`matmul_device_operation.cpp`: Kt must be divisible by `in0_block_w`, and
`per_core_N` must be divisible by `out_block_w`.

Likewise, the old K block calculation gives:

- dense down: `local Kt=17`, candidate block 2, but `17 % 2 != 0`;
- expert down: `local Kt=6`, `6 // grid_x(8) = 0`, so `in0_block_w=0`.

These are helper configuration bugs, not evidence that the shapes are
unsupported.

## The earlier 13-block rejection is refuted

The historical artifact on unmerged commit `c78b0d5959a` reports sliding/full
passing and dense/expert failing because 13 output blocks exceeded grid x=11.
That branch used the role K as `rs_input_shape[3]` and then used that value to
derive `per_core_N`:

```text
dense:  ceil((2176/32) / 11) = 7, then ceil(real Nt 88 / 7) = 13
expert: ceil(( 768/32) / 11) = 3, then ceil(real Nt 88 / 3) = 30
```

The first number in each line is not N geometry. With the correct `Nt=88`,
`per_core_N=ceil(88/11)=8` and `ceil(88/8)=11`, so both roles fit. No
role-specific grid is needed to solve this.

## Proposed exact-shape configuration

### Primary conservative repro

Use this first because it preserves the originally intended Blackhole
placement and keeps K blocks modest:

| Role | Grid | `per_core_M` | `per_core_N` | `in0_block_w` | K blocks | `out_block_h` | `out_block_w` | subblock |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| sliding O | `(11,6)` | 1 | 8 | 2 | 16 | 1 | 8 | `1x1` |
| full O | `(11,6)` | 1 | 8 | 4 | 16 | 1 | 8 | `1x1` |
| dense down | `(11,6)` | 1 | 8 | 1 | 17 | 1 | 8 | `1x1` |
| expert down | `(11,6)` | 1 | 8 | 1 | 6 | 1 | 8 | `1x1` |

Common collective settings:

- `rs_input_shape=[1,1,32,2816]`;
- persistent intermediate `[1,1,32,2816]` in BF16 tile-layout DRAM;
- persistent output `[1,1,32,704]` in BF16 tile-layout DRAM;
- `dim=3`, TP ring size 4, `num_links=2`;
- `reduce_scatter_core_grid_offset=(0,6)`;
- BF16 A, BFP8_B W, BF16 matmul/RS output; and
- HiFi2, approximate math, FP32 destination accumulation, packer L1
  accumulation, matching the current helper.

`matmul_reduce_scatter_async` currently fixes one RS worker per direction per
link. With two links this requests four RS workers. Starting at `(0,6)` places
them on the first four cores below the matmul rectangle on an 11x10 Blackhole
worker grid, so the MM and RS core sets are disjoint.

The conservative K-block rule is:

```python
local_k_tiles = global_k // TP_SIZE // 32
limit = max(1, min(max_in0_block_w, max(1, local_k_tiles // grid_x)))
in0_block_w = max(d for d in range(1, limit + 1) if local_k_tiles % d == 0)
```

The `/ grid_x` limit is a conservative helper heuristic, not an operator
contract. The actual contract is nonzero and divisibility by local Kt.

Set `out_block_w=per_core_N` (or omit it so nanobind applies that documented
default). This is always a legal divisor and avoids the current odd-half bug.
For the primary grid it is 8. `out_block_w=4` is also statically legal, but is
not needed for the first correctness repro.

### Secondary legal block sweep

After the conservative table is stable, these larger role-specific K blocks
are also statically legal and correspond more closely to the maxima already in
the model test:

| Role | Local Kt | Larger legal `in0_block_w` | K blocks |
| --- | ---: | ---: | ---: |
| sliding O | 32 | 4 | 8 |
| full O | 64 | 8 | 8 |
| dense down | 17 | 1 | 17 |
| expert down | 6 | 2 | 3 |

Do not combine this throughput sweep with the initial correctness repair. Keep
only a larger block after isolated correctness and latency evidence.

### 8-column diagnostic fallback

If it is useful to minimize helper changes or compare with the helper's legacy
grid, `(8,6)` is also statically legal for every shape when
`per_core_N=out_block_w=11` and K blocks are selected as divisors. A
conservative table is `[4,8,1,1]` for sliding/full/dense/expert respectively.
This is a useful A/B, but `(11,6)` should remain the primary target repro
because that is the original Blackhole candidate and gives an even
`per_core_N=8`.

## Smallest code path for the local repro

The least duplicated test-only route is to append backward-compatible options
to `run_reduce_scatter_impl` rather than copying the full CCL harness into the
Gemma test:

1. restore an optional `matmul_core_grid=(8,6)` parameter and pass `(11,6)`
   from the Gemma cases;
2. derive persistent buffer widths and N geometry from
   `mm_weights_shape[3]`, never from K;
3. choose a nonzero divisor of rank-local Kt, preferably the conservative rule
   above for the first run;
4. use `out_block_w=per_core_N`;
5. assert the exact local A/W/partial/RS shapes before launch; and
6. wrap sub-device stall-group/manager teardown in `finally`.

The current caller's `rs_input_shape=[1,1,32,HIDDEN_SIZE]` already expresses
item 2 for these cases. Making the helper derive it independently prevents a
future caller from recreating the same K/N confusion.

An even more explicit model-only interface would pass the exact
`in0_block_w` values from the primary table. That avoids disguising a tuned
program configuration as a generic `max_*` heuristic, but adding a program
config override to the shared helper is a slightly larger API change.

## Fused operation contract bug that will likely surface next

`MatmulReduceScatterAsyncDeviceOperation::compute_output_specs` currently does
this:

```cpp
ReduceScatterMinimalAsyncInputs reduce_scatter_tensor_args{
    input_tensors[0], std::nullopt, std::nullopt};
```

That is input A. The program factory, correctly, passes `output_tensors.mm` as
the reduce-scatter input. For nonsquare matmuls the two widths differ. The
current reported RS widths therefore follow K/4 rather than N/4:

| Role | Width inferred from A | Correct RS width |
| --- | ---: | ---: |
| sliding O | 1024 | 704 |
| full O | 2048 | 704 |
| dense down | 544 | 704 |
| expert down | 192 | 704 |

Unmerged commit `c78b0d5959a` changed the spec source to
`tensor_args.persistent_intermediate`, whose required shape is the materialized
matmul output. That is the smallest semantically correct repair available to
the current API, and it should be re-tested rather than rediscovered from the
old branch. A robust follow-up should also validate that the caller-provided
persistent intermediate and persistent output specs match the matmul output
and its scattered result; current fused validation checks the matmul operands
and 2D config but does not establish those persistent-buffer shape contracts.

Do not make the local test K-shaped merely to agree with the current bad output
spec. That recreates the nonsquare bug and is not an exact Gemma repro.

## Artifact audit

The current artifacts do not prove a correct fused RS result:

- `fused_rs_repro.xml`: all four cases fail at Python argument binding because
  the helper has no `matmul_core_grid` keyword;
- `fused_rs_repro_v2.xml`: sliding/full fail during fabric route construction;
  dense fails `Kt(17) % in0_block_w(2)`; expert fails with
  `in0_block_w=0`;
- `fused_rs_sliding_after_reset.xml`: the op gets far enough to return, but the
  host comparison still compares a 4096-wide K-derived tensor with the
  2816-wide matmul oracle; and
- `fused_rs_sliding_shape_fixed.xml`: the corrected width reaches the
  `per_core_N(11) % out_block_w(5)` validation error.

The fabric failure disappearing after a reset is not correctness evidence. The
helper must guarantee teardown after all exceptions, and each crash/assert
investigation should begin from a freshly recovered board.

The historical `c78b0d5959a` JSON says two roles passed after adaptations, but
that branch is not an ancestor of the current checkout and its dense/expert
geometry was still K-derived. It is useful as a source of hypotheses, not as a
current regression result.

## Blackhole correctness/race evidence

### #46181 is not the same exact shape or implementation

`models/demos/gpt_oss/tt/attention/operations.py` gates
`minimal_matmul_strided_reduce_scatter_async` off on all Blackhole devices. The
associated commit `268c5212a17f` records fixed-seed, repeated-run evidence at
`M_tiles=32` (`M=1024` elements): the RS result was the only nondeterministic
op, with cross-run PCC near zero and non-finite-scale garbage around `1e13`.

Gemma's present repro has `M=32` elements, which is **one M tile**, and calls
`matmul_reduce_scatter_async`, not the GPT-OSS minimal strided op. Therefore
#46181 does not by itself prove that the Gemma case races. It does prove that a
single successful Blackhole fused-MM+RS run is an inadequate safety gate.

### Current generic `matmul_reduce_scatter_async` has closer negative evidence

Commit `ab4b9f78dff` (2026-09-04, #55375) added an unconditional
`noc.async_atomic_barrier()` to the 2D matmul in1 sender/writer used by the
generic fused op. Its commit evidence says the prior fused signal atomic could
remain unacknowledged when the kernel retired, producing a watcher-reported
inter-kernel race on a 1x4 Blackhole mesh.

That fix is present in this checkout. However, the same commit explicitly says
the fused repro then reaches a separate deterministic fault: the second RS
worker launches with zeroed runtime arguments. BF16 can issue an illegal read;
FP32 asserts. The fault reproduced with both one and two links, Ring and
Linear, several offsets, with and without a sub-device manager. No passing
regression test was landed because of that second failure.

A source-history check after #55375 finds RS performance changes but no commit
or regression test explicitly closing that second fused-op fault. The current
program still launches multiple RS worker kernels and assigns their runtime
arguments in `reduce_scatter_minimal_async_program.cpp`. Static inspection
cannot establish that the runtime failure has disappeared.

This closer evidence is enough to block production integration even if the
primary Gemma configuration compiles and happens to pass once with watcher
disabled.

## Ranked hypotheses

1. **Verified: helper geometry is the immediate validation failure.** The
   hard-coded 8-wide grid, non-divisor K block calculation, and halved odd N
   block explain the current logs exactly.
2. **Verified: the prior dense/expert core-count rejection is false for the
   corrected output shape.** Correct N geometry makes all four roles 11 N
   blocks on an 11-wide grid.
3. **Verified by source: fused output-spec derivation is wrong for nonsquare
   matmuls.** A/B output widths differ for all four roles.
4. **High risk, requires hardware: the current generic fused op may still hit
   the known second-RS-worker fault or another ordering bug on Blackhole.**
   Existing one-run artifacts never validate corrected RS contents.
5. **Moderate risk: exception cleanup and board state explain some topology
   noise.** `finally` cleanup is necessary, but a reset-only improvement must
   not be mistaken for a fused-op fix.
6. **Integration blocker independent of micro-op correctness:** the 704-wide
   result needs a coherent fractured-residual consumer path (or an explicit
   all-gather measured as part of the full candidate). Dynamic active-expert
   sparse consumption is not covered by the fixed rank-4 expert repro.

## Focused experiments and acceptance gates

Run these serially on Blackhole. A device assert, hang, or watcher report
requires board recovery before the next case.

### Experiment 1: static/config launch isolation

Run one pytest parameter per fresh process with the primary table, tracing off
and watcher on. First prove that each case reaches device execution with exact
buffer shapes. This distinguishes remaining API/spec failures from runtime
failures.

Expected result: no Python binding error; no matmul config fatal; reported and
actual output shape `[1,1,32,704]` on each rank.

If the next error is an output-spec mismatch derived from A, apply only the
spec-source repair described above and rerun. Do not alter shapes to conceal it.

### Experiment 2: one-shot numerical controls

For every role, use identical host A/W data for:

1. Torch global matmul;
2. non-fused rank-local matmul plus reduce-scatter; and
3. fused matmul+reduce-scatter.

Check the rank-local partial matmul before reduction and the concatenated
704-wide RS shards after reduction. Require finite outputs, exact logical
shapes, no untouched/zero regions, and at least the repository's existing PCC
criterion (prefer an explicit `>=0.99` gate). This catches a case where matmul
is good but RS is corrupt.

### Experiment 3: fixed-input dispatch stress

For each role independently:

- compile once, then execute at least 100 eager invocations using fresh/rotated
  semaphores and persistent buffers;
- compare every output with Torch and with run 0;
- require bitwise repeatability where the ring schedule is deterministic, or
  otherwise cross-run PCC indistinguishable from 1 with stable max error; and
- record min/max/absmax so #46181-style explosive garbage cannot hide behind a
  single aggregate PCC.

Then cycle all four shapes for at least 100 rounds in one process to exercise
program-cache transitions and verify `finally` cleanup.

### Experiment 4: trace replay stress

Repeat the exact production capture pattern: eager compile, trace capture,
then at least 1000 fixed-input trace replays for each role and for the mixed
four-role cycle. Validate every replay or copy each result into a rotating host
audit set; checking only the final replay can miss intermittent corruption.

Run watcher-enabled stress first. Watcher-disabled timing is allowed only after
the same binary/config is clean with watcher.

### Experiment 5: fault matrix and controls

The integration target is two-link Ring, but run one-link Ring as a diagnostic
control because #55375's second fault reproduced in both. Also compare the
legacy `(8,6)` config. If only one grid or link count survives, treat that as a
localized hypothesis requiring another stress run, not a general fused-op
validation.

Use `M=1024`/`M_tiles=32` only as an additional Blackhole negative-control
stress. It is useful for comparison with #46181, but it must not replace the
exact Gemma `M=32` evidence.

### Experiment 6: whole-chain gate

After micro-op stress passes, measure and validate the full semantic chain:

```text
row-parallel producer -> RS shard -> distributed norm/consumer(s)
-> residual update -> next layer, including dynamic top-k experts
```

If the only compatible chain immediately all-gathers the 704-wide output, its
full latency must be compared against the incumbent matmul + all-reduce. Never
claim a performance improvement from the fused producer alone.

## Final decision rule

The primary table is a valid candidate for the **local opt-in repro**. It is
not a production fix.

Production integration is allowed only if all four roles pass one-shot and
stress correctness, eager and trace determinism, watcher, repeated program
cache transitions, and the whole-chain contract. Any nonfinite output,
cross-run drift, watcher race, illegal read, device assert, hang, stale-buffer
region, or need to disable watcher is an immediate rejection.

Given the unresolved current-tree evidence, the present decision is:

**restore and run the exact repro; do not integrate the fused Blackhole path.**

## 2026-09-05 addendum: unfused composable producer

The stage review identified a gap in the experiment matrix: the fused producer
had negative evidence and the distributed consumer had positive evidence, but
no test connected the supported unfused producer to that consumer. Source
inspection finds no API blocker to that chain.

`reduce_scatter_minimal_async` computes its output spec from the tensor it
actually receives. A local row matmul produces `[1,1,32,2816]` on every TP4
rank, so scattering dimension 3 produces `[1,1,32,704]`. Its output-topology
implementation explicitly installs `Shard(dim=3)` on cluster axis 1. This is
the same layout consumed by `rms_norm_pre_all_gather`,
`rms_norm_post_all_gather`, and `all_gather_matmul_async` in the already-passing
consumer test. It also avoids the fused op's nonsquare output-spec defect.

The shared CCL test helper cannot be reused by changing `use_non_fused` to
true: that branch still passes the removed `persistent_intermediate_buffer`
and `persistent_output_buffer` keywords. The current binding accepts the
single `persistent_output_buffers` list. The focused model test therefore
calls the current primitive directly and lets it allocate its contiguous ring
staging buffers for the first correctness probe.

The previous `fused_agmm_repro.xml` attention cases were not actually
shape-faithful. The correct projection widths are:

| Consumer | Prior test width | Actual Gemma width |
| --- | ---: | ---: |
| sliding QKV | 5120 | 8192 = 4096 Q + 2048 K + 2048 V |
| full QKV logical | 8192 | 10240 = 8192 Q + 1024 K + 1024 V |
| full QKV physical on TP4 | 8192 | 12288 = 8192 Q + two duplicated 1024-wide K/V pairs |

The consumer parameters now use 8192 for sliding and the actual TP4 physical
12288 for full attention. The old five-pass artifact remains evidence for the
distributed-normalization API and the dense/router/fixed-expert cases. The two
corrected attention cases pass in `artifacts/fused_agmm_qkv_corrected.xml`, so
8192/12288 are now the selected exact-shape attention evidence.

The new opt-in
`test_tp4_unfused_row_reduce_scatter_distributed_consumer_exact_shape_repro`
uses all four row-parallel K widths (4096, 8192, padded 2176, padded 768). For
each, it checks the concatenated reduce-scatter result against the Torch global
row matmul, adds a fractured 704-wide residual skip, performs distributed
RMSNorm, and consumes the result with the exact packed-dense projection width
4352 through fused AGMM. The final concatenated output is also checked against
Torch. Sub-device cleanup is in `finally`.

The serialized P300C QB2 run passed all four cases with fallback throwing;
`artifacts/unfused_fractured_chain.xml` preserves the JUnit evidence. Both the
concatenated RS tensor and final concatenated AGMM tensor cleared PCC 0.99 in
every case, all asserted local shapes matched, device teardown was clean, and
the post-run health check listed all four P300Cs healthy and resettable.

This is intentionally a micro-chain decision gate, not production integration.
The pass establishes that the ordinary row matmul/RS output is numerically and
layout-composable with one exact downstream consumer. It does not solve
the decoder's replicated public input/output boundary, prove a dynamic rank-5
sparse-expert AGMM consumer, establish trace determinism, or demonstrate a
latency improvement. Production remains blocked until those contracts and an
end-to-end timing comparison pass independently.

A warmed latency comparison was not added to this one-shot correctness repro.
The direct RS call deliberately lets the primitive allocate staging, while a
production timing candidate must use persistent/rotating buffers and include
both the boundary conversion and dynamic expert path. Timing the current test
would mix compilation, allocation, and fixture setup and would not answer the
integration decision.
