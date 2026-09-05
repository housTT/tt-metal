# AutoFix: TP4 fractured residual through routed experts

Date: 2026-09-05

Status: **measured rejection at the complete-layer gate**.  The indexed
fractured expert/boundary is legal, correct, watcher-clean, and 1.741x faster as
a coherent micro-chain, but the smallest complete sliding decoder is 0.77236x
the incumbent because maintaining the fracture requires thirteen collectives
per layer.  All temporary product-source changes were reverted; the selected
decoder source is restored byte-for-byte to its pre-experiment SHA256
`7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`.

## Finding under review

Stage review asked whether the viable unfused TP4 path can keep the hidden state
K-fractured through Gemma 4's dynamic top-8 routed expert and the next-layer
boundary. Existing device evidence already covers

`row matmul -> reduce_scatter_minimal_async -> fractured residual -> distributed RMSNorm -> all_gather_matmul_async`

for the four exact row-matmul K shapes used by the decoder. It does not cover the
production MoE consumer, which calls `ttnn.sparse_matmul` for gate, up, and down.
This note diagnoses that missing case and defines the minimum shape-faithful
experiment needed to accept or reject it.

## Source-only diagnosis

### The current placement is incompatible with a fractured input

Production weights are placed as follows on TP4:

| Tensor | Global shape | Placement | Per-device shape |
| --- | --- | --- | --- |
| expert gate | `[1,128,2816,768]` | shard N | `[1,128,2816,192]` |
| expert up | `[1,128,2816,768]` | shard N | `[1,128,2816,192]` |
| expert down | `[1,128,768,2816]` | shard K | `[1,128,192,2816]` |

The production input to gate/up is replicated with local shape
`[1,1,32,2816]`. A fractured hidden state instead has local shape
`[1,1,32,704]`. The sparse-matmul validator requires padded local
`A[-1] == B[-2]`. Directly substituting the fractured activation into the
current N-sharded gate/up therefore fails exactly: local K is 704 versus 2816.
This is a weight-placement contract failure, not evidence that sparse matmul
requires a replicated activation.

Relevant source:

- `multichip_decoder.py`: expert placement and the three production
  `ttnn.sparse_matmul` calls.
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`:
  local padded-K validation and indexed-mode validation.

### A K-sharded, indexed sparse consumer has no static contract blocker

Re-place expert gate/up over K rather than N:

| Tensor | Global shape | Candidate placement | Per-device shape |
| --- | --- | --- | --- |
| expert gate | `[1,128,2816,768]` | shard K | `[1,128,704,768]` |
| expert up | `[1,128,2816,768]` | shard K | `[1,128,704,768]` |
| expert down | `[1,128,768,2816]` | shard K | `[1,128,192,2816]` |

Now the fractured input K=704 matches gate/up K=704 locally. TILE inputs and
interleaved device memory are supported by the sparse operation. A tensor can be
mesh-sharded while remaining interleaved within each device, so the mesh
placement does not violate that local-memory restriction.

Indexed sparse matmul is essential to the candidate. It accepts a UINT16,
ROW_MAJOR list of dynamic expert IDs and returns only the selected experts. The
TTNN unit tests cover arbitrary, non-monotonic IDs for both gate/up and down,
changed index buffers under program-cache reuse, and a preallocated indexed
output. `ttnn.topk` over 128 router logits produces UINT16 indices, which can be
reshaped to `[1,1,1,8]` and converted to ROW_MAJOR without a host round trip.

There is consequently no exact source-level sparse-matmul contract that forces
restoring hidden-state replication. The remaining uncertainties are device
behavior for the candidate shapes, mesh metadata after the collectives,
persistent high-rank output buffers, dynamic indices during trace replay, and
whole-chain performance. Those require the focused device experiment below.

The expanded sparse mode is not a competitive substitute. It materializes all
128 expert outputs even though only 8 are active. The experiment must use
indexed mode and preserve gate-selected top-8 execution.

## Candidate dataflow and exact shapes

Logical batch is 1, decode tile M is 32, hidden H is 2816, intermediate I is
768, expert count E is 128, active experts T is 8, and mesh size D is 4.

| Step | Global/logical tensor | Per-device tensor after the step | Collective |
| --- | --- | --- | --- |
| input residual | `[1,1,32,2816]` | `[1,1,32,704]` | none; shard H |
| router logits | `[1,1,32,128]` | replicated `[1,1,32,128]` | small TP sum, then top-k |
| selected IDs | `[1,1,1,8]` | replicated UINT16 ROW_MAJOR | none |
| selected scores | `[1,1,32,8]` | replicated | none |
| indexed gate partial | `[1,8,32,768]` | `[1,8,32,768]` partial sum | none |
| indexed up partial | `[1,8,32,768]` | `[1,8,32,768]` partial sum | none |
| gate complete/fractured | `[1,8,32,768]` | `[1,8,32,192]` | RS over I |
| up complete/fractured | `[1,8,32,768]` | `[1,8,32,192]` | RS over I |
| GeGLU | `[1,8,32,768]` | `[1,8,32,192]` | none |
| indexed down partial | `[1,8,32,2816]` | `[1,8,32,2816]` partial sum | none |
| weighted expert sum | `[1,1,32,2816]` | `[1,1,32,2816]` partial sum | local weighting/sum |
| expert output | `[1,1,32,2816]` | `[1,1,32,704]` | RS over H |
| final residual | `[1,1,32,2816]` | `[1,1,32,704]` | none |
| distributed RMSNorm | `[1,1,32,2816]` | `[1,1,32,704]` | scalar-stat TP sum |
| next QKV consumer | model QKV output | consumer-defined TP output | AGMM, or next candidate row path |

The two I reductions are required before GeGLU: applying the nonlinearity to a
K-partial gate or up value is algebraically wrong. After reducing/scattering I,
the existing K-sharded down weights match local K=192 exactly. Indexed down
preserves the order of the supplied IDs, so each compact expert output can be
multiplied by its corresponding top-k score before summation.

Do not bake an inferred sparse-output rank into the implementation. The
operation documentation describes an expanded logical form, while the current
profiler records canonical local outputs such as `[1,128,32,N]`; indexed unit
tests use `[1,8,32,N]`. The repro must assert the exact tensor shape returned by
the checked-out build and reduce-scatter its last dimension. The ND
reduce-scatter mapper folds leading dimensions and maps the last dimension to
its internal 4D form, so no static rank limit was found in source.

## Router choice

The first experiment should use the mathematically direct row-sharded router:
each device multiplies local H=704 by its local router rows to produce 128
partial logits, followed by a small BF16 or FP32 all-reduce. Top-k then runs on
the replicated logits. Compare router logits, IDs, and scores against the
incumbent before entering experts; routing is discontinuous, so an aggregate
output PCC alone is insufficient. If the reduced logits alter selected IDs,
increase router precision or use the already-supported hidden AGMM router as a
correctness control.

## Minimal shape-faithful experiment

### Gate A: legality and correctness

Run on the required TP4 mesh with the existing fabric/subdevice setup.

1. Create a global TILE hidden tensor `[1,1,32,2816]`, mesh-shard dimension 3,
   and verify each device sees `[1,1,32,704]` in interleaved memory.
2. Produce router logits on device, run `topk(k=8)`, retain the top values, and
   reshape/convert the indices to UINT16 ROW_MAJOR `[1,1,1,8]`. Use input data
   that selects eight distinct, non-monotonic expert IDs.
3. Place gate/up as K-sharded `[1,128,704,768]` per device and down as
   K-sharded `[1,128,192,2816]` per device.
4. Call indexed `ttnn.sparse_matmul` for gate and up. Assert compact local
   outputs `[1,8,32,768]`; record any actual canonical-rank variation rather
   than reshaping it away before validation.
5. Reduce-scatter gate and up over their last dimension to
   `[1,8,32,192]`, then apply the production GeGLU.
6. Call indexed down with the same dynamic indices, producing partial
   `[1,8,32,2816]`. Apply the matching top-k scores, sum the compact expert
   axis, and reduce-scatter H to `[1,1,32,704]`.
7. Add the fractured residual, run distributed RMSNorm, and feed the exact next
   layer QKV boundary. Cover both the sliding-layer QKV N=8192 path and the
   physical TP4 full-attention N=12288 path, or the exact next decoder path if
   that is cheaper to instantiate.
8. Compare router logits/IDs/scores, gate RS, up RS, GeGLU, down partial, final
   fractured output, and next-consumer output against Torch and the incumbent
   replicated TTNN path. Capture fallback-throw and watcher-clean evidence.

This gate is deliberately end-to-end across the expert and the next-layer
consumer. A standalone sparse-matmul pass does not settle mesh layout,
collective, or boundary correctness.

### Gate B: persistent trace and performance

Only proceed if Gate A passes.

For each of gate RS, up RS, and down RS, create its own persistent triple with
`reduce_scatter_minimal_async_create_intermediate_buffer` and pass
`[intermediate, caller_output, penultimate]` to
`persistent_output_buffers`. Do not share a staging tensor across asynchronous
operations. The expected logical/output shapes are:

| Collective | Input on each device | Caller output on each device |
| --- | --- | --- |
| gate RS | `[1,8,32,768]` | `[1,8,32,192]` |
| up RS | `[1,8,32,768]` | `[1,8,32,192]` |
| down RS | `[1,1,32,2816]` | `[1,1,32,704]` |

Capture the coherent chain in a warmed decode trace. Refresh hidden input,
positions, page-table state needed at the layer boundary, and dynamic router
inputs between replays. At minimum, make two replays select different expert
ID lists and prove both correctness and program-cache reuse; then run at least
100 deterministic/stress replays. A trace that accidentally reuses captured
expert indices is a failure.

Compare the candidate to the current replicated/all-reduce implementation at
the same B=1, S=33/current-position=32 workload. Use at least five warmups and
30 timed traced replays, and report median and a tail percentile. Measure the
entire expert plus next-layer boundary (preferably a complete layer or two
adjacent layers), not the three sparse calls in isolation. Collect a separate
profiler run and generate human-readable and CSV `tt-perf-report` artifacts.
Inspect sparse kernels, CCL, DRAM, compute, host fallback, and layout/data-move
rows before accepting the path.

## Communication estimate and rejection criterion

At BF16, a compact gate or up partial `[1,8,32,768]` is 393,216 bytes and an H
tile `[1,1,32,2816]` is 180,224 bytes. For a four-device ring, the rough
per-rank traffic is:

| Traffic | Approximate bytes per rank |
| --- | ---: |
| current down all-reduce over H | 270,336 |
| candidate gate RS + up RS | 589,824 |
| candidate down RS over H | 135,168 |
| candidate expert total, excluding router | 724,992 |

A fractured full layer saves the all-gather half of the attention, dense, and
expert H reductions, roughly 405,504 bytes per rank, but adds the two compact I
reduce-scatters, roughly 589,824 bytes. The first-order full-layer balance is
therefore about 184,320 bytes per rank worse before router traffic. This does
not prove a latency loss—indexed sparse geometry, overlap, and eliminated data
motion can matter—but it makes a whole-chain benchmark mandatory.

Expanded all-expert outputs would be 6,291,456 bytes each for gate and up at
BF16 and are rejected structurally. The candidate is accepted only if indexed
top-8 execution is correct under changing routes, trace replay is clean, and
warmed whole-chain latency beats the incumbent. If it is slower, record it as
a measured rejected alternative rather than landing extra topology and buffer
complexity.

## Source evidence inspected

- `models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py`
- `tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul_indexed.py`
- `ttnn/cpp/ttnn/operations/matmul/device/sparse/sparse_matmul_device_operation.cpp`
- `ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/`
- Existing multichip profiler CSVs and `AUTODEBUG_FUSED_RS.md`

## Hardware execution

All commands below ran on the required TP4 proxy, a 1x4 mesh of this P300C QB2,
with `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`.  The
candidate used real layer weights and recorded layer activations whenever PCC
was evaluated.

### Gate A: the exact sparse consumer is legal

The exact dynamic consumer returned canonical rank six, not the rank-four form
inferred from older profiler output:

`[1, 1, 1, 8, 32, 1536] -> reshape [1, 8, 32, 1536]`.

The explicit metadata reshape is therefore part of the candidate contract.
After reduce-scatter over N, each device holds `[1,8,32,384]`, split into
gate/up `[1,8,32,192]` tensors.  The existing K-sharded down projection accepts
that local K=192 and the weighted compact result reduce-scatters to local
`[1,1,32,704]`.  Active expert IDs were the non-monotonic set
`[7,108,79,18,21,85,111,96]`.

Correctness against Torch was:

| Boundary | PCC |
| --- | ---: |
| packed indexed gate/up | 0.99996768 |
| GeGLU, worst rank | approximately 0.99992 |
| compact weighted expert output | 0.99990682 |
| next QKV boundary | 0.99994526 |

An apparent physical-tile discrepancy was confined to padded rows.  The valid
logical decode row matched.  Evidence is in `gate_a_contract.json`,
`gate_a_contract.xml`, and `gate_a_packed_diagnostic.pt`.

### Gate B: dynamic routing, persistent trace, and micro-chain performance

The accepted next boundary is the sliding QKV K-shard with local weight
`[1,1,704,8192]`, followed by an N reduce-scatter to local
`[1,1,32,2048]`.  It uses an 11x6 grid, `in0_block_w=2`, per-core N=24 tiles,
and output subblock width 8.  Persistent triples are distinct for packed
gate/up, down, and QKV reductions.

The no-watcher correctness/trace run was:

```bash
GEMMA4_MULTICHIP_FRACTURED_SPARSE_REPRO=1 \
GEMMA4_MULTICHIP_FRACTURED_SPARSE_TRACE=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
timeout 900 python_env/bin/python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_tp4_fractured_indexed_sparse_expert_contract \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse/gate_b_kshard_qkv_rs.xml
```

It passed in 9.16 seconds.  A second route
`[119,80,37,21,30,64,59,97]` also passed, proving that replay did not freeze
the captured index buffer.  A first dynamic-route attempt had PCC 0.2849; the
smallest discriminator localized it to refreshing individual device shards
through `copy_host_to_device_tensor`.  Refreshing one host mesh tensor, created
with the mesh mapper, restored hidden-input PCC to 1.0.

| Gate-B result | Value |
| --- | ---: |
| candidate vs incumbent PCC | 0.9999794060 |
| candidate vs Torch PCC | 0.9999428656 |
| incumbent vs Torch PCC | 0.9999422605 |
| changed-route PCC | 0.9999471114 |
| stress replays | 100/100 bit exact |
| program cache | 22 before, 22 after |
| candidate warmed latency | 0.258290 ms |
| incumbent warmed latency | 0.449673 ms |
| micro-chain speedup | **1.74096x** |

Watcher ran separately with `TT_METAL_WATCHER=10` and
`TT_METAL_WATCHER_DISABLE_ETH=1`; it passed cleanly and all devices closed.
The separate Tracy/tt-perf run used three warmups/replays and produced the raw
CSV with SHA256
`3dd24e226a2bf33193a85081b402a564c0028b62e91bd4c442f89f1124316e50`.
Human-readable tables, selected CSV rows, byte accounting, and provenance are
under `artifacts/fractured_sparse/tt_perf_report/`.

| Per-device maximum | Candidate | Incumbent |
| --- | ---: | ---: |
| total selected device time | 237.192 us | 431.740 us |
| CCL | 57.687 us | 19.080 us |
| matmul/sparse matmul | 111.412 us | 219.090 us |
| RMSNorm | 33.816 us | 26.677 us |
| data movement | 2.968 us | 10.793 us |
| modeled DRAM bandwidth | 93 GB/s (18.2%) | 48 GB/s (9.3%) |

### Rejected AGMM boundary

The functional AGMM continuation was not accepted.  Under watcher, the 11x4
ceiling-divided configuration asserted while reserving a circular buffer at
the first synchronization.  A nominally exact-divisor 8x8 variant deadlocked
and required a bounded targeted reset of UMD IDs 0--3.  A standalone N=8192
AGMM reproduced the watcher circular-buffer assertion, proving the fault was at
the operator boundary rather than in the sparse expert.  The deadlocking 8x8
configuration was not rerun.  Diagnostic logs are `gate_b_watcher.log`,
`gate_b_watcher_exact_divisor_diagnostic.log`, and
`standalone_n8192_agmm_watcher.log`.

### Gate C: product-shaped two-layer boundary

A temporary, opt-in product method/resource bundle instantiated the exact
rank-six reshape, packed indexed experts, independent persistent reduction
buffers, distributed norm, and a real layer-5 full-QKV next boundary.  It
passed eager/trace correctness and then passed watcher cleanly:

| Gate-C result | Value |
| --- | ---: |
| active IDs | `[25,93,85,49,71,36,46,97]` |
| fractured residual vs Torch PCC | 0.9998241690 |
| layer-5 full-QKV vs Torch PCC | 0.9996876410 |
| trace replay | 5/5 bit exact |
| program cache | 60 before, 60 after |
| local residual | `[1,1,32,704]` |
| local next-QKV | 3072 columns per rank |

The temporary source SHA256 was
`75fa40d4b43947e1b00a05770fa210d140e2e0fb330be7c4b2307e0ffc1aac28`.
This API was intentionally removed after Gate D rejected the topology; the
opt-in test-local Gate-C repro is preserved with its JSON, JUnit, and watcher
log.

### Gate D: complete sliding decoder discriminator

The smallest complete composition was:

`input distributed RMSNorm -> K-sharded QKV -> QKV RS -> paged sliding
attention/KV cache -> K-sharded O -> O RS -> distributed RMSNorm -> residual
-> shared-FF distributed RMSNorm -> packed dense gate/up -> gate/up RS ->
GeGLU -> dense down -> down RS -> distributed RMSNorm -> router K-shard + tiny
AR/top-k -> packed indexed gate/up -> gate/up RS -> GeGLU -> indexed down ->
route score/sum -> down RS -> distributed RMSNorm -> sum -> distributed RMSNorm
-> final residual scalar`.

This is the actual dynamic top-8 consumer, real paged cache/current-position
path, all residual/norm boundaries, and a trace-safe complete layer.  The first
functionally complete run passed no-fallback after four independently localized
setup defects in the temporary code: rotary head-memory placement, dense-down
local K block width, the norm's logical-row reshape, and conditional expert-row
padding.  Independent rereview then found that this v5 run retained two trace
IDs during readback and used only one warmup/two aggregate iterations.  Its
timing is superseded rather than selected evidence.

Gate-D v6 fixes exactly those evidence defects.  Incumbent and candidate each
use a separate capture, five blocking warmups, and 30 individually timed
blocking replays.  Each trace is released before output readback or the next
capture.  Candidate determinism uses another fresh trace whose preallocated
output is replayed 20 times, released, and only then read.  The run log has no
active-trace allocation/corruption warning.

| Complete-layer result | Value |
| --- | ---: |
| candidate vs selected incumbent PCC | 0.9998051981 |
| stress replays | 20/20 bit exact |
| program cache | 135 before, 135 after |
| warmups / individual timing samples per path | 5 / 30 |
| incumbent mean / median / p95 | 0.652837 / **0.652086** / 0.657987 ms |
| fractured mean / median / p95 | 0.844540 / **0.844279** / 0.847750 ms |
| median / p95 speedup | **0.772358x** / 0.776157x |
| median regression | **29.474% slower** |

The fixed execution order was incumbent, candidate, then candidate stress.
The median gap is approximately 29%, while each 30-sample range is below
0.014 ms, so an order-reversal run cannot plausibly change the decision.

The complete path needs six reduce-scatters, six scalar-stat all-gathers, and
one router all-reduce per layer.  Physical BF16 traffic represented by the
local tensors is 2,129,920 bytes of RS inputs, 532,480 bytes of RS outputs,
12,288 bytes of local norm statistics, 49,152 bytes of gathered statistics,
and 8,192 bytes for router reduction.  For comparison, each incumbent hidden
all-reduce tensor is 180,224 physical bytes.

The temporary packed/product resources also added approximately 163.47 MiB per
device when all experimental Gate-C and Gate-D alternatives were resident:
140.25 MiB expert gate/up, 8.77 MiB Gate-C full-QKV, 11.00 MiB sliding QKV,
3.10 MiB dense gate/up, 0.344 MiB router, and 6.9 KiB norms.  This cost and the
thirteen-collective critical path make the micro-chain win non-composable.

Gate D's temporary source SHA256 was
`d769d045af22574934b49a5bfab10c9ec84816e7ce53b825b36d8c7b63776f42`.
Exact output, cache, trace, and latency provenance is in
`gate_d_complete_sliding_layer_layer_discriminator_v6.json` (SHA256
`1558f9cb583c41b9d609c39e82b8498596cb43c1d8c01c293ccc939e5d2050ca`);
JUnit and stdout are beside it.  The measured test SHA256 is
`9634ba5e5440096d1565f890b0731e5b488a9d74672607e0f9dd8dc7032f26b4`.

The exact final rerun was:

```bash
GEMMA4_MULTICHIP_FRACTURED_LAYER_REPRO=1 \
GEMMA4_MULTICHIP_ARTIFACT_SUFFIX=_layer_discriminator_v6 \
GEMMA4_FRACTURED_LAYER_WARMUPS=5 \
GEMMA4_FRACTURED_LAYER_ITERATIONS=30 \
GEMMA4_FRACTURED_LAYER_STRESS_REPLAYS=20 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
timeout 900 python_env/bin/python -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_tp4_fractured_complete_sliding_layer[blackhole-mesh_device0-device_params0]' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse/gate_d_complete_sliding_layer_discriminator_v6.xml
```

This AutoFix hypothesis was verified: the v5 readback ordering could allocate
device buffers while another captured trace remained live.  Sequential trace
lifetimes eliminate that condition; v6 passed warning-clean without changing
model execution, so the existing separate Gate-C watcher evidence remains
applicable and watcher was not repeated.

## Reproducing the rejected product candidates

The final tree intentionally omits these methods, but the opt-in Gate-C and
Gate-D tests are independently rerunnable using exact unified diffs preserved
beside their results.  Both patches apply directly to the selected source
SHA256
`7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`;
they are alternatives and must not be stacked.

| Candidate | Patch artifact | Patch SHA256 | Applied source SHA256 |
| --- | --- | --- | --- |
| exact Gate C | `artifacts/fractured_sparse/rejected_fractured_gate_c_candidate.patch` | `36804439afa86045706669b5d7bb490995b992188511b2f17fc1e91f0de657d7` | `75fa40d4b43947e1b00a05770fa210d140e2e0fb330be7c4b2307e0ffc1aac28` |
| exact Gate D, including the Gate-C API/resources | `artifacts/fractured_sparse/rejected_fractured_gate_c_d_candidate.patch` | `fed639dadf7b8c3b2f3923b89a7ddc0597d94336990268edffd5b23fa41eeb81` | `d769d045af22574934b49a5bfab10c9ec84816e7ce53b825b36d8c7b63776f42` |

From the repository root, recreate either measured source with:

```bash
source_path=models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py
candidate_patch=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fractured_sparse/rejected_fractured_gate_c_d_candidate.patch
sha256sum "$source_path"
git apply --check "$candidate_patch"
git apply "$candidate_patch"
sha256sum "$source_path"
```

Use `rejected_fractured_gate_c_candidate.patch` instead when reproducing the
Gate-C-only source.  After the opt-in test, restore and verify the selected
source with:

```bash
git apply --check --reverse "$candidate_patch"
git apply --reverse "$candidate_patch"
sha256sum "$source_path"
```

Host verification applied each patch in isolation, matched the two exact
source hashes above, passed `git apply --check --reverse`, reverted each patch,
and recovered the selected SHA byte-for-byte.  A direct `git apply --check`
also passed against the actual final worktree.  This provenance repair did not
modify the product source or tests and did not use hardware.

## Final verdict

There is no TTNN sparse-matmul contract blocker: the K-fractured tensor is
legally consumable by the actual dynamic indexed top-8 expert when the
rank-six output is explicitly reshaped and gate/up weights are K-sharded.  It
is also correct, deterministic, cache-stable, traceable, watcher-clean, and
substantially faster over the isolated expert/next-QKV micro-chain.

It is nevertheless rejected because the smallest complete decoder layer is
29.474% slower by the warning-clean 30-sample median than the selected
replicated/all-reduce implementation.  All temporary product APIs and
resources were removed.  The final selected source is byte-identical to SHA256
`7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`;
there is no dead product API, no runtime selector, and no change to the public
layout or context contract.  The opt-in test-local Gates A--D and their exact
artifacts remain as a reproducible rejected-alternative record.

The final restored-source reproduction ran the selected P150x4 sliding path at
logical prefill length 33 with the same no-fallback policy.  Five warmed and 30
timed trace replays passed bit-exact across repeats and all ranks, with
replicated page table/current position and local paged KV shape
`[4,2,64,256]`.  It measured 0.651610 ms against the recorded 0.765691 ms
single-chip optimized baseline (1.17508x speedup, 29.377% TP efficiency).  The
exact record is
`artifacts/trace_sliding_attention_batch1_fractured_rejection_restored.json`
(SHA256
`0cb3da9b57f4152355abbcfa2b5be56432758917e8d103876239bb2629221120`),
with JUnit and stdout in `artifacts/fractured_sparse/`.

Final v6 health was clean: bounded `tt-smi -ls --local` showed UMD IDs
0--3, the bounded 1x4 open/close smoke printed `MESH_SMOKE_OK`, and the
following device list still showed all four P300C boards.  The logs are
`artifacts/fractured_sparse/final_mesh_smoke_v6.log` and
`artifacts/fractured_sparse/final_device_health_v6.log`.
