# Gemma 4 26B A4B optimized full model

| Profile | Warmed prefill-to-logits before -> after | Public-generator TTFT BFP8 -> BF16 | Traced token-out before -> after | Final decode t/s/u | Stack lower bound | Final gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 (1 chip) | 154.223 -> 148.127 ms | 188.518 -> 182.831 ms | 32.701 -> 26.841 ms | 37.257 | 23.416 ms | 14.62% |
| P150x2 (2 chip) | 130.349 -> 126.901 ms | 148.851 -> 147.120 ms | 24.296 -> 21.491 ms | 46.532 | 19.787 ms | 8.61% |
| P150x4 (4 chip) | 106.076 -> 106.326 ms | 139.205 -> 138.561 ms | 21.490 -> 20.102 ms | 49.747 | 19.203 ms | 4.68% |

These are same-workload, full 30-layer, batch-1, prompt-128 results with five
decode warmups and 128 nonblocking replays. Token-out includes final norm,
vocabulary-sharded LM head, split on-device sampling, `tt_out_tok` feedback,
device position/RoPE advance, and unchanged page tables. Its measured loop has
zero host readbacks and zero host synchronizations. The first timing column is
the low-level warmed prefill-to-sampler-ready-logits boundary. Public TTFT is
the actual first-token latency from `Gemma4Generator.generate()` on the same
prompt-128 workload; its controlled BFP8/BF16 comparison uses otherwise final
source. The 0.24% P150x4 low-level prefill delta is noise-level; decode improves
21.8%, 13.0%, and 6.9% in t/s/u on P150/P150x2/P150x4.

This stage optimizes the completed TTNN full-model/generator path and does not
start vLLM integration.

## Selected full-path optimization

The selected change stores the hidden-dimension-sharded embedding table as
BF16 row-major. The prior BFP8 tiled table forced `ttnn.embedding` to convert
the complete 262,144 by 2,816 table to row-major on every decode replay. The
new representation is consumed directly. In a reduced real-shape P150x4 Tracy
capture the embedding family is 9.09 us and the only remaining untilize is
4.37 us, replacing the inherited approximately 1.4 ms full-table conversion.

This is a terminal representation optimization, not a broad datatype search.
The vocabulary-sharded LM head remains BFP8_B. The complete decoder's selected
per-profile weights, math fidelity, BF16 activations, BF16 paged KV cache, BF16
CCL payload, persistent collective resources, and replicated BF16 TILE DRAM
inter-layer residual remain unchanged. No rejected lower-precision policy or
replicated stream was reintroduced.

Actual complete-path measurement initially exposed an avoidable P150
fragmentation failure at 50,623 tokens. The full-attention chunk path retained
its complete `q_heads` allocation until after allocating an identically shaped
concat result. Releasing `q_heads` after the final chunk dispatch and before
concat provides an exact contiguous reuse block. The final-source 50,623-token
nonaligned prefill, final traced position 50,624, and safe rejection of the
following position now pass across all 30 layers. P150 therefore preserves
50,624 tokens; P150x2 and P150x4 preserve 262,144 tokens.

## Mesh, sharding, and collectives

All mandatory proxy profiles use the intended mesh rather than a replicated
single-chip fallback:

| Profile | Full-path policy |
| --- | --- |
| P150 | 1x1, identity reduction, hidden-sharded embedding degenerates to one shard, vocabulary-sharded LM head degenerates to one shard |
| P150x2 | 1x2 TP, persistent one-link Linear BF16 all-reduce, hidden-sharded embedding, vocabulary-sharded LM head/logits |
| P150x4 | 1x4 TP, persistent two-link Ring BF16 all-reduce, hidden-sharded embedding, vocabulary-sharded LM head/logits |

TP2 and TP4 reuse exactly three preallocated all-reduce buffer/semaphore sets
across all 30 layers. Decoder contractions, program configs, sparse-expert
kernels, R22/R0 profile choices, and inter-layer layout are inherited from the
optimized multichip decoder's measured winner. There is no inter-layer gather,
reshard, reduce-scatter, or host materialization.

## Serving-ready traced generator contract

The optimized generator retains explicit caller-visible cache, page-table,
position, prompt-length, active-row, fixed-slot, and sampling state. Prefill
pads internally, so public prompt lengths need not align to tiles, cache pages,
or chunks. Fixed-slot batch 32 with four mixed prompt families passes on the
complete P150x4 stack; every row selects a global-max logit and the minimum
B1/B32 logits cosine is 0.99550. Separate mixed-state coverage retains inactive
rows and changed-only page-table adoption.

Decode remains two coordinated traces:

1. The model trace ends in sampler-ready vocabulary-sharded logits and advances
   persistent position and RoPE inputs on device.
2. The sampling trace keeps local top-32 candidates, implements semantic greedy
   with `k=1`, and aliases its output to the next model trace's `tt_out_tok`.

This same sampler supports top-k/top-p sampling. Trace replay is nonblocking.
Steady token-out makes no host token, position, RoPE, cache, page-table,
synchronization, or readback call. Scheduler page tables are copied only when
changed, into stable device buffers. The public `generate()` compatibility API
still returns a host token list; its separately reported request-level numbers
include the expected 128 token readbacks and one terminal synchronization.

## Greedy sampling decision

`Sampling1D` split local-top32 plus `k=1,p=0,temp=1` remains the default. All
selected and force-argmax outputs are valid global maxima on the capped probe
logits. Force-argmax gathers the full vocabulary and costs 2.731 ms on P150x2
and 2.297 ms on P150x4, versus 0.760 and 0.446 ms for the split path. P150 is a
1.502 versus 1.483 ms near-tie, below the 3% selection threshold, so the unified
top-k/top-p-capable split contract is retained. Force-argmax is not accepted as
a workaround for a malformed sampled path, and no full-vocabulary gather is in
the measured token-out loop.

## Accuracy and qualitative evidence

The reference is AIME24 prompt 0 rendered with the pinned Gemma chat template
at checkpoint revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

| Profile | Prefill top-1 / top-5 / top-100 | Traced teacher top-1 / top-5 / top-100 | Teacher decode t/s/u |
| --- | ---: | ---: | ---: |
| P150 | 95% / 100% / 100% | 96% / 100% / 100% | 34.494 |
| P150x2 | 96% / 100% / 100% | 96% / 100% / 100% | 41.447 |
| P150x4 | 97% / 100% / 100% | 95% / 100% / 100% | 43.350 |

Every required top-5 gate exceeds 98% and every top-100 gate is 100%.
Teacher-forcing numbers include host teacher feedback and are kept separate
from the device token-out measurements above.

A fresh P150x4 traced greedy autoregressive run uses the same 161-token AIME24
chat prompt and generates 100 tokens. It stays on-task in coherent English and
begins a valid algebraic setup; both fixed-length HF and TT completions end
mid-solution. The shared mechanical-degeneracy checker finds no advisory or
critical finding. Token agreement with HF is informational only (9/100) and is
not substituted for qualitative review.

The optimized path also freshly reruns the shared six-prompt chat suite for 64
greedy tokens per prompt with allocation tracking and reset between requests.
Haiku, learning explanation, inventor story, thermodynamics, French
translation, and Fibonacci outputs are all coherent and task-aligned, with no
mechanical repetition, wrong-language drift, prompt echo, or cross-request
leakage. Same-revision HF controls, rendered prompts, token IDs, outputs, and
per-case assessment are retained together.

## Context and nonaligned prompts

| Profile | Supported context | Largest tested nonaligned prompt | Final legal traced position | Result |
| --- | ---: | ---: | ---: | --- |
| P150 | 50,624 | 50,623 | 50,624 | full 30-layer pass; following position safely rejected |
| P150x2 | 262,144 | 262,143 | 262,144 | pass; following position safely rejected |
| P150x4 | 262,144 | 262,143 | 262,144 | pass; following position safely rejected |

P150 uses all 30 decoder layers plus real terminal weights for the nonaligned
prefill and final legal traced sampling replay. The measured loop has no host
token read, synchronization, or page-table refresh, and the next position is
rejected safely. P150x2/P150x4 retain representative sliding/full-layer
nonaligned boundary probes plus exact full-stack allocation because their
multi-gigabyte headroom is not limiting. `../context_contract.json` records the
preserved per-profile limits and exact artifacts.

## Lower bound and gap closure

The inherited best multichip layer latencies yield
`25 * sliding + 5 * full` lower bounds of 23.4163, 19.7875, and 19.2025 ms for
P150/P150x2/P150x4. Position-matched logits-only traces measure 25.3161,
20.7366, and 19.6471 ms. Adding sampling and feedback gives 26.8409, 21.4906, and
20.1018 ms. Token-out is therefore 14.62%, 8.61%, and 4.68% above the decoder
stack lower bound; no profile exceeds the requested 10-15% closure band.

The matched token-out minus logits-only increments are 1.5249, 0.7541, and
0.4548 ms. Both traces cover positions `[134,262)`, use five warmups and 128
nonblocking replays, and advance position on device. The increment includes
split sampling, token alias feedback, and the second trace replay rather than
attributing the entire terminal path to the sampler.

## `tt-perf-report` conclusions

The retained reduced P150x4 capture uses real shapes and weights for layers 0
(sliding), 5 (full), final norm, LM head, sampling, persistent CCL, cache
updates, and position/RoPE advance. Separate warmed-prefill and steady-decode
signposts are processed with `tt-perf-report` 1.2.9 using
`python_env/bin/tt-perf-report --arch blackhole --active-experts 8`. Prefill
spans 9.657513 ms host-side and 7,657.06 us summed device operations, with a
9.6% modeled DRAM roofline (49 GB/s). Decode spans 3.581892 ms host-side and
2,591.76 us summed device operations, with a 19.7% modeled DRAM roofline (101
GB/s). Both replay sessions are complete and repeatable: device 0 records 195
model operations and 12 sampler operations in each session. Their kernel sums
are 2,148.878/2,143.172 us for the model and 416.253/418.336 us for sampling.

Decode's dominant families are DRAM matmuls at 691.37 us, the split sampler's
local-vocabulary `TopkLargeIndicesDeviceOperation` at 279.48 us, sparse matmuls
at 247.21 us, width-sharded matmuls at 136.54 us, and six persistent
all-reduces at 107.96 us. The final sampling choice is 27.38 us. The two generic
`TopKDeviceOperation`s total 44.67 us and belong to MoE routing in the model
trace, not the sampler. The large-index input is only the local
`[1,1,32,65536]` vocabulary shard; there is no full-vocabulary gather or
force-argmax in token-out. The BF16 row-major embedding closes the inherited
full-table terminal conversion. No evidence supports changing the preserved
decoder matmul, CCL, program-config, or kernel winners in this full-model-only
pass.

## Runtime and hardware audit

All retained accuracy, capacity, latency, sampler, state, and qualitative paths
run with `throw_exception_on_fallback=true`; the measured path has no runtime
fallback. Tracy and allocation tracking were run separately. Device commands
were serialized on one four-chip Blackhole P300C QB2, and post-run health checks
showed all four devices available and reset-capable.

The initial watcher sweep exposed an invalid one-chunk scatter-state setup in
the all-gather device helper used for UINT32 sampler indices. Watcher caught the
otherwise-disabled assertion before the selected unicast transfer executed.
AutoFix guarded unused scatter initialization in both multicast and unicast
helpers; no sampler shape, router payload, or model policy workaround was
accepted. Reduced P150x2/P150x4 controls then passed, followed by the complete
30-layer P150/P150x2/P150x4 sweep (3 passed in 187.31 s) with periodic watcher
polling. The failing and fixed console/watcher logs are retained.

## Evidence index

- `baseline/` and `final/profiles_refresh/`: same-workload before/after
  warmed prefill-to-logits and fully traced token-out results.
- `final/logits_only_matched/`: position-matched sampler-ready logits boundary.
- `final/public_generator/` and `final/public_generator_bfp8_control/`:
  host-visible 128-token public `generate()` selected/control results.
- `final/accuracy/`: all-profile AIME24 prefill and traced teacher-forcing gates.
- `final/qualitative/`: fresh AIME24 and six-prompt shared-suite HF/TT outputs,
  degeneracy report, prompt metadata, and review.
- `final/full_stack_context_tp1/lifetime_fix_50624/`, `final/capacity/`,
  `final/nonaligned/`, and `final/batch32/`: P150 full-stack boundary, other-profile maximum context,
  nonaligned boundary, and serving-state capability evidence.
- `final/sampler/`: all-profile split-greedy versus force-argmax A/B results.
- `final/profiler_final/`: one compressed raw Tracy CSV plus separate processed
  warmed-prefill and decode operation CSVs, reports, tables, and plots.
- `operation_topology.json`, `perf_summary.json`, and `provenance.json`:
  compact topology, performance, and exact-source/artifact provenance.
- `work_log.md`: commands, checklist closure, watcher AutoFix, and commit SHAs.

The stage includes a CCL device-kernel C++ repair. The required repository
build wrapper was attempted but access to the Docker daemon socket is denied on
this host; this is recorded as unverified host compilation. Device JIT rebuilt the
affected kernels for the watcher-qualified TP2/TP4 runs.
