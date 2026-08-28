# AutoFix record

The fresh AutoDebug runner could not create its Bubblewrap sandbox on this
host. AutoFix therefore continued with isolated fresh-context audits and direct
A/B validation rather than treating the runner failure as a model blocker.

## Host/cache boundary

The audit verified two defects: PLE synchronized the mesh after each upload
despite CQ0 producer/consumer ordering, and expert source packing occurred
lazily in decode although the exact 68.080 GB packed store was already within
the contract.

Retained fixes:

- defer PLE completion to the next exact route boundary while retaining the
  host source; `close()` performs the final sync;
- expose completed per-token wall/read/sync, layer, cache, trace, PLE, and host
  counter deltas;
- prepack all 512 exact experts for each of 48 layers at model construction;
- measure pure completed H2D staging separately from full cache service.

Frozen cold/warm runs preserve exact routes and tokens. Warm packing and PLE
table reads are zero. The instrumented 126-token path reports 41,607 exact
misses, 115.035 GB owner H2D, equal peer-zero D2D, zero source pack, and zero
expert/PLE completion synchronizations.

The lower-bound audit also refuted the earlier physical claim. Serial depth-1
cache service is 6.820373 GB/s, but includes route/index/control, exact peer
D2D, and completion; it is not link bandwidth. Pure staging reaches 6.127343
GB/s for one owner and 6.497517 GB/s for concurrent owners, only 41.244% of
the 15.753846 GB/s raw two-link ceiling.

AutoFix A/B tested staging depths 1/2/10, owner partitioning, owner coalescing,
and a prestarted owner thread pool. Every grouping/threading policy is slower;
depth 10 adds 1.194 GB/rank without a p50 improvement. Serial depth 1 remains
selected and creates no executor. The next credible candidate requires native
batched TTNN H2D submission. No safe Python-level speedup survived AutoFix.

## LM head and sampler

The audit confirmed a memory-bandwidth-dominated LM head and required the
DRAM-sharded advice to be tested rather than dismissed. Exact results:

- selected interleaved BFP8/HiFi2: 0.972307 ms, PCC .999922, top-5 5,
  top-100 98, exact device greedy, valid top-k/top-p;
- DRAM s1/c40: static-CB compile reject;
- DRAM s4/c40: CB/L1 overlap compile reject;
- DRAM s5/c40: fully correct at 1.330563 ms, 36.24% slower;
- BFP4/LoFi -> TILE BF16 boundary: sampler fixed, but PCC .976245, top-5 4,
  top-100 67, so rejected for accuracy.

The specialized full-vocabulary device argmax and generic local-top32/k=1
both match host argmax. Specialized is selected because frozen A/B is 0.664
versus 0.899 ms. This is not force-argmax. `TopKDeviceOperation` in full token-
out reports is the expert router, not the terminal sampler.

## Preserved decoder policy

The proposed GDN BFP8 HiFi2-to-LoFi switch is not a current full-path A/B:
the retained candidate has different route/miss work and no full-model LoFi
accuracy gate. The user required preserving the decoder fidelity policy and
assigned Pareto precision selection to `$datatype-sweep`, so inherited HiFi2
remains selected without manufacturing a rejection claim from incomparable
evidence.

## Final gates

Frozen digest `e85ca93a2fc788bcd70095d93284f538ab2f2ae30754b0ed7523c1bbe6a631d6`
passes 36/36 static contracts, exact cold/warm prefill, AIME prefill, 99-row
teacher forcing, 100-token autoregression, split greedy and top-k/top-p traces,
mixed/inactive/fixed-slot state, eager batch 32, context 262144, fresh
qualitative prompts, selected endpoint, four isolated profiler captures, and
full-48 Watcher plus trace-allocation tracking.

Reportable token-out is 231.490 ms/token. Its optimistic physical lower bound
is 176.814 ms/token; the 30.923% gap is not declared closed. Exact Python-level
AutoFix variants failed, and the remaining native-batched-H2D limitation is
recorded with both raw-ceiling and measured pure-staging evidence.
