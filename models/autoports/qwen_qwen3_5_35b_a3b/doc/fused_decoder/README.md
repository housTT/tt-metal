# Qwen3.5-35B-A3B fused decoder — stage 02

This stage delivers `tt/fused_decoder.py` for the pinned real
`Qwen/Qwen3.5-35B-A3B` checkpoint, revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, on one Blackhole ASIC (1x1 mesh).
The host exposes four ASICs on two P300c cards (`device_list.log`). The functional
runtime is unchanged. Later optimized-decoder, multichip, full-model and vLLM
stages were not started. Independent stage review returned **clean-pass** with
no required work remaining; see `stage_review.md` and `work_log.md` for closure.

## Final measurements

Exact workload:4096 input positions,128 decode steps, batch1, one request, one
Blackhole ASIC. Prefill is warmed; decode is the mean of128 warmed traced
**whole-layer device spans**. Values below are **milliseconds of device time**.

| Layer kind | Baseline prefill | Fused prefill | Speedup | Baseline decode | Fused decode | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| linear_attention | 1743.870 | 764.672 | 2.28x | 2.190 | 1.400 | 1.56x |
| full_attention | 1720.532 | 697.718 | 2.47x | 1.609 | 0.979 | 1.64x |

| Layer kind | Baseline prefill PCC | Fused prefill PCC | Baseline min decode PCC | Fused min decode PCC |
|---|---:|---:|---:|---:|
| linear_attention | 0.999701798 | 0.999676347 | 0.997864842 | 0.997841656 |
| full_attention | 0.998596847 | 0.998591542 | 0.995969236 | 0.995754361 |

`final_results.json` joins the exact sources. Baseline is
`../functional_decoder/tracy/{linear_attention,full_attention}`. Accepted final
profiles are `profile_final/tracy/linear_attention` and
`profile_placement_final/tracy/full_attention`; each has `ops.csv.gz`,
`measurement.json`, `roofline.json`, `processing_provenance.json`, and both
`prefill_perf_report.{csv,txt}` / `decode_perf_report.{csv,txt}`.
Final accuracy and unprofiled host timings are `headline_final/linear.json` and
`placement_final/full.json`:1259.719us and920.198us host replay medians.
These reproduce or beat the best correct screened candidates within run noise.

Prefill PCC deltas are below0.000026 and minimum decode deltas below0.000215.
The small rounding changes come from native convolution/norm/gating and native
expert accumulation, plus full-attention RoPE arithmetic. Fused-to-functional
minimum decode PCC is0.999664128 linear /0.999609709 full. All per-token gates pass.

The perf tables confirm103 versus166 baseline device rows per linear decode and
65 versus101 per full decode. Final prefill has3588 /1013 device rows. Every
measured program hash appears in preceding warmup, and every measured device row
has complete firmware timestamps. Native compact FFN and projection work now
dominate decode; remaining GDN movement implements precise recurrent state and
kernel layouts. Prefill still executes dense/tile-union expert work: the packed
SwiGLU topology wins despite doing more arithmetic than useful active8 work.
Firmware intervals of neighboring operations overlap; table row totals and
apparent movement wait times must not be summed into a layer denominator.
Each accepted profile root's `graph_analysis.json` records actual counts and
warmup checks.

Runtime rows confirm BF16/HiFi4 GDN projections and BF16 convolution taps,
BFP4/LoFi packed/shared prefill weights, BFP8 down/full-attention projections,
and BFP8 compact FFN weights. The custom FFN CSV omits its math-fidelity field;
LoFi is backed by the exercised source/config, not invented profiler metadata.
Whole-layer nominal-peak useful-FLOP ratios are0.103960% linear /0.128960% full;
estimated compulsory-DRAM ratios are16.549971% /13.490976%. These ratios include all layer time,
use one ASIC's663.552TFLOP/s LoFi nominal peak and512GB/s DRAM, and are not
per-op efficiency averages. Sparse useful work, higher-fidelity GDN operations,
extra dense prefill work, movement and gaps explain the low useful-FLOP ratio.

## Delivered graph

`FusedDecoder.DEFAULT_FUSIONS` selects the qualified path. The instance reuses the
functional constructor, arbitrary-length request planner and ownership protocol,
then installs `_FusedGDN` or `_FusedAttention`, `_DedicatedRouter`,
`_PackedPrefillExperts` with `_CompactExperts` decode, and `_FusedShared`.
The delivered device tests assert these actual component classes. Explicit
`--fusions` switches retain unselected research controls; they are not the
qualified default and may fail acceptance.

- Routing uses the generalized native MoE gate. Decode passes top8 scores and
  IDs directly to eight output regions of the fused SwiGLU/down kernel and its
  weighted reduction. All256 real expert descriptors remain resident; only8
  selected experts execute per token. Grid8x2, LoFi, BFP8 native weights and
  internal partials, BF16 output. Gate/up BFP4 coefficients are preserved by
  upcasting because this native op requires homogeneous weight types.
- Prefill tile-pairs gate/up weights without changing BFP4 exponent groups and
  uses8 groups of32 native split minimal-matmul SwiGLU outputs. It executes dense
  gate/up work for all256 experts, faster than the baseline sparse tile-union
  dispatch, followed by the existing BFP8 down projection. Useful-FLOP estimates
  still count only8 active experts. Shared gate/up uses the same native SwiGLU
  epilogue and binary sigmoid-input gating.
- Full attention packs Q/K/V/gate, uses native head splitting and indexed partial64 RoPE,
  preserves the untouched192 head coordinates, and fuses the paired BF16 paged
  cache update on disjoint K/V shard cores. Native causal paged SDPA remains.
  Odd cache capacities use the existing32-token decode chunk contract. Q/K
  projection columns, norm weights and RoPE tables are permuted in16-channel
  groups at construction. The native kernel rotates within each32-channel tile
  and copies the192-channel tail directly, eliminating separate concat. Both
  prefill and decode use the same permuted K basis. Caller-owned caches remain
  BF16 with identical page ownership/capacity, but populated functional and fused
  K caches are not interchangeable without converting the first64 channels.
- GDN uses causal convolution+SiLU, direct Q/K/V outputs, direct prefill history
  tail, RMS-based L2 normalization, fused exact chunk GDN with head-major output,
  and prefill gated normalization. Decode retains FP32 SFPU recurrence, grouped
  Q/K broadcast and in-place addcmul state update. Native TF32 recurrence fails
  real B32 accuracy and is excluded. No persistent state dtype is reduced.

`graph_inventory.md` records the complete original graph, movement at every
boundary and closure of every graph-fusing skill pattern. `candidate_results.csv`
indexes measured control/candidate records. Exact commands, failure exits,
per-token PCC and source hashes are in the corresponding `commands.json` and
result JSON. Dedicated kernels, structural rewrites and epilogues were retried
with valid layouts/packing before rejection. No applicable untested graph-fusion
candidate remains in this inventory. General kernel geometry/precision sweeps
belong to the explicitly excluded next stages.

## Numerical and semantic gates

The acceptance bar remains PCC >=0.995 for each request and decode token, against
HF using actual checkpoint weights. Fused-versus-functional saved-tensor tests
also enforce this bar. Headline and final device measurements appear above.

`positions_final/linear.json` and `positions_placement_final/full.json` validate32 independent requests, heterogeneous prefixes
1..1006, reversed request slots, per-row positions, eight decode steps, repeated
bit-identical replay and paged-cache sentinels. `reuse_final/linear.json` and `reuse_placement_final` validate17 cases
per kind with one live trace, including unseen logical length36, physical
padding128, continuation cuts1/3/35/36/127/129/160/513 and page sizes32/128.
Tracking is enabled with program-cache allocations included; only the exact
trace output is acknowledged. No blanket allocation scope is exempted.

`pytest_indexed_final/device.log` records4 passing actual fused device tests, real and
synthetic weights on layers0/3. `cpu_contract_final.log` records15 passing contract
checks. `watcher_final/linear.json` and `watcher_placement_final` contain separate watcher correctness runs:
both kinds129/8 and full-attention odd pools257 batch2 and65 batch1, each31 decode
steps. Raw watcher dumps and `watcher_summary.json` show clean completion without
assertions, invalid accesses or corruption reports. No profiler overlaps watcher.

Native262144 and nonaligned262143+1 decode are requalified in `context_final`
(linear) and `context_indexed_final` (full).
Every TT token executes the complete real layer; HF compares16 selected prefill
outputs while retaining every recurrence/KV contribution. This selected-reference
method was controlled against dense HF at1153 in the functional stage. Native
context is batch1; batch32 short-context qualification is a separate bound, not
a claim that the Cartesian product fits. KV remains BF16 and context capability
is unchanged; `../context_contract.json` records both stage evidence sets.

The final indexed promotion changes only full attention. The earlier GDN gates
remain backed by the exact archived `qualified_a4c8_fused_decoder.py.gz` and
`indexed_production.diff`; final headline/device tests cover the current default. The later
`qualified_eb835_fused_decoder.py.gz` and `placement_production.diff` isolate a
constructor-only allocation-lifetime fix; native-context math/capacity are unchanged.

The runtime audit forbids torch work and from_torch/to_torch in measured forwards.
Construction-only weight repacking and host input preparation/readback are outside
those forwards. Remaining prefill untilize/gather/slice-write/tilize operations
serve the functional arbitrary-tail planner and logical output contract; native
head/RoPE/cache layouts serve their kernel contracts. The final indexed kernel copies partial-RoPE tails itself; no post-RoPE concat
remains. Decode K/V shard changes
are necessary for native paired cache update. No host fallback or redundant
round-trip is used. No-op `to_layout` warnings indicate already-correct layout,
not an actual conversion; the profile tables record device work.

## Selection, rejected options and anomalies

Host replay medians are screening measurements only. The selected linear final
candidate (`residual_linear4096/kda_tail.json`) is1259.203us, versus1924us
functional; indexed full (`indexed_rope_native4096.json`) is920.399us,
versus1513us functional. Final default device spans, rather than those host
numbers, are the reported optimized performance above.

Outer affine RMSNorm folding has no speed gain. Expert-major decode, padded
native decode head creation/concatenation, shared sigmoid epilogue, and FIR
ternary/consolidated-layout alternatives are slower after adaptation. Full256
region native FFN passes after a capacity repair but loses to the compact eight
regions. Unsplit packed prefill and single-call32-split pass but lose to grouped32.
Padded decode KDA normalization offers no speed gain and lowers accuracy.
The reviewer-required shared-down/gate/routed-add native epilogue was adapted
with BF16-upcast down weights and an expanded sigmoid gate. Both4096/128 paths
execute. Linear passes but is slower1279.977us versus1259.137us; full is slower
944.960us versus924.030us and fails minimum PCC0.9945282. See
`shared_epilogue4096/commands.json` and `tests/run_shared_epilogue_trial.py`.

Corrected projection-residual fusion passes linear accuracy but is slower
(1275.400us versus1259.203us). Full-attention residual fusion requires BF16-upcast
weights, is slower and fails headline minimum PCC0.9948604. Native recurrence
passes4096/128 but the final surrounding graph fails real B32 minimum PCC0.9884342
(`native_positions_final_candidate.json`), versus0.9977745 for selected FP32.
This is a measured semantic failure, not a synthetic-only precision rejection.

BF16 KDA convolution, normalization rounding and BFP8 native FFN intermediates
introduce small deltas; their joint effects pass the qualified cases. The report
does not claim bitwise equality with functional outputs. Some aggregate PCCs
round to1.0 in FP32 correlation; this is not proof of exact tensor equality.

Two experimental hangs were recovered with preserved triage, owned-process
termination, one bounded reset each, all four ASICs visible and successful1x1
mesh smoke. `AUTOFIX.md` and `triage/*` contain details. Native256-output split
exceeded the reader stack; grouped32 packing repairs it without editing shared
C++. Initial routing used a host-created zeros tensor during capture; resident
tiled templates plus device zeros_like repair it. Neither failure-triage report
is presented as clean watcher evidence. UMD statistics shared-memory permission
warnings also occur in passing controls; they do not prevent device execution.

The first production indexed default reproduced accuracy but regressed to962.790us
host decode. Matched `placement_controls4096` isolates construction lifetime:
production962.360us, equivalent test adapter920.675us, retained-input control
920.449us, old partial-RoPE path924.276us. Retaining original projection/norm
allocations until repacking finishes restores final default920.198us
(`placement_final/full.json`). Physical DRAM placement is the likely mechanism;
addresses were not sampled, so no exact bank-conflict cause is claimed. References
are released before runtime; no additional persistent memory or cache capacity
is required. Final affected correctness/profile checks use this retained lifetime.
The earlier `profile_final` full-attention capture is diagnostic and superseded
by `profile_placement_final`; its linear profile remains applicable.

## Host timing screens (4096/128, batch1)

These comparable tracked, unprofiled replay medians select graph candidates;
columns are **host microseconds**, not device time or telemetry rooflines.

| Kind | Functional warmed prefill | Selected warmed prefill | Functional traced decode | Selected traced decode |
|---|---:|---:|---:|---:|
| linear_attention | 1742026.472 | 764010.552 | 1924.349 | 1259.203 |
| full_attention | 1718918.327 | 698318.796 | 1513.266 | 923.816 |

Sources: `residual_linear4096/commands.json` and
`split32_full4096/commands.json`. The subsequent indexed partial-RoPE candidate improves full decode to920.399us
after removing the caller head-layout round trip (`indexed_rope_native4096.json`);
its warmed prefill697554.472us differs by only0.10%, within timing noise.
The final-default reproduction is reported
with the device results above. No comparison mixes profiler-enabled and plain
host times.

## Reproduction

From the checkout, source `../scripts/environment.sh`. Use a fresh destination
for each acceptance group. Hardware jobs must remain serialized.

```bash
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_fused_acceptance \
  --group positions --directory models/autoports/qwen_qwen3_5_35b_a3b/doc/fused_decoder/positions_new
# Repeat for reuse, pytest, watcher, context with fresh directories.

TT_METAL_RUNTIME_ROOT=/tmp/q35-functional-profiler-root \
TT_METAL_KERNEL_PATH=/tmp/q35-functional-profiler-root \
TT_METAL_CACHE=/tmp/q35-functional-profiler-cache-cwd \
python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.run_profiles \
  --decoder fused --evidence-dir models/autoports/qwen_qwen3_5_35b_a3b/doc/fused_decoder/profile_new

python -m models.autoports.qwen_qwen3_5_35b_a3b.tests.process_profiles \
  --policy fused --layer-type linear_attention \
  --evidence-dir models/autoports/qwen_qwen3_5_35b_a3b/doc/fused_decoder/profile_new
# Repeat processing for full_attention.
```

The profiler root must be a symlink to this exact checkout (not a copied source
version). It avoids a16-bit instrumentation source hash collision diagnosed in
the functional stage; profile commands record root/cache/CWD. Profiling drains
bounded buffers between warmup/measured passes and all128 decode replays.
Processing validates a timestamp for every measured device op, takes first-start
to last-end for the whole layer including gaps, and averages128 complete decode
spans. Host input refresh/readback is outside those replay spans. Roofline uses
useful active-MoE FLOPs and an explicitly estimated compulsory DRAM lower bound,
with the theoretical peak of the single participating ASIC; it is not a mean of
per-op utilization and never a matmul-only denominator.

Machine-local `.pt` reference tensors, Tracy binaries, kernel caches and duplicate
raw profiler exports are not portable artifacts. The archived `ops.csv.gz`,
processing provenance, full measurement JSON and prefill/decode tt-perf-report
CSV/text preserve the reported measurements. Console/watcher logs may be stored
as `.log.gz` in the local commit; decompress to recover exact original content.
The compact telemetry packet is outside the repo at
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/telemetry/packets/b499af22-a351-44ce-8553-1dfe93ca6aac.json`.
