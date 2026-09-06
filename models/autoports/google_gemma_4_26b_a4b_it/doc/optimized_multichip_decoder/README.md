# Gemma 4 26B A4B optimized multichip decoder

This stage optimizes the completed `MultichipDecoder` in place for the three
release-blocking P150 proxy profiles. It does not contain full-model,
generator, LM-head, sampling, or vLLM work.

## Selected implementation

All profiles use real tensor parallel execution on the target mesh. TP2 and
TP4 use three trace-safe persistent asynchronous BF16 hidden all-reduces per
decode layer; they are not replicated single-chip fallbacks.

| Profile | Mesh | Selected residual/collective family | Additional selection |
| --- | --- | --- | --- |
| P150 | 1x1 | R22 L1 decode residual; identity reduction converts directly to the next residual layout | coherent R22 dense; packed active-expert gate/up; unused packed dense storage omitted |
| P150x2 | 1x2 submesh | R22 L1 decode residual; persistent Linear BF16 all-reduce, one link | coherent R22 dense; packed experts; full-attention O uses one-reader DRAM sharding; unused packed dense storage omitted |
| P150x4 | 1x4 | R22 for sliding and correctness-gated R0 for full; persistent Ring BF16 all-reduce, two links | packed experts; sliding QKV plus inherited O/packed-dense/down DRAM-sharded roles |

The packed expert tensor is rank-local `[up, gate]`, feeds one sparse matmul,
and is shared by prefill and decode. The separate gate/up device tensors are
released. This is capacity-neutral on TP1/TP2 and reclaims the former 4.412 GB
redundant TP4 packed copy. Decode now passes the exact runtime TopK indices and
compact scores to indexed sparse gate/up and down matmuls. There is no fixed
`nnz=8` metadata claim: the fallback path asks TTNN to infer nonzeros at
runtime, while the default indexed path executes exactly the eight indices
selected by the gate. No dense all-expert path is used.

## Inter-layer contract

The public decoder output remains replicated BF16 TILE DRAM with logical shape
`[1, 1, logical_M, 2816]`. There is no gather, reshard, reduce-scatter,
all-gather, or all-reduce between decoder layers. Decode-local R22 layouts are
implementation details: each row-parallel contraction completes its required
in-layer reduction and hands the result directly to the next residual layout.
Full-model bringup must preserve this replicated inter-layer contract.

Public sequence lengths need not be tile aligned. The decoder owns physical
padding, cache masking, and logical slicing; S=33 prefill followed by
cache-consuming eager/captured/replayed decode passes for sliding and full
attention.

## Final correctness

Final PCC uses real checkpoint weights and prompt-derived inputs recorded after
HF layers 0 and 4. The artifact records the checkpoint revision, token IDs,
prompt hash, tensor shapes, and content hash. The measured implementation is
`MultichipDecoder` on each target mesh; HF is only the numerical oracle.

| Profile / oracle | Sliding prefill / decode PCC | Full prefill / decode PCC |
| --- | --- | --- |
| P150 / HF | 0.999744 / 0.999773 | 0.999594 / 0.999510 |
| P150x2 / HF | 0.999775 / 0.999783 | 0.999831 / 0.999859 |
| P150x4 / HF | 0.999744 / 0.999849 | 0.999812 / 0.999801 |

All six pass at PCC 0.995 with `throw_exception_on_fallback=true`. With
prompt-derived B32 inputs and policy-matched single-chip references, TP4 scores
0.999764 sliding and 0.999833 full; TP2 scores 0.999854 and 0.999895. AutoFix
also retained the deterministic 0.994989 failure against the prior stage's
BFP8-dense oracle and proved it was a reference-policy mismatch, not watcher
drift: unchanged LoFi TP4 full reaches 0.999833 against the matching BF16-dense
oracle. Mixed sliding-to-full stacked traces pass 20 bit-exact
replays on TP2 and TP4; the accepted chained PCCs remain above their inherited
0.98 chained threshold. The stacked artifacts record exactly eight selected
experts and persistent-resource reuse without replay dispatch.

## Warmed performance

Measurements use real checkpoint weights, S=1024, B1, one untimed prefill
warmup before the timed prefill, and five requested decode warmups plus trace
validation before 30 trace replays. Prefill is warmed synchronous host
latency; decode is traced warmed host latency. Numbers below come from the
final default path.

| Profile | Layer kind | Baseline prefill ms | Final prefill ms | Speedup | Baseline decode ms | Final decode ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P150 | sliding | 96.431 | 80.051 | 1.205x | 0.964689 | 0.759652 | 1.270x |
| P150 | full | 107.014 | 78.799 | 1.358x | 1.080224 | 0.885008 | 1.221x |
| P150x2 | sliding | 70.734 | 52.654 | 1.343x | 0.845147 | 0.646949 | 1.306x |
| P150x2 | full | 77.816 | 57.377 | 1.356x | 0.943894 | 0.722745 | 1.306x |
| P150x4 | sliding | 78.695 | 63.521 | 1.239x | 0.677389 | 0.585234 | 1.157x |
| P150x4 | full | 87.299 | 70.092 | 1.245x | 0.984365 | 0.914335 | 1.077x |

Every final host number improves. The selected path also removes 4.412 GB of
redundant TP4 persistent storage. `perf_summary.json` includes same-run device
operation time, per-device imbalance, and DRAM roofline. The maximum profiled
decode imbalance is 1.0062x.

## Profiler and advice closure

Each profile/layer-kind pair has its own Tracy raw CSV, merged prefill/decode
tables, operation summary, and unmerged per-device decode CSV under
`profiler/`. Reports were generated with `tt-perf-report --arch blackhole
--active-experts 8` and exact signpost ranges.

The reports identify active-expert sparse matmuls as the dominant prefill
family. Decode is distributed among sparse matmuls, attention/dense matmuls,
layout operations, norms, SDPA, and the three all-reduces. All report advice is
closed:

- DRAM-sharded decode matmuls: legal 1/2/3-reader TP1 O variants were timed;
  all six layer-kind/reader measurements are slower than the selected default,
  so TP1 rejects them on whole-layer latency rather than capacity.
  The report-advised R22 full-attention QKV, gate, up, and down roles were then
  run in isolation with recorded activations. TP1 down was the only repeatable
  nominal winner (0.8810/0.8823/0.8812 ms in three candidate runs versus
  0.8853/0.8843/0.8844 ms default), but the 0.23-0.48% delta is sub-noise and
  requires a retained 59.5 MB five-layer BF16 copy, so it is rejected. QKV
  regressed, gate/up were noise-level, and TP2 down regressed to 0.7639 ms.
  TP2 full O remains selected. A final TP4 sliding QKV trial passed PCC and
  improved decode from 0.594262/0.594501 ms to 0.584680/0.585390 ms in two
  alternating controls, so it is selected. Full-attention QKV is inapplicable
  outside R22 and is not allocated. The 25-layer sliding copy costs
  288,358,400 bytes/device and leaves 5.026 GB conservative headroom. TP4
  retains the other inherited measured roles;
  multi-reader use on a multi-device `MeshDevice` has an exact primitive
  contract failure already preserved by the completed multichip stage.
- 88-core prefill grid: a legal 110-core 2D QKV/O family was run on all six
  release cases. It ranges from 0.24% faster to 0.71% slower, regresses both
  TP4 cases, and is rejected as noise/regression. Combining that grid with L1
  QKV/O inputs also ran all six; its TP4 full case regresses 1.20%.
- L1 inputs/subblocks: decode R22 roles already consume L1-sharded inputs with
  tuned legal subblocks. The report's material sparse-prefill input was moved
  to L1 on all six cases; results range from 0.18% faster to 0.40% slower and
  are noise/regression. The extra conversion is rejected. Packed sparse gate/up
  is the dominant prefill win.
- Fidelity/precision: attention, dense-MLP, MoE, residual/norm, and CCL dtypes
  are independently controllable and were isolated on TP2 and TP4. Individual
  BFP8 roles were noise-level or slower. The combined BFP8 family sped prefill
  but slowed decode and failed TP4 B1 PCC (0.97485/0.97670), TP2 full decode
  PCC (0.99117), and TP4 B32. BF16 remains the selected activation/CCL policy.
  Prompt-derived input validation additionally exposed the BFP8 full-attention
  dense MLP at 0.992689 prefill PCC. The selected shared BF16 dense-weight
  policy covers gate, up, and down and reaches 0.999831; attention-only,
  expert-only, and fidelity-only probes did not fix the miss. BFP4 and
  LoFi/HiFi variants retain the completed-stage evidence.

## Collective and residual alternatives

BF16 CCL is selected. A fresh TP2 BFP8 CCL retry uses matching preallocated
buffer and payload dtypes. With recorded inputs it passes but is slower at B32
(6.5485/6.6342 ms versus 6.5356/6.6256 ms); the seeded stress case also has
only 0.994785 sliding decode PCC. TP4's earlier BFP8 CCL evidence misses PCC.
TP4 one-link Ring is 6.0%/4.4% slower than two links. Disabling
persistent resources changes latency by only 0.2-0.3% while losing explicit
trace-safe buffer/semaphore ownership, so persistence remains selected.

Fused matmul+reduce-scatter, all-gather+matmul consumers, and a complete
fractured-residual decoder were already adapted to exact Gemma shapes in the
completed multichip stage. The full real-weight chain passes PCC but needs six
reduce-scatters and six norm-stat all-gathers and is 29.474% slower
(0.844279 ms versus 0.652086 ms). Those op shapes and the replicated public
contract did not change in this pass, so that exact evidence remains the
rejection basis.

TP4 coherent R22 was retried after replacing the first illegal local-K block
with a legal one-tile block. Sliding then regressed 1.9%. Full attention was
retried with BF16 attention, HiFi2 MLP/expert math, and BF16 experts; all four
completed results remain below PCC 0.995. R0 therefore remains the measured
TP4 full winner rather than restoring R22 immediately to the old contract.

## Capacity and context

KV cache stays caller-owned BF16 with unchanged TP-local heads, blocks,
page-table semantics, and maximum contexts:

| Profile | Supported tokens | Conservative full-stack headroom/device |
| --- | ---: | ---: |
| P150 | 50,624 | 268,670,464 B |
| P150x2 | 262,144 | 861,289,472 B |
| P150x4 | 262,144 | 5,026,182,144 B |

`capacity_projection.json` and `../context_contract.json` record the revised
persistent accounting. Omitting the packed dense tensor that coherent R22
never consumes offsets most of the five full layers' BF16 gate/up/down
precision delta. The final-source real-weight 50,624-token boundary was rerun
for both layer kinds after direct packed-expert host upload removed the
285,474,816-byte constructor peak. Both pass with fallback throwing.

## Validation and artifacts

- `operation_topology.json`: pre-change topology audit and action map.
- `perf_summary.json`: exact baseline/final host and device accounting.
- `baseline/`: frozen starting measurements. `final/{p150,p150x2,p150x4}/`
  contains the six current default-path measurements used above; each JSON
  embeds the command, device IDs, code hashes, and fallback policy.
- `profiler/`: losslessly xz-compressed raw Tracy CSV, `tt-perf-report`
  tables/advice, summaries, and unmerged device CSVs. Compression keeps each
  provenance artifact within the repository's 500 KB file-size policy.
  `profiler/provenance.json` ties every raw capture to its same-run timing JSON
  and records hashes, operation totals, device imbalance, and the TP4-full
  final-source capture.
- `candidates/`: before/after candidate timings, PCC failures, and retry
  evidence.
- `final/{current_pcc_p150,current_pcc_p150x2,current_correctness}/`: final
  prompt-derived HF PCC for both layer kinds on TP1/TP2/TP4.
  `final/tp4_full_b32_matched_reference/` retains the policy-aligned
  prompt-derived B32 single-chip references.
- `final/watcher_current/`: B32, nonaligned, stacked-trace, and the final
  12-case worker/idle-Ethernet watcher
  evidence. Active-Ethernet instrumentation is not
  legal for this fabric binary: `final/watcher_eth/watcher.log` records the
  physical 30,064-byte program versus 26,624-byte watcher buffer failure.
  `final/tt_smi_post_final.json` shows four healthy devices and zero GDDR
  errors after that expected failure and the clean retry.

The implementation and documentation are Python/JSON/Markdown only; per
`AGENTS.md`, no C++ build is required. Exact commands and the optimization
checklist are retained in `work_log.md`.
