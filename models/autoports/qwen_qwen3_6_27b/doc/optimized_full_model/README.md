# Qwen/Qwen3.6-27B optimized full model

## Results first

Target: four Blackhole P300c devices, `1x4` tensor parallelism, physical
`FABRIC_1D_RING`, batch 1, prompt 128 / generate 128 unless stated otherwise.
All latency is warmed. Accuracy uses the AIME24 chat-template reference.

| Metric | Completed full-model baseline | Optimized full model | Change |
|---|---:|---:|---:|
| Prefill top-1 / top-5 / top-100 | 97% / 100% / 100% | 97% / 100% / 100% | preserved |
| Teacher-forcing top-1 / top-5 / top-100 | 97% / 100% / 100% | 97% / 100% / 100% | preserved |
| Same-process warmed TTFT | 679.059 ms | 655.160 ms | 3.52% faster |
| 64-layer model trace | 42.016 ms | 42.015 ms | stable |
| Split greedy sampler trace | 2.417 ms | 1.478 ms | 38.83% faster |
| Device model + sampler token-out | 44.432 ms, 22.506 t/s/u | 43.488 ms, 22.995 t/s/u | 2.17% higher throughput |
| Caller-visible autonomous token-out | 49.357 ms, 20.260 t/s/u | 48.273 ms, 20.716 t/s/u | 2.25% higher throughput |
| Steady traced teacher-forcing replay | not isolated | 22.558 t/s/u | 98 post-capture replay intervals |
| Teacher decode including trace setup | 19.22 t/s/u (prior runtime) | 19.934 t/s/u (current runtime) | setup retained separately: 622.0 ms |

The prior stage's separately inherited representative TTFT was 742.651 ms.
The same-process 679.059 ms control above is the fair optimization comparison:
the test warms and measures the old allocate/concatenate/copy prefill-state
policy and the selected persistent-state policy with the same process, exact
shape, weights, mesh, and program cache.

Teacher forcing is not token-out. Its loop writes the ground-truth token from
the host after every prediction and its readiness run includes a much longer
161-token AIME prompt. The autonomous device attribution instead replays the
model trace and sampler trace without token readback or host feedback. The
caller-visible number separately includes one compact sampled-ID observation
per token for the Python response API. The inherited and refreshed teacher
rates also use different installed runtime commits, so that row is evidence of
the required separate path, not a source A/B claim. The final runner performs
one explicit full-reference warmup before measurement so kernel compilation is
excluded. Because reset correctly releases request-owned traces, the runner
also reports trace setup separately and starts steady replay timing at the
first decoded callback.

## What changed

The selected optimization removes avoidable terminal-path work without
changing the decoder's selected numerical or distributed policy:

- batch-1 prefill writes each linear-attention layer's result into its existing
  persistent recurrent-state tensor, avoiding 48 fresh state allocations,
  one-element concatenations, and copies;
- batch greater than one retains per-slot materialization and concatenation so
  mixed prompts and independent fixed-slot state remain correct;
- low-level decode may return the compact sampled TT tensor directly with
  `read_from_device=false`; `read_decode_output(async_read=true)` lets a serving
  scheduler defer its only compact-token observation;
- processing a deferred observation updates the Python penalty-history mirror;
  the captured sampler has already advanced the authoritative device history;
- greedy sampling now replaces the inherited full-vocabulary Ring all-gather
  with exact local winners and compact candidate all-broadcast, while the
  traced stochastic top-k/top-p path remains unchanged;
- exact-shape warmup was added to the benchmark so reported TTFT excludes
  compilation and can compare both prefill-state policies in one run.

This is not vLLM integration. The serving-ready generator API was preserved and
made schedulable, but no plugin registration, vLLM adapter, server, or serving
benchmark was added.

## Preserved full-path contract

The full path remains replicated BF16 embedding, 64-layer TP4 decoder,
replicated final RMSNorm, TP4 vocabulary-column BFP8 LM head, split sampler,
paged KV/recurrent state, and generator orchestration. The following selected
decoder policy is unchanged:

| Contract | Selected policy |
|---|---|
| Inter-layer residual | replicated BF16 TILE/DRAM `[B,1,S,5120]` |
| Attention weights | BFP8 |
| Linear-attention MLP | BFP4 gate/up/down |
| Full-attention MLP | BFP4 gate/up, BFP8 down |
| KV cache | BFP8 paged, one local KV head per TP device |
| Activations | inherited BF16 policy |
| CCL payload | BF16 attention/full MLP; BFP8 linear MLP |
| Fidelity | inherited selected LoFi projection/MLP groups |
| Collectives | persistent asynchronous physical-Ring all-reduce/all-gather |

The rejection ledger remains binding. This stage did not run a datatype
frontier and did not select a faster rejected dtype, fidelity, fractured
inter-layer residual, replicated model stream, or generic collective policy.
Datatype Pareto selection belongs to the later datatype-sweep stage.

The generator still owns explicit cache, page table, current-position,
prefill-position, RoPE, prompt-length, batch, active-row, seed, and penalty
state. It supports mixed prompt lengths, inactive rows, fixed slots, reset, and
externally supplied low-level state. Arbitrary positive prompt lengths remain
publicly valid: chunk/page/tile padding is internal, and the hardware gate
exercises non-aligned lengths 65 and 67 together.

## Split token-out and sampler selection

The canonical greedy loop is:

`nonblocking model trace -> TP logits -> captured split greedy -> tt_out_tok`

The model trace consumes the persistent token buffer, advances device position
and RoPE state, updates recurrent/KV cache, and leaves TP vocabulary logits on
device. The sampler computes each vocabulary shard's local maximum, exchanges
only compact candidates on the physical Ring, chooses the global winner, and
writes the token into `tt_out_tok`. An unchanged page table causes no copy; a
changed table is copied only at its explicit request boundary.

The selected split greedy trace is 1.478412 ms and returns token 225721, equal
to host greedy. It computes local max/argmax, packs one candidate tile per
device, all-broadcasts only candidate values and indices, and reconstructs the
exact global ID. Generic `Sampling1D` also returns 225721 but takes 10.736105
ms, 7.26x longer, so it is rejected. The inherited full-vocabulary force-argmax
control took 2.417212 ms and is also rejected. The selected path is not an
incorrectly shaped generic sampled path and does not perform a full-vocabulary
all-gather. The
stochastic contract remains top-k/top-p capable: it gathers TP logits once,
runs four local 65,536-wide top-k operations, concatenates 128 candidates, and
uses the common captured sampler with persistent seed and penalty state.

`evidence/candidates.csv` is the compact candidate and rejection ledger.

## Lower bounds and gap accounting

The optimized multichip decoder medians give the stack lower bound:

`48 * 0.718857 ms + 16 * 0.476422 ms = 42.127888 ms`.

The measured 64-layer model trace is 42.014693 ms, within measurement noise of
that independently derived floor. Model plus sampler is 43.488361 ms, only
1.360473 ms or 3.23% above the decoder-stack floor. That residual is fully
explained by the 1.478412 ms terminal sampler measurement and is below the
10-15% investigation threshold. No dominant force-argmax, generic top-k, or
full-vocabulary gather gap remains to close.

A second theoretical bound divides the 12,935,561,216-byte per-device full
model weight image by the inherited Blackhole 512,000 bytes/us peak DRAM model,
giving 25.264768 ms. The 42.014693 ms model trace is 60.14% of this idealized
weight-read roofline. This is a hard lower-bound convention, not a runtime
prediction; state/cache traffic, CCL, dependencies, and kernel utilization make
the measured result larger. Machine-readable assumptions and results are in
`evidence/final/perf_summary.json`.

## Accuracy and qualitative evidence

Fresh `run_prefill_check` and `run_teacher_forcing` both pass 97/100 top-1,
100/100 top-5, and 100/100 top-100. After one explicit full-reference warmup,
steady teacher-forcing replay reports 22.558 t/s/u; capture-inclusive decode is
19.934 t/s/u with 622.0 ms of trace setup. Both are kept separate from
autonomous token-out. A cold control measured 3.046 t/s/u because first-use
sampler trace compilation landed inside the readiness decode interval; the
Autofix report and 4-layer boundary control document that diagnosis.

The refreshed AIME24 chat-template autoregressive run produces 100 HF and 100
TT tokens. Both completions are coherent English step-by-step work and are
intentionally truncated at the evidence limit. The mechanical gate reports 63
TT words, zero adjacent duplication, 0.0476 repeated-trigram fraction, no
degenerate output, and 40/100 informational token agreement with HF.

The six-prompt shared qualitative controls and manual verdict are recorded
under `evidence/final/qualitative/`.

## Context, capacity, and runtime boundary audit

`../context_contract.json` still advertises the full 262,144-token context and
batch 32. It is the total shared physical KV-token pool, not 262,144 tokens per
slot. No new maximum-shape allocation was introduced, so there is no physical
basis for a capability reduction. The optimized section records persistent
prefill-state reuse, deferred compact-token readback, mixed/non-aligned prompts,
and unchanged page-table semantics.

`evidence/runtime_fallback_audit.md` records a clean measured path. There is no
host argmax, `to_torch`, `.cpu()`, synchronization, token feedback, position or
RoPE refresh, unchanged page-table copy, replicated decoder fallback, or
single-device fallback in `_decode_traced_device`. Host logits exist only at the
initial public prefill/readiness boundary and in explicit compatibility mode.

## Multichip and optimization checklist closure

| Area | Full-model evidence and decision |
|---|---|
| Mesh/topology | exact 4-device `[1,4]` TP mesh; physical Ring configured before open; no smaller-mesh or replicated-model fallback |
| Embedding/norm | replicated BF16 embedding and final norm retained; both operation classes appear in focused reports |
| LM head/logits | vocabulary-column TP4 BFP8 head retained; logits stay sharded into split greedy; no greedy full-vocabulary gather |
| Residual layouts | selected replicated BF16 TILE/DRAM inter-layer contract retained with no extra inter-layer collective |
| Cache/state | BFP8 paged KV remains KV-head sharded; recurrent state stays persistent; changed-only replicated page tables and device position/RoPE advance pass Watcher |
| CCL | focused token report contains four persistent async all-reduces and two compact candidate all-broadcasts; prefill contains six reduce-scatters and six all-gathers |
| Matmuls | selected decoder projection/MLP dtype, fidelity, DRAM-sharded weights, width-sharded activations, and explicit configs remain binding; report FLOP/DRAM rows match them |
| Program configs/kernels | real layer 0/3 reports cover SDPA/GDN, paged fill/update, embedding, norms, LM head, sampling, layout, and CCL kernels; advice triage is in profiler `capture.md` |
| Warm methodology | exact-shape compile plus warm precedes timing; old/new TTFT policies are compared in one process; device and caller-visible decode are separate |
| Correctness | AIME gates, mixed/non-aligned prompts, fixed/inactive slots, stochastic controls, deferred read, reset, and safe Watcher pass |
| Bound/gap | 42.127888 ms stack bound; final terminal overhead 3.23%, below the investigation threshold and explained by the measured sampler |
| Scope | no datatype frontier and no vLLM work; clean measured-path fallback audit |

## Profiling and evidence map

Focused Tracy captures use real checkpoint layer 0 (linear attention) and layer
3 (full attention), plus final norm, LM head, sampling, cache, and CCL terminal
work. Prefill and token-out are captured in separate profiler processes; safe
Watcher is also a separate process. Advice-enabled `tt-perf-report` tables,
compact CSVs, capture provenance, and conclusions live under
`evidence/final/profiler/`.

The refreshed token-out capture sums to 3.381 ms: local reduction is 24.84%,
ArgMax 13.83%, compact candidate all-broadcast 0.53%, matmuls 36.16%, and four
async all-reduces 1.97%; it contains both paged-cache updates, one SDPA decode,
no TopK, and no all-gather. The prefill capture sums
to 5.746 ms: matmuls are 41.65%, norms 9.45%, reduce-scatter/all-gather 7.11%,
and both paged cache fills are present. `tt-perf-report` models overall DRAM at
141 GB/s for token-out and 102 GB/s for prefill. These reduced profiles identify
cost structure; the uninstrumented full64 timings at the top are authoritative.

Primary artifacts:

- `evidence/baseline/token_out_metrics.json`
- `evidence/final/token_out_metrics.json`
- `evidence/final/perf_summary.json`
- `evidence/final/prefill_metrics.json`
- `evidence/final/teacher_forcing_metrics.json`
- `evidence/final/autoregressive/`
- `evidence/final/qualitative/`
- `evidence/final/profiler/`
- `evidence/candidates.csv`
- `evidence/runtime_fallback_audit.md`
- `evidence/artifact_manifest.md`
- final reduced split-trace and safe-Watcher JUnit XML files

Exact commands, environment provenance, limitations, review closure, and local
commit SHAs are in `work_log.md`.
