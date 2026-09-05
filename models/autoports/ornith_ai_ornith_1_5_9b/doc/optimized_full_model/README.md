# Ornith-1.5-9B optimized full model

**Completed; independent stage review: clean-pass.** Full-model
measurements below use all32 layers, batch1, prompt128/generate128, native262144
cache, TP4 1x4 ring on **four Blackhole chips on two physical P300c boards**.
The before control is the completed full model at `2e4b8f828c`; each headline
selects median TTFT from five warmed requests and decode from that same request.

**Warmed token-out decode:81.55→83.31 t/s/u (+2.16%). Median TTFT:
47.07→29.59ms (37.14% lower).**

| Metric | Completed full model | Optimized default |
|---|---:|---:|
| Warm request-inclusive TTFT, median of5 |47.065ms |29.587ms |
| Token-out with requested output collection |81.552t/s/u |83.315t/s/u |
| Plain token-out, no loop readback, median of3 |81.585t/s/u |83.377t/s/u |
| Logits-only trace, no sampling/feedback |85.593t/s |87.612t/s |
| Traced readiness teacher-forcing decode |80.849t/s/u |82.489t/s/u |
| Decode output reads/waits,127-step window |127 |1 final history read |
| First request after model construction |455.968ms |743.741ms |

[Exact before](baseline_repeated_v2.json) and [optimized default](perf_prefill_trace_release_v2.json)
retain all five warmed samples, trace counters and three plain-replay windows.
The [explicit eager-prefill control](perf_prefill_eager_control_v1.json) measures
46.737ms before the duplicate-reset removal, with prefill tracing disabled. The
[trace-only intermediate](perf_prefill_trace_release_v1.json) is32.413ms; the
final private reset skip brings that to29.587ms. Every optimized warm request uses
one prefill replay and one first-sampler replay, zero captures/eager-prefill calls,
and zero page writes when unchanged.

First-request timing excludes model loading. The initial trace setup is slower,
and changing the cached prefill shape rebuilds all four traces. The warmed gain
applies to reuse of the measured shape. At prompt2048, the
[additional full-model run](perf_context2048_prefill_trace_v2.json) measures
108.334ms TTFT and82.482t/s/u, compared with111.830ms/82.481t/s/u before prefill
tracing. This is a single warmed2048 request, separate from the five-run headline.


TTFT includes request reset/setup, prefill, first-token sampling and its read.
Token-out decode includes sampling, device token feedback and output collection.
The baseline reads every token; the optimized generator collects UINT32 history
in the sampling trace and reads once per128-step window. Plain token-out replay
has no host readback. Logits-only trace and readiness teacher-forcing timings
are separate workloads, not interchangeable token-out throughput claims.

## Model, precision and capability

Pinned `ornith-ai/Ornith-1.5-9B` revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, snapshot
`/home/hous/dev/ornith-1.5-9b/upstream`:24 linear-attention plus8 full-attention
layers, hidden4096, valid vocabulary248320. Software profile `p150x4` names the
mesh; these are P300c measurements. No full-model TP1 speedup is claimed.

The completed decoder's [policy and rejection ledger](../optimized_multichip_decoder/optimization_evidence.md)
remain selected: BFP4/LoFi projections, BFP8/LoFi decode QKVG, packed GDN and MLP
gate/up, BF16 activations/CCL, BFP8 paged KV with BF16 updates, FP32 recurrent
state. Replicated hidden4096 passes directly between layers: B1 uses32-core L1
width-sharding; B2..32 uses the validated compact/DRAM boundary. No inter-layer
conversion is introduced; the last decoder also feeds the terminal directly. Prior slower coherent collective families and rejected
head dtypes remain rejected; no broad datatype frontier or vLLM work was done.

[Context contract](../context_contract.json) retains262144 tokens. Final native
capacity, non-aligned logical lengths, mixed prompts, fixed slots, inactive rows,
and caller-owned cache/page/position state are verified in the artifacts below.
The exact old-hop/direct-terminal control atB1/B4/B32 supports retained prior
batch/state evidence; `terminal_boundary_watcher_v1.json` passes all rows under
watcher/allocation tracking. The new prefill integration separately passes exact
full32-layer/eager comparisons under watcher and allocation tracking.
Native context is a batch1 validation;32 simultaneous native-length requests are
not advertised. Optional million-token YaRN remains prior table-only evidence.

## Selected full-path changes

- Reuse one fresh owned-cache B1 prefill shape, logical1..2048 including non-aligned
  lengths. Persistent tokens/pages feed a captured complete prefill graph, which
  copies into canonical sampler-ready logits; first-token sampling replays its
  existing trace. Public callers receive owned logits clones. Shape replacement
  releases and rebuilds all four traces together. Mixed batches, continuation,
  long prompts, live shape misses and external caches retain the validated eager
  path. [Integration evidence](AUTOFIX_prefill_integration.md) proves changed
  inputs, live sampling, public ownership and teardown without tracker suppression.
- Keep final padding and RMSNorm in8x4 L1, then reshard normalized output to an
  8x8 head input. Common `LMHead1D` consumes materialized `LazyWeight` wrappers
  around existing BF16/HiFi4 TP4 weights, two32768-column chunks per rank, K1,
  two readers, per_core_N16. The precision-locked64-core head has bit-identical
  recorded-hidden logits and improves paired terminal1.371676→1.137086ms.
  [AutoFix head comparison](AUTOFIX_head_geometry.md) includes adapted padded
  three-reader controls, smaller-core controls and exact L1 blockers.
- Keep sampler-ready local65536 logits and mask invalid vocabulary IDs. Semantic
  greedy k1/p0/temperature1 uses local physical top32 and gathers128 candidates.
  [Correct greedy comparison](sampler_greedy_comparison.json): split0.572ms,
  force-argmax2.744ms, both exact CPU32-row choices. The selected path neither
  gathers the whole vocabulary nor uses host argmax. Top-k/top-p, penalties,
  seeds and `tt_out_tok` remain owned by common `SamplingGenerator`.
- Preallocate fixed decode embedding and named candidate-gather outputs, with
  disposable clones for consumer ownership. Both program signatures warm before
  capture. [Paired CCL controls](ccl_persistence_v2.json) show small improvements;
  the decoder's previously slower persistent family remains rejected.
- Keep nonblocking model and sampling traces separate. Persistent token feedback,
  position/RoPE device advance and changed-only page tables eliminate per-token
  host input traffic. `decode_forward(read_from_device=False)` and
  `replay_decode(steps)` return persistent device output; the caller owns scheduling
  and bounds, and subsequent replay overwrites the returned buffer.
- Collect output history in a sampling-trace variant. High-level generation reads
  once per128-step window, with index reset only between windows. First-token TTFT
  read and explicit teacher-forcing/host-compatibility boundaries remain visible.
  [Output AutoFix](AUTOFIX_output_collection.md) proves exact UINT32 storage and
  long seeded/greedy parity. EOS slicing follows fixed-window execution; early
  compute reclamation and streaming serving are not newly implemented.
- Reset recurrent/conv state for each request; prefill overwrites live KV prefixes,
  so new generation skips whole-KV clearing. Explicit `reset()` still clears KV.
  [Full-stack stale-page control](request_reset_full_v1.json) poisons KV and
  permutes pages over nine aligned/non-aligned lengths. Short alternating reset
  timing showed a small gain; five-request controls before the head change showed
  essentially unchanged TTFT, so no isolated reset speedup is claimed.
- Skip the second hybrid-state clear inside traced B1 generation after the public
  request reset already cleared the same buffers. A private per-call flag leaves
  public prefill/reset and eager paths unchanged. The [paired reset control](AUTOFIX_prefill_reset.md)
  removes96 duplicate mesh operations and saves2.7–2.8ms at logical128/131 while
  preserving exact full-stack logits, states, RNG, penalties and tokens.
- Release internal traces through a private lifecycle method independent of a
  caller's public teardown hook. [Trace lifecycle evidence](AUTODEBUG_trace_lifecycle.md)
  verifies eight constant-allocation recaptures and zero TRACE bytes after cleanup.
  Output-history scratch is released inside capture; no tracker suppression.

[Topology/candidate ledger](optimization_evidence.md), [checklist](checklist.md)
and [runtime fallback audit](runtime_audit.md) cover the complete path.

## Correctness and state evidence

| Pinned AIME24 check | Completed full model top1/top5/top100 | Final top1/top5/top100 |
|---|---:|---:|
| Prefill |96% /100% /100% |95% /100% /100% |
| Traced teacher forcing |94% /100% /100% |94% /100% /100% |

Final [prefill](prefill_prefill_trace_release_v1.json) and [teacher forcing](teacher_prefill_trace_release_v2.json)
meet top5>=98% and top100=100%. Baselines are
`../full_model/prefill_final_v2.json` and `../full_model/teacher_final_v2.json`.
Teacher-forcing decode is82.489t/s/u with its explicit reference-token/logit
callback boundary. All seven final autoregressive outputs are byte-identical to
`qualitative_release_v2`, including the French case; direct inspection and the
zero-finding degeneracy report are in `qualitative_prefill_trace_release_v2/`.

These are token-ranking checks over the pinned100-position AIME24 chat reference,
not mathematical answer accuracy over the AIME dataset. The exact checkpoint,
chat template, rendered token IDs and reference hashes are preserved.

| Final artifact | Scope |
|---|---|
| `prefill_integration_quick_v2.json` | Real layers0/3,16 exact eager-control comparisons including logical1/2048, changed tokens/pages, live shape miss/reuse, owned logits, live sampling, four-trace cleanup; watcher/tracker |
| `prefill_integration_full32_v2/summary.json` plus both lane JSON.gz files | Final reset policy, all32/native cache,13 exact eager-control comparisons, logical128/131, live seeded/penalty changes and greedy/sample/penalty generation; watcher/tracker |
| `selected_head_contract_quick_v2.json` | Reduced real layers0/3; greedy/sample/greedy and public device-output API, worker watcher plus allocation tracker |
| `prefill_integration_long_v2.json` | Exact eager-prefill control: logical1/2048, live seeded/penalty changes, greedy8, seeded and penalized128, greedy260; final logits/state/tokens and history counters |
| `full_batch32_prefill_trace_release_v1.json` | All32 layers/B32, mixed131/127/3 prompts, exact duplicate/permuted-page full logits and tokens, prior watcher/allocation controls also retained |
| `trace_batch32_release_v1.json` | Reduced layers/B32, fixed inactive slots, changed-only tables, scheduler inputs, mixed prompts and state preservation |
| `scheduler_prefill_trace_release_v1.json` | Reduced layers/B4, live sampling changes, seeded new/joined/reused slots, ongoing-state preservation and large UINT32 IDs |
| `cache_prefill_trace_release_v1.json` | External cache survives warmup; continuation127+3, duplicate logits, explicit reset, host compatibility |
| `native_context_prefill_trace_release_v1.json` | All32 layers with maximum2048 prefill trace resident, prefill262143/262144 and last-position decode advancing262143→262144; allocations recorded |
| `qualitative_prefill_trace_release_v2/` | Shared six chat prompts plus AIME100 autoregression, exact HF controls/format metadata, direct output review and degeneracy check |
| `logs/host_prefill_reset_v1.log` |39 host orchestration/lifecycle tests, no TTNN import; `--noconftest` |
| `logs/precommit_prefill_final_python_v1.log` | Applicable Python formatting and lint hooks pass; Python-only change needs no C++ build |

Synthetic token-ID/reduced-layer fixtures test state contracts, not text quality.
The shared suite is limited to128 generated tokens per prompt, AIME to100. HF and
TT commonly stop inside reasoning within these windows; completed haiku, story,
code, or AIME answer is not claimed. Actual outputs and case-specific comparisons
are retained, including the previously failing French register and seventh-request
trace lifecycle cases.

## Performance accounting and reproducibility

The optimized standalone layer estimate is24×0.355672+8×0.268965 =
**10.687848ms/token** (93.564t/s before terminal work). Adding the measured
terminal1.137086ms, split sampler0.569313ms and embedding0.060434ms gives
**12.454681ms**. Final full-model prompt2048 token-out is**12.123793ms/token**;
there is no positive gap above the10–15% closure threshold. The standalone
layers each begin inDRAM and use separate trace boundaries; the full stack
reuses L1 residuals, so this additive estimate is not a strict physical bound.

[Machine-readable accounting](perf_summary.json) reports the optimistic full
weight/KV-read bandwidth floor3.037673ms separately. The actual reduced same-run
triplet is**1.176782ms floor /2.430452ms device /2.440547ms host**. The10us
host-minus-device difference is small; the remaining device work and report
advice are tied to actual rows and the preserved decoder candidate ledger.

[Reduced profiler reports](tracy/README.md) include real embedding, one layer of
each kind, norm, full head, sampling/history and token feedback, native cache and
unchanged shapes. All32-layer profiling is deliberately avoided; full-stack
end-to-end timing is measured separately. Per-chip clocks are independent. Raw
rows are retained, with only each first pre-signpost gap excluded from the timed
window. Advice and measured dtype/program/collective metadata are classified.

Each `logs/*.provenance.json` records the exact command, environment, immutable
source snapshot/hash, native library hashes, exit status and log hash. Main
reproduction commands use `python_env/bin/python -m` followed by
`models.autoports.ornith_ai_ornith_1_5_9b.doc.optimized_full_model.run_checks`
with `prefill`, `teacher`, or `perf --output <new-path>`. Qualitative runs use
`run_qualitative --sharded-final-norm --output <new-directory>`; the harness also
records the actual norm/head/trace policy to prevent the preserved wrong-flag case. Set:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
```

Serialize hardware jobs. Watcher/allocation tracking and Tracy are separate runs.
Immutable labels, recovery details and exact profiling commands are in the
[work log](work_log.md). [Artifact manifest](artifact_manifest.json) indexes compact
checkpoint evidence and large raw captures/tensors retained only in this workspace.
[Independent review](STAGE_REVIEW.md) returns **clean-pass**. Local checkpoint
receipts are recorded in the work log; nothing is pushed.
