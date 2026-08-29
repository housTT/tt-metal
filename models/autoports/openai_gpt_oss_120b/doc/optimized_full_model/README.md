# GPT-OSS 120B optimized full model

Status: **complete; independent `$stage-review` clean-pass**.

## Headline before/after

Warmed batch-1 P150x4, pinned AIME24 chat prompt, 214 prompt tokens and 100
output tokens:

| Path | Stage | TTFT | Decode | Change |
| --- | --- | ---: | ---: | ---: |
| Split token-out, sampling/feedback included | completed full-model baseline | 3.7757 s | 51.6229 t/s/u | baseline |
| Split token-out, sampling/feedback included | optimized final | **3.6359 s** | **62.7665 t/s/u** | TTFT 3.70% lower; decode 21.59% higher |
| Traced teacher forcing | completed full-model baseline | 3.7515 s | 52.1145 t/s/u | baseline |
| Traced teacher forcing | optimized final | **3.7470 s** | **52.3066 t/s/u** | TTFT 0.12% lower; decode 0.37% higher |

Token-out is the fixed-length fast path: prefill returns GPT-OSS's inherited
pre-sampled channel-token scalar, model and canonical split-sampler decode
traces feed back on device, and the caller collects the final scalar token.
Baseline and optimized TTFT therefore use the same model-specific shortcut;
prefill logits accuracy is validated separately.  Teacher forcing refreshes
the caller-provided next token each step and is deliberately reported
separately.  The final
prompt-128/output-128 split run measured 0.4848 s TTFT and 62.8792 t/s/u, with
its first and final tokens exact against the synchronous arm.

## Accuracy and qualitative gates

| Gate | Before top-1/top-5/top-100 | Optimized top-1/top-5/top-100 | Result |
| --- | ---: | ---: | --- |
| AIME24 chat-template prefill, 100 positions | 94% / 100% / 100% | **94% / 100% / 100%** | pass |
| AIME24 traced teacher forcing, 100 positions | 95% / 100% / 100% | **95% / 100% / 100%** | pass |

The refreshed autoregressive run produced 100 TT tokens at 3.7463 s TTFT and
52.1617 t/s/u and passed degeneration checks.  The shared six-prompt qualitative
suite also has no degeneration findings; every TT prompt-token sequence is
identical to the pinned HF chat-template control.  HF/TT token agreement is
retained as informational, not used as a greedy correctness threshold.

Principal artifacts:

- `artifacts/prefill_readiness.json` (`c2cb6615...`)
- `artifacts/teacher_forcing_readiness.json` (`66ba2548...`)
- `artifacts/autoregressive/autoregressive_meta.json` (`f457341d...`)
- `qualitative/qualitative_tt_chat.json` and
  `qualitative/qualitative_hf_tt_comparison.json`

## Target, mesh, and preserved decoder policy

- Model: `openai/gpt-oss-120b`, revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Stage base: `5058406ce09eb40a1148b91d4f0f7cc485b1115e`.
- Hardware: four Blackhole boards reporting `p300c`, used as the requested
  P150x4 semantic `(1,4)` mesh.  P150 and P150x2 are represented in the mesh
  and capacity contract but cannot host the resident 120B model.
- Parallelism: 1D TP on mesh axis 1.  Logical `[1,1,batch,2880]` BF16 residuals
  stay replicated between layers, L1-interleaved for decode and
  DRAM-interleaved for prefill, with no inter-layer collective.
- Attention/cache: BFP8 attention weights and paged local-head KV cache, LoFi
  decode projections, HiFi2 prefill packed QKV, LoFi output projection, BFP8
  decode attention CCL, and BF16 prefill attention CCL.
- MoE: replicated BF16 router, top-4 indexed `sparse_matmul`, BFP4/LoFi expert
  weights, BF16 expert CCL, and the selected 45/15/45-core program geometries.
- TP4 attention reduces physical width 2944 and slices to logical 2880.  The
  complete optimized-multichip rejection ledger remains authoritative.

No rejected faster dtype/fidelity/activation/CCL policy, replicated stream, or
dense-all-expert path was promoted.  A broad datatype frontier was not run;
that is owned by the later datatype-sweep stage.

## Optimized full path

The production path covers embedding, both decoder kinds, final norm, sharded
LM head, sampler-ready logits, sampling, token feedback, position/RoPE,
KV/page state, trace lifecycle, and generator orchestration.

1. Prefill accepts logical non-aligned prompts and returns the inherited
   pre-sampled GPT-OSS channel token; this is not presented as generic sampler
   execution.  The sampler-ready prefill logits are checked separately.
2. Model decode replays over persistent token, position/RoPE, page table,
   KV-cache, residual, logits, and CCL state.
3. Canonical `models.common.sampling.SamplingGenerator` replays over TP vocab
   shards and writes through `tt_out_tok` into the persistent next-token input.
4. Position and RoPE advance on device.  Unchanged page tables are reused and
   changed content is refreshed only at the explicit request boundary.
5. Replay submission is nonblocking.  The fixed-length API exposes first and
   final scalar collection without per-token host synchronization.

The final prompt-128/output-128 trace evidence records 127 model replays and
device token-out submissions, one token/position/page/sampling setup refresh,
126 fixed greedy sampler replays, 126 unchanged page-table reuses, zero steady
token/position/page refreshes, two scalar synchronizations, zero full-logit
reads, and zero host argmax calls.

The serving-ready generator contract remains explicit: caller-supplied cache,
page table, position, prompt length, batch, fixed slots, and inactive `-1` rows
still work.  A mixed prompt-7/prompt-5 batch-2 trace preserved the inactive row
at position `-1` while advancing the active row from 7 to 8.  The all-36-layer
batch-2 full-logit gate is bitwise exact across rows and runs after bounded
device recovery (`588e9848...`).

## Non-aligned prompts and capacity

The public generator still accepts non-aligned logical lengths.  Physical KV
allocation rounds to the 128-token decode K chunk and then uses 64-token pages;
the logical prompt contract is unchanged.  Passing evidence includes mixed
lengths 5/7, prompt 8 through output position 129, and the full 214-token AIME
chat prompt through 100 output tokens.

| Target | Batch-1 full-context bytes/device | GiB/device | Resident result |
| --- | ---: | ---: | --- |
| P150 | 74,895,259,776 | 69.751646 | impossible; fixed state exceeds 32 GiB, feasible context 0 |
| P150x2 | 39,272,108,928 | 36.575002 | impossible; fixed state exceeds 32 GiB, feasible context 0 |
| P150x4 | 21,543,073,152 | 20.063550 | fits at the advertised 131072 tokens |

P150x4 supports batch 10 at full context, batch 11 through 130816, and batch 32
through 44928.  The batch-1 advertised context remains 131072; the physical
limits on P150/P150x2 are recorded rather than hidden behind a runtime fallback.

## Greedy and top-k/top-p sampling

The selected greedy path remains semantic top-k=1/top-p=0 through the canonical
split sampler.  It does not all-gather full vocabulary logits.  A real-weight
two-layer P150x4 A/B produced 32 exactly matching tokens:

| Candidate | Decode | Decision |
| --- | ---: | --- |
| Canonical split greedy | **154.2817 t/s/u** | selected |
| Generic force-argmax/full-vocab gather | 139.1412 t/s/u | rejected; 9.81% slower by decode time |

The eight-case sampler-only trace matrix covers feedback on/off, fixed versus
refreshed sampling state, and per-step versus final-only collection; every
128-step case is exact.  A production full-path top-k/top-p trace with
temperature 0.8, top-k 20, and top-p 0.9 records four sampling replays,
`tt_out_tok` feedback identity, persistent token/position/RoPE state, two scalar
reads, and zero logit reads.  Greedy therefore stays on device without accepting
force-argmax as a shortcut.

## LM head, profiler, and lower bound

Five final-source reduced full-path phases contain raw profiler CSVs,
`tt-perf-report` tables, CSV reports, phase evidence, logs, and hashes under
`artifacts/profiler/final_source/`.  The reduced model has a real embedding, one
sliding layer, one full-attention layer, final norm, LM head, canonical sampler,
feedback, trace replay, and real cache/page shapes.

| Policy/phase | Signpost window | LM head | Complete sampler | Result |
| --- | ---: | ---: | ---: | --- |
| Selected interleaved prefill | 29,698.179 us | 527.122 us | 352.354 us | pass |
| Selected teacher-forcing decode | 2,339.117 us | 707.135 us | 351.306 us | pass |
| Selected split token-out | **2,335.916 us** | 707.269 us | 351.678 us | pass |
| DRAM-sharded candidate prefill | 29,852.084 us | 676.357 us | 352.376 us | 0.518% slower |
| DRAM-sharded candidate split | 2,342.560 us | 674.740 us | 351.686 us | 0.284% slower |

The candidate performs eight 8192-column BFP8 matmuls plus eight conversions,
retains the base weight, and adds 200,540,160 bytes (191.25 MiB) per device.
It preserves sampled tokens but loses end to end, so the selected one-matmul
65536-column DRAM-interleaved head remains the default.  A 16384-column split
was rejected because its 2,229,248-byte static CB request exceeds the
1,572,864-byte limit.

In the selected split window, the LM head is 30.278% and the complete sampler
chain (`TopkLargeIndices`, both `TopK` rows, and `Sampling`) is 15.055%.  The
sampler is not dominant; force-argmax and full-vocabulary all-gather are absent.
The advice table still flags the LM head as the largest terminal optimization
target, which agrees with the rejected candidate study.

The optimized multichip layer medians give this conservative floor:

| Term | ms/token |
| --- | ---: |
| 18 sliding layers | 7.13544534 |
| 18 full-attention layers | 7.13436516 |
| Decoder stack | 14.26981050 |
| Selected LM head | 0.70726900 |
| Complete sampler chain | 0.35167800 |
| Stack plus named terminal floor | **15.32875750** |
| Measured prompt-128 split token-out | **15.90350675** |
| Conservative residual | **0.57474925 (3.75%)** |

Embedding, final norm, feedback helpers, and residual op gaps remain in that
positive residual, so the closure is conservative and comfortably inside the
15% gate.  Exact arithmetic and sources are in
`artifacts/full_path_lower_bound.json`.

## Correctness recovery, watcher, and fallback audit

An earlier source-unchanged all-layer run exposed stale external fabric/CCL
state: repeated sampler-ready logits diverged while the sampler chose each
capture's correct maximum.  A bounded physical reset restored the unchanged
source.  The final all-layer acceptance then passed repeated synchronous runs,
an exact pre-unseen-prefill control with release count zero, one safe trace
release for the unseen prefill bucket, and split endpoints 7, 8, 122, and 128.
The strict bitwise batch-2 gate was retained; no semantic relaxation remains.

The fully enabled watcher initially found an unused scatter-state initialization
for the sampler's one-page UINT32 all-gather.  When `use_scatter_write` is false,
multicast and standard-unicast constructors now compile out that invalid setup;
the active unicast path is unchanged.  Isolated sampler and UINT32 CCL tests
pass, and the final reduced full path passes with watcher disabled features
`None`, including greedy, top-k/top-p, feedback, cache, async collection, and
teardown.  See `artifacts/watcher/final_source/provenance.json`.

`runtime_fallback_audit.md` is clean for the measured path.  Host argmax and
full-logit conversion exist only behind explicit compatibility/validation
branches.  The optimized split path is fully traced and has no per-token host
boundary.

## Checklist evidence

The applicable `$multichip` items are covered by the TP1/TP2/TP4 mesh/capacity
plan, TP4 resident full stack, sharded embedding/LM-head/cache layouts, explicit
collective and residual boundaries, bitwise batch-2 replay, CCL profiler rows,
and fully enabled watcher evidence.  The hard P150/P150x2 capacity limits are
reported with the largest feasible resident context rather than masked.

The applicable `$optimize` items are covered by warmed end-to-end before/after
measurements, trace-boundary counters, LM-head/sampler A/Bs, retained decoder
program-config/kernel ledger, complete profiler tables/CSVs, lower-bound closure,
watcher separation, and the clean fallback audit.  No broad datatype search was
performed.

## Limitations

- P150 and P150x2 cannot host the fixed resident 120B state in 32 GiB/device.
- Fixed-length split token-out reads first/final tokens only and cannot stop on
  EOS without an explicit caller collection boundary.
- The shared generic high-bandwidth CCL fixture's unrelated fabric-firmware
  size failure is outside this measured model path; the production UINT32 CCL,
  sampler, and full reduced watcher paths pass.
- Datatype Pareto selection and vLLM integration are later stages and were not
  started here.

Commands, recovery provenance, exact logs, and SHA inventory are recorded in
`work_log.md`, `artifact_inventory.json`, and the artifact-local provenance
files.  Independent `$stage-review` returned `clean-pass`; the stage-owned
changes are checkpointed locally and are never pushed by this workflow.
