# Qwen3.8-Flash-Next full-model readiness

## Readiness performance

P300 Blackhole dies 0 and 1, `1x2` `FABRIC_1D`, batch 1, real 48-layer
checkpoint, exact host-backed experts/PLE:

| Path | Workload | TTFT | Trace capture | Steady decode | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| Optimized token-out | prompt 128, generate 128 | 104.640 s | 4.227 s | 0.608555 s/token over 126 tokens | **1.643 t/s/u** |
| Teacher-forcing compatibility | AIME24 chat prompt 201, 99 decode rows | reported separately | 3.664 s | 0.511971 s/token over 98 tokens | **1.953 t/s/u** |

The teacher-forcing number reads the full logits tensor for the explicit test
compatibility contract. The optimized number uses trace-captured device
`Sampling1D`, writes its result directly to `tt_out_tok`, and reuses that
device token as the next embedding input. It performs no host argmax,
full-logits readback, untraced sampling, per-token position upload, or host
token-feedback reconstruction. Its compact one-token D2H shadow is used for
caller output and the declared PLE n-gram lookup only.

Primary evidence: `batch1_prompt128_generate128_performance_final.xml` and
`aime24_teacher_99_l1_workspace_final.xml`. The complete three-token,
48-layer token-out stack is watcher-clean in
`full48_tokenout_watcher_fixed.xml` and passes trace-allocation tracking in
`full48_tokenout_trace_alloc_tracker.xml`.

## Result

`tt/model.py` and `tt/generator.py` implement the complete Hugging Face text
path: token embedding, all 48 optimized multichip decoder layers, final
hyperconnection mixer and norm, vocabulary-sharded LM head, KV/index cache,
sampling, fixed-slot generation, and chat-template entry point. This stage
does not contain vLLM code.

The fresh reference is the first DeepSeek AIME24 prompt encoded with the
checkpoint tokenizer's chat template. It has 201 prompt tokens, 100 greedy HF
tokens, and the exact HF top-100 set at every generation position. Its model,
revision, tokenizer file hashes, template hash, source hash, command, and
bounded exact expert/PLE oracle counters are in `readiness_aime24_chat.json`;
the tensor artifact is `readiness_aime24_chat.refpt`.

Final accuracy on current model source:

| Phase | Rows | Top-1 | Top-5 | Top-100 | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill | 1 | 100% | 100% | 100% | `aime24_teacher_99_l1_workspace_final.xml` |
| Traced decode, teacher forced | 99 | 91.92% | **100%** | **100%** | `aime24_teacher_99_l1_workspace_final.xml` |

The required top-5 >=98% and top-100=100% gates pass for both phases.

## Preserved optimized path

The decoder is `MultichipDecoder.from_checkpoint_host_backed`, not a
single-chip, replicated, CPU projection, or functional fallback. The
stack-internal residual stays mesh-fractured BF16
`[1,1,4*logical_sequence_or_batch,1280]` across all 48 layers. Stack ingress
partitions once and stack exit gathers once; there is no inter-layer gather,
reshard, all-reduce, activation D2H, or activation re-upload.

Selected precision and topology are inherited without broad relaxation:

- routed experts: exact top-10 EP2 (`expert_id % 2`), BFP4/LoFi, full-K owner
  shard plus exact-zero peer, 10 fixed slots per layer and rank;
- shared projections: BFP8/LoFi; GDN projections: BFP8/HiFi2; QSA input and
  output: BF16/HiFi2; row-parallel collective partials: BF16;
- QSA paged K/V and raw-index cache: BFP8; compressed index cache: BF16;
- QSA decode 1D sharding: `qsa_input:110,attn_out:20`; the rejected
  `gdn_qkv_b_a@0:55` role was removed because a progressing-HF/full-stack
  isolate proved numerical corruption;
- `FABRIC_1D`, two collective links, 8192-byte packet payload; synchronous
  collectives; no CPU expert/PLE projection.

Canonical GDN/PLE state is DRAM-owned and is hydrated into one declared
shared L1 batch-1 trace workspace. This avoids per-layer full residency while
keeping the traced compute path and persistent state ownership explicit.

## Context and serving contract

The full stack preserves the advertised **262,144-token** context. A complete
48-layer construction at that value allocated all 12 QSA cache sets, page
table `[1,4096]`, and replicated RoPE tables `[1,1,262144,64]` without an
allocation failure (`advertised_context_construction_final.xml`). The
recomputed per-device plan is 10,170,438,744 bytes against 34,225,520,640
bytes DRAM, leaving 24,055,081,896 bytes headroom. No capability reduction is
declared; exact arithmetic is in `../context_contract.json` and
`../host_weight_contract.json`.

The low-level generator surfaces explicit model-owned KV cache, stable page
table, token and position buffers, prompt lengths, request IDs, active mask,
and fixed slots. `prefill_forward` accepts ragged or padded valid prompts and
owns 128-token internal padding, masks, cache fill, position initialization,
and output slicing. Reduced real-weight tests cover mixed lengths 1 and 33,
an inactive third row, positions `[2,34,-1]`, unchanged inactive PLE history,
unchanged-page skips, and in-place changed page-table publication. The
all-48 real generator also passes batch 32 at context 4096 with active
endpoint slots 0 and 31, 30 inactive rows, distinct/flipped page rows, two
eager device-sampled tokens, and a second request-ID epoch on the same model.
Position, page ownership, PLE history, deterministic reset, and
no-host-feedback counters all pass in
`full48_batch32_eager_fixed_slots.xml`. Its conservative plan is
23,558,946,904 bytes/die, leaving 10,666,573,736 bytes/die.

Batch-1 segmented traces are the optimized host-backed serving path.
Host-backed batch >1 uses the explicit eager low-level path because one
cohort cannot share the fixed expert-service trace safely; this is a trace
implementation boundary, not a full-model batch-capability reduction.

## Trace and sampling evidence

Decode is split into fixed-address traces for embedding ingress, every
host-service front/back segment, final mixing/LM head, sampling, and position
advance. Only declared route IDs and PLE token history cross the host boundary
between traces. A reduced trace test proves capture position 4, replay
position 5, exact equality with host argmax, direct `tt_out_tok` feedback, no
new token/position copies after capture, one unchanged page-table skip, and
one changed page table updated without rebuilding traces. A full 48-layer
three-token smoke passes under watcher after both the expert-slot lifetime
repair and the generic Linear all-gather endpoint repair.

The active-trace allocation warning was investigated with AutoDebug and then
rerun under `TT_METAL_TRACE_ALLOC_TRACKING=1` plus watcher. Both the reduced
split trace and original full-48 token-out trace pass without an unsafe
allocation error (`reduced_split_trace_alloc_tracker.xml` and
`full48_tokenout_trace_alloc_tracker.xml`). This proves the younger snapshot
allocations are released or marked safe before replay; the generic untracked
warning is classified rather than dismissed.

Non-greedy `top_k=4`, `top_p=0.95`, temperature 0.8 split sampling is also
exercised with a real reduced stack. Explicit seed 12345 is expanded through a
per-request host RNG and published to the existing persistent device seed
buffer before each non-greedy sampling execution. The test proves changing
seeds, top-k membership, direct `tt_out_tok` feedback, positions 4 through 7,
changed/unchanged page tables, and sampled -> greedy -> sampled trace release
and rebuild (`non_greedy_split_trace_final.xml`). This declared compact seed
control/H2D is absent from the optimized greedy measurements.

The 100-token AIME24 token-out audit records 98 steady trace replays, two
request-initialization token copies, two capture/reset position copies, one
page-table copy, 100 compact token readbacks, no host sampling/argmax, and no
host feedback reconstruction (`aime24_autoregressive_100_final.json`).

Both common samplers were reviewed. `TTSampling` pads to a generic maximum
batch and uses constructor-bound parameters; it is a poorer fit for this
fixed batch-1 vocabulary-sharded path. `Sampling1D` consumes the `LMHead1D`
shards directly, accepts persistent per-call parameters, and supports
`tt_out_tok`. Within `Sampling1D`, exact full-vocabulary force-argmax measured
0.6658 ms versus 0.9070 ms for local-top32 k=1 in the refreshed A/B; both
matched host argmax. Local-top32 was also rejected because it lacks the
generic sampler's lowest-global-index tie adjustment. Force-argmax is selected
and no custom sampler was written. At roughly 0.14% of a full token-out step,
sampling is not the decode bottleneck.

Explicit `sampling_mode="host"` and
`host_sampling_compatibility=True` retain the full-logits host-sampling path
needed by common correctness tests. It is never used for optimized metrics.

## Exact host-weight contract

All 512 expert IDs remain addressable. On a miss the per-layer mmap source
reads one exact BF16 expert, packs only its EP2 BFP4 owner shard plus the
shared exact-zero peer, uploads through one fixed staging expert per rank,
and publishes the LRU generation only after both projections/ranks succeed.
Hits reuse stable fixed slots and update only a compact slot-index row. Prefill
services the routed union in bounded waves of at most 10; it never runs all
512 experts.

PLE owns an exact 102,400,491,520-byte mmap table, EOS-aware 3-gram hashing,
two-token history per request, exact selected BF16 rows, an 8192-row host LRU,
and stable 128-row/one-row TT staging. Projection, gating, nine-tap
convolution, recurrence, and residual addition remain on TT.

The final prompt-128 cold/warm run measured 10,994 packed-host expert misses
and 92.197 s source packing in 104.769 s cold, then 10,994 packed-host hits,
zero misses and zero source packing in 11.316 s warm. Exact expert H2D remains
60,792,422,400 bytes in each pass because routed weights must populate wave
slots. PLE read 1,816 rows/581,120 bytes cold and zero table rows warm. Both
passes produced token 248046; equality is asserted in
`cold_warm_chunked_prefill_final.xml`.

Reset closes trace ownership, clears expert publication state and PLE request
history, restores device token/position/page buffers, and isolates mixed
requests. The CPU contract suite covers exact misses, hits, evictions,
capacity-one thrash, stale generations, failure invalidation, real PLE row
hashes, EOS/history carry, chunk carry, cancel/reset, cache hits, and
request isolation.

## Qualitative result

Free-running traced greedy generation first diverges from HF at token 5. Both
outputs remain fluent English continuations of the Aya walking-speed problem.
The TT completion has no adjacent-token repeats, a 0.06 dominant-token
fraction, 12 repeated four-grams, no language drift, no topic drift, and no
mechanical collapse. The exact HF/TT completions and the human coherence,
repetition, language, topic, and divergence verdict are retained in
`aime24_autoregressive_100_final.json`; exact HF/TT token IDs, generator
metrics, and the complete fallback audit are in
`aime24_autoregressive_100_report_final.json`.

The shared qualitative gate adds fresh exact HF controls for explanation,
coding, and summarization prompts rendered with the checkpoint chat template,
then runs 128 traced TT tokens for each. TT is coherent, English, on-topic,
and mechanically non-degenerate on all three; the summarization answer is
complete and correct. Explanation and coding remain semantically appropriate
but exhaust the 128-token ceiling before completion, consistent with the HF
control's long visible xhigh reasoning. Exact prompt formatting, raw HF/TT
tokens and outputs, and the prompt-by-prompt human verdict are in
`qualitative_prompt_format.json`, `qualitative_shared_suite_final.json`, and
`QUALITATIVE_REVIEW.md`. The restored common readiness-check degeneracy script
consumes `autoregressive_meta.json` and `tt_completion.txt` and passes with
adjacent duplication 0.0 and trigram-loop fraction 0.0759. The model-local
`_degeneracy` metrics (dominant token, adjacent repeats, repeated four-grams,
and Latin letter fraction) plus direct human review remain additional evidence.

## Performance interpretation and limitations

The inherited per-layer warmed medians imply a 111.818 ms device-decoder
lower bound (35 GDN, one PLE+GDN, 12 QSA), or 8.943 tokens/s before endpoints
and host service. Measured full token-out is 608.555 ms. A representative
final submit is 487.758 ms and includes 397.370 ms exact expert service plus
0.887 ms PLE service; the remaining full-step time includes endpoint work,
TT trace segments, compact control DMA, gathers, and Python host orchestration.
The canonical sampler is not dominant.

Known limitations are therefore explicit: exact host-backed prefill is slow
when its routed union is cold; batch-1 token-out is host-service limited;
pinned Torch host allocation is unavailable on this CPU-only Torch build;
the shared qualitative suite can hit its maximum token window while the
checkpoint exposes xhigh reasoning; and the runtime emits deprecated
CCL-argument warnings. There is no runtime
CPU math fallback, undeclared full weight residency, context reduction, or
vLLM integration in this stage.

Commands, rejected candidates, profiler provenance, failure/fix history, and
artifact mapping are in `work_log.md` and `AUTOFIX.md`.
