# Full-path optimization evidence

The strongest completed baseline is `../full_model`, commit `2e4b8f828c`.
Its lower-level policy and measured rejection ledger are in
`../optimized_multichip_decoder/optimization_evidence.md`. This stage does not
repeat or override the datatype frontier. Four Blackhole chips on P300c boards,
1x4 TP ring; the software profile is called p150x4.

## Operation topology and tensor contracts

| Boundary | Current shape/layout and work | Candidate/action |
|---|---|---|
| Embedding | BF16 table 248320x1024 per rank; local lookup, hidden all-gather to4096 | One gather at model entry matches selected replicated decoder stream; selected preallocated decode output with disposable clone; prefill outputs temporary; logical ragged lengths preserved |
| Linear-attention stack (24 layers) | Local packed GDN projections, FP32 recurrence; BF16 replicated4096 residual, B1 width-sharded8x4 L1; B2..32 compact/padded rows distinct from active users | Preserve selected BFP4/LoFi and cumulative local geometry; no layer-boundary gather/reshard added |
| Full-attention stack (8 layers) | Packed QKVG BFP8/LoFi; local Q/K/V heads, BFP8 paged KV64 with BF16 updates, explicit SDPA; BFP4/LoFi output | Preserve cache/read-window contract and selected native collective family |
| MLP in both kinds | Packed BFP4/LoFi gate/up, DRAM32cores K4 reader3; down8cores K6 reader2 | Preserve precision-locked packed/split, reader and working-shard winners from predecessor |
| Row-parallel projection collectives | Native two-link ring all-reduce; identical residual before/after next layer | Prior adapted RS/sharded-residual/distributed-norm, AG-MM, fused MM-RS and persistence candidates measured and rejected in linked ledger; their policy stays intact |
| Final norm | Selected L1-sharded pad/RMSNorm on8x4; explicit output reshard to8x8 head | Real-hidden norm comparison and all-layer accuracy; no DRAM roundtrip for padding |
| LM head | BF16/HiFi4 FP32destacc; vocabulary-sharded65536/rank; two32768 chunks, K1,2 readers/bank | Selected64 input cores8x8, per_core_N16. Keep passing precision. Prior 8192/K4/r1 and16384/K4/r2, larger K2/K4 allocation failures and French controls remain evidence, not discarded search history |
| Logits/sampling | BF16 local width65536 (power-of-two), invalid vocabulary masked, physical local top32 then candidate all-gather128; semantic greedy k1/p0/temp1 | Preserve common SamplingGenerator, compare to exact greedy force-argmax; no full-vocabulary gather in selected path |
| Token/position/RoPE | Persistent UINT32 token tile32; INT32 positions per logical slot; UINT32 RoPE; device plus_one | Preserve two-trace split sampling with tt_out_tok pointing directly to decode input |
| Page tables/cache | Caller-owned hybrid/cache state and changed-only table copies | Retain explicit ownership, inactive masks, mixed prompts, partial continuation and non-aligned public lengths |
| Prefill dispatch | Exact logical B1/start0/slot0 length1..2048; persistent UINT32 IDs and INT32 page row; unchanged decoder and BF16/HiFi4 terminal | One reusable prefill shape, captured after the three decode/sampler traces; temporary logits copied into canonical output and freed. Other capabilities retain eager prefill; final full32/native gates pass |
| Output boundary | Baseline async host token copy + event wait for every token | Public nonblocking device-output API and traced UINT32 output history, one read per128-step window |
| Request reset | Clear full KV plus recurrent/conv state at request boundary | Selected generate reset skips KV clear after stale-page/permutation controls; explicit reset still clears all KV |

## Baseline measurements

`baseline_v1.json`: warmed B1 prompt128/generate128, native262144 cache,
46.7836 ms request-inclusive TTFT,81.5559 t/s/u with127 output reads and waits.
`baseline_no_readback_v1.json`: three otherwise identical127-step no-readback
windows,12.2568/12.2571/12.2575 ms per token (81.59 t/s/u). There are no
per-step token/position/RoPE/table refreshes, readbacks, waits or global syncs.
Two synchronizations bracket each timed window; a final validation read is outside.
This refutes host-readback latency as the largest performance gap.

The inherited optimized-layer accounting is24*0.355672+8*0.268965 =10.687848
ms/token at the predecessor's2048-prompt context. At that same context the
full-model baseline was12.381586 ms/token. The selected terminal-only path costs
about1.404 ms and split sampler0.574 ms; their sum already covers the observed
full-model overhead. This is a layer-stack lower-bound estimate, not a measured
same-run device time. Final stage profiling/accounting below records the terminal
and full-path numbers separately.

## Initial candidate ledger

- Final norm, recorded real French hidden: DRAM1.40325/1.40376 ms terminal;
  L1-sharded1.37437/1.37440 ms. Greedy identical but max logit difference0.75.
  AIME and prompt-correct full-model controls subsequently qualify this norm layout.
- First norm harness run failed because the saved hidden artifact is a dict.
  Corrected access to `hidden[0]`; no implementation failure. All devices closed;
  bounded reset completed and four-chip discovery passed before the rerun.

## Terminal and CCL candidate closure

`terminal_contract_v1` attempted to expose the physically padded row tile with
reshape; TTNN correctly rejects changing logical volume1->32. This was host
argument validation before launching the candidate. The adapted
`terminal_contract_v2` uses native L1-sharded padding followed by sharded norm
and `models.common.modules.lm_head.lm_head_1d.LMHead1D`. It matches all valid
vocabulary logits bitwise on recorded real French hidden states, eager and
traced. Alternating terminal times are1.37655/1.37709ms for the initial sharded
norm path and1.36788/1.36771ms for direct L1 padding plus common head. The original
DRAM-norm terminal was1.40325/1.40376ms. The final implementation uses the common
module with materialized LazyWeight wrappers around the existing packed device
weights: no reupload, precision change, or new weight packing. It preserves the
BF16/HiFi4 K1/32768/two-reader program. The later precision-locked input-grid
comparison selects64 cores8x8 (per_core_N16), keeping norm8x4 unchanged.
`head_geometry_c64_k1_r2_v1.json` measures1.371676->1.137086ms paired terminal
with bit-identical all248320 real-hidden logits. The adapted three-reader
33024-column padded/trimmed families are legal but slower. The exact L1 blocker
for64-core K2/two-reader is30464 bytes;16-core K1/two-reader is slower. See
`AUTODEBUG_head_geometry.md` and the AutoFix closure report for all controls.

`ccl_persistence_v2` holds payloads, ring, one link, axis/dim, DRAM memory,
semaphore manager and sampler semantics fixed. The persistent variant separates
collective output ownership from the consumer's disposable result via clone.
All128 traced repetitions match eager/CPU greedy on all four ranks. Alternating
sampling times are0.57024/0.57009ms default and0.56935/0.56928ms persistent;
embedding is0.06082/0.06083ms default and0.06048/0.06039ms persistent. These are
small component improvements, not a material whole-model speedup claim.
The final helper enables preallocated outputs for fixed decode embedding and
named sampler buffers only; arbitrary prefill shapes remain temporary. Existing
native decoder CCL persistence remains rejected by the predecessor's matched
whole-layer family measurements (above), preserving its winning policy.

The first CCL candidate warmed the allocation form but attempted to capture the
preallocated form without compiling it. TTNN rejected the new program during
capture; cleanup could not finish with a partial capture. Task-owned PID94205
was terminated, bounded list/reset/list and mesh smoke passed, and the adapted
candidate explicitly warmed both signatures. `logs/ccl_persistence_v1.*` and
`logs/*after_ccl_warm_failure*` preserve the failure and recovery. The final
helper warms the preallocated signature on its initial call, before capture.

## Reusable prefill trace integration

The [prefill prototype](prefill_gaps_v1.json) isolates eager submission gaps on
the actual reduced layers0/3, native262144 cache and TP4 mesh. At logical128,
the unchanged public prefill-plus-sampling window measures4.872266ms; the exact
prepared traced prototype measures4.224427ms, a0.648ms opportunity in that reduced
window. Its prepared eager control measures6.799340ms and is slower than the
original public path; the2.575ms difference against that control is not the
public-path gain. Changed tokens, reversed physical pages, return to the original
inputs, full recurrent/conv state and next-decode logits are exact at128 and131,
also under the [separate watcher run](prefill_gaps_watcher_v1.json).

The prototype's128 capture alone takes9.230ms, excluding input allocation and
warmup, and uses122880 TRACE bytes per bank. These are component measurements,
not final integrated full-model TTFT or combined four-trace memory. The integration
must amortize preparation on repeated shapes; no final integrated speedup is
claimed from the prototype.

The generator's default `use_prefill_trace=True` selects an owned-cache, device
sampling B1 fresh prompt in slot0, with exact logical length1..2048. The explicit
`False` control reaches the same shared validation and device chunk implementation
through eager execution. Nonaligned131 remains eligible. Continuations, mixed
and inactive fixed slots up to32, external caches, larger/native-context prompts,
all-logits requests and host sampling remain supported by eager prefill. Neither
selected decoder/head policy nor cache capacity changes.

One key contains cache identity, logical length, start, slot and page-table shape.
Changing token or table contents reuses persistent inputs; unchanged tables skip
transfer. Shape replacement releases all four traces before allocating inputs;
live shape misses use eager execution. Exact warmup precedes coordinated capture
of model, plain sampler, history sampler, then prefill. Prefill copies into the
existing canonical logits and frees its temporary before capture ends. Program
and sampling changes rebuild the complete family, and public teardown frees its
inputs. This bounds the resident shape family within the existing100MB trace
region without allocation-tracker suppression.

Public device-logits results remain independently owned clones. The private
`generate` path borrows canonical logits and uses the existing plain sampler
trace for its first token, after request seed/penalty preparation and replay safety
checks. Recapture cannot occur between filling canonical logits and consuming
them without losing that output; the first-sampler helper therefore falls back to
the same eager sampler on a later program mismatch. It does not advance decode
positions or append the first token to output history.

`perf.request_counters` records request setup through first-token read, separate
from decode `loop_counters`. Final warm same-shape/same-mode/unchanged-page runs record:

| Request counter | Measured warm delta |
|---|---:|
| `prefill_replays` / `prefill_sampling_replays` |1 /1|
| `prefill_captures` / `prefill_trace_misses` / `prefill_eager_calls` |0 /0 /0|
| `prefill_token_refreshes` / `prefill_page_table_refreshes` |1 /0|
| `readbacks` / `model_replays` / `history_replays` |1 /0 /0|

Cold allocation, exact warmup and capture remain inside request-inclusive TTFT.
Historical [quick integration](prefill_integration_quick_v1.json) passes greedy8/sample8/
greedy8 readback parity and six public no-read replays with worker watcher10,
ETH instrumentation excluded and allocation tracking enabled. Its first cold
request records two prefill captures; the sampling-mode transitions record one
each. [Host integration](logs/host_prefill_integration_v1.log) passes38 tests for
ownership, validation, shape eviction, private trace release, seed state and
existing public behavior. Exact commands and source snapshots are linked from
their [device provenance](logs/prefill_integration_quick_v1.provenance.json) and
[host provenance](logs/host_prefill_integration_v1.provenance.json).

## Final reset repair and integration evidence

[AutoFix reset evidence](AUTOFIX_prefill_reset.md) identifies duplicate fresh B1
reset work:96 mesh multiplies at request reset followed by the same96 in
prefill. The isolated [full32 paired control](prefill_reset_full_v2.json) passes
all12 exact128/131, changed-token/page and greedy/seeded-penalty comparisons.
Skipping only reset2 reduces median TTFT31.455467→28.718005ms at128 and
38.061322→35.215203ms at131. The invalid mixed seed/temperature fixture in
reset-v1 remains a historical failure receipt, not a reset correctness result.

The promoted private `_prefill(state_already_reset=False)` skips its reset only
when `generate` passes true after its existing request reset. Public prefill,
explicit reset, eager capabilities, seeds, graph capture and model implementation
are unchanged. No persistent flag or tracker exception is introduced.
[39 final host checks](logs/host_prefill_reset_v1.log.gz) cover both flag values
and prove that a later public prefill still performs its own reset.

The final selected full32/native B1 prompt128/gen128 comparison is:

| Metric | Repeated completed-stage baseline | Final selected implementation |
|---|---:|---:|
| Warm request TTFT |47.065316ms|29.586688ms|
| Decode throughput |81.551810t/s/u|83.314796t/s/u|
| First request after model construction |455.968ms|743.741ms|
| Decode output reads/waits per127 steps |127 /127|1 /1|

[Final performance](perf_prefill_trace_release_v2.json) and
[immutable provenance](logs/perf_prefill_trace_release_v2.provenance.json) preserve
all five warm samples. Warm TTFT falls37.14% and throughput rises2.16%; the
first request is slower because it includes prefill warmup/capture. Neither
first-request number includes model loading. [Performance summary](perf_summary.json)
separately reports plain token-out, traced logits-only, teacher forcing, request
counters, optimistic bandwidth floor and the inherited stack budget; no synthetic
full-stack device time is labeled as measured.

[Full32 watcher integration](prefill_integration_full32_v2/summary.json) passes13
exact comparisons. [Reduced long/edge integration](prefill_integration_long_v2.json)
passes16, including128/260-token output windows. The
[integration summary](prefill_integration_summary.json) links source/provenance,
changed tokens/pages, fresh/live shape handling, seed/penalty reconfiguration,
owned public logits, next-decode state and four-trace cleanup.
[Native capacity](native_context_prefill_trace_release_v1.json) keeps the maximum
2048 prefill family resident through262143-plus-decode and262144 prefill;
TRACE stays26,542,080 bytes/device below100MB. Native length is a B1 validation;
B32 is tested at short context. Prior final B32, scheduler and cache controls
remain applicable: the only later reset change is private to the eligible B1
generate call, and those public/eager paths are unchanged.

[Teacher forcing](teacher_prefill_trace_release_v2.json) gives94/100/100
top1/top5/top100 and82.488863 decode t/s/u; the unchanged public
[all-logits prefill path](prefill_prefill_trace_release_v1.json) gives95/100/100.
[Selected qualitative v2](qualitative_prefill_trace_release_v2/qualitative_review.json)
checks all seven texts and metadata against their selected-policy controls,
with sharded norm true. Qualitative-prefill-v1 accidentally selected the supported
DRAM norm override and is excluded from final evidence. Bounded reasoning-window
results do not claim complete answers outside their token budget.

Final profiles live in `tracy/prefill_trace_release`, with separate watcher and
profiler runs. [Decode split accounting](tracy/prefill_trace_release/decode_split_accounting.json)
records119 model operations and37 sampler/history operations per iteration;
both graphs are traced. Same-run reduced host2.440547ms versus slowest rank
2.430452ms leaves10.095us host difference per token.
[Prefill split accounting](tracy/prefill_trace_release/prefill_split_accounting.json)
shows103 traced prefill and34 traced first-sampling operations, with eight
explicit request-boundary operations. This is reduced real-shape profiling,
separate from the uninstrumented full-model headline. Existing terminal/CCL v2
and prototype receipts above remain historical evidence for their recorded
sources. Final technical gates pass; [independent stage review](STAGE_REVIEW.md) returns clean-pass.
