# Optimized runtime audit

The selected path remains TP4 with partitioned embedding, attention, MLP and
vocabulary weights. The24 linear and8 full-attention decoder objects are imported
from the completed optimized multichip implementation. Their BF16 replicated
hidden4096 residual passes directly between layers; B1 remains L1 width-sharded,
B2..32 retains its validated compaction/DRAM contract. No new inter-layer gather,
reshard, weight replication or rejected precision policy is introduced.

The terminal consumes the existing B1 L1 shard, pads logical rows on device in
that shard, runs explicit sharded RMSNorm, and calls common LMHead1D with existing
TP4 vocabulary-packed weights and the selected BF16/HiFi4 K1/32768/two-reader
program on64 input cores (8x8, shard[32,64], per_core_N16). The final
norm retains8x4/shard[32,128] and explicitly reshards only its output. Local logits are BF16 width65536, with valid vocabulary248320 separately
recorded. Sampling masks invalid columns before local top32 and gathers only128
candidates. Semantic greedy is k1/p0/temperature1. Common SamplingGenerator owns
parameter, penalty and seed state; top-k/top-p and seeded sampling use the same
split trace wrapper. Force-argmax and full-vocabulary gathering are absent from
this measured path.

Model trace writes sampler-ready logits and advances position/RoPE on device.
The separate sampler trace writes directly into persistent `tt_out_tok`, which
is the next decode input. A second sampler variant additionally appends UINT32
output history and advances its device index; each step still submits exactly
one model trace and one sampler trace, both blocking=False. Plain token-out
`replay_decode(steps)` has no host boundary. `decode_forward(...,
read_from_device=False)` returns that same persistent tensor, whose contents
are overwritten by subsequent replay. Caller scheduling owns context bounds.

Reusable prefill tracing is enabled by the generator-only `use_prefill_trace=True`
option; `False` selects the same implementation's eager control. Eligibility is
device sampling with an owned B1 cache, one fresh request in slot0 at start0,
and logical length1..2048 (the configured prefill chunk limit). Length131 is
eligible without changing its logical length: the existing decoder owns padding,
trimming and last-valid-token selection. Shared `validate_prefill` checks token
IDs, lengths, slots, context bounds and physical page addresses before either
execution path changes state. Continuations, ragged/mixed B1..32 slots,
caller-owned caches, longer prompts, all-logits requests and host sampling retain
their validated eager prefill paths. Native262144 cache capacity is unchanged.

Only one exact prefill shape is resident. Its key binds cache identity, logical
length, start, slot and table shape; token and page contents are refreshed in
persistent UINT32 row-major `[1,length]` and INT32 row-major `[1,page_width]`
inputs. Unchanged prefill page tables skip transfer, including across request
reset. A new shape releases the whole trace family before allocating replacement
inputs and warming. Live shape misses conservatively use eager prefill. Model,
plain sampler, history sampler and prefill form one bounded four-trace family
within the existing100MB trace region; program or sampling changes recapture it
together. Prefill is captured last, copying temporary terminal logits into the
canonical decode-logits buffer and freeing the temporary inside capture.

Public `prefill_forward(return_device_logits=True)` returns an owned clone when
traced; callers release it before decode. Private `_prefill(...,
borrow_logits=True)` lets `generate` consume the canonical buffer without freeing
it. Request seed and penalty preparation finishes before the prefill replay safety
check, because recapture replaces that logits buffer. `_sample_first_token`
reuses the existing plain sampler trace only when its populated canonical logits
and program cache remain valid; otherwise it invokes the same common sampler
eagerly. The first-token path does not execute decode, advance positions or append
history, and preserves the original caller-visible token read for TTFT.

High-level autonomous `generate` retains one initial-token read for TTFT and
collects subsequent outputs in128-step device windows, reading once per completed
window. There is no per-token token, position, RoPE or page-table host refresh.
The fixed-step loop does not read tokens to feed decode. Longer calls reset the
history index at window boundaries; EOS slices returned output after the fixed
requested window and does not reclaim finished-row compute early. This preserves
the completed full-model contract and is not a serving/streaming implementation.

Explicit host boundaries are checkpoint/tokenizer loading, request input upload,
scheduling changes, caller-requested logits/readbacks, readiness teacher forcing,
and `sampling_mode='host'` compatibility. Host sampling still accepts an explicit
callback; otherwise it allows only unpenalized greedy. These paths are separate
from the device token-out benchmark. Public logprob output or vLLM async-serving
support is not newly advertised.

New generation requests clear hybrid state and overwrite their entire live KV
prefix in prefill. Finite stale-page/permutation controls prove future KV is
masked; explicit public reset() continues to clear every KV/state buffer.
The [duplicate-reset repair](AUTOFIX_prefill_reset.md) retains that request reset
and passes private `state_already_reset=True` only to the traced B1 prefill call.
It removes a second96 in-place multiplies over the same24 recurrent and72
convolution buffers. Public prefill defaults to resetting its own fresh state;
eager capability paths, seed preparation and trace graphs are unchanged. The
paired full-model control verifies exact state/output behavior and a2.74–2.85ms
TTFT reduction at128/131; final request measurements below include the repair.

Cache, page tables, positions, prompt lengths, batch slots and hybrid state remain
explicit. Unchanged page tables skip transfer. Caller-owned warmup snapshots and
restores caches. New-request and continued-prefill seed behavior remains lane
scoped. Inactive position-1 rows preserve recurrent and convolution state; live
sampling changes preserve cached state before warming/recapturing a new mode.
Internal `_release_traces` is independent of a caller's public teardown hook,
so repeated shared-runner requests release superseded model and both sampler
traces plus the optional prefill trace. Public teardown also releases the retained
prefill inputs. The history temporary is deallocated inside capture after its copy;
no allocation-tracker suppression or corruptible-buffer exception was added.

Persistent decode embedding and named candidate-gather outputs are separately
owned by SamplingCCL and cloned for consumers that deallocate their results.
Both allocation and preallocated program signatures warm before capture.
Arbitrary prefill lengths do not accumulate large persistent output buffers.
The decoder's native collective/persistence decisions are preserved from its
matched whole-layer rejection ledger. Native collectives use cycling semaphores;
watcher and profiler are collected separately.

The generic warning about allocations after trace capture is controlled through
allocation-tracked repeated generation/recapture and scheduler tests. New prefill
programs cause boundary recapture; temporary request tensors are released before
replay. All final run commands and source/library hashes are in immutable
`logs/*.provenance.json`; README lists the completed validation artifacts.

`perf.request_counters` spans request setup through the first token read; decode
still has its separate `loop_counters`. A warm eligible request with unchanged
shape, sampling mode, programs and pages records `prefill_replays=1`,
`prefill_sampling_replays=1`, `prefill_captures=0`, `prefill_trace_misses=0`,
`prefill_eager_calls=0` and `prefill_page_table_refreshes=0`. Cold shape allocation,
warmup and capture remain included in request TTFT. First request preparation can
cause an additional safe recapture; these costs must not be presented as warm
replay latency.

Final integration passes [39 host checks](logs/host_prefill_reset_v1.log.gz),
[13 full32-layer comparisons](prefill_integration_full32_v2/summary.json), and
[16 reduced long/edge comparisons](prefill_integration_long_v2.json), including
128/260 output windows. The [integration summary](prefill_integration_summary.json)
links watcher10/ETH-exclusion/allocation-tracker provenance, separate eager
controls, changed tokens/pages, shape eviction, live reconfiguration, exact
hybrid state and next decode, and four-trace cleanup. Historical quick-v1 and
38-test receipts describe earlier sources, not the final gate.

[Final native-capacity validation](native_context_prefill_trace_release_v1.json)
retains the maximum2048-token prefill trace family while executing262143-token
prefill plus final-position decode and262144-token prefill. TRACE remains
26,542,080 bytes per device, below the100MB region. Native capacity is validated
at B1; B32 is validated at short context, not32 simultaneous native-length
requests. The [B32](full_batch32_prefill_trace_release_v1.json),
[scheduler](scheduler_prefill_trace_release_v1.json) and
[cache controls](cache_prefill_trace_release_v1.json) retain their applicable
coverage because the final private flag changes none of those execution paths.

[Final performance](perf_prefill_trace_release_v2.json) measures29.586688ms
warm request TTFT and83.314796 decode t/s/u on full32/native B1 prompt128/gen128:
37.14% lower TTFT and2.16% higher throughput than the repeated baseline. The
first request after model construction takes743.741ms including setup/warmup/
capture; model loading is excluded. [Performance summary](perf_summary.json)
keeps all warm samples, request/loop counters and the distinct plain token-out
and logits-only controls. The [final reduced profile](tracy/prefill_trace_release/decode_split_accounting.json)
records119 model and37 sampler/history operations per decode iteration; both
graphs are traced. Same-run host minus slowest-device time is10.095us per token.
Prefill and first sampling are also traced; necessary reset/seed/input/read
boundaries remain explicit in the [prefill accounting](tracy/prefill_trace_release/prefill_split_accounting.json).

[Final teacher forcing](teacher_prefill_trace_release_v2.json) passes94/100/100
top1/top5/top100 and82.488863 decode t/s/u. [All-logits prefill](prefill_prefill_trace_release_v1.json)
passes95/100/100 on its unchanged public path. The [final qualitative review](qualitative_prefill_trace_release_v2/qualitative_review.json)
verifies all seven selected-policy texts exactly, with `sharded_final_norm=True`;
the wrong-flag qualitative-v1 control is excluded. These are bounded reasoning
windows with HF controls, not claims of completed answers beyond those windows.
Technical gates are complete; [independent stage review](STAGE_REVIEW.md) returns clean-pass.

The scheduler test intentionally retains the predecessor's failing UINT32-predicate
lane-merge control beside the passing INT32-predicate implementation. Large-ID
rounding/wrong lanes in that named negative control do not describe the measured
path; `scheduler_release_v1.checks.isolated_large_uint32_lane_merge` records both.
Seed initialization, live parameter changes, partial prefill, continuing RNG state
and newly joined slots pass on the selected generator. Synthetic token-ID slot
fixtures and reduced-layer repetition are state tests, not language-quality tests.
