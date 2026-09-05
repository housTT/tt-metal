# Runtime audit

The all-layer wrapper imports the existing `MultichipDecoder` directly. Each
returned residual feeds the next decoder with no boundary conversion. Projection,
activation, CCL, BFP8 paged-cache/BF16-update split and FP32 recurrent policies
remain in that implementation; no copied decoder or replicated-weight fallback
exists. TP4 shape is checked at construction. Embedding weights shard hidden;
the embedding result is gathered once to the decoder's replicated residual.
Final norm and vocabulary-sharded LM head produce local65536-column BF16 logits.
The common sampler gathers only top candidates, with padded vocabulary masked.

There are explicit host boundaries: checkpoint/tokenizer setup, prefill token
upload and scheduling, readiness `return_all_logits`/`return_logits`, token
readout required by the caller, and `sampling_mode='host'` compatibility. The
optimized token-out path calls none of the host-logit/argmax paths. Host sampling
accepts an explicit callback with full logits, parameters, prompt/history and
step; without it only greedy unpenalized sampling is accepted.

The public device API returns tokens or requested logits. Common-sampler logprob
computation is available internally but logprob results are not exposed by this
generator contract; no public logprob-output support is claimed.

The model and sampling traces are separate. Sampling writes the persistent
`tt_out_tok` input; model replay advances current position and rotary indices on
device. Seed initialization is a request-boundary host copy; RNG counters advance
inside sampling trace. Inactive sampler counters may also advance; a new-request
prefill reinitializes its selected lane before sampling, while continuation
prefill preserves it. A scheduler may explicitly replace tokens/positions/page
table; unchanged page tables compare equal and skip transfer. Prefill shapes that
introduce programs cause boundary recapture before replay. No host masks, page
tables or feedback tokens are rebuilt between optimized decode replays. Token
reads are pipelined behind the next queued step and do not feed the next input.

`ModelCache` is explicit and caller-ownable. A generator binds one cache before
capture. External cache warmup snapshots and restores all KV/recurrent/conv
buffers; explicit reset clears them. Mixed prompts use separate fixed slots,
with B1 prefill scratch state copied into the selected batch rows. Inactive rows
keep position-1 and preserve recurrent/conv state. B1 standalone generation
requires an active request. EOS output slicing occurs after the fixed trace
window; this preserves returned autoregressive semantics but does not reclaim a
finished request's compute early. Future serving scheduling may inactivate rows
at explicit scheduling boundaries. No serving adapter is part of this stage.

Validation artifacts and final tracker/watcher verdicts are listed in README.

Final cache/state probes use watcher and trace allocation tracking independently
of profiler runs. `trace_b32_canonical_norm` covers the model and sampling replay
with31live/1inactive and32live slots; `scheduler_sampling_final_v3` covers live parameter
changes, partial prefill and new-slot join. Predicate dtypes deliberately match
the selected WHERE LLK: BF16 for conv histories, FP32 for recurrent state,
INT32 for integer state and UINT32 token/seed payload selection. This changes
predicate representation only, not stored state precision.

The native width8 Q/K RMSNorm receiver originally packed partial sums in a
cyclic order that varied by worker-owned row group. BF16 accumulation made
identical batch rows differ, and the all-layer B32 test exposed a greedy token
change. The repaired single-stage RMSNorm receiver retains cyclic NoC read
issue order but stores partials in canonical peer order. Frozen real Q/K inputs
now give bitwise identical32 rows and preserve the original B1 anchor on all
four ranks; the full32-layer test checks exact duplicate/permuted full-vocabulary
logits and greedy outputs. ROW_MAJOR and COL_MAJOR native regressions pass under
worker watcher. Precision, layout and fidelity are unchanged. See
[AUTOFIX_norm_row_order.md](AUTOFIX_norm_row_order.md) and
[full_batch32_canonical_norm.json](full_batch32_canonical_norm.json).

Native prefill emits TT-Metal's generic warning about allocations after trace
capture. Temporary prefill/seed-merge tensors are released before replay, and
new compiled programs cause boundary recapture. The warning is controlled by
source inspection, allocation-tracked cache/scheduler/full-batch probes, and
actual native boundary execution; it is not a detected corruption event.

Final selected-policy all-layer verification is `full_batch32_final_v2`:32 live
slots, mixed131/127/3-token prompts, exact duplicate and physical-page-permuted
full-vocabulary logits, seven model/sampler replays with zero per-token host
refreshes, worker watcher and trace allocation tracking. Its instrumented batch
timing is diagnostic and is not a serving-throughput claim.
