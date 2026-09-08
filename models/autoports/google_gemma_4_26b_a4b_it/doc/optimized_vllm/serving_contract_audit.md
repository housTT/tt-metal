# Serving contract and optimize checklist

Scope: optimize the existing selected-policy generator through the real vLLM TT
plugin and `tt/generator_vllm.py`. The mathematical decoder, terminal graph,
weight/activation/cache/CCL datatypes and mesh strategy are inherited unchanged.
[Inherited device evidence](inherited_device_contract.md) describes their actual
layouts and the limits of older profiler evidence. This stage collects no
profiler, Tracy, adapter instrumentation, or device-profiler reads.

| Relevant requirement | Implementation and evidence |
| --- | --- |
| Real serving, traced token-out decode | The manifests select `TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport`; server logs resolve the adapter module and async scheduling. `decode_forward` rejects `enable_trace=False`. The canonical generator replays its model and sampling traces using `blocking=False`. |
| Async boundary | With device sampling and `read_from_device=False`, the adapter returns a TT tensor. The plugin then calls `read_decode_output(async_read=True)`, which enqueues the token tensor's `cpu(blocking=False)` and records a completion event. Host formatting occurs after that event. The optional logprob/unsupported-feature compatibility branch is separate from measured requests. |
| Persistent inputs and feedback | Token, current-position, RoPE, per-layer page buffers, externally owned KV cache, and sampler parameter buffers retain device addresses. The sampler writes `tt_out_tok=token_input`; the graph advances positions. Host token/current/RoPE inputs refresh at scheduler boundaries, not on steady feedback steps. [Current hardware proof](trace_contract_summary.json) covers each mesh independently. |
| Page refresh | Compare unique current/snapshot pairs; share host snapshots only for actual source-object aliases. Upload changed layers only into existing per-layer device buffers. Real HMA has six five-layer groups; do not collapse distinct sliding groups. [Audit](page_table_audit.md), [host proof](page_table_host_after.json), and current hardware page-change checks cover this. |
| Safe trace lifetime | Reuse across warmed same-shape prefill. Retire traces when prefill compiles new programs, explicit host compatibility is used, row remapping occurs, or physical execution shape changes. Program-cache entry counting is a host metadata query. Allocation tracking remains enabled for the focused regression; no new corruptible allocation scopes were added. |
| Greedy split sampling | Preserve temperature-zero semantics after plugin parameter normalization. The canonical generator selects its existing semantic greedy key and Sampling1D local top-32 split sampler; no adapter argmax, full-vocabulary readback, or eager sampler is used by greedy benchmark requests. Per-request unseeded epochs no longer force distinct greedy trace keys. |
| Larger batch correctness | One active request uses genuine B1; more active requests use the existing padded lane space. Tests cover cold and warmed B2/B3/B2 transitions and async admission/early completion. The CI workload requests 32 sequences, and full plugin sampling tests include mixed and over-capacity request batches. No aligned-only or B1-only restriction is added. |
| Cache ownership and context | The adapter allocates and consumes vLLM-owned per-layer caches. It does not assume the standalone generator's cache. No new persistent device buffer shapes are introduced; existing trace allocation lifetimes are extended. [Context contract](../context_contract.json) retains 50,624 / 262,144 / 262,144 for P150/P150x2/P150x4. |
| Runtime cleanup | Explicit idempotent worker shutdown closes model submeshes, selected mesh and physical parent through the existing helper. Router teardown clears completed NoC packet tags before firmware handoff. [Hardware process-lifetime proof](kernel_cleanup_experiment.md), [worker tests](worker_cleanup_experiment.md), and [real-serving shutdown/reopen](cleanup_evidence.json). |
| Performance accounting | Paired `before_warmed` / `after_warmed` use the shared runner, identical settings and successful explicit warmups. Primary 128/128/1 supplies TTFT, TPOT, ITL, aggregate output throughput, and `1000/mean(TPOT)` decode t/s/u. CI 100/100/32 is secondary capacity evidence. Device time and roofline fields are null for this serving stage. Final performance reconciliation is in `perf_summary.json`. |

The current hardware regression uses two real layer kinds with the real terminal
path. It proves allocator, persistence, refresh and replay behavior; full-stack
API/sampling/qualitative benchmarks supply serving evidence. The short probe's
allocation size is not a reduction of served, evaluated, or advertised context.

Retaining traces across prefill changes the lifetime overlap even though tensor
shapes are unchanged. [Independent per-profile accounting](context_lifetime_audit.md)
includes that overlap, all-row prefill logits, the actual hybrid-cache blocks,
rounded trace reservation and the recorded 2,048-token scheduler chunk budget.
It preserves positive headroom without claiming an identical peak allocation
or a measured allocator high-water mark.

Unsupported seeded sampling, penalties, logprobs and structured-output cases
retain the integration's explicit compatibility path. Those tests validate API
capability; they are not evidence of all-device execution for those unsupported
features. Measured primary/CI greedy requests and supported sampled qualitative
requests use the device sampler. Full-logit diagnostic controls are labelled
separately in their artifacts.

Broad decoder precision/geometry/collective-family searches are not rerun or
claimed as new serving experiments. The selected datatype-sweep policy and
strongest token-out full-model controls remain the comparison baseline. No
candidate is rejected here on synthetic precision evidence or on an unrelated
older profiler's dtype. The operation-topology audit in [work_log.md](work_log.md)
identifies the adapter boundaries changed in this stage.
