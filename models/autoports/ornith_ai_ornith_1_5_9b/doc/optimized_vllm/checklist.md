# Serving optimization checklist

Scope: stage10, full32-layer Ornith, selected datatype-sweep policy, TP4 on four
Blackhole chips on P300c boards. All optimization evidence below is complete. Independent review returned `clean-pass`; local checkpoint receipts are in work_log.md.

| Optimize requirement | Serving-path evidence and action |
| --- | --- |
| Operation topology | Work log audit: unchanged decoder stack/terminal, external-cache prefill and first-sampling overhead investigated separately with AutoFix |
| Decoder fully traced, device feedback | Existing nonblocking model + canonical sampling traces; before_async_worker_contract covers stale host tokens/positions, changed/unchanged pages and remapping; final corrected rerun passes: after_async_worker_contract_v2.json |
| Residual/norm/shard continuity | Unchanged model and decoder: B1 L1 width-sharded hidden4096; B2..32 validated compact/DRAM contract. No adapter layout conversions introduced |
| Prefill DRAM/2D matmuls, optimized composites | Reuse same model prefill body and SDPA/paged KV/GDN kernels. Trace selection changes dispatch, not tensor math |
| Coherent topology, lower-movement residuals, CCL fusion | Preserve completed optimized-multichip candidate ledger: ../optimized_multichip_decoder/packed_default_families_resumed.json, ../optimized_multichip_decoder/validated_full_attention_families.json, ../optimized_multichip_decoder/optimization_evidence.md. No new topology/precision family proposed in this serving pass |
| Packed projections, matmul geometry, independent fidelities | Preserve selected BFP4/LoFi body, BFP8/LoFi decode QKVG, layer31 exceptions and head32-core/K4/two-reader geometry. ../datatype_sweep/selected_precision_config.json and final_full_model_tokenout.json runtime summary validate materialized tensors/configs |
| Blackhole reader/grid/padding trials | Inherited decoder and datatype head geometry search, same selected runtime policy; no generic reader default changes |
| Persistent CCL resources | Existing embedding/candidate-gather buffers and chosen decoder collectives unchanged. Prior adapted slower CCL alternatives remain documented in optimized-full-model and multichip ledgers |
| LM head and canonical split sampling | LMHead1D, local65536 logits, invalid-ID mask, physical top32 per shard,128 gathered candidates; semantic greedy1. Full-model sampler comparison rejects slower force-argmax. H2 removes eager first-token sampling; h2_sampling_b1.json and h2_sampling_b32_final.json prove exact canonical replay |
| Trace lifetime and persistent buffers | H1 external prefill tracker probe covers native pool, logical128/131, changed inputs/pages, external sentinels, public output ownership, repeated snapshot/recapture and zero trace allocation after teardown. H2 B1/B32 exact sampler proofs and final full32-admission rerun pass |
| Context and non-aligned capability | Native262144 remains advertised/allocated. H1 reduced server first131 and65 requests pass exact usage; all32 native external capacity262143/262144 and last-position decode pass; after_b1_nonaligned.json and after_b32_nonaligned.json pass logical131/65, repeats and concurrent A/B/A against prior token controls |
| Async split/no host fallback | Adapter returns device token buffer, queues one token shard host copy, plugin waits/formats afterward. Explicit logprob compatibility is separate; both performance servers disable it |
| Correctness/PCC/stress | H1 reduced18 exact complete-logit/state/next-decode comparisons pass (stronger than PCC tolerance); all-layer H1 and H2 B1/B32/full32-admission pass;87 final CPU regressions pass |
| Watcher | Baseline worker Watcher10 and native tracker pass; Ethernet watcher exclusion follows exact firmware overflow28720>26624 and successful recovery. Final candidate worker watcher and native allocation tracking pass (async, sampler, native-capacity artifacts) |
| Before/after and selected default | Before primary128/128/1 and CI100/100/32, one warmup + three measured repeats archived. Final primary_comparison.json and ci_comparison.json pass; perf_summary.json asserts exact server argv/environment equality within each before/after pair |
| Full-model comparison/accounting | Current standalone final_full_model_tokenout.json:29.013ms/87.959t/s/user at128/128/B1/native262144. Final serving35.249ms/87.700t/s/user on128/128/1 is within0.3% decode; teacher forcing and CI burst excluded from this comparison. perf_summary.json records exact metrics |
| Profiler | Intentionally no Tracy, tt-perf-report, adapter/device profiler or ReadDeviceProfiler. perf_summary.json device-time/profile fields are null with the no-profiler reason; prior non-serving device evidence supplies context only |
| Qualitative | Preserve exact upstream chat template and shared six prompts; prior HF/full-model/integration controls available. qualitative_prompt_format.json verifies template/tokenIDs; all12 outputs read in qualitative_review.md/json, all6greedy outputs exactly match prior control. vLLM-scope degeneracy check passes; controlled haiku/truncation limits explicit |
| Batch capability | Before/after32-request bursts pass, as do final B32 native sampler, nonaligned/concurrent and stale-slot checks. Canonical full sampling profile passes72/1expected skip on final_b32_compat (final_sampling.log) |
| MoE/sparse nnz | Not applicable: dense hybrid attention model, no routed-expert MLP |
| Build, independent review, commits | Python-only; no C++ build required. 87 final host regressions and Python hooks pass; stage_check.log exits0. Final process_cleanup audit and bounded four-chip health pass; STAGE_REVIEW.md returns clean-pass; local stage-owned checkpoint receipts are in work_log.md; no push |

Predecessor evidence is inherited only where the measured model implementation
and selected policy are unchanged. Historical BF16-head geometry/precision
claims in optimized-full-model are superseded by datatype-sweep; serving uses
the selected BFP4 head and layer31 exceptions.
