# Full-path optimization checklist

Scope is the completed TP4 full model on four Blackhole chips on P300c boards.
No TP1 full-model implementation existed at stage entry; no full-model TP1
speedup/efficiency is claimed. The exact completed TP4 implementation is the
before control. Existing single-chip decoder controls remain in the predecessor
stage, and are not relabeled as a full-model baseline.

Final technical gates pass, including [39 host checks](logs/host_prefill_reset_v1.log.gz),
[13 full32 watcher comparisons](prefill_integration_full32_v2/summary.json),
[16 reduced long/edge comparisons](prefill_integration_long_v2.json), native
capacity, accuracy, selected-policy qualitative and final profiling. The
[performance summary](perf_summary.json) records29.586688ms warm TTFT and
83.314796 decode t/s/u. Existing terminal/CCL v2 and intermediate receipts remain
historical evidence. [Independent stage review](STAGE_REVIEW.md) returns clean-pass.

| Applicable requirement | Evidence and decision |
|---|---|
| Complete path and topology | `optimization_evidence.md` boundary table; runtime embedding, both decoder kinds, terminal, sampler and feedback included in reduced profile; full32-layer warmed generation measured separately |
| Selected precision reaches runtime | Final raw/report rows are checked against BFP4/LoFi decoder projections, BFP8/LoFi QKVG, BF16 activation/CCL and head BF16/HiFi4 FP32 accumulation; inherited policy/rejection ledger unchanged |
| Residual/shard continuity | B1 L1 width-sharded4096, B2..32 compact/DRAM interface; `model.decode_forward` directly consumes each previous returned hidden; no new inter-layer conversion |
| Coherent collective families | Inherited `../optimized_multichip_decoder/packed_default_families_resumed.json` and `validated_full_attention_families.json`: native, carried hidden shards, delayed gather, fused AG-MM/MM-RS, dtype and persistence crossed under selected packed/QKV8 policy; native remains selected |
| Lower-movement consumers adapted | Predecessor's carried1024-hidden families pass through distributed norms, residuals and MLP without restoring old replicated layout in the timed layer; linked family tables record their slower complete-layer results |
| Dominant decoder program search | Predecessor `optimization_evidence.md`, `candidate_measurements.csv`, `packed_mlp_final_matrix.json`: precision-locked core/K/subblock/output-block/readers and packed/separate families; current default roles preserved |
| DRAM sharding and readers | Gate/up32cores K4 R3; down8cores K6 R2; QKVG32cores K4 R1. GDN/output projection multicast8x4 K8 remains the measured winner against adapted alternatives in predecessor ledger |
| Larger grids/shards | Predecessor includes16/32/64 residual grids and8x2/8x4/8x8/11x10 projection families. This stage additionally investigates precision-locked LM-head input shards/readers with full resident L1 reservation |
| Terminal layout and module | Real-hidden `terminal_contract_v2` verifies L1 padding/norm and common LMHead1D; weights are materialized LazyWeight cache entries, with existing packed TP4 ownership |
| Head geometry/fidelity | Prior BF16 head8192/16384/32768 chunk/K/readers ledger preserved. This stage's head-fidelity trial on recorded hidden finds at most2us terminal saving for HiFi2; selected HiFi4 retained. `AUTOFIX_head_geometry.md` selects64-core K1/R2, exact recorded logits; adapted3-reader and16-core controls are slower,64-core K2/R2 has a measured30464-byte collision |
| Head dtype rejection | `../full_model/AUTOFIX_french_head.md`: lower head dtypes fail real free-running French register control; frozen-hidden CPU terminal distinguishes hidden drift from local head defect. No rejected policy is selected here |
| Power-of-two logits and sampling | Each local logical shard65536; invalid tokenizer IDs masked before physical top32; gather128 candidates only. Correct greedy k1/p0/temp1 split0.572ms versus force-argmax2.744ms, both exact CPU32-row choices |
| Common sampled path | SamplingGenerator selected for parameter/penalty/seed/state/tt_out_tok contract; Sampling1D examined, no faster/slower claim without a like-for-like implementation. Greedy/top-k/top-p/seed changes tested through generator traces |
| Persistent CCL buffers | Fixed embedding and named sampler outputs preallocated, disposable ownership copies measured in `ccl_persistence_v2`. Decoder's slower persistent family remains rejected by matched whole-layer measurements; no winning decoder optimization disabled |
| Device state and trace replay | `decode_forward(read_from_device=False)` and `replay_decode`; persistent token/position/RoPE/table, nonblocking two-trace replay, device position and RNG advance, changed-only table copies |
| Reusable prefill eligibility | Generator-only `use_prefill_trace=True` default; owned B1/device/start0/slot0 exact logical1..2048, including131. `False` is the eager control. Shared validation and unchanged decoder/head policy; continuation/mixed slots/external cache/all-logits/host/long prompts remain eager capability paths |
| Prefill ownership and shape lifetime | One exact shape and persistent ID/page inputs; changed-only page refresh. Release all traces before shape replacement, eager fallback for live misses. Capture prefill last into canonical logits with temporary freed; public results cloned/owned, private generate borrows. Existing100MB region, no tracker suppression |
| Prefill request accounting | Final `perf_prefill_trace_release_v2` request counters prove warm prefill replay1/first-sampler replay1/capture0/miss0/eager0/unchanged-page refresh0.29.586688ms warm TTFT; first request743.741ms includes setup/warmup/capture but excludes model loading |
| Host boundaries | No per-token host refresh/read/wait in plain replay. High-level output history uses one end-of-window read per128 decode tokens; first-token TTFT read retained. Teacher forcing and explicit host compatibility separately labeled |
| Trace lifetime | Prior three-trace probe retains eight constant-allocation recaptures then zero TRACE bytes. Final four-trace full32 watcher comparisons and teardown pass; maximum2048 prefill family remains26,542,080 TRACE bytes/device through native262144 capability checks. Private release owns all traces; scratch freed inside capture, no tracker exception |
| Cache reset | Stale finite KV/page permutation controls pass9 lengths atB4/reduced andB1/full32/native; warmed alternating reset experiment. New generation skips whole-KV clear; explicit reset() still clears all state/KV |
| Duplicate hybrid reset | `AUTOFIX_prefill_reset.md`: full32 paired exact control verifies192→96 reset multiplies and2.74–2.85ms TTFT reduction at128/131. Private per-call flag after generate reset skips only traced B1 duplicate; public prefill/reset, eager paths, seeds and trace graphs unchanged |
| Logical prompt lengths | Tests include3,63/65,127/129/131,2047/2049 and262143/262144; public model/generator owns padding/masking/slicing |
| Batch/state contract | `full_batch32_prefill_trace_release_v1`, `scheduler_prefill_trace_release_v1`, `cache_prefill_trace_release_v1`: B32 short-context logits and fixed-slot/inactive/continuation/external cache controls. Final private B1 reset flag changes none of these paths; full32 integrationv2 additionally checks live seeded/penalty reconfiguration |
| Optimized composites | Existing SDPA and paged KV updates retained; packed GDN/MLP and fused SiLU/gating already selected. No hand-built attention fallback introduced |
| Watcher/stress | Separate watcher10/ETH exclusion/allocation tracking: full32 integrationv2 passes13 comparisons; uninstrumented reduced longv2 passes16 including128/260 output windows. `prefill_integration_summary.json` links provenance, exact state/next-decode checks and cleanup |
| Accuracy/qualitative | Final teacher_prefill_trace_release_v2 top1/top5/top10094/100/100,82.488863 decode t/s/u; unchanged public all-logits prefill95/100/100. Qualitative_prefill_trace_release_v2 verifies seven exact selected texts with normTrue; wrong-flagv1 excluded. Bounded reasoning windows, HF controls and degeneration review retained |
| Capability | `native_context_prefill_trace_release_v1`: full32 B1 native262143+decode and262144 prefill with maximum2048 trace family resident;26.54MB TRACE/device. Native context is B1 coverage, not32 simultaneous native-length requests |
| Before/after and default reproduction | Five-warmed-request B1 prompt128/gen128/native: baseline47.065316ms/81.551810t/s/u → final29.586688ms/83.314796t/s/u;37.14% lower TTFT,2.16% higher throughput. `perf_summary.json` preserves samples and slower first-request setup separately |
| Stack budget | 24×0.355672+8×0.268965=10.687848ms standalone estimate; full2048-context token-out compared with terminal/sampler work, with standalone-input-boundary difference explicit |
| Performance accounting/advice | Final `tracy/prefill_trace_release`:119 model+37 sampler/history decode ops, traced model/sampling; same-run reduced host2.440547ms/device2.430452ms,10.095us difference. Prefill103+sampler34 traced ops and8 request-boundary ops. Tables/CSV/provenance and optimistic floor remain distinct from full-model uninstrumented performance |
| Runtime fallback | `runtime_audit.md`; final profile proves eligible prefill and first sampling traced, without host argmax/full-vocab gathering or precision fallback. Other public shapes retain validated eager prefill; same common eager sampler remains a boundary fallback if canonical logits/program binding is unavailable |
| MoE/sparse-only items | Not applicable: this is a dense9B MLP model; no routed-expert frontier or sparse nnz choice |
| Serving-specific items | Deferred by user: no vLLM/API/streaming adapter or serving-profiler work; generator's explicit state contract preserved for later integration |
| Build/lint/independent review | Python-only stage needs no C++ build;39 host tests and formatting/syntax checks pass. Fresh xhigh [independent stage review](STAGE_REVIEW.md) returns clean-pass; local checkpoint receipts are in the work log; never push |
