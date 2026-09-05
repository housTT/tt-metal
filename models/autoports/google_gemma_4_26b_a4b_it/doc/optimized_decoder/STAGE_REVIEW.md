# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Validate the actual BFP8 cache through numerical traced correctness gates.
  Evidence: `tests/test_optimized_decoder.py:176` installs BFP8 caches only when explicitly called. The traced batch wrapper at line 540, mutable-buffer wrapper, and wrap-stress wrapper never call it. Consequently, the trace entries in `candidate_runs/review_selected_gate22_down11_bfp8_cache.json` advertise a BFP8 environment override but execute the oracle's BF16 caches; their PCCs match the BF16 controls exactly. The final BFP8 boundary test at lines 421–533 uses real BFP8 caches but checks only shape, source dtypes, and finite eager output. Final BFP8 profiler rows prove the performance path uses BFP8, but the performance test has no numerical oracle.
  Why this matters: The recommended fastest cache policy lacks the numerical, cache-consuming trace coverage required by the cumulative optimized contract. Finite output does not establish PCC 0.995 or correct replay with refreshed cache state.
  Required next step: Wire explicit cache dtype selection into the traced correctness harness and assert the instantiated cache dtype. On final code, run numerical prefill-to-traced-decode and repeated/mutable replay coverage for both layer kinds, including non-aligned logical lengths, sliding wrap, and natural/shared full-cache views. Preserve batch coverage for the advertised supported policy. Stamp the actual exercised dtype and results.

- P2: Complete the selected BFP4 expert geometry and packed/separate comparison.
  Evidence: `candidate_runs/expert_geometry_n1_gate44_down22_perf.json` measures gate block 44 only with BFP8/LoFi. The BFP4/LoFi matrix contains block 11, block 22, and N2 candidates, but no block-44 trial or exact blocker under BFP4. The final gate row remains `SLOW`: sliding decode row 1488 is 84.907 us, 48 cores, block 22, subblock 1x1, and about 39% modeled DRAM utilization. The preserved packed/separate expert comparison predates the selected BFP4 gate policy; no corresponding tuned BFP4 separate-gate/up candidate is recorded.
  Why this matters: OPT-010/014 require packing and material geometry to be evaluated under the selected precision policy. A larger-block BFP8 result cannot reject the BFP4 geometry, and an earlier packed BFP8 win does not establish the final BFP4 packed winner.
  Required next step: Compare block 44 against the selected block 22 under the final BFP4/LoFi cumulative path, keeping down geometry fixed; record an exact blocker if illegal. Compare packed BFP4 gate/up with the best legal separate BFP4 gate/up plus fused GeGLU, including all splitting/layout costs and whole-layer traced latency. Keep the fastest candidate that passes the real-weight gates.

- P2: Earn the DRAM-sharding and dense-packing rejections within the final R22 contract.
  Evidence: The QKV/O reader artifacts use `residual_shard_cores=0`; their recorded attention defaults are sliding HiFi4 and full LoFi, whereas the final R22 path uses HiFi2. The dense-reader artifacts explicitly disable the residual chain. In final source, R22 attention at `tt/optimized_decoder.py:1997` and the dense path at line 2270 bypass `decode_dram_weights`; the alternate `_linear` path at line 1506 immediately restores DRAM-interleaved output. Thus the existing knobs do not measure DRAM-sharded weights consumed through the selected sharded residual chain. The README's claim that packed dense gate/up is incompatible with R22 likewise has no preserved adapted R22 attempt or exact op-contract repro.
  Why this matters: These experiments establish results for a different topology and, for attention, different fidelity. They do not reject the compatible combination requested by the cumulative optimization contract, while final QKV/O rows still recommend DRAM sharding.
  Required next step: Measure the material DRAM-sharded projection candidates through R22 with final per-role precision/fidelity and sharded consuming boundaries, including legal one/two/three-reader choices and their required padding. Compare a legal packed dense R22 family against the tuned separate family, or preserve an exact blocker after adapting packing/splitting/layout. Do not rank an unused environment knob as a tested runtime candidate.

- P2: Correct prefill expert accounting and close the remaining movement/advice ledger.
  Evidence: All final prefill CSVs label sparse projections `active=8/128`, and the documented report command applies `--active-experts 8` to both phases. However, `_moe_prefill_chunk` at line 2475 forms the union of routes over a 32-token chunk and uses runtime-inferred `nnz`; eight routes per token does not mean eight experts per chunk. Also, final prefill O rows 757/766 retain `in0_block_w=1` with actionable larger-block advice, and sliding QKV row 746 recommends a larger grid. No matching prefill attention configuration trials or precise blockers are recorded. The movement claim at `work_log.md:52` is contradicted by four untilize rows and a tilize row in each final decode report, around routing/scatter/sparse metadata.
  Why this matters: The prefill expert count misstates the measured work and affects modeled bandwidth/FLOPs and advice. The current checklist claims all actionable advice and layout round trips are closed without accounting for these concrete rows.
  Required next step: Generate or clearly qualify prefill reports using the actual per-chunk route union rather than the per-token top-k override. Try the remaining applicable prefill block/grid/memory recommendations, or record exact adapted blockers. Reconcile each retained routing/layout conversion with its API contract and any attempted removal; correct the zero-round-trip claim. Classify plainly inapplicable advice, including BFP8-fidelity advice emitted for the FP32 router, with the actual dtype reason.

- P2: Reconcile final evidence references, commands, and numerical labels.
  Evidence: `final_perf_results.xml` is the old 08:00:18 run lasting 43.427 s; the claimed final 48.156 s run exists as `perf_results.xml` at 11:09:39. `post_watcher_health_summary.json` contains only the 08:47 run with old hashes and eight tests, not the claimed final 11:13 health record. `work_log.md:228` uses `GEMMA4_OPT_KV_CACHE_DTYPE=BFLOAT8_B`, which the helper silently ignores; actual final JSON uses `bfp8`. The documented prefill signpost `OPTIMIZED_PREFILL_BEGIN` is absent from raw captures: they use `PERF_PREFILL_layer0_sliding_attention_seq1024_batch1` and the corresponding full-layer name. README row 151 and the work log call 0.988527/0.991454 minimum-user PCCs; they are aggregates. The actual G22+BFP4/G22+BFP8 minima are 0.771947899/0.782363706, both at user 15. The router-L1 artifact and XML contain performance tests only, so they do not support the claim that its numerical correctness passed.
  Why this matters: Reproduction commands currently select the wrong cache mode or signpost, final references include stale records, and the B32 anomaly is materially understated.
  Required next step: Point to the existing correct final XML, preserve the already-observed final health record if available, use executable cache/signpost commands, and distinguish aggregate/minimum-user metrics. Describe the rejected router candidate as runtime-tested unless numerical evidence exists. Label top-level timing JSONs as candidate outputs or restore final snapshots; they currently contain the reverted router-L1 candidate. These documentary corrections do not require repeating measurements whose valid final artifacts already exist.

## Other Concerns

- The selected code and final profiler rows agree on BF16/HiFi2 sliding attention, BFP8/HiFi2 full attention, BFP4/LoFi B1 expert gate/up, BFP8/LoFi expert down, and FP32 routing. All four final profiler op sums and gap sums recompute exactly.
- The final default suite records 33 passes and 12 gated skips; context records four passes; `perf_results.xml` records four passes; BFP8 performance and serving prefill each record two passes; watcher records 11 passes. The separate context/performance/serving runs cover the advertised categories of skipped long checks.
- Persistent buffer accounting sums correctly to 2,116,256,768 and 2,111,524,864 bytes, and the 25/5 projection sums to 63,464,043,520 bytes. It includes the separate 588,251,136-byte B32 packed expert allocation and deduplicates attention aliases.
- Precision probes use real checkpoint weights with seeded `torch.randn` hidden states. These are useful layer-isolation tests, but the report should distinguish them from recorded target-model activations.

## Hard-Check Gaps

- Full-attention B32 HF correctness repeats one prefix and one decode vector across all users. Its identical per-user PCCs are expected from that construction. The distinct-input mutable test compares optimized eager execution with optimized trace execution; it does not independently establish HF correctness for distinct full-attention users. Broader per-user HF coverage would strengthen the batch contract, but no current cross-user correctness failure was established by this inspection.
- Candidate provenance stores constructor defaults and environment intent, not every resolved runtime property. The BFP8 trace gap above demonstrates where intent can differ from execution.
- This review ran no hardware or performance experiments. Historical and final runtime results were assessed from source and preserved artifacts.

## Anomaly Ledger

- Observed anomaly: Full-attention SDPA rejected obsolete cache-write keywords.
  Evidence: `AUTODEBUG.md`; current attention source; final natural/shared-cache PCC results.
  Affected path: Full-attention paged decode.
  Control or comparison: Correct functional typed geometry helper.
  Likely subsystem: Python caller/API migration.
  Investigation performed: Inspected separate cache-write and SDPA argument expansion.
  Resolution: fixed.

- Observed anomaly: Sharded QKV splitting tripped watcher bounds assertions.
  Evidence: `AUTOTRIAGE.md`, `AUTOFIX.md`, focused watcher XMLs, and final watcher XML.
  Affected path: Residual-sharded attention head creation.
  Control or comparison: Interleaved splitter and residual-disabled controls.
  Likely subsystem: Upstream sharded-reader terminal coordinate lookup.
  Investigation performed: Checked the retained L1-interleaved conversion, immediate release, and runtime splitter assertion.
  Resolution: fixed within this decoder; the upstream defect remains.

- Observed anomaly: One 262143-token prefill stalled during host upload.
  Evidence: `triage/context_capacity_tt_triage.txt`, compact summary, pinned/unpinned control XMLs, and final four-case context XML.
  Affected path: Large allocation/upload before model execution.
  Control or comparison: Fresh unpinned 40.90 s and pinned 41.38 s passes; final combined context pass.
  Likely subsystem: Pinned-host transport/prefetch.
  Investigation performed: Inspected captured dispatch/prefetch classification and control outcomes.
  Resolution: controlled; the precise transient transport cause remains unproven.

- Observed anomaly: G22 B32 attention caused one severe per-user PCC loss.
  Evidence: `final_selected_default_b32_trace.json` and `isolate_b32_g22_bfp8_expert.json` show user-15 minima 0.771947899 and 0.782363706.
  Affected path: Rejected G22 multi-user attention family.
  Control or comparison: Final G8+BFP8 minimum-user PCC 0.995802051; identity mapping passes for sliding users.
  Likely subsystem: Attention geometry/numerics.
  Investigation performed: Recomputed minima and compared the G8/BFP4/BFP8 isolation matrix.
  Resolution: controlled by the selected batch-aware policy; numerical labels require correction.

- Observed anomaly: BFP8-labeled trace correctness still exercises BF16 caches.
  Evidence: Cache-selector call sites and `review_selected_gate22_down11_bfp8_cache.json`.
  Affected path: Cache precision acceptance evidence.
  Control or comparison: Actual BFP8 performance rows and finite-only BFP8 boundary test.
  Likely subsystem: Test policy propagation.
  Investigation performed: Traced cache construction through each wrapper and the reused oracle.
  Resolution: more-work-needed.

- Observed anomaly: Prefill profiles assume eight experts while runtime infers a route union; the movement audit omits visible tilize/untilize rows.
  Evidence: Final prefill/decode CSVs and routing/MoE source.
  Affected path: Topology, traffic, and advice accounting.
  Control or comparison: Decode's single-user exact eight-route contract differs from prefill's grouped union.
  Likely subsystem: Profiler annotation and report interpretation.
  Investigation performed: Inspected sparse arguments and counted final layout operations.
  Resolution: more-work-needed.

## Scope Inspected

- Goal/skill paths: Supplied optimized-decoder contract; `.agents/skills/stage-review/SKILL.md`, `.agents/skills/optimize/SKILL.md`, `.agents/skills/tt-device-usage/SKILL.md`; section 4 of `tech_reports/LLMs/llms.md`.
- Artifact paths: `models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/`, including README/work log, candidate matrices, final JSON/XML, all four final prefill/decode reports, allocation/profiler summaries, debugging reports, and context triage; `doc/context_contract.json`; matching raw final captures under `generated/profiler/`; fused-stage timing controls.
- Code paths: `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and relevant inherited helpers/oracles in `tt/functional_decoder.py`, `tests/test_functional_decoder.py`, and `tests/test_trace_mutable_buffers.py`.
- Frozen state: Live dirty worktree at `6e67efeb6251e655cb27c39153db5af1da90d68b`; decoder SHA-256 `452aa16b14c5e52bdc86793d0c0d8e4bf874dc35e43bac818ce0fc3fed2756c7`; test SHA-256 `8bbd16e1fe6845fdb6651e7ec9a37edb22e31f1dc639ba7ee5c15de6ea8a0cb0`. Both hashes were rechecked at review end.
- Commands run: Read-only `rg`, `sed`, `nl`, `sha256sum`, `git status`, `git rev-parse`, and `git diff --check`; Python standard-library AST/JSON/XML/CSV analyses. Both Python files parse; all 175 stage JSON files and 67 XML files parse; scoped whitespace check passes. No TT devices, servers, resets, or hardware tests were started. Only this review report was written.

## Residual Risk

- Passing isolated decoder PCC does not establish full-stack or generated-text accuracy; those stages were correctly left outside scope.
- The 262144-token capability was not reduced. Final capacity evidence establishes allocation/execution and finite output at the boundary, not full-context HF numerical equivalence.
- Hardware results are accepted as recorded evidence, not independently reproduced. The transient upload stall and upstream sharded-splitter defect remain relevant for future hardware investigation.
