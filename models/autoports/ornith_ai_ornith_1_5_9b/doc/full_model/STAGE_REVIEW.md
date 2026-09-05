# Stage Review

Verdict: clean-pass

Independent review of **full-model** for `ornith-ai/Ornith-1.5-9B`, pinned HF revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. The supervisor supplied the named
stage and full contract but no numeric ordinal. Reviewed the live worktree on
`hous/ornith-1.5-9b`, based on `c61ea5a4ca101af524f7518923b993b71225c81c`,
before stage checkpoint commits. This is a fresh independent reviewer, not a
self-review or a hardware rerun.

## Required Work

- None. The two RNG findings raised during review are fixed and have focused
  device evidence. The visible French regression and full-batch logit
  discrepancy were investigated and corrected; passing aggregate metrics were
  not accepted as substitutes for those repairs.

## Other Concerns

- The shared qualitative check supports coherence and regression assessment
  within six 128-token windows. Both HF and TT mostly remain in reasoning;
  completed poem, story, code and final-answer quality are not established.
- Native 262144 evidence proves full-stack capacity, positions and execution.
  It does not establish HF agreement over an entire native-length rollout.
  Batch 32 uses shorter contexts; 32 simultaneous native caches are not claimed.
- Public token and optional logits outputs are implemented. Internally available
  common-sampler logprobs are not exposed through this generator's return
  contract. EOS trims returned IDs after fixed-window computation, without
  reclaiming finished-request compute early.
- Inactive seed counters can advance. Admission of a new or reused request
  resets only its selected RNG lane; continuation preserves the existing state.
  Pause/resume equivalence to a stream that never paused is not advertised.
- Checkpoint commits follow this review; no push or vLLM integration is part of
  the reviewed stage.

## Hard-Check Gaps

- The required CI-image C++ build was attempted through
  `.github/scripts/copilot-build.sh` (including the TTNN-test build attempt)
  but Docker is unavailable. The complete host build remains **unverified**,
  as allowed when explicitly disclosed by the repository instructions. Both
  modified device kernels JIT-compiled and ran; the generic RMSNorm ROW_MAJOR
  and COL_MAJOR regressions passed with worker watcher. This is narrower
  evidence than a successful complete CI build.
- Worker watcher and allocation tracking cover the final all-layer B32 path;
  Ethernet watcher is disabled in that run. Profiling is separate. These
  checks and source inspection do not claim exhaustive coverage of every
  native normalization geometry or deployment scheduler.
- The final selected BF16/HiFi4 head has measured legal geometry comparisons
  and failed larger-K controls. BF16 lower-fidelity speed/quality is unmeasured.
  No faster passing BF16 lower-fidelity candidate was rejected, and no claim
  that further optimization is impossible is supported or needed for this stage.

## Anomaly Ledger

1. **Observed anomaly:** scalar seeds were broadcast and omitted seeds collapsed
   to a repeated zero-derived stream; newly admitted slots inherited advanced
   inactive-lane RNG values.
   **Evidence:** `AUTOFIX_request_seeds.md`, current
   `tt/generator.py`, `scheduler_sampling_final_v3.json`, its source/provenance
   archive, and `logs/host_tests_final_v2.log.gz`.
   **Affected path:** sampled request initialization and fixed-slot admission.
   **Control or comparison:** real common formatter/SeedManager definitions,
   explicit list/scalar/None initialization, and fresh/joined/reused requests.
   **Likely subsystem:** generator adaptation of common request seed ownership.
   **Investigation performed:** source inspection identified both defects;
   formatted lane seeds and common initialization helpers now initialize selected
   requests at scheduling boundaries. Signed-predicate device merge preserves
   ongoing counters. Lists repeat; a scalar repeats lane zero only; omitted
   seeds differ by lane/request. Fresh, joined and reused explicit requests
   produce seed 275415 and token 25 after prefill. Continuation advances without
   resetting. The scheduler snapshot predates only the final head-default change;
   generator and native kernel hashes match current sources, and final B32
   checks cover the selected head.
   **Resolution:** fixed.

2. **Observed anomaly:** a generated French explanation labeled Bonjour
   “informal,” despite an HF same-prefix preference for formal/greeting.
   **Evidence:** actual `qualitative_final_v1/prompt_4/tt_completion.txt`,
   `AUTOFIX_french_head.md`, `french_head_v1/`,
   `french_head_geometry_v1/`, and final
   `qualitative_final_v2/prompt_4/tt_completion.txt`.
   **Affected path:** full-model free-running text.
   **Control or comparison:** CPU HF at the exact generated TT prefix; frozen
   actual TT hidden state through precision/fidelity candidates and CPU FP32
   norm/head oracle; full original-prompt continuations.
   **Likely subsystem:** accumulated full-model numerical differences and their
   effect on autoregressive branch selection.
   **Investigation performed:** the frozen-hidden oracle also ranks the wrong
   continuation first, refuting the proposed local terminal-kernel explanation.
   BFP4/BFP8 original-prompt candidates retain the visible error. The selected
   BF16/HiFi4 head changes an earlier branch and correctly labels Bonjour as
   formal/greeting and Salut as informal in the final shared suite. It does not
   claim to repair the frozen-prefix local ranking. Selected N32768/K1/readers2
   is measured at 1.404 ms, versus 1.454 ms for N16384/K4/readers2 and 1.859 ms
   for N8192/K4/reader1; N32768/K2 and K4 have retained concrete L1 failures.
   **Resolution:** fixed for the observed original-prompt regression; causal
   interpretation controlled.

3. **Observed anomaly:** duplicate B32 inputs produced unequal full-vocabulary
   logits, eventually changing a greedy token in slot 6.
   **Evidence:** `AUTOFIX_norm_row_order.md`, boundary captures,
   `norm_rows_controls_v3.json`, `norm_rows_canonical.json`,
   `norm_rows_canonical_b1_watcher.json`, generic norm regression logs and
   `full_batch32_final_v2.json`.
   **Affected path:** width-sharded Q/K RMSNorm.
   **Control or comparison:** persistent trace boundary captures, frozen real
   Q/K inputs repeated in every row, CPU norm oracle, and original B1 anchors.
   **Likely subsystem:** worker-dependent cyclic order of BF16 partial sums.
   **Investigation performed:** differences first appear at Q/K norm, before
   RoPE/cache writes. The receiver retains cyclic NoC issue order but stores
   partials at canonical peer offsets. This changes no selected dtype, fidelity,
   shard layout or communication count. Duplicate rows and original B1 anchors
   become bitwise exact on all ranks; generic ROW/COL tests pass. Final all-layer
   B32 duplicate/repeated/permuted logits and tokens are exact with watcher and
   allocation tracking. Isolated norm timing differs by about 0.09%.
   **Resolution:** fixed.

4. **Observed anomaly:** inactive recurrent/conv or sampler state changed, and
   large UINT32 values lost high bits under some WHERE predicate dtypes.
   **Evidence:** `AUTOFIX_batch32.md`, `trace_b32_masks_fixed.json`,
   `scheduler_sampling_final_v3.json`, state-merge source.
   **Affected path:** fixed-slot state preservation.
   **Control or comparison:** frozen state selections and high-valued integer
   payload controls, followed by mixed/inactive/joining requests.
   **Likely subsystem:** WHERE LLK selection from predicate representation.
   **Investigation performed:** predicates match BF16 conv and FP32 recurrent
   representations; INT32 predicates select UINT32 tokens/seeds exactly. Stored
   payload precision is unchanged. Untouched rows remain exact.
   **Resolution:** fixed.

5. **Observed anomaly:** worker watcher aborted on a 36864-byte NoC transfer
   issued through a 16384-byte single-burst specialization.
   **Evidence:** `AUTOFIX_watcher.md`, `AUTOTRIAGE_watcher.md`, native
   split-bank reader diff and terminal B1/B32 watcher controls.
   **Affected path:** DRAM-sharded terminal matmul reader.
   **Control or comparison:** original non-watcher output hashes, exact
   before/after geometry and watcher execution.
   **Likely subsystem:** wrong specialized NoC read contract.
   **Investigation performed:** use the existing arbitrary-length packetizing
   NoC read overload while retaining transaction IDs and layout. Fixed outputs
   match controls; final selected-policy gates exercise the repaired reader.
   **Resolution:** fixed.

6. **Observed anomaly:** full-vocabulary head allocation exceeded L1; later a
   single-part concat/deallocation caused a segfault.
   **Evidence:** terminal probe logs, terminal implementation, geometry controls
   and stage work log.
   **Affected path:** full-model LM head construction/output ownership.
   **Control or comparison:** real terminal weights, chunked output and
   single-part/multipart candidates.
   **Likely subsystem:** circular-buffer capacity and alias ownership.
   **Investigation performed:** bounded vocabulary chunks fit legal storage;
   a single output part is returned directly rather than deallocating its alias.
   Final two-part head passes eager/traced full-model execution and quality gates.
   **Resolution:** fixed.

7. **Observed anomaly:** native full-stack prefill exceeded L1 despite
   decoder-only capacity evidence.
   **Evidence:** `AUTOFIX_context_l1.md`, `context_l1_probe_v2.json`,
   `native_context_final_v3.json`, `../context_contract.json`.
   **Affected path:** large GDN output matmul with 24 resident recurrent states.
   **Control or comparison:** real projection inputs with preserved K16
   accumulation, all-layer allocator measurements and boundary execution.
   **Likely subsystem:** aggregate persistent-state plus temporary output storage.
   **Investigation performed:** a model-only large-prefill output-block cap of
   six tiles leaves prior decoder defaults unchanged and preserves exact
   projection outputs. All 32 layers execute lengths 262143 and 262144; decode
   at position 262143 advances to 262144. Final DRAM is 5,445,806,592 bytes/rank,
   with trace storage separately accounted.
   **Resolution:** fixed without capability reduction.

8. **Observed anomaly:** allocator warns that allocations after trace capture
   could be corrupted on replay.
   **Evidence:** final native/accuracy/performance logs, `runtime_audit.md`,
   generator prefill/recapture code, final B32 and scheduler tracked runs.
   **Affected path:** scheduling-boundary prefill and temporary state merges.
   **Control or comparison:** temporary lifetimes, recapture after new compiled
   programs, tracked cache/scheduler/B32 execution and native boundary outputs.
   **Likely subsystem:** generic active-trace allocation warning.
   **Investigation performed:** buffers that overlap trace temporaries are
   released before replay; stable persistent inputs survive capture; new program
   cache entries cause recapture. No concrete unclosed allocation hazard appears
   in the inspected source or retained tracked runs.
   **Resolution:** controlled.

9. **Observed anomaly:** decode profiler collection closed devices normally,
   then report generation asserted missing device timing for captured trace 0.
   **Evidence:** `AUTOFIX_tracy_unreplayed.md`,
   `tracy_decode_coverage_final.json`, original raw host/device logs,
   Tracy parser diff and regression tests.
   **Affected path:** offline profiler metadata enrichment.
   **Control or comparison:** BEGIN/END/RELEASE without REPLAY for initial
   traces 0/1; actual replay records for traces 2/3.
   **Likely subsystem:** treating unused trace definitions as executions.
   **Investigation performed:** skip enrichment only when device rows are absent
   and available host metadata shows no replay. Missing executed, nontrace or
   unknown-metadata data still raises; existing measurements remain. CPU tests
   report 10 passed and one unchanged pre-existing skip. Independent raw/report
   comparison verifies all 5772 decode and 3920 prefill execution identities and
   firmware/kernel durations/start/end cycles exactly. Measured windows contain
   2416/564 timed rows. Required reports retain advice and both signposts.
   **Resolution:** fixed; original capture recovered without recollection.

10. **Observed anomaly:** terminal report displays about 118% modeled FLOPs;
    some smaller preserved decoder matmuls receive SLOW heuristic labels.
    **Evidence:** reduced per-rank reports,
    `tracy/head_roofline_classification.json`, installed report source,
    prior optimized multichip geometry/fidelity rejection ledger.
    **Affected path:** performance interpretation.
    **Control or comparison:** actual 16 head compute workers versus the
    installed report's eight-worker model; measured lower-movement/geometry
    families from the completed decoder stage.
    **Likely subsystem:** reporting roofline assumptions and generic advice.
    **Investigation performed:** head utilization is about 59% under the
    16-worker model and about 80% of modeled DRAM bandwidth; raw timings remain
    unchanged. Smaller decoder rows retain measured prior policy decisions.
    No false physical utilization or unmeasured speedup is claimed.
    **Resolution:** controlled.

11. **Observed anomaly:** short reasoning windows, cached/full-prefill HF
    rank-one disagreement, and periodic synthetic-workload outputs.
    **Evidence:** all twelve final HF/TT completion files, qualitative metadata,
    AIME reference controls, native/performance/batch input configuration.
    **Affected path:** interpretation of quality and workload evidence.
    **Control or comparison:** exact HF chat template and shared 128-token
    budget; HF cached/full-prefill comparison; explicit repeated-token inputs.
    **Likely subsystem:** generation budget, floating-point reassociation and
    deliberately synthetic structural/performance workloads.
    **Investigation performed:** all actual final completions were read.
    Coherent task-related reasoning is retained, with no unexplained language
    drift or degeneration. HF's one rank-one discrepancy is rank two, giving
    99/100/100%; synthetic repeated inputs are not language-quality evidence.
    **Resolution:** controlled within the documented assessment window.

12. **Observed anomaly:** miscellaneous setup/reporting warnings and artifact
    formatting changes.
    **Evidence:** final logs, work log, raw token/metadata archives and final
    qualitative identity checks.
    **Affected path:** environment metadata, optional profiler viewer export,
    reference setup and evidence preservation.
    **Control or comparison:** four-chip physical topology; retained custom
    profiler output; exact raw/report coverage; final token decoding and original
    French metadata hash.
    **Likely subsystem:** motherboard tray-ID lookup, optional CPU HF kernels,
    pandas mixed columns, default viewer-export path and whitespace/EOF hooks.
    **Investigation performed:** bus-ID fallback affects tray metadata, not
    model execution. Viewer-copy and mixed-column warnings leave captures and
    required reports intact. CPU-only HF optional fast-kernel warnings do not
    indicate TT host fallback. The initial Torch cache setup failure is replaced
    by a successful documented persistent-cache run. Formatting-altered raw
    newlines/JSON serialization were restored from retained IDs/metadata; the
    qualitative reviewer verified all 12 final completion bytes and six prompt,
    template and HF-reuse identities. No generated tokens changed.
    **Resolution:** controlled.

## Scope Inspected

- **Goal/skill paths:** the supplied original full-model contract, applicable
  root/autoport AGENTS instructions, and `.agents/skills/{stage-review,
  full-model,tt-device-usage,tt-enable-tracing,qualitative-check,multichip,
  optimize}/SKILL.md`.
- **Code paths:** complete `tt/model.py` and `tt/generator.py`; stage diff and
  relevant state/prefill/decode paths in `tt/optimized_decoder.py` plus existing
  decoder/config/RoPE implementation; both common sampler contracts; restored
  `models/common/readiness_check/` contract, runners and tests; native
  `reader_bmm_tile_layout_in1_sender_dram_sharded.cpp` and
  `reader_mcast_receiver_unary_sharded_ln.cpp` with relevant NoC/norm helpers;
  generic RMSNorm regression; `tools/tracy/process_ops_logs.py` and its tests;
  stage probes, source/provenance recorder, reduced profiler and report renderer.
- **Artifact paths:** final README/work log/runtime audit/context contract;
  pinned HF reference metadata and loading controls; final prefill, teacher,
  performance128/2048, native context, B32 and scheduler JSON/log/provenance/
  source archives; all twelve actual final qualitative completions and earlier
  French controls; semantic greedy sampler comparison; repair reports and
  frozen controls; prior optimized multichip rejection evidence; reduced raw
  profiler CSVs, compressed operations, per-rank reports and accounting.
- **Artifact integrity:** all 773 entries in `artifact_manifest.json` were
  independently checked for file existence, exact byte size and SHA256, with
  zero errors. All 578 entries labeled `repository_checkpoint` are in the Git
  index; 195 entries are explicitly retained locally. The manifest hash at
  review is `94e13b1765f7c49c496cf3e651a700550558fdcbff4d38f15f9641e51ae21066`.
  It excludes itself and mutable editorial/status documents. Final source and
  artifact pre-commit logs pass; original generated-text and CSV whitespace
  remains intentionally unnormalized.
- **Commands run by this reviewer:** read-only `rg`, `sed`, `cat`,
  `git diff`, `git status`, and small standard-library Python analyses of
  JSON/gzip/CSV/text/hash artifacts. These analyses independently matched final
  source hashes, reference/template metadata, report hashes, raw execution
  identities and timing fields. The reviewer imported no TTNN, opened no device,
  launched no server, reset no hardware and ran no long tests. Test/device
  results above are inspected retained stage evidence, not reviewer executions.
- **Accepted final gates:** prefill top1/top5/top100 = 96/100/100%; traced
  teacher = 94/100/100% for 100 predictions; native262144; exact all-layer B32
  duplicate/permuted logits; 46 host tests; reviewed shared six-prompt HF/TT
  suite. Warm B1 prompt128/generate128 TTFT is 47.462 ms and traced token-out
  decode 81.558 tokens/s/user. The 127-step loop has 127 model/sampler replays,
  zero host feedback/position/RoPE/page refreshes and zero global syncs.
  Separate logits-only is 85.609 tokens/s. Reduced decode sampling is about
  0.508 ms kernels plus 0.071 ms gaps per rank and does not dominate the full
  token-out path. Device timings are not summed across ranks.

## Residual Risk

This verdict covers the delivered full-model stage and the evidence limits
above. It is not a serving/release certification, broad evaluation of task
accuracy, complete native-length quality study or proof of all normalization
geometries. Large frozen tensors and original profiler captures remain at the
manifest's persistent local paths; the checkpoint carries compact evidence. Future deployment integration must own its scheduler/logprob/EOS
resource contracts and complete the unavailable CI-image build. Local
checkpoint commits should contain only stage-owned changes and follow this
review; no push is authorized.
