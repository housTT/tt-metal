# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The KDA sigmoid-gated RMSNorm finding is closed. The reversible patch hash is `c61dcb930126d53e365f1f0a4fb7713ce03cea5c47f71e5c2931f5035c07fb61`; `git apply --check` succeeds against delivered source `3f033b75...`, and independently applying it as a stream reconstructs candidate SHA-256 `e2918ebaccc6bb331d3703352cb1de6fef99b06e2e139e1a16bc4b687c0028cc`. The patch changes only prefill to request `[B*H,T,V]`, invoke `sigmoid_gated_rms_norm`, and keep decode on the primitive epilogue. The primary real-weight JUnit proves L0 and L3 pass while the normal caller-retained L1 path fails traced decode at 0.86216938; a focused first-eager artifact fails at 0.86030912. The saved recurrent, convolution, and PLE states compare bit-identically. Source inspection independently confirms bounded KDA input/gate/output pages 0..767 and weight pages 0..3 for `B=1,H=48,T=128,V=128`, with only the newly allocated output writable. L1 output placement, deferred lifetimes, and program-cache clearing remain failing. This replaces the invalid old “prefill state perturbation” explanation with an evidence-backed supported-lifetime/allocator-sensitive composition failure. Because the candidate cannot satisfy legal caller lifetime semantics and the scoped repair variants did not repair it, retaining the correct primitive epilogue is an earned rejection despite the candidate's measured 0.60/0.65 ms GDN-prefill gain.
- The final bounded candidate-provenance finding is closed. Both journal JSONLs parse record-by-record, and every retained record matches an original session-journal line byte-for-byte: 32/32 for the KDA causal-convolution sweep and 14/14 for the indexer candidate. The KDA journal contains the initial/adaptation patches, 320/640/1280 mutations, raw commands, complete outputs, exit statuses, PCC, timing, and the later layer-1 confirmation; its immutable base blob exists. The indexer journal contains both forward patches, raw real-weight PCC output, all three timing samples, the exact reverse patch, and exit statuses. The standalone indexer unified patch and its base blob are present. All journal/patch hashes match `provenance_manifest.md`; promoted paths are additionally covered by the frozen final-source gates.
- The equal-sampling host-performance finding is closed. The functional baseline now has an unedited seven-run transcript at ten decode replays, exit 0 with 21 passes, source/perf-test hashes, and manifest hash `388cd823...`. The functional and fused runners differ only by decoder type, environment/signpost names, and fused type assertion; sequence, inputs, cache geometry, warmup, trace capture/replay, replay count, and synchronization placement match. Independent median recomputation gives functional to fused prefill/decode milliseconds: L0 `24.047593/5.730349 -> 17.015016/3.883350`, L1 `27.598879/6.701143 -> 20.517265/4.321484`, and L3 `408.372929/11.341126 -> 43.610069/5.539352`. README, work log, performance summary, final timing log, and provenance manifest now consistently describe seven samples. The distinct correct rejected indexer candidate is slower in every retained decode sample (`5.587990` to `5.590499` ms) than every final decode sample (`5.535979` to `5.540231` ms).
- Frozen implementation/test hashes remain exactly `3f033b75dd81ee615bc8309e8e24dbc7aa43f3a6699dfc3bba596879fb7b9d3b`, `a06d7e0adfb2323021e7ece8d36f4847183d9ba485816a5a936807712b7c8b24`, and `f6d263a3bd1b888da6bd551b907e631634bb6f72cbe11ff5933f084ab32ae884`. No implementation or test edit was made during review.
- The existing final gates remain source-current and internally consistent: 35 non-long correctness/trace-allocation tests, seven exact/non-aligned advertised-context tests, nine repeated trace/determinism tests, and 35 watcher-enabled tests all have zero failures/errors. Real-weight final PCC remains L0 `0.99842572/0.99702853`, L1 `0.99911219/0.99988902`, and L3 `0.99668270/0.99977344`, all above 0.995. The context contract remains 262144 tokens, page size 64, chunk size 128, and tested batch 32, including 262143 non-aligned lengths and traced maximum-position decode.
- The watcher v5 log hash matches the manifest, contains no audited error signature, and ends in dump completion plus device-0 detach. Stress and post-run device-health artifacts remain clean. All six frozen-source Tracy raw/stable and filtered hashes still match the manifest, with the previously reconciled device sums and op counts.
- Source inspection still shows the required final topology: one group-major A-sparse/B-dense expert-down call; routed/shared gating before down; selected-softmax routing; KDA causal-convolution prefill; split PLE/GDN decode state; dedicated HF rotary, decode SDPA, and fused paged K/V update; legal retained V sharding; static block RoPE and persistent compressed index keys. No runtime host-conversion or concatenate-heads fallback appears in `FusedDecoder`.

## Hard-Check Gaps

- The lower-level cause of the rejected KDA epilogue candidate's allocator/address sensitivity is not repaired in TTNN. It is not a final-stage correctness gap: the delivered graph does not use that op, passes the standard retained-input path, and the candidate now has an understood, source-linked, reproducible correctness blocker after scoped AutoFix variants.
- The batch-32 gate proves execution/output shape for all representative layers; exact per-user page/cache-slot semantics remain numerically checked at batch two. This is unchanged inherited functional coverage and does not contradict the advertised batch-32 execution contract.
- The watcher device log is under a gitignored `generated` directory. It exists and hashes correctly, but the post-review checkpoint must explicitly force-add it or move it to a trackable evidence path.

## Anomaly Ledger

- Observed anomaly: A mathematically exact prefill-only KDA epilogue changes subsequent L1 decode based on whether valid caller-owned decode input objects remain live.
  Evidence: `gdn_kda_sigmoid_gated_rms_norm_candidate.xml` fails the standard traced path at 0.86216938; `gdn_kda_sigmoid_gated_rms_norm_eager_live_output_same_inputs.xml` fails first eager decode at 0.86030912; throwaway-input controls pass; state equality and variant XMLs are hash-linked in `provenance_manifest.md`.
  Affected path: Rejected KDA prefill epilogue candidate only.
  Control or comparison: Frozen primitive final graph passes L1 traced decode at 0.99988902. Saved states are bit-identical, kernel page bounds are valid, and L1-output/deferred-lifetime/cache-clear variants still fail.
  Likely subsystem: TTNN allocation/address-sensitive composition behavior, not semantic prefill-state mutation.
  Investigation performed: Reconstructed the exact patch/source hash; inspected primary eager/traced/state/variant artifacts; audited program factory, reader, writer, and unit-test immutability/cache contracts; independently checked page bounds.
  Resolution: controlled by rejecting the candidate and retaining the correct primitive path.

- Observed anomaly: The three final bounded candidates previously had prose-only summaries.
  Evidence: New KDA/indexer journal JSONLs, KDA/indexer patches, candidate XMLs/raw timing transcript, and expanded candidate index/manifest.
  Affected path: Graph-exhaustion and best-correct-candidate provenance.
  Control or comparison: Exact original session journal; all 46 selected JSONL records match it byte-for-byte. Final promoted graph remains validated by current final gates.
  Likely subsystem: Experiment evidence retention.
  Investigation performed: Validated JSONL syntax, call IDs, commands, patches, outputs, exit statuses, original-line identity, file hashes, base blobs, and the sigmoid patch's reconstructed source hash.
  Resolution: fixed.

- Observed anomaly: The prior README mixed three-run functional and seven-run fused medians while calling both three-run.
  Evidence: New `functional_perf_10replay_count7.raw.log`, its source-linked summary, updated docs, and unchanged seven-sample fused log.
  Affected path: Warmed prefill/traced-decode before/after comparison.
  Control or comparison: Runner diff and independent median recomputation for all six windows.
  Likely subsystem: Performance evidence refresh.
  Investigation performed: Compared runner source, verified transcript/hash/exit, extracted and sorted all 42 functional/fused measurements, and checked all current documentation for stale sample-count wording.
  Resolution: fixed.

- Observed anomaly: Six filtered device-report row sums differ from the report totals by at most 0.035 us.
  Evidence: Prior independent summation of each three-decimal `Device Time` column.
  Affected path: Device-time display only.
  Control or comparison: Raw/filtered hashes and op counts match; differences fit cumulative per-row decimal rounding.
  Likely subsystem: CSV presentation precision.
  Investigation performed: Rechecked unchanged profiler hashes and manifest mapping during this bounded rereview.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied graph-fused-decoder contract; `.agents/skills/stage-review/SKILL.md`; the three findings in the previous `stage_review_clean.md` only. No new candidate family was introduced.
- Artifact paths: frozen source/tests; current README, work log, graph inventory, AUTODEBUG report, candidate index, provenance manifest, performance summary; the three bounded candidate logs; KDA/indexer patches and journal JSONLs; all KDA sigmoid PCC/state/variant/performance XML and raw artifacts; functional/fused seven-run timing evidence; unchanged correctness/context/stress/watcher/health and six Tracy artifacts.
- Code paths: `tt/fused_decoder.py`, both functional/fused perf runners, inherited correctness flow, KDA sigmoid-gated RMSNorm program factory/reader/writer/compute kernels, and its unit-test immutability/cache-rebinding coverage.
- Commands run: read-only `sed`, `rg`, `find`, `sha256sum`, `wc`, `jq`, `grep`, `diff`, `sort`, `awk`, `git status/check-ignore/cat-file/apply --check`, and stream-only patch reconstruction. No pytest, Tracy, watcher, TT device, `tt-smi`, reset, reservation, server, vLLM, or later-stage command was run.

## Residual Risk

- The rejected KDA epilogue exposes a lower-level TTNN lifetime/address defect worth a separate kernel/runtime issue, but it is absent from the delivered decoder and is not required to pass this graph-fusing stage.
- Real-weight PCC remains measured at a short non-aligned prompt plus traced decode; advertised-context runs prove full execution, state/cache capacity, paging, and shape semantics rather than full-length real-weight numerical equivalence.
- Performance uses the stage's zero-input harness, and sparse-matmul profiling does not model real routed-expert utilization. The evidence proves the required harness latency/topology, not workload-distribution performance.
