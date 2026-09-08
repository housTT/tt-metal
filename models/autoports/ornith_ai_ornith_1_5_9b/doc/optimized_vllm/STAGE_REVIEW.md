# Stage Review

Verdict: clean-pass

Independent review of stage 10, optimized-vLLM, for
`ornith-ai/Ornith-1.5-9B`, on 2026-09-08 UTC. Reviewed the live
`hous/ornith-1.5-9b` checkout at starting HEAD `539734fbb6` with the stage's
uncommitted changes. The two user-owned AGENTS.md changes are excluded.
The vLLM checkout remains clean at the recorded predecessor revision.

## Required Work

- None for this optimization stage. Local checkpoint commits and recording their
  SHAs follow this review under the stage-review workflow; this verdict does not
  claim those commits already exist.

The final full sampling profile completed with **72 passed, 1 skipped**, and
the final server has stopped. Source inspection and independent artifact
checks support the claimed optimization without a serving correctness or
capability regression:

- Primary **128 input / 128 output / 1 request**, concurrency 1 and
  `max_num_seqs=1`: the median-TTFT repeat changes from **49.956 ms / 87.884
  decode tokens/s/user** to **35.249 ms / 87.700 tokens/s/user**. I checked all
  three measured repeats on each side against their raw results and archive
  hashes, including TTFT, TPOT, ITL, completed requests, token counts and
  aggregate throughput. The 29.4% TTFT improvement is supported; decode is
  effectively unchanged.
- Secondary **100 input / 100 output / 32 requests**, unrestricted admission
  and `max_num_seqs=32`: every before/after repeat completes 32 requests and
  3,200 output tokens. Headline selection is the median-TTFT repeat on each
  side, with every reported metric taken from that same repeat. This evidence
  is correctly separated from the single-user decode headline.
- Server argv and recorded environments match exactly within each B1/B32
  comparison pair. Final source hashes match the inspected checkout. Actual
  server logs show the real Ornith TT-plugin adapter, `trace_mode=all`,
  `sample_on_device_mode=all`, async scheduling and native context 262144.
- The comparable current standalone **128 input / 128 output / B1** generated
  token run measures **87.959 tokens/s/user**, making serving decode 0.295%
  slower. Its runtime policy equals the selected datatype configuration;
  materialized head weights, LoFi compute and layer 31 exceptions agree.
  Model/decoder math and precision are unchanged by this stage.
- H1's all-32-layer external-cache probe passes 18 exact comparisons of full
  logical logits, recurrent/conv state, next-decode logits/tokens and state.
  The tests cover resident 128/131 shapes, literal first 131 after startup,
  changed tokens/pages, live shape misses, continuation, caller ownership,
  unreferenced physical-page sentinels and teardown.
- H2's B1 and final B32 probes pass eight exact sampler cases each with real
  reduced-model prefill plus production-shape sampler controls. They cover
  transient and canonical logits, forced recapture, greedy/seeded/penalty
  transitions, partial/full admission, exact UINT32 seeds, all-rank history
  preservation and stable persistent buffers. The final full-32-lane shortcut
  follows logits staging and does not alter partial admission.
- Decode replays persistent model and canonical sampler traces with
  `blocking=False`; device token feedback and position/RoPE advance are
  retained. The plugin queues a minimal token-shard copy before buffer reuse,
  then waits/formats beyond the async boundary. The focused deferred-stale
  probe matches synchronous results, performs no token/position/RoPE/table
  refresh in its unchanged two-step window, and performs exactly one page
  refresh for each changed-table/remap window. It checks physical KV writes.
  Real performance-server counters establish that the plugin exercises the
  async split and records zero host decodes and explicit generator syncs.
- Public serving requests at logical lengths **131 and 65** retain exact
  predecessor token prefixes and usage; B32 also passes concurrent A/B/A.
  The full-32-layer native external-pool capacity probe completes logical
  **262143 and 262144** prefill, advances the last valid decode position from
  262143 to 262144 on all ranks, preserves cache/sampler buffer identities,
  and reaches zero TRACE allocation at teardown. The context contract includes
  the new persistent allocation and retains 262144.
- I read all twelve current shared-suite texts, all six HF/selected-TT short
  controls, and the extended HF/standalone haiku controls. All six current
  greedy texts independently compare byte-for-byte equal to the prior serving
  snapshot. Prompt metadata records the original pinned chat template and
  matching rendered prompts/token IDs. No new mechanical repetition, wrong
  language, gibberish or cross-request contamination is visible.
- Logged verification contains 87 passing host regressions, passing applicable
  pre-commit hooks, passing scope-specific degeneracy and stage checks,
  stopped-server receipts, and the final four-chip health result. Changes are
  Python/docs only, so no C++ build is required. I did not rerun these tests or
  access hardware during review.

## Other Concerns

- The selected full-model path still makes the controlled haiku syllable error;
  it differs from the pinned HF final answer. This is not newly introduced by
  serving optimization, and this pass is not a general task-accuracy verdict.
- Several 256-token shared-suite responses end during reasoning or before
  completing code. Their matching predecessor controls establish regression
  stability, not completed-answer accuracy.
- H2 has no independently demonstrated serving speedup beyond H1. The final
  report correctly uses the reproduced final number and identifies H2's value
  as traced first-token sampling and persistent state preservation.
- TP1/TP2, prefix caching, million-token YaRN and 32 simultaneous native-length
  requests remain broader release work. The current TP4 optimization neither
  validates those profiles nor lowers an already validated capability.

## Hard-Check Gaps

- The canonical full-profile skip is `test_chat_logprobs_all_vocab`. Its source
  skips specifically when the API rejects the requested logprobs count as above
  the allowed maximum; the final server log records the corresponding chat
  HTTP 400. Ordinary logprob and host-only sampling compatibility tests pass.
  No all-vocabulary logprob capability is claimed here.
- The full sampling server explicitly enables host compatibility for tests that
  require it. Both before/after performance pairs disable that mode. A passing
  mixed canonical suite is not evidence that every optional sampling control
  runs on device.
- The context check's 2048 advisories refer to historical CPU registry probes
  and nested prior logs. Current server manifests and native capacity evidence
  show 262144; the checker ends with the full HF context contract passing.
- Device profiler timing is intentionally absent under the original goal and
  vLLM-stage skills. No new profiler artifact is needed for this verdict.

## Anomaly Ledger

- Observed anomaly: allocator.cpp warns that allocations after trace creation
  may be corrupted on replay.
  Evidence: before/after B1/B32 server logs and the final compatibility log.
  Affected path: eager shape-miss prefill, sampler configuration and recapture.
  Control or comparison: the warning exists in the untouched baseline.
  Likely subsystem: conservative trace-allocation warning.
  Investigation performed: inspected H1 all-layer, H2 B1/final-B32, native
  capacity and deferred-input probes with native allocation tracking including
  program caches; verified transient retirement/persistent allocation ordering
  in source and zero TRACE allocation where measured.
  Resolution: controlled warning; no observed corruption is being waived.

- Observed anomaly: full Ethernet Watcher startup exceeded firmware capacity,
  28720 bytes versus a 26624-byte buffer.
  Evidence: `before_async_contract.log/json` and recovery work-log receipts.
  Affected path: mesh opening before model execution.
  Control or comparison: reset/list/mesh open-close recovered; worker Watcher 10
  probes pass with Ethernet instrumentation disabled.
  Likely subsystem: Watcher firmware size, explicitly covered by Optimize's
  documented scoped retry.
  Investigation performed: reviewed the failure, recovery and resumed probes.
  Resolution: controlled instrumentation limit; worker asserts remain enabled.

- Observed anomaly: reduced smoke min_p request killed its no-host server.
  Evidence: `h2_reduced_sampling.log`, `h2_reduced_b4.server.log` and work log.
  Affected path: a host-only test attached to the performance configuration.
  Control or comparison: corrected compatibility smoke and final full profile
  pass; benchmark servers still reject host fallback.
  Likely subsystem: invocation/configuration mismatch.
  Investigation performed: checked explicit compatibility guard and final
  full-suite disposition.
  Resolution: fixed invocation; no device min_p claim introduced.

- Observed anomaly: first native-capacity probe inspected buffer_address on a
  HOST tensor before the long-context window.
  Evidence: `after_native_capacity.json/log` and corrected v2 artifacts.
  Affected path: probe metadata traversal.
  Control or comparison: corrected walker excludes HOST storage; v2 completes
  both native windows under unchanged runtime code.
  Likely subsystem: diagnostic tensor classification.
  Investigation performed: inspected probe source and successful native results.
  Resolution: fixed probe.

- Observed anomaly: haiku counting errors and unfinished responses at 256 tokens.
  Evidence: current twelve texts and extended predecessor HF/TT haiku outputs.
  Affected path: selected model's generated text.
  Control or comparison: all six greedy texts equal predecessor serving;
  predecessor standalone and API haiku match all 390 generated token IDs,
  while HF finishes a valid 5/7/5 haiku.
  Likely subsystem: existing selected-model behavior; no specific quantized
  group is proven responsible by these controls.
  Investigation performed: direct text reading and independent greedy equality
  checks, with source verification that model math/precision are unchanged.
  Resolution: controlled serving regression result; accuracy limitation retained.

- Observed anomaly: nanobind reports leaked 984 types and 4503 functions at
  interpreter shutdown.
  Evidence: before_b1/before_b32/after_b1/after_b32/final_b32_compat runner logs.
  Affected path: Python binding teardown after worker/server completion.
  Control or comparison: counts are identical in all five logs; runners exit 0,
  meshes close, native TRACE allocation reaches zero and final OS owner scan
  is empty.
  Likely subsystem: inherited binding registration/reference teardown.
  Investigation performed: compared exact warning counts and cleanup receipts.
  Resolution: controlled unchanged shutdown warning.

## Scope Inspected

- Goal/skill paths: original
  `/home/hous/dev/ornith-1.5-9b/state/multigoal/10-10-optimized-vllm.prompt.txt`;
  repository/model AGENTS.md; `.agents/skills/{stage-review,vllm-integration,
  optimize,tt-enable-tracing,tt-device-usage,qualitative-check}/SKILL.md`;
  relevant tracing/data-movement advice in `tech_reports/LLMs/llms.md`.
- Artifact paths: current README, work log, checklist, context contract,
  performance summary/comparisons, before/H1/after raw repeats and manifests,
  source snapshot, execution counters/server logs, all H1/H2/async/native
  proof artifacts and provenance, nonaligned requests, current qualitative
  artifacts and predecessor controls, final full-model token-output run,
  final sampling/host/hook/stage logs, cleanup and health receipts.
- Archive verification: independently checked all four core archive parts,
  combined core hash and all **150 member hashes**; checked the final archive
  hash and all **15 member hashes**. Plain original artifacts were available
  during review. Fresh checkouts must use the documented restore commands.
- Code paths: `tt/generator.py`, `tt/generator_vllm.py`, relevant model and
  common sampling contracts; changed host tests; external-prefill, sampler,
  native-capacity and adapter device probes; stage server/benchmark wrappers;
  pinned plugin registration, async submission/finalization, page tables and
  canonical all-vocabulary logprob skip logic.
- Commands run: read-only `git status`, `git diff`, `rg`, `cat`, `sed`, `tail`
  and small Python standard-library artifact scripts for hashes, archive
  members, metric rederivation, median-repeat selection, source provenance and
  output equality. No TTNN import, server start, device listing/reset, hardware
  test or profiler was performed by this reviewer. Only this report was written.

## Residual Risk

- Three warmed single-request repeats characterize this fixed workload; their
  request P50/P99 values coincide and do not estimate production tail latency.
- Native-window repeated-token probes establish allocation and position/cache
  capability, not long-context semantic quality. Later release evaluations
  remain required.
- The final default behavior and evidence are tied to the recorded pinned
  environment, TP4/P300c hardware and selected precision. This clean-pass is
  limited to stage 10's optimization contract.
