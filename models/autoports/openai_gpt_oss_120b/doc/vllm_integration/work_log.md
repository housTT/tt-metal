# GPT-OSS 120B vLLM integration work log

## Scope

Continued from the completed datatype sweep and its selected `ds00_baseline`
policy. All checkouts, Python packages, weights, tensor caches, logs, and temp
files remained below `/home/ttuser/dev/gpt-oss-20b`. Device work used the
official `vllm-project/vllm` checkout plus the standalone
`tenstorrent/vllm-tt-plugin`; the vLLM checkout was never modified.

## Implementation

- Added `tt/generator_vllm.py` as a thin scheduler/cache adapter over the
  existing full-model generator and canonical split sampler/token-out path.
- Bound vLLM-owned hybrid per-layer KV buffers directly; no standalone serving
  cache is created.
- Registered the dedicated `TTGptOss120BForCausalLM` alias in
  `platform.py::register_tt_models()` while retaining the existing 20B route.
- Declared async decode support and retained B1/B32 decode traces. Scheduler
  removal-only resets keep a surviving request in its live trace bucket; this
  fixes order-dependent seeded-penalty output without introducing host token
  feedback.
- Added/updated adapter and plugin tests for cache ownership, async output,
  stale token/position/page-table state, mixed seeds, registration, and
  full-capacity structured output.
- Updated the shared qualitative runner to request GPT-OSS low reasoning effort
  so a fixed 256-token qualitative budget reaches final content. The shared bad
  words test checks both GPT-OSS reasoning and final text. These are response
  interpretation/test-workload changes, not sampling-path changes.

## Debug and AutoFix history

Earlier host-compatibility tests exposed trace lifetime and page-table routing
issues. The retained lifecycle releases device traces before explicit eager
host-logits compatibility work, then recaptures the prepared device traces and
forces an authoritative reset before returning to on-device sampling. Prefill
tables are request-sized; persistent decode tables retain fixed trace width.

The last full-profile attempt initially had two harness failures: valid
GPT-OSS responses exhausted their token budgets in the parsed `reasoning`
field, leaving `content=null` in bad-words and structured-output tests. A
fresh-context AutoFix review confirmed that both routes were host-sampled and
that no TT token was missing. The bad-words check now inspects reasoning plus
content, and the structured workload uses explicit low reasoning effort with a
128-token budget. The full rerun passed.

Mixed seeded batches were separately reproduced ten consecutive times on the
full 36-layer model. The root cause was an unsafe B32->B1 shape change after a
removal-only scheduler reset while another request survived. Making that live
bucket sticky fixed all ten repetitions; the complete full sampling profile
then passed 73/73 runnable cases with one expected skip.

## Final checks and results

- Import origins: TTNN, vLLM, plugin, and adapter all below the workspace;
  `ORIGIN_CHECK=PASS` before each server start.
- Adapter/full-model host coverage: 27 adapter tests passed; focused common
  allocation coverage 2 passed.
- Focused adapter/readiness host coverage: 32 passed after final formatting.
- Post-format sampling/trace allocation coverage:
  85 passed, 8 skipped, 4 warnings. The old-path log-probs test is now skipped
  on this 4-device P150x4 mesh because that runtime path only supports 8- or
  32-device sharded meshes.
- Plugin host suite excluding live-server `tests/tt`: 331 passed.
- Black checks passed for touched tt-metal and plugin Python files after
  mechanical formatting of touched files.
- `git diff --check` passed in both tt-metal and vLLM TT plugin.
- Final multigoal stage check passed; log:
  `readiness_vllm/stage_09_check.log`.
- Targeted plugin runner/state coverage: 25 passed.
- Full 36-layer mixed-parameter stress: 10/10 passed in `220.61 s`.
- Full sampling profile: 73 passed, 1 skipped in `1502.77 s`.
- Non-aligned request: HTTP 200, 65 prompt tokens, eight output tokens.
- Qualitative: six prompts x greedy/sampled; all 12 manually reviewed and
  passed coherence, topic, repetition, gibberish, language, and contamination.
- Degenerate-output checker: exit 0, no findings.
- Repository build gate: the wrapper reported missing
  `GARAGE_S3_ACCESS_KEY` / `GARAGE_S3_SECRET_KEY`, so a full cold compile was
  not attempted. The stale host-path `build_Release` cache was moved aside
  temporarily so `.github/scripts/copilot-build.sh --configure-only` could
  complete successfully with Clang 20 and warnings-as-errors; log:
  `readiness_vllm/configure_only_build.log`. The pre-existing compiled
  `build_Release` was then restored for local runtime imports, and the
  configure-only directory was preserved at
  `/home/ttuser/dev/gpt-oss-20b/.runtime/preserved_build_dirs/build_Release_configure_only_20260831T2156Z`.
- Primary benchmark, 1 x (128->128), concurrency 1, temperature 0:
  TTFT P50/P99 `514.11/514.11 ms`; TPOT mean/P50/P99
  `21.44/21.44/21.44 ms`; ITL P50/P99 `20.72/34.22 ms`; output throughput
  `39.54 tok/s`; TPOT-derived decode `46.64 t/s/u`.
- CI burst, 32 x (100->100), unbounded admission, temperature 0:
  32/32 complete; TTFT P50/P99 `15.269/15.270 s`; TPOT mean/P50/P99
  `589.40/589.39/589.59 ms`; ITL P50/P99 `589.39/595.41 ms`; aggregate output
  throughput `43.47 tok/s`. Its TPOT-derived value is secondary only.

The exact server and runner commands are in `README.md`; the raw commands are
also embedded in the benchmark JSON. The primary benchmark was measured first
on a clean server. The final performance request used async traced on-device
sampling with no host argmax, generic top-k fallback, full-logits readback, or
Python token-feedback loop. The datatype-sweep `61.04 t/s/u` teacher-forcing
number is retained only as a lower-bound reference to a different,
caller-authoritative workload.

## Runtime and cleanup audit

The sampling/qualitative and clean-benchmark servers both shut down after all
requests completed. vLLM logs an `EngineDeadError` in the API output task only
after its explicit zero-timeout abort-shutdown markers; it is not a request
failure. Post-shutdown process searches found no API server, `vllm serve`, or
EngineCore process. `tt-smi -s` reported all four p300c boards healthy,
`dram_status=true`, zero corrected/uncorrected GDDR errors, and only the probe
itself as a temporary holder. No reset was needed. Full details are in
`readiness_vllm/runtime_cleanup_audit.md`.

## Review and commits

Fresh-context stage review first returned `more-work-needed` because the
context contract retained the older max-seqs=8 metrics. After updating it to
the final max-seqs=32/B1+B32 sampling and benchmark evidence, rereview returned
`clean-pass` with no remaining blocker.

- vLLM TT plugin implementation: `414b870`
- tt-metal implementation/evidence: recorded by the following provenance-only
  commit after this implementation commit is created.

Nothing was pushed.
