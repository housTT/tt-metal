# GPT-OSS 120B optimized vLLM work log

## Scope and provenance

This stage started from completed vLLM integration commit
`fc5237d4d252409deadc45c5427da6a7ba8c748d` and the selected datatype policy
`ds00_baseline`. It did not modify either external serving repository:

- official vLLM checkout:
  `/home/ttuser/dev/gpt-oss-20b/vllm`, v0.26.0,
  `568afb3a13806beb53bb2e6bd518269357b237c0`;
- standalone TT plugin:
  `/home/ttuser/dev/gpt-oss-20b/vllm-tt-plugin`,
  `414b870b57844115d37e8b8abf3ef0e8b62765bf`.

All source, Python environments, weights, Hugging Face caches, model caches,
temporary files, logs, and artifacts stayed below
`/home/ttuser/dev/gpt-oss-20b`. Before every hardware server start,
`.agents/scripts/gpt_oss_workspace_env.sh` was sourced and import origins were
asserted below that root for TTNN, `vllm.entrypoints.openai.api_server`,
`vllm_tt_plugin`, and the GPT-OSS adapter.

The context contract stayed unchanged: P150x4, 36 resident layers,
`max_model_len=131072`, `max-num-seqs=32`, page size 64, B1/B32 decode traces,
and valid non-aligned prompt support. P150/P150x2 were not substituted and the
context was not reduced.

## Before measurement

The unchanged integration was rerun with the final workload and configuration
before editing the serving path. Results were copied to
`artifacts/before_reproduced/`.

| Workload | Before result |
| --- | ---: |
| P150x4 128 prompt / 128 output / 1 request, concurrency 1 | TTFT 510.297 ms; TPOT 21.691 ms; ITL P50/P99 21.243/35.636 ms; output 39.200 tok/s; decode 46.103 t/s/u |
| P150x4 100 prompt / 100 output / 32 requests, unbounded admission | 32/32; TTFT P50/P99 15.297/15.299 s; TPOT mean 589.401 ms; ITL P50/P99 589.270/595.592 ms; output 43.450 tok/s |

The exact runner command is reproduced in `README.md` and embedded in both
benchmark JSON files.

## Audit and hypothesis

The real path was confirmed as:

1. vLLM's TT runner calls
   `models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py`.
2. The adapter calls the autoport full-model generator with
   `read_from_device=False`.
3. Model and sampler traces submit via
   `ttnn.execute_trace(..., blocking=False)`.
4. The plugin calls `read_decode_output(..., async_read=True)` and synchronizes
   its returned event during finalization.
5. The shared read path called `.cpu(blocking=False)` on the full mesh token
   tensor.
6. GPT-OSS `process_output_decode()` later selected only the first TP replica
   because `users_row_sharded=False` on P150x4.

Thus every token transferred four replicated token shards and discarded
three. This was the only boundary defect selected for hardware optimization.
The model, LM head, sampler, token feedback, cache, page table, trace inputs,
precision policy, and scheduling configuration were left intact.

## Retained implementation

`tt/generator.py` now:

- recognizes device-token output only when there is one data-parallel result
  and no logprob payload;
- selects one TP replica for non-row-sharded sampling, or one replica per
  distinct mesh row when row-sharded;
- starts `.cpu(blocking=False)` only for those selected device tensors;
- records the mesh event after the selected reads;
- converts the host-resident shards after plugin synchronization;
- falls back to the generic collector for logprobs, logits, multiple DP
  outputs, or an unrecognized structure;
- records minimal-read, transferred-shard, and skipped-replica evidence.

`tt/generator_vllm.py` now implements `release_persistent_capture()`. The
plugin's normal worker shutdown calls it while the mesh is still open. It
writes the final capability/counter snapshot, releases model/sampler traces,
and clears the adapter's generator reference before mesh close. The method is
idempotent.

Stage review then required a production-depth allocation-safety proof because
the earlier integration probe used only two layers. The first full 36-layer
tracker run failed the first B1 replay and identified two live program-cache
buffers:

- `get_block_size()` treated both cache container forms as nested and indexed
  a TT tensor, materializing a `SliceDeviceOperation` device buffer;
- generic warmup used persistent max-width page tables, so it did not compile
  the explicit B1 per-request `[1, 2]` page-table signature used by vLLM's
  `PagedFillCacheDeviceOperation`.

`models/tt_transformers/tt/common.py` now distinguishes Python cache
containers without indexing a device tensor. The vLLM adapter also runs one
exact explicit per-layer B1 paged-fill warmup before decode capture. No
allocation-tracker exclusion or program-cache skip was added. The repeated
36-layer tracker run completed the primary and CI workloads with 226 model and
226 sampler replays and zero unsafe survivors.

Tests cover one-replica-per-row selection, invalid mesh geometry, final
snapshot-before-teardown ordering, and idempotent shutdown. The existing
stale-state tests continue to cover changed token/current-position inputs,
changed and unchanged page tables, sampling reuse, slot remaps, removal-only
resets, and B1/B32 trace buckets.

## Candidate and final measurements

The first isolated clean-server A/B after the minimal read change measured
511.228 ms TTFT, 18.229 ms TPOT, 18.033/30.158 ms ITL P50/P99,
45.285 tok/s output, and 54.859 t/s/u. The 32-request burst remained
43.449 tok/s. This proved the hypothesis and the candidate was retained.

After the allocation-safety remediation, the final clean-server rerun
measured:

| Workload | Final result | Before/after interpretation |
| --- | ---: | --- |
| P150x4 128/128/1, concurrency 1 | TTFT 502.928 ms; TPOT 17.339 ms; ITL P50/P99 16.238/29.191 ms; output 47.316 tok/s; decode 57.674 t/s/u | decode +25.10%; TPOT -20.06%; TTFT flat |
| P150x4 100/100/32, unbounded admission | 32/32; TTFT P50/P99 15.272/15.273 s; TPOT mean 589.409 ms; ITL P50/P99 589.396/594.845 ms; output 43.464 tok/s | capacity unchanged; secondary evidence only |

The final clean capability snapshot reports 226 device-sampled decodes, 226
async reads, 226 model and 226 sampler trace submissions, 224 fixed sampling
replays, two sampling pushes, zero host-sampled decodes, zero host argmax,
zero full-logits readbacks, zero unclassified execute submissions, 226 token
shards transferred, and 678 redundant TP replicas skipped. Two full-input
refreshes initialize the B1 and B32 phases; 221 decode calls reuse the resident
page table.

The comparable optimized full-model prompt-128/output-128 split token-out
result is 15.9035 ms/token or 62.8792 t/s/u. Final vLLM is 91.7% of that rate;
vLLM ITL P50 is only 0.3343 ms higher. A long-lived pre-remediation server after all
sampling and qualitative validation measured 16.927 ms TPOT and 59.077 t/s/u;
it is retained as warmed-after-validation evidence, not substituted for the
clean-server A/B headline.

## Final correctness and serving gates

- Full TT plugin serving sampling profile: 73 passed, 1 expected skip, 2
  warnings, 1506.58 s.
- Focused adapter and full-model host suite: 62 passed, 12 hardware-gated
  skips, 4 warnings.
- Pre-commit over all touched Python sources/tests: passed.
- No C++ or CMake file changed, so the repository build table requires no
  compile; Python formatting/lint and live serving tests are the applicable
  checks.
- Six qualitative prompts × greedy/sampled: 12/12 manually reviewed as
  coherent, relevant, non-degenerate, correctly languaged, and uncontaminated.
  The regenerated exact-output comparison classifies both thermodynamics
  responses as incomplete at the 256-token cap because they omit the Third
  Law; it no longer overclaims completeness.
- Scoped degeneracy checker: exit 0, no findings.
- Runner stage-10 check: passed; scoped degeneracy remained clean and the
  optimized-vLLM context contract reported target=supported=131072.
- Primary benchmark: 1/1 request, 128/128 output tokens.
- CI burst: 32/32 requests, 3200/3200 output tokens.
- Full 36-layer allocation tracker (`TRACKING=1`, `TRACEBACKS=1`, program-cache
  skip unset): primary 1/1 and CI 32/32, zero live unsafe buffers over 226 B1/B32
  model and sampler replays.
- Final orderly shutdown: capability snapshot flushed, traces released, no
  vLLM API/EngineCore process left, all four devices listed by `tt-smi`, no
  reset required.

The live 65-token non-aligned completion from the completed integration remains
applicable because the retained change starts only after sampling. The final
host suite also passes direct 65- and 97-token non-aligned prefill routing.

## Commands

Host tests and formatting:

```bash
source .agents/scripts/gpt_oss_workspace_env.sh
python -m pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  models/autoports/openai_gpt_oss_120b/tests/test_generator_vllm.py

pre-commit run --files \
  models/tt_transformers/tt/common.py \
  models/autoports/openai_gpt_oss_120b/tt/generator.py \
  models/autoports/openai_gpt_oss_120b/tt/generator_vllm.py \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  models/autoports/openai_gpt_oss_120b/tests/test_generator_vllm.py
```

Full live validation used the same server/benchmark arguments as the final
clean command in `README.md`, with:

```bash
--stages serve,sampling,qualitative,benchmark --sampling-profile full
```

Qualitative degeneracy check:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --scope vllm \
  --missing-artifacts critical \
  --json models/autoports/openai_gpt_oss_120b/readiness_vllm/degenerate_check.json
```

## Decisions and rejected options

- **Kept:** select distinct token shards before async readback. It directly
  removes three redundant transfers per P150x4 decode and improved the first
  isolated candidate by 18.99% and the final exact reproduction by 25.10% in
  decode rate.
- **Rejected:** change the shared generic generator for every model. Output
  replication and data-parallel row semantics are model-specific; the
  autoport wrapper has the necessary GPT-OSS contract and retains a safe
  fallback for other payloads.
- **Rejected:** host greedy/top-1, force-argmax, full-logit readback, or a
  second adapter sampler. These violate the serving contract and would bypass
  the selected full-model split sampler.
- **Rejected:** make the read blocking in the adapter. The plugin's async
  finalization contract already owns the synchronization boundary.
- **Rejected:** aligned-only prompts, smaller benchmark contexts, or lower
  `max_model_len`. They violate the context and serving-shape contracts.
- **Rejected:** change the datatype policy, LM head, sampler, trace buckets,
  page size, mesh, or scheduling mode. The measured bottleneck was an
  adapter-side redundant transfer, and the retained fix closes serving to
  91.7% of the already optimized full-model rate without changing numerical
  work.
- **Rejected:** silence allocation tracking with
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=1` or broadly acknowledge the real
  prefill buffers as corruptible. The final fix prevents those allocations
  after trace capture and passes with full program-cache tracking enabled.
- **Not run by design:** Tracy, `tt-perf-report`, live-server device profiler,
  adapter profiler, and `ReadDeviceProfiler`. Same-harness JSON and runtime
  counters are the required evidence for this stage.

## Applicable optimization checklist

| Checklist item | Result and evidence |
| --- | --- |
| Real vLLM TT plugin path | Pass: server command loads standalone plugin and adapter `tt/generator_vllm.py`; all requests complete through it. |
| Decode fully traced and async | Pass: 226 model + 226 sampler submissions, `blocking=False`, 226 async reads, zero unclassified submissions. |
| Persistent token/position/RoPE/page/cache/sampler state | Pass: two B1/B32 initializations, 224 fixed sampling replays, 221 page-table reuses; stale-state host tests pass. |
| No host sampler/argmax/full logits in measured path | Pass: measured snapshot has 226 device and zero host-sampled decodes, zero host argmax, zero full-logit readbacks. |
| LM-head/sampling/feedback included | Pass: adapter delegates to the optimized full-model split sampler and persistent `tt_out_tok` feedback. |
| Operation/topology audit | Pass for serving boundary: model graph is inherited unchanged; four replicated token transfers were reduced to one before the async boundary. |
| Best-candidate and final-default comparison | Pass: 46.103 baseline, 54.859 first candidate, 57.674 final clean reproduction. |
| Same-harness primary and CI before/after | Pass: identical shapes/configuration and `run_vllm_server`; raw commands embedded in JSON. |
| Full-model comparison | Pass: vLLM 57.674 vs comparable full-model 62.879 t/s/u; ITL P50 within 0.334 ms of full-model ms/token. |
| Batch capability | Pass: B1 headline and 32/32 CI burst; full sampling structured-output capacity case passed. |
| Qualitative output | Qualified pass: 12 coherent outputs plus clean degeneracy checker; two thermodynamics responses explicitly incomplete at the fixed cap. |
| Trace allocation safety | Pass: full 36-layer tracker with program-cache tracking enabled completed 226 B1/B32 replays with zero survivors. |
| Runtime cleanup | Pass: terminal snapshot, trace teardown, timestamped post-final-run empty process audit, and four healthy devices. |
| Performance accounting | Pass: `perf_summary.json` records end-to-end/burst metrics; device/roofline values are null with the mandated no-profiler reason. |

Decoder math/layout, MoE, collective, SDPA, LM-head, dtype/fidelity, and
terminal-sampler tuning checklist items are inherited from the completed
optimized-full-model and datatype-sweep stages. This stage verified that vLLM
reuses that path and changed only its redundant host-read boundary.

## Review and commits

The first independent review returned `more-work-needed` for full-depth trace
allocation safety, stale cleanup evidence, an overstated qualitative
comparison, and raw-log whitespace. After AutoFix remediation and exact-path
reruns, a fresh xhigh independent rereview returned `clean-pass` with no
required work. `stage_review.md` records both review phases.

Local implementation, tests, measurements, artifacts, and review evidence are
committed as `bcb3f87dd50d4813bb2c917701dedf3f06b13545` (`Optimize GPT-OSS 120B vLLM
serving`). The follow-up documentation-only closure commit records that SHA.
Nothing was pushed.
