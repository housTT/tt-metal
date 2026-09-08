# vLLM integration work log

## Scope and source state

- Model: `google/gemma-4-26B-A4B-it`.
- Stage input: completed datatype sweep,
  `selected_canonical_profile_policy`.
- Checkpoint revision:
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
- Scope: OpenAI-compatible text serving only; image and video inputs excluded.
- Hardware: four P300C Blackhole devices used as 1/2/4-chip
  P150/P150x2/P150x4 proxies, firmware 19.13.1.
- Starting tt-metal commit: `5e18deb453d`.
- Starting vLLM commit: `5ffebf4128f`.
- Final local implementation commits: recorded after clean stage review.

## Implementation

1. Added the autoport selector and model target to
   `vllm_tt_plugin.platform.register_tt_models()`.
2. Kept `tt/generator_vllm.py` as an adapter over the full-model generator and
   canonical split-sampling token-out path.
3. Bound the adapter to vLLM-owned heterogeneous KV cache tensors and page
   tables. Fixed HMA sharing so equal tensor indices with different physical
   TP4 block volumes cannot alias.
4. Advertised the profile-specific context contract: P150 50,624; P150x2/x4
   262,144.
5. Advertised async decode and device-sampler capabilities. Added explicit
   shared host-sampler eligibility for penalties, explicit seeds, stochastic
   top-k above 32, and existing unsupported logprob/structured modes.
6. Fixed optional host-logit reshaping for the generator's physical 32-row LM
   head; this path is not used in performance runs.
7. Preserved device token feedback by sending an active mask only at scheduler
   ownership boundaries. Added trace release before a B1/multi-request shape
   recapture.
8. Added physical-parent/submesh lifecycle support to the shared TT worker for
   QB2 TP2 serving. A standalone 1x2 fabric cannot initialize the unopened
   router partners; opening the 2x2 parent and selecting a 1x2 submesh is the
   proven topology.
9. Added OpenAI feature checks, async overlap checks, parent-mesh tests,
   sampling eligibility tests, and Gemma 4 reasoning-parser coverage. Updated
   the streaming tool-parser test for typed `DeltaFunctionCall` results.
10. Fixed the shared qualitative runner's no-chat-template path so it selects
    raw completions before attempting local template rendering; a host test
    verifies the endpoint and retained prompt metadata.

## Final server commands

Common environment and flags:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TT_METAL_HOME=/home/hous/dev/tt-metal \
  TT_GEMMA4_TEXT_VER=google_gemma_4_26b_a4b_it_autoport \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it \
  --max-num-seqs 32 --block-size 64 \
  --additional-server-args \
    '--async-scheduling --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4'
```

Profile additions:

```text
P150:   --mesh-device N150 --max-model-len 50624 --port 8000 --output-subdir P150 --tt-config '{"trace_region_size":220000000}'
P150x2: --mesh-device N300 --max-model-len 262144 --port 8001 --output-subdir P150x2 --tt-config '{"trace_region_size":220000000,"fabric_config":"FABRIC_2D","parent_mesh_shape":[2,2],"submesh_offset":[0,0]}'
P150x4: --mesh-device P300x2 --max-model-len 262144 --port 8002 --output-subdir P150x4 --tt-config '{"trace_region_size":220000000,"fabric_config":"FABRIC_1D_RING"}'
```

Every final server reached `/health`, served requests, received SIGTERM through
the readiness runner, closed devices, and left no vLLM/EngineCore process.

## Commands and gates

Feature checks, repeated with the profile URL and expected context:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_feature_checks.py \
  --server-url http://127.0.0.1:<port> \
  --model google/gemma-4-26B-A4B-it \
  --expected-max-model-len <50624|262144> \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/<profile>/openai_feature_checks.json
```

Result: pass on P150/P150x2/P150x4. Each artifact records the served maximum,
29-token non-aligned prompt, parsed `get_weather` tool call, and parsed reasoning
response.

Sampling:

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages sampling --server-url http://127.0.0.1:<port> \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 \
  --sampling-profile full --output-subdir <profile>
```

- P150 full: 72 passed, one expected all-vocabulary chat-logprobs skip, in
  592.78 seconds.
- P150x2 full: 72 passed, one expected all-vocabulary chat-logprobs skip, in
  626.31 seconds.
- P150x4 full: 72 passed, one expected all-vocabulary chat-logprobs skip, in
  693.68 seconds.

Qualitative:

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages qualitative --server-url http://127.0.0.1:<port> \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 \
  --output-subdir <profile>
```

Result: six prompts, twelve completions per profile. All outputs were read in
full and passed coherence, topic, repetition, gibberish, language drift, and
request-contamination review. Exact verdicts are in each
`qualitative_verdict.json`. Greedy and sampled cases explicitly used
device-compatible `top_k=1` and `top_k=32`; long answers may end at the
intentional 256-token cap while remaining coherent. Every profile also passed
the automated degeneracy checker.

Logit determinism:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_logit_determinism.py \
  --server-url http://127.0.0.1:<port> \
  --model google/gemma-4-26B-A4B-it \
  --standalone-baseline models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/standalone_baselines/<profile>/logit_oracle_tp<1|2|4>.json \
  --standalone-batch-control models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/standalone_baselines/P150x4/logit_oracle_tp4.json \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/<profile>/logit_determinism.json
```

Result: pass on all profiles. Each profile's fresh real-weight B1 oracle uses
the exact two-token serving trajectory: a retained 32-token prompt, its prefill
argmax as the decode input, and the resulting decode argmax. Sequential vLLM
responses match that oracle exactly at both positions (zero selected-logprob
and common-top-10 deltas, 10/10 overlap). Repeated runs and duplicate prompt
groups in different positions of the same concurrent batch are also
bit-identical on every profile.

B1 and multi-request serving deliberately select different optimized decode
kernels. That graph-shape comparison is kept separate from determinism: every
profile preserves the exact selected token and text, while the artifact reports
the distribution movement without fitting a pass tolerance to it. A fresh TP4
full-vocabulary control repeats four prompt/token groups in homogeneous and
mixed B32 layouts. All 32 rows select the corresponding B1 global maximum and
same-group top-10/logprobs are bit-identical across row positions and batch
composition. The worst group has raw/centered logits cosine 0.8968/0.9587 and
probability total variation 0.2706, classifying the residual as deterministic,
input-dependent B1/B32 kernel-shape sensitivity rather than row packing, cache
ownership, or request contamination. P150 cannot allocate the standalone
full-stack B32 state at this test's physical `max_seq_len=2048`; its live B32
selected-token and same-graph position gates remain profile-local.

Async overlap:

```bash
python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tests/run_vllm_async_overlap.py \
  --server-url http://127.0.0.1:<port> \
  --model google/gemma-4-26B-A4B-it \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/<profile>/async_overlap_state_test.json
```

Result: pass on all profiles. The long request crossed a 64-token page while a
short request stopped and vacated a slot. Both retained a stable initial prefix,
topic, and isolation. Exact long-text equality is diagnostic only because the
isolated request executes the B1 graph while overlap deliberately transitions
to the padded multi-request graph; both continuations remained coherent and
uncontaminated.

Direct mutable traced-state check:

```bash
env GEMMA4_MIXED_PROBE=1 \
  GEMMA4_MIXED_STATE_OUTPUT=models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/P150x4/mutable_decode_state.json \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_mixed_prompt_and_inactive_slot_probe
```

Result: one passed. The JSON records stale-token/current-position/page-table
and inactive-slot assertions, stable buffer addresses, trace reuse, and refresh
counters.

Host checks:

```bash
python_env/bin/pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_vllm_adapter_contract.py
# 20 passed

cd /home/hous/dev/vllm
/home/hous/dev/tt-metal/python_env/bin/pytest -q \
  plugins/vllm-tt-plugin/tests \
  --ignore=plugins/vllm-tt-plugin/tests/tt
# 118 passed

python_env/bin/pytest -q \
  models/common/readiness_check/test_run_vllm_server.py
# 1 passed
```

The plugin's `tests/tt` directory requires a live server URL/model name; its
sampling and penalty paths are covered by the per-profile 72-pass full live
suite above rather than counted as host-only tests.

Final-source formatting and repository policy checks:

```bash
# Run in each repository; the file list includes modified and untracked files.
<repo-python>/bin/pre-commit run --files \
  $(git diff --name-only) $(git ls-files --others --exclude-standard)
```

Result: every tt-metal hook passed after the hooks' Black/isort/EOF fixes and a
second run; every vLLM hook passed on the first run. The 20-test and 118-test
host suites above were rerun after those formatting changes and passed on the
final source.

## Benchmarks

Benchmark stage command, repeated against each warmed server:

```bash
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages benchmark --server-url http://127.0.0.1:<port> \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 \
  --output-subdir <profile>
```

The primary workload is greedy 128/128/1, concurrency one. P150/P150x2/P150x4
reported TTFT P50 235.1/235.5/203.3 ms and decode 36.2/44.0/48.5 t/s/u. Full
TTFT/TPOT/ITL P50/P99 and throughput are at the top of `README.md` and in each
`vllm_benchmark.json`; raw output is `vllm_result.json`.

The secondary CI serving-burst is greedy 100/100/32 with unbounded client
concurrency. All profiles completed 32/32. P150/P150x2/P150x4 output throughput
was 100.5/122.2/129.5 tok/s. Full metrics are in
`vllm_ci_serving_benchmark.json`, backed by `vllm_ci_serving_result.json` and the
benchmark log. Burst TPOT is not used as headline decode performance.

## Recovery and cleanup audit

- Standalone TP2 `FABRIC_1D` and `FABRIC_2D` opens failed because router peers
  outside the 1x2 logical mesh were unopened. A direct 2x2-parent/1x2-submesh
  smoke succeeded; the shared worker implementation and unit tests now encode
  that lifecycle.
- A mesh-topology transition can leave an active-Ethernet core requiring a
  board reset. The earlier P150x4 determinism launch retained one such heartbeat
  failure in `server_failed_heartbeat_20260908.log.gz`; it failed during device
  open before model construction. A bounded `tt-smi -r` restored firmware
  19.13.1 and the immediate identical launch and determinism run passed.
- The pre-unified-rerun 2x2 mesh smoke encountered the same class of heartbeat
  timeout before any vLLM server launch. With no owner processes, a bounded
  reset/list and repeated exact 2x2 mesh open/close succeeded. The recovery is
  recorded in `readiness_vllm/hardware_recovery_20260908.json`.
- Split trace capture is protected by `corruptible_allocation_scope`.
  Correctness evidence is the explicit release/recapture lifetime, stable
  buffer-address checks, page-table refresh counters, and stale-feedback
  assertions; allocator-warning counts are diagnostic only.
- Final `pgrep` found zero API-server/EngineCore/readiness-server processes.
  `tt-smi -ls --local` listed all four P300C devices. See
  `readiness_vllm/runtime_cleanup_audit.json`.
- P150x4 emitted one recovered vLLM DecodeStream invalid-prefix warning at
  18:33:59 UTC during full sampling for request
  `cmpl-b675f7e6ff20f971-0-84e2e2dc`. The framework reset that request's
  tokenizer stream and retried the same token. Sampling still passed 72/72
  runnable tests with one expected skip, with no HTTP 500 or fatal marker. The
  stress request text was not retained or manually inspected; the runtime
  audit records the narrow recovery classification rather than claiming a
  qualitative verdict for it.

## Artifacts

Primary files under `readiness_vllm/<profile>/`:

- `server.log.xz` (losslessly compressed to satisfy the repository artifact
  size gate)
- `openai_feature_checks.json`
- `sampling_tests.log`
- `vllm_qualitative_outputs.json` and `qualitative_verdict.json`
- `qualitative_degeneracy_check.json` and `logit_determinism.json`
- `async_overlap_state_test.json`
- `vllm_result.json`, `vllm_benchmark.json`, `vllm_benchmark.log`
- `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.json`,
  `vllm_ci_serving_benchmark.log`
- P150x4 only: `mutable_decode_state.json`

Cross-profile evidence under `readiness_vllm/standalone_baselines/` contains
the profile-local B1 oracle/JUnit pairs and the TP4 B1/B32 full-vocabulary
localization. P150x4 also retains
`server_failed_heartbeat_20260908.log.gz`, the classified pre-model startup
failure recovered by board reset.

The three final server lifetimes ran all gates sequentially against one API
PID per profile, then shut down cleanly:

| Profile | UTC lifetime on 2026-09-08 | API PID | POSTs | HTTP 200 / expected 400 / 500 |
| --- | --- | ---: | ---: | ---: |
| P150 | 17:49:09–18:04:15 | 1696137 | 1,581 | 1,580 / 1 / 0 |
| P150x2 | 18:06:03–18:21:27 | 1699866 | 1,581 | 1,580 / 1 / 0 |
| P150x4 | 18:23:11–18:39:39 | 1703317 | 1,581 | 1,580 / 1 / 0 |

`readiness_vllm/unified_gate_manifest.json` records the common flags, exact
gate order, configurations, lifetimes, request/status counts, cleanup audit,
and SHA256 for every primary artifact. The three sampling logs and six
benchmark logs are force-added despite the repository-wide `*.log` ignore
rule. `doc/vllm_integration/AUTODEBUG.md` and `AUTOFIX.md` retain the evidence
provenance diagnosis and its completed repair.

## Independent stage review

The required fresh xhigh review returned **`clean-pass`** with no unresolved
blocking or non-blocking findings. The reviewer independently matched all 39
manifest hashes, decompressed and audited the three complete server lifetimes,
recomputed raw benchmark metrics and deterministic controls, read all 36 final
qualitative completions, and reran the 21 tt-metal plus 118 plugin host tests.
The full verdict, anomaly ledger, evidence boundaries, and residual risks are
retained in `doc/vllm_integration/STAGE_REVIEW.md`.

## Local stage commits

- tt-metal implementation: `40569292be814f72c62ef30fd06c8b8245957517`
- tt-metal initial stage documentation:
  `c718255087bc62907ac2882fdfe3fc7eda613c42`
- tt-metal evidence closure after clean stage review:
  `cedc5ce7a858738b18ad7216943df2fab4c7edc7`
- vLLM TT plugin: `2f81f493b969da7ce3cd64c0f6bc1895b8f229cb`

Both commits are local only. No branch was pushed and no pull request was
opened against `main`.
