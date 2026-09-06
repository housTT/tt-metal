# Datatype sweep work log

## Scope, gates, and hardware

This stage covers only `google/gemma-4-26B-A4B-it` datatype selection; vLLM
work was not started. Skills used: `datatype-sweep`, `tt-device-usage`,
`qualitative-check`, and `stage-review`. Accuracy gates are top-1 >= 90%,
top-5 >= 98%, and top-100 = 100% on the pinned AIME24 chat-template readiness
reference with 100 generated tokens (SHA256
`a7cad8aabafa0bff94a8b38f430d3468de9ca1d292b56cc15a7f4f19b844b5f8`).

Four P300C Blackhole devices passed health and mesh checks (firmware 19.13.1,
KMD 2.8.0, DRAM healthy, zero uncorrectable errors). Hardware jobs were
serialized. No reset was needed. An intentionally interrupted diagnostic run
was followed by a clean health check before measurement resumed.

## Reproducible commands

All commands use repository root `/home/hous/dev/tt-metal`. Every row in
`sweep_results.json/csv` carries a complete profile-specific command, config,
reference, output directory, and JUnit destination. Canonical form:

```bash
cd /home/hous/dev/tt-metal && env \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_PRECISION_CONFIG=models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/configs/experts_bfp4_lofi.json \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_READINESS_REFERENCE=models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts/gemma4_aime24_chat.refpt \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/experts_bfp4_lofi \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150x4]' \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/experts_bfp4_lofi/junit.xml
```

The final default readiness run omitted `GEMMA4_PRECISION_CONFIG` and selected
all three profile IDs. Post-selection token-out used the same default path:

```bash
cd /home/hous/dev/tt-metal && env \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_PREFILL_BENCH=1 GEMMA4_NO_HOST_TOKEN_OUT_BENCH=1 \
  GEMMA4_NO_HOST_WARMUPS=5 GEMMA4_NO_HOST_ITERATIONS=128 \
  GEMMA4_PROBE_PROMPT_LEN=128 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/selected_token_out \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/selected_token_out/junit.xml
```

Qualitative and artifact generation:

```bash
cd /home/hous/dev/tt-metal && env HF_HOME=/home/hous/.cache/huggingface \
  HF_HUB_OFFLINE=1 python_env/bin/python \
  models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/run_selected_qualitative.py

cd /home/hous/dev/tt-metal && python_env/bin/python \
  models/common/readiness_check/check_degenerate_output.py \
  models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/selected_qualitative \
  --scope autoregressive --missing-artifacts critical \
  --json models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/artifacts/selected_qualitative/degeneracy_check.json

cd /home/hous/dev/tt-metal && MPLBACKEND=Agg python_env/bin/python \
  models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/generate_sweep_artifacts.py
```

## Results and selection

Refreshed canonical baseline teacher top-1 is 96/96/94%, top-5/top-100 is
100%, and traced teacher decode is 34.522/41.472/43.431 t/s/u. The final
default path passes at 96/96/95% top-1 and 34.508/42.139/45.015 t/s/u; TP4 is
the median of 45.613/45.015/44.987. Selected token-out is
37.256/46.531/52.202 t/s/u with warmed prefill
148.127/126.560/102.998 ms, 134 replays, and zero readbacks.

All-expert BFP4+LoFi passes every profile but is selected only on TP4:
31.692/26.449/45.015 t/s/u for P150/P150x2/P150x4. P150 and P150x2 retain
their faster BFP8/mixed policies. The one-chip policy still uses BFP4 gate/up
on all five full-attention expert layers and is physically feasible at the
advertised 50,624-token limit.

The first independent review returned `more-work-needed`. It found that full
dense BFP8 policy was overwritten to BF16, packed expert gate/up ignored its
separate fidelity, commands were not executable, post-selection prefill TTFT
was omitted, plots were not embedded, and active-trace warnings lacked
classification. Fixes and refreshed evidence:

- explicit dense dtype now precedes the full-layer fallback; corrected BFP8
  full-dense LoFi/HiFi2 results show BFP8 live tensors and 43.145/42.987 t/s/u;
- packed/unpacked expert gate/up prefill/decode calls consume
  `expert_gate_compute_config`; down remains independent;
- BFP8 expert HiFi2 (37.343), BFP4 HiFi2 (44.349), and BFP4 LoFi (45.015
  median) were rerun through actual consumers;
- every aggregate row has a complete executable command; prefill initial and
  warmed latency is linked separately from token-out decode;
- charts label principal competitors and are embedded in README;
- FP32 cache-update overrides are rejected until implemented.

## Context, qualitative, and anomaly ledger

Selected KV dtype/layout/chunking remains BF16 and unchanged. Recomputed
`../context_contract.json` preserves 50,624/262,144/262,144 tokens. Therefore
the existing current-head non-aligned probes at 50,623/262,143/262,143 remain
applicable; no datatype-triggered rerun was required. The evaluated BFP8 cache
passed accuracy but regressed to 22.317 t/s/u.

The final selected TP4 policy passed all six chat-template prompts against
matching HF controls. The degeneracy checker exited 0 with no findings.

Observed anomaly: active-trace allocation warning in untracked logs. The
warning is the runtime's conservative mode when tracking is disabled; it does
not itself identify corruption. Control: reran P150 with
`TT_METAL_TRACE_ALLOC_TRACKING=1`, which turns trace allocation tracking into
an enforced check. It completed the full prefill, five warmups, 128 timed
replays (134 total), and zero readbacks without an unsafe-allocation failure.
All transient allocations are inside `ttnn.corruptible_allocation_scope` and
the output token is read only after replay. The diagnostic measured 2.63
t/s/u because tracking overhead is large, so it is retained only under
`artifacts/trace_allocation_tracker_check/` and intentionally interrupted
after the passing P150 case. Untracked matched results are the performance
record.

Observed anomaly: TP4 timing varied across close candidates. Accuracy and
trace counts were stable. Three matched samples define the selected median;
sub-percent points are treated as noise.

Observed anomaly: the initial full-dense and expert-fidelity rows did not
consume their JSON. They were invalidated, runtime plumbing was fixed, host
propagation checks added, and all affected rows rerun. The checked-in aggregate
contains only the corrected measurements. Their obsolete pre-fix compressed
logs and superseded JUnit files were removed; the corrected per-run JSON and
JUnit evidence is retained.

Observed anomaly: the first historical degeneracy invocation used raw text
instead of the supported structured artifact and exited 2. The retained
structured rerun and final-policy rerun both exit 0.

## Verification and limitations

Completed checks: three-profile selected AIME24 readiness; three-profile
post-selection benchmark; final six-prompt qualitative suite; degeneracy
check; tracker-enforced trace-lifetime check; policy/consumer host tests;
Python compilation; JSON parsing; generated JSON/CSV agreement; visual plot
inspection; targeted pre-commit; and `git diff --check`. This stage changes
Python and artifacts only, so AGENTS.md does not require the C++ build.

Measurements are specific to the four local P300C devices used as P150-family
proxies. Context inheritance is valid only while selected cache dtype/layout
and chunking remain unchanged. No vLLM or serving integration was attempted.

## Review and commits

Initial independent stage review: `more-work-needed`; all findings above were
worked. Final independent rereview: `clean-pass`; no required work remains.
The full handoff verdict is in `stage_review.md`.

Base SHA: `ca7ee6b88f1503f24f36195e1e4ade36b8144852`.

The local implementation checkpoint SHA is appended after commit. No push is
performed.
