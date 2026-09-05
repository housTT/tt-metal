# Fused-decoder work log

## 2026-09-05: starting point and scope

Started from clean branch `hous/gemma-4-26b-a4b-it` at functional-decoder
checkpoint `a80e3d2911f488c7e207b6cfcfcaedeb9d20be57`. Work was restricted to
`tt/fused_decoder.py`, its tests, and `doc/fused_decoder/`; no later autoport
stage was started.

`timeout 60 tt-smi -ls --local` reported four Blackhole P300 devices. A TTNN
1x1 mesh open/close smoke on physical device 1 printed `MESH_SMOKE_OK`.
Hardware commands were run serially. Watcher and Tracy were never combined.

Functional evidence used as the baseline:

| Layer kind | Prefill PCC | Decode PCC | Prefill device | Traced decode device |
| --- | ---: | ---: | ---: | ---: |
| sliding attention | 0.999163 | 0.999739 | 1,242.489 ms | 3.012 ms |
| full attention | 0.998457 | 0.999860 | 1,243.618 ms | 3.207 ms |

## Graph audit and candidate loop

The functional graph already used dedicated RMSNorm, QKV split/create-head,
head concat, HF rotary, paged prefill/decode SDPA, top-k, numeric-stable
softmax, and paged cache update/fill operations. The remaining plausible
shared-LHS, fold, structural, and producer-consumer candidates were isolated
with environment switches. Every candidate artifact records its exact switches
and code hashes.

Representative sliding host screen (ms, prefill / 20-replay traced decode):

| Candidate | Result | Decision |
| --- | ---: | --- |
| all fusions off | 1,242.926 / 3.058 | baseline |
| packed dense, manual activation | 1,243.067 / 3.023 | packed projection viable |
| packed dense + dedicated GeGLU | 1,243.116 / 3.025 | selected with dedicated op |
| folded router only | 1,242.519 / 3.048 | selected; helps combined graph |
| branch add + RMSNorm only | 1,243.056 / 3.091 | rejected |
| packed experts, accurate GeGLU | 705.608 / 2.510 | selected; dominant gain |
| combined including branch norm | 705.124 / 2.494 | rejected versus no-branch |
| combined without branch norm | 705.123 / 2.456 | selected topology |
| combined without router fold | 705.653 / 2.515 | router fold retained |
| fast expert `ttnn.geglu` | 704.731 / 2.513 | rejected: no decode win, lower PCC |
| routed-reduction matmul | 721.214 / 2.460 | rejected: prefill regression |
| selected + one final concat | 704.606 / 2.463 | selected structural simplification |

Fast expert GeGLU initially exposed a rank-3 composite-shape limitation. A
metadata-only rank-4 adaptation made the candidate runnable; it passed PCC at
0.999205 prefill and 0.999694 decode. It was still rejected because it did not
beat accurate GeGLU on traced decode and reduced accuracy.

The first packed-expert 1,024-token attempt requested 256 down-matmul M blocks
from an eight-core sparse program. Adapting only the internal implementation to
32-token tiles fixed the kernel geometry while keeping public logical lengths
arbitrary. A list concat then reduced 31 growing output concats to one.

The routed-reduction matmul candidate passed at 0.999312/0.999722 PCC but
regressed prefill. The branch/RMSNorm candidate was numerically correct but
made the selected traced graph slower. The available fused Q/K rotary kernel
was rejected before runtime because its Llama adjacent-pair/transform-matrix
contract is not Gemma's HF rotate-half contract.

## Final correctness and contract commands

The regular final suite command is:

```bash
source python_env/bin/activate && \
timeout 1800 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py
```

It exercises real weights, both meaningful layer kinds, both full-attention
cache views, paged prefill boundaries, batch-2 prefill, traced batch 1/32
decode, mutable stable trace buffers, and the 1,104-replay bounded ring-cache
stress. Opt-in perf and capacity cases are run separately.

Advertised-context traced decode:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 GEMMA4_RANGE_DOWNLOAD=1 \
timeout 1800 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_advertised_context
```

Result: 2 passed. Both layer kinds allocated the 262,144-token cache, decoded
at position 262,143 through a trace, preserved cache sentinels, produced finite
output, and repeated at PCC 1.0.

Physical real-weight prefill capacity, run once for each length:

```bash
GEMMA4_PREFILL_CAPACITY_LENGTH=262143 timeout 3600 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_prefill_capacity_probe

GEMMA4_PREFILL_CAPACITY_LENGTH=262144 timeout 3600 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_prefill_capacity_probe
```

Results: 2 passed at each length. At 262,144 tokens, sliding/full host elapsed
times were 181.1/266.6 seconds inside the measured calls. The functional
context contract is preserved without modification.

## Final performance

Final unprofiled, current-source A/B commands set
`GEMMA4_FUSED_DECODER_PERF=1` and
`GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=20`. The baseline additionally set
all seven `GEMMA4_FUSED_*` candidate switches to `0`; the selected run used
defaults. Results:

| Layer kind | Baseline host prefill/decode | Selected host prefill/decode |
| --- | ---: | ---: |
| sliding attention | 1,243.063 / 3.052 ms | 704.928 / 2.456 ms |
| full attention | 1,244.202 / 3.230 ms | 706.109 / 2.653 ms |

Final Tracy commands and raw paths are recorded in `perf/README.md`. Both
pytest workloads and post-processing passed. `tt-perf-report` signpost slices
show:

| Layer kind | Prefill device total | Traced decode device total |
| --- | ---: | ---: |
| sliding attention | 708.482 ms / 494 ops | 2.417 ms / 72 ops |
| full attention | 709.012 ms / 494 ops | 2.600 ms / 74 ops |

The tables were inspected for host conversion/fallback, layout churn, reshard,
and sparse topology. No host/Torch fallback or reshard exists. Remaining
sharding/layout operations belong to attention head consumers and the
router/scatter/sparse-mask contract. The selected path reduces sparse matmuls
from 96 to 64 in prefill and 3 to 2 in decode.

## Watcher

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/watcher \
timeout 900 pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_real_weights_prefill_decode[blackhole-sliding_attention_shared_cache-device_params0-mesh_device0]' \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_real_weights_prefill_decode[blackhole-full_attention_natural_cache-device_params0-mesh_device0]'
```

Result: 2 passed in 11.27 seconds. Watcher attached to all four visible devices
and stopped cleanly. The 1,220-line log SHA-256 is
`3e990c41981943935ccccc99e5b370ed02f6bce78982970b115bd0f1f5a9c5b1`;
the fatal-pattern scan had no matches. `watcher/summary.json` is the durable
summary; generated watcher internals remain ignored.

## Formatting, review, and commits

The Python files pass:

```bash
source python_env/bin/activate && pre-commit run --files \
  models/autoports/google_gemma_4_26b_a4b_it/tt/fused_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py
```

No build is required: this stage changes only Python, JSON, CSV, text reports,
and Markdown. Stage-review verdicts and local commit SHAs are appended below.
Nothing is pushed.

## Independent review remediation and final selection

The first fresh xhigh stage review returned `more-work-needed` and is retained
as `stage_review.md`. It accepted the functional-path identity, PCC, context,
watcher, and initial profile evidence, but requested direct functional/fused
isolation, matched activation measurements, more expert geometry work,
router-per-expert and shared-normalization folding, stronger default-path
assertions, and final durable artifact provenance.

Direct functional/fused equivalence was added for dense-only, router-only,
expert-only, and selected graphs across layers 0 and 5. Each case constructs
distinct decoder instances and distinct KV caches, runs prefill, and compares
traced decode:

```bash
source python_env/bin/activate && \
pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_direct_functional_fused_equivalence \
  -s
```

Result: 8 passed. The selected graph directly matches functional at
0.999652/0.999866 PCC (sliding prefill/decode) and
0.999700/0.999902 (full prefill/decode).

The accurate expert GELU was folded into the following binary multiply using
`input_tensor_a_activations=[UnaryWithParam(GELU, 0.0)]`. A matched fast
`ttnn.geglu` candidate, including the final padding, scale folds, shared norm,
and sparse geometry, passed both kinds but lost:

| Candidate | Sliding prefill/decode | Full prefill/decode |
| --- | ---: | ---: |
| accurate binary activation | 278.484 / 1.319 ms | 279.685 / 1.496 ms |
| fast composite activation | 279.989 / 1.320 ms | 281.256 / 1.497 ms |

Fast sliding PCC was also lower at 0.999209/0.999659 versus
0.999309/0.999711. The accurate binary activation is selected.

Folding router per-expert scale into expert down weights passed at
0.999307/0.999743 and 278.627/1.387 ms. Sharing the unweighted pre-FFN RMSNorm
and folding the dense/expert norm weights passed at 0.999307/0.999699 and
278.552/1.320 ms. Their combination passed at 0.999309/0.999711 and
278.484/1.319 ms, so both are selected.

Expert sparse geometry screens used the final accurate graph:

| Width / K block | Prefill | Traced decode | Decision |
| --- | ---: | ---: | --- |
| 1,536 / 1 | 515.524 ms | 1.875 ms | viable, underblocked K |
| 1,792 / 1 | 536.992 ms | 1.910 ms | rejected |
| 2,048 / 1 | 557.227 ms | 1.965 ms | rejected |
| 1,536 / 2 | 346.854 ms | 1.524 ms | viable |
| 1,536 / 4 | 278.652 ms average | 1.398 ms average | selected |
| 1,536 / 8 | 280.161 ms average | 1.395 ms average | slower prefill |
| 1,536 / 11 | 283.484 ms average | 1.397 ms average | slower prefill |

Block widths 4, 8, and 11 each used three runs with five trace warmups and 20
measured replays. Padding 1,408 to 1,536 changes the gate/up sparse geometry
from 4 to 48 cores. With the adapted `per_core_M`, larger internal groups were
also tried: 64/128/256/1,024 tokens produced
392.297/619.982/1,072.950/3,800.761 ms prefill. All are slower than the
selected 32-token group's 278.476 ms, so 32 remains an internal choice rather
than a kernel-shape blocker.

## Final post-review evidence

After freezing source and tests at SHA-256
`a9f3d0b776674ecad286500fbd86a5d2cf6bb2fbc1d606312724d484c8aea4ea`
and `fffff4af877208551844a2bd935c86dc4573a54c491c53de1716969db0aeb364`,
the complete regular suite passed 25 tests and skipped only the six explicit
perf/context opt-ins:

```bash
source python_env/bin/activate && timeout 1800 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py
```

Final context commands:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 GEMMA4_RANGE_DOWNLOAD=1 \
timeout 1800 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_advertised_context

GEMMA4_PREFILL_CAPACITY_LENGTH=262143 timeout 3600 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_prefill_capacity_probe

GEMMA4_PREFILL_CAPACITY_LENGTH=262144 timeout 3600 pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k fused_prefill_capacity_probe
```

Advertised decode: 2 passed. Physical prefill: 2 passed per length. Artifact
elapsed times were 71.999/157.489 seconds at 262,143 and
72.000/157.482 seconds at 262,144 (sliding/full). The context contract remains
unchanged.

Final watcher command:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/google_gemma_4_26b_a4b_it/doc/fused_decoder/watcher/definitive \
timeout 900 pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_real_weights_prefill_decode[blackhole-sliding_attention_shared_cache-device_params0-mesh_device0]' \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_real_weights_prefill_decode[blackhole-full_attention_natural_cache-device_params0-mesh_device0]'
```

Result: 2 passed in 6.53 seconds. The 1,220-line watcher log SHA-256 is
`959b7e198a53fad82725cb29d800f4e0006a98bc5c2fe922c9bdb153c036badb`;
the fatal-pattern scan is empty.

The rereview identified two final adjacent-op opportunities. For dense MLP,
the composite `ttnn.geglu` and fast GELU folded into the following binary
multiply were measured for 200 traced replays. Sliding was
278.501/1.31600 versus 278.475/1.31564 ms; full was 279.652/1.50615 versus
279.706/1.49672 ms. The binary-input activation is selected: it removes the
standalone GELU and improves full decode by 0.63% for negligible prefill
movement. The learned final residual scalar was then fused into residual add
with `MUL_UNARY_SFPU`; the isolated fused candidate measured
278.437/1.30943 ms sliding and 279.634/1.48683 ms full while retaining
0.999307/0.999716 and 0.998005/0.999900 prefill/decode PCC. It is selected.

Final unprofiled timings, after 20 warmups and over 200 measured trace
replays, are 278.476/1.311 ms sliding and 279.607/1.495 ms full. Definitive
Tracy captures and `tt-perf-report` commands are in `perf/README.md`; device
totals are 277.751/1.269 ms with 456/65 ops and 278.891/1.451 ms with 456/67
ops. This supersedes the earlier pre-review final performance section in this
log.

The repository pre-commit command for the scoped files passed. No C++ or CMake
file changed, so the AGENTS.md verification matrix does not require a build.
Fresh rereview verdict and local stage commit SHAs follow below; nothing is
pushed.
