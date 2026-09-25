# DEVSTACK-294: stochastic sampling collapse — evidence

Ticket finding: under the shipped `sample_on_device_mode: "all"`, hard prompts sampled with the
checkpoint's default preset (temperature 1.0, top_k 20, top_p 0.95) collapsed into phrase loops
(`4*8=32? 4*8=32? ...`) until `max_tokens`; the vLLM host sampler did not (32 clean runs). Greedy
was never affected. Same class as DEVSTACK-309 (GLM-4-7-flash).

## Why nothing caught it

- vLLM applies `generation_config.json` (top_k 20) to every request that omits sampling fields, and
  top_k 20 is inside the model's `device_sampling_max_top_k: 32`, so every default request ran on
  the bounded `ttnn.sampling` path.
- Every shipped quality gate ran greedy (`top_k=1` → exact argmax, which bypasses that path). The
  only stochastic check in tt-metal (`models/common/tests/modules/sampling/test_sampling_1d.py`)
  tests set membership with a 6/32 tolerance and skips on Blackhole.
- Two harness defects made stochastic runs in the direct harness misleading: unseeded
  `set_sampling_params` dropped the RNG streams so `_advance_sampling_seeds` silently reused the
  static `arange` seed every step (fixed-quantile decode), and the readiness compile hooks bypassed
  the serving bridge. Both fixed in `tt/model.py` / `tt/generator.py`. Neither reached the served
  path: `_apply_serving_sampling_params` has synthesized entropy seeds and the traced decode step has
  refreshed the device seed buffer since the first commit (2026-08-28).

## Measurements

### Device sampler distribution (`tests/test_device_sampler_distribution.py`)

8,192 draws per case, 20 candidates at random ids on the 248,320 vocabulary (they land on different
shards), (4,1) mesh, fresh replicated seeds every call, reference = vLLM/HuggingFace top-k → top-p →
softmax semantics on the same bf16 logits. Candidate probabilities are tie-free (LM-like: top-1 0.40
with a geometric tail; peaked: top-1 0.90 with a geometric tail; flat: 20 near-equal), so "outside the
nucleus" cannot be a tie-break artefact. "Noise" is the same metric on exact draws from the reference.

| case (k / p / T) | metric | approximate exp (shipped kernel) | precise exp (tried, reverted) |
| --- | --- | ---: | ---: |
| LM-like, 20 / 0.95 / 1.0 | TVD (noise 0.011) | 0.015 | 0.013 |
| LM-like | argmax rate device / reference | 0.403 / 0.412 | 0.411 / 0.412 |
| LM-like | draws outside the top-p set | 0 / 8,192 | 0 / 8,192 |
| peaked (top-1 0.90), 20 / 0.95 / 1.0 | TVD (noise 0.003) | 0.017 | 0.015 |
| peaked | argmax rate device / reference | 0.954 / 0.937 | 0.948 / 0.937 |
| peaked | draws outside the top-p set | 0 / 8,192 | 0 / 8,192 |
| flat (20 near-equal), 20 / 0.95 / 1.0 | TVD on the sorted profile (noise 0.019) | 0.055 | 0.052 |
| LM-like, 20 / 1.0 / 1.0 (top-k only) | TVD (noise 0.012) | 0.022 | 0.019 |
| LM-like, 32 / 1.0 / 1.0 | TVD (noise 0.013) | 0.020 | 0.016 |
| LM-like, vocab 4,096, 20 / 0.95 / 1.0 | TVD (noise 0.012) / chi-square p | 0.019 / 0.02 | 0.008 / 0.64 |

Files: `approx_exp/sampler_distribution_4x1.json` (shipped kernel), `sampler_distribution_4x1.json`
(precise exp). An earlier revision of this test used tied candidate probabilities and reported 1.8 % of
peaked draws "outside the nucleus"; that reading was the tie-break and is withdrawn.

Reading: the device sampler is close to the reference and, if anything, slightly *too greedy*: on the
peaked row it over-weights the argmax by 1.1–1.7 points, and it never emitted a token the host sampler
would have excluded. The one real defect is on flat rows: the sampler draws a single bf16 uniform per
row and cuts bf16 probabilities (`writer_interleaved.cpp`), which quantizes a 20-way flat CDF to ~1/256
steps (TVD 0.05, reported but not gated). The precise-exp kernel changes nothing measurable and is not
shipped. All four device shards return identical tokens (shard divergence ruled out; `copy_to_device`
broadcasts a one-shard host tensor). Sampler numerics therefore do not explain a phrase-loop collapse;
the endpoint A/B below is the decisive test of the ticket's report.

### Model-level stochastic free run (`tests/test_stochastic_free_run.py`, `stochastic_free_run.json`)

AIME24 "Aya walk" chat prompt, 1,024 new tokens, temperature 1.0 / top_k 20 / top_p 0.95, seed synthesis
fix in place (first run on the precise-exp kernel; repeated on the shipped kernel in `approx_exp/`):

| run | degenerate (precise exp / shipped kernel) | repeated 4-grams / tokens (precise / shipped) | seed refreshes | note |
| --- | --- | ---: | ---: | --- |
| device, seed 17 | no / no | 0.218 / 0.166 | 1,024 | coherent algebra |
| device, seed 2024 | no / no | 0.200 / 0.185 | 1,024 | coherent algebra |
| device, seed 90210 | no / no | 0.227 / 0.155 | 1,024 | reaches 180 + 24 |
| device, unseeded | no / no | 0.249 / 0.106 | 1,024 | answers `\boxed{204}` (correct) |
| host control, seed 17 | no / no | 0.215 / 0.215 | 1 | reference |

Sampling parameters and seed buffers identical on all four shards; the seed counter advanced once
per token in every device run.

### Endpoint A/B (`endpoint_stochastic_*_20260916.json`)

Same prompt through `/v1/chat/completions`, 8 runs each, `max_tokens 2048`, thinking on:

| server | degenerate rate | empty content | `finish_reason == length` |
| --- | ---: | ---: | ---: |
| host sampler (shipped mitigation) | 0 / 8 | 0 / 8 | 0 / 8 (933–1,486 tokens; 8 / 8 answer 204) |
| device sampler, precise exp (`QWEN38_DEVICE_SAMPLING_MAX_TOP_K=32`) | 0 / 8 | 0 / 8 | 0 / 8 (1,076–1,278 tokens; 8 / 8 answer 204) |
| device sampler, approximate exp (the shipped kernel) | 0 / 8 | 0 / 8 | 0 / 8 (900–1,458 tokens; 8 / 8 answer 204) |
| **shipped 2026-09-11 image as installed** (`tt-model serve tt-hous/qwen3.8-flash-next-p300x2`, device sampler on) | 0 / 8 | 0 / 8 | 0 / 8 (995–1,702 tokens; 8 / 8 answer 204) |

Reading: on the new revision the ticket's collapse does not reproduce — 16 / 16 device-sampled runs and
8 / 8 host-sampled runs with the checkpoint's default preset finish on their own with the correct answer
and the same repeated-4-gram ratio (0.17–0.25). The device arms were also faster end to end (mean 48.7 /
51.6 s vs 73.7 s host) because of the ~18 ms/token host-sampling cost. What differs from the tester's
setup is unknown until the request bodies arrive: their prompt set, any `presence_penalty`/
`repetition_penalty` on the host arm (which vLLM routes to the host regardless of the mode), — not the
shipped build itself: the installed 2026-09-11 image, served unchanged with its device sampler on, also
completes 8 / 8 runs of this prompt (`endpoint_stochastic_device_shipped_20260911build.json`).

### Greedy endpoint regression (`doc/correctness/gsm8k_endpoint_20260916_ship.json`)

GSM8K-20 through the dev endpoint on the shipped configuration (128-row microchunks, greedy on device):
18 / 20 (0.90), 2 length-truncated, no degenerate or empty replies; the 2026-09-11 baseline was 45 / 50
(0.90). Boundary check `readiness_vllm/non_aligned_prompt_check_20260916.json`: 14 lengths from 1 to
1,537 tokens, all pass.

### Cost of host sampling (`bench/`)

`vllm bench serve`, one user, `ignore_eos`:

| ISL / OSL | TPOT greedy (device argmax) | TPOT temperature 1.0 (host) |
| --- | ---: | ---: |
| 128 / 128 | 41.4 ms (24.2 tok/s) | 59.1 ms (16.9 tok/s) |
| 4,096 / 256 | 76.3 ms (13.1 tok/s) | 94.2 ms (10.6 tok/s) |

The host path costs a constant ~18 ms per generated token (full-vocabulary bf16 logits readback, ~0.5 MB
per step, plus vLLM's CPU top-k/top-p sampler and the token upload); TTFT is unchanged (338 vs 339 ms,
11.22 vs 11.31 s). The decode trace also switches variant (`token_out` ↔ `model_only`, ~0.5 s recapture)
once whenever consecutive requests alternate between greedy and stochastic sampling.

## Staged package smoke (2026-09-16, `tt-model serve <staged>/tt_kernel_manifest.json --port 20020`)

Image `tt-model/qwen3.8-flash-next-p300x2:1e0e37dca22f` staged at
`/home/ttuser/dev/qwen3.8-flash-next/tt-metal/build/qwen3.8-flash-next-p300x2` (manifest env
`QWEN38_PREFILL_CHUNK=128`, `device_sampling_max_top_k == 0` in the image, upstream `sampling.cpp`):

| check | result |
| --- | --- |
| server log | `Default vLLM sampling parameters have been overridden by the model's generation_config.json` (temperature 1.0, top_k 20, top_p 0.95) |
| boundary prompts 1–1,537 tokens (`readiness_vllm/non_aligned_prompt_check_20260916_staged.json`) | 14 / 14 pass |
| tool calling (`readiness_vllm/run_toolcall_smoke.py`) | single, parallel, streaming, result round-trip, no spurious call: all pass |
| GSM8K-20 greedy (`doc/correctness/gsm8k_endpoint_20260916_staged.json`) | 18 / 20, 0 degenerate, 0 empty |
| AIME x8, default preset, host-sampled (`endpoint_stochastic_host_staged_20260916.json`) | 0 / 8 degenerate, 7 / 8 answer 204 and stop; 1 / 8 hit `max_tokens 2048` mid-answer (coherent, not a loop) |
| probe rows (`doc/prefill_sweep/probe_rows_staged_20260916.json`) | 128/128: TTFT 0.34 s, TPOT 41.0 ms; 4,096/256: 11.24 s, 76.3 ms; 65,536/256: 222 s, 76.9 ms (within noise of the card table) |

## Latency sweep of the staged package (2026-09-17, community harness)

`bench-sweeps/rerun/rerun_qwen38_staged.py` drives `sweep.py`'s own code paths against the staged
image `tt-model/qwen3.8-flash-next-p300x2:1e0e37dca22f` (greedy, one closed-loop user, `ignore_eos`,
one unrecorded warm-up per input length). Rows in
`/home/ttuser/dev/bench-sweeps/results/tt-hous__qwen3.8-flash-next-p300x2__long-context/rows.jsonl`
(the 2026-09-11 rows are archived under `results_superseded_20260917T113845Z/`); they are the card table.

| ISL | OSL | TTFT | prefill tok/s | TPOT | decode tok/s | E2EL | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 128 | 0.33 s | 383 | 41.0 ms | 24.4 | 5.5 s | ok |
| 1,024 | 256 | 2.30 s | 446 | 41.0 ms | 24.4 | 12.8 s | ok |
| 4,096 | 256 | 11.26 s | 364 | 76.2 ms | 13.1 | 30.7 s | ok |
| 16,384 | 256 | 52.44 s | 312 | 76.6 ms | 13.0 | 72.0 s | ok |
| 32,768 | 256 | 108 s | 303 | 76.8 ms | 13.0 | 128 s | ok |
| 65,536 | 256 | 222 s | 295 | 76.8 ms | 13.0 | 242 s | ok |
| 131,072 | 256 | 458 s | 286 | 76.9 ms | 13.0 | 478 s | slow |

Identical to the 2026-09-11 package within noise: this revision changes sampling, documentation and gates,
not the greedy path. The stochastic (host-sampled) path is not covered by the sweep; see the bench table above.

## What ships

- `tt/generator_vllm.py`: `device_sampling_max_top_k: 0` — every `temperature > 0` request is
  sampled by vLLM on the host; greedy stays on the device argmax. `QWEN38_DEVICE_SAMPLING_MAX_TOP_K=32`
  restores the device sampler for A/B runs.
- Seed synthesis for unseeded stochastic sampling, hard error instead of a silent no-op, compile
  hooks through the serving bridge, 4-D-logit-safe host compatibility sampler.
- Regression gates: `test_device_sampler_distribution.py` (gated: TVD <= 0.03 and no draw outside the
  nucleus on tie-free rows; the flat row is report-only), `test_stochastic_free_run.py`, the
  `repeated_four_grams` rule in `demo/full_model.py::_degeneracy`, and `--temperature/--top-p/--top-k/
  --repeats/--prompt-file` plus a `degeneracy` field in `demo/run_gsm8k_endpoint.py`.
- No ttnn kernel change: the precise-exp variant of `sampling.cpp` was measured and reverted.
- Card text: recommended sampling preset, which parameters are host-sampled, known limitations.

Re-enabling the device stochastic path needs the endpoint A/B to show the same degenerate rate as
the host sampler over a larger sample (the ticket's 8 collapsed runs vs 32 clean), plus a decision on
the flat-row quantization (fp32 uniform draw in `writer_interleaved.cpp`), which is a ttnn-wide item
shared with DEVSTACK-309.
