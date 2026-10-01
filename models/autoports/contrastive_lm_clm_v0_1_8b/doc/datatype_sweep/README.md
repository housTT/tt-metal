# Stage 8: datatype (precision policy) sweep

Metric mapping, recorded as the plugin asks: the decoder stage's top-1 and top-5 token accuracy do not exist for an
encoder. The axes used instead are the full-encoder fidelity against the HF fp32 reference on the 308-text corpus
(`tests/run_fidelity.py`) and the traced encode latency (`tests/bench_encoder.py`). Gate: cosine mean >= 0.99,
cosine min >= 0.97, head-projection cosine min >= 0.95, no NaN. Selection: fastest passing policy by 128-token
batch-1 latency; candidates within 1 percent of the fastest are a tie, and the higher minimum cosine wins
(`tests/datatype_sweep.py`, `sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`).

## Candidates

| policy | attention weights, KV | attention math | MLP weights | MLP math |
|---|---|---|---|---|
| accuracy (stock) | bf16 | HiFi4 | bfp8 | HiFi2 |
| bfp8_attn (this port) | bfp8 | HiFi4 | bfp8 | HiFi2, fp16 accumulate |
| bfp8_attn_hifi2 (this port) | bfp8 | HiFi2 | bfp8 | HiFi2, fp16 accumulate |
| performance (stock) | bf16 | HiFi4 | bfp4 (w1, w3) | LoFi |

A BFP4 + LoFi candidate exists for the one BFP4 tensor group (the stock `performance` policy), as the stage asks.

## Results (p150, one chip, 2048-token context, batch 1 / 4 / 8 traces)

| policy | cosine mean | cosine min | cosine p05 | head cos min | 128 tok b1 | 128 tok b8 | 1024 tok b1 | 2048 tok b1 | gate |
|---|---|---|---|---|---|---|---|---|---|
| accuracy | 0.99909 | 0.99588 | 0.99753 | 0.99429 | 57.6 ms | 162.0 ms | 170.5 ms | 322.8 ms | pass |
| bfp8_attn | 0.99901 | 0.99379 | 0.99750 | 0.99278 | 54.0 ms | 142.4 ms | 148.1 ms | 286.2 ms | pass |
| bfp8_attn_hifi2 | 0.99891 | 0.99310 | 0.99697 | 0.99297 | 53.9 ms | 142.1 ms | 147.2 ms | 277.0 ms | pass |
| performance | 0.98742 | 0.92386 | 0.95780 | 0.91279 | 48.9 ms | 119.9 ms | 125.1 ms | 237.2 ms | fail |

Plots: `cosine_mean_perf_pareto.png`, `cosine_min_perf_pareto.png` (selected point in red, dotted gate line).

## Selection

`bfp8_attn`. It is 6.4 percent faster than the stock accuracy policy at 128 tokens and 13 percent faster at 1024
and 2048 tokens, passes every gate, and beats `bfp8_attn_hifi2` on fidelity at the same speed (the HiFi4 to HiFi2
change on the attention matmuls saved 0.02 ms, inside the 1 percent tie band, while costing minimum cosine).
The stock `accuracy` policy remains available as the `p150-accuracy` serve profile for consumers who want the
highest fidelity. The `performance` policy is rejected on real-weight, real-text evidence: a 0.924 minimum cosine
and 0.913 head-projection minimum would change answer probabilities materially for a verifier whose output is the
vector itself.

## Decision-level cross-check

Decision agreement between the selected policy and the CPU reference on the Typed Decisions subset is measured
against the served package in `../release/`.

## Files

`fidelity_<policy>.json` and `bench_<policy>.json` (accuracy's live in `../full_model/` and
`../optimized_full_model/`), `fidelity_*_tt_single.npy` / `_tt_batched.npy` raw vectors (not committed),
`sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, the two PNGs.
