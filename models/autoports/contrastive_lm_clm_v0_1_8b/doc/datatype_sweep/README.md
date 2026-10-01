# Stage 8: datatype (precision policy) sweep

Metric mapping, recorded as the plugin asks: the decoder stage's top-1 and top-5 token accuracy do not exist for an
encoder. The axes used instead are the full-encoder fidelity against the HF fp32 reference on the 308-text corpus
(`tests/run_fidelity.py`) and the traced encode latency (`tests/bench_encoder.py`). Gate: cosine mean >= 0.99,
cosine min >= 0.97, head-projection cosine min >= 0.95, no NaN. Selection: fastest policy that passes every gate, by 128-token
batch-1 latency; candidates within 1 percent of the fastest are a tie, and the higher minimum cosine wins
(`tests/datatype_sweep.py`, `sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`).

## Candidates

Exact definitions (`models/tt_transformers/tt/model_config.py` `ModelOptimizations._default_settings`, `.accuracy`,
`.performance`; `tt/encoder.py` `CUSTOM_POLICIES`). The stock default is bfp8 weights everywhere, HiFi2 for the
linears (fp16 accumulate in the MLP), HiFi4 for prefill SDPA.

| policy | attention weights and KV | QKV / wo prefill math | SDPA prefill | MLP weights | MLP math |
|---|---|---|---|---|---|
| bf16_all (this port) | bf16 | HiFi4 | HiFi4 | bf16 | HiFi4 |
| accuracy (stock) | bf16 | HiFi4 | HiFi4 | bfp8 | HiFi2 fp16 acc |
| bfp8_attn (this port, equals the stock defaults) | bfp8 | HiFi2 | HiFi4 | bfp8 | HiFi2 fp16 acc |
| bfp8_attn_hifi2 (this port) | bfp8 | HiFi2 | HiFi2 | bfp8 | HiFi2 fp16 acc |
| bfp8_lofi_mlp (this port) | bfp8 | HiFi2 | HiFi4 | bfp8 | LoFi |
| performance (stock) | bfp8 | HiFi2 | HiFi4 | bfp4 (w1, w3), bfp8 (w2) | LoFi (w1, w3), HiFi2 fp16 acc (w2) |

An earlier version of this table described bfp8_attn as "HiFi4 attention math" and performance as "bf16 attention,
HiFi4"; both were wrong and were corrected after the stage review. `bfp8_attn` is exactly the stock default policy;
`bfp8_attn_hifi2` differs from it only in SDPA prefill fidelity.

Gate (added after review C): Typed Decisions argmax agreement with the fp32 reference on the 40-case subset must be
at least 98 percent over the 188 decisions whose reference top-2 margin is at least 0.10
(`tests/decision_agreement.py`, `agreement_<policy>.json`), in addition to the cosine gates above.

## Results (p150, one chip, 2048-token context, batch 1 / 4 / 8 traces; regenerated on the final code 2026 Oct 1 23:11 to 23:19 UTC)

| policy | cosine mean | cosine min | cosine p05 | head cos min | agreement all / margin >= 0.10 | 128 tok b1 | 128 tok b8 | 1024 tok b1 | 2048 tok b1 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16_all | | | | | | | | | | infeasible |
| accuracy | 0.99910 | 0.99596 | 0.99756 | 0.99434 | 95.5 % / 98.9 % | 57.6 ms | 162.0 ms | 170.5 ms | 322.8 ms | pass |
| bfp8_attn | 0.99902 | 0.99384 | 0.99749 | 0.99286 | 94.5 % / 97.3 % | 54.0 ms | 142.4 ms | 148.1 ms | 286.2 ms | fail (agreement) |
| bfp8_attn_hifi2 | 0.99891 | 0.99310 | 0.99697 | 0.99297 | 91.5 % / 95.7 % | 53.9 ms | 142.1 ms | 147.2 ms | 277.0 ms | fail (agreement) |
| bfp8_lofi_mlp | 0.99902 | 0.99564 | 0.99753 | 0.99110 | 94.0 % / 95.7 % | 54.7 ms | 123.6 ms | 130.1 ms | 250.4 ms | fail (agreement) |
| performance | 0.98744 | 0.92390 | 0.95782 | 0.91248 | 79.0 % / 82.4 % | 48.9 ms | 119.9 ms | 125.1 ms | 237.2 ms | fail (all gates) |

Agreement columns: Typed Decisions argmax agreement with the fp32 reference over the 200 subset decisions, and over
the 188 decisions whose reference top-2 margin is at least 0.10 (`agreement_<policy>.json`). `bf16_all` cannot run:
the stock prefill MLP program config sizes its circular buffers for bfp8 weights, and with bf16 w1/w3 the first
1024-token matmul asks for 2.01 MB of L1 per core against the 1.5 MB maximum (`infeasible_bf16_all.json`,
exact error text and log path). Making it fit needs a different block split in `model_config.matmul_config`, a
framework change outside this port.

Plots: `cosine_mean_perf_pareto.png`, `cosine_min_perf_pareto.png` (selected point in red, dotted gate line).

## Selection

`accuracy` (stock policy: bf16 attention weights and KV, HiFi4 attention linears and SDPA, bfp8 MLP with HiFi2).
It is the only candidate above 98 percent decision agreement on confident decisions (98.9 percent). The 2 of 188
confident disagreements have reference margins of 0.13 and 0.32 and total-variation distances of 0.24 and 0.28, so
on those two the port's probabilities differ materially from the fp32 reference, not only the argmax; the other 7 of
the 9 disagreements over all 200 decisions have reference margins below 0.09. Accuracy against the gold labels on
the subset: 0.375 (port) vs 0.370 (reference). It costs 6.4 percent at 128
tokens and 13 percent at 1024 and 2048 tokens against the stock-default `bfp8_attn` policy, which stays available as
the `p150-fast` serve profile (97.3 percent agreement). An earlier version of this sweep, before the agreement gate
was wired in, selected `bfp8_attn`; review C flagged that the plan's decision-agreement gate had been deferred, and
the rerun with the gate reversed the choice. `bfp8_lofi_mlp` is the interesting runner-up: LoFi on the bfp8 MLP
keeps the vector cosine (min 0.9956) and is 24 percent faster at 1024 tokens, but its head-projection minimum and
decision agreement drop to the `bfp8_attn_hifi2` level, so it fails the same gate. The `performance` policy (bfp4
w1/w3) is rejected on real-weight, real-text evidence: 0.924 minimum cosine, 0.912 head-projection minimum, 79
percent agreement.

## Decision-level cross-check

The agreement numbers above are computed from the sweep's own saved vectors (`fidelity_<policy>_tt_single.npy`)
through the published heads. The same comparison against the served package (400-case benchmark run, 40-case
reference subset) is in `../release/RUN_NOTES.md`.

## Files

`fidelity_<policy>.json` and `bench_<policy>.json` (accuracy's live in `../full_model/` and
`../optimized_full_model/`), `fidelity_*_tt_single.npy` / `_tt_batched.npy` raw vectors (not committed),
`sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, the two PNGs.
