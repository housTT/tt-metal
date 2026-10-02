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

## Results (p150, one chip, 2048-token context, batch 1 / 4 / 8 traces)

| policy | cosine mean | cosine min | cosine p05 | head cos min | agreement all / margin >= 0.10 | 128 tok b1 | 128 tok b8 | 512 tok b8 | 1024 tok b1 | 2048 tok b1 | workload sum | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16_all | | | | | | | | | | | | infeasible |
| accuracy (stock configs; `p150-accuracy`) | 0.99910 | 0.99596 | 0.99756 | 0.99434 | 95.5 % / 98.9 % | 57.8 ms | 162.1 ms | 624.5 ms | 170.4 ms | 321.7 ms | 1336.5 ms | pass |
| accuracy_lofi_mlp (stock configs) | 0.99914 | 0.99584 | 0.99787 | 0.99414 | 96.0 % / 98.9 % | 58.5 ms | 144.0 ms | 554.6 ms | 152.0 ms | 285.5 ms | 1194.6 ms | pass |
| accuracy_lofi_mlp + program-config overrides (shipped default `p150`) | 0.99916 | 0.99599 | 0.99787 | 0.99310 | 97.5 % / 98.9 % | 53.1 ms | 135.7 ms | 510.9 ms | 143.3 ms | 266.0 ms | 1109.0 ms | pass, selected |
| accuracy + program-config overrides (not shipped) | 0.99897 | 0.99426 | 0.99679 | 0.99237 | 94.0 % / 97.3 % | 52.5 ms | 154.4 ms | 583.3 ms | 162.4 ms | 306.8 ms | 1259.4 ms | fail (agreement) |
| bfp8_attn (stock configs; `p150-fast`) | 0.99902 | 0.99384 | 0.99749 | 0.99286 | 94.5 % / 97.3 % | 54.0 ms | 142.4 ms |  | 148.1 ms | 286.2 ms |  | fail (agreement) |
| bfp8_attn_hifi2 | 0.99891 | 0.99310 | 0.99697 | 0.99297 | 91.5 % / 95.7 % | 53.9 ms | 142.1 ms |  | 147.2 ms | 277.0 ms |  | fail (agreement) |
| bfp8_lofi_mlp | 0.99902 | 0.99564 | 0.99753 | 0.99110 | 94.0 % / 95.7 % | 54.7 ms | 123.6 ms |  | 130.1 ms | 250.4 ms |  | fail (agreement) |
| performance | 0.98744 | 0.92390 | 0.95782 | 0.91248 | 79.0 % / 82.4 % | 48.6 ms | 120.3 ms | 467.9 ms | 127.0 ms | 241.8 ms | 1005.6 ms | fail (all gates) |

The 512-token batch-8 and workload-sum columns exist only for benches taken on the five-bucket encoder (the nine-variant
benches pad 512 tokens to 1024); the workload sum is 128 b1 + 128 b8 + 512 b8 + 1024 b1 + 2048 b1 (plan amendment
2026 Oct 2 00:55). Rows marked `_pc` ran with the stage 3 program-config overrides and the block-sharded norm
(`../optimized_full_model/README.md`).

Provenance: the fidelity and agreement columns of every runnable policy were regenerated on the final encoder code
(host-side norm, two-phase warmup) on 2026 Oct 1 between 23:10 and 23:17 UTC (`/home/hous/dev/clm-v0.1-8B/logs/fidelity_<policy>_final.log`).
The latency columns come from `bench_<policy>.json`: accuracy 22:17, bfp8_attn 22:15, bfp8_attn_hifi2 22:21 (the
final forward path, nine variants), bfp8_lofi_mlp 23:17, and performance re-run on 2026 Oct 2 on the final code
(`/home/hous/dev/clm-v0.1-8B/logs/bench_performance_final.log`; the earlier 21:58 bench was taken on the first
encoder generation with the device-side norm tail). Agreement columns: Typed Decisions argmax agreement with the fp32
reference over the 200 subset decisions, and over the 188 decisions whose reference top-2 margin is at least 0.10
(`agreement_<policy>.json`). The gate is evaluated on the single-text vectors (`fidelity_<policy>_tt_single.npy`),
the serving path for a request with one new text; on the batched vectors the confident-decision count moves by up to
five for one policy (batch-variant reduction order, tt-metal 47238), and one decision is 0.53 points of the 188, so
the gate has a resolution of about half a point. `bf16_all` cannot run: the stock prefill MLP program config sizes
its circular buffers for bfp8 weights, and with bf16 w1/w3 the first prepared variant (128 tokens, batch 1, the 8x4
core grid) asks for 2.01 MB of L1 per core against the 1.5 MB maximum (`infeasible_bf16_all.json`, exact error text
and log path). Making it fit needs a different block split in the MLP program config; the port overrides program
configs only where the stage 3 experiment found a measured gain, and this candidate is slower than the selected one
in every row where it could be compared, so it was not pursued. `fidelity_bfp8_attn_twophase_limit96.json` (96
texts, 22:23 UTC) is a smoke run of the two-phase warmup, not a sweep input.

Plots: `cosine_mean_perf_pareto.png`, `cosine_min_perf_pareto.png` (selected point in red, dotted gate line).

## Selection

`accuracy_lofi_mlp` with the program-config overrides (bf16 attention weights and KV, HiFi4 attention linears and
SDPA, bfp8 MLP weights with LoFi math; QKV block shape at 128 tokens, 11x10 MinimalMatmul grid above 128 tokens,
block-sharded RMSNorm up to 512 rows). It passes every gate (cosine min 0.99599, head-projection min 0.9931, 98.9
percent confident-decision agreement, 97.5 percent plain) and has the lowest served-workload latency sum of the
passing rows: 1,109 ms against 1,195 ms for the same policy on the stock configs and 1,337 ms for the stock
`accuracy` policy (the previous default; 17 percent slower). History of this choice: the first sweep (vector gates
only) selected `bfp8_attn`; review C added the agreement gate, which reversed it to `accuracy`; review C2 asked for
the LoFi-MLP lever under the accuracy attention policy, which passes and is faster everywhere except 128-token
batch 1, and the selection rule was widened from that one cell to the served-workload sum; review A2's program-config
experiment then added the overrides, which are neutral for this policy's fidelity and remove another 7 percent.
The stock `accuracy` policy stays available as the `p150-accuracy` profile and the stock bfp8 policy as `p150-fast`
(97.3 percent confident-decision agreement in this sweep, 95.7 percent measured from a served package; it fails the
gate and the card says so). `bfp8_lofi_mlp` keeps the vector cosine (min 0.9956) but its head-projection minimum
and decision agreement fall to the `bfp8_attn_hifi2` level. The `performance` policy (bfp4 w1/w3) is rejected on
real-weight, real-text evidence: 0.924 minimum cosine, 0.912 head-projection minimum, 82 percent agreement.
The `accuracy` policy with the overrides is the one combination where the overrides hurt (98.9 to 97.3 percent);
it is recorded and not shipped.

## Decision-level cross-check

The agreement numbers above are computed from the sweep's own saved vectors (`fidelity_<policy>_tt_single.npy`)
through the published heads. The same comparison against the served package (400-case benchmark run, 40-case
reference subset) is in `../release/RUN_NOTES.md`.

## Files

`fidelity_<policy>.json` and `bench_<policy>.json` (accuracy's live in `../full_model/` and
`../optimized_full_model/`), `fidelity_*_tt_single.npy` / `_tt_batched.npy` raw vectors (not committed),
`sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, the two PNGs.
