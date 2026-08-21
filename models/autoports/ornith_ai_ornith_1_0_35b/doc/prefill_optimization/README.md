# Ornith-1.0-35B prefill-throughput optimization

This stage promotes `C25-prefill-sdpa-qk128` as the model's default precision policy and records the
separate, opt-in gathered-MoE serving result. C25 preserves every arithmetic, activation, KV-cache,
and sampling choice from the selected C06 datatype policy. Its only default-policy change is
`prefill.sdpa_q_k_chunk: 256 -> 128`.

Gathered MoE is enabled only by `ORNITH_MOE_GATHER=1`. The combined sweep below therefore measures
the deployable C25 plus gathered-MoE variant, not the environment-free default by itself. Each cell
is a single exact-length request wave with `ignore_eos`; the two 128-token cells remain on the sparse
MoE path because they are below the 1,024-token gather threshold.

## Acceptance evidence

- C25 full-model accuracy passes both readiness paths: prefill top-1/top-5/top-100 is
  `0.92/1.00/1.00`; teacher-forcing accuracy is `0.93/1.00/1.00`.
- A C25-only 128K/512 batch-1 vLLM A/B reduces TTFT from 71.760s to 66.511s (-7.32%) and E2EL from
  86.306s to 81.071s (-6.06%). Median ITL is effectively flat, as expected for a prefill-only
  policy change.
- The combined C25 plus gathered-MoE 8K correctness gate reaches PCC `0.999159520` against HF for
  prefill and `0.999307333` for decode. C25 versus the prior selected policy reaches prefill PCC
  `0.999994355`.
- Focused gathered-MoE tests cover BF8 collective input, batch-8 geometry, mixed gathered/sparse
  tails, expert-63 cache address replacement, and cache-hit validation failures.
- The final vLLM sweep completes all 10 requested cells with zero failed requests. Independent
  validation confirms exact request lengths, concurrency, token totals, median aliases, policy,
  capability state, and immutable run inputs.

## Final vLLM latency sweep

Hardware is four Blackhole `p300c` chips in a `(1,4)` mesh. Concurrency 1 and 8 use separately built
servers with matching `max_num_seqs`. `tok/s/u = 1000 / median ITL`; aggregate throughput is
`concurrency * tok/s/u`; E2EL is median request latency. Parentheses show percent delta versus the
frozen pre-optimization baseline.

| concurrency | ISL | OSL | tok/s/u | tok/s agg | E2EL |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 43.14 (+0.27%) | 43.14 (+0.27%) | 3.11s (-0.12%) |
| 8 | 128 | 128 | 31.83 (-0.13%) | 254.66 (-0.13%) | 5.25s (+0.83%) |
| 1 | 16384 | 512 | 44.39 (+6.54%) | 44.39 (+6.54%) | 17.85s (-8.13%) |
| 8 | 16384 | 512 | 32.27 (+8.79%) | 258.15 (+8.79%) | 67.22s (-13.47%) |
| 1 | 32768 | 512 | 43.18 (+6.84%) | 43.18 (+6.84%) | 24.53s (-10.86%) |
| 8 | 32768 | 512 | 30.29 (+7.48%) | 242.31 (+7.48%) | 120.62s (-15.34%) |
| 1 | 65536 | 512 | 40.86 (+6.09%) | 40.86 (+6.09%) | 38.75s (-14.06%) |
| 8 | 65536 | 512 | 26.93 (+7.22%) | 215.46 (+7.22%) | 234.87s (-17.32%) |
| 1 | 131072 | 512 | 36.92 (+5.44%) | 36.92 (+5.44%) | 70.62s (-18.17%) |
| 8 | 131072 | 512 | 22.03 (+5.23%) | 176.26 (+5.23%) | 493.11s (-19.17%) |

At concurrency 8 and 128K ISL, E2EL falls from 610.08s to 493.11s, saving 116.97s (-19.17%). The
final first token advances by 116.51s while the request wave shrinks by 117.13s, so the improvement
is overwhelmingly faster serialized prefill/queue drain. The apparent decode-rate change is a
secondary run-level effect and is not attributed causally to the prefill-only optimizations.

The short 128-token cells are controls. They do not invoke gathered MoE and are statistically flat.
Each cell is one request wave, so the table is acceptance evidence rather than a distributional
benchmark. Repeat waves should be used for production confidence intervals.

## Provenance

- Final run ID: `combined-c25-qk128-moe-gather-20260821T183746Z`
- C25 candidate: `../datatype_sweep/candidates/C25-prefill-sdpa-qk128.json`
- C25 full-model run: `../datatype_sweep/runs/C25-prefill-sdpa-qk128.json`
- `RUN_CONFIG.json` SHA-256: `c1d0b5e181fe1b83b2008494635b76affc4dbe1bba24551a81e7c1b89ecd1079`
- Raw 10-JSON set SHA-256: `be804048480d39a29316c5ad18db6f0de8a9e5c400425b6487a57d67b265157b`
- Final `RESULTS.md` SHA-256: `a732039e26ecb9a7bf85d73480f723626c5940f80acaeca375d19f1c000030ba`

The serving capability archives attest that C25 q/k=128 was built and gathered MoE was selected,
loaded, ready, and enabled for all 40 layers at both `max_num_seqs=1` and 8. They attest capability,
not per-call invocation counts.
