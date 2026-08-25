# Ornith-1.0-35B post-optimization evaluation

Status: **functional pass with warnings**. This is a compact handoff artifact; `results.json` is canonical.

## Quality subset

| task | samples | current | historical context | delta | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| IFEval | 28/541 | 85.88 | 82.08 | +3.80 | four-metric mean; measured only |
| GPQA-Diamond | 10/198 | 20.00 ± 13.33 | 50.00 | -30.00 | measured only; 8/10 parsed final responses were empty |

These are fixed CI subsets (IFEval doc IDs 0–27; GPQA doc IDs 0–9), not full-benchmark scores. The historical run used the same IDs but different sampling and max-seqs settings, so it is context rather than a graded A/B.

Eight GPQA responses exhausted their reasoning budget without parsed final content; the 20% score therefore diagnoses the selected batch-8 serving profile, not just subject knowledge.

## Correctness and serving

- API gate: **7/7 passed** (health, model discovery, short greedy repeatability, reasoning parser, tool parser, streaming, concurrency 8).
- Device gate: **6 tests passed**; the full 40-layer B1/B2/B4 proof recorded 280 native calls/subchunks, 120 layer calls, and zero fallbacks.
- Host/static gates: **186/186 passed**.
- Production source/evidence contract: **pass**.

## Selected-profile performance

The finalized 10-cell sweep passed strict provenance. At concurrency 8, 131,072 input tokens and 512 output tokens: median TTFT **195.53s**, median E2EL **285.27s**, aggregate decode **167.26 tok/s**, zero failed requests. E2EL is 42.15% below the gathered-MoE baseline and 53.24% below the original sparse path.

## Provenance and limits

- weights: `ornith-ai/Ornith-1.0-35B` @ `5df2ed3f675c7beaa490328cc70bb573b65fb660`
- tt-metal runtime: `824072e81e99af0cacb36adb6a33e271cd66c47f`; evaluation parent: `6c9442c8d6ce1ba2c42c13c826b29ce0cabdd531`
- vLLM: `a887998646dc4e6f192bce8d485bf89f4596ca2f`
- hardware: four Blackhole p300c chips, `(1, 4)` ring (P300_X2)
- selected server: C25 Q/K=128, native MoE 2K span, max-seqs 8, four API frontends, 2.0s coalescing ceiling
- text-only port; no full IFEval/GPQA or GPU-reference result is claimed
- short thinking-off generations repeated byte-for-byte; longer thinking-on greedy generations did not, matching the carried-forward padded-decode reproducibility limitation
