# Qualitative review

Verdict: pass with a fixed-generation-length limitation. The evidence uses the
checkpoint tokenizer's chat template with `add_generation_prompt=True`, the
recorded `xhigh` system instruction, greedy HF controls, and traced on-device
TT greedy sampling. This is an instruction-tuned chat model; the prompts and
comparison mode are appropriate.

The AIME24 free-running sample is a valid shifted-left autoregressive
comparison: both completions start after the same 201-token chat prompt and
each model consumes its own prior token. The first divergence at generated
token 5 is expected free-running behavior, not an alignment error. Both texts
remain coherent English math reasoning, preserve the problem facts, and are
mechanically non-degenerate; TT has no adjacent repeats, a 0.06 dominant-token
fraction, and no wrong-language behavior.

The shared suite was reviewed prompt by prompt:

- `explanation`: TT correctly identifies Rayleigh scattering and proposes an
  age-appropriate analogy. It remains in visible reasoning at token 128.
- `coding`: TT correctly frames an order-preserving set/list solution and the
  possible hashability tradeoff. It remains in visible reasoning at token 128.
- `summarization`: TT emits a correct one-sentence summary retaining increased
  attendance, volunteer staffing, permanent city funding, and preservation of
  weekday service.

All TT samples are coherent, English, and mechanically non-degenerate. The HF
controls for explanation and coding are also cut off by the same 128-token
budget, so the absence of a final answer in those two samples is a generation
budget limitation rather than TT-specific degeneration. No prompt-format fix,
language override, or host-sampling workaround is warranted. For a
customer-facing qualitative demo, raise the generation budget so visible
reasoning can close before evaluating final-answer helpfulness.
