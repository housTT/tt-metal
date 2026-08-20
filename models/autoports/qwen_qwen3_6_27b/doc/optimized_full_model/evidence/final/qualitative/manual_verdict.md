# Qualitative manual verdict

Verdict: pass.

The six prompts use the checkpoint chat template with an assistant generation
prompt and deterministic greedy generation. HF and TT each generate 64 tokens.

- Every TT completion is coherent and specific to its prompt.
- There is no wrong-language drift, replacement character, control token,
  prompt echo, or cross-request leakage.
- The translation prompt remains English while planning the requested French
  translation, matching the HF base-model control at this truncation length.
- The story, thermodynamics, machine-learning, and Fibonacci prompts retain the
  expected topic and structure.
- HF/TT matching prefixes range from 11 to 63 tokens. Divergence remains a
  plausible continuation of the same reasoning, not a topic or language break.
- One story continuation repeats the word "thinking" once; its maximum
  identical-token run is two and it is neither persistent nor degenerate.
- The automatic report passes all six prompts. Maximum repeated-trigram
  fraction is 0.1129, maximum identical-token run is two, and no completion
  leaks a control/replacement token.

Qwen3.6-27B is a base/reasoning checkpoint whose visible completion begins with
planning. At 64 tokens, both HF and TT controls are normally truncated before a
user-facing final answer. This is a limitation of the evidence length, not a TT
quality regression; TT tracks the HF control behavior closely.
