# Shared qualitative-suite verdict

- Checkpoint: `Qwen/Qwen3.6-27B`, revision
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Prompt mode: the checkpoint tokenizer's chat template with
  `add_generation_prompt=True`.
- Generation: greedy, 64 tokens, six prompts from
  `models/common/readiness_check/vllm_prompts.txt`.
- Automated degeneracy verdict: pass. Matching HF/TT prefixes are 11, 55,
  13, 43, 19, and 63 tokens. Maximum identical-token run is two; no control
  token or Unicode-replacement leakage occurs.

Manual review: pass. Every TT completion is grammatical, relevant to its own
request, and in the expected language. `shared-0` discusses machine-learning
concepts for the requested haiku; `shared-1` contrasts supervised and
unsupervised learning; `shared-2` continues the inventor-story setup;
`shared-3` discusses the laws of thermodynamics; `shared-4` reasons about the
requested French translation; and `shared-5` plans a Python Fibonacci
function. There is no prompt echo, cross-request leakage, wrong-language
drift, corrupt first token, or mechanical repetition.

Both HF and TT expose the checkpoint's verbose “thinking process” behavior.
In `shared-2`, both controls contain the same adjacent word duplication
(`thinking thinking`), so it is checkpoint/control behavior rather than a TT
regression. The completions end mid-reasoning only because the inspection
budget is exactly 64 tokens.
