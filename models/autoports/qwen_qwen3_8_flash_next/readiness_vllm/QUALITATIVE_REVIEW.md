# Qwen3.8-Flash-Next vLLM qualitative review

## Primary prompt-correct suite

The primary artifact is `qualitative_tt_chat.json`. It reruns the three shared
full-model prompts with the checkpoint's real Qwen chat format:

- tokenizer: `Qwen2Tokenizer`, checkpoint revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`;
- chat-template SHA-256:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`;
- rendering: `tokenizer.apply_chat_template(add_generation_prompt=True)`;
- transport: the exact rendered token IDs through `/v1/completions`, avoiding
  any server-side double templating;
- generation: greedy canonical traced device argmax, no logprobs, 256-token
  cap;
- HF control: `doc/full_model/qualitative_shared_suite_final.json` using the
  same rendered prompts and token IDs.

Every rendered prompt and exact prompt/output token stream is retained in the
artifact. Prompt lengths were 76, 89, and 94 tokens and matched vLLM usage.

| Case | Finish | Human verdict | Control comparison |
| --- | --- | --- | --- |
| Twelve-year-old sky explanation | Stop, 147 served tokens | Pass. The final answer correctly explains preferential blue scattering with one simple crayon analogy. It is concise, topical, coherent, and free of repetition, gibberish, language drift, or request contamination. | The served reasoning shares a 40-token prefix with HF, then gives a different but equally valid analogy. |
| Python order-preserving deduplication | Length, 256 served tokens | Limited pass. The visible reasoning contains a correct `deduplicate_preserving_order` implementation and the expected `[3, 1, 2]` example, but it over-deliberates hashable versus unhashable inputs and reaches the cap before emitting its final answer. The content remains coherent and uncontaminated; truncation is a real usability limitation. | It shares a 42-token prefix with HF. HF also chooses the set-and-list implementation, but its stored 128-token control is itself length-capped. |
| One-sentence library summary | Stop, 104 served tokens | Pass. The final sentence preserves the successful trial, higher attendance, volunteer support, permanent city funding, and no weekday cuts. No repetition, gibberish, drift, or contamination is present. | It does not share a token prefix with HF, but independently produces the same correct summary. Unlike the 128-token HF control, it stops cleanly without trailing conversation-marker autocomplete. |

The automated review also finds no adjacent-token loops or mechanical
degeneracy in any case; all three have a Latin-letter fraction of 1.0.

## Secondary raw-continuation stress

`vllm_qualitative_outputs.json` contains six untemplated `/v1/completions`
prompts, each served greedily and with top-p sampling. Because this checkpoint
has a non-empty chat template, those twelve outputs are **not** the primary
quality pass/fail evidence. They remain useful as a secondary raw-continuation
stress test of device sampling and request isolation.

All twelve raw continuations were manually read. They stay coherent and on
topic without gibberish, wrong-language drift, mechanical loops, or
cross-request contamination. Seven reach the configured 256-token cap. Learned
Q&A continuation and visible `<think>` text in that untemplated mode are not
used to excuse or judge primary chat quality.

## Verdict

**Pass with one documented length-cap limitation.** The exact prompt-correct
chat path produces two complete, correct responses and one coherent response
whose reasoning contains the correct code and example but is cut before its
final answer. This is acceptable serving-path coherence evidence, while the
coding truncation remains explicitly visible rather than being classified as a
fully successful user response.
