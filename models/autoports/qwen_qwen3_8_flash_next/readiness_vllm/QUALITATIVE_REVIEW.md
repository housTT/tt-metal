# Qwen3.8-Flash-Next vLLM qualitative review

## Scope and provenance

This review independently reads every current output in both serving artifacts:

- `qualitative_tt_chat.json`: 3 prompt-correct chat-template responses, SHA-256
  `b813290555652ab02cd4c804e274ce50707190eef64561cff99e9a477c5a2b8c`;
- `vllm_qualitative_outputs.json`: 6 raw prompts with greedy and sampled
  continuations (12 outputs), SHA-256
  `182008f62c383f6291998f4da822fcece24f872ac4019e6b4fa36cacb6de48de`.

The primary control is
`doc/full_model/qualitative_shared_suite_final.json`, SHA-256
`9e8eaa14fb2ea31f5107cfea2110dfdf06928312abbe5dd2ae8ca48420d59487`.

## Prompt format and usage verification

The exact local checkpoint revision is
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. Its `Qwen2Tokenizer` has a
non-empty chat template with SHA-256
`c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`,
so chat is the required quality mode. For all three primary cases, an
independent local re-render with
`tokenizer.apply_chat_template(add_generation_prompt=True)` exactly matched
both the stored rendered string and every stored prompt token ID. Those IDs
were sent directly through `/v1/completions`, which avoids a second template
application. Generation was temperature 0, no logprobs, at most 256 tokens,
through the canonical traced on-device exact global-argmax path.

The API usage agrees exactly with every prompt length and total:

| Case | Prompt tokens | Visible response IDs | API completion usage | API total | Finish |
| --- | ---: | ---: | ---: | ---: | --- |
| explanation | 76 | 154 | 155 | 231 | stop |
| coding | 89 | 147 | 148 | 237 | stop |
| summarization | 94 | 81 | 82 | 176 | stop |

The one-token difference between visible response IDs and API completion
usage is the terminal stop token, which is consumed but absent from decoded
text. All three responses include the checkpoint's coherent reasoning section
because the exact template opens `<think>`; their final-answer sections are
the portions judged for requested brevity.

## Primary prompt-correct judgments

| Case | Coherence and topic | Repetition and gibberish | Language and contamination | Verdict |
| --- | --- | --- | --- | --- |
| Twelve-year-old sky explanation | Correctly explains preferential blue scattering and uses one flashlight/dust scattering analogy; the final answer is concise and complete. | No mechanical repetition, loop, doubled subword, or gibberish. | English as requested; no prompt echo, foreign-language drift, or content from another request. | Pass |
| Python order-preserving deduplication | Gives a correct `dict.fromkeys` implementation and the correct example output `[3, 1, 2, 4]`. | No mechanical repetition, loop, doubled subword, or gibberish. Repeated code identifiers account for benign repeated four-grams. | English/Python as requested; no prompt echo or request contamination. | Pass |
| One-sentence library summary | One grammatical sentence preserves the successful trial, increased attendance, volunteer support, permanent city funding, and unchanged weekday service. | No mechanical repetition, loop, doubled subword, or gibberish. | English as requested; no prompt echo, language drift, or request contamination. | Pass |

All three return HTTP 200, stop naturally, have a Latin-letter fraction of
1.0, and are marked non-degenerate in the artifact. The explanation and code
responses share long sound prefixes with the HF control; the summary diverges
early but is independently correct. No control discrepancy indicates a TT
serving defect.

## Secondary raw-continuation judgments

The runner's six plain prompts are deliberately untemplated and therefore are
not pass/fail quality evidence for this chat-template checkpoint. Each prompt
was sent twice through `/v1/completions`: greedy with temperature 0 and sampled
with temperature 0.7/top-p 0.9, both with a 256-token maximum. The runner
artifact does not retain API usage or finish reasons; token counts below are
offline counts of the saved visible text with the exact checkpoint tokenizer,
not reconstructed API usage.

| Prompt / output | Visible tokens | Coherence and topic | Repetition / gibberish | Language / request contamination |
| --- | ---: | --- | --- | --- |
| Haiku, greedy | 196 | Coherent reasoning followed by a valid topical 5-7-5 haiku. | None. | English; no cross-request content. Balanced visible `<think>` markup is a raw-format limitation. |
| Haiku, sampled | 21 | Concise, topical 5-7-5 haiku. | None. | English; no cross-request content. It starts with an unmatched `</think>` marker, a raw-format artifact. |
| Supervised vs. unsupervised, greedy | 256 | Correct supervised explanation and starts the unsupervised contrast, but the answer is cut before completing it. | None. | English; no cross-request content. Visible reasoning and length-cap truncation. |
| Supervised vs. unsupervised, sampled | 256 | Correct supervised explanation, then stops immediately after introducing unsupervised learning; incomplete at the cap. | No mechanical loop or gibberish. | English; no cross-request content. |
| Inventor story, greedy | 256 | Opens the requested story coherently, then spends the remaining budget planning it and never supplies a complete narrative. | None. | English; no other request appears. Visible reasoning and cap truncation. |
| Inventor story, sampled | 256 | Coherent plan, but the final section merely echoes the supplied story opening and ends at `discovered`; it does not deliver the story. | None. | English; explicit prompt echo, but no cross-request leakage. Visible reasoning and cap truncation. |
| Thermodynamics, greedy | 256 | Correctly states the three requested laws, then continues into invented heat/temperature and entropy Q&A and is cut off. | No loop or gibberish. | English; learned follow-up-Q&A contamination, not content from another live request. |
| Thermodynamics, sampled | 256 | Correctly states the three laws, then invents entropy/equilibrium Q&A and is cut off. | No loop or gibberish. | English; learned follow-up-Q&A contamination, not cross-request leakage. |
| French translation, greedy | 191 | Produces the correct polite French translation and a coherent register note. | None. | Target French is correct; English reasoning/note reflects raw formatting rather than wrong-language drift. |
| French translation, sampled | 256 | Produces the same correct translation, then its English explanatory note is cut at the cap. | None. | Correct French plus source-language reasoning; no unrelated request content. |
| Fibonacci, greedy | 111 | Valid iterative function for the nth Fibonacci number, a reasonable interpretation of the prompt. | None. | English/Python; no contamination. |
| Fibonacci, sampled | 153 | Valid iterative function returning the first `n` Fibonacci numbers. | None. | English/Python; no contamination. Empty visible `<think>` markup is a raw-format artifact. |

Offline visible token counts by prompt are `8/196/21`, `15/256/256`,
`23/256/256`, `9/256/256`, `14/191/256`, and `10/111/153` for
prompt/greedy/sampled respectively. Thus 7 of 12 continuations reach the
visible 256-token cap, and 8 of 12 expose `<think>`/`</think>` markup. The
mechanical-degeneracy checker reports no degenerate output across all 12:
there is no doubled-token failure, repetition loop, or gibberish. There is
also no wrong-language drift or evidence that one live request leaked into
another. The prompt echo and learned follow-up Q&A noted above are genuine raw
continuation behavior and must not be presented as clean chat responses.

## Verdict

**Pass for the required prompt-correct chat path: 3/3.** Every primary response
is coherent, correct, topical, non-repetitive, non-gibberish, in the requested
language, and free of prompt/request contamination. The secondary raw stress
is mechanically healthy (12/12) but is explicitly non-gating and carries the
documented formatting, truncation, prompt-echo, and learned-Q&A limitations.
