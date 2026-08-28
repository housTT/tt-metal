# Shared qualitative suite review

The fresh reference and TT run use the exact checkpoint revision,
`Qwen2Tokenizer`, checkpoint chat template, and three fixed prompts recorded in
`qualitative_prompt_format.json`. Both sides use greedy generation for 128
tokens. Raw messages, rendered prompts, prompt tokens, HF/TT tokens,
completions, divergence, degeneracy counters, trace metrics, and runtime audits
are retained in `qualitative_shared_suite.json` and
`qualitative_shared_suite_final.json`.

| Prompt | HF/TT matching prefix | Coherence and task | Repetition/language | Completion verdict |
| --- | ---: | --- | --- | --- |
| Explanation | 40 tokens | Both correctly reason about preferential blue-light scattering and attempt a child-friendly analogy. | No adjacent repeats, language drift, topic drift, or mechanical loop. | Bounded pass: HF reaches the answer and TT remains in a correct formulation, but both are truncated by the 128-token ceiling. |
| Coding | 42 tokens | Both choose a first-seen `set`/result-list implementation; TT begins a valid function and HF reaches the loop body. | No adjacent repeats, language drift, topic drift, or mechanical loop. | Bounded pass: neither reaches the requested example before the 128-token ceiling. |
| Summarization | 0 tokens | TT independently produces a correct one-sentence summary covering trial success, attendance, volunteer staffing, permanent city funding, and preserved weekday service. | English throughout; automated review is non-degenerate. The nine adjacent token repeats are formatting/special-token pairs, not a text loop. | Pass: TT completes the requested answer. HF also completes it, then its fixed-length control continues after EOS; post-EOS HF text is excluded from the semantic verdict. |

Overall verdict: **bounded pass**. TT is coherent, non-repetitive, English, and
on-topic across all prompts. There is no qualitative evidence of mechanical
collapse. The checkpoint chat template makes its xhigh reasoning visible, so
two short tasks exhaust the skill's maximum 128-token qualitative window
before a final answer is complete. This is a response-budget/template
limitation shared with the HF control, not TT language drift. The separate
100-token AIME24 run likewise stays coherent and on-topic after divergence at
token 5.

Every prompt's optimized runtime audit reports false for host expert/PLE
projection, activation round trips, host KV/recurrence, host sampling or
argmax, token-feedback reconstruction, per-token position refresh, and
unchanged page-table refresh.
