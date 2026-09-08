# Final serving qualitative reading

All12 actual texts in `full_b32_verified_qualitative.json`
were read after the all-layer full_b32_verified run. The JSON companion records
its byte hash and comparison against the archived earlier shared suite. All
six greedy texts are exactly unchanged, so the completed standalone/HF haiku
controls apply directly. The original chat template, messages and token IDs
remain those in qualitative_prompt_format.json. Profiles are greedy0 and
sampled0.7/top_p0.9, each with256 generated tokens maximum.

| Prompt | Greedy verdict | Sampled verdict |
| --- | --- | --- |
| Machine-learning haiku | Repeats the controlled selected-TT6-syllable first-line miscount; this is not a correct haiku verdict. | Drafts a valid5/7/5 poem, then labels the7-syllable refinement “Feeding data to the mind” as5; cutoff before a final answer, so no completed-haiku claim. |
| Supervised vs unsupervised | Correct labeled/unlabeled distinction and apt examples; coherent. | Correct distinction with driving/language/dog instruction and unfamiliar-city/room exploration examples; coherent, ends during reasoning. |
| Inventor story | Coherent relevant story planning; no completed narrative in budget. | Begins an engaging narrative about a brass compass pointing toward questions; cutoff before completed story. |
| Thermodynamics | Correct first/second-law summaries and reasonable general third-law statement; on-topic. | Relevant first/second-law summaries including sign convention; ends during second-law discussion, before the third law. |
| French translation | Correct final French translation after English thinking. | Correct final French translation after English thinking. |
| Fibonacci | Relevant algorithm discussion and start of answer; no complete executable function. | Correct sequence and relevant algorithm discussion; no complete function before cutoff. |

No mechanical repetition, doubled subwords, gibberish, wrong-language drift,
or request contamination was found. Repeated self-checking in the haiku is
reasoning, not a token-collapse loop. The long thinking prefix matches the
original HF template and controls; it consumes the finite request budget.
This is a serving coherence/regression assessment, not proof of complete
long-form task accuracy. The substantive greedy haiku error is reproduced
exactly through EOS in the selected standalone generator; the pinned BF16 HF
control answers correctly. See qualitative_extended_control_report.md for
that controlled quality limitation and the archived earlier twelve-text review.
