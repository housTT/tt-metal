# Final serving qualitative reading

All12 actual texts in `full_b32_release_qualitative.json`
were read after the all-layer full_b32_release run. The JSON companion records
its byte hash and comparison against the archived earlier shared suite. All
six greedy texts are exactly unchanged, so the completed standalone/HF haiku
controls apply directly. The original chat template, messages and token IDs
remain those in qualitative_prompt_format.json. Profiles are greedy0 and
sampled0.7/top_p0.9, each with256 generated tokens maximum.

| Prompt | Greedy verdict | Sampled verdict |
| --- | --- | --- |
| Machine-learning haiku | Repeats the controlled selected-TT6-syllable first-line miscount; this is not a correct haiku verdict. | Drafts a correct first line, miscounts words in the proposed second line, then corrects individual counts; reasoning ends at cutoff, so no completed-haiku claim. |
| Supervised vs unsupervised | Correct labeled/unlabeled distinction and apt examples; coherent. | Correct distinction and teaching/sorting analogies; coherent, ends during reasoning. |
| Inventor story | Coherent relevant story planning; no completed narrative in budget. | Coherent relevant brainstorming and a brass-device concept; cutoff before completed story. |
| Thermodynamics | Correct first/second-law summaries and reasonable general third-law statement; on-topic. | Correct first/second-law summaries including sign convention; third-law description truncated. |
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
