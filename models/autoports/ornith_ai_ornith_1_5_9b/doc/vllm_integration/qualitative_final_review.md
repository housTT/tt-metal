# Final serving qualitative reading

All12 actual texts in `../../readiness_vllm/vllm_qualitative_outputs.json`
were read after the all-layer full_b32_startup_final run. The JSON companion records
its byte hash and comparison against the archived earlier shared suite. All
six greedy texts are exactly unchanged, so the completed standalone/HF haiku
controls apply directly. The original chat template, messages and token IDs
remain those in qualitative_prompt_format.json. Profiles are greedy0 and
sampled0.7/top_p0.9, each with256 generated tokens maximum.

| Prompt | Greedy verdict | Sampled verdict |
| --- | --- | --- |
| Machine-learning haiku | Repeats the controlled selected-TT6-syllable first-line miscount; this is not a correct haiku verdict. | Initially miscounts a6/8/7 draft as5/7/5, then notices its first line has6 syllables; cutoff before a final answer. |
| Supervised vs unsupervised | Correct labeled/unlabeled distinction and apt examples; coherent. | Correct labeled/unlabeled distinction and teacher/animal/sorting analogies; awkward wording “show a dog a picture” in the animal example; ends during reasoning. |
| Inventor story | Coherent relevant story planning; no completed narrative in budget. | Coherent relevant planning and discovery ideas; cutoff before the story itself. |
| Thermodynamics | Correct first/second-law summaries and reasonable general third-law statement; on-topic. | Relevant zeroth/first/second/third-law summaries, then begins its final answer before cutoff. |
| French translation | Correct final French translation after English thinking; recognizes that formality is unspecified. | Correct final French translation after English thinking; its reasoning incorrectly assumes English “you” specifies singular/informal address. |
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
