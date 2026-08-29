# Qualitative verdict

Verdict: pass.

The exact HF AIME control and TT free-running completion were read directly.
Both are coherent English analyses, identify the walking-time equation, and
remain on the requested problem.  The first token divergence is at zero-based
index 18 (the 19th token); 51 of 100 positions match.  The divergence is a
reasonable wording change, not an early semantic collapse.  Neither completion
shows repetition, wrong-language drift, malformed text, or stale-token behavior.
Both stop at the requested 100-token evidence limit before completing the proof.

The shared six-prompt suite was also read directly:

- haiku: on-topic syllable-count reasoning, truncated at 128 tokens;
- supervised/unsupervised learning: clear labeled-versus-unlabeled setup;
- story continuation: coherent creative setup, truncated at 128 tokens;
- thermodynamics: on-topic discussion of the conventional law numbering;
- French translation: correct formal translation and clean `<|return|>` EOS;
- Fibonacci: coherent Python-answer setup, truncated at 128 tokens.

The exact-checkpoint HF controls for those same six prompt-token sequences were
read side by side with the TT outputs.  All pairs stay in English where
expected, follow the same task, and enter compatible answer structures.  Their
common-prefix lengths are 38, 3, 3, 17, 3, and 26 tokens respectively; after
divergence, the differences are ordinary greedy wording/organization choices.
Neither side shows semantic collapse, phrase looping, stale-token behavior, or
wrong-language drift.  The TT French result is complete and correct at its
81-token `<|return|>` stop, while the HF control continues explaining both
formal and informal variants until the fixed 128-token evidence cutoff.

Control provenance is exact and replayable: checkpoint revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, Transformers 5.12.1, one greedy
six-row batch, explicit CPU/NVMe dispatch, and the command embedded in
`qualitative_hf_tt_comparison.json`.  HF artifact SHA-256 is
`6cb7583a5d129c30ee81e96189198fcafd366ee42ed1070498e1fd3c7cace3f2`;
comparison SHA-256 is
`67b5c8cf65c666fed837ff473eb100d3c3191618bcb8e21fdb0ea471b247550f`.

The first suite run deliberately forced all prompts to 128 tokens and exposed a
post-EOS repeated-control-token tail on the translation.  AutoDebug proved that
the harness had set `stop_on_eos=False`; the trace/token-feedback evidence was
normal.  The corrected run honors the checkpoint generation stop set and the
translation ends at 81 tokens.  The readiness degeneracy checker then reports no
findings across the six prompts and AIME output: replacement-character fraction
is zero, adjacent duplication is below threshold, and no critical phrase loop is
present.
