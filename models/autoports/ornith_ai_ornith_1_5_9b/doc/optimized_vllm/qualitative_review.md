# Final qualitative review

pass for serving regression/coherence; constrained task accuracy and finished answers are not established.

shared six chat prompts; maximum256 generated tokens each; greedy0 and sampled0.7/top_p0.9; max-num-seqs1; native262144; TP4; selected precision.

All twelve texts were read directly. Every greedy text is byte-for-byte equal to the previous integration snapshot. Sampled outputs are unseeded; exact sampled-text equality is not claimed. The current template and all six rendered token sequences match the pinned controls.

- Prompt0 (haiku): Greedy exactly repeats controlled selected-generator syllable error; sampled draft lines scan6/9/6 despite claimed5/7/5 and remain inside unfinished thinking. No new meter-accuracy claim. Control: ../vllm_integration/qualitative_extended_control_report.md; ../vllm_integration/haiku_standalone_serving_exact_comparison.json.
- Prompt1 (supervised): Both distinguish labels/answers from unlabeled pattern discovery correctly; sampled analogy truncates. Control: previous-stage identical greedy and pinned shared HF controls.
- Prompt2 (story): Both plan a coherent inventor story; token budget expires before narrative. Exact greedy control has the same behavior. Control: previous-stage identical greedy.
- Prompt3 (thermodynamics): Greedy gives energy conservation, entropy and absolute-zero summary; sampled organizes law numbering and begins the first law before truncating. Control: previous-stage identical greedy; shared HF controls.
- Prompt4 (French): Both finish a valid Bonjour, comment allez-vous aujourd'hui? translation, with expected reasoning delimiter. Control: previous-stage identical greedy.
- Prompt5 (Fibonacci): Both discuss appropriate approaches; greedy stops at code fence and sampled begins a function before truncation. No completed-code correctness claim. Control: previous-stage identical greedy; shared HF controls.

Haiku remains an accuracy limitation of the selected path: the predecessor extended HF control completed5/7/5 while the selected standalone TT generator and serving matched all390 token IDs and completed6/7/5. Current greedy text reproduces that control exactly. The sampled draft has its own counting mistakes before revising; it is not scored as a finished haiku. The current sampler exact seeded/eager controls and unchanged model math support a serving-regression pass, not a claim that precision preserves all constrained writing accuracy.

There is no mechanical repetition, cross-request contamination, wrong-language drift or gibberish. Reasoning delimiters follow the checkpoint template. The scope-specific degeneracy checker exited0; raw prose131/65/repeated requests also match prior token controls.
