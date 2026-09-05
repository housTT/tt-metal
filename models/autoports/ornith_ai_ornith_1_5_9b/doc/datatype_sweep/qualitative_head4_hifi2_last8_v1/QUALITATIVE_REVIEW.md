# BFP4/HiFi2 head with BFP8 last layer: qualitative review

Verdict, 2026-09-05: passes the recorded bounded qualitative smoke. All seven
TT outputs are token-for-token identical to the qualified head4/LoFi last-layer
control, and were read against pinned HF outputs. No new concrete quality
regression was found. Parent selects the LoFi policy following timing repeats;
this HiFi2 policy also qualifies on quality. Review used light text/JSON reads
without TTNN imports, hardware access or production edits.

[Metadata](qualitative_prompt_format.json) records checkpoint
`ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, pinned local snapshot and
`Qwen2Tokenizer` using the original chat template with a user message and
generation prompt. Generation is greedy, with 128 tokens for each of the six
shared prompts and 100 for AIME. One reused generator uses traced prefill/decode
and context 2048.

Policy `head4_hifi2_last8` uses BFP4 head weights with HiFi2 compute and BFP8
attention/MLP weights only in layer 31. Other decoder projections retain
BFP4/LoFi; decode QKVG and KV remain BFP8. The recorded `repeat5_v2` control
passes AIME 92/100/100 in every sample with median 84.262 t/s/u. The corresponding
LoFi median is 84.371; these are teacher-forcing measurements, not token-out
results, and their timing ranges overlap.

| Prompt | Actual observation, identical to LoFi last-layer control |
| --- | --- |
| 0, haiku | Coherent 5/7/5 planning; no draft line or completed count mismatch appears before cutoff. |
| 1, learning types | Correctly distinguishes labeled supervised data from unlabeled unsupervised data. |
| 2, story | Coherent inventor-story planning. HF starts a story within the budget; prior TT also remains in planning. |
| 3, thermodynamics | Recognizes the zeroth-law naming issue, then starts the first-law heading. |
| 4, French | `"Bonjour" (good day) or "Salut" (informal hello)` and appropriate formal/informal question forms. |
| 5, Fibonacci | Coherent list of ordinary iterative, recursive and memoized approaches; no complete code to execute. |
| AIME | Uses walking time `4 - t/60` and distance/speed `9/s` consistently; no final answer is reached. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-request leakage was observed. All outputs remain reasoning prefixes,
so this is not completed-answer or longer-budget validation.

Light artifact checks verified all nine input/HF-control fields for all seven
prompts against both the pinned previous-stage HF artifacts and
`../qualitative_head4_lofi_last8_v1`. The TT token lists match LoFi exactly.
Revision/template/suite hashes agree with the pinned controls. Exact command,
source provenance and exit0 are recorded in
`../logs/qualitative_head4_hifi2_last8_v1.provenance.json`.

This review applies to that recorded explicit-config run. Parent's final
selected-artifact loader change and fresh default suite require separate
evidence; historical source comparisons do not assert final live-file identity.
See [the AutoFix ledger](../AUTOFIX_french.md).
