# BFP4 head with BFP8 last layer: qualitative review

Verdict, 2026-09-05: passes the recorded bounded qualitative smoke. All six
shared prompts and AIME were read against matching pinned HF and prior TT
outputs. No new concrete quality regression was found. This is a qualifying
finalist; timing repeats and final selection belong to the parent. Review
used light text/JSON reads without TTNN imports or hardware access.

[Metadata](qualitative_prompt_format.json) records
`ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, pinned local snapshot and
`Qwen2Tokenizer` with the original chat template. Greedy generation uses
128 new tokens for shared prompts and 100 for AIME. A single generator is
reused with traced prefill/decode and context 2048.

Policy `head4_lofi_last8` uses a BFP4/LoFi head and BFP8 attention/MLP weights
only in layer 31. The other decoder projections remain BFP4/LoFi; decode
QKVG and KV remain BFP8. Initial AIME teacher forcing passes 92/100/100 at
84.523 t/s/u. This single timing is not a final ranking or token-out result.

| Prompt | Actual observation and comparison |
| --- | --- |
| 0, haiku | Coherent planning for a 5/7/5 structure; ends before a draft line is generated. No completed count mismatch occurs in the prefix. |
| 1, learning types | Correctly contrasts labeled supervised data with unlabeled unsupervised data and supplies sensible examples. |
| 2, story | Coherent planning for an inventor's discovery. HF begins its story within 128 tokens, while prior TT also remains in planning. |
| 3, thermodynamics | Recognizes the zeroth-law naming issue and begins the first-law heading, consistent with controls. |
| 4, French | `"Bonjour" (good day) or "Salut" (informal hello)` and appropriate formal/informal question forms. The reported raw-head error is absent. |
| 5, Fibonacci | Lists ordinary iterative, recursive and memoized approaches coherently, as in controls. No function is completed or executed. |
| AIME | Uses walking time `4 - t/60` hours and distance/speed `9/s` consistently. No final answer appears within the budget. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-request leakage was observed. All TT outputs end during reasoning;
this is not completed-response or longer-budget validation.

Light artifact checks passed: all seven input/HF-control metadata records
match `../../optimized_full_model/qualitative_prefill_trace_release_v2`,
including rendered prompts, input IDs, tokenizer, chat decision and budgets.
The revision/template/suite hashes match the pinned controls. Exact command,
source provenance and exit0 are recorded in
`../logs/qualitative_head4_lofi_last8_v1.provenance.json`.

The BFP8-head last-layer-only control fails French, so this success applies
to the full BFP4-head policy tested here. See [the AutoFix ledger](../AUTOFIX_french.md)
for neighboring rejections and the precision/context contract.
