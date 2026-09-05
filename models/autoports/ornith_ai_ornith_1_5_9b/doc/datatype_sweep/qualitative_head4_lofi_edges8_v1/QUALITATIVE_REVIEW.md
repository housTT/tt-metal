# BFP4 head with BFP8 edge layers: qualitative review

Verdict, 2026-09-05: passes the recorded bounded qualitative smoke. All six
shared prompts and AIME were read against the matching pinned HF and prior TT
controls. No new concrete quality regression was found. This does not select
a final precision winner or establish complete-answer quality beyond the
recorded budgets. Review was host-only without TTNN imports or hardware access.

[Metadata](qualitative_prompt_format.json) identifies checkpoint
`ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, and the pinned local snapshot.
`Qwen2Tokenizer` uses the original chat template with a user message and
`add_generation_prompt=True`. Generation is greedy, 128 tokens for shared
prompts and 100 for AIME, using one reused generator with traced prefill/decode.

Policy `head4_lofi_edges8` uses BFP4/LoFi head weights and BFP8 attention/MLP
projection weights only in layers 0 and 31. Other projection groups remain
BFP4/LoFi, decode QKVG and KV remain BFP8, and other runtime policy fields are
unchanged. AIME teacher forcing passes 94/100/100 at 84.137 t/s/u; that timing
is distinct from the parent's pending matched token-out comparison.

| Prompt | Actual observation and comparison |
| --- | --- |
| 0, haiku | Coherent planning for 5/7/5 lines; no completed draft count or suspicious arithmetic in this prefix. HF and prior TT also spend this budget planning. |
| 1, learning types | Describes labeled supervised data and mapping inputs to known outputs, with sensible examples; begins unsupervised learning before cutoff. |
| 2, story | Coherent planning for a discovery fitting the inventor story. HF starts a story within the budget while prior TT also remains in planning. |
| 3, thermodynamics | Gives the zeroth-law equilibrium relation and begins energy conservation; no new concrete error relative to controls. |
| 4, French | `"Bonjour" (good morning/day) or "Salut" (informal hello)`; appropriate formal/informal question forms. The raw-head Bonjour error is absent. |
| 5, Fibonacci | Lists ordinary iterative, recursive and memoized approaches. The preference for iteration over naive recursion matches the HF discussion; no function is completed or executed. |
| AIME | Coherent restatement of the walking/coffee conditions, identical to the head8 edge control; no final-answer claim. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-request leakage was observed. All TT outputs end during reasoning.
The haiku verdict covers this prefix only; it does not imply the ungenerated
poem will have correct syllable counts.

Host evidence checks passed: all seven input/HF-control metadata records
match `../../optimized_full_model/qualitative_prefill_trace_release_v2`,
decoded token IDs reproduce the saved HF/TT text, and the tokenizer/template is
pinned. During the historical preselection audit on 2026-09-05, all ten inspected
runtime Python source hashes matched provenance; this does not assert that
the later default-loader source is unchanged. Exact
command and exit status are in
`../logs/qualitative_head4_lofi_edges8_v1.provenance.json`. The
[AutoFix ledger](../AUTOFIX_french.md) records the neighboring candidate
rejections and remaining selection work.
