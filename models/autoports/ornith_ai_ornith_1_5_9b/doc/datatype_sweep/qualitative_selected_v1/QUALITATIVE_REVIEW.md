# Selected default qualitative review

Verdict: passes the recorded bounded qualitative default-path check. The run
completed with exit0 on **2026-09-05 at 23:08:11 UTC**. All seven TT token lists
and saved completion texts are identical to the qualified explicit
`head4_lofi_last8` control. All seven actual outputs were read against that
control and pinned HF. No new concrete quality regression was found. Review
used light text/JSON reads without TTNN imports, hardware access or production edits.

## Default path and controls

The [run provenance](../logs/qualitative_selected_v1.provenance.json) records
the normal runner command without a precision or head override:

```bash
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep.run_qualitative --reuse-hf models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/qualitative_prefill_trace_release_v2 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/qualitative_selected_v1
```

[Metadata](qualitative_prompt_format.json) records selected policy
`head4_lofi_last8`, equal to `../selected_precision_config.json` at review:
BFP4/LoFi head, BFP8 attention/MLP weights only at layer 31, ordinary remaining
projections BFP4/LoFi, decode QKVG and KV BFP8. Other dtype/state/CCL fields
match the qualified explicit-config control. The recorded head geometry is
32768 local columns, K1 and two readers, with sharded final norm and traced
prefill/decode. The request cache context is 2048.

Checkpoint `ornith-ai/Ornith-1.5-9B` is pinned to revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53` and the local upstream snapshot.
`Qwen2Tokenizer` uses the original chat template with a user message and
generation prompt. Generation is greedy: 128 tokens for each shared prompt
and 100 for AIME. A single generator is reused across all seven requests.

All nine input/HF-control metadata fields match both
`../qualitative_head4_lofi_last8_v1` and
`../../optimized_full_model/qualitative_prefill_trace_release_v2`: snapshot,
user text, chat decision/mode, tokenizer class, rendered prompt, input token
IDs, generation budget and HF tokens. Saved HF texts match as well. The
seven TT token lists and texts exactly match the explicit-config LoFi control.

## Actual output review

| Prompt | Observation and bounded verdict |
| --- | --- |
| 0, haiku | Coherent planning for a 5/7/5 structure; no draft line or completed count mismatch appears before cutoff. |
| 1, learning types | Correctly contrasts labeled supervised data with unlabeled unsupervised data and supplies sensible examples. |
| 2, story | Coherent inventor-story planning. HF starts a story within the budget while prior TT also remains in planning; no new regression. |
| 3, thermodynamics | Recognizes the zeroth-law naming issue and begins the first-law heading, as in the qualified control. |
| 4, French | `"Bonjour" (good day) or "Salut" (informal hello)` and appropriate formal/informal question forms. The original French failure remains corrected. |
| 5, Fibonacci | Coherent ordinary iterative/recursive/memoized approaches; no completed function to execute. |
| AIME | Uses walking time `4 - t/60` and distance/speed `9/s` consistently; no final answer within the budget. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-request leakage was observed. All TT outputs remain reasoning
prefixes. This pass covers the recorded budgets and geometry, not complete
answers or a later geometry change.

This is new evidence for the final default loader, separate from historical
live-source comparisons. The parent's geometry follow-up and other stage
checks are outside this qualitative verdict. See [the AutoFix ledger](../AUTOFIX_french.md).
