# Final selected C32/K4 default qualitative review

Verdict: passes the bounded final default-path qualitative check. The run
completed with exit0 on **2026-09-05 at 23:44:26 UTC**. All seven TT token lists
and saved texts exactly match the qualified explicit-config
`head4_lofi_last8_c32_k4_r2` suite. All actual outputs were read against that
control and pinned HF; no new concrete quality defect was found. Review used
light text/JSON reads without TTNN imports, hardware access or production edits.

## Consumed default policy

The [run provenance](../logs/qualitative_selected_v2.provenance.json) records
the ordinary runner without a precision or head override:

```bash
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.datatype_sweep.run_qualitative --reuse-hf models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/qualitative_prefill_trace_release_v2 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/datatype_sweep/qualitative_selected_v2
```

[Metadata](qualitative_prompt_format.json) records
`head4_lofi_last8_c32_k4_r2`, equal to the selected artifact and qualified
explicit config at review. The head uses BFP4/LoFi, 32 cores, 32768 columns,
K4 and two readers. The runtime program records K4/per-core-N32/two readers,
confirming consumption of the selected geometry. Layer 31 attention/MLP
weights are BFP8; remaining ordinary projections are BFP4/LoFi and decode
QKVG/KV are BFP8. Other dtype/state/CCL fields match the qualified control.
Final norm is sharded, prefill/decode are traced, and request context is 2048.

Checkpoint `ornith-ai/Ornith-1.5-9B` remains pinned to revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53` and the local upstream snapshot.
`Qwen2Tokenizer` uses the original chat template with a user message and
generation prompt. Greedy generation uses 128 tokens per shared prompt and
100 for AIME; one real generator is reused across all seven requests.

## Exact controls and actual outputs

All nine input/HF-control metadata fields match both
`../qualitative_head4_last8_c32_k4_v1` and
`../../optimized_full_model/qualitative_prefill_trace_release_v2`, including
rendered prompts, input token IDs, tokenizer, chat mode, budgets and HF tokens.
Saved HF texts also match. All seven TT token lists and texts exactly match
the qualified explicit C32/K4 last-layer control.

| Prompt | Actual observation and bounded verdict |
| --- | --- |
| 0, haiku | Coherent planning for a 5/7/5 structure; no draft line or completed count mismatch before cutoff. |
| 1, learning types | Correctly contrasts labeled supervised data and unlabeled unsupervised data with sensible examples. |
| 2, story | Coherent inventor-story planning, matching the newly qualified K4 branch. HF starts its story within the budget; prior TT also remains in planning. |
| 3, thermodynamics | Recognizes the zeroth-law naming issue and begins the first-law heading, matching the qualified control. |
| 4, French | `"Bonjour" (good day) or "Salut" (informal hello)` and appropriate formal/informal question forms. The original reported error is absent. |
| 5, Fibonacci | Coherent ordinary iterative/recursive/memoized approach discussion; no completed function to execute. |
| AIME | Consistent walking time `4 - t/60` and distance/speed `9/s`; no final answer within the budget. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-request leakage was observed. All outputs remain reasoning prefixes;
this result covers the recorded budgets and geometry, not complete answers
or untested later changes.

This is new evidence for the final C32/K4 selected default and does not rely on
historical live-source equality or the earlier K1 default pass. The bounded
qualitative disposition is complete. The parent owns overall stage closure;
its other passing checks are outside this review. See [the AutoFix ledger](../AUTOFIX_french.md).
