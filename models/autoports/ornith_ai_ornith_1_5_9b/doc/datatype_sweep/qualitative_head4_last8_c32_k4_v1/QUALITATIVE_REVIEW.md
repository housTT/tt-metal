# BFP4/LoFi last-layer BFP8 C32/K4/R2 qualitative review

Verdict: passes the recorded bounded qualitative smoke. All seven outputs
were read against pinned HF and the qualified K1 last-layer control. French
is correct, and no new concrete quality defect was found. This qualifies the
new explicit-config geometry; parent owns promotion and the new final default
checks. Review used light text/JSON reads without TTNN imports, hardware access
or production edits.

[Metadata](qualitative_prompt_format.json) records
`head4_lofi_last8_c32_k4_r2`: BFP4/LoFi head, 32 head cores, 32768 columns, K4,
two readers, and BFP8 attention/MLP weights only at layer 31. The older K1
control used 64 head cores, so this tests the complete changed geometry.
Other projections remain BFP4/LoFi, decode QKVG and KV BFP8; state/CCL formats
and tracing contracts are preserved. Five AIME repeats retain 92/100/100 with
median 87.288 t/s/u, a teacher-forcing metric rather than token-out throughput.

Checkpoint `ornith-ai/Ornith-1.5-9B` remains pinned to
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. `Qwen2Tokenizer` uses the original
chat template and generation prompt. Greedy budgets are 128 tokens per shared
prompt and 100 for AIME, with one reused traced generator and context 2048.

| Prompt | Actual observation versus qualified K1 last-layer control |
| --- | --- |
| 0, haiku | Token-identical coherent 5/7/5 planning; no draft line or completed count mismatch before cutoff. |
| 1, learning types | Token-identical distinction between labeled supervised and unlabeled unsupervised data. |
| 2, story | Changes at generated token 40; the new branch coherently plans a discovery for the inventor. HF begins its story within the budget; prior TT also remains in planning. |
| 3, thermodynamics | Token-identical zeroth-law naming discussion and first-law heading. |
| 4, French | Token-identical correct `"Bonjour" (good day) or "Salut" (informal hello)` and appropriate question forms. |
| 5, Fibonacci | Token-identical coherent ordinary approach discussion; no function is completed or executed. |
| AIME | Token-identical consistent walking-time/equation setup; no final answer within the budget. |

Six of seven TT token lists exactly match K1; the story is the only changed
completion. No wrong-language drift, mechanical repetition, corrupt/control-token
leakage or cross-request leakage was observed. All outputs remain reasoning
prefixes, so this is not completed-answer or longer-budget validation.

All nine input/HF-control metadata fields match both pinned HF and the old
last-layer control. The recorded policy equals its explicit config at review;
the runtime head program records K4, per-core-N32 and two readers. Exact
command and exit0 are in
`../logs/qualitative_head4_last8_c32_k4_v1.provenance.json`, spanning
2026-09-05 23:33:16..23:34:34 UTC. See [the AutoFix ledger](../AUTOFIX_french.md).
