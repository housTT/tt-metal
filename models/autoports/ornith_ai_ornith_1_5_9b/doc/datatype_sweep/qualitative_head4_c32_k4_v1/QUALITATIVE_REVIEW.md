# Raw BFP4/LoFi C32/K4/R2 qualitative review

Verdict: quality fail. The new geometry run itself still says
`"Bonjour" (informal) or "Salut" (casual)` in prompt 4, while pinned HF and the
qualified last-layer controls disagree. This rejection comes from the new
outputs, not an assumption that the earlier K1 verdict transfers to K4.
All seven outputs were read. Review used light text/JSON checks with no TTNN
imports, hardware access or production edits.

[Metadata](qualitative_prompt_format.json) records policy
`head4_lofi_c32_k4_r2`: BFP4/LoFi head, no layer exceptions, 32 head cores,
32768 columns, K4 and two readers. The older raw K1 control used 64 head cores,
so both core count and accumulation block differ. Ordinary decoder projections
remain BFP4/LoFi, decode QKVG and KV BFP8, and other state/CCL contracts match.
The new geometry has a five-repeat AIME result of 93/100/100 at median 87.391
t/s/u; this does not override the qualitative failure.

Checkpoint `ornith-ai/Ornith-1.5-9B` remains pinned to
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. `Qwen2Tokenizer` uses the original
chat template with generation prompt. Greedy budgets are 128 tokens for the
six shared prompts and 100 for AIME, using one traced reused generator.

| Prompt | Actual observation |
| --- | --- |
| 0, haiku | Coherent planning; identical to raw K1, with no completed draft count before cutoff. |
| 1, learning types | Coherent supervised-learning explanation; changes from raw K1 at token 85 without a concrete error. |
| 2, story | Coherent story planning, identical to raw K1. HF begins its story within the budget; prior TT also remains in planning. |
| 3, thermodynamics | Same coherent naming discussion and first-law heading as raw K1. |
| 4, French | Incorrectly labels Bonjour informal; all 128 tokens match rejected raw K1. This is the disqualifying error. |
| 5, Fibonacci | Same coherent ordinary approach discussion as raw K1; no function is completed or executed. |
| AIME | Coherent initial equation setup, identical to raw K1; no final answer within 100 tokens. |

No broader mechanical repetition, wrong-language drift, corrupt/control-token
leakage or cross-request leakage was observed. This is a reasoning-prefix
quality failure, not a claim about a completed final answer.

All nine input/HF-control fields match the pinned previous-stage suite. The
recorded policy equals its explicit config at review; the runtime head program
records K4, per-core-N32 and two readers. Exact command and exit0 appear in
`../logs/qualitative_head4_c32_k4_v1.provenance.json`, spanning
2026-09-05 23:31:59..23:33:16 UTC. See [the AutoFix ledger](../AUTOFIX_french.md).
