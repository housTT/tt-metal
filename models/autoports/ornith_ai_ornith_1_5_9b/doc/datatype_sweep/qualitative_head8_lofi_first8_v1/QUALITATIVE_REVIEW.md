# First-layer BFP8 qualitative review

Status, 2026-09-05: quality fail. The first-layer-only exception fixes French,
but the 256-token same-prompt control confirms repeated incorrect haiku
counting while HF counts its draft correctly. This candidate is rejected
overall. The failure is in generated reasoning; neither 256-token output
reaches a final poem. Review was host-only with no TTNN import or hardware access.

## Format and controls

Model: `ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, pinned local `upstream` snapshot.
`Qwen2Tokenizer` uses its original nonempty chat template, rendered with
`apply_chat_template([{"role": "user", ...}], tokenize=False,
add_generation_prompt=True)`. Generation is greedy, 128 new tokens for each
of the six shared prompts and 100 for AIME. The runner reuses one real TT
generator across requests with prefill/decode traces and context 2048.

[Prompt-format metadata](qualitative_prompt_format.json) records policy
`head8_lofi_first8`: head BFP8/LoFi; only layer 0 attention, MLP gate/up and MLP
down weights increase to BFP8. Other projection groups remain BFP4/LoFi,
decode QKVG and KV remain BFP8, and activation/state/CCL contracts are unchanged.
The actual AIME runtime artifact records that policy on all 32 layers and
passes 93/100/100. Its 84.607 t/s/u teacher-forcing result is separate from the
parent's pending matched token-out comparison.

The seven current prompt renderings/token lists were regenerated using the
pinned local tokenizer. All nine input/HF-control metadata fields match
`../../optimized_full_model/qualitative_prefill_trace_release_v2`; its actual
HF and TT outputs, plus edge8/raw-head8 controls, were reviewed. Raw completion
text matches decoded output token IDs. During the historical preselection
audit on 2026-09-05, all ten inspected runtime Python source hashes matched
this run's provenance; this is not a claim about the later default-loader
source. Exact command and exit0 are recorded in
`../logs/qualitative_head8_lofi_first8_v1.provenance.json`.

## Actual output review

| Prompt | Observation versus HF/prior TT | Verdict within recorded budget |
| --- | --- | --- |
| 0, haiku | The 128-token count fragment is completed and incorrectly reaffirmed in the 256-token control. HF counts its own draft correctly. | Fail: persistent incorrect counting, not merely the cutoff. |
| 1, learning types | Distinguishes labeled supervised data and unlabeled unsupervised data, with sensible examples. | No concrete regression found; unfinished reasoning. |
| 2, story | Coherent planning for the inventor's discovery. HF starts the story by 128 tokens; prior TT also remains in planning. | Existing budget limitation, no new defect found. |
| 3, thermodynamics | Gives a coherent zeroth-law statement, then reaches the first-law heading. Prior TT similarly includes the zeroth law. | No concrete regression found; incomplete answer. |
| 4, French | `"Bonjour" (formal/greeting) or "Salut" (informal)` and appropriate formal/informal question forms. | Reported French error corrected. |
| 5, Fibonacci | Correct sequence prefix and recurrence; output is token-identical to prior TT. | No new regression; no generated function to execute. |
| AIME | Coherently restates both walking/coffee conditions; identical to edge8's 100-token output. | No concrete regression found; no final-answer claim. |

No wrong-language drift, mechanical repetition, corrupt/control-token leakage
or cross-prompt leakage was observed. These are bounded reasoning-prefix
checks, not completed-response validation.

## Haiku continuation resolves the cutoff

The exact final tokens of prompt 0 are indices 120..127:
`Patterns`, ` in`, ` the`, ` data`, ` flow`, `"`, ` (`, `5`.
Index 127 is token 20; the metadata records exactly 128 generated tokens, matching
the generation budget. There is no closing parenthesis or `</think>` in the
recorded TT text. Seven syllables would be Patterns 2 + in 1 + the 1 + data 2 +
flow 1. The 128-token fragment alone did not prove a retained incorrect count.

A search of all same-prompt autoregressive artifacts under this autoport found
13 runs, every HF/TT output capped at 128. Parent then generated the
[256-token control](../qualitative_first8_haiku256_v1/prompt_0/autoregressive_meta.json)
from the original prompt with unchanged first8 policy and a fresh pinned HF
control. Host checks verify identical prompt text/rendering/token IDs,
tokenizer/template, revision and precision policy. Both HF and TT preserve
their earlier 128-token prefixes exactly and produce 256 tokens. During that
historical 2026-09-05 audit, all ten inspected runtime source hashes matched
`../logs/qualitative_first8_haiku256_v1.provenance.json`,
which records the exact command and exit0.

The [extended TT text](../qualitative_first8_haiku256_v1/prompt_0/tt_completion.txt)
first completes `"Patterns in the data flow" (5)`, then claims to recount it:

> Pat-ters (2) - in (1) - the (1) - da (1) - flow (1) = 5 syllables ✓

This omits part of `data`; even the listed numbers sum to six, not five. The
full line has seven syllables. In contrast, the
[extended HF text](../qualitative_first8_haiku256_v1/prompt_0/hf_completion.txt)
correctly counts `Data shapes the mind` as 5, `Patterns emerge from the noise`
as 7, and `Learning, line by line` as 5. The confirmed TT counting error persists
beyond the old cutoff and has no matching HF-control behavior. First8 therefore
fails the quality comparison despite its French fix and AIME accuracy pass.
Neither output reaches `</think>` in 256 tokens; this verdict does not claim
that a final emitted poem was invalid.

With the BFP8/LoFi head, last-layer-only is independently rejected: its French still says
`"Bonjour" (informal greeting)` despite AIME 93/100/100. See
[the AutoFix ledger](../AUTOFIX_french.md) for source/precision evidence,
reproducible host checks and remaining localization limits.
