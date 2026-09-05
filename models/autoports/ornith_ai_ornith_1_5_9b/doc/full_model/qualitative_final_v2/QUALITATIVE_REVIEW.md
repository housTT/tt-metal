# Final qualitative review: selected BF16/HiFi4 head

Verdict: pass for the six-prompt, 128-token chat-format coherence and regression
check. The French label regression is absent from the new full-prompt output.
This verdict does not claim completed answers where the generation budget ends
inside reasoning.

## Prompt and artifact verification

Read all twelve actual HF/TT completion files in `prompt_0` through `prompt_5`.
Verified each user prompt, rendered template and prompt token IDs using the
pinned local tokenizer for `ornith-ai/Ornith-1.5-9B`, revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. The tokenizer class is `Qwen2Tokenizer`;
its nonempty chat template opens assistant reasoning with `<think>\n`.

All twelve completion files exactly equal decoding their retained token IDs.
Every side has 128 generated tokens. All six reused HF prompt/token sequences
and completion bytes match `../qualitative_v1`; the template and shared-suite
source hashes match `qualitative_prompt_format.json`. These checks were performed
with `HF_HUB_OFFLINE=1`, local-only tokenizer loading, and no TTNN import.
`comparison_summary.json` records exact artifact hashes and per-prompt findings.

The selected head is BF16/HiFi4, local columns 32768, K block 1, two readers.
Run commands, environment and source snapshot are in
`../logs/qualitative_final_v2.provenance.json` and its source archive. This
review is about the observed text; numerical/performance gate acceptance remains
with the parent and independent stage reviewer.

## Actual output comparison

- `prompt_0`, haiku: HF and TT identify the 5-7-5 form and discuss machine-learning
  themes. TT progresses to a proposed first-line theme; neither completes the
  poem by token 128. Repeated planning phrases are finite and task-related,
  without a mechanical loop. Final syllable-count correctness is not established.
- `prompt_1`, supervised learning: both describe labeled examples and learning
  with a teacher/answers. TT accurately supplies classification/regression examples
  and reaches the unsupervised heading at the limit. No factual contradiction
  or unexplained language drift appears in the inspected material.
- `prompt_2`, story: both plan a fantasy invention story. HF closes its reasoning
  span and starts the story; TT is still elaborating the setup at the limit.
  The quoted story opening and HF `</think>` correspond to the prompt/task and
  template, rather than a TT echo or control-token corruption. Finished story
  quality remains unmeasured by this window.
- `prompt_3`, thermodynamics: TT notes the requested three laws, then accurately
  states the zeroth-law thermal-equilibrium relation and its role in temperature.
  HF also considers the zeroth law before beginning the first. Both windows are
  incomplete explanations; no claim that all requested laws were delivered.
- `prompt_4`, French: TT now says `"Bonjour" (formal/greeting) or "Salut" (informal)`
  and supplies `Comment allez-vous?`, `Comment vas-tu?` and `aujourd'hui` correctly.
  This removes the prior specific claim that Bonjour was informal. The English
  explanatory reasoning matches the HF control, and the French phrases match
  the requested translation. The full response is still budget-limited.
- `prompt_5`, Fibonacci: HF and TT state the correct sequence and recurrence,
  then discuss iterative, recursive and generator approaches. The adjacent `1, 1`
  is mathematically correct and also appears in HF; it is not token duplication.
  Neither window contains a complete function, so no code-execution result is claimed.

First HF/TT generated-token divergence indices are 18, 23, 33, 16, 14 and 24.
Subsequent free-running wording differs while remaining coherent; raw positional
agreement is not used as teacher-forcing accuracy evidence.

## French anomaly resolution and limits

The earlier BF4/LoFi output in `../qualitative_final_v1` wrongly labeled Bonjour
informal. Its exact-prefix CPU HF control did not reproduce the mistake. The
subsequent investigation in `../AUTOFIX_french_head.md` and its raw controls
refuted a local terminal-kernel/quantization explanation for the already-reached
hidden tensor: all terminal candidates, including a CPU FP32 norm/head oracle,
still chose `inform` on that frozen hidden state.

Full-prompt controls are distinct evidence. BF4/BF8 candidate runs retained the
wrong label, whereas BF16/HiFi4 selected a different earlier free-running branch
and produced the correct register wording. The fastest qualifying legal BF16
geometry, 32768/K1/two readers, now reproduces that corrected behavior in this
complete six-prompt run. Resolution: fixed in the delivered free-running path.
No claim is made that its head changes the frozen-hidden next-token ranking.
The optimized decoder precision, cache, residual and collective strategy were
preserved by this terminal-policy selection.

## Final degeneracy checks

Both final-only and model-wide checks exited zero with no degenerate output:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.5-9B \
  --model-dir models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v2 \
  --missing-artifacts critical --scope autoregressive \
  --json models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v2/degeneracy_final.json

TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.5-9B \
  --model-dir models/autoports/ornith_ai_ornith_1_5_9b \
  --missing-artifacts critical --scope autoregressive \
  --json models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v2/degeneracy_all.json
```

Logs: `degeneracy_final.log` and `degeneracy_all.log`. French repeated phrase
fragments are finite quoted translation variants; the Fibonacci adjacent-word
metric is explained by the required repeated number 1. No unexplained collapse,
doubled subwords, gibberish, cross-request text, or wrong-language drift was
observed. No unresolved qualitative finding remains within the stated window.
