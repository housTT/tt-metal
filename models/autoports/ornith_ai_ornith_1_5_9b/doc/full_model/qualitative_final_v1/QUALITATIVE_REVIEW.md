# Final six-prompt qualitative review and French branch control

Verdict: more-work-needed for the French register-label anomaly. The six TT
outputs are coherent task-related text, with no mechanical degeneration, but
the focused HF control does not reproduce the incorrect label.

## Paired output inspection

All six prompts used the checkpoint's exact chat template and 128-token greedy
continuations; the template opens an assistant `<think>` span. Both HF and TT
therefore expose reasoning before a final answer. I read every saved HF and TT
completion in `prompt_0` through `prompt_5`.

- `prompt_0`, haiku: both discuss machine-learning themes and the 5-7-5 structure.
  Neither reaches a finished haiku by token 128. TT repeats the planning phrase
  several times but does not enter a mechanical phrase loop; HF also plans and
  counts syllables. Final poem quality is not established by this window.
- `prompt_1`, supervised/unsupervised learning: both correctly distinguish labeled
  versus unlabeled data and use the teacher analogy. TT gives classification and
  regression examples. No wrong-language drift or factual contradiction observed.
- `prompt_2`, story: both plan a whimsical invention story. HF closes `</think>`
  and begins the story before the limit, while TT is still planning at the limit.
  The literal tag in HF is consistent with the rendered reasoning template;
  it is not evidence of a TT decode loop failure. Finished story quality remains
  outside this 128-token window.
- `prompt_3`, thermodynamics: both distinguish the zeroth law from the requested
  first/second/third laws. TT reaches the first law as energy conservation. Neither
  presents all three laws within the window; no contradiction observed in the
  inspected material.
- `prompt_4`, French translation: both produce appropriate formal/informal
  sentence variants. TT additionally labels `Bonjour` itself as `(informal)`,
  while HF describes it as a general greeting. That local label is incorrect and
  required the focused control below. English reasoning about French translation
  matches HF's prompt-correct behavior; it is not unexplained wrong-language drift.
- `prompt_5`, Fibonacci: both plan a Python answer and discuss implementation
  alternatives; TT describes iterative versus exponential recursive complexity.
  Neither reaches actual function code within 128 tokens. No code correctness
  result is claimed from these reasoning-only continuations.

HF/TT first divergence indices are respectively 18, 3, 33, 16, 14 and 23. These
are zero-based generated-token indices, not prompt positions. Lexical differences
such as adding "me to" can change the rest of free-running text; token-wise
free-run agreement is not a teacher-forcing accuracy metric. No sampled token
feedback, cache correctness or device determinism claim is inferred from prose.

## French anomaly: exact-prefix HF control

The original hypothesis was that HF might reproduce the incorrect register label
when conditioned on the same preceding TT wording. Tested with the same pinned
local checkpoint, CPU BF16 `Qwen3_5ForCausalLM`, Transformers 5.12.1, all 427 weight
keys loaded with no missing/unexpected/mismatched keys or errors.

Executed:

```bash
HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
OMP_NUM_THREADS=8 PYTHONPATH=. python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v1/french_branch_control.py
```

The control verifies the 24 prompt IDs against the exact rendered chat template,
then conditions HF on the first 34 generated TT IDs, ending immediately before
the error at `"Bonjour" (`. It does **not** force the erroneous label. HF independently
continues `formal/greeting) or "Salut" (informal)`. The hypothesis that this control
reproduces TT's label is refuted.

Conditional scoring under the TT prefix places TT's token `inform` at HF rank 7
at generated index 34 (absolute logit position 57). `al` at index 35 then ranks 1,
which is expected after `inform` was already conditioned on. The first lexical
branch at index 14, TT ` into` versus HF ` to`, is rank 2. Across the full 128-token
TT continuation, conditional HF top-1/top-5/top-100 membership is
94.53125% / 99.21875% / 100%; these are localization measurements, not a substitute
for the stage's readiness gate.

Required follow-through: investigate the real TT logits/hidden state at the
24+34-token prefix and compare the local `form`/`inform` ranking to this HF
control. The parent must decide and test a supported model correction or another
non-tautological control; the existing paired outputs do not support dismissing
the error as demonstrated HF behavior. A longer TT/HF completion may establish
final-answer behavior but cannot by itself erase the recorded reasoning error.

Artifacts: `french_branch_control.py`, `.log`, `.json`, and
`french_branch_hf_continuation.txt`. The JSON retains prompt and branch token IDs,
source/input/template hashes, pinned checkpoint/revision, loading diagnostics,
conditional ranks/logits and the 96-token HF continuation. CPU runtime: 52.057 s.
No TTNN import or accelerator command occurred.

## Degeneracy checks

Both commands returned zero with no degenerate output detected:

```bash
python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.5-9B \
  --model-dir models/autoports/ornith_ai_ornith_1_5_9b \
  --missing-artifacts critical --scope autoregressive \
  --json models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v1/degeneracy_all.json

python_env/bin/python models/common/readiness_check/check_degenerate_output.py \
  --hf-model ornith-ai/Ornith-1.5-9B \
  --model-dir models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v1 \
  --missing-artifacts critical --scope autoregressive \
  --json models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/qualitative_final_v1/degeneracy_final.json
```

Logs: `degeneracy_all.log` and `degeneracy_final.log`. The final six all have zero
adjacent word duplication; trigram loop fractions range from 0.0333 to 0.1047.
The machine-clean result does not override the factual label finding above.

Additional previous-stage control read: `../qualitative_v1/prompt_4/tt_completion.txt`
says `"Bonjour" (formal/greeting) or "Salut" (informal)`. It therefore supports
investigating the new final-path label as a regression, not waiving it as an
unavoidable model behavior. The parent delegated a terminal-policy probe to
`watcher_repair`; the exact prompt/prefix/control paths were sent to that agent.
The shared summary in `comparison_summary.json` remains `more-work-needed`
until corrected output is produced and re-reviewed. Reused HF prompt IDs and
completion bytes were checked against the earlier control for all six prompts.
