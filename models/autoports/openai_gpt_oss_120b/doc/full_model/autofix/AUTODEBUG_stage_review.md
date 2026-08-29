# AutoDebug: full-model stage-review evidence gaps

## Starting evidence

- Source: `doc/full_model/stage_review.md`, verdict `more-work-needed`.
- Scope of this fresh-context pass: the P1 missing same-format HF controls,
  exact AIME reference provenance, and repeated-run/batch-position logit
  reproducibility.  No TT hardware was used and no implementation file was
  changed.
- The current six-prompt artifact is TT-only:
  `qualitative/qualitative_tt_chat.json`.  The only stored HF completion is the
  separate AIME control under `artifacts/autoregressive/`.

## Hypothesis experiments

### 1. The qualitative prompt format is aligned, but the model control is absent

- Hypothesis: the six TT requests used the pinned checkpoint's Harmony chat
  format correctly, but no HF model output exists for the same requests.
- Experiment: load the tokenizer from the pinned snapshot, render each prompt
  with `apply_chat_template([{"role": "user", ...}],
  add_generation_prompt=True, tokenize=True)`, normalize the returned
  `BatchEncoding`, and compare every token id to the stored TT artifact.  Search
  the full-model evidence tree for HF/control artifacts.
- Result: all six token-id arrays match exactly.  Prompt lengths are
  `75, 81, 90, 76, 81, 77`.  The tokenizer declares a chat template.  No HF
  control exists for these six prompts.
- Verdict: **verified**.  This is an evidence/harness gap, not evidence of a TT
  prompt-format bug.

### 2. The AIME artifact is pinned, but the generation command is not recoverable
from the artifact

- Hypothesis: the reference proves its payload identity but not its generation
  procedure.
- Experiment: inspect the `.refpt`, its SHA-256, and
  `qualitative_prompt_format.json`.
- Result: the reference SHA-256 is
  `7e722ad241eee84148ed62b5accee20bc642a4a1de4cab98ae146a166ee9d2bc`;
  it stores model/revision, `k=100`, the 214 prompt tokens, 100 generated
  tokens, top-100 tensors, and BOS/EOS/PAD ids.  Neither artifact stores an
  exact command, Python/package versions, loader/offload arguments, or the
  rendered-prompt hash.  The Harmony template injects the current date via
  `strftime_now`, so the recorded `2026-08-29` rendered prompt must be pinned
  explicitly for future reproduction.  Also, the metadata says
  `PreTrainedTokenizerFast`, while the current `transformers==5.12.1` loader
  reports `TokenizersBackend`; the prompt ids still match, but the actual
  generation environment must be recorded rather than inferred.
- Verdict: **verified**.  The existing reference itself need not be distrusted,
  but command-level provenance is missing.

### 3. Existing neighboring tests do not prove logit reproducibility

- Hypothesis: split-greedy equality, mixed slots, and readiness accuracy do not
  establish deterministic full logits across resets and physical batch rows.
- Experiment: inspect the tests and all full-model JSON artifacts for repeated
  full-logit hashes/equality results.
- Result: no determinism/reproducibility artifact exists.  The mixed batch test
  checks finite logits and inactive-row position state only; the split-greedy
  artifact compares sampled token ids between device and host paths.
- Verdict: **verified**.  A focused validation-only host-logit probe is needed.

## Required fixes

### A. Generate and inspect same-format HF controls for all six prompts

Add one repo-local, opt-in HF evidence helper (suggested path:
`doc/full_model/references/generate_hf_evidence.py`) that reuses the already
proven 120B CPU/NVMe loading policy.  Do not use the readiness runner's ordinary
`.to(cpu)` loader: it is not viable for this checkpoint.  For each prompt the
helper must:

1. resolve and assert the exact snapshot revision
   `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`;
2. call the exact Harmony chat template with one user message and
   `add_generation_prompt=True`;
3. assert the resulting prompt ids exactly equal the ids already stored in
   `qualitative_tt_chat.json`;
4. generate independently with `do_sample=False`, `num_beams=1`,
   `max_new_tokens=128`, `use_cache=True`, `pad_token_id=199999`, and the full
   EOS set `[200002, 199999, 200012]`;
5. decode with `skip_special_tokens=False`; and
6. write a small `qualitative/qualitative_hf_chat.json` containing prompt id,
   raw prompt, rendered-prompt SHA-256, prompt token ids, completion token ids,
   completion text, stop reason, model/revision, package versions, exact argv,
   and the load/offload policy.

The helper should keep one HF model loaded but generate prompts independently,
matching the TT generator's reset-per-request semantics.  A proposed invocation
is:

```bash
python_env/bin/python \
  models/autoports/openai_gpt_oss_120b/doc/full_model/references/generate_hf_evidence.py \
  --snapshot /home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --model-id openai/gpt-oss-120b \
  --revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --shared-prompts models/autoports/openai_gpt_oss_120b/doc/full_model/prompts/shared_qualitative_prompts.txt \
  --tt-artifact models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_tt_chat.json \
  --max-new-tokens 128 \
  --cpu-layer-count 27 \
  --offload-dir /tmp/gpt_oss_120b_hf_qualitative_offload \
  --output models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_hf_chat.json
```

The option names above are the proposed helper contract.  The final work log
must record the exact command actually run, not merely this template.  Add a
host-only contract test that pairs the HF and TT rows by id and asserts prompt
text, rendered prompt ids, revision, greedy settings, generation budget, and
EOS set match.

Then read every HF and TT completion and update `qualitative/verdict.md` with a
six-row prompt-by-prompt semantic comparison.  Token equality is not the gate;
coherence, correct language/task, repetition, control-token leakage, and
material degradation relative to HF are.  Rerun the degeneracy checker after
the HF artifact is present so both controls and TT outputs are covered.

### B. Store truthful, machine-readable AIME reference provenance

Add `references/aime24_chat_100_top100.provenance.json` (or equivalent fields in
a single evidence manifest) with:

- exact command argv that actually produced the checked-in payload;
- resolved model id, snapshot path, and revision;
- Python, PyTorch, Transformers, Accelerate, and tokenizer class/version from
  that run;
- CPU/NVMe device map and dtype/load arguments;
- source file and prompt index, chat-template flag, injected date
  `2026-08-29`, rendered prompt path/hash, prompt-token hash/count;
- generation length 100, top-k 100, greedy/EOS/PAD settings; and
- output path, size, and the existing SHA-256.

Do not invent the historical command.  If it is still present in the parent
agent's execution transcript, record it and validate every resulting metadata
field against the payload.  If it cannot be recovered exactly, extend the HF
helper with an AIME mode, regenerate the reference, run the existing readiness
gates if the payload changes, and record that executed command.  Merely writing
the skill's generic `models.common.readiness_check.generate` command is not
valid here because its stock loader does not encode the proven 120B offload
path.

### C. Add full-model repeated-run and batch-position logit evidence

Add an opt-in P150x4 acceptance test, preferably
`test_real_weight_36_layer_batch2_logit_reproducibility`, using the complete
36-layer model with `max_batch_size=2`.  Duplicate one non-aligned Harmony
prompt into two physical page-table rows and perform this sequence twice with
`generator.reset()` between runs:

1. explicit `prefill_forward(..., sampling_mode host boundary)` and retain both
   last-token full-logit rows;
2. feed the same greedy token to both rows at the same position and call
   `decode_forward(..., sampling_mode="host", reset_batch=True,
   force_host_tokens=True)`;
3. assert finite logits, bitwise equality between batch rows, and bitwise
   equality for each row across the two reset/reuse runs.

Write only compact evidence to `artifacts/logit_reproducibility.json`: model
revision/layer count/mesh, prompt length/hash, page rows, tensor shapes/dtypes,
argmax/top-100 token hashes, raw tensor SHA-256 hashes, `torch.equal` booleans,
and maximum absolute differences.  Do not store full-vocabulary tensors.  The
host full-logit reads are an explicit validation compatibility boundary and
must not be included in optimized token-out performance claims.

Run a two-layer probe first only to validate the test mechanics, then the
complete acceptance test for stage evidence:

```bash
GPT_OSS_120B_FULL_MODEL_PROBE=1 \
GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k two_layer_batch2_logit_reproducibility -s

GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 \
GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k real_weight_36_layer_batch2_logit_reproducibility -s
```

If bitwise equality fails, do not weaken the assertion to a loose tolerance and
call it deterministic.  Record the first differing boundary and continue the
AutoFix localization loop (prefill versus decode, row versus reset, then cache
page/position/terminal head) before accepting the stage.

## Optional strengthening

- Generate the six HF controls twice and compare completion-token hashes.  This
  is useful but not required once deterministic greedy settings, exact argv,
  environment, and prompt hashes are stored.
- Repeat the logit probe after a fresh model rebuild/process, in addition to
  reset/reuse within one model.  The reset/reuse test is the minimum relevant
  serving-path evidence; a fresh-process match is stronger provenance evidence.
- Store rendered prompt text as a separate compact JSON artifact in addition to
  the hashes.  Prompt token ids already make this inspectable, so this is not a
  blocker if the aggregate HF artifact includes the rendered text.

## Final status

**Still failing stage review until A, B, and C are implemented and evidenced.**
The P1 issue is not resolved by the existing AIME HF completion, because it is
a different prompt.  No runtime fallback or optimized decoder policy change is
indicated by this investigation.
