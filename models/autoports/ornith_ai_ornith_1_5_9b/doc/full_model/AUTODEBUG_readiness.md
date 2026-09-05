# AutoDebug: missing readiness runners

Inspection date: 2026-09-05. Inspected checkout: `c61ea5a4ca101af524f7518923b993b71225c81c`.
Workflow source: `70a596f92229ada922fba743cd0cd9d2658a5c1c`.
Prior Ornith reference: `f7662055fe4ae3d66509335d96a7c74acd53911b`.
This investigation made no implementation changes, loaded no model weights, and did not import
TTNN or open/list/reset any accelerator. The main agent was independently authoring the full model.

## Findings

1. **The readiness package is absent, rather than merely misconfigured.**
   `models/common/readiness_check` does not exist at the inspected checkout. Its 21 files are
   present in the pinned workflow commit, while this branch imported only `.agents/` from that
   commit. The prior reference checkout is sparse and also lacks this directory on disk, but
   `git show f7662055:models/common/readiness_check/...` resolves the prior implementation.
   Restoring the non-serving dependency closure below is sufficient to resolve the missing
   package. No C++ change or additional external dependency follows from this finding.

2. **Verbatim restoration then fails on chat-template tokens with installed Transformers.**
   Workflow `generate.py:121-135` iterates the return value of
   `tokenizer.apply_chat_template(..., tokenize=True)` as token IDs. In this environment,
   Transformers **5.12.1**, the pinned local checkpoint loads a `Qwen2Tokenizer` with a nonempty
   chat template; the call returns `BatchEncoding` with `input_ids` and `attention_mask`.
   The workflow normalization raises `ValueError: invalid literal for int() with base 10:
   'input_ids'`. The prior commit's `generate.py:128-140` already normalizes mappings, tensors,
   and nested lists. Port that focused change, or explicitly request the list return form
   and verify both API forms. This is a directly reproduced host-side failure.

3. **Restored HF entry points do not themselves pin a remote revision or preserve enough
   reference provenance.** Workflow `generate.py:322-323` and
   `run_autoregressive.py:90-91,136` call `from_pretrained` without a revision; neither CLI
   accepts one. `schema.py:96-115` stores model ID, K, tokens, and special-token IDs, but no
   revision, tokenizer identity, prompt source, chat flag, or command. Supplying the already
   pinned local snapshot `/home/hous/dev/ornith-1.5-9b/upstream` as `--hf-model` is the smallest
   compliant intervention for this model: every existing load then resolves locally. Record
   canonical model ID `ornith-ai/Ornith-1.5-9B`, revision
   `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, snapshot path and hashes in an autoport sidecar.
   If the CLI must accept a remote model ID, add and propagate `--revision` to **every** config,
   model, and tokenizer load instead. Adding it only to the model load is insufficient.

4. **The autoregressive runner does not render chat prompts or run the shared suite.**
   Workflow `run_autoregressive.py:132-137` always encodes raw file text. There is no chat flag.
   Raw default output is continuation stress coverage only for this checkpoint. A small
   model-local driver can render the exact checkpoint template to a file, verify that the
   runner's subsequent encoding reproduces the template's token IDs exactly, and invoke
   `run_autoregressive` with that file. Alternatively add an explicit chat mode to the common
   runner and reuse the normalized prompt helper. Preserve both the original user prompt
   and the rendered prompt/tokens in metadata. The shared suite is the six nonempty lines of
   `vllm_prompts.txt`; using those prompts through the TT generator does not require restoring
   or running `run_vllm_server.py`.

5. **The restored runners provide metrics, but do not enforce accuracy thresholds.**
   Prefill and teacher-forcing CLIs print/return results and can exit zero for low accuracy.
   The model-local evidence driver must retain top-1/top-5/top-100 counts, denominator, K, and
   explicit stage-threshold verdicts. It must require `k == 100` for a top-100 claim: the
   `top100` dictionary key actually means top-K for whatever K the reference contains.
   Teacher forcing already rejects an incomplete callback sequence and requires the
   explicit `enable_trace` parameter; preserve those checks. Its timing measures token-out
   generation plus callback overhead, not a logits-only path or necessarily warmed latency.

6. **The common CLI has no four-chip profile and lacks model-specific device-open controls.**
   Workflow `mesh_device.py:14-19` only maps `N150=(1,1)`, `N300=(1,2)`, `T3K=(1,8)`, and
   `TG=(8,4)`. It opens the mesh with only `mesh_shape` at line 63. A model-local driver can
   open the validated 1/2/4-chip profile with the established trace/fabric configuration and
   call the runners' programmatic APIs, which accept `mesh_device` and `build_kwargs`.
   This is narrower than changing shared mesh behavior. Report physical P300c hardware and
   `p150`, `p150x2`, or `p150x4` as the logical profile; do not use a Wormhole label as hardware
   provenance. Device behavior was outside this investigation.

## Minimal source/data closure

Restore these files from the workflow commit. Keeping the original `__init__.py` requires
`contract_vllm.py`; it is a lightweight typing/torch contract and imports no vLLM package.

```text
models/common/readiness_check/
  __init__.py
  contract.py
  contract_vllm.py
  schema.py
  teacher_forcing.py
  generate.py
  mesh_device.py
  run_prefill_check.py
  run_teacher_forcing.py
  run_autoregressive.py
  check_degenerate_output.py
  autoregressive_prompt.txt
  vllm_prompts.txt
```

Restore the existing host tests alongside them: `test_generate.py`, `test_schema.py`,
`test_run_prefill_check.py`, `test_run_teacher_forcing.py`, and
`test_check_degenerate_output.py`. Do not restore the serving runner/test merely to use the
prompt file. `references/.gitignore` is optional because this model's artifacts belong under
its own autoport. The existing shared package initializer eagerly imports `generate`, so
even a contract import requires Transformers as well as torch; these are already installed.
`tqdm` is optional. TTNN is imported lazily only by the mesh open/close functions.

The AIME24 data already exists at
`models/demos/deepseek_v3/demo/aime_under_8k_prompts.json`, contains five prompts, and is
byte-identical to the workflow version. SHA-256:
`ced18521fe26b8e304b3d9900e9c05e33387a3967dbb964c941eaca77321ef99`.
No DeepSeek code imports are needed. The optional book file also already exists.

## AIME24 chat100/top100 contract

Use prompt source `aime24`, prompt index 0, `--chat-template`, `--gen-len 100`, and
`--top-k 100`. The current pinned tokenizer renders prompt index 0 to **161 tokens**.
The reference command after restoration and prompt normalization can be:

```bash
HF_HUB_OFFLINE=1 python_env/bin/python -m models.common.readiness_check.generate \
  --hf-model /home/hous/dev/ornith-1.5-9b/upstream \
  --prompt-source aime24 --aime24-prompt-index 0 --chat-template \
  --gen-len 100 --top-k 100 \
  --output models/autoports/ornith_ai_ornith_1_5_9b/readiness_aime24_chat.refpt
```

This command was **not run**; it loads all HF weights. Record generation dtype, model class,
Transformers version, actual special-token stop IDs, tokenizer/chat-template identity, exact
command and prompt source hash in the sidecar. `gen-len` is a maximum and HF can stop on EOS;
check the actual generated length before describing the reference as a 100-token gate.

For a 100-token continuation, prefill checks a 261-token concatenated input and scores logits
at positions `[160:260]`. Teacher forcing prefills the original 161 tokens, then requests 100
predictions: one prefill prediction and 99 decode predictions. Its callback records each TT
prediction and returns that step's HF continuation token as the next input. The implementation
must invoke it exactly 100 times and must not stop early on EOS during teacher forcing.
It must return its own predictions, not the forced tokens. Scoring compares each TT argmax
with the reference rank-1, ranks 1-5, and ranks 1-100; it is not a comparison of TT top-K sets.

The full-model driver must independently record all three profiles and enforce the active
stage's accuracy thresholds. Keep trace evidence separate from the runner's signature check:
an accepted keyword alone does not prove the generator actually replays traces.

For free generation, use all six shared prompts with exact chat rendering and a pinned HF
control, at least 64-128 generated tokens when feasible, and repeated reuse of the same TT
generator to expose stale request state. Leave prompt-format metadata, rendered prompts/IDs,
HF and TT completions, and a concrete per-prompt review. Write standard
`autoregressive_meta.json`/`tt_completion.txt` artifacts or explicitly include the qualitative
outputs in the degeneracy checker: arbitrary JSON filenames are not automatically discovered.
Run the restored checker by file path with explicit `--model-dir`, `--scope autoregressive`,
and `--missing-artifacts critical`. Its exit codes are 0 clean, 1 advisory, 2 critical, and 3
checker error; a machine-clean result still requires qualitative review.

## Refuted inherited hypothesis: wrong HF architecture necessarily means random weights

The prior Ornith work log and `hf_model.py` describe `AutoModelForCausalLM` silently loading
unmatched multimodal checkpoint keys. The current checkpoint indeed declares
`Qwen3_5ForConditionalGeneration`, stores text weights under `model.language_model.*`, and
maps through `AutoModelForCausalLM` to `Qwen3_5ForCausalLM`. Those facts alone do **not** prove
the old failure here.

A cheap current-environment control constructed `AutoModelForCausalLM.from_config(config)`
under `torch.device('meta')`, then ran the installed loader's key-renaming functions against
the safetensors index. The model owns `Qwen3_5TextConfig` with type `qwen3_5_text`.
Transformers `conversion_mapping.py:782` supplies
`PrefixChange(prefix_to_remove='language_model', model_prefix='model')`; it maps all **427**
model state keys, with **zero missing mapped keys**. For example,
`model.language_model.embed_tokens.weight` becomes `model.embed_tokens.weight` and
`lm_head.weight` remains unchanged. No tensor storage or checkpoint weights were loaded.
The text-only parameter count was 8,953,803,264.

Thus importing the old architecture resolver is not a proven mandatory repair. Selecting
the checkpoint-declared class can still be an explicit reference policy, and the prior helper
is a useful starting point; if used, fix its `AutoConfig.from_pretrained` call to receive the
same revision/local-only policy as its model call. Do not copy its outdated random-weights
claim as evidence about this environment. Full loading diagnostics and a real HF generation
remain required; this meta control proves key coverage, not numerical model correctness.

## Experiments and focused follow-through

Executed host-only checks:

- `git ls-tree`/`git show` verified both pinned commits and AST import scanning established
  the dependency closure. Exact current/pinned AIME bytes were compared and hashed.
- With `HF_HUB_OFFLINE=1`, the actual local config/tokenizer reproduced the BatchEncoding
  normalization error and established the 161-token prompt without loading any model weights.
- The meta-model key-conversion control above refuted the inherited unmatched-weights claim.
- The original scoring modules were loaded directly from `git show` into isolated in-memory
  module objects. A four-position K=100 reference with predictions at reference ranks 1, 2,
  6, and outside the row produced `top1=0.25`, `top5=0.50`, `top100=0.75`, `total=4`, `k=100`
  from both `_run_one_entry` and `_run_one_entry_prefill`. The teacher callback returned the
  expected four ground-truth next-input IDs. `ttnn` was absent from `sys.modules` afterward.

An initial plain interpreter import encountered the host's missing passwd entry for UID 1002
inside torch's default cache-directory lookup. Setting
`TORCHINDUCTOR_CACHE_DIR=/tmp/ornith_readiness_inductor` allowed these host-only probes. This
is environment setup, not a readiness source defect; use the established stage environment
for real checks.

Suggested repair experiment, in order:

1. Restore only the closure and five host tests listed above; run imports/`--help` and existing
   tests with `pytest --noconftest` so unrelated repository device fixtures are not involved.
2. Apply only the verified prompt-result normalization change. Add a focused test using a
   `BatchEncoding` plus the legacy list return shape; prove both produce identical IDs.
3. Use the pinned local snapshot and emit a provenance sidecar. If adding remote revision
   support, use mocked model/config/tokenizer loaders to prove every call receives the pin.
4. Have a model-local evidence driver own profile opening and build kwargs, exact chat
   rendering, threshold assertions and machine-readable metric artifacts. Test its scoring
   and metadata path with a fake generator before any hardware run.
5. Main agent: run the reduced model smoke, then the actual fresh HF reference and each
   required all-layer/profile gate. Preserve failure evidence and only claim completion after
   numerical gates, real trace execution, qualitative review and degeneracy checks pass.

No accelerator correctness, latency, memory capacity, or full-model readiness result is
claimed by this report.
