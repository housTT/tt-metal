# AutoFix Report: readiness dependency

## Starting Evidence

- Diagnosis: `AUTODEBUG_readiness.md` in this directory, inspected checkout
  `c61ea5a4ca101af524f7518923b993b71225c81c`.
- The common readiness package was absent. Pinned workflow source:
  `70a596f92229ada922fba743cd0cd9d2658a5c1c`.
- Scope: restore the diagnosis's 13 non-serving source/data files and five host
  tests, then verify and repair tokenizer output normalization. No model or
  generator implementation changed in this repair.

## Hypothesis Experiments

### Missing dependency

- Hypothesis: restoring the pinned non-serving dependency closure resolves imports
  and provides working existing host checks.
- Experiment: restored the exact 18 files listed in `AUTODEBUG_readiness.md`.
  Ran all five test files with `pytest --noconftest`; 19 passed in 1.85 s.
- Verdict: verified. Log: `readiness_restored_tests.log`.
- Serving runner and its tests were not restored. `contract_vllm.py` is the existing
  lightweight package dependency and imports no vLLM runtime.

### BatchEncoding normalization

- Hypothesis: the existing helper iterates BatchEncoding keys as token IDs.
- Experiment: called `_chat_or_plain_prompt_tokens` with actual Transformers
  `BatchEncoding({'input_ids': [3, 5, 8], 'attention_mask': [1, 1, 1]})`.
  The unmodified helper raised `ValueError: invalid literal for int() with base
  10: 'input_ids'`. Log: `readiness_normalization_repro.log`.
- Verdict: verified.
- Fix: extract `input_ids` from mapping output, normalize tensor output to a list,
  unwrap one batch row, and reject multiple rows rather than silently dropping
  prompts. Existing bare list output is preserved.
- Regression: parameterized legacy list, nested list, tensor, BatchEncoding list,
  and BatchEncoding tensor cases; empty output and unexpected batch rejection.
- Verification: 28 host tests passed in 1.83 s. Log:
  `readiness_repaired_tests.log`.
- Actual pinned local tokenizer control: snapshot
  `/home/hous/dev/ornith-1.5-9b/upstream`, offline/local-only load, AIME24 prompt 0
  produced exactly 161 chat-template tokens, equal to the direct tokenizer
  `input_ids`. Log: `readiness_host_checks.log`.

## Commands

From repository root, both baseline and final test runs used:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m pytest --noconftest \
  models/common/readiness_check/test_generate.py \
  models/common/readiness_check/test_schema.py \
  models/common/readiness_check/test_run_prefill_check.py \
  models/common/readiness_check/test_run_teacher_forcing.py \
  models/common/readiness_check/test_check_degenerate_output.py -q
```

Host helper checks imported every restored source module and asserted `ttnn` was
absent from `sys.modules`. CLI checks used the same environment and
`HF_HUB_OFFLINE=1`, with this command for each entry point:

```bash
python_env/bin/python -m models.common.readiness_check.generate --help
python_env/bin/python -m models.common.readiness_check.run_prefill_check --help
python_env/bin/python -m models.common.readiness_check.run_teacher_forcing --help
python_env/bin/python -m models.common.readiness_check.run_autoregressive --help
python_env/bin/python -m models.common.readiness_check.check_degenerate_output --help
```

All returned zero. Black check passed on the two modified files:

```bash
python_env/bin/python -m black --check \
  models/common/readiness_check/generate.py \
  models/common/readiness_check/test_generate.py
```

`python_env/bin/python -m isort --check-only --profile black` on those same files
could not run: isort is not installed. No dependency was installed. No build was
required for these Python/data-only changes. The tests emit existing SWIG
DeprecationWarnings, with no functional test failures.

## Final Status

Fixed within this repair's bounded scope. Sixteen restored files are byte-identical
to the pinned workflow commit; only `generate.py` and `test_generate.py` differ.
No weights loaded, no TTNN import, no device command, no serving work, no remote
checkpoint load, and no reference generation occurred. Main stage owner must
review this diff, then produce the HF reference with pinned local snapshot and
provenance sidecar and conduct full-model hardware gates. These host checks do
not prove accelerator correctness or model quality.

## Authorized end-to-end reference verification (2026-09-05)

After the main agent reviewed and kept the repair, it authorized the original
HF reference generation. This follow-through supersedes the earlier bounded
repair statement that no weights/reference were loaded/generated.

Executed:

```bash
HF_HUB_OFFLINE=1 \
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
OMP_NUM_THREADS=8 PYTHONPATH=. python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/generate_hf_reference.py

TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/verify_hf_reference.py
```

Both returned zero. The first script calls the restored `generate_reference`
with AIME24 prompt 0, chat template, `gen_len=100`, `top_k=100`, and explicit CPU.
It observes the original AutoModel loader with `output_loading_info=True`; no
architecture/loader replacement, dtype override, remote load, or dependency
installation was used. The installed Transformers loader selected
`Qwen3_5ForCausalLM`, BF16, with 8,953,803,264 parameters and zero missing,
unexpected, or mismatched keys and no load errors. Generation and scoring took
47.372 seconds; this CPU setup-inclusive reference duration is not TT performance.

Saved artifacts (relative to the autoport):

- `readiness_aime24_chat.refpt`: fresh 161-token prompt, exactly 100 generated
  tokens, and top-100 IDs at all 100 scoring positions.
- `readiness_aime24_chat.meta.json`: canonical HF ID and revision, local snapshot,
  checked local download revision receipts, tokenizer/class/template identity,
  source and metadata hashes, exact arguments/command, rendered prompt/token IDs,
  completion, load diagnostics, and reference SHA-256.
- `doc/full_model/aime24_chat_prompt.txt` and `aime24_hf_completion.txt`.
- `doc/full_model/readiness_hf_generation.log` and
  `readiness_hf_verification.log`.
- `doc/full_model/generate_hf_reference.py` and `verify_hf_reference.py` preserve
  the exact executed scripts. Source hashes refer to their executed contents.

Read the actual HF completion under `$qualitative-check`: it is coherent English
mathematical setup, deriving `9/s = 4 - t/60`, with no mechanical duplication,
collapse, wrong-language drift, or control-token leakage observed. It ends
mid-sentence because the requested 100-token limit is reached. This is a partial
reasoning control, not proof of a final correct AIME answer. HF's optional fast
path warning is explained by the CPU reference using the installed torch
implementation; it is not a TT runtime fallback. No TT comparison or six-prompt
shared suite is claimed here; those remain full-model stage work.

HF self-control observation: cached greedy generation versus full-prefill scoring
agrees at 99/100 top-1 positions and 100/100 top-5/top-100 positions. The only
rank-1 difference is generated index 82 (logit position 242): cached generation
ID 13 is rank 2 under full-prefill scoring, whose rank 1 is ID 3979. This difference
is present wholly within the pinned BF16 HF control, without TT execution. Both
the generated teacher tokens and the independent full-prefill top-k ranking are
preserved in the reference. The sidecar records the exact position and rank;
this does not justify waiving any TT top-5/top-100 gate. Its numerical mechanism
was not investigated because the HF reference meets those rank thresholds.

## Finding 4 follow-through: chat-correct autoregressive runner

Hypothesis: passing a pre-rendered chat prompt file through the existing runner's
`.strip()` changes the checkpoint template's input IDs. Verified offline with
the pinned local tokenizer: `Explain why the sky is blue.` renders with suffix
`<|im_start|>assistant\n<think>\n`. The exact chat prompt has 18 tokens; stripping
it removes final token 198 and leaves 17. Artifact:
`readiness_ar_strip_repro.log`.

Smallest repair: added explicit `chat_template=False` to programmatic
`run_autoregressive`, plus CLI `--chat-template`. Chat mode reads the raw user
prompt file without stripping it, then calls the existing normalized
`_chat_or_plain_prompt_tokens` helper. Both HF and TT receive the same exact IDs.
Metadata adds `chat_template`, `prompt_mode`, `tokenizer_class`, and the exact
`rendered_prompt`, alongside the existing raw `prompt_text` and token IDs. The
legacy default retains its stripped plain prompt and completion behavior.

Added `test_run_autoregressive.py`: mocked model/tokenizer loaders exercise the
real HF runner and real shared prompt helper, assert identical HF/TT token IDs,
check the legacy default, preserve generation/teardown behavior, and verify
saved rendered prompt, metadata, and HF/TT outputs. Two focused cases passed.
A separate pinned actual-tokenizer control with mocked HF/TT generation verified
all 18 chat IDs reached both paths unchanged. It loaded no weights and asserted
TTNN was absent from imports. Fake completion artifacts were removed immediately
after assertions to prevent later qualitative discovery treating host fixtures
as real generated output; their log names the temporary paths.

Executed host verification:

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m pytest --noconftest \
  models/common/readiness_check/test_run_autoregressive.py -q

TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m pytest --noconftest \
  models/common/readiness_check/test_generate.py \
  models/common/readiness_check/test_schema.py \
  models/common/readiness_check/test_run_prefill_check.py \
  models/common/readiness_check/test_run_teacher_forcing.py \
  models/common/readiness_check/test_check_degenerate_output.py \
  models/common/readiness_check/test_run_autoregressive.py -q

TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m models.common.readiness_check.run_autoregressive --help
```

Results: focused tests 2 passed; all host tests 30 passed (2.12 s); CLI exposes
`--chat-template`. Black formatted the new test and left the runner unchanged.
Artifacts: `readiness_ar_tests.log`, `readiness_all_host_tests.log`,
`readiness_ar_actual_tokenizer.log`, `readiness_ar_cli_help.log`.
Status: finding 4 input-contract bug fixed with host evidence. This repair does
not claim full HF/TT autoregressive qualitative gate completion.
