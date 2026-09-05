# AutoFix: explicit host sampling and EOS return contract

## Starting evidence

Inspected `tt/generator.py`, the restored readiness `Generator` contract, and
`models/common/sampling/generator.py`'s `SamplingParams`. The original host
compatibility branch called host `argmax` for prefill and every decode token,
even when non-greedy sampling parameters or penalties were supplied. Those
parameters were sent only to the device sampler, which this mode never executed.
This was a source-verified ignored-parameter defect.

The original high-level default was `stop_on_eos=False`; its optional truncation
supported only one tokenizer EOS and would also truncate teacher-forcing output
if enabled. The pinned model/tokenizer have distinct stop IDs. The readiness
contract permits EOS termination for free generation and requires all own
predictions/callbacks for teacher forcing.

## Change

- Added optional `host_sample` at construction and on `generate`. Its contract is
  `host_sample(logits, *, sampling_params, step, prompt_token_ids,
  generated_token_ids)`, returning one integer vocabulary ID per user. Histories
  are copied into per-user lists. The callback owns filtering, penalties, RNG,
  and logprob state; the generator validates returned IDs and feeds them back.
- Host mode with no callback retains plain greedy argmax. Non-greedy parameters,
  penalties, or requested logprobs require an explicit callback and otherwise
  fail before resetting request state. Device mode rejects a supplied host
  callback. No custom non-greedy sampler was introduced.
- Host mode no longer configures the unused device sampler or resets its
  request seed/penalty state. Existing model trace capture/replay remains.
- Free generation defaults to `stop_on_eos=True`, slicing each returned row at
  its first tokenizer or text-model EOS (scalar or list IDs). The internal
  generation window remains fixed, including replay/read behavior. Explicit
  `stop_on_eos=False` returns the whole window for performance measurements.
  Teacher forcing ignores EOS slicing and always returns/invokes every step.
- No changes to device trace capture, replay, token feedback, position advance,
  or page-table mechanics were made.

## Focused verification

```bash
TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache \
python_env/bin/python -m pytest --noconftest \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py -q
```

Six tests passed in 0.56 s. The test compiles the actual source functions/class
via AST while excluding runtime imports and supplies fake TT/model boundaries;
it exercises pure validation and the actual high-level host generation method.
Coverage includes unsupported params, callback history/params, invalid IDs,
callback-controlled feedback, default multi-EOS slicing, fixed-step output, and
complete teacher-forcing callbacks despite EOS predictions. Log:
`generator_host_contract_tests.log`. An initial test collection failed because
pytest reserves the parametrization name `request`; renaming that test variable
resolved the test-only issue before the passing run.

No TTNN import, device command, HF weight load, or build occurred. These tests
prove source-level host orchestration only. The parent stage owner must perform
the final hardware regression of both host and optimized device paths.
