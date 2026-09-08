# AutoDebug: penalty activation when admitting a prefill

Source and CPU experiment, 2026-09-08, before runtime modification.

## Verified failure

The actual adapter `prefill_forward` calls `_sampling(params, slots=rows)`
before the actual generator `prefill_forward`. No prompt/generated histories
are supplied. The pinned plugin `model_runner.py::submit_prefill` passes compact
sampling fields and `empty_slots`, but does not pass histories either.

The generator's `configure_sampling` intentionally requires complete histories
when a live stream activates previously disabled penalty tracking. Once any
request has run, `_live=True` remains until a reset, including when that request
has finished. Admitting a fresh penalized request therefore raises before its
prefill can populate the new request's prompt and empty output histories.

`tests/test_prefill_penalty_admission.py` executes the actual adapter class,
generator class, common `SamplingParams`, and common parameter formatter from
their source AST. Only TT kernels and unrelated setup are replaced with torch
boundaries. All six admission cases fail with the predicted
`ValueError: Enabling live penalties requires prompt_token_ids and generated_token_ids`:
presence, frequency, or repetition penalties, each with other rows active or
all prior rows finished. No device import or access occurred.

```bash
USER=hous python_env/bin/python -m pytest -q \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_prefill_penalty_admission.py \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/tests \
  -k adapter_admits --tb=short
```

## Minimal boundary repair

Introduce explicit `fresh_slots` in generator `configure_sampling`. Without
full histories, allow the transition only if **every currently penalized lane
is a freshly admitted slot**. Global penalty tracking was previously disabled,
so these are also exactly the newly penalized lanes. The adapter derives fresh
slots only from prefill rows whose `start_pos == 0` and forwards that declaration
through `_sampling`. Continuation prefills and live parameter changes retain
the existing history requirement. Validate slots before any sampler mutation.

This exception does not invent previous histories or erase ongoing state:
fresh requests have no prior generated tokens, and the immediately following
generator prefill already resets their output counts and loads their prompts
using its existing per-slot sampler preparation. Existing neutral requests
continue with neutral parameters. The existing sampling-only warmup snapshot
and restore behavior remains intact.

Tests also require that wrong/missing fresh-slot declarations still fail,
another live penalized lane cannot borrow a fresh lane's declaration, invalid
slots fail before mutation, and warmup preserves every tracked tensor.
The supervising agent owns the adapter edit and serialized device follow-up.
