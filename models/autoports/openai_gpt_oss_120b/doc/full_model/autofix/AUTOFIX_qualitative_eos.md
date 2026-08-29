# AutoFix: qualitative EOS tail

## Failure

The first six-prompt qualitative artifact failed the machine checker on the
French translation with `adjacent_duplication=0.1084`.  The visible translation
was correct, followed by repeated `<|return|>` markers and a second assistant
turn.

## Isolated diagnosis

Fresh-context AutoDebug produced `AUTODEBUG_qualitative_eos.md`.  It confirmed
that the harness called `generate(..., stop_on_eos=False)` and asserted exactly
128 tokens.  The first real EOS was `<|return|>` (`200002`) at token 104.  Trace
evidence remained healthy: one initial refresh, 126 steady page-table reuses,
zero host argmax calls, and zero full-logit reads.  The earlier split-greedy test
also remained exact against host argmax.

The checkpoint generation config declares EOS ids `200002`, `199999`, and
`200012`.  `<|end|>` (`200007`) is a message boundary and is intentionally not
treated as EOS.

## Retained fixes

- The qualitative suite now requests `stop_on_eos=True` and accepts a completion
  length from 1 through 128.
- `Generator._eos_token_ids()` now prefers the complete HF generation-config
  stop set, with model-config and tokenizer fallbacks.
- A host test fixes the expected stop set `{200002, 199999, 200012}`.

## Verification

The focused host EOS/sampler tests passed.  The full six-prompt hardware suite
then passed; the translation stopped at 81 tokens.  The all-scope degeneracy
checker reported `No degenerate output detected.`  All four devices remained
healthy with zero uncorrectable GDDR errors and no external device process.
