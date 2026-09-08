# AutoDebug: multi-token bad words lack output history

Date: 2026-09-08. Separate finding from the original singleton-greeting test.
No runtime implementation edits or server requests have been made by this
investigator.

## Verified source boundary

The TT plugin's `model_runner.py:1231` builds decode output history only under
`not input_batch.no_penalties and not is_prompt`. A nonempty bad-word dictionary
does not activate this path. The host sampling preparation at line 3012 then
passes an empty output history to upstream `apply_bad_words`, even when an
active request already generated the prefix of a multi-token banned sequence.

CPU experiment: AST-extracted and executed the actual runner history block with
a fake batch whose current output history is `[3446]` (the token for `New`).
With neutral penalties and a bad-word sequence `[3446, 4121]` (`New York`),
neither history helper was called and `output_tokens` remained `None`. With
non-neutral penalties, the same block copied the actual history and padded it
with `-1`, as expected. Artifact: `bad_words_history_source_probe.json`.

Independent CPU experiment using the actual upstream
`vllm/v1/sample/ops/bad_words.py`: an empty history leaves token 4121 available;
history `[3446]` masks token 4121 to negative infinity. Artifact:
`bad_words_host_sampler_cpu_probe.json`, `multi_token_history_control`.

Verdict: verified integration defect for multi-token bad words with neutral
penalties. This cannot explain the original six greeting words, whose 12
encoded variants are all singletons.

## Smallest proposed fix, not applied

Allow the existing decode history preparation path when either penalties are
non-neutral or `input_batch.sampling.bad_words_token_ids` is nonempty. This
preserves the current per-request row selection and padding. It also copies
prompt history on the optional bad-word host path; that is unnecessary but
harmless. A more surgical version can copy only output history for bad words,
with corresponding independent padding, but introduces more code.

The normal on-device serving/benchmark path has neutral penalties and no bad
words, so its history-free path remains unchanged.

## Required verification after parent integration

- Rerun the actual source-block CPU probe: neutral penalties plus a multi-token
  bad-word entry must preserve the real prefix history. Neutral parameters
  without bad words must still avoid copying histories.
- A live original-versus-banned greedy chat request encouraging `New York` (or
  another phrase confirmed to occur in the unbanned control), with
  `return_token_ids=true`, must never generate the entire banned token sequence.
  Use the actual generated IDs, not re-encoded output text. The unbanned control
  must first demonstrate that the phrase is reachable.
- Rerun the original host-only tests and final canonical sampling suite.

## Applied fix and CPU regression

The parent authorized the proven fix. `model_runner.py` now activates the
existing history path for nonempty bad words as well as non-neutral penalties.
Only the condition and explanatory comment changed.

New durable regression:
`../vllm/plugins/vllm-tt-plugin/tests/test_bad_words_history.py` relative to the
tt-metal checkout. The tests compile the actual runner history block without
TTNN imports. They verify neutral-penalty multi-token masking using the actual
upstream bad-word processor, per-request row reordering, `-1` padding,
preservation of penalty history, exclusion during prefill, and no history
helper calls for the ordinary device path.

Command from the tt-metal root:

```sh
USER=hous ../state/serving-env/bin/python -m pytest ../vllm/plugins/vllm-tt-plugin/tests/test_bad_words_history.py -q
```

Before the implementation fix: **1 failed, 3 passed**, with the expected missing
history assertion (`bad_words_history_before.log`). After: **4 passed**
(`bad_words_history_after.log`). The first source probe remains preserved as
pre-fix evidence.

## Live pre-fix confirmation

The parent ran `bad_words_history_request_probe.py` against the existing
all-layer32 TP4/P300c async server before restarting it to load the source fix.
The probe returned **exit 1**, confirming the source-localized defect on the
real serving path. Artifacts: `bad_words_history_before_server.json` and
`bad_words_history_before_server.log`.

The unbanned greedy control produced IDs
`[3446, 4121, 1478, 4121, 1478, 4121]`: `New York New York New York`.
With bad words `["New York"]`, the otherwise identical request produced
**the same forbidden sequences**. Both tokenizer variants `[3446, 4121]` and
`[1478, 4121]` were therefore violated. A separate one-token control with only
final ID 4121 available correctly returned ` York`, proving that the intended
fix must be sequence-dependent rather than globally banning the final token.
All three cases had neutral penalties and explicit optional host sampling
through logit bias/allowed token IDs. Actual generated API IDs were inspected.

## Live post-fix confirmation and final disposition

After restarting the full server to load the fix, the parent reran the same
client and received **exit 0**. Artifacts:
`bad_words_history_after_server.json` and
`bad_words_history_after_server.log`. The inspected generated token IDs were:

- Unbanned control: `[3446, 4121, 1478, 4121, 1478, 4121]`, unchanged from
  before the fix (`New York New York New York`).
- Banned phrase: `[3446, 1478, 1478, 1478, 1478, 1478]`
  (`New New New New New New`), with neither forbidden token sequence present.
- Final token alone: `[4121]` (` York`), still legal without the prefix.

The source-localized missing-history defect is **fixed and verified on the
actual serving path**, with the unchanged positive control and legal-final-token
control ruling out a globally banned final token or an unreachable phrase.
The repeated `New` output is intentional under this six-token forced-vocabulary
diagnostic; it is not a qualitative generation result.

The parent is running the canonical targeted tests and full sampling suite.
Their broader stage verdict is recorded separately. No singleton-greeting test
change is included in the runtime history fix. See `AUTOFIX_bad_words.md` for
the consolidated repair disposition.
