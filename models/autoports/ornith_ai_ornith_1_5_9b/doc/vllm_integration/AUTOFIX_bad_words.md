# AutoFix: bad-word serving integration

Date: 2026-09-08.

## Starting evidence

- `AUTODEBUG_bad_words.md`: original full-suite greeting assertion failure.
- `AUTODEBUG_bad_words_history.md`: separately verified missing decode output
  history for multi-token bad words with neutral penalties.
- `full_b32_sampling_full_v1.log`: 69 passed, 3 failed, 1 skipped. Two failures
  concern presence penalties and are investigated separately.

## Hypothesis experiments

**Greeting mask failure — refuted.** The parent repeated the original five
requests with only generated token-ID response metadata added. The same seed-4
text assertion failed, but all five actual token-ID streams contained zero
forbidden IDs. The apparent `hello` was generated as 71 (`h`) and 4638 (`ello`),
not either banned singleton. CPU execution of the actual upstream tokenizer
update method and bad-word masking code confirmed all 12 forbidden variants
were masked. Evidence: `bad_words_original_token_ids.json`,
`bad_words_host_sampler_cpu_probe.json`, and `bad_words_tokenizer_probe.json`.

The canonical test was corrected to assert actual contiguous generated token
sequences against variants from
`SamplingParams.update_from_tokenizer(get_tokenizer(tt_model_name))`. All five
requests and nonempty content checks remain; generated token IDs must also be
present and nonempty. Negative controls reject actual banned singleton and
multi-token sequences while accepting the proven alternate segmentation.
No model-specific exception or skipped check was introduced.

**Missing multi-token output history — verified and fixed.** The TT plugin
only prepared decode output history for non-neutral penalties. The existing
history condition now also includes nonempty bad words. The ordinary device
sampling path still makes no history copies. CPU regression before the fix:
1 failed, 3 passed; afterward: 4 passed. Adding assertion negative controls
produced 7 passing CPU tests. Evidence: `bad_words_history_before.log`,
`bad_words_history_after.log`, `bad_words_host_regressions.log`, and
`bad_words_history_source_probe.json`.

The parent ran this exact focused API client before and after restarting the
all-layer TP4/P300c async server to load the source fix:

```sh
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_history_request_probe.py --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_history_before_server.json
USER=hous ../state/serving-env/bin/python models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_history_request_probe.py --output models/autoports/ornith_ai_ornith_1_5_9b/doc/vllm_integration/bad_words_history_after_server.json
```

Before: **exit 1**. Both unbanned and banned requests generated
`[3446,4121,1478,4121,1478,4121]`, violating the forbidden `New York` sequences.
After: **exit 0**. The unbanned control remained identical, while the banned
request generated `[3446,1478,1478,1478,1478,1478]`, containing neither forbidden
sequence. In both runs, the final-token-alone control correctly returned
`[4121]`. Raw request parameters, responses, and exact IDs are preserved in the
two JSON files and matching `.log` files.

The focused requests use neutral penalties and explicit optional host controls
(logit bias and allowed token IDs). They do not alter the measured on-device
sampling path. Forced-vocabulary diagnostic text is not qualitative evidence.

## Final status

**Runtime defect fixed with CPU and live serving evidence.** The greeting
failure is resolved at the assertion boundary using the documented token
contract and negative controls. Parent-owned canonical targeted tests and the
full sampling suite remain the broader stage gate; their result must be read
before declaring the stage complete.

CPU verification command:

```sh
USER=hous ../state/serving-env/bin/python -m pytest ../vllm/plugins/vllm-tt-plugin/tests/test_bad_words_history.py -q
```

No device operations or requests were performed by this source-investigation
agent; the parent serialized both API experiments. No runtime/source changes
were made after inspecting the post-fix evidence.
