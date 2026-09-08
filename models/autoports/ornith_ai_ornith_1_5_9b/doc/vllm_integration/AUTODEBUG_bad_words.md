# AutoDebug: bad-words sampling gate

Date: 2026-09-08. Source-only fresh investigator; no server requests, TTNN
imports, device operations, or runtime implementation edits during diagnosis.

## Starting evidence

The full 73-test sampling run on the all-layer, TP4, batch-capacity-32 async
server reports
`test_host_only_params.py::TestHostOnlyParameters::test_bad_words FAILED`.
The suite is still running as this initial report is written, so the failure
traceback and exact response are not yet available. Later host-only and logprob
tests pass, and the parent reports the server remains alive. The original log
is `../../readiness_vllm/sampling_tests.log` relative to this directory.

The tested request is five concurrent chat requests, prompt `Say hello to me`,
100 output tokens, temperature 1.0, seeds 0 through 4, and bad words
`hello`, `Hello`, `hi`, `Hi`, `hey`, `Hey`. The helper supplies no other sampling
restrictions. Explicit host compatibility is enabled in the adapter. The
plugin routes bad words to the host sampler and disables the steady async
shortcut for such requests.

## Findings and hypotheses

1. **Exact symptom classification remains pending.** The test checks words
   after stripping punctuation except `>`. This is stronger than the pinned
   vLLM `SamplingParams.bad_words` token-sequence contract. The test already
   exempts BPE-merged `>Hello`, but other merged tokens and multi-token alternate
   segmentations exist. This is a possible false positive, not a waiver.

   CPU inspection of the pinned local snapshot shows all six words, both
   unprefixed and space-prefixed, encode to singleton IDs. Executing the actual
   `SamplingParams.update_from_tokenizer` method and actual
   `vllm/v1/sample/ops/bad_words.py` code on CPU masks all 12 IDs to negative
   infinity, including with empty output history. However, vocabulary tokens
   `_hi` (45649), `_hello` (93383), and `>Hello` (76759) remain allowed under that
   exact token contract. The original test strips `_` and would reject the
   first two. Prefix fragments can also construct the same text with different
   token IDs. This observation alone does not classify the actual failure.

   Evidence: `bad_words_tokenizer_probe.json` and
   `bad_words_host_sampler_cpu_probe.json`. Both probes used
   `USER=hous ../state/serving-env/bin/python` and only CPU Transformers/Torch;
   the source method was AST-extracted to avoid plugin/device imports.

2. **Separate likely defect: multi-token bad words lose their decode history
   when penalties are neutral.** In the pinned TT plugin
   `model_runner.py:1227`, `output_tokens` is populated only when
   `not input_batch.no_penalties`. In `_get_output_tokens` at line 3012, missing
   `model_input.output_tokens` yields empty histories passed into the host
   sampler. Actual upstream bad-word masking at
   `vllm/v1/sample/ops/bad_words.py:15` requires preceding tokens for sequences
   longer than one token. A CPU control with `New York` IDs `[3446, 4121]`
   confirms empty history leaves 4121 available while history `[3446]` masks it.
   This cannot explain leakage of any of the 12 singleton IDs in finding 1.
   Confirm the full lowered history boundary before implementing a fix.

## Focused next experiments, serialized by parent

1. Read the final original traceback and exact response first. Rerun the five
   original requests with the same parameters and only add
   `return_token_ids=true`, an output metadata field supported by the pinned
   chat API. Save every raw response. Compare generated IDs against all 12
   banned IDs and against the exact textual assertion. Do not infer generated
   token IDs by re-encoding returned text.
2. If a banned ID actually appears, inspect the prepared per-row bad-word dict,
   host sampler mask, and scheduler row attribution. Keep the test unchanged.
   If no banned sequence occurs but the textual assertion fails, demonstrate
   the exact alternate token segmentation with the original API token IDs and
   CPU masking control before proposing a token-contract-aware test.
3. Independently run a CPU extraction test through the actual runner's history
   preparation block, with neutral penalties and a multi-token bad-word entry.
   Only after confirming the missing history should the runner be changed to
   carry output history for bad words as well. Then retest the source boundary
   and a targeted live multi-token phrase, plus the original sampling gate.

## Exact serving follow-up

The completed original suite is archived as `full_b32_sampling_full_v1.log`:
69 passed, 3 failed, 1 skipped. The two presence-penalty failures are investigated
separately. The bad-word failure is seed 4 and quotes `"hello to me."` in the
generated reasoning text.

The parent ran the original five requests through `bad_words_request_probe.py`
with only `return_token_ids=true` added. Artifact:
`bad_words_original_token_ids.json` (raw responses and exact generated IDs),
with console summary in `bad_words_original_token_ids.log`. It reproduced the
same seed-4 text and original assertion failure. None of the five outputs
contains a banned ID. At zero-based output positions 8 and 9, the actual IDs
are **71 (`h`) followed by 4638 (`ello`)**. They form `hello` without producing
the banned singleton 14556 (`hello`) or 23066 (` hello`). Re-encoding output
text would conceal this distinction, so the actual API IDs are essential.

Verdict for the original greeting failure: the text assertion rejects a valid
alternate token segmentation under the documented pinned vLLM token-sequence
contract. No serving mask defect is demonstrated by that assertion. The
canonical test should inspect actual generated token subsequences against the
actual bad-word token sequences and retain nonempty response/content checks.
A negative-control host regression must reject a genuine banned singleton or
multi-token sequence. This is a proposed assertion correction, not a skipped
test or a model-specific exemption.

The parent authorized the assertion correction after inspecting this exact
counterexample. The canonical test now gets variants through the actual
`SamplingParams.update_from_tokenizer(get_tokenizer(tt_model_name))`, retains
all five requests, requires nonempty text and generated token IDs, and checks
every banned contiguous token sequence. Only response metadata was added to
the requests. `tests/tt/utils.py` gained the explicit chat response metadata
flag; no sampling parameter changed.

`tests/test_bad_words_history.py` includes negative controls that reject a real
banned singleton and a real multi-token sequence, plus the actual allowed
`h` + `ello` counterexample. Combined with the separate history tests, all
**7 CPU tests pass** (`bad_words_host_regressions.log`). No test is skipped or
given a model-specific exception. Canonical serving rerun is delegated to the
parent.

The independent multi-token history defect now has a proven minimal source fix
and live before/after confirmation; see `AUTODEBUG_bad_words_history.md` and
`AUTOFIX_bad_words.md`. The exact API probe failed before the server restart
and passed afterward: the unbanned `New York` control is unchanged, the banned
sequence disappears, and its final token remains legal alone. Raw evidence is
`bad_words_history_before_server.json` and
`bad_words_history_after_server.json`. The parent owns the remaining canonical
targeted and full-suite reruns.
