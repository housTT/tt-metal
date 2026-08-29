# AutoDebug: qualitative EOS degeneracy

## Summary

The failed degeneracy gate is best explained by the qualitative harness forcing
fixed-length generation after the model has already produced its Harmony response
terminator. This is not evidence of a trace/token-feedback defect.

`test_shared_qualitative_chat_suite` calls:

```python
generator.generate(prompt_tokens, 128, enable_trace=True, sampling_mode="device", stop_on_eos=False)
```

and then asserts `len(completion_tokens) == 128`. For prompt 4, the completion is
valid through the first `<|return|>` token, but the harness keeps decoding. The
recorded output then emits repeated `<|return|>` markers and starts another
assistant turn, which is exactly the failure shape that
`check_degenerate_output.py` flags.

## Direct observations

- Artifact:
  `models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_tt_chat.json`
  prompt 4 decodes a complete French answer before the repeated special-token
  tail.
- Prompt 4 has 128 completion tokens. Token `200002` (`<|return|>`) appears at
  positions 104 through 113. Token `200007` (`<|end|>`) appears at positions 81
  and 114.
- Truncating prompt 4 at the first `200002` gives 105 output tokens including
  EOS. A local static metric check on that truncated text gives rough
  `adjacent_duplication=0.0` and `trigram_loop_fraction=0.0`; the recorded full
  artifact reports `adjacent_duplication=0.1084` and
  `trigram_loop_fraction=0.1071`.
- `models/autoports/openai_gpt_oss_120b/tt/generator.py` already supports
  early stop: it builds `eos_ids` from `self.model.hf_config.eos_token_id` and
  returns/breaks when `stop_on_eos` is true and the predicted token is in that
  set.
- The pinned checkpoint `config.json` has `eos_token_id: 200002`.
  `tokenizer.json` maps `200002` to `<|return|>`, `200007` to `<|end|>`,
  `199999` to `<|endoftext|>`, and `200012` to `<|call|>`.
- The pinned `generation_config.json` has
  `eos_token_id: [200002, 199999, 200012]`, `pad_token_id: 199999`, and
  `bos_token_id: 199998`.
- Trace evidence for prompt 4 is internally consistent with normal traced
  on-device sampling: `decode_calls=127`, `trace_replays=127`,
  `full_input_refreshes=1`, `page_table_reuses=126`,
  `sampled_token_readbacks=128`, and no host argmax/full-logit readbacks.
- The split-greedy comparison artifact shows on-device sampled predictions
  exactly match host argmax for 16 forced steps, which directly exercises the
  sampling/token-feedback path.
- `models/tt_transformers/tt/generator.py` binds the sampling output token back
  into the decode token input (`tt_out_tok`) for traced on-device sampling, and
  preserves that device-produced token/position during steady replays.

## Root cause

The qualitative test asks for a fixed 128 generated tokens with
`stop_on_eos=False`. GPT-OSS/Harmony completions are allowed to terminate before
that length. Prompt 4 does terminate: it produces `<|return|>` at token index
104 after a complete final answer. Continuing past that terminator makes the
model generate control-token tail content, including repeated `<|return|>`,
which pushes the adjacent-duplication metric above the gate threshold.

## Ranked hypotheses

1. **Confirmed: harness ignores EOS and requires fixed output length.** This
   explains the complete answer followed by repeated `<|return|>` markers, the
   exact prompt 4 failure, and the gate metric. Minimal fix is in the
   qualitative harness.
2. **Confirmed follow-up gap: generator stop ids are narrower than the checkpoint
   generation config.** `Generator.generate` uses `hf_config.eos_token_id`
   (`200002`) and not `generation_config.eos_token_id`
   (`[200002, 199999, 200012]`). This does not explain prompt 4, because prompt
   4's first real terminator is `200002`, which is covered. It is still a
   robustness gap if a future completion emits `<|endoftext|>` or `<|call|>` as
   an EOS condition.
3. **Low likelihood: `<|end|>` should be treated as EOS.** `<|end|>` is a
   Harmony message-boundary token (`200007`), not listed in
   `generation_config.json` EOS. The artifact contains `<|end|>` before the
   assistant final message, so stopping on every `<|end|>` would truncate before
   the visible answer. Do not add `200007` blindly to EOS for chat generation.
4. **Refuted by current evidence: trace/token-feedback stale-token defect.** The
   recorded evidence shows steady traced sampling behavior, prior artifacts show
   device-vs-host greedy parity under forced decode, and the repeated tail begins
   only after a valid terminator that the harness intentionally ignores.

## Minimal fix

Change only the qualitative gate harness:

- Call `Generator.generate(..., stop_on_eos=True)`.
- Replace the fixed `len(completion_tokens) == 128` assertion with a variable
  length assertion, for example `1 <= len(completion_tokens) <= 128`.
- Keep the non-collapse guard (`len(set(completion_tokens)) > 8`) or adapt it so
  very short but valid EOS completions are not rejected accidentally.

Optional generator hardening:

- Load or store the checkpoint generation EOS set so `stop_on_eos=True` checks
  `{200002, 199999, 200012}` rather than only `config.json`'s `{200002}`.
- Do not include `200007` (`<|end|>`) unless a separate chat-template contract
  proves that it is safe to stop there in all expected generation states.

## Minimal repro from existing artifact

Prompt 4 in `qualitative_tt_chat.json` is already a repro:

```text
Translate the following to French: "Hello, how are you today?"
```

The generated sequence contains the first `<|return|>` (`200002`) at completion
token index 104. With `stop_on_eos=True`, `Generator.generate` would return at
that point with a 105-token completion. With the current test setting
`stop_on_eos=False`, generation continues to 128 tokens and produces the
repeated `<|return|>` tail that trips the degeneracy check.

## Verification command

No hardware was used in this investigation. The hardware verification should
rerun the qualitative gate after the harness change:

```bash
GPT_OSS_120B_FULL_MODEL_QUALITATIVE=1 pytest -q models/autoports/openai_gpt_oss_120b/tests/test_full_model.py::test_shared_qualitative_chat_suite
python models/common/readiness_check/check_degenerate_output.py \
  --input models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_tt_chat.json \
  --output models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/degenerate_check.json
```

Expected result: prompt 4 stops at the first `200002` terminator, the artifact
accepts variable completion length, and the degeneracy check no longer reports
the repeated `<|return|>` tail as a critical finding.

## Scope notes

The repo-local fresh AutoDebug runner was started as requested, but its nested
Codex session hit a local sandbox/read failure and then stalled on delegated
inspection. Per the priority update, it was interrupted and this report was
finalized from direct source/artifact inspection in the parent session.
