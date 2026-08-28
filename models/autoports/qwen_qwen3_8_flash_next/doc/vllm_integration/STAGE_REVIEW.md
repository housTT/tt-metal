# Qwen3.8-Flash-Next vLLM stage review

## Final verdict

**clean-pass**

The final independent xhigh review found no P0, P1, or P2 findings. It inspected
the vLLM, host-weight-cache, TT-device, and qualitative skill contracts; the
adapter/canonical model path; plugin registration and sampling routing; async
decode; readiness logs and JSON; README/work log; lifecycle/cleanup; benchmark
provenance; and the actual prompt-correct outputs. It made no edits and did not
touch TT hardware.

The review specifically closed the three findings from the first pass:

1. Prompt format: `qualitative_prompt_format.json` and
   `qualitative_tt_chat.json` use the checkpoint's exact
   `apply_chat_template(add_generation_prompt=True)` prompt/token IDs with the
   existing HF control. Raw untemplated completions are explicitly secondary
   stress evidence.
2. Active batch/context language: `max_model_len` remains the full 262,144.
   `max_num_seqs_limit.json` classifies active traced batch one as a current
   segmented expert/state ABI limit, not a physical/device/context limit, and
   does not misrepresent the 32-request queued burst as active batching.
3. Trace allocation: the direct full-48 tracker-enabled vLLM rerun completed
   seven HTTP-200 requests and 1,033 trace replays with zero generic active-
   trace warnings, zero unsafe-live-allocation errors, no model-only or host-
   sampling fallback, and a clean post-stop process/device audit.

## Non-blocking residuals

- Multi-active traced serving requires a new multi-user expert-wave,
  recurrent-state, and slot-lifecycle ABI. The adapter fails fast at one active
  sequence; eager batch 32 at context 4,096 remains a capacity/control result,
  not a serving claim.
- The prompt-correct coding case reaches its 256-token cap after producing the
  correct implementation and example in visible reasoning but before its final
  answer. The other two cases stop cleanly. This usability limitation is
  retained in the primary qualitative verdict.
- The final server command used the shared readiness runner in the
  `/home/ttuser/dev/muse-glimmer/tt-metal` checkout while binding the current
  model and plugin through `PYTHONPATH`; release handoff should retain that
  provenance.

## Review history

The first review returned `more-work-needed` for invalid raw-prompt qualitative
evidence, unsupported physical-limit wording around active batch one, and an
unclassified trace-allocation warning. The AutoFix loop investigated each
hypothesis, retained the proven paging/sampling/reset fixes, produced exact
chat-template and tracker-enabled server evidence, corrected shutdown and
capacity language, and then requested this fresh rereview.
