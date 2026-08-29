# AutoDebug: GPT-OSS 120B Teacher-Forcing Accuracy

Date: 2026-08-29
Mode: source-only; no hardware reproduction was run.

The repo-local AutoDebug runner was invoked, but its fresh worker could not
read the checkout because every local shell command failed inside its own
bubblewrap sandbox with `bwrap: loopback: Failed RTM_NEWADDR: Operation not
permitted`. I therefore verified the report claims directly against the local
source in this session. No implementation or test code was intentionally edited
as part of this investigation.

## Headline finding

The earliest likely semantic boundary for the reported failure is the
teacher-forcing call into traced decode with on-device sampling enabled:
`models/autoports/openai_gpt_oss_120b/tt/generator.py::generate` calling
`decode_forward`, which then delegates to
`models/tt_transformers/tt/generator.py::Generator.decode_forward`.

The shared traced decode path has production free-running semantics for
on-device sampling. On a reset step, once a decode trace exists, it may read the
sampled token and current position already stored in the persistent trace input
buffer and treat them as authoritative when their positions are continuous with
the host view. That is correct for vLLM async/free-running decode, where the
host token may lag the device by one step. It is wrong for teacher forcing,
where the callback's returned ground-truth token is the authoritative next input
even when the persistent device token position matches the host position.

The causal chain is:

1. The readiness driver calls `generator.generate(..., next_input=..., enable_trace=True)`.
   Its callback records the TT prediction and returns the reference token for
   the same generated-step cursor.
2. The model-specific generator predicts the prefill token, calls
   `next_input(0, predicted)`, then enters the decode loop with the returned
   forced token and `start_pos = len(prompt)`.
3. Each teacher-forced decode step is passed with `reset_batch=True`, so the
   traced decode path should refresh token, current-position, RoPE-index, and
   page-table inputs from host before replay.
4. Before that refresh, the shared generator's async-ahead merge can replace the
   host token with the previous device-sampled token when `dev_pos == host_pos`
   or `dev_pos == host_pos + 1`.
5. Once a predicted token differs from the reference forced token, subsequent
   decode runs on a free-running prefix while the teacher-forcing accuracy
   harness still compares predictions against the reference prefix. A large
   top-5 drop is then expected even though prefill accuracy and hardware health
   are fine.

This explains the contrast in the prompt: full-sequence prefill, which reads
host-gathered logits for every position, can pass `top5=1.0`; traced
teacher-forcing can fail because the consumed decode token stream has silently
fallen back to device free-running after the first relevant divergence.

## Current local source state

The current workspace already contains the small local intervention that
matches this diagnosis:

- `models/autoports/openai_gpt_oss_120b/tt/generator.py` passes
  `force_host_tokens=teacher_forced` from `generate`.
- The same file's `decode_forward` accepts `force_host_tokens` and, when true,
  marks all local slots in `self._inner._slots_prefilled_since_decode` before
  calling the shared generator. The shared merge already treats those marked
  slots as host-authoritative and forces `use_dev=False` for them.

The existing short contrast artifact
`doc/full_model/artifacts/split_greedy_host_comparison.json` now shows
`exact_match: true` for a 16-token, two-layer AIME teacher-forcing run:
device split-greedy predictions match host-argmax predictions exactly, with
15 decode calls, 15 trace replays, 15 full input refreshes, and 15 forced-token
refreshes. That is strong evidence that the host-authoritative marker fixes the
earliest teacher-forcing semantic drift for the short trace path. It is not a
substitute for rerunning the full 36-layer readiness gate.

## Boundary checks

### Callback alignment

The shared readiness callback is aligned. `next_input(step, predicted)` records
the current TT prediction via `TokenAccuracy.collect_predicted_tokens`, then
returns `generated_tokens[cursor]` and advances the cursor. In the generator,
step 0 is the prefill prediction and decode step 1 consumes generated token 0
at position `len(prompt)`. That is the standard teacher-forcing alignment.

### reset_batch and teacher-token refresh

`reset_batch=True` is passed for every teacher-forced decode step. In the shared
trace path, `reset_inputs = reset_batch or not on_device_sampling or
sampling_mode_changed`, so a teacher-forced step reaches the full
`prepare_decode_inputs_host` plus `copy_host_to_device` refresh. The bug is not
that reset refresh is absent; the bug is that, without the host-authoritative
marker, the async-ahead merge can mutate `tokens` before the refresh happens.

### Current-position advancement

For on-device logits, GPT-OSS pads the logits batch for `TTSampling`, then calls
`_increment_decode_positions_device(current_pos, rot_mat_idxs)`. The host loop
also increments `start_pos += 1` after each step. With the forced-token marker,
the next reset refresh restores the intended host position and token. Without
it, the position can remain continuous while the token is taken from the device
free-running buffer, which is exactly the dangerous mixed semantic state.

### Cache and page-table updates

The page table is fixed for the single-entry readiness run. The traced path only
uses page-table-only refresh when the page table changes and the token/position
state is intentionally device-owned. In the teacher-forced run, all decode
steps are full refreshes. The KV cache is updated at the token/current-position
pair selected by the merge. Therefore cache divergence is a consequence of the
wrong token becoming authoritative, not the earliest cause.

### Production free-running semantics

Production on-device free-running intentionally differs from teacher forcing.
In free-running mode there is no `next_input`; the sampled token written into
the persistent decode token buffer is the right next token, and page-table-only
or reuse paths avoid unnecessary host traffic. Teacher forcing must override
only that token-ownership rule while preserving trace replay, current-position
advance, cache update, and page-table behavior.

## Smallest correct fix

For the autoport path, the smallest correct fix is the current local
host-authoritative marker:

1. Thread a `force_host_tokens` boolean from `generate` when `next_input is not
   None`.
2. Before delegating to the shared generator, mark the active slots as freshly
   host-supplied so the shared async-ahead merge keeps the host token and host
   position for those slots.

That keeps production free-running behavior unchanged and changes only
teacher-forced reset steps.

The cleaner long-term shared fix would be to make
`models/tt_transformers/tt/generator.py::Generator.decode_forward` accept an
explicit `force_host_tokens` or `host_tokens_authoritative` parameter and apply
it directly in the async-ahead merge. The current local marker is smaller and
uses the shared generator's existing "freshly prefilled slots use host tokens"
contract, but it is implicit.

## Minimal verification plan

1. Run the two-layer short-reference contrast:
   `test_real_weight_two_layer_split_greedy_matches_host_argmax`.
   Expected: device predictions equal host-argmax predictions, and trace
   evidence shows a full input refresh for every forced decode step.
2. Add temporary token-index instrumentation at the shared merge and refresh
   boundary: log `step`, host token, device token, host position, device
   position, `use_dev` before slot marking, and final consumed token. A failing
   pre-fix run should show `use_dev=True` on the first post-capture forced step
   where the consumed token equals the previous device prediction rather than
   the reference token. A fixed run should show `use_dev=False` for the forced
   slot.
3. Run a 36-layer short slice of the AIME reference, preferably 8 to 16 tokens,
   with the same instrumentation. This checks that the first full-stack miss no
   longer changes the consumed prefix.
4. Rerun the canonical full 36-layer gate
   `tests/test_full_model.py::test_run_teacher_forcing_aime24_top100`.
   Expected: `top5 >= 0.98` and `top100 == 1.0`.
5. Run or inspect the production token-out smoke without `next_input`.
   Expected: no forced-token marker is set, page-table reuse behavior remains,
   and sampled-token feedback still drives free-running generation.

## Demoted hypotheses

- Callback off-by-one: demoted. The driver/generator step mapping is coherent,
  and the short artifact's host/device teacher-forcing match would not recover
  from an off-by-one label bug.
- Missing trace input refresh: demoted. `reset_batch=True` forces full refresh;
  the issue is token ownership before the refresh.
- Page-table drift: demoted for the single-entry readiness run because the page
  table is stable and all teacher-forced decode steps are full refreshes.
- Split-sampler argmax mismatch: demoted as the headline after the current
  short artifact shows device split-greedy equals host argmax for 16 forced
  tokens. If the full 36-layer run still fails after the host-authoritative
  marker, this becomes the next boundary to inspect by comparing host logits,
  gathered top-k candidates, normal device sampling, and force-argmax sampling
  at the first failing token.
- Decode-trace capture precompile clobbering the token feedback buffer: demoted.
  The first real replay is a full reset and should overwrite the persistent
  token/current-position inputs before execution. It remains worth inspecting
  only if instrumentation shows cache divergence before any forced-token mismatch.
