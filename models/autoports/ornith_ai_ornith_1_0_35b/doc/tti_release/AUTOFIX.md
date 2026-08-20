# AutoFix Report — Stage 11 (TTI release) EngineCore deadlock

Subject: `VLLM::EngineCore` spins forever in `_process_engine_step` with `model_executed=False`
while one request stays RUNNING, during the TTI sweep point isl=16384 osl=128 max-concurrency=16.

## Starting Evidence

- No `AUTOTRIAGE.md` / `AUTODEBUG.md` existed. tt-triage was already captured for the live hang at
  `doc/tti_release/triage/tt-triage.txt` + `triage-summary.txt`: all four Blackhole devices IDLE,
  `dump_running_operations` pass, eth/arc pass. A fresh triage pass was therefore *not* run — the
  captured evidence already localises the stall to the host, and the process is dead.
- `/home/ttuser/dev/ornith/tti-release/ornith-1-0-35b/HANG_BRIEF.md` (symptom brief).
- `evidence/hang/server_hang_16511.log`, `pyspy_enginecore{,_locals}.txt`, `failing_bench_command.txt`.
- Original failing command: `vllm bench serve --backend openai-chat --dataset-name random
  --max-concurrency 16 --num-prompts 32 --random-input-len 16384 --random-output-len 128 ...`
  (reached 30/32, then hung).

**Pre-existing working-tree state (important):** a *previous session* of this stage had already left an
unverified candidate fix on disk before this AutoFix run started (mtimes 18:09–18:11Z):
`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py` (+65),
`.../model_runner.py` (+25), and a staged tt-metal test file (+106). None of it was authored by this
AutoFix run and none of it had been verified against the failure. This run treated it as a hypothesis.

## Root cause (verified)

Under async scheduling the plugin's `TTScheduler._preempt_request` throws away the decode tokens still in
the pipeline for the preempted request (they were computed against KV the preempt just freed), but
`TTModelRunner` had already appended those same tokens to its own
`CachedRequestState.output_token_ids` and never reconciles on resume. The runner's copy of the request is
then one token longer than the scheduler's, so on the resume prefill
`prompt_lens (= num_computed + num_scheduled) < input_batch.num_tokens`, the prefill is misclassified as
an unfinished chunk (`intermediate_prefill_mask`), the runner publishes **no** sampled token (and falls
back to host sampling, which is why exactly those prefills logged `host sampling`), the async
`num_output_placeholders` for that step is never retired, and
`Scheduler.schedule()`'s max-tokens guard
(`num_computed_tokens + 2 - num_output_placeholders >= num_prompt_tokens + max_tokens`)
then skips the request on every subsequent step: one request RUNNING, zero tokens scheduled, devices
idle, forever.

Two amplifiers specific to this configuration:

1. At isl=16384 with concurrency 16 the KV pool (4128 blocks x 64 = 264,192 tokens) is almost exactly
   saturated by 16 x 16384 = 262,144, so block-boundary crossings during decode force preemption.
2. `TTScheduler._schedule_prefill_only` rebuilds `self.running` as `partial_prefills + pure_decodes`,
   which moves the *oldest* decode requests to the tail of the running list. The base scheduler's
   preemption victim is `self.running.pop()`, i.e. the request **closest to finishing** — exactly the
   request for which the max-tokens guard bites.

The two odd prefill lengths in the hang log are explained exactly: with `truncate_prompt_tokens=16384`
every prompt is 16384, so 16448 = 16384+64 and 16511 = 16384+127 can only be resume prefills of
preempted requests carrying 64 and 127 accepted output tokens. Only the 16511 one hung, because
16511 + 2 - 1 = 16512 >= 16384 + 128 = 16512 (guard fires), while 16448 + 2 - 1 = 16449 < 16512
(guard does not fire, so that request recovered with one lost step). Nothing about "not a multiple of
128" is involved.

## Hypothesis Experiments

### H1 — async-decode output accounting drops/duplicates a sampler output
- **Experiment:** built a host-only driver (`doc/tti_release/repro/hang_scheduler_sim.py`) that runs the
  **real** `TTScheduler` with `async_scheduling=True` against a faithful mock of the runner's
  bookkeeping: the real `apply_cached_req_state_update`, the runner's unconditional append of every
  sampled token, `InputBatch.num_tokens` seeding on `add_request`, the
  `prompt_lens < num_tokens -> intermediate chunk -> no token` rule, a depth-1 async output pipeline, and
  the benchmark client's continuous-arrival pattern. Instrumented every resume with
  `len(runner.output_token_ids)` vs `CachedRequestData.num_output_tokens`.
- **Result:** every preempt-with-a-token-in-flight produced `drift=1` at resume, and each such resume
  produced an empty prefill output. In the canonical config the run then deadlocks with
  `running=[('r4', num_computed=386, num_tokens=386, placeholders=1, out=34, max_tokens=35,
  prompt=352)]` — 386 + 2 − 1 = 387 >= 352 + 35 = 387, i.e. the identical guard arithmetic and the
  identical "1 RUNNING, 0 scheduled, forever" signature, after 31 of 32 requests completed
  (production: 30 of 32).
- **Verdict:** **verified** (in the specific form "runner runs ahead of scheduler after a discarded
  in-flight token", not "a sampler output is lost on a device/host sampling switch").
- **Evidence:** `doc/tti_release/repro/ab_sweep.txt`.

### H2 — `num_computed_tokens` rounded/padded for a prompt length that is not a multiple of 128
- **Experiment:** derived the two host-sampling lengths from the log against
  `truncate_prompt_tokens=16384`; reproduced the same hang in the simulator with `block_size=16` and
  lengths such as 386/387 that have nothing to do with 128.
- **Result:** 16448 and 16511 are prompt+output of resumed preempted requests, not user prompt lengths;
  the hang reproduces at arbitrary alignments.
- **Verdict:** **refuted.** No alignment/rounding is involved and nothing was padded or bucketed.

### H3 — slot/lane bookkeeping leak around the host-sampling prefill
- **Experiment:** the simulator contains no slot/lane logic whatsoever and reproduces the failure
  exactly; the server log also shows the slots being reused normally (`slots [1]`, then `slots [2]`,
  `remap_state_slots: remapped 3 recurrent state row(s)` after the stall began).
- **Verdict:** **refuted.** The stall is entirely in vLLM scheduler/runner state.

### H4 — the device→host→device `check_perform_device_sampling` transition corrupts sampling state
- **Experiment:** read `check_perform_device_sampling` and its caller. `perform_device_sampling` is
  forced `False` at `model_runner.py:1234` *because* `intermediate_prefill_mask.any()`; the mask is
  computed from `prompt_lens < input_batch.num_tokens`.
- **Result:** `host sampling` is a **downstream symptom** of the same drift, not a cause; the trace
  re-capture and `reset_trace` lines are ordinary consequences of the following prefill.
- **Verdict:** **refuted as a cause.** (Consistent with it: exactly the two resumed prefills, and no
  others, took the host path.)

## Fix kept

Both files are in `/home/ttuser/dev/ornith/vllm` (authored by the previous session; kept because this run
verified the cause and A/B-verified the change, reviewed line by line below).

- `plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py` — `apply_cached_req_state_update` gains
  optional `num_output_tokens` / `all_token_ids` / `in_persistent_batch` / `async_scheduling` and
  reconciles `req_state.output_token_ids` against the scheduler's authoritative count: truncate in place
  for a request that still holds a persistent-batch row, rebuild from `all_token_ids[-n:]` for a request
  that does not. Returns whether it changed anything. Mirrors
  `vllm/v1/worker/gpu_model_runner.py::_update_states` lines 1051–1084 (upstream does exactly this; the
  TT plugin's copy of that loop had simply omitted it).
- `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` — sets
  `input_batch.async_scheduling` once at construction, passes the new arguments from both `_update_states`
  call sites, and refreshes `InputBatch.num_tokens` when the output history was trimmed.
- `models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py` (tt-metal, already staged) —
  two unit tests for the resync.

Nothing was reverted; no speculative edit was added by this run. `max_model_len` (262144), the served
context, the benchmark ISL/OSL/concurrency, request lengths, host sampling, async decode, tracing and
on-device sampling are all untouched.

## Source review of the fix (requested)

- **(a) `in_persistent_batch` differs between call sites — correct, not a bug.**
  `model_runner.py` passes `req_index is not None`; `input_batch.py` (lane path) passes
  `req_index is not None and not resumed_from_preemption`. The two batches remove the row at different
  times. In the non-lane `_update_states` a preempted request is dropped earlier in the same function as
  an *unscheduled* request, so by resume `req_index` is already `None` (upstream asserts exactly this:
  `assert req_index is None` in the resumed branch). In `TTLaneInputBatch.update` the row is removed
  *after* the call (`if resumed_from_preemption: ... self.remove_request(req_id)`), so `req_index` can
  still be non-None there and the extra `and not resumed_from_preemption` is required to select the
  rebuild-from-scheduler branch instead of truncate-in-place. Both formulations are equivalent to
  upstream's intent for their own path.
- **(b) Can `all_token_ids.get(req_id)` be `None` exactly when needed? No — this is closed by an
  invariant in vLLM itself.** In `Scheduler._make_cached_request_data` the population rule is
  `if not scheduled_in_prev_step: all_token_ids[req_id] = req.all_token_ids.copy()`, and immediately
  above it, for every resumed request, `assert not scheduled_in_prev_step`. So a resumed request always
  gets its entry. That assert holds because the base scheduler gates the whole waiting loop with
  `if not preempted_reqs:` (scheduler.py:535), so a request can never be preempted and resumed in the
  same step; and `prev_step_scheduled_req_ids` is rewritten at the end of every `schedule()`. A request
  that is merely "not scheduled last step" also satisfies `not scheduled_in_prev_step`. Were the
  invariant ever violated, the failure mode is a loud `AssertionError` inside vLLM, not a silent
  no-op resync.
- **(c) `num_tokens = num_prompt_tokens + num_output_tokens` for a mid-chunked-prefill request — safe.**
  That line only runs when `output_tokens_changed` is True, which only the truncate branch can return,
  which requires `num_output_tokens < len(req_state.output_token_ids)`; a request still in prefill has
  both at 0, so the branch cannot fire for it. The value is also the same expression
  `InputBatch.add_request` uses to seed the row (`CachedRequestState.num_tokens`), so it is consistent
  with the rest of the batch for a chunked request too (whole prompt present, partially computed).
- The plugin's `InputBatch` has no `num_tokens_no_spec`, so upstream's extra line there has no analogue
  to keep in sync. `self.input_batch` is constructed exactly once, and the `async_scheduling` flag is set
  unconditionally after both construction branches, so it cannot be silently left `False`.

## Verification

Done in this AutoFix run (host-only, no device):

1. Deterministic repro + A/B, 36 configurations
   (`cd /home/ttuser/dev/ornith/vllm && python doc/.../repro/hang_scheduler_sim.py --seed S --prompt 320
   --prompt-jitter 40 --concurrency 6 --num-blocks NB --max-tokens MT [--patched]`):
   **unpatched 22/36 HANG, patched 0/36 HANG, 36/36 COMPLETED 32/32.**
   Full table: `doc/tti_release/repro/ab_sweep.txt`.
2. Unit tests:
   `python -m pytest models/autoports/ornith_ai_ornith_1_0_35b/tests/test_generator_vllm.py
   -k "discarded_async_token or resync_leaves_a_healthy" -q` -> **2 passed** (2.23 s).

End-to-end hardware verification: **run by the parent agent** (this agent was instructed mid-run to stop
all device work and hand hardware back). Required command, against a freshly started
`launch_autoport_server.sh` server on port 8100:

```
.../.venv_llm_vllm/bin/vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
  --model ornith-ai/Ornith-1.0-35B --dataset-name random --max-concurrency 16 --num-prompts 32 \
  --random-input-len 16384 --random-output-len 128 --percentile-metrics ttft,tpot,itl,e2el \
  --save-result --save-detailed --result-filename <path>.json --host 127.0.0.1 --port 8100 \
  --extra-body '{"truncate_prompt_tokens": "16384"}' --header Accept-Encoding=identity
```

> RESULT (parent agent, 2026-08-19 20:13:23Z - 20:18:03Z, after the ERISC recovery below):
> **PASS. `Successful requests: 32`, `Failed requests: 0`, benchmark duration 279.82 s**
> (previously: hung at 30/32 and never returned).
> Artifacts: `doc/tti_release/fixverify/bench_isl16384_osl128_conc16_n32.json`
> (`completed=32 failed=0 num_prompts=32 duration=279.8s`),
> `doc/tti_release/fixverify/bench_console.txt`,
> `doc/tti_release/fixverify/server_fixverify.log.gz`.
>
> The resume-from-preemption path was genuinely exercised, so this is a real test of the fix and
> not a run that happened to avoid preemption: the server log records prefills at logical lengths
> **16448, 16450 and 16511** - i.e. 16384+64, 16384+66 and 16384+127, resumed requests carrying
> their already-accepted output tokens. 16511 is the exact length that hung before.
>
> The mechanism-level signature also flipped: this run logs **0 `host sampling` prefills**, against
> 2 in the hanging run. `host sampling` was the visible symptom of a resumed re-prefill being
> misread as an unfinished chunk (`intermediate_prefill_mask`); with the runner's output history
> reconciled against the scheduler's, the resumed prefills are complete prefills again and take the
> normal device-sampling path.
>
> Device state: the first launch attempt at 20:05:47Z died during device init with
> `TT_THROW: Device 0: Timed out while waiting for active ethernet core 29-25 to become active
> again` (llrt.cpp:594) - an ERISC/infrastructure fault, unrelated to this fix. Recovered per
> `$tt-device-usage`: stale processes killed, `tt-smi -ls / -r / -ls` all exit 0 with all four
> p300c boards present (`doc/tti_release/logs/device_reset_erisc_recovery.txt`), 1x4 mesh
> open/close smoke `MESH_SMOKE_OK` (`doc/tti_release/logs/mesh_smoke_after_erisc.txt`). The server
> then came up cleanly at 20:13:15Z and the benchmark above ran on it.

## Final Status

- **Fixed and confirmed end-to-end on hardware** (see the RESULT block above). Root cause verified, fix A/B-verified
  against a deterministic reproduction of the exact production signature, and reviewed against upstream.
- Commands that prove the current state: the two listed under Verification.
- No process was left running by this agent; no `tt-smi -r` was issued by this agent.

### Remaining risks / uncertainty

1. **`num_output_tokens == 0` at resume is still not reconciled.** `apply_cached_req_state_update`
   (like upstream `gpu_model_runner`, which has the identical `num_output_tokens > 0` guard) leaves a
   stale token in place if a request is preempted before *any* output token was accepted while one was in
   flight. The resulting drift can only deadlock when `max_tokens <= 1` (the guard needs
   `P + 2 - 1 >= P + max_tokens`); for larger `max_tokens` it costs one wasted step and one token of
   output corruption. It is additionally hard to reach under `TTScheduler`, whose running-list rebuild
   puts freshly admitted requests at the *head*, while the preemption victim is popped from the *tail*.
   Upstream-shared, narrow, and not observed in any of the 36 simulated runs; worth a follow-up.
2. The simulator models the runner's bookkeeping, not the device. It cannot show a device-side effect of
   a resumed request (state-slot remap, trace re-capture) — those paths are only covered by the hardware
   run above.
3. Output *correctness* after a preemption+resume is not asserted anywhere: the resumed request
   re-prefills prompt+accepted-tokens, which is correct by construction, but no test compares generated
   text across a preemption. The benchmark only checks completion.
4. Minor, unrelated: when `TTScheduler` falls back from `_schedule_prefill_only` to
   `_schedule_decode_only` in the same step, the second `super().schedule()` sees a
   `prev_step_scheduled_req_ids` set already overwritten by the first call, so
   `all_token_ids[req_id] = req.all_token_ids.copy()` is taken for every decode request in that step —
   a full copy of a 16k-token list per request per such step. Correctness is unaffected; it is a
   throughput wart worth a look if TTFT/TPOT regress.
