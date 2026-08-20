# Independent stage review 4

Date: 2026-08-20 EDT

Verdict: **more-work-needed**

The fresh reviewer inspected the current full-model code, stage reviews 1--3,
context/capacity contract, structured metrics, JUnit files, qualitative and
autoregressive outputs, profiler reports, artifact manifest, and postflight
log. Manifest verification and `git diff --check` passed. Accuracy,
qualitative behavior, context arithmetic, optimized decoder preservation,
greedy Ring sampling, page-table handling, and selected profiler evidence were
otherwise judged strong.

Required findings:

1. Remove the allocator warning emitted while constructing the stochastic
   sampler trace with the model trace resident. Identify and preallocate the
   operation, preserve the warning-free console log, and rerun safe Watcher.
2. Complete low-level `decode_forward` request state for explicit stochastic
   sampling: seed reset/advancement, prompt and output penalty history, mixed
   active rows, inactive slots, and reset controls.
3. Add a representative full-64-layer prompt-128/generate-128 token-out
   measurement that includes the caller-required sampled-ID readback. Relabel
   the existing 10-iteration trace-pair figure as device-only.
4. Refresh bounded device-list and mesh-open/close postflight evidence after
   the final remediation run.

The reviewer also required the stochastic console log to preserve selected
tokens, seed counters, topology output, and the absence of the allocator
warning. Commit creation remains correctly deferred until a clean rereview.
