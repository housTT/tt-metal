# AutoFix report: split-channel depthwise conv1d

## Starting evidence

- Source report: `AUTODEBUG.md`, finding 1.
- Review gate: `stage_review.md`, P1.
- Hypothesis: four 2,560-channel BF16 depthwise `ttnn.conv1d` calls could replace the spelled chunk/token convolution, preserve rolling state and fused SiLU, and improve both measured phases. Eight 1,280-channel calls were the resource/contract fallback.

## Hypothesis experiments

- Hypothesis: split-4 fits and expresses B=1/L=67 prefill.
  Experiment: real layer-0 weights, BF16 HiFi4, fused SiLU, repeated prepared-weight invocation, explicit versus direct DRAM output, and comparison with Torch plus the current TTNN expression.
  Result: finite; Torch PCC 0.999771533448292; current-expression PCC 0.999769934209639; conversion-route PCC 1.0.
  Verdict: verified for isolated prefill geometry.
  Evidence: `candidates/split_conv1d/probe_split4_l1small.log`.

- Hypothesis: physical B=32/L=4 is accepted and prepared weights are geometry-independent.
  Experiment: split-4 host and prefill-prepared weights, isolated decode-first process, split-8 fallback, and both NHWC/NLC input ranks.
  Result: every physical B32 form fails before math with `Reader indices buffer page size 132 exceeds worst-case CB size 64`; prefill-prepared weights are not reusable at decode geometry.
  Verdict: refuted.
  Evidence: `probe_split4_geometry_specific.log`, `probe_split4_decode_only.log`, `probe_split8_decode_only.log`, and `probe_split8_decode_nlc.log`.

- Hypothesis: a structurally exact logical-B32 adaptation works.
  Experiment: flatten 32 four-row states to physical B=1/L=128, convolve, select starts 0,4,...,124, reuse prepared weights, capture, and replay twice.
  Result: PCC 0.9997603085001467 against Torch, finite, reuse PCC 1.0, trace determinism PCC 1.0.
  Verdict: verified in isolation.
  Evidence: `probe_split4_decode_flattened.log`.

- Hypothesis: the candidate preserves the real decoder acceptance bar.
  Experiment: integrate native host weight packing and persistent geometry-keyed prepared weights, then run the real-weight non-aligned linear prefill/decode test. The no-L1-small DRAM-config route and the standard 16 KiB L1_SMALL route were both tested; HiFi4 exact/FP32-accumulating and standalone-SiLU controls followed.
  Result: the DRAM-config route stalls device 3 (zero heartbeat, all NoCs hung). Fused conv1d SiLU completes with PCC 0.9575315732886901; FP32 accumulation gives 0.9564177132706132. Moving SiLU out of `Conv1dConfig` passes at 0.9978466372080389, with subsequent checks 0.9994128027425493, 0.9996440458669663, and 1.0.
  Verdict: fused conv activation refuted; standalone-SiLU split conv verified for short and native-context correctness.
  Evidence: `short_linear_correctness_dram_config.log`, `triage/`, `short_linear_correctness_l1small.log`, `short_linear_correctness_l1small_fp32acc.log`, and `short_linear_correctness_standalone_silu.log`.
  Fix: none retained; implementation and tests reverted to frozen hashes. The exact passing diff is retained for follow-up.

## Final status

Native-context gates passed for all 262,144 prefill tokens and native-prefix traced decode. The dedicated op is mathematically expressible, and standalone SiLU preserves correctness; putting SiLU inside the convolution does not. The no-L1-small DRAM metadata route hard-stalls, while the passing route requires the standard 16 KiB L1_SMALL reservation.

The original 5,737.134/2,983.133-us measurements are retained only as historical exploration because their 16-KiB profiler fixture was not retained and their controls used zero L1_SMALL. P1 was closed with an exact 16-KiB A1/B/A2 under `candidates/split_conv1d/l1_16k_ab/`. The base implementation/test identities were `f0778a63...`/`ea3c4007...`; the candidate implementation/test identities were `f6901109...`/`ea3c4007...`; the same profiler runtime, isolated P300 board, cache, replay count, node, and signposts were used throughout. Prefill A1/B/A2 is 5,505.526/5,635.236/5,500.267 us (159/159/159 ops), making B 132.3395 us or 2.405% slower than the 5,502.8965-us median base. Traced decode is 2,625.831/2,873.108/2,625.095 us (71/94/71 ops), making B 247.645 us or 9.432% slower with 23 extra ops. Since both required phases lose, Watcher was not warranted and the frozen spelled TTNN path remains selected.

## Hypothesis group: review provenance and device recovery

- Hypothesis: the passing split-conv candidate might win when base and candidate both reserve 16 KiB L1_SMALL. Experiment: exact same-fixture A1/B/A2 for linear prefill and traced decode. Result: candidate is correct but slower by 2.405% and 9.432%. Verdict: refuted. Evidence: `candidates/split_conv1d/l1_16k_ab/`.
- Hypothesis: the old prose recovery record could be made exact from retained history. Experiment: search all existing stage evidence and provenance. Result: exact historical PID/list/reset/list/lock/mesh transcript was not retained. Verdict: limitation recorded without invented details. Fix: `candidates/split_conv1d/recovery/` retains a fresh bounded pre-remediation process/lock check, targeted topology-complete free-board reset, list, second-reset decision, and installed-runtime mesh smoke. No process was killed and no lock cleared.

## Hypothesis group: missing frozen-base controls

- Hypothesis: FunctionalDecoder reshape/concat Q/K replication might beat direct `repeat_interleave` on the final graph. Experiment: one-change real linear correctness plus prefill/decode A/B/A. Result: correct, but 5,863.601 us versus 5,607.296 us median base prefill and 2,742.536 us versus 2,726.932 us median base decode. Verdict: refuted; direct repetition remains selected. Evidence: `candidates/repeat_interleave_ab/`.
- Hypothesis: dedicated `transformer.concatenate_heads` might beat primitive full-prefill permute/reshape on the final graph. Experiment: both merge sites changed, real full correctness, full-prefill A/B/A. Result: correct, but 2,365.869 us versus 2,357.464 us median base. Verdict: refuted; primitive merge remains selected. Evidence: `candidates/concat_heads_prefill_ab/`.
- Hypothesis: lane-axis dedicated partial decode RoPE might beat the current primitive partial RoPE on the final graph. Experiment: only `_full_qkv_decode` changed, non-aligned and paged-trace correctness, full-decode A/B/A. Result: correct and six fewer ops, but 2,346.163 us versus 2,337.998 us median base. Verdict: refuted; primitive decode RoPE remains selected. Evidence: `candidates/decode_rope_lane_axis_ab/`.

All three candidates were reverted. Frozen implementation/test SHA256 values remain `f0778a63a47a844621cffa3a251345333e563105cabf708e5abe0af0dbce3a14` and `05c38b764320bb682248df78d00ff950a1602b8aeee9b2e40cb25fd690a41dad`.
