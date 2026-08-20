# AUTOTRIAGE

## Diagnosis

- Two independent defects caused the max-context Watcher failure, and both are now fixed and verified.
- First, split-trace setup failed to restore token, position, rotary, and linear-recurrence state between warmup and capture. At the final supported input position this made capture attempt position 262144/KV block 4096. Restoring the request-boundary state before capture fixes that correctness contract.
- Second, common force-argmax sampling unconditionally demoted every sub-eight-device group to Linear topology and supplied a Linear completion barrier. This target is not a logical Galaxy subgroup: it is a physical four-device P300c `FABRIC_1D_RING`. Allowing this model to retain Ring and omitting the Linear barrier fixes the sampler CCL failure. The final safe-Watcher max-context gate passes.

## Triage Evidence

- `tt/generator.py:269-318` now restores the snapshotted linear recurrence, current position, rotary position, and token buffer immediately after warm synchronization and again after capture. Diagnostic `max_context_fixed.junit.xml` intentionally failed only because its then-stale test expected one capture boundary; its failure records the actual corrected sequence `[262143, 262143]` for model and sampler capture.
- Linear force sampling failed safe Watcher on device 0 BRISC at `all_gather_async/.../minimal_default_writer.cpp:119`. That line loads `chunks_per_sync`; it is a stop location, not an assert statement.
- Removing `DUMP_ALL` did not cure the failure. Standard top-k also failed its Linear gather at writer line 260, so unsafe polling and force-argmax arithmetic were refuted.
- Forcing `cluster_axis=1` left the force failure at the same line 119, refuting implicit axis selection.
- The held-constant physical-ring fix retains one link, `chunks_per_sync=10`, one worker/link, two buffers, logits shape/dtype/memory config, and semantic force argmax. It changes the resolved route from Linear to Ring and omits the barrier that is valid only for Linear completion.
- Final gate: `evidence/max_context_watcher_ring_sampler.junit.xml`, SHA-256 `608e8af191bee7d3733e0902e8ba57c48e321301b65c83b8fc3161dab00eba5d`, records one test passed in 49.909 seconds. Runtime diagnostics reported `cluster_axis=None`, `topology=Ring`. The test covers initial position 262143, both trace-capture boundaries, on-device force argmax, split traces, token feedback, and final position 262144.

## Source Evidence

- `QwenFullModel.decode_device` consumes current/rotary positions and advances them in place (`tt/model.py:596-637`). `_capture_split_traces` now calls `restore_capture_inputs()` before `begin_trace_capture` and after both captures (`tt/generator.py:269-318`).
- The target topology was established before this stage: `doc/multichip_decoder/work_log.md` records four P300c devices, `MeshShape([1,4])`, `FABRIC_1D_RING`, and a bitwise-passing one-link Ring all-gather. The decoder consistently preserves Ring for all-reduce, reduce-scatter, and distributed-norm gathers (`tt/multichip_decoder.py:690-788`).
- `TTSampling` now reads the opt-in `allow_small_ring` model setting. Its generic `<8 => Linear` protection remains the default for logical submeshes; the Qwen model opts in because its four devices form a physical ring (`models/common/sampling/tt_sampling.py:170-187,359-371`; `tt/model.py:317-334`).
- The force gather constructs common kwargs and adds `barrier_semaphore` only when the resolved topology is Linear (`tt_sampling.py:463-492`). This matches the canonical `Sampling1D` Ring branch, which uses Ring without a barrier (`models/common/modules/sampling/sampling_1d.py:288-308`).

## Producer/Consumer Ledger

| Resource | Producer/owner | Consumer | Verified contract |
|---|---|---|---|
| trace inputs | generator request-boundary snapshots | captured decoder and sampler | restored before both capture boundaries; both see 262143 |
| vocab logits | TP4 vocab-sharded LM head | force gather | dimension 3 over all four devices |
| route | model sampling metadata | physical P300c fabric | `cluster_axis=None`, Ring, one link |
| AG semaphores | shared TT_CCL axis-none pool | Ring reader/writer kernels | rotating AG handles; no Linear barrier |
| chunk cadence | model metadata | Ring reader/writer | 10 chunks/sync, one worker/link, two buffers |
| sampled token | device argmax | persistent next-token input | remains on device through canonical split sampling |

## Rejected Hypotheses

- **Residual max-position overrun after restoration:** refuted by both capture boundaries observing 262143.
- **Unsafe `DUMP_ALL` polling:** refuted by failure under safe Watcher without `DUMP_ALL`.
- **Force-argmax arithmetic/argmax kernel:** refuted because standard top-k also failed in its Linear gather.
- **Implicit versus explicit cluster axis:** refuted because explicit axis 1 reproduced line 119.
- **`chunks_per_sync=10` itself:** no supporting evidence; line 119 only loads the argument, and the passing Ring path retains value 10.
- **Custom sampler required:** refuted; the selected common `TTSampling` force path passes after correcting its target topology contract.

## Proposed Fix

- Final implemented fix is the smallest verified pair:
  1. restore all mutable request-boundary trace inputs after warmup and before capture;
  2. add an explicit model opt-in for physical rings smaller than eight devices, preserve Ring for that target, and omit the Linear barrier on Ring.
- Keep generic small logical submeshes on their existing Linear default. Do not infer Ring solely from device count.

## Final Status

- **Fixed and Watcher-verified.** The original max-context split-sampling test passes on the exact physical 1x4 P300c ring with safe Watcher.
- The standard top-k Linear Watcher failure is retained as rejected-alternative evidence; it is not the selected optimized token-out path. The selected semantically greedy common force path is Ring, device-resident, and passing.

## Uncertainty

- The A/B changed topology and barrier as the supported protocol pair, so it does not attribute the old halt to one of those two variables independently. That distinction is unnecessary for the selected path because the canonical Ring contract omits the Linear barrier.
- The retained failed Watcher console messages are summarized here but were not embedded in JUnit. Preserve exact console logs in future CCL investigations.
