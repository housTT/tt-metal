# AutoFix — fused graph repair

Contract: preserve PCC >=0.995 per decode token, real checkpoint semantics,
unrestricted valid logical tails, paged cache ownership, and device-only trace
replay. Candidate failures are work; no candidate is selected on topology alone.

1. Expert-major decode reduction: fast_reduce_nc exposes physical row32 after
   reduction. Slicing to the single logical row before the shared-expert add
   fixes the reshape mismatch. `three_linear_129_fixed.json` passes. It is
   slower than the original compact SwiGLU layout and remains unselected.
2. Dedicated decode RoPE returns its padded input shape. Explicit logical head
   slicing fixes concatenation with the unchanged non-RoPE coordinates.
   Fused K/V update validation requires disjoint core grids: key core(0,0),
   value core(1,0) fixes the failed overlap validation. The adapted suite
   `full_trials129_adapted/commands.json` passes all candidates, including the
   combined attention path; it compares real HF and saved functional outputs.
3. Generalized MoE gate adapter: `ttnn.zeros` invokes host creation, so it fails
   during capture even after warmup. Source: `creation.cpp:269` uses full_impl;
   its full_like fast path at lines238–248 uses device SFPU for same-dtype tiled
   tensors. Persistent zero templates for lengths1/128/512 plus zeros_like fix
   the prohibited write. The harness now ends/releases a failed capture before
   propagating its exception, avoiding a teardown wait on an open capture.

## Preserved hang and hardware recovery

The first dedicated-gate command is recorded in `extra_trials129_queue.log`;
the active child used `run_layer --layer3 --length129 --steps8 --trace --audit
--decoder fused --fusions moe_gate`. The fatal was
`Writes are not supported during trace capture`, at 01:41:53 UTC. Its parent
queue PID283544 was stopped to prevent a subsequent device job.

`timeout 120 python tools/tt-triage.py --llm-output ...` exited0 before cleanup.
Output and summary are under `triage/moe_gate_capture/`; triage reported cores
whose halt state changed during collection. These are failure/recovery evidence,
not a clean watcher result or a model numerical result.

Only owned child PID283867 and queue PID283544 received SIGTERM; the stopped
queue received SIGCONT to exit. No unrelated process was terminated. Bounded
`tt-smi -ls --local`, `tt-smi -r`, `tt-smi -ls --local` succeeded. One reset was
sufficient; all four ASICs reappeared. No locks were manually removed.
`timeout60` 1x1 open/close smoke exited0 and printed `MESH_SMOKE_OK`.
Exact logs: `triage/moe_gate_capture/{console,list_before,reset,list_after,mesh_smoke}.log`.
The retry queue is `extra_trials129_fixed_queue.log`, followed serially by the
GDN grouped/ternary trials. No profiler or watcher overlapped recovery.

## Isolated source investigation

AutoDebug's first preflight failed because this host disallows unprivileged
user namespaces; no model started. Existing danger-full-access execution and
the user's subagent authorization permit the documented retry with
`AUTODEBUG_SKIP_CHILD_SANDBOX=1`. Logs: `autodebug_launcher.log` and
`autodebug_retry.log`; report destination `AUTODEBUG.md`. Installed skills and
shared source are unchanged. The investigator is inspection-only.

## Native FFN and prefill packing repair

The FFN reader bounds output regions against activation row capacity even with
read_x_at_offset=False. Increasing backing capacity repairs the missing-expert
outputs (`ffn_hypothesis.md`, `ffn_repaired129`). The full256-region version is
slower; compact eight-slot mapping and direct router top8 remove zero filling,
dense scatter and repeated topk. `compact_trials4096` passes and wins.

Native minimal_matmul_split with256 output pointers hangs in the reader. Triage
was preserved before terminating owned child353611. `triage/prefill_split/`
contains capture, bounded list/reset/list and successful MESH_SMOKE_OK. One reset,
all four ASICs, no manual lock removal. Independent `AUTOTRIAGE.md` traces the
5168-byte compiled reader frame against the5056-byte limit and overlapping NoC
counter globals; the exact counter value was not sampled and remains inference.
Eight groups of32 outputs avoid that frame size. Both adapted split layouts pass;
the grouped32 adapter wins (`split32_full4096`, `split32_linear4096`). Shared C++
source remains unchanged.

KDA direct tuple, prefill history tail and prefill normalization/gate fusion all
pass at4096/128 (`kda_final4096`, `residual_linear4096`). Projection-residual was
repaired after a missing token-return hook; its corrected linear path passes
but is slower. Full attention passes shape adaptation (BF16 projection weight)
but fails headline PCC0.9948604 and is slower (`residual_full4096`).

Native recurrence was retried in the final surrounding graph, not rejected from
synthetic evidence: `native_positions_final_candidate.json` fails real B32
minimum decode PCC0.9884342. Precise default passes0.9977745 under identical
prefixes and seed. Native TF32 matrix source-register rounding is the prior
source-supported cause; no available mode preserves the FP32 SFPU state contract.

All observed adapter bugs and recoverable hardware faults were resolved. The
final default passes positions, reuse, four device tests and separate watcher
correctness. Native capacity/profile and independent review are recorded in the
README/work log. No AutoFix failure is claimed.

## Independent-review closure additions

The review exposed a missing shared-expert epilogue candidate and incorrect4-byte
fused FIR tap estimate. `shared_epilogue4096/commands.json` records the fully
adapted native shared down/gate/routed-add trial: slower on both real layer kinds,
and full decode fails0.995. Its separate tests-only harness preserves the rejected
experiment. The estimator now counts BF16 KDA taps (65536-byte correction).

Indexed partial RoPE had no exact API blocker after coherent16-group packing.
The initial tests-only adapter failed a frozen dataclass assignment before device
forward; dataclasses.replace repaired it. The first natural-layout launch was a
CPU argument error after an edit-script guard failed; corrected command is in
indexed_trial_commands.json. Neither is a kernel/numerical rejection. Adapted
129 and4096 runs pass; removing the unnecessary caller transpose round trip wins
920.3985us versus924.030us. This path is promoted and qualified in the indexed
final suites. Shared C++ remains unchanged. No AutoFix failure is claimed.

Production indexed timing regressed4.6% despite identical results. Independent
AST review found no dispatch change; a matched lifetime control isolates retention
of original Q/K norm and packed projection buffers through repacking. All four
real4096/128 controls are in placement_controls4096/commands.json. Retention
restores920.4485us versus962.3595us without it; final production920.198us. The
likely mechanism is allocation placement; no exact address/bank explanation is
claimed. This transient constructor fix adds no runtime buffer. Targeted final
correctness/replay/watcher and full device profiles follow; no failure is waived.
