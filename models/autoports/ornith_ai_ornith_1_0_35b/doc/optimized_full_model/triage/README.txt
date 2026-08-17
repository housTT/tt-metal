tt-triage for the hang described in work_log.md §5.1 and README §9 (the reverted traced-first-token
optimization: writing prefill logits into the trace-region `_trace_logits` buffer from outside a
replay).

  tt-triage.txt      the full LLM-output report
  triage-summary.txt the per-script pass/fail summary
  triage-console.txt the console, including check_binary_integrity's kernel .text mismatches

What it shows: all four devices stuck on the same `ReshapeViewDeviceOperation`
(`dump_running_operations`, `dump_op_mesh`), and `check_binary_integrity` failing with
"Data mismatch in section .text" against the cached kernel ELFs on many worker cores - the signature
of kernel binaries allocated after trace capture being overwritten by a replay.

bisect_after/ is a second, different hang, from the deliberate §5.1-of-the-full-model reproducer.

Recovery for this one: kill -> `tt-smi -ls --local` (8 boards) -> `tt-smi -r` -> `tt-smi -ls --local`
(8 boards) -> mesh open/close smoke printing MESH_SMOKE_OK. Infrastructure recovery, not a model
result.
