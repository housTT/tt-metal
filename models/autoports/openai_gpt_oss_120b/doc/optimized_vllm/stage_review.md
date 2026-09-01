# Optimized vLLM independent stage review

## Initial verdict: more-work-needed

The first fresh-context review identified three release-blocking evidence
issues and one hygiene issue:

1. The normal 36-layer server log contained an active-trace allocation
   warning, while the available allocation tracker covered only two layers.
2. Cleanup and device-health evidence predated the final measured run.
3. The qualitative comparison incorrectly implied that both thermodynamics
   responses covered the Third Law, although both ended at the fixed
   256-token cap after the Second Law.
4. Two retained raw server logs contained trailing whitespace.

## Remediation

The allocation finding was handled through the `$autofix` workflow and a
fresh-context AutoDebug investigation. The exact full-depth tracker initially
reproduced two unsafe program-cache survivors before the first B1 replay:

- a device slice created when `get_block_size()` indexed a TT tensor; and
- a paged-fill program compiled only when real vLLM supplied its explicit
  per-request page-table shape.

The shared block-size helper now examines Python cache containers without
indexing a device tensor. The adapter now compiles one exact B1, 128-token,
per-layer paged-fill signature before capturing decode traces. The repeated
36-layer tracker run kept program-cache tracking enabled and completed the
primary and CI workloads, including 226 B1/B32 model and sampler replays, with
zero unsafe survivors. Its summary and raw evidence are in
`artifacts/trace_allocation_full36/`.

The exact final-code normal run was then repeated. Its primary/CI results,
terminal capability snapshot, compressed server log, empty post-run process
audit, and four-device DRAM-health snapshot are in
`artifacts/after_final_remediation/`.

The exact-output qualitative comparison was regenerated from the retained
12-output artifact. Both thermodynamics responses are now explicitly marked
`incomplete-at-cap`; no Third-Law completeness claim remains. Formatting
hygiene is enforced by the final staged pre-commit pass.

## Rereview

Verdict: **clean-pass**.

The fresh-context xhigh rereview found no required work. It independently
checked the staged implementation, official-checkout provenance, exact
before/after benchmark shapes, full sampling log, all 12 qualitative outputs,
host-test XML, context contract, terminal serving counters, full-depth tracker
log and summary, final cleanup audit, and device-health JSON. It classified:

- the remaining normal-mode allocation warning as controlled reserved trace
  storage because the exact full-36 tracker completed every replay with zero
  unsafe survivors;
- the cleanup finding as fixed by the post-final-run empty process audit and
  four-device DRAM-health snapshot;
- the thermodynamics truncation as controlled by the regenerated
  `incomplete-at-cap` comparison and verdict; and
- the raw-log whitespace finding as fixed by a clean staged diff check.

Residual risk is limited to the noisy generic normal-mode warning; the tracker
control closes the corruption risk for the measured B1/B32 serving path. The
review itself was read-only and ran no hardware or server commands.
