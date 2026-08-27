# Multichip decoder stage review

Date: 2026-08-27

Final verdict: **more-work-needed**

## Required work

The original resident full-stack contract remains impossible on the fixed P300
`1x2` target.  The audited plan leaves 26,889,461,760 bytes/device for experts,
while ordinary TP2 BFP4 experts need 33,973,862,400 bytes/device.  The available
compressed kernel's bank-padded count is 66,846,720 physical expert tiles/device
and permits only a 0.3213106043198529 BFP4 fraction, requiring 67.86894%
BFP2/zero.  The tested fit candidates do not pass PCC.

The required next step must change a hard constraint: use a larger mesh/board,
relax the resident full-stack requirement, or add a compatible sub-BFP4
active-expert prefill and decode representation that passes PCC.  No further
repairable evidence gap remains on the present board and software stack.

## Rereview closure

The first independent review also found that lower-precision candidate
failures were summary-only.  This was repaired by adding
`capacity_candidate_probes.log` with timestamps, command-execution IDs,
selected expert IDs, raw stdout PCC/error metrics, and exit status provenance
for all five candidate families.  The log is linked from `AUTOFIX.md`,
`mesh_plan.md`, and `work_log.md` and included in
`evidence_manifest.sha256`.

The rereview verified the manifest, inspected the raw candidate log and the
correctness, context, watcher, latency, and Tracy summaries, and reported no
other concern or hard-check gap.  The per-layer multichip implementation and
evidence are sufficient; only the physical resident-stack blocker prevents a
clean pass.

The review was read-only and did not open TT hardware.
