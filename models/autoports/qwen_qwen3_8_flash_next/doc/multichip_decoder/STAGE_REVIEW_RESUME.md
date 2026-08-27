# Resume-1 stage review

Date: 2026-08-27

Verdict: **more-work-needed**

The independent `$stage-review` rereview found no remaining repairable
code, test, documentation, or provenance issue.  The first review's only
repairable finding was that `MultichipMemoryPlan` omitted the layer-1 PLE
prefill/decode staging allocation.  The implementation, static test, context
contract, host-weight contract, mesh plan, and README now all charge 819,200
bytes/die for that staging.  The corrected full-stack plan is
8,066,785,280 bytes/die, and the refreshed `host_backed_static.xml` records
19 passes.  Every entry in `evidence_manifest.sha256` verifies.

## Required work

Progressing host-backed GDN trace replay still fails the original warmed
decode-trace gate.  The isolated four-token comparison produced token-four
output PCC 0.92648160 and final recurrent-state PCC 0.82660490.  Direct eager
host-backed correctness and changing-input QSA segmented trace replay pass,
but 36 of 48 layers use GDN.  `SEGMENTED_TRACE_AUTOFIX.md` records the
individually refuted repair variants, and the delivered API rejects this known
corrupt trace mode explicitly.

The reviewer classified this as the sole remaining P1 blocker and found no
other concern or hard-check gap.  A runtime/kernel trace fix or a new
baseline-preserving GDN trace strategy is required before the stage can
receive `clean-pass`.

## Scope inspected

- authoritative resume prompt and `$stage-review` instructions;
- `tt/multichip_decoder.py`, `tt/host_weight_cache.py`, and their tests;
- README, work log, mesh/context/host-weight contracts, host-backed XMLs,
  AutoFix report, and SHA-256 manifest;
- XML result counts and `sha256sum -c` output.

The rereview was read-only and used no TT hardware.
