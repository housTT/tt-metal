# Stage 11 independent review — metric-integrity remediation

Date: 2026-09-03

Verdict: `more-work-needed`

The independent reviewer audited committed handoff state `9a5d198` and found
the following blocking report-integrity and copy-back issues:

- IFEval timing used a stale 28-sample denominator. The authoritative full-run
  value is `13045.45777320399 / 541 = 24.11360032015525s` per request.
- GPQA's lm-eval aggregate advertised the 198-row dataset size even though the
  runtime spec selected exactly IDs 0 through 6. The authoritative selected-run
  value is `2246.9087665929983 / 7 = 320.986966656143s` per request. The merger
  also needed fail-closed raw/runtime provenance validation and an exact PASS
  requirement.
- The single-row benchmark waiver described aggregate-throughput defects but
  did not disclose the two unmet aspirational TTFT tiers, for five failed
  subchecks in total.
- All 21 benchmark blocks exposed `error_request_count=null` although the raw
  artifacts proved zero failures and zero errors.
- The intended small benchmark CSV and smoke log were ignored rather than
  committed, generated `__pycache__` files remained, and `RUN_NOTES.md` still
  recorded completed work as pending.

The same review independently confirmed the generated-autoport imports,
external no-Docker runtime wiring, exact 131072 context, non-aligned 10000-token
coverage, AIME 13/15, GPQA flexible extraction 7/7, MMLU 84.6732% over 2,127
samples, canonical IFEval 463/541 with explicit PASS, all 21 raw benchmark rows,
22/22 chat/spec tests, prompt/qualitative evidence, forbidden-data exclusions,
and clean device/server shutdown.

This is a provisional finding record, not the required final `clean-pass`.
