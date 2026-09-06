# Full-model stage review

Verdict: **clean-pass**

## Required work

None. The sampler-transition finding and the related page-table/RNG lifecycle
issues discovered during review are resolved in the final working tree.

## Verified gates

- `artifacts/final_profiles.xml` records all three complete-stack proxy
  profiles passing: three tests, zero failures/errors/skips, 176.743 seconds.
- All six source hashes in `artifacts/provenance.json` match the final files.
- All-layer capacity construction, non-aligned public prefill, batch-32
  correctness, explicit serving state, traced feedback, sampling, and reset
  coverage have retained evidence.
- The fresh AIME24 reference contains 161 prompt tokens, 100 generated tokens,
  and `[100,100]` top-k data. TP1/2/4 prefill and teacher forcing achieve
  0.95/0.96/0.95 top-1 and 1.0 top-5/top-100.
- Performance documentation distinguishes public generation, logits-only
  traces, device token-out traces, and allocation-tracked teacher forcing.
- The optimized multichip decoder policy is preserved and vLLM integration is
  untouched.

## Resolved anomalies

- Sampling-parameter recapture formerly risked consuming stale host tokens or
  overwriting retained device feedback during capture. The final generator
  warms a separate sampler output and backs up/restores the live token on
  device. The stale-host-token regression records two feedback reuses and two
  restores; both transitions preserve top-1/top-100 with cosine 0.9999671 and
  0.9999622, and sampled tokens match.
- Replacing page-table tensor identities formerly risked trace recapture and
  RNG disruption. New mappings now copy into stable tensors. Both trace IDs
  and buffer addresses remain unchanged; repeated sources stay at one copy,
  while a second source produces exactly two total copies. Logits cosine is
  0.9999576 with matching token/top-1/top-100, and sampled RNG continuation is
  exact.
- Public stochastic generation reproduces the same 16-token seed-42 sequence
  across reset with identical trace IDs; seed 43 differs and both streams
  advance.
- Greedy sampler candidates return different IDs only because terminal softcap
  creates equal global maxima. Canonical split sampling is semantically greedy
  and measures 0.446 ms versus 2.296 ms for force-argmax.
- Raw completion repetition and haiku end-of-turn tokens match HF behavior;
  the six-prompt chat suite remains coherent and task-aligned.

## Residual risk

Results use P300C QB2 proxy profiles. Maximum-context execution combines
representative real-layer probes with complete-stack capacity construction;
batch-32 evidence uses short prompts. Qualitative evidence covers six shared
prompts plus additional 64-token controls. These disclosed limits do not
contradict the completed stage contract.

The reviewer made no edits and used no hardware, servers, commits, or pushes.
