# Independent stage review 2

Verdict: **more-work-needed**.

The second independent review confirmed that the prior reset, maximum-context,
page-table, physical-Ring CCL, qualitative, and batch findings were remediated,
but identified three remaining closure gaps:

- public stochastic generation configured sampling parameters but selected
  token zero with unconditional host argmax; host compatibility also argmaxed
  every later token;
- final Ring benchmark and teacher-forcing values existed only in prose because
  their JUnit files omitted stdout and the retained logs described the earlier
  top-k revision;
- the compact profiler report covered rejected Linear force argmax rather than
  selected Ring force argmax.

AutoFix remediation:

- request-scoped deterministic host sampling now applies temperature, top-k,
  top-p, presence/frequency, repetition, and seed from the prefill token. Host
  compatibility uses the same policy throughout and explicitly feeds back the
  selected token; optimized greedy device decode remains split traced.
- readiness prefill/teacher runners now accept `--output-json`; the 64-layer
  A/B benchmark accepts `QWEN36_TOKEN_OUT_METRICS_JSON`. Structured artifacts
  include accuracy, latency, selected/control tokens, topology, layer count,
  iterations, and runtime metadata.
- a fresh selected Ring profiler capture produced signpost-bounded compact
  artifacts: 3,962.61 us total, argmax 1,417.43 us, Ring all-gather 883.30 us,
  21 width-sharded matmuls 988.52 us, and no top-k.

Focused AutoFix tests passed 10/10. Final hardware JSON reruns and the next
clean-pass review are recorded after they complete.
