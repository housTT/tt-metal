# Direct-slot plus async retained-source repeat A

This directory preserves the first exact retained-source primary and CI
benchmark JSON plus the derived host-window snapshot. The source is the same
as the definitive `after/` implementation after the PLE candidate was
reverted.

The raw `vllm_result.json` and `vllm_ci_serving_result.json` are present and
support the benchmark summaries. The large live server log was subsequently
overwritten by the PLE A/B and is not claimed as available here; consequently
`serving_host_metrics.json` is an archived normalized snapshot whose embedded
`source` path no longer identifies its original bytes. It is useful as repeat
context only. All final host/async assertions and the reportable raw server log
come from `../../after/serving_host_metrics.json` and `../../after/server.log`.
