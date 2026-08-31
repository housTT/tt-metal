# GPT-OSS 120B vLLM integration stage review

Reviewer: fresh `$stage-review` subagent `/root/stage_review_vllm_final`.

## Initial verdict

`more-work-needed`

Required work: `doc/context_contract.json` contained stale vLLM integration
state from an earlier `max-num-seqs=8` run, contradicting the final README and
readiness artifacts that record `max-num-seqs=32`, B1/B32 decode buckets,
73 passed / 1 skipped sampling, and the final primary and CI benchmark metrics.

## Remediation

The live context contract was updated to match the final serving evidence:

- `status`: `vllm_integration_complete`
- `serving.max_num_seqs`: `32`
- `serving.decode_trace_buckets`: `[1, 32]`
- `validation.sampling_tests_passed`: `73`
- `validation.sampling_tests_skipped`: `1`
- primary 1 x 128->128 benchmark metrics from
  `readiness_vllm/vllm_benchmark.json`
- secondary 32 x 100->100 CI serving-burst metrics from
  `readiness_vllm/vllm_ci_serving_benchmark.json`

After remediation, JSON parsing, the final `09-vllm.check.sh` gate, and both
repos' `git diff --check` gates passed.

## Rereview verdict

`clean-pass`

The rereview found no remaining concrete blocker for the context-contract
finding. The updated `doc/context_contract.json` matches the final README and
readiness JSON for `max-num-seqs=32`, B1/B32 decode buckets, 73 passed / 1
skipped sampling, primary 128->128 benchmark metrics, and secondary
32 x 100->100 CI burst metrics.
