# AutoFix: full-36 logit reproducibility

## Outcome

Resolved without changing model math, precision, tensor parallelism, cache
layout, or the exact acceptance gate.

The first full-36/full-context batch-2 run produced finite but
invocation-dependent logits.  Prefill and decode greedy tokens remained stable
and exactly matched the first two pinned HF reference tokens.  The two-layer
control was bitwise exact after its separate sparse-MLP row-materialization
fix.  Fresh-context AutoDebug therefore retained exact equality and prescribed
bounded external device recovery before any source change.

## Recovery and proof

All four boards reported healthy DRAM and zero uncorrectable errors before
recovery.  The serialized recovery was:

```bash
tt-smi -r all
tt-smi -ls
```

The unchanged full gate then passed twice in separate fresh processes:

```bash
GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k real_weight_36_layer_batch2_logit_reproducibility -s
```

Durations were 352.32 s and 352.76 s.  Both prefill and traced decode are now
bitwise identical across the two physical page rows and across both
reset/reuse runs: maximum difference 0, different values 0, and identical raw
and top-100 hashes.  The refreshed artifact is
`artifacts/logit_reproducibility.json` with SHA-256
`588e98487aad4491d204dcac2ba12ab1dae4ec1068f2876606583ae30d79882d`.

The post-run `tt-smi` snapshot showed healthy DRAM and zero uncorrectable GDDR
errors on all four boards.  The unchanged source plus two clean-process passes
identifies stale external fabric/collective state as the failed-run condition;
it does not justify a model fallback or a relaxed logit invariant.

## Accepted source change from the earlier two-layer localization

The preceding two-layer investigation did find one real batch-2 decoder bug:
sub-tile `ttnn.split` views assigned distinct physical tile rows to logically
identical sparse-MLP inputs.  The production B>1 path now explicitly slices
each logical row, materializes it through row-major storage, retiles it, and
zeros implicit padding before invoking the unchanged batch-one sparse expert
graph.  That path stays on device and preserves all selected weight, activation,
CCL, fidelity, residual, and cache policies.  The batch-one measured token-out
branch is unchanged.
