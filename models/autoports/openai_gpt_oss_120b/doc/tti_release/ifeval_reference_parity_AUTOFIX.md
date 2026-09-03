# IFEval reference-parity AutoFix

## Starting evidence

- The full canonical Google IFEval run scored 397/541 strict prompts with
  `reasoning_effort=low` and `max_gen_toks=1280`.
- The cited 423/541 reference used `max_gen_toks=4096` and did not override
  GPT-OSS reasoning effort; the official model and Harmony defaults are medium.
- The repair must not change the score, tolerance, dataset, sample scope, or
  supported context.

## Verified repair

- The GPT-OSS `meta_ifeval` gate maps transparently to canonical `ifeval` and
  now makes the parity policy explicit: medium reasoning, 4096 generated
  tokens, deterministic sampling (`do_sample=false`, `temperature=0`), seed
  42, and `max_length=131072`.
- Full and ci-nightly command tests assert the exact policy. Full remains
  unbounded; ci-nightly retains only its configured `--limit 0.05`.
- The release-report merger rejects IFEval artifacts unless both the top-level
  run config and canonical task config preserve the policy and full context.
  Its handoff metadata records only the bounded policy and scope, not samples.

## Host verification

```text
python -m pytest -q test_regenerate_release_report.py
15 passed

PYTHONPATH=tt-inference-server python -m pytest -q \
  tt-inference-server/tests/test_module/llm_tests/test_llm_eval_tests.py \
  tt-inference-server/tests/workflows/test_workflow_dispatch_routing.py
160 passed

python -m py_compile <four changed Python files>
git -C tt-inference-server diff --check
PASS
```

The active full medium+4096 hardware run is the remaining end-to-end accuracy
confirmation; this host-side repair did not access its server, session, or
devices.
