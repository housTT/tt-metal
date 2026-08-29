# Full-model work log

## Scope and implementation

Built only the repo-local full-model stage; no vLLM integration was started.
`tt/model.py` now streams the exact 36-layer checkpoint into the optimized
multichip decoder policy and maintained GPT-OSS terminal stack.  `tt/generator.py`
implements the standard readiness builder, explicit serving prefill/decode, fixed
cache slots, mixed prompts, inactive rows, reset/teardown, traced device sampling,
and an explicit host-sampling compatibility mode.

The model revision is
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.  Hardware runs used a `(1,4)`
Blackhole mesh on four healthy boards reporting `p300c`; this is the requested
P150x4 semantic topology.  Device tests were serialized with
`scripts/run_safe_pytest.sh`, followed by `tt-smi` health checks.

## Reference and capacity

The fresh AIME24 reference was generated with the exact HF tokenizer/chat
template, 214 prompt tokens, 100 generated tokens, and top-100 sets.  CPU/NVMe
offload was used for the exact HF model because the 120B checkpoint does not fit
host RAM as one resident model.  Artifact SHA-256:
`7e722ad241eee84148ed62b5accee20bc642a4a1de4cab98ae146a166ee9d2bc`.

Exact weight/KV/page/reserve accounting is stored in `doc/context_contract.json`.
P150 and P150x2 are hard physical rejections; P150x4 retains the full 131072
context.  Batch-10 full context fits, batch-11 reaches 130880, and batch-32
reaches 44992 under the same accounting.

## Commands and results

Host contract suite (final command is rerun after formatting/review):

```bash
python_env/bin/pytest -q models/autoports/openai_gpt_oss_120b/tests/test_full_model.py
```

Reduced real-weight trace, changed-page, mixed prompt, and inactive-row probes:

```bash
GPT_OSS_120B_FULL_MODEL_PROBE=1 \
GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k 'real_weight_two_layer' -s
```

The split trace probe and batch-2 mixed prompt/fixed-slot/inactive-row tests
passed.  A 16-step device-greedy versus host-argmax comparison is exact.

Full resident 36-layer/full-context smoke:

```bash
GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k real_weight_36_layer_full_context_token_out_smoke -s
```

Passed with all 36 layers and configured context 131072.  The short cold/capture
smoke produced four tokens and proved device feedback; its compile-inflated
performance is not the headline metric.

Readiness:

```bash
GPT_OSS_120B_FULL_MODEL_READINESS=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k run_prefill_check_aime24_top100 -s

GPT_OSS_120B_FULL_MODEL_READINESS=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k run_teacher_forcing_aime24_top100 -s
```

Prefill passed 0.94/1.00/1.00 top-1/top-5/top-100.  After the documented
teacher-forcing AutoFix, the original full gate passed 0.95/1.00/1.00, TTFT
3751.46 ms, and 52.1145 decode t/s/u.

Autoregressive comparison:

```bash
GPT_OSS_120B_FULL_MODEL_AUTOREGRESSIVE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k run_autoregressive_aime24_hf_and_tt -s
```

Passed with 100 HF and 100 TT tokens.  TT TTFT is 3.7757 s and decode is
51.6229 t/s/u.  Trace counters are 99 replays, one full refresh, 98 unchanged
page-table reuses, zero forced refreshes, zero full-logit reads, and zero host
argmax calls.  The two completions were read and received a coherent/on-topic,
non-repetitive, correct-language verdict.

Shared qualitative suite and AutoFix rerun:

```bash
GPT_OSS_120B_FULL_MODEL_QUALITATIVE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k shared_qualitative_chat_suite -s

python_env/bin/python /home/ttuser/dev/scratch/tt-metal/models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/openai_gpt_oss_120b --scope all \
  --missing-artifacts critical \
  --json models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/degenerate_check.json
```

The corrected suite passed all six prompts; the EOS-aware translation ends at
81 tokens and the checker reports no degeneration.

Exact same-format HF controls for all six TT prompts:

```bash
python_env/bin/python \
  models/autoports/openai_gpt_oss_120b/doc/full_model/scripts/generate_hf_qualitative_controls.py \
  --snapshot /home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  --prompts models/autoports/openai_gpt_oss_120b/doc/full_model/prompts/shared_qualitative_prompts.txt \
  --tt-control models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_tt_chat.json \
  --output models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_hf_chat.json \
  --comparison-output models/autoports/openai_gpt_oss_120b/doc/full_model/qualitative/qualitative_hf_tt_comparison.json \
  --offload-folder /tmp/gpt_oss_120b_hf_explicit_offload_20260829
```

The exact 120B model ran as one six-prompt greedy batch with 27 layers on CPU
and 9 on explicit NVMe offload.  Prompt token sequences match every TT control.
All six HF/TT pairs were read and are coherent/on-task with no repetition,
wrong-language drift, or semantic collapse.  HF artifact/generator
script/comparison SHA-256:
`6cb7583a5d129c30ee81e96189198fcafd366ee42ed1070498e1fd3c7cace3f2`,
`6f4c9f1280463c796dea2c491da838b79ae4a1c96c8cfb91b91915d1977d2a28`,
and `67b5c8cf65c666fed837ff473eb100d3c3191618bcb8e21fdb0ea471b247550f`.

Full-stack batch-2 full-logit reproducibility and recovery:

```bash
GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k real_weight_36_layer_batch2_logit_reproducibility -s
```

The initial full-36 run exposed invocation-dependent finite logits while both
greedy tokens still exactly matched HF.  A fresh-context AutoFix retained the
bitwise gate and prescribed bounded physical recovery.  After `tt-smi -r all`,
the unchanged test passed twice in separate processes (352.32 s and 352.76 s).
Prefill and decode now have zero differing values and zero maximum difference
across both physical page rows and both reset/reuse runs; artifact SHA-256 is
`588e98487aad4491d204dcac2ba12ab1dae4ec1068f2876606583ae30d79882d`.

The resident full-context token-out smoke was then rerun with explicit trace
counters and passed in 353.28 s.  It records one initial token/position/page
upload, zero steady-state uploads, four caller-visible scalar-token syncs, and
zero validation full-logit syncs.  Artifact SHA-256 is
`7834b079dd6fee9e7dbc445eb802d6db0001841f37b1f94f17b43ea3bab63737`.

Reduced profiler:

```bash
GPT_OSS_120B_FULL_MODEL_PROBE=1 GPT_OSS_120B_SNAPSHOT=... \
scripts/run_safe_pytest.sh --profile models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -k real_weight_two_layer_full_model_split_sampling_probe -s

/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report \
  --start-signpost FULL_MODEL_TOKEN_OUT --end-signpost FULL_MODEL_TOKEN_OUT_END \
  --active-experts 4 --csv token_out_perf_report.csv --no-summary --no-color \
  generated/profiler/reports/2026_08_29_06_25_16/ops_perf_results_2026_08_29_06_25_16.csv
```

Passed.  The retained losslessly xz-compressed raw CSV and report show 708.553
us for the sharded LM head and 27.500 us for `SamplingDeviceOperation`;
canonical sampling is not the bottleneck.  Eighteen sliding plus eighteen full
layers give a 14.2698 ms/token optimized-layer lower bound.  Against the
19.3712 ms/token token-out wall result, full-model-only work is bounded at
5.1014 ms/token: 0.7086 ms LM head, 0.0275 ms sampling, and a 4.3653 ms wall
residual covering final norm, orchestration, synchronization, scalar readback,
and timer gaps.  The raw artifact records the uncompressed SHA-256 and source.

## AutoFix ledger

- mixed prefill page row: retain the sliced physical row and use local user 0;
- decode batch propagation: derive actual batch from the hidden-state row axis;
- teacher forcing: explicitly refresh caller-authoritative tokens and unwrap
  host-sampler tuples;
- qualitative EOS: honor the complete HF generation stop set and do not force
  post-EOS tokens.
- batch-2 sparse MLP row isolation: replace sub-tile `split` views with explicit
  device-side row materialization through row-major storage, restoring bitwise
  identical logical rows without changing the selected sparse graph or policy;
- full-36 repeated invocation: preserve the exact gate; bounded device reset
  proved stale external fabric/collective state, and the unchanged source passed
  twice afterward.

The corresponding fresh-context reports are under `autofix/`.

## Runner-side gate remediation (2026-08-29)

The independent runner reran
`.agents/prompts/model_bringup_multigoal/06-full-model.check.sh` after the
original completion claim and failed with exit 2 because this checkout lacked
`models/common/readiness_check/check_degenerate_output.py`.  The original
qualitative command above had accidentally invoked the checker from the
separate `/home/ttuser/dev/scratch/tt-metal` checkout, so it did not prove that
the runner-visible path existed here.

The standalone checker was restored byte-for-byte from repository blob
`18865610c5fec9f1f0cd8c27ff915d99492cc35c` (source commit
`8ca2878926da1cd60d6467b92b5235356779b61c`; file SHA-256
`4216f5456f9367c881e8f25b5a26305971dad95a612b296a13038d4cbfec213e`).
After that repair exposed the second half of the gate, the context checker
reported that the JSON used non-canonical top-level field names.  The contract
now records `hf_advertised_context=131072` and
`current_supported_context=131072`, derived from the checked-in HF config's
`max_position_embeddings=131072`; the detailed per-stage capacity accounting
is unchanged.

Focused host verification passed:

```bash
python -m py_compile models/common/readiness_check/check_degenerate_output.py
pre-commit run --files models/common/readiness_check/check_degenerate_output.py
python models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --missing-artifacts critical --scope autoregressive
python .agents/scripts/check_context_contract.py \
  --model-dir models/autoports/openai_gpt_oss_120b \
  --hf-model openai/gpt-oss-120b --stage full-model --require-contract
MODEL_DIR=models/autoports/openai_gpt_oss_120b \
HF_MODEL=openai/gpt-oss-120b \
bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh
```

The final stage gate exits 0.  It measures the retained 100-token TT
autoregressive artifact at zero adjacent duplication, zero replacement-character
corruption, and trigram-loop fraction 0.0896, then reports target and supported
context both 131072.  These changes affect only the host-side runner utility and
contract metadata; no model, generator, trace, cache, sampler, CCL, or device
runtime path changed, so no device reset or silicon rerun was warranted under
`$tt-device-usage`.

The fresh remediation `$stage-review` independently reran the exact gate and
the stricter context-cap check, inspected the generated outputs and full-model
evidence, and returned `clean-pass` with no required work.  Its report is
`stage_review_remediation_20260829.md`.

## Artifacts and review/commit ledger

Exact artifacts are listed in `README.md`.  Independent `$stage-review`
returned `clean-pass` after rereading the corrected implementation, artifacts,
AutoFix evidence, capacity contract, and documentation.  The stage-owned
implementation and evidence are committed locally as
`0a98f5de75fefbf73dede0dcf7d592e8afbc1cab`; the ledger-only follow-up commit
is `fad485049d995e62b35873148049dcd7037e65e0`.  The runner-gate remediation,
canonical context fields, and independent clean-pass review are committed on
`main` as `d6d486ea2673e58426d0537497b328719b49a753`; its ledger-only follow-up is
reported in the final handoff.  Nothing is pushed.
