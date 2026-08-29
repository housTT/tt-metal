# Datatype-sweep work log

Date: 2026-08-29 UTC

Model: `openai/gpt-oss-120b` at revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`

Target: P150/P150x2/P150x4.  Full-model measurements use a 1x4 mesh of four
physical p300c devices as the P150 semantic target.  P150 and P150x2 are
capacity-accounted because the fixed resident 120B model state exceeds
32 GiB/device.

Starting repository HEAD: `8166f54d`.  Completed optimized-full-model stage
commit present in history: `1989f631`.

## Stage contract and inventory

The completed optimized full model, AIME24 reference, selected generator path,
context contract, prior qualitative controls, and optimized token-out benchmark
were inspected before changing precision behavior.  This stage intentionally
does not begin vLLM integration.

Accuracy gates:

- top-1 >= 0.90;
- top-5 >= 0.98;
- top-100 recorded but not gated;
- final ranking: maximum warmed trace-verified teacher-forcing decode
  tokens/s/user among configs that pass both accuracy gates.

Reference:
`doc/full_model/references/aime24_chat_100_top100.refpt`, chat-template prompt
length 214, 100 generated/teacher-forced tokens, top-k 100.

## Device safety

All pytest hardware work ran through `scripts/run_safe_pytest.sh`, which owns
`/tmp/tt-device.lock`.  Standalone health/recovery checks used the same lock.
The initial and post-candidate health checks reported four visible devices and
zero device errors.

Exact bounded recovery commands used after an OOM/invalid-collective sequence:

```bash
flock /tmp/tt-device.lock timeout 60 tt-smi -ls --local
flock /tmp/tt-device.lock timeout 180 tt-smi -r
flock /tmp/tt-device.lock timeout 60 tt-smi -ls --local
```

The recovery was followed by a locked Python mesh smoke using
`ttnn.open_mesh_device(ttnn.MeshShape(1, 4))`; it opened and closed cleanly with
`MESH_SMOKE_OK`.  No hardware tests were run concurrently.

## Baseline refresh

The final baseline refresh used the normal selected-config construction path,
the main AIME24 chat-template reference, 100 teacher-forced tokens, one
discarded same-generator warmup, and two timed repetitions:

```bash
env PATH=$PWD/python_env/bin:$PATH \
  GPT_OSS_120B_FULL_MODEL_READINESS=1 \
  GPT_OSS_120B_DATATYPE_SWEEP_SELECTED=1 \
  GPT_OSS_120B_DATATYPE_SWEEP_REPETITIONS=2 \
  LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:$PWD/build/lib \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k run_teacher_forcing_aime24_top100 -s
```

Result: top-1/top-5/top-100 = 0.95/1.00/1.00, TTFT 3634.9465 ms,
traced teacher-forcing decode median 60.8008 t/s/u (60.8041, 60.7975).
The artifact proves that the measured path loaded
`selected_precision_config.json`, config hash
`b8e1e655581ffca37dd8b841285940c494a5e43d6b5deec3ebc10078ddb52c57`.
Each timed 99-step decode began with one captured model and one captured
sampling trace and submitted each trace exactly 99 times, with zero
unclassified submissions.  The measurement source-state hash is
`5f7cc129329864bf6c5977fede08235ad3b90f3fc12a4da61340f69d625a7e96`;
the external readiness runner SHA-256 is
`11f812bd493c687e61710bf3f81e0df749f88f14f47677c6497394d3cc1537a8`.

An earlier cold result (11.6368 t/s/u) was diagnosed during the first stage
review as lacking same-generator warmup and sufficient trace/source provenance.
It is retained only as historical evidence and is excluded from ranking and
final baseline reporting.

## Runtime precision plumbing

Added a strict versioned precision loader with default path
`doc/datatype_sweep/selected_precision_config.json` and explicit candidate
override.  The policy is consumed by the 36-layer constructor and terminal
path, covering:

- embedding, attention, router, expert, norm, and LM-head weights;
- per-layer exceptions;
- attention/SDPA/expert/router/LM-head compute fidelity;
- residual, attention projection input, and expert intermediate dtypes;
- attention/expert CCL dtypes;
- KV-cache dtype and dtype-aware capacity;
- LM-head output/logit-shard and device sampling accumulator assumptions.

Runtime validation emits actual constructed policy groups and terminal sampling
buffer dtypes.  Hard-coded behavior that disagrees with a selected field raises
during construction.

## Candidate sweep

Each candidate used this command with `<candidate>` replaced by its repo-local
candidate JSON path; every fully expanded command and measurement regime is
stored in the corresponding `sweep_results.json`/CSV row:

```bash
env PATH=$PWD/python_env/bin:$PATH \
  GPT_OSS_120B_FULL_MODEL_READINESS=1 \
  GPT_OSS_120B_DATATYPE_SWEEP_CONFIG=models/autoports/openai_gpt_oss_120b/doc/datatype_sweep/candidates/<candidate>.json \
  LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:$PWD/build/lib \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k run_teacher_forcing_aime24_top100 -s
```

`GPT_OSS_120B_DATATYPE_SWEEP_REPETITIONS=2` was added for `ds00`, `ds11`,
and `ds12`; all other measured candidates used one timed repetition after the
discarded same-process warmup.  Every ranked repetition has exact runtime
trace proof: handles existed before timing, 99 model and 99 sampling
`execute_trace` submissions occurred, and no submission was unclassified.

Executed candidate files:

`ds00_baseline`, `ds01_attention_bfp4_lofi`,
`ds02_attention_bfp4_hifi2`, `ds03_expert_bfp4_hifi2`,
`ds04_canonical_bfp8_hifi2`, `ds05_kv_bfp4`, `ds06_kv_bf16`,
`ds07_expert_ccl_bfp8`, `ds08_all_ccl_bfp4`, and
`ds09_expert_intermediate_bfp8`, `ds10_attention_bfp8_hifi2`,
`ds11_lm_head_lofi`, and `ds12_attention_hifi2_lm_head_lofi`.

Results:

| Config | Top-1/top-5/top-100 | TTFT ms | traced TF t/s/u | Status |
| --- | --- | ---: | ---: | --- |
| `ds00_baseline` | .95/1/1 | 3630.9383 | 61.0382 | pass/selected |
| `ds01_attention_bfp4_lofi` | .89/1/1 | 3639.1191 | 60.8324 | fail top-1 |
| `ds02_attention_bfp4_hifi2` | .89/1/1 | 3635.0639 | 60.8710 | fail top-1 |
| `ds03_expert_bfp4_hifi2` | .95/1/1 | 3599.7911 | 60.5643 | pass/slower |
| `ds04_canonical_bfp8_hifi2` | unavailable | unavailable | unavailable | physical OOM |
| `ds05_kv_bfp4` | .78/.95/1 | 3632.8593 | 60.9527 | fail both gates |
| `ds06_kv_bf16` | .94/1/1 | 3637.0268 | 60.8398 | pass/slower |
| `ds07_expert_ccl_bfp8` | .92/1/1 | 3630.6817 | 58.0519 | pass/slower |
| `ds08_all_ccl_bfp4` | unavailable | unavailable | unavailable | exact TTNN blocker |
| `ds09_expert_intermediate_bfp8` | .94/1/1 | 3964.6754 | 54.2010 | pass/slower |
| `ds10_attention_bfp8_hifi2` | .94/1/1 | 3630.3374 | 60.7639 | pass/slower |
| `ds11_lm_head_lofi` | .95/1/1 | 3631.7833 | 61.0300 | pass/slower |
| `ds12_attention_hifi2_lm_head_lofi` | .95/1/1 | 3631.0349 | 60.8092 | pass/slower |

The exact `ds04` allocation error and `ds08` layernorm legality fatal are stored
under their artifact directories with raw-log SHA-256, measurement-time source
state, runner provenance, command, and post-failure health.  `ds08` proves that
the BFP4 residual produced at the attention CCL/normalization boundary is not a
legal layernorm input; it is not a BFP4 matmul-group waiver.  Expert BFP4 uses
LoFi in `ds00` and HiFi2 in `ds03`; attention BFP4 uses LoFi in `ds01` and HiFi2
in `ds02`.  The first review's missing isolated BFP8-attention HiFi2 and LM-head
LoFi axes were added as `ds10` and `ds11`, plus combined `ds12`.

The deterministic result generator was run after all candidate artifacts:

```bash
python_env/bin/python3 \
  models/autoports/openai_gpt_oss_120b/doc/datatype_sweep/generate_results.py
```

It generated both tables and both pyplot charts and asserted that the selected
config is the fastest passing trace-verified row.  Both plots were visually
inspected: evaluated points, Pareto line, red selected star, and dotted accuracy
threshold are present.

## Selection and context

Selected config: `ds00_baseline`.  The selected JSON is identical to the
measured `ds00` candidate and is automatically loaded by the normal generator
construction path.  Semantic config SHA-256:
`b8e1e655581ffca37dd8b841285940c494a5e43d6b5deec3ebc10078ddb52c57`.
The selected policy is BF16 embedding/router/norm weights, BFP8 attention and
LM-head weights, BFP4 expert weights, LoFi decode-attention/expert math, HiFi2
prefill-attention/router/LM-head math, HiFi4 SDPA, BF16 residual/expert
intermediate, BFP8 attention projection and attention CCL, BF16 expert CCL,
BFP8 KV cache, BFP8 LM-head output, BF16 top-k values/sampling accumulator,
and no layer exceptions or materialized full-vocabulary logit gather.

`doc/context_contract.json` was recomputed for BFP4/BFP8/BF16 KV caches across
P150/P150x2/P150x4.  P150x4 preserves 131,072 context for all candidates.  The
P150/P150x2 fixed resident model footprint exceeds 32 GiB/device for all three,
so largest feasible context is zero and the hard physical limit is recorded.

## Post-selection token-out and AutoFix

Exact strict command:

```bash
env PATH=$PWD/python_env/bin:$PATH \
  GPT_OSS_120B_FULL_MODEL_ACCEPTANCE=1 \
  GPT_OSS_120B_FULL_TRACE_LIFECYCLE_ISOLATION=1 \
  GPT_OSS_120B_FULL_ASYNC_ISOLATION_LENGTHS=7,8,122,128 \
  GPT_OSS_120B_DATATYPE_SWEEP_SELECTED=1 \
  LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:$PWD/build/lib \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k real_weight_36_layer_full_context_token_out_smoke -s
```

The first invocation completed the non-aligned prompt-214/gen-100 section but
reported an async endpoint mismatch at prompt length 128 after the earlier
candidate OOM and invalid collective.  Two fresh-context AutoFix investigations
reviewed the unchanged trace/token-feedback path and prior optimized-stage
history.  The evidence matched a stale device/fabric state signature; no source
change was proposed or retained.  The bounded reset and 1x4 mesh smoke above
restored the environment.

The exact unchanged strict command then passed twice in fresh processes.  It
was rerun once more after the final provenance/schema edits and passed:
`1 passed, 41 deselected` in 375.85 s.  The final-source artifact records
prompt-214/gen-100 TTFT 3.6304 s and 62.7727 t/s/u, plus prompt-128/gen-128 TTFT
0.4835 s and 62.8877 t/s/u.  First/final tokens match synchronous readback, only
two caller-visible token reads occur on the 128-token split path, and all
isolation lengths 7, 8, 122, and 128 pass.

## Selected qualitative check

Exact command:

```bash
env PATH=$PWD/python_env/bin:$PATH \
  GPT_OSS_120B_FULL_MODEL_QUALITATIVE=1 \
  GPT_OSS_120B_DATATYPE_SWEEP_SELECTED=1 \
  LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:$PWD/build/lib \
  scripts/run_safe_pytest.sh \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py \
  -q -k shared_qualitative_chat_suite -s
```

Final-source result: `1 passed, 41 deselected` in 365.23 s.  All six
selected-path prompt token sequences exactly match the
exact-checkpoint HF control.  All generated completions begin at the shifted
assistant analysis-channel boundary.  Direct review found coherent, on-task,
correct-language output with no phrase loop, stale-token behavior, replacement
characters, or malformed control tail.  The detailed review is
`artifacts/selected/qualitative_review.json`.

## Verification and review

Python/host verification:

```bash
env LD_LIBRARY_PATH=/tmp/gptoss-libnsl2:/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib \
  python_env/bin/python3 -m pytest \
  models/autoports/openai_gpt_oss_120b/tests/test_full_model.py -q \
  -k 'full_stack_capacity or kv_candidate_capacity or selected_precision or capacity_failure or reduced_stack or generator_page_table or streaming_checkpoint or aime_reference or layer_adapter or readiness_generate or sampling_decision or generator_uses_complete or shared_qualitative_hf or trace_evidence or reset_clears or unseen_prefill or split_greedy_submission or mixed_prefill' -ra
```

Final post-format result: 30 passed, 12 hardware-gated skips in 6.93 s.  This includes the
selected-default, all-candidate schema, all KV dtype capacities, AIME provenance,
HF qualitative-control, trace lifecycle, fixed-slot, and non-aligned page-table
host contracts.

`pre-commit run --files ...` passed for every stage-owned source, test, JSON,
CSV, markdown, and PNG artifact after its first isort fix.  `git diff --check`,
all datatype-sweep JSON parse checks, context-contract JSON validation,
deterministic result regeneration, and the fastest-passing selection assertion
also passed.  This is a Python/docs-only source change, so the repository
AGENTS.md matrix does not require a C++ build.

The hardware artifacts retain the exact measurement-time source manifest.  The
final pre-commit pass subsequently changed only Python import ordering in
`tests/test_full_model.py`: its measured hash is
`04b2b9d784e4e093b8cfb10bc0261f9b9aba4b2d39f237d7c461c4a5dbbd160c`,
its formatted hash is
`b34faf0f582818874840e4633ef67b6dcf704a954198c1732df49493629d56ad`,
and the post-format runtime-source-manifest hash is
`01810c66559ef8a6a29b79df03f42040c3d6425742b019739792e974493bc29e`.
No runtime model/generator/decoder source changed after measurement, and the
complete 30-test host suite passed on the formatted tree.

Final locked `tt-smi -s`: four p300c devices visible, DRAM status true on every
device, GDDR corrected/uncorrected error counters zero, therm-trip counts zero,
and no process remained after the health command exited.

Independent fresh-context stage review: `clean-pass`; no required work.  The
reviewer independently recomputed the winner, checked all 13 result/config
hashes and ranked trace submissions, inspected runtime consumption and capacity
evidence, rendered both plots, audited both blockers and the anomaly ledger, and
confirmed that no vLLM work entered the scoped diff.  The complete retained
verdict is `stage_review.md`.

Stage-owned commit: pending.

Commit-log update: pending.
