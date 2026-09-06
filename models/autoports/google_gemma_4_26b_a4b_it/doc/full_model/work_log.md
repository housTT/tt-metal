# Full-model work log

## Scope and starting state

Started from branch `hous/gemma-4-26b-a4b-it` at
`e9ac7fc33287019b2808b741830fa9b595671743` (`Record Gemma 4 optimization
commit`). This stage changes only the full-model wrapper, generator, focused
tests, context contract, and full-model evidence. It does not modify
`generator_vllm.py` or begin vLLM integration.

The selected starting block is the complete optimized multichip decoder at
`d03bc791f7c`. Its dtype/fidelity, dynamic top-8 sparse routing, packed expert,
profile-specific retained-copy, BF16 activation/cache/CCL, persistent
collective, and replicated residual contracts were treated as invariants.
The full inherited candidate/rejection ledger remains in
`doc/optimized_multichip_decoder/README.md`, `AUTOFIX.md`, and its artifact
tree; no rejected decoder alternative was silently reintroduced.

## Implementation sequence

1. Generalized the former TP4-only wrapper to the supported TP1/TP2/TP4
   meshes and retained the established 50,624/262,144/262,144 profile limits.
2. Sharded the tied embedding and LM head over hidden/vocabulary axes and used
   physical BFP8_B terminal storage on all profiles. The LM head leaves logits
   vocabulary-sharded for split sampling.
3. Instantiated all 30 optimized decoder layers. Persistent TP2/TP4 collective
   buffers are deliberately allocated after weight setup to avoid pinning the
   L1 region needed for DRAM-to-sharded conversion, then shared by every layer.
4. Allocated the exact BF16 cache split: 25 sliding layers with 64-token pages
   and 1,024-token slots, plus five full layers with 128-token pages through
   the profile context. Added explicit stable page tables, slot lengths,
   positions, active rows, and in-place reset.
5. Implemented logical prompt padding/masking/slicing, mixed prompt lengths,
   the full HF terminal path, and low/high-level generator interfaces.
6. Selected common `Sampling1D` after comparing both common samplers. Added
   split model/sampling traces, direct `tt_out_tok` device feedback, traced
   position advance, stable page-table refresh, sampled-mode trace keys, and
   explicit host compatibility mode.
7. Built a reduced two-real-layer probe for trace, cache, state, performance,
   sampler, and profiler debugging, then returned to all-layer gates.

## AutoDebug and first stage review

The `$autofix` loop ran a fresh-context AutoDebug pass; its complete report is
retained as `AUTODEBUG.md`. The first independent `$stage-review` returned
`more-work-needed` for eight concrete issues: padded prompt tokens reaching
the decoder, capture-time KV mutation and end-of-context overflow, stale reset
and active-mask positions, sampled seed restart, unstable wrappers for raw
caller KV, batched all-logits shape, incomplete B32 row coverage, and missing
public-generator performance evidence.

The repair pass now slices hidden state and positions to logical prefill length,
captures every model trace with all rows inactive, restores request state after
capture, advances canonical host positions only after replay, uses an in-trace
UINT32 seed-skip sentinel, memoizes borrowed caller state by cache/page-table
identity, returns one padded host logits tensor for mixed batches, and tests all
32 sampler rows against B1 and global-logit controls. It also added the public
`generate()` benchmark and isolated initial-token sampling allocations so a
retained trace can safely survive reset and a new prefill.

Targeted allocation-tracked hardware gates passed for logical prompt tails
1025/1055, mixed active/inactive transitions and teacher-trace reuse, sampled
RNG continuation, complete B32, and the two-request qualitative reset path.
The last path first exposed five short-lived first-token sampler buffers; the
buffers were placed in an explicit corruptible lifetime scope and the exact
failing two-request run then passed.

## Hardware validation

All device commands ran serially. The hardware is one four-chip Blackhole
P300C QB2 used as accepted P150-family proxies: 1x1, 1x2 submesh under
FABRIC_2D, and 1x4 Ring. The checkpoint is the offline snapshot revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

The complete stack load/prefill/decode gate passed independently on all three
profiles. The final-source all-layer TP4 B32 rerun also checked every sampled
row against its global-max logit and measured minimum B1/B32 cosine 0.9929.
Representative focused commands were:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=/tmp/gemma4_full_model_evidence \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150]'

env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_BATCH32_PROBE=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=/tmp/gemma4_full_model_evidence \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150x4]'
```

The retained capacity gates construct the full stack at every maximum profile.
Public-generator long probes passed at 50,623 tokens on TP1 and 262,143 on
TP2/TP4, with physical padding hidden from the API, one remaining legal replay,
and rejection of the following out-of-capacity step.

Allocation tracking was run separately from Tracy. Every final profile, all
three 100-token teacher-forcing gates, B32, sampled RNG, mixed/reset state, and
refreshed qualitative generation completed under
`TT_METAL_TRACE_ALLOC_TRACKING=1`. Generic non-tracked runs can emit the
allocator's conservative active-trace warning while capturing the second split
trace; the tracked controls establish that retained buffers are safe.

## Accuracy, quality, and performance

The historical readiness package was materialized from the parent revision in
`/tmp/gemma4-readiness.T6WBn5` because it was removed from the current source
tree. The exact-revision reference metadata matches the pinned checkpoint,
AIME24 prompt 0, tokenizer chat template, 100 generated tokens, and top-k 100.
Prefill and traced teacher forcing scored 0.95/0.96/0.95 top-1 on
P150/P150x2/P150x4; every profile scored 1.0 top-5 and top-100.

The prompt-quality pass used the `$qualitative-check` contract. The shared
six-prompt readiness source was recovered from the exact historical package,
then every prompt was rendered with the pinned checkpoint chat template and
run for 64 greedy tokens through both HF and the full traced TT generator.
The same TT model was reset between prompts under allocation tracking. All six
outputs passed human review for task alignment, coherence, repetition,
wrong-language drift, and suspicious early divergence; the source, rendered
prompts, token IDs, outputs, and per-case assessment are retained in
`artifacts/qualitative/shared_readiness_suite/`. A focused runner also applied
the checkpoint's chat template to three user prompts and compared 64-token
greedy HF and traced TT completions. These outputs were
refreshed after the trace-state repair, including two requests through one
model and retained trace with `reset()` between them. The shared mechanical
degeneracy checker returned exit 0 on the main chat and raw artifacts. Exact
metadata, rendered prompts, controls, outputs, and verdict are retained under
`artifacts/qualitative/`.

The shared-suite invocation was:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_shared_qual.py
```

The runner read the exact six prompts from
`/tmp/gemma4-readiness.T6WBn5/models/common/readiness_check/vllm_prompts.txt`.
That file, its SHA-256, rendered prompts, token IDs, and outputs are retained in
the suite directory, so the evidence no longer depends on the temporary source.

```bash
python_env/bin/python \
  /tmp/gemma4-readiness.T6WBn5/models/common/readiness_check/check_degenerate_output.py \
  --model-dir models/autoports/google_gemma_4_26b_a4b_it \
  --hf-model google/gemma-4-26B-A4B-it --scope autoregressive \
  --missing-artifacts critical \
  --json models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts/qualitative/degeneracy_final.json
```

Primary P150x4 performance is the public B1 prompt-128/generate-128 contract:
139.029 ms TTFT, 44.433 end-to-end t/s/u, 46.322 request decode t/s/u, and
46.327 warmed decode t/s/u. It includes first-token sampling, the caller-visible
Python list, and 128 minimal token readbacks. The prompt-128 prefill component
is 106.118 ms warmed. Separate position-32 component probes measure
20.9238 ms/47.7925 decode/s through sampler-ready logits and
21.3730 ms/46.7880 t/s/u through split on-device sampling and feedback. They
are not subtracted as a matched overhead experiment. The optimized token-out
measurement loop performs no host readback or synchronization.

The reduced layers 0/5 Tracy command was:

```bash
env -u TT_METAL_WATCHER HF_HOME=/home/hous/.cache/huggingface \
  HF_HUB_OFFLINE=1 GEMMA4_FULL_MODEL_PROBE=1 \
  GEMMA4_FULL_MODEL_DEVICE_PROFILE=1 \
  python_env/bin/python -m tracy -r -p \
  -o gemma4_full_model_reduced_tp4 -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150x4]'

python_env/bin/tt-perf-report <raw-ops.csv> --arch blackhole \
  --active-experts 8 --start-signpost FULL_MODEL_REDUCED_DECODE \
  --end-signpost FULL_MODEL_REDUCED_DECODE_END \
  --csv reduced_tp4_decode_ops.csv \
  --summary-file reduced_tp4_decode_summary
```

The raw report is losslessly compressed under `artifacts/profiler/`. Tracy and
watcher/allocation tracking were never combined.

The fresh stage rereview confirmed the original eight findings were repaired
and requested explicit terminal trials. A serialized TP4 reduced-real-shape
campaign tested four embedding/LM-head alternatives. Persistent BF16 row-major
embedding cut 1.430 ms to 0.136 ms but its TP1 incremental storage exceeds the
268,670,464-byte full-context headroom. Generic BFP8 tiled gather produced
non-finite values and was 76.050 ms. A monolithic DRAM-sharded LM head failed
an exact L1 circular-buffer allocation check; the valid mature 8,192-column
split took 0.1231 ms, making eight TP4 linears a 0.9850 ms lower bound before
concat versus the 0.5904 ms incumbent. The mature interleaved TP4 program
config changed latency by only 0.43%, while pre-scaled softcap variants changed
two entries in at least one top-100 set. All alternatives were rejected and
the capacity-required BFP8 terminal contract remains unchanged. Exact samples,
numerics, configurations, and blockers are in `artifacts/terminal_trials.json`;
the main boundaries are summarized in `perf_summary.json`.

The serialized terminal commands all used the same strict fallback and
allocation-tracking environment:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_terminal_trials.py

env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_embedding_gather_trial.py

env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_lm_config_trial.py

env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_softcap_fusion_trial.py

env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_lm_split_trial.py
```

## Final verification and commits

The final source was exercised with fallback exceptions enabled in one
serialized three-profile invocation:

```bash
env TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts/final_profiles.xml
```

All three distinct P150-family proxy profiles passed on the final source: P150
in 52.10 s, P150x2 in 53.48 s, and P150x4 in 67.26 s (3 passed in 176.74 s). The retained
JUnit report is `artifacts/final_profiles.xml`. A final
`timeout 60 tt-smi -ls --local` showed all four Blackhole P300C devices
available and reset-capable after the run.

The final independent review found one sampler-transition lifecycle bug: when
`top_k`/`top_p`/temperature changed during a steady device request, recapture
could initialize the new trace from the intentionally stale host token. The
fix retains the previous trace's aliased device-feedback tensor across trace
release and uses it as the new trace input; sampler warmup now writes to a
separate temporary output. Because trace capture itself executes the aliased
sampler output, the generator also backs up and restores the retained token
device-to-device around capture. The focused regression exercised
greedy -> sampled -> greedy with host token 0 on both transitions and a
correct-token control. Both transitions matched top-1 and the top-100 set with
cosines 0.9999671 and 0.9999622, and the sampled token matched exactly. It ran
with strict fallback rejection and allocation tracking:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_SAMPLED_TRACE_PROBE=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/artifacts \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150x4]'
```

The same regression covers the related page-table lifecycle boundary.
Identity-distinct scheduler tables are adopted by copying once into the
existing stable tensors, preserving their device addresses and both trace IDs.
Repeated use of the same source performs no additional copy, and a second
source increments the refresh count exactly once. A fixed-logit sampled
control proves the next random token is identical to the uninterrupted stream.

The public stochastic lifecycle was then rerun through `generate()`: seed 42
reproduced the same 16-token sequence before and after `reset()` while retaining
the same model/sampling trace IDs; seed 43 differed, and each sequence advanced.
The retained report is `artifacts/public_sampling_rng_tp4.json`.

`python_env/bin/pre-commit run --files ...` passed all applicable hooks after
the final source formatting. The final host-side contract invocation reported
13 passed and 5 explicitly environment-gated hardware tests skipped. Every
retained JSON artifact passed `jq empty`, and `git diff --check` passed. No C++
or CMake file changed, so the repository `AGENTS.md` matrix does not require a
build.

The fresh independent `$stage-review` returned `clean-pass` with no required
work or missing hard gate. Its retained report is `stage_review.md`. Local
commit records follow below.

## Local commit record

- Starting commit: `e9ac7fc33287019b2808b741830fa9b595671743`.
- Full-model implementation, tests, evidence, and clean review:
  `769a0c593b56e55f36b477bb19321d6bde9bb831`.
- No commit was pushed. The documentation-only commit that records this SHA is
  reported in the final handoff.

## Runner-side gate repair (2026-09-06)

The post-completion runner check
`.agents/prompts/model_bringup_multigoal/06-full-model.check.sh` failed with
exit 2 before inspecting any model artifact:

```text
python: can't open file '/home/hous/dev/tt-metal/models/common/readiness_check/check_degenerate_output.py': [Errno 2] No such file or directory
```

Root cause: the stage had run the degeneracy check from a temporary historical
checkout (`/tmp/gemma4-readiness.T6WBn5`) because the shared readiness package
was removed from the current source tree. The completion evidence therefore
did not validate the runner's actual in-tree entry point, even though the
runner script still invoked it. This was a gate-packaging defect, not a TT
generation or device-runtime failure.

The repair restores the standalone, standard-library-only
`models/common/readiness_check/check_degenerate_output.py` entry point and adds
focused tests for clean output, doubled-token critical failure, model-scoped
autoregressive artifact discovery, empty TT payload rejection,
text-plus-token-ID collapse detection, malformed JSON-schema rejection, and
advisory/critical missing-artifact policy. It does not change the model,
generator, precision, multichip, trace, sampling, or capacity paths.

Verification from the repository root:

```bash
python_env/bin/pre-commit run --files \
  models/common/readiness_check/check_degenerate_output.py \
  models/common/readiness_check/test_check_degenerate_output.py

python_env/bin/pytest -q \
  models/common/readiness_check/test_check_degenerate_output.py

python_env/bin/pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py

env MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it \
  HF_MODEL=google/gemma-4-26B-A4B-it \
  bash .agents/prompts/model_bringup_multigoal/06-full-model.check.sh

git diff --check
```

All pre-commit hooks passed, the focused suite reported `11 passed`, and the
exact runner gate exited 0. It measured both retained 64-token free-running
artifacts: the chat completion has adjacent duplication 0.0000 and trigram-loop
fraction 0.0612 by words and 0.0469 by token IDs; the raw continuation stress
artifact has adjacent duplication 0.0000 and trigram-loop fraction 0.2941 by
words and 0.2344 by token IDs. The context subcheck independently reported
target/support 262,144 (`full HF context`).

The first fresh independent rereview returned `more-work-needed` because the
historical checker semantics allowed an empty TT payload to count as evidence
and skipped token-ID checks whenever nonblank text existed. Both findings were
reproduced and fixed: the configured missing-artifact severity now applies to
empty payloads, and decoded words plus token IDs are both checked when both are
available. The two new regression tests prove that an empty payload and a
64-token single-token collapse hidden behind fused nonblank text each return
exit 2. The first review and exact reproductions are preserved in
`stage_review_runner_recheck.md`; a clean rereview is required after this
remediation.

The next rereview probe exposed one further false-pass family: syntactically
valid but schema-invalid metadata (`null`, string or object `token_ids`, and
`[null]`) could be treated as an inspectable artifact. The loader now
distinguishes parse failure from JSON `null`, requires object-shaped
autoregressive/TT metadata, and accepts TT token IDs only as a list of
non-negative JSON integers. Each reproduced payload has a regression test that
returns exit 2. Equivalent basic list/item/completion-string validation was
added for the checker's shared vLLM artifact path, without changing the stage-6
`autoregressive` scope.

The fresh post-remediation independent review returned `clean-pass` with no
required work. It independently reran all 11 focused cases and the exact
stage-6 runner, checked the final checker/test hashes, inspected the retained
Gemma chat/story and shared-suite outputs, and revalidated source provenance.
The final report is `stage_review_runner_rereview.md`; the earlier
`more-work-needed` report remains beside it as the remediation audit trail.

No TT device command was needed to diagnose or verify the checker itself. A
subsequent unfiltered host-contract pytest invocation reported
`13 passed, 5 skipped`; its standard fixtures briefly opened and cleanly closed
the available four-device mesh before each environment-gated hardware case
skipped, but no real-weight model probe ran. The existing serialized all-profile
and watcher/allocation evidence therefore remains the applicable model-hardware
record.

### Runner-gate repair checkpoint

- Repository: `/home/hous/dev/tt-metal`
- Branch: `hous/gemma-4-26b-a4b-it`
- Repair, regressions, and independent review commit:
  `50ddb4d999a2bca24bf951dd89083730b58860bb`
- No commit was pushed. This documentation-only follow-up records the repair
  checkpoint and is reported in the final handoff.
