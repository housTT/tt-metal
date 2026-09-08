# AutoFix: penalty activation for fresh prefills

## Starting evidence

`AUTODEBUG_prefill_penalty_admission.md` records the pre-edit diagnosis.
Six source-executed adapter admissions failed at the real generator guard:
presence, frequency, and repetition penalties, each after neutral requests
with other rows either active or finished. Plugin prefill carries compact
sampling fields and new slots, but no full request histories.

## Repair and verification

The generator now accepts `configure_sampling(..., fresh_slots=None)`. It
validates unique integer slot indices before changing sampler state. When
global penalty tracking was inactive, missing full histories are permitted
only if every penalized lane is declared fresh. The caller must immediately
prefill these lanes at position zero; existing prefill code initializes their
prompt masks and output counts. Live/continuation requests still require real
histories, and a fresh lane cannot authorize a different live lane's penalties.
No histories are fabricated and no warmup preservation logic is removed.

The supervising agent added adapter forwarding: prefill passes only rows with
`start_pos == 0` through `_sampling`; decode leaves the declaration absent.

The exact six originally failing source-executed admissions now pass. Nine
generator-focused tests additionally verify fresh admission, preservation of
all warmup-mutated tensors, refusal of missing/wrong declarations and another
live penalized lane, and invalid-slot rejection before mutation. The combined
run passes 40 tests (15 new admission tests plus 25 existing serving/generator
checks). An additional adapter continuation-prefill regression then passed
separately, bringing the covered tests to 41.

```bash
USER=hous python_env/bin/python -m pytest -q \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_prefill_penalty_admission.py \
  models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_serving_contract.py \
  models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/test_generator_host_contract.py \
  --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/tests --tb=short
```

Black checks and byte compilation passed for the generator and new test file.
This Python-only change needs no C++ build. The subagent used AST execution and
torch stubs only; it did not import TTNN or touch devices.

## Live verification and remaining scope

The supervising lane's real reduced-layer `reduced_v4` async server, batch 4,
passed sequential 131-token prompts with eight generated tokens each through
neutral → presence penalty 1 → frequency penalty 1 → repetition penalty 1.5
→ explicit host logprobs 3 → device neutral. Evidence is
`reduced_v4_transitions_v2.json` and `.log`. Every response is HTTP 200 with a
real choice and usage showing 131 prompt / 8 completion / 139 total tokens;
targeted transition assertions also pass. The first harness version compared
the entire usage dictionary and falsely rejected the optional
`prompt_tokens_details: null` field; the corrected harness checks the required
usage fields without suppressing response errors or token-count mismatches.
Async concurrent nonaligned requests also pass in that reduced server.

Status: the originally predicted admission failure is repaired and verified
through the actual serving path. Source checks cover preservation when other
neutral rows remain live and refusal of continuation-as-fresh declarations.
Full-model mixed-penalty concurrency and all-profile release gates remain at
the parent stage; these reduced-layer structural responses do not establish
qualitative model accuracy.
