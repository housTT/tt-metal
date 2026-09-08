# H2 prefill sampling trace candidate

Starting evidence: `AUTODEBUG_prefill.md` H2 in the supervising checkout. The
serving `_sample_prefill_device` unconditionally submitted eager sampling.
Its request-local backup clones also outlived an older sampler trace, so merely
switching that call to replay would violate native allocation tracking.

Candidate: allocate logits staging, token/UINT32-seed/history backups and INT32
lane/row masks before initial capture. Warm the exact copies and restoration
programs then. At prefill, stage and retire transient logits before recapture,
copy into the current canonical logits afterwards, copy state into resident
backups, require canonical sampler replay, and restore untouched lanes exactly.
A late program-cache miss raises; serving does not fall back to eager sampling.
The standalone `_sample_first_token` default remains unchanged. Teardown releases
traces before resident sampling buffers. No H1 selector/startup change is included.

Host evidence, run in the sampling worktree on 2026-09-08:

```bash
python -m pytest -o addopts='' -q --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/tests models/autoports/ornith_ai_ornith_1_5_9b/tests/test_prefill_sampling_trace.py models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_serving_contract.py models/autoports/ornith_ai_ornith_1_5_9b/tests/test_prefill_penalty_admission.py
# 60 passed
python -m pytest -o addopts='' -q --confcutdir=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/test_prefill_trace_contract.py models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/test_trace_lifecycle.py
# 23 passed
```

The shared lifecycle fixture bypasses `__init__`; it now initializes the three
new attributes. Seven initial teardown failures were caused solely by that
fixture omission. The new host contracts execute actual generator methods,
cover B1/B32 partial/all admissions, penalties on/off, canonical/external logits,
program-count changes, exact large token/seed/history values, freeing external
logits before replay, stable backup/mask buffers and teardown order. Python
compilation, deferred-import probe `--help`, Black targeting Python3.10 and
`git diff --check` passed. No build is required for these Python-only changes.

Prepared serialized device commands (not run by the authoring subagent):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.prefill_sampling_device_probe --model-path "$ORNITH_MODEL_PATH" --batch 1 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/h2_sampler_b1.json
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.prefill_sampling_device_probe --model-path "$ORNITH_MODEL_PATH" --batch 32 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/h2_sampler_b32.json
```

The probe uses real layers0/3, native logical context and an external reduced
physical pool. It exercises ordinary public model prefill for each greedy →
seeded → penalties → greedy transition, then compares synthetic exact-shape
sampling with the eager common sampler. It checks all physical ranks and
untouched lanes, populated canonical logits through forced recapture, transient
external-logits retirement under the native guard, resident buffer addresses,
and zero trace bytes after teardown. Runtime eager sampling inside the prefill
helper is forbidden; warm/capture and the explicit numerical oracle are allowed.
DRAM/L1/TRACE memory views and per-buffer logical/padded element bytes are
recorded. The prepare allocation delta includes helper program-cache warmup;
element byte counts do not claim allocator bank/alignment overhead.

Status: candidate ready for serialized native-tracker device qualification.
No device/TTNN import, performance claim, commit or push was performed here.
Full-model serving and capacity/accounting evidence remain with the supervisor.

Follow-up movement review: the separate `optimized-vllm-h2-mask-refresh.patch`
caches sorted admission membership and refreshes the two resident masks only
when it changes. Cache initialization matches the all-one device masks; it
survives trace recapture and clears with buffer teardown. Two added host tests
count copies through repeated/reordered/changed admission and recapture: the
host suites now pass 62 + 23 tests. Preservation remains unchanged: B1 row0
still has 31 untouched physical sampler lanes, so a logical full-batch shortcut
would violate the exact state contract. A future all32-lane shortcut is possible
but is not included in this candidate.

Probe API review corrected missing required `top_p=1.0` arguments in its three
greedy parameter constructors. All four actual mode expressions were then
executed from AST against the actual common `SamplingParams` dataclass without
TTNN import; construction succeeded. The corrected probe patch supersedes its
initial version.
