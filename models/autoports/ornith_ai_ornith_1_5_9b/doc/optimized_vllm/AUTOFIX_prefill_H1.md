# AutoFix H1: external B1 prefill trace selection

Starting evidence: `AUTODEBUG_prefill.md` in the supervising main checkout.
The source-only selector experiment verified that `owns_cache` excluded every
external serving request from the existing B1 prefill trace body.

The candidate removes that eligibility exclusion, preserves external ownership
and exact cache identity, and retains the first external trace shape across
both live and reset shape misses. Startup selects its exact128 shape before
`ensure_traces(preserve_cache=False)`, avoiding a native-pool snapshot/rebuild
on the first real prefill. B>1, continuation, all-logits, and long prompts retain
the eager path. H2 sampling implementation is unchanged.

Host verification in the isolated `optimized-vllm-prefill-worktree`:

```bash
/home/hous/dev/ornith-1.5-9b/state/serving-env/bin/python -m pytest --noconftest -q models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_full_model/test_prefill_trace_contract.py models/autoports/ornith_ai_ornith_1_5_9b/tests/test_serving_startup_warmup.py models/autoports/ornith_ai_ornith_1_5_9b/tests/test_generator_serving_contract.py
python -m py_compile models/autoports/ornith_ai_ornith_1_5_9b/tests/external_prefill_trace_device_probe.py
python models/autoports/ornith_ai_ornith_1_5_9b/tests/external_prefill_trace_device_probe.py --help
black --line-length 120 models/autoports/ornith_ai_ornith_1_5_9b/tests/external_prefill_trace_device_probe.py
git diff --check
```

Result: **43 passed in 1.13s**; compilation, deferred-import help and whitespace
checks passed. Black formatted the probe; a follow-up `--check --target-version py310 --line-length 120` passed for all three modified/new test files.
Python-only change; no build needed. No TTNN imports or hardware commands were
performed by this resumed authoring agent.

The new `tests/external_prefill_trace_device_probe.py` is ready for the serialized
hardware lane. It uses the real adapter allocation helper with4098 blocks,
B1/native262144, the selected precision unchanged, sequential candidate/eager
pools, layers0/3 by default and all32 with `--full32`. It covers literal first131
after startup128, independent fresh131 capture, token A/B/A and page P/Q/P,
resident shape reuse, non-live/live misses, continuation, complete logical
logits and next-decode logits/tokens, all-rank recurrent/conv hashes, owned
public-result deallocation, stable persistent/cache addresses, nonzero page4097
sentinels, repeated default-preserving rebuilds, and zero trace bytes at release
and teardown. `--require-trace-allocation-tracking` requires native tracking
including program-cache allocations; set the tracker environment before Python.
It stores incremental JSON plus complete host tensor artifacts for comparisons.
`--tensor-dir` is required so those bulky artifacts can remain outside source docs.

Device probe command for the supervising runtime (not yet run by this agent):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python models/autoports/ornith_ai_ornith_1_5_9b/tests/external_prefill_trace_device_probe.py --require-trace-allocation-tracking --tensor-dir /home/hous/dev/ornith-1.5-9b/state/optimized-vllm-h1-tensors --output models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/h1_external_reduced.json
```

Status: candidate and host contract verified; external replay device proof,
all32 evidence, sampling reconfiguration checks and serving A/B/burst gates
remain the supervising agent's work. No performance improvement is claimed.
The probe's default-preserving native rebuild intentionally exercises the full
cache snapshot and can expose a memory-fit limitation; shape misses themselves
are separately asserted to make no snapshot calls.

## Supervising hardware result: reduced external probe

The supervising lane ran the above probe in its serving runtime with
Watcher10, `TT_METAL_WATCHER_DISABLE_ETH=1`, and native trace-allocation
tracking including program-cache allocations. Read-only inspection of
`h1_external_reduced.json` and `.log` confirms **pass**, clean device closure,
and no fixture/source failure. See `.provenance.json` for exact argv,
environment and runtime source hashes.

All four sequential lanes (resident128/131, traced/eager) completed nine cases;
all18 candidate/eager comparisons passed exactly for complete logical logits,
next-decode logits/tokens, and all-rank recurrent/conv state. Each lane passed
two default-preserving native-pool snapshots, sentinel preservation, and zero
trace bytes at teardown. Both literal-first131 boundary comparisons also
passed. This verifies the reduced external-cache replay hypothesis with the
selected policy and exact4098-block pool. It does not establish all32 accuracy,
sampling-reconfiguration coverage or a serving performance improvement.
