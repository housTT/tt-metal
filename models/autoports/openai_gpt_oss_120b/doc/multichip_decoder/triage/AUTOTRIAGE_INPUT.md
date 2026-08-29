# Multichip decoder hang: AutoTriage input

## Reproducer

```bash
PYTHONPATH=$PWD/ttnn \
LD_LIBRARY_PATH=$PWD/build/lib \
GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
GPT_OSS_120B_MULTICHIP_ACCEPTANCE=1 \
GPT_OSS_120B_MULTICHIP_TRACE_REPEATS=2 \
timeout 1800 scripts/run_safe_pytest.sh \
  'models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py::test_real_weight_optimized_baseline_paged_cache_and_traced_decode[blackhole-1x4-sliding]' -sv
```

## Primary evidence

- Safe-runner LLM triage: `generated/tt-triage/triage.csv` (118 KiB,
  captured 2026-08-28 21:44 UTC).
- Failure site: `ttnn.synchronize_device(baseline_mesh)` immediately after
  `baseline.prefill_forward(...)` at
  `tests/test_multichip_decoder.py::test_real_weight_optimized_baseline_paged_cache_and_traced_decode`.
- The parent `(1,4)` ring initialized successfully. The TP=4 decoder and its
  full-context local KV caches constructed successfully. The single-chip
  `OptimizedDecoder` baseline was then constructed on a `(1,1)` submesh of the
  same parent and its prefill timed out.
- The single-chip optimized stage previously passed this exact baseline on a
  standalone `(1,1)` mesh. The new test is the first time it runs concurrently
  with a TP=4 module on a submesh carved from the same fabric-enabled parent.
- The safe runner reset all four devices after triage.

## Triage facts to explain

- `dump_running_operations.py` found dispatch blocked and multiple fabric ERISC
  router call stacks; device 3 became unreadable during triage.
- Fast dispatch on devices 0--2 waited in `cq_dispatch`; teardown later found
  dispatch cores `14-3,14-2` stuck across devices.
- The log immediately before the timeout showed compilation of the RM gather
  reader/writer used by the optimized full-local MoE baseline.
- No multichip forward had executed yet. Therefore this capture cannot prove a
  TP attention/expert collective bug; it can prove that the current concurrent
  baseline/submesh harness is unsafe or that construction disturbed the parent
  fabric before the baseline operation.

## Relevant paths

- `models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py`
- `models/autoports/openai_gpt_oss_120b/tt/multichip_decoder.py`
- `models/autoports/openai_gpt_oss_120b/tt/optimized_decoder.py`
- `models/autoports/openai_gpt_oss_120b/tt/fused_decoder.py`
- `models/demos/gpt_oss/tt/experts/`
- `models/demos/gpt_oss/tests/test_factory.py`
- `models/common/tests/conftest.py` (parent-mesh/submesh fabric policy)

## Requested diagnosis

Determine the exact most-supported stop-site/root-cause class and the smallest
safe test/implementation change. In particular, test the hypothesis that the
optimized baseline must run in a separate standalone `(1,1)` process and leave
a host baseline artifact, while the TP=2 target must be a submesh of an opened
four-device parent rather than a directly opened two-device subset.
