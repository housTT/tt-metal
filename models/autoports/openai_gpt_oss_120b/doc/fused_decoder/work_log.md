# GPT-OSS 120B fused decoder work log

Status: implementation and all requested hardware gates are complete. The
final fresh independent stage review returned `clean-pass`; the local stage
commit is pending.

## Scope and target

- Model: `openai/gpt-oss-120b`, pinned revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`.
- Stage source: `tt/fused_decoder.py` only.
- Stage tests: `test_fused_decoder.py`, packed/indexed A/B, direct prefill A/B,
  direct FullLocal real-weight A/B, and FullLocal reducer A/B.
- Stage docs/evidence: `doc/fused_decoder/` only.
- Hardware: four local p300c Blackhole boards, firmware 19.13.1, treated as
  P150-class. Runtime mesh remains `1x1`; P150x2/P150x4 host support means one
  selected device in this stage. No optimized-decoder, multichip-decoder,
  full-model, or vLLM work was started.
- Functional parent commits: `acfbdee5` (implementation) and `51905378`
  (stage ledger).

The canonical hardware environment used by the final gates was:

```bash
env \
  -u TT_METAL_SIMULATOR \
  -u TT_METAL_SIMULATOR_HOME \
  -u TT_METAL_SLOW_DISPATCH_MODE \
  -u TT_METAL_DISABLE_SFPLOADMACRO \
  -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_WATCHER \
  -u TT_METAL_WATCHER_DUMP_ALL \
  TT_VISIBLE_DEVICES=0,1,2,3 \
  TT_METAL_HOME=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_CACHE=/home/ttuser/dev/gpt-oss-20b/tt-metal/.tt_metal_cache \
  TTNN_CONFIG_OVERRIDES='{"cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache","model_cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache/models","tmp_dir":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_tmp"}' \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn \
  LD_LIBRARY_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  COMMAND
```

Every hardware command was wrapped in:

```bash
flock -x /tmp/gpt_oss_120b_fused_hw.lock -c 'EXPANDED_COMMAND'
```

The retained run logs begin with shell tracing or print every selector, input,
path assertion, and checkpoint, so they are the exact expanded transcripts.

## Baseline and implementation sequence

The accepted functional-stage Tracy baseline at sequence 128 was:

| Layer kind | Prefill device sum | Traced decode device sum |
| --- | ---: | ---: |
| sliding | 137.401 ms | 2.060 ms |
| full | 137.418 ms | 2.047 ms |

Implementation proceeded as follows:

1. Cloned the functional public boundary, paged KV-cache contract, shared
   attention, layer norms, residual flow, and arbitrary logical-length padding.
2. Replaced prefill's per-expert primitive graph with
   `unified_routed_expert_moe` and exact OAI-SwiGLU/bias semantics.
3. Replaced invalid `1x1` fabric dispatch/combine with device-local masked
   bincount, tile-aligned offsets, sort, gather/scatter, and inverse mapping.
4. Added private 64-row routing padding and 4096-token internal chunking while
   returning the exact logical length.
5. Packed decode gate/up weights and biases, then replaced dense projections
   with compact top-4 indexed sparse gate/up and down matmuls.
6. Investigated public FullLocal `moe_compute`; corrected an expert-major score
   oracle bug, integrated the public fused score reducer, tested memory/output
   variants, and exhaustively swept real layers.
7. Replaced device BF4 typecast with public host quantization after it proved
   materially more accurate. Applied only the real-layer batch-1/batch-2
   intersection and independent-seed-qualified marginal set.
8. Added exact checkpoint, layer, ring-8, and configured-max-batch <=2 gates.
   All other inputs keep the best correct indexed graph, preserving batch 32.
9. Fixed tensor ownership at the attention/MLP and split-user boundaries,
   retained alias-safe delayed cleanup where immediate view deallocation
   changed output, and serialized setup packing/quantization to bound peak.
10. Corrected the profiler constructor drain to use chunk-unique cache prefixes
    and forwarded dimensions; a host test proves all 128 expert inputs remain
    distinct and ordered.

## Candidate ledger

| Candidate | Correctness | Performance / topology | Decision |
| --- | --- | --- | --- |
| Dedicated unified prefill MoE | functional-to-fused PCC 0.999770207; Torch PCC 0.985837; exact trace | 205.288 to 32.829 ms wall; 205.203 to 32.623 ms Tracy | Applied. |
| Local regroup | Exact slot/order equivalence | Avoids invalid remote fabric waits on `1x1` | Applied. |
| Packed gate/up | PCC 1.0 to prior decode | 1.755/46 ops to 1.537/42 ops | Applied in indexed. |
| Indexed packed gate/up + down | PCC 1.0 to packed; 0.999708177 to Torch; exact 40-replay stress | 1.531 to 0.993 ms | Applied fallback. |
| FullLocal with device BF4 typecast | Initial corrected candidates failed the 0.995 bar on many layers | Kernel itself was fast | Rejected on precision. |
| FullLocal with public host BF4 quantization | 25 real layers pass batch 1 and 2; marginal set second-seed checked | 500-replay whole decode 46.68-47.46% below indexed | Applied behind gates. |
| Fused FullLocal score reducer | PCC >0.999996 to primitive transpose/mul/sum | Faster at batch 1/2/32; removes primitive chain | Applied. |
| FullLocal L1 output | Bitwise equal to DRAM | Two-layer A/B 1.443122 vs 1.440433 ms DRAM | Rejected end to end. |
| Zero/dummy reducer indices | Bitwise equal | TILE-to-one-core row-major fold violates tile shard shape | Rejected; no op removed. |
| FullLocal batch 3-32 | Sampled direct kernel often accurate | Whole batch 16/32 nondeterministic; batch 9 blocked in shared attention | Rejected; cap 2, indexed preserves 32. |
| Generalized MoE gate | Runtime math viable | Trace output-buffer write unsupported | Rejected. |
| DeepSeek fabric dispatch/combine | Source requires neighbor | `1x1` remote handshake timed out | Rejected. |
| Unified expert kernel for decode | Correct path | 36.565 ms plus maps | Rejected on latency. |
| `moe_gpt` | Missing GPT-OSS expert biases | N/A | Rejected on semantics. |
| `post_combine_reduce` / ordinary fast reduce for indexed | PCC 0.999948 / 0.99999765 | 5.51% / 1.00% slower | Rejected. |
| Shared attention bias folds | Correct cases profiled | Outside authorized file; decode o-proj also slower | Not integrated. |

The full pattern mapping is in `graph_fusion_assessment.md`. Failed experiments
are retained rather than overwritten in
`candidates/full_local_moe_compute/` and the other candidate directories.

## FullLocal qualification

The direct assessment command was repeated over all layer indices at logical
batch 1 and 2:

```bash
GPT_OSS_120B_FULL_LOCAL_REAL_AB=1 \
GPT_OSS_120B_FULL_LOCAL_WEIGHT_QUANTIZER=host_quant \
GPT_OSS_120B_FULL_LOCAL_ASSERT_ALLOWLIST_PCC=1 \
GPT_OSS_120B_FULL_LOCAL_REAL_TOKENS=BATCH \
GPT_OSS_120B_FULL_LOCAL_REAL_REPEATS=2 \
pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder_full_local_real_ab.py::test_full_local_moe_compute_real_weight_assessment \
  -k 'LAYER_SELECTION'
```

The first seed produced 27/36 passing layers at batch 1 and 26/36 at batch 2.
The intersection was 26. Independent seed 820511 was then run on marginal
layers 21, 26, 28, and 33; layer 28 fell to 0.994515765 and was removed. Final
set: `{0-13,21,23-27,29,31,33-35}` (25 layers). Lowest first-seed accepted PCC
was 0.995360570; all traces were deterministic. Exact values and selected path
are in `host_quant_layer_qualification.csv`; source logs are the five
`host_quant_{prior_allowlist,rejected_layers,all_layers,marginal_second_seed}*.log`
files.

## Final correctness, capacity, stress, and watcher

Final exact-revision whole-layer command:

```bash
pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_paged_prefill_and_traced_decode
```

Result: layer-0 sliding prefill/decode PCC 0.978123157/0.990064440;
layer-1 full 0.990889623/0.955780929. Both constructed FullLocal, captured and
replayed a complete trace, and passed determinism.

Batch and boundary commands:

```bash
pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_batch_two_full_local_paged_prefill_and_traced_decode

GPT_OSS_120B_RUN_BATCH32=1 pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_batch_32_indexed_fallback_paged_prefill_and_traced_decode_capacity

pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_prefill_tile_page_and_window_boundaries
```

Results: real batch-2 FullLocal PCC 0.992492223/0.992720098; exact-revision
batch-32 selected indexed and passed 131072-context-per-user capacity; logical
lengths 1, 31/32/33, 63/64/65, and 127/128/129 passed both layer kinds. Earlier
unchanged prefill evidence covers 4095/4096/4097 and 131071/131072. The 500-
replay performance A/B below is also the final repeated-run stress.

Watcher command added only these watcher variables to the canonical environment:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_APPEND=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_paged_prefill_and_traced_decode
```

Result: 2/2 passed; qualified scan found no watcher, kernel, NoC, circular-
buffer, semaphore, or trace-allocation error. Profiler was explicitly unset and
was never combined with watcher.

## AutoFix: layer-5 indexed fallback equivalence

The fresh review found that the retained
`indexed_layer5_functional_equivalence.log.gz` still described a pre-final-
source failure: functional-to-HF prefill/decode PCC 0.987181853/0.917455178,
indexed-to-HF 0.351548308/0.371811276, and indexed-to-functional prefill PCC
0.355932135. AutoFix treated the artifact as a hypothesis, not as a current
runtime result.

1. Reproduction on final `fused_decoder.py` SHA
   `66876eaed3856fda62d05e6425477e88ad8d3c18328dd854f185a41675d2c675`
   passed: indexed-to-functional prefill/decode PCC
   0.998950699/1.0, with bitwise-equal decode and replay.
2. The program-cache-alias hypothesis predicted that removing the already-
   present clear between the independently authored paths would restore the
   failure. A controlled local removal followed by the same command also
   passed at 0.998950699/1.0, refuting that hypothesis. The clear remains as
   defensive A/B isolation.
3. The failed log was timestamped 07:11; the final implementation was modified
   at 08:32 and has the SHA above. It was stale, not evidence against the final
   runtime. The test now labels functional and indexed HF measurements
   separately, while retaining the real unqualified-revision selector and
   concrete indexed-weight construction assertions.

The final repeat gate, in the canonical hardware environment and serialized
by the stage hardware lock, was:

```bash
GPT_OSS_120B_FULL_LOCAL_FALLBACK_EQUIVALENCE=1 \
pytest -q -s --count=3 \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder_full_local_real_ab.py::test_layer5_indexed_fallback_preserves_functional_real_outputs
```

All 3/3 runs produced identical indexed-to-functional PCC
0.998950699/1.0 and bitwise-equal traced decode/replay. The log attests source
SHA above and test SHA
`8bbf908cf0d96ee4344b6946cc48627778ce8b2eee6a195c28b7549e6249af33`.
Evidence:

- final repeat:
  `candidates/full_local_moe_compute/indexed_layer5_functional_equivalence.log.gz`,
  SHA-256 `dabf23f03ca60cff2b3235061f76283c6b3873d6a00a7be7f82694d53c96f393`;
- original failure, preserved:
  `candidates/full_local_moe_compute/indexed_layer5_functional_equivalence_before_final_source.log.gz`,
  SHA-256 `838caa784970ec3c0274c0fbd87d921034997c8a9d40881e487ca7fed9b86c52`;
- no-clear control:
  `candidates/full_local_moe_compute/indexed_layer5_no_program_cache_isolation_control.log.gz`,
  SHA-256 `aa4c791670e6e426688a0e439de2db30c469927eb9665b4a190e2a31592f3d1e`.

No production runtime code changed, so a new watcher run was not warranted;
the existing final-source 2/2 watcher-clean evidence remains applicable.

## Final whole-decoder latency A/B

```bash
GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF=1 \
GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF_REPEATS=500 \
pytest -sv \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_whole_traced_decode_path_performance
```

The harness supplies the exact checkpoint revision for both paths and forces
only the indexed selector in the baseline cases. Construction and output-layout
assertions prove no fallback substitution.

| Layer | FullLocal wall | Indexed wall | Change | Replays |
| --- | ---: | ---: | ---: | ---: |
| sliding 0 | 0.719116 ms | 1.368594 ms | 47.46% lower | 500 each |
| full 1 | 0.728915 ms | 1.367084 ms | 46.68% lower | 500 each |

All four cases were bitwise deterministic.

## Final Tracy and `tt-perf-report`

Each of the four exact node IDs was captured separately with watcher unset:

```bash
TT_METAL_PROFILER_DIR=doc/fused_decoder/tracy/final_full_local/KIND/PATH/generated/profiler \
GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF=1 \
GPT_OSS_120B_FULL_LOCAL_WHOLE_PERF_REPEATS=1 \
timeout 900 ./python_env/bin/python -m tracy -r -p -v \
  --dump-device-data-mid-run --op-support-count 5000 -m pytest -q \
  'models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_real_weight_whole_traced_decode_path_performance[NODE_ID]' \
  -s
```

Node IDs were `blackhole-1x1-{full_local,indexed}-{sliding,full}`. Every raw
CSV had exactly one start/end signpost pair for prefill and decode, no dropped-
marker warning, and a path/checkpoint assertion in its run log.

Each raw report was losslessly compressed and filtered with:

```bash
/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --csv prefill_perf_report.csv --summary-file prefill_perf_report_stacked \
  --no-advice --no-color

/home/ttuser/dev/ornith/ornith-pyenv/bin/tt-perf-report ops_perf_results.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --active-experts 4 --csv decode_perf_report.csv \
  --summary-file decode_perf_report_stacked --no-advice --no-color
```

Text tables used the same signposts plus `--no-summary`. Final totals:

| Kind/path | Prefill device / ops / host | Decode device / ops / host |
| --- | ---: | ---: |
| sliding FullLocal | 36.220 ms / 66 / 0 | 0.678 ms / 34 / 0 |
| sliding indexed | 36.396 ms / 66 / 0 | 1.312 ms / 67 / 0 |
| full FullLocal | 35.558 ms / 66 / 0 | 0.689 ms / 34 / 0 |
| full indexed | 35.596 ms / 66 / 0 | 1.320 ms / 67 / 0 |

Compared with functional, final prefill is 73.64-74.12% lower and FullLocal
decode is 66.35-67.10% lower. Compared with the best correct indexed candidate,
FullLocal decode is 47.80-48.32% lower. `comparison.csv` and `validation.txt`
record exact path/layout/host-op conclusions.

## Device recovery record

All correctness, watcher, and profiler captures completed and closed devices.
A post-profile mesh smoke accidentally inherited `TT_METAL_SLOW_DISPATCH_MODE`
and left device 0 firmware/remote Ethernet state unhealthy. The failure did not
affect any capture because every profiler command explicitly unset slow
dispatch. Following `tt-device-usage`:

1. verified no live pytest/Tracy owner;
2. enumerated all four boards and ran a bounded reset;
3. preserved the failed first corrected smoke, which showed an inactive remote
   Ethernet heartbeat;
4. captured focused `tt-triage` Ethernet and ARC evidence (16/16 links up,
   heartbeat true, retrain 0; ARC ~9.998 heartbeats/s);
5. ran a second bounded reset, enumerated all four boards, and ran a fully
   pinned mesh open/close that printed `MESH_SMOKE_OK` and closed cleanly.

Artifacts are in `tracy/final_full_local/recovery/`. No reset was hidden, and
no profiler/watcher evidence was collected during recovery.

## Host checks and limitations

The final source/tests are Python-only, so AGENTS.md does not require the CI
container build. Checks run:

```bash
python -m py_compile \
  models/autoports/openai_gpt_oss_120b/tt/fused_decoder.py \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder*.py

pytest -q \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_profile_constructor_drains_use_128_distinct_expert_cache_inputs \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_fused_implementation_has_no_functional_runtime_fallback \
  models/autoports/openai_gpt_oss_120b/tests/test_fused_decoder.py::test_full_local_activation_requires_exact_revision_layer_and_ring

pre-commit run --files STAGE_SOURCE_TEST_DOC_FILES
git diff --check
```

Limitations:

- FullLocal BF4 is checkpoint/layer sensitive; qualification is intentionally
  explicit. Indexed remains correct and faster than functional elsewhere.
- FullLocal is capped at configured maximum batch 2. This does not reduce the
  public batch-32/context contract because larger configurations use indexed.
- Ring-7 P150 variants use indexed until independently calibrated.
- The `1x1` stage does not use all devices on P150x2/P150x4; that is the next
  multichip stage.
- On-device evidence is specific to P150-class Blackhole/f19.13.1; no claim is
  made for another architecture.

## Review and commit ledger

- Initial stage review: `more-work-needed`; findings were real-checkpoint
  FullLocal evidence, trace A/B overlap, durable direct prefill evidence, and
  missing opt-ins in the documented indexed stress command.
- AutoFix: all four findings fixed. The trace candidates are now captured and
  released serially; direct prefill A/B and tt-perf evidence are durable; exact
  opt-ins are logged; FullLocal was fully investigated and became a larger,
  faster production optimization.
- Second fresh stage review: `more-work-needed`; its sole finding was the stale
  layer-5 indexed-fallback failure artifact.
- Layer-5 AutoFix: final-source reproduction and three-repeat gate passed at
  0.998950699/1.0 to functional with bitwise decode; the original failure and
  refuted no-program-cache-clear control are both preserved.
- Final fresh stage rereview after AutoFix: `clean-pass`.
- Stage implementation commit: pending; never push.
- Commit-ledger documentation commit: pending; never push.
