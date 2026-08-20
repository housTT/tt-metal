# Optimized multichip decoder work log

## Scope and starting point

- Model: `Qwen/Qwen3.6-27B`, dense 64-layer decoder (48 linear-attention,
  16 full-attention layers).
- Hardware: four Blackhole P300c devices, `1x4` ring, TP=4.
- Starting completed multichip stage: commit `79d14961c44`; prior implementation
  commit `65ac60e460f`.
- Scope remained the repo-local multichip decoder. No full-model, generation,
  vLLM, or release work was started.
- TT commands were serialized and launched from `/tmp` with
  `PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal`.

## Baseline and topology audit

The initial real-weight warmed baseline command was equivalent to:

```bash
cd /tmp
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_PERF_PHASE=both QWEN36_DECODE_REPLAYS=50 \
pytest -q -s /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py \
  -k multichip_real_weight_performance
```

It measured linear 6535.723/774.659 us and full 1778.039/553.373 us
(prefill/traced decode). The topology audit then examined packed and repeated
same-input projections, two material row-parallel collective boundaries,
standard all-reduce layout conversions, fractured residual/norm consumers,
fused matmul-CCL APIs, CCL placement/dtype, DRAM-sharded decode matmuls,
precision/fidelity, program geometry, and persistent resources. The resulting
audit and decisions are in `README.md`; numerical candidate rows are in
`candidates/index.csv`.

## Candidate sequence

1. Swept global and per-boundary CCL BFP8, ring links, projection BFP4/HiFi2,
   full-layer MLP per-role BFP4, MLP HiFi2, packed gate/up, and exact BFP4/LoFi
   block geometry. Correctness and performance logs are under `candidates/`.
2. Re-ran the residual-layout family with the fractured residual carried
   through residual add and distributed RMSNorm into the next projection.
   Fresh evidence is `candidates/residual_layout_final.log`; replicated won.
3. Used `$autofix` after the fused family crossed TTNN op/layout/runtime
   boundaries. Exact fused shapes were made numerical, logical padding was
   sliced, production weights were reshaped rank-4 without a copy, and L1 and
   DRAM persistent-output contracts were retried. The complete report and
   executable probes are under `autofix/fused_persistent/`.
4. Selected lazy mesh-scoped persistent async all-reduce. Exact collective
   tests measured 60.082 to 25.641 us on 8 cores and 60.422 to 23.943 us on 16
   cores. Same-mesh sequential layers retained identical buffers and passed.
5. Ran contemporaneous old-policy controls and three fresh final-default runs.
   After the final layer-specific CCL selection, authoritative no-override
   medians are linear 6680.525/718.857 us and full 2008.705/476.422 us. The
   files `autofix/fused_persistent/authoritative_default_run{1,2,3}.log` include
   exact command, environment, branch, links, payload, and pool provenance.
6. Stage review found the first BFP8 trials were non-persistent. `$autofix`
   added true BFP8 input/output persistent buffers and reran exact, attention,
   MLP, and global policies with printed command/runtime provenance. Exact PCC
   is 0.9999418. BF16 measured 721.502/476.934 us linear/full; attention BFP8
   722.918/479.149, MLP BFP8 718.964/479.494, and global BFP8
   719.703/480.809. All correctness gates pass. A second interleaved three-cycle
   family then isolated linear-layer policy: BF16 median 721.843 us, linear-MLP
   BFP8 719.125 us, and linear-global BFP8 719.971 us. MLP-only BFP8 beat BF16
   in all three matched cycles and is selected for linear layers; full-layer
   attention and MLP boundaries remain BF16.

Candidate commands used the same pytest performance node with the environment
overrides documented in the implementation (`QWEN36_MC_*`). Older candidate
logs preserve the pytest node and result but do not all echo their shell
environment; their candidate directory, source control, and adjacent
correctness/performance pair establish the policy. The material AutoFix reruns
preserve an explicit command or runtime-policy dump. The selected default
requires no override.

## Correctness and stress gates

The final broad acceptance command selected local tensor/state contracts,
non-aligned linear and full paths, paged trace replay, forced chunking, and
both fractured controls. The fallback and batch-32 nodes were selected in the
separate command recorded verbatim by their log:

```bash
cd /tmp
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
pytest -q -s /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py \
  -k 'source_runtime_fallback_audit or local_tensor_and_state_contracts or non_aligned or paged_decode_trace or forced_chunk or batch_32 or fractured'
```

The final exact mixed-default run is `final/acceptance_mixed_default.log`:
8 passed, 4 intentional opt-in topology/performance tests skipped, 14
deselected. The separately selected source fallback and batch-32 tests are in
`final/fallback_batch32_mixed_default.log`: 3 passed, 23 deselected. The shared
pool opt-in test was run in the AutoFix sequence with
`QWEN36_RUN_SHARED_CCL_POOL=1`; it passed sequential linear/full layers,
stable pool identity, exactly three slots, and trace repeat PCC 1.0.

Native context commands exercised full prefill at 262144, full traced decode at
position 262143, and linear 262144 prefill plus final traced decode. Evidence:
`final/context/full_native.log`, `full_native_decode.log`, and
`linear_native_mixed_default.log`. The final linear command ran from a scrubbed
environment with only `QWEN36_CCL_PROVENANCE=1`, passed in 692.46 seconds, and
printed BF16 attention/BFP8 linear-MLP payloads, explicit persistent async CCL,
one prefill link, two decode links, and the exact final pool allocation. The
older `linear_native.log` predates mixed-policy selection. Batch 32 passes for
both meaningful layer kinds.
After the final watcher and again after the refreshed native-context run,
`final/postflight_tt_smi.log` and `final/postflight_after_native_tt_smi.log`
record all four P300c devices visible and reset-capable; no reset was needed.

## Profiler and watcher

Profiler capture used one real layer of each kind, separately signposted
prefill and a single warmed trace replay. From the repository root, the exact
successful decode-report commands were:

```bash
PANDAS_FUTURE_INFER_STRING=0 tt-perf-report \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/linear/ops.csv \
  --start-signpost LINEAR_ATTENTION_DECODE_TRACE_START \
  --end-signpost LINEAR_ATTENTION_DECODE_TRACE_END \
  --csv models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/linear/decode_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/linear/decode_summary

PANDAS_FUTURE_INFER_STRING=0 tt-perf-report \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/full/ops.csv \
  --start-signpost FULL_ATTENTION_DECODE_TRACE_START \
  --end-signpost FULL_ATTENTION_DECODE_TRACE_END \
  --csv models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/full/decode_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/profiler/full/decode_summary
```

The human-readable stdout is retained as `decode_report.txt`; command status
and generated-file messages are in each `decode_report_generation.log`.
Advice-enabled tables, CSVs, summaries, images, capture logs, and raw
provenance are retained under `profiler/{linear,full}`.
The reports profile the selected math/layout/trace/persistent-async topology,
but their linear MLP collective row predates the final BFP8 selection and is
BF16. Exact final mixed-policy runtime/provenance comes from the authoritative
default and variance-cycle logs. All applicable report advice was tried or tied
to measured inherited evidence as recorded in `README.md`.

Watcher was a separate run. The original level-1 evidence is retained as
provenance; closure reran the same risk-focused selection at the required level
10:

```bash
cd /tmp
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_DUMP_ALL=1 \
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
pytest -q -s /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_multichip_decoder.py \
  -k 'linear_non_aligned_prefill_decode or full_attention_real_weight_paged_decode_trace'
```

Result: 2 passed, 24 deselected in 27.71 s. The authoritative final mixed-policy
files are `final/watcher/pytest_level10_mixed_default.log` and
`watcher_level10_mixed_default.log`; the latter has
1092 lines, no error/assert/hang/stuck/mismatch signature, and clean detach of
devices 0--3. Full trace PCC is 0.9954127239 with repeat PCC 1.0.

## Fused-op hang classification and recovery

An adapted `all_gather_matmul_async` DRAM-output retry stopped making host
progress after mesh construction. `$autotriage` was used. Inspector data was
not available and the process exited before the focused attachment, so no
device-kernel deadlock is claimed. `timeout 60 tt-smi -ls --local` subsequently
listed all four devices and a minimal 1x4 mesh open/close passed. No reset or
reboot was required. Commands and raw evidence are in
`triage/fused_agmm_hang/`; the rejected candidate source is preserved in
`autofix/fused_persistent/fused_candidate_source_snapshot.patch` and is absent
from the default runtime.

## Final checks and commits

- `python -m json.tool doc/context_contract.json`: passed.
- `python -m py_compile tt/multichip_decoder.py tests/test_multichip_decoder.py`:
  passed.
- `git diff --check`: passed.
- Independent `$stage-review` pass 1: `more-work-needed`; required coherent
  persistent-BFP8 CCL evidence, watcher level 10, same-run profiler accounting,
  corrected batch-32 ranges, and cumulative precision-locked geometry evidence.
  All findings were remediated.
- Independent `$stage-review` pass 2: `more-work-needed`; required an actually
  tested layer-specific CCL policy, explicit command/runtime provenance, and
  corrected profiler signpost commands. All findings were remediated with the
  interleaved variance family, exact provenance logging, final mixed-default
  reruns, and corrected report commands. All findings were remediated.
- Independent `$stage-review` pass 3: `more-work-needed`; found that retained
  native linear-context evidence predated the final mixed CCL policy. The exact
  final-default 262144-token prefill plus position-262143 trace was rerun with
  command/runtime provenance and passed.
- Independent `$stage-review` pass 4: `clean-pass`, with no required work. The
  fresh read-only reviewer was `/root/optimized_mc_context_rereview`; it checked
  the remediation log, final source/tests, context contract, postflight, and
  retained artifact inventory directly.
- Local stage commit: pending.
- Nothing is pushed.
