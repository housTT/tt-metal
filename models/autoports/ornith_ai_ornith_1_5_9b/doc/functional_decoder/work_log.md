# Functional decoder work log

Target: ornith-ai/Ornith-1.5-9B, revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
Branch: hous/ornith-1.5-9b. Initial checkout: `565b0aedf3` (clean).
Stage status: complete; independent review clean-pass. Local checkpoint recorded below.

## Initial inspection — 2026-09-04

Read the local AGENTS contract and functional-decoder, tt-device-usage, autofix,
stage-review skills. Inspected pinned HF config through AutoConfig and installed
Transformers 5.12.1 Qwen3_5 decoder, attention, DeltaNet, norm, MLP and rotary source.
Read the user-requested 35B reference decoder, RoPE, configuration, HF harness and
functional tests from the pinned reference checkout. Adapted the shared mechanisms;
replaced MoE with dense SwiGLU, hidden width 4096, intermediate width 12288, KV heads 4.
No earlier model metrics are used as evidence.

`timeout 60 tt-smi -ls --local`: four Blackhole P300c chips visible (`logs/device_list.log`).
`timeout 90 python /home/hous/dev/ornith-1.5-9b/bin/mesh-smoke.py --chips 1`:
passed (`logs/mesh_smoke.log`). All device workloads serialized in this runner.

HF layer-only loading reads only layer 0/3 entries from local safetensors shards;
full model instantiation is unnecessary. Strict reference load passed for both kinds;
`logs/weight_load.log` and `weight_stats_layer0.json`, `weight_stats_layer3.json` record keys,
shapes, dtypes and moments. `hf_config.json` is the exact pinned config for offline CI.

Environment repair: importing torch._dynamo without a cache override fails because
container uid 1002 has no passwd entry (`getpwuid(): uid not found: 1002`), masked by
a subsequent duplicate precompile registration on Transformers import. Setting
`TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache` resolves the
initialization and real HF reference loading. No package or implementation change.

All test commands use the active source-built `python_env`, `OMP_NUM_THREADS=8`,
and the persistent TORCHINDUCTOR_CACHE_DIR above. Explicit real-weight runs set
`ORNITH_WEIGHTS=real`; normal CI defaults to deterministic synthetic weights from recorded stats.
HF's missing FLA/causal-conv warning selects its CPU oracle; it does not describe
TTNN runtime fallback. Device topology's unknown motherboard warning uses PCI bus IDs;
the single-chip open/close check passed.

## Initial verification and new gates

`ORNITH_WEIGHTS=real pytest .../tests/test_functional_decoder.py -k real_weights_pcc -x -v -s`
passed 2 tests (47.69s including cold JIT); `logs/real_weights_pcc.log`:
layer0 prefill/decode PCC .999710/.999630, layer3 .999401/.999528.
`... -m 'not long' -k 'not perf and not real_weights' -x -v -s` passed all
52 selected tests in 158.77s (`logs/functional_matrix.log`). This initial matrix
includes batch32, ragged positions, shuffled pages, boundary prompts, determinism,
traced replay and runtime guard controls. It predates the continuation extension
and input-scale calibration; final evidence must rerun the changed paths.

Added arbitrary-offset prefill continuation, changed-page-table trace replay,
trace repeated-state determinism, and optional YaRN using HF setup-time RoPE
initialization. `logs/trace_continuation.log` passed the changed-table trace,
linear trace determinism and million-position YaRN table checks, then failed
linear continuation at split63 because the persistent position ramp was freed.
AutoFix independent diagnosis requested; this is required repair, not a waiver.

`embedding_stats.json`: actual checkpoint embedding rows 1000:2024 std
.014285416342318058. Replaced inherited .5 input scale with the measured value.
`logs/calibrated_real_weights.log` passes both real-weight kinds at that scale:
layer0 .999366/.999433, layer3 .999459/.998906 (prefill/decode).

`logs/native_decode_oracle.log`: real layer3 weights plus exact-shape synthetic
BF16 history with shuffled pages, traced decode at262143 fails PCC .94535284.
This run used the inherited .5 input scale. Requested independent AutoFix
diagnosis and focused controls; no context reduction is authorized by this failure.

## AutoFix ownership repair and native capacity

`AUTODEBUG.md` independently identified the full-extent ramp slice alias. AutoFix
experimental agent proved it with allocation/address controls before editing;
`AUTOFIX_ramp.md`, `logs/ramp_alias_probe.log`. Kept the minimal `_slice_owned`
conditional-deallocation fix. `logs/ramp_continuation_fix.log`: all4 real-weight
linear split63/65/128/129 continuations pass (.999328 minimum PCC).
`logs/full_continuation.log` and its `.provenance.json`: all4 full-attention
continuations pass after arbitrary-offset support (9.98s).

`logs/native_prefill.log`: all4 native-capacity tests passed in178.14s using
calibrated real-weight inputs: lengths262143/262144 for both kinds; traced decode
at262143 matched eager PCC1.0. This is capacity/trace evidence, not full-length
HF prefill PCC. The independent exact-cache HF native-decode anomaly remains
under AutoFix (`AUTODEBUG_long_decode.md`) until its verified repair is recorded.


## Final functional checks and resolved investigations

All source-changing repairs stay under this autoport. `tools/record_run.py` stores
exact argv, environment, before-run source SHA256, timestamps, exit status and log
SHA256 beside each final run. Earlier exploratory logs are labeled above and are
not substituted for final evidence.

- `logs/final_functional.log` and `.provenance.json`: all 76 non-long tests passed
  in 87.70 s, with 9 long tests deselected. Real-weight PCC: layer 0 prefill/decode
  .999366/.999433; layer 3 .999459/.998906. B=1/4/32 traced, changed-input decode
  passes; three identical-state eager and trace runs are bit-identical.
- `logs/final_long.log` and `.provenance.json`: all 9 long tests passed in 404.77 s.
  Both kinds complete 262143/262144-token decoder prefill. Decode after actual
  prefill to position 262143 has eager/replay PCC 1.0. Two chunk sizes (2048/1024)
  give tail PCC 1.000000. HF comparisons at 8001 tokens are .999393/.999319 for
  linear and .999530/.999734 for full attention. Native exact-cache full-attention
  HF/traced decode PCC is .99965012 at position 262143.
- `logs/context_gate.log`: strict context-contract gate passes with advertised
  and supported context 262144 and no reduction. No full-HF native prefill claim.

AutoFix native-decode investigation is complete in `AUTOFIX_long_decode.md` and
its listed artifacts. The historical .94535 result did not reproduce under the
unchanged SDPA path. Twelve scale/policy diagnostic controls and seven strict
reruns pass, including the exact historical .5 activation formula in normal,
watcher and cold-JIT processes. All six original-scale strict reruns give
.9995616861; cold JIT has 0/237 cache hits. Cache write/readback and FP32 same-cache
attention controls pass, and eager/replay are bit-identical. No attention/numerical
change was kept. The original failure is preserved, with unknown historical cause;
this is controlled current-state evidence, not a claimed bug fix.

A final source audit found an independent continuation padding error: aligning
only to the 64-token page boundary could make a final SDPA chunk's physical
128-token padding exceed an exactly allocated cache. The geometry is in
`continuation_capacity_geometry.json`. Added `test_unaligned_continuation_to_capacity`
with real full-layer weights, context allocation 4096, split 63. The unchanged
source fails with `Ends65 must be <= shape64` in
`logs/continuation_capacity_before.log`. Aligning the device-decode prefix to 128
fixes this without changing fresh-prefill/decode. `logs/continuation_capacity_after.log`
and provenance: 11 tests pass in 18.29 s, including all continuation splits for
both kinds, the capacity regression (HF PCC .99951140), and both runtime-guard
cases. The guard now covers unaligned continuation and forbids
`copy_host_to_device_tensor` as well as the original Torch and transfer APIs.

## Profiling collection

The first Tracy command used a whitespace-containing pytest `-k` expression.
Tracy reconstructs its child command without preserving that argument, producing
`file or directory not found: and`; no test ran. The preserved
`logs/profile_linear_attention_prefill.log` records exit 4. Using an explicit
pytest test node and a single layer-kind selector repairs only the invocation.
Final captures use `--check-exit-code` so a failed child cannot pass as profiling.
All four captures run serially with watcher disabled. Each test warms before its
signposts and checks the measured output against HF after the measured window.


Final prefill captures (`logs/profile_v2_<kind>_prefill.log`) have complete marker
coverage and measured-output HF PCC .99948959 (linear), .99949539 (full).
The initial 32-replay decode captures explicitly warn `Profiler DRAM buffers were
full, markers were dropped`; they are invalid for timing and retained under
`tracy/<kind>/rejected_v2/`, with `performance_v2_rejected.json` labeled rejected.
The original report command also lacked `--tracing-mode`, required because capture
timestamps must not reorder replay rows. Corrected that flag and added optional
`ORNITH_PERF_DECODE_ITERS` (normal default stays 32). Four measured replays after
four warm replays fit the profiler buffers without changing decoder computation.
`logs/profile_v3_<kind>_decode.log`: both pass with no dropped markers. Coverage
checks prove sessions 5/6/7/8 each contain all 78 linear or 69 full ops. Filtered
report times equal raw kernel durations after ns-to-us conversion.

Final kernel sums: linear prefill 40.075804 ms, traced decode 1.60966675 ms/replay;
full prefill 35.104062 ms, traced decode 1.42231950 ms/replay. Decode measured-output
HF PCC .99980134/.99902570. These are functional baselines, not optimization claims.
`performance.json` and four human-readable tables/CSVs carry the final measurements.
The optional GUI copy warning concerns Tracy's global path when `-o` is used;
actual per-run raw captures and ops CSVs exist, and coverage checks pass. Pandas'
mixed-type-column warning is a parser warning; all measured rows have device times.
Task-owned GUI PIDs 91457 and 126289 were stopped after their captures.

Exact profiling invocation (run serially for each kind):

```bash
python -m tracy -r -p -v --check-exit-code --web-app-port 18940 \
  -o models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/tracy/linear_attention/raw/prefill \
  -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py::test_perf_prefill \
  -k linear_attention -v -s
ORNITH_PERF_DECODE_ITERS=4 python -m tracy -r -p -v --check-exit-code --web-app-port 18940 \
  -o models/autoports/ornith_ai_ornith_1_5_9b/doc/functional_decoder/tracy/linear_attention/raw/decode_v3 \
  -m pytest models/autoports/ornith_ai_ornith_1_5_9b/tests/test_functional_decoder.py::test_perf_decode_traced \
  -k linear_attention -v -s
python models/autoports/ornith_ai_ornith_1_5_9b/tools/render_perf.py linear_attention prefill
python models/autoports/ornith_ai_ornith_1_5_9b/tools/render_perf.py linear_attention decode --capture decode_v3 --iterations 4
python models/autoports/ornith_ai_ornith_1_5_9b/tools/summarize_perf.py
```

Repeat the first four commands replacing `linear_attention` with `full_attention`;
run the summary after both kinds. `record_run.py` wrappers and all actual argv/env
are recorded in matching log provenance JSONs.

## Final watcher and runtime audit

`logs/watcher_final.log` and provenance: 23 tests pass in 169.82 s under
TT_METAL_WATCHER=10 and a separate TT_METAL_LOGS_PATH, without profiling.
Selection is `real_weights_pcc or traced_decode_pcc or no_host_fallback or
continuation or native_context_decode_oracle or trace_repeated_identical_state or
trace_changed_page_table`. Both layer kinds and traced batches 1/4/32 pass;
native HF/traced .99965012 and capacity continuation .99951140 repeat successfully.
`watcher_audit.json` records clean generated logs and their SHA256. The watcher file
is rewritten on device reopen and therefore retains the last fixture; full suite
stdout preserves each preceding check and result. Expected host-fallback guard
exceptions are test positive controls, not runtime fallbacks or watcher errors.
The only test-source change after this run makes profiler iteration count selectable;
none of this watcher's selected tests use that setting.

Exact stdout is archived as `.log.gz`, with originals ignored and left locally.
The README gives lossless hydration instructions. Ops CSV archives and watcher
archives likewise preserve raw bytes. Human tables remain plain text for review.
No C++/CMake files changed, so no build is required; Python checks and actual device
runs validate this stage, including successful device-kernel JIT compilation.


## Host checks before independent review

`python .agents/scripts/check_context_contract.py --model-dir models/autoports/ornith_ai_ornith_1_5_9b --hf-model /home/hous/dev/ornith-1.5-9b/upstream --stage functional-decoder --require-contract --strict-caps`
passes again (`logs/context_gate_final.log` and provenance). `pre-commit run`
initially formats import spacing in diagnostic/report tools and strips trailing
spaces from rendered text tables; no decoder/test semantics change.
The rerun passes every applicable hook (`logs/precommit_final_retry.log`).
`evidence_integrity.json` records verified log hashes and lossless archive parity.
Fresh xhigh stage-review inspected the live implementation, original stage
contract, skills, raw evidence and final docs. Its final verdict is clean-pass
after the evidence corrections recorded below.


Independent review identified a packaging gap: global repository ignore rules had
excluded the locally present filtered performance CSVs. Explicitly staged all four
final reports and the two rejected-v2 reports. README links and performance.json
hashes now resolve in the staged checkout, and the byte hashes still match.
`logs/precommit_csv_packaging.log`: all applicable pre-commit hooks pass again.

Review also removed an unsupported allocation-counter wording from the README.
The traced tests establish stable buffers, changed-input HF parity and deterministic
replay; they do not measure a runtime allocation counter. No new test is claimed.

The report CLI emits CRLF CSVs. Evidence-local `tracy/.gitattributes` preserves
those exact bytes (`-text whitespace=cr-at-eol`) without changing report hashes.
`git diff --cached --check` passes; its empty-success log is archived.


## Independent review and checkpoint

`STAGE_REVIEW.md`: fresh xhigh independent reviewer, **clean-pass**, no required
work after CSV packaging and allocation-count prose corrections. The reviewer
re-derived source hashes, checkpoint/config parity, run results, measured replay
coverage and timings, watcher hashes, and original-scale anomaly controls.
No additional device rerun was requested. Native-length full HF prefill and
million-token full-layer validation remain explicitly unclaimed; native 262144
capability is tested without reduction. No later pipeline stage was started.

Only stage-owned files in `models/autoports/ornith_ai_ornith_1_5_9b` are checkpointed
in tt-metal on branch `hous/ornith-1.5-9b`; no other repo was changed. Local commits
only; never pushed. The following metadata commit records the stage checkpoint.

`pre-commit run` also passes after the final review report and completion metadata
are staged (`logs/precommit_reviewed.log`). Final performance CSV bytes and all
reviewed implementation hashes remain unchanged.


| Repository | Branch | Reviewed stage checkpoint |
| --- | --- | --- |
| tt-metal | hous/ornith-1.5-9b | `3bdcbf715694d8ac2df1c94e5ca108960d4aa178` |

Checkpoint message: `Add Ornith-1.5-9B functional TTNN decoder and validation`.
The commit hook reran all applicable pre-commit checks successfully. This follow-up
commit changes only this log to record the exact checkpoint SHA; its own SHA is
reported in the final handoff. No pushes or PRs were performed.
