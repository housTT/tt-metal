# Optimized full-model work log

## Scope and starting point

- Model: `google/gemma-4-26B-A4B-it`
- Starting branch/commit: `hous/gemma-4-26b-a4b-it` at
  `6740ab49d22` (`Record full-model gate repair commit`)
- Checkpoint revision:
  `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- Stage boundary: optimize the completed full model and generator on 1/2/4-chip
  P150 proxies; no vLLM code or registration work.
- Selected skills: `multichip`, `optimize`, and `tt-device-usage`, with
  `full-model`, `tt-enable-tracing`, `qualitative-check`, `autofix`, and
  `stage-review` used for their applicable gates.

The worktree was clean at the start. Device commands were serialized. The
hardware was one four-chip Blackhole P300C QB2, using 1x1, a 1x2 submesh of a
FABRIC_2D parent, and 1x4 FABRIC_1D_RING.

## Baseline and hypothesis

The inherited full model already had a real 30-layer tensor-parallel decoder,
BF16 paged caches, persistent CCL resources, vocabulary-sharded LM head,
split model/sampling traces, `tt_out_tok` alias feedback, device position/RoPE
advance, and changed-only page tables. The inherited profiler identified an
approximately 1.4 ms decode-time untilize of the complete BFP8 tiled embedding
table. A prior terminal trial had measured BF16 row-major embedding near 0.136
ms, but the full-model stage rejected it using a conservative P150 source-live
projection rather than actual maximum-context construction.

The baseline was frozen before changing the default:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_PREFILL_BENCH=1 GEMMA4_NO_HOST_TOKEN_OUT_BENCH=1 \
  GEMMA4_NO_HOST_WARMUPS=5 GEMMA4_NO_HOST_ITERATIONS=128 \
  GEMMA4_PROBE_PROMPT_LEN=128 GEMMA4_EMBEDDING_STORAGE=bfp8_tile \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=<optimized_full_model/baseline/profile> \
  python_env/bin/pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-<profile>]'
```

Baseline warmed prefill-to-logits/token-out were 154.223/32.701 ms on P150,
130.349/24.296 ms on P150x2, and 106.076/21.490 ms on P150x4.

## Implementation and selection

`Gemma4FullModel` now resolves profile embedding storage to BF16 row-major on
TP1/TP2/TP4 and keeps an explicit `bfp8_tile` override for controlled A/B
measurement. The embedding cache key includes topology and storage format.
LM-head weights remain vocabulary-sharded BFP8_B. No decoder precision,
fidelity, CCL, cache, residual, matmul, program-config, or kernel policy was
changed.

The candidate was rerun on the same full-stack workload, then exact
maximum-context allocation was measured:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_CAPACITY_ONLY=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/capacity \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/capacity/junit.xml
```

Construction passed at the inherited limits, but independent review correctly
noted that this did not prove peak prefill memory. Complete 30-layer P150
prefill at 50,623 tokens initially failed with a DRAM fragmentation OOM. The
failure was localized to `_full_chunked_prefill_attention`: all full-attention
Q chunks had been dispatched, but the complete `q_heads` allocation remained
live while `ttnn.concat` requested an identically shaped output. Deallocating
`q_heads` after the final chunk dispatch and before concat creates an exact
contiguous block for reuse; it changes no attention math, precision, or layout.

The final default-source P150 rerun used all 30 layers, strict fallback, and
allocation tracking. Its nonaligned 50,623-token prefill, final legal traced
position 50,624, and following-position capacity rejection passed in 117.36
seconds. The trace recorded zero host reads, synchronizations, and page-table
refreshes. Post-boundary allocation is 29,437,724,160 bytes, leaving
4,673,897,984 bytes free and a 579,556,608-byte largest contiguous block per
bank. The authoritative artifacts are in
`final/full_stack_context_tp1/lifetime_fix_50624/`. P150 therefore preserves
50,624 tokens; P150x2/P150x4 remain 262,144.

## Final performance and lower bound

After the watcher-discovered CCL fix below, the final-source performance sweep
was refreshed with this exact command:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_PREFILL_BENCH=1 GEMMA4_NO_HOST_TOKEN_OUT_BENCH=1 \
  GEMMA4_NO_HOST_WARMUPS=5 GEMMA4_NO_HOST_ITERATIONS=128 \
  GEMMA4_PROBE_PROMPT_LEN=128 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/profiles_refresh \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/profiles_refresh/junit.xml
```

Result: 3 passed in 182.68 s. Final warmed prefill-to-logits/token-out are
148.127/26.841 ms (P150), 126.901/21.491 ms (P150x2), and
106.326/20.102 ms (P150x4). Measured-loop counters remain zero for host token
readback, host synchronization, and page-table refresh.

The decoder stack lower bound uses the optimized multichip layer winners:

- P150: `25 * 0.759652 + 5 * 0.885008 = 23.416340 ms`.
- P150x2: `25 * 0.646949 + 5 * 0.722745 = 19.787450 ms`.
- P150x4: `25 * 0.585234 + 5 * 0.914335 = 19.202525 ms`.

The initial logits-only probe used an unmatched position window and is retained
only as superseded evidence. It was rerun with initial position 128, explicit
device `plus_one`, five warmups, and the same measured positions `[134,262)` as
token-out. Final comparable logits-only traces are 25.316054, 20.736552, and
19.647055 ms. Full
token-out gaps over the layer lower bound are 14.62%, 8.61%, and 4.68%; none
exceeds the requested 10-15% band. Sampling/feedback increments relative to
the matched logits trace are 1.5249, 0.7541, and 0.4548 ms.

The public host-visible generator was also measured separately:

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_GENERATE_BENCH=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/public_generator \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/public_generator/junit.xml
```

It returns 128 host tokens and measures warmed decode at 37.110, 46.266, and
49.441 t/s/u. This request boundary has the expected 128 readbacks and is not
reported as the no-host token-out loop.

A controlled otherwise-final rerun with `GEMMA4_EMBEDDING_STORAGE=bfp8_tile`
provides a same-workload public-generator baseline. BFP8-to-BF16 public TTFT is
188.518 -> 182.831 ms, 148.851 -> 147.120 ms, and 139.205 -> 138.561 ms on
P150/P150x2/P150x4. Warmed request decode improves from
30.532/40.967/46.300 to 37.110/46.266/49.441 t/s/u. The control JUnit records
three passes in `final/public_generator_bfp8_control/`.

## Sampler A/B

```bash
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_SAMPLER_AB_BENCH=1 \
  GEMMA4_SAMPLER_WARMUPS=5 GEMMA4_SAMPLER_ITERATIONS=128 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/sampler \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/sampler/junit.xml
```

Result: 3 passed in 17.49 s. Split local-top32 semantic greedy costs
1.502/0.760/0.446 ms on TP1/TP2/TP4. Force-argmax full-vocabulary gather costs
1.483/2.731/2.297 ms. TP1 is a sub-3% near-tie and retains the unified split
contract; TP2/TP4 strongly reject force-argmax. All choices equal the trusted
global maximum. Top-k/top-p remains supported by the same traced sampler.

## Accuracy and generation quality

The exact chat-template AIME24 reference from the pinned checkpoint was reused:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/python /tmp/run_gemma4_readiness_profile.py prefill <tp>

env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  python_env/bin/python /tmp/run_gemma4_readiness_profile.py teacher <tp>
```

Prefill top-1 is 95%/96%/97%; traced teacher-forcing top-1 is 96%/96%/95%.
Every profile is 100% top-5 and top-100 in both modes, exceeding the required
top-5 >=98% and top-100=100% gates. Teacher decode (which includes host teacher
feedback) is 34.494/41.447/43.350 t/s/u and is reported separately.

Free-running P150x4 greedy generation used the exact 161-token chat prompt and
100 tokens:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_aime24_autoregressive.py

python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/qualitative/aime24_autoregressive \
  --scope autoregressive --missing-artifacts critical \
  --json /tmp/gemma4_optimized_degeneracy.json
```

The output is coherent, on-task English math reasoning with no mechanical
degeneracy. The checker exits 0 with no findings. HF/TT token agreement is
9/100 and recorded only as an informational comparison.

The shared six-prompt suite was freshly rerun on the selected optimized path:

```bash
env PYTHONPATH=/home/hous/dev/tt-metal \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_TRACE_ALLOC_TRACKING=1 \
  python_env/bin/python /tmp/gemma4_shared_qual.py
```

Each pinned-chat-template prompt generated 64 greedy tokens through HF and the
traced TT generator, with TT reset between requests. All six pass task
alignment, coherence, repetition, language-drift, prompt-echo, and
cross-request-leakage review. Outputs, prompt IDs, retained prompt source,
metadata, and per-case assessment are under
`final/qualitative/shared_readiness_suite/`.

## Serving state, batch, and nonalignment

Allocation-tracked final-source gates cover full-stack TP4 batch 32 with four
mixed prompt families, explicit fixed slots, every row selecting a global-max
logit, and minimum B1/B32 cosine 0.99550. `mixed_state_tp4.json` covers active
and inactive rows, stable trace IDs, and changed-only page tables.

P150 ran its complete 30-layer path one token below the selected limit:
nonaligned prefill 50,623 followed by final traced position 50,624. The next
position is safely rejected before allocation. P150x2/P150x4 retain
the earlier real-terminal representative-layer 262,143-token nonaligned probes,
final position 262,144, and multi-gigabyte full-stack allocation headroom.

## Tracy and `tt-perf-report`

Watcher and profiler were run separately. One final reduced real-weight P150x4
Tracy capture includes separate warmed-prefill and steady-decode signposts for
layers 0 and 5 plus terminal and split-sampling work:

```bash
env -u TT_METAL_WATCHER HF_HOME=/home/hous/.cache/huggingface \
  HF_HUB_OFFLINE=1 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_MODEL_DEVICE_PROFILE=1 \
  GEMMA4_PREFILL_BENCH=1 GEMMA4_PROBE_PROMPT_LEN=128 \
  python_env/bin/python -m tracy -r -p \
  -o gemma4_optimized_full_model_complete_replay_20k_tp4 -m pytest -q -s \
  'models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe[blackhole-p150x4]'

python_env/bin/tt-perf-report <raw_ops.csv> \
  --arch blackhole --active-experts 8 \
  --start-signpost FULL_MODEL_REDUCED_PREFILL \
  --end-signpost FULL_MODEL_REDUCED_PREFILL_END ...

python_env/bin/tt-perf-report <raw_ops.csv> \
  --arch blackhole --active-experts 8 \
  --start-signpost FULL_MODEL_REDUCED_DECODE \
  --end-signpost FULL_MODEL_REDUCED_DECODE_END ...
```

The actual executable is `python_env/bin/tt-perf-report`, package version 1.2.9.
The retained raw CSV and separate processed CSV/report/PNG pairs are under
`final/profiler_final/`. Mid-run dumps plus the 20,000-program support count
prevent the profiler-buffer truncation found in the superseded capture. Device
0 contains exactly 195 model-trace and 12 sampling-trace operations in both
replay sessions. Prefill spans 9.657513 ms between host signposts and sums
7,657.06 us across 239 merged device operations; its modeled DRAM roofline is
9.6% (49 GB/s). Decode spans 3.581892 ms and sums 2,591.76 us across 207
operations; its modeled DRAM roofline is 19.7% (101 GB/s).

The split sampling trace sums 416.253/418.336 us across the two sessions. Its
279.48 us `TopkLargeIndicesDeviceOperation` consumes the local
`[1,1,32,65536]` vocabulary shard and is the largest sampler subcomponent, not
MoE routing. The final sampling choice is 27.38 us. The two generic TopK
operations total 44.67 us and belong to MoE routing in the model trace, not the
sampler; six persistent all-reduces total 107.96 us. There is no full-vocabulary
gather or force-argmax in the measured path. The inherited full-table embedding
untilize remains absent. Processed CSV `Advice` columns and the inherited
decoder rejection ledger are retained.

## AutoFix: watcher-discovered CCL defect

The first complete watcher sweep passed P150, then aborted on P150x2. The
initial non-watcher paths and device health were clean, so AutoDebug first
ranked a watcher/runtime interaction. A focused reduced P150x2 run with
`PYTHONFAULTHANDLER=1` captured the actual exit 134 and watcher assertion:

```text
multicast_writer.cpp: BRISC tripped assert line 279
TT_THROW: Watcher detected tripped assert and stopped device
```

The included assertion is `tt_metal/fabric/hw/inc/api_common.h:279`, requiring
scatter chunk count in `[2,4]`. The sampler's UINT32 top-k index tile is 4,096
bytes while the fabric packet is 4,352 bytes, so `pages_per_packet=1` and the
all-gather helper selects unicast. Its constructor nevertheless initialized an
unused scatter header with one chunk. The fix wraps unused scatter-state setup
in `if constexpr (use_scatter_write)` in both multicast and symmetric unicast
helpers. It does not change router payload, sampler layout, or valid transfer
behavior.

After `timeout 180 tt-smi -r` and a clean four-device list, the exact reduced
TP2 and TP4 watcher controls passed. The complete final gate was:

```bash
env PYTHONFAULTHANDLER=1 TT_METAL_WATCHER=10 \
  TT_METAL_WATCHER_DISABLE_ETH=1 \
  HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}' \
  GEMMA4_FULL_MODEL_PROBE=1 GEMMA4_FULL_STACK_PROBE=1 \
  GEMMA4_FULL_MODEL_PROBE_OUTPUT_DIR=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/watcher/full_stack_fixed \
  python_env/bin/pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_full_model_contract.py::test_reduced_real_weight_full_model_probe \
  --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_full_model/final/watcher/full_stack_fixed/junit.xml
```

Result: all three complete 30-layer profiles passed in 187.31 s with watcher
polling and normal teardown. The failing console, fixed reduced logs, and full
fixed logs are retained under `final/watcher/`.

## Skill checklist closure

### `multichip`

- Distinct 1x1, 1x2, and 1x4 real TP paths: evidenced by final profile,
  capacity, and watcher artifacts.
- Weight/embedding/LM-head sharding: embedding is hidden-sharded, LM/logits are
  vocabulary-sharded, decoder policy unchanged.
- CCL: TP2 Linear one-link and TP4 Ring two-link persistent BF16 reductions;
  sampler uses candidate gathers, never a full-vocabulary gather.
- Inter-layer residual: replicated BF16 TILE DRAM unchanged with no between-
  layer collective.
- Explicit serving state, fixed slots, mixed prompts, inactive rows, nonaligned
  prefill, paged caches, and changed-only tables all pass.

### `optimize`

- Baseline frozen before tuning; one isolated full-path hypothesis selected.
- Warmed prefill-to-logits, actual public-generator TTFT, and traced token-out
  measured on every required profile.
- Decoder matmuls, fidelity, sharding, CCL, program configs, sparse kernels,
  cache dtype, and residual layout retain their prior measured winners.
- `tt-perf-report` raw/CSV/table/plot and provenance retained; terminal
  conversion closed and sampler dominance refuted.
- No broad datatype frontier search was performed.
- Strict runtime fallback audit is clean for every measured path.

### `tt-device-usage`

- Initial/final four-device listing passed; commands serialized.
- Watcher and Tracy were separate runs.
- The watcher assert was captured before reset; bounded reset/list recovered all
  four devices; reduced controls preceded the full retry.
- No performance number was taken under watcher or allocation tracking.

## Host verification and commit ledger

The C++ build command required by `AGENTS.md` was attempted:

```bash
.github/scripts/copilot-build.sh
```

The wrapper exited immediately with `ERROR: docker is not available` because
the Docker CLI cannot access `/var/run/docker.sock` on this host (`docker info`
returns permission denied). Therefore the host C++ build is unverified for an
environmental reason. Device JIT compilation did rebuild the affected kernels
and the watcher-enabled real-model gates passed, but this is not represented as
a replacement for the required host build.

Final verification before independent review:

- `python_env/bin/pre-commit run --files <stage source/docs>`: all applicable
  hooks passed, including Black/isort/autoflake, clang-format, Metalium include
  validation, and the 500 KiB artifact limit.
- `python_env/bin/pytest -q .../test_full_model_contract.py -k 'not reduced and
  not public_nonaligned'`: 14 passed, 5 deselected in 3.25 s; JUnit is
  `final/host_unit_junit.xml`.
- All optimized-full-model JSON plus `../context_contract.json`: `jq empty`
  passed.
- Direct AIME24 degeneracy check: exit 0, no advisory or critical finding.
- `MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it bash
  .agents/prompts/model_bringup_multigoal/07-optimized-full-model.check.sh`:
  exit 0, no degenerate output.
- Final `timeout 60 tt-smi -ls --local`: all four P300C devices visible and
  reset-capable.

Large plain-text watcher logs were losslessly compressed with `xz`; auxiliary
Inspector/fabric duplicates were moved to
`/tmp/gemma4-watcher-generated.JDIbVa`. The old incomplete canonical profiler
capture was moved recoverably to `/tmp/gemma4-profiler-incomplete.uMCrPu`;
generated profiler roots were moved to
`/tmp/gemma4-profiler-generated.8IZXa5` after the complete raw CSV and
processed reports were retained in `final/profiler_final/`.

The independent stage-review result is appended after its fresh-context run.

Stage commit SHAs are appended after the local commits are created. Nothing is
pushed.
