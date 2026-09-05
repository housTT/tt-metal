# AutoFix: canonical RMSNorm partial reduction order

Status: **fixed and verified, including exact full-stack batch logits**.

The initial controlled-numerics conclusion was insufficient: the all-32-layer
B32 gate (`full_batch32_final_v1`) diverged at token 4 in slot 6, despite the
reduced model's matching greedy winners. The retained native kernel repair
now makes duplicate full-model logits and token sequences bitwise identical
across all 32 slots and repeated physical page permutations. This supersedes
the earlier numerical-order qualification in `AUTOFIX_batch32.md` and the
historical diagnosis recorded below.

## Retained repair and final verification

In `reader_mcast_receiver_unary_sharded_ln.cpp`, single-stage RMSNorm receivers
still issue reads in cyclic peer order, but store each partial into the CB
offset for its canonical peer index. Compute consequently accumulates peers
in the same order for every row group. The existing `start_x/start_y` runtime
arguments determine that index; no factory/library API or new argument is
needed. Sender peer order is already canonical for these shard groups.

The fix changes neither precision/fidelity nor input/output shard layout,
communication count, NoC issue order, compute algorithm or cache strategy.
It is scoped to single-stage RMSNorm; LayerNorm and two-stage reductions are
unchanged. The FP32 accumulation and height-shard experiments below remain
diagnostic/rejected alternatives, not runtime fallbacks.

Final artifacts establish:

- `norm_rows_canonical.json`: frozen Q and K inputs produce bitwise equal
  outputs in all 32 rows on all four ranks, matching the original slot-0
  anchor exactly. The former maximum discrepancy was 0.0625.
- `norm_rows_canonical_b1_watcher.json`: explicit batch-one Q/K outputs remain
  bitwise equal to those original anchors on every rank; worker watcher passes.
- `trace_b32_canonical_norm.json`: the reduced B32 mixed/fixed-slot/feedback/
  sampling/page-table contract passes with worker watcher and allocation
  tracking enabled.
- `full_batch32_canonical_norm.json`: all 32 layers and all 32 active slots
  have exact duplicate full-vocabulary logits, exact repeated/permuted logits,
  identical token sequences and the expected device-only loop counters.
  Slot 6 now matches `[240,129,240,129,240,129,240,129]`; these synthetic token
  inputs are a structural contract check, not a language-quality claim.
- `logs/norm_duplicate_rows_pytest.log`: the generic native RMSNorm regression
  passes for both ROW_MAJOR and COL_MAJOR eight-core shard orientations under
  worker watcher: **2 passed**.

The same 128-replay isolated norm timing is Q 0.024835 ms / K 0.024857 ms
after, versus Q 0.024813 ms / K 0.024835 ms before (about 0.09% difference).
There is no material measured operation slowdown or claimed improvement.
These are operation diagnostics, not full-model performance figures.

The changed native device source JIT-compiled and executed in these checks.
The required `timeout 60 .github/scripts/copilot-build.sh` attempt failed
because Docker is unavailable; `logs/norm_copilot_build.log` records the
environment limitation. A complete CI-image host build is **unverified**.
`clang-format --dry-run --Werror` and `git diff --check` pass for the native
patch, and Black passes for the Python regression/probes. No reset was needed.

## Historical first-difference localization

`probe_batch_boundaries.py` allocates persistent diagnostic buffers before
capture, copies into them within the model trace, and reads them only after
replay. `batch32_boundaries_norm.json` and `.pt` retain selected slots
0/3/6/30 on all four ranks:

1. Embedding, linear layer 0 output, QKVG projection and split Q/K norm inputs
   are bitwise identical for duplicate prompts.
2. Q/K RMSNorm outputs first differ; rotary and SDPA consume those differences.
3. Values entering the paged V cache remain exact. The K-cache differences
   originate upstream of cache update; they are not evidence of an incorrect
   physical page write.
4. Full-vocabulary logits have maximum cross-slot difference 0.0625 and
   minimum PCC 0.9999645084; the largest absolute logit magnitude is 18.125.
   The same slots are bitwise identical across A/A2/B
   in the prior diagnostic, and every sampled token equals CPU greedy on all
   four replicas.

## Source mechanism and decisive control

`tt/optimized_decoder.py::_norm` reshards Q/K across eight width shards, using
`block_h=batch`, `block_w=1`, `subblock_w=1`. This is the preserved optimized
decoder implementation, not a newly selected full-model fallback.

The native `rmsnorm_default_compute_config` selects HiFi4,
`math_approx_mode=true`, `fp32_dest_acc_en=false`. In
`sharded_layernorm_factory_helpers.cpp` (around lines 1788–1805), each
all-to-all worker gets `start_x` from its own core coordinate. In
`kernels/dataflow/layernorm_dataflow_utils.h::compute_single_stage_noc_addrs`
(lines 121–151), peer traversal starts there and wraps cyclically. Different
workers own different tile-row groups. The receiver gathers the partials in
that explicit order and `kernels/compute/layernorm_sharded.cpp` reduces them
in stream order. Thus different row groups use different floating-point
partial-sum orders. This is deterministic ordering, not network arrival
order or an RNG-dependent explanation.

`probe_norm_rows.py` loads the saved real Q/K inputs, copies each rank's anchor
into **all 32 rows**, and executes the exact eight-core width-sharded norm
program with the real folded BF16 norm weight. `norm_rows_controls_v3.json`
proves both Q and K:

- The original operator reproduces every selected output row of the full
  traced diagnostic **bitwise on every rank**, including the row differences.
  This rules out cache, masking, prior token feedback, input packing and trace
  lifetime as explanations for this discrepancy.
- A diagnostic with the same HiFi4/approximate mode, BF16 input/output and
  eight-core layout but FP32 destination accumulation gives exact duplicate
  outputs across all tested row positions. It is not selected for runtime,
  because the user requires preserving the established precision policy.
- The proposed direct height-sharded norm alternative is concretely rejected
  by native validation: `Height sharded inputs are not supported.` Both Q and
  K rejections are retained. No speculative height-layout change remains.

CPU oracle calculations use `x * rsqrt(mean(x*x) + 1e-6) * folded_weight`
on the actual dequantized norm input and BF16 folded weight. Full per-rank
metrics for both original and FP32 diagnostic are in
`norm_rows_controls_v3.json`; the trace-input CPU evaluation is also preserved
in `batch32_norm_cpu_oracle.json`. The original norm agrees with the CPU
oracle at minimum PCC 0.9999920531 (Q) and 0.9999918242 (K) across ranks and
all 32 frozen rows. The FP32 diagnostic reaches 0.9999983357 (Q) and
0.9999981220 (K). These correlations alone were not used to dismiss the discrepancy—the
isolated bitwise reproduction and source traversal establish its cause.

Original width-norm trace timing, 128 replays without profiler or watcher,
is 0.024813 ms for Q and 0.024835 ms for K. These are isolated operation
diagnostics, not a full-model improvement claim.

## Commands and provenance

All commands run from repository root with:

```bash
export TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache
export OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
```

Final repair commands (each was run through `record_run`; the labels below
have matching immutable `.provenance.json` and `.sources.json.gz` files):

```bash
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run norm_rows_canonical \
timeout 120 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_norm_rows \
--expect-canonical --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/norm_rows_canonical.json

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run trace_b32_canonical_norm \
timeout 240 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.trace_contract \
--batch 32 --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/trace_b32_canonical_norm.json

TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run full_batch32_canonical_norm \
timeout 480 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.full_batch_contract \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/full_batch32_canonical_norm.json

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run norm_rows_canonical_b1_watcher \
timeout 120 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_norm_rows \
--expect-canonical --output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/norm_rows_canonical_b1_watcher.json

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run norm_duplicate_rows_pytest \
timeout 180 python_env/bin/python -m pytest tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py \
-k duplicate_batch_rows -q --disable-warnings
```

The source snapshots include the changed RMSNorm receiver and the sender/
coordinate helper; the generic regression snapshot also includes its test
file. The following commands are historical pre-repair localization controls:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 \
python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run batch32_boundaries_norm \
timeout 90 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_batch_boundaries \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/batch32_boundaries_norm.json

python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.record_run norm_rows_controls_v3 \
timeout 90 python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.doc.full_model.probe_norm_rows \
--output models/autoports/ornith_ai_ornith_1_5_9b/doc/full_model/norm_rows_controls_v3.json
```

The earlier ladder commands and their source archives are retained under
`logs/batch32_boundaries_v1.*` and `logs/batch32_boundaries_sdpa.*`. The first
standalone invocation used a nonexistent Python `BlackholeComputeKernelConfig`
class and stopped before the norm controls; it was corrected to the installed
`WormholeComputeKernelConfig`, which configures this Blackhole backend. The
second invocation established the Q controls and then stopped at the explicit
height-shard validation. The third records that rejection and completes both
Q/K controls. All runs closed normally; no reset, profiler, hardware downgrade,
decoder policy change or custom normalization kernel was used.
