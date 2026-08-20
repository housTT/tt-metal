# Optimized full-model work log

Date: 2026-08-20 EDT

## Scope and starting point

- Model: `Qwen/Qwen3.6-27B`, snapshot
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Stage input commit: `d75b892909a` (`Record Qwen3.6 context gate repair checkpoint`).
- Stage: optimized full model/generator only. No vLLM integration or datatype
  frontier was run.
- Target: 4x Blackhole P300c, mesh `[1,4]`, TP=4,
  `FABRIC_1D_RING`, trace region 1,500,000,000 bytes/device.
- Preserved unrelated dirty state: nested Tracy and UMD submodule changes and
  an untracked `tt-cluster-descriptors` directory. They are not stage-owned and
  will not be committed.

The default installed Python runtime pointed at an older source tree and failed
a mesh smoke while compiling `cq_dispatch.cpp` (`init_telemetry` header/source
mismatch). This was an environment mismatch, not a device failure. All accuracy,
correctness, and uninstrumented performance commands therefore use the working
runtime at `/home/ttuser/dev/tt-metal`, commit
`9b415f82002af5d9040eca389d703690e405d91f`. The Python model under test always
comes first from this checkout.

```bash
export TT_METAL_HOME=/home/ttuser/dev/tt-metal
export PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal:/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal
```

## Baseline and selected optimization

The inherited full model already had the correct TP4 complete path and canonical
split sampling. Its independent optimized-layer medians establish the
42.127888 ms stack lower bound. A fresh full64 baseline measured 42.016435 ms
model trace, 2.417212 ms sampler trace, 44.432408 ms device token-out, and
49.357466 ms caller-visible decode.

The only repeatable avoidable full-path gap was prefill state materialization.
For batch 1, every linear layer allocated a fresh recurrent-state tensor,
wrapped it in a one-element concat, then copied it into the persistent state.
The selected path writes directly into the already-owned persistent layer state.
Batch greater than one preserves per-user tensors and concat, so mixed prompts
and fixed-slot isolation are unchanged.

The final benchmark toggles this model-local policy in one process and
exact-shape warms both variants. Prompt-128 TTFT improved from 679.058899 ms to
655.159903 ms (3.52%). Decode remains at the layer lower bound. Low-level decode also gains
a deferred compact-token API for serving schedulers; the measured autonomous
loop itself remains the existing fully device-resident split-trace path.

## Commands and outcomes

### Host/static contracts

```bash
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py \
  -k 'static_contracts or public_sampling_params or device_sampling_initializes or low_level_device_sampling_reset or host_sampling_top_p'
python -m json.tool models/autoports/qwen_qwen3_6_27b/doc/context_contract.json >/dev/null
git diff --check
```

The focused host suite passed 5/5. JSON and whitespace checks pass.

### Reduced full-model contract

```bash
QWEN36_RUN_FULL_MODEL_SMOKE=1 pytest -q -s \
  models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_full_model_prefill_decode_and_split_trace \
  --junitxml=models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/baseline/reduced_trace.junit.xml
```

Baseline and optimized runs pass mixed non-aligned prompts, changed/unchanged
page tables, direct low-level state, split greedy and stochastic traces, device
feedback, reset, fixed slots, and inactive rows. The final safe-Watcher rerun is
listed below.

### Full64 warmed performance and sampler choice

```bash
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=64 \
QWEN36_TOKEN_OUT_METRICS_JSON=models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/token_out_metrics.json \
pytest -q -s models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_token_out_latency_breakdown \
  --junitxml=models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/token_out.junit.xml
```

Selected results: 42.014693 ms model trace, 1.478412 ms sampler,
43.488361 ms / 22.994658 t/s/u device token-out, and 48.272701 ms /
20.715642 t/s/u caller-visible decode. Selected compact all-broadcast greedy
and generic Sampling1D both returned token 225721; Sampling1D took 10.736105 ms
and was rejected as 7.26x slower. Device token-out is only 3.23% above the independent
decoder-stack lower bound, fully attributable to terminal sampling.

### AIME24 prefill and teacher forcing

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 \
  --output-json models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/prefill_metrics.json

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --reference models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 \
  --warmup-repeats 1 \
  --output-json models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/teacher_forcing_metrics.json
```

Both pass top-1 97/100, top-5 100/100, and top-100 100/100. Steady traced
teacher-forcing replay is 22.558017 t/s/u over 98 post-capture intervals.
Capture-inclusive decode is 19.934041 t/s/u over 99 intervals, with trace setup
reported separately as 622.025 ms. This path remains separate because its
accuracy loop reads the prediction and overwrites the device token with ground
truth on every step. A cold final-code control measured 3.045838 t/s/u;
the log showed first-use sampler kernel compilation inside the decode interval.
One full-reference warmup on the same generator removes that confound.

### AIME24 autoregressive quality

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model /home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --prompt-file models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_autoregressive_prompt.txt \
  --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/autoregressive \
  --max-new-tokens 100 --mesh-device P300 \
  --fabric-config FABRIC_1D_RING --trace-region-size 1500000000

python models/common/readiness_check/check_degenerate_output.py \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/autoregressive/autoregressive_meta.json
```

HF and TT each generated 100 tokens. TT has 63 measured words, adjacent
duplication 0.0, trigram-loop fraction 0.0476, and 40/100 informational token
agreement. Manual inspection finds coherent English step-by-step reasoning,
close to HF through the visible prefix, truncated by the explicit 100-token
evidence limit.

### Shared qualitative suite

```bash
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend hf --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/qualitative --max-new-tokens 64
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend tt --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/qualitative --max-new-tokens 64
python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
  --backend check --output-dir models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/qualitative
```

Final automatic and manual verdicts are stored beside the output JSON files.

### Safe Watcher and profiler

Safe Watcher is intentionally run without the profiler. Token-out and prefill
Tracy captures are separate and use real checkpoint layer indices 0 and 3.
The unscoped Watcher attempt failed before model setup because the instrumented
fabric program was 27,920 bytes for a 25,600-byte ACTIVE_ETH config buffer. The
documented scoped retry disables Ethernet instrumentation while retaining
Tensix Watcher coverage:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN36_RUN_FULL_MODEL_SMOKE=1 pytest -q -s \
  models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_full_model_prefill_decode_and_split_trace \
  --junitxml=models/autoports/qwen_qwen3_6_27b/doc/optimized_full_model/evidence/final/reduced_trace_watcher_distributed_greedy_final.junit.xml
```

The final Watcher run passes in 191.16 seconds and cleanly detaches all four
devices. It includes deferred asynchronous batch-3 readback with 29 inactive
fixed slots, mixed prompt lengths 65/67/65, batch-32 decode, changed-only page
tables, stochastic sampling, reset, and traced distributed greedy feedback.
The first attempt reached that deferred path without a Watcher fault but had an
incorrect assertion about the history container type; the assertion was
corrected and the complete gate rerun.

Final postflight `tt-smi -s` reports all four Blackhole p300c devices with
`dram_status=true`, synchronized heartbeat `26922`, and 44.7--46.6 C ASIC
temperatures. No reset or recovery was required.

Token-out and prefill profiles both pass. The refreshed token report has 228
rows and 3,380.81 us summed device time; the prefill report has 532 op rows and 5,746 us.
Advice-enabled `tt-perf-report` CSV tables, summaries, checksums, exact raw
paths, commands, and conclusions are in `evidence/final/profiler/capture.md`.
Two initial prefill captures passed execution but failed Tracy merge after the
compile warmup filled profiler DRAM. Flushing immediately after compile warmup
produced the complete final host/device ledger.

## Autofix closure after initial stage review

The first independent review correctly found that inherited `force_argmax`
still gathered all 262,144 logits and that teacher/TTFT comparisons mixed
runtime and cold-cache conditions. Fresh-context AutoDebug localized the
issues; the full report is `AUTODEBUG.md` and the experiment ledger is
`AUTOFIX.md`.

The adapted distributed argmax computes local BF16 max/argmax, packs exact
candidate values and indices, reconstructs global IDs without TF32 precision
loss, and writes the persistent RM uint32 `tt_out_tok`. Three candidate Ring
all-gather protocols (async, async with barrier, and synchronous) were rejected
after safe-Watcher writer stalls. Compact `all_broadcast` plus concat passed
eager/trace correctness and safe Watcher. Full64 sampler latency fell from
2.417212 to 1.478412 ms. The final profile proves two compact all-broadcasts,
local reductions/argmax, no all-gather, and no TopK.

For teacher forcing, a reduced 4-layer same-process control measured 5.061 ms
autonomous token-out, 5.318 ms with compact read/callback/forced-token copy, and
5.380 ms with an additional full-mesh sync. This refuted host synchronization
as the large regression. The cold full64 run showed sampler kernel compilation
inside its measured interval. The readiness runner now accepts
`--warmup-repeats`; its unit tests pass. A rereview then found that reset
correctly releases request traces, so capture was still inside the old metric.
The runner now retains capture-inclusive time and starts its steady metric at
the first decoded callback. The final warmup-1 AIME run measures 22.558017
t/s/u steady and 19.934041 t/s/u including 622.025 ms setup while retaining
97/100 top-1 and 100/100 top-5/top-100. The timing lifecycle suite passes 6/6.

One overly broad local pytest selection accidentally collected hardware cases
under the default older runtime and aborted during mesh open. All four devices
subsequently reported healthy DRAM and synchronized heartbeats; no reset was
required. The intended readiness-runner unit suite then passed under the scoped
command; after adding direct timing separation it covers warmup/reset,
measurement/teardown, and trace-interval arithmetic (6/6).

## Context and limitations

`doc/context_contract.json` preserves the full 262144 logical context, batch 32,
shared-cache capacity semantics, non-aligned prompt support, selected precision,
replicated residual contract, BFP8 paged KV policy, persistent CCL pool, and
physical Ring topology. No capability was reduced.

The model is intentionally fixed to a four-device P300c Ring. The initial
prefill/readiness boundary returns host logits, and the caller-visible Python
generator observes one compact token per step. The separately measured device
token-out path has neither boundary. Explicit host sampling remains compatibility
only. No vLLM work or broad datatype search is included.

## Review and commits

The first post-Autofix review found that the warmed teacher metric still
included reset-owned trace recapture. After the interval split and refreshed
AIME artifact, a second fresh-context `$stage-review` returned `clean-pass`
with no required work; its complete verdict is `stage_review.md`.

Local checkpoint commits are recorded below. Nothing is pushed.
