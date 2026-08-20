# Full-model work log

## Scope and starting state

- Model/checkpoint: `Qwen/Qwen3.6-27B`, snapshot
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- Hardware: four Blackhole P300c devices, `1x4`, TP=4,
  `FABRIC_1D_RING`.
- Starting optimized multichip checkpoint: `55e728d6635` (implementation
  checkpoint `0dc44c70024`).
- Added only the repo-local full model, generator, tests, readiness-runner
  support, reference/evidence, and the GDN correctness repair required by the
  full stack. No vLLM files or registration were touched.
- Hardware commands were serialized. Because the checkout and installed TTNN
  extension otherwise select mismatched build headers, commands ran from
  `/home/ttuser/.local/lib/model-bringup/tt-metal` with
  `PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal`.

## Implementation sequence

1. Constructed the replicated BF16 embedding, all 64 existing
   `MultichipDecoder` layers, replicated final RMSNorm, and TP4 vocabulary BFP8
   LM head in `tt/model.py`.
2. Added explicit `FullModelState` ownership for BFP8 paged caches, linear
   recurrent state, page table, positions, tokens, prompt lengths, active
   slots, and capacity. Public prefill owns non-aligned/mixed padding, masking,
   cache fill, and slicing.
3. Added `tt/generator.py` with standard readiness discovery, low-level
   prefill/decode, explicit host compatibility, in-place reset, and canonical
   model-trace plus sampler-trace replay.
4. Compacted batch-1 linear-attention decode to the active rows inside the
   trace while retaining the full-attention batch-32 contract. This reduced a
   four-layer model trace from 77.272 ms to 3.595 ms; inactive rows no longer
   execute 31 redundant recurrent paths.
5. Extended the shared readiness runner with the `P300` mesh label, explicit
   trace-region sizing, Transformers-5 chat-template `BatchEncoding` handling,
   and preservation of serialized prompt whitespace.

## AutoFix correctness investigation

The first full-stack prefill produced only 1/100 top-1, 1/100 top-5, and 8/100
top-100. Layer-by-layer real-weight isolation found the divergence in the
linear GDN triangular inverse. The polynomial 64-wide factorization was exact
in real arithmetic but numerically unstable at actual embedding scale.

Controls established:

- host exact inverse restored the layer, proving the rest of the full stack;
- 16-token TT chunks were correct but rejected as a less-optimized execution
  fallback;
- the dedicated GDN op remained rejected from the optimized stage because it
  expands the layer to 344 ops and 6482.619 us;
- a four-block hierarchical inverse preserved the 64-token outer chunk and the
  same ten TT matmuls while avoiding unstable high powers.

Real TP4 cumulative PCC after layers 0--3 became 0.99898648, 0.99892616,
0.99827313, and 0.99906021. The final full 64-layer prefill then passed at
97/100 top-1 and 100/100 top-5/top-100. No host inverse, higher-fidelity rescue,
single-chip path, or smaller production chunk remains.

## Reference and correctness commands

The fresh reference was generated with the equivalent command:

```bash
python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3.6-27B \
  --prompt-source aime24 --aime24-prompt-index 0 --chat-template \
  --gen-len 100 --top-k 100 \
  --output models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt
```

It has one 161-token chat-template prompt, exactly 100 greedy continuation
tokens, and 100 reference candidates at every step.

Final prefill:

```bash
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
python -m models.common.readiness_check.run_prefill_check \
  --model-dir /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b \
  --reference /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING
```

Result: top-1 97/100, top-5 100/100, top-100 100/100.

Final traced teacher forcing:

```bash
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b \
  --reference /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000
```

Final structured result: top-1 97/100, top-5 100/100, top-100 100/100,
TTFT 14,620.75 ms, traced teacher-forcing decode 19.22 t/s/u, and end-to-end
5.06 t/s/u. The selected sampler logged `cluster_axis=None`, one link, and
`Topology.Ring` during trace setup.

The reduced state/trace/mixed-prompt gate used:

```bash
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_RUN_FULL_MODEL_SMOKE=1 \
pytest -q -s /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py \
  -k reduced_full_model_prefill_decode_and_split_trace
```

It passed token feedback, active/inactive position coherence, changed and
unchanged page tables, in-place reset, non-aligned prompt length 3, mixed
lengths `[3,5]`, compact linear state, and mixed decode positions `[4,6]`.

## Sampling and performance ledger

The 64-layer split timing command was:

```bash
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=64 \
pytest -q -s /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py \
  -k reduced_token_out_latency_breakdown
```

The test name is historical; `QWEN36_BENCH_LAYERS=64` constructs the complete
production stack. The final structured Ring force-argmax result is 42.141 ms
model trace, 2.416 ms selected sampler trace, 44.556 ms combined, and 22.444 t/s/u.
Sampling is 5.4%, so it does not dominate token-out. Machine-readable final
evidence is `evidence/token_out_ring_metrics.json`; pass status is in
`evidence/full64_ring_argmax_metrics.junit.xml`.

| Candidate | Semantically greedy | Trace latency | Decision |
|---|---|---:|---|
| Common `SamplingGenerator` / `TTSampling` Ring force argmax | yes | 2.416 ms | selected; fastest common path, cached split trace, direct device feedback, and full wrapper lifecycle |
| Common `Sampling1D` | yes; same selected token 225721 | 10.797 ms | rejected; 4.47x slower and less wrapper lifecycle support |
| Common standard top-k through Linear gather | yes (`top_k=1`) | 10.797 ms before Watcher | rejected; slower, and safe Watcher reproduced an async all-gather writer stall |
| Common force argmax through Linear gather | yes | 3.134 ms without Watcher | rejected; safe Watcher reproduced an async all-gather writer stall; physical hardware is a 1x4 Ring |
| Custom local top-1 plus gather | yes | 10.871 ms | rejected; slower than the common selected path, so no custom sampler was retained |
| Custom max plus argmax | yes | 32.838 ms | rejected; slower than both common paths |
| LM head 8192 columns / 32 cores | n/a | reduced model trace 77.206 ms versus 77.278 ms at 4096/16 | rejected; equal performance and larger working geometry |
| LM head 16384 columns / 64 cores | n/a | illegal | rejected; DRAM-sharded matmul reports uneven K sharding (`K:160`, `per_core_K:3`) |

The selected fixed LM-head geometry is 4096 local vocabulary columns over 16
cores. Its real-weight PCC is 0.99982172 with 96/100 top-100 overlap; the legal
8192/32 alternative is 0.99983120 with the same overlap and no latency win.

### Terminal device-op profile

The final selected Ring capture reused the command below with
`TT_METAL_CACHE=/tmp/qwen36_full_model_ring_tracy_cache` and output directory
`evidence/profiler/raw_ring_argmax`. The signpost-bounded report contains 145
merged rows and 3,962.61 us summed device time: argmax 1,417.43 us (35.77%),
Ring all-gather 883.30 us (22.29%), and 21 width-sharded matmuls 988.52 us
(24.95%). No top-k operation is present. Compact outputs use the `ring_`
prefix; `ring_capture.md` records provenance and raw-intermediate deletion.

The successful compact Tracy capture used the profiler-enabled runtime, one
real linear-attention layer, the full final norm/TP LM head, and common force
argmax through the subsequently rejected Linear route. It used C++ runtime
analysis and flushed setup records before the signpost:

```bash
PANDAS_FUTURE_INFER_STRING=0 \
TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler \
TT_METAL_CACHE=/tmp/qwen36_full_model_tracy_cache \
PYTHONPATH=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/ttnn:/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/tools:/home/ttuser/dev/qwen-perf/tt-metal \
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=1 \
QWEN36_PROFILE_TOKEN_OUT=1 \
python -m tracy -p -r -v --check-exit-code --dump-device-data-mid-run \
  --tracy-tools-folder=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler/build/tools/profiler/bin \
  -o models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/profiler/raw_token_out \
  -m pytest -q -s models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py \
  -k reduced_token_out_latency_breakdown

PANDAS_FUTURE_INFER_STRING=0 tt-perf-report <generated-ops.csv> \
  --start-signpost QWEN36_FULL_MODEL_TOKEN_OUT_START \
  --end-signpost QWEN36_FULL_MODEL_TOKEN_OUT_END \
  --csv models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/profiler/token_out_report.csv \
  --summary-file models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/profiler/token_out_summary
```

The merged signpost report totals about 4.596 ms of device time. All-gather is
1.518 ms (33.02%), argmax is 1.417 ms (30.84%), and 21 width-sharded matmuls
total 0.990 ms (21.53%). This proved force argmax removed the former top-k
bottleneck, but safe Watcher then rejected the Linear routing contract.
Four-layer profiler attempts overflowed the legacy marker buffer, so production
evidence uses the direct final 64-layer Ring timing: model 42.141 ms, sampling
2.416 ms. The compact files remain clearly labeled rejected-path diagnostics.
Raw Tracy intermediates were deleted after distillation. `token_out_report.txt`
retains category warnings emitted while the tool scans the source capture;
the signpost-bounded CSV and summary CSV contain no `TopKDeviceOperation` and
are the authoritative compact outputs.

## Qualitative command and verdict

```bash
PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
python -m models.common.readiness_check.run_autoregressive \
  --model-dir /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b \
  --hf-model /home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --prompt-file /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_autoregressive_prompt.txt \
  --output-dir /home/ttuser/dev/qwen-perf/tt-metal/models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/autoregressive \
  --max-new-tokens 100 --mesh-device P300 \
  --fabric-config FABRIC_1D_RING --trace-region-size 1500000000
```

The prompt file's 161 token IDs exactly equal the fresh reference prompt. HF
produced 100 tokens exactly equal to the fresh reference. TT also produced 100
tokens, matched the first 40, then changed “coffee break” to “coffee shop
visit.” Its remaining visible text stays grammatical, English, non-repetitive,
and focused on identifying the 9 km distance and timing equations. There is no
wrong-language drift, topic loss, pathological repetition, or suspicious early
divergence; both files end mid-outline only because the inspection budget is
100 tokens. Verdict: coherent pass.

## Runtime audit and limitations

- `QwenFullModel` rejects every mesh except `1x4`; no fallback decoder exists.
- The full model instantiates only `MultichipDecoder` and retains the selected
  residual, dtype, fidelity, cache, CCL, and layout policies.
- Decode has no `ttnn.to_torch`, host argmax, or logits readback. Host sampling
  is explicit compatibility only. Autonomous feedback is sampler-to-token
  buffer on device; Python only reads output IDs.
- Readiness prefill returns host logits by contract. The first post-prefill
  token is selected at that boundary. Teacher forcing intentionally writes
  ground-truth tokens from its callback; neither behavior is included in the
  autonomous token-out latency.
- Cache/state allocations are stable. External page tables update only when
  changed. Reset uses `ttnn.fill(..., output_tensor=...)` rather than replacing
  tensors.
- Active decode slots are fixed contiguous prefix slots. Sparse arbitrary slot
  IDs are not advertised. The physical KV budget is 262,144 tokens total.
- vLLM integration is explicitly out of scope.

## Artifacts and finalization

- `evidence/final_validation.md`
- `evidence/token_out_ring_metrics.json`
- `evidence/teacher_forcing_ring_metrics.json`
- `evidence/prefill_ring_metrics.json`
- `evidence/full64_ring_argmax_metrics.junit.xml`
- `evidence/reduced_trace_ring_autofix_final.junit.xml`
- `evidence/max_context_watcher_ring_sampler.junit.xml`
- `evidence/run_autoregressive.log`
- `evidence/postflight_tt_smi.log` (all four P300c devices visible and reset-capable; no reset needed)
- `evidence/autoregressive/{hf_completion.txt,tt_completion.txt,autoregressive_meta.json}`
- `../../readiness_aime24_chat.refpt`
- `../context_contract.json`

Static closure uses `python -m json.tool`, `python -m compileall`,
`git diff --check`, and the static contract pytest. Independent stage-review
verdicts, remediations, and the final stage commit SHA are appended here during
closure. No push is performed.

The chronological `run_prefill_check.log`, `run_teacher_forcing.log`,
`token_out_64_layer.log`, and `reduced_trace_mixed.log` predate the final Ring
AutoFix. They are retained as earlier evidence, while `final_validation.md`
contains the post-fix correctness and performance results.

## 2026-08-20: max-context Watcher AutoFix

- AutoTriage first verified that warm trace setup advanced persistent token,
  position, rotary, and linear state before capture. The generator now restores
  those request-boundary tensors after warmup and after capture. Diagnostic
  `evidence/max_context_fixed.junit.xml` records both capture boundaries at the
  valid final input position 262143; its failure was only a stale one-boundary
  test expectation.
- Safe Watcher without `DUMP_ALL` refuted the polling hypothesis: Linear force
  sampling still stopped at all-gather writer line 119. Standard top-k failed
  its Linear gather at line 260, and explicit force axis 1 still failed at line
  119, refuting force-only and axis-selection hypotheses.
- The retained fix matches the exact hardware contract. The common force
  sampler has a model-scoped `allow_small_ring` opt-in; Qwen uses it for the
  physical 1x4 P300c ring, resolves `cluster_axis=None`, retains one-link Ring,
  and omits the Linear barrier. Generic small logical submeshes remain Linear.
- Final safe-Watcher max-context gate: 1 passed in 49.909 seconds. Evidence is
  `evidence/max_context_watcher_ring_sampler.junit.xml`, SHA-256
  `608e8af191bee7d3733e0902e8ba57c48e321301b65c83b8fc3161dab00eba5d`.
  `AUTOTRIAGE.md` and `AUTOFIX.md` contain the complete hypothesis ledger.

## 2026-08-20: final Ring sampler and state closure

- The production 64-layer benchmark passed in 472.85 seconds. Its exact split
  was model 42.141 ms, Ring force-argmax sampling 2.416 ms, combined 44.556
  ms/token, and 22.444 t/s/u. The semantically greedy common `Sampling1D`
  control selected the identical token 225721 but took 10.797 ms.
- `evidence/reduced_trace_ring_final.junit.xml` records the pre-review-2
  expanded hardware gate passing in 177.44 seconds; the post-AutoFix rerun is
  `evidence/reduced_trace_ring_autofix_final.junit.xml` (181.75 seconds). They
  cover alternating host/device sampling,
  trace teardown and recapture, deterministic reset, persistent tensor
  identity, zero reads for an unchanged page table and one read for a changed
  table, mixed non-aligned prompts `[65,67,65]`, inactive rows, duplicate-row
  determinism, explicit `start_pos=17`, and batch-32 fixed-slot/page isolation.
- The max-context test instruments both model and sampler trace-capture
  boundaries and observes position 262143 at both. The pre-capture restore is
  the same mechanism used for token, rotary, and recurrent linear state, so
  trace construction does not consume a user token or cross the context limit.
- A six-prompt shared chat-template qualitative suite was run for HF and TT at
  64 greedy tokens. Prefix agreement by prompt was 11, 55, 13, 43, 19, and 63
  tokens. Manual review passed every TT output for coherence, relevance,
  language, leakage, and repetition; the one flagged phrase in the reasoning
  control appears in both HF and TT. Evidence is under `evidence/qualitative/`.

  ```bash
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
    --backend hf --max-new-tokens 64
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
    --backend tt --max-new-tokens 64
  PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
  python models/autoports/qwen_qwen3_6_27b/tests/run_qualitative_suite.py \
    --backend check --max-new-tokens 64
  ```

## 2026-08-20: review-2 AutoFix and structured reruns

The second review's sampling-boundary finding was reproduced with CPU fake
logits: seed 0 with top-k 2 selected token 1 while greedy selected token 0.
The retained fix applies the requested policy from prefill token zero and
throughout host compatibility mode. Ten focused sampling/schema/CLI/static
tests passed together.

Final metric artifacts were generated by the executing paths themselves:

```bash
QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=64 \
QWEN36_TOKEN_OUT_METRICS_JSON=.../evidence/token_out_ring_metrics.json \
pytest -q -s .../tests/test_full_model.py::test_reduced_token_out_latency_breakdown \
  --junitxml=.../evidence/full64_ring_argmax_metrics.junit.xml

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir .../qwen_qwen3_6_27b --reference .../readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 \
  --output-json .../evidence/teacher_forcing_ring_metrics.json

python -m models.common.readiness_check.run_prefill_check \
  --model-dir .../qwen_qwen3_6_27b --reference .../readiness_aime24_chat.refpt \
  --mesh-device P300 --fabric-config FABRIC_1D_RING \
  --trace-region-size 1500000000 \
  --output-json .../evidence/prefill_ring_metrics.json
```

Structured results: token-out 42.141 ms model + 2.416 ms sampler =
44.556 ms/token (22.444 t/s/u), `Topology.Ring`, one link, 64 layers, and
host/device token 225721. Teacher forcing is 97/100 top-1, 100/100 top-5 and
top-100, TTFT 14,620.75 ms, traced decode 19.22 t/s/u, end-to-end 5.06 t/s/u.
Prefill is 97/100 top-1 and 100/100 top-5/top-100.

## 2026-08-20: review-3 stochastic trace closure

Review 3 identified missing device seed/history lifecycle and unsafe standard
top-k routing. The generator now installs prompt plus token-zero history,
resets each selected slot seed, advances it once per decoded token, restores
the real history after trace setup, and clears all request mirrors on reset.
Qwen opts into common-sampler seeded trace replay; generic seeded users keep
their existing direct path.

Four rejected hardware experiments isolated the topology and shape failures:
synchronous candidate Ring, exact asynchronous candidate Ring, and adjusted
candidate cadence stopped under Watcher; a proven full-logit Ring gather
completed but a single 262,144-wide top-k did not finish within five minutes.
The retained standard sampler gathers full logits with the already-proven Ring
protocol, runs four 65,536-wide top-k operations with persistent global index
chunks, locally concatenates 128 candidates, and invokes `ttnn.sampling`.

The final command was:

```bash
env TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
    TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
    PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
    TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1 \
    QWEN36_RUN_FULL_MODEL_SMOKE=1 \
    /home/ttuser/.tenstorrent-venv/bin/pytest -q -s \
    models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_full_model_prefill_decode_and_split_trace \
    --junitxml=models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/reduced_trace_ring_stochastic_final.junit.xml
```

Historical result: one passed in 180.47 seconds. The test covers seeded repeatability
across reset, temperature/top-k/top-p, presence/frequency/repetition penalties,
seed advancement, prompt/output history, unseeded stochastic capture/replay,
greedy restoration, token feedback, positions, page tables, and inactive rows.
A single allocator warning occurred after sampler replay when Python dispatched
penalty-history maintenance with traces resident. Stage-review 4 correctly kept
that lifecycle open; the following closure removes it.

## 2026-08-20: review-4 AutoFix and representative token-out

AutoFix moved penalty output-history maintenance into the captured sampler
operation and made compile warmup history-neutral. The warning-free safe-Watcher
rerun is `evidence/reduced_trace_ring_stochastic_autofix4.log` plus JUnit: one
passed in 179.70 seconds (176.91-second test call). Grep found no unsafe-buffer
allocation warning, Watcher error/fatal, failure, or failures section.

Low-level `decode_forward` now requires an explicit device-sampling request
start for new serving state. It accepts per-row prompt/output history, owns
active prefix slots, resets and advances their seeds exactly once per decoded
token, preserves inactive rows, rejects mid-request parameter changes, and
clears state on reset. The focused sampling/readiness suite passes 14/14.

The final full-64-layer benchmark retained the ten-iteration direct split for
attribution (42.137 ms model, 2.417 ms Ring sampler, 44.556 ms combined,
22.444 t/s/u) and added the public boundary: prompt 128, generate 128, 127
post-prefill autonomous decode intervals, sampled-ID readback included, no host
token feedback. Result: 49.220 ms/token, 20.317 t/s/u. Full log is
`evidence/token_out_ring_metrics_autofix4.log`; structured results remain in
`evidence/token_out_ring_metrics.json`.

Final postflight `tt-smi -s` sees four P300c boards with healthy DRAM and zero
uncorrectable GDDR errors. A separate `MeshShape([1,4])` open/close smoke prints
`mesh_open_ok ... devices=4` and `mesh_close_ok`; evidence is in
`evidence/postflight_tt_smi.log` and `evidence/postflight_mesh_open_close.log`.

## 2026-08-20: review-5 closure

Review 5 found that request reset cleared only active seed slots. AutoFix made
`SeedManager.reset_request_state()` clear all 32 seed/counter/RNG request
mirrors and made generator reset unconditional, without allocating new device
tensors. The common unit regression and the reduced hardware sequence seeded
batch 2, reset, then ran unseeded batch 1 for three tokens. The hardware gate
passed in 178.05 seconds with first request tokens `[26, 30, 318]`, identical
seeded replay after reset, and a clean unseeded request; no allocator or Watcher
warning was present. The focused reset/sampling suite passed 5/5.

The final matched representative workload is prompt 128 / generate 128. Warmed
TTFT is 742.651 ms. Across its 127 autonomous intervals, caller-visible decode
is 49.204778 ms/token (20.323229 t/s/u), including one sampled-ID readback and
no host token feedback. Direct device attribution is 42.140081 ms for the
64-layer model trace plus 2.416293 ms for the canonical Ring sampler, or
44.552574 ms/token (22.445392 t/s/u). Rejected `Sampling1D` is 10.798086 ms.

Steady-state greedy host counters are: model replay 1, sampler replay 1,
caller-visible sampled-ID readback 1 (device-only attribution 0), token refresh
0, current-position refresh 0, RoPE refresh 0, unchanged-page-table refresh 0,
mask rebuild 0, and explicit synchronization 0. A changed page table is copied
once at the request boundary. Unseeded stochastic mode performs two setup seed
copies and zero per-token copies; explicitly seeded stochastic mode retains the
common sampler's one seed-tensor copy per token. Token feedback, positions, and
page tables remain device-owned.

The selected compact profiler rerun uses real checkpoint layers 0 and 3, so it
contains both unique layer kinds plus final norm, TP LM head, and Ring argmax.
Its 195 merged rows sum to 4.361462 ms: one SDPA decode row, two paged-cache
updates, four AllReduce rows, 29 matmuls (1.221205 ms), argmax (1.418030 ms),
Ring all-gather (0.883663 ms), and no top-k. Artifacts use the
`evidence/profiler/two_kind_ring_` prefix. The optimized layer medians give the
independent lower bound `48*0.718857 + 16*0.476422 = 42.127888 ms`; the measured
42.140081-ms model trace is only 0.012193 ms (0.029%) above it. Combined latency
minus that bound is 2.424686 ms, matching the 2.416293-ms sampler.

## Local commits

- `5c658065f3d` — `Add Qwen3.6-27B TP4 full model`: stage implementation,
  tests, readiness helpers, reports, and sealed evidence package.
- `765ce8e58e6` — `Record Qwen3.6 full-model stage completion`: records the
  implementation SHA and independent `clean-pass` in the stage work log.

No commit was pushed. Unrelated Tracy, UMD, and cluster-descriptor worktree
changes were excluded.

## 2026-08-20: runner-side context-contract gate repair

The independent post-completion runner gate
`.agents/prompts/model_bringup_multigoal/06-full-model.check.sh` exited 2 after
the autoregressive degeneracy check passed. The reproduced critical diagnostic
was:

```text
models/autoports/qwen_qwen3_6_27b/doc/context_contract.json does not record the current supported context.
```

The capacity result itself was present and unchanged under
`full_model.maximum_logical_sequence_length=262144`, together with the
20,975,165,440-byte/device full-model plan and 13,384,572,928 bytes/device of
unreserved DRAM. The bug was that the stage document did not also expose the
runner schema's canonical top-level fields. Added
`hf_advertised_context=262144` and `current_supported_context=262144`; this is
a schema repair only, with no capability reduction or runtime-policy change.

Reverification command:

```bash
MODEL_DIR=models/autoports/qwen_qwen3_6_27b \
HF_MODEL=Qwen/Qwen3.6-27B \
.agents/prompts/model_bringup_multigoal/06-full-model.check.sh
```

Final result: exit 0. The autoregressive artifact remained non-degenerate
(`num_tokens=63`, adjacent duplication `0.0`, trigram-loop fraction `0.0476`),
and the context checker reported:

```text
Context contract OK for models/autoports/qwen_qwen3_6_27b: target=262144, supported=262144 (full HF context).
```

The context checker also exits 0 when invoked with only `--model-dir` and when
invoked with only `--hf-model`, proving that the repaired top-level contract is
self-describing rather than relying on the locally cached HF configuration.

Fresh independent `$stage-review` task `/root/full_model_gate_review` returned
`clean-pass` with no required work, other concerns, or hard-check gaps. The
reviewer independently inspected the original goal and failure log, checker,
model/generator context bounds, capacity evidence, generated outputs, and
sealed artifact manifest; it reran the full runner and all three context-check
argument modes successfully. Its anomaly ledger classified the failure as an
evidence-schema compatibility bug, fixed by the canonical top-level fields.

Local repair commit: `8a742b6219f` (`Fix Qwen3.6 full-model context contract
gate`). No commit was pushed; unrelated Tracy, UMD, and cluster-descriptor
worktree changes remain excluded.
