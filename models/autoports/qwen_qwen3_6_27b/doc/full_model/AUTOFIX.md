# AutoFix Report

## Starting Evidence

- Source diagnosis: `AUTOTRIAGE.md` in this directory.
- Original failure: `test_reduced_full_model_max_context_decode` under Watcher stopped during split-trace setup. Initial evidence exposed an un-restored max-position warmup; after that repair, eager common sampling still stopped in the asynchronous all-gather writer.
- Target contract: four Blackhole P300c devices, `MeshShape([1,4])`, `FABRIC_1D_RING`, TP4, common on-device force-argmax split sampling.

## Hypothesis Experiments

### Warmup advances capture beyond the context

- Hypothesis: warm decode mutates persistent token/position/rotary/linear state, so capture runs a different logical step and position 262144 at the context boundary.
- Experiment: snapshot and restore all mutable request state before capture; instrument both `begin_trace_capture` boundaries.
- Result: both boundaries observe 262143. `evidence/max_context_fixed.junit.xml` records the diagnostic sequence `[262143, 262143]`; its failure is only the then-stale expected list `[262143]`.
- Verdict: **verified and fixed**.
- Fix: `restore_capture_inputs()` runs after eager warmup synchronization and after capture.
- Verification: final max-context Watcher gate below exercises the corrected boundaries.

### Unsafe full Watcher dumps cause the CCL stop

- Hypothesis: `TT_METAL_WATCHER_DUMP_ALL=1` perturbs active CCL kernels.
- Experiment: rerun with safe Watcher and no `DUMP_ALL`.
- Result: force sampling still stopped at writer line 119.
- Verdict: **refuted**.
- Fix: none.

### Force argmax or implicit CCL axis is invalid

- Hypothesis: the force-only path or its implicit axis causes the stop.
- Experiments: run standard top-k; force axis 1 explicitly.
- Results: standard top-k also stopped in its Linear gather at writer line 260; explicit-axis force still stopped at line 119.
- Verdict: **refuted**.
- Fix: removed the diagnostic axis override; no sampler-algorithm change retained.

### Generic Linear route mismatches the physical P300c ring

- Hypothesis: the common sampler's unconditional `<8 devices => Linear` rule describes logical subgroups but conflicts with this physical 1x4 Ring; Ring must use its own completion protocol without the Linear barrier.
- Experiment: retain common force argmax, logits contract, one link, `chunks_per_sync=10`, one worker/link, and two buffers; opt this model into small physical Ring, resolve `cluster_axis=None`, and omit `barrier_semaphore` for Ring. Run the original final-position split-trace test under safe Watcher.
- Result: runtime reported `cluster_axis=None topology=Ring`; one test passed in 49.909 seconds.
- Verdict: **verified and fixed**.
- Evidence: `evidence/max_context_watcher_ring_sampler.junit.xml` (SHA-256 `608e8af191bee7d3733e0902e8ba57c48e321301b65c83b8fc3161dab00eba5d`).
- Fix: model-scoped `allow_small_ring`; generic small-submesh Linear behavior remains default; Ring gather omits the Linear barrier.
- Verification command:

  ```bash
  env TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
      TT_METAL_RUNTIME_ROOT=/home/ttuser/.local/lib/model-bringup/tt-metal \
      PYTHONPATH=/home/ttuser/dev/qwen-perf/tt-metal \
      TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1 \
      QWEN36_RUN_CONTEXT_WRAPPER=1 \
      /home/ttuser/.tenstorrent-venv/bin/pytest -q -s \
      models/autoports/qwen_qwen3_6_27b/tests/test_full_model.py::test_reduced_full_model_max_context_decode \
      --junitxml=models/autoports/qwen_qwen3_6_27b/doc/full_model/evidence/max_context_watcher_ring_sampler.junit.xml
  ```

## Final Status

- **Fixed.** Trace setup restores all request-boundary mutable state, and the selected common force sampler now follows the exact physical-ring topology contract.
- Final max-context safe-Watcher result: 1 passed in 49.91 seconds.
- The production path remains semantically greedy, device-resident, common-sampler based, and split traced. No host, replicated, single-chip, or custom-sampler fallback was introduced.
- Remaining risk: the standard top-k Linear route still has failed Watcher evidence and remains rejected for this model; it is not the selected optimized token-out path.

## Stage-review AutoFix closure

The second independent review found three additional closure defects.

### Sampling parameters bypassed at token zero

- Hypothesis: only later device sampling honored `SamplingParams`; prefill and
  host compatibility hard-coded argmax.
- Isolation: CPU fake logits with top-k 2 and seed 0 select token 1, while
  greedy would select token 0.
- Result: verified.
- Fix: request-scoped host sampling now implements temperature, top-k, top-p,
  presence/frequency/repetition penalties, and seed. It governs the prefill
  token in both modes and every token in explicit host compatibility mode.
  Optimized greedy decode after prefill remains common-sampler split traced.
- Verification: focused stochastic boundary, deterministic sequence, top-p,
  host-feedback, and static tests pass.

### Final metrics were prose-only

- Hypothesis: the Ring JUnit proves pass status but cannot recover stdout
  metrics, while chronological logs describe the earlier top-k revision.
- Result: verified.
- Fix: readiness prefill/teacher runners accept `--output-json`; the token-out
  A/B accepts `QWEN36_TOKEN_OUT_METRICS_JSON`. Reports contain per-entry and
  aggregate accuracy/performance plus runtime, topology, layer/iteration, and
  selected-token metadata.
- Verification: nonhardware schema/CLI tests pass; final hardware JSON
  artifacts are listed in `artifact_manifest.sha256` after reruns.

### Compact profiler covered rejected Linear topology

- Hypothesis: the old compact profile cannot establish the selected Ring
  terminal graph.
- Result: verified.
- Fix: captured the same real one-layer terminal stack with selected one-link
  Ring force argmax and generated signpost-bounded compact artifacts.
- Verification: 145 merged operations, 3,962.61 us summed device time; argmax
  1,417.43 us, Ring all-gather 883.30 us, width-sharded matmuls 988.52 us, and
  no top-k. The final 64-layer direct split places sampling at 5.4%.

## Stage-review 3 AutoFix closure

The third review found that device stochastic requests did not initialize or
advance seeds and penalty history, and that standard top-k still selected the
known-unsafe small-mesh Linear candidate collective.

### Request-scoped stochastic state

- Fix: device request setup now resets the selected slot's seed, installs
  prompt tokens plus the prefill-selected output in the common sampler, and
  advances the seed once per device-decoded token. Trace warmup/capture restores
  the real history, and reset clears the model-owned mirrors.
- Fix: this model opts into common-sampler seeded trace replay. A persistent
  seed tensor is copied before replay, while generic seeded users retain the
  existing direct execution contract.
- Evidence: the expanded reduced hardware test uses `top_k=16`, `top_p=0.9`,
  `temperature=0.8`, all three penalty classes, and seed 17. Two runs separated
  by reset reproduce the same tokens, seed count, prompt mask, and history.

### Safe stochastic Ring topology

Five isolated alternatives were tested under Watcher:

1. A synchronous Ring candidate gather stopped in its writer.
2. The exact proven asynchronous Ring protocol still stopped when gathering
   candidate payloads.
3. Payload-derived cadence did not resolve that stop.
4. A proven full-logit Ring gather completed, but one 262,144-wide top-k did
   not terminate within five minutes.
5. The retained path performs the proven full-logit Ring gather, four
   65,536-wide top-k operations with persistent global index chunks, and a
   local 128-candidate concat; the complete seeded/penalized trace gate passed.

Historical evidence is `evidence/reduced_trace_ring_stochastic_final.junit.xml`:
one test passed in 180.47 seconds under safe Watcher, but the console exposed a
trace-resident allocator warning. Stage-review 4 therefore kept this item open.

## Stage-review 4 AutoFix closure

### Trace-resident penalty allocation

- Hypothesis: sampling replay returned to Python and then dispatched
  `update_output_tokens`; its scatter/count temporaries were allocated while
  both model and sampler traces were resident.
- Isolation: a fake trace regression marks replay as trace-active and rejects
  any Python-side penalty update. The old placement fails that contract.
- Fix: penalty application, token sampling, and output-history update are one
  `_run_sampling` sequence captured in the sampler trace. Compile warmup skips
  history mutation, and replay performs no allocating Python-side update.
- Result: `evidence/reduced_trace_ring_stochastic_autofix4.log` contains no
  unsafe-allocation warning, Watcher error/fatal, or pytest failure. The paired
  JUnit records one pass in 179.70 seconds under safe Watcher.

### Low-level stochastic request ownership

- Fix: `decode_forward` exposes an explicit request-start boundary with
  sampling parameters, per-row prompt history, optional output history, active
  prefix slots, seed reset, and exactly-once advancement. Inactive rows remain
  untouched; mid-request parameter replacement is rejected; reset clears all
  state.
- Verification: mixed two-row active state with remaining inactive rows,
  different seeds and penalties, repeated decode, reset, and deterministic
  restart pass in the focused CPU suite. The high-level seeded/penalized path
  passes the hardware gate above.

### Representative token-out boundary

- Fix: a non-mutating `token_observer` records caller-visible outputs without
  triggering teacher-forcing feedback. The full-64-layer benchmark now runs
  prompt 128 / generate 128 and measures the 127 post-prefill autonomous decode
  intervals including sampled-ID readback.
- Result: 49.220 ms/token, 20.317 t/s/u. The 44.556 ms/token, 22.444 t/s/u
  direct trace pair is retained only as device-side attribution.

## Stage-review 5 AutoFix closure

### Complete seed-request reset

- Hypothesis: reset cleared seed mirrors only for the previously active prefix,
  so an inactive seeded slot could leak into a later smaller unseeded request.
- Isolation: seed batch 2, reset, then start unseeded batch 1 and decode three
  tokens. The prior implementation retained stale slot-1 seed state.
- Fix: `SeedManager.reset_request_state()` clears every one of the 32 seed,
  counter, and RNG request mirrors without replacing persistent device tensors;
  generator reset invokes it even before full-model state exists.
- Result: the common reset regression plus the reduced hardware gate pass. The
  final safe-Watcher run passes in 178.05 seconds with no trace-resident
  allocation or Watcher warning.

### Matched boundary and two-kind profile

- The representative prompt-128/generate-128 run now records its warmed TTFT,
  742.651 ms, beside its 49.204778-ms caller-visible decode interval.
- The selected compact profiler explicitly instantiates checkpoint layers 0
  and 3, covering linear and full attention. It confirms the selected Ring
  sampler, one SDPA decode, both cache paths, CCL rows, and no top-k.
- The optimized per-kind medians predict 42.127888 ms for all 64 layers; the
  measured model trace is 42.140081 ms, a 0.029% delta.
