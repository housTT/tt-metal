# Final validation on selected Ring sampler

Date: 2026-08-20 EDT

Runtime: installed Blackhole runtime at
`/home/ttuser/.local/lib/model-bringup/tt-metal`, repo source on `PYTHONPATH`,
four P300c devices, `MeshShape([1,4])`, `FABRIC_1D_RING`.

## Full 64-layer prefill

Command: `python -m models.common.readiness_check.run_prefill_check` with the
repo-local model and fresh `readiness_aime24_chat.refpt` reference.

- top-1: 97/100
- top-5: 100/100
- top-100: 100/100
- exit status: 0

The final post-AutoFix rerun writes the complete per-entry and aggregate record
plus runtime metadata to `prefill_ring_metrics.json`.

## Full 64-layer token-out split

Command: `QWEN36_RUN_TOKEN_OUT_BENCHMARK=1 QWEN36_BENCH_LAYERS=64 pytest
-q -s ...::test_reduced_token_out_latency_breakdown`.

- warmed TTFT for the same prompt-128/generate-128 workload: 742.651 ms
- model trace: 42.140081 ms
- selected common Ring force-argmax trace: 2.416293 ms
- device-only combined trace pair: 44.552574 ms/token, 22.445392 t/s/u
- caller-visible prompt-128/generate-128: 49.204778 ms/token, 20.323229 t/s/u
- rejected common `Sampling1D` greedy trace: 10.798086 ms
- both common paths selected token 225721
- exit status: 0

The selected compact Ring profiler instantiates checkpoint layers 0 and 3 and
therefore covers both linear- and full-attention decoder kinds. It reports
4.361462 ms summed device time across 195 merged rows: argmax is 1.418030 ms
(32.51%), Ring async all-gather is 0.883663 ms (20.26%), and 29 matmuls total
1.221205 ms (28.00%). It includes one SDPA decode, two paged-cache updates,
four AllReduce rows, and no top-k. Artifacts use the `two_kind_ring_` prefix.

The optimized decoder medians independently predict
`48*0.718857 + 16*0.476422 = 42.127888 ms` for the 64-layer stack. The measured
model trace is 0.012193 ms (0.029%) above that lower bound. Combined token-out
minus the bound is 2.424686 ms, consistent with the 2.416293-ms sampler trace.

Exact timing, topology, layer count, iterations, representative workload shape,
readback boundary, and host/device selected-token equality are preserved in
`token_out_ring_metrics.json`; pass status and wall duration are in
`full64_ring_argmax_metrics.junit.xml`. The representative interval contains
127 autonomous device-feedback decode steps after the prefill token and
includes each caller-visible sampled-ID readback.

Steady-state greedy host-work counters are one model replay, one sampler replay,
one caller-visible sampled-ID readback (zero in device-only attribution), zero
token/position/RoPE refreshes, zero unchanged-page-table refreshes, zero mask
rebuilds, and zero explicit synchronizations. A changed page table copies once
at the request boundary. Unseeded stochastic requests make two seed copies at
setup and none per token. Explicitly seeded stochastic requests retain the
common sampler contract's one seed-tensor copy per token; token feedback,
positions, and page tables stay on device.

## Full 64-layer traced teacher forcing

Command: `python -m models.common.readiness_check.run_teacher_forcing` with
the same model/reference, `--trace-region-size 1500000000`, and Ring fabric.

- top-1: 97/100
- top-5: 100/100
- top-100: 100/100
- TTFT: 14,620.75 ms
- traced teacher-forcing decode: 19.22 t/s/u
- end-to-end: 5.06 t/s/u
- sampler trace logged `cluster_axis=None`, one link, `Topology.Ring`
- exit status: 0

The complete per-entry and aggregate record, including elapsed seconds and
runtime metadata, is `teacher_forcing_ring_metrics.json`.

## Static closure

- `python -m compileall`: pass for model, generator, optimized decoder, tests,
  readiness helpers, and common sampling change
- `python -m json.tool doc/context_contract.json`: pass
- `git diff --check`: pass
- `pytest ...::test_full_model_static_contracts`: 1 passed in 1.47 seconds
- integrated sampling/schema/CLI/static AutoFix suite: 10 passed in 2.55 seconds
- post-AutoFix expanded Ring trace/state gate: 1 passed in 181.75 seconds
- warning-free seeded/penalized stochastic Ring trace gate under safe Watcher:
  1 passed in 178.05 seconds (`reduced_trace_ring_stochastic_autofix4.junit.xml`)
- final focused sampling, model-contract, reset, and readiness suite: 19 passed
  in 3.75 seconds

The stochastic gate proves reset-stable seed reproducibility, once-per-token
seed advancement, prompt/output penalty history, unseeded stochastic trace
capture/replay, greedy restoration, and the selected full-logit Ring plus four
65,536-wide top-k contract. Penalty history now updates inside the captured
sampler operation, so its preserved console contains no trace-resident allocator
warning. Final `tt-smi -s` and explicit 1x4 mesh open/close also pass.

The post-review reset regression additionally proves that seeded state in an
inactive slot cannot survive reset into a later smaller unseeded request. The
focused common/model reset and sampling suite passes 5/5.

## Fresh post-fix shared qualitative suite

The six TT controls were regenerated with 64 greedy tokens on the selected
Ring sampler, then compared to the existing exact-checkpoint HF controls.

- automatic verdict: pass
- HF/TT matching prefixes: 11, 55, 13, 43, 19, 63 tokens
- all TT completions: 64 tokens
- mechanical failures: 0/6
- manual reread: all six are coherent, relevant, in the expected language,
  and free of prompt leakage or pathological repetition
- the adjacent `thinking thinking` in prompt 2 remains present in both HF and
  TT controls and is not a TT regression
