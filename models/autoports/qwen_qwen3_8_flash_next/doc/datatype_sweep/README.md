# Qwen3.8-Flash-Next datatype sweep

## Result

The selected default is `qsa_bfp8_hifi2_lm_head_bf16_hifi2`. It is the fastest evaluated full-model configuration that passes the accuracy, traced-decode, source-provenance, and runtime precision-propagation gates.

The gate is top-1 >= 90%, top-5 >= 98%, and top-100 = 100% on the AIME24 chat-template readiness reference (201 prompt tokens and 99 scored teacher-forcing rows). Pareto ranking uses only 98 verified model-only trace replays. Eager, untraced, and autoregressive token-out numbers are not used to select a dtype.

The selected current-source median is 4.384021 traced teacher-forcing tokens/s/user at 91.919% top-1, 100% top-5, and 100% top-100. The optimized baseline median is 4.306269 tokens/s/user at the same accuracy, so the selected median is 1.806% faster. The final normal-construction selected refresh measured 4.386243 tokens/s/user, 227.986 ms/token, and 12.744 s TTFT with the same accuracy and all 61 policy leaves consumed.

## Selected policy

- Routed experts: BFP4 TILE weights with LoFi compute; exact BF16 checkpoint source is prepacked to BFP4 TILE host entries, staged as BFP4 TILE, and executed as BFP4. All 512 experts/layer are prepacked; each layer/rank has 10 exact device slots and one staging allocation.
- Shared projections: BFP8 with LoFi. GDN projections, QSA input/output, attention output, and the final hyperconnection down/up weights: BFP8 with HiFi2.
- LM head: BF16 with HiFi2. Embeddings, residuals, general matmul outputs, PLE activations, logits, and sampling are BF16. Norms are BF16 with HiFi4; router/top-k outputs are BF16.
- CCL: BF16 payload, linear topology, two links, 8192-byte packets.
- KV cache: BFP8 TILE, 64-token page blocks, BF16 updates.
- PLE: BF16 row-major mmap table, BF16 host assembly, BF16 TILE staging and execution, 8192-row host cache, and 128-row prefill staging chunks.
- Sampling: device full-vocabulary argmax with BF16 logits and BF16 sampling input.

The normal `Qwen38FullModel` and `Qwen38Generator` constructors load `selected_precision_config.json` when no override is supplied. Runtime evidence observes the live decoder tensors, compute fidelities, residual and CCL boundaries, live KV tensors and updates, endpoint tensors, sampler mode, fixed expert cache/staging objects, and PLE store/staging objects. The selected teacher-forcing and token-out reports both prove `all_fields_consumed=true` with 61/61 leaves. `host_weight_contract.json` separately records the selected expert and PLE source/packed/staging/execution representations and the successful upload-preservation checks.

## Baseline refresh

The optimized baseline was refreshed with the main AIME24 chat-template readiness reference. Its primary traced teacher-forcing sample measured 91.919%/100%/100% top-1/top-5/top-100, 12.294 s TTFT, 232.707 ms/token, and 4.297257 tokens/s/user. A same-source repeat measured 4.315280 tokens/s/user, giving the 4.306269 median used in the sweep.

The required 100-generated-token autoregressive refresh is embedded in `full_runs/baseline_optimized_bfp4lofi_bfp8hifi2/candidate_result.json`: 9.625 s TTFT, 221.048 ms/token, 4.523909 tokens/s/user, traced device sampling, and a non-degenerate TT completion. This autoregressive result is readiness evidence only, not a Pareto input.

## Full-model sweep

| Config | Top-1 | Top-5 | Traced TF t/s/u | Samples | Gate | Decision |
|---|---:|---:|---:|---:|---|---|
| `qsa_bfp8_hifi2_lm_head_bf16_hifi2` | 91.919 | 100.000 | 4.384021 | 3 | pass | selected |
| `qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2` | 91.919 | 100.000 | 4.366890 | 2 | pass | rejected: slower |
| `shared_bfp4_hifi2` | 84.848 | 98.990 | 4.365348 | 1 | fail | rejected: top-1 |
| `canonical_accuracy_bf16cache` | 91.919 | 100.000 | 4.360255 | 2 | pass | rejected: slower |
| `qsa_bfp8_hifi2` | 91.919 | 100.000 | 4.327310 | 2 | pass | rejected: slower |
| `shared_bfp8_hifi2` | 90.909 | 100.000 | 4.315543 | 1 | pass | rejected: slower |
| `lm_head_bfp8_lofi` | 91.919 | 100.000 | 4.311844 | 1 | pass | rejected: slower |
| `baseline_optimized_bfp4lofi_bfp8hifi2` | 91.919 | 100.000 | 4.306269 | 2 | pass | rejected: slower |
| `ccl_bfp8` | 92.929 | 100.000 | 4.284524 | 1 | pass | rejected: slower |
| `expert_bfp4_hifi2` | 91.919 | 100.000 | 4.166757 | 1 | pass | rejected: slower |
| `lm_head_bf16_hifi2` | 91.919 | 100.000 | 4.014166 | 1 | pass | rejected: slower |
| `residual_bfp8` | 91.919 | 100.000 | 3.967161 | 1 | pass | rejected: slower |
| `kv_bf16_control` | 91.919 | 100.000 | 3.951342 | 1 | pass | rejected: slower |
| `gdn_bfp8_lofi` | 91.919 | 100.000 | 3.862568 | 1 | pass | rejected: slower |
| `shared_bfp4_lofi` | 84.848 | 98.990 | 3.677590 | 1 | fail | rejected: top-1 |
| `qsa_bfp8_lofi` | 91.919 | 100.000 | 3.194514 | 1 | pass | rejected: slower |

Both material BFP4 groups were paired as required. Routed experts have the baseline BFP4+LoFi policy and the `expert_bfp4_hifi2` control. Shared projections have both `shared_bfp4_lofi` and `shared_bfp4_hifi2`; both failed top-1 at 84.848%. No TTNN/runtime blocker or AutoFix substitute was needed because all BFP4+LoFi pair candidates executed successfully.

The top-1 Pareto frontier contains the selected point and `ccl_bfp8`: the latter has higher top-1 but lower decode throughput. The selected point is the top-5 frontier because it has 100% top-5 and the highest valid throughput. The PNGs plot all 16 full-model candidates, draw these frontiers, mark the selected point in red, and draw the vertical minimum-accuracy line.

Independent review challenged the one-sample rows with elevated expert H2D time. Current-source controls were collected for GDN LoFi, BF16 KV, BF16 LM head, QSA+LM, QSA LoFi, and residual BFP8. The first QSA+LM control recovered to 4.376718 tokens/s/user, so it received a second control and final normal-construction sample; their median is the selected 4.384021. The previously selected HiFi2 shared-projection policy received two equal-source controls and medianed to 4.366890. All challenged controls keep the same fixed workload and record exact expert misses, H2D bytes/time, and real PLE rows.

## Fixed host-backed measurement regime

Every ranking run used P300 Blackhole dies 0 and 1 as a 1x2 `FABRIC_1D` mesh, batch 1, the same 201-token AIME24 prompt/reference continuation, 99 teacher-forcing rows, and 98 measured trace replays. The expert policy, top-10 routing width, 10-slot device capacity, all-512 prepack, staging depth, construction-time cold state, PLE table/cache/staging policy, and token workload were fixed. Candidate numerics determine the realized expert ids; the reports preserve realized routes through exact miss/hit, H2D byte/time, and PLE row/lookup counters, so the full cost is included rather than normalized away.

The baseline primary sample recorded 49,959 exact expert misses, 138,126,643,200 expert H2D bytes, 16.733 s expert H2D submission time, 102 PLE lookups, and 3,480 real PLE table rows. The final selected validation recorded 50,069 misses, 138,430,771,200 H2D bytes, 16.805 s, 102 PLE lookups, and the same 3,480 real rows. For the displaced HiFi2 shared-projection policy, a preserved same-source transient and normal control have identical 50,059 misses and 138,403,123,200 bytes but 47.826 versus 16.603 s H2D time; the slow 3.938815 tokens/s/user sample is retained as host-timing variance and is not a Pareto input.

## Context and non-aligned support

The BFP8 and BF16 KV candidates were each reconstructed through all 48 layers at the advertised 262,144-token capacity. BFP8 charges 2,340,421,632 cache bytes/device and leaves 24,220,969,896 bytes/device headroom. BF16 charges 4,227,858,432 bytes/device and leaves 22,333,533,096 bytes/device. Both fit physical DRAM, so `context_contract.json` continues to advertise 262,144 tokens with no capability reduction.

The selected BFP8 path also passed a 129-token non-aligned prompt with three traced tokens. The BF16 KV control ran the same reduced non-aligned check. Cache dtype/layout and trace-buffer changes therefore preserve non-aligned prompt handling.

## Post-selection token-out

The normal selected-config construction path was rerun with the optimized-full-model warmed batch-1 prompt-128/generate-128 no-readback workload. It measured 7.787 s TTFT, 231.594 ms/token, and 4.317901 tokens/s/user across 126 measured traced replays using device sampling. This is recorded separately from teacher forcing and is the token-out number later serving/vLLM comparisons must consume.

The run revalidated 53,543 exact expert misses, 148,035,686,400 expert H2D bytes, 17.842 s expert H2D submission time, 129 PLE lookups, 2,512 real PLE table rows, 1.849 s PLE lookup time, and 2,621,440 PLE staging H2D bytes. All prohibited host-work flags are false. No vLLM integration was started.

## Commands

All hardware commands use:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

Candidate template:

```bash
RUN_QWEN38_DATATYPE_SWEEP=1 \
QWEN38_PRECISION_CONFIG=models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/candidates/<config-id>.json \
QWEN38_EVIDENCE_DIR=models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/full_runs/<config-id> \
timeout 2400 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_datatype_sweep_candidate
```

Post-selection token-out uses the default selected config, with no precision override:

```bash
RUN_QWEN38_PERF=1 \
QWEN38_EVIDENCE_DIR=models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/post_selection/token_out \
timeout 1800 pytest -q -s --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py::test_full_model_batch1_prompt128_generate128_performance
```

Ledger and capacity regeneration:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py --check
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/recompute_context_contract.py
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/update_host_contract.py
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/finalize_sweep.py
```

## Artifacts and limitations

- Machine-readable results: `sweep_results.json`, `sweep_results.csv`, immutable `baseline_precision_config.json`, `candidate_matrix.json`, `candidates/*.json`, and each candidate's `full_runs/<config-id>/candidate_result.json`. `make_candidates.py --check` proves all 17 generated matrix artifacts reproduce exactly from the immutable baseline seed; finalization also rejects any candidate/result policy mismatch.
- Consumed default: `selected_precision_config.json`; runtime proof: `post_selection/precision_smoke/precision_propagation_smoke.json` and `post_selection/teacher_forcing_selected/candidate_result.json`.
- Plots: `top1_perf_pareto.png` and `top5_perf_pareto.png`.
- Context: `context_contract_candidates/kv_bfp8.json`, `context_contract_candidates/kv_bf16.json`, both full construction reports under `full_runs/context_*`, and `../context_contract.json`.
- Host policy: `../host_weight_contract.json` and post-selection teacher/token-out host totals.
- Qualitative review: `post_selection/qualitative/qualitative_shared_suite_final.json` and `qualitative_review.md`.
- Detailed chronology, commands, source variance, test evidence, review verdict, and commit SHAs: `work_log.md`.

The isolated BF16/HiFi2 LM-head candidate's current-source control recovered from 2.704361 to 4.014166 tokens/s/user but remained slower than the finalists; the sweep therefore uses measured full configurations and makes no additive per-group speed assumption. The fixed 128-token qualitative reference truncates some HF controls and one TT coding response before a final answer; this is documented but does not replace or weaken the full-model top-1/top-5 gate. Host H2D submission time remains a measurable source of wall-clock variance, as preserved in the displaced-policy transient and all anomaly-control counters. Unqualified top-level post-selection XML/log aliases now copy the final selected runs; displaced-policy runs live only under `anomaly_controls/qsa_bfp8_hifi2_shared_bfp8_hifi2_lm_head_bf16_hifi2/` with historical/control labels.
