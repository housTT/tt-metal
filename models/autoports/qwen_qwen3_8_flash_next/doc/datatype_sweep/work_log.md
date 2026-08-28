# Datatype-sweep work log

## Scope and starting point

- Model: `Qwen/Qwen3.8-Flash-Next`, checkpoint revision `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Starting branch/HEAD: `hous/qwen3.8-flash-next` at `2ee658a6a9760215ea5357b91c801c14638ff1c1`.
- Input stage: completed optimized full model. This stage created only datatype-sweep/default-policy plumbing and evidence; no vLLM integration was started.
- Hardware: P300 Blackhole board, dies 0 and 1, 1x2 `FABRIC_1D`, 2-link linear collectives, 8192-byte packets. Runs were serialized. `tt-smi -ls --local` remained healthy before/after long runs; no reset or recovery was required.
- Accuracy thresholds: top-1 >= 90%, top-5 >= 98%, top-100 = 100%.
- Selection metric: median traced teacher-forcing tokens/s/user, requiring 98 model-only trace replays and complete precision propagation. A one-sample candidate uses that sample as its median. Review-challenged host-timing rows use their current-source anomaly-control cohort rather than the historical primary sample.

## Implementation

1. Added `tt/precision_config.py` to validate and load the repo-local selected default, with `QWEN38_PRECISION_CONFIG` as an explicit candidate override.
2. Threaded weight groups, compute fidelities, activation/residual, CCL, KV-cache/update, logits/sampling, expert host representations, and PLE table/assembly/staging/execution fields through normal full-model construction.
3. Added a live 61-leaf propagation report. It inspects decoder weight tensors and fidelities, norms/router boundaries, residual and CCL wrappers, live QSA cache tensors and update policy, embedding/final norm/LM-head tensors, generator sampler defaults, fixed expert host/cache/staging objects, and PLE mmap/cache/staging objects.
4. Added candidate, selected teacher-forcing, selected token-out, qualitative, precision-smoke, and advertised-context evidence writers without changing vLLM code.
5. Added deterministic candidate generation from immutable `baseline_precision_config.json`, context capacity recomputation, host-contract projection, ledger/CSV generation, and pyplot Pareto generation scripts. `make_candidates.py --check` verifies all 17 matrix artifacts byte-for-byte without writing; finalization independently rejects any candidate JSON whose config ID or 61 expected policy leaves differ from a retained result.

## Sweep execution

Generated 16 policies with:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/make_candidates.py
```

Every reduced candidate first passed a real-weight non-aligned 129-token construction/trace smoke. Full candidates then used the command template recorded in `README.md` and embedded per row in `sweep_results.json`/`.csv`. The fixed regime was batch 1, the same 201-token AIME24 chat prompt/reference continuation, 99 scored rows, 98 trace replays, all 512 exact experts/layer prepacked, 10 device slots/layer/rank, one BFP4 TILE staging object, BF16 mmap PLE with an 8192-row cache and 128-row staging chunks, and the same construction-time cold state.

The first canonical result was generated before test-harness provenance was added. It is preserved as `full_runs/canonical_accuracy_bf16cache/candidate_result_initial_dormant_test_digest.json` and its XML; the canonical primary was rerun on the common sweep source. No unsourced result participates in selection.

Initial finalist repeats were collected for the baseline, canonical policy, QSA BFP8/HiFi2, and the initial QSA+shared-HiFi2+LM-head combination. Fourteen configs passed; both shared-projection BFP4 variants failed top-1 at 84.848%. Every material BFP4 group had a runnable LoFi pair, so AutoFix was not invoked for a missing BFP4+LoFi candidate.

The first independent review challenged the one-sample passing rows whose low throughput coincided with elevated expert H2D time. Current-source controls were collected serially for `gdn_bfp8_lofi`, `kv_bf16_control`, `lm_head_bf16_hifi2`, `qsa_bfp8_hifi2_lm_head_bf16_hifi2`, `qsa_bfp8_lofi`, and `residual_bfp8`. GDN, KV, QSA LoFi, and residual controls reproduced slower transfer/service classes; the LM-head control recovered to 4.014166 tokens/s/user but stayed slower. QSA+LM recovered to 4.376718 and therefore received a second control at 4.384021. Equal-current-source controls for the initial shared-HiFi2 finalist measured 4.388817 and 4.344964, median 4.366890. A final normal-construction QSA+LM sample measured 4.386243, making its three-sample current-source median 4.384021. That faster passing policy, `qsa_bfp8_hifi2_lm_head_bf16_hifi2`, became the consumed default.

## Baseline and post-selection refreshes

- Baseline teacher-forcing primary: 91.919%/100%/100%, 12.294 s TTFT, 232.707 ms/token, 4.297257 tokens/s/user. Repeat: 4.315280; median: 4.306269.
- Baseline 100-token AIME24 autoregressive: 9.625 s TTFT, 221.048 ms/token, 4.523909 tokens/s/user; traced device sampling; non-degenerate.
- Final selected normal-construction teacher forcing: 91.919%/100%/100%, 12.744 s TTFT, 227.986 ms/token, 4.386243 tokens/s/user, 98/98 trace replays, 61/61 policy leaves.
- Final selected normal-construction token-out: batch 1 prompt128/generate128, 7.787 s TTFT, 231.594 ms/token, 4.317901 tokens/s/user, 126 traced measured tokens, device sampling, no activation/logit readback, 61/61 leaves.

One current-source run of the displaced shared-HiFi2 policy passed correctness but measured 3.938815 tokens/s/user because exact expert H2D submission accumulated 47.826 s. Its exact miss count (50,059), H2D bytes (138,403,123,200), accuracy, routes/workload, and PLE real rows (3,480) match the preserved normal 4.388817 control with 16.603 s H2D. The slow report/log/XML retain the `_transient_h2d` suffix and remain variance evidence, not a Pareto input.

An earlier current-compute baseline token-out control measured 3.459951 tokens/s/user and is preserved under `post_selection/token_out_baseline_control`. It was diagnostic only and is not a ranking input. Later serving comparisons must use the selected post-selection 4.317901 token-out result, not teacher forcing or this control.

## Host and context evidence

- Selected teacher forcing: 50,069 exact expert misses, 138,430,771,200 H2D bytes, 16.805 s H2D, 102 PLE lookups, 3,480 real table rows.
- Selected token-out: 53,543 exact misses, 148,035,686,400 H2D bytes, 17.842 s H2D, 129 PLE lookups, 2,512 real rows, 1.849 s lookup, 2,621,440 PLE staging bytes. Every prohibited host-work flag is false.
- BFP8 advertised-context construction: all 48 layers, 36 QSA cache tensors, 262,144 tokens, 2,340,421,632 cache bytes/device, 24,220,969,896 bytes/device headroom.
- BF16 selected-plus-KV-control construction: all 48 layers, 36 BF16 QSA cache tensors, 262,144 tokens, 4,227,858,432 cache bytes/device, 22,333,533,096 bytes/device headroom.
- Both KV candidates fit, so `doc/context_contract.json` retains the 262,144-token capability. The selected and BF16 control non-aligned smokes passed; no chunking/layout regression was found.
- `doc/host_weight_contract.json` now records the exact selected expert/PLE representations, upload-preservation checks, capacity arithmetic, teacher metrics, and token-out host totals.

## Validation commands

Core hardware commands and candidate templates are in `README.md`. Generated artifacts were rebuilt with:

```bash
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/recompute_context_contract.py
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/update_host_contract.py
python models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/finalize_sweep.py
```

Static validation uses the autoport environment:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_full_model.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py
```

An initial collection attempt without sourcing `ttenv.sh` failed with `ModuleNotFoundError: models.autoports`; it did not open a device or execute model code. A later sourced command omitted the two-device visibility/descriptor exports, opened a 1x1 P150 fixture, and failed seven 1x2 reshapes; `static_contracts_wrong_mesh.xml` preserves that command error. The same run exposed and fixed one real stale assertion that expected the pre-sweep hard-coded sampling default instead of the selected-config default. The sourced 1x2 rerun is the authoritative static result.

The first final-source static rerun found one stale host-contract assertion naming the displaced shared-HiFi2 policy (22 passed, 13 skipped, 1 failed). The assertion was updated to the promoted config and now also proves `shared_projection={dtype:bfp8, compute_fidelity:lofi, policy:bfp8_lofi}`. After the second review, a candidate-generation/result-policy guard was added. Authoritative result: `static_contracts_final.xml`, 24 passed and 13 explicitly gated tests skipped in 52.80 seconds on the correct 1x2 fixture. Python compilation and `git diff --check` also pass.

## Qualitative and limitations

The selected three-prompt chat-template suite is mechanically non-degenerate and fallback-clean, but manual review found fixed-cap truncation. The HF explanation/coding controls are themselves truncated at 128 tokens; selected TT reaches only a weak explanation sentence and remains in reasoning for coding, while summarization is correct and complete. See `qualitative_review.md`. This diagnostic does not replace the accuracy gate.

The isolated BF16/HiFi2 LM-head point recovered materially under a current-source control but remained slower than the finalists. Full-policy measurements, rather than additive group estimates, are therefore authoritative. Host H2D timing remains variable, but exact counts/bytes and the fixed workload are retained in every report. The selected 128-token qualitative rerun remains mechanically non-degenerate and fallback-clean; explanation and coding are still truncated/incomplete as recorded in `qualitative_review.md`.

## Independent review and commits

The first `$stage-review` returned `more-work-needed`. Its P1 finding was that several passing one-sample rows had anomalously high host H2D time while a transient slow selected sample had been excluded, so the ranking needed current-source controls. Its P2 finding was a stale BFP8 non-aligned evidence path in `context_contract.json`. The controls and equal-evidence finalist comparison described above resolved P1 and changed the winner to the faster shared-LoFi policy. `recompute_context_contract.py` now emits the existing `smokes/baseline.xml` BFP8 path and `smokes/kv_bf16_control.xml` BF16 path; both resolve and both preserve 129-token non-aligned support. Duplicate host-contract limitations and stale token-out references were also removed.

The second fresh `$stage-review` also returned `more-work-needed`: top-level post-selection XML/log aliases and two `result_current_exact.xml` files still described the displaced policy, and `make_candidates.py` incorrectly used the mutable selected config as its base. Every displaced artifact was preserved under the explicitly named old-policy anomaly-control directory, removed from final-looking selected directories, and the top-level aliases were replaced by the final selected XML/logs. Candidate generation now uses the immutable optimized-baseline seed; `make_candidates.py --check` reports `{"artifacts": 17, "reproducible": true}`, and `finalize_sweep.py` checks every retained result's config ID and expected runtime propagation leaves against its candidate JSON.

The final fresh `$stage-review` returned `clean-pass` with no required work. It independently verified 31 primary/replicate/anomaly/selected result artifacts against config IDs and all 61 precision leaves, byte-identical final post-selection aliases, deterministic 17-artifact candidate regeneration, capacity-equivalent context provenance, both Pareto plots, and the authoritative 24-pass/13-skip static result. The isolated stage checkpoint SHA is appended by the follow-up ledger commit below. No push was performed.
