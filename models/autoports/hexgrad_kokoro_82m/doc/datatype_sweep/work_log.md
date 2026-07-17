# Kokoro-82M — Datatype Sweep work log (stage 07)

Target: `hexgrad/Kokoro-82M` plbert full model, 4× Blackhole p300c
(`ClusterType.P300_X2`), `(1,4)` ring mesh. Env: dev checkout recipe
(`TT_METAL_HOME=/home/ttuser/dev/tt-metal`,
`PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal`).

## 0. Context / architecture reality

Kokoro's plbert is a **non-autoregressive, bidirectional, stateless** ALBERT
encoder: no KV cache, no paged cache, no MoE, no token-by-token sampling. The
datatype-sweep axes therefore map as: weight dtype (attn / mlp / map groups),
activation dtype, compute fidelity (matmul / sdpa / norm), CCL payload dtype
(all_gather / reduce_scatter), logits/readout dtype, greedy argmax. **KV-cache
dtype = N/A** (0 bytes). The precision policy is fully parameterized by
`PrecisionPolicy` (tt/optimized_decoder.py) + `OptConfig` (tt/optimized_multichip_decoder.py),
both already consumed by the runtime path (verified by reading device tensor
dtypes; see propagation_check + per-candidate `dtype_summary`).

## 1. Device health

`tt-smi -ls --local` → 4× Blackhole p300c visible and resettable. No reset/recovery
needed during the stage.

## 2. Baseline refresh (100 gen tokens)

- Regenerated the shared readiness references fresh from pinned HF weights:
  `doc/full_model/make_references.py` → `readiness_recon_{prefill,tf}.refpt`
  (4 IPA phoneme sentences, K=100, **222 generated positions total ≥ 100**).
  AIME24 chat-template is N/A for this TTS model (established stages 01–06); the
  IPA phoneme reconstruction is the model-appropriate main readiness reference.
- Ran the baseline candidate on the current checkout (`run_candidate.py --id baseline`):
  PCC min 0.996999, prefill top-1/5/100 = 1.0, TF top-1 0.9865 / top-5 0.9955 /
  top-100 1.0, traced TF decode 347.8 t/s/u, token-out 558.9/392.3 t/s/u @T128/512.
  Matches stage-06 → baseline reproduced on the current tree.

## 3. Sweep harness

- `run_candidate.py`: one candidate per **subprocess / device job** (per
  $tt-device-usage: one hardware command at a time, clean mesh + 90 MB trace region
  each). Builds through `build_generator(build_kwargs={policy,opt})` — the SAME path
  the readiness runners and vLLM use — then: (1) fast PCC smoke vs HF at all lengths
  incl. non-aligned; (2) `run_prefill_check` + `run_teacher_forcing` (trace-verified;
  full-model top-1/5/100 + TTFT + decode t/s/u); (3) warmed min-of-3×30 traced
  token-out @128/512; (4) `dtype_summary` = readback of the real device tensor dtypes
  + kernel-config fidelities (propagation proof). Writes `candidates/<id>.json`.
- `sweep_driver.py`: sequences the matrix (subprocess each) then aggregates to
  `sweep_results.{json,csv}`.
- Smoke-first note (skill): no candidate needed a semantic code change — all
  dtype/fidelity/CCL fields are already plumbed (`_dtype`, kernel configs, CCL
  typecast path). The KV-cache `paged_fill`/`paged_update` dtype caveat is N/A (no
  cache). BFP8/BFP4 weights and BFP8 CCL all constructed + ran without a TTNN
  blocker; failures are pure accuracy, not op-contract.

## 4. Candidate matrix (10 configs, one device job each)

baseline, bf16_weights, bfp8_lofi, bfp8_hifi4, mlp_bfp4_lofi, mlp_bfp4_hifi2,
attn_bfp4_lofi, all_bfp4_lofi, ccl_bfp8, ccl_ag_bfp8. Results table in README.

Gate: top-1 ≥ 0.90, top-5 ≥ 0.98, last_hidden_state PCC ≥ 0.995 (model-specific
output-fidelity bar carried from stages 01–06 — last_hidden_state is Kokoro's real
output). Ranking metric: trace-verified teacher-forcing decode t/s/u.

Result: **baseline (BFP8/HiFi2, CCL bf16) is the fastest gate-passing config** on
both the ranking metric (347.8 t/s/u, highest among passing) and the stable
token-out metric (558.9 t/s/u @128). The model is launch-bound, so lower precision
buys ≤ ~1–2% token-out and nothing reliable on traced TF decode while eroding PCC:
BFP8+LoFi, all BFP4 groups (+LoFi and +HiFi2), and BFP8-reduce_scatter all fail the
fidelity/accuracy gate; bf16 weights and HiFi4 pass but are slower.

Process note: `mlp_bfp4_hifi2` first errored (`NameError: load_selected`) because I
added the `precision_config` import to generator.py *while* that candidate was
mid-import — rebuilt it after the edit landed; rc=0, status=fail (PCC 0.928). All
other post-edit candidates passed explicit policy/opt so were unaffected. Lesson:
don't edit imported source mid-sweep.

## 5. Selected config wiring (consumed-by-default proof)

- `tt/precision_config.py::load_selected` reads
  `doc/datatype_sweep/selected_precision_config.json` `construct.policy`/`construct.opt`
  → `PrecisionPolicy`/`OptConfig`. `tt/generator.py::build_generator` now loads this
  by default when `policy`/`opt` are not passed (explicit kwargs still override; file
  absent → dataclass defaults == selected baseline; delete file to revert).
- `propagation_check.py` builds via `build_generator` with **no** policy (so the file
  is loaded) and asserts the on-device weight dtypes + kernel fidelities match the
  selected config → `all_consumed: true`, `mismatches: []`
  (`propagation_check.json`).
- `selected_precision_config.json` records weight groups, layer exceptions (none —
  12 weight-tied layers), compute fidelities, activation/residual dtype, CCL dtype,
  KV-cache dtype (N/A), logits/sampling dtype (bf16 readout + greedy argmax).

## 6. Plots, context contract, non-aligned, post-selection

- `make_plots.py` → `top1_perf_pareto.png`, `top5_perf_pareto.png` (all configs
  plotted; non-dominated Pareto frontier; selected point red ★; vertical dotted
  min-accuracy line at 0.90 / 0.98; pass=teal / fail=gray).
- `context_contract.json` → new `datatype_sweep` section: KV-cache dtype N/A,
  context 512 preserved (advertised = supported), `capability_reduction=false`, no
  capacity change (selected config = baseline dtype/layout). Context-contract gate
  `check_context_contract.py --stage datatype-sweep` rc=0.
- `post_selection.py` (default `build_generator` path): warmed token-out no-readback
  benchmark **560 t/s/u @128 · 395 t/s/u @512, eager TTFT 10.7 ms** (recorded
  separately from teacher-forcing in `post_selection_tokenout.json`) + non-aligned
  check (31/33/127/200/511 all PCC ≥ 0.9970, `non_aligned_check.json`).
- `test_full_model.py`: **19/19 pass** with the generator wiring change.

## 7. Stage review + commit

- `$stage-review`: see section below (fresh xhigh subagent).
- Commit: stage-owned files only (doc/datatype_sweep/*, tt/precision_config.py,
  tt/generator.py wiring, doc/context_contract.json, refreshed references). Not
  pushed. SHA logged below.

### Stage-review verdict
**clean-pass** (fresh independent xhigh subagent, read-only). No required work. The
reviewer re-derived the ranking from `sweep_results.json` and confirmed: baseline is
the fastest gate-passing config under the real gate; propagation is genuinely proven
(device tensor dtypes + kernel fidelities read back, `all_consumed: true`); BFP4
coverage complete with real-weight full-model rejections (not synthetic PCC); KV-cache
N/A correct; context 512 preserved; non-aligned re-run; post-selection token-out
recorded separately; plots meet all four requirements; CSV↔JSON↔candidates consistent.
One non-blocking documentation nuance (a BFP4 config within 0.014% / noise of baseline
under a hypothetical top-1/top-5-only gate) was addressed by tightening the README
Pareto interpretation.

### Commit SHA
See below (stage-owned files only; .agents/ pipeline-setup changes excluded; not pushed).
