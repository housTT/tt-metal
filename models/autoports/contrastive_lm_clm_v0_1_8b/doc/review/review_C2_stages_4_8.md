# Stage Review: checkpoint C2, stages 4 to 8 (re-review after the review C responses)

Reviewer mode, independent, read-only. Written 2026 Oct 1, 20:41 ET.
Target: Contrastive-LM/CLM-v0.1-8B (frozen Qwen3-8B encoder, last-token pooling after the final RMSNorm, L2 normalized, two MLP heads; score = 100 x cosine of head projections).
Autoport: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b`, branch `hous/clm-v0.1-8b`, HEAD `2c710b113b` (2026 Oct 2 00:25 UTC). `git status` is empty for the autoport directory; the unrelated dirty files elsewhere in the worktree were ignored. `/home/hous/dev/clm-v0.1-8B/REPORT.md` and `/home/hous/dev/clm-v0.1-8B/STATUS.md` changed on disk during this review (REPORT moved from DRAFT to "final, 2026 Oct 2 00:35 UTC"); every statement below is against the versions read after that change.
Times quoted from artifacts are UTC as stamped in the files. Eastern Time is UTC minus 4 h.

Acronyms: TT = Tenstorrent. HF = Hugging Face. TTNN = the tt-metal neural network library. CPU = central processing unit. MLP = multilayer perceptron. RMSNorm = root mean square normalization. SDPA = scaled dot-product attention. KV = key-value. CCL = collective communication library. TP = tensor parallel. BFP8 / BFP4 = block floating point with 8-bit / 4-bit mantissa. bf16 = bfloat16. fp32 = 32-bit float. HiFi / LoFi = high / low math fidelity. L1 = the per-core on-chip memory. TV = total variation distance between two probability vectors. JSON, CSV, PNG = file formats. UTC = Coordinated Universal Time. ET = Eastern Time. SHA = commit hash. MD5 = file hash used below to compare vector files.

Verdict: more-work-needed

Summary. Review C's P1 is closed: the decision-agreement gate exists (`tests/decision_agreement.py`, `GATE["decision_agreement_margin_0p10_min"] = 0.98` in `tests/datatype_sweep.py`), I reproduce every agreement number in `doc/datatype_sweep/agreement_*.json` from the saved vectors and the head checkpoint, and the selection (`accuracy`) follows the stated rule. The five-bucket encoder's claims reproduce: single-text vectors are byte-identical to the three-bucket run, the fifteen-variant trace-safety JSON matches its log, and the `bf16_all` infeasibility record matches its log. Policy descriptions now match the code. What remains is four P2 items, all documentation or one short measurement: the stage 6 README still shows the superseded fidelity table and omits the gate it is judged by, the "all regenerated on the final code" claim is wrong for one latency column, the MLP fidelity lever was never measured under the selected attention policy, and `selected_precision_config.json` is still missing the fields the skill lists. No P1.

## Required Work

- P2: The stage 6 README still carries the generation-A fidelity table, does not report the decision-agreement gate, and PLAN row 6 was not amended
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/README.md` table: cosine 0.99909 mean, 0.99588 min, 0.99753 p05; state head 0.99744 / 0.99429; action head 0.99870 / 0.99573; alone vs batched 0.99934 / 0.99643. The JSON beside it, `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/fidelity_accuracy.json` (timestamp 2026-10-01T23:10:14Z, `trace_lens [128, 1024, 2048]`, byte-identical to `doc/datatype_sweep/fidelity_accuracy.json`), and my recomputation from `fidelity_accuracy_tt_single.npy` give 0.99910 / 0.99596 / 0.99756; 0.99746 / 0.99434; 0.99871 / 0.99571; 0.99934 / 0.99644. `/home/hous/dev/clm-v0.1-8B/REPORT.md` section 3 uses the JSON values, so the report disagrees with the stage 6 README it summarizes. `git diff fe0b69f03e HEAD` on the README touches only the implementation bullets, the new "Serving nondeterminism" section and the context paragraph; review C asked for both README tables to be updated from the JSON and only the stage 8 one was.
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4 row 6 names "decision agreement (argmax) >= 98 % on Typed Decisions cases scored by both" as a stage 6 gate. The stage 6 README does not report it; its only mention defers to `doc/release/`. The gate is defined only in `doc/datatype_sweep/README.md` and `tests/datatype_sweep.py`. Measured against the plan's literal wording every policy fails (accuracy 191/200 = 95.5 percent plain argmax); the margin-aware reading (186/188 = 98.9 percent over decisions with reference margin >= 0.10) is what passes. PLAN section 4 carries dated amendments for stages 2 and 7 but none for the stage 6 gate form, the stage 6 isolation gate (now applied as a mean), or the stage 8 candidate set and tie rule.
  Why this matters: the stage's headline table is contradicted by its own JSON and by the report; the gate that decided the default serve profile is defined outside the stage that owns it; the plan still states a threshold nothing meets.
  Required next step: replace the stage 6 table with the JSON values; add an agreement row (plain 95.5 percent, 98.9 percent over the 188 confident decisions; 8 of 200 reference margins below 0.05, 12 below 0.10) with the one-paragraph justification for the margin-aware form; add a dated PLAN section 4 amendment for rows 6 and 8. No device time.

- P2: "Six candidates, all regenerated on the final code" is not true for the `performance` latency column
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/bench_performance.json` has `timestamp 2026-10-01T21:58:06Z`. The host-side norm path landed in commit `498325e8ed` at 22:08 UTC (work log: 22:05) and the two-phase warmup in `fe0b69f03e` at 22:27 UTC. That bench therefore ran on the first encoder generation with the device-side slice / norm tail that the trace tracker flagged at 22:00 UTC (43 buffers). The other latency rows: `bench_accuracy.json` 22:17, `bench_bfp8_attn.json` 22:15, `bench_bfp8_attn_hifi2.json` 22:21 (the `fe0b69f03e` forward path, unchanged since for nine variants), `bench_bfp8_lofi_mlp.json` 23:17. The fidelity rows for all five runnable policies were regenerated between 23:10 and 23:17 UTC; each `/home/hous/dev/clm-v0.1-8B/logs/fidelity_<policy>_final.log` carries the two-phase `warmup done ... (prepare ... capture ...)` line. The sweep README header says "regenerated on the final code 2026 Oct 1 23:11 to 23:19 UTC" over a table that includes the latency columns; REPORT section 4 says "all regenerated on the final code"; the work log says, correctly, "(where missing) benches regenerated". Review C asked for exactly this re-run or relabel.
  Why this matters: a provenance claim is contradicted by the artifact it points at, and the two Pareto PNGs plot one point from a different code path. The gate outcome does not change (`performance` fails every gate).
  Required next step: either re-run `tests/bench_encoder.py --precision performance` on the final code, or label that row's regime in `doc/datatype_sweep/README.md` and REPORT section 4 and drop the word "all".

- P2: The MLP fidelity lever was never measured under the selected attention policy
  Evidence:
  Every custom candidate changes one axis starting from `bfp8_attn` (SDPA fidelity, MLP fidelity, MLP dtype); none starts from the selected `accuracy` policy. `bfp8_lofi_mlp` (bfp8 attention + LoFi MLP) is 23.7 percent faster than `accuracy` at 1024 tokens (130.1 vs 170.5 ms) and fails the gate at 180/188, but `bfp8_attn` alone already fails at 183/188, so the LoFi effect on the gate is not isolated. The LoFi-only latency delta, measured between `bfp8_attn` and `bfp8_lofi_mlp`, is 12.2 percent at 1024 tokens (148.1 to 130.1 ms) and 12.5 percent at 2048 (286.2 to 250.4 ms). The layer-0 perf report (`doc/functional_decoder/tracy/layer0/prefill_perf_report.csv`, rows 31, 32, 34 and 127, 128, 130) puts the three MLP matmuls at 729 of 1557 us (47 percent) at 128 tokens and 2151 of 4519 us (48 percent) at 1024 tokens. `doc/optimized_full_model/README.md` closes with "the attention math fidelity lever is rejected by the decision-agreement gate ... What remains is kernel-level work on the 128-token matmuls and the 4-core norms". The datatype-sweep skill asks for BFP8+LoFi against BFP8+HiFi2 on the dominant groups; the optimize skill asks to move one tensor group at a time from the selected baseline.
  Why this matters: `accuracy` + LoFi MLP is the one legal candidate that could be faster than the shipped default and still pass; until it is measured, the "fastest passing policy" and "levers exhausted" statements are not earned. With the gate's resolution (one decision = 0.53 points on 188), the outcome is not predictable from the existing rows.
  Required next step: add `accuracy_lofi_mlp` to `CUSTOM_POLICIES` in `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt/encoder.py` (`WQKV`, `WO`, `KV_CACHE` BF16; `LI_QKV_PREFILL`, `LI_O_PREFILL`, `SDPA_PREFILL` HIFI4; `LI_FF1_FF3`, `LI_FF2` LOFI), run `tests/run_fidelity.py`, `tests/decision_agreement.py` and `tests/bench_encoder.py` for it, and add the row to the sweep. About five minutes of device time. A failure is then an earned, recorded rejection.

- P2: `selected_precision_config.json` still lacks the fields the skill lists and review C named
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/selected_precision_config.json` now has `precision`, `env.CLM_PRECISION`, `gate`, `selected_row` and `policy_spec` (five `TensorPrecision` groups, eight `OpFidelity` groups). I confirmed `policy_spec` matches `ModelOptimizations.accuracy` for Qwen3-8B (`models/tt_transformers/tt/model_config.py` lines 275 to 290, the generic `else` branch). Still missing: activation and residual dtype (`TensorGroup.ACTIVATION: None`, that is the bf16 input dtype), layer exceptions (none: `DecodersPrecision.__init__` applies `decoder_conf` to all 36 layers), CCL dtype (none on 1x1; stock all-gather dtype on 1x4), final norm and heads placement (host fp32, `TtQwen3Encoder._pool_and_norm` and `tests/decision_agreement.py`), the weight dtype passed to `create_tt_model` (`ttnn.bfloat8_b`), the runtime flag `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`, and a pointer to the consumption evidence (`attention.py` lines 116 to 150, `mlp.py` lines 91 to 95, 176, 340; perf report rows below).
  Why this matters: the skill says a selected config that only names weight dtypes is incomplete and must be consumable mechanically. The host-side fp32 norm and heads are part of the numerical policy a consumer must reproduce to get the recorded agreement numbers.
  Required next step: add those keys to the JSON (about a dozen lines). No device time.

## Other Concerns

- `bf16_all` record: the failing variant is the 128-token batch-1 trace, not "the first 1024-token matmul" as `doc/datatype_sweep/README.md` says. Evidence: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_bf16_all_final.log` lines 523 to 525 show `rotary_embedding_llama ... input_Ht=4` (4 tiles = 128 tokens) immediately before the throw, and `TtQwen3Encoder.warmup` prepares `128_0_1_sp0` first. The core range `[0-0 - 7-3]` is the 8x4 grid the stage 3 audit found for 128 tokens. `infeasible_bf16_all.json` says "first prefill variant", which is correct but vague. The README's "a framework change outside this port" is inconsistent with the port already editing `model_config.py` for the Qwen3-8B trace entries. A higher-fidelity probe without the L1 problem (bfp8 MLP weights with `HIFI4_FP16` math) was not tried; it is not needed for the selection, so this is not required work.
- The agreement gate is evaluated on single-text vectors. Recomputed on the batched vectors (the path a multi-text request takes), `bfp8_lofi_mlp` reaches 185/188 = 98.4 percent and would pass; `accuracy` passes in both modes (186/188 single, 185/188 batched on the nine-variant run, 186/188 on the five-bucket run); `bfp8_attn` (180/188) and `bfp8_attn_hifi2` (183/188) fail in both. Record which vector set the gate uses and why, and state the gate's resolution (one decision = 0.53 points).
- The two Pareto PNGs plot cosine against latency with the cosine gate line; the metric that decides the sweep (agreement) is not plotted, so three failing policies sit above the drawn gate line. Add an agreement-vs-latency plot or mark failing points.
- Direction of the latency comparison in `doc/datatype_sweep/README.md`: "costs 6.4 percent at 128 tokens and 13 percent at 1024 and 2048 tokens against bfp8_attn". From `sweep_results.json`: `accuracy` is 6.8 / 15.1 / 12.8 percent slower than `bfp8_attn`; `bfp8_attn` is 6.4 / 13.1 / 11.3 percent faster than `accuracy`. Pick one direction.
- REPORT section 5 describes the nine-variant encoder: it cites `perf_summary_accuracy.json` (22:17 UTC) and `perf_summary.json`, omits the 256 and 512 rows, and says "nine trace captures: 6.2 s". The shipped encoder has fifteen variants (`perf_summary_accuracy_buckets5.json`: 57.7 / 92.1 / 161.6 ms at 128; 72.1 / 162.4 / 306.7 at 256; 93.8 / 312.5 / 624.5 at 512; 170.4 / 648.6 / 1300 at 1024; 321.7 / 1300 / 2576 at 2048; warmup 17.6 s in the fidelity run, 8.9 s in the replay run, 16.1 s in the container). Sections 2 and 7 are correct; section 5 should cite the five-bucket summary as the final default-run numbers. Also in that JSON the `notes` string still quotes 1.609 / 4.692 ms per layer while `per_layer_device_ms_from_tracy` says 1.557 / 4.519, and REPORT's "+2.8 percent" is relative to the bound while "+4.6 percent" is relative to the measurement (4.8 percent relative to the bound).
- Multi-chip (carried from review C, unchanged): the plan's stage 4 gate, cosine against the one-chip TTNN output >= 0.999, is still not evaluated in any artifact (review C measured mean 0.99929, min 0.99547, 68 of 308 texts below 0.999); the README still states both "the residual stream comes back width-sharded across the four devices" and "the residual is replicated on every device after each all-gather"; stage 5 is "Not done beyond the stock TP plan" while `p150x4` ships. The card's `status: Experimental community bring-up` covers the package, not that profile specifically.
- Head placement is still asserted from the cache-hit path, not measured (carried from review C).
- The README-example qualitative substitute in the stage 6 README and REPORT section 3 still comes from `doc/probe/probe_full_encoder.json` (probe code, HF bf16 control, `max_seq_len 1024`). The final-path numbers for the selected policy can be computed offline from the saved vectors, as review C did (noul 0.852, billing 0.991, frustration 2.000, tides Moon 0.992).
- `TtQwen3Encoder._select_trace_lens` validates the multiple-of-128 rule on the env entries and then appends `max_seq_len` unchecked (`tests/test_encoder_buckets.py` asserts `[128, 256, 512, 640]` for 640; a value such as 600 would pass through and contradict the contract text "multiples of 128 only"). `from_env` does not reject `CLM_MAX_TOKENS` above `CLM_MAX_SEQ_LEN`; a longer text would then fail at `prefill_ids[i, :len(ids)]` at request time. Default settings are safe; this is a guard, not a bug in the shipped configuration.
- `doc/datatype_sweep/fidelity_bfp8_attn_twophase_limit96.{json,npy}` (96 texts, 22:23 UTC) is still not mentioned in any README (review C: document or remove).
- Work logs: one consolidated `doc/full_model/work_log.md` covers stages 4 to 8; `doc/multichip_decoder/`, `doc/optimized_full_model/` and `doc/datatype_sweep/` have none, although PLAN section 4 and the datatype-sweep skill name a per-stage file. Acceptable if recorded as a deviation.
- REPORT section 1 still says "head `a3df3fd2ee`" one line before "built ... from tt-metal `2c710b113b`"; `git log` shows HEAD `2c710b113b`. Section 7 "Publish" carries an unfilled `PULL_CHECK_REPORT` placeholder while the status line says "final".

## Hard-Check Gaps

- Policy consumption for the selected policy is now shown at the layer level: `doc/functional_decoder/tracy/layer0/prefill_perf_report.csv` rows 15 / 39 / 63 / 87 (QKV, `HiFi4 BF16 x BF16`), 28 / 52 / 76 / 100 (WO, `HiFi4 BFP8 x BF16`), 31 / 32 / 34 (w1, w3, w2, `HiFi2` with BFP8 weights), and 111 / 124 / 127 / 128 / 130 at 1024 tokens. These rows are consistent with `accuracy` and inconsistent with `bfp8_attn`. No profile exists for the fifteen-variant traced path itself; not required, the per-variant device times reconcile with the layer stack.
- No watcher-clean run for stages 6 to 8 (stage 1 has one). Carried from review C.
- The fifteen-variant trace-safety check ran only for `accuracy` on 1x1. The `p150-fast` and `p150x4` profiles ship the same fifteen buckets with a different policy or mesh; no `TT_METAL_TRACE_ALLOC_TRACKING=1` run exists for either. The allocation pattern does not depend on dtype, so the risk is low, but the 1x4 path has never been under the tracker at all (carried from review C).
- The decision-agreement reference covers 40 cases / 200 decisions; the 98 percent bar on 188 decisions is decided by one to three decisions. Extending the CPU fp32 reference to more cases is CPU-only work and would make the gate less sensitive to batch composition.
- A batched-vector agreement artifact per policy is not saved (numbers below are from this review).
- Vectors remain gitignored and no SHA-256 is recorded in the JSON; all re-derivation here depends on local files (carried from review C).

## Anomaly Ledger

- Observed anomaly: single-text vectors are byte-identical across four runs and three code generations (22:07 host-norm run, 23:11 final run, 23:41 five-bucket run; MD5 `550516dc4f22` for all three `*_tt_single.npy`), and the 22:07 and 23:11 batched vectors are byte-identical too (`2346293aa12f`).
  Evidence: MD5 and `np.array_equal` over `doc/optimized_full_model/fidelity_accuracy_hostnorm_tt_*.npy`, `doc/datatype_sweep/fidelity_accuracy_tt_*.npy`, `doc/full_model/fidelity_accuracy_tt_*.npy`, `doc/optimized_full_model/fidelity_accuracy_buckets5_tt_single.npy`.
  Affected path: `TtQwen3Encoder.embed_ids`, all three code generations since the host-norm rewrite.
  Control or comparison: the review C generation-A vectors differed at cosine 0.999994 (device-side norm); the batched five-bucket vectors differ for 128 of 308 texts (min cosine 0.99863) because the bucket grouping changes batch-mates.
  Likely subsystem: deterministic device kernels for a fixed shape; the tracker's flags on the 22:07 generation were about allocation safety, not values.
  Investigation performed: hashes and exact comparison in this review.
  Resolution: controlled; this is positive evidence for determinism and for the "bit-identical" claim.

- Observed anomaly: the same text alone and inside a batch differs by up to 1 minus cosine = 3.6e-3; 60 of 308 texts below 0.999 (56 on the five-bucket run); argmax flips on 4 to 5 of 200 subset decisions between the two modes, and for `bfp8_lofi_mlp` the gate verdict depends on the mode (180/188 single, 185/188 batched).
  Evidence: `fidelity_*_tt_single.npy` vs `fidelity_*_tt_batched.npy`, every policy; agreement recomputation in this review.
  Affected path: batched traced prefill on P150 (tt-metal issue 47238).
  Control or comparison: repeated replay of the same input reproduces to >= 0.99999976 (fifteen variants); different inputs give cosine 0.58 to 0.98; the selected policy passes the gate in both modes.
  Likely subsystem: batch-count-dependent reduction order in the prefill matmul and SDPA kernels.
  Investigation performed: stage recorded it in the stage 6 README and the card limitations; this review quantified the gate sensitivity.
  Resolution: controlled and disclosed; record the gate's vector set and resolution (Other Concerns).

- Observed anomaly: decisions reverse against the fp32 reference at state cosine above 0.999 under every policy; for `accuracy` two of the 188 confident decisions reverse (`invoice_processing_000063` and `_000096 / discrepancy_severity`, reference margins 0.13 and 0.32, TV 0.24 and 0.28).
  Evidence: `doc/datatype_sweep/agreement_accuracy.json` disagreements; recomputed identically here (191/200, 186/188, mean TV 0.038, max TV 0.291).
  Affected path: encoder output through the host fp32 heads, 100 x cosine scoring.
  Control or comparison: CPU bf16 control at cosine 0.99998 vs fp32 (review C); the `performance` policy shows the same mechanism at 155/188.
  Likely subsystem: BFP8 weights and HiFi2 fp16-accumulate MLP math, amplified by the head scoring on small-margin decisions.
  Investigation performed: stage measured it per policy and disclosed the two confident reversals in the sweep README.
  Resolution: controlled by the margin-aware gate and the card text; the gate definition itself still needs to land in the stage 6 README and the plan (P2).

- Observed anomaly: `bf16_all` cannot be captured: `Statically allocated circular buffers on core range [0-0 - 7-3] grow to 2012160 B which is beyond max L1 size of 1572864 B`.
  Evidence: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_bf16_all_final.log` lines 522 to 553 (policy `bf16_all`, 24.8 s load, throw inside `mlp.py` line 194 during `_prepare_trace_prefill` of the first variant); `infeasible_bf16_all.json` quotes the same string, the same frame and `2026-10-01T23:15:59Z`.
  Affected path: w1 / w3 prefill matmul program config with bf16 weights on the 8x4 grid at 128 tokens.
  Control or comparison: every bfp8-MLP policy captures the same variant without error.
  Likely subsystem: `model_config.matmul_config` circular-buffer sizing assumes bfp8 weights.
  Investigation performed: recorded with the exact error; no program-config adaptation attempted.
  Resolution: controlled (the candidate is slower than the selected one, so no selection is affected); README mislabels the variant (Other Concerns).

- Observed anomaly: the stage 7 layer-stack "lower bound" exceeded the measurement in review C (57.92 vs 57.64 ms).
  Evidence: `perf_summary_accuracy_buckets5.json` now uses layer ops only: 56.05 ms at 128 tokens vs 57.70 measured (2.9 percent), 162.68 vs 170.40 at 1024 (4.5 percent of the measurement).
  Affected path: performance reconciliation only.
  Control or comparison: eager and traced parity (57.45 vs 57.65 ms) in `doc/fused_decoder/README.md`.
  Likely subsystem: profiler accounting (full per-layer time vs layer ops only).
  Investigation performed: layer-only accounting adopted after review A.
  Resolution: fixed; the `notes` string in the JSON is stale (Other Concerns).

- Observed anomaly: stage 6 README fidelity table contradicts the JSON next to it and REPORT section 3.
  Evidence: values listed under the first P2.
  Affected path: evidence integrity for stage 6.
  Control or comparison: the two value sets differ by at most 8e-5 in cosine (generation A vs host-norm path, review C: cosine 0.999994 between the vector sets); no gate outcome changes.
  Likely subsystem: documentation not regenerated.
  Investigation performed: `git diff`, JSON and vector recomputation.
  Resolution: more-work-needed (P2).

- Observed anomaly: the 1x2 mesh (P300) does not open, `Fabric Router Sync: Timeout after 10000 ms on Device 0`.
  Evidence: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_accuracy_1x2.log`, `bench_accuracy_1x2.log`.
  Affected path: optional 1x2 profile, not published.
  Control or comparison: 1x4 opens and serves on the same boards.
  Likely subsystem: fabric or control plane for a sub-mesh of the discovered 1x4 system mesh, or a physical link.
  Investigation performed: documented, not pursued (optional per PLAN row 4).
  Resolution: controlled (carried from review C).

- Observed anomaly: the build 7 `p150x4` profile did not boot inside the container (fabric router kernel source mode 0660).
  Evidence: `doc/multichip_decoder/README.md` "In-container 1x4 serving"; `doc/release/RUN_NOTES.md` eighth and ninth attempts; `tt-model.yaml` gained a `verify:` line that opens the file.
  Affected path: packaging of the multi-chip profile; single-chip profiles never JIT-compile fabric kernels.
  Control or comparison: build 9 `p150x4` boots and serves (README example cold 174.6 ms, new state 34.1 ms).
  Likely subsystem: file modes carried from the dirty worktree through `pre-commit` stash and restore.
  Investigation performed: root cause named, build script and manifest changed.
  Resolution: fixed.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/clm-v0.1-8B/PLAN.md` (section 4 rows 4 to 8 and amendments); `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/{stage-review,multichip,full-model,optimize,datatype-sweep}/SKILL.md`; `/home/hous/dev/clm-v0.1-8B/STATUS.md`; `/home/hous/dev/clm-v0.1-8B/REPORT.md` (read twice, before and after the on-disk change).
- Artifact paths (under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/`): `doc/context_contract.json`; `doc/review/review_C_stages_4_8.md`; `doc/full_model/{README.md,work_log.md,fidelity_accuracy.json}` and its two `.npy`; `doc/multichip_decoder/{README.md,fidelity_accuracy_1x4.json,bench_accuracy_1x4.json}` and its two `.npy`; `doc/optimized_full_model/{README.md,bench_accuracy.json,bench_accuracy_buckets5.json,perf_summary_accuracy.json,perf_summary_accuracy_buckets5.json,fidelity_accuracy_buckets5.json,fidelity_accuracy_hostnorm.json,replay_trace_check_accuracy.json,replay_trace_check_accuracy_buckets5.json,replay_trace_check_bfp8_attn.json}` and the `_buckets5` and `_hostnorm` `.npy` pairs; `doc/datatype_sweep/{README.md,sweep_results.json,sweep_results.csv,selected_precision_config.json,infeasible_bf16_all.json,agreement_*.json,fidelity_*.json,bench_*.json}` and every `fidelity_*_tt_{single,batched}.npy`; `doc/functional_decoder/README.md` and `tracy/layer0/prefill_perf_report.{csv,console.log}`; `doc/release/RUN_NOTES.md` (lines 60 to 110); `tt-model.yaml`. Reference data: `/home/hous/dev/clm-v0.1-8B/reference/{fidelity_corpus.json,hf_embeddings.npy,typed_decisions_subset_reference.json}`; `/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt`. Logs: `/home/hous/dev/clm-v0.1-8B/logs/{fidelity_bf16_all_final.log,fidelity_*_final.log,fidelity_accuracy_buckets5.log,replay_trace_check_buckets5.log}` plus the directory listing with timestamps.
- Code paths: `tt/encoder.py` (full file and `git diff fe0b69f03e HEAD`), `tests/datatype_sweep.py`, `tests/decision_agreement.py`, `tests/run_fidelity.py`, `tests/replay_trace_check.py`, `tests/test_encoder_buckets.py`; `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` (lines 73 to 102, 213 to 420, 728 to 750, 4852 to 4930), `generator.py` (trace key construction, lines 784 to 847), `attention.py` (lines 116 to 150), `mlp.py` (lines 91 to 95, 173 to 177, 340 to 341), `model.py` (lines 251 to 263). Git: `git log`, `git show --stat` for `a3df3fd2ee`, `a890a5ab05`, `b39ef75ab6`, `2c710b113b`, `git status`, `git diff` on three files.
- Commands run: read-only `cat`, `sed`, `grep`, `ls`, `cmp`, `git`; one analysis script (`/tmp/claude-1002/-home-hous-dev-clm-v0-1-8B/0082bb62-0b82-4079-9fa4-87d59d66a985/scratchpad/c2/recompute.py`) with `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` that recomputed decision agreement for nine vector sets in single and batched mode, hashed every `.npy`, compared the five-bucket and three-bucket vectors, recomputed the fidelity summaries and head-projection cosines, and read the bench JSON headers; one short script that summarised the replay JSON files and the perf report rows. No device was opened, no `ttnn` import, no server or container started, no test run, no implementation file modified.

Recomputed decision agreement (argmax vs fp32 reference; all 200 / the 188 with reference margin >= 0.10; mean TV):

| vectors | single | batched |
|---|---|---|
| accuracy (sweep 23:11, full_model copy, hostnorm 22:07; identical files) | 191 / 186 / 0.038 | 191 / 185 / 0.038 |
| accuracy five-bucket (23:41) | 191 / 186 / 0.038 | 192 / 186 / 0.039 |
| bfp8_attn (23:13) | 189 / 183 / 0.046 | 186 / 180 / 0.048 |
| bfp8_attn_hifi2 (23:14) | 183 / 180 / 0.045 | 191 / 183 / 0.043 |
| bfp8_lofi_mlp (23:17) | 188 / 180 / 0.050 | 194 / 185 / 0.041 |
| performance (23:15) | 158 / 155 / 0.176 | 161 / 156 / 0.176 |
| 1x4 accuracy (22:14) | 190 / 185 / 0.036 | 192 / 186 / 0.034 |

The single-mode column reproduces `agreement_<policy>.json` exactly (0.955 / 0.9894, 0.945 / 0.9734, 0.915 / 0.9574, 0.940 / 0.9574, 0.790 / 0.8245); the reference scorer matches `predicted_label` on 200 of 200 and gives 0.370 against gold. Selection: only `accuracy` passes, so the tie rule is not exercised and `selected = accuracy` follows the stated rule. Fidelity recomputation from the `.npy` files reproduces every `cosine_single_vs_hf` and `head_projection_cosine` field in the five runnable policies' JSON to the printed precision. The 2047 and 2048-token corpus texts score 0.99969 and 0.99961 on the five-bucket run, consistent with `context_contract.json` ("cosine >= 0.9996").

## Residual Risk

- The default profile passes the gate by two confident decisions (single vectors) or one to two (batched); batch composition alone moves one policy by five decisions. The gate is sound in direction but thin in resolution until the fp32 reference covers more cases.
- `p150-fast` ships a policy that fails the stage's own gate (183/188). It is disclosed in the card and the manifest with its agreement number; a consumer who picks it for speed gets about 2.7 percent confident-decision disagreement instead of 1.1 percent.
- `p150x4` ships without the stage 5 CCL audit, without any trace-safety run on the 1x4 mesh, and with the residual-layout description contradicting itself.
- The MLP fidelity lever under the selected attention policy is unmeasured (P2); the shipped default may be leaving about 12 percent of latency on the table at 1024 and 2048 tokens, or may not.
- All fidelity re-derivation depends on gitignored local vectors.
- The worktree is live: REPORT and STATUS changed during this review and REPORT still has a placeholder; HEAD did not move.
