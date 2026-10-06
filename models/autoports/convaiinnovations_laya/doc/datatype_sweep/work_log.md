# Stage 8 work log (Track T3, UTC)

- 2026 Oct 5 23:30 `tests/datatype_sweep.py` written: one `run_fidelity.py` process (200 gate decisions, 40 hidden-state
  cases, final buckets) and one `bench_latency.py --mode fresh` process (the four published cells captured) per policy,
  then the aggregate, the selection rule, `sweep_results.json/.csv`, `selected_precision_config.json`, two Pareto PNGs.
  `run_fidelity.py --hidden-cache` caches the CPU fp32 hidden states of the 40 cases (written by the stage 7 gate run,
  `/home/hous/dev/laya/state/tt_cache/fidelity_hidden_ref_gate40.pt`), so each policy costs about 2.5 min of device
  time and no CPU reference forwards. Two cheap erf variants added to `tt/model_config.py: POLICIES`
  (`bf8w_hifi2_erf`, `bf8w_lofi_mlp_erf`).
- 23:50 `s8_sweep` launched (`/home/hous/dev/laya/logs/p3_s8_sweep_20261005T235020Z.log`, subprocess log
  `sweep_subprocess.log`), chained behind the stage 7 measurements.
- 23:58 `bf16_hifi4` fidelity failed in warmup at the first block-sharded bucket (L1 clash in the sharded GeGLU down
  projection with bf16 weights); `mlp_shard_plan` now declines for bf16 weights; `s8_sweep2` re-run with
  `--skip-existing` filled the file (`/home/hous/dev/laya/logs/p3_s8_sweep2_20261005T235827Z.log`).
- 2026 Oct 6 00:00 Sweep aggregate: 7 of 9 policies pass the stage 6 gates; `bf8_act` is the fastest passing policy
  (228.5 ms over the four cells) and is selected by the plan's rule; its margins on the median (0.0188 against 0.02),
  the scorer PCC (0.9904) and the encoder hidden PCC (0.9902) are under 10 percent of the gate width.
- 00:00 Amendment A11 received from the orchestrator (binding): thin margins require a served confirmation of the
  candidate against the runner-up (the shipped `bf8w_hifi3_erf`): full E1 (488 items, wire and tensor) and full E2 (400
  cases) each; gates: E2 accuracy within 0.010 of the CPU fp32 row, the other four E2 metrics within 0.015; E1 confident
  agreement at least 98 percent and argmax agreement at least 95 percent over the 488 items. `datatype_sweep.py --only
  confirm` implements the gates and writes the `confirmation` block into `selected_precision_config.json`,
  `sweep_results.json` and `confirmation.md`.
- 00:01 The `bf8_act` full-suite host serve started before the rule arrived (`s7_serve_p150`, results
  `/home/hous/dev/laya/evals/results/host_tt_p150_b1_20261006T000056Z`); it doubles as the candidate's confirmation run
  (E1 and E2 are the first two steps). Chain `t3_chain_confirm2.sh`: invariance test for `bf8_act`, the runner-up's
  full-suite serve, the confirmation, then the `p150x4` E5 column with the selected policy.
- 00:03 `bf8_act` full-suite serve done in 2.5 min (`host_tt_p150_b1_20261006T000056Z`): E1 tensor path 475 of 488 argmax,
  401 of 403 confident; E2 0.3605 / 0.3312 / 0.3133 / 0.1730 / 0.6903 (CPU fp32 0.3615 / 0.3315 / 0.3155 / 0.1747 /
  0.6937); E3 0.953 / 0.588; E5 10.3 / 22.8 / 39.5 / 185.3 ms client p50, 213 to 271 questions per second. The A11
  confirmation gates all pass for the candidate.
- 00:04 `s8_invariance_cand` (`decision_agreement.py --policy bf8_act`, final buckets): the same argmax in 16 of 16
  questions, but max abs delta p alone against in batch 0.0381 (B 2 and B 4 0.0335, mixed B 8 0.0199, B 64 0.0381)
  against the stage 6 gate of 0.01 (`bf8w_hifi3_erf`: 0.0090). bfp8 activations make the probabilities depend on the
  bucket's matmul blocking by up to 0.038. The A.7 sweep gate list does not contain this gate, but PLAN.md section 5 row
  6 does, and the stage 7 task requires the stage 6 gate scripts to pass on the shipped configuration. Decision:
  `bf8_act` is recorded as "fastest; passes the A.7 gates and the A11 served confirmation; fails the stage 6
  alone-versus-in-batch gate"; the runner-up `bf8w_hifi3_erf` ships and `DEFAULT_POLICY_NAME` is unchanged.
  `datatype_sweep.py --only confirm --invariance <json>` folds this gate into the confirmation block. The chain that
  would have selected the candidate was stopped before its confirm step; the runner-up's full-suite serve
  (`s7_serve_p150_runnerup`) continues and is the build 1 run of record.
- 00:10 Chain 3: confirm with both served runs and the invariance file, then the `p150x4` E5 column with the shipped
  policy, then an informational invariance run for `bf8w_hifi2_erf` (the fastest policy that passes every A.7 gate
  with no thin margin, 246.6 ms, 7.4 percent under the shipped policy; not selectable under the plan's rule and A11,
  recorded for the orchestrator).
- 00:07 Runner-up full-suite serve done (`host_tt_p150_b1_20261006T000400Z`, the build 1 run of record): E1 476 of 488
  argmax, 403 of 403 confident (tensor and wire); E2 0.3590 / 0.3315 / 0.3109 / 0.1716 / 0.6892; E3 0.955 / 0.593; E5
  11.0 / 26.5 / 46.0 / 212.9 ms client p50, 187 to 231 questions per second; every call at a 128 or 256 bucket.
  Confirmation block written (`confirmation.md`, `selected_precision_config.json`, `sweep_results.json`): both runs
  pass the A11 gates; the candidate fails the stage 6 invariance gate; selected `bf8w_hifi3_erf`, default unchanged.
- 00:07 `p150x4` E5 column served with the shipped policy (`host_tt_p150x4_b1_20261006T000708Z`; mesh load 11.6 s, warmup
  8.1 s plus 0.7 s for 30 buckets on four chips): 13.3 / 16.8 / 26.5 / 85.4 ms client p50; 299 to 712 questions per
  second batched; merged into the build 1 SUMMARY as the `p150x4` column.
- 00:08 Informational: `bf8w_hifi2_erf` also fails the stage 6 invariance gate (0.0289 against 0.01, same argmax 16 of
  16), so among the measured policies only HiFi3 with bf16 activations (`bf8w_hifi3_erf`) keeps the probabilities
  placement invariant within 0.01; HiFi2 and bfp8 activations do not. Device work of stages 7 and 8 complete; devlock
  free; no server process left.
