# Stage 8: datatype sweep (precision policies on the final buckets)

Plugin stage "datatype-sweep" mapped to Laya (PLAN.md section 5 row 8, Appendix A.7, amendments A9, A10 and A11). Nine
precision policies ran the stage 6 gate (200 typed-decisions decisions against CPU fp32, 40 hidden-state cases) and the
four published latency cells (1, 5, 10 and 50 questions of the STATE_EN speed-table shape) on the stage 7 deployment set
(seq buckets 128, 256, 512; row buckets 1, 2, 4, 5, 8, 10, 16, 32, 50, 64). PCC = Pearson correlation coefficient;
dp = the probability of an option after the temperature softmax.

## Result in one paragraph

`bf8_act` (bfp8 matmul weights and intermediates, bf16 residual, HiFi3, tanh GELU) is the fastest policy that passes
the Appendix A.7 gates (228.5 ms over the four cells, 14 percent under the shipped policy) and it also passes the
amendment A11 served confirmation (full E1 and E2 against the runner-up), but it fails the stage 6 alone-versus-in-batch
gate: the same question moves by up to 0.038 in probability between buckets (gate 0.01; the shipped policy 0.009).
The shipped policy `bf8w_hifi3_erf` therefore stays the default (`DEFAULT_POLICY_NAME` unchanged); `bf8_act` is recorded
as "fastest, passes A.7 and A11, fails the stage 6 invariance gate". The next fastest clean policy, `bf8w_hifi2_erf`
(246.6 ms, every A.7 gate with a wide margin), fails the same invariance gate (0.029). Among the measured policies only
HiFi3 with bf16 activations keeps the decision probabilities placement invariant within 0.01.

## Files

| file | role |
|---|---|
| `tests/datatype_sweep.py` | the driver: one `run_fidelity.py` process and one `bench_latency.py --mode fresh` process per policy, the aggregate, the selection rule, the thin-margin test, the A11 confirmation (`--only confirm`), the Pareto plots |
| `fidelity_<policy>.json` | the stage 6 gate run per policy (`tests/run_fidelity.py --items gate --hidden-cases 40`, final buckets, cached CPU reference hidden states) |
| `latency_<policy>.json` | the four published cells per policy (`tests/bench_latency.py --mode fresh`, the four cell buckets captured, 3 warm, p50 of 20) |
| `sweep_results.json`, `sweep_results.csv` | the table below plus the thin margins, the selection and the confirmation block |
| `selected_precision_config.json` | the shipped policy, the rule applied, the gates, the latency cells, the `env` block for the manifest, the `confirmation` block |
| `confirmation.md` | the A11 confirmation table |
| `decision_agreement_bf8_act.json`, `decision_agreement_bf8w_hifi2_erf.json` | the stage 6 alone-versus-in-batch test for the candidate and the next fastest clean policy |
| `pareto_latency_vs_confident_agreement.png`, `pareto_latency_vs_median_dp.png` | the Pareto plots (star = the sweep's fastest passing policy, cross = fails a gate) |
| `sweep_subprocess.log` | the subprocess output of every run |
| `work_log.md` | timeline and decisions |

## Method

- Gate (Appendix A.7, as stage 6): confident argmax agreement (CPU reference top-1 minus top-2 >= 0.10; 149 of the
  200 decisions) >= 98 percent; median over decisions of max abs dp <= 0.02; scorer-logit PCC over the gathered markers
  >= 0.99; hidden-state PCC (encoder output and head output, real positions, 40 cases pooled) >= 0.99; no NaN. Reported,
  not gated: plain argmax agreement, act-head argmax agreement, p95 and max of max abs dp. The 40 five-question gate
  calls ran at 5x256 (10 calls) and 5x512 (30 calls) for every policy.
- Latency: the four published cells on the final buckets (1x256, 5x256, 10x256, 50x256), end to end in process (input
  write, trace replay, two readbacks, host tail, temperature softmax), 3 warm calls and p50 of 20, one process per
  policy with the four buckets captured. Loads per cell are in the table (all under 8).
- Selection (plan): the fastest passing policy by the sum of p50 over the four cells; ties within 1 percent go to the
  higher confident agreement. Amendment A11: when the fastest passing policy clears any gate by less than 10 percent of
  the gate's width, it is confirmed on the full served workload against the runner-up (the shipped policy): served E2
  accuracy within 0.010 of the CPU fp32 row and soft accuracy, Brier, ECE and score MAE each within 0.015; E1 confident
  agreement over the 488-item corpus >= 98 percent and argmax agreement >= 95 percent. The stage 6 alone-versus-in-batch
  gate (same argmax, max abs dp <= 0.01 across the alone, B 2, B 4, mixed B 8 and B 64 placements of 16 questions) was
  run for the candidate because the stage 7 task requires the stage 6 gate scripts to pass on the shipped configuration.

## Sweep table (`sweep_results.json`; 200 gate decisions; latency in ms, p50 of 20)

| policy | A.7 gates | confident agree | plain agree | act argmax | median max abs dp | p95 | max | scorer PCC | hidden PCC enc / head | 1 q | 5 q | 10 q | 50 q | sum | loads (gate run; cells) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf8w_hifi3_erf` (shipped) | pass | 149 of 149 | 195 of 200 | 100 percent | 0.0110 | 0.0300 | 0.076 | 0.9965 | 0.9957 / 0.9996 | 9.42 | 22.97 | 40.65 | 193.32 | 266.4 | 2.5; 2.3 to 2.5 |
| `bf16_hifi4` | pass | 149 of 149 | 195 of 200 | 100 percent | 0.0085 | 0.0233 | 0.080 | 0.9975 | 0.9961 / 0.9997 | 10.56 | 25.25 | 44.85 | 214.20 | 294.9 | 2.0; 2.7 to 3.0 |
| `bf8w_hifi3` (upstream default, tanh) | pass | 148 of 149 | 193 of 200 | 100 percent | 0.0175 | 0.0561 | 0.122 | 0.9909 | 0.9903 / 0.9993 | 9.06 | 21.69 | 38.04 | 179.34 | 248.1 | 2.7; 2.4 to 2.5 |
| `bf8w_hifi2` | FAIL (hidden PCC 0.9898) | 148 of 149 | 193 of 200 | 100 percent | 0.0177 | 0.0503 | 0.120 | 0.9911 | 0.9898 / 0.9993 | 8.86 | 19.94 | 34.70 | 165.42 | 228.9 | 2.7; 2.6 to 2.9 |
| `bf8w_lofi_mlp` | FAIL (median 0.0251, scorer PCC 0.9852, hidden PCC 0.9842) | 147 of 149 | 191 of 200 | 100 percent | 0.0251 | 0.0741 | 0.110 | 0.9852 | 0.9842 / 0.9989 | 8.90 | 20.11 | 35.11 | 163.99 | 228.1 | 2.5; 2.4 to 2.5 |
| `bf8_act` | pass (thin margins) | 147 of 149 | 192 of 200 | 100 percent | 0.0188 | 0.0580 | 0.112 | 0.9904 | 0.9902 / 0.9992 | 8.69 | 19.72 | 34.55 | 165.54 | 228.5 | 4.0; 3.5 to 3.8 |
| `bf8w_hifi3_head_bf16` | pass | 148 of 149 | 193 of 200 | 100 percent | 0.0176 | 0.0567 | 0.123 | 0.9909 | 0.9903 / 0.9993 | 9.03 | 21.71 | 38.28 | 180.23 | 249.2 | 4.2; 3.7 to 4.1 |
| `bf8w_hifi2_erf` | pass | 149 of 149 | 196 of 200 | 100 percent | 0.0148 | 0.0389 | 0.093 | 0.9948 | 0.9938 / 0.9994 | 9.32 | 21.25 | 37.19 | 178.88 | 246.6 | 3.5; 3.0 to 3.3 |
| `bf8w_lofi_mlp_erf` | FAIL (scorer PCC 0.9893, hidden PCC 0.9887) | 149 of 149 | 192 of 200 | 100 percent | 0.0196 | 0.0572 | 0.148 | 0.9893 | 0.9887 / 0.9991 | 9.32 | 21.48 | 37.97 | 177.26 | 246.0 | 3.1; 2.5 to 2.7 |

NaN rows: 0 for every policy. Readings:

- The erf GELU is worth more than any weight or fidelity change, as stage 3 found: `bf8w_hifi3` to `bf8w_hifi3_erf`
  lifts the confident agreement from 148 to 149 of 149, the median from 0.0175 to 0.0110 and the scorer PCC from 0.991
  to 0.9965 for 7 percent of time; `bf8w_hifi2` fails only on the hidden PCC and `bf8w_hifi2_erf` passes with margin.
- The head policy (`bf8w_hifi3_head_bf16`) changes nothing measurable against `bf8w_hifi3`: the head and scorer are
  not where the error is (the hidden PCC after the head is 0.9993 for every bfp8 policy).
- LoFi for Wi and Wo does not hang on p150 (stage 3) but fails the gates with both GELUs.
- `bf16_hifi4` is the reference-quality row (median 0.0085) and the slowest (294.9 ms); with bf16 weights the
  block-sharded GeGLU plan overflows L1 at 1024 rows, so it runs the interleaved config everywhere (stage 7 README).
- Thin margins (under 10 percent of the gate width) of the passing policies: `bf8_act` on the median (0.0012 of 0.02),
  the scorer PCC (0.0004 of 0.01) and the encoder hidden PCC (0.0002 of 0.01); `bf8w_hifi3` and `bf8w_hifi3_head_bf16`
  on the scorer PCC and the encoder hidden PCC; `bf8w_hifi3_erf`, `bf16_hifi4` and `bf8w_hifi2_erf` have none.

Selection by the plan's rule: `bf8_act` (228.5 ms; no other passing policy within 1 percent). Its thin margins
triggered the A11 confirmation.

## Amendment A11 confirmation (served host TT backend, final buckets; `confirmation.md`)

Candidate `bf8_act` served with the full suite (`/home/hous/dev/laya/evals/results/host_tt_p150_b1_20261006T000056Z`,
load 1.2 to 2.5); runner-up `bf8w_hifi3_erf` served with the full suite (`.../host_tt_p150_b1_20261006T000400Z`, load
1.4 to 2.1; this is the build 1 run of record). CPU fp32 row: `.../cpu_reference_cpu_b0_20261005T210448Z`.

| gate | bf8_act | bf8w_hifi3_erf | threshold |
|---|---|---|---|
| E2 accuracy | 0.3605 (delta -0.0010, pass) | 0.3590 (delta -0.0025, pass) | within 0.010 of CPU fp32 0.3615 |
| E2 soft accuracy | 0.3312 (delta -0.0003, pass) | 0.3315 (delta +0.0000, pass) | within 0.015 of 0.3315 |
| E2 Brier (soft) | 0.3133 (delta -0.0022, pass) | 0.3109 (delta -0.0046, pass) | within 0.015 of 0.3155 |
| E2 ECE | 0.1730 (delta -0.0017, pass) | 0.1716 (delta -0.0031, pass) | within 0.015 of 0.1747 |
| E2 score MAE | 0.6903 (delta -0.0034, pass) | 0.6892 (delta -0.0045, pass) | within 0.015 of 0.6937 |
| E1 tensor path, confident agreement (488 items) | 99.50 percent (401 of 403) | 100.00 percent (403 of 403) | >= 98 percent |
| E1 tensor path, argmax agreement | 97.34 percent (475 of 488) | 97.54 percent (476 of 488) | >= 95 percent |
| E1 wire path, confident agreement | 99.50 percent (401 of 403) | 100.00 percent (403 of 403) | >= 98 percent |
| E1 wire path, argmax agreement | 97.34 percent (475 of 488) | 97.54 percent (476 of 488) | >= 95 percent |
| A11 confirmation | pass | pass | all of the above |
| stage 6 alone versus in batch (16 questions, 5 placements) | max abs dp 0.0381 (B 2 / B 4 0.0335, mixed B 8 0.0199, B 64 0.0381), same argmax 16 of 16: FAIL | 0.0090, 16 of 16: pass (stage 7 README) | <= 0.01 and the same argmax |

Other served numbers of the candidate run, for the record (not gates): E1 tensor path median max abs dp 0.0167 and
scorer PCC 0.9837 (shipped: 0.0081 and 0.9937); E2 agreement with the CPU reference decisions 0.9565 argmax and 0.9967
confident over 2,000 (shipped: 0.9775 and 1.000); E3 AG News 0.953 and Emotion 0.588 (shipped: 0.955 and 0.593; CPU
0.950 and 0.595); E5 client p50 10.3 / 22.8 / 39.5 / 185.3 ms and 213 to 271 questions per second (shipped: 11.0 /
26.5 / 46.0 / 212.9 ms and 187 to 231).

Decision: `bf8_act` passes the A11 gates and would have shipped under A11 alone, but it fails the stage 6 invariance
gate by a factor of 3.8. A served decision that changes by 0.038 depending on how many other questions share the call
is not acceptable for a calibrated decision model, and the stage 7 task requires the stage 6 gate scripts to pass on the
shipped configuration. The runner-up `bf8w_hifi3_erf` ships; `tt/model_config.py: DEFAULT_POLICY_NAME` is unchanged.
`bf8w_hifi2_erf`, the fastest policy with no thin margin (7.4 percent under the shipped policy), fails the same gate
(0.0289; `decision_agreement_bf8w_hifi2_erf.json`), so no faster policy is available without a change to the invariance
behaviour itself (the deltas come from different matmul blockings per bucket amplified by lower-precision
accumulation or bfp8 activation quantization).

`selected_precision_config.json`: `selected_policy` `bf8w_hifi3_erf`, `sweep_fastest_passing` `bf8_act`,
`default_changed` false, `env` {`LAYA_PRECISION` bf8w_hifi3_erf, `LAYA_SEQ_BUCKETS` 128,256,512, `LAYA_ROW_BUCKETS`
1,2,4,5,8,10,16,32,50,64}, the `confirmation` block above. Named profile candidates: `bf8w_hifi3_erf` (shipped),
`bf16_hifi4` (reference quality, 11 percent slower, no thin margin), `bf8w_hifi2_erf` and `bf8_act` (faster, fail the
invariance gate; a manifest profile could expose them for workloads that always send one question per call, where the
placement does not vary).

## What proved wrong or incomplete in the plan

- The Appendix A.7 gate list omits the stage 6 alone-versus-in-batch gate; the two fastest candidates pass A.7 (and
  one passes the A11 served confirmation) while failing it. The invariance test belongs in the sweep gate.
- `bf8_act` did not "break the head" as upstream saw for the MLM head: with the fp32 scorer output it passes every
  accuracy gate; its problem is placement dependence, which single-shape benchmarks never show.
- `bf16_hifi4` cannot run the block-sharded GeGLU plan (L1 clash with bf16 weights); the plan now declines for bf16
  weights.
- The plan expected the sweep to run on seq 512 cells; the final cells are 256-token buckets, and the latency ranking
  of the policies is the same at both (stage 3 measured the same ordering at 1x512, 8x512 and 64x512).

## How to run

```
source /home/hous/dev/laya/bin/ttenv.sh; cd $TT_METAL_HOME; A=models/autoports/convaiinnovations_laya; DL=/home/hous/dev/laya/bin/devlock
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/datatype_sweep.py --row-buckets 1,2,4,5,8,10,16,32,50,64 --seq-buckets 128,256,512 --hidden-cases 40 --threads 6 --skip-existing
TT_METAL_VISIBLE_DEVICES=0 $DL python $A/tests/decision_agreement.py --policy bf8_act --seq-buckets 128,256,512 --row-buckets 1,2,4,5,8,10,16,32,50,64 --out $A/doc/datatype_sweep/decision_agreement_bf8_act.json
python $A/tests/datatype_sweep.py --only confirm --candidate bf8_act --candidate-dir <served run of bf8_act> --runner-up bf8w_hifi3_erf --runner-up-dir <served run of bf8w_hifi3_erf> --invariance $A/doc/datatype_sweep/decision_agreement_bf8_act.json --row-buckets 1,2,4,5,8,10,16,32,50,64 --seq-buckets 128,256,512
```

The served runs: `LAYA_PRECISION=<policy> LAYA_SEQ_BUCKETS=128,256,512 LAYA_ROW_BUCKETS=1,2,4,5,8,10,16,32,50,64
LAYA_RAW_FORWARD=1 /home/hous/dev/laya/bin/devlock bash /home/hous/dev/laya/bin/serve-tt.sh`, then
`TARGET=host_tt PROFILE=p150 BUILD=1 BASE_URL=http://127.0.0.1:8710 RAW_FORWARD=1 bash /home/hous/dev/laya/bin/run-evals.sh`
(`/home/hous/dev/laya/scratch/t3_serve_evals.sh` wraps both and stops the server by PID).
