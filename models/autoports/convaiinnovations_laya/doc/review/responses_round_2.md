# Responses to review R2 (stages 4 to 8 and the host-served parity and evaluation)

Review: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/doc/review/review_R2_stages_4_8.md`
(verdict more-work-needed, one Required Work item at P2). Dispositions by the orchestrator, 2026 Oct 6 00:45 UTC.

## Orchestrator dispositions

| review item | disposition | owner |
|---|---|---|
| Required Work P2: `bf8w_hifi3` and `bf8w_hifi3_head_bf16` pass A.7, are 6.5 to 6.9 percent faster, and have no invariance measurement or A11 confirmation; README claims no faster policy exists | fix: measure invariance for both; A11 confirmation for any that passes; record in the sweep files; correct the README sentences and `cmd_confirm`; the orchestrator then decides the selection and records it as an amendment | Track T3, orchestrator |
| Stage 7 README misquotes the B 2 / B 4 invariance value (0.0087 instead of 0.00587) | fix | Track T3 |
| The `bf8_act` served run is named build 1 and matched by the context contract glob | fix: NOTE.md in the run directory, exact path in `doc/context_contract.json` | Track T3 |
| REPORT.md cites the removed `doc/functional_decoder/layer_pcc.json` | fixed by the orchestrator (now the regenerated stage 1 file) | orchestrator |
| Served overhead above device time (1.8 to 21 ms) | fixed by Track S before this review closed (server host path 17.6 to 2.0 ms at 50 questions, byte-identical wire output, commit `9727209227`); measured again on build 2 | Track S, orchestrator |
| About 140 MB of Tracy binaries committed on the branch | fix: binaries over 1 MB leave the git tree before the branch is pushed (kept under `/home/hous/dev/laya/evidence/tracy/`); the unpushed commits are rewritten so the push stays small | orchestrator |
| Hard-Check Gap: E1 served parity stores aggregates and the worst 20 only | fix: `parity/decisions.jsonl` written on every run from now on; the build 2 and later package runs carry it | Track E |
| Hard-Check Gap: 1x4 marker logits not stored; host-tail equality by inspection; latency samples not stored; consumed dtype verified for the shipped policy only; pooled hidden PCC | accepted as residual (the indirect checks reproduce; the gates pass) | orchestrator |
| Other concerns: unfused head ReLU, LayerNorm on 8 cores at 256 rows, elementwise share, placement spread 0.0116 between batched placements, manifest profiles for faster policies | accepted and documented; not pursued in this bring-up | orchestrator |

## Track T3 responses

(appended by Track T3)

## Track E responses

(appended by Track E)

## Track T3 responses

(appended by Track T3, 2026 Oct 6 00:50 UTC)

- Required Work P2, invariance of the two unmeasured faster policies: `tests/decision_agreement.py` on the final buckets
  (job `r2_invariance`, `/home/hous/dev/laya/logs/p3_r2_invariance_20261006T003901Z.log`, load 1.7 to 2.5):
  `bf8w_hifi3` max abs delta p alone against in batch 0.0150 (B 2 / B 4 0.0128, mixed B 8 0.0088, B 64 0.0150; same
  argmax 16 of 16), `bf8w_hifi3_head_bf16` 0.0154 (0.0129 / 0.0097 / 0.0154; 16 of 16); both fail the 0.01 gate, so
  no A11 served confirmation was run for them (files `doc/datatype_sweep/decision_agreement_bf8w_hifi3.json`,
  `decision_agreement_bf8w_hifi3_head_bf16.json`). All four A.7-passing policies faster than the shipped one now have
  an invariance measurement (0.0381, 0.0289, 0.0150, 0.0154); the shipped `bf8w_hifi3_erf` has 0.0090.
- Required Work P2, sweep files: `sweep_results.json` (`confirmation.policies` with the per-policy status in sweep
  order, `rule_selection_after_confirmation`), `sweep_results.csv` (columns `thin_margins`, `invariance_max_abs_dp`,
  `invariance_pass`, `a11_pass`, `a11_e2_accuracy`, `a11_e1_confident_rate`, `a11_e1_argmax_rate`, `status`),
  `selected_precision_config.json` (`confirmation` block with `sweep_order`, `runner_up`, `policies`, `rule_selection`,
  `decision_pending`), `confirmation.md` (A11 table plus the status table). `DEFAULT_POLICY_NAME` unchanged;
  `selected_policy` stays `bf8w_hifi3_erf`; `rule_selection` is `bf8w_hifi3_erf` with `decision_pending` false (the
  rule, with the A11 and A12 gates, now has no undecided faster policy).
- Required Work P2, `cmd_confirm` (`tests/datatype_sweep.py`): the runner-up is the next policy in sweep order
  (ascending latency sum among A.7-passing policies) unless `--runner-up` is given; served runs are passed as
  `--run NAME=DIR` (any number); every `decision_agreement_<policy>.json` in the sweep directory (and the stage 7 file
  for the shipped policy) is read; the script never changes the shipped policy.
- Required Work P2, README sentences: the result paragraph now states which policies were measured and their values
  ("every other A.7-passing policy that is faster than the shipped one was then measured on the invariance gate and
  fails it: `bf8w_hifi3` 0.0150, `bf8w_hifi3_head_bf16` 0.0154, `bf8w_hifi2_erf` 0.0289"); the decision paragraph
  states that the four faster policies were each measured and each fails, in place of "no faster policy is available
  without a change to the invariance behaviour itself"; the "only HiFi3 with bf16 activations" sentence is replaced by
  the measured list (`bf16_hifi4` is noted as not measured on this gate, being slower). A table of all invariance
  measurements (five placements, medians, PCC, loads) was added; the "manifest profile for the faster policies"
  suggestion was removed as the review asked.
- Stage 7 README invariance table: B 2 / B 4 corrected to 0.00587 (mixed B 8 0.00890, B 64 0.00897) from
  `decision_agreement_final_buckets.json`; the 0.0087 came from the 512-only protocol file.
- `bf8_act` run labelling: `/home/hous/dev/laya/evals/results/host_tt_p150_b1_20261006T000056Z/NOTE.md` written (A11
  candidate run, not build 1 of record, do not quote its E3, E5 or feed numbers as build 1);
  `doc/context_contract.json` now carries `served_results` with the exact directories
  (`build1_of_record_p150` = `host_tt_p150_b1_20261006T000400Z`, `build1_p150x4_e5_column` =
  `host_tt_p150x4_b1_20261006T000708Z`, `a11_candidate_run_bf8_act_not_build1` = `host_tt_p150_b1_20261006T000056Z`)
  and the status line names them.
- Not changed: `DEFAULT_POLICY_NAME` (`bf8w_hifi3_erf`), the manifests, any stage 1 to 6 document beyond the dated notes
  already recorded. Device time used for this round: 2 invariance runs (about 25 s). No commit.

## Orchestrator closure, 2026 Oct 6 00:55 UTC

The Required Work is closed by measurement: `bf8w_hifi3` and `bf8w_hifi3_head_bf16` fail the placement-invariance gate (max abs dp 0.0150 and 0.0154 against 0.01; `doc/datatype_sweep/decision_agreement_bf8w_hifi3.json`, `decision_agreement_bf8w_hifi3_head_bf16.json`), so every A.7-passing policy faster than `bf8w_hifi3_erf` fails invariance and the shipped policy is the rule's selection; no accuracy-preference amendment is needed. The Tracy binaries left the branch by a history rewrite of the unpushed commits (tip `d30b32e311` at the rewrite, old tip tagged `laya-pre-rewrite-20261006T0044`, copies under `/home/hous/dev/laya/evidence/tracy/`). Track E writes `parity/decisions.jsonl` from the build 2 run onward. Track S's host-path trim is in build 2 (served client minus device now 1.5 to 5.9 ms).
