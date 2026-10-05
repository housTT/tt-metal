# Responses to review R1 (P1 and stages 1 to 3)

Review: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/doc/review/review_R1_p1_stages_1_3.md`
(verdict more-work-needed, one Required Work item at P2). Dispositions by the orchestrator, 2026 Oct 5 22:59 UTC.

## Orchestrator dispositions

| review item | disposition | owner |
|---|---|---|
| Required Work P2: stage 1 `layer_pcc.json` overwritten by the stage 3 run | fix: copy the shipped run to `doc/optimized_decoder/layer_pcc_shipped.json`, regenerate the stage 1 file with the stage 1 policy, policy-specific file names in `tests/test_ttnn_encoder.py` | Track T1 |
| Load rule statement in the stage 3 README | fix: state the real range and name the hot cells | Track T1 |
| Stale cross-reference in the stage 2 README; allocator warning undocumented | fix | Track T1 |
| Erf cost at B 2 and B 4; B 1 scatter about 5 percent | fix the README statements | Track T1 |
| Stale statements in the serving README (clamp rule, thread default) | fixed by the orchestrator in this round | orchestrator |
| E0.md "up to 8 questions each" | fixed by the orchestrator ("4 to 5 questions each, kmax 6") | orchestrator |
| Eager versus SDPA reference noise (1.3e-4 in scorer logits) | recorded here; passed to Track T2 so stage 6 does not mistake it for a device error | orchestrator |
| Hard-Check Gap: tracked replay over all seven buckets in one process | deferred to stage 7 (the plan's gate for all buckets); Track T3 | orchestrator |
| Hard-Check Gap: no end-to-end PCC at B 2 and B 4; invariance test covers B 8 and B 64 only | passed to Track T2: stage 6 adds B 2 and B 4 end-to-end rows and buckets 2 and 4 in `test_invariance.py` | Track T2 |
| Hard-Check Gap: Wo threshold pair at B 2 and B 4 confounded; GeGLU plan at B 2 and B 4 decided under tanh | accepted as residual (stake 1 to 3 percent at two buckets); a clean pair is queued for stage 7 if device time allows | orchestrator |
| Hard-Check Gap: `test_host_engine.py` had no log | closed by the review's own run (25 passed); later runs keep a log | orchestrator |
| Hard-Check Gap: policy table and fp32-residual conclusion rest on logs and scratch JSON | accepted: the logs under `/home/hous/dev/laya/logs/` are the evidence; the scratch per-layer files are copied under `doc/optimized_decoder/policy_probe/` by Track T1 if still present | Track T1 |

## Track T1 responses

(appended by Track T1)

- Required Work P2, shipped per-layer trace: copied the erf run (196 rows, `bf8w_hifi3_erf`, shipped port, L1 chain 4096) to `doc/optimized_decoder/layer_pcc_shipped.json` and referenced it from the correctness paragraph of `doc/optimized_decoder/README.md` (files: `doc/optimized_decoder/layer_pcc_shipped.json`, `doc/optimized_decoder/README.md`).
- Required Work P2, per-layer file name: `tests/test_ttnn_encoder.py` now writes `doc/functional_decoder/layer_pcc_<policy>_<port label>.json`; `tests/conftest.py` gained `LAYA_PORT=stage1` (selects `model_config.STAGE1_PORT`) and `port_label()`; `LAYA_LAYER_PCC_OUT` still overrides (files: `tests/test_ttnn_encoder.py`, `tests/conftest.py`).
- Required Work P2, stage 1 regeneration: job `s1_regen_layer_pcc` (`LAYA_POLICY=bf8w_hifi3 LAYA_PORT=stage1`, one `test_ttnn_encoder.py` run on chip 0 under devlock, log `/home/hous/dev/laya/logs/p3_s1_regen_layer_pcc_20261005T230058Z.log`) writes `doc/functional_decoder/layer_pcc_bf8w_hifi3_stage1port.json`; the stage 1 README table header, gate row and "How to run" now name that file; the mislabeled `doc/functional_decoder/layer_pcc.json` is removed once the regenerated file is verified against the README table (status line below).
- Required Work P2, stage 3 work log: the 21:53 entry now cites the stage 1 log and the regenerated stage 1 file for the tanh values 0.9805 and 0.9610, and `layer_pcc_shipped.json` for the erf run (file: `doc/optimized_decoder/work_log.md`).
- Load rule statement: `doc/optimized_decoder/README.md` A/B section now states 46 of 132 cells at or above 8.0, 12 at 9.3 to 10.5 with the six files named, shipped cells at 7.45 to 8.18.
- B 1 scatter: same section now says about 5 percent, from the two identical pairs (`sdpa_128` and `rotary_never_sharded` against `default` at 1x512, 13.28 against 13.92 ms) and C3 against `qkv_minimal_only` (11.86 against 12.28 ms).
- Erf cost at B 2 and B 4: policy section of `doc/optimized_decoder/README.md` states 21.01 / 36.71 ms against 18.94 / 32.12 ms (+10.9 / +14.3 percent), the Wo-lever confound, and that the sharded-GeGLU erf cost is not separately measured.
- Stage 2 stale cross-reference: `doc/fused_decoder/README.md` re-validation section now quotes the 22:32 UTC JSON (deltas 3.282 / 3.323 / 3.691, warmup 0.61 / 0.07 s, final shipped port) and names the superseded 22:24 run and its log.
- Allocator warning: one paragraph in the same section explains the once-per-process Metal `allocator.cpp:130` warning of the untracked multi-bucket benches (second-bucket trace outputs allocated while the first trace exists; tracked runs pass with no error and bit-identical replays).
- Policy probe evidence: the scratch per-layer and PCC files of the policy probes are copied to `doc/optimized_decoder/policy_probe/` with `INDEX.md` mapping each file to its policy and source log.
- Deferred to stage 7 (not done in this round): tracked replay over all seven buckets in one process; a clean Wo-threshold pair at B 2 and B 4 (shipped port, `wo_minimal_min_rows` 4096 against 0, same policy); the GeGLU plan at B 2 and B 4 under the erf policy.
- No shipped configuration was changed in this round (`DEFAULT_PORT`, `DEFAULT_POLICY_NAME` untouched).
- Status of the stage 1 regeneration (23:04 UTC): `s1_regen_layer_pcc` passed (9 tests, 25 s, `/home/hous/dev/laya/logs/p3_s1_regen_layer_pcc_20261005T230058Z.log`); `doc/functional_decoder/layer_pcc_bf8w_hifi3_stage1port.json` (196 rows, policy `bf8w_hifi3`, `STAGE1_PORT`) reproduces the stage 1 log to six digits at every shape (0.997114, 0.992017, 0.994863, 0.995009, 0.992006, 0.992817, 0.997302; worst layers 13 / 20 / 20 / 20 / 20 / 19 / 13) and the README per-layer table (layer 19: fill row 0.9666, B 8 0.9838, S 1024 0.9996); the mislabeled `doc/functional_decoder/layer_pcc.json` was byte-identical to `layer_pcc_shipped.json` and has been removed.
