# Stage 6 work log (Track T2, UTC)

- 22:53 `tt/engine.py` written: `LayaEngine` with the server contract, env knobs of PLAN.md section 8, `host_tail`
  mirroring `DecisionModel.forward` after the gather, `act_head` rebuilt in fp32 from the checkpoint. CPU check:
  `host_tail` against `reference/laya_reference.py: _tail` on random inputs gives bit-identical logits and act logits.
  Decision: report `pad_rows_keep_one_token: False` (the device masks are -1e30, SDPA stays finite on an all-zero
  attention row, verified by T1); the server's pad rows then carry no live token, which is the plan's rule.
- 22:54 First fidelity run failed on a script bug (infinite recursion in the per-type breakdown); fixed, self-tested on
  CPU, relaunched 22:56.
- 22:58 `fidelity_bf8w_hifi3_erf_1x1.json` (log `p3_s6_fidelity_erf_20261005T225618Z.log`): every gate passes on the
  shipped policy; 100 calls at 64.6 ms device p50; 40 CPU reference forwards for the hidden states in 97.9 s at 6
  threads; load 1.6 to 1.7. Decision: `bf16_hifi4` not run (fallback only after a failed gate).
- 22:59 `decision_agreement.json` (log `p3_s6_invariance_erf_20261005T225844Z.log`), three placements: 16 of 16 same
  argmax, max abs delta p alone versus in-batch 0.0093 (gate 0.01), median 0.0026.
- 22:59 Server started under devlock (`p3_serve_tt_20261005T225920Z.log`): loaded in 6.6 s, sanity ok, seven traces
  warm. `server-smoke.sh` passed (`smoke_20261005T225920Z`). `run-evals.sh` with `TARGET=host_tt RAW_FORWARD=1` started
  (`p3_served_evals_20261005T225920Z.log`, results `host_tt_p150_b0_20261005T225931Z`). E1 tensor path reproduces the
  in-process gate numbers exactly (475 of 488, 403 of 403, median 0.00793, PCC 0.9956).
- 23:00 Orchestrator addition from review R1 received: B 2 and B 4 end-to-end rows and B 2 and B 4 placements in the
  invariance test. `run_fidelity.py --chunk N` and the `b2`, `b4` placements in `decision_agreement.py` added; the
  device jobs are queued behind the served evals (`scratch/t2_post_evals.sh`: stop the server by PID, then
  `s6_fidelity_erf_b2`, `s6_fidelity_erf_b4`, `s6_invariance_erf5`).
- Observations recorded for stage 8: the noul scorer PCC (0.988 on the gate subset) is a scale effect of saturated
  logits (deltas of 2 to 7 on logits of 9 to 19 move the probability by at most 0.024); the encoder hidden-state PCC
  (0.9955 pooled, 0.9912 worst case) is the gate with the smallest margin.
- 23:04 Served evals done (`host_tt_p150_b0_20261005T225931Z/SUMMARY.md`, rc 0): E1 tensor path equals the in-process
  run; E2 0.359 / 0.331 / 0.311 / 0.171 / 0.689; E3 0.953 / 0.593; E5 14.5 / 68.9 / 142.9 / 545.1 ms client p50, 70 to
  117 questions per second batched (every row padded to 512). Server stopped by PID (uvicorn 1342484, then the flock
  holder) at 23:04:27.
- 23:04 to 23:06 `s6_fidelity_erf_b2`, `s6_fidelity_erf_b4`, `s6_invariance_erf5` (five placements) all pass; B 2 and
  B 4 are bit identical to each other and at least as close to fp32 as bucket 8 on the same rows.
- 23:08 Token-length histogram of the served requests computed in the eval venv (`served_token_lengths.json`): E2 rows
  p50 308, E3 rows p50 103 and 51, E5 rows 194 to 205; recorded as the stage 7 seq-bucket input.
- 23:10 README sections for B 2 / B 4, the five-placement invariance and the served numbers written;
  `doc/context_contract.json` updated (engine, pad rule, mesh, bucket evidence, status).
