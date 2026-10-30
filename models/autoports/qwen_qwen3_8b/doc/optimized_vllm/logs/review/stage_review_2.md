# Stage Review (review 2, fresh read-only subagent, HEAD 1ae8602de6, 2026 Oct 30 18:30 to 18:45 ET)

Verdict: more-work-needed

Every review-1 fix is verified against the artifacts, the contract items of the stage are met and re-derived from JSON and logs, no plugin or vLLM change exists, and nothing needs human authorization. One P2 remains: the stage added new device execution paths (batched prefill ops, the 1D multicast O and down kernels, the penalties-on sampling trace) and left no watcher evidence for any of them, which the `optimize` skill lists as a mandatory evidence item with an explicit vLLM-stage form. It is a short reduced-layer probe run, not a server run.

## Required Work

- P2: No watcher run covers the device paths this stage added.
  Evidence: no `watcher` or `TT_METAL_WATCHER` mention in the stage-10 README, work log, logs or sidecars. The last watcher run in this bringup is stage 7 (`doc/optimized_full_model/watcher/summary.json`: two-layer reduced model, `test_split_sampling_trace or test_host_work_counters or test_multi_row_prefill_equals_solo or test_tracked_replay_workload`, 0 fatal-pattern matches), which predates `_linear_batched` (2D and 1D multicast kernels over row pieces through the batch dimension), `_attention_prefill_batched` (`nlp_create_qkv_heads` on `[rows, 1, S, W]`, batched causal SDPA, `paged_fill_cache` with `batch_idx_tensor` over a table whose dummy rows and padded tails name the scratch block, `nlp_concat_heads` on the batched layout), the `stage10` solo table's 1D O and down kernels at 64 to 256 rows, and the penalties-on sampling trace with the counted `_run_sampling` warmup. The bit-equality evidence checks logits and the K/V pages of layers 0 and 35 at the real positions only; a stray NoC write into the scratch block, a padded tail or another page is invisible to it.
  Why this matters: `optimize` "Evidence To Leave" marks the watcher item MUST, names multi-row prefill in the battery, and states the vLLM-stage form (a watcher run of the changed path on a reduced-layer generator probe; not on the live server). The stage records no reason for skipping it.
  Required next step: one chip, no server, `TT_METAL_WATCHER=10` with the stage environment, following the stage-7 pattern: `QWEN3_FULL_MODEL_LAYERS=2 QWEN3_EVIDENCE_SUFFIX=watcher pytest -k 'test_batched_prefill_equals_solo or test_batch_decode or test_split_sampling_trace'` plus `logs/penalty_probe.py --layers 2`, each under `timeout`. Save the watcher log and a `summary.json` with the pattern counts under `doc/optimized_vllm/watcher/`, record the command and result in the work log and README section 8, and add a sidecar run entry.

## Other Concerns

- README section 1 and work log section 8 cite `prefill call=207 rows=31 ... ms=421.7` as the 32 x 128 burst's batched call. Call 207 is the CI serving burst's 31-row call (100-token prompts padded to the 128 bucket). The `prefill_burst` stage's calls are 224 and 226 (`ends=[128, ...]`, 422.0 and 421.8 ms). Same padded shape and time, wrong citation.
- README section 3.2 cites `prefill call=5 rows=1 ... ms=36.4` as "single-row 128-token prefill 36.5 ms per row". Call 5 is a 3-token prompt (bucket 32). The 128-token single-row calls in the same log are 205, 223 and 225 at 36.9, 37.4 and 37.1 ms, about equal to the stage-7 "about 37 ms", so the served-level sentence "the stage-10 solo table is faster" is not supported by that line; the generator-level pair and the generator TTFT 42.1 to 37.9 ms remain the evidence.
- `readiness_vllm/vllm_prefill_burst_benchmark.log` is the tracked2 run's log (median TTFT 487.41 ms) while the JSON beside it is the final untracked run (480.26 ms); `served_final/` holds no `.log` files. Restore the final run's log or note the mismatch.
- `logs/tracked_served2/sampling_smoke.log.status` is untracked in git.
- `test_mixed_params_batch` stays a known reproducibility-only failure; carry it to the next stage's work log explicitly.
- The warmup-time allocator warning ("Allocating device buffers is potentially unsafe due to the existence of an active trace", one line during `warmup_model_decode(enable_trace=True)`) is also present in stage 9's final server log; the tracked runs report no live buffer at replay. Not new to this stage.

## Hard-Check Gaps

- The stage gate exits 1 on the lowered-threshold resume advisory; the default-threshold run and the stage-7 margin probe are beside it. Gate-wording limit.
- The pinned plugin documents no `model_capabilities` key by name; `supports_device_penalties` is read at `platform.py` 745 to 747 with default True; the serving contract records this.
- Both tracked `prefix_cache` stages ran without the shared-prefix bench; the untracked final run carries it (hit rate 0.823).
- The burst serial reference is stage 9's file (operator-directed; `config_matches` true); the generator-level pair on the final code is the like-for-like control.
- The resume check's "identical to stage 9 to the digit" claim was not re-derived per start by the reviewer; both default files carry status `fail` under the same thresholds and the lowered run passes.

## Anomaly Ledger

- `test_mixed_params_batch` fails in the full, smoke, tracked and tracked2 runs with a different seeded row each time (full: request 0 `temp=0.5 seed=42`; smoke: request 2 `temp=0.5 seed=42`; tracked: request 1 `temp=1.0 seed=7`; tracked2: request 1 `temp=0.5 seed=42`); server-log calls 70 to 73 and 11 to 14 with the decode-mode windows. Affected path: plugin per-step routing for a batch holding a `top_k 100` request. Controls: greedy repeats and same-seed identical in the mixed-shape check, 71 other tests pass. Resolution: controlled (reproducibility-only); the record defect of review 1 is fixed.
- Resume check at default thresholds fails one gate (top-1 31 of 32 at start 16352); both controls fail at every window; stage-7 margin probe. Resolution: controlled.
- First batched kernel policy not bit equal above 256 dense rows; bisect isolated the down projection under `minimal_matmul`; fixed with the 2D/1D table (22 of 22 shapes bit equal). Resolution: fixed.
- Single-user TTFT moved more than 5 percent (49.299 to 44.592 ms, an improvement from the contracted 1D O and down kernels); TPOT 20.714 to 20.707. Resolution: controlled.
- Documentation cites call 207 and call 5 for numbers that belong to other calls. Resolution: documentation fix.

## Scope Inspected

Goal and skill paths; every artifact of the brief under `doc/optimized_vllm/`, `readiness_vllm/`, `doc/full_model/`, `readiness/`, stage-9 and stage-7 documents, the plugin checkout (`README.md`, `docs/`, `platform.py`, `model_runner.py`); code diff `534fb6fda1..HEAD`; read-only commands only; no device opened, no server started, no file modified. Review-1 fixes verified one by one (work log 8.1 and README sections 1, 5 (c), 6 against the four sampling logs and the server-log calls; commit table; 14 sidecar regime fields; `qualitative_verdict.json` prompt-format path; contract rows; the tracker log line at `logs/tracked_served2/server.log` line 150 in `EngineCore_DP0` with the same workload after it). Re-derived numbers: single-user TTFT p50 49.299 to 44.592 ms (minus 9.55 percent), TPOT 20.714 to 20.707 ms, ITL p50 20.674 to 20.677, throughput 47.754 to 47.856 tok/s; CI burst TTFT p50 1217.9 to 477.5 ms, TPOT 25.06 to 24.82, throughput 873.2 to 1094.4 tok/s, 32 of 32; burst 32 x 128 TTFT p50/p99/mean 480.26/481.05/467.00 ms against serial 1231.11/1231.87/1194.54, ratios 0.3901/0.3905/0.3910, `config_matches` true; tracked 0.4026/0.4034, tracked2 0.3963/0.3969; batched equals solo 7 cases untracked and 7 tracked, every row bit equal, program cache 645 (650 tracked) flat; batch decode identical for all 32 tokens; 22 shapes bit equal with gains 1.19x to 9.29x; served program cache 651 on all 6909 adapter lines, 33 calls with 31 batched rows, 189 penalised device steps; mixed shape 47 of 47 with 0 captures; prefix cache pass; sampling 71 passed, 1 failed, 1 skipped; qualitative greedy identical to stage 9; checkouts clean at `f96662a7d`. Push state: the supervisor pushes stage commits; the stage agent never pushes.

## Residual Risk

- The new batched device paths have no watcher coverage until the P2 run lands.
- Seeded requests sharing a batch with a request above the 32-candidate device top-k still change sampler mid-stream.
- Device penalties depend on the pinned plugin's per-step `prompt_tokens` / `output_tokens` and on the common sampler's private `_run_sampling(count_tokens=True)` warmup.
- Batched prefill covers fresh rows up to the 512 bucket with 2 to 32 rows (8 rows at 512, 16 at 256); longer prompts, prefix-cache hits and larger groups still serialize.
- Headline single-user numbers were measured beside two busy chips, the same regime as stage 9.
- The burst gate's denominator is stage 9's serial run on the stage-7 solo table; the generator-level pair on the final code shows the same ratio.
