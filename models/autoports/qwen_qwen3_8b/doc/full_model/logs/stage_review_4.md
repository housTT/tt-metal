# Stage review 4 (fresh subagent, read-only, 15:19 to 15:36 ET), HEAD 814e401a7b

Operator-granted extra review (one beyond the two the goal allows). The
reviewer read `stage-review/SKILL.md` directly and the brief in
`stage_review_prompt.md`; no device was opened, no test was run, no file
was modified. Verbatim verdict follows.

# Stage Review

Verdict: more-work-needed

Round 4, HEAD `814e401a7b` on `hous/qwen3-8b-autoport`, live worktree. The device reruns that reviews 1 to 3 required all ran on chip 2 at `6e3b243e25`, every evidence JSON carries `provenance.checkout_git_sha 6e3b243e25` and `visible_devices "2"`, and the three source files hash to the `6e3b243e25` blobs (`generator_sha256 5204daef...`, `model_sha256 2d3c442b...`, `test_sha256 ee8d2f6a...` match `git show 6e3b243e25:` of each file; no code changed between `6e3b243e25` and HEAD). Accuracy, resumed prefill, split sampling, host-work counters, the tracker gate with program-cache allocations, the same-bucket batch controls, the corrected capacity probe and its bisect twin all pass on the evidence, not only on the prose. One coverage gap in a declared serving path remains, so the stage is not ready to close.

## Required Work

- P2: Decode buckets 16 and 24 are declared supported serving batch sizes but have never been captured, replayed or checked in the full model.
  Evidence: `tt/model.py` `DECODE_BUCKETS = (1, 8, 16, 24, 32)`; `tt/generator.py` line 132 builds `self.decode_buckets` from it and `prepare()` warms all five. `tests/test_full_model.py` `@pytest.mark.parametrize("batch", [8, 32])` is the only multi-user decode test. `doc/full_model/tracked_replay_workload.tracked.json` `traces."16".decode_traces: {}` and `traces."24".decode_traces: {}` with `counters.decode_trace_captures: 4` (bucket 1 device and host, bucket 8, bucket 32). `doc/full_model/serving_contract.json` row `tt_supported_decode_batch_sizes` cites as evidence only "each prepared before capture". The pinned plugin pads every decode step to the smallest declared bucket that holds the live users (`/home/hous/dev/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` lines 1219 to 1229, "Pad decode to the lane/rank wire capacity", positions padded with -1), so 9 to 24 concurrent users select these two untested execution signatures. The only earlier evidence is layer level: `doc/optimized_decoder/sharded_residual_full_attention.json` line 49 `"physical_batch": 16` (batch 13, one layer). Nothing anywhere in the autoport exercises physical batch 24 (an 8x3 core rectangle per `decode_physical_batch` in `tt/functional_decoder.py` lines 1022 to 1047).
  Why this matters: the SDPA decode work split, the fused cache update and the head create/concat ops change with the physical batch; `$tt-enable-tracing` requires every trace variant to be exercised in the generator's replay order, and the full-model skill defines `supported` as "evidence names the test or artifact". Warmup is not that evidence. A stage-9 server with 9 to 24 users runs these traces first.
  Required next step: add 16 and 24 to the `test_batch_decode` parametrization (the solo arm already decodes in the same physical bucket with -1 rows), run it untracked on chip 2 at HEAD (`QWEN3_FULL_MODEL_LAYERS=36 ... -k test_batch_decode`, about two minutes of device time per bucket), record `batch_decode_16.json` and `batch_decode_24.json`, and cite them in the `tt_supported_decode_batch_sizes` row. The tracked twin of the same run is optional for this stage, since allocation safety does not depend on the bucket width, but if it is run it needs `--timeout 1800` like batch 32.

## Other Concerns

- Continuation prefill (`start_pos > 0`) compiles per-offset programs after decode traces are live, and the tracked workload never exercises it. `tt/model.py` `prefill_hidden`: `compute_len = prefill_compute_len(seq_len) if start_pos == 0 else seq_len`; `tt/generator.py` `prefill_forward` lines 553 to 562 pass each continuation row at its exact length; `tests/test_full_model.py::test_tracked_replay_workload` has no nonzero `start_pos` call. The README limitations bullet 1 and the `supports_prefix_caching` row disclose it and the skill's `prefill-features.md` assigns the runtime-indexed fix to stage 9/10, so I do not count it as required work here. It is the top carry-forward risk: the first prefix-cache hit under live decode traces in stage 9 is a program-cache allocation of the kind `TraceCorrectness.md` calls unsafe. Stage 9 must run its mixed-shape trace-reuse check with continuation starts while other requests decode, and a cheap pre-check is one `prefill_forward(start_pos=68)` appended to `test_tracked_replay_workload` so the tracker says whether the compile actually trips it.
- Qualitative suite and free-running autoregressive evidence predate the generator changes. `doc/full_model/logs/qualitative_tt.log` ran 16:01 to 16:03 UTC and `doc/full_model/logs/run_autoregressive.log` 15:53 to 15:59 UTC, on the first build; `doc/full_model/qualitative/tt_counters.json` records `program_cache_entries_after_capture: 344` against 451 at `6e3b243e25`. `git diff --stat 2eaaa0b3f9 6e3b243e25` changed `tt/generator.py` by 250 lines (seed admission in the `generate` loop, per-row page tables, `rows_to_tile`). The outputs themselves are clean (every TT output coherent, English, `<think>` structure like HF, first divergence at tokens 34, 4, 32, 39, 40, 22; autoregressive 256 tokens coherent, degeneracy check exit 0), and the final code is covered by the 64-token coherent generation in `doc/full_model/host_work_counters.json`, the bit-equal batch controls and teacher forcing at `6e3b243e25`, so this is not required work. Rerun `logs/qualitative_tt.py 128` at HEAD in the same chip-2 chain as the bucket run (three minutes) or label both artifacts with their build in the README evidence table.
- The canonical resume check lowers `pcc_min_min` to 0.90 although the block-32 geometry never needs it. `readiness/prefill_resume_check.json` `thresholds: {pcc_mean_min 0.99, pcc_min_min 0.9, top1_min 0.93}`; every case has `pcc_min >= 0.9897`, and the only default-gate miss is `top1_agreement 0.96875` at `L=2560 s=2528` (one near-tie flip, margins 0.016 TT and 0.032 HF in `doc/full_model/resume_margin_probe_L2560_s2528_fp32.json`). The controls still fail both the PCC-mean and top-1 gates by a wide margin (`no_history` 0.78 to 0.93, `fresh_suffix` 0.934 to 0.955, top-1 0.67 to 0.91), so the check discriminates. Keep `--top1-min 0.93` with the margin evidence and restore `--pcc-min-min 0.98` on the next rerun; review 1 raised this and it is unchanged.
- The 2 GiB runtime reserve is not what remains after build. `doc/full_model/capacity_probe_budget.json` `dram_view_after_build.total_bytes_free_per_bank 253777280` x 8 banks = 2.03 GB = 1.89 GiB, below `RUNTIME_RESERVE_BYTES 2_147_483_648` (`tt/generator.py` line 811), because the formula subtracts only the KV pool and not the generator's own buffers (scratch block, zero rows, page tables, sampler state, allocator padding, about 120 MB). The 40960-token prefill and the bucket-8 decode fit in the 1.89 GiB, so the budget held; state the measured headroom in `doc/context_contract.json` `full_model.runtime_reserve_note` or subtract the owned buffers in `max_tokens_all_users`.
- `_redirect_table` clamps instead of raising. `tt/generator.py` line 344 `valid = min(width, math.ceil(valid_tokens[row] / self.block_size))`: a caller table narrower than the request's block count silently sends later K/V rows to the scratch block. The pinned plugin always sends full-width tables, so no current path is affected; `AUTODEBUG.md` section 7 item 1 already names it. Raise when the needed columns exceed the width.
- `Qwen3Generator.max_tokens_all_users(model=None, *, mesh_device=None)` (line 815) does not match the plugin hook's positional shape `get_max_tokens_all_users(model_name, num_devices, tt_data_parallel, max_model_len, max_num_seqs)` recorded in the serving contract row; the stage-9 wrapper must adapt it and must call `release_owned_cache()` before the budget is measured, or the owned 6 GB pool is counted as allocated. Reviews 1 and 2 recorded this; unchanged.
- Prompt lengths 4097 to 40959 compute whole 4096-token chunks (`prefill_compute_len` in `tt/model.py`), so TTFT steps up by a full chunk at every 4096 boundary (a 4097-token prompt costs an 8192-row prefill). Correct, warmed, and recorded as a stage-7 lever in README limitations bullet 2; serving TTFT at 4K to 8K prompts will show the cliff.
- Seeded requests run the sampler eagerly by the common sampler's contract: `doc/full_model/seeded_sampling.json` `host_work_seeded.seeded_eager_sampling_steps 22`, `sampling_trace_replays 0`. Reproducible and isolated from later unseeded requests (greedy after seeded: 11 trace replays, 0 eager), but a seeded served request pays eager sampling per token. Record it as a stage-9 performance expectation.
- Document nits: `doc/full_model/README.md` "Performance accounting" still says "Full model token-out: 21.79 ms per token" while the headline and `perf_batch1_128_128.json` say 21.86; `doc/context_contract.json` `full_model.hardware` says "readiness runners on TT_VISIBLE_DEVICES=0 and 1" although the recorded reruns ran on chip 2; `doc/full_model/qualitative/verdict.json` `overall` cites the book-text resume margin probes as the near-tie evidence for chat-prompt divergence (the teacher-forced AIME chat run is the control that applies).

## Hard-Check Gaps

- No HF-checked full-model output beyond prefill 16384 (resume check) and decode position 314 (autoregressive). The context-end decode at positions 40955 to 40958 is executed, not checked (`doc/context_contract.json` `full_model.layer_kinds.full_attention.full_context_decode_status: "demonstrated without an output check"`); the row-0 text " had eyes to see," is plausible book continuation only. Layer-level stage-1 to stage-3 checks at 40959 are the backing.
- The tracked workload has no continuation-after-capture case (see Other Concerns).
- `/home/hous/dev/skills/plugins/tt-model-bringup/scripts/check_serving_contract.py` checks structure only and does not verify that cited evidence paths exist; the `tt_supported_decode_batch_sizes` row passed with warmup as evidence.
- The tracker positive control (`doc/full_model/trace_alloc_tracking/positive_control.log`, "positive control raised as expected: Found 1 device buffer(s) still alive before trace replay") is a standalone 13-kernel script on chip 0, not the full-model process. The tracked batch-32 traceback in `doc/full_model/logs/pytest_full_model_36_tracked_rerun.log` (`UnsafeAllocationTracker(device).verify_before_replay`) does show the tracker active inside the full-model runs.
- Readiness runners open the mesh with `trace_region_size=0` (`/home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/mesh_device.py` `ttnn.open_mesh_device(mesh_shape=...)` with no trace region), so the teacher-forcing numbers were taken with two traces in dynamic trace storage; operator constraint, recorded.
- The gdb captures of the hung probes hold no Python frames; which of replays 1 to 4 stalled is unknown and stays unknown.
- No watcher run of the 36-layer model; `doc/full_model/watcher/summary.json` covers the two-layer reduced model only (10 dumps, zero patterns).

## Anomaly Ledger

- Observed anomaly: capacity probe hung on chip 1 (12:11 ET) and chip 0 (13:02 ET); both chips wedged afterward.
  Evidence: `doc/full_model/logs/capacity_probe.log` and `capacity_probe_chip0.log` (last line the allocator's "active trace" warning, `exit=143`); `doc/full_model/triage/capacity_probe_gdb_bt.txt`; `doc/full_model/logs/mesh_smoke_chip{0,1}.log` exit 124.
  Affected path: bucket-8 traced decode with row 0 at positions 40959 then 40960 to 40962 (advanced by `plus_one` on device), -1 rows, 3510-block pool.
  Control or comparison: corrected probe and stage-by-stage bisect at `6e3b243e25` on chip 2 (`doc/full_model/logs/capacity_probe_chip2.log`, four steps at 40955 to 40958, exit 0 in 137 s; `capacity_bisect_chip2.log`, every stage synchronized, tokens `[6414, 48409, 1817]`, `[311, 89852, 1008]` bit-equal to the probe).
  Likely subsystem: SDPA decode reader and fused KV update indexing the 1280-entry page-table stick at 1280 and beyond for positions at or past 40960 (`doc/full_model/triage/AUTODEBUG.md` section 2).
  Investigation performed: tt-triage (device health checks pass, no Inspector data), gdb, static `$autodebug` with the erratum appended, guard in `decode_forward` (line 641 rejects positions `>= 40960`) and in `generate` (line 762), corrected probe and bisect.
  Resolution: controlled. The out-of-contract step is now rejected on the host, the in-contract configuration runs clean, and decode at input position 40959 is unreachable from `generate` and from vLLM (a request at `max_model_len` finishes after prefill). Not a positive reproduction: the hung step was never re-executed and the code changed between the hung and clean runs, as the README states.

- Observed anomaly: resume check top-1 misses the 0.98 bar at the 32-position `prefix_cache_hit` windows.
  Evidence: `readiness/prefill_resume_check_block32_bf16_logits.json` reasons (0.9688 at 2528, 0.9375 at 16352); canonical fp32 run 0.96875 at 2528, 1.0 at 16352.
  Affected path: `prefill_logits_continuation` through `logits_all_rows`.
  Control or comparison: `doc/full_model/resume_margin_probe_L2560_s2528_fp32.json` (flip at 2531, ref margin 0.0156, continuation 0.0156, HF 0.0316; PCC 0.9999 at the hit); both controls fail at every 257-position window.
  Likely subsystem: BF16 near-tie at the LM head, not cache addressing or positions.
  Investigation performed: margin probes with BF16 and fp32 readiness logits, float32 readiness logits adopted.
  Resolution: controlled.

- Observed anomaly: one position at PCC 0.917 at the lcm-256 start.
  Evidence: `readiness/prefill_resume_check_block32_256_default_thresholds.json` reasons `pcc_min 0.9169` at `s=256`.
  Affected path: same as above.
  Control or comparison: `doc/full_model/resume_margin_probe_L2560_s256_fp32.json`: the fresh TT path is at 0.899 against HF at that position, the continuation at 0.942, all three top-1 equal; window means against HF 0.99066 and 0.99067.
  Likely subsystem: per-position precision floor of the BFP8/BFP4 LoFi policy, shared by both paths.
  Investigation performed: margin probe over the 2304-position window.
  Resolution: controlled; the floor itself belongs to the datatype-sweep stage.

- Observed anomaly: batch-32 user 18 diverged at token 6 (review 1).
  Evidence: earlier `batch_decode_32.json`; now `doc/full_model/batch_decode_32.json` `matching_tokens_per_user` 32 of 32 for every user, `first_divergence_per_user` all null, `solo_arm` "one live row decoded in physical batch 32 with 31 inactive rows at position -1".
  Affected path: bucket-32 traced decode.
  Control or comparison: same-bucket solo arm (the earlier arm compared bucket 1 against bucket 32).
  Likely subsystem: bucket-dependent accumulation order, not a cache or feedback defect.
  Investigation performed: test rewritten, rerun untracked and tracked on chip 2.
  Resolution: fixed (controlled by construction).

- Observed anomaly: sampler ran eagerly after the first seeded request, so earlier perf and trace evidence measured the eager path (review 1).
  Evidence: `doc/full_model/perf_batch1_128_128.json` `sampler_traced: true`, `counters.seeded_eager_sampling_steps 0`; `host_work_counters.json` 63 sampling-trace replays, 0 seeded eager steps; `seeded_sampling.json` isolation.
  Affected path: `_apply_sampling_params`, `_replay`.
  Control or comparison: seeded test runs last; greedy after seeded replays the trace.
  Likely subsystem: seed admission.
  Investigation performed: admission-time registration, counters split into traced and seeded-eager.
  Resolution: fixed.

- Observed anomaly: greedy TT output diverges from HF after 4 to 40 tokens (chat suite) and after 18 tokens (autoregressive), 22 of 256 tokens matching.
  Evidence: `doc/full_model/qualitative/verdict.json` `first_divergence_index`; `readiness_autoregressive/hf_completion.txt` and `tt_completion.txt`.
  Affected path: free-running `generate`.
  Control or comparison: teacher forcing top-1 0.980, top-5 1.000, top-100 1.000 on the same model path (`doc/full_model/logs/run_teacher_forcing_rerun.log`); both completions coherent and on topic; degeneracy check adjacent duplication 0.0, trigram loop 0.014.
  Likely subsystem: expected BF16/BFP drift at near-ties; no feedback, position or cache signature (no doubled tokens, no prompt echo, no language drift).
  Investigation performed: read both outputs, degeneracy check.
  Resolution: controlled.

- Observed anomaly: program cache grew after the first capture in the first build (344 to 387).
  Evidence: work log section 5; now `doc/full_model/tracked_replay_workload.tracked.json` `program_cache_entries` before 451, after 451 across a 3-row prefill, a 32-row prefill and an 8192-token generation after capture.
  Affected path: multi-row prefill, long prompts.
  Control or comparison: `prepare()` warms 17 prefill lengths and one 32-row call; `rows_to_tile` replaces the per-row-count concat.
  Likely subsystem: compile-time program keys.
  Investigation performed: warmup list extended, test assertion added.
  Resolution: fixed for row counts and chunk multiples; continuation starts remain open and are recorded for stage 9/10 (Other Concerns).

- Observed anomaly: chip-1 profile stall, chip 3 held by another run's process.
  Evidence: `doc/full_model/logs/profile_reduced.log` (exit 3 after SIGTERM) versus `profile_reduced_chip0.log` (`PROFILE_OK` in 30 s); `mesh_smoke_chip3.log`.
  Affected path: infrastructure.
  Control or comparison: identical command passed on chip 0.
  Likely subsystem: device state, not the model.
  Investigation performed: rerun on another chip; no reset issued per the operator rule.
  Resolution: controlled as infrastructure; chips 0 and 1 await an operator reset.

## Scope Inspected

- Goal/skill paths: `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_06-full-model.txt`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/stage-review/SKILL.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/full-model/SKILL.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-enable-tracing/SKILL.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-device-usage/SKILL.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/qualitative-check/SKILL.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/skills/vllm-integration/references/prefill-features.md`; `/home/hous/dev/skills/plugins/tt-model-bringup/prompts/model_bringup_multigoal/06-full-model.check.sh`; `/home/hous/dev/skills/plugins/tt-model-bringup/scripts/check_serving_contract.py`; `/home/hous/dev/skills/plugins/tt-model-bringup/scripts/check_context_contract.py`; `/home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/{contract,run_teacher_forcing,run_prefill_resume_check,run_autoregressive,mesh_device}.py`.
- Artifact paths (under the autoport root): `README.md`; `doc/context_contract.json`; `doc/full_model/README.md`; `doc/full_model/work_log.md` (sections 1 to 9); `doc/full_model/serving_contract.json`; `doc/full_model/logs/stage_review_{1,2,3}.md` and `stage_review_prompt.md`; `readiness/prefill_resume_check.json`, `prefill_resume_check_block32_bf16_logits.json`, `prefill_resume_check_block32_256_default_thresholds.json`, `readiness_rerun_chip2.json`; `doc/full_model/resume_margin_probe_*.json`; `doc/full_model/logs/run_prefill_check_rerun.log`, `run_teacher_forcing_rerun.log`, `run_autoregressive.log`, `qualitative_tt.log`, `generate_reference.log`, `generate_reference_wrapper.py`, `qualitative_tt.py`, `stage_env.sh`; `readiness_autoregressive/{autoregressive_meta.json,hf_completion.txt,tt_completion.txt}`; `doc/full_model/degenerate_output_autoregressive.json`; `doc/full_model/qualitative/{qualitative_prompt_format,rendered_prompts,hf_outputs,tt_outputs,verdict,tt_counters}.json`; `doc/full_model/split_sampling_trace.tracked.json`, `host_work_counters.json`, `host_work_counters.tracked.json`, `tracked_replay_workload.tracked.json`, `seeded_sampling.json`, `multi_row_prefill.json`, `batch_decode_{8,32}.json`, `batch_decode_{8,32}.tracked.json`, `perf_batch1_128_128.json`, `perf_batch1_128_128_force_argmax.json`, `capacity_probe_budget.json`; `doc/full_model/logs/pytest_full_model_36_rerun.log`, `pytest_full_model_36_tracked_rerun.log`, `pytest_full_model_36_tracked_batch32_rerun.log`, `capacity_probe.py`, `capacity_bisect.py`, `capacity_probe_chip2.log`, `capacity_bisect_chip2.log`, `capacity_probe.log`, `capacity_probe_chip0.log`, `rerun_chain_chip2.status`, `gate_06_full_model.log`, `gate_06_full_model_rerun.log`; `doc/full_model/trace_alloc_tracking/positive_control.log`; `doc/full_model/watcher/summary.json`; `doc/full_model/profile/decode.txt`, `ops_perf_results.source.txt`; `doc/full_model/triage/AUTODEBUG.md`, `capacity_probe_gdb_bt.txt`, `triage-summary-capacity-probe.txt`; `doc/optimized_decoder/sharded_residual_full_attention.json` (batch coverage grep).
- Code paths: `tt/generator.py` (all 876 lines); `tt/model.py` (all 489 lines); `tt/hf_weights.py`; `tests/test_full_model.py` (all 631 lines); `git diff db7a12e7c4 -- tt/optimized_decoder.py tt/functional_decoder.py tt/fused_decoder.py` (only the `rope_tables` loader argument, as claimed); `tt/functional_decoder.py` `decode_physical_batch`; `/home/hous/dev/tt-metal-wt-qwen3/models/common/sampling/generator.py` `format_sampling_params` (temperature 0 maps to temp 1, k 1, p 0, which matches the README's greedy description); pinned plugin `/home/hous/dev/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/{model_runner,async_decode,input_batch}.py` (`submit_prefill` kwargs, `decode_forward` kwargs, `reset_batch` on every persistent batch layout change, decode padding to the bucket with -1 positions; HEAD `f96662a7d239c077e1435b658c0e7d316f9c4799`, `docs/` holds only `SCHEDULING.md`).
- Commands run (read-only, inside `/home/hous/dev/tt-metal-wt-qwen3` for git): `git log --oneline db7a12e7c4..814e401a7b`, `git status --short --ignored`, `git diff --stat db7a12e7c4 814e401a7b`, `git diff --stat 2eaaa0b3f9 6e3b243e25 -- tt tests`, `git diff --stat 6e3b243e25 814e401a7b -- tt tests` (empty), `git show --stat 6e3b243e25` and `814e401a7b`, `git show 6e3b243e25:<file> | sha256sum` against `sha256sum` of the worktree files; `cat`, `grep`, `sed -n`, `ls -la` over the artifacts and logs; `python3 -I` one-liners that only load and print JSON. No device was opened, no test was run, no file was modified.

## Residual Risk

- Continuation prefill under live decode traces is the first thing stage 9's prefix-cache path will exercise, and it has no trace-safety evidence; the runtime-indexed prefill is the planned fix.
- The context-end hang was attributed statically and guarded, not reproduced; chips 0 and 1 stay wedged until an operator reset, and the one configuration never re-executed at full-model level (decode input position 40959) is unreachable through `generate` and vLLM but still accepted by `decode_forward`.
- Readiness numbers were taken with `trace_region_size=0` (dynamic trace storage with two live traces); the focused tests and the future server reserve a trace region, so the served numbers may differ slightly.
- Full-model decode output past position 314 and prefill past 16384 are checked only against earlier layer-level references; a long-prompt HF comparison belongs to the benchmark stage.
- TTFT steps up by a whole 4096-token chunk at each chunk boundary and the LM head stays at HiFi2 (about 2.15 ms of the 21.86 ms step); both are stage-7 levers.
- Seeded served requests sample eagerly per token by the common sampler's contract.
