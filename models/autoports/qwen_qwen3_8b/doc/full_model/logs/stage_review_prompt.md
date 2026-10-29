Use $stage-review as an independent reviewer for one TTNN model bringup stage.
You are the fresh review subagent. Do not spawn another reviewer. Do not modify
implementation files. You may run read-only local commands and small
artifact-analysis scripts over existing artifacts. Do not start servers, open TT
devices, reserve hardware, or run hardware/vLLM experiments.

Stage under review: 6, full model (single chip; stages 4 and 5 were skipped by the operator, so "optimized multichip decoder" means models/autoports/qwen_qwen3_8b/tt/optimized_decoder.py on one Blackhole P300 chip, and single-chip is the target topology, not a fallback)
Target model: Qwen/Qwen3-8B
Autoport/evidence root: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b (live worktree, branch hous/qwen3-8b-autoport)
Original stage goal contract: /home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_06-full-model.txt (read it in full)

Review round: 4 (the operator granted one review beyond the two the goal allows). HEAD is 814e401a7b on branch hous/qwen3-8b-autoport (stage commits 2eaaa0b3f9 .. 814e401a7b over base db7a12e7c4). Earlier verdicts: logs/stage_review_1.md, stage_review_2.md, stage_review_3.md (all more-work-needed; review 3 left P1-A host fixes, landed in 6e3b243e25, and P1-B "every device rerun open"). Since review 3, every device rerun the reviews called unexecuted ran on chip 2 at 6e3b243e25 and is recorded in work_log.md section 9: the focused module (logs/pytest_full_model_36_rerun.log), the tracker gate (logs/pytest_full_model_36_tracked_rerun.log plus logs/pytest_full_model_36_tracked_batch32_rerun.log), run_prefill_check (logs/run_prefill_check_rerun.log), run_teacher_forcing (logs/run_teacher_forcing_rerun.log), the perf test (perf_batch1_128_128.json), the same-bucket batch-32 control (batch_decode_32.json, batch_decode_32.tracked.json), the corrected capacity probe (capacity_probe_budget.json, logs/capacity_probe_chip2.log) and its stage-by-stage twin (logs/capacity_bisect_chip2.log). Judge the evidence, not the prose written to answer earlier findings. *.log files are gitignored by the repo and live only in the worktree.

Hardware state the review must respect: chips 0 and 1 are wedged (pending operator reset); chip 3 is held by another run's process; chip 2 was the only usable chip. Do not open any device. If a finding needs hardware to confirm, report it as required follow-up.

Operator constraints that bind the review (recorded decisions, not the stage owner's theory):
- Serving stack pin is the nested plugin /home/hous/dev/vllm/plugins/vllm-tt-plugin at vLLM fork commit f96662a7d239c077e1435b658c0e7d316f9c4799; its docs/ holds only SCHEDULING.md at this pin.
- The stage-3 review carried one P2 (prefill RMSNorm and head-op grids on short chunks) into stage 7; it is out of scope here.
- The readiness runners open the mesh with trace_region_size=0; the focused tests reserve 512 MB.

Stage skill(s) to compare against:
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/full-model/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-enable-tracing/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-device-usage/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/qualitative-check/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/vllm-integration/references/prefill-features.md (serving contract and resume check requirements)

Key artifacts to inspect first (paths under the autoport root unless absolute):
- doc/full_model/README.md, doc/full_model/work_log.md
- doc/full_model/serving_contract.json, doc/context_contract.json
- readiness/prefill_resume_check.json (canonical) and readiness/prefill_resume_check_block32_bf16_logits.json, readiness/prefill_resume_check_block32_256_default_thresholds.json, doc/full_model/resume_margin_probe_*.json
- doc/full_model/logs/run_prefill_check_rerun.log, logs/run_teacher_forcing_rerun.log, readiness/readiness_rerun_chip2.json (chip-2 reruns at 6e3b243e25); logs/run_prefill_check_final.log, logs/run_teacher_forcing_final.log (2eaaa0b3f9), logs/run_autoregressive.log, readiness_autoregressive/, doc/full_model/degenerate_output_autoregressive.json
- doc/full_model/qualitative/*.json
- doc/full_model/split_sampling_trace.json, host_work_counters.json, tracked_replay_workload.json, seeded_sampling.json and the *.tracked.json twins (provenance.checkout_git_sha and provenance.visible_devices in each), logs/pytest_full_model_36_rerun.log, logs/pytest_full_model_36_tracked_rerun.log, logs/pytest_full_model_36_tracked_batch32_rerun.log, trace_alloc_tracking/positive_control.log
- doc/full_model/multi_row_prefill.json, batch_decode_8.json, batch_decode_32.json
- doc/full_model/perf_batch1_128_128.json, perf_batch1_128_128_force_argmax.json, profile/
- doc/full_model/capacity_probe_budget.json, logs/capacity_probe_chip2.log, logs/capacity_bisect_chip2.log, logs/capacity_probe.py, logs/capacity_bisect.py, logs/rerun_chain_chip2.status (UTC times), the hung first-probe logs logs/capacity_probe.log, logs/capacity_probe_chip0.log, triage/ (AUTODEBUG.md with erratum)
- doc/full_model/watcher/summary.json
- implementation: tt/model.py, tt/generator.py, tt/hf_weights.py, tests/test_full_model.py, and the one-argument change in tt/optimized_decoder.py (git diff db7a12e7c4 -- models/autoports/qwen_qwen3_8b/tt/)
- README.md (top level), doc/full_model/README.md status, headline, context and open-items sections, doc/full_model/work_log.md section 9

Review task:
1. Read the stage skill(s) and the stage evidence.
2. Compare the delivered artifacts against the goal contract.
3. Inspect generated outputs directly if present.
4. Look for bugs, oddities, contradictions, stale artifacts, weak dismissals, or
   suspicious behavior that hard checks might have missed.
5. Return the required verdict format. Do not summarize only; lead with
   findings.
