Use $stage-review as an independent reviewer for one TTNN model bringup stage.
You are the fresh review subagent. Do not spawn another reviewer. Do not modify
implementation files. You may run read-only local commands and small
artifact-analysis scripts over existing artifacts. Do not start servers, open TT
devices, reserve hardware, or run hardware/vLLM experiments.

Stage under review: 6, full model (single chip; stages 4 and 5 were skipped by the operator, so "optimized multichip decoder" means models/autoports/qwen_qwen3_8b/tt/optimized_decoder.py on one Blackhole P300 chip, and single-chip is the target topology, not a fallback)
Target model: Qwen/Qwen3-8B
Autoport/evidence root: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b (live worktree, branch hous/qwen3-8b-autoport)
Original stage goal contract: /home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_06-full-model.txt (read it in full)

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
- doc/full_model/logs/run_prefill_check_final.log, logs/run_teacher_forcing_final.log, logs/run_autoregressive.log, readiness_autoregressive/, doc/full_model/degenerate_output_autoregressive.json
- doc/full_model/qualitative/*.json
- doc/full_model/split_sampling_trace.json, host_work_counters.json, tracked_replay_workload.json and the *.tracked.json twins, logs/pytest_full_model_36.log, logs/pytest_full_model_36_tracked.log, trace_alloc_tracking/positive_control.log
- doc/full_model/multi_row_prefill.json, batch_decode_8.json, batch_decode_32.json
- doc/full_model/perf_batch1_128_128.json, perf_batch1_128_128_force_argmax.json, profile/
- doc/full_model/capacity_probe_budget.json (if present), logs/capacity_probe*.log, triage/
- doc/full_model/watcher/summary.json
- implementation: tt/model.py, tt/generator.py, tt/hf_weights.py, tests/test_full_model.py, and the one-argument change in tt/optimized_decoder.py (git diff db7a12e7c4 -- models/autoports/qwen_qwen3_8b/tt/)

Review task:
1. Read the stage skill(s) and the stage evidence.
2. Compare the delivered artifacts against the goal contract.
3. Inspect generated outputs directly if present.
4. Look for bugs, oddities, contradictions, stale artifacts, weak dismissals, or
   suspicious behavior that hard checks might have missed.
5. Return the required verdict format. Do not summarize only; lead with
   findings.
