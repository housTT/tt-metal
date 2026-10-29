Use $stage-review as an independent reviewer for one TTNN model bringup stage.
You are the fresh review subagent. Do not spawn another reviewer. Do not modify
implementation files. You may run read-only local commands and small
artifact-analysis scripts over existing artifacts. Do not start servers, open TT
devices, reserve hardware, reset devices, or run hardware/vLLM experiments.
Read the skill from disk: /home/hous/dev/skills/plugins/tt-model-bringup/skills/stage-review/SKILL.md
(do not use the Skill tool for tt-model-bringup skills; it serves stale text).

Stage under review: 7, optimized full model
Target model: Qwen/Qwen3-8B
Autoport/evidence root: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b (live worktree, branch hous/qwen3-8b-autoport)
Hardware context (operator decisions, not negotiable): one Blackhole P300 chip (TT_VISIBLE_DEVICES=2) is the target topology; stages 4 and 5 (multichip) were skipped by the operator; chips 0 and 1 are wedged pending an operator reset; chip 3 belongs to another run. Evidence on chip 2 only.
Original stage goal contract: /home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_07-optimized-full-model.txt
Items carried into this stage from earlier reviews: (a) prefill RMSNorm and head ops on 64 to 1024-row chunks measured against full-grid or sharded alternatives; (b) LM head HiFi2 versus LoFi; (c) the chunk-boundary TTFT step; (d) the resumed-prefill check rerun at the runner's default thresholds on the final code (if it fails only on 32-position cache-hit windows from BF16 near-tie flips, that is recorded with margin probes and the lowered-threshold run kept beside it).

Stage skill(s) to compare against (read from disk):
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/optimize/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/full-model/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-enable-tracing/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/qualitative-check/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-device-usage/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/vllm-integration/references/prefill-features.md

Key artifacts to inspect first:
- models/autoports/qwen_qwen3_8b/doc/optimized_full_model/README.md and work_log.md
- models/autoports/qwen_qwen3_8b/doc/optimized_full_model/{perf_summary.json,ttft_sweep_before.json,ttft_sweep_after.json,probe_candidates.*.json}
- models/autoports/qwen_qwen3_8b/doc/optimized_full_model/{profile/,qualitative/,watcher/,logs/}
- models/autoports/qwen_qwen3_8b/doc/full_model/*.json (focused-test evidence regenerated on this stage's code; check provenance.checkout_git_sha and visible_devices)
- models/autoports/qwen_qwen3_8b/readiness/{readiness_stage7_chip2.json,prefill_resume_check*.json}
- models/autoports/qwen_qwen3_8b/readiness_autoregressive/, doc/optimized_full_model/degenerate_output_autoregressive.json
- models/autoports/qwen_qwen3_8b/tt/{model.py,generator.py,optimized_decoder.py}, tests/test_full_model.py (git log for the stage commits)
- previous stage: models/autoports/qwen_qwen3_8b/doc/full_model/README.md (before numbers), doc/optimized_decoder/README.md (decoder policy)

Review task:
1. Read the stage skill(s) and the stage evidence.
2. Compare the delivered artifacts against the goal contract.
3. Inspect generated outputs directly if present.
4. Look for bugs, oddities, contradictions, stale artifacts, weak dismissals, or
   suspicious behavior that hard checks might have missed.
5. Return the required verdict format. Do not summarize only; lead with
   findings.
