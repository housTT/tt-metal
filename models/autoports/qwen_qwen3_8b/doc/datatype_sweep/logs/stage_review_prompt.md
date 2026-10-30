Use $stage-review as an independent reviewer for one TTNN model bringup stage.
You are the fresh review subagent. Do not spawn another reviewer. Do not modify
implementation files. You may run read-only local commands and small
artifact-analysis scripts over existing artifacts. Do not start servers, open TT
devices, reserve hardware, or run hardware/vLLM experiments. Read the skill from
disk at /home/hous/dev/skills/plugins/tt-model-bringup/skills/stage-review/SKILL.md
(do not use the Skill tool for tt-model-bringup skills; it serves stale text).

Stage under review: 8, datatype sweep
Target model: Qwen/Qwen3-8B
Autoport/evidence root: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b
Evidence directory: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/datatype_sweep
Branch/commit: hous/qwen3-8b-autoport, live worktree /home/hous/dev/tt-metal-wt-qwen3 (stage started at 2e52cebf61; stage commits listed in work_log.md section 7)
Hardware: one Blackhole P300 chip per process; chip 2 for every ranking run, chips 0 and 1 (released by the operator at 10:47 ET) for accuracy-only robustness runs. Stages 4 and 5 (multichip) were skipped by the operator; single chip is the target topology, not a fallback, so there is no CCL dtype.

Original stage goal contract (verbatim from the goal file
/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_08-datatype-sweep.txt):
- The baseline optimized full model accuracy and performance are refreshed using the main AIME24 chat-template readiness reference with 100 generated tokens.
- Full-model candidate results are captured in doc/datatype_sweep/sweep_results.json and sweep_results.csv, including config id, dtype policy, compute-fidelity policy, top-1/top-5/top-100, TTFT, trace-verified teacher-forcing decode t/s/u, measurement regime, command, hardware, mesh, and pass/fail status.
- The selected precision config is written to doc/datatype_sweep/selected_precision_config.json and is either the model default or a required config artifact that later full-model/vLLM construction paths actually consume by default. It must include weight groups, layer exceptions, compute fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, and logits/sampling dtype assumptions. The stage evidence must prove these fields are consumed by the measured runtime path; a JSON field ignored by hard-coded model code does not satisfy this requirement.
- Recompute doc/context_contract.json for every selected KV-cache dtype candidate that could change memory capacity. Do not complete after reducing advertised capability unless a hard physical device limit requires it and evidence records the largest feasible value.
- Datatype changes preserve valid non-aligned prompt-length support; if a KV-cache or trace-buffer dtype/layout changes chunking, rerun the non-aligned prompt check.
- Two pyplot charts exist: top1_perf_pareto.png and top5_perf_pareto.png. Each chart plots evaluated full-model configs, draws the Pareto frontier, marks the selected point in red, and shows a vertical dotted line at the minimum allowed accuracy.
- README.md and work_log.md record thresholds, selected config, rejected configs, Pareto interpretation, commands, plots, performance, accuracy, limitations, and exact artifacts.
- The final selected config is the fastest evaluated config that satisfies the full-model top-1/top-5 gate using traced teacher-forcing decode performance. Eager or untraced decode numbers are not valid for Pareto ranking or final selection. The sweep includes likely lower-precision wins and canonical policies, with evidence for rejected configs. For every material BFP4 matmul group considered or selected, the sweep includes a BFP4+LoFi candidate for that same group, or an exact TTNN/runtime blocker plus $autofix evidence.
- After selecting the winning config, run the same warmed token-out no-readback benchmark used by optimized full model through the normal selected-config construction path. Record it separately from teacher-forcing as post-selection token-out TTFT/decode t/s/u.
- $stage-review returns clean-pass; findings fixed/rereviewed. Local commit stage-owned touched-repo changes; never push; log SHAs.

Operator decisions that constrain the review (recorded facts, not my theory):
- Stage 3 rejected the BFP8 KV cache on a short-prefix per-user bar (0.9942); any sweep candidate that touches the cache dtype must include the short-prefix and batch-32 same-input cases.
- Hard wall-clock cap 8 hours from 10:00 ET (ends 18:00 ET).
- The supervisor pushes stage commits; the stage agent never pushes.
- No code comments in source (repository rule); docstrings are allowed.

Stage skill(s) to compare against:
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/datatype-sweep/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/tt-device-usage/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/qualitative-check/SKILL.md
- /home/hous/dev/skills/plugins/tt-model-bringup/skills/stage-review/SKILL.md

Key artifacts to inspect first (all under the evidence directory unless absolute):
- README.md, work_log.md
- sweep_results.json, sweep_results.csv, selected_precision_config.json, configs/*.json
- top1_perf_pareto.png, top5_perf_pareto.png
- readiness_c*_chip2.json (main-reference sidecars), readiness_*_serial_chip2.json (serialized re-measurements), readiness_*_aime_p1_chip1.json and readiness_*_book32_chip{0,2}.json (robustness references), readiness_selected_default_chip2.json
- propagation_check_reduced.json, propagation_check_selected_36.json
- batch32_same_input_*.json
- qualitative/ (prompt format, rendered prompts, HF control, TT outputs, verdict)
- profile/decode.txt, profile/prefill_128.txt
- logs/ (every command log, chain scripts, status files)
- /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/context_contract.json (datatype_sweep section)
- /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/full_model/*.stage8.json (post-selection module artifacts: perf_batch1_128_128.stage8.json is the token-out benchmark)
- /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/readiness/prefill_resume_check_stage8_default_thresholds.json
- Implementation: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/tt/precision.py, tt/generator.py (build_generator, teardown, max_tokens_all_users), tt/model.py (layer_policies, kv_bytes_per_token), tt/optimized_decoder.py (OptimizedPolicy.prefill_kernel), tests/test_full_model.py (_provenance)
- Stage 7 context: /home/hous/dev/tt-metal-wt-qwen3/models/autoports/qwen_qwen3_8b/doc/optimized_full_model/README.md, doc/optimized_decoder/README.md (stage-3 dtype ledger)

Review task:
1. Read the stage skill(s) and the stage evidence.
2. Compare the delivered artifacts against the goal contract.
3. Inspect generated outputs directly if present.
4. Look for bugs, oddities, contradictions, stale artifacts, weak dismissals, or
   suspicious behavior that hard checks might have missed.
5. Return the required verdict format. Do not summarize only; lead with
   findings.
