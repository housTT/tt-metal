# Stage 05: optimized-multichip-decoder

Target: ornith-ai/Ornith-1.5-9B, pinned revision489cb97981b8654bcfcf30ce1f94ed1b62e07b53. Optimize the completed decoder in place on the physical four-chip Blackhole P300c 1x4 mesh. No full-model or vLLM work. Original user requires optimize and tt-device-usage skills, all applicable optimizations tried with evidence, no deferrals, and stopping only after AutoFix fails or an external blocker requires operator intervention.

- Target-mesh measurements; no single-chip or replicated-weight fallback as completion.
- Accepted prefill/decode PCC for linear and full attention.
- Before/after warmed prefill and traced warmed decode from final default path.
- Advice-enabled tt-perf-report tables, CSV, logs and provenance; actionable advice tried, material rejections adapted and retried or AutoFixed.
- Initial operation-topology audit: repeated same-input matmuls, material collectives, reshard/layout conversions, packing, fused matmul-CCL, lower-movement residuals, actions and evidence.
- Coherent topology families: residual layout, collective placement, fused CCL/matmul, packing, activation/CCL dtype and persistent buffers. Sharded residual families measured without immediate restore to replicated boundary.
- Model-applicable async CCL, fused paths, collective placement, preallocation, residual/activation sharding, DRAM matmuls and reduced precision/fidelity all have evidence.
- Preserve native262144 context and batch1–32; update context_contract.json for allocation/layout changes. Any reduction requires physical evidence and largest feasible capacity.
- Own internal padding/masking/slicing for valid non-aligned logical lengths.
- No inter-layer collective/layout conversion unless measured faster overall; explicit residual interface contract for downstream full-model stage.
- Clean runtime fallback audit, appropriate stress, separate watcher-clean evidence.
- README.md/work_log.md include commands, before/after PCC/performance, CCL findings, rejected options, limitations and artifact links.
- MoE active-expert requirement does not apply: checkpoint has dense MLP.
- Fresh xhigh independent stage-review clean-pass, findings fixed and rereviewed. Local commits only, never push; log SHAs.

Skills: .agents/skills/optimize/SKILL.md, tt-device-usage/SKILL.md, autofix/SKILL.md, stage-review/SKILL.md. Repository/model AGENTS.md instructions also apply.
