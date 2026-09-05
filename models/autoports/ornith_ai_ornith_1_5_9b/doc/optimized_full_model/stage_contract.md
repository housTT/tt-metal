# Optimized full-model acceptance contract

User-requested stage: optimized-full-model for pinned ornith-ai/Ornith-1.5-9B,
after completed full-model and before datatype-sweep/vLLM. Use multichip,
optimize and tt-device-usage. Target is native1x4 TP on four Blackhole chips on
physical P300c boards; p150x4 is the software profile. No vLLM integration or
broad full-model datatype frontier search. Checkpoint revision:
489cb97981b8654bcfcf30ce1f94ed1b62e07b53.

Required scope and gates, transcribed from the user:

- Optimize embeddings, norms, LM head, logits, sampling, cache, trace replay,
  collectives, residual layouts, host boundaries and generator orchestration.
- Preserve canonical split sampling: sampler-ready logits, persistent tt_out_tok
  feedback, device position/RoPE advance, changed-only page tables, traced greedy
  and top-k/top-p-capable sampling.
- Preserve explicit cache/page-table/position/prompt-length/batch state, mixed
  prompts, fixed slots and inactive rows. Greedy decode stays on device; benchmark
  semantically correct sampler choices on this mesh. Fix split-greedy shape
  problems rather than accepting force-argmax.
- Preserve the completed decoder's selected dtype/fidelity/KV/activation/CCL
  policy, rejection ledger and inter-layer residual layout. Do not select faster
  rejected policies or change to a replicated stream. The actual predecessor's
  winning stream is replicated hidden4096 with B1 L1 width-sharding; preserve it.
- Preserve/improve context_contract.json. Reduce capability only for a hard
  physical limit with evidence and the largest feasible value. Preserve valid
  non-aligned prompt lengths through the public optimized generator.
- Refresh AIME24 chat-template prefill, teacher-forcing decode and autoregressive
  evidence. Standard run_prefill_check/run_teacher_forcing require top5>=98%
  and top100=100%. Run/read shared qualitative suite with controls.
- At the README top, report before/after warmed B1 TTFT and trace-verified t/s/u.
  Keep traced teacher forcing and token-out numbers separate when sampling,
  feedback and readback differ.
- Compute the decoder-stack lower bound from optimized multichip layer latencies.
  Token-out uses persistent token/position/RoPE/page inputs, nonblocking replay,
  no per-token host sync/readback, and persistent-buffer/CCL optimizations.
- If token-out exceeds layer-stack estimate plus terminal work by10–15%, split
  and close the largest avoidable gap. Dominant force-argmax, full-vocab gather,
  generic TopKDeviceOperation or other sampler bottlenecks must be fixed.
- Leave advice-enabled tt-perf-report tables, CSVs and provenance for the complete
  path. Follow optimize's reduced-layer profiling requirement, never all32 layers.
  Account for sharding, CCL, matmul, program-config and kernel checklist items.
- Clean runtime fallback audit; measured token-out fully traced with split
  sampling and no host boundary inside device replay.
- README and work_log record commands, before/after accuracy/performance,
  mesh/optimization decisions, perf conclusions, limitations and exact artifacts.
- Fresh independent xhigh stage-review must return clean-pass; findings are work
  and require fix/rereview. Local commit stage-owned changes, never push, log SHAs.
- Unmet requirements, failed gates and findings remain work; stop only after
  AutoFix fails or an unrecoverable external dependency prevents progress.

Skill sources: `.agents/skills/{multichip,optimize,tt-device-usage,full-model,
tt-enable-tracing,qualitative-check,autofix,stage-review}/SKILL.md`.
Autoport `AGENTS.md` and the supervising user's stage scope remain authoritative.
