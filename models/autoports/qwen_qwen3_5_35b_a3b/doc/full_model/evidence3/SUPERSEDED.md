# Superseded

- Superseded by `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence4/`, which ran on the final source.
- `m1/sampling_audit.json` and `m4/sampling_audit.json` say `"passed": false`: the old exact top-32 support
  check failed on a BF16 tie at the 32nd candidate. The check is now tie-aware.
- `failed_audit_rerun1/`: p150x2 failed the old per-run check `sampled_b4_differs_from_greedy` by chance.
- `audit_rerun2/`: passed on the chain 3 source. The final source adds sampling validation and inactive-row
  handling, so evidence4 reruns it.
- Still cited as diagnostics: `hf_trace_prompt_margin.json` (HF nucleus check), `sampling_repeat/` (p150x2
  repeat test against TT probabilities).
