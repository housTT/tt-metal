# Partly superseded

Superseded by `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/` (via evidence4, also superseded):

- `m*/sampling_audit.json`: says `"passed": true`, but ran with the defective sampler. Invalid.
- `m*/split_sampling_trace.json`: sampled leg ran with the defective sampler.
- `m*/full_evidence.json`, `m*/teacher_forcing_*.log`, `m*/prefill_check_aime24.log`, `m*/autoregressive/`,
  `qualitative/`, `degenerate_check.*`: rerun on the final source in evidence4 and evidence5 with identical greedy results.

Still current (no rerun on the final source; the final changes are host-side sampling and validation only):

- `m*/watcher_probe.*`: 2-layer watcher runs.
- `m*/profile/`, `m*/profile_run.json`: reduced-model Tracy profiles.
- `continuation/`: HF KL probe for continuation prefill.
- `slots/`: p150 32-slot build and its failed attempts.
