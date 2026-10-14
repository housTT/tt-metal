# Superseded

- Superseded by `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/full_model/evidence5/`, which ran on the final source.
- Evidence4 ran on `tt/generator.py` before the review 3 slot fixes (`empty_slots`, `slot_remap`, live-slot guard).
- Evidence5 outputs are identical to evidence4: full evidence tokens, autoregressive, qualitative and calibration.
- Evidence5 adds the hidden-live-slot case and `batch_perf.json` per profile.
