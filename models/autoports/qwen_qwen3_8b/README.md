# Qwen/Qwen3-8B TTNN autoport (one Blackhole P300 chip)

Full model (stage 6), batch 1, real weights, greedy on-device split sampling:

| Metric | Value | Workload |
| --- | ---: | --- |
| TTFT | 54.5 ms | 128-token prompt, prefill plus first sampled token |
| Token-out decode, trace-verified | 21.79 ms/token (45.9 t/s/u) | 128 generated tokens |
| Teacher-forced decode | 21.81 ms/token (45.9 t/s/u) | same traces, forced next tokens |
| Readiness accuracy | prefill top-1 0.990 / top-5 1.000 / top-100 1.000; teacher forcing top-1 0.980 / top-5 1.000 / top-100 1.000 | AIME24 chat-template reference, 100 tokens |

Stage reports: `doc/functional_decoder/`, `doc/fused_decoder/`,
`doc/optimized_decoder/`, `doc/full_model/` (README, work log, serving
contract), `doc/context_contract.json`. Generator entry point:
`tt/generator.py` (`build_generator`, `readiness_check.contract.Generator`).
