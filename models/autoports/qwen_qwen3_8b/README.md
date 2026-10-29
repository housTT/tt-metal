# Qwen/Qwen3-8B TTNN autoport (one Blackhole P300 chip)

Full model (stage 6), batch 1, real weights, greedy on-device split sampling:

| Metric | Value | Workload |
| --- | ---: | --- |
| TTFT | 80.5 ms (158-token AIME24 chat prompt); 54.5 ms (128-token book prompt) | prefill plus first sampled token (chip 2, commit `6e3b243e25`) |
| Teacher-forced decode, traced decode and sampler (readiness harness) | 22.3 ms/token (44.9 t/s/u) | AIME24 chat prompt, 100 forced tokens |
| Token-out decode, 128/128 | 21.86 ms/token (45.7 t/s/u), decode trace and sampling trace replayed every step (`sampler_traced true`) | 128 generated tokens |
| Readiness accuracy | prefill top-1 0.990 / top-5 1.000 / top-100 1.000; teacher forcing top-1 0.980 / top-5 1.000 / top-100 1.000 | AIME24 chat-template reference, 100 tokens |

Stage reports: `doc/functional_decoder/`, `doc/fused_decoder/`,
`doc/optimized_decoder/`, `doc/full_model/` (README, work log, serving
contract), `doc/context_contract.json`. Generator entry point:
`tt/generator.py` (`build_generator`, `readiness_check.contract.Generator`).
