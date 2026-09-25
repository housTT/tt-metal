# Final resident TP4+EP4 vLLM latency sweep

> Historical virtual-slot sweep. The selected batch-one path and current
> measurements are in `../batch1_optimization/README.md`.

Date: 2026-09-03

This is the deployment-gate endpoint sweep for the four-device `4x1`
TP4+EP4 implementation. All 512 routed experts are resident on TT device
DRAM, with 128 complete BFP4 experts on each device. Only the PLE n-gram
lookup and selected-row upload use the host during generation.

## Method

- Server: vLLM 0.24.0, OpenAI-compatible `/v1/completions` endpoint.
- Model: `Qwen/Qwen3.8-Flash-Next` at revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Hardware: two P300 boards / four Blackhole devices, mesh `4x1`.
- Requests: exact random token lengths, greedy temperature 0, EOS ignored.
- Each `vllm bench serve` invocation performed its unmeasured initial probe;
  the table contains only the subsequently measured requests.
- Decode tok/s/user is `1000 / mean TPOT ms`. Aggregate output throughput
  includes TTFT and request admission.

## Results

| ISL | OSL | Max concurrency | Requests | Median / p99 TTFT | Mean / p99 TPOT | Decode tok/s/user | Aggregate output tok/s | Median E2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 3/3 | 504 / 3,183 ms | 181.61 / 182.09 ms | 5.506 | 5.229 | 23.630 s |
| 1,024 | 128 | 1 | 3/3 | 3,818 / 6,136 ms | 181.70 / 182.19 ms | 5.504 | 4.625 | 26.952 s |
| 4,096 | 128 | 1 | 3/3 | 15,052 / 17,434 ms | 181.81 / 182.32 ms | 5.500 | 3.289 | 38.200 s |
| 128 | 128 | 2 | 4/4 | 1,955 / 2,745 ms | 368.36 / 368.36 ms | 2.715 | 5.253 | 48.736 s |

Concurrency 2 does not increase aggregate decode throughput over concurrency
1. The two virtual request slots preserve request state over the physical-B1
trace, but they do not form a physical batch-2 kernel. Use concurrency 1 for
the headline per-user decode value.

The first cold generation after server startup returned HTTP 200 in 128.451 s
and included TT program compilation and decode-trace capture, so it is not a
steady-state result. A later eight-token completion returned coherent text and
HTTP 200 in 3.363 s.

## Runtime placement audit

Post-sweep `QWEN38_VLLM_METRICS` reported:

- `resident_expert_layers=48`;
- `resident_expert_bytes_per_device=16,986,931,200`;
- `resident_expert_host_store_bytes=0`;
- expert H2D bytes, expert service time, and route-read/stall time all zero;
- PLE n-gram row lookup, host assembly, and selected-row H2D active as the
  declared host-backed path;
- attention/KV cache owned by vLLM and recurrent state owned by the model.

No expert projection, activation round trip, KV/recurrence fallback, host
sampling, or token-feedback reconstruction was reported.

## Raw artifacts

- `isl128_osl128_c1.json` — SHA256
  `0a2bf0398f763481a7a07c94eee4b6b84a7e309a8f1e76994f49e76a447dff74`
- `isl1024_osl128_c1.json` — SHA256
  `30b8350aa17b2bbb0cd1bc633d481b46ce856ef6f426d6b89b95648e9f4ee8cc`
- `isl4096_osl128_c1.json` — SHA256
  `041b1464cfeb37046aa44c0625c138642e01b7a27872d78d3340f3f2d6ccf4f9`
- `isl128_osl128_c2.json` — SHA256
  `87f7d49c984e9a904685c15560267d41dc3492e0fbb0d55a50986b6841e2f5d5`

Package, publication, pull, cached-restart, API, and digest evidence is recorded
in `DEPLOYMENT.md`.
