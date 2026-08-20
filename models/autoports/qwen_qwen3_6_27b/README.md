# Qwen3.6-27B TTNN autoport

## Full-model status

The full `Qwen/Qwen3.6-27B` autoregressive path is ready on a `1x4` Blackhole
P300c mesh with TP=4 and `FABRIC_1D_RING`. On the fresh 100-token AIME24
chat-template reference, full-model prefill and traced teacher-forcing decode
both reach **97% top-1, 100% top-5, and 100% top-100**.

| Batch-1 metric | Result |
|---|---:|
| Full-model TTFT (teacher-forcing runner) | 14,620.75 ms |
| Warmed TTFT, prompt 128 in representative 128/128 run | 742.651 ms |
| Teacher-forcing traced decode | 19.22 t/s/u |
| Caller-visible autonomous token-out, prompt 128 / generate 128 | 20.323 t/s/u (49.205 ms/token) |
| Device-only model + canonical sampler trace pair | 22.445 t/s/u (44.553 ms/token) |
| Model trace only | 42.140 ms |
| Canonical Ring force-argmax sampling trace | 2.416 ms |

Teacher forcing includes the explicit host compatibility feedback required by
the accuracy harness. The caller-visible token-out number measures 127
autonomous decode steps after the prefill-selected token and includes the
sampled-ID readback returned by the public generator. It uses device-resident
greedy split sampling and device token feedback; no decode logits, host argmax,
or host token feedback participate. The device-only trace pair is retained for
attribution: sampling is 5.4% of that latency. The rejected common `Sampling1D`
greedy path selected the same token but took 10.798 ms.

The public context remains 262,144 total physical KV tokens. The calculated
maximum full-model plan is 19.535 GiB/device, including full weights, BFP8
paged KV cache, batch-32 linear state, persistent CCL buffers, and a 4 GiB
trace/activation/fragmentation reserve on each 32 GiB device.

Implementation and evidence are in [tt/model.py](tt/model.py),
[tt/generator.py](tt/generator.py), and [doc/full_model/README.md](doc/full_model/README.md).
This stage intentionally does not include vLLM integration.
