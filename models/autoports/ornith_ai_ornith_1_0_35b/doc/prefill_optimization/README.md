# Ornith-1.0-35B prefill-throughput optimization

This stage promotes `C25-prefill-sdpa-qk128` as the model's default precision policy, adds true
device-side prefill batches of 1, 2, and 4 users, and adds the opt-in top-k-native routed-MoE prefill
path. C25 preserves every arithmetic, activation, KV-cache, and sampling choice from the selected
C06 datatype policy. Its only default-policy change is `prefill.sdpa_q_k_chunk: 256 -> 128`.

The final sweep uses the production-default C25 policy with `ORNITH_MOE_TOPK_NATIVE=1`,
`ORNITH_MOE_GATHER` unset, and `TT_MAX_PREFILLS_PER_STEP=4`. The vLLM engine coalesces each arriving
request wave and the model executes synchronized prefill chunks as physical B1/B2/B4 device calls.
The previous C25 plus gathered-MoE sweep is the frozen baseline below. Each cell is one exact-length
request wave with `ignore_eos` and the matching compiled `max_num_seqs` of 1 or 8.

## Acceptance evidence

- C25 full-model accuracy passes both readiness paths: prefill top-1/top-5/top-100 is
  `0.92/1.00/1.00`; teacher-forcing accuracy is `0.93/1.00/1.00`.
- The isolated runtime gate executes exact physical B1, B2, and B4 waves at 2,048 input tokens. Its
  histogram is `{1: 1, 2: 1, 4: 1}`: 3 device invocations, 2 batched invocations, 7 logical users,
  14,336 logical tokens, and zero prefill fallbacks.
- The same gate proves the native MoE path actually ran: 560 calls, 560 subchunks, 120 layer calls,
  and zero fallbacks, with every one of the 40 layers ready.
- The combined C25 plus gathered-MoE 8K correctness gate reaches PCC `0.999159520` against HF for
  prefill and `0.999307333` for decode. C25 versus the prior selected policy reaches prefill PCC
  `0.999994355`.
- Focused tests cover BF8 collective input, batch-8 geometry, mixed tails, state-pack restoration,
  top-k dispatch/combine geometry, cache address replacement, and fail-closed capability evidence.
- The final vLLM sweep completes all 10 requested cells with zero failed requests. Independent
  validation confirms exact request lengths, concurrency, token totals, median aliases, policy,
  capability state, native invocation counters, runtime binaries, source revisions, and immutable
  run inputs.

## Final vLLM latency sweep

Hardware is four Blackhole `p300c` chips in a `(1,4)` mesh. Concurrency 1 and 8 use separately built
servers with matching `max_num_seqs`. `tok/s/u = 1000 / median ITL`; aggregate throughput is
`concurrency * tok/s/u`; E2EL is median request latency. Parentheses show percent delta versus the
accepted C25 plus gathered-MoE sweep that immediately preceded this work.

| concurrency | ISL | OSL | tok/s/u | tok/s agg | E2EL |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 42.91 (-0.53%) | 42.91 (-0.53%) | 3.12s (+0.32%) |
| 8 | 128 | 128 | 31.74 (-0.28%) | 253.94 (-0.28%) | 6.31s (+20.11%) |
| 1 | 16384 | 512 | 44.42 (+0.08%) | 44.42 (+0.08%) | 14.85s (-16.79%) |
| 8 | 16384 | 512 | 32.29 (+0.06%) | 258.31 (+0.06%) | 39.37s (-41.43%) |
| 1 | 32768 | 512 | 43.24 (+0.14%) | 43.24 (+0.14%) | 18.74s (-23.60%) |
| 8 | 32768 | 512 | 30.27 (-0.06%) | 242.17 (-0.06%) | 67.95s (-43.66%) |
| 1 | 65536 | 512 | 40.89 (+0.08%) | 40.89 (+0.08%) | 27.45s (-29.17%) |
| 8 | 65536 | 512 | 26.95 (+0.08%) | 215.63 (+0.08%) | 127.60s (-45.67%) |
| 1 | 131072 | 512 | 36.89 (-0.10%) | 36.89 (-0.10%) | 48.31s (-31.60%) |
| 8 | 131072 | 512 | 22.06 (+0.12%) | 176.47 (+0.12%) | 264.37s (-46.39%) |

At concurrency 8 and 128K ISL, E2EL falls from the gathered baseline's 493.11s to 264.37s, saving
228.74s (-46.39%). Relative to the original sparse 610.08s result, it saves 345.71s (-56.67%). The
median TTFT is 148.03s, down from the original 335.51s (-55.88%), while decode throughput remains
flat against the gathered baseline. This is the intended signature of removing serialized prefill
work rather than trading away token-out performance.

All throughput floors and long-context E2EL gates pass. The sole formal miss is the cold B8
128/128 control: 6.307s against a 5.356s cap. All eight requests arrived within 0.826ms and the first
B4 device prefill began less than 24ms later, so the 250ms coalescing deadline was not paid. The
extra second is attributable to cold B4x128 program compilation, one recurrent-state remap, and two
decode-trace recaptures. Future warmup should cover B2/B4 at the 128-token tail and precompile the
non-identity eight-row remap; this sweep deliberately remains the measured artifact rather than a
post-hoc adjusted result.

## Provenance

- Final run ID: `final-topk-native-default-single-engine-20260821T231204Z`
- C25 candidate: `../datatype_sweep/candidates/C25-prefill-sdpa-qk128.json`
- C25 full-model run: `../datatype_sweep/runs/C25-prefill-sdpa-qk128.json`
- Paired source revisions: tt-metal `0382df3379b4bd5ddd09504b21ae411df7c61ab0`; vLLM
  `ea902ddc49deff61777679c71108b2b299548fd5`
- `RUN_CONFIG.json` SHA-256: `48f8256496c646176b1e40b095c18f262c9851e6bd7de7677e197e3c076c76bc`
- Raw 10-JSON set SHA-256: `f8fb730ad1563cbbc8760e9ef0d2d0202e846caeed892d9aa66bfce0b291a96f`
- Final `RESULTS.md` SHA-256: `22486dd0ef95a46be41c1a4a0e8990e00b7af511865e577a0fecaabb807e45a1`
- `FINALIZATION.json` SHA-256: `134872eea7d0c1037e7c750789f1031ecff3391f9dc546f12f584206b45e0220`
- Isolated runtime `VALIDATED.json` SHA-256:
  `6dcc05ab9d1584e05d07be78d80d6bdca343a33178d098c1b412da5447a19cd7`

The maxseq-8 evidence archive proves exact B1/B2/B4 device invocation and native top-k counters.
The maxseq-1 warm/final pair shares runtime session `a32b2522155f482c8b1fc7df6d293439` and binds
exactly to the five B1 cells: 121 prefill calls, 245,888 logical input tokens, 2,171 device decodes,
9,600 top-k calls/subchunks, 4,800 layer calls, and zero fallbacks. A post-measurement harness defect
had incorrectly required maxseq-1 to advertise maxseq-8's `[1,2,4]` capability. Finalization copied
the already-written pair without rerunning hardware, recorded that correction explicitly, and the
next manifest schema now records maxseq-1 `[1]` and maxseq-8 `[1,2,4]` separately.
