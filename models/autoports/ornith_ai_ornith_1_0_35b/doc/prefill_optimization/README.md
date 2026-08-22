# Ornith-1.0-35B prefill-throughput optimization

This stage promotes `C25-prefill-sdpa-qk128` as the model's default precision policy, adds true
device-side prefill batches of 1, 2, and 4 users, and adds the opt-in top-k-native routed-MoE prefill
path. C25 preserves every arithmetic, activation, KV-cache, and sampling choice from the selected
C06 datatype policy. Its only default-policy change is `prefill.sdpa_q_k_chunk: 256 -> 128`.

The selected production profile runs the native routed-MoE path over a **2,048-token sub-chunk span**
with `ORNITH_MOE_TOPK_NATIVE=1`, `ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048`, `ORNITH_MOE_GATHER` unset,
and `TT_MAX_PREFILLS_PER_STEP=4`. The concurrency-8 server additionally runs **four API-server
frontends** and a **2.0 s maximum prefill coalescing window**, so an arriving request wave is
admitted together and the model executes it as physical B4 device prefills instead of serialized
single-user ones. The window is a ceiling on one absolute deadline, not a fixed wait: a wave that
reaches the batch target exits coalescing immediately. The previous C25 plus gathered-MoE sweep is
the frozen baseline below. Each cell is one exact-length request wave with `ignore_eos` and the
matching compiled `max_num_seqs` of 1 or 8.

## Acceptance evidence

- C25 full-model accuracy passes both readiness paths: prefill top-1/top-5/top-100 is
  `0.92/1.00/1.00`; teacher-forcing accuracy is `0.93/1.00/1.00`.
- The 2,048-token native path matches the gathered reference on real weights: layer 0
  (`linear_attention`) PCC `0.999999868` and layer 3 (`full_attention`) PCC `0.999999665`.
- The isolated runtime gate executes exact physical B1, B2, and B4 waves at 2,048 input tokens. Its
  histogram is `{1: 1, 2: 1, 4: 1}`: 3 device invocations, 2 batched invocations, 7 logical users,
  14,336 logical tokens, and zero prefill fallbacks.
- The same gate proves the native MoE path actually ran at the 2,048-token span: **280 calls, 280
  subchunks, 120 layer calls, and zero fallbacks**, with 280 observed composites and every one of the
  40 layers ready. The subchunk count equals the call count because one 2,048-token sub-chunk covers
  the full span; the rejected 1,024-token arm needed two subchunks per call.
- The native span is faster than the gathered path on the same real layer: gathered `25.074 ms`
  versus native `9.995 ms`, ratio `0.3986`, over five samples each with under 0.2% spread.
- The four serialized post-install hardware gates all pass: full 40-layer B1/B2/B4 (1 passed,
  304.46 s), focused correctness (3 passed, 4 deselected, 43.77 s), promotion timing (1 passed,
  10.70 s), and the production fabric ring discriminator (1 passed, 2.81 s).
- Focused tests cover BF8 collective input, batch-8 geometry, mixed tails, state-pack restoration,
  top-k dispatch/combine geometry, cache address replacement, and fail-closed capability evidence.
- The final vLLM sweep completes all 10 requested cells with zero failed requests. Independent
  validation confirms exact request lengths, concurrency, token totals, median aliases, policy,
  capability state, native invocation counters, runtime binaries, source revisions, and immutable
  run inputs: `validated_cells: 10`, `strict_latency_provenance: pass`.

## Final vLLM latency sweep

Hardware is four Blackhole `p300c` chips in a `(1,4)` mesh. Concurrency 1 and 8 use separately built
servers with matching `max_num_seqs`. `tok/s/u = 1000 / median ITL`; aggregate throughput is
`concurrency * tok/s/u`; E2EL is median request latency. Parentheses show percent delta versus the
accepted C25 plus gathered-MoE sweep that immediately preceded this work.

| concurrency | ISL | OSL | tok/s/u | tok/s agg | E2EL |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 43.14 (+0.01%) | 43.14 (+0.01%) | 3.11s (+0.11%) |
| 8 | 128 | 128 | 31.69 (-0.45%) | 253.52 (-0.45%) | 4.73s (-9.89%) |
| 1 | 16384 | 512 | 41.67 (-6.12%) | 41.67 (-6.12%) | 16.01s (-10.31%) |
| 8 | 16384 | 512 | 29.77 (-7.73%) | 238.19 (-7.73%) | 44.98s (-33.09%) |
| 1 | 32768 | 512 | 40.62 (-5.93%) | 40.62 (-5.93%) | 20.47s (-16.56%) |
| 8 | 32768 | 512 | 28.26 (-6.70%) | 226.08 (-6.70%) | 75.16s (-37.69%) |
| 1 | 65536 | 512 | 38.49 (-5.79%) | 38.49 (-5.79%) | 30.46s (-21.41%) |
| 8 | 65536 | 512 | 25.13 (-6.69%) | 201.05 (-6.69%) | 139.83s (-40.47%) |
| 1 | 131072 | 512 | 35.00 (-5.20%) | 35.00 (-5.20%) | 53.91s (-23.67%) |
| 8 | 131072 | 512 | 20.91 (-5.10%) | 167.26 (-5.10%) | 285.27s (-42.15%) |

At concurrency 8 and 128K ISL, E2EL falls from the gathered baseline's 493.11s to 285.27s, saving
207.84s (-42.15%). Relative to the original sparse 610.08s result it saves 324.81s (**-53.24%**). The
median TTFT at that cell is 195.53s, down from the original 335.51s (-41.72%). Every long-context
concurrency-8 cell improves by 33% or more, and the cold B8 128/128 control now improves as well
(4.73s, -9.89%), which the superseded sweep did not achieve.

Decode throughput gives back 5-8% per user at long context. That trade is deliberate and it is the
signature of the change: the 2.0 s coalescing window and the four frontends hold a wave together so
prefill runs as B4 device work, and the concurrency-8 capability report confirms the result --- all
242 measured prefills executed at physical batch 4 (`histogram {4: 242}`), with zero fallbacks. The
aggregate long-context E2EL win is far larger than the per-user decode cost.

## Selected versus rejected sub-chunk span

Both spans were measured on identical frozen runtime revisions, the same harness, the same frozen
baseline, and the same 10-cell grid, so they compare directly.

| sub-chunk span | run ID | B8 128K E2EL | B8 128K TTFT | native counters | verdict |
|---|---|---:|---:|---|---|
| 2,048 | `final-topk-native-2k-throughput-shared-geometry-20260822T191635Z` | **285.27s** | 195.53s | 280/280/120/0 | **selected** |
| 1,024 | `final-topk-native-1k-throughput-selected-20260822T194934Z` | 348.08s | 242.27s | 560/560/120/0 | rejected |

The 2,048-token span is 62.81s faster at the decisive cell (-18.04%). The 1,024-token run is kept
intact as rejected apples-to-apples evidence; both runs carry a complete `FINALIZATION.json` with
`status: pass` and 10 validated cells. Note that the rejected run's directory name contains the word
`selected` --- that string was fixed in the run ID at launch, before the comparison concluded, and it
does **not** indicate the selection. The `verdict` column above is authoritative.

## Superseded earlier sweep

An earlier run, `final-topk-native-default-single-engine-20260821T231204Z`, reported 264.37s at the
B8 128K cell. That figure is **not** the reference result and must not be quoted as a regression
against the 285.27s above. It is not comparable:

- it was measured on the older runtime pair, tt-metal `0382df3379b4` and vLLM `ea902ddc49de`, before
  the shared runtime geometry work;
- its receipt is finalization schema `/1` and records no `run_id`, no pinned revisions, and no
  `topk_native_sub_chunk`;
- it has no `FINALIZATION.sha256` sidecar, so its receipt is not hash-anchored;
- its own receipt carries a `harness_correction` block recording that `run_latency_sweep.sh` changed
  after measurement (`hashes_differ: true`), so its artifacts were not produced by the current
  harness;
- its acceptance carried a genuine miss --- the cold B8 128/128 control at 6.307s against a 5.356s
  cap --- which the selected sweep clears at 4.73s.

## Reproducing the selected profile

Check out the paired revisions, reset the mesh, then launch the matching server. Always reset before
a launch or a benchmark.

```bash
git -C tt-metal checkout 824072e81e99af0cacb36adb6a33e271cd66c47f
git -C vllm     checkout a887998646dc4e6f192bce8d485bf89f4596ca2f
tt-smi -r
```

Common serving environment for both concurrencies:

```bash
export ORNITH_MOE_TOPK_NATIVE=1
export ORNITH_MOE_TOPK_NATIVE_SUB_CHUNK=2048
export ORNITH_VLLM_PREFILL_WARMUP=all
export TT_MAX_PREFILLS_PER_STEP=4
export TT_INTERLEAVE_PREFILL_CHUNKS=1
unset ORNITH_MOE_GATHER
```

Concurrency 8, the throughput profile --- four frontends and the 2.0 s coalescing window:

```bash
ORNITH_VLLM_PREFILL_PROFILE=throughput \
python -m vllm.entrypoints.cli.main serve ornith-ai/Ornith-1.0-35B \
  --api-server-count 4 --block_size 64 --max_num_seqs 8 --port 8100 --max_model_len 262144 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000, "l1_small_size": 32768, "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192, "input_queue_batching_delay": 2.0}}'
```

Concurrency 1, a single frontend and the default window:

```bash
python -m vllm.entrypoints.openai.api_server --model ornith-ai/Ornith-1.0-35B \
  --block_size 64 --max_num_seqs 1 --port 8100 --max_model_len 262144 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 200000000, "l1_small_size": 32768, "fabric_config": "FABRIC_1D_RING", "fabric_router_max_packet_bytes": 8192}}'
```

`input_queue_batching_delay` in `--additional-config` takes precedence over
`ORNITH_VLLM_PREFILL_PROFILE`; the sweep set both to the same 2.0 s so either route reproduces it.
Setting the profile alone is sufficient, since `throughput` resolves to the same 2.0 s window.

## Provenance

- Selected run ID: `final-topk-native-2k-throughput-shared-geometry-20260822T191635Z`
- C25 candidate: `../datatype_sweep/candidates/C25-prefill-sdpa-qk128.json`
- C25 full-model run: `../datatype_sweep/runs/C25-prefill-sdpa-qk128.json`
- Paired source revisions: tt-metal
  [`824072e81e99af0cacb36adb6a33e271cd66c47f`](https://github.com/housTT/tt-metal/commit/824072e81e99af0cacb36adb6a33e271cd66c47f);
  vLLM
  [`a887998646dc4e6f192bce8d485bf89f4596ca2f`](https://github.com/housTT/vllm/commit/a887998646dc4e6f192bce8d485bf89f4596ca2f)
- Supporting documentation commit: tt-metal
  [`5672083478cf7440b029ee5e074581d9d5697451`](https://github.com/housTT/tt-metal/commit/5672083478cf7440b029ee5e074581d9d5697451)
- `RUN_CONFIG.json` SHA-256: `ad5e1a0108536859d794df763f1411353e4085007fe2c811419065a6bc2c4b47`
- Raw 10-JSON set SHA-256: `8b1a8b7a6a5208aaf332cdb5a4cb05cece2d7e75f75cb22291e136237ec01e25`
- Frozen baseline 10-JSON set SHA-256:
  `be804048480d39a29316c5ad18db6f0de8a9e5c400425b6487a57d67b265157b`
- Final `RESULTS.md` SHA-256: `5fc2ee85bdf4182d2fb75f657be8578601a54c35b5da4fdd49df473da7e30b92`
- `FINALIZATION.json` SHA-256: `e98dd0f714ccdfa2c3b13f13df618fef87d8ee02f4e4fc637bcea1e48a70fdb5`
- Isolated runtime `VALIDATED.json` SHA-256:
  `d19875b7ea1305fb8529c60656699f7f6ac1285eb61c3973c3a4363ec72eb8e1`
- Rejected 1,024-token arm `FINALIZATION.json` SHA-256:
  `8e739d569084c857f89b32c8610ea56433b3d9b6f71b478c8d87fd81584c4b44`

The maxseq-8 evidence archive proves exact B1/B2/B4 device invocation and native top-k counters at
the 2,048-token span. The maxseq-1 warm/final pair shares runtime session
`1f4e54e2f28d44c0ac1d97993559658a` and binds exactly to the five B1 cells: 121 prefill calls all at
physical batch 1, 245,888 logical input tokens, 2,171 device-sampled decodes, 4,800 top-k
calls/subchunks, 4,800 layer calls, and zero fallbacks. The manifest schema records maxseq-1 `[1]`
and maxseq-8 `[1,2,4]` separately, so the capability requirement is capacity-aware.
