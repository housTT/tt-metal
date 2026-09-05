# Gemma 4 26B-A4B-IT functional decoder

Status: functional-decoder gates pass on one Blackhole P300 chip. This stage
does not claim optimized-decoder, multichip, full-model, or vLLM readiness.

## Runtime contract

`tt/functional_decoder.py` implements one Hugging Face text decoder layer from
canonical checkpoint keys and supports every meaningful layer kind in the
target configuration:

| Layer kind | Representative layer | KV geometry | Attention contract |
| --- | ---: | --- | --- |
| sliding attention | 0 | 8 heads, dim 256, 64-token pages | 1024-token sliding window |
| full attention | 5 | 2 heads, dim 512, 128-token pages | global causal attention; natural or HMA-shared physical cache |

Prefill accepts `[batch, 1, seq_len, 2816]`, a paged KV cache, and page-table
rows. It serializes users on the single-device functional path, preserves the
logical length for non-tile-aligned inputs, and uses exact paged updates for a
1..31-token tail. Inputs longer than 32768 use chunked attention.

Decode accepts `[1, 1, batch, 2816]`, device-resident `current_pos[batch]`, a
paged KV cache, and page tables. A caller may keep these tensors at stable
addresses, update their contents, capture `decode_forward` once, and execute it
entirely through `ttnn.execute_trace`. Full-attention reads use one atomic
`ttnn.PagedCacheGeometryOverride(block_size=128, num_kv_heads=2)` so an
HMA-shared physical cache is interpreted consistently.

The complete tensor and cache contract is documented in the module docstring.

## Correctness evidence

The acceptance threshold is PCC >= 0.995. Canonical real checkpoint weights
were used for both representative layers.

| Gate | Sliding attention | Full attention |
| --- | ---: | ---: |
| real-weight prefill PCC, seq 32 | 0.999163 | 0.998457 |
| real-weight decode PCC, position 32 | 0.999739 | 0.999860 |
| traced batch-32 decode PCC vs HF | 0.999455 | 0.999860 |
| trace replay vs eager / repeat replay | 1.0 / 1.0 | 1.0 / 1.0 |
| lowest boundary prefill PCC | 0.995127 | 0.997776 |
| 32800-token attention PCC range | 0.996791..0.999091 | 0.999972..0.999983 |

Boundary coverage includes logical lengths 1/31/32/33, sliding pages
63/64/65, full pages 127/128/129, the sliding-window boundary
1023/1024/1025, the chunking cliff 32767/32768/32799, and advertised-context
lengths 262143/262144. Page tables are deliberately permuted in boundary and
capacity tests. The bounded sliding cache was also traced across positions
1023, 1024, 1025, and 1103 with PCC 1.0 against its unbounded counterpart.
An additional batch-32 A/B/A trace test overwrote stable hidden, RoPE,
current-position, page-table, and K/V-cache buffers between replays for both
layer kinds; eager-control PCC and repeat PCC were 1.0.

The HF-advertised 262144-token capability is retained. Real-weight full-layer
prefill completed at 262144 and at the non-aligned length 262143 for both layer
kinds. Advertised-context traced decode ran at current position 262143 with
device-initialized history, a rolled page table, sentinel readback, stable
current-position handling, and repeat replay PCC 1.0. See
`../context_contract.json`.

## Performance evidence

Warmed batch-1, sequence-1024 measurements on one P300 chip:

| Layer kind | Warmed prefill | Warmed traced decode |
| --- | ---: | ---: |
| sliding attention | 1,242.489 ms device / 1,243.350 ms host | 3.012 ms device / 3.112 ms host |
| full attention | 1,243.618 ms device / 1,244.512 ms host | 3.207 ms device / 3.304 ms host |

Human-readable tables, complete measured-phase CSVs, signposts, and commands
are under `perf/`. These are functional baselines only.

## Safety and fallback gates

- Static runtime audit: clean. A measured `prefill_forward` or
  `decode_forward` pass contains no `torch`, `ttnn.from_torch`,
  `ttnn.to_torch`, host-backed `ttnn.full`, or other host fallback, including
  the delegated canonical sparse expert prefill path. Non-aligned prefill tail
  indices use device-side `ttnn.moreh_full`.
- Determinism: eager-to-trace and repeated trace replay PCC are 1.0; the
  advertised-context repeat replay PCC is also 1.0 for both layer kinds.
- Watcher: the separate traced batch-1 run passed both layer kinds; the watcher
  log's fatal-pattern scan is empty. See `watcher/summary.json`.
- Real-weight inventory: `real_weight_stats.json` records name, real shape,
  dtype, element count, mean, standard deviation, range, and deterministic
  synthetic seed for every tensor used by representative layers 0 and 5.

## Reproduction

Activate the checkout environment first:

```bash
source python_env/bin/activate
```

Default functional suite:

```bash
python -m pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py
```

Advertised-context decode and real-weight prefill-capacity probes:

```bash
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k advertised_context_traced_decode

GEMMA4_PREFILL_CAPACITY_LENGTH=262144 python -m pytest -q -s \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k prefill_capacity_probe
```

The 262143 probes use the same command with
`GEMMA4_PREFILL_CAPACITY_LENGTH=262143`. The >32768 attention regression uses
`GEMMA4_LONG_ATTN_TEST=1`. Profiler commands are recorded in `perf/README.md`.

## Artifact index and limitations

- `pcc_layer*.json`: canonical real-weight HF-vs-TTNN prefill/decode PCC.
- `prefill_boundaries_*.json`: tile/page/window logical-length PCC matrix.
- `trace_*.json` and `trace_replay_pcc.json`: batch/current-position/page-table
  trace contracts and determinism.
- `trace_mutable_buffers_*.json`: batch-32 stable-buffer A/B/A replay with
  independently permuted per-user page tables.
- `prefill_capacity_*_{262143,262144}.json`: largest feasible real-weight
  prefill evidence.
- `advertised_context_decode_*.json`: 262144-context traced decode evidence.
- `long_prefill_attention_*.json`: chunking-cliff attention correctness.
- `triage/`, `AUTOTRIAGE.md`, and `AUTOFIX*.md`: failed-gate diagnoses and
  proven repairs.

The implementation is intentionally a correctness-first single-device layer.
Prefill batches are serialized, and no throughput or multi-device claim is
made. Full 262144-token capacity probes check successful finite output and
cache/page-table execution; the detailed HF-vs-TTNN PCC bar is established by
real-weight full-layer tests plus boundary and long-attention tests because a
full 262144-token HF reference materialization is not physically practical on
this runner.
