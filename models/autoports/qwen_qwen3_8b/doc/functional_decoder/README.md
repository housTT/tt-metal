# Qwen3-8B functional decoder

Status: functional-decoder gates pass on one Blackhole P300 chip (1x1 mesh).
This stage claims nothing about optimized-decoder, multichip, full-model or
vLLM readiness.

Abbreviations: HF (Hugging Face), PCC (Pearson correlation coefficient),
KV (key/value), RoPE (rotary position embedding), SDPA (scaled dot-product
attention), MLP (multi-layer perceptron), DRAM (device dynamic RAM), L1
(per-core local memory), GQA (grouped-query attention).

## Runtime contract

`tt/functional_decoder.py` implements one HF `Qwen3DecoderLayer` from the
canonical checkpoint keys. Qwen3-8B has one layer kind:

| Layer kind | Representative layers | Geometry | Attention |
| --- | ---: | --- | --- |
| full attention | 0, 17 (of 36) | 32 query heads, 8 KV heads (GQA), head_dim 128, per-head Q/K RMSNorm, RoPE theta 1e6 | global causal, paged KV cache with 32-token blocks |

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
mesh_device, weight_dtype=bfloat16, activation_dtype=bfloat16,
tensor_cache_path=None, prefill_chunk_size=4096)` accepts full
`model.layers.{i}.` keys or layer-local keys, validates every shape against
the config, fuses QKV, and builds the RoPE cos/sin tables for all 40960
positions on device.

`prefill_forward(hidden_states, *, page_table, kv_cache, user_id=0,
start_pos=0, seq_lens=None)`: `hidden_states` is `[B, 1, S_max, 4096]`;
each row carries `seq_lens[b]` valid tokens of any positive length, starts at
absolute position `start_pos[b]` (0 for a fresh request, otherwise an explicit
continuation over a cache that already holds `[0, start_pos[b])`), and writes
page-table row `user_id[b]`. Rows run one at a time in 4096-token physical
chunks; tile-unaligned starts are front-padded on device and the rows that
share a cache block with live prefix rows are written token by token so the
prefix is never overwritten. Output is `[B, 1, S_max, 4096]` with zero padding
past each row's logical length.

`decode_forward(hidden_states, *, current_pos, page_table, kv_cache)`:
`hidden_states` is `[1, 1, B, 4096]` for `1 <= B <= 32`, `current_pos` is a
device int32 `[B]` tensor that drives cache update, attention length and the
on-device RoPE lookup, `page_table` is `[B, blocks]` int32. The decode ops map height-sharded
users to cores in rows of 8 (measured: 8-wide rectangles and single rows up
to 8 users are exact, other rectangles corrupt users outside an 8-wide row
mapping; see `tt_metal_decode_grid/`), so up to 8 users run as one row and
a larger batch runs as `round_up(B, 8)` users on `8 x n` cores; the padded
users get position `-1`, which makes the cache update and attention skip
them, a zero page-table row, and their rows are sliced away before
returning.
The method is trace-safe: capture once over stable buffers and replay with
`ttnn.execute_trace` after refreshing their contents.

`kv_cache` is a caller-owned `(key, value)` pair of `[num_blocks, 8,
block_size, 128]` TILE DRAM tensors (`FunctionalDecoder.allocate_kv_cache`
zero-fills one on device). The full tensor and cache contract is in the module
docstring.

### Kernel constraint handled inside the decoder

The chunked prefill SDPA and the paged decode SDPA pad their K window up to a
multiple of their chunk size and walk the page table that far. Entries past
the row end come from stale memory, and non-finite data in the padded region
corrupts every query row even though those keys are causally masked
(`tt_metal_sdpa_padded_k/` holds deterministic reproducers for both kernels
and the in-decoder sequence that first exposed it). The decoder therefore
chooses the prefill chunk so `chunk_start + round_up(len, chunk) <=
page_table_extent` and the decode K chunk as the largest power of two (<= 128)
that divides the page-table extent. Callers get the fastest chunks when the
page-table extent is a multiple of 256 positions (8 blocks of 32); any
multiple of 32 is correct. The cache must hold finite data in every block a
page table references.

## Correctness evidence

Acceptance bar: PCC >= 0.995 against a float32 HF layer reference. All tests
also report the PCC of the layer contribution (`output - input`, "delta"),
because the residual passthrough makes whole-output PCC insensitive to
attention errors. Real checkpoint weights were used for every row below;
layer-0 inputs are real text (tale of two cities) through the real embedding,
layer-17 inputs are the real hidden states captured after HF layers 0..16.

| Gate | Layer 0 | Layer 17 | Artifact |
| --- | ---: | ---: | --- |
| real-weight prefill PCC (1024 / 1000 tokens) | 0.999982 (delta 0.999981) | 0.999999 (delta 0.999424) | `pcc_layer{0,17}_full_attention.json` |
| real-weight decode PCC, next two positions | 0.999993, 0.999986 | 0.999988, 0.999991 | same |
| repeated identical prefill bit-equal | yes | yes | same |

Layer-0 coverage (`prefill_boundaries_full_attention.json`, permuted 4-user
page table, slot 2): lengths 1, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023,
1024, 1025, 4095, 4096, 4097, 8192, 8225 (three physical chunks with a 33-token
tail); worst prefill PCC 0.999957, worst decode-after-prefill PCC 0.999957.

Continuation (`prefill_continuation_full_attention.json`, 4500 tokens, starts
32, 37, 1000, 4096, 4101, 4499): continuation per-row delta PCC minimum over
`[start, start+256)` plus the last row >= 0.99991 against HF and the window
mean >= 0.99997; the no-history and fresh-suffix negative controls score
0.63 to 0.89 on the same metric, so the check sees a broken resume. The
same continuation shape with 64-token cache blocks (27 token-by-token rows
before the first block boundary) is in `block64_full_attention.json`.

Multi-row prefill (`prefill_multi_row_full_attention.json`): one call with
rows of lengths 5, 200, 4097 and 33 (the last a continuation at 70) on
page-table rows 3, 0, 2, 1: PCC >= 0.99998 per row, zero padding verified, a
batch-4 decode over those rows PCC >= 0.999969, and two rows (33 and 60
tokens) that pad into the same tile run twice on the same input tensor
(PCC >= 0.999981) as the regression for an aliasing bug found during bringup.

Traced decode (`trace_full_attention_batch{1,13,31,32}.json`): per-user
prefills of 1 to 600 tokens on permuted page tables, eager decode PCC >= 0.999965 per
user, trace replay equals eager (PCC 1.0), repeat replay bit-equal, and two
further steps with refreshed tokens and positions change the output and match
HF (PCC >= 0.999971). The same tests pass under
`TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
(`trace_alloc_tracking/`, with a positive control proving the tracker raises)
and under `TT_METAL_WATCHER=10` (`watcher/summary.json`, fatal-pattern scan
empty).

Advertised context (`prefill_capacity_full_attention_{40960,40959}.json`,
`advertised_context_decode_full_attention.json`): real-weight prefill at
40960 and at the non-aligned 40959 (10 chunks), PCC >= 0.999883 against the
chunked torch reference on the first, middle and last 256 rows; traced decode
at position 40959 PCC 0.999935 with bit-equal repeat replay; batch-32 traced
decode at position 40959 on a 5.37 GB full-context cache, live user PCC
0.999921, all users finite. The HF-advertised 40960-token context is retained
in full (`../context_contract.json`).

## Performance evidence

Warmed batch-1, 1024-token measurements on one chip (`perf/README.md`,
`perf/full_attention/{prefill,prefill_continuation,decode}.{txt,csv}`):
prefill 10.956 ms device (21 ops) / 11.349 ms host; continuation prefill of
100 tokens at start 37 2.873 ms device (381 ops, dominated by the 27
token-by-token leading-row writes) / 5.529 ms host; traced decode 1.266 ms
device (35 ops) / 1.335 ms host. HiFi4 with fp32 accumulation everywhere;
functional baselines only.

## Safety and fallback gates

- `test_hot_path_fallback_audit`: no `torch`, `ttnn.from_torch`,
  `ttnn.to_torch`, host-backed `ttnn.full`, `.cpu()`, `ttnn.to_device` or
  synchronization in any method a measured prefill or decode pass runs.
  Runtime index tensors use device-side `ttnn.moreh_full`. The Tracy windows
  confirm it at run time: every op in the prefill, continuation-prefill and
  traced-decode windows is a device op (`perf/full_attention/*.txt`, "0 host
  ops"). `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'` was
  set for every run as the environment prescribes, but this checkout only
  declares that field (`ttnn/api/ttnn/config.hpp`) and nothing reads it, so
  it is not part of the evidence.
- Determinism: repeated identical prefill bit-equal; repeated trace replay
  bit-equal at positions 257 to 601 and at 40959.
- Real-weight inventory: `real_weight_stats.json` records name, shape, dtype,
  element count, mean, standard deviation, range and a deterministic seed for
  every tensor of layers 0 and 17, plus the input-activation statistics.

## Capability contract

| Claim | Evidence | Remaining risk |
| --- | --- | --- |
| 40960-token context, no reduction | `prefill_capacity_full_attention_{40960,40959}.json`, `advertised_context_decode_full_attention.json`, `../context_contract.json` | one layer only; a 36-layer batch-32 full-context cache (36 x 5.37 GB) exceeds the 34.2 GB of device DRAM, so the full-model stage sizes batch against context |
| one layer kind, full causal GQA with Q/K norms | `pcc_layer{0,17}_full_attention.json` (real weights, real activations) | layer-17 delta PCC 0.99942 is the lowest number in the suite; watch it when precision is lowered |
| any logical prefill length 1..40960 | `prefill_boundaries_full_attention.json` (19 lengths incl. 33 after 8225), `prefill_multi_row_full_attention.json` | lengths between 8226 and 40958 are covered by construction (same chunk plan) and by the two 40K probes, not individually |
| explicit continuation at any start, prefix preserved | `prefill_continuation_full_attention.json` (6 starts, negative controls fail), `block64_full_attention.json` | a start that is not a multiple of 256 forces the SDPA chunk down to the largest power of two dividing the start (32 for starts 32, 37, 4128); correct but slower, and vLLM prefix-cache hits resume at such starts |
| multi-row prefill call with per-row lengths, starts and page-table rows | `prefill_multi_row_full_attention.json` | rows run serially; dense work is not batched across rows |
| paged decode, batch 1..32, per-user positions and permuted pages | `trace_full_attention_batch{1,13,31,32}.json` (13 runs as 16 on 8x2, 31 as 32 on 8x4), host test of the rectangle choice for every batch 1..32 on three grids, `tt_metal_decode_grid/` probe of 14 rectangles | `current_pos` is not validated against the page-table extent on the tensor path; padded users cost their share of the decode ops |
| traced decode with refreshed inputs, repeat replay, allocation tracking | `trace_*.json`, `trace_alloc_tracking/` (gate plus positive control) | positions advance on the host in this harness; the generator stage owns on-device advancement |
| cache block size any multiple of 32 | block 32 everywhere, block 64 in `block64_full_attention.json`, host tests for 64 and 128 | block 128 is host-tested only |
| watcher clean | `watcher/summary.json` | watcher covered the traced-decode tests, not the 40K prefill |

## Reproduction

From the checkout root with the stage environment (`TT_METAL_HOME`,
`PYTHONPATH`, `TT_MESH_GRAPH_DESC_PATH` for the p150 profile,
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'`,
`TT_VISIBLE_DEVICES=<chip>`):

```bash
python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -p no:cacheprovider
QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k advertised_context -p no:cacheprovider
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_traced_decode -p no:cacheprovider
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8b/doc/functional_decoder/watcher python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_traced_decode -p no:cacheprovider
```

Profiler commands are in `perf/README.md`; every command actually run, with
timestamps and logs, is in `work_log.md`.

## Artifact index and limitations

- `pcc_layer*.json`, `prefill_boundaries_*.json`, `prefill_continuation_*.json`,
  `prefill_multi_row_*.json`, `block64_*.json`, `trace_*_batch{1,13,31,32}.json`:
  default-suite evidence written by the default-suite run, each with
  provenance (decoder/test/ttnn hashes, git SHA, device, exact command,
  environment). The watcher and allocation-tracking runs write their own
  copies (`*.watcher.json`, `*.tracked.json`) through `QWEN3_EVIDENCE_SUFFIX`
  and never overwrite these.
- `prefill_capacity_*_{40959,40960}.json`, `advertised_context_decode_*.json`:
  advertised-context evidence.
- `perf/`: Tracy ops CSV, tt-perf-report tables, host timings.
- `watcher/`, `trace_alloc_tracking/`: debug-gate runs.
- `tt_metal_sdpa_padded_k/`: tt-metal defect reproducers and logs.
- `layer_inputs/`: real token ids (`tale_token_ids.json`) and
  `capture_layer17_inputs.py`, which regenerates the gitignored 8.2 MB
  `layer17_tale_1024.pt` (deterministic float32 HF layers 0..16, about 20 s on
  CPU). Without that file the layer-17 test falls back to Gaussian inputs with
  the recorded statistics; the committed `pcc_layer17_full_attention.json`
  was produced with the real capture present.

Limitations: rows in a multi-row prefill run serially (the dense work is not
batched across rows); prefill SDPA chunk starts use the scalar
`chunk_start_idx` path (one program per distinct start, fine for eager prefill,
to be revisited when prefill is traced); the SDPA kernel derives its Q-chunk
offset as `chunk_start_idx / q_chunk_size` with integer division, so the
chunk must divide the start and a continuation start that is only a multiple
of 32 runs the whole physical chunk with 32-wide SDPA chunks; the decoder
zero-fills the implicit tile padding of each prefill input before padding it,
because those rows become cache and attention inputs; the compute
configuration is HiFi4/fp32-accumulate everywhere and no memory layout beyond
DRAM interleaved is used. No tt-metal issue has been filed for the padded-K
SDPA behaviour yet; the reproducers under `tt_metal_sdpa_padded_k/` are the
hand-off, and the workaround can be retired when the kernels clamp their K
walk to the page-table extent.
