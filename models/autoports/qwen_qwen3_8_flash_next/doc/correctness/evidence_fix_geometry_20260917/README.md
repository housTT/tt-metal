# Compressed-index key cache page geometry (fixed 2026-09-17)

## The defect

`fused_index_key_cache` (the per-block keys the QSA selector scores every token) was allocated as
`(max_num_blocks, 1, 16, 128)` in TILE layout: 16 block keys per 64-token page. The two writers,
`_compressed_index_prefill` (`paged_fill_cache`) and `_compressed_index_decode` (`paged_update_cache`), let the
op derive the page size from the cache's **padded** tile height, which is 32, while the reader
(`_physical_compressed_ids` + `reshape(cache, (max_num_blocks * 16, 128))`) addressed 16 keys per page.

Effect, reproduced by `tests/test_fused_decoder.py::test_compressed_key_cache_page_geometry_roundtrip` before the
fix: a two-page fill wrote keys 0-15 to the first page, dropped keys 16-31 into tile padding and left the second
page zero. So with 128-row microchunks half of every chunk's block keys were invisible to the selector; with
256/512-row microchunks tile *i* of a fill also landed on page *i* instead of page *2i*, aliasing blocks onto
other blocks' keys; decode's own key updates were misplaced the same way. Prefill logits below 2,048 tokens were
unaffected (the dense prefill path never reads this cache), which is why the all-position prefill gate did not
see it, and why the damage appeared as (a) ~75 % long-document decode agreement (perf-plan item C5) and (b) a
decode regression that grew with the microchunk size (item C6).

## The fix

One 32-row tile per page (`COMPRESSED_KEY_ROWS_PER_PAGE` in `tt/fused_decoder.py`): the prefill writer pads each
page's 16 keys to a full tile before `paged_fill_cache`; the decode writer maps group `g` to row
`(g >> 4) * 32 + (g & 15)`; the reader strides pages by 32; `_physical_compressed_ids_host` (`tt/model.py`) takes
`rows_per_page`; the vLLM cache adoption allocates 32 rows. No extra DRAM (the padding rows existed physically).

## Tests

- `test_compressed_key_cache_page_geometry_roundtrip`: op-level round trip through the reader's view.
- `test_compressed_key_cache_is_chunk_size_invariant[128|256|512]`: layer 3, shuffled page table, the 3k
  document's chunk plan; every complete block reads back non-zero and distinct, keys bitwise identical across
  chunk sizes, a decode update changes exactly its own block. All pass after the fix (10.7 s).
- Teacher-set gate gained a `long_context` domain gate (top-1 >= baseline - 3.0) and a per-document decode
  divergence report.

## Model-level results (48 layers, BF16 HF reference, greedy)

Long documents, decode top-1 / top-5 after prefill (99 teacher-forced steps each):

| document | before (baseline 2026-09-11) | after, 128-row microchunks | after, 512 rows + adaptive + slabs |
| --- | ---: | ---: | ---: |
| 3k (3,156 tokens) | 72.7 / 85.9 | 98.0 / 100.0 | 98.0 / 100.0 |
| 6k (6,228) | 74.7 / 89.9 | 89.9 / 99.0 | 90.9 / 100.0 |
| 12k (12,372) | 75.8 / 87.9 | 91.9 / 100.0 | 92.9 / 100.0 |
| pooled | 74.4 / 87.9 | 93.3 / 99.7 | 93.9 / 100.0 |

The 512-row microchunk regression is gone: 512 rows now score at or above 128 rows on every document.

Other gates at 128 rows: all-position prefill agreement (12 prompts, 4,594 positions) top-1 85.76 → **88.68**,
reference top-1 in TT top-5 97.39 → **98.96**, mean top-100 PCC 0.908 → **0.938**; the 3k document's bins above
2,048 tokens went from 76.0 / 73.4 to 84.8 / 90.6 (below 2,048 unchanged at 87-88). AIME24 teacher forcing
96.97 → 97.98 top-1 (top-5 / top-100 100). Full 14-prompt teacher set (1,386 decode positions), pooled top-1 / top-5 / top-100:

| configuration | teacher set | all-position prefill (top-1 / top-5 / PCC) | AIME24 TF | free run |
| --- | ---: | ---: | ---: | --- |
| before (baseline 2026-09-11, 128 rows) | 91.92 / 97.40 / 99.42 | 85.76 / 97.39 / 0.908 | 96.97 / 100 / 100 | prefix 5 |
| after, 128 rows (`full_c128/`) | 96.18 / 99.93 / 100.00 | 88.68 / 98.96 / 0.938 | 97.98 / 100 / 100 | — |
| after, 512 rows + adaptive + slabs (`full_c512_a1_s1/`, **shipped**) | 96.32 / 100.00 / 100.00 | 88.83 / 98.85 / 0.938 | 97.98 / 100 / 100 | prefix 5, coherent |

Every short-prompt domain is unchanged to the decimal between the two chunk sizes and versus the 2026-09-11
baseline; the whole gain is the long-context domain (74.4 → 93.3 / 93.9). `doc/correctness/teacher_set/baseline.json`
and `baseline_allpos.json` are re-recorded from the shipped configuration (the previous files are kept as
`*_20260911_pre_geometry_fix.json`). GSM8K-100 and the endpoint checks on the rebuilt package: see the P1 section.

## P1: 512-row adaptive microchunks + MoE slabs shipped (community sweep, 2026-09-17)

`bench-sweeps/rerun/rerun_qwen38_staged.py` against the rebuilt package (greedy, one user, `ignore_eos`); rows in
`bench-sweeps/results/tt-hous__qwen3.8-flash-next-p300x2__long-context/rows.jsonl` (128-row rows archived in
`results_superseded_20260917T152048Z/`).

| ISL | TTFT before (128 rows) | TTFT after (512 rows) | speed-up | prefill tok/s | TPOT before → after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.33 s | 0.34 s | 1.0× (short prompts keep the 128-row path) | 379 | 41.0 → 41.2 ms |
| 1,024 | 2.30 s | 1.59 s | 1.44× | 642 | 41.0 → 41.3 ms |
| 4,096 | 11.26 s | 7.99 s | 1.41× | 513 | 76.2 → 72.1 ms |
| 16,384 | 52.4 s | 38.3 s | 1.37× | 428 | 76.6 → 72.4 ms |
| 32,768 | 108 s | 79 s | 1.37× | 413 | 76.8 → 72.4 ms |
| 65,536 | 222 s | 163 s | 1.36× | 402 | 76.8 → 72.5 ms |
| 131,072 | 458 s | 338 s | 1.35× | 388 | 76.9 → 72.4 ms |

Decode above the 2,048-token budget also got ~4 ms/token faster: the selector now reads a tile-aligned key
table instead of a padded 16-row view.

Endpoint checks on the rebuilt package (`tt-model serve <staged>/tt_kernel_manifest.json`, port 20020):
boundary prompts 1–1,537 tokens (`readiness_vllm/non_aligned_prompt_check_20260917_p1.json`) 14 / 14 pass; tool
calling (single, parallel, streaming, result round-trip, no spurious call) all pass; GSM8K-100 greedy
(`doc/correctness/gsm8k_endpoint_20260917_p1_100.json`) 92 / 100, 8 cut at the 512-token cap, 0 degenerate;
AIME24 ×8 with the default preset (host-sampled) 8 / 8 finish, 0 degenerate. Layer tests:
`test_two_user_prefill_then_batched_decode[3]` passes; `[0]` fails only because the fused GDN prefill norm
rejects the checkpoint-less test layer's zero weight shape (`sigmoid_gated_rms_norm: weight must be [V]`) and
passes with `QWEN38_GDN_PREFILL_FUSED_NORM=0` — pre-existing, unrelated to this fix.

Host-sampling cost on the rebuilt package (`vllm bench serve`, one user, `ignore_eos`, `bench/`): 128/128 greedy
41.3 ms vs temperature 1.0 host-sampled 59.2 ms; 4,096/256 71.9 vs 91.0 ms. The constant ~18 ms/token host cost
is unchanged; the greedy base above 2k is 4 ms lower than before the fix.
