# Qwen3-8B functional decoder: work log

Stage 1 (functional decoder) for `Qwen/Qwen3-8B` in
`/home/hous/dev/tt-metal-wt-qwen3` (branch `hous/qwen3-8b-autoport`, started
at HEAD `4f3c2b13fa`). Hardware: Blackhole P300, one chip per process
(`TT_VISIBLE_DEVICES=0` or `1`), 1x1 mesh. Times are Eastern (ET).

## 1. Discovery (2026 Oct 27, 20:05 to 20:25 ET)

- Skills read: `model-bringup`, `functional-decoder`, `tt-device-usage`,
  `tt-enable-tracing`, `stage-review` from
  `/home/hous/dev/skills/plugins/tt-model-bringup` (plugin version 0.1.21),
  plus `skills/vllm-integration/references/prefill-features.md` and the stage
  prompt `/home/hous/.claude/jobs/1099b3df/tmp/prompts/qwen3_01-functional-decoder.txt`.
- `python /home/hous/dev/skills/plugins/tt-model-bringup/scripts/environment.py`
  validated `TT_AUTODEBUG_ROOT=/home/hous/dev/skills/plugins/tt-autodebug`
  (tt-autodebug 0.1.8 with `autodebug`, `autotriage`, `autofix`).
- HF model: snapshot
  `/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`,
  transformers 5.12.1, torch 2.11.0+cpu. Config: `model_type=qwen3`, 36
  layers, all `full_attention`, hidden 4096, intermediate 12288, 32 query
  heads, 8 KV heads, head_dim 128, `rms_norm_eps=1e-6`, RoPE default with
  theta 1e6, `max_position_embeddings=40960`, no attention bias, no sliding
  window. One layer kind.
- `timeout 60 tt-smi -ls --local`: four Blackhole p300c chips (ids 0..3);
  chips 0 and 1 used here, chips 2 and 3 belong to another job.
- Mesh smoke (chip 0): `ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)`
  opened and closed; `MESH_SMOKE_OK`. DRAM:
  `ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)` reports 8 banks x
  4272341376 bytes free (34178731008 bytes allocatable).
- API probes on chip 0 before coding: `ttnn.pad` rejects front padding on
  TILE tensors (row-major detour used); `ttnn.embedding` needs uint32
  indices (`ttnn.typecast` on a ROW_MAJOR int32 tensor works);
  `ttnn.SDPAProgramConfig` grid is a `CoreCoord`; the chunked SDPA `scale`
  must be float32-representable (nanobind rejects `128 ** -0.5` as a double;
  the module rounds it); chunked SDPA accepts `Sq` that is not a multiple of
  `q_chunk_size`; `ttnn.moreh_full` allocates a 1280-block cache in 0.25 s.

## 2. Implementation

- `tt/functional_decoder.py`: raw TTNN layer (no `models/common` module
  reuse), `from_state_dict` fuses QKV, transposes projections, reshapes norm
  weights, and builds HF rotate-half cos/sin tables on device for all 40960
  positions. Prefill runs each row in 4096-token physical chunks:
  `rms_norm`, `linear`, `nlp_create_qkv_heads`, per-head `rms_norm` for Q/K,
  `rotary_embedding_hf`, cache write (`paged_update_cache` for rows that share
  a block with live prefix rows, `paged_fill_cache` from the first block
  boundary), `chunked_scaled_dot_product_attention` over the paged cache,
  `nlp_concat_heads`, `linear`, residual, `rms_norm`, SwiGLU MLP
  (`ttnn.mul` with fused SiLU). Decode: `nlp_create_qkv_heads_decode`,
  Q/K norms in L1, RoPE lookup from `current_pos` by `ttnn.embedding`
  (device-side typecast to uint32), `paged_update_cache`,
  `paged_scaled_dot_product_attention_decode`, `nlp_concat_heads_decode`.
  Compute config HiFi4 with fp32 accumulation everywhere (correctness first).
- `tests/weights.py`: real safetensors loader (only the shards that hold the
  layer), deterministic synthetic weights from `real_weight_stats.json`.
- `tests/reference.py`: HF `Qwen3DecoderLayer` reference in float32 with a
  `DynamicCache`, plus a chunked torch reference for 40960-token inputs
  (validated against the HF layer at PCC 0.9999999 in
  `test_chunked_reference_matches_hf_host`).
- Real activations: `layer_inputs/tale_token_ids.json` (first 40960 tokens
  of `models/tt_transformers/tests/tale-of-two-cities.txt.bz2`) drives layer-0
  inputs through the real embedding; `layer_inputs/layer17_tale_1024.pt`
  holds the real hidden state entering layer 17 for the first 1024 tokens
  (HF layers 0..16 in float32, 19 s on CPU). Layer 17 inputs have RMS 7.99
  and max abs 12726 (massive activations), which Gaussian inputs do not
  reproduce.

## 3. First device smokes (chip 0, 20:28 to 20:33 ET)

Scratch scripts outside the repo (`/tmp/qwen3_scratch/smoke.py`,
`smoke2.py`): layer 0 real weights, S=100 on a permuted 4-user page table in
slot 2: prefill PCC 0.99997, decode PCC 0.99998 (position 100) and 0.99997
(101). Two-chunk prefill at 4097: PCC 0.99997. Continuation at tile-unaligned
start 37: PCC 0.99997 vs HF and 0.99999 vs full TT prefill. Traced decode
batch 4 with `trace_region_size=0`: replay equals eager (PCC 1.0), repeat
replay bit-equal, refreshed tokens/positions change the output and match
eager (PCC 1.0).

Finding: the "no prefix" negative control still scored PCC 0.9997 on the
layer output because the residual passthrough dominates a single layer. The
tests therefore gate continuation on the layer contribution
(`output - input`) over the resumed window `[start, start + 256)` plus the
last row, with natural-text inputs, as `prefill-features.md` prescribes.

## 4. Suite run 1 (chip 0, 20:35 ET) and the SDPA padded-K defect

Command:

```bash
TT_VISIBLE_DEVICES=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k 'not host and not audit' -p no:cacheprovider
```

Result: 3 failed, 4 passed (`/tmp/qwen3_scratch/logs/default_suite_run1.log`).

1. `test_prefill_logical_boundary_lengths`: length 8225 (chunks 4096, 4096,
   64) gave prefill PCC 0.00045 while the decode right after it was correct
   (PCC 0.99994); every other length passed at PCC >= 0.99997.
2. `test_prefill_continuation`: continuation correct everywhere, but the
   negative controls at starts 32 and 37 did not fail (see section 3).
3. `test_real_weights_prefill_decode_pcc[17]`: prefill PCC 0.999996, decode
   PCC 0.007 at position 1024 (Gaussian fallback inputs, 33-block table).

Investigation (scratch scripts `narrow*.py`, all on chip 0):

- Neither failure reproduced in a fresh process; the 8225 case reproduced
  deterministically only after an 8192-token prefill on the same cache and
  page table (`narrow5.py`): normed input PCC 1.0, Q after RoPE PCC 0.99999,
  K cache rows PCC 1.0, chunked SDPA output garbage (max 1.4e38). Re-running
  only chunk 3 reproduced the garbage; re-running the whole prefill passed.
- `narrow6.py` (kept as
  `tt_metal_sdpa_padded_k/repro_in_decoder_sequence.py`): same failing state,
  chunk 3 re-run with a 264-entry page-table row (6 entries of slack) is
  correct, with chunk 64 or 32 is correct, with the 258-entry row and chunk
  256 is garbage every time.
- Standalone reproducers (kept under `tt_metal_sdpa_padded_k/`):
  `repro_chunked_prefill_sdpa.py` case B: a 266-entry row whose entries 258..265
  point at blocks holding 3e38, all beyond `chunk_start + Sq` and causally
  masked, gives PCC -0.002 and max 8.9e35; with the exact 258-entry row the
  same call is correct (PCC 0.9997 against a float32 torch reference with
  random unit-variance K/V). `repro_paged_decode_sdpa.py` case B: decode at
  `cur_pos=1024` with `k_chunk_size=128` and a 41-entry row whose entries
  33..40 point at 3e38 blocks gives PCC 0.012; with `k_chunk_size=32`
  (which divides the 1056-position extent) the poisoned row is harmless
  (PCC 0.9997).

Root cause (tt-metal, `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp`
lines 326-360): the chunked kernel pads `Sk = chunk_start_idx + Sq` to a
multiple of `k_chunk_size` and relies on the causal mask for the padded
keys. The mask is applied by addition, so non-finite K/V in the padded
region corrupt every valid query row, and the reader follows the page table
into entries past the row end to fetch that region, where stale memory
supplies arbitrary block ids. The decode kernel walks K in `k_chunk_size`
steps the same way. Validation only checks `kv_length >= Sq + chunk_start`.

On-branch workaround in the decoder: `prefill_sdpa_chunk_size` picks the
largest chunk (<= 256) that divides `chunk_start` and satisfies
`chunk_start + round_up(phys_len, chunk) <= page_table_extent`;
`decode_sdpa_k_chunk_size` picks the largest K chunk (<= 128) that divides
the page-table extent. Both are host tests (`test_sdpa_chunk_selection_host`,
`test_prefill_plan_host`).

Also fixed in the suite: layer-0 inputs now come from real text, the layer-17
real test uses 1000 tokens plus two decode steps so every row is a real
activation, and the continuation gate uses the resumed window.

## 5. Suite run 2 (final code)

See README.md for the result tables. Commands and logs are listed in
section 7.

## 6. Hash note

`black` reformatted the four Python files at 20:42 ET while suite run 1 and
the first context run were already executing the pre-format code. Behaviour
was identical, but every evidence JSON in the final set was regenerated from
the final files so the recorded `functional_decoder_sha256` matches HEAD.

## 7. Commands and logs

Run 2 (20:52 ET, chip 0, `default_suite_run2.log`): 5 passed, 2 failed.
`test_prefill_continuation`: the no-history control at start 32 scored 0.997
on the window mean, so the gate moved to the per-row minimum of the
layer-contribution PCC over the window (row 32 without its history attends to
one token instead of 33 and must fail hard). `test_real_weights_prefill_decode_pcc[17]`
(1000 tokens): "Tensor is not allocated" on the second `prefill_forward` over
the same input tensor. Cause: for a length whose tile-padded shape already
matches the padded shape, `ttnn.pad` returns a view that aliases the caller's
buffer, and `_prefill_row` deallocated it as if it were its own copy. Fix:
`_same_buffer` (buffer-address comparison) guards every deallocation of a
possibly-aliased tensor (`_prefill_row`, `_pad_rows`); the multi-row test
gained two rows (33 and 60 tokens) that pad into the same tile and run twice on
one input tensor as the regression.

Run 3 (21:01 ET, chip 0, `default_suite_run3.log`): 7 passed, 4 skipped
(opt-in context and perf tests). Context tests run 3 (chip 1,
`context_run3.log`): 3 passed.

Run 4 (21:13 ET): after the pre-commit hooks reordered imports in the test
file and the `expect_error` fixture replaced `pytest.raises`, every device
run was repeated so each evidence JSON records the committed test file:
default suite on chip 0 (`default_suite_run4.log`, 7 passed, 4 skipped,
123.7 s), context tests on chip 1 (`context_run4.log`, 3 passed, 59.9 s),
Tracy and watcher on chip 1 (`perf_run4.log`, `watcher_run4.log`), trace
allocation tracking and its positive control on chip 0
(`trace_alloc_tracking4.log`). All evidence JSON carry
`functional_decoder_sha256 = 0fd1cb7dde0f7cbc...` and
`test_sha256 = 4247601ddaa4f1af...` (HEAD).

Results are tabulated in `README.md`; the raw numbers are in the JSON files
listed there.

## 7. Commands and logs

All commands ran from `/home/hous/dev/tt-metal-wt-qwen3` with
`HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TT_METAL_HOME=/home/hous/dev/tt-metal
PYTHONPATH=/home/hous/dev/tt-metal-wt-qwen3:/home/hous/dev/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime
TT_MESH_GRAPH_DESC_PATH=/home/hous/dev/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback":true}'` and the
checkout `python_env`. Scratch console logs live in `/tmp/qwen3_scratch/logs/`
(not committed); the evidence directory keeps the JSON, tables and the
watcher, tracker and reproducer logs.

| Step (ET) | Command | Result |
| --- | --- | --- |
| 21:12 host tests | `python -m pytest -q models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k 'host or audit' -p no:cacheprovider` | 16 passed |
| 21:13 default suite, chip 0 | `TT_VISIBLE_DEVICES=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k 'not host and not audit' -p no:cacheprovider` | 7 passed, 4 skipped (123.7 s) |
| 21:13 context tests, chip 1 | `TT_VISIBLE_DEVICES=1 QWEN3_CONTEXT_TESTS=1 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k advertised_context -p no:cacheprovider` | 3 passed (59.9 s) |
| 21:14 Tracy, chip 1 | see `perf/README.md` | 1 passed; CSV `generated/profiler/qwen3_functional/reports/2026_10_27_21_14_29/` |
| 21:16 tt-perf-report | see `perf/README.md` | `perf/full_attention/{prefill,decode}.{txt,csv}` |
| 21:15 watcher, chip 1 | `TT_VISIBLE_DEVICES=1 TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8b/doc/functional_decoder/watcher python -m pytest -q -s '...::test_traced_decode[blackhole-batch1-...]' '...::test_traced_decode[blackhole-batch32-...]' -p no:cacheprovider` | 2 passed; `watcher/summary.json` clean (0 fatal-pattern matches, 3 dumps) |
| 21:16 trace allocation tracking, chip 0 | `TT_VISIBLE_DEVICES=0 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m pytest -q -s models/autoports/qwen_qwen3_8b/tests/test_functional_decoder.py -k test_traced_decode -p no:cacheprovider` | 2 passed; `trace_alloc_tracking/console.log` |
| 21:17 tracker positive control, chip 0 | `TT_VISIBLE_DEVICES=0 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python models/autoports/qwen_qwen3_8b/doc/functional_decoder/trace_alloc_tracking/positive_control.py` | raised "Found 1 device buffer(s) still alive before trace replay" |
| 21:07 kernel reproducers, chip 1 | `TT_VISIBLE_DEVICES=1 python models/autoports/qwen_qwen3_8b/doc/functional_decoder/tt_metal_sdpa_padded_k/<script>.py` | logs beside each script; defect reproduced, workaround correct |
| 21:08 context contract | `python /home/hous/dev/skills/plugins/tt-model-bringup/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_8b --hf-model Qwen/Qwen3-8B --stage functional-decoder --require-contract --strict-caps` | `Context contract OK ... target=40960, supported=40960 (full HF context)` |

Host-work counters for the traced decode loop in `test_traced_decode`: one
capture per batch size, three replays measured (one initial, one repeat, two
with refreshed inputs), token and current_pos refreshed by
`ttnn.copy_host_to_device_tensor` only when the test changes them, page table
never refreshed after setup, no synchronization inside the replayed region.
Positions advance on the host in this layer-level harness; the generator
stage owns on-device advancement.
