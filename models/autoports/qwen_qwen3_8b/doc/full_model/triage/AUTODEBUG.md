# AutoDebug: Qwen3-8B capacity probe hang (static diagnosis)

Scope: the device-side hang of `models/autoports/qwen_qwen3_8b/doc/full_model/logs/capacity_probe.py` on
chips 1 and 0 (2026 Oct 29, 12:11 and 13:02 ET), analysed from source only. No device was available.
Code read at commit `1cccb01d89` (the version that ran) and at `c1f0a5ee3c` (review-2 fixes, committed
while this analysis ran). Abbreviations: SDPA (scaled dot-product attention), KV (key/value), CB
(circular buffer), NoC (network on chip), RoPE (rotary position embedding), DRAM (device memory).

## Verdict

| Rank | Hypothesis | Status | Mechanism found in source |
| --- | --- | --- | --- |
| 1 | (a) decode position at or past the 40960-token page-table extent | **Confirmed contract violation from the probe's second step onward.** Only hypothesis with a concrete out-of-bounds access. Whether the hang occurred at step 2 or later cannot be read from the evidence, but no step-1 mechanism exists. | SDPA decode reader and fused KV update both index the 1280-entry page-table stick at index 1280 and beyond; the block id they read is unspecified, and every K, V and cache-write page id derived from it is unspecified. RoPE gather reads one row past its table. |
| 2 | (c) full-context SDPA decode (K chunk 128, 1280-block table) in physical batch 8 | No mechanism. Per-core work is the same single-core loop the passing batch-32 layer test ran four times per core. | none |
| 3 | (b) inactive rows at position -1 with a scratch page table in the traced full model | No mechanism, and the premise "first full-model run" is wrong: `test_batch_decode`'s solo arm ran this exact configuration at buckets 8 and 32 and passed. | none |
| 4 | (d) sampler or LM head on this batch | No mechanism. All sampler loops are bounded; the documented sampler deadlock needs a single-tile candidate row, this model has four tiles. | none |

Root cause of the probe failure: the probe decoded four steps from a 40959-token prompt in a
40960-token context. Position 40959 is the last valid decode position. The traced graph advances the
position on device (`plus_one`), so the second replay ran row 0 at position 40960, the third at 40961,
the fourth at 40962. Nothing on the host rejected this: `decode_forward` had no context check and
`_redirect_table` clamps the valid column count to the table width instead of raising. The device
kernels have no bounds check either; the paged-cache op documents that callers must keep every
accessed entry inside the table.

## 1. Direct observations versus interpretations

Observations (from the logs, gdb captures and scripts):

- Both runs printed the allocator's "active trace" warning once, then nothing. `capacity_probe.py` at
  `1cccb01d89` printed nothing per step.
- `prefill_forward` with `sampling_params` ends in `_read_tokens`, a blocking `ttnn.to_torch`
  (`tt/generator.py:538` at `1cccb01d89`, line 569 at `c1f0a5ee3c`). The prefill therefore completed
  on the device before any decode step.
- The main thread blocked in `ttnn.from_device` -> `FDMeshCommandQueue::wait_for_outstanding_reads`;
  the completion-queue reader thread polled. The gdb capture has no Python frames, so the step index is
  unknown.
- tt-triage: DRAM, NoC, Ethernet, ARC and CB checks pass; no Inspector data, so no device callstacks.
- After SIGTERM both chips failed a one-op mesh smoke (`logs/mesh_smoke_chip{0,1}.log`), so device
  kernels or the dispatch path were still stuck.

Interpretations:

- The work log's "the first decode step never completed" is unsupported; the evidence places the hang
  somewhere in the four-step loop. The log at `c1f0a5ee3c` already corrects this wording.
- The hung-device state after kill is consistent with a kernel spinning in a NoC barrier on a
  transaction the device never completes, or with corrupted device memory. It does not identify the
  step.

## 2. What each decode step asks of the kernels (instantiated values)

Generator state at the first `decode_forward` (3 rows, bucket 8): positions written to the device are
`[40959, -1, 200, -1, -1, -1, -1, -1]`; page table rows: row 0 has 1280 real blocks (slot 0 was given
`_ensure_blocks(0, 40960)`), row 1 is all scratch (`valid_tokens = -1 + 1 = 0`), row 2 has 7 real
blocks then scratch, rows 3 to 7 are scratch (`_redirect_table`, `_write_page_table`).

SDPA decode factory values for this call (`sdpa_decode_program_factory.cpp:194-208`): grid 8x8
(`decode_sdpa_grid`), B = 8, 8 KV heads, so `num_cores_per_batch = 8`, `num_cores_per_head = 1`,
`num_heads_per_core = 1`, `num_tree_reduction_rounds = 0`, `use_k_mcast = false`
(`use_col_major_group_indexing` needs `q_heads_parallel_factor > 1`, which is MLA only). Each core
owns one (user, head) pair and nobody waits on another core. `St = 1280`, `Sk_chunk_t = 4`.

| Step | Row 0 position on device | SDPA (`get_workload_for_core`, k chunk 128) | Fused KV update | RoPE gather |
| --- | --- | --- | --- | --- |
| 1 | 40959 | `valid_seq_len = 40960`, 320 chunks, tile rows 0..1279, `page_table[0..1279]`, `PSt = 1280 = St`; mask tile 3 of the last chunk, `cur_pos_in_tile = 31` | `virtual_block_id = 1279`, in bounds | index 40959, last row of the 40960-row table |
| 2 | 40960 (advanced by `plus_one` inside the trace) | `valid_seq_len = 41088`, 321 chunks; chunk 320 maps tile rows 1280..1283 through `virtual_seq_tile_id_to_physical_tile_id`, which reads `page_table_ptr[1280..1283]` | `virtual_block_id = 40960 / 32 = 1280`, reads `page_table_ptr_u32[1280]` in both reader and writer | index 40960, one row past the table |
| 3, 4 | 40961, 40962 | same as step 2 | same as step 2 | two and three rows past |

Row 2 (position 200, 201, ...) and the five inactive rows are in bounds at every step.

Why the step-2 accesses are out of bounds and unspecified:

- The page-table CB in the SDPA reader holds exactly one stick (`page_table_cb_size =
  page_table_stick_size` for a DRAM page table, `sdpa_decode_program_factory.cpp:569`; the stick is
  1280 x 4 bytes). `read_page_table_for_batch` reads that one stick. Index 1280 is the first word past it
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:64-68`).
- The fused update's page-table CB is likewise one stick (`num_pages_page_table = 1`,
  `paged_tiled_fused_update_cache_program_factory.cpp:137,281-290`). Reader and writer both compute
  `virtual_block_id = update_idx / block_size` and index the stick with no bound
  (`reader_paged_fused_update_cache_interleaved_start_id.cpp:132-140`,
  `writer_paged_fused_update_cache_interleaved_start_id.cpp:102-109`). The writer then writes `Wt`
  tiles per head to cache page ids derived from that value. The op states the caller's duty:
  "The table spans virtual blocks and may be wider than the physical pool. Callers must map each
  accessed entry to an in-bounds physical cache block"
  (`paged_fused_update_cache_device_operation.cpp:238-239`).
- The embedding reader passes the token straight to the weight accessor with no range check
  (`ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp:71-80`).
- Neither op can validate positions on the host: `cur_pos_tensor` and `update_idxs_tensor` live on
  the device and are read by the kernels (`reader_decode_all.cpp:119-156`).

Consequence: from step 2 the kernels read K and V from, and write K and V to, cache pages chosen by
whatever word follows the page-table stick in L1. That is outside the ops' contracts. Such accesses can
corrupt other device buffers or target memory the device never acknowledges; either ends in the
observed signature (a replay that never completes, a device that stays stuck after the process dies).
The exact failing transaction cannot be determined statically.

## 3. Hypothesis (b): inactive rows at -1 with a scratch table

Kernel handling of position -1 (`UINT32_MAX`):

- SDPA reader: pushes both position CBs, reads the position, returns (`reader_decode_all.cpp:150-156`).
  Writer: pops `cb_cur_pos`, returns (`writer_decode_all.cpp:126-137`). Compute: waits, pops, returns
  (`sdpa_flash_decode.cpp:154-161`). With `num_cores_per_head = 1` no reducer, output core or
  multicast partner waits on these cores. The user's output rows are left unwritten.
- Fused KV update: reader sets `skip_update`, still pushes `cb_index` and `Wt` stale cache tiles per
  head; writer pops the index, skips the DRAM write; compute runs unconditionally. CB counts balance
  (`reader_paged_fused_update_cache_interleaved_start_id.cpp:98-101,157-171`,
  `writer_paged_fused_update_cache_interleaved_start_id.cpp:80-84,120-160`).
- `ttnn.relu` on int32 lowers to `relu_tile_int32` (`unary_op_utils.cpp:766-768`), so the RoPE index of
  a -1 row is 0, not a reinterpreted bit pattern.
- `plus_one` with `skip_negative_entries` leaves -1 alone (`reader_plusone_interleaved.cpp:45-49`).

Passing contrast that uses exactly this path: `tests/test_full_model.py::test_batch_decode` solo arm
(lines 311-328 at `1cccb01d89`): one live row, `pos = -1` for the others, a full-width table whose
inactive rows are the scratch block, 31 traced decode steps per prompt, eight prompts, buckets 8 and
32. Recorded as passing in `doc/full_model/batch_decode_8.json` and `batch_decode_32.json` (the sampler
ran eagerly in those artifacts because of the seed defect fixed in review 1; the decode trace itself
was replayed, `decode_trace_replays == steps - 1` is asserted). The layer-level
`test_optimized_decoder.py::test_sharded_residual_contract` also passed batch 13 as physical 16 with
-1 rows and zero page-table rows. The review-2 note "the inactive-row path in the traced full model
has no passing run" is therefore too strong; only the sampler trace with -1 rows lacks a run.

Residual risk, not a hang: the unwritten SDPA output rows of -1 users feed `nlp_concat_heads_decode`,
the O projection and the rest of the layer. Every op downstream is row-independent (RMSNorm,
DRAM-sharded matmuls, elementwise, embedding, LM head, per-row top-k), so the garbage stays in rows the
host never reads. The design accepts this ("their rows of the result are unspecified",
`tt/optimized_decoder.py:720-722`).

## 4. Hypothesis (c): full-context SDPA in physical batch 8

Per-core work for user 0 at step 1: 320 chunks x (4 K tiles x 4 head-dim tiles + 4 V tiles x 4) in a
single `for k_chunk` loop, one head, no reduction. The passing batch-32 layer tests
(`test_functional_decoder.py::test_advertised_context_batch32_traced_decode`, artifacts
`doc/{functional,fused,optimized}_decoder/advertised_context_decode_full_attention.json`) ran the same
reader, compute and writer with `num_heads_per_core = 4`, that is the identical inner loop four times
per core at position 40959 with k chunk 128 and a 1280-block table, eager and replay bit-equal. The
batch-1 `ctx40959` test ran the 8-cores-per-head tree-reduction variant at the same position. Nothing
in the three kernels branches on `num_heads_per_core == 1` except the `cur_head` loop bound. No
mechanism.

## 5. Hypothesis (d): sampler or LM head

- LM head: ten DRAM-sharded matmuls, run in every decode test and in `prepare()` for bucket 8.
- Sampler compute (`reduction/sampling/device/kernels/compute/sampling.cpp`): top-k sort and merge
  with compile-time iteration counts; softmax over a fixed tile count. No data-dependent loop.
- Sampler writer (`writer_interleaved.cpp:152-200`): the top-p and draw loops iterate over at most `k`
  candidates and `break`; with non-finite values the comparisons are false and the loops run to their
  fixed bound. Greedy parameters (`k=1`, `p=0`) cut off at the first candidate.
- The known deadlock (`tt_sampling.py:923`, tenstorrent/tt-metal#53781) needs a single-tile candidate
  row; this sampler uses four vocabulary splits x 32 = 128 candidates.
- Bucket-8 sampling ran in `prepare()` (eager) and in `test_batch_decode` (eager, with garbage rows
  from the -1 users present).

No mechanism.

## 6. What settles each hypothesis on a device

| Hypothesis | Host-side guard or probe change | Expected result |
| --- | --- | --- |
| (a) | Already in `c1f0a5ee3c`: `decode_forward` raises for any position >= 40960 (`tt/generator.py:636-637`); `capacity_probe.py` and `capacity_bisect.py` prefill row 0 with 40955 tokens so four steps end at 40958; a print after every step. | The corrected probe completes all four steps. If it does, (a) is confirmed as the cause, since step 1 of the original probe was in bounds and nothing else changed. If it still hangs, the fault is in the step-1 configuration and the bisect below applies. |
| (b) | Rerun `test_batch_decode -k 8` as the control (same -1 rows, same scratch table, short prompts). Then the corrected probe with row 1 live at a short length instead of -1. | Both pass, as the recorded artifacts already did for the decode trace. |
| (c) | Bisect arm: one row at position 40955..40958 in bucket 1, then the same row in bucket 8 with seven -1 rows, then the full corrected probe. A layer-level analogue of `test_advertised_context_batch32_traced_decode` with batch 8 isolates the SDPA op. | Pass; differs from the passing batch-32 case only in the per-core head count. |
| (d) | Run the corrected probe in host-sampling mode (`sampling_params=None`, logits returned) and with `allow_force_argmax=True`. | Pass in both modes, which excludes the sampler. |

Also recommended for the first device run after the reset: `TT_METAL_WATCHER=10` on the corrected
probe. The watcher's NoC sanitizer reports a transaction outside the valid DRAM range, which would
confirm or refute the step-2 mechanism directly if anyone re-runs the original out-of-contract probe
on purpose.

## 7. Other potential issues (not causes of this hang)

1. `_redirect_table` clamps `valid = min(width, ceil(valid_tokens / 32))` (`tt/generator.py:339`).
   Before `c1f0a5ee3c` this turned an out-of-context position into a silently full table. The new
   `decode_forward` guard covers the generator path; the layer-level `OptimizedDecoder.decode_forward`
   and `Qwen3Model.ttnn_decode_forward` still have no position check and rely on their callers.
2. `prefill_forward` accepts `end == max_context` (`tt/generator.py:525`). A 40960-token row can be
   prefilled but cannot take a single decode step; the probe handled this by marking row 1 inactive.
   `generate()` reserves `len(ids) + max_new_tokens` blocks but does not reject a request whose last
   decode position would reach 40960; the new guard now raises mid-generation instead of at admission.
3. The original `capacity_bisect.py` used `positions + 1` for the second step, which turned row 1's -1
   into 0: a live decode at position 0 against a 40960-token cache, overwriting that slot's first cache
   row. Fixed in `c1f0a5ee3c` (explicit `[40956, -1, 201]`).
4. SDPA decode output rows for -1 users are never written (section 3). Harmless for the host today;
   any future cross-row op on the residual (a batch norm, a reduction over users) would propagate it.
5. The device kernels trust the page-table index without a bound on both the SDPA decode and
   paged-cache paths. A debug-build `ASSERT(virtual_block < max_blocks_per_seq)` in
   `virtual_seq_tile_id_to_physical_tile_id` and in the fused-update reader/writer would turn this
   class of caller error into a watcher-visible assert instead of an unexplained hang. This is a
   tt-metal change, outside the model directory.

## 8. Claims re-checked against the code

- Step 1 in bounds everywhere: verified with `get_workload_for_core` (`rt_args_common.hpp:35-93`),
  `virtual_seq_tile_id_to_physical_tile_id`, the fused-update reader, `generate_mask`
  (`cur_pos_in_tile = 31` takes the early return in `fill_tile_partial`), and the 40960-row RoPE tables
  (`hf_rope_tables`).
- The position advances on device: `ttnn.plus_one(current_pos, skip_negative_entries=True)` is inside
  the captured graph (`tt/model.py:473-475`); `decode_forward` does not rewrite positions on steady
  steps (`reload` false after the first device-mode step, `tt/generator.py:603-608` at `1cccb01d89`,
  lines 638-643 at `c1f0a5ee3c`).
- Work split for B = 8 on 8x8: computed from `sdpa_decode_program_factory.cpp:194-208`; `use_k_mcast`
  is `use_col_major_group_indexing` (line 682), false here.
- `test_batch_decode` solo arm at `1cccb01d89` uses `pos = -1` and a scratch-filled wide table
  (`tests/test_full_model.py:317-319`); artifacts exist and the test asserted trace replays.
- The probe's prefill completed on device before decode: `prefill_forward` returns
  `_read_tokens(...)`, a blocking readback.

Unverified without hardware: which replay (2, 3 or 4) the device stopped in, and which out-of-range
transaction or corrupted buffer produced the non-completing read. Both require the watcher or Inspector
on a live run.

## Erratum (stage owner, after review 3)

Section 3 states that `test_batch_decode`'s solo arm ran the -1-row configuration at buckets 8 and 32 and that
`doc/full_model/batch_decode_{8,32}.json` record it. The recorded artifacts came from the earlier bucket-1 solo arm
(`gen.generate`); the same-bucket arm with -1 rows is the review-2 change and has not run on a device. The
contrast in section 3 therefore does not exist yet; hypothesis (b) stays excluded by the kernel reading only.
