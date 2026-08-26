# AUTOTRIAGE

## Diagnosis

- The watcher failure was caused by routing QSA embedding lookups through TTNN's `EmbeddingsTilizedIndicesProgramFactory` with a TILE-layout UINT32 index tensor.  The Metal2 tilized-index reader sizes its private `INDEX_SCRATCH` DFB as 16 UINT32 elements (512 bytes), but reads a full TILE index page (32 x 32 UINT32 = 4096 bytes) into that scratch buffer.  Watcher correctly reports the 4096-byte NOC read overflowing the circular buffer.  The model-scope fix is to convert QSA embedding index tensors to ROW_MAJOR before `ttnn.embedding`, which selects the fused/RM embedding path while preserving tiled outputs and logical QSA paging.

## Triage Evidence

- Failing command artifact: `watcher_final/pytest.log`.
- Stop site: `test_real_weights_hf_prefill_decode_pcc[3]`, QSA real-weight prefill, inside `FunctionalDecoder._selected_virtual_tokens`.
- Watcher direct evidence:
  - Device/core: device 0 worker core `(x=0,y=0)` / virtual `(x=1,y=2)`.
  - Fault: NCRISC using NOC0 tried to unicast-read 4096 bytes to local L1 `0x01b600` from DRAM `0x00d91e80`.
  - Watcher classification: `NOC transaction overflows a circular buffer`.
  - Active kernels: BRISC `writer_unary_stick_layout_interleaved_start_id_metal2.cpp`; NCRISC `ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp`.
  - Python stack: `functional_decoder.py` `_selected_virtual_tokens` -> `_qsa_prefill` -> `prefill_forward`.
- Passing/failing contrast:
  - The same QSA real-weight test passed without watcher and produced `prefill_pcc=0.99679226`, `decode_pcc=0.99988198`.
  - Layers 0 and 1 passed under watcher before layer 3 entered QSA, so the fault is isolated to the QSA embedding/index path, not shared HyperConnection, MoE, GDN, or PLE.
- Downstream symptoms:
  - Process termination and Python abort are watcher consequences after the NOC sanitizer fault.  There was no remaining live hang to triage.

## Source Evidence

- Model source:
  - `_selected_virtual_tokens` pooled QSA raw-index cache rows to `[max_num_blocks * 16, 128]`, flattened physical compressed ids to `[1, batch * compressed_blocks]`, and called `ttnn.embedding(..., layout=ttnn.TILE_LAYOUT)`.
  - `_gathered_qsa_attention` similarly flattened physical token ids and called `ttnn.embedding` to gather K/V cache rows.
- TTNN dispatch route:
  - `ttnn/cpp/ttnn/operations/embedding/embedding.cpp` routes TILE-layout input indices to the prim embedding op without row-major reshaping.
  - `embedding_device_operation.cpp` selects `EmbeddingsTilizedIndicesProgramFactory` whenever `input_tensor_arg.layout() == TILE_LAYOUT`.
- CB/page ledger for the failing pooled-key embedding:
  - Inspector/generated compile-time args for `embedding_ind_tilized` included `row_length=1024` and `weight_stick_size=256`; the companion QSA cache-gather embedding used `row_length=1024` and `weight_stick_size=128`.
  - The tilized-indices factory creates `INDEX_SCRATCH` with `entry_size = FACE_HEIGHT * round_up_to_mul32(input_element_size)` = `16 * 32` = `512` bytes for UINT32 indices.
  - `embedding_ind_tilized.cpp` obtains `input_page_size = input.get_aligned_page_size()` and issues `noc.async_read(input, ..., input_page_size, page_id=curr_tile)`.
  - A TILE UINT32 index page is `32 * 32 * 4 = 4096` bytes, matching the watcher-reported transaction size and exceeding the 512-byte scratch DFB entry.
- The previous single-row flattening workaround fixed a row-order correctness bug but left the indices in TILE layout, so it still selected this unsafe tilized-index kernel.

## Downstream Effects

- The fault is a source-level DFB sizing/read-size contract mismatch in the selected TTNN embedding implementation.  The observed device stop, abort, and test interruption are downstream watcher actions after a real NOC/CB overflow was detected.
- The model's logical QSA arithmetic was not wrong in this watcher failure; normal PCC remained above the functional bar.  The failure was that the chosen kernel path was not watcher-safe for TILE UINT32 index pages.

## Proposed Fix

- In `models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py`, convert every QSA embedding index tensor to `ttnn.ROW_MAJOR_LAYOUT` immediately before `ttnn.embedding`, while requesting `layout=ttnn.TILE_LAYOUT` for the output.
- This keeps the decoder fully on device and traceable, avoids `ttnn.from_torch`/`ttnn.to_torch`/host gather fallback, and forces TTNN to use the fused/RM embedding route instead of `embedding_ind_tilized`.
- Apply the same helper to RoPE row embedding, pooled compressed-key embedding, and main K/V cache embedding so the QSA path has no remaining `embedding_ind_tilized` program.

## Uncertainty

- This is a model-scope workaround, not a TTNN core fix.  The underlying `EmbeddingsTilizedIndicesProgramFactory` still appears to need a source fix: either size `INDEX_SCRATCH` for the actual TILE page size or read only the intended face-sized index chunk.
- Focused verification after the workaround:
  - `pytest -q -s ...::test_real_weights_hf_prefill_decode_pcc[3]` passed with `prefill_pcc=0.99679226`, `decode_pcc=0.99988198`.
  - Watcher rerun in `watcher_qsa_fix_20260826_1251/` passed and `rg` found no `critical`, `TT_THROW`, `NOC transaction`, `overflows`, `sanitize`, `fatal`, `Aborted`, `FAILED`, `ERROR`, or `embedding_ind_tilized` entries.
