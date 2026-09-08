# AutoDebug: generator serving contracts

Source inspection and CPU AST probe, 2026-09-08, before implementation.
No device was opened. This is the serving-contract hypothesis group delegated
by the supervising AutoFix agent; adapter/plugin work is separate.

## Evidence and predictions

- `OrnithModel.allocate_cache` accepts only batch/context and computes
  `num_blocks_for_context(context, page_block_size) * batch_size`. An explicit
  scheduler physical pool cannot be represented; passing `num_blocks` raises
  `TypeError`. Preserve the native logical context while allocating the exact
  requested block count, and require a scheduler page table for a shared pool.
- Generator AST has no `read_output_async`, `tokens_from`, `logits_from`,
  `remap_serving_slots`, or `refresh_serving_inputs`. The pinned plugin's
  `async_decode.py` submits decode then calls adapter `read_decode_output`;
  output must be copied before the next replay overwrites persistent buffers.
- Pinned plugin `model_runner.py::_decode_state_slot_remap` emits a full
  permutation, including off-batch holders. Recurrent/conv state, token and
  current/RoPE inputs, seeds and penalty histories currently remain in old
  rows. A swap followed by decode would associate requests with another row's
  state. All sources must be saved before writing destinations.
- `_replay` unconditionally submits the sampling trace in device mode.
  `return_logits=True` changes only readback, so optional host sampling still
  mutates device feedback/RNG/history. An explicit per-step sampler switch must
  leave captured trace identities intact.

## Probe

`USER=hous python_env/bin/python` with `ast.parse` over the two source files
printed their actual method signatures, missing methods, cache allocation
expression, and replay condition. All four predictions are directly confirmed
at the Python contract boundary. Follow-up focused tests will execute these
actual methods with torch-backed TT boundaries, including cyclic row moves,
large exact integer seeds, stale scheduler inputs, and queued read snapshots.

## Reference and limits

The pinned Ornith-1.0-35B generator's deferred copy/event split and model's
snapshot-before-write row selection establish existing repository patterns.
9B uses batch axis zero for all hybrid state, 32 sampler lanes, and its own
canonical split sampler. Preserve those contracts. CPU tests can prove Python
orchestration and selections; hardware must separately validate TT gather/
where layouts, trace recapture after boundary programs, and async read ordering.
