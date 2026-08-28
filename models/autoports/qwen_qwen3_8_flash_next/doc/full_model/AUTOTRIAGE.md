# AUTOTRIAGE

## Diagnosis

- The non-mux Linear `all_gather_async` writer unconditionally retrieves the selected forward/backward fabric connection before checking whether that endpoint direction has any targets.  A line endpoint legitimately has no outward connection, so the getter trips `ASSERT(has_forward_connection())` (or its backward equivalent).

## Triage Evidence

- The original full-48 token-out watcher run stopped device 0, worker core `(0,0)`, in BRISC kernel `minimal_default_writer.cpp`; the focused one-layer split-greedy trace reproduced the same core, kernel, and reported line.  This refutes decoder depth, expert service, PLE, and the long full-stack CCL schedule.
- The host stack stopped in `sampled_tokens_to_torch`; that read is only the synchronization surface for the earlier asynchronous failure.
- DWARF for the cached BRISC kernel maps the reported writer site to `fabric_connection_manager.hpp:119`, `ASSERT(has_forward_connection())`.

## Source Evidence

- `Sampling1D._argmax_all_gather` calls experimental Linear `all_gather_async` with `argmax_num_workers_per_link=1`.
- A single worker per direction disables `USE_WORKER_MUX`; the program factory still launches both directional workers.
- `minimal_default_writer.cpp` opens the connection manager and immediately calls either `get_backward_connection()` or `get_forward_connection()` unconditionally.
- The later barrier, local-send, and forwarding paths already use `detail::valid_targets(direction)` or compile-time target counts.  For the disconnected outward endpoint direction, `writes_expected` is zero, so no fabric sender is consumed.

## Downstream Effects

- Watcher intentionally stops the device at the assertion.  The Python abort, failed compact token readback, and teardown loss are downstream consequences, not separate model failures.

## Proposed Fix

- In the generic non-mux writer, retrieve a directional connection only when the connection manager reports it exists.  Preserve an assertion that a null pointer is valid only when that direction has no targets.  This leaves all valid sender paths unchanged and permits the required no-op outward worker at a Linear endpoint.

## Uncertainty

- The focused and original watcher checks must be rerun after rebuilding the kernel.  The existing non-watcher passes do not prove the assert path is repaired because watcher is what exposes device assertions.
