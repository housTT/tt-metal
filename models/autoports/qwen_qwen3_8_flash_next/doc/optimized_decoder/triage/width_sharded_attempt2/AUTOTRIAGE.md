# AUTOTRIAGE

## Diagnosis

- The width-sharded residual candidate deallocated the sharded projected residual without preserving a possible output alias from `ttnn.mac`. The following hyperconnection then received an unallocated tensor. Pytest's failure handling stalled after the host-side `Tensor is not allocated` exception; this was not a running device-kernel deadlock.

## Triage Evidence

- `dump_running_operations` and `dump_op_mesh` showed no running operation and an idle device.
- Dispatch cores were waiting normally for upstream host work; watcher, L1, ARC, and active-CB checks passed.
- The console stopped immediately after compiling the sharded ternary writer and emitted `TT_THROW: Tensor is not allocated`.
- Binary-integrity mismatches and Ethernet NoC counters were observed after the exception, but no functional worker was running. They are teardown/cache-state symptoms rather than the first stuck point.

## Source Evidence

- `OptimizedDecoder._hyper_inject` created `out = ttnn.mac(projected_sharded, ..., memory_config=the_same_sharded_config)` and then unconditionally deallocated `projected_sharded`.
- TTNN view/output aliasing must be guarded with the decoder's `_free(source, live_output)` helper; the surrounding fused graph consistently uses that rule.
- The first attempt's 10x4 program grid versus 8x5 output-shard grid was separately fixed before this run, so it does not explain the second stop.

## Downstream Effects

- Pytest could not finish the traceback/teardown after the unallocated tensor reached the next decoder subgraph.
- Dispatch and device-idle observations are downstream of the host exception and do not indicate a worker-kernel producer/consumer mismatch.

## Proposed Fix

- Replace unconditional `ttnn.deallocate(projected_sharded)` with `_free(projected_sharded, out)` so an aliased output buffer stays live.
- Rerun the exact candidate, then retain it only if real-weight PCC and traced latency beat the interleaved indexed-expert control.

## Uncertainty

- The candidate still needs a successful retry to prove that TTNN selected an aliasing implementation and that no later sharded-layout constraint fails.
