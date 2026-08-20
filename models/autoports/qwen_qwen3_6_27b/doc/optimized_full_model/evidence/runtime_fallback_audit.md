# Runtime fallback audit

Measured path: `Generator._decode_traced_device` with the selected TP4 full
model and split greedy sampler.

- Model replay is `ttnn.execute_trace(..., blocking=False)` followed by the
  captured sampler replay. There is no synchronization between the pair.
- The sampler receives TP vocabulary shards, performs local max and argmax per
  shard, packs exact global-index candidate tiles, all-broadcasts only those
  compact values/indices on the physical Ring, chooses the global winner, and
  copies it into persistent `tt_out_tok`.
- The final profiler contains `AllBroadcastDeviceOperation`, local reductions,
  and argmax. It contains no `AllGatherAsyncDeviceOperation`, generic
  `TopKDeviceOperation`, or full-vocabulary collective.
- The next model replay consumes that same token buffer. Position, RoPE,
  recurrent/KV state, and unchanged page tables advance or remain on device.
- Page-table host comparison/copy is restricted to explicit setup or a
  changed-table boundary. The steady-state measured loop does not invoke it.
- `model.decode_device` contains no `ttnn.to_torch`, `.cpu()`, host argmax,
  replicated decoder, or single-device fallback.
- `_decode_traced_device` contains no `ttnn.to_torch`, `.cpu()`,
  `ttnn.synchronize_device`, token copy, position copy, RoPE copy, page-table
  copy, or full-vocabulary readback.
- The caller-visible `generate` metric intentionally reads one compact sampled
  ID per token for the Python API response; it is reported separately from the
  no-readback device token-out metric.
- The low-level serving contract can set `read_from_device=false`; a scheduler
  may later issue `read_decode_output(async_read=true)` for the compact
  replicated token only. Processing that deferred read keeps the host penalty
  history mirror coherent without changing the already-advanced device state.
- Host-logit sampling remains an explicit compatibility option and the initial
  prefill/readiness boundary. It is not selected by any optimized measurement.

Audit result: clean for the measured token-out path. No runtime fallback or
per-token host boundary is reachable without choosing an explicit public
observation/compatibility API.

Teacher forcing is intentionally outside this clean autonomous boundary: it
reads the compact predicted ID and writes a ground-truth ID each token. A
4-layer control measures 5.318 ms/token with that boundary versus 5.061 ms for
autonomous token-out; adding an explicit mesh synchronization changes it to
5.380 ms and is rejected. This does not alter the selected serving path.
