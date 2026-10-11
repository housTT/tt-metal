# Split sparse SiLU diagnosis

Source-only AutoFix discriminating check, 2026-10-10. No TTNN import, device access, test execution, or implementation edit was performed.

**Verdict:** the inspected sparse matmul path does ignore an explicit `program_config.fused_activation`. A split implementation that relies on it and then multiplies the raw gate and up results computes `gate * up`, not `silu(gate) * up`. The current consumer-side SiLU restores the intended expression, and its gate/up ordering is correct. This proves the source defect; attributing the historical PCC collapse solely to it still requires a matched device check and runtime source provenance.

## Exact lowering

- `ttnn/cpp/ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:56-78` constructs `MatmulParams` with `user_fused_activation = nullopt`, but that is **not by itself the cause**: `device/config/matmul_program_config.cpp:1022-1023` returns an explicitly supplied program config unchanged. The source comment in `tt/optimized_decoder.py` should describe the missing sparse-kernel activation lowering rather than imply that `nullopt` overrides the explicit config.
- The sparse factory extracts tiling fields from the config around lines 86-96 but never reads its `fused_activation`. It sets `FUSE_ACTIVATION = 0` at line 525 and never emits `SFPU_ACTIVATION` or named `activation_type`/parameter arguments. At lines 898-905 it passes these definitions to `bmm_large_block_zm_fused_bias_activation_metal2.cpp`.
- That compute kernel guards activation initialization and application with **`SFPU_ACTIVATION`**, at lines 22, 288, 467, and 654. These branches are absent from the sparse build. `FUSE_ACTIVATION=0` is supporting evidence, not the actual guard consumed by this kernel. Dense matmul's corresponding factory does emit `SFPU_ACTIVATION` and activation arguments, providing a concrete comparison.
- The current `ttnn.multiply(gate, up, input_tensor_a_activations=[SILU])` passes the activation list as mathematical LHS activations through `eltwise/binary/binary.cpp`. `binary_ng/device/binary_ng_program_factory.cpp:1017` emits the LHS activation chain. The no-broadcast compute path calls `PREPROCESS(LHS, ...)` before `BINARY_OP` (`eltwise_binary_no_bcast.cpp:39-52`); the helper applies the chain to gate tiles and packs their intermediate result. `unary/common/unary_op_utils.cpp:917` maps SILU to `silu_tile_init()` / `silu_tile(...)`. Thus the consumer computes `silu(gate) * up`, with device activation approximation and intermediate/output rounding still requiring numerical validation.

The sparse factory inspected here has SHA-256 `b93d3c1374a34dfc3b1bbf3336ea3eb44d1a921e85d9d7d72ee5025553b02d5f`.

## Weight ordering and observed failure

The checkpoint's fused expert tensor is `[gate, up]` on its output-row dimension. `models/demos/blackhole/qwen36/tt/moe/weights.py:75-79` separates those halves and transposes them, then line 142 explicitly concatenates **`[up, gate]`** on the TT tensor's last dimension. Consequently `optimized_decoder.py` taking `[..., :512]` as up and `[..., 512:]` as gate is correct for the real 512-wide expert intermediate. TTNN SwiGLU likewise computes first-half times SiLU(second-half), as shown by `eltwise/unary/device/unary_composite_op.cpp:301-309`.

`expert_real/split_bfp4.json` records prefill PCC `0.9993682504`, overall decode PCC `0.6710391641`, and minimum token PCC `0.3326610923` at zero-based token 5. The log ends in the `decode_pcc_min >= 0.995` assertion. Repeatability was reported true. These measurements fit a deterministic decode-only missing nonlinearity, but do not uniquely establish it. The historical implementation hash is `4a86f49d2424bbc74408747ff919cd0d4360ad9e836a147041fcf656b002ea8a`; current source has changed. The experiment-provenance handoff says those harness hashes were computed after execution, so they cannot prove the exact imported historical bytes.

## Smallest discriminating device check for the owner

1. Freeze source, build identity, policy, and input manifest before launching. Keep the failing policy's BFP4 gate/up, BF16 outputs, 8 gate cores, K block 16, subblock 2, and eight active experts in the non-indexed sparse path (`nnz=8`). Use identical uploaded tensors and routing for all arms.
2. For one gate projection with positive and negative values, compare sparse results with config activation `None` and `SILU`; also compare to an explicit `ttnn.silu(raw_gate)`. The hypothesis predicts that the two sparse outputs agree, while the explicit SiLU differs. Inspect absolute errors and negative entries; PCC alone can conceal missing nonlinearities for unsuitable inputs.
3. On the same gate/up outputs compare plain multiply, consumer-fused SiLU multiply, explicit SiLU followed by multiply, and SwiGLU of `[up, gate]`. The corrected consumer should agree within the expected numerical tolerance with the latter two and disagree materially with plain multiply. This also detects a half-order reversal independently of the projection.
4. Replay the recorded layer-3 fixture through the original 16-token decode and compare a frozen control with the corrected split candidate, changing only activation placement. Preserve all per-token PCCs, especially token 5, and require successful process exit. A restored intermediate expression alone does not establish the full-layer `0.995` gate or latency improvement.

No device recovery, PCC improvement, or performance improvement is claimed by this report.
