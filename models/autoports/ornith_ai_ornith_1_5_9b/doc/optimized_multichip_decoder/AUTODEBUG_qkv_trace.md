# AutoDebug: final B32 traced full-attention PCC

Source-only investigation, 2026-09-05. This is the fresh isolated investigator;
the coordinating agent owns all hardware experiments. No model implementation
was edited during diagnosis.

## Starting evidence

`logs/final_default_v1_watcher_contracts.log` fails
`test_multichip_decoder.py::test_traced_decode_pcc[blackhole-32-full_attention-mesh_device0-device_params0]`
at the unchanged per-user PCC gate: `0.9942707596653978 < 0.995`.
No step summary precedes the failure, so this is **step 0, position 63**;
the assertion does not report the user. Positions 64/65 were never adjudicated.
The same log passes B1/B4 full-attention and all three linear-attention batches.
The historical and current ordinary B32 inputs differ from this trace contract.

## Exact contract and precision ledger

- Layer 3; recorded real layer activations. Prefix indices are
  `(arange(63) + 31 + user * 137) % source_length`; decode step `s` uses
  `(3100 + s + user * 137) % source_length`.
- TP4, B32, context allocation 1024, identity disjoint page-table rows,
  model `num_blocks_for_context`, page size and local head counts.
- Prefill attention/QKVG weights: **BFP4**, from `PrecisionPolicy.attention`.
  MLP gate/up/down weights BFP4; projection activations BF16; LoFi,
  `math_approx_mode=False`, `fp32_dest_acc_en=False`, `packer_l1_acc=True`.
- Decode QKVG: BFP4, DRAM width-sharded, 8 input cores, K block 16,
  2 readers/bank; output storage default `ceil(local_N/(32*8)) = 10` tiles/core.
  K/V caches remain BFP8. Native CCL carries the existing BF16 residuals.
- MLP decode uses DRAM roles, 8 input cores, gate/up K block 8,
  down K block 6, 2 readers/bank.
- Prefill uses `paged_fill_cache`; decode uses `paged_fused_update_cache`
  and `paged_scaled_dot_product_attention_decode`, SDPA K chunk 256.
- Trace fixture snapshots each rank's distinct state with device clones,
  copies state back before capture and replay, and verifies exact residual
  replication on host read. Inputs, positions and RoPE rows are copied into
  persistent device buffers before each replay.

## Ranked hypotheses and decisive experiments

1. **Decode QKVG quantization is sensitive to an omitted real input.**
   Current change lowered decode QKVG; first replay failure occurs before any
   low-precision decode history accumulates. A matched BFP8 run at **8 cores,
   block 16, readers 2**, sharing BFP4 prefill and BFP8 cache, isolates this
   boundary. `production_candidate` additionally changes geometry to 32/block4
   and alone cannot distinguish precision from geometry. Localize with separate
   Q, K, V and gate substitutions using a raw-weight BFP8 projection if needed.

2. **DRAM geometry or compute accumulation changes the result enough to fail.**
   BFP4 is not rejected by the first failure. Compare 8/block8 and 32/block4
   against 8/block16 at unchanged dtype/readers. Compare QKVG-only HiFi2/HiFi4
   and FP32 destination controls independently. Per-field raw-Torch projection
   comparisons distinguish quantization from arithmetic error.

3. **Trace state/input corruption.** The existing single-chip helper explicitly
   avoids device snapshots because of historical trace-pool aliasing; TP fixture
   intentionally retains all local ranks using device clones. This historical
   comment is not evidence that current live clones alias. Compare eager and
   trace from the exact same prefilling snapshot, check snapshot content digests
   after compile/capture/replays, and repeat changed-input replay after eager.
   Exact eager/trace outputs and unchanged snapshot hashes refute this mechanism
   for the failing inputs.

4. **Cache row/page boundary error.** Position 63 is the last row of the first
   64-token page, with ample allocation for rounded SDPA reads, not an
   allocation edge. The helper allocates 32 blocks/user for context 1024.
   Still record all positions 63/64/65 and page/tile/chunk coordinates. The
   existing cache report's longer-prompt exact-row evidence does not substitute
   for this exact input. If eager and trace differ, or failure has a boundary
   cliff, run the exact same-cache SDPA oracle and fused-vs-independent update
   probe before attributing it to cache precision. KV dtype must stay BFP8 in
   QKVG controls.

## Runtime adjudication by the coordinating agent

`qkv_trace_b4_b32.json` reproduces the original value exactly, now localized to
**user 31, step 0**, on every rank. Initial replay, restored eager execution,
and replay after eager produce bitwise-identical outputs and final K/V caches.
The eight saved per-rank cache payloads retain their initial digests through
warmup, capture and all three phases. Positions 64/65 pass every user. This
refutes trace-state corruption and a page-64 failure cliff for the exact input.

`qkv_trace_b4_user31.json` extracts that same request into B1 and still fails
step 0 at 0.9943646417. A batch-dependent precision fallback is insufficient.
`qkv_trace_b4_hifi2_b32.json` leaves all step minima unchanged from LoFi.
`qkv_trace_b8_b32.json` passes all users/steps, with minima
0.9964473790 / 0.9985126454 / 0.9977029616.

The QKV4/QKV8 and LoFi/HiFi2 comparisons have identical input hashes and
bit-identical initial K/V payloads on all four ranks. Their cache dtype, layout,
allocation, local heads and page tables match. The first-step rescue therefore
localizes sensitivity to decode QKVG with an unchanged prefill cache.

Raw-Torch comparison over the actual normalized TT input reports user 31,
step 0 field PCC ranges across ranks: Q 0.99425–0.99655,
K 0.99512–0.99614, V 0.99343–0.99690, gate 0.97043–0.98902.
Some passing users have lower gate PCC, so this alone does not assign output
causality. Follow-up experiments establish that restoring only the gate from
BFP8 is sufficient: `qkv_trace_gate8_b32.json` passes all users/steps, minima
0.9966528050 / 0.9984664285 / 0.9971419024. Its initial **and final** K/V
payload hashes are bit-identical to the failing BFP4 run across every phase.
This corrects the output without changing cache contents or Q/K/V precision.

QKVG HiFi4 plus FP32 destination accumulation still fails user 31 at
0.9948409006; BFP4 32-core/block4/reader1 geometry fails at 0.9945455234.
These are successful executions that fail accuracy, not resource-error
rejections. The actual mixed QKV4/gate8 implementation passes the original B32
trace selector and all 14 measured whole-layer configurations. The best split
result is 0.275439845 ms; packed QKV8 at 32/block4/reader1 is 0.271286342 ms
(8/block16/reader1 is effectively tied at 0.271270063 ms). The split family is
therefore a verified numerical repair with a measured latency cost.

Retain packed decode QKV8 for all batches and BFP8 KV caches. Gate-only
correction remains test evidence, not a slower production fallback. No
trace/cache/partition source bug remains supported by these experiments.
The coordinating agent retains the established 32/block4/reader1 packed
interface and owns the final complete watcher/topology/performance reruns.
