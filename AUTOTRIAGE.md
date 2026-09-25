# Qwen3.8 vLLM repeated non-aligned prefill hang

> Historical diagnosis, superseded by the verified resolution below. The
> intermediate statements that no fix was yet claimed describe the captured
> failure window, not the final vLLM stage.

## Final resolution

The final primitive split isolated a tiled route-sparsity reshape after the
max reduction. The retained production repair performs `max -> ROW_MAJOR ->`
metadata reshape, removing the hazardous four-page-to-one-page tiled reshape.
The independently verified sparse-matmul FP32 intermediate-CB sizing fix was
also applied. Diagnostic fences and progress hooks were removed. Focused TT
gates for row-major-first reuse, repeated non-aligned embedding/adapter calls,
and sparse FP32 CB sizing pass, followed by the final all-48 server run:
logical prompt lengths `1, 63, 64, 65, 67, 127, 129` each passed twice, the
full 72-test sampling profile passed, and qualitative/benchmark/lifecycle
workloads completed without a recurrence. Final evidence is under
`models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/`; the detailed repair
ledger is `autofix_full_output_stall_triage.md`.

## Diagnosis

The all-48 vLLM path deterministically wedges both P300 devices on the second
logical-length-63 prefill. Five repeated one-token requests pass, and the first
63-token request passes. The failure is inside layer 0's second routed-expert
wave, before either sparse expert matmul for that wave is launched.

The current narrow boundary is route-sparsity preparation after all-miss expert
slot H2D/D2D and both expert-bank concats have completed. The remaining
operations are dynamic route-index upload, routing-weight gather, grouping
views, sparsity max, and row-major conversion. A final primitive split is in
progress; no production fix is claimed yet.

## Triage evidence

- Failed server logs:
  - `models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/final_virtual_b2_upsample/server.log`
  - `models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_full_output_debug/server.log`
  - `models/autoports/qwen_qwen3_8_flash_next/readiness_vllm/autofix_expert_stage_split/server.log`
- Host stacks first showed `ttnn.to_torch -> _read_compact_route_ids`; later
  diagnostic fences proved that read was only the first downstream completion
  boundary.
- Live `tt-triage` could not read either device after the wedge: both devices
  returned all-NoC MMIO timeouts at TENSIX `(0,0)` address `0xffb121b0`.
- The most precise completed marker sequence on the failing request is:
  `embedding`, layer-0 router, wave-0 upload/concat/route preparation/both
  sparse matmuls/reduction, wave-1 upload, and wave-1 bank concat. The next
  aggregate route-sparsity fence never completes.
- The layer-0 wave-1 plan is all misses and is disjoint from the capacity-10
  first wave.
- Every wedged run was stopped before reset; process audits found no vLLM or
  EngineCore holder. Devices 0 and 1 required and received a paired reset, then
  returned healthy DRAM and heartbeats.

## Source evidence

- `Qwen38FullModel.prefill_forward` reaches layer 0 and uses the host-backed
  routed-expert path.
- `MultichipDecoder._routed_experts` services unique routed experts in
  capacity-10 waves. `_routed_expert_wave` materializes gate/down banks and
  gathers per-wave routing weights before sparse matmul.
- H2D directly enqueues the live destination; D2D copy and concat register and
  repatch all live buffer addresses on cache hits. Tensor specs, including
  logical shapes, participate in cache keys, refuting a simple variable-wave
  width cache collision.
- Current HEAD lacks upstream `1f4441216b3`, which sizes the sparse-matmul FP32
  intermediate CB from its actual FP32 format instead of BF16. Qwen's selected
  BFP4/LoFi expert policy exercises that two-times underallocation. Stage
  fences show the final failing wave has not launched sparse matmul, so this is
  a verified correctness defect but not yet the direct hang trigger.

## Downstream effects

- The EngineCore worker remains inside synchronous prefill; the API request
  never completes. This is not an SSE, AsyncLLM, scheduler, or final-output
  delivery problem.
- Both device NoCs become unreadable and require paired recovery after process
  cleanup.
- Final non-aligned serving, qualitative, benchmark, and stage-review gates
  cannot be accepted until the repeated-length repro passes without diagnostic
  fences.

## Proposed fix

Complete the route-sparsity primitive split. Keep only a fix whose isolated
prediction is observed on hardware, add a repeated all-48 regression, remove
all diagnostic fences from the production path, and separately apply upstream
`1f4441216b3` with its sparse-matmul regression because the selected precision
path provably exercises the underallocated FP32 intermediate CB.

## Uncertainty

The exact failing route-sparsity primitive is not yet known. The leading
lifetime hypothesis is the transient device gather-index tensor being
deallocated immediately after asynchronous gather and reused under cached fast
submission, but that remains a hypothesis until the next marker run.
