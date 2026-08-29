# Optimized multichip decoder work log

## 2026-08-29: stage start and audit

- Starting commit: `b0d2fb3c` (`Record GPT-OSS multichip decoder stage commit`).
- Local checkpoint cache: `openai/gpt-oss-120b`, revision
  `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, reported by `hf cache list`;
  no network download was required.
- Hardware inventory: four local Blackhole `p300c` devices, IDs 0--3, all
  reported resettable by `tt-smi -ls --local`. These are the four physical
  devices used for the P150-family 1x1/1x2/1x4 contracts in this checkout.
- Read the completed multichip implementation, correctness/performance logs,
  profiler CSVs, context contract, and prior clean stage review. Read the
  repository LLM optimization guidance and relevant 1D attention/MLP/RMSNorm
  implementations.
- Wrote `operation_topology_audit.md` before implementation changes. It fixes
  the full-family measurement rules for residual layout, collective placement,
  fused CCL+matmul, packed projections, dtype/fidelity, and persistent buffers.
- No implementation changes have been made yet. Fresh baseline commands and
  results follow below.

## 2026-08-29: fresh accepted baseline

- Wrote fresh P150 layer-0/layer-1 and batch-2 producer artifacts from the
  cached real checkpoint.
- P150 control: sliding prefill/decode 36.594683/0.530660340 ms; full
  31.935862/0.528483690 ms.
- Completed-stage TP2 baseline: sliding 47.748477/0.752601540 ms with
  prefill/decode PCC 0.991984446/0.983523316; full
  47.438676/0.752338440 ms with PCC 0.991051914/0.960386940.
- Completed-stage TP4 baseline: sliding 25.895578/0.652817770 ms with PCC
  0.991819332/0.984312881; full 25.544376/0.653009110 ms with PCC
  0.991578193/0.962574523.
- Every latency used 100 warmed traced replays and five samples. The public
  prefill length was the non-aligned logical length 127.

## 2026-08-29: movement and residual contract

- Added a decode backend that borrows the input residual, writes each residual
  add into its branch output, and leaves the final BF16 L1 tensor for the next
  layer. This removes the per-layer DRAM clone without changing ownership.
- Replaced both single-core decode norms with 10-core L1 width-sharded norms.
- Repeated the lower-movement family after adapting the first validation
  failures. Results through the next norm and packed QKV were:
  physical-replicated 0.112158940 ms, async RS+AG 0.117812920 ms, persistent
  async 0.113428670 ms, reduce-scattered residual plus distributed norm plus
  fused AG+QKV 0.135073110 ms, and persistent fused 0.131030210 ms. PCC was at
  least 0.9999136.
- Kept the physical-replicated boundary: no layer-to-layer collective, 1.1%
  faster than persistent async, and 16.8% faster than the stack-compatible
  sharded/fused path. The sharded candidate was not restored immediately to
  the old layout for timing.
- Expert logical-width CCL was 0.054117040 ms versus runtime-padded physical
  width 0.096322570 ms (1.7799x). Kept logical 2880.
- Evidence: `artifacts/20260829_topology/bf16_async_fused_persistent.log.gz`.

## 2026-08-29: projection topology and fused CCL

- DRAM-sharded QKV was adapted to legal rank, memory, and program configs and
  lost on TP2/TP4. The final-policy TP4 comparison was about 0.438091 ms versus
  about 0.4275 ms.
- Explicit O projection also lost: final-policy TP4 was 0.428899 ms versus
  about 0.4275 ms.
- Three separate real-weight Q/K/V matmuls plus concat produced TP4 sliding
  prefill 24.916073 ms and decode 0.500701600 ms (PCC 0.981589952). Kept packed.
- Separate gate/up sparse matmuls produced TP4 sliding prefill 23.720292 ms and
  decode 0.506294350 ms. Kept packed gate/up.
- TP4 fused O+MMRS passed at adapted 3072 and native 2944. The isolated region
  improved 0.234033 -> 0.080770 ms (adapted) and 0.261526 -> 0.082997 ms
  (native), but the integrated traced layer regressed to 0.414726 ms from
  about 0.3966 ms after its compatible gather/residual boundary.
- TP2 fused MMRS was retried at padded 3072 and native 2880; both hung. Triage
  and bounded reset/list/mesh-smoke recovery completed. TP4 is the passing
  source-backed control.

## 2026-08-29: dtype and fidelity matrix

- Selected BFP8 attention weights and KV, LoFi decode projections, prefill
  HiFi2 packed QKV and LoFi O projection, BFP4/LoFi expert weights, BF16
  router, BFP8 decode attention CCL, BF16 prefill attention CCL at the measured
  logical length 127, and BF16 expert CCL in both phases.
- Attention BFP4 on the final packed real-weight topology failed prefill PCC at
  0.829482726. HiFi4 decoded at 0.479016840 ms; HiFi2 also lost to LoFi.
- BFP8 experts gave 19.239024 ms prefill and 0.414414070 ms decode. BF16 experts
  gave 27.932937/0.468664700 ms. Both passed PCC but lost to BFP4.
- Router BFP8/BFP4 decoded at 0.396339970/0.396170450 ms. BFP4 reduced prefill
  PCC to 0.985043263; neither materially beat BF16.
- Decode attention BFP8 CCL passed all four mesh/layer cases. This sweep acts
  on the custom decode tail only; inherited prefill selects BF16 for its
  attention projection and fused RS+AG at length 127, as the final profiler
  rows prove. Expert BFP8 regressed TP4 decode to 0.464593140 ms. Decode
  attention BFP4 was adapted past its first API error but trace-refresh PCC was
  0.885471. Expert/global BFP4 executed but prefill PCC was 0.92932186.

## 2026-08-29: sparse geometry and DRAM-sharded decode

- Preserved router-selected top-4 indexed sparse execution throughout.
- Started from gate/up 12 cores and down 30. Tried gate/up 9, 15, 30, and 45
  cores and down 15, 18, 45, and 48. Initial 30/48 validation failures were
  followed by legal shape/subblock adaptations.
- Representative TP4 decode: gate/up 15 cores 0.408534610 ms, 9 cores
  0.434787660, 45 cores 0.402051970; down 45 cores 0.429580430, 15 cores
  0.426184870, 18 cores 0.426371150. Selected 45-core gate/up and 15-core down.
- `tt-perf-report` exposed TP2's legal wider gate/up subblock. TP2 `1x2`
  improved sliding/full to 0.428611430/0.425780430 ms without a PCC delta and
  was promoted. TP4 `per_core_N=1` makes width 2 illegal, so `1x1` is its exact
  legal maximum.
- Added decode-only DRAM-width-sharded O with internal 3072 padding/slicing.
  TP4 whole-layer core sweep: 16 0.396698770, 8 0.398640400, 4 0.400045950,
  2 0.396732730; native won. TP2: 16 0.480186320, 8 0.480278960,
  4 0.483059400 before the final sparse improvement. Promoted TP2 16-core only.
- Source audit found no DRAM-sharded `ttnn.sparse_matmul` factory. The routed
  implementation uses interleaved
  `SparseMatmulMultiCoreReuseMcast1DProgramFactory`, so the supported sparse
  core/block/subblock space was swept instead.

## 2026-08-29: autofix and hardware recovery

- A speculative phase-specific prefill geometry produced exact rank-replication
  failure. `$autofix` isolated it, proved the failure bit-identical, refuted the
  hypothesis, and reverted it.
- Async/persistent experiments then left stale device/fabric state and the
  unchanged default reproduced the failure. Ran `timeout 180 tt-smi -r`,
  `timeout 60 tt-smi -ls --local`, and a 1x4 mesh smoke. All four devices and
  `MESH_SMOKE_OK` returned; unchanged default passed.
- Report: `autofix/AUTOFIX_prefill_replication.md`.

## 2026-08-29: pre-review default, stress, watcher, and profiler

- Pre-review 100x5 default, superseded by the review refresh below: TP2
  sliding 24.305916/0.428666630 ms, TP2 full
  23.956320/0.425765000, TP4 sliding 15.470742/0.396577620, TP4 full
  15.594027/0.396495060. All PCC, cache, refresh, and rank checks passed.
- 1000x3 stress passed all four. TP4 sliding/full reproduced
  0.396258894/0.396098337 ms. Batch-2 high-position passed all four, including
  page-table-only refresh and replay determinism.
- Fully enabled `TT_METAL_WATCHER=10` first failed before model execution: the
  inlined ACTIVE_ETH program was 29072 bytes for a 26624-byte kernel-config
  buffer. The adapted no-Ethernet run kept Tensix, dispatch, and asserts and
  passed all four, but was rejected because it disabled Ethernet watcher.
- `$autofix` proved that global and selective no-inline candidates admitted the
  program but failed after real model traffic. An instrumented TP2-full run
  passed PCC, then watcher caught a subordinate ACTIVE_ETH core at kernel
  handoff with `DebugAssertNCriscNOCPacketTagClearedTripped`; write/atomic NoC
  packet tags remained sticky after router teardown.
- Retained the minimal repair: after write and atomic barriers, router teardown
  calls `noc_clear_packet_tags(noc_index)`. Reverted every diagnostic waypoint,
  no-inline, sanitizer, and 30-KiB HAL candidate.
- The repository-supported `TT_METAL_FABRIC_OPT_LEVEL=Os` override admits the
  normally inlined router without disabling watcher features. The final
  `TT_METAL_WATCHER=10` matrix passed TP2/TP4 sliding/full in one process. Every
  reopen reported `disabled features: None`; the process and cluster closed
  cleanly, and all four devices remained visible/resettable. Evidence:
  `artifacts/20260829_watcher/autofix_watcher_fabric_os_packet_tag_clear_all.log.gz`
  and `autofix/AUTOFIX_watcher_active_eth.md`.
- Pre-review raw profiler and `tt-perf-report` CSVs/tables exist for TP2/TP4 and both
  layer kinds under `artifacts/20260829_profiler_release/`. Device sums are
  387.724/388.227 us for TP2 and 361.848/361.478 us for TP4.
- Current advice was reviewed. TP2's small sparse subblock was fixed. TP4's
  `1x1` is its exact legal maximum; HiFi advice lost traced latency; DRAM
  sharding and fused-CCL advice have the evidence above.

## 2026-08-29: repository checks

- `python_env/bin/pytest -q models/autoports/openai_gpt_oss_120b/tests/test_multichip_decoder.py`:
  5 passed, 14 hardware/acceptance probes skipped by their explicit gates.
- The retained fabric router source was JIT-compiled and exercised on real
  hardware by the strict watcher tests. The final-source prescribed wrapper
  configure check passed:
  `.github/scripts/copilot-build.sh --build-dir build_copilot_optimized_multichip_final --configure-only`.
  Evidence is
  `artifacts/20260829_watcher/final_wrapper_configure_only.log.gz`. A full
  wrapper compile remains unverified: Garage credentials are unavailable and
  the wrapper warned that a cold build takes over an hour and will most likely
  not finish.

## 2026-08-29: first review findings and final-policy refresh

- The first independent `$stage-review` returned `more-work-needed`: the
  prefill expert geometry had not been swept through the full layer, router
  profiler advice lacked explicit evidence, trace allocation tracking was
  claimed without the environment variable, and fidelity wording conflated
  decode with prefill.
- Added phase-specific expert down geometry and measured down-30 and down-45
  through TP2/TP4 sliding/full. Down-30 regressed TP2 to
  27.096822/26.747726 ms while improving TP4 to 14.700000/14.436056. Down-45
  won all four at 22.139411/21.656049 and 14.346312/14.010952 ms, so the final
  default uses 45-core prefill down and retains 15-core decode down.
- Tried the profiler's explicit router recommendation with a legal 4x4,
  `in0_block_w=2` config. It was at most 1.1% faster but reduced prefill PCC in
  three of four cases. An explicit DRAM-to-L1 copy was neutral/slower when
  paired with that config. The L1-only isolation preserved PCC but was mixed:
  21.956250/21.700090 ms on TP2 and 14.455629/14.062288 on TP4. Retained the
  automatic router config and DRAM input.
- Removed the unsupported trace-allocation-tracking claim and qualified
  fidelity by phase in the README and context contract.
- Authoritative refreshed 100x5 default: TP2 sliding
  21.988971/0.428405910 ms, TP2 full 21.679801/0.425538640, TP4 sliding
  14.374113/0.396413630, TP4 full 14.041818/0.396353620. PCC is unchanged from
  the accepted final baseline. Final 1000x3 stress and batch-2 high-position
  matrices passed all four cases.
- Repeated fully enabled watcher validation with `TT_METAL_WATCHER=10` and
  `TT_METAL_FABRIC_OPT_LEVEL=Os`; all four cases passed with no disabled
  features and clean shutdown. Evidence:
  `artifacts/20260829_watcher/default_after_review_watcher10_all.log.gz`.
- Fresh final-default profiler evidence is under
  `artifacts/20260829_profiler_after_review/`. Decode device sums are
  389.214/387.509 us on TP2 and 360.311/362.191 us on TP4; prefill sums are
  21.121/21.070 ms and 13.311/13.268 ms. `tt-perf-report` tables and CSVs were
  regenerated for every mesh/layer pair.

## 2026-08-29: independent stage-review closure

- The fresh xhigh rereview found one remaining documentation-contract issue:
  decode-only BFP8 attention CCL and L1 residual behavior was described without
  phase qualification even though the inherited measured prefill path uses
  BF16 collectives and a DRAM-interleaved boundary.
- Corrected the README, topology audit, context contract, work log, and runtime
  optimization manifest to state the phase-specific contracts. Final profiler
  rows prove BF16 fused RS+AG and BF16 DRAM output for logical prefill length
  127, versus BFP8 attention CCL and BF16 L1 output for decode.
- Repeated the host policy/fallback tests and formatting checks after the
  correction. The independent rereview returned `VERDICT: clean-pass`.

## Stage commits

- Starting multichip-decoder stage: `b0d2fb3c`.
- Optimized implementation and evidence: `7ce49646`.
- The final log-only follow-up commit cannot contain its own SHA; that SHA is
  reported in the stage handoff.
