# AutoFix report: fused and persistent CCL

## Starting evidence

- Fresh source diagnosis: `AUTODEBUG.md` in this directory.
- Prior generic fused-op evidence:
  `doc/multichip_decoder/fused_ccl_probe.log` and
  `doc/multichip_decoder/autofix/fused_decoder/AUTOFIX.md`.
- Original selected decode used sharded `ttnn.all_reduce`, whose composite
  implementation converts sharded input to interleaved and back.

## Hypothesis experiments

### Explicit persistent all-reduce removes material conversions

- Hypothesis: the explicit-buffer `all_reduce_async` overload accepts the
  decoder's BF16/TILE/L1 width-sharded `[1,1,32,5120]` row partial and is
  faster than the standard composite operation.
- Experiment: `test_decoder_shape_persistent_all_reduce` in
  `test_ccl_candidates.py`, on the target 1x4 P300c Ring with two links, exact
  8-core `[32,640]` and 16-core `[32,320]` output shards, a four-times-larger
  persistent shard, 10 warmups, and 100 traced replays.
- Result: all four replicas passed at PCC 0.9999957. Eight cores measured
  60.082 us standard versus 25.641 us persistent (2.343x); sixteen cores
  measured 60.422 us versus 23.943 us (2.524x).
- Verdict: **verified**.
- Evidence: `persistent_attempt1.log`.
- Fix: decode-only persistent all-reduce with caller-owned buffers,
  semaphores, and a full-worker subdevice. Resources are mesh-scoped and
  shared by every decoder layer. Three independent boundary slots are kept:
  attention/16-core, full-MLP/16-core, and linear-MLP/8-core. This is a fixed
  3.75 MiB/device/mesh reserve rather than a per-layer reserve. The selected
  default remains disable-able with `QWEN36_MC_PERSISTENT_CCL=0`.
- Verification: `lazy_shared_pool_sequential.log` constructs and runs linear then
  full decode on the same mesh, proves stable buffer identity and exactly
  three slots, and passes trace repeat PCC 1.0. `shared_pool_full_non_aligned.log`
  passes logical sequence length 33 prefill/decode PCC 0.996686/0.997455.

### Real-layer performance benefit survives integration

- Hypothesis: the isolated collective win improves complete traced decode.
- Experiment: default path, 50 warmed trace replays for both meaningful layer
  kinds (`QWEN36_PERF_PHASE=both QWEN36_DECODE_REPLAYS=50`).
- Result: final lazy shared-pool default is 721.692 us linear and 476.388 us
  full. The exact current-source, links=2 non-persistent control was 765.223 us
  and 524.213 us, so the improvements are 5.69% and 9.12%. Lazy decode-only
  initialization leaves prefill on its original worker/collective contract;
  prefill measured 6409.383 us linear / 1995.723 us full in the selected
  both-phase run.
- Verdict: **verified and selected**.
- Evidence: `lazy_shared_pool_default_perf.log` and
  `current_source_nonpersistent_control.log`.

### Fused matmul plus CCL is dimensionally and numerically feasible

- Hypothesis: earlier generic shapes hid a decoder program-family blocker,
  but adapted MinimalMatmul programs can execute exact decode dimensions.
- Experiment: exact attention row projection
  `[32,1536]@[1536,5120]` through `matmul_reduce_scatter_async`; exact fractured
  `[32,1280]` gather feeding MinimalMatmul local widths 3584, 4352, and 4608.
  The first non-fused helper controls were refuted as invalid controls: the RS
  helper used stale persistent-buffer keyword names, and the AGMM helper's
  1x4 non-fused composer produced zero-width host slices. They are not used as
  performance evidence. The fused calls were retried independently with the
  legal two-link/four-worker and MinimalMatmul contracts.
- Result: fused RS passed PCC 0.999962; fused AGMM passed every local width and
  replica at PCC approximately 0.999849.
- Verdict: **API/shape feasibility verified; no speed claim**.
- Evidence: `fused_exact_attempt1.log`. The retained candidate test now
  contains only the valid fused probes; production non-fused evidence comes
  from the real decoder reports rather than the broken helper controls.
- Coherent integration: an opt-in real-layer candidate kept both row outputs
  fractured, fused both row matmuls with reduce-scatter, kept distributed
  RMSNorm outputs fractured, fused their next packed column consumers with
  all-gather+matmul, and reused the MLP gate call's gathered lhs for up. It did
  not immediately restore the replicated residual contract.
- Adaptations tried:
  - the fixed 32-row fused-RS scratch initially leaked padding into one-token
    linear decode; slicing back to the caller's logical rows fixed it and all
    four devices then passed at PCC 0.999917;
  - AGMM rank-2 production weights were reshaped to rank-4 views, clearing the
    first API error;
  - the exact four-core L1 gathered buffer reached kernels but overlapped the
    matmul static circular-buffer region (L1 allocation 953664, CB end 968000);
  - retrying with `persistent_output_buffer=None` and interleaved DRAM avoided
    that allocation contract but failed to make host progress. AutoTriage
    could not attach Inspector data, so this is recorded as a host-progress
    blocker, not asserted to be a device-kernel deadlock.
- Whole-layer evidence: the fused-RS coherent family already lost before AGMM
  could recover it: linear 960.261 us versus 730.243 us replicated (0.760x),
  and full 710.184 us versus 485.937 us (0.684x). The AGMM adaptations did not
  produce a complete measurable real layer. Rejected experimental production
  code was removed; its exact source snapshot remains in
  `fused_candidate_source_snapshot.patch`, with the focused executable probes
  retained in `test_ccl_candidates.py`.
- Verdict: **rejected after coherent integration and adapted retries**.
- Evidence: `fused_rs_real_attempt1.log`, `fused_rs_linear_attempt2.log`,
  `fused_agmm_real_attempt1.log`, `fused_agmm_real_attempt2.log`,
  `fused_agmm_linear_attempt3_dram.log`, and the repo-level
  `doc/optimized_multichip_decoder/triage/fused_agmm_hang/` report.

### Persistent BFP8 follow-up

- Review finding: the original BFP8 attention/MLP candidates took the
  composite `ttnn.all_reduce` branch, so they did not test the selected
  explicit-buffer async family.
- Adaptation: the mesh-scoped pool now keeps dtype-indexed tensors beneath the
  same three stable boundary slots. A decoder requesting BFP8 lazily allocates
  a true `BFLOAT8_B`, TILE, L1 width-sharded persistent buffer with the same
  `[1,1,32,20480]` logical shape and 8/16-core geometry. `_all_reduce_partial`
  always uses the explicit `all_reduce_async` decode branch and passes matching
  input, output, and buffer dtype; BF16 remains the default.
- Exact contract: on `[1,1,32,5120]`, all four replicas passed at PCC
  0.9999418 for both 8 and 16 cores. Persistent BFP8 measured 18.343/18.460 us
  versus persistent BF16 25.541/23.933 us. Returned output and persistent
  buffer both report `DataType.BFLOAT8_B`; this is not composite evidence.
- Real-weight correctness:
  - attention-only: linear nonaligned prefill/decode 0.997619/0.997573; full
    nonaligned 0.996681/0.997514; full trace 0.995436, repeat 1.0;
  - MLP-only: linear nonaligned prefill/decode 0.997960/0.997461; full
    nonaligned 0.996728/0.997324; full trace 0.995199, repeat 1.0;
  - global BFP8: linear nonaligned prefill/decode 0.997570/0.997475; full
    nonaligned 0.996731/0.997409; full trace 0.995243, repeat 1.0.
- Fifty-replay traced decode, same-source persistent BF16 control:

  | persistent payload | linear us | full us | versus BF16 |
  |---|---:|---:|---|
  | BF16/BF16 | 721.502 | 476.934 | control |
  | BFP8 attention | 722.918 | 479.149 | 0.20% / 0.46% slower |
  | BFP8 MLP | 718.964 | 479.494 | 0.35% faster / 0.54% slower |
  | BFP8 global | 719.703 | 480.809 | 0.25% faster / 0.81% slower |

- Initial verdict: **correct and coherently measured, but inconclusive as a
  common default**. Every common BFP8 family regressed the full-attention
  layer, while MLP-only BFP8 showed a small linear-only gain. This finding was
  subsequently resolved with a layer-kind-specific policy and interleaved
  variance controls below.
- Evidence: `persistent_bfp8_exact_attempt1.log`,
  `persistent_bfp8_attention_correctness.log`,
  `persistent_bfp8_mlp_correctness.log`,
  `persistent_bfp8_real_correctness.log`, and
  `persistent_bfp8_perf_{bf16,attention,mlp,global}.log`.

### Layer-kind policy and variance-controlled selection

- The policy now has explicit layer-kind/role precedence:
  `QWEN36_MC_{LINEAR,FULL}_{ATTENTION,MLP}_CCL_DTYPE`, then the role-wide
  override, then the global override. The no-override default is BF16
  attention plus BFP8 MLP for linear-attention layers, and BF16/BF16 for
  full-attention layers. This avoids imposing the linear-layer win on the
  full layer that regressed in the initial common-policy sweep.
- Three interleaved 50-replay traced-decode cycles were run from `/tmp` with
  `env -i`, so the logged environment proves no unrelated `QWEN36_MC_*`
  override leaked into controls. Every log records the instantiated layer
  kind, selected policies, explicit persistent-async branch, links, pool ID,
  allocations, and actual input/output/buffer dtypes.

  | candidate | cycle 1 us | cycle 2 us | cycle 3 us | median us | range us |
  |---|---:|---:|---:|---:|---:|
  | BF16 attention / BF16 MLP | 722.130 | 721.745 | 721.843 | 721.843 | 0.385 |
  | BF16 attention / BFP8 MLP | 718.853 | 719.125 | 719.492 | 719.125 | 0.639 |
  | BFP8 attention / BFP8 MLP | 719.851 | 720.224 | 719.971 | 719.971 | 0.373 |

- MLP-only BFP8 beat BF16 in every matched cycle by 3.277, 2.620, and
  2.351 us; its median gain is 2.718 us (0.377%). Global BFP8 was slower than
  MLP-only in two of three cycles, so it remains rejected.
- Clean no-override correctness after selecting the mixed default passed:
  linear logical-length-65 prefill/decode/trace-repeat PCC
  0.997960/0.997461/0.998331 (repeat identity 1.0), and full decode trace PCC
  0.995413 (repeat identity 1.0). Runtime provenance confirms true BFLOAT8_B
  input/output/persistent-buffer dtype only at the linear MLP boundary and
  BF16 at linear attention plus both full-layer boundaries.
- Evidence: `variance_cycle{1,2,3}_{bf16,linear_mlp_bfp8,linear_global_bfp8}.log`
  and `selected_default_correctness.log`.
- The mesh-shared pool resolves and allocates all semantic slots before first
  decode, independent of layer construction order. The final default owns
  exactly attention/16-core BF16 (1,310,720 bytes), full-MLP/16-core BF16
  (1,310,720 bytes), and linear-MLP/8-core BFP8 (696,320 bytes): 3,317,760
  bytes/device/mesh total. `final_shared_pool_sequential.log` records payload
  dtypes, tensor identities, device-0 addresses, the 640-tile byte calculation,
  and proves identities do not grow or change after sequential linear/full
  execution. `doc/context_contract.json` carries the reduced fixed reserve;
  maximum context is unchanged because the pool remains within the existing
  4 GiB trace/activation/CCL/fragmentation reserve.

### Authoritative no-override final default

- Three independent both-phase runs used a clean environment, warmed prefill,
  and 50 traced decode replays for both layer kinds. No `QWEN36_MC_*`
  override appears in command provenance.

  | layer | metric | run 1 | run 2 | run 3 | median |
  |---|---|---:|---:|---:|---:|
  | linear | prefill us | 6680.525 | 6636.792 | 6698.480 | 6680.525 |
  | linear | decode us/replay | 719.235 | 718.842 | 718.857 | 718.857 |
  | full | prefill us | 2121.701 | 1994.819 | 2008.705 | 2008.705 |
  | full | decode us/replay | 476.323 | 476.643 | 476.422 | 476.422 |

- Evidence: `authoritative_default_run{1,2,3}.log`. These are the final
  current-source default-path numbers, not an earlier candidate result.
- Final runtime fallback source audit passed; evidence is
  `final_fallback_audit.log`.

## Final status

- **Fixed:** explicit persistent/preallocated async all-reduce is the default
  traced decode path with bounded mesh-scoped resources, trace-safe stable
  addresses, accepted PCC, non-aligned prefill coverage, and reproducible
  whole-layer speedups.
- Fused RS and fused AGMM are numerically supported at exact decoder shapes,
  but the coherent real-layer family is rejected by measured RS regressions
  plus the documented L1/host-progress blockers after layout and weight-shape
  adaptations. The rejected path is not reachable from production defaults.
- BFP8 CCL candidates use the same explicit persistent-buffer family as BF16.
  Variance-controlled evidence selects BFP8 only for the linear-attention MLP
  boundary; attention boundaries and the full-attention layer remain BF16.
- Hardware postflight: `timeout 60 tt-smi -ls --local` listed all four P300c
  devices after recovery and cleanup (`final_tt_smi.log`).
