# Multichip decoder work log

## 2026-09-05: baseline and topology selection

- Read `.agents/skills/multichip/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`, `.agents/skills/optimize/SKILL.md`,
  and `tech_reports/LLMs/llms.md` section 3.3.
- Starting branch: `hous/gemma-4-26b-a4b-it`; starting worktree was clean.
- Baseline: `OptimizedDecoder` from local checkpoint `7f1f91d3167`, with
  BF16 KV cache, gate-selected sparse experts, arbitrary logical prefill
  lengths, and traced decode.
- `timeout 60 tt-smi -ls --local`: four P300C Blackhole devices visible.
- 1x4 mesh open/close smoke: `MESH_SMOKE_OK`.
- Host-only multichip contract subset: initially 1 failed / 5 passed because
  decode-only weight selection still depended on the parent `_in_decode_forward`
  flag.  The focused repair keys selection to the explicit multichip phase;
  rerun: 6 passed.
- 1x4 exact hidden all-reduce smoke under `FABRIC_1D_RING`: passed.
- Captured fresh layer-0 and layer-5 optimized single-chip reference tensors:
  `artifacts/optimized_reference_layer{0,5}.pt`; JUnit is
  `artifacts/single_chip_reference.xml` (2 passed).
- Initial TP construction failed before kernels because inherited row-major
  routing requires R22 while the replicated TP candidate selects zero residual
  cores.  AutoFix/AutoDebug started; see `AUTODEBUG.md` when complete.
- Fused all-gather/matmul exact residual-consumer probes passed for sliding QKV,
  full QKV, packed dense gate/up, router, and a fixed expert projection;
  `artifacts/fused_agmm_repro.xml` (5 passed).
- The first fused matmul/reduce-scatter run exposed stale helper API and mesh
  health/config evidence.  Following `$tt-device-usage`, all devices were reset,
  listed, and mesh-smoked.  After reset the discovered topology changed from a
  line degree histogram to `{2:4}`, confirming the initial forwarding error was
  infrastructure state rather than a model conclusion.
- The shape-adapted RS repro now reaches the matmul program validator but the
  helper hard-codes an 8x6 grid, producing illegal `per_core_N=11,
  out_block_w=5` for hidden 2816.  AutoFix/AutoDebug is deriving the intended
  role-specific exact configs; see `AUTODEBUG_FUSED_RS.md`.
- Chosen profile/tensor/capacity plan is `mesh_plan.md`.  Final implementation
  work begins only after this recorded choice.

## 2026-09-05: inherited constructor policy AutoFix

- `AUTODEBUG.md` verified cross-stage constructor drift: the R0 multichip raw-
  weight loader inherited an R22-only row-major routing default and four enabled
  graph-fold flags from the optimized single-chip baseline.
- A host inspection probe resolved the effective pre-fix policy to row-major
  routing true, `residual_shard_cores=0`, and all four folds true.
- Added a multichip-local construction wrapper.  It explicitly disables the
  folds, supplies row-major routing false only when the caller left the env
  variable unset, restores the environment on exit, and preserves explicit
  caller values.  `optimized_decoder.py` remains unchanged.
- `py_compile`: passed for the implementation and test module.
- Focused host-only multichip tests: 7 passed, 31 deselected.
- `pre-commit` on the touched implementation/test and AutoDebug report: passed.
- The 1x4 constructor/PCC command was not run in this no-device AutoFix pass;
  hardware verification remains pending.

## 2026-09-05: phase-owned attention configuration AutoFix

- The unchanged hardware test advanced past construction for both layers, then
  failed at the first multichip prefill projection.  Evidence:
  `artifacts/optimized_pcc_v2.xml` (2 failures).
- A host AST/source probe found four stale `attention_compute_config` accesses:
  QKV and O in each of `_attention_prefill` and `_attention_decode`.  The base
  constructor now owns only `prefill_attention_compute_config` and
  `decode_attention_compute_config`.
- Replaced those four accesses with the matching phase-owned configuration.
  No baseline code or other multichip behavior changed.
- Post-fix AST probe and `py_compile`: passed.  Focused host-only multichip
  tests: 7 passed, 31 deselected.
- Hardware rerun remains pending to prove execution progresses beyond prefill
  QKV and to expose any next independent failure.

## 2026-09-05: paged cache API split AutoFix

- `artifacts/optimized_pcc_v3.xml` records the full-attention decode passing
  loose `block_size`/`num_kv_heads` to paged SDPA, whose binding accepts a
  grouped `paged_cache_geometry` instead.
- Verified the root patch against both current baseline call sites and the API:
  cache updates retain loose `block_size=128`, `num_kv_heads=1`, while SDPA gets
  `PagedCacheGeometryOverride(block_size=128, num_kv_heads=1)`.  Sliding
  attention gets neither override.
- Added a device-free helper/wiring regression.  The focused test passed;
  `py_compile` passed; the host-only multichip subset passed 8 tests with 31
  deselected.
- The patch is verified for the v3 full-attention TypeError.  The v3 sliding
  failure (`decode_dram_padded_input_widths` missing) is independent and remains
  for a separate hypothesis.

## 2026-09-05: inherited factory state and v7 numerical triage

- Audited `optimized_pcc_v4.xml` through `v7.xml`.  The v4/v5 packed-dense
  failures were missing padded-input and logical-output metadata maps.  V6 was
  the inherited packed-expert default trying to consume factory-only packed
  tensors that the raw TP loader never builds.  The current multichip setup
  initializes all reachable factory state and forces both packed-expert flags
  off.
- Added host regressions covering all six raw-weight policy flags, reachable
  factory-state assignments, and exact zero-only tail padding.
- V7 reaches correctness: both prefill checks pass; decode PCC is
  `0.9892006673654702` for sliding and `0.9949203490627379` for full.  The
  sliding value corrects the shorthand that both cases were approximately
  0.9949.
- Source comparison found the leading sliding mismatch: optimized defaults its
  sliding attention weights to BF16 and full attention to BFP8, while v7
  multichip used BFP8 for both.  The leading shared decode mismatch is the
  multichip-only independent BFP4 packed dense gate/up copy.
- Expert dtype control `optimized_pcc_v8.xml` was refuted: baseline-like
  BFP4 gate/up plus BFP8 down worsened decode to `0.9885124312118698` sliding
  and `0.9930889096865416` full, and was reverted.
- Combined HiFi4 (`optimized_pcc_hifi4.xml`) passed full but worsened sliding to
  `0.9848282846715828`, so a ranked one-variable matrix now separates attention,
  dense, expert gate, expert math, DRAM roles, packed storage, packing, and CCL.
- This investigator opened no TT device and ran no hardware command.

## 2026-09-05: generalized TP profile host coverage

- Replaced TP4-only test imports with values derived from `_profile_for_tp(4)`
  so existing hardware cases remain profile-backed.
- Parameterized the host shape contract over mandatory TP1/TP2/TP4 profiles,
  including exact padded/local dense and expert widths and local Q, sliding-KV,
  and full-KV head counts.
- Added unsupported-profile checks for TP0/3/8, frozen-profile and tile-alignment
  checks, source wiring checks, and the TP4 duplicated full-KV pair contract.
- Parameterized paged-cache full-KV geometry and packed gate/up rank pairing
  across all supported TP sizes.
- `py_compile` passed.  Focused host-only result: 14 passed, 41 deselected.  No
  TT device was opened.

## 2026-09-05: TP2 sliding decode wait AutoDebug

- Fresh live markers place the first missing decode exit in the dense-down
  all-reduce: attention, cache, SDPA, O projection, and the preceding attention
  reduction finish.  TP2 defaults to ordinary non-persistent Linear/one-link
  collectives, while a standalone call passes.  The leading hypothesis is
  therefore a back-to-back completion/resource-reuse hazard; see
  `AUTODEBUG_TP2_DECODE.md` for the persistent, serialized, and repeated-smoke
  experiments.
- Found a second deterministic TP2-only decode bug in host source geometry.
  Local expert width 352 is 11 tiles, so the old `expert_gate_per_core_n=2`
  default is illegal in the strict optimized decode builder.  TP1's 22 and
  TP4's 6 tiles are legal with 2; TP2 is legal with 1.
- Changed the default to derive 2 for even expert tile counts and 1 for odd
  counts, preserving explicit overrides.  Added a host regression over all
  three supported profiles.
- Paged-cache geometry, TP2 default DRAM roles, and dense matmul dimensions are
  source-consistent and are ruled out as the observed dense CCL wait.
- This investigator opened no device.

## 2026-09-05: selected profile correctness and context gates

- Selected TP1/TP2/TP4 profiles use the optimized decoder as their baseline,
  keep a replicated residual, and shard every eligible QKV/O, dense, and
  active-expert projection. TP4 load-time pads dense 2112 -> 2176 and expert
  704 -> 768. Full-attention TP4 duplicates its two KV heads by rank pair.
- Final direct optimized-baseline PCC artifacts are
  `artifacts/pcc_tp1_capacity_selected.xml`, `artifacts/pcc_tp2_final.xml`, and
  `artifacts/pcc_tp4_final.xml`. All 12 representative prefill/decode checks
  clear 0.995; exact PCC values are in `artifacts/pcc_tp{1,2,4}_layer{0,5}.json`.
- TP2 uses a 2x2 `FABRIC_2D` parent and a Linear/one-link 1x2 compute submesh.
  The final rotating persistent all-reduce policy and odd-expert geometry fix
  close the wait documented in `AUTODEBUG_TP2_DECODE.md`.
- TP4 direct HF-oracle comparison passed both representative layers:
  `artifacts/hf_correctness_tp4.xml`.
- Logical S=33 prefill plus decode trace replay passed both layer kinds with
  exact output replicas, exact repeated replay, replicated page table/current
  position, and local cache ownership:
  `artifacts/nonaligned_trace_tp4_final.xml`.
- Last-valid-position traced decode (`current_pos=262143`) passed both layer
  kinds with cache sentinels and exact repeated replay:
  `artifacts/advertised_context_trace_tp4_final.xml`.
- Real-weight physical prefill at logical length 262143 passed both layer kinds
  (`artifacts/prefill_capacity_262143_tp4.xml`), and the bounded sliding modulo
  tail gate passed at 1025 (`artifacts/bounded_modulo_tail_tp4.xml`). The
  advertised 262144 context is unchanged.

## 2026-09-05: capacity precision selection

- A 30-layer decoder-stage placement projection includes all rank-local and
  replicated weights, padding, packed dense storage, TP4 retained decode
  copies, full BF16 KV cache, and persistent CCL resources. It excludes the
  terminal embedding/LM head, traces, and transient/full-model allocations.
- The initial all-BFP8 TP1 projection left too little operating headroom. The
  narrow selected exception is BFP4 only for full-attention expert gate/up
  weights on TP1. Representative TP1 layer 5 passes prefill PCC 0.9970008655
  and decode PCC 0.9985550317; TP1 sliding remains BFP8 and passes
  0.9981487298/0.9951300642.
- Final decoder+KV+CCL projections per 32 GiB device are 29.444 GiB TP1,
  15.354 GiB TP2, and 10.095 GiB TP4. The ledger and scope are
  `capacity_projection.json`; a host test verifies every subtotal and headroom
  identity.

## 2026-09-05: performance and profiler decisions

- Captured exact optimized single-chip B1 and B32 baselines, then measured the
  TP4 policy using warmed trace replay. At the current recorded hashes, B32
  sliding is 8.8175026 ms versus 12.1991918 ms (1.3835x) and full is
  9.1740186 ms versus 12.1998448 ms (1.3298x). At B1 S=33, sliding is
  0.6521392 ms versus 0.7656910 ms (1.1741x), while full is 0.9524278 ms
  versus 0.8120687 ms (0.8526x). These are the post-HF-AutoFix final artifacts;
  no blanket speedup claim is made.
- Reduced B1 Tracy captures and advice-enabled `tt-perf-report` output are
  retained under `tt_perf_report/sliding_b1` and `tt_perf_report/full_b1`.
  Per-op times/configurations are valid; profiler readback gaps make aggregate
  percentage columns unsuitable for whole-layer attribution.
- The B32 profiler dropped a device marker during the four-device merge. Its
  JUnit and compact dropped-marker CSV are retained; the valid unprofiled B32
  trace remains the end-to-end result.
- Historical report-driven candidates were tested one at a time and rejected:
  QKV DRAM
  sharding 10.6055 ms (within noise plus extra retained storage), QKV
  `block_w=4` 10.6153 ms, and router-L1 staging 10.6345 ms. The final default
  for that experiment was restored in
  `artifacts/batch32_tp4_final_restored.xml`; the later optimized-contract
  AutoFix section supersedes its policy and timing.

## 2026-09-05: final host and watcher gates

- JSON parsing and `py_compile` passed for the implementation and test module.
- Focused profile/padding/cache/packing/phase/no-host-hot-path/capacity host
  gate: 22 passed, 39 deselected.
- Full watcher instrumentation cannot coexist with this checkout's active-ETH
  fabric firmware: it expands a 29,072-byte program beyond the 26,624-byte
  config buffer before model execution. The failure is preserved as
  `artifacts/watcher_active_eth_overflow.xml`.
- Following `$tt-device-usage`, reran independently from profiling with only
  active-ETH watcher instrumentation disabled:

  ```bash
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  GEMMA4_RANGE_DOWNLOAD=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'test_multichip_matches_optimized_single_chip or test_multichip_non_aligned_prefill_and_decode_trace' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/watcher_tp4_final.xml
  ```

  Result: 4 passed, 57 deselected in 74.83 s. Watcher reported no Tensix or
  dispatch error. `tt-smi -ls --local` listed all four P300C devices before and
  after the run. Active-ETH behavior is independently covered by the ordinary
  correctness, context, trace, and performance gates.
- No C++ or CMake file changed, so `AGENTS.md` does not require a build. Final
  pre-commit and independent stage review are recorded below when complete.

## 2026-09-05: full-stack capacity AutoFix

- Stage review invalidated the decoder-only capacity conclusion: TP1's
  2,744,465,408 B decoder/KV/CCL headroom is smaller than the 2,952,790,016 B
  required by separate BF16 embedding and LM-head tensors before final norm or
  runtime reserve.
- Following `$tt-device-usage`, `timeout 60 tt-smi -ls --local` found four
  healthy P300C devices. Two candidate runs were then serialized on one device
  and closed it cleanly after each run.
- Expert-down BFP4 was refuted in the actual TP1 multichip path: full-attention
  prefill/decode passes at 0.9962860301/0.9977604982, but sliding decode is
  0.9941133959. HiFi2 expert-down math is also refuted at 0.9941319551. JUnit
  evidence is `artifacts/pcc_tp1_capacity_down_bfp4.xml` and
  `artifacts/pcc_tp1_capacity_down_bfp4_hifi2.xml`. No candidate production
  change was kept.
- Retained the already-proven TP1 decoder policy and repaired the analytical
  full-stack contract instead. Downstream embedding and LM-head storage must be
  BFP8_B: physical 1,088-byte tile accounting gives 1,568,669,696 B on TP1,
  rather than the optimistic one-byte-per-element approximation.
- Added replicated BF16 final norm (180,224 B) and an exact 1 GiB per-device
  operating reserve. TP1's reserve is 64 MiB trace + 676,331,520 B for one
  `[30720,2816]` BF16 residual and one `[30720,8192]` BF16 QKV output +
  330,301,440 B allocator slack. TP2/TP4 use the same source-backed chunk and
  smaller TP-local QKV terms; all three reserve totals are exactly 1 GiB.
- Audited the selected persistent packed-expert decode copy with host-only
  physical-tile arithmetic. Its per-layer shape is
  `[1,128,2816,2*E_local]`: TP2 `E_local=352` uses 269,615,104 B/layer or
  8,088,453,120 B across 30 layers; TP4 `E_local=192` uses 147,062,784 B/layer
  or 4,411,883,520 B across 30 layers. These are separate from TP4's prior
  642,611,200 B of O/dense retained copies.
- TP1 packed expert is disabled by a hard capacity limit. Its all-BFP8
  candidate is 16,176,906,240 B, while even an all-BFP4 physical-tile lower
  bound is 8,564,244,480 B, larger than the 2,744,465,408 B decoder-stage
  headroom.
- With packed expert selected only for TP2/TP4, projected full-stack
  totals/headroom are 34,257,864,704/101,873,664 B TP1,
  26,433,070,080/7,926,668,288 B TP2, and
  16,717,143,040/17,642,595,328 B TP4 on a 32 GiB basis.
- This decoder stage records but does not implement the downstream BFP8_B
  terminal contract. Terminal/logit correctness and measured full-model peak
  allocation remain full-model gates.
- This packed-copy follow-up did not use hardware or modify production code;
  the capacity JSON and host regression derive every selected subtotal.

## 2026-09-05: selected profiler accounting audit

- After the HF-oracle repair, profiled the final TP4 S=1024 B1 sliding and full
  paths serially with watcher unset. The frozen SHA256 values were
  `34225bd3fa03639ecbc9311264e420d7a70c99772e6991f339cbe7107dca6f76`
  for `tt/multichip_decoder.py` and
  `8338455c27a473c506486a1f1c4c864699ebfc234c018e540e0606bdf90c1ba8`
  for `tests/test_multichip_decoder.py`; both hashes were unchanged before,
  between, and after the two runs. All four P300C devices were visible and the
  1x4 mesh opened and closed before profiling; all four were healthy after.
  No watcher ran concurrently.
- Exact sliding profile command (one passed; JUnit suite/testcase time
  19.433/18.656 s):

  ```bash
  env -u TT_METAL_WATCHER GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
    GEMMA4_RANGE_DOWNLOAD=1 \
    GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=2 \
    GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=3 timeout 900 \
    python_env/bin/python -m tracy -r -p \
    -o generated/profiler/gemma4_multichip_final_selected_sliding_b1 \
    -m pytest -q -s \
    'models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_perf_profile[blackhole-sliding_attention-batch1-mesh_device0-device_params0]' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/profiler_sliding_b1_final_selected.xml
  ```

- Exact full profile command (one passed; JUnit suite/testcase time
  17.244/16.485 s):

  ```bash
  env -u TT_METAL_WATCHER GEMMA4_FUNCTIONAL_DECODER_PERF=1 \
    GEMMA4_RANGE_DOWNLOAD=1 \
    GEMMA4_FUNCTIONAL_DECODER_TRACE_WARMUPS=2 \
    GEMMA4_FUNCTIONAL_DECODER_TRACE_ITERATIONS=3 timeout 900 \
    python_env/bin/python -m tracy -r -p \
    -o generated/profiler/gemma4_multichip_final_selected_full_b1 \
    -m pytest -q -s \
    'models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_perf_profile[blackhole-full_attention-batch1-mesh_device0-device_params0]' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/profiler_full_b1_final_selected.xml
  ```

- Regenerated the human-readable prefill/decode reports with
  `tt-perf-report --arch blackhole --active-experts 8` and derived exact
  device ledgers with the same signposts plus `--no-merge-devices
  --no-host-ops --no-summary --no-advice`. The retained raw captures have
  SHA256 `3854f4e6...` sliding and `8bd2592f...` full. There are exactly 258
  sliding and 261 full rows per device over three replays, or 86/87 ops per
  replay/device. The final JUnits are selected by `artifact_manifest.json` and
  were also copied to canonical
  `artifacts/profiler_sliding_b1.xml` and `artifacts/profiler_full_b1.xml`,
  whose post-normalization SHA256 values are `71921f30...` and `458fef00...`;
  all provisional and
  pre-HF profiler JUnits are superseded.
- TP4 prefill is 80.030248/88.843240 ms for sliding/full versus the optimized
  single-device 96.365912/107.636814 ms: 1.204119x/1.211536x, or
  30.1030%/30.2884% TP efficiency. This efficiency is speedup divided by four,
  not measured utilization; the baseline and TP4 runs have different checkout
  SHAs but the same TTNN extension SHA.
- Decode host e2e is 760.727/1077.322 us. Per-replay device-op sums span
  629.075-633.008/942.862-947.451 us, leaving 127.719 us (16.79%) and
  129.871 us (12.05%) against critical device 1. Its three all-reduces total
  56.238/57.739 us, 8.88%/6.09% of critical-device ops and 7.39%/5.36% of
  host e2e. `tt_perf_report/selected_accounting.csv` is the actual compact CSV
  derivation and also records overall modeled DRAM and representative roofline
  rows.
- The existing merged decode CSVs are useful for representative op rows but
  not per-device sums: the report tool keeps the maximum-duration device for
  ordinary ops and averages devices for collectives. Profiler readback inside
  the signposted loop makes `Total %` unsuitable for whole-layer attribution.
- Provenance limitations are explicit in `perf_summary.json`. The TP4 host JSONs
  record the delegated functional helper command rather than the outer
  multichip entrypoint proven by the profiler JUnit testcase names; the exact
  outer commands above are authoritative. Provisional and pre-HF captures are
  explicitly excluded. The
  profiler-instrumented B32 host JSON remains excluded because its four-device
  merge dropped an operation marker; verified unprofiled B32 JSONs are the
  throughput proxy at their recorded hashes.
- The selected sliding B1 O row uses the repaired `block_w=2` logical-B1
  program; the B32 path uses `block_w=4`. After profiling, Black removed only
  redundant parentheses around one assignment-call, changing the source hash
  to `4617f559feedd1dfef9633ea47063283b9dd0eed2319cfc831f254271817daef`.
  The AST and profiler behavior are unchanged. The formatted source passed
  `py_compile` and the final four-case watcher gate, so the capture is final
  selected-policy evidence compatible with `artifact_manifest.json`.

## 2026-09-05: optimized-contract AutoFix and final policy gates

- Host/source experiments verified that optimized folds commute with TP weight
  fracture only when applied first, R22 needs explicit DRAM projection
  boundaries, and expert gate/up packing must be rank local. Added regressions
  for those contracts, row-major environment restoration and R0 rejection,
  reachable batch-32 inherited attributes, and DRAM reader ownership/padding.
- Isolated TP4 direct candidates for router fold, shared FFN norm, expert-scale
  fold, fused final scalar, packed expert, R22, cumulative folds, and row-major
  routing all passed the 0.995 layer-kind gate. Candidate JUnit and suffixed PCC
  JSON files are retained as `artifacts/candidate_*.xml` and
  `artifacts/pcc_tp4_layer{0,5}_*.json`.
- The cumulative TP4 B32 full policy failed at 0.994389. Higher math fidelity
  and BF16 attention/expert/dense candidates were refuted. R0 raw passed at
  0.995053; R0 + packed expert + fused scalar passed at 0.995057; adding folded
  expert scale remained passing at 0.995025 and 9.173047 ms. Adding router fold
  (0.994890) or shared FFN norm (0.994940) failed. The maximal passing subset is
  the selected TP4/full exception.
- Selected defaults: TP1/TP2 and TP4 sliding use R22, all four graph folds, and
  row-major routing. TP4 full uses R0, raw router, separate FFN norms, folded
  expert scale, fused final scalar, and non-row-major routing. Packed expert
  decode is off on TP1 by capacity and on for TP2/TP4. TP4 decode DRAM roles are
  O, packed dense gate/up, and dense down with one reader per bank.
- Final direct command:

  ```bash
  GEMMA4_RANGE_DOWNLOAD=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'test_multichip_matches_optimized_single_chip or test_p150_proxy_matches_optimized_single_chip or test_p150x2_proxy_matches_optimized_single_chip' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/direct_pcc_selected.xml
  ```

  Six passed. `pcc_tp{1,2,4}_layer{0,5}.json` records TP1
  0.998077/0.999262 and 0.997276/0.998433, TP2 0.997943/0.999121 and
  0.998801/0.999741. The post-HF-fix TP4 rerun is
  0.997090/0.999319 and 0.998620/0.999875.
- Final TP4 nonaligned trace command used the same base environment with
  `-k test_multichip_non_aligned_prefill_and_decode_trace` and
  `--junitxml=.../artifacts/trace_selected.xml`. The post-HF-fix canonical
  rerun is `trace_selected_final.xml`: two passed at 0.652139 ms sliding and
  0.952428 ms full, with 30 bit-exact replays and exact replicas.
- Final B32 artifacts are `batch32_selected_final.xml` and
  `multichip_batch32_layer{0,5}.json`: sliding 0.995238/8.817503 ms versus
  12.199192 ms, full 0.995025/9.174019 ms versus 12.199845 ms. Both repeat bit
  exactly.
- Final TP2 chained stress is `stacked_tp2_selected.xml` and
  `stacked_tp2_mixed_trace.json`. Twenty replays and replicas are exact. Layer
  5 differs by one of eight routes after consuming layer 0, so the chained
  final PCC 0.983471 uses a dedicated 0.98 threshold while direct TP2 layer
  gates stay above 0.995.
- Workers 2 and 3 both reached the inherited worker-hop primitive's hard
  `mesh->num_devices() == 1` contract before timing; evidence is
  `trace_dram_workers2.xml` and `trace_dram_workers3.xml`. Default workers
  remain one and TP2/TP4 now reject larger values before device setup.
- Final watcher command:

  ```bash
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  GEMMA4_RANGE_DOWNLOAD=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'test_multichip_matches_optimized_single_chip or test_multichip_non_aligned_prefill_and_decode_trace' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/watcher_selected.xml
  ```

  Four passed without watcher error. All four P300C devices were listed healthy
  before and after the bounded run, and the serialized hardware slot was
  released. No further device commands were run during final host verification.

## 2026-09-05: unfused fractured-residual micro-chain

- Source audit found that direct `reduce_scatter_minimal_async` computes the
  correct `[1,1,32,704]` output and `Shard(dim=3)` topology from a
  `[1,1,32,2816]` row-matmul partial. The shared CCL helper's `use_non_fused`
  branch is stale and cannot be selected because it passes two removed buffer
  keywords instead of `persistent_output_buffers`.
- Corrected the consumer geometry: sliding QKV is 8192, while full QKV is
  logically 10240 and physically 12288 on TP4 due to pairwise KV duplication.
- Added an opt-in four-role chain covering row K widths 4096, 8192, 2176, and
  768, followed by RS, a fractured residual add, distributed RMSNorm, and fused
  packed-dense AGMM at N=4352. Both RS and final output are compared with Torch.
- Host gate: `py_compile` passed and focused collection found four new unfused
  cases plus five consumer cases.
- Exclusive hardware command:

  ```bash
  GEMMA4_MULTICHIP_UNFUSED_FRACTURED_REPRO=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  timeout 240 python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k test_tp4_unfused_row_reduce_scatter_distributed_consumer_exact_shape_repro \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/unfused_fractured_chain.xml
  ```

  Four passed. Every role cleared PCC 0.99 at the concatenated RS boundary and
  final concatenated AGMM output, with exact local shapes.
- Corrected attention-consumer command:

  ```bash
  GEMMA4_MULTICHIP_FUSED_AGMM_REPRO=1 \
  TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
  timeout 240 python_env/bin/python -m pytest -q -s \
    models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py \
    -k 'test_tp4_fused_all_gather_matmul_exact_residual_consumer_repro and (sliding_qkv or full_qkv_physical)' \
    --junitxml=models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/artifacts/fused_agmm_qkv_corrected.xml
  ```

  Both corrected cases passed. Device processes closed normally and the final
  health check listed all four P300Cs healthy/resettable before slot release.
- No warmed latency is claimed: the correctness repro intentionally uses
  primitive-allocated RS staging. A meaningful comparison must first add
  persistent/rotating buffers and include the replicated boundary and dynamic
  sparse-expert path.

## 2026-09-05: TP4 sliding HF-oracle AutoFix

- The selected TP4 sliding path missed the independent HF decode gate at
  0.9947442319 while prefill and the full-layer control passed. Isolated MoE,
  fold, fidelity, collective-link, and persistent-collective changes did not
  recover it. Exact candidates and JUnit/JSON paths are recorded in
  `AUTOFIX_HF_ORACLE.md` and `artifacts/hf_candidates/`.
- Omitting only the O DRAM-sharded role raised HF decode to 0.9995321511.
  Keeping that role and changing its block width from 4 to 2 produced the same
  passing result, localizing the miss to the sliding O program's accumulation
  grouping. Block 2 alone scored only 0.9949080351 at B32, so the repair keeps
  one BF16 sharded O weight and dispatches block 2 for logical B1, block 4 for
  logical multi-user decode. The known logical batch is used because padded
  physical row count is 32 for both modes.
- `hf_oracle_selected_final.xml`: two passed; sliding prefill/decode
  0.9984599504/0.9995321511, full 0.9985002091/0.9997050636.
- `direct_pcc_tp4_after_hf_fix.xml`: two passed; sliding
  0.9970896450/0.9993185418, full 0.9986196042/0.9998745871.
- `batch32_selected_final.xml`: two passed; sliding PCC 0.9952382122 at
  8.8175026 ms and full PCC 0.9950253440 at 9.1740186 ms.
- `trace_selected_final.xml`: two passed with bit-exact replay and replicas;
  sliding/full S=33 replay is 0.6521392/0.9524278 ms.
- Post-run `tt-smi -s` showed all four P300C devices with healthy DRAM, zero
  GDDR errors, and temperatures below 38 C. Decoder/test SHA256 is
  `34225bd3...`/`8338455c...`; the focused host suite passed 31 tests.

## 2026-09-05: final integrated acceptance

- The final profiler reran both selected TP4 S=1024 B1 cases at decoder/test
  SHA256 `34225bd3...`/`8338455c...`. The sliding/full profiler JUnits each
  passed. Final prefill latency is 80.030248/88.843240 ms and profiled decode
  latency is 760.727/1077.322 us. The derived 62-row CSV and exact per-device,
  collective, and roofline accounting are in `perf_summary.json` and
  `tt_perf_report/selected_accounting.csv`.
- The separate unprofiled B32 throughput gate decodes at position 32 with a
  256-token cache per user. Full attention prefills S=32; sliding attention
  starts from a zero-initialized cache. Earlier README/perf labels calling this
  S=1024 were incorrect and are superseded by the exact JSON metadata.
- The integrated direct and stacked command used the same no-fallback
  environment as the direct command above, selected all three profile gates
  plus `test_tp2_stacked_mixed_attention_shared_persistent_ccl_trace`, and
  wrote `artifacts/direct_and_stacked_selected_final.xml`: 7 passed.
- The final context/cache command set
  `GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1`,
  `GEMMA4_PREFILL_CAPACITY_LENGTH=262143`, `GEMMA4_RANGE_DOWNLOAD=1`, and
  `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`; selected
  advertised-context trace, prefill-capacity, and bounded-modulo-tail tests;
  and wrote `artifacts/context_cache_selected_final.xml`: 5 passed.
- Black removed only redundant parentheses around the batch-32 O-program
  assignment, changing the decoder SHA256 to final `4617f559...` without an
  AST or policy change. The formatted source passed `py_compile` and the final
  watcher command wrote `artifacts/watcher_selected_final.xml`: 4 passed.
- Final host verification on the formatted source passed 31 focused tests
  with 39 deselected. `artifact_manifest.json` parses every selected JUnit as
  failure/error-free, and distinguishes deliberately failing or superseded
  candidate artifacts from the authoritative final set.
- The final `tt-smi -s` health check listed all four P300C devices with healthy
  DRAM and zero corrected or uncorrected GDDR errors. The serialized device
  slot was released. No C++ or CMake file changed, so `AGENTS.md` does not
  require a build.

## 2026-09-05: stage-review P2 stacked-oracle repair

- Review found that the stacked test applied its `0.98` final-discontinuity
  allowance to layer 0, even though layer 0 has identical inputs and 8/8 route
  agreement. The layer-5 route change cannot explain layer-0 PCC 0.991223.
- The test now requires every identical-input layer-kind prefill/decode
  boundary to clear 0.99. Layer 5 is run independently with the same input on
  both paths and fresh cache state. Only the final layer-5 output from the
  intentionally divergent chain retains the separately labelled 0.98 gate.
- `tp2_hf_same_input_stacked_final.xml`: three passed with no runtime fallback.
  Identical-input Optimized/TP2 prefill is 0.997563/0.997879 and decode is
  0.991223/0.995297 for sliding/full. Chained prefill remains above 0.99;
  final eager/trace remains 0.983471 with one of eight layer-5 routes changed.
- Independent TP2/HF prefill is 0.999024/0.998534 and decode is
  0.999566/0.999711 for sliding/full, all above 0.995. Exact JSON is in
  `artifacts/tp2_hf_oracle/`; the strengthened stacked JSON explicitly records
  same-input and chained thresholds.
- The run used the unchanged decoder SHA256 `4617f559...`; test SHA256 was
  `374b7e03...`. Post-run health listed all four P300C devices with healthy
  DRAM, zero GDDR errors, and temperatures from 31.2 to 36.0 C.
- A later capacity repair is expected to remove TP2's retained packed-expert
  decode copy. The combined gate must be rerun after that source-policy change;
  this evidence is not a substitute for the final-policy run.

## 2026-09-05: conservative prefill-lifetime capacity repair

- Stage review invalidated the earlier 30,720-token chunk reserve.
  `_attention_prefill` materializes full-length QKV before chunked SDPA, the
  attention helpers retain output chunks through concat, and the inherited
  prefill block retains full residual/branch tensors through attention and
  FFN. No unimplemented deallocation is credited.
- At S=262144 the conservative source-live peaks are 22,817,013,760 B TP1
  (attention concat), 15,651,045,376 B TP2 (MoE), and 17,001,611,264 B TP4
  (MoE). The MoE bound includes both accumulated chunks and the new concat
  result. With the configured 64 MiB trace region and preserved allocator
  slack, packed expert must be disabled on TP2 as well as TP1. TP2/TP4 totals
  are 33,570,989,056/33,419,910,144 B with 788,749,312/939,828,224 B
  headroom on a 32 GiB basis.
- TP1 cannot retain the advertised prefill context under the
  correctness-proven precision policy: its S=262144 projection is
  56,398,546,944 B. The selected full-attention expert gate/up is already BFP4,
  while broader expert-down BFP4 candidates failed PCC. The exact profile
  limit is therefore 50,624 tokens and is enforced before allocation.
- The contiguous-bound formula block-rounds KV to 128 tokens, tile-rounds
  activation tensors to 32, and retains caller-owned unpadded hidden/cos/sin
  tensors for nonaligned inputs. S=50623 is the tightest accepted case at
  34,358,700,544 B (1,037,824 B headroom); aligned S=50624 leaves 389,822,464
  B; S=50625 is the first rejected case at 673,280 B over capacity.
- Focused host capacity/profile gates passed six tests. The dedicated TP1
  real-weight shape control passed both representative layer kinds at S=53343
  and S=53344 with finite last-token output and fallback throwing (four tests
  total). A second lifetime audit then found the missing MoE concat-output
  term, so those artifacts are superseded as boundary evidence. Corrected
  S=50623/S=50624 probes then passed both layer kinds with real weights, finite
  last-token output, and fallback throwing. Authoritative evidence is
  `artifacts/p150_prefill_capacity_{50623,50624}.xml` plus the four corrected
  length/layer JSONs. Post-run health showed four healthy P300Cs, zero GDDR
  errors, and a 44 C maximum. Fresh TP2 no-packed PCC, stacked, and performance
  evidence remains required before this repair is accepted.

## 2026-09-05: final capacity-policy acceptance

- Reran the final TP1/TP2 policy suite with fallback throwing. The command
  selected both direct optimized-baseline layer kinds for TP1/TP2, both TP2/HF
  layer kinds, the strengthened same-input/chained TP2 stack, and all four
  TP1/TP2 warmed trace timings. `artifacts/final_profiles_after_capacity.xml`
  records 11 passed. Final no-packed TP2 direct PCC is
  0.997943/0.999121 sliding and 0.998801/0.999741 full; TP2/HF remains
  0.999024/0.999566 and 0.998534/0.999711.
- The corrected P150 boundary is authoritative:
  `artifacts/p150_prefill_capacity_50623.xml` and
  `p150_prefill_capacity_50624.xml` each record two real-weight layer-kind
  passes with finite last-token output. The host guard rejects 50,625 before
  allocation. Passing 53,343/53,344 artifacts are retained but superseded as
  boundary evidence after the MoE concat lifetime correction.
- Reran the four mandatory TP1/TP2 S=33 B1 timings on frozen source SHA256
  `7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`
  and test SHA256
  `9125621e2e5a7e7fe72d73fc5fb6bb6498775c573b643e97425e1dcc9144178b`.
  `artifacts/warmed_required_profiles_final.xml` records four passes. TP1
  sliding/full is 0.929880/1.048610 ms (0.823/0.774x; 82.34/77.44%
  efficiency); TP2 is 0.816643/0.913994 ms (0.938/0.888x; 46.88/44.42%).
  All use five warmups and 30 trace replays and are bit-exact across replay and
  replicas. `perf_summary.json` records the exact command and artifact hashes.
- Final watcher coverage used `TT_METAL_WATCHER=10` with active-ETH watcher
  instrumentation disabled for the documented firmware-size limit, fallback
  throwing, and eight TP1/TP2/TP4 direct/nonaligned-trace cases.
  `artifacts/watcher_final_capacity_policy.xml` records eight passes with no
  Tensix or dispatch error. Its measured source differs from the final SHA only
  in the subsequent TP1 constant 53,344 -> 50,624 correction; the selected hot
  path is identical. The exact final source then passed both 50,623/50,624
  hardware probes and the four warmed trace runs. Post-run health showed all
  four P300Cs with healthy DRAM and zero corrected/uncorrected GDDR errors.

## 2026-09-05: final stage-review AutoFix

- The first independent `$stage-review` verdict was `more-work-needed`. It
  identified stale TP2 status/profiler hashes, missing TP4 mixed-layer stack
  evidence, missing real-weight BFP4 attention trials, and an incompletely
  decided fractured-residual candidate. Metadata and delegated-entrypoint
  provenance were reconciled before the new device work.
- TP4 stack AutoFix captured a separate one-chip optimized oracle, then ran the
  selected layer-0 sliding R22 path directly into the layer-5 full R0 path on
  the 1x4 target proxy with one shared three-slot CCL bundle. Same-input
  prefill PCC is 0.997792/0.998964 and decode PCC is 0.999117/0.995288.
  Chained prefill is 0.997792/0.991588 and chained eager/trace decode is
  0.999117/0.986078 at the documented 0.99/0.98 divergent-input gates. Twenty
  trace replays, replicas, persistent slot order/index, local KV shape and
  content, full-KV rank-pair duplication, page tables, current positions, and
  public replicated layouts all passed. Evidence is
  `AUTOFIX_TP4_STACK.md`, `stacked_tp4_reference_capture.xml`,
  `stacked_tp4_mixed_trace.xml`, and `stacked_tp4_mixed_trace.json`.
- BFP4 attention AutoFix ran four process-isolated real-weight TP4 modes at
  position 32 with BF16 paged cache, five warmups, 30 trace replays, and
  runtime dtype-observation assertions. QKV BFP4 scored 0.984498 sliding and
  0.978292 full; O BFP4 scored 0.991448 sliding and 0.998744 full. QKV and
  sliding O fail PCC 0.995. Full-only O clears PCC but is only 0.39% nominally
  faster and needs another retained tensor, so the selected policy is
  unchanged. `AUTOFIX_BFP4_ATTENTION.md` and
  `artifacts/bfp4_attention_summary.json` preserve commands and hashes.
- Fractured-residual AutoFix first proved the actual K-fractured dynamic
  indexed top-8 expert consumer. Its packed gate/up -> two intermediate RS ->
  GeGLU -> indexed down -> score/sum -> hidden RS -> distributed norm ->
  K-sharded next-QKV chain passed changed-route and 100-replay stress, watcher,
  and profiler gates. It measured 0.258290 ms versus 0.449673 ms for the
  incumbent micro-chain (1.74096x), with candidate/selected PCC 0.999979.
- The decisive Gate-D experiment integrated the same topology through one
  complete real-weight paged sliding layer. PCC versus the selected layer is
  0.999805 and 20 extra replays are bit-exact, but six reduce-scatters, six
  norm-stat all-gathers, and one router all-reduce cost a warning-clean median
  0.844279 ms versus 0.652086 ms for the incumbent: 0.772358x, or 29.474%
  slower. Both paths used separate trace lifetimes, five warmups, and 30
  individually timed blocking replays; every readback occurred after release,
  and p95 was 0.847750 versus 0.657987 ms. The earlier v5 aggregate is
  superseded because it retained active traces during readback. All temporary
  product source was removed. The selected source returned byte-for-byte to
  SHA256 `7279e13a379bd9260006f2ab220a1fa7b35d01ccec1ccc79dddd17bab058f3bf`.
  A final restored-source S=33 run measured 0.651610 ms over 30 replays and
  passed fallback, cache, control-replication, replay, and replica checks.
  The decision, human/CSV profiler tables, exact commands, transient candidate
  hashes, and focused setup failures are under
  `AUTOFIX_FRACTURED_RESIDUAL.md` and `artifacts/fractured_sparse/`.
- Every hardware command was serialized. The final bounded device list showed
  UMD IDs 0--3, and a bounded 1x4 open/close smoke printed `MESH_SMOKE_OK`.
  No hardware process remains. The only reset in this repair loop followed a
  deliberately rejected watcher AGMM boundary that left device 0 firmware
  stuck; targeted UMD 0--3 reset and mesh smoke recovered the board before any
  further evidence was collected.
- The rejected Gate-C and Gate-D product sources are reproducible from
  `artifacts/fractured_sparse/rejected_fractured_gate_{c,c_d}_candidate.patch`.
  Each patch applies independently to selected source SHA `7279e13a...`,
  recreates its recorded temporary source hash, passes reverse-check, and
  restores the selected source byte-for-byte. The patches must not be stacked.
- Final host-only regression selected 32 profile, capacity, padding, cache,
  packing, phase, fallback, and active-expert contract cases: 32 passed and 58
  hardware cases were deselected.

## 2026-09-05: independent final review

- A fresh xhigh `$stage-review` independently checked the frozen final tree,
  recomputed Gate-D v6 statistics from all 30 raw samples, parsed all 26
  selected JUnits, resolved all 43 selected machine-data entries, verified the
  62-row accounting ledger and immutable hashes, applied both rejected-source
  patches in memory, and inspected the complete profile/capacity/cache/stack/
  watcher evidence.
- Verdict: **clean-pass**. No required work or hard-check gap remains. Generic
  motherboard, L1 semaphore-placement, and fabric packet-size advisories are
  present in both Gate-D paths, but the v5 active-trace allocation/corruption
  warning is absent from selected v6 and v5 is explicitly superseded.
- Per `AGENTS.md`, no C++/CMake build is required because this stage changes
  Python, tests, documentation, and evidence only. The final repository checks
  are pre-commit, `py_compile`, the 32-case host suite, JSON/XML/provenance
  validation, candidate-patch applicability, warning scan, and
  `git diff --check` as recorded above.
