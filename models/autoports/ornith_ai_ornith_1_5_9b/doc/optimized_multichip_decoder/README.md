# Optimized multichip decoder

Stage 05 for `ornith-ai/Ornith-1.5-9B`, pinned revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`. The target is a native 1x4 tensor-parallel ring on **four Blackhole chips on two physical P300c boards**. `p150x4` is a software profile name. This stage optimizes the completed decoder in place; it does not implement a full model or vLLM adapter.

**Status: complete; independent review clean-pass.** The 101-case suite, explicit async watcher checks, native capacity, direct composition, final timings and all four final profiles pass. [Independent review](STAGE_REVIEW.md) reports no required work. Local checkpoint SHAs are recorded in the [work log](work_log.md#local-checkpoints).

## Measured before/after

All latency rows refer to actual TP4 execution at batch 1 and 2048-token prefill/decode position 2048. Prefill uses synchronized warmed forwards; decode uses warmed nonblocking trace replay with synchronization outside each 32-replay group. State restoration and numerical comparison are outside timing. Each final timing uses the same probe as its baseline, with 16 prefill iterations and 5 groups of32 traced decode replays.

| Layer kind | Before prefill ms | After prefill ms | Before decode ms | After decode ms | Before → after prefill / decode PCC |
| --- | ---: | ---: | ---: | ---: | --- |
| Linear attention, layer0 | 3.463762 | 3.443891 | 0.371151 | 0.355672 | 0.999963081 → 0.999963081 / 0.999987989 → 0.999990048 |
| Full attention, layer3 | 2.858084 | 2.690952 | 0.283669 | 0.268965 | 0.999977247 → 0.999977247 / 0.999695988 → 0.999697387 |

Final default decode latency falls4.17% and5.18%. Linear prefill’s0.57% delta is within observed synchronized-host timing variation; its earlier candidate medians around3.33ms are not substituted for the final3.443891ms. Full-attention prefill medians differ5.85%, but its prefill math/topology is unchanged; this measurement does not establish an algorithmic prefill gain. Traced decode is the optimization target and the reproducible improvement claim. [Measurements JSON](measurements.json) links the exact final default records.

PCC in the timing probe compares with the optimized single-chip numerical control. That control supplies no stage latency claims. The real-weight HF contract suite separately enforces its unchanged 0.995 output threshold; state/cache comparison remains 0.99.

## Selected default path

The topology audit preceded local tuning. Packed GDN combines QKV/A/B/Z from raw, rank-partitioned HF weights, preserves tile-aligned A/B fields, and fuses the Z activation into the gated multiply through the verified compact BF16-left/FP32-output adaptation. Decode MLP gate/up are packed into one projection. A precision-locked22-case geometry sweep and matched repeated whole-layer timings select32 input cores, K-block4 and3 DRAM readers after counting output slices, fused SiLU/multiply and movement. Separate prefill projections share the original policy. Decode packing replaces both separate decode copies without increasing stored weight bytes.

| Decode projection | Weight/fidelity | Program and weight layout |
| --- | --- | --- |
| GDN QKV/A/B/Z | BFP4 / LoFi | Shared DRAM-interleaved weight;8x4 multicast compute; K block 8 |
| GDN output and attention output | BFP4 / LoFi | Shared DRAM-interleaved weight;8x4 multicast compute; K block 8 |
| MLP gate/up | BFP4 / LoFi | One packed DRAM-width-sharded decode allocation; 32 input cores, K block4,3 readers |
| MLP down | BFP4 / LoFi | DRAM-width-sharded decode copy; 8 input cores, K block 6,2 readers |
| Attention QKVG | BFP8 / LoFi | DRAM-width-sharded decode copy; 32 input cores, K block 4, 1 reader |

Prefill retains DRAM-interleaved BFP4 projections and the large 11x10/block 16 program. Norms/residuals stay BF16, recurrence stays FP32, and KV stays BFP8 with 64-token pages and one local 256-wide KV head. Decode SDPA retains grid 8x8/chunk 256. The native reader implementation now chooses bank/worker coordinates for each physical mesh device; the generic reader-count default remains one.

[Optimization evidence](optimization_evidence.md) records paired coherent-family tables, communication bytes, reader geometry, dtype/fidelity trials, adapted rejections and provenance. [Candidate CSV](candidate_measurements.csv), [JSON](candidate_measurements.json) and [table](candidate_measurements.md) include failed and ineligible runs. The current packed-path32-case topology matrix and19 QKV8 controls pass. Earlier38-case topology and19-case fidelity/async/SDPA matrices remain explicitly labeled historical QKV4-era evidence. All34 final packed fidelity, fused-norm, packet-size and prefill-input advice trials also pass; none displaces the default. Native replicated reductions win the complete decoder comparisons. Lower-movement1024-wide residual families are measured through compatible residual/norm/MLP consumers, with host gathering only after timing.

## Interlayer residual contract

Weights and heads are tensor-parallel; the selected residual is replicated BF16 hidden 4096 on every rank. Prefill returns tiled DRAM-interleaved `[B,T,4096]`. Batch 1 decode returns L1 width-sharded `[1,1,4096]`, using 32 row-major 8x4 cores with shard `[32,128]`. Batches 2–32 return DRAM-interleaved `[B,1,4096]` after the decoder-owned folded residual addition.

Feed each returned tensor directly into the next decoder. **No gather, all-reduce or reshard belongs between layers.** The two row-projection reductions are inside each layer and are counted in all timings. Layer-local caches/recurrent state remain private. See the exact interface and alternative-family contracts in [the evidence document](optimization_evidence.md#interlayer-residual-contract).

## Validation and capacity

The earlier QKV4 candidate smoke passed 6 tests: real-HF batch 4/batch 32 checks for both kinds, plus fixed-shape prefill trace replay with refreshed input, exact all-rank output/state and immutable input checks. `final_default_v1` passes native capacity and the native-cache oracle but fails a batch32 full-attention traced HF output (PCC0.99427076 below0.995). Completed AutoFix isolates gate-projection precision loss, reproduces the request at batch1, and verifies identical eager/trace/cache behavior. Higher fidelity and alternate geometry still fail. A real QKV4/gate8 split passes but loses across14 configurations. Packed QKV8 is restored for all batches; The subsequently promoted packed-MLP/QKV8 default passes10 real-HF batch/trace smoke tests including that original failing selector; **Final default validation now passes101 tests**. Both async-CCL watcher probes pass. Direct three-decoder composition at logical prefill131 and eight traced decode steps uses zero boundary conversions, with minimum decode PCC0.9999204. The native cache oracle reaches position262143 at HF PCC0.9983894 with exact eager/trace results.

The [context contract](../context_contract.json) preserves native 262144, arbitrary valid logical sequence lengths, and batches 1–32 at shorter per-user contexts. The decoder owns alignment, page padding, masking, continuation and output slicing. The [updated capacity plan](memory_capacity_plan.json) accounts for 1,744,175,104 projection bytes/device and 13,595,475,968 conservative estimated total bytes/device. Its native test reserves 7,153,385,472 DRAM bytes/device plus 24 recurrent-state allocations before executing a native-context decoder. Reservations are capacity evidence, not full-model execution.

The passing final validation includes non-aligned/native prefill and decode, cache continuation and permuted/changed page tables, poisoned free pools, repeated restored traces, ragged batches and prime batch boundaries, no-host-fallback guards with positive controls, direct decoder composition, and the native cache oracle. Worker watcher runs separately from profiling, including explicit async-CCL checks. Full Ethernet watcher instrumentation retains the [prior measured kernel-buffer/teardown limitation](../multichip_decoder/README.md#investigations-and-limits); `TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1` scopes the clean claim to worker instrumentation.

## Profiling, repairs and rejected options

Advice-enabled before reports: [linear decode](tracy/linear_attention/before/decode_device0_report.txt), [linear prefill](tracy/linear_attention/before/prefill_device0_report.txt), [full decode](tracy/full_attention/before/decode_device0_report.txt), [full prefill](tracy/full_attention/before/prefill_device0_report.txt). Every directory includes all four device tables, CSVs, compressed operations and window accounting. Final default reports: [linear decode](tracy/linear_attention/after/decode_device0_report.txt), [linear prefill](tracy/linear_attention/after/prefill_device0_report.txt), [full decode](tracy/full_attention/after/decode_device0_report.txt), [full prefill](tracy/full_attention/after/prefill_device0_report.txt). [Paired performance accounting](performance_accounting.md) and its [JSON](performance_accounting.json)/[CSV](performance_accounting.csv) reconcile all eight profiles and32 rank rows, including actual dtype/fidelity/programs, both norms, SDPA, movement and collective metadata. Device clocks are independent; chip durations are never summed as layer latency, and only the first gap beginning outside the signpost is removed.

The final same-profile decode accounting is:

| Kind | Modeled weight/KV floor µs | Max independent rank span µs | Same-run host µs |
| --- | ---: | ---: | ---: |
| Linear attention | 60.192 | 368.621 | 385.308 |
| Full attention | 70.050 | 303.530 | 314.762 |

The floor includes BFP headers and assumes512GB/s per chip; it is not measured bus utilization and omits recurrent-state/activation/CCL traffic. Profiles use four replays with instrumentation; headline timings use32 replays per window without profiling. Linear/full device operation counts fall62→59 and47→46. All interior gaps remain counted (about80/49µs); the extra host time outside rank spans is16.7/11.2µs per profiled replay. Shared/packed projections, phase-specific shards, fused collectives and persistent buffers were compared as complete paths to reduce these costs. The retained small-op/collective costs and the rejected alternatives are quantified in the linked evidence. Full-attention prefill has unchanged38-op topology and nearly unchanged profiled device span, supporting the limited prefill claim above.

- [Reader AutoFix](AUTOFIX_reader_mesh.md): mesh-coordinate-aware 1/2/3-reader paths built and tested; padded output storage and per-role geometry adapted.
- [CCL AutoFix](AUTOFIX_ag_trace.md): full 11x10 semaphore coverage; gathered/fused/persistent/async families retried after repair.
- [Z-fusion AutoFix](AUTOFIX_z_fusion.md): correct compact operand adaptation at B1/B4/B32, exact targeted B32/T2048 user31 HF control, and unchanged-source controls for a historical non-reproducing anomaly. No unsupported causal attribution.
- [QKV trace AutoFix](AUTOFIX_qkv_trace.md): real batch32 and extracted batch1 precision failure; matched state/eager/trace controls, fidelity/geometry adaptations, gate localization and14 true split-projection timing trials justify retaining packed decode QKV8.
- [Cache AutoFix](AUTOFIX_cache_precision.md): five matched TP4 diagnostics prove exact cache addressing/update and independent SDPA consumption, while K4 and V4 fail unchanged state PCC; B32 KV4 also fails two HF users. Retain K8/V8.
- [SDPA AutoFix](AUTOFIX_sdpa_chunk1024.md): buffer-lifetime/DRAM-query and reduction-scratch adaptations make chunk 1024 pass, but it is slower than selected 256. Oversized packed-prefill K64/K128 also pass after output-block adaptation and lose on measured latency.

The [warning ledger](warning_ledger.md) classifies active-trace allocation warnings, topology metadata fallback, BFP8 packet advice and the interrupted runner case with controls and follow-up evidence.

## Reproduction and provenance

Run from the repository root with `TORCHINDUCTOR_CACHE_DIR=/home/hous/dev/ornith-1.5-9b/state/torch-cache` and `OMP_NUM_THREADS=8`. Hardware jobs must remain serialized.

```bash
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder
python "$D/run_candidates.py" "$D/packed_default_families_resumed.json"
python "$D/run_candidates.py" "$D/validated_full_attention_families.json"
python "$D/run_candidates.py" "$D/packed_final_precision.json"
python "$D/run_validation.py" NEW_LABEL
python "$D/run_profiles.py" NEW_LABEL
```

Candidate matrices skip previously successful immutable records. For a new measurement, copy the matrix with unused run names. Validation/profile labels must also be new. Historical policy reproduction requires the archived source and recorded native binary; a new run of current defaults is a different measurement. `record_run.py NAME timeout SECONDS python -m ...` stores exact command/environment, timestamps, log hashes, archived Python/context sources, native-source hashes and loaded-library hashes. [Work log](work_log.md) records the decisions and local commit SHAs; [stage contract](stage_contract.md) is the acceptance authority.

The required `.github/scripts/copilot-build.sh --build-ttnn-tests` could not run because this build container has no Docker binary. The installed-toolchain fallback **`timeout 1200 cmake --build build_Release --target ttnncpp ttnn -j 2` passed 14 targets**, followed by successful `tar` and `tt_pybinds` install components. No compiler or dependency was installed. See [build provenance](logs/reader_native_build.provenance.json). Python/native formatting and repository pre-commit checks pass after default selection. [Artifact integrity](artifact_integrity.json) verifies466 completed records,933 archives and36,591 archived source entries, plus10 final runs against current runtime source/binary hashes; one explicitly recorded interruption is excluded from passing results. The [independent review](STAGE_REVIEW.md) returns `clean-pass`.

Raw multi-GB Tracy captures and tensor dumps stay local and excluded from commits. Compact advice tables, CSVs, compressed operation/log archives and provenance are retained. Native-context tests do not claim a complete native-prefix HF rollout; the last-position oracle uses an explicitly constructed paged history. Optional million-token YaRN, full-model construction and serving remain separate pipeline stages.
