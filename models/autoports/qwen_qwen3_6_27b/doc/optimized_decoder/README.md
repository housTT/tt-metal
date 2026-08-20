# Qwen3.6-27B optimized decoder

This stage delivers the single-device `tt/optimized_decoder.py` path. It preserves the fused decoder API, prefill/decode semantics, paged state ownership, deterministic trace replay, non-aligned logical lengths, batch-32 coverage, and the 262,144-token context contract. The tests substitute `OptimizedDecoder` into the real-weight functional harness and assert optimization counters, so a functional fallback cannot satisfy them.

## Selected policy

| Group | Linear attention | Full attention | Selected runtime policy |
|---|---|---|---|
| Projections | packed BFP8 input; BFP8 DRAM-sharded output | packed BFP8 DRAM-sharded input/output | TTNN automatic fidelity; explicit LoFi/HiFi2 lose on the final topology |
| MLP prefill | BFP4/BFP4/BFP4 | BFP4/BFP4/BFP8 | real-weight-selected precision; separate gate/up |
| MLP decode | BFP4/BFP4/BFP4 | BFP8/BFP8/BFP8 | 32-core DRAM-width-sharded weights, physical row height 32 |
| GDN | FP32 recurrent state | n/a | explicit reuse configs for all eight recurrence and ten inverse matmuls; `reuse96m` decode update; tuned row/state reads |
| Cache | n/a | BFP8 paged K/V | device-side fill casts, TILE/DRAM, unchanged capacity |
| Residual path | BF16 | BF16 | decode entry converts once to 32-core width-sharded L1; input/post-attention RMSNorm and MLP remain sharded; final add emits public DRAM output |

Full decode mixed BFP4/BFP4/BFP8 crossed with the selected sharding and failed real-weight trace PCC at 0.994284, so BFP8 decode copies remain necessary. BFP4 full projections fail badly (0.975903). Synthetic/random PCC did not veto any real-weight win.

## Operation-topology audit and dispositions

| Region | Measured topology / concern | Candidate and action | Evidence |
|---|---|---|---|
| Same-input projections | packed full Q/K/V/gate and packed GDN inputs | retained packing; separate paths add ops; full input+output and linear output DRAM sharding selected | final full 31-op prefill / 47-op decode; linear padded-input candidate is legal but loses 1561.389 vs 1555.930 us |
| GDN chunk prefill | eight recurrence plus ten inherited inverse FP32 matmuls dominated linear prefill | copied the fused recurrence exactly and supplied legal full-output reuse configs to all 18 matmuls | real-weight PCC passes; inverse auto/reuse host 6643.707 -> 5604.958 us; final kernels 4394.522 -> 2751.686 us |
| GDN decode | generic recurrent update and row/state reads | selected `reuse96m` update plus inherited tuned read configs | actual wired matrix: auto about 1556, reuse48 1501.806, reuse96m 1497.917 us; reuse96n invalid because N=4 is not covered by per-core N=2 |
| Dedicated GDN kernel | issue #50475 requests optimized GDN/gated-attention kernels | installed `gated_delta_attn_seq` exercised beyond its first API error; not retained | kernel row 141.937 us, but adapter conversions/inverse/layout expanded the path to 344 ops / 6482.619 us |
| Decode MLP | interleaved weights and repeated movement | selected 32-core DRAM-sharded matmuls; tested BFP4/LoFi across material geometries | 8-core block-1 is legal but 1749.951 us; 16-core block-1 is legal but 1689.438 us; padded 64-core block-1 and adapted 3/3/9 variants hit `bad optional access`; 32-core wins |
| Projection fidelity | final rows were previously mislabeled | crossed LoFi, HiFi2, and auto on both final sharded layer kinds | linear 1557.698/1597.812/1555.930; full 1214.390/1341.780/1213.799 us; auto selected |
| Decode input norm | one-core DRAM RMSNorm cost about 85.2 us on both kinds | selected 32-core width-sharded residual and RMSNorm; projection and residual tail reuse that layout | conversion 1.4 us plus RMSNorm 6.7 us; linear/full decode kernels 1218.077/1067.191 us |
| Full Q/K layout | head creation and decode RoPE/cache require height sharding, while normalization caused a round trip | actual native height-sharded RMSNorm candidate reached the installed op and failed | TTNN assertion: `Height sharded inputs are not supported`; conversions retained as exact installed-layout boundary |
| Full attention | cache fill/update, SDPA/paged SDPA, fused gate | retained optimized TTNN composites and BFP8 cache | full prefill/decode kernels 1196.264/1067.191 us |
| Large prefill | generic projection/MLP rows carried 2D/DRAM advice | wired a phase-scoped 2D config and crossed block widths; retained TTNN auto because every legal candidate lost | linear seq128 auto 10619.324 vs 2D 11848.602-12502.804 us; full chunked auto 9363.200 vs 2D 12276.011 us |
| Movement | report contains layout conversions | audited every reported family | decode boundary joins, head/cache formats, GDN convolution/state formats, and public-output layout are required; no host fallback or torch/from/to-torch exists in measured methods |

The 64-core MLP was not rejected on its original tail error: weights and activations were padded to 6144/18432 and both block-1 and adapted role-specific blocks were retried. The installed DRAM-sharded program still rejects that padded geometry internally. Likewise, the 40-core padded linear input was repaired through slice and interleaved-consumer errors until it ran, then rejected on performance.

There are no collectives or MoE experts in this dense, single-device decoder. Multichip, full-model, LM-head, sampling, and vLLM work are outside this stage.

## Correctness and contract

The final acceptance threshold is PCC >= 0.995. Exact-source results are in `final_selected_v5/`.

| Gate | Result |
|---|---:|
| Linear non-aligned prefill / first decode / traced decode (65) | 0.997581 / 0.997539 / 0.998381 |
| Linear trace determinism | 1.000000 |
| Full non-aligned prefill/decode (33), BFP8 cache | passed |
| Full paged traced decode | passed, deterministic |
| Full forced chunk, 257 logical / 384 physical | 0.996460 |
| Linear/full batch 32 | passed; all per-user PCC above 0.995 |
| Full native-position decode/replay | 0.999708 / 1.000000 |
| Linear native 262,144 prefill/decode | finite, traced, passed |
| Full 32,769 and 262,144 prefill | finite, passed |
| Entire optimized non-perf suite | 11 passed, 6 deselected, 306.16 s |
| Watcher suite | 6 passed, 11 deselected, 144.19 s; 1,107-line log clean |

The BFP8 cache reduces bytes and does not reduce capacity, so `doc/context_contract.json` continues to advertise 262,144 tokens. Padding/chunking remains internal; there is no public sequence-alignment restriction.

## Performance

Device numbers are sums of `DEVICE KERNEL DURATION` inside warmed signposts. Decode is trace replay; full decode is divided across ten replays and linear decode uses one replay because the legacy profiler buffer overflows at ten. Fused controls are the best correct traced fused-stage baseline.

| Layer | Phase | Fused kernels | Final kernels | Improvement | Final ops |
|---|---|---:|---:|---:|---:|
| Linear | prefill | 5605.562 us | 2751.686 us | 50.91% | 159 |
| Linear | traced decode | 2728.252 us | 1218.077 us | 55.35% | 73 |
| Full | prefill | 2362.104 us | 1196.264 us | 49.36% | 31 |
| Full | traced decode | 2334.152 us | 1067.191 us | 54.28% | 47/replay |

Five-sample unprofiled host medians compare the unchanged fused controls to final v5:

| Layer | Phase | Fused host | Final host | Improvement |
|---|---|---:|---:|---:|
| Linear | prefill | 7319.387 us | 5629.998 us | 23.08% |
| Linear | traced decode | 2911.526 us | 1456.961 us | 49.96% |
| Full | prefill | 2461.871 us | 1622.408 us | 34.10% |
| Full | traced decode | 2301.998 us | 1135.458 us | 50.68% |

The representative seq-128 linear prefill is 10885.080 us under Tracy with 6628.735 us of kernels / 307 ops. The warmed 257-logical/384-physical full chunked prefill is 9556.945 us under Tracy with 5339.789 us of kernels / 112 ops. Advice-enabled `tt-perf-report` CSVs, human-readable tables, summaries, and plots exist for all four canonical phases and both large-prefill paths under `final_selected_v5/profiler/`. Advice to DRAM-shard decode MLP/projections was implemented. The installed DRAM-sharded programs require the decode physical-row contract and cannot replace height-64/128 interleaved prefill rows. A phase-scoped 2D replacement was implemented beyond its first errors and lost on both layer kinds, so TTNN auto is retained. HiFi advice was measured and rejected. The report tool still infers Wormhole because its raw CSV omits `DEVICE ARCH`; the explicit Blackhole floor in `final_selected_v5/blackhole_required_byte_floor.csv` is authoritative.

Reported conversions are required by installed composite bindings. `nlp_create_qkv_heads_decode` first emits HEIGHT_SHARDED Q/K; native RMSNorm explicitly rejects that layout, so Q/K must become interleaved for normalization/partial RoPE and return to height sharding for paged cache/SDPA. GDN convolution and recurrent state use fixed formats, and the public result returns to DRAM. The decode residual now converts once at entry and remains width-sharded through both norms and MLP. The dedicated GDN adapter proved that explicit preparation, tilize/untilize, padding, inverse, and transpose traffic raises the path to 344 ops / 6482.619 us despite a 141.937-us kernel. Candidate-only padded input conversion is not present in the selected path. No measured runtime method contains torch, `from_torch`, `to_torch`, host fallback, tilize/untilize calls, or an explicit reshard call.

## Optimize checklist

- [x] Optimized path is directly exercised; PCC, paged cache, trace determinism, non-aligned lengths, batch 32, native context, repeated runs, and Watcher pass.
- [x] Warmed prefill and traced decode before/after tables exist; final source beats the correct fused baseline and all materially different accepted candidates.
- [x] Topology audit covers repeated same-input matmuls, packing, composites, precision/fidelity, sharding, programs, cache policy, and runtime movement.
- [x] Real-weight BFP4/LoFi MLP trials cover role policies and 8/16/32/64-core geometries; synthetic PCC did not veto real-weight results.
- [x] Projection input/output/both, padded legal linear input, fidelity, GDN prefill/decode configs, cache dtype, SDPA, and large prefill were measured beyond first errors.
- [x] Advice-enabled raw and filtered profiler artifacts exist for every required phase; accepted and rejected advice has before/after evidence.
- [x] No applicable MoE/CCL opportunity exists; issue #50475's unavailable optimized integrated kernels are recorded as the remaining upstream limitation.
- [x] Profiler and Watcher ran separately on the isolated board; postflight was healthy and no reset was needed.

Exact commands, artifact paths, candidate failures/repairs, and hardware discipline are in `work_log.md`.
