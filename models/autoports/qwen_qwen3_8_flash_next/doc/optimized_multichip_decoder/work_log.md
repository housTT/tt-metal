# Optimized multichip decoder work log

## Scope and provenance

- Model: `Qwen/Qwen3.8-Flash-Next`, checkpoint revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Target: P300 Blackhole dies 0 and 1, fixed `1x2` `FABRIC_1D` mesh.
- Starting implementation: `82af47d73235744357fd4ac62d33ef97eb04dd56`;
  stage-start HEAD: `7ec7d69fcd4f13506e36cfeefdf3f50d8426a38e`.
- Scope: the completed multichip decoder only. No full-model or vLLM work was
  started.
- Selected mode: exact host-backed routed-expert EP2 and exact host-backed PLE
  rows. Router, active experts, shared expert, projections, attention, state,
  residual, and collectives execute on TT. Routed MoE remains gate-selected
  top-10; dense all-expert execution is not present.
- Active skills: `$optimize`, `$host-weight-cache`, and `$tt-device-usage`.
  `$autofix` was used for the fused matmul-reduce-scatter hang, and
  `$stage-review` is the independent final gate.

All hardware commands below used:

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ttenv.sh
export TT_VISIBLE_DEVICES=0,1
export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_mesh_graph_descriptor.textproto
```

TT jobs were serialized. `tt-smi -ls --local` listed four local P300c devices,
and source-backed `1x2` mesh open/close checks passed before the first test and
after watcher stress. Watcher and profiler were never enabled together.
The final `tt-smi -s` health snapshot at 20:18 local time reported all four
P300c devices, live heartbeats, zero corrected/uncorrected GDDR errors, and
37.7–39.3 C ASIC temperatures.
Pre-existing untracked artifacts under `doc/multichip_decoder/` were preserved
and are not part of this stage.

## Baseline

The inherited host path used ordered slot replacement and only 16 packed host
experts. The fresh baseline command was:

```bash
QWEN38_MC_PERF_DECODE_REPLAYS=100 timeout 7200 pytest -q -s \
  --tt-arch blackhole --capture=tee-sys -o junit_logging=all --count=7 \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/baseline_perf_count7.xml
```

Result: 21 passed. Medians are in `baseline_perf_medians.csv`.

| Layer kind | Prefill PCC | Decode PCC | Warmed prefill ms | Segmented decode ms |
| --- | ---: | ---: | ---: | ---: |
| GDN layer 0 | 0.99942303 | 0.99996978 | 2065.630386 | 3.427797 |
| PLE+GDN layer 1 | 0.99949104 | 0.99984211 | 1526.015021 | 4.118693 |
| QSA layer 3 | 0.99972457 | 0.99988294 | 1969.037359 | 3.059096 |

`baseline_real_pcc.xml` reconfirmed every meaningful layer kind with real
weights/activations. `baseline_static_contracts.xml` and
`baseline_host_service_windows.xml` captured the starting context, fallback,
and host-accounting behavior. The initial counters were cumulative and could
not support hit-rate claims; the harness was changed to report measured-window
deltas.

## Host-weight path work

The selected host path now has a generation-checked LRU, stable slot indices,
a persistent replicated `uint16` slot-index row, ten device expert slots, a
512-expert packed host cache per layer, and one shared packed exact-zero peer
shard. An exact cache miss still loads the checkpoint row, packs the owner
shard, and uploads it; no approximate or stale-hit behavior is allowed.

PLE retains safetensors mmap handles and exact EOS/request history semantics.
It deduplicates requested rows with `torch.unique`, uses per-shard batched
`index_select`, caches 8,192 exact rows, uploads in 128-row padded chunks, and
reuses fixed TT staging. Logical non-aligned lengths are padded, masked, and
sliced internally.

Targeted CPU evidence was collected with:

```bash
pytest -q models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py \
  -k 'ple_row_ids or ple_chunk_history or ple_non_aligned or ple_duplicate or contract_json' \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/host_ple_exactness.xml
```

Result: 5 passed. A separate 129-token repeated-lookup sweep retained exact
outputs and wrote `ple_row_cache_sweep.csv`: capacity 0 reread 2,064 rows,
capacity 256 reread 1,808 rows, and capacity 8,192 reread zero rows while
reducing warm lookup from 2.197265 to 0.541153 ms. `host_pinning_probe.txt`
records that this CPU-only Torch build raises `Need to provide pin_memory
allocator to use pin memory`; TT staging is therefore the declared boundary.

Measured candidate processes used the baseline command with `--count=1`, a
single layer selected by `QWEN38_MC_PERF_LAYERS`, and one candidate override.
The full values and XML mapping are in `optimization_matrix.csv`:

- Stable indexed slots, packed capacity 512, and the shared zero shard won.
- Device slot capacity 20 was adapted to fit the available DRAM and rerun. It
  improved prefill to 153.253806 ms but slowed decode to 2.351580 ms.
- Capacity 12 slowed decode and changed the accepted layer-1 PCC.
- Threaded rank uploads retained PCC but did not beat serial fenced uploads;
  overlap remains truthfully reported as zero.
- The final layer-1 prefill window has 12 expert waves, 114 misses, 114 packed
  hits, 630,374,400 physical H2D bytes in 98.814318 ms (6.379 GB/s), 528
  selected/528 unique PLE rows, no table reads, 337,920 logical PLE bytes in a
  1,310,720-byte padded upload, 0.371790 ms lookup, and 0.227715 ms PLE H2D.
  Prefill `stall_ms=0` is the non-negative accounting remainder when recorded
  operations fill the service window, not an independent stall timer.
- Its 100-token decode window has 976 hits/24 misses, 23 packed hits/one exact
  checkpoint miss, 9,830,400 checkpoint bytes, 132,710,400 expert H2D bytes at
  6.393 GB/s, 1,440 slot-index bytes, 1,600 selected/unique PLE rows, no table
  reads, 1,024,000 PLE H2D bytes, 3.362758 ms lookup, 8.634629 ms PLE H2D,
  27.768330 ms unattributed stall, and 261.315892 ms end to end. Median
  checkpoint read/packing is 1.723209/8.035640 ms.

## Device optimization experiments

Every candidate was run in a new process against the same real-checkpoint
layer harness. The environment-to-candidate mapping is encoded in
`test_multichip_decoder_perf.py`; the retained XML is the command provenance.
The common form was:

```bash
QWEN38_MC_PERF_LAYERS=<layer> QWEN38_MC_PERF_DECODE_REPLAYS=<replays> \
QWEN38_MC_<CANDIDATE_OVERRIDE>=<value> timeout 2400 pytest -q -s \
  --tt-arch blackhole --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode \
  --junitxml=<candidate-artifact.xml>
```

The coherent optimization families and decisions were:

- Residual layout: the selected inter-layer ABI is mesh-sharded BF16
  `[1,1,4*M,1280]`, shard dimension 3, local DRAM interleaved. A real hyper
  consumer measured fractured 0.233722 ms versus replicated 0.221804 ms.
  Replication was rejected because it reinstates an all-gather/reshape at
  every layer boundary. No gather, reshard, or all-reduce remains between
  decoder layers. Evidence: `residual_topology_current.xml`.
- Collective placement: one/two links, 4,352/8,192-byte payloads, and their
  combination were measured for layers 0, 1, and 3. The mixed one-shot result
  was not dismissed: the combination was rerun seven times for every layer
  kind in `candidate_links2_packet8192_perf_count7.xml`. Its medians were
  157.797183/2.042372, 121.089902/2.626367, and 162.836228/3.025393 ms, so two
  links plus 8,192 bytes was promoted. `final_default_perf_count7.xml` then
  reproduced the no-override default, and `final_fabric_contract.xml` read the
  live payload as 8,192 bytes and asserted two links.
- Async CCL: the model shapes were ported to preallocated semaphores, then
  adapted and retuned rather than rejected at an API error. The best adapted
  run was 161.128189 ms prefill and 4.824999 ms decode for layer 0, slower than
  the 157.879783/4.783126 ms synchronous control. Evidence: all five
  `candidate_async_ccl_*layer0.xml` files and the byte-for-byte transcript
  recovery in `candidate_async_ccl_recovered_provenance.txt`.
- Activation layout: QSA decode 1D width sharding
  `qsa_input:110,attn_out:20` won its synchronous A/B and is selected.
  The retained stdout reruns measured 162.378467/3.037342 ms for 1D versus
  162.525198/3.046772 ms without it; prefill 2D was 163.180329/3.042779 ms and
  slower. Evidence: `candidate_qsa_decode_1d.xml`,
  `candidate_qsa_decode_1d_sync_control.xml`, and
  `candidate_qsa_prefill_2d.xml`.
- DRAM-sharded decode matmuls: QSA input/output were adapted at HiFi2 and
  retried at HiFi4. Decode PCC was 0.99907207 and 0.99908733 respectively;
  retained reruns measured 162.125484/3.021874 ms and
  165.421701/3.097239 ms; HiFi4 was also slower. Evidence:
  `candidate_qsa_dram_sharded.xml` and
  `candidate_qsa_dram_sharded_hifi4.xml`.
- Precision: BF16 row-parallel partials remain selected. FP32 payload was
  rerun at 158.402200/4.780256, 124.299125/2.911613, and
  163.584140/3.029378 ms; it was slower and changed layer-3 decode PCC. GDN
  BFP8, QSA BFP8, QSA input BFP4,
  attention-output BFP4, and shared-projection BFP4 were tried on real
  activations; all changed the accepted baseline or failed the 0.995 gate.
  Evidence: the five `candidate_*bfp*xml` files covering eight measured
  precision-candidate rows, plus
  `candidate_row_parallel_fp32_ccl.xml`; exact QSA BFP8 stdout and transcript
  record hashes are in `candidate_qsa_bfp8_recovered_provenance.txt`.
- Packed/fused projections: selected weights remain packed for GDN QKV, QSA
  Q/K/V/index, hyper down+injection, router+shared input, shared gate+up, and
  active-expert gate+up. Profiler evidence shows these packed projections in
  the final graph; no unpacked split removed enough movement to win.
- Persistent buffers: the selected path retains expert slots, fixed upload
  staging, route-index row, PLE staging, KV/index/recurrent state, shared GDN
  workspace, and two fixed-address decode traces.

### Fused matmul-reduce-scatter AutoFix

The initial stock and adapted fused MMRS programs hung. Triage was captured in
`fused_mmrs_hang_triage.txt` and `fused_mmrs_adapted_hang_triage.txt`; both had
completed matmul kernels and ring readers waiting on remote-intermediate
semaphores. `$autofix` identified and tested source-level defects rather than
stopping at the TTNN failure:

1. Linear decoder fabric was dispatched through the Ring reduce-scatter
   builder.
2. The non-square result spec was derived from input-A K rather than matmul N.
3. Linear needs a doubled persistent intermediate, while the semantic output
   remains N/TP.
4. The fused factory did not forward the one-worker setting.
5. The producer needed an atomic flush after the operation signaler.

Temporary core adaptations implemented all five points and rebuilt `_ttnn.so`.
Stock 8x6/offset `(0,6)` was used first, then the 8x1/offset `(0,1)` model
shape. Decode produced PCC 1.00000012 but measured 1.241605 ms fused versus
0.715216 ms synchronous (1.736x slower). M=32 produced PCC 1.00000119 but
measured 1.175380 versus 0.715399 ms (1.643x slower). Fabric ERISC teardown
also remained unhealthy. Evidence is in the three
`candidate_fused_mmrs_autofix_*xml` files and four retained watcher logs. All
temporary core and qwen36-helper edits were then reverted and `_ttnn.so` was
rebuilt; the final source contains no rejected fused experiment.

## Final default and correctness

The final performance command was the baseline command with output changed to
`final_default_perf_count7.xml` and no candidate override. The selected two
links/8,192-byte payload came from source defaults. Result: 42 passed, 42
skipped in 316.094 seconds, including 21 host-backed measured windows. Final
medians and the exact comparison are in `final_default_perf_medians.csv` and
`before_after.csv`.

| Layer kind | Prefill PCC | Decode PCC | Prefill before / after ms | Decode before / after ms |
| --- | ---: | ---: | ---: | ---: |
| GDN layer 0 | 0.99942303 | 0.99996978 | 2065.630386 / 158.877355 | 3.427797 / 2.082220 |
| PLE+GDN layer 1 | 0.99949104 | 0.99984211 | 1526.015021 / 120.853601 | 4.118693 / 2.613159 |
| QSA layer 3 | 0.99972457 | 0.99988294 | 1969.037359 / 162.370719 | 3.059096 / 3.027236 |

PCC is unchanged from the accepted baseline for every meaningful layer kind.
The final values reproduce the final default, not an earlier candidate.

The static/fallback gate was:

```bash
pytest -q \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_host_weight_cache.py \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_class_and_memory_contract \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_rank_local_config_contract \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_rank_local_checkpoint_shapes_without_allocating_weights \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_runtime_collective_has_no_host_fallback \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_runtime_boundary_whitelist \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_multichip_prefill_plan_preserves_non_aligned_contract \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/final_static_and_fallback.xml
```

Result: 32 passed. It audits the host runtime whitelist, absence of host
fallback collectives, cache arithmetic, exact packing/miss behavior, PLE
semantics, and non-aligned logical plans from 1 through 262,144.

`doc/context_contract.json` retains a physically backed 262,144-token limit,
BFP8 paged QSA KV/index cache, and 24,122,217,472 planned TT DRAM headroom per
device. The QSA sharding is temporary intra-layer state. The packed host cache
maximum is 1,418,342,400 bytes per layer and 68,080,435,200 bytes for the
48-layer maximum, declared in `doc/host_weight_contract.json`; it does not
consume KV capacity. There is no context reduction.

## Profiling

Fresh final-default captures were made independently for layers 0, 1, and 3:

```bash
QWEN38_MC_PROFILE_DIRECT_DECODE=1 QWEN38_MC_PROFILE_HOST_ONLY=1 \
QWEN38_MC_PERF_DECODE_REPLAYS=1 QWEN38_MC_PERF_LAYERS=<0|1|3> \
timeout 2400 python -m tracy -p -r --op-support-count=2000 \
  --dump-device-data-mid-run --check-exit-code -o <tracy-output> -m pytest -q \
  --tt-arch blackhole \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder_perf.py::test_host_backed_warmed_prefill_and_segmented_decode
```

All three tests passed with no full profiler buffer, unmatched operation, or
missing device data. For each layer and phase:

```bash
tt-perf-report <raw.csv> \
  --start-signpost MC_HOST_<PREFILL|DECODE>_L<layer> \
  --end-signpost MC_HOST_<PREFILL|DECODE>_L<layer>_END \
  --no-color --no-host-ops --active-experts 10 \
  --csv <phase>_report.csv --summary-file <phase>_summary.csv
```

The same six reports were generated with `--no-summary` to retain complete
human tables and actionable advice. All twelve commands passed. Raw hashes,
commands, detailed CSVs, summary CSVs/PNGs, and human tables are under
`profiler_provenance.txt` and `tracy_final/`. `profiler_family_summary.csv`
reports dominant coherent families: final decode layout/TM is 28.88–60.78%,
dense matmul 12.37–25.61%, and CCL 4.80–8.07%; prefill layout/TM is
27.70–47.09% and active sparse matmul 11.57–20.46%. Advice to try DRAM sharding and lower
movement is resolved by the adapted experiments above.
`profiler_operation_audit.csv` separately records repeated dense/active-sparse
matmul, all-gather, reduce-scatter, broadcast, copy, layout conversion, and
mesh-partition counts for each phase so family percentages do not hide op
multiplicity.

After the capture logs were checked, redundant `.logs`, device-timeline CSVs,
and host Tracy intermediates were moved to the desktop trash. Raw TT op
reports were retained losslessly as `.csv.xz`; `profiler_provenance.txt`
records both content and packed hashes plus the unpack command. This reduced
the retained profiler package from roughly 11 GiB to 4.4 MiB without removing any
`tt-perf-report` input/output used by the claims.

## Watcher stress and final health

The final selected default was stressed with:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
QWEN38_MC_TRACE_STRESS_STEPS=100 timeout 2400 pytest -q -s \
  --tt-arch blackhole --capture=tee-sys -o junit_logging=all \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_gdn_segmented_trace_progression \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_segmented_trace_replay_matches_direct_qsa_decode \
  models/autoports/qwen_qwen3_8_flash_next/tests/test_multichip_decoder.py::test_host_backed_shared_workspace_segmented_trace_stack \
  --junitxml=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/final_watcher_stress100.xml \
  2>&1 | tee models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/final_watcher_stress100.log
```

Result: four passes in 109.233 seconds: 100 changing tokens for GDN layer 0,
PLE+GDN layer 1, and paged QSA layer 3, plus two live segmented GDN traces
sharing the workspace. The 2,027-line log is retained as
`final_watcher_stress100.log.gz`; its uncompressed-content SHA-256 is
`6a5a24028026cf0c98b24b3cae6c669f62222544d57fe8989b67f97870658c4d`;
the packed-file SHA-256 is
`d3f796ba898c1a39eae22fecddb6c85ae5ec325f3bc1a9ce1eba895e8d2ce45b`.
It has no watcher, NoC, assert, panic, hang, or fatal signature. The XML
SHA-256 is
`2e42c9e17b40698859a5d4ba261ae421f83974f72cc5bba01607a679f76d208b`.
`TT_METAL_WATCHER_DISABLE_ETH=1` is the
inherited P300 instrumentation control; Tensix, NoC, CB, and stack watching
remain active. A post-stress source-backed mesh open/close passed.

## Limitations, review, and commits

- Exact host-backed decode is batch one; the resident decoder retains its
  previously validated batch-32 contract.
- Expert/PLE host service is serialized. The parallel transfer candidate did
  not win, so measured overlap is zero rather than inferred.
- Torch pinned allocation is unavailable in the installed CPU-only build.
- Intra-layer collectives and stack ingress/exit remain; no collective or
  layout conversion remains between decoder layers.
- Full-model assembly and vLLM remain deliberately out of scope.

Independent review result: `clean-pass` in `STAGE_REVIEW.md`.

Stage-owned payload commit SHA: `5b8898664b3bb09c3db14ec80999c52859967a9d`.
No push was performed.
