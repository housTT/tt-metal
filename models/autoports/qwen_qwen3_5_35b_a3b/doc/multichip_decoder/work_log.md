# Stage 4 work log: multichip decoder

Times are Eastern (UTC-4). Evidence roots:
`/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/` (exploration, tensors) and
`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b/doc/multichip_decoder/evidence4/`
(final frozen chain). `evidence1/` to `evidence3/` are superseded chains, kept.

## 2026 Oct 11

### Mesh setup (`runs/main/stage04/micro/ccl_x{2,4}.json`, logs)

- Host: four Blackhole chips on two P300c cards; control plane sees one 2x2 mesh.
- `MeshShape(1,4)` with `FABRIC_1D_RING` opens as device IDs [3,2,1,0].
- Bare `MeshShape(1,2)` fails: fabric router handshake timeout on device 2
  (`ccl_x2_attempt1_fabric_timeout.log`, `ccl_x2_attempt2_fabric_timeout.log`; a reset did not help).
  Routers toward the unopened card wait for peers.
- Fix: `TT_VISIBLE_DEVICES=0,1` or `2,3` gives a 1x2 mesh (`fabric_probe.py`).
- `TT_VISIBLE_DEVICES=<one id>` fails (CUSTOM cluster needs a mesh graph descriptor,
  `moe_split_attempt1_visible_devices.log`); p150 uses `physical_device_ids=[3]`.
- `ttnn.GetPCIeDeviceID(d) == d`: logical ID equals PCIe ID.

All-reduce of a decode row [1,32,2048] BF16, traced, per op:

| Mesh | Topology | Links | all_reduce | reduce_scatter + all_gather |
|---|---|---:|---:|---:|
| 1x4 | Ring | 1 | 20.7 us | 19.0 us |
| 1x4 | Ring | 2 | 16.8 us | 18.4 us |
| 1x4 | Linear | 2 | 26.2 us | 26.2 us |
| 1x2 | Linear | 2 | 17.5 us | 17.6 us |

4096x2048 BF16 eager: 1x4 Ring 2 links 402 us; 1x2 Linear 2 links 482 us.
Selected: `ttnn.all_reduce`, Ring on 1x4, Linear on 1x2, 2 links.

### MoE split (`micro/moe_split.json`, `micro/moe_ep_decode.json`, one chip, synthetic weights)

| Phase | Option | x1 | x2 | x4 |
|---|---|---:|---:|---:|
| Decode, indexed sparse (8 experts) | expert-TP (intermediate / tp) | 91.4 us | 87.2 us | 69.2 us |
| Decode, masked sparse over local experts | expert-parallel | n/a | n/a | 195-248 us (0-8 active) |
| Prefill 4096, `moe_fused_swiglu` | expert-TP | 7161 us | 5736 us | 4859 us |
| Prefill 4096, `moe_fused_swiglu` | expert-parallel | 7161 us | 3476 us | 1752 us |

- Indexed `sparse_matmul` has no skip: every listed id is computed. Expert-parallel
  decode needs the masked form, which expands to all local slots.
- Masked output zero-fills skipped slots (verified: no nonzero, no non-finite).
- Selected: decode expert-TP, prefill expert-parallel.

### Bring-up failures and fixes (logs under `runs/main/stage04/bringup/`)

| Attempt | Failure | Cause | Fix |
|---|---|---|---|
| x2 layer 3 #1 | `NotImplementedError` one device | harness used FunctionalDecoder | harness class selection |
| x2 layer 3 #2 | reshape size error in `_Projection` | `ConcatMeshToTensor(dim=0)` readback doubles shape | read device-0 replica |
| x2 layer 3 #3 | `AttributeError` in harness report | wrapper attribute `module` shadowed | rename to `reduced_module` |
| x2 layer 0 #1 | `to_torch` on mesh tensor in qwen36 GDN loader | discarded qwen36 GDN built on mesh | layer shell without qwen36 GDN |
| x2 both | decode PCC 0.985-0.99, prefill fine | fused reduce `cluster_axis=0` zeroes device-1 scores | `cluster_axis=1` on 1xN |
| x2 memory | 1580 MiB/device (expected ~790) | loop variable kept last replica | build replicas in a helper |

Eager decode reproduced the decode error, so it was not a trace issue
(`x2_layer0_eager.json`). Token 128 (eager decode path inside prefill) was correct because
its route happened to avoid the error pattern; the fused reduce source
(`deepseek_moe_fast_reduce_nc_fused_reader.cpp:181-203`) shows the on-axis rule.

### Base-code generalization (behavior preserving)

- Edits: readbacks use device-0 replica; `_FusedShared` packing, indexed RoPE column map,
  sparse expert width and GDN a/b column offsets derive from shapes; fused-reduce axis attribute;
  `replicated_mesh` flag; dispatch `restore_table` hook; `dispatch_weights` policy field.
- Single-chip 129/4 outputs bitwise identical before and after (`runs/main/stage04/control_pre_edit/`,
  `regression_post_edit/`).

### Expert weight representation (`micro/stacked_moe.json`, `op_tests/`)

- Stage 3 single chip held two routed-expert copies: decode stacked 560 MiB plus dispatch
  lists 853 MiB per layer. Forty layers need about 60 GiB; one chip has 31.8 GiB.
- `moe_fused_swiglu` extended: stacked (1,E,K,N) weights, gate/up column offsets, a separate
  down dtype. Kernel parameters are defines; list-mode kernels compile unchanged.
- Results vs BFP8 lists: bitwise identical in every case; faster (4991 -> 4304 us at 512 tokens,
  7036 -> 6884 us at 4096 tokens, 256 experts).
- Existing op tests: 281 of 282 pass. The failure
  (`test_hybrid_routed_expert_config_fits_default_ring`) also fails on the unmodified source
  (`op_tests/hybrid_ring_baseline.log`).
- p150 dispatch now reads the decode weights: outputs bitwise identical to Stage 3;
  DRAM per layer 777/719 MiB (was 1593/1517 MiB).

### Profiling on a mesh (`explore_profile_x4/`, `explore_profile_x4_nopush/`)

- First 1x4 profile: tracy ops CSV kept only 7056 construction ops per device and no signposts.
- C++ device report was complete (21794 ops per device, 130 replay sessions).
- Cause: mid-run dumps also push four chips of device data into the tracy capture.
- Fix: tracy flag `--disable-device-data-push-to-tracy`; then 87600 ops and 260 signposts.
- Raw `profile_log_device.csv` is about 13 GB per run and is deleted after reports.

### Evidence chain 1 (`runs/main/stage04/evidence_chain1.sh`, 02:55-04:03 UTC)

- Source manifest identical at start and end (`evidence1/source_manifest_{start,end}.json`): `SOURCE_STABLE`.
- Headline, lengths, batch 32, reuse, positions, context, device pytest, placement, profiles: see README.
- Failures preserved as run:
  - `accept_watcher_m4`, `watcher_long/`: ERISC fabric router watcher build 29264 B > 26624 B ETH buffer.
    Rerun with `TT_METAL_WATCHER_NOINLINE=1` under the same source manifest (`watcher_m4_retry/source_manifest.json`):
    all clean.
  - `real_inputs_m{4,2,1}/positions_0.json`: row 17 token 4 below 0.995 on every profile and on the Stage 3 single chip.
- The chain copied both baseline `default.json` files into one directory (overwrite). Fixed after the chain by
  copying them with distinct names (`headline_base_logs/`); the source tensors and commands are in
  `runs/main/stage04/evidence1_tensors/`.

### After the chain

- Residual-contract micro benchmark (`micro/residual_contract_bench.py`): sharded residual slower on x2 and x4.
- Router margin (`router_margin_layer0.json`): tokens 950, 637, 1720, 3407 tie exactly at the 8th/9th BF16 router
  logit; 913 of 4096 tokens tie. Explains which tokens fall below 0.995 vs HF and why the set shifts with tiny numeric
  changes.
- Inventory per profile (`inventory/`, `tests/weight_inventory.py --decoder multichip --mesh N`) and capacity model
  (`capacity_model.json`): constants that repeat per layer (RoPE, dispatch maps, zero templates) must be allocated once
  per mesh in the full model.

### Review 1 response (`stage_review.md`, verdict more-work-needed, 3 P2 findings)

- Nondeterminism: per-op capture probe (`evidence2/nondeterminism/probe.py`), B32 full layer, 4 runs per profile.
  - Baseline and p150: bitwise repeatable. p150x2 and p150x4: `moe_fused_swiglu` and `post_combine_reduce`
    outputs drift on one device at a time with identical inputs (op-output max abs 0.0003-0.009; small
    relative to the values, but not bounded to 1-2 ULP).
  - BFP8 lists (no stacked extension) drift the same way, so the extension is not the cause.
  - Standalone op loops (single chip, x4 mesh with fabric, with interleaved all-reduces) are repeatable.
  - AutoDebug (`evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`): no source defect; experiments E0-E6 listed.
  - Classified inherited, controlled for Stage 4, open for Stage 5.
- B32: p150 row added; README reports B32 latencies and the batch-dimension plan.
- Memory handoff recheck: `tests/weight_inventory.py` records allocator and host RSS samples during construction;
  `evidence2/memory_handoff_recheck.json`; replica peaks `evidence2/inventory/replica_peaks/`.
- `functional_gdn.py` NameError in the error path fixed.
- In-tree op test added: `test_moe_fused_swiglu_stacked_mixed_weights` (3 pass).
- Decode `sparse_matmul` gate and down swept independently (`evidence2/micro/sparse_geometry_sweep.json`):
  x4 down 16 -> 64 cores (25.9 -> 15.8 us), x2 gate k 16 -> 64 and down k 4 -> 8.

### Evidence chain 2 (`runs/main/stage04/evidence_chain2.sh`)

- Start and end source manifests equal. Results: README.
- Incident: the replica-peak copy step overwrote `inventory/m{2,4}_layer*.json`. Rerun under the same source manifest
  (`inventory/rerun_source_manifest.json`, `*.rerun.log`).

### Decode projection geometry (`evidence4/micro/projection_geometry_sweep.json`)

- Per-device shapes, selected dtype and fidelity, single chip, traced.
- Selected: x4 qkvg 1D 40 cores k 8; x4 GDN in-projection 1D 20 cores k 2; x2 qkvg 1D 16 cores k 8;
  shared gate/up decode minimal-matmul grids; shared down 1D multicast.
- Layer screen 1 (`micro/projection_screen/`): decode gains as predicted.

### Evidence chain 3 (`runs/main/stage04/evidence_chain3.sh`, superseded, kept)

- `SOURCE_STABLE`. Decode gains reproduced (x4 full 347.8 us, linear 497.8 us).
- Prefill regressed 0.3-1.2 ms on meshes: `shared_minimal_config` also drives the
  prefill shared-expert matmul, so the small decode grid ran at 4096 rows.
- Fix: new `shared_decode_minimal_config`, used only for one-row decode. Stage 3 trial
  matrices keep the old field meaning.
- Layer screen 2 (`micro/projection_screen2/`): prefill back to the chain 2 level; decode unchanged.
- Rerun as chain 4 (`evidence4/`).

### Evidence chain 4 (`runs/main/stage04/evidence_chain4.sh`, final)

- `SOURCE_STABLE`. Every step exits 0 except the 3 expected `real_positions_0_m*` (row 17 token 4).
- Results: README. x4 decode 346.5 / 498.3 us; prefill at the chain 2 level.
- Post-chain processing: `process_mesh_profiles4.sh 4` and `2` (run separately; the `&&` chain stopped after
  the version command failed), `gen_tables.py`, `capacity_model.py`, `memory_handoff_recheck.py`,
  `construction_peaks.py`, `update_context_contract.py evidence4`.
- `tt_perf_report_version.txt` rewritten from the installed dist-info (1.4.1); `--version` is not a valid flag.
- Baseline log overwrite repeated in the chain copy step (one directory, same file names). Fixed after the chain by
  copying both baseline runs from `evidence4_tensors/` with `full_`/`linear_` prefixes.

### Review 2 (`stage_review2.md`, more-work-needed, 1 P2)

- P2: nondeterminism bound and cause overstated. Range across chains, mesh-only rate, and AutoDebug E0/E2 now
  recorded (README "Nondeterminism", `evidence4/nondeterminism/e0e2/`).
- Construction-peak fields renamed and annotated in `memory_handoff_recheck.json` (`field_notes`).
- Row counts, GiB units, B32 union (163), profiler overhead note, placement device note fixed in README.

### Review 3 (`stage_review3.md`, more-work-needed, 1 P2)

- P2: `prefill_repeat_identical` compares the first (compiling) prefill with the warmed prefill, not warm with
  warm; the E2 target drifted only in the cold run. Fixed:
  - Metric renamed in README.
  - E0 warm probe (`evidence4/nondeterminism/e0e2/x4_full_b32_e0e2_warm.json`, reference warm run 1, 7 runs):
    every warm run differs from warm run 1; MoE inputs identical.
  - E2a/E2b on a warm-run alternate (call 10, devices 1 and 2): 600 replays, one value, never the alternate.
  - "Carried state" inference removed; mechanism unknown; owner Stage 5.
- Other concerns recorded as open Stage 5 anomalies: mesh prefill timing variance across chains, GDN layout ops.

### Review 4 (`stage_review4.md`, more-work-needed, 1 P2): mesh prefill at 2176 tokens

- Classification (`evidence5/micro/padding_routes/x4_layer{3,0}_2176.json`, warm runs, 6 per mode):
  - The 2176-row MoE group pads to 4096. Zero rows tie all router logits, so all 1920 padding rows pick the
    same 8 experts. Rank 0 gets 20130 (full) / 19594 (linear) routes; other ranks 3790-4932.
  - Masking padding routes to the dummy slot (`valid_rows`) brings rank 0 to 4770 / 4234 routes, but prefill time
    stays about 20.1 ms. The imbalance is real but not the cost.
  - Cause: EP prefill cost follows the padded group size (4096 rows), not the valid routes.
  - Single timed chain rows (21.6-24.4 ms) are high samples; warm medians are about 20 ms.
- Fix: `prefill_moe_pieces` policy. A padded group runs MoE in exact bucket pieces (at most 2), e.g.
  2176 = 2048 + 128. Padding routes are still masked.
- Measured (`evidence5/micro/padding_routes/pieces_*.json`, medians of 6 warm runs; limit 1 = pad, 2 = pieces):

| Profile | Layer | 1152 | 2176 | 2560 | 3072 |
|---|---|---|---|---|---|
| p150x4 | full | 12621 -> 11138 | 21267 -> 14731 | 20311 -> 16133 | 20601 -> 18448 |
| p150x4 | linear | 12785 -> 11485 | 20556 -> 15754 | 21432 -> 18036 | 20868 -> 19114 |
| p150x2 | full | 13017 -> 12763 | 20510 -> 17024 | 20850 -> 18466 | 21436 -> 20851 |
| p150x2 | linear | 13686 -> 13773 | 23119 -> 19690 | 24034 -> 21555 | 24262 -> 23917 |
| p150 | full | 14777 -> 16909 | 22883 -> 21717 | 23641 -> 23882 | 24528 -> 26861 |
| p150 | linear | 17093 -> 19376 | 27046 -> 26240 | 28296 -> 28925 | 29509 -> 31838 |

- Selected: 2 pieces on p150x2 and p150x4; p150 keeps padding (pieces slower at 3 of 4 lengths).
- PCC of pieces against padding: >= 0.99998 (expected: different MoE grouping and bucket configs).
- Evidence chain 5 (`evidence5/`) adds length rows 3000 and 3968 for the split path.
- Chain 5 (`runs/main/stage04/evidence_chain5.sh`): `SOURCE_STABLE`; every step exits 0 except the 3 expected
  `real_positions_0_m*`. p150x4 2176 prefill 17.5 / 17.6 ms (single rows).
- Repeated prefill timing (`evidence5/micro/prefill_timing/`, median of 6 warm runs, sources equal to chain 5):
  latency reference for all lengths and profiles.

### Review 5 (`stage_review5.md`, more-work-needed, 2 P2)

- P2 1: no controlled row reached the padding mask; 3968 claim wrong (3968 runs one exact 4096-row piece);
  limit 4 equal to limit 2 in every probed case.
  - CPU planner check: with limit 2, the mask was reached only on linear layers, 896 lengths
    (1281-1408, 2305-2432, 2561-2688, 3073-3584).
  - Measured limits 1-4 at linear 1300, 2400, 2600, 3300, 3584 on p150x4 and p150x2
    (`evidence6/micro/padding_routes/pieces3_*.json`, chain 5 code plus logging only):
    3 or 4 pieces never beat 2; 2 pieces lose to padding at 1300, 3300, 3584.
  - New rule in `_moe_pieces`: split only if the pieces total less than the single bucket, and a padded last
    piece only if the single bucket is the largest (4096). Fits all measured cases. The mask runs whenever a piece is
    padded: linear 1281-1408 and 3073-3584 (one padded bucket) and 2305-2432 and 2561-2688 (split); full layers never
    at start 0.
  - `valid_rows` reset in `try`/`finally`. Row JSONs record the MoE pieces (`moe_pieces`).
  - Chain 6 adds length rows 1300, 2400, 2600, 3300.
- P2 2: repeat-rate change on the split path recorded in README (per-path rates).
- Chain 6 (`evidence6/`, superseded, kept): child row `len_m4_linear_2600` exited 1 (the `step` wrapper still
  printed rc 0). Error: `moe_fused_swiglu` static circular buffers (1024-row piece, 4x4 grid, region end 1083584)
  clash with an L1 buffer at 1039104 (`evidence7/micro/l1_clash/chain6_len_m4_linear_2600_failure.log`).
- Cause (`evidence7/micro/l1_clash/x4_layer0_b.json`): `ttnn.all_reduce` creates CCL global semaphores
  (4 bytes per core, all cores) when a program is first built and keeps them. The 2048-row prefill all-reduce
  was first built while large L1 intermediates were live, so its semaphores landed near 1.04 MB.
- Fix: prefill all-reduces (rows > 1) use `ttnn.experimental.all_reduce_async` with semaphores created once per
  mesh at construction (top of L1). Decode keeps `ttnn.all_reduce`: explicit semaphores cost +9 us per traced
  decode all-reduce on p150x4 (25.7 vs 16.8 us, `evidence7/micro/ccl_semaphores/x4.json`); prefill cost is
  equal within 7 us at 4096 rows (x4) and 22 us (x2).
- Probe after the fix: no low L1 buffers; 1000, 2600, 2400 pass. Screen (`runs/main/stage04/screen_semaphores/`):
  headline timing and control PCC unchanged.
- Chain 7 adds a scan of every `commands.json` for child exit codes (`CHILD_FAILURES`).
- Chain 7 (`runs/main/stage04/evidence_chain7.sh`, final): `SOURCE_STABLE`, `CHILD_FAILURES 0`; every step exits 0
  except the 3 expected `real_positions_0_m*`. Length rows 1300, 2400, 2600, 3300 pass on all profiles; 2400 and
  2600 linear run the masked padded piece on p150x2 and p150x4 (`moe_pieces` in the row JSON).
- Repeated prefill timing on chain 7 sources (`evidence7/micro/prefill_timing7/`, manifests equal to chain 7).
- p150x4 B1 prefill repeat: 0 of 64 rows in chains 1-4, 9 of 75 in chains 5-7, all at 2176-2400 tokens; cause
  not separated; bound unchanged (README "Nondeterminism").

### Review 6 (`stage_review6.md`, more-work-needed, 2 P2, documentation)

- Mask coverage statement corrected (README length table and plan; work log above).
- Nondeterminism rates and bound restated over all 7 chains (chain 6 column added; repeat PCC >= 0.9999973;
  p150x2 7 rows in chain 6 vs 1 in chain 7, which differ only in the prefill all-reduce).
- Stale `evidence7/micro/prefill_timing/` (copy of the chain 5 sweep) removed; the chain 5 copy stays in `evidence5/`.
- Limitations list the CCL path selection, decode semaphore timing and per-wrapper ping-pong.
