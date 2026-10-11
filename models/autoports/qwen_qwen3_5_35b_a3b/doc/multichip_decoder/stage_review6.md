# Stage Review

Stage 4, multichip decoder. Review 6, after `stage_review.md` to `stage_review5.md`. Model: Qwen/Qwen3.5-35B-A3B,
revision 59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent. Review date: 2026 Oct 11.
I did not open a device, start a server, build, or run a device test. I ran CPU-only analysis of existing artifacts
and a CPU copy of the prefill planner (the live `_chunk_plan`, `_moe_pieces`, `_bucket_sizes` and `mesh_policy`,
imported without ttnn; `ttnn` was not in `sys.modules`).

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), PCC (Pearson correlation coefficient), HF (Hugging Face reference), B1/B32 (batch 1 /
batch 32), CCL (collective communication), AR (all-reduce), RS (reduce-scatter), AG (all-gather), L1 (core SRAM),
CB (circular buffer).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `MD` = `AP/doc/multichip_decoder`
- `E7` = `MD/evidence7`
- `LOG7` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain7.log`
- Planner scripts (scratch, not artifacts):
  `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/plan6.py` and
  `.../scratchpad/mask6.py`

Verdict: more-work-needed

## Required Work

- P2: README and work log misstate which lengths reach the padding mask.
  Evidence:
  - The mask is active whenever a MoE piece is padded: `moe_rows > rows_here`, then `dispatch.valid_rows =
    rows_here` and `route_maps` masks rows `>= valid_rows` (`AP/tt/optimized_decoder.py:434-448`,
    `AP/tt/multichip_decoder.py:257-258`). This includes the case where `_moe_pieces` returns one padded bucket.
  - The chain-7 row JSONs show it: `E7/len_m{2,4}_linear_1300/default.json` `moe_pieces` = `[[2048, 1312]]` and
    `E7/len_m{2,4}_linear_3300/default.json` = `[[4096, 3328]]`. Both are masked single buckets.
  - CPU planner over start-0 lengths 1 to 4096 on p150x4 (`mask6.py`): linear layers reach the mask at 896 lengths:
    split plus mask at 2305-2432 and 2561-2688 (256 lengths), single padded bucket at 1281-1408 and 3073-3584 (640
    lengths). Full-attention layers never reach it at start 0 (1278 lengths use an exact split).
  - `MD/work_log.md:227-228` says "Mask now reached at linear 2305-2432 and 2561-2688". That is wrong. The README
    non-aligned table describes 1300 as "one 2048-row group" and 3300 as "one 4096-row group" and marks only 2400 and
    2600 as "padding routes masked". Review 5 required step 1 was to state which rows reach the split and which
    reach the mask.
  - The code and the evidence are correct. All four mask ranges have a baseline-controlled row on p150x2 and
    p150x4 (control prefill PCC >= 0.99994, decode >= 0.99924, exit 0).
  Why this matters: later stages (full model, datatype sweep, vLLM) use this README to decide which paths the
  evidence covers. Review 5 made this exact statement a required fix.
  Required next step: correct `MD/work_log.md` and the README non-aligned table (rows 1300 and 3300 linear:
  "single padded bucket, padding routes masked"; full layers: no mask at start 0). No hardware run is required.

- P2: The nondeterminism rate and bound statements contradict the kept chains, including chain 6, which the README
  itself cites.
  Evidence (re-derived from `MD/evidence{1..7}/*/commands.json` and `*_commands.json`):
  - README "Rates by path": "p150x2: rare (1-2 rows per chain, 31-71 rows)". Actual p150x2 drifting rows per
    chain: 2, 1, 1, 2, 3, 7, 1. Chain 6 has 7 (`len_m2_full_1000`, `_full_2400`, `_full_3000`, `_linear_1000`,
    `_linear_2400`, `_linear_2600`, `_linear_3000`; 31 to 94 rows each). Chain 1 has 159 rows at full 1000, which the
    README table itself shows.
  - README bound: "repeat PCC >= 0.9999982". Chain 6 `b32_m4_full` has repeat PCC 0.9999973 (minimum per chain:
    0.9999986, 0.9999983, 0.9999983, 0.9999990, 0.9999983, 0.9999973, 0.9999984).
  - The p150x4 B1 rate line counts chain 6 ("chain 6 kept: 2 of 27"), but the table has no evidence6 column and the
    p150x2 and bound statements leave chain 6 out. The anomaly ledger still says "max abs 0.0156 across 4 chains".
  - Chain 6 and chain 7 differ only in the prefill AR (`ttnn.all_reduce` against `all_reduce_async`; I diffed the
    `MD/evidence6/source_snapshot.tar.gz` sources against the live tree: only `tt/multichip_decoder.py` changed).
    The p150x2 rate went from 7 rows to 1 row. That is a data point for the Stage 5 localization, which now lists
    only "MoE expert compute and combine" as the affected path.
  - Max abs stays at 0.0156 in all chains, so the "controlled by bound" resolution still holds if the PCC bound is
    restated.
  Why this matters: the anomaly rule requires a classification that matches the evidence. Review 5 required the
  per-path rates for the same reason. Stage 6 and the serving stages size their repeatability checks from these
  rates.
  Required next step: add an evidence6 column or state that chain 6 is excluded and why; restate the p150x2 rate
  (1 to 7 rows per chain, 31 to 159 rows) and the repeat-PCC bound (>= 0.9999973); fix "across 4 chains"; record the
  chain-6 to chain-7 p150x2 rate change as an open observation for Stage 5. No hardware run is required.

## Earlier Findings: Status

- Review 5 P2 1 (no controlled row reached the mask; 3968 claim wrong; limit 4 equal to limit 2):
  - 3968 row now says "one exact 4096-row MoE group (no split, no mask)". Correct (planner and
    `E7/len_m*_3968/default.json` `[[4096, 4096]]`).
  - Controlled mask rows added: 1300, 2400, 2600, 3300 on all profiles. All 78 length rows exit 0. Every
    `moe_pieces` value I checked (52 mesh rows) equals my CPU planner result.
  - Limits 1 to 4 measured at 1300, 2400, 2600, 3300, 3584 on p150x4 and p150x2
    (`E7/micro/padding_routes/pieces3_x{2,4}_layer0.json`). The README table matches the JSON medians in all 40
    cells. The selected rule picks the faster of limit 1 and limit 2 in all 10 cases; limits 3 and 4 never win.
  - Piece list recorded per row (`moe_pieces`). Rows that take the tiled `prefill_forward` path (4096, the 2176 =
    128 + 2048 rows) log `[]`, because only `_batched_prefill` writes the log. See Other Concerns.
  - Remaining defect: the mask ranges in the docs (P2 1 above).
- Review 5 P2 2 (repeat rate on the split path): README now gives per-path rates and the p150x4 B1 count (9 of 75 in
  chains 5 to 7; I confirm 3 of 20, 2 of 27, 4 of 28). It says that 2176 = 128 + 2048 drifts without a split piece,
  which is correct: that row uses the tiled path. Remaining defect: P2 2 above.
- Review 5 other concerns: `valid_rows` reset in `finally`: fixed. Stale 3-token ledger row: fixed (5.1 against
  3.8 ms, repeated medians). Exploratory label on the `padding_routes` probes: added. `merge_stats.sharded_bytes`
  still 0 (`E7/headline_p150x4_full/default.json`). Attention-chunk padding rows still not masked. Both are
  cosmetic or cost-only.
- Reviews 1 to 4: still resolved. Headline rows, speedups, controls and HF PCC re-derived below.

## Other Concerns

- Semaphore ping-pong is per wrapper instance (`_AllReduceOutput.semaphore_index`), but all wrappers on a mesh share
  the same two sets (`_CCL_SEMAPHORES[mesh]`). The attention AR and the MoE AR of a chunk therefore use the same set
  back to back (A0 F0 A1 F1 ...). By inspection this is safe: `reduce_scatter_minimal_async` and
  `all_gather_async` run a start barrier on the barrier semaphore when no persistent buffers are passed
  (`/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:902`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:730`), and readers reset their
  semaphores before the op ends. The native-context rows (64 chunks, about 128 back-to-back ARs) pass on p150x2 and
  p150x4. The ping-pong is therefore not what makes reuse safe. Use one counter per mesh, or document the barrier
  dependence, before the full model stacks 40 layers.
- Decode still uses `ttnn.all_reduce`, which creates its global semaphores when its program is first built. Their
  L1 placement depends on the L1 state at first decode compile. This is the same mechanism as the chain-6 clash.
  The layer harness compiles decode with little L1 in use, and rows 3 and 4097 (decode-path tail inside prefill,
  then a warm prefill) pass. The full model must compile decode before it allocates persistent L1 state, or use
  explicit semaphores. Record this for Stage 6.
- Decode against prefill selection uses `partial.shape[-2] == 1`. The profiles confirm the intended split (decode:
  `ReduceScatterMinimalDirect` + `AllGather`; prefill: `ReduceScatterMinimalAsync` + `AllGatherAsync`,
  `E7/profiles/m4_*/`). A Stage 5 batched decode with B rows on dim -2 would silently take the explicit-semaphore
  path, which costs +9 us per traced AR on p150x4 (`E7/micro/ccl_semaphores/x4.json`). Select by mode, not by shape.
- `E7/micro/prefill_timing/` is a byte-identical copy of `MD/evidence5/micro/prefill_timing/` (chain-5 sources)
  inside the final evidence folder, with no label. README cites only `prefill_timing7/`, but the stale folder can
  mislead. Remove it or add a note.
- `moe_pieces` is `[]` for rows on the tiled path (headline 4096 rows, 2176 = 128 + 2048 rows, `watcher_long`
  4096). The MoE still runs there. README "Row JSONs record the MoE pieces" should say that only the planned path
  logs pieces.
- README "3300 and 3968 are 0.6-0.8 ms above 4096 on meshes": the table gives 0.1 ms (p150x2 linear 3300) to 0.9 ms
  (p150x2 linear 3968).
- README plan table: "exact bucket pieces, at most 2" for tp 2 and 4. The rule also allows a padded last piece when
  the single bucket is 4096 (2400, 2600 linear).
- README "p150 outputs are bitwise equal to the Stage 3 baseline in 26 of 28 rows": with the 2 B32 rows it is 28 of
  30 (`control_equivalence`). The 2 exceptions are correct.
- `_CCL_SEMAPHORES` holds semaphores per mesh object and never releases them. A process that closes and reopens a
  mesh keeps the old entries. Harmless in the harness.

## Hard-Check Gaps

- The mask branch still never ran under watcher. Watcher rows use 65, 129, 257, 1000 and 4096 tokens; none reaches a
  padded MoE piece.
- The rule selection rests on exploratory probes (chain-5 code plus logging, no source manifest). The frozen
  `prefill_timing7` table confirms the final default, but has no frozen A/B of the rule against limit 1 or 2.
- The source manifest hashes the ttnn patch text (723ed3a1, unchanged), not the built `_ttnn.so`.
- No mesh test drives an EP rank with zero local routes through a whole layer.
- No probe lists L1 buffers after a decode compile, so the placement of the lazily created decode AR semaphores is
  not measured.

## Anomaly Ledger

- Observed anomaly: chain 6 `len_m4_linear_2600` failed with a static CB clash (CB region end 1083584, L1 buffer at
  1039104).
  Evidence: `MD/evidence6/len_m4_linear_2600/commands.json` (exit 1, kept), `E7/micro/l1_clash/` (probe logs list
  4-byte height-sharded buffers at 1038720 to 1039168 before the fix and only near 1.56 MB after it).
  Affected path: prefill AR semaphores created lazily by `ttnn.all_reduce` while large L1 intermediates were live.
  Control or comparison: 1000 tokens passes in the same process; fixed probe passes 1000, 2600, 2400.
  Likely subsystem: CCL global semaphore allocation.
  Investigation performed: L1 buffer listing per `moe_fused_swiglu` call; construction-time semaphores; full chain 7.
  Resolution: fixed (chain 7 row exits 0; `CHILD_FAILURES 0`). Decode keeps the lazy path (Other Concerns).
- Observed anomaly: docs misstate the mask lengths.
  Evidence: P2 1. Resolution: more-work-needed (documentation).
- Observed anomaly: prefill not bitwise repeatable (p150x4 B32 always; p150x4 B1 9 of 75 rows since chain 5;
  p150x2 1 to 7 rows per chain; single chip rare).
  Evidence: `MD/evidence{1..7}` `prefill_repeat_identical`; E0/E2 (`MD/evidence4/nondeterminism/e0e2/`); AutoDebug
  (`MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`).
  Control or comparison: p150 and baseline B32 repeat bitwise; isolated op replays give one value.
  Likely subsystem: prefill MoE compute and combine in the mesh context; the chain-6 to chain-7 p150x2 drop suggests
  the CCL path can change the rate.
  Investigation performed: E0, E2a, E2b, AutoDebug (earlier); reviewer re-count per chain.
  Resolution: controlled by bound (max abs 0.0156), owner Stage 5; rate and PCC-bound text needs P2 2.
- Observed anomaly: p150x4 linear 2600 decode PCC against HF 0.99911 (p150 0.99971); control decode 0.99924, the
  lowest p150x4 control value.
  Evidence: `E7/len_m4_linear_2600/commands.json`. Control: other mesh rows 0.99909 to 0.99984 against HF; p150x2
  linear 4095 0.99920. Resolution: controlled (inside the mesh range; HF gates pass).
- Observed anomaly: p150x4 full 3-token prefill slower than p150 (5.1 against 3.8 ms, repeated medians).
  Control: every decode-path token pays two ARs. Resolution: expected; Stage 5.
- Observed anomaly: layer-0 recorded-input positions row 17 token 4 (0.99393, 0.99396, 0.99414; the 3 expected
  `rc=1` steps in `LOG7`). Control: Stage 3 single chip 0.99393. Resolution: controlled.
- Observed anomaly: p150x4 watcher build needs `TT_METAL_WATCHER_NOINLINE=1`. All 16 chain-7 `watcher.log` files have
  no error, assert, sanitizer or timeout line. Resolution: fixed.

## Verified Claims

- Chain 7: 172 chain steps; all `rc=0` except `real_positions_0_m{4,2,1}`; `SOURCE_STABLE`, `CHILD_FAILURES 0`,
  `CHAIN_DONE` (`LOG7`). All acceptance groups exit 0 inside their `commands.json`. Device pytest: 12 passed.
  CPU pytest: 1167 tests, 0 failures, 0 errors, 0 skipped (`E7/cpu_pytest.xml`).
- Provenance: `E7/source_manifest_start.json` equals `source_manifest_end.json` and both
  `E7/micro/prefill_timing7/source_manifest_{start,end}.json`. All 41 file hashes match the live worktree. `git_head`
  296a8bff. `git diff HEAD -- ttnn tests/ttnn` hashes to 744236997d8a (manifest value); `E7/ttnn_patch.diff` hashes
  to 723ed3a1. `E7/memory_handoff_recheck.json` `source_sha256` equals the manifest (7 of 9 items addressed, 2 open
  for the full model).
- Headline table (8 rows): prefill, decode, speedups, control PCC and HF PCC match the raw JSON.
- Repeated prefill table: all 104 cells match `E7/micro/prefill_timing7/*.json`; maximum spread 2.09 ms (README
  "at most 2.1 ms").
- All 92 chain-7 `default.json` files: `repeated_identical` true, `output_replicas_identical` true, `runtime_audit`
  true. Trace rows run with `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`
  (`AP/tests/run_optimized_trials.py:72`, `AP/tests/run_fused_acceptance.py:25`).
- First against warm prefill: 113 of 122 rows identical; 7 mesh and 2 baseline exceptions, as README says.
- Reuse (18 cases per layer, 1 capture across requests) and native context 262143 and 262144 pass on all three
  profiles; `AP/doc/context_contract.json` `capability_reduction` is null and its artifacts point to `E7`.
- `_moe_pieces` rule: the CPU planner matches every logged piece list; groups never exceed 4096 rows; all piece
  sizes have prepared EP dispatch buffers.
- Chain 6 to chain 7 source change is only the prefill AR in `AP/tt/multichip_decoder.py`.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`. I
  did not re-read the multichip, optimize and tt-device-usage skills in full; I checked their requirements through
  the goal contract and the earlier reviews.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `MD/stage_review5.md`, `E7/` (manifests, headline, `len_*`,
  `len_base_*`, `b32_*`, `accept_*`, `real_inputs_m*`, `watcher_long`, `profiles/`, `memory_handoff_recheck.json`,
  `cpu_pytest.xml`, `micro/l1_clash/`, `micro/ccl_semaphores/`, `micro/padding_routes/`, `micro/prefill_timing7/`,
  `micro/prefill_timing/`), `MD/evidence{1..6}` repeat fields, `MD/evidence6/len_m4_linear_2600/`,
  `MD/evidence{5,6}/source_snapshot.tar.gz`, `LOG7`,
  `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain7.sh`, `AP/doc/context_contract.json`.
- Code paths: `AP/tt/multichip_decoder.py` (all), `AP/tt/optimized_decoder.py` (`prefill_forward`, `_chunk_plan`,
  `_attention_step`, `_batched_prefill`, `_planned_prefill`, `_moe_pieces`, `_bucket_sizes`),
  `AP/tests/run_layer.py` (prefill, repeat and `moe_pieces` logging), `AP/tests/test_multichip_plan.py`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp`,
  `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/composite_common.cpp`, RS and AG barrier code.
- Commands run: `git status`, `git log`, `git diff | sha256sum`, `sha256sum`, `cmp`, `diff`, `tar -x` (to scratch),
  `grep`, Python over JSON, XML and logs, and the CPU planner copy. No device, server, build or device test.

## Residual Risk

- Mask correctness at start 0 rests on four controlled lengths per mesh and code inspection; it never ran under
  watcher.
- The piece rule is fitted to 5 measured lengths per mesh. Other lengths in the same ranges follow by the same
  piece shapes, but are not timed.
- Prefill repeatability on meshes varies by chain and by CCL path; the mechanism is unknown (Stage 5).
- Full-model L1 headroom: `moe_fused_swiglu` static CBs reach about 1.08 MB on a 1024-row piece. Persistent L1 state
  in the full model, and lazily created decode semaphores, can bring back the clash class.
