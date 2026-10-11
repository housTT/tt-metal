# Stage Review

Stage 4, multichip decoder. Review 7, after `stage_review.md` and `stage_review2.md` to `stage_review6.md`. Model:
Qwen/Qwen3.5-35B-A3B, revision 59d61f3ce65a6d9863b86d2e96597125219dc754. Reviewer: fresh independent subagent.
Review date: 2026 Oct 11. I did not open a device, start a server, build, or run a device test. I ran CPU-only
analysis of existing artifacts and the review 6 CPU copy of the prefill planner (live `_chunk_plan`, `_moe_pieces`,
`_bucket_sizes` and `mesh_policy`; no device opened).

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet, the
linear-attention layer), PCC (Pearson correlation coefficient), HF (Hugging Face reference), B1/B32 (batch 1 /
batch 32), CCL (collective communication), AR (all-reduce), L1 (core SRAM), CB (circular buffer), CPU (host
processor).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `MD` = `AP/doc/multichip_decoder`
- `E7` = `MD/evidence7`
- `LOG7` = `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain7.log`
- Scratch scripts (not artifacts): `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/nd7b.py`
  (drift recount), `.../scratchpad/mask6.py` and `.../scratchpad/plan6.py` (planner)

Verdict: clean-pass

## Required Work

None.

## Earlier Findings: Status

- Review 6 P2 1 (mask lengths misstated): fixed.
  - README lines 249-251 and the non-aligned table rows 1300, 2400, 2600, 3300 now state the mask ranges: linear
    1281-1408 and 3073-3584 (one padded bucket), 2305-2432 and 2561-2688 (split plus padded piece), 896 lengths;
    full layers never at start 0. `MD/work_log.md` review 5 and review 6 sections say the same.
  - I reran the planner on the live sources: linear split plus mask 256 lengths (2305-2432, 2561-2688), single
    padded bucket 640 lengths (1281-1408, 3073-3584); full 0 and 0. This matches the docs.
- Review 6 P2 2 (nondeterminism rates and bound): fixed. I recounted `prefill_repeat_identical` from every
  `commands.json` and `*_commands.json` in `MD/evidence1` to `E7` (`nd7b.py`). Every cell of the README
  nondeterminism table matches:
  - p150x2 B1 drifting rows per chain: 2, 1, 1, 2, 3, 7, 1 (rows differing 64-159, 32, 32, 32-64, 31-71, 31-94, 31).
  - p150x4 B1: 0 of 16 in chains 1-4; 3 of 20, 2 of 27, 4 of 28 in chains 5-7 (9 of 75, all 2176-2400 tokens).
  - p150x4 B32 rows and max abs per chain; p150 chain 6 linear 1000; baseline rows per chain.
  - Minimum repeat PCC 0.9999973 (chain 6), as README states. Max abs 0.0156 in all chains.
  - The chain 6 to chain 7 p150x2 change (7 rows to 1 row) is recorded as an open observation. "Across 4 chains"
    text is gone.
- Review 6 other concerns:
  - Stale `E7/micro/prefill_timing/`: removed. Only `prefill_timing7/` remains.
  - Ping-pong semaphores, lazy decode semaphores, shape-based CCL selection: recorded in README "Limitations and
    Stage 5 work".
  - `moe_pieces` `[]` on the tiled path: README line 248 states it.
  - "0.1-0.9 ms above 4096": matches `prefill_timing7` (I recomputed all eight cells).
  - "28 of 30 rows": fixed.
  - Plan table "exact bucket pieces, at most 2", `merge_stats.sharded_bytes` 0, `_CCL_SEMAPHORES` never released:
    unchanged, cosmetic (see Other Concerns).
- Reviews 1 to 5: still resolved. Implementation and chain artifacts are unchanged since review 6 (below).

## Other Concerns

- README anomaly ledger row "Single chain prefill rows vary up to 3 ms on meshes ... repeated timing spread
  <= 1.3 ms" (line 485) and line 222 ("vary up to 3 ms") have stale numbers. Chain-7 single rows lie 0.19 to
  3.83 ms above the `prefill_timing7` medians (largest: p150x4 linear 2600, +3.83 ms; p150x2 linear 1300,
  +2.77 ms). The repeated spread is at most 2.09 ms (p150x2 linear 1300), which README line 223 itself states.
  The resolution (the repeated table is the latency reference) still holds. Correct the two numbers.
- README "Communication: about 6% of decode" (line 410). From `E7/profiles/*/decode_replay_summary.txt`, the
  reduce-scatter plus all-gather share is 6.3% (p150x4 linear), 9.3% (p150x4 full), 5.4% (p150x2 linear) and
  8.5% (p150x2 full). Say 5-9%. The "not the limiter" conclusion holds.
- `prefill_timing7/timing.py` takes `warm[len(warm) // 2]` (upper median of 6 runs) and calls it the median. The
  difference from the true median is at most 0.1 ms in every cell. State "upper median" or compute the median.
- README reshape, typecast and slice family: 112.6 us (ledger, limitations) against 112.5 us (profile table).
  The summary gives 45.67 + 39.01 + 27.86 = 112.54 us. Rounding only.
- Plan table line 84 still says "exact bucket pieces, at most 2" for tp 2 and 4; the rule also allows a padded
  last piece (lines 141-145 explain this).
- `merge_stats.sharded_bytes` is still 0; `_CCL_SEMAPHORES` holds per-mesh entries forever. Harmless in the
  harness.

## Hard-Check Gaps

- The padding-mask branch never ran under watcher. Watcher rows use 65, 129, 257, 1000 and 4096 tokens.
- The `_moe_pieces` rule rests on exploratory probes (chain-5 code plus logging, no source manifest). The frozen
  `prefill_timing7` table confirms the final default but has no frozen A/B against limit 1 or 2.
- The source manifest hashes the ttnn patch text, not the built `_ttnn.so`.
- No mesh test drives an EP rank with zero local routes through a whole layer.
- No probe lists L1 buffers after a decode compile, so the placement of the lazily created decode AR semaphores is
  not measured.
- No p150 (1x1) tt-perf-report profile in `E7/profiles/`. p150 is bitwise equal to the Stage 3 baseline in 28 of
  30 rows and Stage 3 holds its profiles, so this is not a gap in the measured path.

## Anomaly Ledger

- Observed anomaly: chain 6 `len_m4_linear_2600` static CB clash with a CCL semaphore at 1039104.
  Evidence: `MD/evidence6/len_m4_linear_2600/commands.json` (exit 1, kept), `E7/micro/l1_clash/`.
  Affected path: prefill AR semaphores created lazily by `ttnn.all_reduce`.
  Control or comparison: 1000 and 2400 pass in the same process; probe after the fix passes.
  Likely subsystem: CCL global semaphore allocation.
  Investigation performed: L1 buffer listing; construction-time semaphores (`AP/tt/multichip_decoder.py`
  `_prefill_semaphores`); full chain 7.
  Resolution: fixed (chain 7 `CHILD_FAILURES 0`). Decode keeps the lazy path; recorded for Stage 6.
- Observed anomaly: prefill not bitwise repeatable (p150x4 B32 every chain; p150x4 B1 9 of 75 rows in chains 5-7;
  p150x2 1 to 7 rows per chain; single chip 0 to 2 rows per chain).
  Evidence: `MD/evidence{1..7}` `prefill_repeat_identical`; `MD/evidence4/nondeterminism/e0e2/`;
  `MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`; reviewer recount.
  Affected path: prefill MoE expert compute and combine; rate changes with the CCL path (chain 6 against 7).
  Control or comparison: p150 B32 and baseline B32 repeat bitwise; isolated op replays give one value (E2a, E2b).
  Likely subsystem: MoE prefill on meshes; mechanism unknown.
  Investigation performed: E0, E2a, E2b, AutoDebug; rates and bound re-derived here.
  Resolution: controlled by bound (max abs 0.0156, repeat PCC >= 0.9999973, all control and HF gates pass); owner
  Stage 5; serving stages must run a repeatability check.
- Observed anomaly: layer-0 recorded-input positions, row 17 token 4 decode PCC 0.99393 (p150), 0.99396 (p150x2),
  0.99414 (p150x4); the 3 expected `rc=1` steps in `LOG7`.
  Evidence: `E7/real_inputs_m{1,2,4}/positions_0.json` (`status` `failed_gate`, threshold 0.995; all other rows
  >= 0.9972).
  Control or comparison: Stage 3 single chip 0.99393
  (`AP/doc/optimized_decoder/final_evidence15/real_inputs/positions_0.json`).
  Resolution: controlled (inherited, mesh matches single chip).
- Observed anomaly: p150x4 linear 2600 decode PCC against HF 0.99911; control 0.99924.
  Control: other mesh rows 0.99909-0.99984. Resolution: controlled.
- Observed anomaly: p150x4 full 3-token prefill slower than p150 (5.1 against 3.8 ms, repeated medians).
  Control: decode-path tokens pay two ARs each. Resolution: expected; Stage 5.
- Observed anomaly: single chain prefill rows lie up to 3.83 ms above repeated medians on meshes.
  Evidence: chain-7 `len_m*/commands.json` `prefill_host_us` against `E7/micro/prefill_timing7/*.json`.
  Control: repeated medians with spread <= 2.09 ms. Resolution: controlled (repeated table is the reference);
  README numbers need the correction in Other Concerns.
- Observed anomaly: p150x4 watcher build needs `TT_METAL_WATCHER_NOINLINE=1`.
  Evidence: `MD/evidence1/accept_watcher_m4/linear.log`; the 16 chain-7 watcher logs have no error, assert,
  sanitizer or hang line. Resolution: fixed.

## Verified Claims

- No implementation change since review 6: all 41 file hashes in `E7/source_manifest_end.json` match the live
  worktree; start and end manifests are byte-identical, and both equal the `prefill_timing7` manifests.
  `git diff HEAD -- ttnn tests/ttnn` hashes to 744236997d8a (manifest value); `E7/ttnn_patch.diff` hashes to
  723ed3a1. HEAD 296a8bff; Stage 3 commit 81485827 is its parent.
- No `E7` file changed after `prefill_timing7` (14:16) except the `micro` directory entry (stale copy removed).
  The `prefill_timing7` logs run 14:15 to 14:16, after the chain finished (14:11).
- `LOG7`: 173 `rc=` lines; only `real_positions_0_m{4,2,1}` exit 1; `SOURCE_STABLE`, `CHILD_FAILURES 0`,
  `CHAIN_DONE`.
- First against warm prefill: 113 of 122 rows identical (92 mesh-decoder rows, 30 baseline rows; 9 drift).
- Headline table equals `E7/tables.md`. Efficiency and the 40-layer estimate (25.63, 25.62, 21.02, 18.37 ms;
  1.40x) recompute from the table.
- Repeated prefill table: all cells equal the `warm_median_us` fields in `E7/micro/prefill_timing7/*.json`.
- Native context 262143 and 262144 rows on all three profiles in `AP/doc/context_contract.json`;
  `capability_reduction` null; README PCC values match.
- Trace rows run with `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; no
  corruptible-buffer acknowledgment is used in `AP/tt` or `AP/tests`.
- Profile CCL sums (32.7 us p150x4 linear) and family sums match `decode_replay_summary.txt`.
- All cited evidence paths exist.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`.
  I checked the multichip, optimize and tt-device-usage requirements through the goal contract and earlier
  reviews; I did not re-read those skills in full.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `MD/stage_review6.md`, `E7/` (manifests, headline, `len_*`,
  `len_base_*`, `b32_*`, `real_inputs_m*`, `accept_watcher_*`, `watcher_long`, `profiles/*/decode_replay_summary.txt`,
  `micro/prefill_timing7/`, `micro/padding_routes/`, `nondeterminism/`, `tables.md`), `MD/evidence1` to
  `MD/evidence6` `commands.json` files, `LOG7`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain7.sh`,
  `AP/doc/context_contract.json`.
- Code paths: `AP/tt/multichip_decoder.py` (all), `E7/micro/prefill_timing7/timing.py`, trace-tracking lines in
  `AP/tests/run_optimized_trials.py`, `AP/tests/run_fused_acceptance.py`, `AP/tests/run_positions.py`.
- Commands run: `git status`, `git log`, `git diff | sha256sum`, `git merge-base`, `sha256sum`, `cmp`, `find`,
  `grep`, `zcat`, Python over JSON and logs, and the CPU planner copy. No device, server, build or device test.

## Residual Risk

- Mask correctness at start 0 rests on four controlled lengths per mesh and code inspection; never under watcher.
- The piece rule is fitted to 5 measured lengths per mesh.
- Prefill repeatability on meshes varies by chain and CCL path; the mechanism is unknown (Stage 5). Greedy
  outputs can differ between identical requests until this is resolved.
- Full-model L1 headroom: `moe_fused_swiglu` static CBs reach about 1.08 MB on a 1024-row piece. Persistent L1
  state and lazily created decode semaphores in the full model can bring back the clash class.
- Decode CCL path selection by `partial.shape[-2] == 1` will misroute a batched decode (Stage 5).
