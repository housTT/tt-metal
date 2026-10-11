# Stage Review

Stage 4, multichip decoder. Review 5, after `stage_review.md`, `stage_review2.md`, `stage_review3.md` and
`stage_review4.md`. Model: Qwen/Qwen3.5-35B-A3B, revision 59d61f3ce65a6d9863b86d2e96597125219dc754.
Reviewer: fresh independent subagent. Review date: 2026 Oct 11. I did not open a device, run a server, build, or
run a device test. I ran CPU-only analysis of existing artifacts, plus a CPU copy of the prefill planner
(`_chunk_plan`, `_moe_pieces`, `_bucket_sizes` taken by Abstract Syntax Tree (AST) extraction from the live
`optimized_decoder.py`, without ttnn).

Abbreviations: TP (tensor parallel), EP (expert parallel), MoE (mixture of experts), GDN (Gated DeltaNet),
PCC (Pearson correlation coefficient), HF (Hugging Face reference), B1/B32 (batch 1 / batch 32),
CCL (collective communication), TTFT (time to first token).

Short names:
- `AP` = `/home/hous/dev/qwen3.5-35b-a3b/tt-metal/models/autoports/qwen_qwen3_5_35b_a3b`
- `MD` = `AP/doc/multichip_decoder`
- `E5` = `MD/evidence5`
- `PR` = `E5/micro/padding_routes`
- Planner script (scratch, not an artifact): `/tmp/claude-2005/-home-hous-dev-qwen3-5-35b-a3b/5bead60f-0645-4ca5-b27d-6b0f6acbc38f/scratchpad/plan.py`

Verdict: more-work-needed

## Required Work

- P2: The new padding-route mask branch has no baseline-controlled length row, and README and work log claim
  coverage that the artifacts contradict. The "4 pieces" rejection compares identical plans.
  Evidence:
  - The mask is active only when a MoE piece is padded (`moe_rows > rows_here`, then
    `dispatch.valid_rows = rows_here < length`, `AP/tt/optimized_decoder.py:434-446`,
    `AP/tt/multichip_decoder.py:254-255`). That happens only when the group's padded row count is not a sum of
    at most 2 bucket sizes.
  - 3968 tokens does not reach this branch. The planner makes one masked 4096-row attention chunk (valid 3968),
    so the MoE group is one exact 4096 piece and `valid_rows == length`. The stage's own probe shows this:
    `PR/pieces_x4_layer0.json` and `PR/pieces_x4_layer3.json`, length 3968, limit 2: one `route_maps` call,
    `rows 4096, valid_rows 4096`, identical to limit 1 (PCC 1.0, max abs 0). My CPU planner copy gives the same
    plan for both layer kinds.
  - README contradicts this. The non-aligned table row says "3968 | mesh MoE split 2048 + 2048 (last piece
    padded, padding routes masked)". `MD/work_log.md:211` says chain 5 "adds length rows 3000 and 3968 for the
    split path". 3000 does split (2048 + 1024), but neither piece has MoE padding, so the mask is not active
    there either. Its "last piece padded" means 72 attention-chunk padding rows, which are not masked.
  - Chain-5 lengths 3, 100, 1000, 2176, 2176 = 128 + 2048, 3000, 3968, 4095, 4097, the B32 rows, the positions
    prefixes (1 to 1006), the watcher rows (65, 129, 257, 1000, 4096) and the native-context rows (262143,
    262144): none of them reach the mask branch (CPU planner copy). The only gate that reaches it is the
    `accept_reuse_m{2,4}` case "1153 with cuts [1, 3, 127, 129, 160, 513]". Its continuation pieces of 353
    tokens at start 160 and 640 tokens at start 513 produce padded MoE pieces (for example full: [512, 128 with
    96 valid]). That case passes against HF only (`E5/accept_reuse_m4/full.json`, prefill PCC 0.99841, p150
    0.99839; threshold 0.995 in `AP/tests/run_reuse.py:197`). It has no single-chip control comparison.
  - The branch is common for ordinary start-0 prompts. For linear-attention layers at start 0, 896 of the
    lengths 1 to 4096 reach it: 1281-1408, 2305-2432, 2561-2688 and 3073-3584 (for example 3584 =
    [2048, 2048 with 1536 valid]). For full-attention layers it appears in continuations with unaligned starts.
  - README "Up to 4 pieces gave no further gain on p150x4" rests on `PR/pieces_x*_layer*.json`. In every probed
    case, limit 4 produced exactly the same `route_maps` calls as limit 2, because no probed length needs more
    than 2 pieces. The comparison did not test the 4-piece policy. Where the mask branch is active, a third piece
    removes padding: 3584 linear could run 2048 + 1024 + 512 with no padding rows, against 2048 + 2048 now.
    The stage's own classification says EP prefill cost follows padded rows. So at 3073-3584 the selected limit
    of 2 pads to 4096 rows in two calls, the same row count as limit 1. Neither alternative was measured there.
  Why this matters: the goal requires control PCC for every meaningful path and a public contract for valid
  non-aligned lengths. The stage-review skill requires coverage of changed padding and chunking paths, nearby
  boundary lengths and awkward tails, and earlier evidence does not cover a later change. The README tells
  later stages that 3968 qualifies the masked split. It does not.
  Required next step:
  1. Correct README (non-aligned table rows 3000 and 3968, "MoE group split" bullets) and `MD/work_log.md` to
     state which rows reach the split and which reach the mask. Cite the reuse 1153-cut case as the only current
     mask coverage.
  2. Add baseline-controlled length rows that reach the mask branch on p150x2 and p150x4, at least one linear
     start-0 length from the ranges above (for example 3584 or 1300). Run them under a frozen source manifest
     with the same `--compare-tensors` control as the other length rows.
  3. Measure limit 2 against limit 3 or 4 (and limit 1) at one such length on p150x4. Then select the policy, or
     remove the "4 pieces" sentence. Record the planner's piece list in the row JSON so later reviews do not
     have to re-derive it.

- P2: The new split path changes the prefill non-repeatability rate on p150x4 B1, and the classification does
  not say so.
  Evidence:
  - `prefill_repeat_identical` false rows, all chains (re-derived from `MD/evidence{1..5}/*/commands.json`):
    chains 1 to 4 have no false p150x4 B1 row (4 chains x 4 rows at 2176 tokens, all true). Chain 5 has false
    `len_m4_full_2176`, `len_m4_full_2176_s128` and `len_m4_linear_2176_s128`, plus new p150x2 3000 full and
    linear. These are exactly the lengths that the chain-5 change moved from one padded MoE call to two pieces.
  - README "Rate" and the anomaly ledger still say "p150x4 B32 always, p150x2 and baseline rarely". README
    "Nondeterminism" lists the new rows only under "other mesh rows (evidence5 only has 3000 and 3968 rows)",
    which is unclear (2176 rows exist in every chain), and its header says "same configurations in all four
    chains" above five chain columns.
  - The localization (E0, E2) ran on B32 129 tokens with chain-4 sources. README argues that the split does not
    change that path. That is correct for B32, but the new drift appears on the path that did change.
  - Magnitudes stay within the recorded bound (max abs 0.0078 for full 2176 = 128 + 2048; 0.0020 for full 2176;
    `E5/len_m4_*_2176*/commands.json`). Control PCC is at least 0.99995 and the HF gates pass.
  Why this matters: the anomaly rule requires a classification that matches the evidence. Greedy serving
  repeatability on p150x4, the optimization target, now depends on this path for common prompt lengths. A
  rate statement that omits the changed path misleads the Stage 5 owner and the serving-stage repeatability
  check.
  Required next step: update the classification and ledger with the per-path rate (split path on p150x4 B1: 3 of
  4 rows in chain 5 against 0 of 16 before), and fix the table header and the "other mesh rows" label. Keep the
  "controlled by bound" resolution only if the bound statement covers these rows. That is true today, at max
  abs 0.0078 against the 0.0156 bound. No hardware run is required for this item.

## Earlier Findings: Status

- Review 4 P2 (mesh 2176-token prefill slower than 4096 tokens): classified and fixed for the probed lengths.
  - Classification: `PR/x4_layer3_2176.json` and `PR/x4_layer0_2176.json` show 20130 / 19594 routes on mesh
    device 0 and 3790-4932 on the others without the mask. With the mask, device 0 has 4770 / 4234 routes. The
    median stays 20.1 / 20.6 ms. Masked against unmasked output PCC is 0.9999994 / 0.9999996. This matches the
    README statement that the padded row count, not route imbalance, sets the cost.
  - Fix: `prefill_moe_pieces` = 2 on tp 2 and 4 (`AP/tt/multichip_decoder.py` `MESH_POLICIES`), 1 on tp 1. The
    p150 path has no `expert_dispatch` (the tp 1 branch returns before line 417), so p150 stays bitwise equal to
    the baseline. I confirmed `control_equivalence` prefill and decode identical in all 22 p150 rows.
  - The README "MoE group split" table matches `PR/pieces_*.json` medians for every cell I checked (all 24).
  - The frozen repeated timing table (`E5/micro/prefill_timing/*.json`) matches README to 0.1 ms in all 72
    cells. Its start and end manifests equal `E5/source_manifest_end.json`. The maximum spread is 1.29 ms
    (README: at most 1.3 ms). p150x4 2176 tokens: 15.2 ms full, 15.1 ms linear, against 18.7 / 20.7 ms at 4096
    tokens. The inversion that review 4 found at 2176 tokens is gone.
  - Remaining gap: see P2 1 (mask-branch lengths and piece limit).
- Review 4 other concerns:
  - Device pytest default devices 0,1: README now states that the chain sets 2,3 and the test default is 0,1.
    Resolved as documentation.
  - Short-prefill fixed cost: listed under "Limitations and Stage 5 work". Resolved.
  - `merge_stats.sharded_bytes` still reads 0 (`E5/headline_p150x4_full/default.json`:
    `{'sharded': 11, 'replicated': 87, 'sharded_bytes': 0}`; `AP/tt/multichip_decoder.py:344`). Not fixed.
    See Other Concerns.
  - "repeat" spelled out as "bitwise repeat": resolved.
- Reviews 1 to 3: still resolved. The headline rows in chain 5 equal chain 4 for the 4096-token path (control
  and HF PCC identical to all printed digits), as expected, because that path did not change.

## Other Concerns

- Attention-chunk padding rows are not masked. `valid_rows` is set to `rows_here`, which includes the masked
  tail rows of attention chunks. Examples: 257 full is one 512-row chunk with 255 padding rows; 3968 is a
  4096-row chunk with 128 padding rows; the reuse continuation 640 at start 513 has 511 padding rows in a
  1024-row chunk. In `PR/pieces_x4_layer3.json` at 3968 tokens, the routes per device are 8494, 7319, 7706 and
  9249. The stage's own classification says that the route count does not set the cost, so this is probably
  immaterial. But the mask could use the true valid count. Repeated timings at 3968 against 4096 are similar on
  the baseline (+0.5 / +1.2 ms) and on p150x4 (+1.0 / +0.8 ms).
- `dispatch.valid_rows` is not reset if `feed_forward.forward` raises (no `try`/`finally`,
  `AP/tt/optimized_decoder.py:441-446`). After a caught exception, a later prefill of the same piece length
  would mask real rows. Low risk, but cheap to fix.
- Anomaly ledger row "p150x4 3-token prefill slower than p150 (5.5 ms vs 4.7 ms, full)" is stale. Chain-5 rows
  are 5.55 / 4.41 ms (`E5/len_m{4,1}_full_3/commands.json`); repeated medians are 5.4 / 3.9 ms.
- `merge_stats.sharded_bytes` is always 0, because `items[0]` has no `buffer_size`. Remove or fix the field.
- The selection of 2 pieces for meshes and padding for p150 rests on `PR/` screens that have no source
  manifest. README does not label them exploratory. The final default is confirmed by frozen chain 5 and the
  frozen repeated timing for meshes. The p150 rejection of the split (slower at 3 of 4 lengths) has only the
  exploratory screen. p150 is not the optimization target, so I list this here, not as required work.
- p150x2 linear decode control PCC is 0.99845 in the headline and 0.9993-0.9995 at 2176-4097 tokens. p150x4
  gives 0.99888 and at least 0.99996. It is unchanged since chain 4 and was accepted earlier. HF PCC is unchanged
  by the mesh. Not a new finding.

## Hard-Check Gaps

- The mask branch never ran under watcher. The watcher rows (65, 129, 257, 1000, 4096) reach the split only at
  257 linear (pieces 256 + 128, no MoE padding).
- Each chain row is one timed run. The frozen repeated table is the latency reference, but it has no length in
  the mask ranges.
- The source manifest hashes the ttnn patch text (723ed3a1, unchanged from chain 4), not the built `_ttnn.so`.
- No mesh test drives an EP rank with zero local routes through the whole layer. The masked probe gives
  device 0 its share of routes; no case gives a device none. By code reading, capacity and the zeroed dummy
  row make this safe.
- The device pytest (12 passed) covers lengths 100, 129 and 1000. None of them uses the split or the mask.

## Anomaly Ledger

- Observed anomaly: the mask branch is claimed for 3968 tokens, but 3968 runs one exact 4096-row MoE call.
  Evidence: `PR/pieces_x4_layer{0,3}.json` (3968, limit 2: rows 4096, valid_rows 4096); CPU planner copy;
  README non-aligned table; `MD/work_log.md:211`.
  Affected path: `_batched_prefill` piece loop with `moe_rows > rows_here`; `_ExpertParallelDispatch.route_maps`
  with `valid_rows < length`.
  Control or comparison: the reuse 1153-cut case reaches the branch, against HF only.
  Likely subsystem: evidence selection and documentation. No code defect found by inspection.
  Investigation performed: CPU planner re-derivation over lengths 1 to 4096 at start 0, the reuse cuts and the
  positions prefixes; cross-check with probe JSON.
  Resolution: more-work-needed (P2 1).
- Observed anomaly: the "4 pieces gave no further gain" comparison ran identical plans.
  Evidence: `PR/pieces_x*_layer*.json`. The limit 4 and limit 2 `route_maps` calls are identical in every case.
  Affected path: `_moe_pieces` limit selection.
  Control or comparison: none. No probed length needs 3 pieces.
  Likely subsystem: policy selection.
  Investigation performed: compared the logged calls per case.
  Resolution: more-work-needed (P2 1).
- Observed anomaly: p150x4 B1 prefill at 2176 tokens is now not repeatable (3 of 4 rows; 0 of 16 in chains 1
  to 4).
  Evidence: `MD/evidence{1..5}/len_m4_*_2176*/commands.json` `prefill_repeat_identical`.
  Affected path: two-piece MoE prefill on meshes.
  Control or comparison: p150 and baseline rows at the same length repeat bitwise. The magnitude is within the
  recorded bound.
  Likely subsystem: the same prefill MoE non-repeatability as the B32 rows (mechanism unknown, owner Stage 5).
  Investigation performed: cross-chain comparison by the reviewer only.
  Resolution: more-work-needed (P2 2: classification update; the bound still holds).
- Observed anomaly: mesh 2176-token prefill slower than 4096 tokens (review 4).
  Evidence: `PR/`, `E5/micro/prefill_timing/`.
  Control or comparison: frozen repeated medians; p150 and baseline.
  Investigation performed: per-device route counts, mask on and off, piece limits 1, 2 and 4, frozen repeat.
  Resolution: fixed for the probed lengths (2176: 21.3 to 14.7 ms full, 20.6 to 15.8 ms linear on p150x4).
- Observed anomaly: prefill not bitwise repeatable on B32 p150x4 and rarely elsewhere.
  Evidence and investigation: E0, E2a, E2b, AutoDebug (`MD/evidence4/nondeterminism/e0e2/`,
  `MD/evidence2/moe_nondeterminism_autodebug/AUTODEBUG.md`, `E5/nondeterminism/`).
  Resolution: controlled by bound (max abs 0.0156), root cause owned by Stage 5. The rate statement needs the
  update in P2 2.
- Observed anomaly: layer 0 recorded-input positions, row 17 token 4, PCC 0.99393 to 0.99414 (the 3 expected
  `rc=1` steps in `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain5.log`).
  Control or comparison: Stage 3 single chip 0.99393. Resolution: controlled.
- Observed anomaly: 3968 tokens slower than 4096 tokens on every profile, including the baseline.
  Evidence: `E5/micro/prefill_timing/*.json` (baseline full 26.0 against 25.5 ms; p150x4 full 19.7 against
  18.7 ms). Control or comparison: the baseline shows the same direction and similar size. Likely subsystem:
  masked 4096-row tail chunk. Resolution: controlled by the baseline.
- Observed anomaly: p150x4 watcher build needs `TT_METAL_WATCHER_NOINLINE=1`. All 16 chain-5 `watcher.log`
  files have no error, assert, sanitizer, hang or timeout line. Resolution: fixed.

## Verified Claims

- Chain 5: 138 `rc=0` and 3 `rc=1` (the expected `real_positions_0_m{4,2,1}`); the log ends with
  `SOURCE_STABLE` and `CHAIN_DONE`. `E5/source_manifest_start.json` equals `source_manifest_end.json`. All 41
  file hashes match the live worktree. `git_head` is 296a8bff. The `git diff HEAD` outside the autoport hashes
  to 744236997d8a, equal to the manifest and to `git diff HEAD -- ttnn tests/ttnn`. `E5/ttnn_patch.diff` hashes
  to 723ed3a1. `E5/memory_handoff_recheck.json` `source_sha256` equals the manifest.
- Headline table (all 8 rows), the 40-layer estimate (18.40 ms and 25.63 ms), the speedup and efficiency columns,
  and the minimum control PCC per length row group match the raw `commands.json` data. All 54 length rows and
  18 baseline length rows exit 0.
- First against warm prefill: 82 of 90 rows bitwise identical (80 of 88 plus 2 placement rows), as README says.
- CPU tests: 1163, 0 failures, 0 errors, 0 skipped (`E5/cpu_pytest.xml`). Device pytest: 12 passed
  (`E5/accept_pytest/device.log`).
- Reuse 1153-cut case (mask branch) on all profiles: prefill PCC 0.99839-0.99846 full, 0.99965 linear; decode
  minimum at least 0.9982; `guards_clean` true for full layers; replicas identical.
- Native context 262143 and 262144 on all three profiles (`AP/doc/context_contract.json`
  `multichip_decoder_validation`, `capability_reduction` null, artifacts point to `E5`).
- Code inspection of the mask: the row-index tensor has the same [length, 8] layout as `ids` before the
  optional transpose, and `tokens` uses `% length` in tiles mode. So masking by row is correct in both modes.
  Masked routes go to `capacity - 1`, which `restore_table` zeroes. Counts exclude the dummy column. Row
  indices up to 4096 are exact in FP32.
- `_moe_pieces` unit tests (`AP/tests/test_multichip_plan.py:158-171`) match the function for the listed cases.
  Groups never exceed 4096 rows, because the planner splits groups at `buckets[0]`. So `next(...)` cannot run
  out of sizes.

## Scope Inspected

- Goal/skill paths: `/home/hous/dev/qwen3.5-35b-a3b/runs/main/04-04-multichip-decoder.prompt.txt`,
  `/home/hous/dev/qwen3.5-35b-a3b/AGENTS.md`,
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`.
  I did not read the multichip, optimize and tt-device-usage skills again in full for this review. Their
  requirements were checked through the goal contract and the earlier reviews.
- Artifact paths: `MD/README.md`, `MD/work_log.md`, `MD/stage_review4.md`, `E5/` (manifests, headline, `len_*`,
  `len_base_*`, `b32_*`, `accept_reuse_m*`, `accept_positions_m4`, `accept_watcher_m*`, `watcher_long`,
  `accept_pytest`, `cpu_pytest.xml`, `memory_handoff_recheck.json`, placement rows), `PR/` (all files),
  `E5/micro/prefill_timing/` (all JSON, manifests, `timing.py`), `MD/evidence{1..4}/` repeat flags,
  `AP/doc/context_contract.json`, `/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage04/evidence_chain5.log`.
- Code paths: `AP/tt/optimized_decoder.py` (`_chunk_plan`, `_attention_step`, `_batched_prefill`, `_moe_pieces`,
  `_bucket_sizes`, `_LocalDispatch.__call__`, `_StackedDispatch`), `AP/tt/multichip_decoder.py`
  (`MESH_POLICIES`, `_ExpertParallelDispatch.prepare_maps`, `route_maps`, `restore_table`, `merge_replicas`,
  `from_state_dict` wiring), `AP/tests/test_multichip_plan.py`, `AP/tests/run_reuse.py`.
- Commands run: `git status`, `git log`, `git rev-parse`, `git diff | sha256sum`, `sha256sum`, `cmp`, `grep`, and
  Python over existing JSON, XML and logs; the CPU planner copy. No device, pytest, server or build.

## Residual Risk

- Correctness of the mask branch at start 0 rests on code inspection and one continuation case against HF.
- The piece limit is not tuned for lengths where 2 pieces still pad (for example 3073-3584 linear).
- Mesh prefill non-repeatability now also affects the B1 split path on p150x4. Serving stages need the
  warm-against-warm repeatability check already planned.
- Prefill timing for lengths outside the probed set is not measured.
