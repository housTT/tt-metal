# Stage Review R2: release verification of tt-hous/clm-v0.1-8b-p150 revision 2

Reviewer: independent stage reviewer (read-only, no devices, no servers). Date: 2026 Oct 2.
Scope: Hub revision `2eb9cf5199d2302f5f3e7eeb71d55f39cdacabe5` (image `6cf6949ed327`, built from the clean worktree
`/home/hous/dev/clm-v0.1-8B/worktree/tt-metal` at commit `0cca94bc36`), the report
`/home/hous/dev/clm-v0.1-8B/REPORT.md`, the release notes
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/release/RUN_NOTES.md`, the card text in
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt-model.yaml` (and the Hub README of
revision 2, downloaded read-only), and
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/review/responses_round_2.md`.

Acronyms: HF (Hugging Face), OCI (Open Container Initiative), PCC (Pearson correlation coefficient), KL
(Kullback-Leibler divergence), ECE (expected calibration error), MLP (multilayer perceptron), RMSNorm (root mean square
normalization), bfp8 (8-bit block floating point), LoFi and HiFi (low and high fidelity matmul math modes), p50 and p95
(50th and 95th percentile), UTC (Coordinated Universal Time, the time base of every artifact timestamp quoted below).

Verdict: more-work-needed

The provenance chain is sound: the published manifest and README on the Hub are byte-identical to the staged files, the
image digest and code sha256 match the manifest, the build log and the git history agree with the stated commit and
`dirty: false`, and both kernel sources inside the staged image hash to the committed `0cca94bc36` versions. Almost
every figure in REPORT.md sections 2, 4, 5 and 7 reproduces from its result file. The verdict is driven by the published
card: four of its statements about the shipped default profile are wrong or stale against the same result files the
report cites, one of them in the opposite direction of the measurement, and the release notes still describe the
profile set of build 9.

## Required Work

- P1: Card claims `p150-accuracy` is faster than `p150` at 128 tokens batch 1; the measurement says the opposite.
  Evidence: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt-model.yaml` line 65
  (profile description) and lines 125 to 127 (card performance), identical in the Hub README of revision 2 and in
  `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/tt_kernel_manifest.json`: "1.5 percent faster than p150 at
  128 tokens batch 1 and 10 to 20 percent slower in every other cell". Measured: `p150` (accuracy_lofi_mlp with
  overrides) 128 tokens batch 1 = 53.14 ms
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/bench_accuracy_lofi_mlp_pc.json`);
  `p150-accuracy` (stock accuracy) = 57.76 ms
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/bench_accuracy_buckets5.json`).
  `p150-accuracy` is 8.7 percent slower in that cell, and REPORT.md section 5 line 128 states the default "gained 8 percent
  at 128 tokens batch 1". Served numbers agree: README example cold 219.8 vs 262.9 ms, new state 56.0 vs 60.4 ms
  (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b10_20261002T011041Z/`,
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150-accuracy_b10_20261002T012508Z/`). The 1.5 percent figure belongs
  to `accuracy_lofi_mlp` on stock configs (58.48 ms) against `accuracy` (57.76 ms), the comparison in the PLAN amendment of
  2026 Oct 2 00:55 UTC, before the program-config overrides were added.
  Why this matters: a consumer choosing between the two single-chip profiles reads a performance claim in the published
  card that is reversed in sign; the card also contradicts the report it summarizes.
  Required next step: rewrite the `p150-accuracy` description and the card performance sentence from the measured
  cells (about 9 percent slower at 128 tokens batch 1, 16 to 22 percent slower in the other cells, denominator stated),
  rebuild the manifest from `tt-model.yaml`, and push a revision 3 (or document why the Hub card stays wrong).

- P1: Card limitations quote the wrong disagreement counts for the default profile.
  Evidence: `tt-model.yaml` lines 136 to 137 and the Hub README: "the default profile reverses 8 of those 200 decisions
  (6 of them where the reference's own top-2 margin is below 0.10, 2 of the 188 confident ones)". Measured for the default
  profile (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b10_20261002T011041Z/reference_agreement.json`):
  8 disagreements, 5 with reference margin below 0.10, 3 with margin at or above 0.10 (cases
  `agent_trace_observability_000045` 0.152, `invoice_processing_000063` 0.131, `invoice_processing_000096` 0.324),
  185 of 188 = 98.40 percent. The "6 and 2" split is the build 7 `accuracy` run
  (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b7_20261001T234742Z/reference_agreement.json`: 8 disagreements,
  2 confident, 98.94 percent). The same card sentence earlier states 98.4 percent of 188, which implies 3, so the card
  is internally inconsistent.
  Why this matters: the decision-agreement gate is the port's correctness contract for the shipped policy; the card
  understates the confident disagreements by one third.
  Required next step: change to "5 below 0.10, 3 of the 188 confident ones" (or recompute from the final run) in
  `tt-model.yaml` and the Hub card.

- P2: Card and REPORT.md attribute the `accuracy` policy's README-example probabilities to the shipped package.
  Evidence: `tt-model.yaml` lines 139 to 140 ("this package gives 0.852 / 0.991 / 2.000 on the single-text path and
  0.816 / 0.993 / 2.000 when the request is embedded as one batch") and REPORT.md line 52 ("0.852 / 0.991 / 2.000 on the
  single-text path"). Source of 0.852 / 0.991:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/README.md` lines 76
  to 83, computed from `fidelity_accuracy_buckets5_tt_single.npy`, the stock `accuracy` policy, not the shipped
  `accuracy_lofi_mlp` with overrides. Source of 0.816 / 0.993: the served `accuracy` policy (build 7
  `readme_example.txt`: urgency 0.81610, billing 0.99317; identical in
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150-accuracy_b10_20261002T012508Z/readme_example.txt`). The shipped
  default profile serves urgency 0.87527, billing 0.98646, frustration 1.99997 in build 10, build 11 and the clean pull
  check (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b10_20261002T011041Z/readme_example.txt`,
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b11_20261002T013337Z/readme_example.txt`,
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_pulled_v2_20261002T013530Z/readme_example.txt`). REPORT.md line 52
  gives the served values correctly (0.875 / 0.986 / 2.000) but pairs them with the other policy's single-text values. No
  single-text value for the shipped policy exists in any artifact.
  Why this matters: the card says "this package gives" and then quotes another configuration; the batched value the card
  gives (0.816) differs from what the package returns (0.875) by 0.06 in a probability the card presents as a
  reproducibility statement.
  Required next step: compute the three README answers from
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/fidelity_accuracy_lofi_mlp_pc_tt_single.npy`
  with the `tests/decision_agreement.py` method, put the served default values (0.875 / 0.986 / 2.000) in the batched
  slot, and state the policy for each column in the card and in REPORT.md line 52.

- P2: Card quotes a stale `p150x4` cold latency.
  Evidence: `tt-model.yaml` line 128 and the Hub README: "p150x4 (four chips, tensor parallel) 175 ms cold README example
  and 34 ms new state". REPORT.md lines 187 and 274 and RUN_NOTES.md lines 115 and 150: 169.3 ms and 34.3 ms, from the
  published image (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150x4_b11_20261002T013245Z/readme_example.txt`,
  `vector_cache_table.txt`). The 175 ms is the mean of builds 6 and 9 (176.1 and 174.6 ms,
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150x4_20261001T232047Z/readme_example.txt`,
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150x4_b9_20261002T002940Z/readme_example.txt`), which ran the
  `accuracy` policy on the edited kernels. The same number differs between the card and REPORT.md.
  Why this matters: the card was frozen at commit `0cca94bc36` before the build 11 four-chip run; the published card
  describes a profile configuration (`accuracy`) that revision 2 no longer ships for `p150x4`.
  Required next step: update to 169 ms and 34 ms (build 11) with the next card revision.

- P2: RUN_NOTES.md still describes the build 9 profile set and labels the wrong build "the published build".
  Evidence: RUN_NOTES.md lines 94 to 98 ("Serve profiles" table): `p150` precision `accuracy`; `p150-fast` "6 percent
  faster, 97.3 percent decision agreement"; `p150x4` precision `accuracy`; no `p150-accuracy` row. Shipped
  (`tt_kernel_manifest.json` `serve.env` and `serve_profiles`): `p150` = `accuracy_lofi_mlp` with `CLM_PROGRAM_CONFIGS=1`
  and `CLM_SHARDED_NORM=1`; `p150-accuracy` = `accuracy`, overrides off; `p150-fast` = `bfp8_attn`; `p150x4` =
  `accuracy_lofi_mlp`. Against the shipped default, `p150-fast` is 1.7 percent slower at 128 tokens batch 1 (54.0 vs
  53.1 ms), not 6 percent faster. RUN_NOTES.md line 85: "Ninth attempt (00:26 to 00:29 UTC, `package exit 0`, the
  published build)" while line 80 says "Eleventh attempt (the published build)" and the Publish section records revision
  2 from the eleventh build.
  Why this matters: RUN_NOTES.md is the release record REPORT.md points to; a reader of its profile table gets the
  build 9 configuration.
  Required next step: replace the profile table with the manifest's four profiles and their measured deltas; change
  line 85 to "the first published build (revision 1)".

- P2: REPORT.md section 1 names a stale branch head.
  Evidence: REPORT.md line 12: "head `e72ab8b471`". `git log --oneline -12 hous/clm-v0.1-8b` in
  `/home/hous/dev/ornith-1.5-9b/tt-metal`: head is `16d4e95079`; the published commit `0cca94bc36` is three commits after
  `e72ab8b471`. The branch point `b725040266` is confirmed (direct parent of the first autoport commit `418c0e85bf`).
  Why this matters: a reader checking out "the head" gets neither the published code nor the latest release notes.
  Required next step: state the published commit `0cca94bc36` and the current head `16d4e95079` (or drop the head).

- P2: REPORT.md still carries relative file references.
  Evidence: REPORT.md line 107 "`tests/bench_encoder.py`" (bare relative path); lines 310 to 314 (producer block) run
  `pytest $A/tests/test_host_engine.py ...`, `python $A/tests/run_fidelity.py ...` and similar after a `cd`, which are
  relative paths in commands; line 249 "upstream `examples/t_rex/results/clm_realtime.json`" is a path in the upstream
  repository and is followed by the absolute local copy, which is acceptable. No em or en dash is present (0 matches in
  REPORT.md, RUN_NOTES.md and `tt-model.yaml`).
  Why this matters: the user's house rule requires absolute paths in every file reference, including commands.
  Required next step: write
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tests/bench_encoder.py` on line 107
  and absolute paths in the producer block.

## Other Concerns

- REPORT.md section 5, row "128 | accuracy, stock configs": 57.7 ms and 161.6 ms are the 32-real-token rows of
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/perf_summary_accuracy_buckets5.json`
  (57.70 and 161.65 ms, padded to 128), while the 6,317 tokens/s in the same row is the 128-real-token row (162.11 ms). The
  128-real-token batch-1 value is 57.76 ms, which REPORT.md section 4 and the sweep README print as 57.8 ms. The default
  profile row uses the 128-real-token rows (53.14 and 135.70 ms). The same cell is 57.7 in section 5 and 57.8 in section
  4. The percentage gains quoted (8, 16, 18, 16, 17 percent) hold with either row.
- REPORT.md line 140: "the bound is 2.8 percent below the measurement". `perf_summary_accuracy_buckets5.json` records
  `gap_to_lower_bound_pct` 2.9 (32-token row) and 3.0 (128-token row); (57.70 minus 56.05) / 57.70 = 2.86 percent. The
  2.8 comes from the older 57.6 ms bench in
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_decoder/README.md`.
- REPORT.md line 243 says the build 6 Typed Decisions case cost "1,302 ms" while the table on line 222 says "1,301 ms";
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_final_20261001T232132Z/typed_decisions/score.json` has client
  p50 1300.8 ms (per workflow 1301.0 to 1301.7 ms).
- REPORT.md line 241: "300 of the 400 cases send five state texts of 130 to 300 tokens each". From
  `usage.input_tokens / 5` per case (cases 2 to 100 of each workflow, build 10 `results.jsonl`): invoice_processing 269
  to 366, security_incidents 170 to 303, customer_service 81 to 510 tokens per text. The upper end is understated; the
  bucket argument still holds because every text fits the 512 bucket (customer_service p50 517 ms, invoice 519 ms,
  security 258 ms).
- `responses_round_2.md` line 59: "p150-fast served once from the published image (RUN_NOTES)". RUN_NOTES.md and
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150-fast_b10_20261002T012408Z/` show the `p150-fast` serve came
  from build 10 image `500871f7c383` (same code sha256 `e9ae8402...`, different image), not from the published
  `6cf6949ed327`. RUN_NOTES.md states this correctly; the response document overstates.
- `responses_round_2.md` line 58: "near-tie sentence rewritten with the measured counts". The counts written into the
  card are the build 7 counts (see Required Work).
- `responses_round_2.md` line 45: "the consolidated work log is recorded as a deviation". No such record was found in
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` or `/home/hous/dev/clm-v0.1-8B/STATUS.md` (PLAN.md line 84 records a different
  deviation, one review per stage); the stage 4 to 8 log is one file,
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/work_log.md`.
- RUN_NOTES.md line 11: "the manifest's `pushed: false` refers to the upstream remote". The published manifest says
  `pushed: true`, `remote: git@github.com:housTT/tt-metal.git`, `branch: HEAD`; the sentence applies to builds 1 to 9
  only and should say so.
- The tool-generated Hub README (`/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/README.md`, identical on the
  Hub) contains six em dashes in tt-model-manager boilerplate and the sentence "Runs on **p150** or **p150** or **p150**
  or **p150x4**" (per-profile hardware not deduplicated). Not in the author's card text; a tt-model-manager template
  issue worth reporting there.
- The Hub tags the base-model relation as `base_model:finetune:Contrastive-LM/CLM-v0.1-8B` (model_info tags) while the
  weights are the unmodified upstream weights; RUN_NOTES.md line 154 discloses this.
- The card's "throughput up to 7.7k tokens/s at batch 32" rounds the served maximum 7,753 tokens/s (512 tokens x 32)
  downward; acceptable.

## Hard-Check Gaps

- No artifact holds the shipped policy's single-text README-example probabilities (only `accuracy` policy vectors were
  scored); a 30-second host-only computation closes it.
- Boot-time claims "healthy 50 s after start" (p150x4, build 11) and "healthy 30 s after start" (pulled revision 2) are
  not pinned by a log line that marks the `tt-model serve` start. `/home/hous/dev/clm-v0.1-8B/logs/container_build11_p150x4.log`
  shows ttnn init at 01:32:08.4 and "Application startup complete" after a 16.3 s warmup at about 01:32:39 (31 s); the
  pulled check's serve log was written at 01:35:04.8 and the first evaluation call at 01:35:30.5 (26 s). Consistent in
  magnitude, not exactly verifiable.
- The kernel-source check was made on the staged OCI layout, whose index digest `sha256:6cf6949ed327...` equals the
  manifest's `image_digest` and whose 28 blobs (963.67 MB) match the Hub listing by count and size. The Hub blobs
  themselves were not downloaded and re-hashed.
- REPORT.md sections 3 and 6 were spot-checked only (section 3 fidelity, head and batch rows against
  `fidelity_accuracy.json`; section 6 against the multichip README and `fidelity_accuracy_1x4.json`): all matched.
- REPORT.md line 260 ("the p150-fast run shared the host CPU with two review subagents") is not verifiable from the
  artifacts.

## Anomaly Ledger

- Observed anomaly: the card says `p150-accuracy` is 1.5 percent faster than `p150` at 128 tokens batch 1.
  Evidence: `tt-model.yaml` lines 65 and 125 to 127; `bench_accuracy_lofi_mlp_pc.json` 53.14 ms vs `bench_accuracy_buckets5.json` 57.76 ms; served 219.8 vs 262.9 ms.
  Affected path: published card and `p150-accuracy` profile description (Hub revision 2).
  Control or comparison: REPORT.md section 5 line 128 (8 percent gain for the default) and the served build 10 runs.
  Likely subsystem: documentation carried over from the pre-override comparison (PLAN amendment 00:55 UTC).
  Investigation performed: recomputed both cells from the bench JSON files and the served result files.
  Resolution: more-work-needed.

- Observed anomaly: card disagreement split "6 near ties, 2 confident" does not match its own 98.4 percent.
  Evidence: `package_p150_b10_20261002T011041Z/reference_agreement.json` (8 / 5 / 3); `package_p150_b7_20261001T234742Z/reference_agreement.json` (8 / 6 / 2).
  Affected path: card limitations text.
  Control or comparison: build 7 counts reproduce the card's sentence exactly.
  Likely subsystem: card text written from the build 7 evaluation and not refreshed for build 10.
  Investigation performed: counted disagreements by `ref_top2_margin` for builds 5, 6, 7 and 10.
  Resolution: more-work-needed.

- Observed anomaly: README-example probabilities in the card and REPORT.md line 52 come from the `accuracy` policy.
  Evidence: `doc/full_model/README.md` lines 76 to 83 (source vectors `fidelity_accuracy_buckets5_tt_single.npy`); served default 0.875 / 0.986 in builds 10 and 11 and the pulled check; served `accuracy` 0.816 / 0.993 in build 7 and `p150-accuracy`.
  Affected path: card limitations, REPORT.md section 2.
  Control or comparison: three independent served runs of the default profile agree to five digits.
  Likely subsystem: documentation.
  Investigation performed: traced each quoted value to its result file.
  Resolution: more-work-needed.

- Observed anomaly: card `p150x4` cold latency 175 ms vs 169.3 ms measured from the published image.
  Evidence: `package_p150x4_20261001T232047Z` 176.1 ms, `package_p150x4_b9_20261002T002940Z` 174.6 ms, `package_p150x4_b11_20261002T013245Z` 169.3 ms.
  Affected path: card performance text.
  Control or comparison: REPORT.md and RUN_NOTES.md both carry 169.3 ms.
  Likely subsystem: card frozen at commit `0cca94bc36` before the build 11 run; the `p150x4` policy also changed from `accuracy` to `accuracy_lofi_mlp`.
  Investigation performed: read the three result directories.
  Resolution: more-work-needed.

- Observed anomaly: RUN_NOTES.md profile table and "the published build" label describe build 9.
  Evidence: RUN_NOTES.md lines 85 and 94 to 98 vs `tt_kernel_manifest.json` `serve_profiles`.
  Affected path: release record.
  Control or comparison: the manifest and `tt-model.yaml`.
  Likely subsystem: documentation not refreshed after the second review round.
  Investigation performed: side-by-side comparison.
  Resolution: more-work-needed.

- Observed anomaly: REPORT.md names head `e72ab8b471`; the branch head is `16d4e95079`.
  Evidence: `git log --oneline -12 hous/clm-v0.1-8b`; `git worktree list`.
  Affected path: REPORT.md section 1.
  Control or comparison: manifest `built.tt_metal.sha` `0cca94bc36...`.
  Likely subsystem: documentation.
  Investigation performed: git log and worktree inspection.
  Resolution: more-work-needed.

- Observed anomaly: REPORT.md section 5 mixes 32-real-token latencies with the 128-real-token tokens/s in one row, and the same cell reads 57.7 in section 5 and 57.8 in section 4.
  Evidence: `perf_summary_accuracy_buckets5.json` rows (tokens 32 and 128, padded 128).
  Affected path: REPORT.md section 5 table.
  Control or comparison: `doc/optimized_full_model/README.md` prints both variants in two tables (lines 61 and 119).
  Likely subsystem: table assembled from two rows of the same bench.
  Investigation performed: matched every cell to the JSON rows.
  Resolution: controlled (0.1 to 0.5 ms; the stated gains hold); fix the row label or the value for consistency.

- Observed anomaly: uncommitted Ornith kernel edits were shipped in builds 1 to 10 (review R).
  Evidence: `git diff HEAD` in `/home/hous/dev/ornith-1.5-9b/tt-metal` still shows the two edited files (+3 and +8/-2 lines); the staged image layers `afbafae8ed91...` and `171b9683e700...` contain `opt/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` (sha256 `a9a00cc34beb...`) and `opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` (sha256 `23cba51e0a2c...`), equal to `git show 0cca94bc36:<path>`; zero occurrences of `noc_clear_packet_tags(noc_index)` and `has_backward_connection` in the image copies; modes `-rw-rw-r--`.
  Affected path: `p150x4` profile (fabric router and all-gather writer compile only on the multi-chip path).
  Control or comparison: 1x4 fidelity and agreement with the stock kernels pass (`fidelity_accuracy_lofi_mlp_1x4_stock_kernels.json`: mean 0.99908, min 0.99588; `agreement_accuracy_lofi_mlp_1x4_stock_kernels.json`: 194 of 200, 187 of 188); build 11 `p150x4` boots and serves (169.3 ms cold, 34.3 ms new state, first 100 cases 0.296 at 106 ms).
  Likely subsystem: build provenance (source tree selection).
  Investigation performed: extracted and hashed both files from the OCI layers; compared with the committed blobs.
  Resolution: fixed (image verified clean; manifest `dirty: false`, `sha 0cca94bc36db77ac13aa28b280f578e76c4c8e27`, `pushed: true`).

## Scope Inspected

- Goal/skill paths:
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/tmp/claude-1002/-home-hous-dev-clm-v0-1-8B/0082bb62-0b82-4079-9fa4-87d59d66a985/scratchpad/reviews/common.txt`;
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` (amendments of 2026 Oct 1 23:45 and 2026 Oct 2 00:50 and 00:55 UTC).
- Artifact paths:
  `/home/hous/dev/clm-v0.1-8B/REPORT.md` (re-read after it changed on disk during the review; the current copy uses
  absolute paths where the earlier copy had `.../` forms);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/release/RUN_NOTES.md`;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt-model.yaml`;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/review/responses_round_2.md`;
  `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/tt_kernel_manifest.json`, `README.md`, `image/index.json`,
  `image/manifest.json`, `image/blobs/sha256/*` (22 layers);
  Hub revision 2 via `huggingface_hub` (model_info with file metadata, `README.md` and `tt_kernel_manifest.json`
  downloaded to the scratchpad; both byte-identical to the staged files; 124 files, 1,040,748,044 bytes = 1,041 MB; 28
  image blobs = 963.67 MB; tags blackhole, p150, p150x4, tt-dit-server, tt-model-cache, tt-model-container,
  text-ranking, license apache-2.0, base_model Contrastive-LM/CLM-v0.1-8B; commits `3da3cc872dc3` at 00:31 UTC and
  `2eb9cf5199d2` at 01:34 UTC on `main`);
  `/home/hous/dev/clm-v0.1-8B/logs/package_build.log` (build 10 at 01:08 to 01:10 UTC from `abd3e02365`, tt-model reports
  "dirty tree" because the manifest still named the main checkout; build 11 at 01:29 to 01:32 UTC from `0cca94bc36`,
  "dirty files: 0", "963.7 MB"), `tt_model_push_v2.log`, `tt_model_pull_v2.log`, `evals_pulled_v2.log`,
  `pulled_v2_rank.txt`, `serve_pulled_v2.log`, `container_build10_p150.log`, `container_build10_p150-accuracy.log`,
  `container_build10_p150-fast.log`, `container_build11_p150.log`, `container_build11_p150x4.log`;
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b10_20261002T011041Z/` (every table, `reference_agreement.json`,
  `typed_decisions/score.json`, `results.jsonl`, `meta.json`), `package_p150x4_b11_20261002T013245Z/`,
  `package_p150_b11_20261002T013337Z/`, `package_pulled_v2_20261002T013530Z/`, `package_p150-fast_b10_20261002T012408Z/`,
  `package_p150-accuracy_b10_20261002T012508Z/`, `package_p150_b7_20261001T234742Z/`,
  `package_p150_final_20261001T232132Z/`, `package_p150_20261001T224031Z/`, `package_p150x4_20261001T232047Z/`,
  `package_p150x4_b9_20261002T002940Z/`, `package_pulled_clean_p150_20261002T003519Z/`;
  `/home/hous/dev/clm-v0.1-8B/evals/trex/REFERENCE.md`, `reference/clm_realtime.json`, `results/20261001T225213Z/`,
  `results/20261001T233448Z/`, `results/20261001T235652Z/`, `results/20261002T011822Z/` (`clm_realtime.json`,
  `run_meta.json`);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/` (`README.md`,
  `sweep_results.json`, `sweep_results.csv`, `selected_precision_config.json`, `fidelity_accuracy.json`,
  `bench_performance.json`, `infeasible_bf16_all.json`, all `agreement_*.json`);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/`
  (`README.md`, `bench_accuracy_lofi_mlp_pc.json`, `bench_accuracy_buckets5.json`, `perf_summary_accuracy_buckets5.json`,
  `perf_summary.json`, `fidelity_accuracy_lofi_mlp_pc.json`, `agreement_accuracy_lofi_mlp_pc.json`,
  `replay_trace_check_accuracy_lofi_mlp_pc.json`);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_decoder/` (`README.md`,
  `program_config_experiment_{128,256,512,1024,2048}.json`, `geometry_experiment*.json`);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/multichip_decoder/` (`README.md`,
  `fidelity_accuracy_lofi_mlp_1x4_stock_kernels.json`, `agreement_accuracy_lofi_mlp_1x4_stock_kernels.json`);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/README.md`,
  `doc/fused_decoder/README.md`;
  `/home/hous/dev/clm-v0.1-8B/evals/vendor/CLM/README.md` (lines 86 to 90 and 467 to 473),
  `/home/hous/dev/clm-v0.1-8B/evals/typed_decisions/SCHEMA.md`, `/home/hous/dev/clm-v0.1-8B/evals/typed_decisions/data/README.md`.
- Code paths:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt/encoder.py` (lines 198 to 204,
  230 to 237, 248, 297, 359 to 365), `tests/perf_summary.py` (line 41), `tests/program_config_experiment.py` (exists);
  `/home/hous/dev/ornith-1.5-9b/tt-metal` git history (`hous/clm-v0.1-8b` head `16d4e95079`, `hous/clm-v0.1-8b-release`
  at `0cca94bc36`, worktree `/home/hous/dev/clm-v0.1-8B/worktree/tt-metal` detached at `0cca94bc36`, clean; `git diff
  HEAD` of the two kernel files in the main checkout: +3 and +8/-2 lines).
- Commands run: `cat`, `sed`, `grep`, `ls`, `find`, `file`, `git log`, `git rev-parse`, `git worktree list`,
  `git status --short`, `git diff --stat HEAD`, `git show <commit>:<path> | sha256sum`, `git merge-base --is-ancestor`,
  `tar -tzf` and `tar -xzOf` over the 22 OCI layers, `sha256sum`, and small read-only Python scripts with
  `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` (JSON summarization of the T-Rex, agreement and
  Typed Decisions result files; `huggingface_hub` model_info and two file downloads). No device, server, container,
  `pkill` or `pgrep -f` was used; no implementation file was modified.
- Figures verified as matching their result files (stated = measured): REPORT.md section 1 (revisions `e939398d`,
  `b968826d`, `bb42c6c5`, branch point `b725040266`, image tag and digest, code sha256 `e9ae8402b633...`, commit
  `0cca94bc36`, `dirty: false`, 14 verify assertions, four profiles, first publication 00:31 UTC image `9372e4d3d3c4`);
  section 2 (every row: 98 tokens 219.8 ms, 0 tokens 0.1 ms with 0.8 ms client, 56.0 / 56.0, 0.1 / 0.1, embeddings
  56.6 / 149.2 / 590 and 82.9 / 530 / 2,113 and 148.5 / 1,087 / 4,338 ms with the matching tokens/s, Typed Decisions
  0.345 / 2.039 / 0.628 / 0.503, 258 / 520 ms, T-Rex 3 of 5 / 587 / 2,195 vs 5 of 5 / 697 / 3,342, 16.4 / 1.3 vs
  16.5 / 2.6 ms, agreement 96.0 / 98.4 percent, cosine 0.99916 / 0.99599; RTX 4090 values 38 tokens 58.1 ms, 28.0 /
  28.1, 0.6 / 0.7, 106 tokens cold, 0.41022 / 0.93878 / 1.98386 from the CLM README); section 4 (every cell of all nine
  policy rows against `sweep_results.json`, gates, 188 confident decisions, 7 percent from the overrides, 98.9 to 97.3
  for `accuracy` with overrides); section 5 (all five default rows and tokens/s against `bench_accuracy_lofi_mlp_pc.json`;
  `p150-fast` rows against `perf_summary.json`; the 512 / 1024 / 2048 `accuracy` rows; gains 8 / 16 / 18 / 16 / 17
  percent; 1.557 and 4.519 ms, 56.1 and 162.7 ms, 4.5 percent; eager vs trace 57.45 / 57.65 and 170.21 / 170.45 ms;
  per-op shares and core counts; 72 and 94 vs 170 ms; load 5.3 s, warmup 17.6 s one chip, 26.8 s four chips; rejected
  experiments PCC 0.00 to 0.03, L1 overflow, circular-buffer error, retraction notes naming `lru_cache`); section 7
  (README example rows for all four profiles, vector-cache table, p150-accuracy 60.4, p150-fast 56.4, p150x4 34.3 ms,
  the full 5 x 4 embeddings latency and tokens/s tables, Typed Decisions rows for builds 10, 7, 6 and 5, the CPU
  reference 0.370, the leaderboard rows from the dataset card copy, per-workflow and per-type numbers, 0.232 vs 0.310,
  0.355 to 0.360, 252 / 511 ms host bench, all four T-Rex rows and the authors' row, 124 files / 1,041 MB, 20 s pull,
  Moon 0.994, 219.2 ms / 98 tokens, 55.5 ms, 169.3 and 34.3 ms); RUN_NOTES.md (host, build 6 to 9 commit range, build
  10 and 11 provenance lines, image and digest and code sha256 of the published build, 963.7 MB in 28 blobs, 23.0 s
  upload, the verification table including load times 5.2 / 5.5 / 5.9 / 5.5 s, first-100-case numbers 0.294 at 167 ms,
  0.292 at 147 ms, 0.296 at 106 ms, build 11 `p150` 220.5 / 56.1 / 140 ms, builds 6 and 9 `p150x4` 176.1 / 174.6 and
  33.6 / 34.1 ms, revision 1 pull check 260.9 / 60.5 ms, served agreement 93.0 / 95.7 (build 5), 95.5 / 98.4 (build 6),
  96.0 / 98.9 (build 7), stock-kernel 1x4 fidelity and agreement 97.0 / 99.5, Hub tags); card (220 ms, 0.1 ms, 56 ms,
  58.1 ms, 28.0 and 0.6 ms, embeddings 56.6 / 83 / 149 / 272 and 149 / 530 ms, 7.7k tokens/s, 0.9992 / 0.9960, 96.0 and
  98.4 percent of 188, 0.345 / 0.364 / 0.370 / 0.355, KL 2.04, Brier 0.63, 258 ms, T-Rex 3 of 5 and 2,195 vs 5 of 5 and
  3,342, p150-accuracy 263 / 60 ms, p150-fast 234 / 56 ms and 97.3 / 95.7 percent, 4 to 5 single-vs-batched argmax
  changes per `doc/full_model/README.md` line 68, 1x2 fabric timeout, pinned-memory workaround).
- `responses_round_2.md` claims found in the named files: A2 P1 (experiment script, five JSON files, README table,
  `_install_program_configs` and `_install_sharded_norms` with env toggles, end-to-end validation table); A2 P2 items
  (perf summaries with 1.557 / 4.519, `perf_summary.py` default, `retracted` notes naming `lru_cache` in both
  instance-patch JSON files, HF bf16 layer reference label, fused-decoder typecast and fill-cache row with 0.7 to 1.0
  percent and "unconditional in attention.py"); C2 P2 items (stage 6 README agreement rows 191 of 200 and 186 of 188,
  qualitative substitute table, PLAN amendments for rows 4, 6 and 8, `bench_performance.json` timestamp 2026 Oct 2
  00:50 UTC, `accuracy_lofi_mlp` fidelity, agreement and bench files, selection rule amendment, every listed field in
  `selected_precision_config.json`, bf16_all label, `_select_trace_lens` multiple-of-128 check at `encoder.py` line 365,
  `CLM_MAX_TOKENS` vs `CLM_MAX_SEQ_LEN` check at line 233, twophase_limit96 description, stage 4 gate 0.99929 / 0.99547);
  R items (RUN_NOTES correction, stock-kernel 1x4 run, clean worktree build, 58.1 ms labelling, no "4.5 times" text
  remains, host-bench attribution, p150-fast dual agreement numbers, footnotes (a) and (b), Hub tag list). Not found or
  overstated: the consolidated work log deviation record, "served from the published image" for p150-fast, and the
  near-tie counts (see Other Concerns and Required Work).

## Residual Risk

- Until a revision 3 is pushed, the public card on the Hub carries a reversed profile-speed claim, stale decision counts
  and probabilities, and a stale four-chip latency; the Hub README is generated from the manifest, so `tt-model.yaml`,
  the staged manifest and the Hub must change together.
- The served-package agreement for the shipped default is 98.4 percent on confident decisions (185 of 188), below the
  98.9 percent of the standalone sweep that selected the policy and only 0.4 points above the 98 percent gate; one more
  flipped decision (0.53 points) would put the served number at the gate. The card quotes 98.4 percent correctly, but
  the gate margin from the served path is thin and should be stated.
- The `p150x4` profile's only measurements on the published image are the README example, the vector-cache rows and the
  first 100 Typed Decisions cases; its decision-agreement and fidelity evidence (97.0 / 99.5 percent) comes from a host
  run with the committed kernels, not from the served image.
- The Hub relation tag "finetune" misdescribes an unmodified-weights package; a consumer filtering by that tag will be
  misled unless the card sentence is read.
