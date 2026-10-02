# Stage Review

Review R: release and evaluation review. Scope: the plan's stages 9 to 11 substitutes (FastAPI serving in place of
vLLM, the tt-model v5.1 container in place of the TTI release, evaluation through the served package) and the final
report. Target: Contrastive-LM/CLM-v0.1-8B (frozen Qwen3-8B encoder, last-token pooling after the final RMSNorm, L2
normalized, two MLP heads). Autoport:
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b`, branch `hous/clm-v0.1-8b`, live
worktree at HEAD `e72ab8b471` (on github.com/housTT/tt-metal). Published package: Hub repo `tt-hous/clm-v0.1-8b-p150`,
revision `3da3cc872dc36c4d738bbadd42f921d854ddae04`. Reviewer mode, read-only, 2026 Oct 2. Times are UTC as written
in the logs.

Acronyms: API = application programming interface. CCL = collective communication library. ECE = expected
calibration error. GPU = graphics processing unit. HF = Hugging Face. JIT = just in time (kernel compilation at run
time). KL = Kullback-Leibler divergence. MLP = multilayer perceptron. OCI = Open Container Initiative (the image
layout). p50 / p95 = 50th / 95th percentile. RMSNorm = root mean square normalization. SHA = secure hash algorithm.
TP = tensor parallel. TTI = the tt-inference-server release workflow. TTNN = the tt-metal tensor library. vLLM = the
upstream LLM serving engine. BFP8 = 8-bit block floating point. HiFi2 / HiFi4 / LoFi = math fidelity modes. RTX 4090 =
the Nvidia consumer GPU the CLM authors measured on.

Verdict: more-work-needed

Summary. The published artifact is the evaluated one: the Hub `README.md`, `tt_kernel_manifest.json`,
`requirements.lock`, `image/index.json` and `image/manifest.json` are byte-identical to the staged package; the Hub
`code/` tree has the same 89 files and 77,039,427 bytes; the manifest's image digest, code sha256, tt-metal commit,
version string and dirty flag match the build log and the git history; the clean-pull numbers match the build 7
evaluation. Every number I re-derived from REPORT.md sections 2, 4 and 7 and from the card matches its result file
(list under Scope Inspected). The Typed Decisions scorer is described honestly: KL and Brier reproduce the dataset's
Uniform row, ECE does not, and the report and card say so. What blocks a clean pass: the release notes make a
verifiably false statement about the uncommitted tt-metal edits, which the published image ships on the multi-chip
serving path (P1); the card and REPORT section 2 misread the CLM README's 58.1 ms reference; the card attributes
host-bench and standalone-sweep numbers to the served package and quotes the better of two p150-fast agreement
measurements while that profile was never served from the published image; RUN_NOTES still names the build 6
commit as the final build. (REPORT.md was updated on disk at 00:48 UTC while this review ran; its section 1 head
commit and section 5 capture count were corrected in that update and are not listed below.)

## Required Work

- P1: The release notes misstate the uncommitted tt-metal edits; the published image ships two patched kernel
  sources on the multi-chip serving path, and the pushed branch does not carry them
  Evidence:
  `git status` in `/home/hous/dev/ornith-1.5-9b/tt-metal` lists 21 modified tracked files. Two are kernel sources:
  `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` (+3 lines: `noc_clear_packet_tags(noc_index)`
  added to `teardown`) and
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` (+8 / -2: the
  direction connection pointer becomes null when `has_backward_connection()` or `has_forward_connection()` is false
  instead of being dereferenced unconditionally). A tar scan of the 22 layers of the staged OCI image
  `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/image` (digest `sha256:9372e4d3d3c4...`, identical to the
  Hub copy) finds `opt/tt-metal/tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` in layer 11 and
  `opt/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`
  in layer 12, each with the sha256 of the dirty working-tree file (`6ae17948...`, `9c9e79d9...`), not of the
  committed `HEAD` version (`a9a00cc3...`, `23cba51e...`). Both files are on the `p150x4` path: the fabric router is
  JIT-compiled at `FABRIC_1D` initialization (the build 7 `p150x4` boot failed on exactly this file,
  `/home/hous/dev/clm-v0.1-8B/logs/container_x4_debug.log` line 23), and the shipped code calls
  `ttnn.experimental.all_gather_async`, which compiles `minimal_default_writer.cpp`, in
  `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/code/models/tt_transformers/tt/distributed_norm.py` (lines
  122, 151), `attention.py` (928, 1296), `mlp.py` (322) and `ccl.py` (205, 253, 326, 341, 446).
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/release/RUN_NOTES.md`,
  "Host and tree", says the dirty files are "`AGENTS.md`, two kernel files" and that "none of them is on the CLM
  serving path (the allowlist ships only `models/common` files, `models/tt_transformers/tt` and this autoport)". The
  allowlist governs `code/`; the runtime image also copies the builder's `tt_metal/` and `ttnn/` trees
  (`/home/hous/.cache/tt-model/build/clm-v0.1-8b-p150.log` lines 3317 and 3320), which is where the two files live.
  The stage 4 host runs (`fidelity_accuracy_1x4`, `bench_accuracy_1x4`) and the 1x2 attempt ran in this same working
  tree, so every 1x4 number and the 1x2 "Fabric Router Sync: Timeout" were produced under the patched kernels.
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/multichip_decoder/README.md`
  ("In-container 1x4 serving") mentions the edits only as a file-mode problem. `git ls-remote` shows
  `hous/clm-v0.1-8b` at `e72ab8b471` on github.com/housTT/tt-metal; that commit carries neither edit. The published
  card's Provenance row reads "a local checkout, commit not published (dirty tree, the image includes uncommitted
  changes)".
  Why this matters: PLAN.md row 11 makes `RUN_NOTES.md` the record of commands, SHAs and digests for the shipped
  bits, and its statement is contradicted by the image contents. Anyone who rebuilds `p150x4` from the published
  branch gets different fabric and CCL kernels than the evaluated image, and no artifact records whether the stock
  kernels work on this path. The 1x2 fabric timeout is an uncontrolled anomaly next to a modified fabric router.
  The single-chip profiles compile neither kernel and are unaffected.
  Required next step: (1) Correct RUN_NOTES: name both files, their diffs and the layers that carry them; state that
  the `p150x4` profile and all 1x4 evidence ran on these edits; list the full dirty set and mark which of it is in
  the image (only these two files; `models/` in the runtime image comes from `code/`). (2) Make the image
  reproducible from the branch: commit the two edits on `hous/clm-v0.1-8b` with their Ornith origin recorded, or save
  them as a patch under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/release/`
  that `/home/hous/dev/clm-v0.1-8B/bin/package-build.sh` applies, and point to it from the card description (the
  generator's Provenance row cannot carry it). (3) Record whether `p150x4` boots and answers the README example with
  the committed kernels, or state that it is untested without the edits. Step 3 needs hardware and is outside this
  review.

- P2: The card and REPORT section 2 misread the CLM README's RTX 4090 latency and mix its before- and after-cache
  columns
  Evidence: `/home/hous/dev/clm-v0.1-8B/evals/vendor/CLM/README.md` line 90:
  `print(r.usage.input_tokens, r.latency_ms)  # 38 58.1  (106 tokens on a cold cache: option texts are embedded once)`.
  The 58.1 ms call embedded 38 tokens (the state texts; the option vectors were cached). The README publishes no
  latency for the 106-token cold call. The card (`tt-model.yaml` `card.performance`, Hub README "Expected
  performance") says "the CLM README reports 58.1 ms cold". `/home/hous/dev/clm-v0.1-8B/REPORT.md` section 2 puts
  `38 58.1` in the row "README example, warm cache (state and options cached)" against the p150's `0 0.1` (zero tokens
  embedded) and derives "every cold number here is 2 to 4.5 times the published one" (262.4 / 58.1). Section 7's
  table ("106 cold, 38 warm | 58.1 ms") is the only consistent reading. The TT measurement that matches "state
  embedded, options cached" is the vector-cache row "new state every call": 60.2 ms server p50 at 54.75 tokens per
  call (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b7_20261001T234742Z/vector_cache_table.txt`), which
  is parity with 58.1 ms, not 4.5x. The card also quotes "28.6 / 1.7 ms for the same table" (the README's
  before-cache column, lines 471 to 473) while REPORT section 2 quotes 28.0 / 28.1 and 0.6 / 0.7 (the after-cache
  column); the served package has the vector arena, so the after-cache column is the comparable one.
  Why this matters: the card is the consumer-facing comparison. It understates the port on the only like-for-like
  row and presents a ratio the reference does not support. Card text is outside the code hash (builds 8 and 9 kept
  `c8730871...`), so the fix is a card-only rebuild and republish.
  Required next step: in the card and REPORT section 2, label 58.1 ms as "38 tokens embedded, options cached" and set
  it against the p150 new-state row (60.2 ms); drop "cold" and the "4.5 times" derivation or state that the README
  publishes no cold latency; use one 4090 cache column (after) in both documents.

- P2: The card attributes host-bench and standalone-sweep numbers to the served package, quotes the better of two
  p150-fast agreement measurements, and the p150-fast profile was never served from the published image
  Evidence: card `performance`: "Measured on this package served ... one text up to 128 tokens 57.8 ms, 256 tokens
  72 ms, 512 tokens 94 ms, 1024 tokens 170 ms, 2048 tokens 322 ms; eight 128-token texts 162 ms, eight 512-token
  texts 625 ms; throughput saturates near 6.3 to 6.7k tokens/s." These are the rows of
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/bench_accuracy_buckets5.json`
  (57.76, 72.05, 93.8, 170.4, 321.68, 162.11, 624.53 ms; 6,300 to 6,678 tokens/s), a host process
  (`tests/bench_encoder.py`), not the served container. The served equivalents (server-side p50 in
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_b7_20261001T234742Z/embeddings_table.txt`) are 58.6, 95.2,
  172.2, 324.2 ms at batch 1 and 164.8, 629.5 ms at batch 8; the served throughput tops out at 6,401 tokens/s, so
  "6.7k" has no served evidence. Card and manifest text for `p150-fast`: "6 percent faster than the default at 97.3
  percent decision agreement". 97.3 percent is the standalone single-text sweep
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/datatype_sweep/sweep_results.json`,
  `decision_agreement_margin_0p10` 0.9734 for `bfp8_attn`). The served p150-fast package measured 93.0 percent overall
  and 95.7 percent on confident decisions
  (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_20261001T224031Z/reference_agreement.json`), which RUN_NOTES
  records; for the default profile the card uses the served figures (96.0 / 98.9). That served run used image
  `aa6f0847aa7a` (build 5, three buckets, a different code sha256). No serve of `p150-fast` exists for build 7 or for
  the published image `9372e4d3d3c4` (RUN_NOTES "Verification" table; no fast-profile result directory after 22:40
  UTC under `/home/hous/dev/clm-v0.1-8B/evals/results/`). REPORT section 1 says "both profiles were re-smoked on build
  9", which covers two of three. Card `limitations`: "flips about 2 percent of near-tie decisions". The agreement
  artifact gives 8 of 200 decisions flipped against the fp32 reference (4.0 percent); 6 of the 8 have a reference
  top-2 margin below 0.10, so 6 of the 12 near-tie decisions flip (50 percent) and 2 of the 188 confident ones (1.1
  percent); review C's batch-composition measurement was 4 to 5 of 200 decisions (2 to 2.5 percent of all). The
  sentence matches none of these as written.
  Why this matters: these are the card's accuracy and performance claims (brief points 1 and 5). The encoder-row
  deviations are 1 to 2 percent, but the p150-fast agreement claim and the near-tie sentence are statements a
  consumer uses to choose a profile, and the fast profile on the shipped image has no serve evidence at all.
  Required next step: (a) attribute the encoder rows to the host bench or replace them with the served rows and cap
  the throughput claim at the served maximum; (b) give `p150-fast` the served figures (93.0 / 95.7 percent on
  `aa6f0847aa7a`, three buckets) or state that 97.3 percent is the standalone single-text sweep; (c) rewrite the
  near-tie sentence with the measured numbers (8 of 200 flips, 6 of them below margin 0.10; 4 to 5 of 200 between
  batch compositions); (d) serve `p150-fast` from the published image once (health, README example, new-state row)
  and record it in RUN_NOTES, or say in the card that the profile was evaluated on an earlier build. Step (d) needs
  hardware and is outside this review.

- P2: Stale commit and file-count references in RUN_NOTES.md
  Evidence: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/release/RUN_NOTES.md`
  line 10, "Host and tree": "final build from commit `a3df3fd2ee`", while the same file's "Ninth attempt" bullet, the
  manifest (`built.tt_metal.sha`) and the build log say `2c710b113b`; `a3df3fd2ee` is the build 6 commit, five
  commits behind the branch head `e72ab8b471`. Line 127, "Publish": "image/ (OCI index, manifest and 31 blobs)"; the
  Hub holds 28 blobs plus `index.json`, `manifest.json` and `oci-layout` (31 files), and the push log says "28
  content-addressed blobs". For the record, `/home/hous/dev/clm-v0.1-8B/REPORT.md` had the same stale head commit
  ("head `a3df3fd2ee`") and "nine trace captures: 6.2 s" when this review started; both were corrected in the 00:48
  UTC update (now "head `e72ab8b471`" and "fifteen trace captures: 17.6 s").
  Why this matters: the brief asks that corrections be consistent everywhere, and the "Host and tree" line is the
  release stage's own statement of what was built.
  Required next step: replace `a3df3fd2ee` with `2c710b113b` on line 10 (and name `e72ab8b471` as the branch head at
  publication); change "31 blobs" to "28 blobs plus the OCI index, manifest and layout file".

## Other Concerns
- Publication preceded the earlier-stage re-review. `tt-model push` ran at 00:31 UTC; review A2
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/review/review_A2_stages_1_3.md`,
  written 00:37 UTC, untracked) returned `more-work-needed` with a P1 on the stage 3 geometry conclusion. PLAN.md
  section 4 requires `clean-pass` before a stage closes. The card's `risks` text ("32 of 110 cores", "about 2x an RTX
  4090") rests on that audit and may need an update when A2 is resolved.
- Hub README generator artifacts (tt-model 0.1.0): "Runs on **p150** or **p150** or **p150x4**"; the Provenance row
  "commit not published" although the branch is public on github.com/housTT/tt-metal; "port 20000" in the Quickstart
  while the manifest serves 8700 (tt-model maps the host port, consistent with REPORT section 9). The card
  description is the only author-controlled field that could carry the fork URL and commit.
- Typed Decisions table (REPORT section 7): the TT p50 (312 ms, client on the same host) sits beside Jev's 710 ms,
  which the dataset card defines as hosted end-to-end p50 from a remote client; `SCHEMA.md` ambiguity 4 records this,
  the report does not. The leaderboard ECE values (0.144, 0.088) share a column with the TT ECE (0.484) that the
  report itself declares not comparable; a footnote mark in the table would prevent misreading. The leaderboard rows
  match `/home/hous/dev/clm-v0.1-8B/evals/typed_decisions/data/README.md` lines 74 and 82.
- The ECE caveat is honest and well supported (brief point 2). `/home/hous/dev/clm-v0.1-8B/evals/typed_decisions/SCHEMA.md`
  shows KL (gold || model, natural log, eps 1e-6) and Brier (sum of squared differences against the soft gold)
  reproduce the dataset's Uniform row exactly (0.444, 0.238), that the Prior row is only approximately reproduced
  (built in a way the dataset does not describe), and that eight ECE variants fail to reproduce 0.169;
  `/home/hous/dev/clm-v0.1-8B/evals/typed_decisions/score.py` implements exactly what SCHEMA.md says (10 equal-width
  bins, confidence = max probability, overconfidence component reported); the card omits ECE. No action.
- RUN_NOTES' Hub tag list omits `tt-model-cache` (present on the Hub and in the card frontmatter). RUN_NOTES says
  "the card text says so" about the weights being the unmodified upstream weights; the card implies a port but does
  not state it. The Hub shows one `base_model` tag (`Contrastive-LM/CLM-v0.1-8B`, relation `finetune`) although the
  frontmatter lists two.
- REPORT section 7: "Three p150 runs with the same encoder latency" is not exact; the p150-fast run embeds at 54 ms,
  the default runs at 57.6 ms. The conclusion holds on the per-seed data: seeds 3 and 4 reach 697 in all three runs
  (and seed 2 in the build 6 run); seeds 0 to 2 crash or not; mean decisions 2,039 to 2,165 and discarded answers
  1,983 to 2,004 are stable.
- Unverifiable prose, marked as such and not as wrong: "the p150-fast run shared the host CPU with two review
  subagents" (REPORT section 7); the "Not reproducible here" items (dataset 404, no Terminal-Bench artifacts).

## Hard-Check Gaps
- No serve of `p150-fast` on the published image (see the third P2).
- No saved artifact for the clean-pull `/v1/rank` check ("Moon 0.9948, 157.8 ms", REPORT "Publish" and RUN_NOTES):
  neither `/home/hous/dev/clm-v0.1-8B/logs/` nor
  `/home/hous/dev/clm-v0.1-8B/evals/results/package_pulled_clean_p150_20261002T003519Z/` contains it; the only rank
  outputs on disk are the 22:03 and 22:26 smokes (0.9946, 0.9966). The clean-pull evidence that does exist (README
  example 260.9 ms / 98 tokens, new state 60.5 ms, health with `trace_lens` [128, 256, 512, 1024, 2048]) is
  sufficient on its own.
- The manifest's `code_sha256` is tt-model's own digest; I confirmed the Hub `code/` tree equals the staged tree in
  file count (89) and bytes (77,039,427) but did not recompute the digest.
- On the served path the 256-token bucket was exercised at batch 8 (Typed Decisions) and batch 1 (agent traces)
  only; batch 4 at 256 is covered by the host bench and the fifteen-variant replay check, not by a served request.
  Low risk.

## Anomaly Ledger
- Observed anomaly: the published image contains two kernel sources that differ from the committed tree.
  Evidence: tar scan of the staged OCI layers (layers 11 and 12), `git diff` of
  `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` and
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`.
  Affected path: `p150x4` profile (fabric init, all-gather in `distributed_norm.py`, `attention.py`, `mlp.py`,
  `ccl.py`); all stage 4 host measurements.
  Control or comparison: none; no run of the multi-chip path with the committed kernels exists.
  Likely subsystem: tt-model packaging of a dirty local checkout; release documentation.
  Investigation performed: confirmed presence and identity of the files in the image; confirmed the call sites in
  the shipped code; confirmed the pushed branch lacks the edits; searched the CLM docs for an explanation (only the
  file-mode note in the multichip README).
  Resolution: more-work-needed (P1).
- Observed anomaly: 1x2 mesh fails with "Fabric Router Sync: Timeout after 10000 ms" while the fabric router kernel
  in the tree carries an uncommitted teardown edit.
  Evidence: `/home/hous/dev/clm-v0.1-8B/logs/fidelity_accuracy_1x2.log`, multichip README section "1x2".
  Affected path: two-chip profile (not shipped); listed in the card's limitations.
  Control or comparison: none with the committed kernel; the README attributes it to the 1x4 system mesh and a
  possible physical link problem.
  Likely subsystem: fabric control plane or the modified router kernel.
  Investigation performed: read the logs and README; no hardware run (out of scope).
  Resolution: more-work-needed as part of the P1 (record the edit as a candidate cause or rule it out).
- Observed anomaly: the CLM README's 58.1 ms is quoted as a cold-cache latency.
  Evidence: README line 90 versus the card and REPORT section 2.
  Affected path: card text, REPORT section 2.
  Control or comparison: the TT new-state row (60.2 ms at 54.75 tokens per call) is the like-for-like measurement.
  Likely subsystem: documentation.
  Investigation performed: read the README passage and the harness output format.
  Resolution: more-work-needed (P2).
- Observed anomaly: p150-fast decision agreement is 97.3 percent in the sweep and 95.7 percent (confident) / 93.0
  percent (all) from the served package.
  Evidence: `sweep_results.json` and `package_p150_20261001T224031Z/reference_agreement.json`.
  Affected path: `p150-fast` profile description in the card and manifest.
  Control or comparison: the default profile shows the same direction (sweep 95.5 / 98.9, served 96.0 / 98.9 on
  build 7, 95.5 / 98.4 on build 6), explained by batch composition (REPORT section 3, tt-metal issue 47238).
  Likely subsystem: batch-variant reduction order in batched prefill.
  Investigation performed: compared the three agreement files and the disagreement lists.
  Resolution: controlled as a phenomenon; more-work-needed for the card wording (P2).
- Observed anomaly: T-Rex survival count varies 2 / 3 / 2 across three p150 runs; the published card says "2 of 5".
  Evidence: `/home/hous/dev/clm-v0.1-8B/evals/trex/results/20261001T235652Z`, `20261001T233448Z`, `20261001T225213Z`.
  Affected path: card and REPORT T-Rex rows.
  Control or comparison: per-seed data: seeds 3 and 4 survive at 697 in every run; decisions and discarded answers
  stable; `host_stall_seconds_dropped` 0.0 in all runs.
  Likely subsystem: real-time timing of the game loop against 57.6 ms encoder passes.
  Investigation performed: read all three result files and the reference file.
  Resolution: controlled; the report states the noise and cites the stable signal.
- Observed anomaly: Typed Decisions accuracy and agreement differ between build 6 and build 7 (0.361 vs 0.364; 95.5
  vs 96.0 percent) with bit-identical single-text vectors.
  Evidence: the two `typed_decisions_score.txt` and `reference_agreement.json` files; `fidelity_accuracy_buckets5.json`.
  Affected path: served batched prefill.
  Control or comparison: 308 of 308 single-text vectors identical across bucket sets; batched vectors differ for 128
  texts (min cosine 0.9986).
  Likely subsystem: batch-variant reduction order.
  Investigation performed: read the fidelity evidence and the score files.
  Resolution: controlled; recorded in the report, the card and the context contract.

## Scope Inspected
- Goal/skill paths: `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`,
  `.../tti-release/SKILL.md`, `.../vllm-integration/SKILL.md` (no tt-model skill is installed under
  `/home/hous/.claude/plugins/cache`); `/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4; `/home/hous/dev/clm-v0.1-8B/STATUS.md`.
- Artifact paths: `/home/hous/dev/clm-v0.1-8B/REPORT.md` (read at the start of the review and re-read after its
  00:48 UTC update; the cited passages in sections 2 and 7 are unchanged);
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt-model.yaml`,
  `.../doc/release/RUN_NOTES.md`, `.../doc/serving/README.md`, `.../clm/VENDORED.md`, `.../doc/context_contract.json`,
  `.../doc/datatype_sweep/sweep_results.json`, `.../doc/datatype_sweep/selected_precision_config.json`,
  `.../doc/optimized_full_model/perf_summary_accuracy.json`, `.../doc/optimized_full_model/bench_accuracy_buckets5.json`,
  `.../doc/optimized_full_model/README.md`, `.../doc/multichip_decoder/README.md`,
  `.../doc/review/review_A_stages_1_3.md`, `.../doc/review/review_C_stages_4_8.md`, `.../doc/review/review_A2_stages_1_3.md`;
  staged package `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/` (manifest, README, lock, code tree, OCI
  image); Hub metadata and files of `tt-hous/clm-v0.1-8b-p150` at `3da3cc87...` (model_info, file listing, README,
  manifest, lock, image index and manifest downloaded to the scratchpad); build logs
  `/home/hous/.cache/tt-model/build/clm-v0.1-8b-p150.log`, `/home/hous/dev/clm-v0.1-8B/logs/package_build*.log`; push
  and pull logs `tt_model_push.log`, `tt_model_pull.log`, `tt_model_pull_clean.log`, `evals_pulled_clean.log`,
  `serve_pulled_*.log`, `container_build9_x4.log`, `container_x4_debug.log`, `track_e2.log`; result directories
  `package_p150_b7_20261001T234742Z`, `package_p150_final_20261001T232132Z`, `package_p150_20261001T224031Z`,
  `package_p150x4_b9_20261002T002940Z`, `package_p150x4_20261001T232047Z`, `package_p150_b9_20261002T003001Z`,
  `package_pulled_p150_20261002T003310Z`, `package_pulled_clean_p150_20261002T003519Z` under
  `/home/hous/dev/clm-v0.1-8B/evals/results/`; T-Rex `/home/hous/dev/clm-v0.1-8B/evals/trex/REFERENCE.md`,
  `.../reference/clm_realtime.json`, `.../results/20261001T235652Z`, `20261001T233448Z`, `20261001T225213Z`, `run.sh`;
  `/home/hous/dev/clm-v0.1-8B/evals/README.md`, `.../typed_decisions/SCHEMA.md`, `score.py`, `data/README.md`,
  `data/eval.yaml`; `/home/hous/dev/clm-v0.1-8B/evals/vendor/CLM/README.md`; `/home/hous/dev/clm-v0.1-8B/bin/run-evals.sh`,
  `package-build.sh`.
- Numbers re-derived and found to match: REPORT section 2 and 7 README example (98 / 262.4; 0 / 0.1; 0.8 client),
  vector cache (60.2 / 60.2 / 0.1), every embeddings cell (client p50 and tokens/s), Typed Decisions (0.364 / 2.046 /
  0.625 / 0.484; 312 / 630 ms; per workflow and per type), agreement (96.0 / 98.9; 0.360 vs 0.370; 40 cases, 200
  decisions, 188 confident), fidelity (0.99910 / 0.99596), T-Rex build 7 (2 of 5, 4 deaths, 588.8, 2,082.6, 0.778,
  1,990, 16.4, 1.3), build 6 (3 of 5, 3, 541.2, 2,039.0, 0.781, 2,004), p150-fast (2 of 5, 3, 552.6, 2,165.4, 0.851,
  1,983, 16.4, 1.4), authors (5 of 5, 697.0, 3,341.8, 0.658, 1,250, 16.5, 2.6); build 6 rows (262.4, 60.6, 0.361 /
  2.045 / 0.626 / 0.487 at 1,301 ms, 95.5 / 98.4); p150-fast rows (235.3, 56.7 / 56.6, 0.361 / 2.026 / 0.624 / 0.484
  at 1,140 ms, 93.0 / 95.7); p150x4 build 9 (174.6, 34.1 / 34.2, 0.292 at 110 ms) and build 6 (176.1, 33.6 / 33.8);
  build 9 p150 smoke (262.3, 60.6, 167 ms, 0.294); first pull (260.5, 60.2 / 60.3) and clean pull (260.9, 60.5);
  section 4 sweep table (all six rows against `sweep_results.json`); leaderboard rows (Jev 0.727 / 1.442 / 0.148 /
  0.144 / 710 ms; Prior 0.470 / 0.347 / 0.189 / 0.088); manifest provenance (image tag and digest, code sha256
  `c87308710a85...`, tt-metal `2c710b113b` dirty, `0.65.2.dev9729+g2c710b113b`, created 2026-10-02T00:26:36Z, 14
  `verify` lines, all passed in the build log); Hub (124 files, 1,041,164,071 bytes, created 00:31:30 UTC, public,
  28 image blobs, tags); host facts (hostname, Ubuntu 24.04.3, Docker 29.5.2, KMD 2.10.0, tt-model 0.1.0 at `5caec6c`);
  host test count (20 + 3 = 23).
- Code paths: `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/code/models/tt_transformers/tt/{ccl,distributed_norm,attention,mlp}.py`
  (call sites of `all_gather_async`); the two dirty kernel sources in `/home/hous/dev/ornith-1.5-9b/tt-metal`.
- Commands run (read-only): `git log`, `git status --short`, `git diff`, `git show HEAD:<path>`, `git merge-base
  --is-ancestor`, `git ls-remote` on the tt-metal checkout; `huggingface_hub` `model_info(files_metadata=True)` and
  `hf_hub_download` of README, manifest, lock, image index and manifest; `diff -q` staged vs Hub files; a Python
  `tarfile` scan of the 22 OCI layers for the two kernel paths with sha256 comparison against worktree and `HEAD`; a
  Python read of the T-Rex JSON files; `grep`, `sed`, `awk`, `find`, `wc`, `stat`, `modinfo`, `docker --version`.
  No device was opened, no server started, no ttnn import, no test run.

## Residual Risk
- The multi-chip profile's dependence on the two uncommitted kernel edits is unquantified until a run with the
  committed kernels exists; if the all-gather null-guard is what makes the 1x4 line topology work, the published
  branch cannot reproduce `p150x4` and the 1x2 failure may have the same cause.
- The `p150-fast` profile on the published image has no serve evidence; the risk is low (one environment variable
  selects a policy exercised by the sweep on the final code) but not zero (five buckets with `bfp8_attn` were never
  captured in a container).
- The card is published; every correction above requires a card-only rebuild and a republish to the same repo. The
  evaluated code is unchanged by that, as builds 8 and 9 showed.
- Decision-level nondeterminism across batch compositions (4 to 5 of 200 decisions) is documented but remains a
  property of the served path.
- Stage 3's geometry conclusion is disputed by review A2; the card's "risks" wording may need to follow that
  outcome.
