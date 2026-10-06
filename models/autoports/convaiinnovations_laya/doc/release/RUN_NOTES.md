# Release run notes: tt-hous/laya-p150 (tt-model v5.1 container), staged, not pushed

Host `qb2-120-p11t01` (2x p300c, four Blackhole chips, TT-KMD 2.10.0, firmware 19.15.0, kernel 7.0.0-34), tt-model 0.1.0
(`/home/hous/.tenstorrent-venv/bin/tt-model`), Docker 29.5. Source tree for every build: the clean worktree
`/home/hous/dev/laya/worktree/tt-metal` detached at the branch commit (`dirty: false`). Build driver
`/home/hous/dev/laya/bin/final-build.sh <N>` (package, serve, evaluate, demo evidence, four-chip profile, stop; it never
pushes). Logs under `/home/hous/dev/laya/logs/` (`p4_final_build_<N>.log`, `package_build.log`, `p5_*`), tt-model's own
build log at `/home/hous/.cache/tt-model/build/laya-p150.log`. Hub push and catalog listing are gated on the user's demo
review (PLAN.md, decisions of 2026 Oct 4); the exact commands are at the end of this file.

## Builds

- Build 1, first attempt (2026 Oct 6 00:15 UTC): refused at manifest validation, the fresh worktree had uninitialised
  git submodules (`tt_metal/third_party/umd/CMakeLists.txt` missing). Fixed with
  `git -C /home/hous/dev/laya/worktree/tt-metal submodule update --init --recursive` (43 s).
- Build 1, second attempt (00:15 to 00:18 UTC, `package exit 0`, commit `fed0080908`): image
  `tt-model/laya-p150:9490fc616305`, digest `sha256:9490fc61630537a2bfec437de4a6f8b8e2b95545c453a8a45fb1ea8e313f52f7`,
  code sha256 `03e40358fe6a80376e0f640c6eb1cffe857bacb0e50929ffca32adb9309adcf1`, tt-metal `fed0080908`, `dirty: false`,
  first build with `runtime.packages`; the generated lock (`/home/hous/dev/laya/package/out/laya-p150/requirements.lock`)
  is committed as `requirements.lock` for build 2 onward. All verify lines passed (tt-model refuses the build otherwise).
- Build 2 (00:34 to 00:37 UTC, `package exit 0`, commit `d650825fa8` before the history rewrite that removed the Tracy
  binaries from the branch): the server host path trimmed by Track S (heads built once per request, memoized, batch
  tokenization; wire output byte-identical), `runtime.lock: requirements.lock` (staged lock identical to the repo copy),
  card with the packaged build 1 numbers. Image `tt-model/laya-p150:7e0b39f07e04`, digest
  `sha256:7e0b39f07e04a60d82dc42cf456c949904f9a7c89be5ddf3ba34ac25dabb2d16`, code sha256
  `5e1b584c98456fac5203ffc49eb7cafc41d7dbde0fe66f148656a2e5871be596`, `dirty: false`.
- Build 3 (00:45 to 00:48 UTC, `package exit 0`, commit `4ec75e96f7`): the demo page's `?autorun=decide,feed` option
  for the review screenshot and the card with the packaged build 2 numbers. Image `tt-model/laya-p150:597e9bd0c76b`,
  digest `sha256:597e9bd0c76b7c2ae8ca3ac1b10b3616bcb3357c4e95f40f9a50b298d78f41d8`, code sha256
  `b458861e3783014b1d4268d4b3ee6833ae2947da57ec3406af60fc49b22b7b5e`, `dirty: false`.
- Build 4, the release candidate (00:56 to 00:59 UTC, `package exit 0`, commit `ba429fe433`): card only (the packaged
  build 3 numbers); code sha256 identical to build 3. Image `tt-model/laya-p150:cd1a51a2d8cb`, digest
  `sha256:cd1a51a2d8cb502c6dac4adeffec9e20f69cd8d2923edcd61c367a9ff1302ffe`, code sha256
  `b458861e3783014b1d4268d4b3ee6833ae2947da57ec3406af60fc49b22b7b5e`, tt-metal `ba429fe433` (`dirty: false`). Staged
  bundle: `/home/hous/dev/laya/package/out/laya-p150/` (`tt_kernel_manifest.json`, `README.md`, `requirements.lock`,
  `code/`, `image/`).
- Build 5 (2026 Oct 6 02:34 to 02:37 UTC, `package exit 0`, commit `1a69202bb6`): the review R3 responses (card states
  agreement and placement spread per evidence set and names the `bf16_hifi4` alternative), the demo footer from
  `/v1/health`, live health counters, the shutdown fix for an injected engine, the feed numbers normalized, the sibling's
  per-sequence row buckets (English behaviour unchanged). Image `tt-model/laya-p150:dd4da8467626`, digest
  `sha256:dd4da84676262f1bfaaf05f8eb24b37828fbf26184b95a98eabe4252dc43aead`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b` (the same code as the sibling bundle; the two
  manifests differ in weights, buckets and card), `dirty: false`. Two evaluation runs: the first
  (`package_p150_b5_20261006T023716Z`) overlapped the sibling build 3 compile and shows 205.5 ms at 50 questions, a batch
  low end of 144 questions per second and 11 ms per suite case (amendment A14); the second on a quiet host
  (`package_p150_b5_20261006T024919Z`, load 2.3): E1 wire and tensor 476 of 488 and 403 of 403 (scorer-logit PCC 0.9937),
  single-row E1 792 of 800 and 777 of 779 (median 0.0012, max 0.273), E2 and E3 unchanged, E5 10.9 / 24.6 / 42.5 / 197.5
  ms, 203 to 250 questions per second, feed 20.07 decisions per second, p150x4 (`package_p150x4_b5_20261006T024623Z`)
  13.2 / 16.1 / 22.3 / 68.5 ms and 349 to 940 questions per second. The card quotes the second run.
- Build 6, the English release candidate (2026 Oct 6 02:55 to 02:58 UTC, `package exit 0`, commit `990cea36ba`): card
  only (the packaged build 5 run); code sha256 identical to build 5. Image `tt-model/laya-p150:5c5a30ea0bc8`, digest
  `sha256:5c5a30ea0bc8ea8fd4775b1edc6390fe43bc17b25528be1fd7fe9f8922b77d5c`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b`, `dirty: false`. Staged bundle
  `/home/hous/dev/laya/package/out/laya-p150/` (this build). Verification (`package_p150_b6_20261006T025805Z`,
  `package_p150x4_b6_20261006T030242Z`): every accuracy row equals builds 2 to 5 (E1 wire and tensor 476 of 488 and 403
  of 403; single-row E1 792 of 800 and 777 of 779; E2 0.359 / 0.332 / 0.311 / 0.172 / 0.689; E3 0.955 / 0.593); E5 10.7 /
  24.8 / 42.5 / 197.5 ms, 201 to 249 questions per second (the one-minute load was 5.6 when the speed table started
  because the orchestrator's no-device test run of 02:55 to 02:58 UTC overlapped it; the cells agree with the quiet build 5
  run within 0.3 ms); feed 20.07 decisions per second, agreement 0.382, 0 errors; demo page
  `/home/hous/dev/laya/evidence/demo_laya-p150_b6.png`; p150x4 healthy in 20 s, 11.1 / 16.1 / 23.7 / 68.9 ms, 308 to 936
  questions per second.
- Build 7, the English release candidate (2026 Oct 6 03:28 to 03:30 UTC, `package exit 0`, commit `01e82b93e4`, which
  differs from `578c6dbdb5` only in review documents): card
  only (review R3 re-check corrections: the `bf16_hifi4` sentence scoped to the measured sets, the placement spread
  stated at its measured bucket); code sha256 identical to builds 5 and 6. Image `tt-model/laya-p150:42bac8981309`,
  digest `sha256:42bac89813098e37ed4d405b906e640301f139e80a7722d901362989e6e25b9e`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b`, `dirty: false`. Staged bundle
  `/home/hous/dev/laya/package/out/laya-p150/` (this build). Verification (`package_p150_b7_20261006T033045Z`,
  `package_p150x4_b7_20261006T033521Z`, load 3.1 and 2.2 at the speed tables, nothing else running): every accuracy row
  equals builds 2 to 6 (E1 wire and tensor 476 of 488 and 403 of 403; single-row E1 792 of 800 and 777 of 779; E2 0.359 /
  0.332 / 0.311 / 0.172 / 0.689; E3 0.955 / 0.593); E5 10.8 / 24.7 / 42.7 / 197.2 ms, 202 to 249 questions per second;
  feed 20.07 decisions per second, agreement 0.382, 0 errors; demo page `/home/hous/dev/laya/evidence/demo_laya-p150_b7.png`;
  p150x4 healthy in 20 s, 13.2 / 16.1 / 23.6 / 68.4 ms, 309 to 937 questions per second, parity smoke 97 of 100 and
  82 of 82.

- Build 8 (2026 Oct 6 13:43 to 13:46 UTC, `package exit 0`, commit `5fc55e64e2`): demo page only, after the user's
  review of the build 7 demo: a lede that says what Laya is and is for, with links to the authors' model cards, GitHub,
  docs, blog post, the feed dataset and PyPI; a checkpoint band under the header that names the served checkpoint
  (`ORIGINAL CHECKPOINT` for `convaiinnovations/laya`, `FINE-TUNED CHECKPOINT` for the sibling) with the published
  typed-decisions accuracy of each, because the live feed draws from that dataset and the original checkpoint is near
  chance on it; the server stamps `<title>` and `<body data-model>` from the engine's model id. Model and API paths
  unchanged. Image `tt-model/laya-p150:f5b48d6ee19b`, code sha256 `2ea292bb7326...`. Not served: superseded by build 9
  before deployment.
- Build 9 (2026 Oct 6 13:48 to 13:50 UTC, `package exit 0`, commit `d315a10374`): build 8 plus the removal of the
  act / escalate tile and its caveat from the answer cards (user request; the API still returns
  `action.act_probability`). Image `tt-model/laya-p150:a335363e74e5`, digest
  `sha256:a335363e74e50759a4d71a3de823f95216a6ddb964cccf986cb3ecdac544d450`, code sha256
  `42ab6916a82d736434f7597df64491c4680335d48cf361e96f7194e4665c9206`, tt-metal `d315a10374` (`dirty: false`), 15 verify
  assertions (one new: `id="variant"` in the page). Staged at `/home/hous/dev/laya/package/out/laya-p150/`. The
  model, server and evaluation harness code are identical to build 7 apart from `demo/`, `server/demo.py`,
  `tests/test_demo_page.py` and the two cards, so the build 7 evaluation evidence stands; verification of build 9 is
  limited to the served smoke, the demo screenshots and a 60 s feed run on each port (section below).

## Verification from the served package (build 1, profile p150, chip 0, port 8710)

Results `/home/hous/dev/laya/evals/results/package_p150_b1_20261006T002044Z/SUMMARY.md` (the only source the card and
REPORT.md quote). Server healthy 144 s after start (container log `/home/hous/dev/laya/logs/p5_container_b1_p150.log`).

| check | result |
|---|---|
| E1 parity vs CPU fp32 (488 decisions, wire path) | 476 of 488 argmax, 403 of 403 confident (margin >= 0.10), median max abs dp 0.0082, p95 0.031, max 0.1225, probability PCC 0.9991, 0 NaN |
| E2 typed-decisions (400 cases, 2,000 decisions) | accuracy 0.359, soft 0.332, Brier 0.311, ECE 0.172, score MAE 0.689 (published 0.362 / 0.332 / 0.316 / 0.175 / 0.694; CPU fp32 0.361 / 0.332 / 0.316 / 0.175 / 0.694) |
| E3 AG News / DAIR Emotion (400 each) | 0.955 (ECE 0.035, 9.7 ms per case) / 0.593 (ECE 0.308, 9.6 ms) (authors' CPU run 0.950 / 0.595; Jev 0.910 / 0.480) |
| E5 speed, client p50 for 1 / 5 / 10 / 50 questions | 11.2 / 25.9 / 45.6 / 210.6 ms (server 10.4 / 25.0 / 44.6 / 208.7; device 9.2 / 22.7 / 40.3 / 191.6; T4 39.5 / 84.5 / 158.6 / 771) |
| E5 batched throughput | 192 to 234 questions per second (T4 103 to 332) |
| demo feed (`/home/hous/dev/laya/evidence/demo_feed_b1.json`, 60 s at 4 cases per second) | 241 cases, 1,205 decisions, 20.07 decisions per second, 0 errors, agreement with gold 0.378 |
| demo page | `/home/hous/dev/laya/evidence/demo_p150_b1.png` (health pill: tt ready, bf8w_hifi3_erf, 30 traces; live feed at 20 decisions per second, device p50 45 ms per five-question case) |
| p150x4 profile (`--device-id 0,1,2,3`, port 8711, `package_p150x4_b1_20261006T003234Z`) | healthy 30 s after start; smoke batch 4x256 in 13.8 ms client; parity 20 calls: 97 of 100 argmax, 82 of 82 confident; E2 first 100 cases and E3 first 50 cases ran (AG News 0.960, Emotion 0.700 on those subsets); E5 client p50 13.1 / 16.2 / 25.1 / 82.1 ms for 1 / 5 / 10 / 50 questions (device 10.9 / 12.3 / 19.8 / 62.8), batched 309 to 748 questions per second. The first attempt passed one device id and tt-model refused it ("--device-id gave 1 chip(s) but profile 'p150x4' needs 4"). |

## Verification from the served package (build 2, profile p150, chip 0, port 8710)

Results `/home/hous/dev/laya/evals/results/package_p150_b2_20261006T003732Z/SUMMARY.md`; healthy 20 s after start.
Accuracy rows identical to build 1 (E1 476 of 488 and 403 of 403; E2 0.359 / 0.332 / 0.311 / 0.172 / 0.689; E3 0.955 /
0.593). Latency after the server host-path trim: client p50 10.7 / 24.7 / 42.7 / 197.7 ms for 1 / 5 / 10 / 50 questions
(server 9.9 / 23.9 / 41.7 / 195.8; device 9.2 / 22.7 / 40.2 / 191.8), batched 202 to 250 questions per second; E3
9.4 and 9.3 ms per case. Demo feed (`/home/hous/dev/laya/evidence/demo_feed_b2.json`): 241 cases, 1,205 decisions, 20.08
per second, client p50 48.5 ms per five-question case, 0 errors. Demo page `/home/hous/dev/laya/evidence/demo_p150_b2.png`.
p150x4 (`package_p150x4_b2_20261006T004210Z`): healthy 20 s after start; E5 13.2 / 16.1 / 22.8 / 68.6 ms, batched
308 to 937 questions per second.

## Verification from the served package (build 3, profile p150, chip 0, port 8710)

Results `/home/hous/dev/laya/evals/results/package_p150_b3_20261006T004853Z/SUMMARY.md` (with `parity/decisions.jsonl`
from this build on); healthy 20 s after start. Accuracy rows identical to builds 1 and 2 (E1 476 of 488 and 403 of 403,
median max abs dp 0.0082, p95 0.031; E2 0.359 / 0.332 / 0.311 / 0.172 / 0.689; E3 0.955 / 0.593, 9.4 and 9.3 ms per
case). E5 client p50 10.8 / 24.6 / 42.1 / 197.4 ms for 1 / 5 / 10 / 50 questions (server 9.9 / 23.7 / 41.3 / 195.4;
device 9.2 / 22.7 / 40.2 / 191.7), batched 203 to 249 questions per second. Demo feed
(`/home/hous/dev/laya/evidence/demo_feed_b3.json`): 241 cases, 1,205 decisions, 20.07 per second, client p50 48.6 ms per
five-question case, 0 errors. Demo page with an answered preset and the live feed:
`/home/hous/dev/laya/evidence/demo_p150_b3.png`. p150x4 (`package_p150x4_b3_20261006T005357Z`): healthy 20 s after
start; E5 13.3 / 16.3 / 23.8 / 69.9 ms, batched 301 to 937 questions per second.

## Verification from the served package (build 4, the release candidate, profile p150, chip 0, port 8710)

Results `/home/hous/dev/laya/evals/results/package_p150_b4_20261006T010016Z/SUMMARY.md`; healthy 20 s after start.

| check | result |
|---|---|
| E1 parity vs CPU fp32 (488 decisions, wire path; `parity/decisions.jsonl` stored) | 476 of 488 argmax, 403 of 403 confident, median max abs dp 0.0082, p95 0.031, max 0.1225, probability PCC 0.9991, 0 NaN |
| E2 typed-decisions (400 cases, 2,000 decisions) | 0.359 / 0.332 / 0.311 / 0.172 / 0.689 (published 0.362 / 0.332 / 0.316 / 0.175 / 0.694; CPU fp32 0.3615 / 0.3315 / 0.3155 / 0.1747 / 0.6937) |
| E3 AG News / DAIR Emotion (400 each) | 0.955 (ECE 0.035, 9.4 ms per case) / 0.593 (ECE 0.308, 9.4 ms) |
| E5 client p50 for 1 / 5 / 10 / 50 questions | 10.8 / 24.4 / 42.8 / 197.2 ms (server 9.9 / 23.6 / 41.8 / 195.4; device 9.2 / 22.7 / 40.3 / 191.5; T4 39.5 / 84.5 / 158.6 / 771) |
| E5 batched throughput | 202 to 250 questions per second (T4 103 to 332) |
| demo feed (`/home/hous/dev/laya/evidence/demo_feed_b4.json`) | 241 cases, 1,205 decisions, 20.07 per second, client p50 48.2 ms per five-question case, 0 errors |
| demo page | `/home/hous/dev/laya/evidence/demo_p150_b4.png` (answered preset, tiles, live feed) |
| p150x4 (`--device-id 0,1,2,3`, `package_p150x4_b4_20261006T010523Z`) | healthy 20 s after start; E5 13.1 / 16.1 / 23.6 / 68.7 ms, batched 351 to 939 questions per second |

The card quotes the build 3 run; on the p150 profile build 4 reproduces it within 0.7 ms on every latency cell and
exactly on every accuracy row. On the p150x4 profile the cells differ by up to 1.2 ms (69.9 against 68.7 ms at 50
questions) and the batched low end moved from 301 to 351 questions per second because one noisy 1x5 cell (16.6 against
14.2 ms client p50) sets it; the card keeps the lower figure.

Note on logs: the sibling bundle's first chain run (2026 Oct 6 02:02 UTC) reused the English log names, so
`p5_container_b1_p150.log`, `p5_serve_b1_p150.log`, `p5_evals_b1_p150.log` and `p5_feed_b1.log` now hold the sibling's
run; the English build 1 evidence stays in its result directory and in `p4_final_build_1.log`. Log names carry the
package name from then on.

## Sibling bundle tt-hous/laya-typed-decisions-p150 (staged, not pushed)

Manifest `tt-model-typed-decisions.yaml`: the same server and port, weights `convaiinnovations/laya-typed-decisions` at
revision `e929ae5cf69bc34259cd2f95c9e91145b818b1f0`, sequence buckets 128, 256, 512 and 1024 (rows 1, 2, 4, 5, 8, 10, 16
at 1024; 37 traces, 177.3 MiB), `LAYA_MAX_BATCH_TOKENS` 16384, a sibling sanity reference
(`server/sanity_reference_typed_decisions.json`, selected by `LAYA_SANITY_REFERENCE`). Stage evidence:
`doc/release_typed_decisions/README.md`.

- Build 1 (2026 Oct 6 01:59 to 02:02 UTC, `package exit 0`, commit `6e09a909f5`): image
  `tt-model/laya-typed-decisions-p150:566a0febd4b4`, digest
  `sha256:566a0febd4b4973195531cd104de58cadb0dfcf222369e384a8f95a900027f9f`, code sha256
  `e4ef6027ed9dcd4d305b5a1bd03218dd18e5d89066c1a239b0c613bd6092b957`, `dirty: false`; card with the host-served numbers.
  The chain's first evaluation step did not run (a shell quirk in the chain, fixed) and reused the English log names
  (note above); the evaluation was rerun on the same image at 02:08 UTC.
- Build 2, the sibling release candidate (2026 Oct 6 02:19 to 02:22 UTC, `package exit 0`, commit `8a843f009a`): card with
  the packaged build 1 numbers, demo footer reading the served model from `/v1/health`. Image
  `tt-model/laya-typed-decisions-p150:a4c8f7eecc1b`, digest
  `sha256:a4c8f7eecc1bad867654963c6427a64cf3df97f13d98290afcece2a1485f8533`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b` (the demo page changed, so the code sha differs from
  build 1), `dirty: false`. Staged bundle `/home/hous/dev/laya/package/out/laya-typed-decisions-p150/`. Verification
  (`package_laya-typed-decisions-p150_p150_b2_20261006T022244Z`, `..._p150x4_b2_20261006T022728Z`): E1 485 of 488 and 383
  of 383 (wire and tensor), E2 0.764 / 0.469 / 0.062 / 0.214 / 0.244 with 1,492 of 1,492 confident agreements against the
  sibling CPU reference, E3 AG News 0.953 (400 of 400 argmax, 392 of 392 confident against the sibling CPU reference) and
  Emotion 0.595 (CPU 0.600), E5 10.8 / 24.5 / 42.9 / 197.6 ms, 203 to 249 questions per second, feed 20.07 decisions per
  second at agreement 0.751, p150x4 healthy in 20 s with 12.7 / 16.1 / 23.8 / 68.5 ms and 309 to 936 questions per second,
  parity smoke 100 of 100. The `parity_single_row/` directories under the sibling runs compare against the English
  single-row corpus by mistake (marked with a NOTE.md, not quoted).
- Build 3, the sibling release candidate (2026 Oct 6 02:38 to 02:40 UTC, `package exit 0`, commit `9ed0e2927e`): card only
  (the packaged build 2 numbers and the validated four-chip profile); code sha256 identical to build 2. Image
  `tt-model/laya-typed-decisions-p150:f2d730fa7fed`, digest
  `sha256:f2d730fa7fed5dc150655cc3a1c370c2bc471898b182e09096fa2d316d2f2ef0`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b`, `dirty: false`. Verification
  (`package_laya-typed-decisions-p150_p150_b3_20261006T024203Z`, `..._p150x4_b3_20261006T024743Z`, host load 2.1 at the
  start of the speed table): every accuracy row equals build 2 (E1 485 of 488 and 383 of 383; E2 0.764 / 0.469 / 0.062 /
  0.214 / 0.244; E3 0.953 / 0.595); E5 10.8 / 24.7 / 42.5 / 197.6 ms, 202 to 249 questions per second; feed 20.08
  decisions per second at agreement 0.751, 0 errors; p150x4 healthy in 20 s, 11.4 / 14.9 / 22.2 / 68.6 ms, 342 to 938
  questions per second. Staged bundle `/home/hous/dev/laya/package/out/laya-typed-decisions-p150/` (this build).
- Build 4, the sibling release candidate (2026 Oct 6 03:20 to 03:22 UTC, `package exit 0`, commit `578c6dbdb5`): card
  only (the measured single-row agreement of this checkpoint quoted, the parity flip margin stated as under 0.015) and
  `runtime.lock: requirements.lock` in the manifest; code sha256 identical to builds 2 and 3. Image
  `tt-model/laya-typed-decisions-p150:aed3896b5546`, digest
  `sha256:aed3896b5546096b8aae031a5a258034f6b0cff08a8ee802995f67e0d483f1f8`, code sha256
  `7b36a18faed4998a61c5c6e355eb2bd62cdd9ca9f8ec408bf489b6c30def831b`, `dirty: false`. Staged bundle
  `/home/hous/dev/laya/package/out/laya-typed-decisions-p150/` (this build). Verification
  (`package_laya-typed-decisions-p150_p150_b4_20261006T032237Z`, `..._p150x4_b4_20261006T032658Z`, load 4.1 and 1.9):
  every accuracy row equals builds 2 and 3 (E1 485 of 488 and 383 of 383; E2 0.764 / 0.469 / 0.062 / 0.214 / 0.244; E3
  0.953 / 0.595); E5 10.8 / 24.7 / 42.8 / 197.7 ms, 202 to 249 questions per second; feed 20.07 decisions per second at
  agreement 0.751, 0 errors; p150x4 healthy in 20 s, 13.2 / 16.2 / 23.6 / 68.6 ms, 349 to 942 questions per second,
  parity smoke 100 of 100.

Verification from the served sibling package (build 1, profile p150, chip 0, port 8710;
`/home/hous/dev/laya/evals/results/package_laya-typed-decisions-p150_p150_b1_20261006T020850Z/SUMMARY.md`; healthy 20 s
after start; startup sanity check against the sibling reference ok, max abs dp 0.0093; 37 traces):

| check | result |
|---|---|
| E1 parity vs the sibling CPU fp32 reference (488 decisions; wire and tensor paths) | 485 of 488 argmax, 383 of 383 confident, median max abs dp 0.0039, max 0.040, probability PCC 0.9997, scorer-logit PCC 0.9989, 0 NaN |
| E2 typed-decisions (400 cases at max_len 1024 / head_max_len 256) | 0.764 / 0.469 / 0.062 / 0.214 / 0.244 (published 0.766 / 0.471 / 0.062 / 0.213 / 0.242; CPU fp32 0.766 / 0.471 / 0.061 / 0.213 / 0.242) |
| E3 AG News / DAIR Emotion (400 each, at 1024 / 256) | 0.953 (ECE 0.156) / 0.595 (ECE 0.201); 9.4 and 9.3 ms per case; the CPU fp32 reference for these suites with this checkpoint is produced separately (the first summary compared against the English checkpoint's reference by mistake) |
| E5 client p50 for 1 / 5 / 10 / 50 questions | 10.8 / 24.6 / 42.9 / 197.7 ms (device 9.2 / 22.7 / 40.3 / 191.8); batched 202 to 249 questions per second |
| demo feed (`/home/hous/dev/laya/evidence/demo_feed_laya-typed-decisions-p150_b1.json`) | 241 cases, 1,205 decisions, 20.08 per second, agreement with gold 0.751, 0 errors |
| demo page | `/home/hous/dev/laya/evidence/demo_laya-typed-decisions-p150_b1.png` |
| p150x4 (`--device-id 0,1,2,3`, `package_laya-typed-decisions-p150_p150x4_b1_20261006T021310Z`) | healthy 20 s after start; parity smoke 100 of 100 argmax, 81 of 81 confident; E2 first 100 cases 0.728 (one workflow, partial); E5 13.2 / 14.6 / 23.7 / 68.5 ms for 1 / 5 / 10 / 50 questions, batched 310 to 936 questions per second |

- Build 5 (2026 Oct 6 13:46 to 13:47 UTC, `package exit 0`, commit `5fc55e64e2`): the demo lede and checkpoint band
  (see English build 8); the band reads `FINE-TUNED CHECKPOINT` from this bundle's model id. Image
  `tt-model/laya-typed-decisions-p150:e27fa5720abf`, code sha256 `2ea292bb7326...`. Not served: superseded by build 6
  before deployment.
- Build 6 (2026 Oct 6 13:50 to 13:51 UTC, `package exit 0`, commit `d315a10374`): build 5 plus the removal of the
  act / escalate tile (see English build 9). Image `tt-model/laya-typed-decisions-p150:2d0db36e9ec0`, digest
  `sha256:2d0db36e9ec03b6bbd01d7056a1e52ae423055a5949200262044c1ee83895c6c`, code sha256
  `42ab6916a82d736434f7597df64491c4680335d48cf361e96f7194e4665c9206` (identical to English build 9), tt-metal
  `d315a10374` (`dirty: false`). Staged at `/home/hous/dev/laya/package/out/laya-typed-decisions-p150/`. Model, server
  and harness code identical to build 4 apart from the demo page files and the card, so the build 4 evidence stands;
  build 6 verification is the served smoke, the demo screenshot and a 60 s feed run (section below).

## Verification of builds 9 and 6 from the served images (2026 Oct 6 13:52 to 13:55 UTC)

Both containers started from the staged manifests with `tt-model serve --detach --profile p150`: English build 9 on
port 8710, chip 0 (`tt-model-laya-p150-p150`, image `a335363e74e5`); sibling build 6 on port 8711, chip 1
(`tt-model-laya-typed-decisions-p150-p150`, image `2d0db36e9ec0`). Both reported `ready` within 16 s (device programs
cached from the earlier builds).

| check | English build 9 (port 8710) | sibling build 6 (port 8711) |
|---|---|---|
| `GET /health` | `convaiinnovations/laya` at `7b928d82`, backend `tt` | `convaiinnovations/laya-typed-decisions` at `e929ae5c`, backend `tt` |
| served page stamp | `<title>Laya (original) on p150</title>`, `data-model="convaiinnovations/laya"` | `<title>Laya typed-decisions (fine-tuned) on p150</title>`, `data-model="convaiinnovations/laya-typed-decisions"` |
| `server-smoke.sh` | `SMOKE_DONE`, 0 failures (`/home/hous/dev/laya/logs/p5_smoke_b9b6_8710.log`) | `SMOKE_DONE`, 0 failures (`p5_smoke_b9b6_8711.log`) |
| screenshot (`?autorun=decide,feed&rate=4&seconds=20`) | `/home/hous/dev/laya/evidence/demo_laya-p150_b9.png`: blue `ORIGINAL CHECKPOINT` band, lede with seven links, answer cards without the act tile, feed agreement 0.380 (114 of 300) | `/home/hous/dev/laya/evidence/demo_laya-typed-decisions-p150_b6.png`: green `FINE-TUNED CHECKPOINT` band |
| 60 s feed (`feed_client.py`, 4 cases/s) | `/home/hous/dev/laya/evidence/demo_feed_laya-p150_b9.json`: 241 cases, 1,205 decisions, 20.07 per second, agreement 0.3817 (choice 0.304, noul 0.513, score 0.342), 0 errors, client p50 49.6 ms | `/home/hous/dev/laya/evidence/demo_feed_laya-typed-decisions-p150_b6.json`: 241 cases, 1,205 decisions, 20.07 per second, agreement 0.751 (choice 0.646, noul 0.922, score 0.701), 0 errors, client p50 49.5 ms |

The feed agreement rates equal those recorded for builds 7 and 4 to four decimals (0.3817 and 0.751), as expected for
unchanged model code.

## Publish

English bundle, pushed and listed at the user's request after the demo review (2026 Oct 6):

```
/home/hous/.tenstorrent-venv/bin/tt-model push /home/hous/dev/laya/package/out/laya-p150 --public
/home/hous/.tenstorrent-venv/bin/tt-model publish tt-hous/laya-p150
```

- Push 14:05 UTC (`/home/hous/dev/laya/logs/p9_push_laya-p150_b9.log`): repo `tt-hous/laya-p150` created public,
  `image/` 892.5 MB in 28 content-addressed blobs (layers shared with another model on the same tt-metal commit upload
  once), upload 21.2 s, `pushed tt-hous/laya-p150`. Hub revision `dcd0ad8ca6cd44bfb3b248d75747b9d555382bc0`, 82 files,
  tags `tt-model-container`, `tt-dit-server`, `tt-model-cache`, `tt-model-catalog`.
- Publish 14:06 UTC (`p9_publish_laya-p150.log`): `listed tt-hous/laya-p150 in the community catalog`. Delist with
  `tt-model unpublish tt-hous/laya-p150`.
- Consumers: `tt-model pull tt-hous/laya-p150` then `tt-model serve tt-hous/laya-p150 --device-id 0 --profile p150`
  (or `--profile p150x4 --device-id 0,1,2,3`); the demo is at `/demo/` on the served port.
- Pull check (14:07 to 14:15 UTC): `tt-model pull tt-hous/laya-p150` resolved the Hub revision `dcd0ad8c` to image
  `tt-model/laya-p150:a335363e74e5`, which was already loaded on this box (the image layers were not re-downloaded, so
  the check covers the Hub manifest and bundle, not the image transfer); manifest at
  `/home/hous/.cache/tt-model/pulled/tt-hous__laya-p150/tt_kernel_manifest.json`. The container name is per package and
  profile, so the staged demo container on port 8710 was stopped and the published bundle served in its place with
  `tt-model serve tt-hous/laya-p150 --port 8710 --device-id 0 --detach --profile p150`: ready after about 20 s,
  `/health` ok on backend `tt`, `/demo/` 200 with the title `Laya (original) on p150`, a noul decision answered
  (`urgent` 0.7511). Port 8710 keeps serving the published bundle; port 8711 serves the staged sibling build 6.

Sibling bundle `tt-hous/laya-typed-decisions-p150` (build 6): still staged only, awaiting the user's decision:

```
/home/hous/.tenstorrent-venv/bin/tt-model push /home/hous/dev/laya/package/out/laya-typed-decisions-p150 --public
/home/hous/.tenstorrent-venv/bin/tt-model publish tt-hous/laya-typed-decisions-p150
```
