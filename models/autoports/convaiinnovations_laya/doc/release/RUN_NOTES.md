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
at 1024; 37 traces, 176.8 MiB), `LAYA_MAX_BATCH_TOKENS` 16384, a sibling sanity reference
(`server/sanity_reference_typed_decisions.json`, selected by `LAYA_SANITY_REFERENCE`). Stage evidence:
`doc/release_typed_decisions/README.md`.

- Build 1 (2026 Oct 6 01:59 to 02:02 UTC, `package exit 0`, commit `6e09a909f5`): image
  `tt-model/laya-typed-decisions-p150:566a0febd4b4`, digest
  `sha256:566a0febd4b4973195531cd104de58cadb0dfcf222369e384a8f95a900027f9f`, code sha256
  `e4ef6027ed9dcd4d305b5a1bd03218dd18e5d89066c1a239b0c613bd6092b957`, `dirty: false`; card with the host-served numbers.
  The chain's first evaluation step did not run (a shell quirk in the chain, fixed) and reused the English log names
  (note above); the evaluation was rerun on the same image at 02:08 UTC.

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

## Publish (gated, not run)

```
tt-model push /home/hous/dev/laya/package/out/laya-p150 --public
tt-model publish tt-hous/laya-p150
tt-model push /home/hous/dev/laya/package/out/laya-typed-decisions-p150 --public
tt-model publish tt-hous/laya-typed-decisions-p150
```
