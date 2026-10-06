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

## Publish (gated, not run)

```
tt-model push /home/hous/dev/laya/package/out/laya-p150 --public
tt-model publish tt-hous/laya-p150
```
