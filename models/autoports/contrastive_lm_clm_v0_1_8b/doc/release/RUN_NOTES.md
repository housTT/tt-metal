# Release run notes: tt-hous/clm-v0.1-8b-p150 (tt-model v5.1 container)

Substitute for the plugin's TTI release stage (the model has no vLLM serving path). Everything a consumer gets is
produced by `tt-model package --container` from `tt-model.yaml` in this directory, then verified by serving the
built package on this box and running the evaluation harnesses against it.

## Host and tree

- Host `qb2-120-p11t01`: 2 x p300c (4 Blackhole chips), TT-KMD 2.10.0, firmware 19.15.0, Ubuntu 24.04, Docker 29.5.2.
- tt-metal `/home/hous/dev/ornith-1.5-9b/tt-metal`, branch `hous/clm-v0.1-8b`, final build from commit `a3df3fd2ee`
  (branch pushed to `https://github.com/housTT/tt-metal`; the manifest's `pushed: false` refers to the upstream
  `tenstorrent/tt-metal` remote, which does not carry this branch). The working tree also carries
  unrelated uncommitted edits from the earlier Ornith project (`AGENTS.md`, two kernel files), which is why the
  manifest provenance says `dirty: true`; none of them is on the CLM serving path (the allowlist ships only
  `models/common` files, `models/tt_transformers/tt` and this autoport).
- tt-model 0.1.0 from `/home/hous/dev/tt-model-manager` (`5caec6c`).

## Build

```
umask 0022
tt-model package --container models/autoports/contrastive_lm_clm_v0_1_8b/tt-model.yaml --out /home/hous/dev/clm-v0.1-8B/package/out
```

- Log: `/home/hous/.cache/tt-model/build/clm-v0.1-8b-p150.log`; driver log `/home/hous/dev/clm-v0.1-8B/logs/package_build.log`.
- First attempt failed before docker (root-owned `~/.docker/buildx/activity` from an earlier sudo docker run; chown fixed it).
  Second attempt failed in the in-image `verify.sh` with `Permission denied`: the project directory's umask 007 made the
  staged scripts and `code/` unreadable for the image's unprivileged `tt` user; `umask 0022` fixed it.
- Third attempt: 2026 Oct 1 22:28 to 22:31 UTC. The tt-metal compile layer came from the BuildKit cache (115 s),
  all 13 `verify:` assertions passed (imports from the stripped tree, prefetcher config and playground files present,
  head checkpoint sha256 `b2b4a8c9...4eda5`, transformers 5.x, numpy 1.x, head parameter count 18,887,680).
- Image `tt-model/clm-v0.1-8b-p150:7efb9343d137`, digest `sha256:7efb9343d13738ea96c1b75c188c055bc083e029f0b520ff48737a367514b69e`,
  code sha256 `26ab3f7a2bfd2e509a8c1e01f715a820ccafa0162af347a460294d820aca8a4a`, built 2026-10-01T22:30:46Z.
- Staged repo: `/home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150/` (`tt_kernel_manifest.json`, `README.md`,
  `code/`, `image/`, `requirements.lock`).

- Fourth and fifth attempts (22:36 to 22:38 UTC): the rebuilt image still had group-only directory modes because
  BuildKit's `COPY` cache key ignores directory permissions; changing the content of a shipped file
  (`__version__` markers) invalidated the layer. Image `aa6f0847aa7a` served the single-chip profiles.
- Sixth attempt (23:08 to 23:13 UTC, `package exit 0`): the fifth image's p150x4 profile had failed in-container on an
  unreadable fabric kernel source (two tree files with mode 0660 from the Ornith edits); `chmod -R a+rX` on the
  tt-metal source tree, then rebuilt with the final manifest (default profile `p150` = accuracy policy,
  `p150-fast` = stock bfp8 policy, `p150x4`) and the card text. The tt-metal compile layer rebuilt in 109 s
  (ccache), all 13 `verify:` assertions passed. Image `tt-model/clm-v0.1-8b-p150:38a80e5078b7`, digest
  `sha256:38a80e5078b7e7a303863a18bbf73f4d4bb13041193ec0eff907632d8744a475`, code sha256
  `76a384d3d9d7ea58ae4da3e5f0d1a039440fb43f434ed792135f6efdb3ff2287`, created 2026-10-01T23:10:40Z, tt-metal
  `a3df3fd2ee` (`0.65.2.dev9726+ga3df3fd2ee`). Verified from the served package (table below), then superseded.
- Seventh attempt (23:44 to 23:47 UTC, `package exit 0`): after the build 6 evaluation showed the 1024-token bucket
  cost on Typed Decisions, the encoder gained the 256 and 512 token prefill buckets (`doc/optimized_full_model/README.md`,
  "Prefill buckets"). Image `tt-model/clm-v0.1-8b-p150:a79fd98c9a89`, digest
  `sha256:a79fd98c9a89dfb3e19a6b3150f81259aa45f60ee96b5caefcab29f412aed0c1`, code sha256
  `c87308710a8567474a9cc832873033182fd111e55558ca96cf92b3c6ac34970c`, created 2026-10-01T23:44:45Z. The manifest
  records tt-metal `a890a5ab05` plus `dirty` because the bucket change was committed (`b39ef75ab6`) a few minutes
  after the snapshot; the shipped code is the working tree, which equals that commit. This image was evaluated
  (table below) for the `p150` profile. Its `p150x4` profile did not boot: the fabric router kernel source was
  again mode 0660 inside the image (`Cannot open kernel source file .../fabric_erisc_router.cpp`). Cause: the
  pre-commit hooks stash and restore unstaged files on every commit, and the restore re-creates the Ornith-edited
  kernel files with the project's umask 007, undoing the `chmod` done before build 6. Fix: `package-build.sh` now
  runs `chmod -R a+rX` over the shipped source trees before every build and logs the count of files that are still
  not world-readable, and the manifest gained a `verify:` line that opens that kernel source inside the image.
- Eighth attempt (card text and the mode fix; same code as build 7): recorded under "Publish".

## Serve profiles

| profile | hardware | mesh | precision | notes |
|---|---|---|---|---|
| p150 (default) | p150 | P150 | accuracy | selected by the stage 8 sweep with the decision-agreement gate |
| p150-fast | p150 | P150 | bfp8_attn (stock default policy) | 6 percent faster, 97.3 percent decision agreement |
| p150x4 | p150x4 | P150x4 | accuracy | 1x4 tensor parallel, FABRIC_1D, 1.9x faster at 128 tokens |

```
tt-model serve --port 8700 --device-id 0 --detach --profile p150 <manifest or tt-hous/clm-v0.1-8b-p150>
tt-model logs clm-v0.1-8b-p150 -f
tt-model stop clm-v0.1-8b-p150
```

## Verification from the served package

Final image `a79fd98c9a89` (five prefill buckets), served with `tt-model serve --port 8700 --device-id 0 --detach
--profile p150` and `--profile p150x4 --port 8702 --device-id 0,1,2,3`; harness `/home/hous/dev/clm-v0.1-8B/bin/run-evals.sh`;
results under `/home/hous/dev/clm-v0.1-8B/evals/results/`.

| check | p150 (default, accuracy) `package_p150_b7_20261001T234742Z` | p150x4 `package_p150x4_b7_*` |
|---|---|---|
| `GET /health` | ok, ready, embedder tt, models clm-latest and clm-raw, cache 537 MB reserved | X4_HEALTH_B7 |
| README example, cold (`usage.input_tokens latency_ms`) | 98 262.4; answers urgency 0.816, billing 0.993, frustration 2.000 | X4_README_B7 |
| README example, warm x20 | 0 0.1 (client 0.8 ms) | 0 0.1 |
| vector cache, new state every call, 3 / 50 actions | 60.2 / 60.2 ms | X4_CACHE_B7 |
| vector cache, revisited and repeated states | 0.1 ms | 0.1 ms |
| embeddings table (client p50) | 128 tok x 1: 61.4 ms; 128 x 8: 175.9 ms (5,820 tok/s); 512 x 8: 641 ms (6,393 tok/s); 1024 x 8: 1,320 ms (6,204 tok/s); 2048 x 32: 10,378 ms (6,315 tok/s) | not run |
| Typed Decisions, zero-shot | 400 cases, 2,000 decisions, 0 errors: accuracy 0.364, KL 2.046, Brier 0.625, ECE 0.484, p50 312 ms per case (p95 630 ms) | X4_TD_B7 |
| agreement with the CPU fp32 reference (40-case subset, 200 decisions) | 96.0 percent; 98.9 percent where the reference margin >= 0.10; accuracy vs gold 0.360 vs 0.370 | not run |
| T-Rex, 5 seeds x 60 s, shield on | 2 of 5 survived (seeds 3 and 4 at the 697 course maximum), mean best 589, 2,083 decisions, planner agreement 0.778, answer p50 16.4 ms, model p50 1.3 ms, 1,990 answers discarded, 0 errors (`/home/hous/dev/clm-v0.1-8B/evals/trex/results/20261001T235652Z`) | not run |

Build 6 (image `38a80e5078b7`, same code except the 256 / 512 buckets, `package_p150_final_20261001T232132Z`):
README example cold 98 262.4; new state 60.6 ms; Typed Decisions 0.361 / 2.045 / 0.626 / 0.487 at 1,301 ms per case
(the 1024-token bucket cost); agreement 95.5 percent (98.4 percent confident); T-Rex 3 of 5 survived, mean best 541,
2,039 decisions. The p150x4 profile of build 6 (`package_p150x4_20261001T232047Z`): README example cold 98 176.1,
new state 33.6 ms, first 100 Typed Decisions cases at 110 ms (147 ms on one chip for the same cases).

The `p150-fast` profile (stock bfp8 policy) was evaluated with the full suite on image `aa6f0847aa7a`
(`package_p150_20261001T224031Z`): README example cold 98 235.3; new state 56.7 ms; Typed Decisions 0.361 / 2.026 /
0.624 / 0.484 at 1,140 ms; agreement 93.0 percent (95.7 percent confident); T-Rex 2 of 5 survived. Every serve of an
image exercised the container's weight-cache build and warmup (first serve loads the Qwen3-8B safetensors from the
Hub cache and writes the ttnn weight cache; later serves load in about 25 s plus 18 s of trace preparation and
capture for the fifteen variants).

## Publish

TODO: `tt-model push` result, HF revision, `tt-model pull` from a clean cache and serve check.
