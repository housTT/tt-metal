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
  `a3df3fd2ee` (`0.65.2.dev9726+ga3df3fd2ee`). This is the image that was verified and published below.

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

TODO: fill from `/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_*` (health, README example, vector-cache
table, embeddings table, Typed Decisions 400 cases, T-Rex, reference agreement) and from the other profiles.

## Publish

TODO: `tt-model push` result, HF revision, `tt-model pull` from a clean cache and serve check.
