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

Image `38a80e5078b7`, served with `tt-model serve --port 8700 --device-id 0 --detach --profile p150` and
`--profile p150x4 --port 8702`; harness `/home/hous/dev/clm-v0.1-8B/bin/run-evals.sh`; results under
`/home/hous/dev/clm-v0.1-8B/evals/results/`.

| check | p150 (default, accuracy) `package_p150_final_20261001T232132Z` | p150x4 `package_p150x4_20261001T232047Z` |
|---|---|---|
| `GET /health` | ok, ready, embedder tt, models clm-latest and clm-raw, cache 537 MB reserved | same |
| README example, cold (`usage.input_tokens latency_ms`) | 98 262.4; answers urgency 0.816, billing 0.993, frustration 2.000 | 98 176.1 |
| README example, warm x20 | 0 0.1 (client 0.7 ms) | 0 0.1 |
| vector cache, new state every call, 3 / 50 actions | 60.6 / 60.6 ms | 33.6 / 33.8 ms |
| vector cache, revisited and repeated states | 0.1 ms | 0.1 ms |
| embeddings table (client p50) | 128 tok x 1: 61.4 ms; 128 x 8: 175.9 ms (5,820 tok/s); 1024 x 8: 1,320 ms (6,204 tok/s); 2048 x 32: 10,376 ms (6,316 tok/s) | not run |
| Typed Decisions, zero-shot | 400 cases, 2,000 decisions, 0 errors: accuracy 0.361, KL 2.045, Brier 0.626, ECE 0.487, p50 1,301 ms per case | first 100 cases: accuracy 0.292, p50 110 ms (same 100 cases cost 147 ms on p150-fast) |
| agreement with the CPU fp32 reference (40-case subset, 200 decisions) | 95.5 percent; 98.4 percent where the reference margin >= 0.10; accuracy vs gold 0.355 vs 0.370 | not run |
| T-Rex, 5 seeds x 60 s, shield on | 3 of 5 survived, mean best 541, 2,039 decisions, planner agreement 0.781, answer p50 16.4 ms, model p50 1.3 ms, 0 errors | not run |

The `p150-fast` profile (stock bfp8 policy) was evaluated with the full suite on image `aa6f0847aa7a`
(`package_p150_20261001T224031Z`): README example cold 98 235.3; new state 56.7 ms; Typed Decisions 0.361 / 2.026 /
0.624 / 0.484 at 1,140 ms; agreement 93.0 percent (95.7 percent confident); T-Rex 2 of 5 survived. The container's
serve-time weight download, cache build and warmup were exercised on every serve (first serve of an image loads the
Qwen3-8B safetensors from the Hub cache and writes the ttnn weight cache; later serves load in about 25 s plus 6 s
of trace capture).

## Publish

TODO: `tt-model push` result, HF revision, `tt-model pull` from a clean cache and serve check.
