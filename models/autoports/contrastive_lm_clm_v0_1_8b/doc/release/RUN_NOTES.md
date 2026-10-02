# Release run notes: tt-hous/clm-v0.1-8b-p150 (tt-model v5.1 container)

Substitute for the plugin's TTI release stage (the model has no vLLM serving path). Everything a consumer gets is
produced by `tt-model package --container` from `tt-model.yaml` in this directory, then verified by serving the
built package on this box and running the evaluation harnesses against it.

## Host and tree

- Host `qb2-120-p11t01`: 2 x p300c (4 Blackhole chips), TT-KMD 2.10.0, firmware 19.15.0, Ubuntu 24.04, Docker 29.5.2.
- tt-metal `/home/hous/dev/ornith-1.5-9b/tt-metal`, branch `hous/clm-v0.1-8b`; builds 6 to 9 from commits `a3df3fd2ee` to `2c710b113b`; the published build is recorded under "Publish"
  (branch pushed to `https://github.com/housTT/tt-metal`; the manifests of builds 1 to 9 say `pushed: false`
  because they were taken before the push or checked the upstream remote; the published manifest says `pushed: true`,
  `remote: git@github.com:housTT/tt-metal.git`). The working tree also carries unrelated
  uncommitted edits from the earlier Ornith project (21 tracked files: `AGENTS.md`, Ornith autoport files, and two
  kernel sources), which is why the manifests of builds 1 to 9 say `dirty: true`. Correction after review R: the
  two kernel sources ARE in those images and ARE on the four-chip serving path. `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`
  (+3 lines, `noc_clear_packet_tags(noc_index)` in `teardown`) and
  `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp` (+8 / -2,
  null direction pointers instead of unconditional dereferences) are copied with the builder's `tt_metal/` and
  `ttnn/` trees into the runtime image (OCI layers 11 and 12 of image `9372e4d3d3c4`), the fabric router is compiled
  at `FABRIC_1D` initialization and the all-gather writer by `ttnn.experimental.all_gather_async`, so every 1x4
  measurement up to build 9 (`doc/multichip_decoder/README.md`) ran on the edited kernels; the single-chip profiles
  compile neither. The earlier sentence "none of them is on the CLM serving path" was wrong. Resolution (2026 Oct 2
  00:56 to 00:59 UTC): with the two files stashed to their committed versions, the 1x4 fidelity run of the
  `accuracy_lofi_mlp` policy passes (`/home/hous/dev/clm-v0.1-8B/logs/fidelity_lofi_1x4_stock_kernels.log`,
  `doc/multichip_decoder/fidelity_accuracy_lofi_mlp_1x4_stock_kernels.json`: cosine vs fp32 reference mean 0.99908 / min 0.99588 / p05 0.99763, head-projection minima 0.9917 (state) and 0.9970 (candidate), single vs batched min 0.99664, no NaN; decision agreement
  97.0 percent, 99.5 percent on confident decisions), so the edits are not required by this port, and the published
  build (tenth and later) is produced from a clean worktree of the branch (`dirty: false`), see "Publish".
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
- Tenth attempt (2026 Oct 2 01:08 to 01:10 UTC, `package exit 0`, image `500871f7c383`, code sha256
  `e9ae8402b6337c9cf84fa8c049dea50bcc0422849995c95a95099734a23137c6`, tt-metal `abd3e02365` dirty): the second-round
  review responses (program-config overrides, `accuracy_lofi_mlp` default, `p150-accuracy` profile). Intended to come
  from the clean worktree, but `tt-model` takes the source tree from the manifest's `source.tt_metal` path, which still
  named the main checkout, so this image also carries the two Ornith kernel edits and (after a commit's stash and
  restore) a mode-0660 fabric router source; its `p150x4` profile did not boot. Its `p150`, `p150-fast` and
  `p150-accuracy` profiles were evaluated (table below) and the code sha256 is the one the published build carries.
- Eleventh attempt (the published build): `source.tt_metal` set to the clean worktree
  `/home/hous/dev/clm-v0.1-8B/worktree/tt-metal` (detached at the pushed commit, no uncommitted files), recorded under
  "Publish".
- Twelfth attempt (2026 Oct 2 14:39 to 14:41 UTC, `package exit 0`, Hub revision 4): the T-Rex live demo served at
  `/demo/` (vendored harness `server/trex/`, `server/demo_brain.py`, `server/demo_runner.py`, `server/demo.py`, the page
  under `clm/demo/`, `websockets==16.0` added to the lock, three new `verify:` lines; 17 assertions pass; staged lock
  identical to the repo copy). Image `tt-model/clm-v0.1-8b-p150:73e7a51f46f4`, digest
  `sha256:73e7a51f46f44b53a18e6510c979c63e25d4fc327d15d85ca0f80527589e5b5c`, code sha256
  `9ea9a148d22ee670f0e8bb67d24ddbbaa7db9c4b2fa60047fa0014c5fa408e58`, built from the clean worktree at commit
  `29a1d6a715` (`dirty: false`). Demo and default-profile checks under "Demo verification".
- Thirteenth attempt (14:51 to 14:53 UTC, `package exit 0`, the published build, Hub revision 5): one change on top of
  build 12, `Session.stop()` in `server/demo.py` always joins the finished runner process, so no `<defunct>` child
  remains after a game. Image `tt-model/clm-v0.1-8b-p150:85eee9d0dd8b`, digest
  `sha256:85eee9d0dd8b803d07d2d57f66578c3911a0fde5b3609e2819dc6cb273d05993`, code sha256
  `db04b18c9024e224629cbd6f35779404d71b0430403ea09162fa00ae9fd43c6e`, built from the clean worktree at commit
  `6432259fac` (`dirty: false`), 17 `verify:` assertions pass, staged lock identical to the repo copy.
- Fourteenth attempt (18:37 to 18:40 UTC, `package exit 0`, the published build, Hub revision 6): the demo page defines
  the shield (lede sentence, a help paragraph under the Run controls, a tooltip on the shield-saves counter); no server
  or engine change. Image `tt-model/clm-v0.1-8b-p150:f02b781cf8d0`, digest
  `sha256:f02b781cf8d079f50fa4f045fd81a641d6ef868c5a734502381ae8206196459b`, code sha256
  `949aa72e5600889b125e98cd047df93504a603dc9fcf85e7ea270318a0cd7cb8`, built from the clean worktree at commit
  `2641f871d7` (`dirty: false`; also the head of `hous/clm-v0.1-8b-release`), staged lock identical to the repo copy.
- Eighth attempt (00:24 to 00:26 UTC): card text from the build 7 evaluation and the mode fix; image `f5a706931401`,
  code sha256 unchanged (`c87308710a85...`), which confirms file modes and the card are outside the code hash.
- Ninth attempt (00:26 to 00:29 UTC, `package exit 0`, the first published build, Hub revision 1): adds the fabric kernel `verify:` line
  (14 assertions pass). Image `tt-model/clm-v0.1-8b-p150:9372e4d3d3c4`, digest
  `sha256:9372e4d3d3c488067bdc2a22a23040ca0deed6027f76221b5d36763632be7100`, code sha256
  `c87308710a8567474a9cc832873033182fd111e55558ca96cf92b3c6ac34970c`, created 2026-10-02T00:26:36Z, tt-metal
  `2c710b113b` (dirty: the unrelated Ornith edits). The `p150x4` profile boots and serves from this image (table
  below), and the `p150` profile smoke matches build 7.

## Serve profiles (published revision 2 and later)

| profile | hardware | mesh | policy and configs | measured against the default (host bench) |
|---|---|---|---|---|
| p150 (default) | p150 | P150 | accuracy_lofi_mlp with the program-config overrides and the block-sharded norm | selected by the stage 8 sweep: lowest served-workload sum of the passing rows (1,109 ms) |
| p150-accuracy | p150 | P150 | stock accuracy policy, stock configs (the previous default) | 8.7 percent slower at 128 tokens batch 1, 19 to 22 percent slower in the other cells; passes the gate |
| p150-fast | p150 | P150 | stock bfp8 policy, stock configs | 1.7 percent slower at 128 tokens batch 1 (54.0 vs 53.1 ms) and 3 to 8 percent slower elsewhere; fails the agreement gate (97.3 percent sweep, 95.7 percent served) |
| p150x4 | p150x4 | P150x4 | accuracy_lofi_mlp, 1x4 tensor parallel, FABRIC_1D; overrides do not apply | 169 ms cold README example and 34 ms new state from the published image (vs 220 and 56 ms on one chip) |

Revision 1 (image `9372e4d3d3c4`) shipped `p150` = accuracy, `p150-fast` and `p150x4` = accuracy.

```
tt-model serve --port 8700 --device-id 0 --detach --profile p150 <manifest or tt-hous/clm-v0.1-8b-p150>
tt-model logs clm-v0.1-8b-p150 -f
tt-model stop clm-v0.1-8b-p150
```

## Verification from the served package

Build 10 image `500871f7c383` (code sha256 `e9ae8402...`, the same code as the published build 11), served with
`tt-model serve --port 8700 --device-id 0 --detach --profile p150` and `--port 8702 --profile <other>`; harness
`/home/hous/dev/clm-v0.1-8B/bin/run-evals.sh`; results under `/home/hous/dev/clm-v0.1-8B/evals/results/`.

| check | p150 (default: accuracy_lofi_mlp + overrides) `package_p150_b10_20261002T011041Z` | p150-accuracy `package_p150-accuracy_b10_20261002T012508Z` | p150-fast `package_p150-fast_b10_20261002T012408Z` | p150x4 (build 11) `package_p150x4_b11_20261002T013245Z` |
|---|---|---|---|---|
| `GET /health` and boot | ok, ready, embedder tt; model load 5.2 s, fifteen traces | ok; 5.5 s | ok; 5.9 s | ok; 5.5 s load, healthy 50 s after start (image `6cf6949ed327`) |
| README example, cold (`usage.input_tokens latency_ms`) | 98 219.8; answers urgency 0.875, billing 0.986, frustration 2.000 | 98 262.9 | 98 234.2 | 98 169.3 |
| README example, warm x20 | 0 0.1 (client 0.8 ms) | 0 0.1 | 0 0.1 | 0 0.1 |
| vector cache, new state every call, 3 / 50 actions | 56.0 / 56.0 ms | 60.4 ms | 56.4 ms | 34.3 ms |
| vector cache, revisited and repeated states | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| embeddings table (client p50) | 128 tok x 1: 56.6 ms; 128 x 8: 149.2 ms (6,863 tok/s); 512 x 8: 530 ms (7,723 tok/s); 1024 x 8: 1,087 ms (7,534 tok/s); 2048 x 32: 8,477 ms (7,731 tok/s) | not run | not run | not run |
| Typed Decisions, zero-shot | 400 cases, 2,000 decisions, 0 errors: accuracy 0.345, KL 2.039, Brier 0.628, ECE 0.503, p50 258 ms per case (p95 520 ms) | first 100 cases: accuracy 0.294, p50 167 ms | first 100 cases: accuracy 0.292, p50 147 ms | first 100 cases: accuracy 0.296, p50 106 ms |
| agreement with the CPU fp32 reference (40-case subset, 200 decisions) | 96.0 percent; 98.4 percent where the reference margin >= 0.10; accuracy vs gold 0.355 vs 0.370 | not run (standalone sweep: 95.5 / 98.9) | not run (standalone sweep 94.5 / 97.3; served on build 5: 93.0 / 95.7) | not run (standalone 1x4 run: 97.0 / 99.5) |
| T-Rex, 5 seeds x 60 s, shield on | 3 of 5 survived (seeds 0, 2, 3 at the 697 course maximum), mean best 587, 2,195 decisions, planner agreement 0.757, answer p50 16.4 ms, model p50 1.3 ms, 2,016 answers discarded, 0 errors (`/home/hous/dev/clm-v0.1-8B/evals/trex/results/20261002T011822Z`) | not run | not run | not run |

Earlier full runs (same harness): build 7 `accuracy` default with five buckets (`package_p150_b7_20261001T234742Z`:
README example cold 262.4, new state 60.2 ms, Typed Decisions 0.364 / 2.046 / 0.625 / 0.484 at 312 ms, agreement
96.0 / 98.9 percent, T-Rex 2 of 5); build 6 `accuracy` with three buckets (`package_p150_final_20261001T232132Z`:
Typed Decisions 0.361 at 1,301 ms, agreement 95.5 / 98.4, T-Rex 3 of 5); build 5 `bfp8_attn`
(`package_p150_20261001T224031Z`: 0.361 at 1,140 ms, agreement 93.0 / 95.7, T-Rex 2 of 5); build 6 and 9 `p150x4`
checks (README example cold 176.1 and 174.6 ms, new state 33.6 and 34.1 ms, first 100 cases at 110 ms).

## Demo verification (build 12 and later)

Build 12 image `73e7a51f46f4` served on chip 0 with `tt-model serve --port 8700 --device-id 0 --detach --profile p150`;
games driven over the WebSocket by `/home/hous/dev/clm-v0.1-8B/evals/trex/demo_ws_client.py`, which saves every
message under `/home/hous/dev/clm-v0.1-8B/evals/trex/results/demo_<stamp>/` (`messages.jsonl`, `report.json`). The brain
process inside the container posts to the server's own `/v1/systemone` at `http://127.0.0.1:8700`; 6 requests in
flight, shield on, seed 0.

| check | result |
|---|---|
| `GET /demo` | 307 to `/demo/`; `GET /demo/` 200; `GET /demo/demo.js` 200 |
| 20 s game (`demo_20261002T144221Z`) | survived, best score 193, 365 decisions, planner agreement 0.901, answer p50 16.6 ms (p95 333 ms), model p50 1.4 ms, server p50 0.3 ms (p95 273 ms), 131 late answers discarded, 10 shield interventions, 36 arrival saves, 8 emergency saves, 0 errors, host stall 0.0 s; 845 frames and 40 stats messages received |
| 60 s game (`demo_20261002T144243Z`) | survived at the 697 course maximum, 1,350 decisions, agreement 0.800, answer p50 16.6 ms (p95 234 ms), model p50 1.3 ms, server p50 0.3 ms (p95 217 ms), 513 discarded, 12 interventions, 219 arrival saves, 8 emergency saves, 0 errors; 2,621 frames |
| live page | headless Firefox screenshot of the page during the 60 s game: `/home/hous/dev/clm-v0.1-8B/evidence/demo_p150_build12.png` (16 s in: 678 decisions at 42.5 per second, 72 late answers dropped, 41 shield saves, 93.4 percent agreement, score 146, round trip 16.5 ms, model call 0.9 ms, chip 0.2 ms for a cached state) |
| default profile with the demo code (`package_p150_b12_20261002T144344Z`) | README example cold 98 tokens 218.5 ms, new state 55.8 ms, first 100 Typed Decisions cases 0.294 at 140 ms (matches builds 10 and 11) |
| clean pull of revision 4, 10 s game on port 8703 (`demo_20261002T144536Z`) | `GET /demo/` 200; survived, best 89, 204 decisions, agreement 0.946, answer p50 16.5 ms, model p50 1.1 ms, server p50 0.2 ms, 40 discarded, 0 errors |
| processes in the container | during a game: the server, the runner (`trex-demo-runner`) and the brain (`trex-demo-clm`); 21 s after the 20 s game one `[python] <defunct>` child remained (the finished runner, not yet joined by the server). Fixed in build 13 (`Session.stop()` always joins the runner), see the thirteenth attempt |
| build 13 (published), 20 s game (`demo_20261002T145414Z`) | survived, best 193, 361 decisions, agreement 0.928, answer p50 16.6 ms, model p50 1.5 ms, server p50 0.3 ms, 143 discarded, 0 errors; 3 s after the game the container holds only the server and the multiprocessing resource tracker, `defunct count: 0` |
| build 13 default profile (`package_p150_b13_20261002T145438Z`) | README example cold 98 tokens 218.8 ms, new state 55.8 ms, first 100 Typed Decisions cases 0.294 at 140 ms |
| clean pull of revision 5, 10 s game on port 8703 (`demo_20261002T145626Z`) | image loaded in 18.6 s, healthy 30 s after start, `GET /demo/` 200; survived, best 89, 209 decisions, agreement 0.957, answer p50 16.5 ms, model p50 1.2 ms, server p50 0.2 ms, 42 discarded, 0 errors; `defunct count: 0` after the game |

The games are shorter than the 5 x 60 s evaluation runs and use one seed, so their survival is not a score; the
per-decision latencies (answer p50 16.5 to 16.6 ms, model p50 1.1 to 1.4 ms) match the evaluation runs above
(16.4 and 1.3 ms). The 20 s and 60 s games ran with the headless Firefox and the client on the same host.

## Publish

```
tt-model push /home/hous/dev/clm-v0.1-8B/package/out/clm-v0.1-8b-p150 --public
```

Six revisions of `tt-hous/clm-v0.1-8b-p150` were pushed; all stay in the repo history.

- Revision 1, 2026 Oct 2 00:31 UTC (`/home/hous/dev/clm-v0.1-8B/logs/tt_model_push.log`): repo created public;
  image `9372e4d3d3c4` (build 9, `accuracy` default, built from the dirty main checkout), Hub revision
  `3da3cc872dc36c4d738bbadd42f921d854ddae04`, 124 files, 1,041 MB. Clean pull check passed (`tt_model_pull_clean.log`,
  `evals_pulled_clean.log`: README example cold 260.9 ms / 98 tokens, new state 60.5 ms).
- Revision 2, 2026 Oct 2 01:34 UTC (`tt_model_push_v2.log`), the first clean-tree build: image
  `tt-model/clm-v0.1-8b-p150:6cf6949ed327`, digest `sha256:6cf6949ed327e23b85cabe93b4d45d8520c28f10c7be72d38c94071890ceb4b3`,
  code sha256 `e9ae8402b6337c9cf84fa8c049dea50bcc0422849995c95a95099734a23137c6` (identical to the evaluated build 10),
  built 01:29 to 01:32 UTC from the clean worktree `/home/hous/dev/clm-v0.1-8B/worktree/tt-metal` detached at commit
  `0cca94bc36` of `hous/clm-v0.1-8b` (pushed to github.com/housTT/tt-metal; the manifest says `dirty: false` and
  `branch: HEAD` because the worktree is detached). The image carries only committed sources: the two Ornith kernel
  edits of builds 1 to 10 are gone, and the `p150x4` profile boots from it (table above: healthy 50 s after start,
  README example cold 169.3 ms, new state 34.3 ms, first 100 Typed Decisions cases 0.296 at 106 ms). The `p150`
  profile from this image matches build 10 (README example cold 220.5 ms, new state 56.1 ms, first 100 cases 140 ms).
  Hub revision `2eb9cf5199d2302f5f3e7eeb71d55f39cdacabe5`, 124 files, 1,041 MB; image 963.7 MB in 28 blobs uploaded in 23.0 s.
- Revision 3, 2026 Oct 2 01:59 UTC (`tt_model_push_v3.log`), the current published build: card corrections from review R2
  only (same code sha256 `e9ae8402...`). Image `tt-model/clm-v0.1-8b-p150:4c6e66b2acfc`, digest
  `sha256:4c6e66b2acfca7067e3b39a2aba2dcde979fea9f529cb1d5d528b10d3a1f4108`, built 01:55 to 01:58 UTC from the clean
  worktree at commit `ba61f5a200` (`dirty: false`, `pushed: true`; also the head of `hous/clm-v0.1-8b-release`). Hub
  revision `6796024daaedbcab10ac51314a6894feeb5d0339`, 124 files, 1,041 MB. Default-profile smoke from this image
  (`package_p150_b12_20261002T015817Z`): README example cold 98 219.6, new state 55.8 ms, first 100 Typed Decisions
  cases 0.294 at 140 ms, matching builds 10 and 11. Clean pull check (`tt_model_pull_v3.log`, `evals_pulled_v3.log`,
  `pulled_v3_rank.txt`, `package_pulled_v3_20261002T020032Z`): image loaded in 18.6 s, healthy 30 s after start,
  `/v1/rank` tides example Moon 0.9940 in 137.7 ms, README example cold 98 218.3, new state 56.0 ms.
- Revision 4, 2026 Oct 2 14:44 UTC (`tt_model_push_v4.log`): the T-Rex live demo at `/demo/` (build 12, image
  `tt-model/clm-v0.1-8b-p150:73e7a51f46f4`, digest `sha256:73e7a51f46f44b53a18e6510c979c63e25d4fc327d15d85ca0f80527589e5b5c`,
  code sha256 `9ea9a148d22ee670f0e8bb67d24ddbbaa7db9c4b2fa60047fa0014c5fa408e58`, clean worktree at `29a1d6a715`,
  `dirty: false`). Hub revision `b1e818116df1cc78f30b2016249028abe1585052`. Clean pull check (`tt_model_pull_v4.log`,
  `serve_pulled_v4.log`, `demo_client_pulled_v4.log`): healthy 30 s after start, `GET /demo/` 200, 10 s demo game
  204 decisions at agreement 0.946 (table under "Demo verification").
- Revision 5, 2026 Oct 2 14:55 UTC (`tt_model_push_v5.log`): build 13, the runner join fix
  only (image `tt-model/clm-v0.1-8b-p150:85eee9d0dd8b`, digest
  `sha256:85eee9d0dd8b803d07d2d57f66578c3911a0fde5b3609e2819dc6cb273d05993`, code sha256
  `db04b18c9024e224629cbd6f35779404d71b0430403ea09162fa00ae9fd43c6e`, clean worktree at `6432259fac`, `dirty: false`).
  Hub revision `d57730943327b34c3aaa5753e8a3135bbbfd72f1`, 141 files, 1,041 MB; image 964.0 MB in 28 blobs uploaded in
  20.6 s. Clean pull check (`tt_model_pull_v5.log`, `serve_pulled_v5.log`, `demo_client_pulled_v5.log`): image loaded in
  18.6 s, healthy 30 s after start, `GET /demo/` 200, 10 s demo game 209 decisions at agreement 0.957, no leftover child
  process after the game (table under "Demo verification").
- Revision 6, 2026 Oct 2 18:40 UTC (`tt_model_push_v6.log`), the current published build: build 14, the shield definition
  on the demo page only (image `tt-model/clm-v0.1-8b-p150:f02b781cf8d0`, digest
  `sha256:f02b781cf8d079f50fa4f045fd81a641d6ef868c5a734502381ae8206196459b`, code sha256
  `949aa72e5600889b125e98cd047df93504a603dc9fcf85e7ea270318a0cd7cb8`, clean worktree at `2641f871d7`, `dirty: false`).
  Hub revision `43b663fbac4209ad83323c8352aea21d8a7a0f2b`, 141 files, uploaded in 24.9 s. Checks: the image served on
  chip 0 (`serve_build14_p150.log`) answers `GET /demo/` with the new text, and the Hub copy of
  `code/models/autoports/contrastive_lm_clm_v0_1_8b/clm/demo/index.html` at that revision carries it. No clean pull
  check was run for this revision (two page files changed; the server, engine and lock are those of revision 5).
- Hub tags: blackhole, p150, p150x4, tt-dit-server, tt-model-cache, tt-model-container, text-ranking, license
  apache-2.0, base_model Contrastive-LM/CLM-v0.1-8B (the Hub labels the relation "finetune" by default; the weights
  are the unmodified upstream weights and the card says so). Not listed in the community catalog (`--publish` was not
  passed); `tt-model publish tt-hous/clm-v0.1-8b-p150` adds the catalog pointer if wanted.

Consumer check from a clean local install (`tt_model_pull_v2.log`, `evals_pulled_v2.log`, `pulled_v2_rank.txt`;
every local copy of the image, the `installed.json` entry and the pulled directory removed first):

```
tt-model pull  tt-hous/clm-v0.1-8b-p150
tt-model serve tt-hous/clm-v0.1-8b-p150 --port 8703 --device-id 0 --detach --local-only
```

The pull downloaded and `docker load`ed the image in 20.1 s; the server was healthy 30 s after start; `POST /v1/rank`
on the tides example answers "The Moon's gravitational pull." with probability 0.9940 (136.7 ms, cold cache); the
README example cold call is 98 tokens in 219.2 ms, warm 0.1 ms; a new state against a fixed action set costs 55.5 ms,
cached states 0.1 ms (`/home/hous/dev/clm-v0.1-8B/evals/results/package_pulled_v2_20261002T013530Z`). These match the
build 10 and build 11 numbers above, so the published artifact is the evaluated one.
