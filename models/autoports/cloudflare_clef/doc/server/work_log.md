# Stage 3 (server) work log, host side

Times are ET (UTC-4); the host clock is UTC. Interpreter: `/home/hous/dev/clef/bin/hostrun python` (the clef venv). No device command was run.

## 2026 Oct 05 08:50 ET: reading and probes

Read `/home/hous/dev/clef/AGENT_BRIEF.md`, the plan (stages 3 and 6), `doc/functional/README.md`, `doc/stage0/README.md`, `doc/vision/README.md` (merge and M-RoPE contract), `/home/hous/dev/clef/package/tt-model.yaml` and `package/README.md`, Kev's `tt/server.py`, `tt/api.py`, `tests/test_server_api.py`, `doc/server/README.md`, the release `joint_schema_model.py` (`encode_record`, `_encode_media`, `systemone_answer`, `systemone`), and the current `tt/engine.py` (`ClefEngine`, `prefill_hidden`, `prefill_state(state_ids, slot, key)`, `schema_hidden(handle, schema_ids)`, `cached_hidden`, `probs_for_request`, attributes `snapshot_slots`, `prefix_hidden`, `handles`, `max_tail_len`).

Probe (host): tokenizer load 5.42 s, `Qwen3VLProcessor` load 1.95 s; `encode_record` of the reference PNG `2a7ddcfe...` (323x240) gives 225 tokens with `pixel_values [320, 1536]`, `image_grid_thw [1, 3]`, 83 `mm_token_type_ids`, `token_offset` 36; a 4-frame video as a list of PIL frames or as a `[F, H, W, 3]` array both encode (`pixel_values_videos [392, 1536]`). System prefix 36 tokens, suffix 18. `ttnn.get_device_ids` and `ttnn.get_num_devices` exist in this build.

`fastapi` was not importable from the clef venv (`starlette 1.7.0`, `httpx 0.28.1`, `pydantic 2.9.2`, `pillow 12.3.0` were). Installed the manifest pins: `uv pip install --python /home/hous/dev/clef/tt-metal/python_env/bin/python fastapi==0.142.2 uvicorn==0.54.0` (added `opentelemetry-api 1.45.0`, `typing-inspection 0.4.4`).

## 2026 Oct 05 08:58 ET: files written

- `tt/api.py`: `SystemOneRequest` (`model`, `state`, `questions`, optional `images`, `videos`, `media_kwargs`), `check_release_rules` (the four release messages, then shape rules), `decode_media` (base64, `data:` URL, http(s) with 10 s timeout and 20 MiB cap, `CLEF_ALLOW_REMOTE_IMAGES`), `to_record`, `api_request`.
- `tt/fake_engine.py`: `FakeEngine` with `prefill_hidden`, `prefill_state(state_ids, slot, key, media)`, `schema_hidden`, `prefix_hidden`, `snapshot_slots`, `received` (records every call with the media tensor shapes).
- `tt/server.py`: `Settings`, `parse_mesh_shape`, `mesh_plan`, `open_devices`, `build_engine` (signature-filtered kwargs), `Worker` (LRU prefix cache, per-slot handling), `Server` (encode, limits, dispatch, body, card), routes, middleware, lifespan.
- `tests/test_server_api.py`: 45 tests.

## 2026 Oct 05 09:03 ET: first test run

Command: `cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/hostrun python -m pytest models/autoports/cloudflare_clef/tests/test_server_api.py --timeout=900 -p no:cacheprovider -q`. Outcome: `11 failed, 34 passed in 21.52s`. Two causes: the 422 handler returned pydantic's `errors()` list, whose `ctx` holds a `ValueError` and is not JSON serializable (fixed: the body is `{"detail": "<text>"}` only); the truncate test read slot 0 while the free-slot list pops slot 3 first (test fixed to read the handle from the worker's cache).

## 2026 Oct 05 09:05 ET: second test run

Same command. Outcome: `45 passed, 3 warnings in 20.37s`. Log `/home/hous/dev/clef/logs/stage3_server_api_test.log`.

## 2026 Oct 05 09:06 ET: manifest verify lines

Loop over the four lines that depend on stage 1 to 4 modules through `hostrun python -c`. Log `/home/hous/dev/clef/logs/stage3_manifest_verify_lines.log`: `OK import ... server as s; assert s.app`; `OK ... parse_mesh_shape('1x2', 'P150x2') == (1, 2)`; `OK ... ClefEngine`; `FAIL ... precision_defaults.profile_name() == 'selected'` (stage 4 owns `precision_defaults.py`; `profile_name` does not exist yet).

## 2026 Oct 05 09:07 ET: fake server over HTTP

Command: `cd /home/hous/dev/clef/tt-metal && CLEF_FAKE_ENGINE=1 CLEF_MODEL=<snapshot> HF_HUB_OFFLINE=1 timeout 240 /home/hous/dev/clef/bin/hostrun python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8018 --lifespan on`, log `/home/hous/dev/clef/logs/stage3_fake_server.log`. Startup: `worker 0 ready on fake0, prefix cache 4 state(s), media=True` 3.5 s after `starting`, warmup `latency_ms=61.2`, `Application startup complete`. Requests: `/v1/health` `{"status":"ok","workers":1,"queued":0}`; the README SystemOne example `usage.input_tokens 300`, `latency_ms 57.8` (`/home/hous/dev/clef/logs/stage3_fake_server_checkout.json`); the card's image example with the reference PNG `usage.input_tokens 236`, `latency_ms 43.1` (`stage3_fake_server_image.json`), log line `S=128 tail=108 questions=1 images=1`; a `choice` without criteria `{"detail":"q: criteria must not be empty"}`; `/v1/models` card with `backend fake`, `mesh_shape 1x2`, revision `2f3de3dd...`. SIGTERM: `Application shutdown complete`, `Finished server process`.

## 2026 Oct 05 09:10 ET: docs

Wrote `NOTICE` (Apache-2.0; vendored head classes from the release, Kev server design), `doc/server/README.md` (routes, design, engine interface, env table, run commands, tests), this log.

## 2026 Oct 05 15:05 to 15:35 ET: stage 3 review remediation, host side (review `/home/hous/dev/clef/reports/review_stage3.md`)

Orchestrator decision on P1: the shipped default is the eager engine (`CLEF_TRACED=0`); tracing stays an opt-in for deployments with a fixed set of image grids. Changes in `tt/server.py`, `tt/api.py`, `tt/engine.py`, `tests/test_server_api.py`:

- `CLEF_TRACED` default `1` to `0`. New `CLEF_PLANNER` (default `1`, passed to the engine as `planner=`).
- Traced opt-in: the engine requires `CLEF_VISION_WARM_GRID` (`ValueError` with the reference grid list at build when it is unset or empty), `_vision_request` refuses a grid outside the warm list before any tower op (`check_warm_grids`), the worker reads `engine.vision_warmed_grids` once after the build, `Server.encode` refuses a request whose `image_grid_thw` / `video_grid_thw` is outside the union of the workers' warm lists with 422 (`warm_grid_message`: the grid, the list, the three remedies) on the request thread, and a `ValueError` from `prefill_state` is returned as 422 with the slot back on the free list.
- `/v1/models`: `mode`, `gdn_conv`, `prefix_planner`, `traced_media` (`warm_grids`, `rule`). The ready line and the per-worker ready line print the same values.
- `media_kwargs` folded into the media digest of the prefix-cache key (`tt/api.py` `decode_media`, canonical JSON, only when the request has media).
- Over-long string literals split (`server.py` 422 and log messages, `api.py` noul rule) so black at the repo pin (23.10.1) and the 120-column limit agree; `pytest.raises` replaced by the repo `expect_error` fixture at the four sites (the two `mesh_plan` cases match `CLEF_PARENT_MESH=` and `unsupported mesh shape`).
- New tests: `test_media_kwargs_are_part_of_the_cache_key`, `test_traced_server_refuses_grids_outside_the_warm_list` (fake engine, the worker's warm list set by hand), `test_engine_value_error_is_a_422`; `test_models` checks the new card fields.

Pre-commit at the repo pins on the ten stage 3 files (`black` 23.10.1, `isort` 5.13.2, `autoflake` v2.3.1, `trailing-whitespace`, `end-of-file-fixer`, `prefer-expect-error`): every hook `Passed` on the second pass (`/home/hous/dev/clef/logs/stage3r_precommit.log`; a first pass through `pre-commit run <hook> --files ...` was run per hook). Note: in zsh `$FILES` does not word-split; the first attempt reported `(no files to check)` for every hook until `${=FILES}` was used.

Tests: `cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/hostrun python -m pytest models/autoports/cloudflare_clef/tests/test_server_api.py --timeout=900 -p no:cacheprovider -q`: first run `2 failed, 46 passed` (a two-frame video of the reference PNG has grid `(1, 16, 20)`, the same as the image, so it was in the warm list the test expected to fail; and a slot-accounting assertion that assumed an empty cache), second run `48 passed, 3 warnings in 23.84s` (`/home/hous/dev/clef/logs/stage3r_server_api_test.log`).

## 2026 Oct 05 17:40 to 17:55 ET: second review remediation, host side (review `/home/hous/dev/clef/reports/review_stage3_r2.md`, P2: a multi-image request on the traced opt-in compiled the per-image `ttnn.concat` with traces live)

Decision: refuse more than one image or video grid per request on a traced server (option 2 of the review); multi-image and multi-video requests stay supported on the eager default. Changes:

- `tt/engine.py`: `single_grid_message(n_grids)`; `check_warm_grids` first counts the grid rows and raises `ValueError` with that message when there is more than one, before the warm-list loop. `_vision_request` already calls `check_warm_grids` in traced mode before any tower op, so the refusal happens before `get_image_features` / `get_video_features`.
- `tt/server.py`: the same `single_grid_message`; `Server.check_warm_grids` collects the rows of `image_grid_thw` and `video_grid_thw`, returns 422 with the single-grid message when there is more than one row, then applies the warm-list rule per row; `/v1/models` `traced_media.rule` now starts with "one image or video grid per request".
- `tests/test_server_api.py`: `test_traced_server_refuses_more_than_one_grid_per_request` (two images and two one-frame videos return 200 with the warm list unset and the engine receives `image_grid_thw` of shape `[2, 3]`; with the worker's warm list `[(1, 16, 20)]` both return 422 with the single-grid message and no engine call; the card rule starts with the rule; a one-image request still returns 200).
- `tests/test_engine_traced.py`: `test_two_grid_request_refused` (device side, `doc/optimized/work_log.md`).
- Docs: `doc/server/README.md` (errors paragraph: the two media grid rules; env row `CLEF_TRACED`; test list; the 13:45 ET section no longer calls `CLEF_TRACED=1` the default), `doc/optimized/README.md`, `doc/benchmark/README.md`, `doc/context_contract.json`, `/home/hous/dev/clef/package/README.md` lines 17, 99, 106, 107 and 201 (`CLEF_TRACED` placeholders to `0`), `/home/hous/dev/clef/package/tt-model.yaml` line 204 (the rule text).

Tests: `cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/hostrun env CLEF_MODEL=<snapshot> HF_MODEL=<snapshot> OMP_NUM_THREADS=8 python -m pytest models/autoports/cloudflare_clef/tests/test_server_api.py --timeout=900 -p no:cacheprovider -q`: `49 passed, 3 warnings in 24.35s` (21:49 UTC, `/home/hous/dev/clef/logs/stage3r2_server_api_test.log`). Pre-commit at the repo pins on the eight touched tt-metal files (`python_env/bin/pre-commit run --files ${=FILES}`): `/home/hous/dev/clef/logs/stage3r2_precommit.log`.
