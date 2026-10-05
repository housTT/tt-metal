# Clef stage 3 (server): host side

Date: 2026 Oct 05, 08:50 to 09:10 ET (the host clock is UTC). Host side only: no Tenstorrent device was opened and `devrun` was not used. All paths are absolute. The device side of stage 3 (traces, real-server parity, bench) is owned by the device agent and is not described here.

Code: `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/tt/server.py` (ASGI `app`, FastAPI), `tt/api.py` (request model, release validation rules, image decoding), `tt/fake_engine.py` (CPU stand-in for the engine). Reference it mirrors: `systemone()` and `systemone_answer()` in `/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c/joint_schema_model.py` (lines 523 to 584). Template: `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/server.py` and its `doc/server/README.md`.

Manifest contract this satisfies: `/home/hous/dev/clef/package/tt-model.yaml` (`runtime.app models.autoports.cloudflare_clef.tt.server:app`, `mesh_shape_env CLEF_MESH_SHAPE`, the two server `verify:` lines, the card's routes and env table). The tt-model launcher starts `python -m uvicorn --host 0.0.0.0 --port 8008 --lifespan on models.autoports.cloudflare_clef.tt.server:app` with `HF_MODEL=Cloudflare/clef`, `MESH_DEVICE=P150x2`, `CLEF_MESH_SHAPE=1x2`, and reads readiness from `Application startup complete`, so the server opens the mesh and loads the weights inside the ASGI lifespan.

## Routes

| Route | Body |
|---|---|
| `POST /v1/systemone` | Exactly the body of the release `systemone()`: `{"model", "answers", "usage": {"input_tokens", "output_tokens": 0}}`, plus `latency_ms`. Answer shapes and rounding come from the release `systemone_answer()`, which the server calls on the per-question probabilities: `noul` (`noul`), `choice` (`choice`, `confidence`, `probabilities` in the order of the request's `criteria`), `score` (`score`, `confidence`, `legend`, `probabilities`). With `CLEF_TRUNCATE_STATES=1` the body also carries `truncated` and `usage.state_tokens` / `usage.state_tokens_used`. |
| `GET /v1/models` | One card per name in `CLEF_MODEL_NAMES` (default `clef`): `description`, `weights` (`repo`, `revision`, `snapshot`), `device`, `mesh_shape`, `mesh_plan`, `backend` (`ttnn` or `fake`), `precision` (the `QWEN36_*` knobs), `traced`, `mode` (`eager` or `traced`), `gdn_conv` (the GDN causal conv kernel the engine runs: `fir` in both shipped modes, see `doc/optimized/README.md`), `prefix_planner`, `traced_media` (`null` in eager mode; in traced mode `warm_grids`, the `(t, h, w)` grids compiled before capture, and `rule`, the 422 rule for every other grid), `max_state_tokens`, `max_schema_tokens`, `truncate_states`, `remote_images`, `prefix_cache` (pooled `size`, `hits`, `misses`, `cached_states`), `workers` (per worker `queued`, `cached_states`, `cache_size`, `hits`, `misses`, `requests`, `media`), `requests`, `uptime_s`. |
| `GET /health`, `GET /v1/health` | `{"status": "ok", "workers", "queued"}`. Never behind the API key. |

What differs from the release `systemone()`: nothing in the body except the added `latency_ms` (and the truncation fields when that mode is on). The release function runs on PIL images in a Python process; over HTTP each image is a string (base64, `data:image/...;base64,` URL, or http(s) URL) and each video a list of frame strings, decoded by the server. The release truncates a long state silently to `max_length` 16384; this server returns 422 for a state over `CLEF_MAX_STATE` tokens unless `CLEF_TRUNCATE_STATES=1`, and 422 for a schema over `CLEF_MAX_TAIL` tokens. The release's `output_tokens` is 0 and stays 0 here. `/v1/systemone/permute` and `/v1/systemone/separate` are not served (404).

Every response carries `x-typesafe-request-id` (echoed from the request when given) and `server-timing`. `CLEF_API_KEY` set means `/v1/*` (except the health routes) needs `Authorization: Bearer <key>`, else 401. `latency_ms` is the model time of the request on its worker (state prefill or cache hit, schema continuation, head on the host), not its wait in the queue or the HTTP time.

### Errors (422, `{"detail": "<text>"}`)

The release rules of `systemone()` are checked first, in its order and with its wording: `model and state are required`; `at least one question is required`; `<id>: type must be noul, choice, or score`; `<id>: criteria must not be empty` (for `choice` and `score`). Then the shape rules (own wording): `<id>: choice criteria must be a mapping of option id to description`, `<id>: score criteria must be a list of ordered option descriptions`, `<id>: noul criteria must be a mapping with optional true and false descriptions`. Then media: `images[i]: ...` and `videos[i][j]: ...` for invalid base64, undecodable image bytes, a non-base64 `data:` URL, a fetch failure, an image over 20 MiB, or a remote URL when `CLEF_ALLOW_REMOTE_IMAGES=0`. Then, on a traced server only (`CLEF_TRACED=1`), the two media grid rules, both checked on the request thread from `image_grid_thw` / `video_grid_thw` before any device work. The single-grid rule: `this request carries N image or video grids; a traced server (CLEF_TRACED=1) takes one grid per request, because the tower joins several images with a concat program compiled per image count and no vision program may compile while traces are live: send one image (or one video) per request, or serve with CLEF_TRACED=0 (the default), which accepts several` (`single_grid_message`, from the row count of `image_grid_thw` plus `video_grid_thw`; added for the P2 of the second review, `/home/hous/dev/clef/reports/review_stage3_r2.md`: `tt/vision.py` `forward_with_taps` joins the per-image tower outputs with `ttnn.concat`, a program the one-grid-per-call warm-up never compiles). The warm-list rule: `image grid (t, h, w) (t, h, w patches) is not in this traced server's warm list [...]: resize the image to a warmed grid, add the grid to CLEF_VISION_WARM_GRID at startup, or serve with CLEF_TRACED=0 (the default), which accepts any grid` (`warm_grid_message`). The engine repeats both checks in `_vision_request` (`check_warm_grids`), and a `ValueError` from the engine's `prefill_state` is also returned as 422, so no vision program is compiled while traces are live. The eager default has neither rule: it accepts any grid and several images or videos per request. Then the limits: `state is N tokens (K of them the system prompt and media placeholders), over the 16,384-token limit: ... CLEF_TRUNCATE_STATES=1 ...` and `schema is N tokens, over the 4,096-token limit: ...`. A `ValueError` from the release `encode_record` (for example `schema requires N tokens before state; maximum is M`) is also a 422 with the release text. The FastAPI validation handler flattens pydantic's error list to the first message, so the body is always `{"detail": "<text>"}`.

## Design

```
request thread                       worker thread k (one per TP group)        mesh k (1x2, TP=2)
  SystemOneRequest (422 here)        queue -> LRU lookup on key
  decode images (PIL), media digest           miss: prefill_state(state_ids, slot, key, media)  ---> ttnn
  encode_record (release code)                hit:  reuse StateHandle
  split_for_cache -> state | tail             schema_hidden(handle, tail_ids)                   ---> ttnn
  key = sha1(state ids [+ media])             hidden = prefix_hidden[slot] ++ tail rows
  limits (422)                                probs_for_record (JointSchemaHead, CPU fp32)
  pick worker                        <- done  systemone_answer per question (release code)
  await future, body + latency_ms
```

- Startup runs inside the ASGI lifespan: resolve the snapshot (`CLEF_MODEL`, else `HF_MODEL`, a local directory or a Hub id resolved with `snapshot_download` at `CLEF_REVISION`, `local_files_only` when `HF_HUB_OFFLINE=1`; the resolved directory is written back to `CLEF_MODEL` so `tt/loader.py` finds it), load the tokenizer and the fp32 `JointSchemaHead` on the host, open the mesh, build one engine per TP group on that group's own worker thread, then one warmup request per worker (the blog curl example). The image processor (`Qwen3VLProcessor`) loads lazily on the first request with media (about 2 s). uvicorn prints `Application startup complete` only after that.
- Mesh open (`mesh_plan`, `open_devices`): the target shape is `parse_mesh_shape(CLEF_MESH_SHAPE, MESH_DEVICE)`. For `1x2`: when exactly 2 chips are visible (`ttnn.get_device_ids()`), `FABRIC_1D` and a direct `open_mesh_device(MeshShape(1, 2))` (a real two-chip host); when more are visible, or `CLEF_PARENT_MESH` is set, the stage 0 path A: `FABRIC_1D`, open the `(1, 4)` parent, `create_submesh(MeshShape(1, 2), MeshCoordinate(*CLEF_SUBMESH_OFFSET))` (default `0,0`, chips `[1, 0]` on this box); `CLEF_PARENT_MESH=2x2` selects path B (`FABRIC_2D`, `(2, 2)` parent). For `1x4`: direct open under `FABRIC_1D` and two `(1, 2)` submeshes at offsets `(0, 0)` and `(0, 2)`, one worker each (stage 7, not validated). For `2x2`: `FABRIC_2D` and submeshes at `(0, 0)` and `(1, 0)`. Every open uses `l1_small_size=24576`, `num_command_queues=2`, `trace_region_size=CLEF_TRACE_REGION`. Close order (stage 1 finding): synchronize each group, close `parent.get_submeshes()`, close the parent, `set_fabric_config(DISABLED)`. The plan is a pure function (`mesh_plan(mesh_shape, visible, parent, offset)`) and is unit tested; `open_devices` itself has not run on a device yet.
- One worker thread per TP group. A request goes whole to one worker (Clef decides every field jointly in one pass, so there is no per-question fan-out as in Kev): the worker whose prefix cache holds the state key, else the least loaded (queue depth plus in flight, ties by id). Every ttnn call for a mesh happens on its worker thread only; the main thread opens and closes the meshes.
- Prefix cache: per worker, least-recently-used map from the state key to `(StateHandle, slot)`, at most `CLEF_PREFIX_CACHE` entries (default 4), clamped to `engine.snapshot_slots`. The key is `cache_key(state_ids)` (sha1 over the little-endian int32 ids of the first piece of `split_for_cache`: system prompt, media placeholders, state); when the request has images or videos the key is re-hashed with the sha1 of the raw image bytes and, when present, the canonical JSON of `media_kwargs` (the processor options the release passes into `Qwen3VLProcessor`, which can change pixel values without changing the token count), because the `<|image_pad|>` ids are identical for every image of the same grid (`doc/vision/README.md`, contract item 2). The tests `test_image_bytes_are_part_of_the_cache_key` and `test_media_kwargs_are_part_of_the_cache_key` check that two requests with the same ids and different images, or the same image and different `media_kwargs`, are two misses. A miss takes a free slot, else evicts the least recently used entry and reuses its slot, then calls `prefill_state(ids, slot=slot, key=key)`; if that raises, the slot returns to the free list and the error propagates (500). Only the 128-aligned prefix of a state is cached (`StateHandle.S0`); the remainder is recomputed with the schema tail on every request, as in the engine's `schema_hidden`. `CLEF_PREFIX_CACHE=0` disables the cache and uses `prefill_hidden` over the whole sequence.
- Admission: the request is encoded once with the release `encode_record` (`max_length` set to 2^31 so the release does not truncate), split with `split_for_cache`. `S` = tokens of the first piece, `L` = tokens of the second (schema plus suffix). `S > CLEF_MAX_STATE` gives 422, or with `CLEF_TRUNCATE_STATES=1` a second `encode_record(max_state_tokens=CLEF_MAX_STATE - fixed)` that keeps the first tokens of the state (`fixed` = system prompt plus media placeholder tokens, 36 for text), and the body says `truncated: true`. `L > CLEF_MAX_TAIL` (default 4096, the engine's `max_tail_len`) gives 422. So `S + L <= max_state + max_tail`, the engine's `max_len`.
- Sync entry points for tests and tools: `Server.predict(record) -> (probs, stats)`, `Server.answer(req) -> body`. The HTTP handler awaits a `Future` per worker queue (`answer_async`), so a waiting request holds no thread.
- One log line per request: `worker=<id> S=<state tokens used> tail=<schema tokens> questions=<n> images=<n> videos=<n> latency_ms=<ms> cache_hit=<bool>`. The ready line (`ready: N worker(s) on <device>, mode=<eager|traced>, gdn_conv=<fir|kda>, planner=<bool>, warm_grids=<list or None>`) and the per-worker ready line carry the engine mode, the GDN conv kernel and the warm-grid list, the same values `/v1/models` reports.

## Engine interface

The server constructs the engine through `build_engine(mesh, settings)` and calls only the methods below. The device agent's `tt/engine.py` must keep these names and signatures (they are the ones the file exposes at the time of writing, plus the `media` keyword, which is new). `tt/fake_engine.py` implements the same interface on the CPU and is what `CLEF_FAKE_ENGINE=1` swaps in.

| Member | Used by the server as | Notes |
|---|---|---|
| `ClefEngine(mesh, **kwargs)` | `build_engine`: passes `max_state_len=CLEF_MAX_STATE`, `max_tail_len=CLEF_MAX_TAIL`, `snapshot_slots=max(1, CLEF_PREFIX_CACHE)`, `traced=CLEF_TRACED`, each only if `ClefEngine.__init__` has that parameter (`accepted_kwargs` inspects the signature). | `chunk_size` stays at the engine default (1024, stage 0 finding 2). A `traced` parameter is optional; when it does not exist the engine decides. |
| `engine.snapshot_slots` (int) | clamps the per-worker cache size | the engine's `_fit_slots` result |
| `engine.prefill_state(state_ids, slot=0, key=None, media=None, tail_len=None) -> StateHandle` | cache miss (a `ValueError` is returned to the client as 422; any other exception is a 500 and the slot returns to the free list in both cases) | `state_ids` is `[1, S]` long with `S <= max_state_len`; `key` is the server's cache key (store it on the handle; do not recompute it from the ids, the server's key includes the image bytes); `media` is `EncodedRecord.media` from the release `encode_record` (`pixel_values`, `image_grid_thw`, `pixel_values_videos`, `video_grid_thw`, `mm_token_type_ids`, `token_offset`) or `None` for text. The server passes `media=` only when the signature has that parameter; an engine without it gets 422 `this server build does not accept images or videos` for media requests and works for text. |
| `tail_len` (stage 3 addition) | `Worker._serve` passes `tail_len=len(job.tail_ids)` when the signature has it (`supports_tail_len`) | the engine's prefix planner (`plan_prefix`, eager and traced, `CLEF_PLANNER`) uses it to choose the cached prefix length; ignored by the fake engine. |
| `engine.traced`, `engine.gdn_conv_impl`, `engine.planner`, `engine.vision_warmed_grids` (stage 3 remediation) | read once after the build for the ready log lines, `/v1/models` and the traced warm-grid check | optional; the fake engine has none of them (the card then shows `mode eager`, `gdn_conv null`, `traced_media null`). |
| `engine.schema_hidden(handle, schema_ids) -> Tensor [S - S0 + L, 5120] fp32` | after a hit or a miss | `schema_ids` is `[1, L]` long, the second piece of `split_for_cache` (schema plus suffix). Returns the normalized final-RMSNorm rows for the uncached state remainder plus the tail, exactly as the current engine does. |
| `engine.prefix_hidden[slot] -> Tensor [S0, 5120] fp32` | concatenated before the `schema_hidden` rows to form `[T, 5120]` for the head | indexed by the slot of the handle; a list or a dict. |
| `engine.prefill_hidden(token_ids, slot=0, media=None) -> Tensor [T, 5120] fp32` | only with `CLEF_PREFIX_CACHE=0` | whole-sequence path. |

The head call is `models.autoports.cloudflare_clef.tt.head.probs_for_record(head, hidden, input_ids, encoded, LmHeadRows(snapshot))` on the worker thread; the server owns the head, the tokenizer, the processor and the `LmHeadRows` reader (one per worker), so the engine's own `tokenizer`, `head` and `probs_for_request` members are not used by the server. The server never calls `cached_hidden`, because its key check ignores image bytes.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CLEF_MODEL` | unset | Snapshot directory of `Cloudflare/clef`, or a Hub id (`Cloudflare/clef`, `Cloudflare/clef@<rev>`). Read before `HF_MODEL`. The server writes the resolved directory back into `CLEF_MODEL`. |
| `HF_MODEL` | `Cloudflare/clef` | Fallback for `CLEF_MODEL`; tt-model sets it to `Cloudflare/clef`. |
| `CLEF_REVISION` | `2f3de3dd85f379784083b0814d997ab627200f0c` | Revision used when `CLEF_MODEL`/`HF_MODEL` is a Hub id without `@rev`. |
| `HF_HUB_OFFLINE` | unset | `1` resolves Hub ids from the local cache only. |
| `CLEF_MESH_SHAPE` | unset | Target mesh `RxC`: `1x2` (one TP=2 worker), `1x4` or `2x2` (two TP=2 workers, stage 7). tt-model sets it from the profile's `mesh_device`. |
| `MESH_DEVICE` | unset | When `CLEF_MESH_SHAPE` is unset: `P150x2`, `P300` give `1x2`; `P150x4`, `P300x2` give `1x4`; `QB2` gives `2x2`; anything else `1x2`. |
| `CLEF_PARENT_MESH` | unset (auto) | `1x4` (path A, `FABRIC_1D`) or `2x2` (path B, `FABRIC_2D`): forces the parent-plus-submesh open for a `1x2` target. Auto picks `1x4` when more than 2 chips are visible. |
| `CLEF_SUBMESH_OFFSET` | `0,0` | Offset of the `1x2` submesh inside the parent. |
| `CLEF_TRACE_REGION` | `1073741824` when `CLEF_TRACED=1`, else `0` | `trace_region_size` for the mesh open. |
| `CLEF_TRACED` | `0` (eager; was `1` before the stage 3 review) | Passed to the engine as `traced=`. `0` is the shipped default: the eager engine accepts any image grid. `1` is an opt-in for a deployment with a fixed set of image grids: it needs `CLEF_VISION_WARM_GRID`; an image or video whose grid is not in that list, and a request with more than one image or video, are refused with 422 so that no vision program compiles while traces are live (`doc/optimized/README.md`, "Why eager is the default"). |
| `CLEF_VISION_WARM_GRID` | unset (required when `CLEF_TRACED=1` and the tower is on) | `;`-separated `t,h,w` patch grids the traced server warms before capture and then accepts, for example `1,16,20;1,22,38;1,26,36;1,28,36;1,28,38;1,40,50;2,16,20` (the 8 reference images and the video, `tt/engine.py` `REFERENCE_GRIDS`). The engine refuses to start traced without it. Ignored in eager mode. |
| `CLEF_PLANNER` | `1` | `0` turns the prefix planner off (`ClefEngine.plan_prefix` then returns the stage 1 rule `S0 = (S // 128) * 128`). Reported as `prefix_planner` in `/v1/models`. |
| `QWEN_GDN_CONV` | `fir` when `CLEF_TRACED=1` (set by the engine with `setdefault`); not read by the eager path | The GDN causal conv kernel of the unmasked traced body. The eager masked buckets always run the FIR kernel. A traced server started with `QWEN_GDN_CONV=kda` in its environment runs the fused KDA kernel, which shifted the 16-record parity to max dp 0.1077 with one flip (`doc/optimized/README.md`); the value in use is in the ready log line and in `/v1/models` as `gdn_conv`. |
| `CLEF_PREFIX_CACHE` | `4` | States kept per worker (clamped by `engine.snapshot_slots`); `0` disables the cache. |
| `CLEF_MAX_STATE` | `16384` | Engine `max_state_len` and the 422 limit for the state piece (system prompt and media placeholders included). |
| `CLEF_MAX_TAIL` | `4096` | Engine `max_tail_len` and the 422 limit for the schema plus suffix. |
| `CLEF_TRUNCATE_STATES` | `0` | `1` keeps the first `CLEF_MAX_STATE` tokens instead of refusing and marks the response. |
| `CLEF_API_KEY` | unset | Bearer key for `/v1/*` (health routes stay open). Read at import time; tests patch `server.API_KEY`. |
| `CLEF_ALLOW_REMOTE_IMAGES` | `1` | `0` refuses http(s) image URLs with 422. Fetches use a 10 s timeout and a 20 MiB cap. |
| `CLEF_MODEL_NAMES` | `clef` | Comma-separated names listed by `/v1/models`. |
| `CLEF_FAKE_ENGINE` | `0` | `1` opens no device: `CLEF_FAKE_WORKERS` (default 1) workers return deterministic pseudo-random hidden rows, the real head, tokenizer and processor run. For API tests on a CPU. |
| `CLEF_WARMUP` | `1` | `0` skips the warmup request per worker. |
| `QWEN36_*` | set by `tt/precision_defaults.py` | Dtype knobs applied when `tt/engine.py` is imported (real engine only). |

## Run

Real server on this box (4 chips visible, so path A: `FABRIC_1D`, `(1, 4)` parent, `(1, 2)` submesh at `(0, 0)`), from the tt-metal env, under the device lock; the shipped configuration is the default (`CLEF_TRACED=0`, `CLEF_PLANNER=1`):

```bash
export CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
export HF_HUB_OFFLINE=1 CLEF_MESH_SHAPE=1x2 MESH_DEVICE=P150x2 TT_CACHE_PATH=/home/hous/dev/clef/tt_cache OMP_NUM_THREADS=8
cd /home/hous/dev/clef/tt-metal && nohup /home/hous/dev/clef/bin/devrun timeout 7200 python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage3_server.log 2>&1 &
```

Traced opt-in instead: add `CLEF_TRACED=1 CLEF_VISION_WARM_GRID="1,16,20;1,22,38;1,26,36;1,28,36;1,28,38;1,40,50;2,16,20"` (or the grids of the deployment's images). Path B instead: add `CLEF_PARENT_MESH=2x2`. A real two-chip host (p150x2 or one p300 board with two visible chips) needs no extra variable: the server opens `(1, 2)` directly. Stop with `kill -TERM` to the python process (not the `timeout` wrapper); the lifespan closes the submeshes, then the parent, then disables the fabric.

Fake server (no device; API tests and plumbing checks):

```bash
cd /home/hous/dev/clef/tt-metal && CLEF_FAKE_ENGINE=1 CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c HF_HUB_OFFLINE=1 /home/hous/dev/clef/bin/hostrun python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8018 --lifespan on
```

Request (the HF README's SystemOne example):

```bash
curl -s http://127.0.0.1:8018/v1/systemone -H 'content-type: application/json' -d '{"model": "clef", "state": "Our checkout started returning errors and orders are blocked.", "questions": {"department": {"type": "choice", "instructions": "Which team should handle the message?", "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}}, "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]}, "outage": {"type": "noul", "instructions": "Is a service down?"}}}'
```

Image request: `IMG=$(base64 -w0 /home/hous/dev/clef/reports/reference/images/2a7ddcfe4724ee1403a6291d21347162.png)` then `"images": ["$IMG"]` in the body (the package card's quickstart shows the full command).

## Demo

A stand-alone web page and command-line client for this server (presets, image requests, prefix-cache demonstration, labelled replay) live in `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/`; see `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/README.md`.

## Tests

```bash
cd /home/hous/dev/clef/tt-metal && /home/hous/dev/clef/bin/hostrun python -m pytest models/autoports/cloudflare_clef/tests/test_server_api.py --timeout=900
```

Result on 2026 Oct 05, 09:05 ET: `45 passed, 3 warnings in 20.37s` (`/home/hous/dev/clef/logs/stage3_server_api_test.log`); after the stage 3 review remediation (15:30 ET): `48 passed, 3 warnings in 23.84s` (`/home/hous/dev/clef/logs/stage3r_server_api_test.log`); after the second review's P2 remediation (17:49 ET, the single-grid test): `49 passed, 3 warnings in 24.35s` (`/home/hous/dev/clef/logs/stage3r2_server_api_test.log`; the warnings are the starlette `httpx` deprecation and two SWIG import notices from ttnn). The file sets `CLEF_FAKE_ENGINE=1` itself and uses the local snapshot. Every expected error goes through the repo `expect_error` fixture (`/home/hous/dev/clef/tt-metal/conftest.py`), and the pinned pre-commit hooks pass on the stage 3 files (`/home/hous/dev/clef/logs/stage3r_precommit.log`). It covers:

- the two HF README examples and the blog curl example: 200, the exact field sets of the release body plus `latency_ms`, `output_tokens` 0, option order equal to the request's `criteria`, `choice`/`confidence`/`score` consistent with `probabilities`, and `usage.input_tokens` equal to the CPU reference counts (260, 300, 346 from `/home/hous/dev/clef/reports/reference/readme_examples.json` and `ref_text_bf16.jsonl`), which the engine cannot influence;
- `answers` equal to `release.systemone_answer(question, probs)` computed on the same probabilities (`Server.predict` plus `Server.answer`), so ordering and 4-decimal rounding are the release's;
- a `score` question returns `score`, `legend`, `probabilities`, `confidence`; a one-level score gives `score 0.0`, `confidence 1.0`;
- the four release validation messages (10 request variants), checked both through HTTP (`detail` equals the release text) and against `check_release_rules`, plus a check that the four strings exist in the snapshot's `joint_schema_model.py`;
- oversize state 422 with the `CLEF_TRUNCATE_STATES=1` hint; oversize schema 422; `CLEF_TRUNCATE_STATES=1` path (`truncated: true`, `state_tokens_used` 16384, `handle.S` 16384);
- prefix cache: determinism and hits, eviction over 2 slots, slot return after a failed `prefill_state`;
- bearer auth on and off (health routes stay open);
- image request with a base64 PNG and with a `data:` URL: the fake engine records `pixel_values [320, 1536]`, `image_grid_thw [1, 3]`, `mm_token_type_ids` 83 entries, `token_offset` 36 (the values the processor produced for that PNG in the stage 1 probe); the repeat is a cache hit; two different images with the same ids are two misses; a two-frame video gives `pixel_values_videos`; bad payloads and disabled remote URLs give 422;
- the traced warm-grid rule with the fake engine (`test_traced_server_refuses_grids_outside_the_warm_list`: with the worker's warm list set to `[(1, 22, 38), (2, 16, 20)]` the reference PNG, grid `(1, 16, 20)`, is refused with 422 and the message names the grid, the list and `CLEF_TRACED=0`; `/v1/models` shows the list under `traced_media`; with `(1, 16, 20)` in the list the same request and a two-frame video of it, grid `(1, 16, 20)`, return 200; with the list `[(1, 22, 38)]` the video is refused); the traced single-grid rule (`test_traced_server_refuses_more_than_one_grid_per_request`, second review P2: two images of the reference PNG and two one-frame videos of it return 200 with the warm list unset, the engine receiving `image_grid_thw` of shape `[2, 3]`; with the worker's warm list `[(1, 16, 20)]` both return 422 with the single-grid message and no engine call, `/v1/models` `traced_media.rule` starts with the rule, and the one-image request still returns 200); an engine `ValueError` on a miss returned as 422 with the slot accounting intact (`test_engine_value_error_is_a_422`); `media_kwargs` in the cache key;
- `/v1/models` (including `mode`, `prefix_planner`, `gdn_conv`, `traced_media`) and the health shapes; concurrent requests on one worker; two fake workers;
- `parse_mesh_shape` table (10 rows, including the manifest's verify line), `mesh_plan` table, `accepted_kwargs`.

HTTP smoke with uvicorn (`/home/hous/dev/clef/logs/stage3_fake_server.log`, 09:07 ET): startup 3.5 s (tokenizer 5.4 s in the first probe; cached afterwards), warmup `latency_ms` 61.2, the README SystemOne example `input_tokens` 300 and `latency_ms` 57.8 (`/home/hous/dev/clef/logs/stage3_fake_server_checkout.json`), the image example `input_tokens` 236 (`stage3_fake_server_image.json`), a `choice` question without criteria `{"detail": "q: criteria must not be empty"}`, clean `Application shutdown complete` on SIGTERM.

Manifest verify lines (`/home/hous/dev/clef/logs/stage3_manifest_verify_lines.log`): the two server lines and the `ClefEngine` line pass through `hostrun`; the `precision_defaults.profile_name() == 'selected'` line still fails because stage 4 owns that function (the package README already records this).

## Open items

1. Closed in stage 3: `open_devices` ran on this box with the shipped server (`/home/hous/dev/clef/logs/stage3r_server_shipped.log`: 4 chips visible, FABRIC_1D parent [1, 4], 1 TP group [[1, 0]], clean close). The direct (1,2) open on a two-chip host is still unexercised; stage 6 validates it inside the two-chip container.
2. Closed in stage 2: `prefill_state` and `prefill_hidden` take `media`; image and video requests are served (see "Real server").
3. `fastapi==0.142.2` and `uvicorn==0.54.0` were installed into `/home/hous/dev/clef/tt-metal/python_env` with `uv pip install` (the pins of the manifest); they were missing from the stage 0 venv.
4. Closed in stage 3: `doc/optimized/perf_summary.json` exists (manifest allowlist).

## Real server (device side, 2026 Oct 05, 13:45 ET onward)

Device agent run of the server on this box: path A (4 chips visible, `FABRIC_1D`, `(1, 4)` parent, `(1, 2)` submesh at `(0, 0)`), traced engine (`CLEF_TRACED=1`, the default at that time; since the review remediation of 15:00 ET the shipped default is `CLEF_TRACED=0`, `doc/optimized/README.md` "Why eager is the default"; the trace layer is described in `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/doc/optimized/README.md`). Command (`/tmp/claude-1002/-home-hous-dev-clef/2a64a29f-04fe-4cd3-b935-9017907901d0/scratchpad/start_server.sh`, the "Run" command above with `CLEF_TRACED=1` and `timeout 14400`), log `/home/hous/dev/clef/logs/stage3_server.log`.

### Startup

`starting:` at 17:45:01 UTC, `mesh: 4 chip(s) visible, FABRIC_1D parent [1, 4] chips [1, 0, 3, 2], 1 TP group(s) [[1, 0]]` at 17:45:05 (open item 1 of the host section closed: the log line is the predicted one), `ClefEngine ready: load 102.7 s` (state dict 1.8 s, model build 86.2 s, tower 9 s, tower warm on the 7 default grids 0.6 s, trace warmup 3.4 s, 13 captures 0.8 s, trace region used 130.9 MiB), `worker 0 ready on [1, 0], prefix cache 4 state(s), media=True`, `warmup worker=0 questions=3 latency_ms=271.5`, `ready: 1 worker(s) on ttnn 1x2: 1 TP=2 worker(s)`, then `Application startup complete.` at 17:46:48 UTC: 107 s from `starting` to ready, and the ready line comes after the engine and the warmup request. DRAM free per device after the slots 10.66 GiB, after the traces 10.59 GiB (the 1 GiB trace region is reserved at the mesh open; `get_memory_view` numbers from the engine log). `/health` returns `{"status": "ok", "workers": 1, "queued": 0}`; `/v1/models` reports `device "ttnn 1x2: 1 TP=2 worker(s)"`, `backend ttnn`, `traced true`, `max_state_tokens 16384`.

### HTTP checks (`/home/hous/dev/clef/logs/stage3_server_checks.log`, bodies and responses under `/home/hous/dev/clef/reports/stage3_server/`)

| Request | `usage.input_tokens` | `latency_ms` (pass 1 / pass 2) | Served answers | CPU reference (`/home/hous/dev/clef/reports/reference/readme_examples.json`, `ref_text_bf16.jsonl`) |
|---|---|---|---|---|
| HF README usage example (`readme_invoice`) | 260 | 261.8 / 257.3 | `status` overdue 0.9941, `large` 0.9943 | overdue 0.9942, true 0.9943 |
| HF README SystemOne example (`readme_checkout`) | 300 | 262.0 / 260.5 | `department` technical 0.9138, `urgency` score 1.83 (level 2 0.8706), `outage` 0.9011 | technical 0.9155, score 1.82 (0.8651), 0.8995 |
| blog curl example (`blog_support_triage`) | 346 | 260.6 / 261.3 | `urgent` 0.9895, `team` technical 0.8072, `severity` score 2.9548 (Critical 0.9679) | 0.9906, technical 0.8059, 2.9568 (0.9697) (`ref_text_bf16.jsonl`) |
| image, base64 PNG `2a7ddcfe` (contest 42) | 330 | 489.7 / 229.9 | `caption` D 0.6402 | D 0.6475 (`ref_image_bf16.jsonl`) |
| video, 4 base64 frames of the same cartoon | 433 | 530.5 / 275.8 | `caption` D 0.6503 | D 0.6467 (`ref_video_bf16.jsonl`) |
| oversize state (20,037 tokens) | | 32 ms, HTTP 422 | `state is 20,037 tokens (36 of them the system prompt and media placeholders), over the 16,384-token limit: shorten the state or split it across requests; or start the server with CLEF_TRUNCATE_STATES=1 ...` | |

Every served probability equals the engine's traced probability to the 4-decimal rounding of `systemone_answer` (max |dp| 4.9e-5 over the 118 text, 40 image and 5 video options against `/home/hous/dev/clef/reports/stage3_tt_text_l64_traced_full.jsonl`, `stage3_tt_image_l64_traced_full.jsonl`, `stage3_tt_video_l64_traced_cached.jsonl`).

### Parity over HTTP (`scripts/parity_remote.py`, log `/home/hous/dev/clef/logs/stage3_parity_remote.log`, summary `/home/hous/dev/clef/reports/stage3_http_parity_summary.json`)

Two passes over the 16 text records, the 8 image records and the video record (the second pass in reverse order), rows `/home/hous/dev/clef/reports/stage3_http_{text,image,video}.jsonl` and `..._pass2.jsonl`, `parity_compare.py` against the CPU bf16 references (`/home/hous/dev/clef/reports/stage3_parity_http_{text,image,video}.{json,md}`):

| Set | questions | max dp vs CPU bf16 | mean dp | flips (at margin 0.05) | stage 1 / 2 engine result | pass 2 answers equal | pass 2 cache hits |
|---|---|---|---|---|---|---|---|
| 16 text records | 27 | 0.0722 | 0.0109 | 0 (0) | 0.0722 / 0.0109 / 0 (stage 1, `stage1_parity_l64_full.md`) | 16 / 16 | 4 of 16 (the 4 states the LRU still held) |
| 8 image records | 8 | 0.0463 | 0.0200 | 0 (0) | stage 2 after the tower amendment (`doc/vision/README.md`) | 8 / 8 | 5 of 8 |
| video record | 1 | 0.0126 | 0.0126 | 0 (0) | | 1 / 1 | 1 of 1 |

The gate "within 0.01 |dp| of the stage 1 and stage 2 engine results" holds with 4.9e-5 (the API rounding). Per-request `latency_ms` (model time on the worker, traced engine): text 219 to 467 ms for 150 to 529 tokens, 638 to 900 ms for 720 to 1,774 tokens; images 486 to 605 ms (tower plus prefill); median 395 ms over the 16 text records, 530 ms over the 8 images.

### Prefix cache over HTTP

Same state, different questions: the second pass restores the cached state and runs only the schema tail. `cfpb/3235868` (1,774 tokens, state prefix 1,280 tokens cached): 900.2 ms on the miss, 362.2 ms on the hit; `cfpb/8522318` (720 tokens) 638.3 to 283.3 ms; `cfpb/6165909` (912 tokens) 697.5 to 294.2 ms; image `6900e8fa` 520.5 to 233.3 ms; the video 529.8 to 279.9 ms. A state under 128 tokens has no cached prefix (`S0 = 0`), so the README examples cost the same on a hit (261 to 272 ms). The server log line carries `cache_hit=True` on the hits; `/v1/models` after the parity run: `prefix_cache {"size": 4, "hits": 17, "misses": 44, "cached_states": 4}`, 61 requests. The demo's repeated-state preset shows the same effect on a 2,227-token state (`demo/README.md`, "Real server").
