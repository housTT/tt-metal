# Clef demo and quickstart on two Blackhole chips (TP=2)

Date: 2026 Oct 05 (times in this file are Eastern Time, ET; the host clock is UTC). Code: tt-metal branch `hous/clef-bringup`, autoport `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/`.

This directory is a stand-alone demo of the Clef server (`/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/tt/server.py`). It has a web page and a command-line interface (CLI) that send the same requests to `POST /v1/systemone` and draw the answers. Nothing here is mounted by the server or shipped in the package. The page is plain HTML, CSS and JavaScript with no build step and no network dependency besides the Clef server. The CLI uses only the Python standard library. The preset builder needs Pillow.

Acronyms: API (application programming interface), ARC (AI2 Reasoning Challenge), CLI (command-line interface), CORS (cross-origin resource sharing), HF (Hugging Face), JSON (JavaScript Object Notation), LRU (least recently used), PNG (Portable Network Graphics), SSH (secure shell), TP (tensor parallel), URL (uniform resource locator).

## Files

| File | Purpose |
|---|---|
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/index.html` | The demo page: presets, state, images and questions editors, Answers, Replay and Snippets tabs. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/presets.json` | The demo cases and the labelled replay set. Read by the page and by the CLI. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/assets/` | 11 PNG files: 10 New Yorker cartoons copied from the reference and eval sets, and one synthetic receipt. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/build_presets.py` | Rebuilds `presets.json` and `assets/` from the reference records, the eval samples and the kev bench constants. Deterministic. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/quickstart.py` | The CLI: runs presets or the replay set, prints text bars, latency, token counts and accuracy. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/run.sh` | Serves this directory with `python3 -m http.server` and prints the URL to open. |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/check_core.js` | Node 18 check: the page's request builder against the CLI's request bodies (section 6). |
| `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/README.md` | This file. |

## 1. Start the Clef server

All three commands listen on `http://127.0.0.1:8008`. The full route, environment and error reference is `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/doc/server/README.md`.

Real server on this box (4 chips visible, so the server opens the `(1, 4)` parent under `FABRIC_1D` and takes the `(1, 2)` submesh at offset `(0, 0)`; path A of the server README). Device commands go through `devrun`, which holds the box's device lock:

```bash
export CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
export HF_HUB_OFFLINE=1 CLEF_MESH_SHAPE=1x2 MESH_DEVICE=P150x2 TT_CACHE_PATH=/home/hous/dev/clef/tt_cache OMP_NUM_THREADS=8
cd /home/hous/dev/clef/tt-metal && nohup /home/hous/dev/clef/bin/devrun timeout 7200 python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage8_server.log 2>&1 &
```

Wait for `Application startup complete` in `/home/hous/dev/clef/logs/stage8_server.log`. Stop with `kill -TERM` to the `python -m uvicorn` process (not the `flock` or `timeout` wrapper); the lifespan closes the submesh, the parent, then the fabric.

Published package (`tt-hous/clef`, after stage 6; no checkout needed besides this directory):

```bash
cd /home/hous/dev/tt-model-manager
/home/hous/dev/clef/bin/devrun bash -c "uv run --locked tt-model serve --port 8008 --profile p150x2 tt-hous/clef && sleep infinity"
```

No device at all (fake engine; API and plumbing checks only):

```bash
cd /home/hous/dev/clef/tt-metal && CLEF_FAKE_ENGINE=1 CLEF_MODEL=/home/hous/.cache/huggingface/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c HF_HUB_OFFLINE=1 nohup /home/hous/dev/clef/bin/hostrun python -m uvicorn models.autoports.cloudflare_clef.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/clef/logs/stage8_fake_server.log 2>&1 &
```

The fake engine returns seeded pseudo-random hidden rows through the real tokenizer, image processor and joint head, so every panel works, `usage.input_tokens` is exact, but the answers mean nothing and the replay accuracy is near chance.

## 2. Run the demo

Page:

```bash
/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/run.sh
/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/run.sh http://127.0.0.1:8008
```

It prints `Open: http://127.0.0.1:8080/#server=http://127.0.0.1:8008`. The `#server=` part fills the Server URL field; the field is editable. `DEMO_PORT=<port>` picks another port when 8080 is busy (`run.sh` refuses a busy port with a message). The page must be served, not opened as a file, because it fetches `presets.json` and the bundled images.

The box has no browser. From a laptop, forward both ports over SSH and open the same URL locally, since the page calls the Clef server from the browser:

```bash
ssh -L 8080:127.0.0.1:8080 -L 8008:127.0.0.1:8008 <user>@<box>
```

`DEMO_BIND=0.0.0.0` makes `run.sh` listen on every interface instead; then the Server URL in the page must also be reachable from the laptop.

CLI (Python 3, standard library only):

```bash
cd /home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo
python3 quickstart.py --list
python3 quickstart.py --preset "Support ticket triage"
python3 quickstart.py --preset "Long document"
python3 quickstart.py --all
python3 quickstart.py --replay
python3 quickstart.py --server http://127.0.0.1:8008 --api-key <CLEF_API_KEY> --replay
python3 quickstart.py --show-request --preset "Receipt image"
```

`--preset` takes an exact title or a unique prefix. With no mode flag the CLI runs `--all`. `--show-request` prints the request bodies without sending them (images as base64). Exit code 0 only when every request returned HTTP 200; `--replay` also needs a question accuracy of at least 0.5. That floor is a smoke bar for a working server, not an acceptance bar for the model: the CPU reference scores 95.0 on the ARC sample, 93.0 on the BANKING77 sample and 60.0 on the New Yorker sample (`/home/hous/dev/clef/reports/reference/README.md`), and the fake engine scores near chance, so `--replay` exits 1 against the fake server by design.

## 3. What each panel shows

- Status line: `GET /health` and `GET /v1/models`. Model name, backend (`ttnn` or `fake`), device string, mesh shape, worker count, whether the workers accept images, prefix cache fill, hits and misses, queue depth, state limit, traced or eager.
- Presets: one button per preset. A preset with several cases shows a second row of buttons. Selecting a case fills the State, Images and Questions editors and shows the case's source. A JSON-object state ticks "parse as JSON".
- Editors: the state is free text (or JSON when ticked). Images shows the case's bundled PNG files as thumbnails; the file input attaches more (any image type the browser can read), the `x` on a thumbnail removes one, "Remove all" clears the list. The questions editor is the `questions` object of the request, validated as you type with the release rules (`type` must be `noul`, `choice` or `score`; `choice` needs a non-empty object of options; `score` needs a non-empty list of levels; `noul` criteria, when given, is an object). Run is disabled while invalid. Ctrl+Enter runs.
- Answers tab: one card per question in the order sent. `choice` draws one bar per option sorted by probability, with the chosen option in full colour; the server returns `probabilities` in the request's `criteria` order, while the model itself sees the options sorted by id (release `question_options`). `noul` draws one bar, the probability of true, with a tick at 0.5. `score` draws one bar per level with the legend and a tick for the expected value (`score`) on a scale under the bars. `confidence` is the server's own field (top probability). When the case has labels and the editors were not changed, each card says whether the top answer matches the label.
- Metrics strip: `latency_ms` (the server's model time on its worker), client wall time, `usage.input_tokens`, image count, and a prefix-cache hit or miss derived from the change in `prefix_cache.hits` in `/v1/models` around the request (exact only when nobody else is sending). On the repeated-state preset the hint says to run again or switch cases to see the hit.
- Replay tab: sends the 24 labelled records one after another (one TP group answers one request at a time), with a running accuracy, records passed, mean and p50 `latency_ms`, elapsed time, and one row per record with the token count, latency and the model's top answer next to the label where they differ. Stop ends the loop after the current request.
- Snippets tab: the current request as `curl` and as Python `requests`. For a request with images the shell form reads each file with `base64 -w0` into a variable and the Python form reads the listed paths; the paths are the bundled `assets/...` files or the attached file names, so run the snippets from this directory. Copy uses the clipboard API on a secure origin (`127.0.0.1` is), else it selects the text for Ctrl+C.
- Errors appear under the Run button: 422 with the server's text (release validation rules, oversize state or schema, bad image payload), 401 (hint to fill the API key field), and connection refused (hint to start the server).

## 4. Presets

Every preset is `{"title", "what_it_shows", "cases": [...]}` and every case is `{"name", "source", "state", "questions", "images"?, "labels"?}`. `images` are paths relative to this directory; the page and the CLI read the files and send them as base64 strings in the request's `images` list. `labels` map a question id to the gold answer.

| Preset | Cases | Source | Shows |
|---|---|---|---|
| Support ticket triage | 1 | the blog's curl example (`/home/hous/dev/clef/reports/reference/records_text.jsonl`, `blog_support_triage`) | `urgent` noul, `team` choice, `severity` four-level score, one pass. The server's warmup request is the same record, so its state is a cache hit on the first run. |
| Invoice JSON state | 1 | HF README Usage example (`readme_invoice`) | A JSON object as the state; `status` choice and `large` noul. |
| Checkout outage | 1 | HF README SystemOne API example (`readme_checkout`) | `department` choice, `urgency` score with legend, `outage` noul. |
| Long document, repeated state | 2 | kev `952ce9d` `scripts/serving_bench.py` `PARAGRAPH` x30 with `FIVE`, and with `tone`, `escalate`, `frustration` (`tone` option descriptions written here) | A 2,227-token state piece (2,789 and 2,518 input tokens with the two schemas). The first run prefills the state; the second case, or a second run, hits the prefix cache. |
| New Yorker cartoon | 2 | `/home/hous/dev/clef/reports/reference/records_image.jsonl` rows 497 and 430 (contests 42 and 616), PNGs copied to `assets/` | An image request with the 5-caption `choice` and the gold caption. |
| Receipt image | 1 | synthetic PNG drawn by `build_presets.py` with Pillow (DejaVu Sans Mono, 520x760) | `legible` noul, `total_over_100` noul, `vendor` choice; image reading without a third-party asset. Labels: true, true, `northwind`. |

Replay set: 24 labelled records from the stage 5 eval samples in `/home/hous/dev/clef/evals/`: the first 8 rows of `arc_challenge_test_sample100.jsonl` (ARC-Challenge, 4-option `choice`), 8 rows of `banking77_test_sample100.jsonl` with the shortest distinct intent names (`age_limit`, `change_pin`, `atm_support`, `pin_blocked`, `card_arrival`, `card_linking`, `exchange_rate`, `top_up_failed`; every record carries the full 77-option schema, about 1,830 input tokens), and the first 8 rows of `newyorker_matching_test_sample100.jsonl` (cartoon plus 5 captions, PNGs copied to `assets/`). Each record keeps its `id` as the name and its `_label` values in `labels`; `_source`, `_source_row` and `_contest_number` are stripped from the request and summarized in `source`.

Accuracy in the replay is over questions, the rule of `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/scripts/eval_metrics.py`: `choice` takes `choice`, `noul` is true when `noul >= 0.5`, `score` takes the arg max of `probabilities`. A record passes when every question in it is right.

Licensing: the kev bench text and the layout are Apache-2.0 (listed in `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/NOTICE`). The 10 cartoons are from `jmhessel/newyorker_caption_contest` (Hugging Face dataset card: license `cc-by-4.0`, checked 2026 Oct 05; paper arXiv 2209.06293). The receipt is generated here and carries no third-party content.

## 5. Measured

### Fake engine, 2026 Oct 05, 09:20 ET (wiring check, no device)

Server started as in section 1 (fake engine, port 8008), log `/home/hous/dev/clef/logs/stage8_fake_server.log` (0 tracebacks, 54 responses with status 200, one deliberate 422). CLI logs: `/home/hous/dev/clef/logs/stage8_fake_quickstart_all.log` (`--all`, exit 0, 8 of 8 requests 200) and `/home/hous/dev/clef/logs/stage8_fake_quickstart_replay.log` (`--replay`, 24 of 24 requests 200, 0 failures, accuracy 0.083 with the fake engine, so exit 1 as the floor requires). The `latency_ms` values below are fake-engine times (tokenizer, processor and head on the CPU) and say nothing about the device.

| Case | state tokens (server `S=`) | input_tokens | latency_ms | client wall ms | cache |
|---|---|---|---|---|---|
| Support ticket triage | 48 | 346 | 48.9 | 53 | hit (warmup record) |
| Invoice JSON state | 62 | 260 | 41.4 | 45 | miss |
| Checkout outage | 46 | 300 | 47.6 | 51 | miss |
| Long document, 5 questions | 2,227 | 2,789 | 215.7 | 226 | miss |
| Long document, second question set | 2,227 | 2,518 | 154.2 | 161 | hit |
| New Yorker cartoon, contest 42 | 133 (80 image tokens) | 330 | 46.4 | 930 (first image request loads the processor) | miss |
| New Yorker cartoon, contest 616 | 319 | 511 | 62.1 | 73 | miss |
| Receipt image | 431 | 741 | 84.4 | 95 | miss |

Replay on the fake engine: input tokens 181 to 260 (ARC), 1,828 to 1,847 (BANKING77), 354 to 752 (New Yorker); `latency_ms` mean 88.6, p50 63.5, max 178.2; 2.3 s in total.

Page served from port 8080 (`/home/hous/dev/clef/logs/stage8_demo_http.log`): `index.html` 200 (40,970 bytes, `text/html`), `presets.json` 200 (80,213 bytes, `application/json`), `assets/receipt_northwind.png` 200 (46,449 bytes, `image/png`). The server's CORS preflight for the page origin returned 200 with `access-control-allow-origin: *`. A second fake server started with `CLEF_API_KEY=demo-key` (`/home/hous/dev/clef/logs/stage8_fake_server_apikey.log`) gave 401 on `/v1/systemone` and `/v1/models` without the key, 200 on `/health`, 200 through `quickstart.py --api-key demo-key`, exit 1 with the 401 text from the CLI without the key, and 401 without CORS headers on the browser preflight (section 7).

### Shipped package, eager engine, two chips (TP=2), 2026 Oct 05, 20:14 ET

These are the numbers of the shipped configuration: the `tt-model` container image `tt-model/clef:4fc1683a8975` (profile `p150x2`, `CLEF_TRACED=0`, prefix planner on, chips 0 and 1 of the first p300 board, direct 1x2 mesh open under `FABRIC_1D`) served by `tt-model serve --port 8008 --profile p150x2 --device-id 0,1` during the stage 6 package validation (`/home/hous/dev/clef/package/WORKLOG.md` section 8, container log `/home/hous/dev/clef/logs/stage6_serve_p150x2.log`). The CLI ran right after the quick serving benchmark, so every preset below was a prefix-cache miss except where marked. CLI logs: `/home/hous/dev/clef/logs/stage6_demo_quickstart_all_p150x2.log` (`--all`, exit 0, 8 of 8 requests 200) and `/home/hous/dev/clef/logs/stage6_demo_quickstart_replay_p150x2.log` (`--replay`, exit 0, 24 of 24 requests 200). `latency_ms` is the server's model time on the worker (eager TP=2 engine, host head), client wall is the CLI's round trip.

| Case | input_tokens | latency_ms | client wall ms | cache |
|---|---|---|---|---|
| Support ticket triage | 346 | 279.9 | 283 | miss (short state, no cached prefix) |
| Invoice JSON state | 260 | 268.7 | 271 | miss |
| Checkout outage | 300 | 277.8 | 281 | miss; technical 0.9138, score 1.83, outage 0.9011 (the card's reference answers) |
| Long document, 5 questions (new state, `--all`) | 2,789 | 1,233.1 | 1,244 | miss |
| Long document, second question set (same state, `--all`) | 2,518 | 419.1 | 427 | hit |
| New Yorker cartoon, contest 42 (2a7ddcfe) | 330 | 498.6 | 505 | miss; choice D, label D |
| New Yorker cartoon, contest 616 (6c1478b4) | 511 | 551.2 | 563 | miss; choice B, label B |
| Receipt image | 741 | 2,737.6 | 2,749 | miss; legible true 0.9873, total over 100 true 0.9933, vendor northwind 0.9866 (all labels met) |

The receipt's 2,737.6 ms includes the per-grid program compile of the vision tower: the container boots with a cold kernel cache (`/home/hous/.cache/tt-model/clef/cache`), and the 520x760 grid had not been seen inside it (the traced server below ran this case at 984.2 ms on a box whose cache held the grid). The two cartoons use grids the validation's image request and parity run had already compiled.

Replay on the shipped package: 24 records, 22 passed, 0 request failures, 22 of 24 questions correct, accuracy 0.917 (smoke floor 0.5); `latency_ms` mean 866.2, p50 811.5, max 2,775.0; client wall mean 875 ms; 21.0 s in total. The two misses are the same as on the traced server (ARC-Challenge `Mercury_417143`, answer C against label D; New Yorker `2d63a4a5`, caption A against label E). The ARC records took 224 to 229 ms (one at 401.7 ms), the BANKING77 records (1,828 to 1,847 tokens) 809 to 826 ms (the first at 1,008.3 ms), and the New Yorker records 570 to 2,775 ms, where the 2,287.4, 1,512.7, 2,731.0 and 2,775.0 ms rows are the first requests of four cartoon grids in the container's cold cache (`6f3e6ee6`, `deb0ad8d`, `2d63a4a5`, `e08f4e32`). `/v1/models` after the demo: 884 requests served since boot, `prefix_cache {"size": 4, "hits": 127, "misses": 757}`.

### Previous row: traced engine on the development server (`CLEF_TRACED=1`), 2026 Oct 05, 14:20 to 14:21 ET

Kept for comparison; the traced engine is the opt-in mode, not the shipped configuration. The eager engine's probabilities equal the traced engine's, so the answers in this block are the same as above.


Server started as in section 1 (traced engine, `CLEF_TRACED=1`, path A; the same server process that served the stage 3 parity and benchmark runs, log `/home/hous/dev/clef/logs/stage3_server.log`, started 13:45 ET, `Application startup complete` 107 s after `starting`). The CLI ran right after the serving benchmark, so the prefix cache held the bench's states and every preset below was a miss except where marked. CLI logs: `/home/hous/dev/clef/logs/stage8_device_quickstart_all.log` (`--all`, exit 0, 8 of 8 requests 200), `/home/hous/dev/clef/logs/stage8_device_quickstart_long_1.log` and `..._long_2.log` (`--preset "Long document"` twice, exit 0), `/home/hous/dev/clef/logs/stage8_device_quickstart_replay.log` (`--replay`, exit 0, 24 of 24 requests 200). `latency_ms` is the server's model time on the worker (traced TP=2 engine, host head), client wall is the CLI's round trip.

| Case | input_tokens | latency_ms | client wall ms | cache |
|---|---|---|---|---|
| Support ticket triage | 346 | 272.8 | 276 | miss (short state, no cached prefix) |
| Invoice JSON state | 260 | 261.8 | 265 | miss |
| Checkout outage | 300 | 271.5 | 274 | miss |
| Long document, 5 questions (new state, `--all`) | 2,789 | 1,408.3 | 1,418 | miss |
| Long document, second question set (same state, `--all`) | 2,518 | 384.9 | 393 | hit |
| Long document, 5 questions (same state again, `--preset` run 1 / run 2) | 2,789 | 548.4 / 549.8 | 557 / 559 | hit / hit |
| Long document, second question set (`--preset` run 1 / run 2) | 2,518 | 400.9 / 408.4 | 409 / 416 | hit / hit |
| New Yorker cartoon, contest 42 (2a7ddcfe) | 330 | 488.2 | 494 | miss; choice D, label D |
| New Yorker cartoon, contest 616 (6c1478b4) | 511 | 532.8 | 544 | miss; choice B, label B |
| Receipt image | 741 | 984.2 | 995 | miss; legible true 0.9873, total over 100 true 0.9933, vendor northwind 0.9866 (all labels met) |

The receipt (520x760 PNG, 741 tokens) is the one image whose grid the server had not run before, so its first request includes the per-grid program compile of the vision tower (stage 2 finding; `doc/optimized/README.md`, "Vision under tracing"); the cartoons use grids warmed at startup.

Replay on the device: 24 records, 22 passed, 22 of 24 questions correct, accuracy 0.917 (smoke floor 0.5; the CPU reference scores 95.0 / 93.0 / 60.0 on the full ARC, BANKING77 and New Yorker samples these 8 + 8 + 8 records come from); `latency_ms` mean 773.4, p50 790.8, max 2,671.8; client wall mean 783 ms; 18.8 s in total. The two misses are ARC-Challenge `Mercury_417143` (answer C against label D) and New Yorker `2d63a4a5` (caption A against label E); the 2,671.8, 2,284.9 and 1,528.7 ms outliers are the first requests of three cartoon grids the server had not seen (per-grid tower compile; `6f3e6ee6`, `deb0ad8d`, `2d63a4a5`), the other five New Yorker records took 574 to 963 ms, the ARC records 221 to 226 ms (one at 394 ms) and the BANKING77 records (about 1,830 tokens) 805 to 843 ms. Labelled presets on the device: New Yorker 2 of 2 correct, receipt 3 of 3 questions correct. `/v1/models` after the demo: 3,394 requests served, `prefix_cache {"size": 4, "hits": 128, "misses": 3266}` (the misses are the benchmark's distinct states).

## 6. Request shape and the request-builder check

Both the page and the CLI send `{"model": "clef", "state": <state>, "questions": <questions>}` plus `"images": [<base64>, ...]` when the case has images, to `POST /v1/systemone` with `content-type: application/json` and `authorization: Bearer <key>` when a key is given. The page builds it in `CLEF.buildRequest(state, questions, model, images)` inside the `clef-core` script block of `index.html`; the CLI in `build_request(case, model, base_dir)` in `quickstart.py`. A preset's `state` goes in as is (the page shows it in the editor and reads it back; an object state is shown as JSON and parsed back), `questions` is the preset's object verbatim, and every image file is sent as plain base64 (the server also accepts `data:` URLs and http(s) URLs).

Check: `node /home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/check_core.js` extracts the `clef-core` block from `index.html`, runs it under Node 18 with no DOM, builds the request for every preset case and replay record as the page would (editor round trip and base64 of the bundled files included), and compares each body with `quickstart.py --show-request`. It also checks `parseQuestions` on six invalid inputs, `predicted` for `score` and `noul`, and the two snippet forms with an image. Result on 2026 Oct 05: `request bodies identical: 32 of 32 (8 cases, 24 replay records)`. The page's layout was not screenshot-checked on this box (no browser); open `http://127.0.0.1:8080/#server=http://127.0.0.1:8008` after `run.sh` to see it.

## 7. Known limits

- A server started with `CLEF_API_KEY` answers the browser's CORS preflight (`OPTIONS /v1/systemone`, which carries no `authorization` header) with 401 and without CORS headers (checked against the fake engine on 2026 Oct 05), so the page cannot call such a server from another origin. The CLI with `--api-key` works. The page says so in its error text when a key is filled in.
- The cache hit or miss in the metrics strip and in the CLI is inferred from the `/v1/models` hit counter; with other clients active it can be wrong.
- The first request with an image after a server start loads the image processor (about 1 s on the fake run), so its client wall time is longer than its `latency_ms`.
- The replay sends images of up to 335 KB as base64 in JSON; on a slow link the client wall time grows while `latency_ms` does not.

## 8. Add a preset

Edit `build()` in `/home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/build_presets.py` (a preset is `{"title", "what_it_shows", "cases": [{"name", "source", "state", "questions", "images"?, "labels"?}]}`; `copy_asset(path)` copies a PNG into `assets/` and returns the relative path), then run:

```bash
/home/hous/dev/clef/bin/hostrun python /home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/build_presets.py
python3 -m json.tool /home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/presets.json > /dev/null
node /home/hous/dev/clef/tt-metal/models/autoports/cloudflare_clef/demo/check_core.js
```

For a one-off case, editing `presets.json` by hand with the same shape also works; both the page and the CLI read it at start.
