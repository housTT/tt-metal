# kev-9b demo and quickstart on one P150

Date: 2026 Oct 02 (times in this file are Eastern Time, ET). Code: tt-metal branch `hous/kev-9b-bringup`, autoport `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/`.

This directory is a stand-alone demo of the kev-9b server. It has a web page and a command-line interface (CLI) that send the same requests to `POST /v1/systemone` and draw the answers. Nothing here is mounted by the server or shipped in the package. The page is plain HTML, CSS and JavaScript with no build step and no network dependency besides the kev server. The CLI uses only the Python standard library.

Acronyms: API (application programming interface), CLI (command-line interface), CORS (cross-origin resource sharing), JSON (JavaScript Object Notation), LRU (least recently used), SSH (secure shell), URL (uniform resource locator).

## Files

| File | Purpose |
|---|---|
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/index.html` | The demo page: presets, state and questions editors, Answers, Replay and Snippets tabs. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/presets.json` | The demo cases and the labelled replay set. Read by the page and by the CLI. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/make_presets.py` | Rebuilds `presets.json` from the kev checkout at `/home/hous/dev/kev/kev` (commit 952ce9d). Deterministic. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/quickstart.py` | The CLI: runs presets or the replay set, prints text bars, latency and accuracy. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/run.sh` | Serves this directory with `python3 -m http.server` and prints the URL to open. |
| `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/README.md` | This file. |

## 1. Start the kev server

Pick one of the two commands. Both listen on `http://127.0.0.1:8008`.

Development server from this checkout, one chip (device commands go through `devrun`, which holds the box's device lock):

```bash
cd /home/hous/dev/kev/tt-metal
export HF_MODEL=/home/hous/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/68c46c4b3498877f3ef123c856ecfde50c39f404
export KEV_RUN=/home/hous/.cache/huggingface/hub/models--jaredpalmer--kev-9b/snapshots/db029f08b290afd9fee4aa4bbcd9ae48602d1eb0
export TT_CACHE_PATH=/home/hous/dev/kev/tt_cache KEV_MESH_SHAPE=1x1 HF_HUB_OFFLINE=1
/home/hous/dev/kev/bin/devrun timeout 7200 python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --host 127.0.0.1 --port 8008 --lifespan on > /home/hous/dev/kev/logs/demo_server.log 2>&1 &
```

Wait for `Application startup complete` in `/home/hous/dev/kev/logs/demo_server.log` (35 s with a warm weight cache on 2026 Oct 02; 1 to 2 minutes cold). Stop it with `kill -TERM <pid of the python -m uvicorn process>`, not the `flock` or `timeout` wrapper.

Published package (`tt-hous/kev-9b`, no checkout needed besides this directory):

```bash
cd /home/hous/dev/tt-model-manager
/home/hous/dev/kev/bin/devrun bash -c "uv run --locked tt-model serve --port 8008 tt-hous/kev-9b && sleep infinity"
```

`tt-model serve` returns when the container is ready (about 150 s on the first boot, which writes the tensor cache, and about 100 s after that). The container keeps the chips, so keep the `devrun` session open while you use the demo, and stop with `uv run --locked tt-model stop tt-hous/kev-9b` from `/home/hous/dev/tt-model-manager`. The package work log has the full procedure: `/home/hous/dev/kev/package/README.md`.

No device at all: `KEV_FAKE_ENGINE=1 /home/hous/dev/kev/bin/hostrun python -m uvicorn models.autoports.jaredpalmer_kev_9b.tt.server:app --port 8008` (with the same `HF_MODEL`, `KEV_RUN` and `HF_HUB_OFFLINE=1`). The fake engine returns deterministic noise, so every panel works but the answers mean nothing and the replay accuracy is near chance.

## 2. Run the demo

Page:

```bash
/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/run.sh
/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/run.sh http://127.0.0.1:8008
```

It prints `Open: http://127.0.0.1:8080/#server=http://127.0.0.1:8008`. The `#server=` part fills the Server URL field; the field is editable. Port 8080 is already in use on this box by another service, so here use `DEMO_PORT=8081 /home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/run.sh` and open `http://127.0.0.1:8081/#server=http://127.0.0.1:8008`. `run.sh` refuses a busy port with a message instead of a traceback.

The box has no browser. From a laptop, forward both ports over SSH and open the same URL locally, since the page calls the kev server from the browser:

```bash
ssh -L 8081:127.0.0.1:8081 -L 8008:127.0.0.1:8008 <user>@<box>
```

`DEMO_BIND=0.0.0.0` makes `run.sh` listen on every interface instead; then the Server URL in the page must also be reachable from the laptop.

CLI:

```bash
cd /home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo
python3 quickstart.py --list
python3 quickstart.py --preset "Support ticket triage"
python3 quickstart.py
python3 quickstart.py --replay
python3 quickstart.py --server http://127.0.0.1:8008 --api-key <KEV_API_KEY> --replay
python3 quickstart.py --show-request --preset "Code review"
```

With no `--preset` and no `--replay` it runs every preset. `--show-request` prints the request bodies without sending them. Exit code 0 only when every request returned HTTP 200; `--replay` also needs accuracy of at least 0.7. That floor is a smoke bar for a working server, not an acceptance bar for the model; the model-card numbers are in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/doc/benchmark/README.md`.

## 3. What each panel shows

- Status line: `GET /health` and `GET /v1/models`. Model name, backend (`ttnn` or `fake`), device string, worker count, prefix cache fill, hits and misses, queue depth, state limit.
- Presets: one button per case group. A group with several cases (for example the three complaints) shows a second row of buttons. Selecting a case fills the State and Questions editors and shows the case's source. States that are JSON objects tick "parse as JSON".
- Editors: the state is free text (or JSON when ticked). The questions editor is the `questions` object of the request, validated as you type (type must be `noul`, `choice` or `score`; `choice` needs an object of options; `score` needs a list of levels). Run is disabled while invalid. Ctrl+Enter runs.
- Answers tab: one card per question in the order sent. `choice` draws one bar per option sorted by probability, with the chosen option in full colour. `noul` draws the yes and no bars. `score` draws one bar per level with the legend and a tick for the expected value (`score`) on a scale under the bars. The confidence is the server's own field. When the case has labels and the editors were not changed, each card says whether the top answer matches the label.
- Metrics strip: `latency_ms` (the server's model time), client wall time, `input_tokens`, `output_tokens`, `state_tokens` when the server reports it, and a prefix-cache hit or miss derived from the change in `prefix_cache.hits` in `/v1/models` around the request (exact only when nobody else is sending). The hint under it says to run a long state again to see the hit.
- Replay tab: sends the 24 labelled records one after another (one chip answers one request at a time, so there is no point in parallel requests), with a running accuracy, records passed, mean and p50 `latency_ms`, elapsed time, and one row per record with the model's top answer next to the label where they differ. Stop ends the loop after the current request.
- Snippets tab: the current request as `curl` (heredoc body), Python `requests`, and Python `typesafe_sdk`. Copy uses the clipboard API when the page is on a secure origin (`127.0.0.1` is), else it selects the text for Ctrl+C. All three forms were run against the server on 2026 Oct 02 and returned the same answers as the page.
- Errors appear under the Run button: 422 (state over the limit, or a validation error listed by field), 401 (hint to fill the API key field), and connection refused (hint to start the server).

## 4. Presets

| Preset | Cases | Source | Shows |
|---|---|---|---|
| Support ticket triage | 1 | `scripts/serving_bench.py` `TICKET` and `QUESTIONS` (equal to the playground "Support triage" preset) | Six questions of three types in one request. |
| Long document, repeated state | 2 | `PARAGRAPH x30` with `FIVE`, and the same state with `tone`, `escalate`, `frustration` | 2,392-token state; the prefix cache on the second run. |
| Complaint routing | 3 | `evals/documents-v1/development.jsonl` lines 6, 54, 69 (`cfpb/9046322`, `cfpb/2038700`, `cfpb/8653073`) | Product and issue routing with labels. |
| Answer checking | 2 | `evals/hard-v1/development.jsonl` lines 15, 19 | Yes/no judgement plus a value choice, with labels. |
| Code review | 1 | `evals/devtools-v1/development.jsonl` line 57 (`commitpackft/javascript/14553`) | A diff as the state. |
| News topic | 3 | `evals/v7/decision-v7/development.jsonl` lines 161, 163, 166 | AG News topic plus two yes/no questions, with labels. |
| Review rating | 1 | playground "Review rating" | Score questions with the expected value. |
| News article (object state) | 1 | playground "News article" | A JSON object as the state. |
| Isolation probe | 1 | playground "Isolation probe" | A secret in a sibling question stays invisible. |
| Boundary forgery | 1 | playground "Boundary forgery" | An option text with fake delimiters. |

Replay set: 24 labelled development records, the first string-state records of each file: 8 from documents-v1, 8 from hard-v1, 4 from devtools-v1, 4 AG News records from decision-v7 (records whose state is an object are skipped, and so is `agnews/test/1080`). Every record keeps its `_meta.id` as the name and its `label` values in `labels`; the request itself carries only `type`, `instructions` and `criteria` per question, the same cut as `api_request` in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/tt/api.py`.

Accuracy in the replay is over questions, as kev's own `benchmark.py` scores it: the top answer (`choice`; `noul > 0.5`; the arg max level of a `score`) equals the label. A record passes when every question in it is right.

Paths in the table are relative to `/home/hous/dev/kev/kev`. The playground presets, the bench constants and the bar layout are from the kev repository (Apache-2.0) and are listed in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/NOTICE`.

## 5. Measured on 2026 Oct 02

### Development server, one P150 (10:14 ET)

Server started as in section 1 (`KEV_MESH_SHAPE=1x1`, chip 0, traced prefill, selected precision `mlp_bfp8`), tree at `d36ca74c62c` plus this directory. Logs: `/home/hous/dev/kev/logs/demo_server.log`, `/home/hous/dev/kev/logs/demo_device_presets.log`, `/home/hous/dev/kev/logs/demo_device_long_twice.log`, `/home/hous/dev/kev/logs/demo_device_replay.log`. Every request returned 200; the server log has 0 tracebacks and 0 5xx over 47 requests.

| Case | input_tokens | latency_ms | client wall ms |
|---|---|---|---|
| Support ticket triage | 253 | 606.9 | 611 |
| Long document, 5 questions (new state) | 2,392 | 1,533.1 | 1,538 |
| Long document, 5 questions (same state again) | 2,392 | 526.6 | 532 |
| Long document, second question set (same state) | 2,252 | 315.4 | 319 |
| Complaint routing (3 cases) | 306, 222, 228 | 291.4, 247.2, 246.8 | 294, 250, 249 |
| Answer checking (2 cases) | 127, 140 | 202.4, 202.7 | 205, 205 |
| Code review | 200 | 247.5 | 250 |
| News topic (3 cases) | 121, 133, 151 | 303.7, 303.5, 303.5 | 306, 306, 306 |
| Review rating | 123 | 303.5 | 306 |
| News article (object state) | 135 | 303.3 | 306 |
| Isolation probe | 93 | 202.1 | 205 |
| Boundary forgery | 83 | 101.3 | 103 |

The long-document hit (1,532.1 ms then 526.6 ms, `/home/hous/dev/kev/logs/demo_device_long_twice.log`) matches the model-card row (1,533.5 / 527.0 ms). A hit needs the state to still be in the worker's 8-slot LRU cache: in the first pass over all presets, 17 other states ran between the two long runs and the second run missed (1,530.2 ms).

Labelled presets: complaint routing 5 of 6 questions right (`cfpb/8653073` product `personal_loan`, label `student_loan`); answer checking 4 of 4; code review 1 of 2 (`change_type` `change`, label `docs`, confidence 0.08); news topic 9 of 9.

Replay: 24 records, 17 passed, 0 request failures, 33 of 41 questions right, accuracy 0.805. `latency_ms` mean 344.9, p50 303.4, max 1,693.5 (the 12,769-character `long_policy` record), client wall mean 348 ms, 8.4 s in total.

### Published package `tt-hous/kev-9b` (10:15 ET)

Pulled on the host first with `cd /home/hous/dev/tt-model-manager && uv run --locked tt-model pull tt-hous/kev-9b` (log `/home/hous/dev/kev/logs/demo_pull.log`), because `devrun` sets `HF_HUB_OFFLINE=1`. Then one `devrun bash -c` session ran `uv run --locked tt-model serve --port 8008 tt-hous/kev-9b` (profile `p150`, chip picked by tt-model), `quickstart.py --preset "Support ticket triage"`, `quickstart.py --replay`, and `uv run --locked tt-model stop tt-hous/kev-9b`. Log `/home/hous/dev/kev/logs/demo_package.log`, container log `/home/hous/dev/kev/logs/demo_package_container.log` (0 tracebacks).

| Step | Result |
|---|---|
| `tt-model serve` to `Application startup complete` | 10:15:17 to 10:16:00 ET, 43 s (tensor and kernel caches already present from the stage 7 validation) |
| Support ticket triage | 253 input tokens, `latency_ms` 606.3, client wall 610 ms (development server: 606.9) |
| Replay | 24 records, 17 passed, 0 request failures, 33 of 41 right, accuracy 0.805; `latency_ms` mean 344.7, p50 303.1, max 1,693.2; 8.4 s. Every per-record answer equals the development-server run. |
| `tt-model stop` | clean shutdown in 1.4 s; no `tt-model-kev-9b-*` container left; device lock free at 10:16:11 ET |

## 6. Request shape

Both the page and the CLI send `{"state": <state>, "model": "kev-latest", "questions": <questions>}` to `POST /v1/systemone` with `content-type: application/json`, plus `authorization: Bearer <key>` when a key is given. The page builds it in `KEV.buildRequest(state, questions, model)` inside the `kev-core` script block of `index.html`; the CLI in `build_request(case, model)` in `quickstart.py`. A preset's `state` goes in as is (the page shows it in the editor and reads it back; an object state is shown as JSON and parsed back), and `questions` is the preset's object verbatim.

Check done on 2026 Oct 02: the `kev-core` block was extracted and run under Node 18 to build the request for all 16 preset cases and 24 replay records as the page would (editor round trip included), and compared with `quickstart.py --show-request` output: 40 of 40 bodies identical. The page's layout was not screenshot-checked on this box (no browser); open `http://127.0.0.1:8081/#server=http://127.0.0.1:8008` after `DEMO_PORT=8081 run.sh` to see it.

## 7. Known limits

- A server started with `KEV_API_KEY` answers the browser's CORS preflight (`OPTIONS /v1/systemone`, no `authorization` header) with 401 and without CORS headers (checked against the fake engine on 2026 Oct 02), so the page cannot call such a server from another origin. The CLI with `--api-key` works. The page says so in its error text when a key is filled in.
- The cache hit or miss in the metrics strip is inferred from the `/v1/models` hit counter; with other clients active it can be wrong.
- Port 8080 is taken on this box; use `DEMO_PORT`.
- `typesafe_sdk` is not in the tt-metal environment; the SDK snippet was run from the kev environment (`cd /home/hous/dev/kev/kev && uv run python <file>`).

## 8. Add a preset

Edit `presets` in `/home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/make_presets.py` (a preset is `{"title", "what_it_shows", "cases": [{"name", "source", "state", "questions", "labels"?}]}`), then run `python3 /home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/make_presets.py` and `python3 -m json.tool /home/hous/dev/kev/tt-metal/models/autoports/jaredpalmer_kev_9b/demo/presets.json > /dev/null`. For a one-off case, editing `presets.json` by hand with the same shape also works; both the page and the CLI read it at start.
