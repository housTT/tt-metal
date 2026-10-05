# Laya serving: host-side code, demo page, manifests

Track S deliverables (plan sections 8 and 9, Appendix B, CPU-side tests of A.8). Everything here runs without a
device when `LAYA_BACKEND=cpu`; the TT (Tenstorrent) backend is Track T's `tt/engine.py` and is imported lazily.
Paths are under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/convaiinnovations_laya/` unless absolute.

## Files

| file | role |
|---|---|
| `server/app.py` | module-level `app`, lifespan model load, sanity check, endpoints, headers, bearer auth, one lock around the forward |
| `server/engine.py` | model dir resolution, backend selection, question validation, request to rows, buckets, padding, chunking, `Engine`, `CpuBackend` |
| `server/decode.py` | temperatures, softmax, confidence, expected score, P(true), the pip 0.3.27 answer shape, `usage`, `min_confidence` gate |
| `server/demo.py` | `register_demo(app)`: `/demo` redirect, stamped HTML, `presets.json`, `feed.json`, static mount with `Cache-Control: no-cache` |
| `server/sanity_reference.json` | the STATE_EN / Q_CHOICE answer recorded from the CPU reference; compared at every startup |
| `demo/index.html`, `demo/demo.css`, `demo/demo.js`, `demo/presets.json`, `demo/feed_cases.json` | the page (plain ES2020, no build step, no external assets) |
| `shim/laya_tt_backend.py` | `TtBackend` for a pip `laya` Agent: routes `agent.model.forward` to `/v1/forward` or to an in-process `LayaEngine` |
| `tests/test_host_engine.py`, `tests/test_server_cpu.py`, `tests/test_demo_page.py`, `tests/fixtures/*.json` | CPU-side tests and recorded logits |
| `tt-model.yaml`, `tt-model-typed-decisions.yaml` | the two container manifests (placeholders marked `PLACEHOLDER`) |
| `/home/hous/dev/laya/bin/serve-cpu.sh`, `serve-tt.sh`, `package-build.sh`, `screenshot-demo.sh`, `screenshot_wrapper.py` | workspace scripts (CLM originals with paths changed) |
| `/home/hous/dev/laya/package/BUILD_PLAN.md`, `IMPORT_CLOSURE.md`, `/home/hous/dev/laya/demo-script.md` | build plan, import closure, 60-second script |

## Endpoints

| method and path | purpose | status codes |
|---|---|---|
| `GET /health` | liveness and readiness: `{"status": "ok", "ready": true, "backend", "model", "revision"}` | 200 ready, 503 `{"status": "loading"}` before the lifespan finished or when the model failed to load |
| `GET /v1/health` | everything a client or the demo page needs: backend, model, revision, model_dir, max_len, head_max_len, mesh_shape, precision, seq_buckets, row_buckets, warm_shapes, the backend's `shapes()`, limits, token_budget, temperatures (raw and applied), request and row counters, batch histogram, `sanity`, `raw_forward`, `api_key_required`, uptime | 200, 503 while loading; with a wrong or missing bearer only `status`, `ready`, `backend`, `api_key_required` are returned |
| `GET /v1/models` | one entry: id, revision, backend, precision, max_len, head_max_len, buckets | 200, 401 |
| `POST /v1/systemone` | one state, N questions (1 to `LAYA_MAX_QUESTIONS`) | 200, 400, 401, 413, 422, 500, 503 |
| `POST /v1/systemone/batch` | 1 to `LAYA_MAX_BATCH_STATES` states sharing one question set; `{"results": [...], "total_usage": {...}}` | same |
| `POST /v1/forward` | raw tensors in, raw tensors out; only registered when `LAYA_RAW_FORWARD=1` | same |
| `GET /demo` (307), `GET /demo/`, `GET /demo/presets.json`, `GET /demo/feed.json`, `GET /demo/<static>` | the demo; absent with `LAYA_NO_DEMO=1` | 200, 404 |

Request bodies follow pip `laya` 0.3.27's `laya-serve` (`/home/hous/dev/laya/evals/.venv/lib/python3.12/site-packages/laya/serve.py`):

- `POST /v1/systemone`: `{"state": <str | object | list>, "questions": {id: {"type": "choice"|"score"|"noul", "instructions": <str | object | list>, "criteria": ...}}, "max_len"?, "head_max_len"?, "min_confidence"?}`.
- `POST /v1/systemone/batch`: `{"states": [...], "questions": {...}, "max_len"?, "head_max_len"?, "min_confidence"?, "batch_size"?, "sort_by_length"?}`. `batch_size` and `sort_by_length` are type-checked (422) and otherwise ignored: the server plans its own chunks.
- `POST /v1/forward`: `{"input_ids": [[int]], "attention_mask": [[int]], "marker_pos": [[int]], "marker_mask": [[bool]], "qtype": [int]}` with the shapes of the contract below; the reply is `{"logits": [[float]], "act_logits": [[float]], "batch": "<rows>x<seq>", "rows": N, "kmax": K}` sliced back to the request rows and marker slots.

Status code rules, taken from pip `serve.py`:

- 400: body is not JSON, not an object, lacks `questions` (or `states` on batch), `state` missing or `null`, `questions` not an object, `states` empty or not a list, state not JSON-serializable.
- 413: more than `LAYA_MAX_QUESTIONS` (64) questions, more than `LAYA_MAX_BATCH_STATES` (64) states, state longer than `LAYA_MAX_STATE_CHARS` (50000) characters, more than `LAYA_MAX_CHOICE_OPTIONS` (100) choice options, more than `LAYA_MAX_SCORE_LEVELS` (32) score levels, more than `LAYA_MAX_TOTAL_OPTIONS` (512) options per request, more rows than the largest row bucket on `/v1/forward`.
- 422: a malformed question (unknown type, missing or empty `instructions`, wrong `criteria` shape, null or duplicate or non-scalar choice labels, null score level, noul criteria keyed other than true/false, bad `option_order`, `labels` which this server does not support), `max_len` or `head_max_len` not a positive integer or above the token budget or above the served context, bad `min_confidence`, hook keys in the body (`hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise`, `hooks_timeout`), options that do not fit the head budget, a `/v1/forward` tensor with wrong shapes or values.
- 401: `LAYA_API_KEY` set and `Authorization: Bearer <key>` missing or wrong (constant-time compare, bytes).
- 503: model not loaded. 500: anything else; the traceback goes to the log, the client sees `inference failed`.

Fields pip accepts that this server ignores on purpose: `model` (one checkpoint is served), `task`, `lang`,
`lang_guess` (no router, English checkpoint only), `batch_size`, `sort_by_length`. The pip Router's `routing` block
is not emitted. `LAYA_JEV_STRICT` is not implemented.

## Response shape (pip `laya` 0.3.27 `Agent._decode_answers`)

```
{"model": "laya-rl-agent",
 "answers": {
   "<id>": {"type": "choice", "choice": "<label>", "probabilities": {"<label>": p, ...}, "confidence": c, "answer_confidence": max_p, "action": {"act_probability": a}},
   "<id>": {"type": "score", "score": expected, "legend": {"0": "...", ...}, "probabilities": {"0": p, ...}, "confidence": c, "answer_confidence": max_p, "action": {...}},
   "<id>": {"type": "noul", "noul": p_true, "confidence": max(p_true, 1 - p_true), "answer_confidence": max_p, "action": {...}}},
 "usage": {"input_tokens": n, "output_tokens": 0, "state_tokens": n, "state_tokens_dropped": n, "truncated": bool, "truncated_questions": [ids], "options": {id: {"total", "distinct", "tokens_per_option"}} (only when options collapsed)}}
```

`confidence` is `1 - H(p) / log(k)` (vendored `confidence_from_probs`), `answer_confidence` is `max(p)`, `score` is
`sum(i * p_i)`, `noul` is `p[1]`, `act_probability` is `softmax(act_logits)[0]`, all rounded to 4 decimals as pip does.
Note the plan's field name `questions_truncated` is `truncated_questions` in pip 0.3.27; the server uses pip's name.
Empty `questions` returns `{"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}`
without a forward pass.

With `min_confidence` set (a float in [0, 1] or a map `{"choice:3-5": 0.6, "default": 0.1}` keyed like
`temp_bucket`), every answer gains `abstention` (`passed`, `abstained`, `unevaluated`) and `abstention_threshold`, and
`low_confidence: true` when `answer_confidence` (fallback `confidence`) is below the threshold. With it unset nothing
is added. This is `laya.confidence.apply_confidence_gate` copied line for line into `server/decode.py`.

### Temperature clamp (a deviation from the Hub `rl_agent_api.py`, following pip)

`rl_agent_api.py` divides the logits by the raw temperature of the `(type, option count)` bucket. pip 0.3.27 clamps
every temperature to `[0.5, 5.0]` (`laya.common.clamp_temperature`). The shipped `choice:11+` temperature is 0.1006,
so pip answers a choice question with 11 or more options with temperature 0.5 and the Hub code with 0.1006; all other
buckets lie inside the range and are unchanged. The server follows pip because pip is the wire-format target and the
evaluation baseline; `LAYA_TEMPERATURE_CLAMP=0` restores the Hub behaviour. `/v1/health` reports both tables and the
clamped buckets. Track R's `reference/laya_reference.py` applies the same clamp by default (`clamp=True`; `clamp=False` is the raw Hub rule), see PLAN.md amendments A6 and A7; its `forward` is what the
server calls, so the two agree on logits and differ in decoding only for `choice:11+`.

## Headers on every `/v1/*` response

- `X-Inference-Time-Ms`: time around the engine call (validation excluded, as in pip). The middleware adds it to any
  `/v1/*` response that lacks it (health, models, errors), there measured around the handler.
- `X-Laya-Device-Ms`: sum over chunks of the backend's `last_device_ms` when the backend sets that attribute after
  each forward, else the wall time of the forward calls. For the CPU backend it is the forward wall time.
- `X-Laya-Batch`: the `<rows>x<seq>` buckets run for the request, comma-separated in execution order (`none` for an
  empty question set).
- `Server-Timing: inference;dur=<ms>` as pip emits.

## Environment knobs

| variable | default | meaning |
|---|---|---|
| `LAYA_BACKEND` | `tt` | `tt` = `tt/engine.py: LayaEngine.from_env()`; `cpu` = `CpuBackend` (Track R's `LayaReference`, fp32) |
| `LAYA_MODEL_DIR` | unset | checkpoint directory with `rl_agent_config.json`, `model.safetensors`, `encoder/config.json`, `tokenizer/`; when unset, `snapshot_download(HF_MODEL, revision=LAYA_REVISION, allow_patterns=[*.json, tokenizer/*, encoder/*, model.safetensors], local_files_only=True)` under `HF_HOME` |
| `HF_MODEL`, `LAYA_REVISION`, `LAYA_SUBFOLDER` | `convaiinnovations/laya`, unset, unset | the snapshot to resolve; the launcher sets `HF_MODEL` from the manifest's weights |
| `LAYA_SEQ_BUCKETS`, `LAYA_ROW_BUCKETS` | cpu: multiples of 64 up to `max_len`; `1,2,4,8,16,32,64` | buckets the CPU backend reports; the TT engine reads them itself and reports them through `shapes()` |
| `LAYA_MAX_ROWS` | 64 | rows per forward chunk (capped by the largest row bucket) |
| `LAYA_MAX_BATCH_TOKENS` | 65536 | `rows_bucket * seq_bucket` per chunk |
| `LAYA_MAX_BATCH_STATES`, `LAYA_MAX_QUESTIONS` | 64, 64 | 413 limits |
| `LAYA_MAX_STATE_CHARS`, `LAYA_MAX_CHOICE_OPTIONS`, `LAYA_MAX_SCORE_LEVELS`, `LAYA_MAX_TOTAL_OPTIONS` | 50000, 100, 32, 512 | pip's 413 limits |
| `LAYA_MAX_TOKEN_BUDGET` | largest seq bucket | cap on `max_len` and `head_max_len` in a request (422 above) |
| `LAYA_TEMPERATURE_CLAMP` | 1 | pip clamp of temperatures to [0.5, 5.0] |
| `LAYA_RAW_FORWARD` | 0 | register `POST /v1/forward` |
| `LAYA_API_KEY` | unset | bearer token for `/v1/*` |
| `LAYA_NO_DEMO` | 0 | do not register `/demo` |
| `LAYA_SANITY_CHECK` | 1 | run the STATE_EN / Q_CHOICE check at startup |
| `LAYA_CORS` | 0 | permissive CORS with the three headers exposed |
| `LAYA_CPU_THREADS`, `LAYA_CPU_ATTN`, `LAYA_CPU_IMPL` | unset (`common.DEFAULT_THREADS`, 6), `eager`, `reference` | CPU backend: torch threads, attention implementation passed to `LayaReference`, `vendored` builds the Hub `DecisionModel` with sdpa instead |
| `LAYA_MESH_SHAPE`, `LAYA_PRECISION`, `LAYA_WARMUP_SHAPES`, `LAYA_TRACE`, `LAYA_TRACE_REGION_SIZE`, `LAYA_L1_SMALL_SIZE` | Track T | read by `LayaEngine.from_env()`; the server only reports what `shapes()` returns |
| `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES` | set to 0 by `make_backend("tt")` with `setdefault` | mandatory on this box |

## Request to rows

1. Validate every question (`check_question`, a port of pip `Agent._check_question`), then normalise it
   (`to_internal`: list criteria to `{label: None}`, noul criteria keys lower-cased, non-string instructions to JSON
   with `ensure_ascii=False`, `option_order` kept).
2. Tokenize the state once per state (`tok(serialize_state(state).replace(mask_token, " "))`); conversation lists are
   truncated from the left as pip does (`truncate_left=True` for a list state).
3. For every question call the vendored `build_sequence(tok, state, q, max_len, head_max_len, option_order,
   truncate_left)` from `vendor/rl_common.py` (the Hub file, byte-identical to the download). A marker count below the
   option count is a 422 with pip's message.
4. Truncation statistics: the head length is measured by calling `build_sequence` with an empty state; `room = max_len -
   head_len - 1`, `state_tokens_used = min(state_tokens, room)`, `dropped = state_tokens - used`. Collapsed-option
   statistics replicate pip `build_head`'s capping (48 tokens per option, even shrink when fewer than 16 tokens remain).
5. Items carry `target`, `label`, `episode`, `ep_step`, `ep_len`, `src` because the Hub `collate_items` requires them
   (pip's version made `target` optional).
6. Rows of all states are sorted by length and cut into chunks: a chunk takes rows while `rows <= min(LAYA_MAX_ROWS,
   largest row bucket)` and `row_bucket(rows) * seq_bucket(longest) <= LAYA_MAX_BATCH_TOKENS`; one row always fits.
7. Each chunk is collated with the vendored `collate_items`, padded with `pad_batch` to the row bucket and the seq
   bucket, run through `backend.forward`, and sliced back to `[:N, :kmax]`. Pad rows: pad id everywhere, attention 0
   (token 0 set to 1 when `shapes()["pad_rows_keep_one_token"]` is true, the CPU default, because SDPA and the head
   layers produce NaN for an all-masked row), marker 0 live, qtype 0. This is pip `backends.base.pad_batch`'s rule.
8. Logits are decoded per state in request order; `usage` uses the real token count of the state's rows.

## Backend contract the server expects (CPU reference and `tt/engine.py` alike)

```
forward(input_ids, attention_mask, marker_pos, marker_mask, qtype) -> (logits, act_logits)
  input_ids      torch.int64 [B, S]      pad id 50283 in padding positions, B and S already padded to a bucket
  attention_mask torch.int64 [B, S]      1 real, 0 pad
  marker_pos     torch.int64 [B, kmax]   0 in unused slots
  marker_mask    torch.bool  [B, kmax]   True for live markers
  qtype          torch.int64 [B]         0 choice, 1 score, 2 noul
  logits         float32 [B, kmax]       anything tensor-like; -1e4 where marker_mask is False (the host decodes only [:k] per row)
  act_logits     float32 [B, 2]          raw logits; the host applies softmax and reports column 0 as act_probability
shapes() -> dict with at least one of:
  "seq_buckets": [int], "row_buckets": [int]    or    "warm_shapes": [[rows, seq], ...]
  and optionally "mesh_shape", "precision", "device", "max_rows", "pad_rows_keep_one_token" (default True), "trace"
optional: attribute last_device_ms (float, ms of the last forward on the device) read after every forward
optional: close()
```

The server pads to the smallest row bucket >= N and the smallest seq bucket >= the longest row; a request longer than
the largest seq bucket is a 422. `LayaEngine.from_env()` must read `LAYA_MODEL_DIR` (the server sets it with
`setdefault` to the resolved snapshot before calling `from_env`).

## Startup and shutdown

The lifespan loads the engine (`Engine.from_env()`), then runs the sanity check: `STATE_EN` (the authors'
`bench_latency.py` ticket) with `Q_CHOICE` (billing / technical / sales) through the full request path and compares the
argmax and probabilities with `server/sanity_reference.json` (recorded 2026 Oct 5 from the CPU reference: `billing`
0.8899, technical 0.0417, sales 0.0685). An argmax mismatch logs a warning and is reported in `/v1/health["sanity"]`;
the server still starts. The ready line `laya server ready backend=... seq_buckets=... row_buckets=... warm=...` is
logged before uvicorn's `Application startup complete`, which is what `tt-model serve` waits for. Shutdown calls
`engine.close()` (the TT engine releases traces and closes the mesh there) and drains the single-worker executor.

Concurrency: one `asyncio.Lock` around every forward (systemone, batch, raw); the work runs in a
`ThreadPoolExecutor(max_workers=1)` so health and static requests stay responsive while the device is busy.

## Demo page (`/demo/`)

- Header: name, a health pill polling `GET /v1/health` every 5 s (`<backend> ready`, then precision, mesh, seq and
  row buckets, trace count), a yellow banner when `backend == "cpu"`, a red banner when the server is unreachable.
- Presets (`demo/presets.json`): support ticket triage (STATE_EN, Q_CHOICE, Q_NOUL plus a tone score), email spam and
  phishing (the `email_utils.email_questions()` fan-out on a synthetic email), guardrail (the two held-out toxic-chat
  nouls from `bench_apps.py`, verbatim, on a synthetic prompt carried under both `prompt` and `post`), RAG relevance
  (the `bench_apps.py` relevance noul plus a completeness score), agent trace observability (case
  `agent_trace_observability_000000` with its gold, drawn as ticks).
- Editor: state (JSON or text; JSON is parsed when it parses, else sent as a string), questions JSON, `max_len`,
  `head_max_len`, `min_confidence`, Decide. Server errors (400, 413, 422) are shown inline with the server's `detail`.
- Answer cards: type badge, instructions, per-option probability bars with the argmax in full colour; score cards add
  a ruler with the expected score; noul cards a two-tone false/true bar; `confidence`, `answer_confidence`, the act /
  escalate tile with the authors' caveat (issue #185), `abstention` when a gate ran, gold ticks when the preset has gold.
- Tiles with sparklines over the last 60 calls: client ms, server ms (`X-Inference-Time-Ms`), device ms
  (`X-Laya-Device-Ms`), batch (`X-Laya-Batch`), input tokens, state tokens, truncated.
- Live feed: `demo/feed.json` = `demo/feed_cases.json` (60 typed-decisions test cases, the first 15 of each workflow,
  revision `d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18`, Apache-2.0, with gold); a JSONL upload replaces it (one case
  per line with `state`, `questions`, optional `id`, `workflow`, `gold`). Controls: rate 0.5 to 20 cases/s,
  concurrency 1 to 4, loop, batch mode (up to 8 states with identical questions per `/v1/systemone/batch`; the feed's
  questions are identical within a workflow), seconds (0 = until the feed ends or Stop), Start, Stop, Copy stats JSON.
  The scheduler launches one unit every `1/rate` s (one case, or 8 cases in batch mode) when a concurrency slot is
  free; it never aborts in-flight requests.
- Table (last 300 rows): id, workflow, five `qid=argmax` cells coloured green when the argmax equals the gold label
  and red otherwise (grey without gold), device ms and batch of the request that carried the case.
- Running stats: decisions/s, cases/s, counts, client and server p50/p95, device p50, agreement with gold, errors, in
  flight, batch shapes, wall time. Latency percentiles are per request (a batch request covers up to 8 cases).
- `?autorun=feed&rate=4&seconds=20` starts the feed on load (also `concurrency`, `batch=1`, `loop=1`, `preset=<id>`,
  `autorun=decide`). `window.layaDemo.statsJson()` returns the current stats object.

Agreement rule: choice: `choice == gold.label`; score: `argmax(probabilities) == gold.label` (the dataset's label is
the level index as a string); noul: `(noul >= 0.5 ? "true" : "false") == gold.label.toLowerCase()`.

### Stats JSON (Copy stats JSON; the shape `evals/demo/feed_client.py` should write)

`/home/hous/dev/laya/evals/README.md` did not exist when this was written, so this is the defining description.

```
{"schema": "laya-demo-feed-stats/1", "source": "demo-page" | "feed_client", "page_url" | "base_url": str,
 "server": {"backend", "precision", "mesh_shape", "seq_buckets", "row_buckets", "model", "revision"},
 "config": {"rate", "concurrency", "loop", "batch", "batch_states": 8, "seconds"},
 "feed": {"source", "dataset", "revision", "cases_available"},
 "started_at", "ended_at": ISO 8601 UTC, "wall_s": float,
 "requests", "cases", "decisions", "errors": int,
 "requests_per_s", "cases_per_s", "decisions_per_s": float,
 "latency_unit": "per request ...",
 "client_ms" | "server_ms" | "device_ms": {"n", "p50", "p95", "mean", "max"},
 "agreement": {"decisions", "agree", "rate", "by_type": {"choice" | "score" | "noul": {"n", "agree", "rate"}}, "rule": str},
 "batch_shapes": {"<rows>x<seq>": count},
 "rows": [{"id", "workflow", "answers": {qid: argmax}, "gold": {qid: label}, "agree", "n", "client_ms", "server_ms", "device_ms", "batch", "error"}]}
```

`p50` and `p95` are nearest-rank percentiles (`ceil(p/100 * n)`-th sorted value). `server_ms` comes from
`X-Inference-Time-Ms`, `device_ms` from `X-Laya-Device-Ms`.

## Shim for pip `laya` (`shim/laya_tt_backend.py`)

pip `laya` 0.3.27's wheel does not ship `laya/backends/` (checked: `laya-0.3.27.dist-info/RECORD` has no `backends`
entry, and `laya/agent.py` imports it lazily), while the pinned clone at the same version has
`laya/backends/base.py` with `class Backend`. The shim imports `laya.backends.base.Backend` when present and otherwise
defines a class with the same `install()` / `uninstall()` semantics (swap `agent.model.forward`, restore on
uninstall). `TtBackend(agent, url="http://host:8710").install()` posts the collated tensors to `/v1/forward` (needs
`LAYA_RAW_FORWARD=1` on the server, which the manifests set) and returns `(logits, act_logits)` as float32 tensors on
the agent's device; `TtBackend.in_process(agent)` wraps `LayaEngine.from_env()` instead. Padding to the server's
buckets happens on the server. Not tested against a live pip Agent in this track (the eval venv lacks uvicorn and the
tt-metal venv lacks pip laya); the HTTP path is exercised by `test_raw_forward_matches_wire_path`.

## Tests

Run in the tt-metal venv from the tt-metal root with the repo's `conftest.py` (it imports no device at collection):

```
source /home/hous/dev/laya/bin/ttenv.sh
cd /home/hous/dev/ornith-1.5-9b/tt-metal
LAYA_CPU_THREADS=4 python -m pytest models/autoports/convaiinnovations_laya/tests/test_host_engine.py models/autoports/convaiinnovations_laya/tests/test_server_cpu.py models/autoports/convaiinnovations_laya/tests/test_demo_page.py -q -p no:cacheprovider -o addopts=""
```

- `test_host_engine.py` (no model load): temperature lookup equals the vendored `temp_bucket` rule; the pip clamp;
  `decode_answer` equals the Hub `rl_agent_api.py` arithmetic (softmax over the raw-temperature-scaled logits,
  `confidence_from_probs`, expected score, `p[1]`, `softmax(act)[0]`) on `tests/fixtures/recorded_logits.json` (STATE_EN
  with a 3-way choice, a noul, a 4-level score and a 12-option choice, recorded from the CPU reference padded to 4x512);
  masked marker slots ignored; recorded answers reproduced; usage fields; `pad_batch` rule; buckets and chunk plan;
  14 malformed questions rejected; `min_confidence` validation and gate states; helpers.
- `test_server_cpu.py` (loads the model once per session, `LAYA_BACKEND=cpu`, `LAYA_CPU_THREADS=4`): 503 while
  loading; `/health`, `/v1/health` (buckets, sanity ok against the stored value), `/v1/models`; the STATE_EN example
  with headers, shape and the stored CPU value; a score question with `max_len=128` truncation; batch of 3 states and
  the same batch chunked with `max_rows=2` (two `X-Laya-Batch` entries, same argmax, probabilities within 1e-3);
  `min_confidence` float, 0.0 and map; eight 422 bodies; four 413 bodies; five 400 bodies; empty questions; `/v1/forward`
  against the wire path (probabilities within 2e-3) and a bad qtype; raw forward absent without the flag; bearer auth;
  the module-level `app`.
- `test_demo_page.py`: demo files present and free of em and en dashes, no external URLs in the HTML; presets and feed
  validate (5 presets, 60 cases, 4 workflows, 5 questions each); `/demo` 307, `/demo/` 200 with stamped asset URLs
  and `no-cache`, static files, `presets.json`, `feed.json`, a 404; every preset decides with probabilities summing to
  1 within 1e-3 and gold keys matching; a feed case decides with gold labels inside the answer keys.

Results on 2026 Oct 5 (logs under `/home/hous/dev/laya/logs/p2_tests_*`): `test_host_engine.py` 25 passed,
`test_server_cpu.py` 22 passed, `test_demo_page.py` 5 passed. Details in `work_log.md` next to this file.

## Known gaps and decisions

- The CPU backend speed on this host is dominated by other tracks' jobs (load average 30 to 48 on 16 threads on
  2026 Oct 5 21:20 UTC); timings recorded under that load are not meaningful. Use at most 4 threads for test runs.
- `labels` on noul questions is rejected with 422 because the vendored Hub `render_options` does not render labels;
  pip renders them into the option text.
- The feed file is an object `{"attribution": {...}, "cases": [...]}`, not a bare list; the Appendix B verify line was
  changed to `len(f["cases"])`.
- `LAYA_ROW_BUCKETS_1024` in the sibling manifest is a proposal for Track T's engine; rename if Track T chooses
  another name.
