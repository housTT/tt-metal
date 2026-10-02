# CLM-v0.1-8B serving: host-side code

This document covers the device-independent serving code of the CLM (Contrastive Language Model) autoport:
the vendored `clm` engine, the FastAPI server, the Hugging Face (HF) CPU reference embedder, and the interface that
the Tenstorrent (TT) encoder in `tt/encoder.py` must satisfy.

All paths are under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/`.

| Path | Role |
| --- | --- |
| `clm/` | Vendored upstream package (schema, engine, heads, vector arena, HTTP embedder, client, playground). See `clm/VENDORED.md`. |
| `tt/heads.py` | Thin wrapper over the vendored `HeadPair`: loads `CLM_v0.1-8B.pt`, projects embeddings, reports the head config and clamped scale. |
| `reference/hf_embedder.py` | `HfQwen3Embedder`: Qwen3-8B on CPU through HF transformers, last-token pooling, L2 normalisation. |
| `server/app.py` | FastAPI app factory `create_app(...)`, module-level `app`, `main()`. |
| `server/__main__.py` | `python -m models.autoports.contrastive_lm_clm_v0_1_8b.server`. |
| `tests/test_host_engine.py` | CPU tests with a fake embedder (no Qwen3-8B load). |
| `tests/test_hf_embedder_smoke.py` | Opt-in test (`CLM_RUN_HF=1`) that loads Qwen3-8B on CPU. |

## What the model computes

The encoder is `Qwen/Qwen3-8B` (revision `b968826d`), last-token pooling of the final hidden state after the final
RMSNorm, L2-normalised, 4096 wide. The heads are two MLPs (state head, action head), 9,443,840 parameters each:
`Linear(4096, 1536) -> GELU -> Linear(1536, 1536) -> LayerNorm(1536) -> GELU -> Linear(1536, 512)`. The checkpoint
`/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt` carries `logit_scale = 4.6132`, so the scale is
`exp(4.6132) = 100.81`, clamped to `100.0` by the vendored `HeadPair`. A candidate's logit is
`100.0 * cos(normalize(state_head(e_state)), normalize(action_head(e_candidate))) / temperature`, and the softmax
over a question's candidates is the answer distribution. The state text is `context + "\n\n" + question`; candidate
texts are embedded verbatim (see `clm/schema.py`).

## Endpoints

The ASGI target is `models.autoports.contrastive_lm_clm_v0_1_8b.server.app:app`. Routes match the upstream
`clm-serve` server (`src/clm/server.py` at upstream commit `bb42c6c5`), plus `POST /v1/embeddings`.

| Method and path | Request | Response |
| --- | --- | --- |
| `POST /v1/systemone` | `{"state": str or object or array, "model": "clm-latest", "questions": {id: Question}, "temperature": 1.0}` | `{"model", "answers": {id: Answer}, "usage": {"billing_units", "input_tokens", "output_tokens"}}` |
| `POST /v1/rank` | `{"context": ..., "question": str or null, "answers": [str, ...], "model", "temperature"}` | `{"model", "ranked": [{"rank", "candidate", "prob"}, ...]}` best first |
| `POST /v1/embeddings` | `{"model": str, "input": str or [str] or [int] or [[int]], "encoding_format": "float" or "base64", "truncate_prompt_tokens": int}` | `{"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...] or base64}], "model", "usage": {"prompt_tokens", "total_tokens"}}` |
| `GET /v1/models` | | `{"models": [{"name": "clm-latest", "description", "release_date": "2026-09-19"}, {"name": "clm-raw", ...}]}` |
| `GET /health` | | `{"ok", "ready", "embedder": bool, "embedder_kind", "models": [names], "cache": arena stats or null}` |
| `GET /` | | The playground (static files from `clm/static/`), unless `CLM_NO_UI=1` |
| `GET /demo/` | | The T-Rex live demo page (`clm/demo/`), unless `CLM_NO_UI=1` or `CLM_NO_DEMO=1`; `GET /demo` redirects to it |
| `WS /demo/ws` | `{"type": "start", "seeds", "seed", "duration", "shield", "inflight", "api_key"}`, `{"type": "stop"}`, `{"type": "ping"}` | `hello`, `status`, `course`, `frame`, `stats`, `course_end`, `summary`, `busy`, `error`, `stopped`, `pong` (section "T-Rex live demo") |

Question and Answer objects follow the upstream TypeSafe wire schema (`noul`, `choice`, `score`), see the upstream
README section "API Reference". `temperature` must be in `(0, 100]`.

Every `/v1/*` response carries `X-CLM-Latency-Ms`. For `/v1/systemone`, `/v1/rank` and `/v1/embeddings` it is the
server-side wall time of the engine call, measured exactly as upstream does (body parsing excluded). For other
`/v1/*` responses, including errors, a middleware fills it with the handler wall time.

Errors: `401` bad API key (when `CLM_API_KEY` is set; `/health` is never authenticated), `422` malformed request,
unknown model or unknown question type, `502` embedder failure (`clm.embedder.EmbedderError`), `503` engine not
ready, `501` token-id input with an embedder that has no `embed_ids` and no tokenizer.

### `POST /v1/embeddings` details

- `input` as a string or list of strings is embedded as text. A list of integers is one token-id sequence; a list of
  integer lists is a batch of token-id sequences.
- Token ids go to `embedder.embed_ids(id_lists)` when the embedder has it (the HF reference and the vendored HTTP
  embedder do). Otherwise the server decodes them with a tokenizer (`embedder.tokenizer` if present, else
  `AutoTokenizer.from_pretrained(CLM_TOKENIZER or HF_MODEL or "Qwen/Qwen3-8B")`) and calls `embed`.
- `truncate_prompt_tokens`: `-1` or absent means the embedder's own `max_tokens`. A positive `k` keeps the last `k`
  tokens: applied directly to token-id inputs, and to text inputs only when the embedder has both `tokenize` and
  `embed_ids`; otherwise text inputs are truncated by the embedder at its `max_tokens`.
- `encoding_format: "base64"` returns the float32 little-endian bytes of each vector, base64 encoded (the vLLM and
  OpenAI convention; the vendored HTTP embedder decodes exactly this).
- `model` is echoed back when given, else the embedder's `name` (`qwen3-8b` by default). The server does not reject
  unknown model names on this route.
- `dimensions`, if given, must be `4096`.
- `usage.prompt_tokens` is what the embedder reports: tokens processed after truncation (the HF reference counts all
  inputs; the HTTP embedder forwards vLLM's number, which covers cache misses only).

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLM_EMBEDDER` | `tt` | `tt`: `models.autoports.contrastive_lm_clm_v0_1_8b.tt.encoder.TtQwen3Encoder.from_env()`; `hf`: `reference/hf_embedder.py`; `http`: vendored HTTP embedder against `CLM_EMB_URL`. The `tt` module is imported only in that branch. |
| `CLM_CKPT` | `/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt` | Head checkpoint served as `clm-latest`. |
| `CLM_ACTION_CACHE` | `512MiB` | Vector arena budget: an absolute size (`512MiB`, `2GB`), a fraction of device memory (`0.02`, CPU assumes 8 GiB), or `0` to disable. |
| `CLM_MAX_TOKENS` | `2048` | Encoder truncation length, passed to the `hf` and `http` embedders. `from_env()` of the TT encoder should read it too. |
| `CLM_MESH_SHAPE` | `1x1` | TT encoder mesh (`1x1` one chip, `1x4` four chips tensor parallel); `tt-model` sets it from the profile's `mesh_device`. |
| `CLM_MAX_BATCH` | `8` | Largest prefill batch; traces are captured for batch 1, 4 and this value. |
| `CLM_MAX_SEQ_LEN` | `CLM_MAX_TOKENS` | Longest prefill bucket; must be a multiple of 128 and at least `CLM_MAX_TOKENS`. |
| `CLM_PRECISION` | `accuracy_lofi_mlp` in the package | Precision policy: `accuracy`, `accuracy_lofi_mlp`, `bfp8_attn`, `bfp8_attn_hifi2`, `bfp8_lofi_mlp`, `performance` (`tt/encoder.py` `CUSTOM_POLICIES`, `doc/datatype_sweep/README.md`). |
| `CLM_TRACE_LENS` | `128,256,512,1024,2048` | Prefill buckets (multiples of 128, each captured at every batch size); texts pad to the smallest bucket that fits. |
| `CLM_PROGRAM_CONFIGS` | `1` | `0` restores the stock matmul program configs (QKV block shape at 128 tokens, 11x10 MinimalMatmul grid above 128 tokens); single-chip meshes only. |
| `CLM_SHARDED_NORM` | `1` | `0` restores the stock interleaved RMSNorm (the block-sharded norm is used for 128, 256 and 512 prefill rows); single-chip meshes only. |
| `CLM_TRACE_REGION_SIZE`, `CLM_L1_SMALL_SIZE` | `200000000`, `32768` | Device open parameters (bytes). |
| `CLM_FABRIC_CONFIG` | unset (`FABRIC_1D` in the `p150x4` profile) | Fabric configuration for multi-chip meshes. |
| `CLM_API_KEY` | unset | When set, `/v1/*` routes need `Authorization: Bearer <key>`. |
| `CLM_NO_UI` | unset | `1` disables the playground at `/` and the demo at `/demo/`. |
| `CLM_NO_DEMO` | unset | `1` disables the T-Rex demo (`/demo/`, `/demo/ws`) while keeping the playground. |
| `CLM_DEMO_BASE_URL` | derived | Where the demo's player process posts its `/v1/systemone` requests. Default `http://127.0.0.1:<port the server is bound to>` (read from the WebSocket's own server address, so it is right inside the container whatever port tt-model published). Set it only behind a proxy or in tests. |
| `CLM_CORS` | unset | `1` allows browser requests from any origin and exposes the latency header. |
| `CLM_WARMUP` | `1` | `0` skips the start-up warm-up call (`engine.rank("clm server warm-up", ["clm server warm-up"])`). |
| `CLM_DEVICE` | `cpu` when CUDA is absent | Torch device of the heads and the arena. |
| `PORT`, `CLM_PORT` | `8700` | Port when `--port` is not given (`--port` > `PORT` > `CLM_PORT` > 8700). |
| `CLM_HOST` | `0.0.0.0` | Bind address when `--host` is not given. |
| `CLM_LOG_LEVEL` | `info` | uvicorn log level. |
| `CLM_EMB_URL`, `CLM_EMB_MODEL`, `CLM_EMB_API_KEY` | `http://127.0.0.1:8090/v1/embeddings`, `qwen3-8b`, unset | `http` mode only. |
| `HF_MODEL` | `Qwen/Qwen3-8B` | HF model id for the `hf` embedder and the token-id decode fallback. |
| `CLM_HF_DTYPE` | `float32` | `float32` or `bfloat16` for the `hf` embedder. |
| `CLM_HF_THREADS` | `8` | `torch.set_num_threads` for the `hf` embedder. |
| `CLM_HF_REVISION` | unset | HF revision for the `hf` embedder. |
| `CLM_TOKENIZER` | unset | Tokenizer id for the token-id decode fallback. |

The embedder and the engine are constructed inside the FastAPI lifespan, before uvicorn serves. The lifespan logs
`clm server ready embedder=<kind>` and only then does uvicorn print `Application startup complete.`, so that line
means the model is loaded and warm.

## Running the server in `hf` mode (API testing without a device)

```
source /home/hous/dev/clm-v0.1-8B/bin/ttenv.sh
cd /home/hous/dev/ornith-1.5-9b/tt-metal
CLM_EMBEDDER=hf CLM_HF_THREADS=8 python -m models.autoports.contrastive_lm_clm_v0_1_8b.server --port 8700
```

Qwen3-8B in float32 takes about 32 GB of RAM. Measured on this box (AMD Ryzen 7 9700X, 8 cores, `avx512_bf16`,
8 threads, safetensors already in the page cache) on 2026 Oct 1, see
`/home/hous/dev/clm-v0.1-8B/logs/track_e1.log`:

| | `CLM_HF_DTYPE=float32` (default) | `CLM_HF_DTYPE=bfloat16` |
| --- | --- | --- |
| Tokenizer plus model load | 7.0 s | 1.3 s |
| Two short texts (15 tokens) | 7.1 s | not measured |
| README customer example, cold cache (10 texts, 98 tokens) | 37 to 47 s | 6.6 s |
| Cosine to the float32 vectors, README texts | 1 | >= 0.9999 |

`bfloat16` is the practical setting for interactive API testing; `float32` is the fidelity baseline. A cold load
from disk (16 GB) adds the disk read time.

The published README answers for the customer example (`urgency 0.41022`, `billing 0.93878`, `frustration 1.98386`,
"106 tokens on a cold cache") are not reproduced by this reference: it gives `0.842`, `0.988`, `2.000` and 98 cold
tokens in both dtypes. The token count differs, so the README run embedded different texts (or used a different head)
than the published commit `bb42c6c5` produces; the upstream history is a single squashed commit, so the cause cannot
be checked. The qualitative answers agree (`billing`, very angry).

Then:

```
curl -s http://127.0.0.1:8700/health
curl -s http://127.0.0.1:8700/v1/models
curl -s -D - http://127.0.0.1:8700/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "Customer: my invoice was charged twice and nobody answers the phone!",
  "model": "clm-latest",
  "questions": {
    "urgency": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}}}'
curl -s http://127.0.0.1:8700/v1/rank -H 'Content-Type: application/json' -d '{
  "context": "What causes tides on Earth?",
  "answers": ["The Moon'"'"'s gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."]}'
curl -s http://127.0.0.1:8700/v1/embeddings -H 'Content-Type: application/json' -d '{
  "model": "qwen3-8b", "input": ["Hello world"], "encoding_format": "float"}' | head -c 300
```

The upstream Python client works unchanged against this server:
`from models.autoports.contrastive_lm_clm_v0_1_8b.clm import CLMClient, Noul, Choice, Score`.

Other modes: `CLM_EMBEDDER=http CLM_EMB_URL=http://host:8090/v1/embeddings` points at any OpenAI-compatible
Qwen3-8B pooling server (the upstream `vllm serve Qwen/Qwen3-8B --runner pooling` set-up). `CLM_EMBEDDER=tt` (the
default) is the device path once `tt/encoder.py` exists.

## T-Rex live demo at `/demo/`

The quickstart demo: the CLM repository's T-Rex harness (`examples/t_rex`, vendored unchanged into `server/trex/`, see
`server/trex/VENDORED.md`) runs headless inside the server's container at 60 FPS, the model on the chip chooses jump,
duck or run for every decision, and a browser page renders the game and every decision live.

Processes. `tt-model serve` runs uvicorn (`server/app.py`). A `start` message on `/demo/ws` makes `server/demo.py`
spawn a **runner** process (`server/demo_runner.py`: the upstream `Arena` with a `Pilot` subclass that collects every
decision and shield event, one course per seed) which spawns the upstream **brain** process (`server/demo_brain.py`:
the upstream `RemoteBrain` and `serve()`, with three module globals replaced so that each decision also carries the
server's `X-CLM-Latency-Ms`, the state text and the three option texts). The brain posts to this server's own
`POST /v1/systemone` at `CLM_DEMO_BASE_URL` (default: the server's bound port on 127.0.0.1). The runner writes
compact JSON messages to a pipe; the server relays them to every connected page, coalescing frames when a viewer
lags and never dropping a decision or event. One game runs at a time (`busy` to a second `start`); any number of
pages may watch it; the game stops ten seconds after the last page disconnects, on `stop`, or at server shutdown.
The runner is not a daemon process (it must spawn the brain); the server terminates it on stop and shutdown.

Messages. Client: `start` (seeds 1 to 10, seed, duration 5 to 180 s, shield, inflight 1 to 8, `api_key` when the
server has `CLM_API_KEY`), `stop`, `ping`. Server: `hello` (running flag, config, viewer count, whether a key is
required; followed by the cached `course` and `stats` when a game is live), `status` (phase starting, warming,
playing, between, finished, stopped, error), `course` (index, seed, endpoint, warm-up samples, latency frames),
`frame` (one per 60 FPS tick: game state as compact arrays, the decisions that landed in the tick, pilot events, the
in-flight count), `stats` (every 0.5 s), `course_end` (the same row as `examples/t_rex/run.py` plus `server_ms_p50`
and `server_ms_p95`), `summary` (the same summary plus `server_ms_p50_median`, and all rows), `busy`, `error`,
`stopped`, `pong`. A decision entry carries `seq, frame, state, instructions, criteria, p, proposed, executed, best,
safe, intervened, arrival, agreed, airborne, threat, distance, latency_ms, inference_ms, server_ms, plan_ms,
input_tokens, dropped, error, event`. Timings: `latency_ms` is the pilot's round trip from asking to applying,
`inference_ms` the HTTP call measured in the brain process, `server_ms` the server's header.

Page (`clm/demo/`). Plain HTML, CSS and ES2020 with no build step; shares the playground's palette and theme toggle.
Canvas renderer with vector shapes (no sprite sheet exists upstream) for the dino, cacti, birds, ground, clouds,
night mode and score; a decision card with the state text, the three option texts and probability bars, the model's
pick, the executed action, the planner's best and shield replacements, with sparklines of the three latencies;
counters from `stats`; the event log; controls; and a results table per course in the shape of the upstream result
rows with the authors' RTX 4090 file (`reference_rtx4090.json`, the repository's `results/clm_realtime.json`)
alongside. Relative URLs resolve under `/demo/`, which is why `/demo` redirects.

Effect on other clients. While a game runs, other `/v1` requests queue behind at most `inflight` demo requests
(about one millisecond each on a cached state, about 56 ms on a new state text on one p150). Do not run latency
measurements while the demo plays.

Tests and checks: `tests/test_demo.py` (host only, mock encoder: frame encoding, row keys, routes, WebSocket validation
and API key, a 5 s course end to end behind a real uvicorn, stop mid game); on a device, serve the package and drive a
game with `/home/hous/dev/clm-v0.1-8B/evals/trex/demo_ws_client.py`.

## Tests

`python -m pytest models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_demo.py -q` covers the demo (section above).

```
source /home/hous/dev/clm-v0.1-8B/bin/ttenv.sh
cd /home/hous/dev/ornith-1.5-9b/tt-metal
python -m pytest models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_host_engine.py -q
CLM_RUN_HF=1 python -m pytest models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_hf_embedder_smoke.py -q -s
```

`test_host_engine.py` runs in a few seconds on CPU and does not load Qwen3-8B. It uses a fake embedder
(deterministic random unit vectors keyed by a hash of the text) with the real heads, engine, arena and FastAPI app.

The checkout's root `conftest.py` imports `ttnn` at collection time (through
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/demo/trace_region_config.py`). The tests here do not
need any root fixture, so a process that must not import `ttnn` can add `--noconftest`:
`python -m pytest models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_host_engine.py -q --noconftest`.

## Embedder interface contract (what `tt/encoder.py` must satisfy)

The engine (`clm/engine.py`, `EmbedderLike`) and the server (`server/app.py`) use the embedder through this surface.

Required:

| Member | Contract |
| --- | --- |
| `TtQwen3Encoder.from_env()` | Class method. Builds the encoder from environment variables (`CLM_MAX_TOKENS`, `HF_MODEL` for weights, plus the device variables the TT track defines). Opens the mesh device, loads weights, compiles or traces, so that the server's `Application startup complete` means warm. |
| `max_tokens: int` | The truncation length the encoder applies (2048 by default). Read by the server for `truncate_prompt_tokens` clamping. |
| `embed(texts: list[str]) -> tuple[np.ndarray, int]` | Returns `(vectors, tokens)`. `vectors` is a `np.ndarray` of dtype `float32`, shape `[len(texts), 4096]`, each row L2-normalised (norm within `1e-4` of 1), row `i` for `texts[i]`, no NaN or Inf. `tokens` is an `int`: encoder tokens processed for these inputs after truncation. Must accept any batch size (the engine sends 1 to a few dozen texts per call; the HTTP embedder batches at 32) and empty strings (embed a single space token, like the training recipe). |
| `healthy() -> bool` | `True` when the device is open and the model is loaded; reported by `GET /health` as `embedder`. |

Semantics `embed` must reproduce:

- Tokenise with the Qwen3-8B tokenizer and `add_special_tokens=False` (Qwen3 adds none anyway). Keep the last
  tokens when truncating. The HF reference (`reference/hf_embedder.py`) keeps the last `max_tokens - 1` tokens,
  matching the upstream training recipe (`train/embed_utils.py`); the upstream vLLM serving path keeps the last
  `max_tokens`. Use `max_tokens - 1` to match the reference.
- Pool the last token's hidden state after the final RMSNorm (the `last_hidden_state[:, -1]` of HF `Qwen3Model`),
  cast to float32, L2-normalise.
- Deterministic: the same text alone and inside a mixed-length batch gives the same vector (stage 6 gate: cosine
  `>= 0.999`); identical texts give identical vectors.
- Thread-safe: the server calls `embed` from its default thread pool (`run_in_executor`) and may do so concurrently.
  The encoder must serialise device access itself (a lock around the device call).
- Failures of the backend raise `models.autoports.contrastive_lm_clm_v0_1_8b.clm.embedder.EmbedderError` so the
  server answers `502`. Any other exception becomes a `500`.

Optional, used when present:

| Member | Contract |
| --- | --- |
| `embed_ids(id_lists: list[list[int]])` | Pre-tokenised inputs. Returns either `(vectors, tokens)` like `embed`, or the bare `float32 [n, 4096]` L2-normalised array; the server accepts both and, for the bare form, reports `tokens` as the number of ids it sent. The server truncates every id list to the last `truncate_prompt_tokens` ids, or to the last `max_tokens` ids when the request gives none, before calling `embed_ids`, so the encoder may assume `len(ids) <= max_tokens`. Enables `POST /v1/embeddings` with token-id inputs without a decode round trip. |
| `tokenize(text: str) -> list[int]` | Token ids of `text` (no special tokens; an empty text gives the ids of a single space), truncated or not. With `embed_ids`, enables `truncate_prompt_tokens` smaller than `max_tokens` for text inputs: the server keeps the last `k` of what `tokenize` returns. |
| `tokenizer` | An object with `decode(ids) -> str`; used to decode token-id inputs when `embed_ids` is absent. |
| `name: str` | Served encoder name echoed as `model` in `/v1/embeddings` responses (default `qwen3-8b`). |
| `stats() -> dict` or `info() -> dict` | JSON-serialisable counters or description; `GET /health` reports it as `embedder_stats` (`stats` preferred when both exist). |

The HF reference `HfQwen3Embedder` implements all of the above and is the fidelity baseline: for the same text, the TT
encoder's vector should have cosine `>= 0.99` (mean) and `>= 0.97` (min) with the reference over the fidelity corpus.

### State of `tt/encoder.py` against this contract (read on 2026 Oct 1, not executed by this track)

`TtQwen3Encoder` in `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tt/encoder.py`
provides `from_env()`, `max_tokens`, `embed(texts) -> (array, tokens)`, `embed_ids(id_lists) -> array` (bare form,
no truncation inside), `tokenize(text)` (returns the last `max_tokens` ids), `tokenizer`, `healthy()`, `stats()`,
`warmup()` (also gated by `CLM_WARMUP`) and `close()`. It has no `name`, so `/v1/embeddings` echoes `qwen3-8b`.

One open discrepancy: the TT encoder keeps the last `max_tokens` (2048) tokens, the HF reference keeps the last
`max_tokens - 1` (2047) as this track was specified. The two differ only for texts longer than 2047 tokens; the
fidelity corpus includes 2048-token excerpts, so the owners of the two files must pick one value before stage 6.
