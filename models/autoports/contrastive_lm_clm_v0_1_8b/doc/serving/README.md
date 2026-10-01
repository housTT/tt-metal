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
| `CLM_API_KEY` | unset | When set, `/v1/*` routes need `Authorization: Bearer <key>`. |
| `CLM_NO_UI` | unset | `1` disables the playground at `/`. |
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

## Tests

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
