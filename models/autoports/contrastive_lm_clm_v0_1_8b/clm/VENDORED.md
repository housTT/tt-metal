# Vendored `clm` package

This directory is a vendored subset of the `clm` Python package from the upstream
Contrastive Language Models repository.

| Field | Value |
| --- | --- |
| Upstream repository | https://github.com/Contrastive-LM/CLM |
| Upstream path | `src/clm/` |
| Upstream commit | `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` (2026-09-24 23:09:47 UTC, "README: note how to raise the 2048-token limit") |
| Vendored on | 2026 Oct 1 |
| Upstream license | Apache-2.0. The upstream `LICENSE` file is copied to `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/clm/LICENSE`. |
| Upstream package version | `0.1.0` (`__version__` in `__init__.py`) |
| Upstream release tag served by `GET /v1/models` | `RELEASE = "2026-09-19"`, model names `clm-latest` and `clm-raw` (unchanged) |

## Files

| File | Status |
| --- | --- |
| `__init__.py` | unchanged |
| `py.typed` | unchanged |
| `schema.py` | unchanged |
| `client.py` | unchanged |
| `cache.py` | unchanged. Verified on CPU: `VectorArena("cpu", "512MiB")` allocates 512 MiB and carves a 229,376 x 512 pool plus a 4,096 x 4096 pool; `VectorArena("cpu", "64MiB")` gives 28,672 x 512 plus 512 x 4096; lookups hit on repeated texts. |
| `heads.py` | unchanged. Verified: `default_device()` returns `cpu` when `torch.cuda.is_available()` is false and `CLM_DEVICE` is unset; torch is imported only inside functions. |
| `engine.py` | modified, see below |
| `embedder.py` | modified, see below |
| `static/index.html`, `static/app.css`, `static/app.js` | unchanged (the playground) |
| `LICENSE` | copy of the upstream repository `LICENSE` |

Not vendored: `server.py` (replaced by
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/server/app.py`, which keeps the
same routes and adds `POST /v1/embeddings`), and the upstream `tools/`, `train/`, `evaluation/`, `preprocessing/`
and `examples/` directories.

## Modifications

### `engine.py`

1. Added `EmbedderLike`, a `typing.Protocol` (runtime checkable) that names what the engine needs from an encoder:
   `embed(texts: list[str]) -> tuple[np.ndarray, int]` (float32 `[n, 4096]`, L2-normalised, plus encoder tokens
   spent), `healthy() -> bool`, and a `max_tokens` attribute. `Engine.__init__` is annotated with it.
2. The HTTP `Embedder` is no longer imported at module import time. `Engine.__init__` imports it lazily only when no
   embedder object is passed, so an engine built around a device-resident encoder does not need `requests`.
   Default behaviour (no embedder given, `CLM_EMB_URL` / `CLM_EMB_MODEL`) is unchanged.
3. In `Engine._cached`, the embedder output is coerced with `np.ascontiguousarray(emb, dtype=np.float32)` and the
   token count with `int(spent)` before use, so an embedder that returns a non-contiguous array or a numpy integer
   still satisfies `torch.from_numpy` and JSON serialisation.

### `embedder.py`

1. `Embedder._fetch` accepts `list[str] | list[list[int]]` (the vLLM `/v1/embeddings` endpoint accepts both).
2. Added `Embedder.embed_ids(id_lists)` for pre-tokenised inputs. It bypasses the text LRU cache and returns the same
   `(vectors, tokens)` tuple as `embed`. `POST /v1/embeddings` uses it for token-id inputs.

No other behaviour changed. Docstrings and comments in the vendored files are the upstream ones; new code added to
these two files follows the upstream style.

## Refreshing

```
git -C <clone of https://github.com/Contrastive-LM/CLM> rev-parse HEAD
diff -u <clone>/src/clm/engine.py /home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/clm/engine.py
diff -u <clone>/src/clm/embedder.py /home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/clm/embedder.py
```

Re-apply the two modifications above after copying a newer upstream, update the commit field in this file, and run
`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_host_engine.py`.
