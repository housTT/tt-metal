# Kokoro-82M — Serving via tt-inference-server media server (DONE)

> Provenance note: this documents the tt-inference-server **serving** path, which
> host-vocodes by design (mirroring the SpeechT5 runner) and runs only plbert on
> device. It is not a limit of the tt-metal model — that now runs the **entire**
> pipeline on device via `../../tt/device_pipeline.py` (`synthesize_device`).

The auto-bringup's vLLM stage was **BLOCKED** by design: vLLM only serves
autoregressive causal LMs, and Kokoro-82M is a non-autoregressive StyleTTS2 /
ISTFTNet TTS model (see `../vllm_integration/`). This stage serves Kokoro the
correct way — through **tt-inference-server's media server** (the same MEDIA
inference engine that serves SpeechT5/Whisper) — and proves it end to end:
text in over HTTP, a WAV file back.

## Result

```
POST /v1/audio/speech  ->  HTTP 200, Content-Type: audio/wav
"Hello from Tenstorrent. Kokoro is now serving through the media server."
  -> 4.9 s of 24 kHz mono WAV in ~2.0 s (single-sentence, warmed)
```

Validated: single + multi-sentence (chunked) input, `response_format` wav/json,
voice selection via `speaker_id` (`af_heart` default, `af_bella`, …). No
device-side errors or host fallbacks in the server log.

## Architecture — TT transformer + host vocoder (mirrors the SpeechT5 runner)

Only Kokoro's `bert` (plbert) is a transformer, and it is the component brought
up on device in `../../tt/` (TP=4 head-parallel, `(1,4)` ring mesh across all
four p300c chips, BFP8/HiFi2, PCC ≥ 0.995 vs HF). The rest of the pipeline is
convolutional/recurrent and runs on host in torch:

```
text --EspeakFallback G2P--> phonemes
     --KModel(bert = TT plbert)--> 24 kHz waveform
        bert_encoder / predictor / text_encoder / ISTFTNet decoder  (host torch)
```

The runner (`tt-inference-server/tt-media-server/tt_model_runners/kokoro_runner.py`,
`TTKokoroRunner`) builds the TT generator with
`hexgrad_kokoro_82m.tt.generator.build_generator(model_dir, mesh_device)` and
replaces `kokoro.model.KModel.bert` with `_TTBert`, an `nn.Module` whose
`forward(input_ids, attention_mask)` returns `gen.model.forward(ids)` — the same
`last_hidden_state [1, S, 768]` the stock `CustomAlbert` would. Fidelity check:
TT-vs-CPU bert hidden PCC = 0.9988, identical predicted durations.

### Audio quality / chunk joins

Text up to ~510 phoneme tokens (~30 s) is one plbert pass, so Kokoro renders
inter-sentence pauses itself; the served audio is metrically identical to the
reference model (spectrogram, per-window energy, and impulse-click content all
match misaki+CPU-bert — the TT plbert adds no clicks). Longer text is split at
sentence boundaries into independent passes; to avoid the lost inter-sentence
pause and any boundary transient, chunks are separated by a natural pause
(`INTER_CHUNK_PAUSE_S`) and each chunk edge + every response's absolute
start/end get a short raised-cosine fade (`BOUNDARY_FADE_MS`).

### Why espeak G2P (not misaki's spaCy path)

misaki's English G2P needs spaCy/thinc, whose wheels want a numpy-2 ABI; the
tt-metal `python_env` is pinned to numpy 1.26 (ttnn's ABI). So G2P uses
`misaki.espeak.EspeakFallback` (libespeak-ng via `phonemizer-fork` /
`espeakng-loader`, all pure-python), which maps espeak output straight into
Kokoro's vocab. `spacy` is stubbed only so `import kokoro` succeeds; we never
build `misaki.en.G2P`.

## Wiring (in tt-inference-server, branch `hous/kokoro-8m-p150`)

* `tt-media-server/tt_model_runners/kokoro_runner.py` — the runner (new).
* `tt-media-server/tt_model_runners/runner_fabric.py` — `TT_KOKORO_82M` → runner.
* `tt-media-server/config/constants.py` — `SupportedModels.KOKORO_82M`,
  `ModelNames.KOKORO_82M="Kokoro-82M"`, `ModelRunners.TT_KOKORO_82M="tt-kokoro-tts"`,
  added to `MODEL_SERVICE_RUNNER_MAP[TEXT_TO_SPEECH]` and
  `INFERENCE_MODEL_RUNNER_TO_MODEL_NAMES_MAP`, and a `ModelConfigs`
  `(TT_KOKORO_82M, P300X2)` entry with `device_mesh_shape=(1,4)`,
  `device_ids=DEVICE_IDS_4_GROUP` (one worker owns all four chips as one mesh).
* `workflows/model_spec.py` + `workflows/model_specs/{dev,prod}/audio_tts.yaml` —
  `kokoro_tts` impl + P300X2 template (catalog/docker discovery).

## Device modes (dual-mode runner)

The runner picks the TT plbert path from `settings.device_mesh_shape`:

* **P150 (single chip), `DEVICE=p150`, mesh `(1,1)`** — the single-chip
  `tt/optimized_decoder.py` `OptimizedDecoder` (packed-QKV + fused SDPA + BFP8/
  HiFi2, PCC 0.9988 vs HF). No ring fabric, no CCL. On a multi-chip host the
  worker restricts itself to one chip via `TT_VISIBLE_DEVICES`, so the runner
  sets `TT_MESH_GRAPH_DESC_PATH` to `p150_mesh_graph_descriptor.textproto` before
  opening the `(1,1)` mesh (else `open_mesh_device` asserts on the missing
  descriptor). Warmup ~10 s.
* **p300x2 (4 chips), `DEVICE=p300x2`, mesh `(1,4)`** — the TP=4 multichip ring
  (`OptimizedMultichipDecoder`), 1D ring fabric via auto-discovery. Warmup ~16 s.

Both feed the same host KModel (predictor + ISTFTNet vocoder) and produce
identical-quality 24 kHz audio.

## Launch + curl (direct uvicorn — the MEDIA engine can't use `run.py --local-server`)

```bash
export TT_METAL_HOME=/home/ttuser/dev/tt-metal
export PYTHONPATH=$TT_METAL_HOME/ttnn:$TT_METAL_HOME:/home/ttuser/dev/tt-inference-server/tt-media-server
export MODEL=Kokoro-82M MODEL_RUNNER=tt-kokoro-tts SERVICE_PORT=8000
export DEVICE=p150          # single chip; use DEVICE=p300x2 for the 4-chip ring
cd /home/ttuser/dev/tt-inference-server/tt-media-server
$TT_METAL_HOME/python_env/bin/python -m uvicorn --host 0.0.0.0 main:app --lifespan on --port 8000

curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H 'Authorization: Bearer your-secret-key' -H 'Content-Type: application/json' \
  -d '{"text":"Hello from Tenstorrent.","response_format":"wav"}' --output speech.wav
```

## Host deps added to `python_env` (installed `--no-deps` / constrained to protect torch 2.11 + numpy 1.26)

`kokoro==0.9.4`, `misaki==0.9.4` (no-deps); `phonemizer-fork`, `espeakng-loader`,
`num2words`, `loguru`, `segments`, `language-tags`, `rdflib` (pure-python);
media-server framework: `fastapi`, `uvicorn`, `pydantic-settings`,
`python-multipart`, `prometheus-fastapi-instrumentator`. NOT installed:
`torch==2.7.1`/`torchaudio`/`whisperx`/`silero_vad`/`spacy` (would break ttnn's
torch/numpy; not needed for Kokoro).
