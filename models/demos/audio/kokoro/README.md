# Kokoro-82M

## Platforms:
    Blackhole (p150)

Secondary / experimental: Blackhole p300x2 (4-chip TP=4 ring). The p150 single-chip path is the primary supported target.

## Introduction

[hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a non-autoregressive
StyleTTS2 / ISTFTNet text-to-speech model. Its only attention transformer is a
weight-tied ALBERT encoder (`plbert`, 12 layers, hidden 768); the prosody predictor,
text encoder, and ISTFTNet vocoder are convolutional / recurrent.

This bring-up runs the **plbert encoder on Tenstorrent** (TTNN) and the remaining
prosody + vocoder stages on the host (torch), matching the SpeechT5 TT-transformer +
CPU-vocoder pattern. A fully-on-device pipeline (`tt/device_pipeline.py`) ports every
stage — including the ISTFTNet decoder and iSTFT — to TTNN and is validated per-stage
(reference PCC ≥ 0.999 except the Generator tail ≈ 0.987 audio PCC); it is not yet wired
into an end-to-end entrypoint and is exercised through its stage methods.

**Status: EXPERIMENTAL.**

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Model-specific host dependencies: `pip install -r models/demos/audio/kokoro/requirements.txt`
- System package for grapheme-to-phoneme: `apt-get install espeak-ng`

> Note: `misaki`/`kokoro` pull in `spacy` transitively, which wants a numpy-2 ABI that
> conflicts with the numpy 1.26 TTNN is built against. The G2P path uses
> `misaki.espeak.EspeakFallback` (libespeak-ng) to avoid loading spaCy. See the demo.

## How to Run

### Text-to-speech demo (single-chip p150)

Synthesize speech from text and write a WAV file:

```sh
pytest --disable-warnings models/demos/audio/kokoro/demo/demo.py::test_demo
```

### plbert encoder correctness (PCC vs HuggingFace)

```sh
pytest --disable-warnings models/demos/audio/kokoro/tests/test_optimized_decoder.py
```

### Functional / multichip encoder tests

```sh
pytest --disable-warnings models/demos/audio/kokoro/tests/test_functional_decoder.py
pytest --disable-warnings models/demos/audio/kokoro/tests/test_optimized_multichip_decoder.py
```

## Model precision

The selected weight/activation/fidelity policy is file-backed in
`doc/datatype_sweep/selected_precision_config.json` and loaded by
`tt/precision_config.py`. The default is BFP8 weights / bf16 activations / HiFi2, chosen
from a 10-config sweep as the fastest configuration that clears the accuracy gate.

## Performance

The model is launch/dispatch-bound (small ops, ~5% DRAM utilization), so throughput is
dominated by op-to-op dispatch rather than compute.

Measured on P150 (single Blackhole chip):

| Path | Metric | Value |
|---|---|---|
| plbert encoder | last_hidden_state PCC vs HF (worst, T≤512) | ≈ 0.997 |
| plbert encoder | eager prefill latency (T=128 / T=512) | ≈ 2.5 ms / 3.0 ms |
| **Fully on-device** TTS (`synthesize_device`) | latency (≈2.4 s clip) | ≈ 0.88 s |
| **Fully on-device** TTS (`synthesize_device`) | real-time factor | ≈ 2.7× |
| Fully on-device audio | STFT log-magnitude PCC vs torch | ≈ 0.98 |

Reproduce the perf numbers:

```sh
pytest -m models_performance_bare_metal models/demos/audio/kokoro/tests/test_perf_optimized.py         # encoder
pytest -m models_performance_bare_metal models/demos/audio/kokoro/tests/test_perf_device_pipeline.py   # full TTS
```

## Details

- `tt/optimized_decoder.py` — single-chip plbert encoder (packed QKV, FlashAttention SDPA, BFP8/HiFi2). Used by the p150 demo path.
- `tt/functional_decoder.py` — reference functional plbert encoder.
- `tt/multichip_decoder.py`, `tt/optimized_multichip_decoder.py` — TP=4 head-parallel encoder for p300x2.
- `tt/generator.py`, `tt/model.py` — full-model wrapper + `build_generator` entrypoint (multichip).
- `tt/device_pipeline.py` — full text→audio pipeline stages on-device (experimental).
- `tt/precision_config.py` — loads the selected precision policy.
- `doc/analysis/` — serving-integration and vLLM-integration analysis notes (provenance).
