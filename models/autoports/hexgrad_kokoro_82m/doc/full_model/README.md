# Kokoro-82M — Full Model (TTNN, TP=4)

Stage 05 (full-model) for `hexgrad/Kokoro-82M` on **4× Blackhole p300c**
(`ClusterType.P300_X2`, physical 4-ring exposed as a `(1, 4)` mesh). Assembles
the stage-04 **optimized multichip decoder** into a full-model wrapper
(`tt/model.py::KokoroModel`) + readiness/serving generator
(`tt/generator.py::KokoroGenerator`, `build_generator`).

## Headline performance (warmed, real weights, batch-1, (1,4) ring mesh)

| Metric | Value | Workload |
|---|---|---|
| **TTFT (trace-verified)** | **1.89 ms** | prompt 128, traced encode→readout→argmax |
| **Token-out decode t/s/u (trace-verified)** | **528 t/s/u (1.89 ms)** @T=128 · **348 t/s/u (2.88 ms)** @T=512 | greedy, on-device argmax, no host argmax/logits readback |
| **Teacher-forcing decode t/s/u (trace-verified)** | **314 t/s/u** | `run_teacher_forcing`, growing-prefix re-encode |
| TTFT (eager, non-production) | 11.19 ms | host-dispatch-bound (same framing as stage-04 eager prefill) |

Token-out and teacher-forcing decode differ because teacher forcing re-encodes a
**growing** prefix each step (variable length 2..~65) plus generator-loop
overhead, while the token-out figure is a fixed-`T` traced replay. Lower-bound
check vs the decoder stack: stage-04 decoder-only traced decode is 1.68 ms @T=128;
full-model token-out is 1.89 ms → terminal readout+argmax add ~0.21 ms (~11%),
of which `ArgMaxDeviceOperation` is ~6% — **not** the dominant cost (Matmul/SDPA/
CCL dominate, as in stage-04). See `tracy/tokenout_perf_report.txt`.

## Architecture reality (why this is not a causal-LM full model)

Kokoro-82M is a **non-autoregressive StyleTTS2/ISTFTNet TTS** model. The
checkpoint has five components (`bert`=plbert, `bert_encoder`, `predictor`,
`decoder`=ISTFTNet vocoder, `text_encoder`); its **only** attention-transformer
is `bert` = plbert (HF `AlbertModel`, bidirectional, 12 weight-tied layers,
vocab 178, ctx 512). There is **no causal decoder, no KV cache, no next-token
distribution, and no sampling** anywhere in the real pipeline. Stages 01-04
brought up plbert with four stage-review clean-passes and established the
autoregressive contract items (KV/paged cache, current-position advance, MoE,
token-by-token sampling) as **N/A**, replaced by a stateless *decode == prefill*
proof. This stage is the full-model assembly around that decoder; the full TTS
pipeline (predictor/vocoder/text_encoder → audio) is out of scope for the
start-from-decoder contract (and is itself non-autoregressive).

## What was built

- **`tt/model.py::KokoroModel`** — wraps `OptimizedMultichipDecoder` **verbatim**
  (import, no re-implementation): TP=4 head-parallel attention + sequence-parallel
  FFN + sequence-sharded residual `[b,1,S/TP,H]`, bf16 act / BFP8 linear weights /
  HiFi2 / fp32-dest-acc, block-sharded L1 LayerNorm, persistent CCL, 1 all_gather +
  1 reduce_scatter/layer, **inter-layer residual layout preserved** (no added
  layer-to-layer gather). Adds the terminal gather to a full `[b,S,H]`
  `last_hidden_state` (localized, one all_gather) and a **tied-embedding
  reconstruction readout** (LM-head analog) + on-device greedy argmax.
- **`tt/generator.py::KokoroGenerator` / `build_generator`** — implements the
  `models.common.readiness_check.contract.Generator` ABC. Low level:
  `prefill_forward` (full encode → readout logits) and `decode_forward` (stateless
  re-encode → last-position readout logits or on-device sampled token), both with
  explicit `page_table`/`kv_cache`/`prompt_lens`/batch state (accepted; cache is
  N/A). High level: `generate` (explicit `enable_trace` kw; teacher-forcing
  growing-prefix loop; free-running full-context reconstruction; explicit
  `host_sampling` compat mode). `reset` is a stateless no-op.

### Reconstruction readout (LM-head analog)

plbert ships no LM head. We add `logits = last_hidden_state @ (W_map @ W_word^T)`
using only shipped weights (the ALBERT factorised-embedding tie run in reverse),
folded host-side to one `[H, vocab_padded]` matmul run on the sequence shard;
pad columns masked to −inf so on-device argmax never selects them. This is a
**fidelity/reconstruction probe**, not a claim that Kokoro emits phoneme tokens —
the real output is `last_hidden_state` (PCC-validated). HF and TT apply the
identical readout, so top-1/5/100 over it measures TT-vs-HF **full-model**
numerical fidelity, and the free-running reconstruction is a genuine,
non-degenerate phoneme completion for the degenerate-output gate.

## Correctness / accuracy (real weights)

- **`last_hidden_state` PCC vs HF ≥ 0.995** at every tested length incl.
  non-aligned (worst **0.99700** @T=128; 8/31/33/64/127/200/256/511/512), batch-4
  0.998881. (`results.json → pcc_vs_hf`; `tests/test_full_model.py`.)
- **Prefill reconstruction (full visibility): top-1 = top-5 = top-100 = 1.000**
  (222 positions, `run_prefill_check` on `readiness_recon_prefill.refpt`).
- **Teacher-forcing (growing prefix): top-1 0.9865, top-5 0.9955, top-100 1.000**
  (222 positions, `run_teacher_forcing` on `readiness_recon_tf.refpt`). Bars:
  top-5 ≥ 98% ✓, top-100 = 100% ✓.
- **Free-running / autoregressive analog:** HF-vs-TT token agreement 1.000,
  adjacent-dup 0.000 (`readiness_autoregressive/autoregressive_meta.json`); the
  runner-side degenerate-output gate passes (adjacent_dup 0.0, trigram-loop 0.14).
- **Determinism:** repeated traced decode bit-identical; on-device argmax ==
  host argmax (`split_sampling.json`, `test_split_sampling_trace_feedback`).

Qualitative reading: TT and HF reconstruct the **identical** phoneme string for
the AIME24-substitute IPA sentence (agreement 1.000) — no repetition,
single-token collapse, wrong-language drift, or early divergence. (AIME24
chat-template reference is N/A: Kokoro has no chat template / causal LM; replaced
by real IPA phoneme sentences — see `make_references.py`.)

## Split-sampling / tracing contract

- **On device & traced:** encode → readout → argmax is one captured graph; greedy
  token produced by `ttnn.argmax` with **no host argmax, no full-vocab logits
  readback** (`counters: host_argmax=0, logits_readbacks=0` on the greedy path).
- **Trace feedback / refresh proof:** two decode steps with different contexts
  produce different tokens (trace inputs refreshed); repeated replay deterministic
  (`split_sampling.json`).
- **N/A (documented):** the AR token-feedback loop (`tt_out_tok`→persistent
  decode-token input, device position advance, unchanged-page-table skip) does
  not apply — the encoder input **grows** each step, so there is no fixed-shape
  persistent decode-token tensor or per-token position state. This is the honest
  model-specific equivalent, not a missing optimization.
- **Sampler choice:** on-device `ttnn.argmax` selected for greedy. Both common
  samplers (`models/common/sampling` TTTv1, `models/common/modules/sampling/
  sampling_1d.py` TTTv2) target a large **sharded** vocab with a cross-device
  all-gather; Kokoro's readout is a tiny (178) **replicated** vocab, so both add
  avoidable movement — recorded as rejected-for-greedy in
  `sampler_comparison.json` (TTTv2 retained for a top-k/top-p path if sampled
  TTS-token selection is ever needed). No custom sampler code written.

## Context / batch contract

Advertised = supported = **512**, **no reduction** (`../context_contract.json →
full_model`). Weight+KV DRAM recomputed for the full stack: ~4.69 MB/device
(decoder ~4.39 MB + readout `[768,192]` bf16 ~0.30 MB), KV cache = 0 (N/A). Public
API accepts any logical length 1..512 incl. non-aligned (validated 31/33/127/200/
511). Batch-1 primary; batch dims not hard-coded (batch-4 PCC 0.998881; decoder
stage tested batch up to 32).

## Watcher / fallback

- Watcher clean: 0 fault/assert/overflow markers (`watcher/watcher.log`, ETH
  excluded — infra limit).
- Fallback audit clean (`fallback_audit.txt`): no host fallback in the measured
  token-out path; host boundaries confined to setup, the prefill-check surface,
  and the explicit host-sampling compat mode.

## Reproduce

```bash
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"

# references (HF only, no device):
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/full_model/make_references.py
# full evidence (one mesh session): PCC + prefill/TF top-k + autoregressive_meta + split-sampling + perf:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/full_model/gen_full_model_evidence.py
# tests (19):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_full_model.py -v
# sampler comparison:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/full_model/sampler_comparison.py
# reduced-layer token-out perf report (tracy; separate from watcher):
env $ENV KOKORO_PERF_LAYERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf_full_model.py -k tokenout
# watcher (separate from profiler; ETH excluded):
env $ENV TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_full_model.py -k "pcc_vs_hf or split_sampling or decode_on_device or free_running"
```

## Artifacts

- `results.json`, `split_sampling.json`, `sampler_comparison.json`
- `readiness_recon_prefill.refpt`, `readiness_recon_tf.refpt`, `make_references.py`
- `../../readiness_autoregressive/{autoregressive_meta.json,hf_completion.txt,tt_completion.txt}`
- `tracy/{tokenout_perf_report.txt,tokenout_ops_perf.csv}`, `watcher/watcher.log`
- `fallback_audit.txt`, `logs/`, `gen_full_model_evidence.py`, `probe.py`, `work_log.md`
- `../context_contract.json → full_model`; tests `tests/test_full_model.py`, `tests/test_perf_full_model.py`

## Limitations / deviations

- The "full model" is the plbert encoder + reconstruction readout, not the full
  phonemes→audio TTS pipeline (out of scope; also non-autoregressive).
- Autoregressive/KV/sampling contract items are N/A (bidirectional stateless
  encoder); model-specific equivalents implemented + validated.
- The reconstruction readout is a fidelity probe (no shipped LM head); the real
  model output is `last_hidden_state` (PCC-gated).
- Eager prefill wall-clock is host-dispatch-bound (non-production); the traced
  path is the production path.
