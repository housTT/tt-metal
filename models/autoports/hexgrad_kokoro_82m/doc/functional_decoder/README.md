# Kokoro-82M — Functional Decoder (TTNN)

Stage 01 (functional-decoder) for `hexgrad/Kokoro-82M` on Tenstorrent Blackhole
(p300c, single 1x1 device). Implements
`models/autoports/hexgrad_kokoro_82m/tt/functional_decoder.py`.

## What the "decoder" is for this model

Kokoro-82M is a **non-autoregressive** StyleTTS2 / ISTFTNet **text-to-speech**
model. `KModel.forward_with_tokens` (kokoro `model.py`) runs a single stateless
pass:

```
bert (plbert) -> bert_encoder(Linear) -> ProsodyPredictor(LSTMs)
             -> TextEncoder(CNN+LSTM) -> Decoder(ISTFTNet vocoder) -> audio
```

There is **no causal transformer decoder, no KV cache, and no token-by-token
decode** anywhere in the model. The *only* attention/transformer component — the
one this stage's skill (HF transformer decoder layers: attention, MLP, norms,
head reshapes) targets — is **`plbert`**, a HuggingFace `transformers.AlbertModel`
(`CustomAlbert` in `kokoro/modules.py`). This stage brings that encoder up in
TTNN, end to end, with real weights.

ALBERT ties parameters across all `num_hidden_layers` layers, so the encoder has
exactly **one transformer layer kind** (`AlbertLayer`), applied 12×.
`FunctionalDecoder` represents the full parameter-shared encoder
(embeddings + `embedding_hidden_mapping_in` + 12× shared `AlbertLayer`) so it can
be validated against HF `last_hidden_state`.

### Architecture (from real checkpoint + config)

| Field | Value |
|---|---|
| plbert type | `AlbertModel` (bidirectional encoder) |
| hidden_size | 768 |
| num_attention_heads / head_dim | 12 / 64 |
| intermediate_size | 2048 |
| num_hidden_layers | 12 (1 shared `albert_layer_group`, 1 inner layer → weight-tied) |
| embedding_size (factorized) | 128 → projected to 768 |
| activation | `gelu_new` (matched by ttnn erf gelu, see below) |
| layer_norm_eps | 1e-12 |
| vocab (n_token) | 178 |
| max_position_embeddings | **512** (advertised context; hard positional limit) |

## Prefill / decode contract

The transformer is bidirectional and stateless, so prefill and decode differ
only in *execution strategy*, not autoregressive semantics (there is none):

- `prefill_forward(input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None)`
  — eager full-sequence bidirectional encode. Accepts **any logical
  `seq_len` in 1..512**; `prepare_inputs` owns tile padding (to a multiple of 32)
  and masking, so non-tile-aligned lengths are valid public inputs.
- `decode_forward(input_ids, position_ids, token_type_ids, attention_mask=None, *, batch=None, seq_len=None)`
  — the **same** computation captured into a TTNN trace and replayed for a fixed
  `(batch, padded_seq_len)` shape (lazy capture on first use per shape). Satisfies
  the traced-execution requirement and provides the fast repeated-inference path a
  TTS server uses for fixed-length phoneme windows.
- `from_state_dict(state_dict, *, hf_config, layer_idx=0, mesh_device, **kwargs)`
  — all host→device weight conversion (bf16 tables/weights). Accepts the raw
  Kokoro `bert` sub-dict (with or without `module.` prefix).
- `prepare_inputs(input_ids, mesh_device, *, attention_mask=None)` — host-side
  input construction (the allowed torch boundary): pads ids/positions/type-ids to
  a tile multiple and builds the additive fp32 attention mask.

Output is the tile-padded hidden state `(batch, padded_seq_len, 768)`; callers
slice to the logical `seq_len` at the torch boundary.

### KV-cache / paged-cache / current-position → N/A (architectural)

Paged prefill, paged decode, page tables and current positions are KV-cache
concepts. This model has no KV cache (bidirectional, single-pass). The
model-appropriate replacement is `test_decode_is_stateless`, which proves
`decode == prefill` bit-for-bit for the same input (no hidden state). See the
capability table below and `../context_contract.json`.

## Precision policy

- **Weights: bf16** (ttnn.embedding requires bf16 tables; bf16 linear weights are standard).
- **Activations: fp32.** With bf16 activations, HF-vs-TTNN PCC occasionally dips
  just below 0.995 on atypical short inputs; fp32 activations keep the worst case
  ≥ 0.9958 over a 16-seed representative sweep (real IPA sentences ≈ 0.9994).
  Fidelity is raised here per the functional-decoder skill ("raise fidelity where
  useful"); the optimize stage may lower it.
- **gelu:** HF uses `gelu_new` (tanh approx). ttnn's *accurate erf* gelu
  (`fast_and_approximate_mode=False`) tracks `gelu_new` far better than ttnn's
  fast tanh mode (worst-case PCC 0.9986 vs 0.9942 over T=32/128/512 × 4 seeds).
- Compute kernel config: `HiFi4`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`.

## Capability-contract evidence

| Claim | Evidence | Residual risk |
|---|---|---|
| Advertised context = 512 | `AlbertConfig.max_position_embeddings`; `KModel` asserts `len(ids)+2<=512` | none |
| Prefill supports full 512 (+ non-aligned) | `test_prefill_pcc_real_weights[8,16,31,32,33,64,128,256,500,511,512]` all ≥0.995 | none |
| Decode (traced) supports full 512 + non-aligned | `test_decode_traced_pcc_real_weights[32,64,128,500,511,512]` all ≥0.995 | none |
| Traced decode with batch>1 + nonzero mask | `test_decode_traced_masked_batch` (rows 113,47; trace-baked mask + copy-on-replay) ≥0.995 | none |
| Single layer kind (`AlbertLayer`) | `test_single_layer_kind` (num_hidden_groups=1, inner_group_num=1) | none |
| Batch (1..32) | `test_prefill_batch_pcc[2,4,8,32]` all ≥0.999 | batch >32 untested (not needed for TTS) |
| Padding/masking correct | `test_padding_mask` (variable-length batch vs HF masked) | none |
| No KV cache / stateless | `test_decode_is_stateless` (decode == prefill, bit-identical) | N/A by architecture |
| Real weights pass | every `*_real_weights` test + real IPA sentences (0.9996+) | none |

## Results (real weights; see `pcc_results.json`, `perf_summary.json`)

- **PCC ≥ 0.995 bar met.** Prefill T=8..512 representative: 0.998–0.9997. Traced
  decode T∈{32,64,128,500,511,512} (incl. non-aligned): ≥0.999. Traced decode
  batch>1 + nonzero mask (`masked_decode_traced`): ≥0.999. Real IPA sentences:
  0.9996+. Batch 2/4/8/32: 0.9993+. Worst-case over 16 representative
  seeds/length: **0 below bar** (min 0.9958 @ T=16).
- **Determinism:** prefill and traced decode bit-identical across repeats;
  `decode == prefill`.
- **Perf (warmed, T=512, bf16 w / fp32 act, unoptimized functional path):**
  eager prefill **≈21.0 ms/pass**, traced decode **≈21.0 ms/pass** (wall-clock);
  tt-perf-report device-kernel time ≈21.0 ms (prefill) / ≈20.8 ms (decode). The
  encoder is compute-bound at fp32, so trace (which removes host dispatch) and
  eager are close; dtype/layout optimization is the next stage's job.
  T=128: ≈7.1 ms/pass.
- **Short-length characterization** (`pcc_results.json.short_len_characterization`):
  T=1 ≈0.91 (single token, no cross-position averaging, bf16-weight-bound — and
  below the model's minimum valid input of 3 tokens); T≥3 ≥0.998. Not a bug; out
  of the model's operating domain.

## Reproduce

```bash
# All commands use the dev-checkout ttnn build (headers+kernels+lib consistent):
ENV="TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal"

# Correctness suite (29 tests):
env $ENV python -m pytest models/autoports/hexgrad_kokoro_82m/tests/test_functional_decoder.py -v

# PCC / determinism evidence table:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/functional_decoder/gen_evidence.py

# Warmed wall-clock latency:
env $ENV python models/autoports/hexgrad_kokoro_82m/doc/functional_decoder/gen_perf_latency.py

# Profiling (one measured window per run; KOKORO_PERF_ITERS=2 keeps the profiler
# DRAM marker buffer from overflowing):
env $ENV KOKORO_PERF_ITERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf.py -k prefill
env $ENV KOKORO_PERF_ITERS=2 python -m tracy -r -p -v -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_perf.py -k decode
# then tt-perf-report with --start-signpost PERF_PREFILL/PERF_DECODE (see work_log.md).

# Watcher-clean run (separate from profiler):
env $ENV TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=<dir> python -m pytest \
  models/autoports/hexgrad_kokoro_82m/tests/test_functional_decoder.py -k "512 or determinism or padding"
```

## Artifacts

- `pcc_results.json`, `perf_summary.json`, `weight_stats.json`
- `tracy/albert_layer/{prefill,decode}_perf_report.txt` (human tables),
  `*_perf_report.csv` (filtered), `*_perf_report.console.log`. The raw
  `*_ops.csv` Tracy dumps (~1.5–2.5 MB each) are kept on disk as provenance but
  are excluded from git by the repo's 500 KB large-file pre-commit gate; the
  filtered `*_perf_report.csv` are the committed CSV evidence.
- `logs/` (pytest, pcc, perf-latency console logs)
- `watcher/generated/watcher/watcher.log` (clean)
- `fallback_audit.txt` (clean — no torch/from_torch/to_torch in a measured pass)

## Limitations / deviations from the LLM decoder template

- KV cache / paged cache / current-position / autoregressive decode: **N/A** —
  the model is non-autoregressive and stateless. Documented and proven via
  `test_decode_is_stateless`, not fabricated.
- Perf is unoptimized (fp32 activations, DRAM, separate QKV matmuls, no sharding):
  correctness-first per the functional stage; optimization deferred to stage 02.
- Downstream Kokoro components (prosody predictor LSTMs, text encoder, ISTFTNet
  vocoder) are out of scope for the functional-decoder stage.
