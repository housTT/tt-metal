# Kokoro-82M — vLLM Integration Stage: BLOCKED (fundamental architecture incompatibility)

**Status: BLOCKED.** The shared Tenstorrent vLLM serving path cannot serve
hexgrad/Kokoro-82M's plbert component. This is not a bug that adapter code,
plugin registration, or `$autofix` can resolve — it is a proven, multi-layered
incompatibility between the model class and the TT vLLM backend's core
assumptions. Blocking here needs human/product direction (extend the TT vLLM
backend, or serve Kokoro via a non-vLLM TTS path); it is not a stoppable
model-bringup defect.

No `run_vllm_server` run, benchmark, or served qualitative output exists because
the server cannot start for this model (see Blocker 1). Fabricating a false
causal-LM identity + tokenizer and abusing prefill/decode to produce output
would violate this stage's explicit contract ("no hidden standalone-cache",
"vLLM owns the cache, adapter passes it through", "serves through the shared
vLLM path with evidence [of the actual model]") and would misrepresent a TTS
encoder as a text generator, so it was not done.

## What Kokoro-82M is (carried from stages 01–07, four stage-review clean-passes)

Kokoro-82M is a **non-autoregressive StyleTTS2/ISTFTNet text-to-speech** model.
Its only attention-transformer is `bert` = **plbert**, an HF `AlbertModel`:
**bidirectional, 12 weight-tied layers, hidden 768, 12 heads, vocab 178, ctx
512.** It has **no causal decoder, no KV cache, no next-token distribution, and
no sampling** anywhere in the real pipeline. Stages 01–07 brought up exactly
this encoder and documented the autoregressive contract items (KV cache, paged
cache, current-position advance, token-by-token sampling, token-feedback loop)
as **N/A**, replaced by a *stateless `decode == prefill` re-encode* proof. See
`../context_contract.json` and `../full_model/`.

The TT vLLM backend, by contrast, serves **only autoregressive causal LMs with a
paged KV cache.** These two facts are irreconcilable through an adapter.

## Blockers (each independently fatal; all evidence-backed)

### Blocker 1 — vLLM cannot construct a `ModelConfig` for this model (empirical, server-startup)
`hexgrad/Kokoro-82M`'s `config.json` has **no `model_type` and no
`architectures`** (top-level keys: `istftnet, dim_in, dropout, hidden_dim,
max_conv_dim, max_dur, multispeaker, n_layer, n_mels, n_token, style_dim,
text_encoder_kernel_size, plbert, vocab`). It is a custom Kokoro TTS config, not
a transformers config. Reproduced:

```
AutoConfig.from_pretrained("hexgrad/Kokoro-82M")
  -> ValueError: Unrecognized model in hexgrad/Kokoro-82M.
     Should have a `model_type` key in its config.json ...
```

The HF repo also ships **no tokenizer** (no `tokenizer.json` /
`tokenizer_config.json` / vocab merges — the repo is `config.json`,
`kokoro-v1_0.pth`, voice packs, and docs only). Reproduced:

```
AutoTokenizer.from_pretrained("hexgrad/Kokoro-82M")
  -> ValueError: Couldn't instantiate the backend tokenizer ...
```

vLLM's OpenAI server tokenizes prompts with the served model's HF tokenizer and
resolves the architecture from the HF config **before** the TT plugin/adapter
runs. Both fail. There is no adapter hook that repairs this; it would require
fabricating a false causal-LM `config.json` + a synthetic HF tokenizer for a
phoneme vocab — i.e. inventing a fake model identity. Even then, English prompts
→ IPA-phoneme ids is semantically meaningless (Kokoro's vocab is 178 IPA
phonemes, not text tokens).

### Blocker 2 — vLLM's decode contract cannot feed a stateless bidirectional re-encoder (source-proven)
The TT plugin's `decode_forward` receives only the **single newest token
`[batch,1]`** plus `start_pos [batch]`, `page_table`, and `kv_cache` — never the
growing full context. Evidence (installed plugin
`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:932-936`):

```python
positions_np = input_batch.num_tokens[req_indices] - 1
input_tokens = input_batch.token_ids_cpu_tensor[req_indices, positions_np].view(-1, 1)  # [batch,1]
```

A bidirectional encoder that has no KV cache must **re-encode the whole
sequence** each step, so it needs the full `[batch, cur_len]` context. vLLM
never delivers it, and `kv_cache` holds projected K/V (not raw token ids), so the
context is unrecoverable from what the plugin passes. Reconstructing per-sequence
history inside the adapter is (a) explicitly forbidden by this stage's contract
("no ... Python readback/writeback token-feedback loop", "no hidden
standalone-cache assumptions") and (b) impractical: `decode_forward` gets no
request ids and vLLM remaps/condenses batch slots across steps
(`model_runner.py` `slot_remap`, `Generator.decode_forward` merge logic), so
anonymous reordered `[batch,1]` slices cannot be stitched into stable histories.

Also, the model is **non-autoregressive by nature**: feeding its own predictions
back collapses (documented in `tt/generator.py` — a token-feedback loop yields
degenerate "kkkk" for a bidirectional reconstruction model). vLLM's decode loop
*is* autoregressive token feedback; it cannot be mapped away, because it is the
serving mechanism itself.

### Blocker 3 — a nonzero paged KV cache is mandatory (source-proven)
The TT worker/model_runner hard-require a nonzero paged KV cache and page tables:
`worker.py` `get_kv_cache_spec` always returns a spec;
`determine_available_memory`/`get_num_available_blocks_tt` always override to a
nonzero block count; `initialize_kv_cache` raises `ValueError("kv_cache_config
has no groups")` on an empty spec; `num_kv_heads=0` hits a divide-by-zero in
`_kv_cache_shape`. The plbert encoder has **no KV cache** and cannot declare one
as zero. There is no zero-cache / stateless runner path.

### Blocker 4 — the TT backend disables the one encoder-shaped path (source-proven)
Upstream vLLM can serve encoders via pooling/embedding runners, but the TT worker
sets `is_pooling_model=False` and `get_supported_pooling_tasks() -> []` ("TT
backend does not support pooling/embedding tasks yet"). Every registered TT model
is a `...ForCausalLM`/`...ForConditionalGeneration` with `use_paged_kv_cache=True`.
There is **no** non-causal, encoder-only, or zero-KV TT model served anywhere in
the tree.

## Why `$autofix` was not run as a repair loop
`$autofix` verifies/refutes hypotheses about **bugs** and keeps only fixes proven
against a failing command. There is no bug hypothesis here: the incompatibility
is proven from framework source across four independent dimensions, and the
first failure (Blocker 1) is a deterministic server-startup `ValueError` whose
only "fix" is to invent a fake model identity. Per the `$autofix` stop criteria,
"the report plus experiments show a legitimate limitation that needs
human/product direction" — which is this case.

## What would unblock this (product direction)
1. **Extend the TT vLLM backend** to support a pooling/encoder/non-causal runner
   (enable `is_pooling_model`, a zero/stateless-KV spec, and a full-context
   forward), then represent Kokoro's plbert as an embedding/encoder model. This
   is a vLLM-plugin + tt-metal core change, out of scope for an adapter stage.
2. **Serve Kokoro via a non-vLLM TTS path** (the real product surface: text →
   G2P phonemes → plbert → prosody predictor → ISTFTNet vocoder → audio), which
   is what Kokoro actually is. vLLM's OpenAI text-completion API is not the right
   serving surface for a TTS model.

## Serving-path performance (lower bound, for context only)
No vLLM serving numbers exist (server cannot start). The full-model /
datatype-sweep stages recorded the decoder lower bound on the (1,4) Blackhole
p300c ring mesh with the selected BFP8/HiFi2 policy: traced token-out
**560 t/s/u @ T=128 / 395 t/s/u @ T=512**, eager TTFT ≈ 10.7 ms, teacher-forcing
≈ 347.8 t/s/u (`../datatype_sweep/post_selection_tokenout.json`,
`../context_contract.json` `datatype_sweep`). These are non-vLLM
teacher-forcing/token-out numbers and are not a serving result.

## Device / process hygiene
No vLLM server, `EngineCore`, or benchmark process was launched, so none was left
holding devices. Post-stage `tt-smi -ls --local`: all 4 Blackhole p300c chips
visible and healthy. No reset needed.

## Selected precision config (would be used if serving were possible)
`../datatype_sweep/selected_precision_config.json`, loaded by default via
`tt/precision_config.load_selected` → `tt/generator.build_generator`: attn/mlp/map
weights BFP8, activations bf16, matmul+SDPA HiFi2, fp32 dest acc, LayerNorm HiFi4,
CCL (all_gather + reduce_scatter) bf16, readout/logits bf16, on-device greedy
argmax. KV-cache dtype axis is N/A (no KV cache).
