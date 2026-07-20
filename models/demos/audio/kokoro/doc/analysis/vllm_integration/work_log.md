# vLLM Integration Stage — Work Log (Kokoro-82M)

Verdict: **BLOCKED** — fundamental architecture incompatibility between the
non-autoregressive plbert encoder and the TT vLLM backend (causal-LM + paged-KV
only). See `README.md` for the full determination. This log records exactly what
was investigated and the evidence, so the determination can be audited.

## Skills read
- `$vllm-integration`, `$tt-device-usage`, `$stage-review`, `$autofix`
  (`.agents/skills/*/SKILL.md`).

## Ground-truth reads
- `tt/generator.py`, `tt/model.py`, `tt/precision_config.py` — full-model
  generator/adapter surface (`prefill_forward`/`decode_forward` accept-and-ignore
  `page_table`/`kv_cache`/`start_pos`; stateless `decode==prefill`; on-device
  argmax token-out; explicit N/A documentation for AR contract items).
- `doc/context_contract.json` — advertised = supported context 512, no KV cache,
  `capability_reduction=false` for every prior stage.
- `models/tt_transformers/tt/generator_vllm.py` and `.../generator.py` — the
  reference adapter and `Generator` base (all `...ForCausalLM` with paged KV).
- `tech_reports/LLMs/vLLM_integration.md` — model-integration contract
  (paged attention required; `initialize_vllm_model`/`allocate_kv_cache`/
  `prefill_forward`/`decode_forward` interface).
- Installed TT vLLM plugin
  `/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/`
  (`platform.py::register_tt_models`, `loader.py`, `worker.py`,
  `model_runner.py`, `async_decode.py`) — deep contract read via a read-only
  subagent.

## Environment facts established
- vLLM + `vllm_tt_plugin` are editable-installed in `.tenstorrent-venv`
  (`pip show vllm` → `.local/lib/model-bringup/tt-metal/vllm/vllm`); the plugin
  git tree is a nested repo at `.local/lib/model-bringup/tt-metal/vllm`. There is
  **no** `vllm/` directory in the dev checkout `/home/ttuser/dev/tt-metal`.
- `models.autoports.hexgrad_kokoro_82m` resolves to the dev checkout; ttnn
  resolves to the installed tree (editable finder), the known dev/installed split
  from `[[tt-metal-dev-env]]`.

## Experiments / evidence (no TT device opened, no server launched)
1. HF config parse (what vLLM `ModelConfig` does at startup):
   `AutoConfig.from_pretrained("hexgrad/Kokoro-82M")` →
   `ValueError: Unrecognized model ... Should have a model_type key in its
   config.json`. Config top-level keys are Kokoro TTS fields (`plbert`,
   `istftnet`, `n_token`, `vocab`, ...), no `model_type`, no `architectures`.
2. HF tokenizer load (what vLLM's server does at startup):
   `AutoTokenizer.from_pretrained("hexgrad/Kokoro-82M")` → `ValueError: Couldn't
   instantiate the backend tokenizer ...`. The HF repo ships no tokenizer files.
3. Plugin decode contract (source): `model_runner.py:932-936` builds the decode
   `tokens` tensor as the single last token `.view(-1,1)` → `[batch,1]`. Full
   context is never passed; `kv_cache` holds projected K/V, not token ids.
4. Mandatory paged KV cache (source): `worker.py` always returns a nonzero KV
   spec/block count; `model_runner.py` `initialize_kv_cache` raises on empty
   groups; `num_kv_heads=0` → divide-by-zero. No zero-KV path.
5. Encoder/pooling path (source): TT worker sets `is_pooling_model=False`,
   `get_supported_pooling_tasks() -> []`. No non-causal / zero-KV TT model exists.

## Blockers (each independently fatal)
- **B1**: vLLM cannot construct `ModelConfig` / load a tokenizer for Kokoro
  (no `model_type`/`architectures`, no tokenizer). Server fails before the
  adapter runs.
- **B2**: decode passes only `[batch,1]`; a stateless bidirectional re-encoder
  needs full context → forbidden hidden per-sequence cache; slots are anonymous
  and remapped across steps.
- **B3**: nonzero paged KV cache is mandatory; the encoder has none and cannot
  declare zero.
- **B4**: TT backend disables the pooling/encoder runner; no non-causal precedent.

## Why not `$autofix` / why not fabricate artifacts
No bug hypothesis exists — the incompatibility is proven from framework source
and the first failure is a deterministic server-startup `ValueError` whose only
"fix" is inventing a fake causal-LM identity + tokenizer for a TTS encoder.
Doing so, plus abusing prefill/decode with a hidden per-sequence cache, would
violate this stage's explicit "no hidden cache / vLLM owns the cache / serves
through the shared vLLM path with evidence" contract and misrepresent the model.
Per the `$autofix` stop criteria, this is "a legitimate limitation that needs
human/product direction."

## Not done (blocked before it was possible)
- `tt/generator_vllm.py` — not written (no working adapter can exist; writing a
  non-functional stub would be misleading).
- `platform.py::register_tt_models` registration — not added (registration
  cannot help; the server fails at config/tokenizer resolution first).
- `run_vllm_server`, sampling tests, qualitative outputs, benchmarks,
  `$stage-review` — not run (server cannot start).
- No checkpoint commit created (the goal gates commits on `$stage-review`
  clean-pass, which is not reached). Evidence docs left under
  `doc/vllm_integration/` uncommitted.

## Device hygiene
No server/`EngineCore`/benchmark process launched; none left holding devices.
`tt-smi -ls --local`: all 4 Blackhole p300c chips visible and healthy.
