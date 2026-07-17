# Work Log — Kokoro-82M Functional Decoder (Stage 01)

Date: 2026-07-17
Branch: `agentic-research/hous/kokoro-82m-p150`
Hardware: Blackhole p300c, single device (device_id=0). Bringup on 1x1.
Runtime env (consistent dev-checkout ttnn build; avoids JIT header/kernel mismatch):
`TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=/home/ttuser/dev/tt-metal/ttnn:/home/ttuser/dev/tt-metal`

## Summary

Brought up the Kokoro-82M `plbert` (HF `AlbertModel`) bidirectional encoder in
TTNN as the functional decoder. Kokoro is a non-autoregressive TTS model with no
causal decoder / KV cache; plbert is its only attention-transformer component and
the correct target for this stage. See README.md for the full contract.

## Environment issue resolved (infrastructure)

- Symptom: `import ttnn` + `open_device` → `brisc build failed ... 'init_telemetry'
  was not declared in this scope` (cq_dispatch.cpp).
- Cause: bare `import ttnn` resolved to the older installed tree
  `/home/ttuser/.local/lib/model-bringup/tt-metal` (headers/lib) while JIT read
  the newer dev-checkout dispatch kernel *source* → mismatch.
- Fix: run against the dev checkout's own built ttnn by putting its `ttnn` parent
  dir on PYTHONPATH and setting `TT_METAL_HOME` to the dev checkout (env above).
  Mesh smoke `open_device`+gelu OK.

## Steps & key commands

1. Architecture study: read kokoro 0.9.4 `model.py`/`modules.py`/`istftnet.py`;
   confirmed non-autoregressive, plbert = `AlbertModel`, 12 weight-tied layers.
2. HF reference + real weights: `AlbertModel(AlbertConfig(vocab_size=178,**plbert))`,
   load `kokoro-v1_0.pth['bert']` (strip `module.`), strict=False → 0 missing / 0
   unexpected. Extracted per-tensor stats → `weight_stats.json`.
3. Implemented `tt/functional_decoder.py` (FunctionalDecoder, LightweightModule).
4. Precision decisions (device experiments):
   - gelu: ttnn erf gelu (`fast_and_approximate_mode=False`) beats fast tanh
     (worst PCC 0.9986 vs 0.9942 over T=32/128/512×4).
   - activations fp32 (weights bf16): worst-case ≥0.9958 over 16 representative
     seeds/length vs bf16-act occasional <0.995 dips. embeddings must stay bf16.
5. Tests: `tests/test_functional_decoder.py` (29 tests) — layer kind, prefill PCC
   over T∈{8,16,31,32,33,64,128,256,500,511,512}, real IPA sentences, batch
   {2,4,8,32}, traced decode {32,64,128,500,511,512}, traced masked batch,
   determinism (prefill+decode), decode==prefill (stateless), padding mask,
   synthetic-weights (CI-portable). Result: `29 passed`. Log: `logs/pytest_full.log`.
6. Evidence table: `gen_evidence.py` → `pcc_results.json` + `logs/pcc_evidence.log`.
7. Perf:
   - `gen_perf_latency.py` → `perf_summary.json` (warmed wall-clock).
     T=512 prefill 21.01 ms, traced decode 20.99 ms; T=128 7.10 / 7.06 ms.
   - Tracy (KOKORO_PERF_ITERS=2, one window/run):
     `python -m tracy -r -p -v -m pytest tests/test_perf.py -k prefill|decode`
     then copied newest `ops_perf_results_*.csv` to `tracy/albert_layer/{prefill,decode}_ops.csv`.
   - tt-perf-report 1.2.8:
     ```
     tt-perf-report <ops.csv> --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
       --csv prefill_perf_report.csv --no-advice > prefill_perf_report.console.log
     tt-perf-report <ops.csv> --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
       --no-summary --no-advice > prefill_perf_report.txt
     ```
     (decode: PERF_DECODE signposts, `--tracing-mode`). Device kernel time
     ≈21.0 ms prefill / ≈20.8 ms decode per pass. Human tables in `*_perf_report.txt`.
   - Note: with the default 20 iters the tracy host+device merge asserted / the
     profiler DRAM marker buffer overflowed; 2 iters produces a clean signposted
     window. Recorded here per skill guidance on tooling limitations.
8. Watcher: `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=.../watcher pytest -k "512 or
   determinism or padding or stateless or real_ipa"` → 8 passed. `watcher.log`
   clean: only normal attach/dump/stack-usage(free)/detach; 0 fatal/fault/
   sanitize/tripped markers.
9. Fallback audit: `fallback_audit.txt` — `to_torch` count 0; `from_torch`/`torch.`
   confined to `from_state_dict` (weights) and `prepare_inputs` (inputs); forward
   path pure ttnn; measured decode window is `execute_trace`+`synchronize` only.
10. Context contract: `../context_contract.json` — advertised=supported=512, no
    reduction. `.agents/scripts/check_context_contract.py --model-dir
    models/autoports/hexgrad_kokoro_82m --hf-model hexgrad/Kokoro-82M
    --require-contract` → exit 0.
11. Stage-review (fresh independent subagent, xhigh): first pass more-work-needed —
    traced decode only tested at tile-aligned lengths / all-valid masks. Fixed by
    adding non-aligned traced-decode lengths (500, 511) and a batch>1 masked
    traced-decode test (`test_decode_traced_masked_batch`); re-ran suite and
    re-reviewed → clean-pass.

## Re-verification / finalization (2026-07-17, resumed session)

- Device was found in a `Read 0xffffffff over PCIe ID 0` fault at session start
  (recoverable per $tt-device-usage). Recovery: no stage-owned processes owned the
  device; `tt-smi -r` (exit 0, boards [0,1,2,3] re-init) → `tt-smi -ls --local`
  (all 4 p300c chips visible) → mesh smoke `open_device(0)/close_device` =
  `MESH_SMOKE_OK`. Classified as infrastructure recovery, not a model result.
- Full suite re-run on device 0 (dev-checkout ttnn): **29 passed** in ~12.6s
  (`logs/pytest_full.log` refreshed). PCC bar met at every tested length.
- Fresh watcher-clean run (`TT_METAL_WATCHER=10`, `-k "512 or determinism or
  padding or stateless or real_ipa"`): 7 passed; `watcher/generated/watcher/
  watcher.log` regenerated — 0 fatal/fault/sanitize/assert/tripped/overflow
  markers, clean attach/dump/detach.
- Fallback audit re-verified against current code (line numbers match).
- Doc drift fixed: test count 26 → 29 in README.md and this log.
- Independent $stage-review (fresh Opus subagent, this session): **clean-pass**,
  no P1/P2 required work. Confirmed the KV-cache/paged/current-position "N/A"
  claim from kokoro source (`KModel.forward_with_tokens` single stateless pass;
  `CustomAlbert` is the only attention component). Non-blocking concerns
  (bf16-activation re-measurement, decode mask-None-at-capture dead branch,
  per-call input dealloc) are explicitly deferred to stage 02 / out of scope for
  the functional stage.

## PCC (real weights, bf16 w / fp32 act) — highlights

- Prefill: T=8:0.9995 T=16:0.9996 T=31:0.9997 T=32:0.9995 T=33:0.9995 T=64:0.9981
  T=128:0.9994 T=256:0.9993 T=500:0.9993 T=511:0.9995 T=512:0.9990.
- Traced decode: T=32:0.9997 T=64:0.9996 T=128:0.9996 T=512:0.9994.
- Real IPA sentences: 0.99977 / 0.99961 / 0.99965.
- Worst-case 16-seed sweep: min 0.9958 (T=16), 0 below 0.995 at every tested length.
- Determinism: prefill/decode bit-identical; decode==prefill True.

## Limitations

- KV cache / paged cache / current position / autoregressive decode: N/A
  (non-autoregressive stateless model). Proven via decode==prefill.
- T=1 (below model min input of 3 tokens) is bf16-weight-bound (~0.91); out of domain.
- Perf unoptimized (fp32 act, DRAM, separate QKV, no sharding) — stage 02 scope.
- Only the plbert encoder is in scope; other Kokoro modules are later stages.

## Commit
- See end of stage: local checkpoint commit of stage-owned files under
  `models/autoports/hexgrad_kokoro_82m/`. SHA recorded below after stage-review clean-pass.
