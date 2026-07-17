# Kokoro-82M full-model — work log (stage 05)

Target: `hexgrad/Kokoro-82M`, branch `agentic-research/hous/kokoro-82m-p150`,
4× Blackhole p300c (`ClusterType.P300_X2`, `(1,4)` ring mesh, FABRIC_1D_RING).
Env: `TT_METAL_HOME=/home/ttuser/dev/tt-metal PYTHONPATH=.../ttnn:...` (dev checkout).

## Approach
Kokoro is a non-autoregressive TTS model; its only transformer is plbert
(`AlbertModel`), brought up in stages 01-04. This stage assembled the stage-04
optimized multichip decoder into `tt/model.py` + `tt/generator.py` (Generator
ABC + build_generator), adapting the causal-LM readiness contract to the
bidirectional encoder: AR/KV/paged/position/token-feedback = N/A (documented),
plus a tied-embedding reconstruction readout (LM-head analog, shipped weights
only) so `run_prefill_check`/`run_teacher_forcing`/`run_autoregressive` and
on-device greedy sampling are genuinely exercisable and measure TT-vs-HF
full-model fidelity.

## Decisions
- Validated the reconstruction readout on HF first (CPU): non-degenerate, top-1
  71-82% vs input — meaningful, not trivial-copy.
- Readout folded to one `[H, vocab_padded=192]` matmul run on the sequence shard;
  pad columns masked to −inf; on-device `ttnn.argmax` for greedy (proved ==host).
- Two references (visibility-matched): prefill = full bidirectional; TF =
  growing-prefix (matches the stateless growing-prefix decode loop). AIME24
  chat-template N/A (no chat template / causal LM) → real IPA phoneme sentences.
- Free-running = full-context reconstruction (a token-feedback loop collapses for
  a bidirectional encoder — observed "kkkk"; fixed by using the model-appropriate
  no-feedback reconstruction, which is also why decode-loop degeneracy cannot
  arise here).
- Decoder preserved verbatim (import OptimizedMultichipDecoder + OptConfig +
  PrecisionPolicy); terminal gather only; no inter-layer gather added.

## Results (real weights, warmed, batch-1)
- PCC last_hidden_state vs HF ≥ 0.995 all lengths (worst 0.99700 @128), batch-4 0.998881.
- Prefill top-1/5/100 = 1.000 (222 pos). TF top-1 0.9865 / top-5 0.9955 / top-100 1.000.
- Autoregressive: HF-TT agreement 1.000, adjacent-dup 0.000; degenerate gate PASS.
- Split-sampling: outputs differ across steps, deterministic replay, on-device
  argmax == host, host_argmax=0 / logits_readbacks=0 on greedy path.
- Perf: TTFT (traced) 1.89 ms @128; token-out 528 t/s/u @128 / 348 t/s/u @512;
  teacher-forcing 314 t/s/u. Lower-bound vs decoder-only 1.68 ms → terminal
  readout+argmax ~0.21 ms (~11%; ArgMax ~6%, not dominant).
- 19/19 `tests/test_full_model.py` pass. Watcher clean. Fallback audit clean.

## Gates
- `check_degenerate_output.py --scope autoregressive --missing-artifacts critical` → rc=0.
- `check_context_contract.py --stage full-model --require-contract` → rc=0.

## Commands (see README Reproduce). Key logs in `logs/`.

## Stage review
- Independent $stage-review (fresh subagent, read-only, no device): **clean-pass**,
  no Required Work. Verified: both runner gates rc=0, decoder imported verbatim
  (decoder files unmodified in git), N/A claims true for the architecture,
  reconstruction readout honest (not gate-gaming), BFP8/HiFi2 policy present in
  measured perf rows, watcher 0 markers, TT/HF reconstruct identical coherent
  non-degenerate phoneme string, AIME24/chat-template correctly N/A.
- P3 nits addressed post-review: (1) counter instrumentation — `_free_running`
  now increments `trace_captures` on a new capture; (2) `results.json`
  tt_text/hf_text resynced space-joined to match the completion files (token_ids
  authoritative, agreement 1.0); (3) reduced-layer Tracy re-run with
  KOKORO_PERF_ITERS=4 → 0 dropped profiler markers, complete per-op breakdown
  (Matmul 29.3% dominant; ArgMax fixed 111.7 µs/op = ~25% on the reduced 2-layer
  probe, ~6% projected on the full 12-layer model — not dominant).

## Commit
- Stage-owned changes committed locally on branch agentic-research/hous/kokoro-82m-p150
  (never pushed). SHA recorded in the follow-up commit and in memory.
