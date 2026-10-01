# Work log: stages 4 to 8 (multi-chip, full model, optimized full model, datatype sweep)

All times UTC, 2026 Oct 1. Device commands through `/home/hous/dev/clm-v0.1-8B/bin/devlock`; logs under
`/home/hous/dev/clm-v0.1-8B/logs/`.

- 21:46: probe run 3 (`tests/probe_full_encoder.py`, accuracy, 1x1): full 36-layer encoder works; README examples
  reproduced against HF bf16; 57.8 ms single text (`doc/probe/`).
- 21:51 to 21:55: `tests/run_fidelity.py --precision accuracy` and `--precision performance` on the 308-text
  reference corpus (`fidelity_accuracy.log`, `fidelity_performance.log`). Accuracy passes the cosine gate;
  performance (bfp4 MLP) fails (min cosine 0.924).
- 21:55 to 22:00: `tests/bench_encoder.py` accuracy and performance (`bench_accuracy.log`, `bench_performance.log`).
- 22:00: first `TT_METAL_TRACE_ALLOC_TRACKING=1` replay check fails: 43 program-cache buffers from the eager
  post-trace slice / norm / to_layout (`replay_trace_check.log`).
- 22:05: encoder rewritten to replay traces directly with host-side last-token pooling and fp32 RMSNorm
  (`tt/encoder.py`); fidelity re-run (`fidelity_accuracy_hostnorm.log`, mean 0.99910 / min 0.99596).
- 22:02 to 22:10: 1x2 mesh fails (fabric router sync timeout, `fidelity_accuracy_1x2.log`); 1x4 first attempt fails
  in host readback (width-sharded residual), fixed with `ConcatMeshToTensor`; 1x4 fidelity and bench pass
  (`fidelity_accuracy_1x4_v2.log`, `bench_accuracy_1x4_v2.log`).
- 22:04 to 22:12: bfp8_attn and bfp8_attn_hifi2 fidelity and benches (`fidelity_bfp8_attn*.log`, `bench_bfp8_attn*.log`).
- 22:12: replay check still flags 176 buffers (per-variant prepare+capture); two-phase warmup implemented.
  22:19: 8 buffers left (trace outputs); acknowledged after capture. 22:24 and 22:30: clean pass, nine variants
  (`replay_trace_check_v5.log`, `replay_trace_check_v6.log`, `doc/optimized_full_model/replay_trace_check_bfp8_attn.json`).
- 22:13 to 22:20: consistent host-norm-path benches for accuracy, bfp8_attn, bfp8_attn_hifi2
  (`bench_accuracy_hostnorm.log`, `bench_bfp8_attn_hostnorm.log`, `bench_bfp8_attn_hifi2_hostnorm.log`).
- 22:13: `tests/datatype_sweep.py` selected bfp8_attn (fastest passing within a 1 percent tie band).
- 22:53: independent review C (`../review/review_C_stages_4_8.md`): more-work-needed. P1: the plan's stage 6 gate
  "decision agreement >= 98 percent on Typed Decisions" was deferred and not applied in the sweep; computed from
  the saved vectors it is 95.5 to 96.5 percent for accuracy (98.9 percent on decisions with reference margin >= 0.10)
  and 94.5 percent (97.3 percent) for bfp8_attn. Response: `tests/decision_agreement.py` added, the gate wired into
  the sweep (margin-aware 98 percent), policy descriptions corrected (bfp8_attn equals the stock defaults: bfp8
  weights, HiFi2 linears, HiFi4 SDPA; accuracy adds bf16 attention weights and HiFi4 attention linears;
  performance keeps bfp8 attention and sets bfp4 + LoFi on w1/w3), two candidates added (`bf16_all`,
  `bfp8_lofi_mlp`), and every policy's fidelity regenerated on the final code path. The default serve profile
  follows the re-run sweep.
- Other review C items addressed: `doc/context_contract.json` updated to 2048 with evidence paths; the
  batched-vs-single nondeterminism (batch-variant reductions; argmax differs on 4 to 5 of 200 subset decisions
  between a text embedded alone and in a batch) is now recorded in the full-model README and the card limitations.
- 23:11 to 23:19: fidelity, decision agreement and (where missing) benches regenerated on the final code for all six
  candidates (`fidelity_<policy>_final.log`, `bench_bfp8_lofi_mlp.log`); `bf16_all` is infeasible (L1 overflow in the
  stock prefill MLP program config, `doc/datatype_sweep/infeasible_bf16_all.json`); the sweep with the agreement gate
  selects `accuracy` (98.9 percent on confident decisions; every bfp8-attention policy is at or below 97.3 percent).
- 23:08 to 23:13: container build 6 (`package exit 0`, image `38a80e5078b7`) with the `accuracy` default profile.
- 23:21 to 23:40: full evaluation suite against the served build 6 default profile (`/home/hous/dev/clm-v0.1-8B/evals/results/package_p150_final_20261001T232132Z`);
  the Typed Decisions trace shows 300 of 400 cases paying one batch-8 1024-token pass (1,302 ms) for five 130 to
  300 token texts.
- 23:25 to 23:45: encoder owns its prefill buckets (128 / 256 / 512 / 1024 / 2048, `CLM_TRACE_LENS`); fidelity
  (`fidelity_accuracy_buckets5.log`: single-text vectors bit-identical to the nine-variant encoder), bench
  (`bench_accuracy_buckets5.log`: 256 tokens 72 ms, 512 tokens 94 ms, 512 x 8 625 ms) and the trace-safety check
  (`replay_trace_check_buckets5.log`, 15 variants, pass) run on chip 0 after the evaluation chain released it. A
  first attempt to use chip 1 concurrently failed: `TT_METAL_VISIBLE_DEVICES=1` still makes UMD wait for chip 0's
  `CHIP_IN_USE` lock held by the container, and `TT_VISIBLE_DEVICES=1` requires a custom fabric mesh graph descriptor
  (`tt_cluster.cpp:277`).
- 23:44 to 23:45: class-level grid experiment (`grid_experiment_class.log`): forced 4x9 / 4x10 / 4x11 grids are
  rejected by `matmul_config` ("Input width must be divisible by tile size times grid size"); the earlier
  instance-level forced-grid timings are retracted (`../optimized_decoder/README.md`).
- 23:44: container build 7 started with the five-bucket encoder.
