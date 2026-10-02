# Responses to the second-round reviews (A2, C2, R)

Written 2026 Oct 2 (UTC times). Each finding names the review, the action, and where the evidence now lives.
Paths are under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/` unless absolute.

## Review A2 (stages 1 to 3)

- P1, geometry conclusion unsupported, no measured candidate. Accepted. The forced-grid experiment had asked for
  11 rows instead of 11 columns and the "divisibility contract" conclusion was wrong. New experiment
  `tests/program_config_experiment.py` (instance-level getter overrides, consumed config recorded, HF bf16 layer
  reference) at 128 / 256 / 512 / 1024 / 2048 tokens: `doc/optimized_decoder/program_config_experiment_<len>.json`,
  table and findings in `doc/optimized_decoder/README.md`. Kept: QKV `in0_block_w 4` / out subblock 1x4 at 128
  tokens (-3.1 percent of the layer), `MinimalMatmul` on an 11x10 grid for QKV and w2 above 128 tokens (-2 to -4.5
  percent each), block-sharded RMSNorm for 128 to 512 rows (-1 to -2.3 percent); together -5.4 to -6.7 percent per
  layer. Rejected with evidence: uneven per-core N splits run but return wrong numbers (PCC 0.00 to 0.03) on every
  shape; `in0_block_w 16` overflows L1 for wo and w1/w3; the sharded create-heads path fails in the matmul's circular
  buffer config. Shipped through `tt/encoder.py` (`_install_program_configs`, `_install_sharded_norms`, env toggles);
  end-to-end validation in `doc/optimized_full_model/README.md`.
- P2, stale reconciliation JSON and tool defaults. Done: `perf_summary*.json` regenerated with 1.557 / 4.519 ms,
  `tests/perf_summary.py` default and note changed.
- P2, retraction consistency and cause. Done: both instance-patch JSON files carry `retracted` notes naming the
  `lru_cache` mechanism; README labels the reference as the HF bf16 layer; the recomputed grid fields are explained.
- P2, typecast and fill-cache ops. Recorded in `doc/fused_decoder/README.md` with the exact reason they cannot be
  skipped from the autoport (unconditional in `attention.py`, 0.7 to 1.0 percent of the layer); not measured.
- Other concerns: audit shares recomputed over the layer-only denominator; 1024-token core counts corrected; REPORT
  section 5 rewritten for the fifteen-variant encoder; the gap convention is stated (relative to the measurement).

## Review C2 (stages 4 to 8)

- P2, stage 6 README table and gate. Done: table replaced with the JSON values, agreement rows added with the
  margin-aware justification, qualitative substitute recomputed from the final-path vectors; PLAN section 4 amended
  for rows 4, 6 and 8.
- P2, "all regenerated" provenance. Done: `bench_performance.json` re-run on the final code (2026 Oct 2); the sweep
  README states the provenance of every column.
- P2, MLP fidelity lever under the selected attention policy. Done: `accuracy_lofi_mlp` added, measured
  (`doc/datatype_sweep/fidelity_accuracy_lofi_mlp.json`, `agreement_...json`, `bench_...json`): passes every gate
  (98.9 percent confident-decision agreement, cosine min 0.99584) and is faster in every cell except 128-token
  batch 1 (+1.5 percent). The selection rule was amended to a served-workload sum (PLAN amendment 2026 Oct 2 00:55)
  and the default profile changed to this policy.
- P2, `selected_precision_config.json` fields. Done (activation dtype, layer exceptions, CCL dtype, final norm and
  heads placement, weight dtype passed, runtime flag, consumption evidence, gate vector set and resolution).
- Other concerns: bf16_all variant label corrected (128-token batch-1 variant); gate vector set and resolution
  stated; latency direction wording fixed; `_select_trace_lens` now rejects a `max_seq_len` that is not a multiple
  of 128 and `from_env` rejects `CLM_MAX_TOKENS` above `CLM_MAX_SEQ_LEN`; the twophase_limit96 file is described;
  the consolidated work log records itself as a deviation from the per-stage log convention (first paragraph of
  `doc/full_model/work_log.md`); the multichip README's residual statements are reconciled
  and the stage 4 gate outcome (mean 0.99929, min 0.99547) is recorded. Not done: a Pareto plot of agreement vs
  latency (the table carries the agreement columns); extending the fp32 reference beyond 40 cases.

## Review R (release and evaluation)

- P1, uncommitted kernel edits in the image. Accepted. RUN_NOTES corrected (files, diffs, layers, affected path).
  The committed kernels were tested on the 1x4 mesh (fidelity and agreement pass, `doc/multichip_decoder/README.md`),
  and the published build is produced from a clean worktree of the branch, so the image is reproducible from the
  pushed commit (`dirty: false`). See "Publish" in `doc/release/RUN_NOTES.md`.
- P2, the CLM README's 58.1 ms. Done in the card and REPORT section 2: labelled as 38 tokens embedded with cached
  options, set against the p150 new-state row; the "4.5 times" derivation removed; one 4090 cache column used.
- P2, card attribution and p150-fast wording. Done: encoder rows attributed to the host bench, served rows given
  separately, throughput capped at the served maximum; p150-fast given both agreement numbers; near-tie sentence
  rewritten with the counts of the final default-profile run (8 of 200, 5 below margin 0.10, 3 confident);
  p150-fast served once from build 10, which has the same code sha256 as the published image (RUN_NOTES).
- P2, stale commit and blob count in RUN_NOTES. Done.
- Other concerns: Typed Decisions table carries footnotes on the latency protocol and the ECE column; the T-Rex
  wording about "same encoder latency" corrected; the Hub tag list completed.

## Review R2 (release verification, 2026 Oct 2 01:50 UTC)

- P1, reversed `p150-accuracy` speed claim and P1, wrong disagreement split: both came from the pre-override and
  build 7 numbers. Card and release notes rewritten from the final bench and the build 10 agreement file (8.7 and
  19 to 22 percent slower; 8 of 200 reversed, 5 below margin 0.10, 3 of 188 confident = 98.4 percent); revision 3
  pushed with the corrected card.
- P2 items: README-example probabilities now from the shipped policy (single-text 0.844 / 0.991 / 2.000; served
  0.875 / 0.986 / 2.000); p150x4 latency from the published image (169 ms); release-notes profile table replaced;
  "the published build" label moved to the eleventh attempt; REPORT head commit, relative paths, the 57.8 ms cell,
  the 3.0 percent bound gap, 1,301 ms and the per-text token range corrected.
