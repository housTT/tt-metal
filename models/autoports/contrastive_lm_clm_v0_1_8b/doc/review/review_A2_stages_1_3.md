# Stage Review

Review A2: re-review of stages 1 (functional decoder), 2 (graph fusing) and 3 (optimized decoder) after the
responses to review A and the grid-experiment correction of 2026 Oct 1 23:45 UTC.
Target: Contrastive-LM/CLM-v0.1-8B (frozen Qwen3-8B encoder, last-token pooling, two MLP heads).
Autoport: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b`, branch `hous/clm-v0.1-8b`,
live worktree at commit `b39ef75ab6` (the autoport tree has one uncommitted edit, `doc/release/RUN_NOTES.md`, outside
this scope). Reviewer mode, read-only. Reviewed 2026 Oct 2.

Acronyms: PCC = Pearson correlation coefficient. HF = Hugging Face. KV = key/value. RMSNorm = root mean square
normalization. SDPA = scaled dot-product attention. RoPE = rotary position embedding. MLP = multilayer perceptron.
QKV = query/key/value projection. DRAM = device memory. L1 = core-local memory. HiFi2/HiFi4/LoFi = math fidelity
modes. BFP8 = 8-bit block floating point. BF16 = bfloat16. p50 = 50th percentile. CSV/JSON = file formats.
TTNN = the tt-metal tensor library. Times are UTC as written in the logs (Eastern Time is UTC minus 4 hours).
CSV row numbers in this review are the `ID` column values of `prefill_perf_report.csv` (review A used 0-based line
indices, two lower).

Verdict: more-work-needed

Summary. Points (1) and (2) of the re-review brief verify: the corrected stage 3 audit table and the layer-only
reconciliation (1.557 / 4.519 ms per layer, +2.8 / +4.6 percent) re-derive from the Tracy CSV and the benchmark
JSON to within rounding, with small denominator inconsistencies listed below. Point (3) verifies only in part: the
retraction of the instance-level forced-grid timings is justified by the artifacts, but the replacement conclusion
("divisibility contract, admissible column counts 1, 2, 4, 8, kernel-level work needed") does not follow from
`matmul_config` and `find_prefill_grid`. The experiment forced the wrong grid axis, the assertion it hit is a
Python helper convention that the helper itself lets a caller bypass, the TTNN operation accepts the uneven split
the stage says is impossible, and the largest single op (QKV) never passes through the patched function at all.
Point (4): REPORT.md section 5 repeats that conclusion and cites two JSON files that still carry the retracted
reconciliation. The dominant 128-token matmul rows therefore remain `SLOW` with `in0_block_w=1` on the largest one
and no measured candidate, which the stage-review skill treats as required work.

## Required Work

- P1: The stage 3 geometry conclusion is not supported; dominant 128-token rows still have no measured candidate
  Evidence:
  - `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` lines 3537 to 3575
    (`matmul_config`): `per_core_M = ceil(m / (32 * grid_size[1]))`, `per_core_N = ceil(n / (32 * grid_size[0]))`,
    and the assertion `k % (32 * grid_size[1]) == 0` runs only when `in0_block_w is None`. `grid_size` is passed to
    `compute_with_storage_grid_size`, so `grid_size[0]` is x (columns, the N split) and `grid_size[1]` is y (rows,
    the M split). The assertion constrains the row count, not the column count.
  - `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tests/grid_experiment.py`
    line 69: force mode returns `(min(rows, r), cols)` = `(4, 11)`, which `get_attn_wo_program_config`
    (`model_config.py` lines 2146 to 2156) passes straight through as `grid_size`, so the experiment requested
    x = 4 columns and y = 11 rows. That exceeds the 10-row grid and trips the row-count assertion
    (`geometry_experiment_class_patch.json`, traceback at `model_config.py` line 3561). A grid of x = 11 columns
    and y = 4 rows was never requested. For wo (`k = 4096`), `grid_size = (11, 4)` gives `per_core_M = 1`,
    `per_core_N = 12`, `4096 % 128 == 0`, `in0_block_w = find_largest_divisor(32) = 8` (line 3685),
    `out_subblock_w = 4` (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/common.py` line 673):
    the helper accepts it. The same holds for w1/w3 (`k = 4096`) and w2 (`k = 12288`, 384 tiles, `384 % 4 == 0`).
  - The TTNN operation does not require the N split to divide evenly.
    `/home/hous/dev/ornith-1.5-9b/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp`
    lines 486 to 568 require `Kt % in0_block_w == 0`, the subblock divisibilities and the destination-register
    bound, nothing about `N % per_core_N`.
    `/home/hous/dev/ornith-1.5-9b/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
    line 195 computes `num_blocks_x = ceil(N / per_core_N)` and line 1142 `last_per_core_N = N % per_core_N`
    with partial-block handling. So "uneven per-core N" is supported by the op; the only exact contract is
    `Kt % in0_block_w == 0`, and `in0_block_w` is a caller-supplied Python field.
  - The largest 128-token op never passes through `find_prefill_grid` or `matmul_config`.
    `model_config.py` lines 1808 to 1835 (`get_attn_qkv_program_config`, prefill, `seq_len <= 128`) return a
    hard-coded `MatmulMultiCoreReuseMultiCastProgramConfig` with grid `(8, 10)` on Blackhole, `in0_block_w=1`
    (comment: "FIXME: optimize this config for prefill"), `out_subblock 1x1`, `per_core_M = 1`,
    `per_core_N = ceil(6144 / 32 / 8) = 24`: 4 x 8 = 32 of the 80 grid cores. This matches CSV rows 15, 39, 63, 87
    (`Cores=32`, `Inner Dim Block Size=1`, subblock `1 1`, advice "in0_block_w=1 is small, try in0_block_w=2 or
    above", "Output subblock 1x1 is small"). The experiment's `qkv_grid` field is computed from
    `find_prefill_grid(4, 192)`, a call the QKV path does not make, and in the forced runs the error was raised in
    wo, which executes after QKV, so QKV ran with its stock config in every row of every geometry file.
  - The README's mechanism statement is also wrong in detail: `mlp1_3_grid` and `mlp2_grid` (`model_config.py`
    lines 882 to 891) and the wo config call `find_prefill_grid(self.prefill_rows = 8, k_tiles)`, which returns
    `(8, 8)` (the class-patch log `/home/hous/dev/clm-v0.1-8B/logs/grid_experiment_class.log` line 43 prints
    "MLP prefill grids @ 32: w1/w3: (8, 8), w2: (8, 8)"). The 4-row footprint comes from `per_core_M = 1` at
    M = 4 tiles, not from `find_prefill_grid` "yielding a 4x8 grid"; raising the 8x8 cap cannot change anything
    because the row argument is the constant 8, and the column argument (`k_tiles`) is already divisible by 8.
  - Compute grid: `/home/hous/dev/ornith-1.5-9b/tt-metal/tt_metal/core_descriptors/blackhole_140_arch.yaml`
    "2xharvested" range `[0,0]` to `[10,9]` = 11 columns x 10 rows = 110 cores, matching the 110-core elementwise
    rows in the CSV. x = 11 is a legal grid extent.
  - Untouched 4-core ops: RMSNorm x2, NlpCreateHeads, NLPConcatHeads = 262.9 us in pass 2 = 16.9 percent of the
    1557 us layer (CSV rows 38, 40, 51, 54). The README records "fixed framework configs and were not changed" and
    no candidate.
  Why this matters: the stage-review skill requires, for a dominant row left `SLOW` or with `in0_block_w <= 2`, a
  precision-locked measurement of the larger legal divisors of the tiled K dimension and of the residual-grid
  candidates, or an exact L1, divisibility, padding or op-contract blocker. The blocker recorded here is a
  framework helper's assertion on the wrong axis, bypassable by the helper's own `in0_block_w` argument, and the
  op's actual contract (`Kt % in0_block_w == 0`) is satisfied by every candidate below. At the 128-token headline
  shape 72.6 percent of layer time is matmul on 32 of 110 cores and 16.9 percent is 4-core ops; the "optimized
  decoder" label and the stage 7 statement "The matmul core grid lever is exhausted without kernel changes"
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/README.md`)
  rest on this conclusion. The autoport already overrides `ModelArgs` behavior from `tt/encoder.py` (line 179
  `optimizations=` callable, line 190 `trace_prefill_supported_seq_lens`), so a per-model program-config override
  needs no framework edit.
  Required next step: under the `accuracy` policy at 128 tokens (and 1024 where the same role applies), measure
  and tabulate (config, layer ms p50, PCC, kept or rejected, exact reason) at least: (a) QKV with `in0_block_w` in
  {2, 4, 8} and `out_subblock_w` in {2, 4} on the stock 32-core footprint, and with the N split widened to
  x = 11 (`per_core_N = 18`, 44 cores); (b) wo, w1/w3, w2 through `matmul_config` with `grid_size = (11, 4)` or
  `(11, 8)` and an explicit `per_core_N` that keeps `out_subblock_w = 4` (for example 36 for N = 384 tiles);
  (c) `MinimalMatmul` for QKV and w2 at 128 tokens, which the framework itself already selects at 128 tokens for one
  Galaxy configuration (`model_config.py` lines 1839 to 1845) and for every length above 128; (d) one multi-core
  or sharded program config for the two RMSNorms and one for the head reshapes, or an exact op-contract error per
  candidate. Record the Tracy rows of any kept candidate, the resulting end-to-end latency, and rewrite the
  "Matmul geometry and small-grid ops" section with the correct mechanism (per_core_M, row-count assertion, QKV
  hard-coded config). If every candidate is rejected, the table of exact errors is the deliverable. Do not use
  `tests/grid_experiment.py` as is: patch the four `@lru_cache` getters (`get_attn_qkv_program_config` line 1784,
  `get_mlp_ff1_3_prg_config` 1393, `get_mlp_ff2_prg_config` 1449, `get_attn_wo_program_config` 2107) or construct
  a fresh `ModelArgs` per candidate, and record the program config actually consumed (Tracy `Inner Dim Block
  Size`, `Cores`, subblock columns), not a recomputed `find_prefill_grid` value.

- P2: The retracted reconciliation still lives in the JSON files that REPORT.md section 5 cites, and in the tool
  Evidence:
  `/home/hous/dev/clm-v0.1-8B/REPORT.md` lines 95 to 96 cite
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/perf_summary_accuracy.json`
  and `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/perf_summary.json`
  as the sources of the latency table. Both files (22:25 UTC, unchanged since) carry
  `per_layer_device_ms_from_tracy: 1.609 / 4.692`, `layer_stack_lower_bound_ms: 57.92 / 168.91` and
  `gap_to_lower_bound_pct: -0.4, -0.5, 0.7, 0.9` (accuracy) and `-7.5` to `-14.0` (bfp8_attn), plus the note "24 ops
  per layer pass". The report text two lines below says 1.557 / 4.519 ms and +2.8 / +4.6 percent.
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_full_model/perf_summary_accuracy_buckets5.json`
  has the corrected field values (1.557 / 4.519, gaps 2.9 / 3.0 / 4.5) but its `notes` string still reads "1.609 ms at
  128 tokens, 4.692 ms at 1024; 24 ops per layer pass".
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tests/perf_summary.py` line 41
  defaults `--layer-ms` to `128=1.609,1024=4.692`, line 70 hard-codes the same note, and `layer_pass_times` (lines
  14 to 28) splits passes at `BinaryNg` boundaries so its automatic path sums all 24 ops including the harness
  Tilize and Typecast.
  Why this matters: the brief asks that retractions be consistent across README, report and JSON notes. A reader
  who opens the report's cited sources finds a negative gap and a bound that exceeds the measurement, the opposite
  of the corrected claim, and the tool regenerates the stale note on every run.
  Required next step: regenerate `perf_summary.json` and `perf_summary_accuracy.json` with
  `--layer-ms 128=1.557,1024=4.519` (or delete them and point the report at the buckets5 file), fix the default and
  the note in `tests/perf_summary.py`, and make `layer_pass_times` exclude the FP32 Tilize/Typecast harness rows so
  the automatic path matches the manual number.

- P2: The retraction is not applied consistently to the instance-patch cap file, and its stated cause is not the
  operative one
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_decoder/geometry_experiment.json`
  (23:22 UTC) was produced by the same instance-patch script as the retracted
  `geometry_experiment_forced.json` (the pre-`a890a5ab05` version of `tests/grid_experiment.py`, which built one
  `ModelArgs` before the loop and patched the instance per spec). The README cites its rows as "Cap mode ... Layer
  time 1.608 to 1.616 ms, PCC 0.99979 for all caps", and the file carries no `retracted` or "stock-grid repeats"
  note. Both instance-patch files show PCC 0.9997860599702103 in every row, identical to the stage 1 row for layer 0
  at 128 tokens in `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/functional_decoder/layer_pcc.json`.
  The retraction note attributes the failure to patching "on the ModelArgs instance after construction". The program
  configs are fetched lazily at forward time, so an instance patch would normally take effect; the operative cause
  is that the four config getters are `@lru_cache` methods keyed on the shared instance
  (`model_config.py` lines 1393, 1449, 1784, 2107), so after the first stock iteration every later iteration reused
  the cached stock configs. The committed script creates a fresh `ModelArgs` per spec, which is why the class-patch
  run reached `matmul_config`. The committed script cannot reproduce the two instance-patch files (it always writes
  `mode` and `patch` fields that those files lack).
  Also in this section: the README says "PCC against the HF fp32 layer reference"; `reference/hf_layer_reference.py`
  line 24 loads the HF model in `torch.bfloat16` and `run_layer` casts to `model.dtype`, so the reference is the
  bf16 layer, as the stage 1 README correctly states. The recorded `mlp13_grid` and `qkv_grid` fields are computed
  from `find_prefill_grid(4, 384)` and `find_prefill_grid(4, 192)`, calls the framework does not make (it uses
  `(8, 128)`, `(8, 384)` and no call at all for QKV).
  Why this matters: the cap-mode "measurements" are stock-grid repeats presented as measurements of three caps;
  the analytical conclusion for cap mode is right for the wrong reason (the row argument is a constant 8, see P1),
  and the stated cause of the retraction would mislead the next experimenter into the same trap.
  Required next step: add the same `retracted` or "stock-grid repeats" note to `geometry_experiment.json` (or
  delete it), correct the cause statement to the `lru_cache` mechanism, label the reference as bf16, and either
  drop the recomputed grid fields or record the consumed program config from Tracy.

- P2: Review A's stage 2 item on the identity typecasts and the dead KV-cache writes has no response
  Evidence: review A (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/review/review_A_stages_1_3.md`
  lines 101 to 121) required: "evaluate removing the identity casts and the dead fill-cache writes for the encoder
  (or record a minimal repro if the framework path cannot skip them) and record the before/after". The stage 2
  README (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/fused_decoder/README.md`)
  changed only the gate sentence (`git show a3df3fd2ee`); it still lists no typecast or fill-cache rows and still
  claims the audit is complete. The stage 3 README notes "two dead paged_fill_cache writes per layer (KV is never
  read back)" and stops. The ops are still in the measured graph: CSV rows 45 to 48 (pass 2: 2.6 + 2.8 us
  `Typecast BF16 => BF16`, 2.7 + 2.3 us `PagedFillCache`), rows 117 to 120 at 1024 tokens (10.8 + 10.4 us, 13.0 +
  13.1 us). In `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/attention.py` the typecasts (lines
  1136 and 1144) are unconditional and a fill runs on every path (paged when `page_table` is given, `fill_cache`
  otherwise, lines 1159 to 1210), so skipping them needs a per-model override or a framework flag; that is the
  finding to record.
  Why this matters: about 0.7 percent of layer time at 128 tokens and 1.0 percent at 1024, but the stage claims
  "nothing else left to try" and the previous review's required item was neither done nor answered.
  Required next step: measure the layer with the two casts and two fills removed (a subclass of `Attention` or a
  flag in the autoport's `optimizations` callable), record before and after, or record the exact reason the stock
  path cannot skip them; list the four ops in the stage 2 audit table either way.

## Other Concerns

- Stage 3 audit table denominators. The "share" column is computed over the 24-op pass total (1609.2 us, pass 2:
  QKV 263.3 / 1609.2 = 16.4 percent as printed) while the text says "the layer itself is 1.557 ms"; over the
  layer-only 1557.2 us the shares are QKV 16.9, w1 16.0, w3 16.0, w2 14.6, wo 9.1, NlpCreateHeads 5.2, RMSNorm
  4.2 + 4.2, concat 3.3, RoPE 4.1, SDPA 1.4 percent. "Matmuls are 70 percent of the layer" (also in REPORT.md line
  111) is 70.3 percent of the 24-op total and 72.6 percent of the layer-only time (1130.9 / 1557.2). The last row
  "residual adds, q/k norms, typecasts, paged fill cache | 72 us | 4.5 percent" sums to 78.6 us in pass 2 (the
  5.9 us `BF16 => BFP8` typecast before SDPA, row 49, is missing). Per-op times otherwise match pass 2 of the CSV
  (rows 36 to 59) to the microsecond.
- "At 1024 tokens the matmuls use 64 cores at 79 to 90 percent FLOPs utilization" (stage 3 README; "64 cores at
  1024 tokens" in REPORT.md line 112): CSV row 111 is the QKV `MinimalMatmul 1024 x 4096 x 6144` on 80 cores at
  71.2 percent FLOPs and the w2 `MinimalMatmul` on 64 cores at 85.1 percent; the quoted range covers only wo and
  w1/w3. REPORT.md line 112 also generalizes "43 to 45 percent DRAM, 58 to 64 percent FLOPs" to all 128-token
  matmuls; those are the MLP rows (QKV is 39 / 55, wo 47 / 68).
- REPORT.md line 122: "nine trace captures: 6.2 s". The shipped encoder captures fifteen variants
  (`doc/optimized_full_model/README.md`: "warmup 17.6 s: prepare 16.9 s, capture 0.7 s"); the report's headline
  describes the fifteen-variant build, so this line is stale.
- Stage 7 README reconciliation says "2.9 and 4.7 percent above the bound" from the buckets5 bench; its own
  `perf_summary_accuracy_buckets5.json` gives 2.9 / 3.0 and 4.5 percent (gap over measured). 4.7 is the gap over
  the bound ((170.40 - 162.68) / 162.68). One convention should be used; the stage 3 README and REPORT.md use the
  gap over the measured value.
- `layer_pcc.json` cosine column: 11 of 33 rows exceed 1.0 (maximum 1.00285 at layer 0 and 17, 2048 tokens).
  Review A listed this (counting 7); nothing changed. The PCC gate column is unaffected.
- The stage 1 to 3 work log
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/functional_decoder/work_log.md`)
  ends at 22:50 with "results recorded in `../optimized_decoder/geometry_experiment.json` and README when run" and
  never records the three grid runs (22:57 instance cap, 23:11 instance force, 23:45 class) or the retraction;
  those appear only in the stages 4 to 8 log
  (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/full_model/work_log.md`,
  entry "23:44 to 23:45") and in `/home/hous/dev/clm-v0.1-8B/STATUS.md`.
- The grid experiment timed an eager, untraced layer with per-call `synchronize_device`, so its "layer ms" (1.61 ms)
  includes dispatch and is not comparable with the Tracy device time (1.557 ms). The README does not say which it is
  comparing against.

## Hard-Check Gaps

- No Tracy profile of any alternative matmul or norm geometry exists; every geometry file row that ran used the
  stock program configs.
- `geometry_experiment.json` and `geometry_experiment_forced.json` are not reproducible from the committed
  `tests/grid_experiment.py`; the script version that produced them is only in git history
  (`git show a3df3fd2ee:models/autoports/contrastive_lm_clm_v0_1_8b/tests/grid_experiment.py`).
- The grid experiment logs (`/home/hous/dev/clm-v0.1-8B/logs/grid_experiment.log`, `grid_experiment_forced.log`,
  `grid_experiment_class.log`) are outside the repository and not cited from the README or the JSON files.
- The two stale `perf_summary*.json` files and the `perf_summary.py` note remain the only machine-readable record of
  the reconciliation for the nine-variant benches that REPORT.md section 5 tabulates.
- The Tracy capture still has no signposts and no `prefill_perf_report.txt` (review A); acceptable because the four
  128-token passes agree within 0.3 percent, but unchanged.
- No watcher run covers the traced full-encoder path (review A, belongs to the stage 6/7 review).

## Anomaly Ledger

- Observed anomaly: forced-grid rows in `geometry_experiment_forced.json` report `status: ok` with PCC identical to
  16 digits across 4x8, 4x9, 4x10, 4x11 and to the stage 1 layer-0 row.
  Evidence: the four rows all read 0.9997860599702103; `layer_pcc.json` layer 0, 128 tokens, users 0 to 3 read the
  same value; `grid_experiment_forced.log` line 34 prints stock grids at construction.
  Affected path: stage 3 geometry evidence.
  Control or comparison: the class-patch rerun (`geometry_experiment_class_patch.json`) raises the assertion for the
  same grids, so the ops did not run with them in the instance run.
  Likely subsystem: `@lru_cache` on the four program-config getters keyed on one shared `ModelArgs` instance.
  Investigation performed: read both script versions (`git show a890a5ab05 -- tests/grid_experiment.py`), the
  getters' decorators, and the three logs.
  Resolution: controlled (retraction justified); more-work-needed for the cause statement and for the cap file,
  which has the same defect and no note (P2).

- Observed anomaly: force mode asked for 11 rows, not 11 columns, and the "admissible column counts" conclusion is
  derived from the row-count assertion.
  Evidence: `grid_experiment.py` line 69 returns `(4, 11)`; `matmul_config` uses `grid_size[1]` for `per_core_M`
  and for the K-divisibility assertion; `grid_size[0]` only sets `per_core_N` by ceiling; the C++ validation has no
  N-divisibility rule; the factory handles `N % per_core_N` at line 1142.
  Affected path: stage 3 conclusion, stage 7 "lever is exhausted" statement, REPORT.md lines 115 to 117.
  Control or comparison: `matmul_config(m=128, k=4096, n=4096, grid_size=(11, 4))` passes every assertion by
  inspection; not executed by this review (read-only).
  Likely subsystem: experiment design (axis confusion between `find_prefill_grid`'s `(rows, cols)` and
  `CoreCoord(x, y)`).
  Investigation performed: code read of `matmul_config`, `find_prefill_grid`, `get_attn_wo_program_config`, the
  MLP getters, the C++ validation and factory, and the core descriptor.
  Resolution: more-work-needed (P1).

- Observed anomaly: the QKV prefill matmul at 128 tokens is outside the patched function and keeps `in0_block_w=1`.
  Evidence: `model_config.py` lines 1808 to 1835; CSV rows 15, 39, 63, 87 (`Inner Dim Block Size=1`, subblock 1x1,
  32 cores, `SLOW` advice); in the forced runs the error is raised at `attention.py` line 1316 (wo), after QKV.
  Affected path: largest single op of the 128-token layer (263 us, 16.9 percent).
  Control or comparison: at 1024 tokens the framework switches QKV to `MinimalMatmul` on 80 cores (row 111).
  Likely subsystem: stock `tt_transformers` QKV prefill config, marked FIXME upstream.
  Investigation performed: code read and CSV cross-check.
  Resolution: more-work-needed (P1).

- Observed anomaly: `perf_summary.json` and `perf_summary_accuracy.json` encode a negative gap to the layer-stack
  bound while the report text says +2.8 / +4.6 percent.
  Evidence: fields quoted in P2; REPORT.md lines 95 to 96 and 107 to 109.
  Affected path: stage 3 and 7 reconciliation, REPORT.md section 5.
  Control or comparison: `perf_summary_accuracy_buckets5.json` fields (not its note) and this review's CSV sums
  (layer-only 1555.8 to 1559.2 us for the four 128-token passes, 4518.7 us at 1024; with harness 1607.5 to 1611.1
  and 4691.7 us).
  Likely subsystem: artifacts not regenerated after the correction; tool defaults.
  Investigation performed: JSON read, `tests/perf_summary.py` read, CSV re-sum.
  Resolution: more-work-needed (P2).

- Observed anomaly: two `Typecast BF16 => BF16` and two `PagedFillCache` ops per layer in an encoder that never reads
  the KV cache, unchanged since review A.
  Evidence: CSV rows 45 to 48 and 117 to 120; `attention.py` lines 1136, 1144, 1159 to 1203.
  Affected path: every layer, both buckets.
  Control or comparison: SDPA input dtypes (row 50: `BFP8, BF16`) show K/V heads are consumed directly.
  Likely subsystem: stock attention prefill path; audit omission.
  Investigation performed: CSV and code read; diff of the stage 2 README since review A.
  Resolution: more-work-needed (P2).

- Observed anomaly: stage 3 audit shares use the 24-op denominator while the text defines the layer as 22 ops.
  Evidence: QKV 263.3 / 1609.2 = 16.4 percent (printed) versus 263.3 / 1557.2 = 16.9 percent; remainder row 72 us
  versus 78.6 us summed.
  Affected path: stage 3 README table, REPORT.md "70 percent".
  Control or comparison: pass 2 per-op values match the CSV exactly; only the shares and one sum differ.
  Likely subsystem: documentation.
  Investigation performed: per-op share computation over both denominators.
  Resolution: more-work-needed (low; fold into the P1 rewrite of the README).

- Observed anomaly: `layer_pcc.json` cosine values above 1.0 in 11 rows (maximum 1.00285).
  Evidence: JSON scan of `doc/functional_decoder/layer_pcc.json`.
  Affected path: evidence metric only; gate is PCC.
  Control or comparison: PCC column within [0.99972, 1.0]; 33 of 33 rows pass at 0.995.
  Likely subsystem: float32 `cosine_similarity` accumulation (unverified, as in review A).
  Investigation performed: JSON scan.
  Resolution: controlled (not the gate); unchanged from review A.

- Observed anomaly: REPORT.md section 5 "nine trace captures: 6.2 s" for a fifteen-variant package.
  Evidence: REPORT.md line 122; stage 7 README "Prefill buckets" section.
  Affected path: report only.
  Control or comparison: `bench_accuracy_buckets5.json` `trace_lens` lists five lengths x three batches.
  Likely subsystem: report not updated after the bucket change.
  Investigation performed: text comparison.
  Resolution: more-work-needed (low; one-line fix).

## Scope Inspected

- Goal/skill paths:
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/functional-decoder/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/graph-fusing/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md`;
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` section 4; `/home/hous/dev/clm-v0.1-8B/STATUS.md`;
  `/home/hous/dev/clm-v0.1-8B/REPORT.md` sections 2, 5 and 7.
- Artifact paths (under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/` unless absolute):
  `doc/functional_decoder/README.md`, `work_log.md`, `layer_pcc.json`,
  `doc/functional_decoder/tracy/layer0/prefill_perf_report.csv` (130 rows, 11158.0 us);
  `doc/fused_decoder/README.md`, `eager_vs_trace_128.json`, `eager_vs_trace_1024.json`;
  `doc/optimized_decoder/README.md`, `geometry_experiment.json`, `geometry_experiment_forced.json`,
  `geometry_experiment_class_patch.json`;
  `doc/optimized_full_model/README.md`, `bench_accuracy.json`, `bench_accuracy_buckets5.json`,
  `perf_summary.json`, `perf_summary_accuracy.json`, `perf_summary_accuracy_buckets5.json`,
  `replay_trace_check_accuracy.json`, `replay_trace_check_accuracy_buckets5.json`, `replay_trace_check_bfp8_attn.json`
  (all three `pass: true`); `doc/full_model/work_log.md`; `doc/datatype_sweep/README.md`,
  `selected_precision_config.json`, `sweep_results.csv`; `doc/context_contract.json` (now 2048 with evidence
  paths); `doc/review/review_A_stages_1_3.md`;
  `/home/hous/dev/clm-v0.1-8B/logs/grid_experiment.log`, `grid_experiment_forced.log`, `grid_experiment_class.log`.
- Code paths:
  `tests/grid_experiment.py` (committed and the `a3df3fd2ee` version), `tests/perf_summary.py`,
  `tests/eager_vs_trace.py`, `tests/test_functional_decoder.py` (unchanged since `fe0b69f03e`),
  `reference/hf_layer_reference.py`, `tt/encoder.py`;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` lines 842, 880 to 891,
  1393 to 1510, 1784 to 1845, 2107 to 2158, 3537 to 3575, 3619 to 3643, 3685 to 3689;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/common.py` line 673;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/attention.py` lines 1066 to 1075, 1120 to 1210, 1316;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` lines 486 to 568;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
  lines 190 to 200, 340 to 360, 1140 to 1160;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/tt_metal/core_descriptors/blackhole_140_arch.yaml`.
- Commands run (read-only): `cat`, `sed -n`, `grep`, `find`, `ls -la`, `awk`, `git log/status/show/ls-files/diff
  --stat` in the worktree; three `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` heredoc scripts over
  `prefill_perf_report.csv` (row listing; per-pass sums with and without the FP32 harness rows; per-op mean over the
  four 128-token passes; matmul and 4-core shares over both denominators; 1024-token matmul cores and FLOPs),
  `layer_pcc.json` (per-layer min/mean, cosine > 1 scan, layer 0 128-token rows) and the three replay-check JSON
  files (`pass` flags). No device opened, no `ttnn` import, no test or server run, no implementation file modified.
  Only this report file was written.

Re-derived numbers that match the stage docs: layer-only per pass 1559.2 / 1557.2 / 1558.2 / 1555.8 us (mean 1557.6)
at 128 tokens and 4518.7 us at 1024; harness 51.2 to 52.1 us and 173.1 us; 36 x 1.557 = 56.05 ms versus 57.64 ms
(`bench_accuracy.json`, 128 tokens batch 1) = +2.8 percent over measured; 36 x 4.519 = 162.68 ms versus 170.52 ms =
+4.6 percent. Eager versus traced 57.45 / 57.65 ms and 170.21 / 170.45 ms match the JSON. The REPORT.md section 5
latency table matches `bench_accuracy.json` and `perf_summary.json` row for row, including tokens per second.
Stage 1: 33 rows, all pass at 0.995, minimum 0.999724 (layer 35), coverage layers {0, 17, 35} x lengths {32, 33, 127,
128, 129, 500, 1024, 2048}. Stage 3 per-op times equal CSV pass 2 (rows 36 to 59) to the microsecond; the 1024-token
w1/w3 attribution (13.1 percent, rows 127 and 128) and the 32-core counts at 128 tokens are now correct in both
READMEs. Review A's other P2 items are closed: `context_contract.json` records 2048, the largest tested prefill and
the product-decision reason; `replay_trace_check_accuracy.json` is a passing 22:57 run; the stage 1 README carries
written substitutes for the embedding, final-norm and batch items; work logs exist and the perf CSV and console log
are force-added (`git ls-files`).

## Residual Risk

- The headline-shape optimization has still not been attempted on the measured path. Until P1 is done, every
  128-token latency in the report (57.6 ms batch 1, 162 ms batch 8) is a baseline on 32 of 110 cores with an
  `in0_block_w=1` QKV, and the stage 7 statement that the grid lever is exhausted is unsupported.
- The expected gain is bounded and not proven: the 128-token matmuls sit at 39 to 47 percent DRAM and 55 to 68
  percent FLOPs on 32 cores, so widening to 44 cores or raising the QKV block size may give a 1.1x to 1.3x on 73
  percent of the layer, or nothing if L1 or DRAM bandwidth binds first. Only a run decides this.
- Two JSON artifacts cited by the final report contradict it until regenerated.
- The eager-versus-trace parity and the layer-level correctness evidence are unchanged since review A and remain
  sound; batch 4 and 8 traces are still not compared with eager.
- The full traced encoder path still has no watcher-clean run (stage 6/7 scope).
