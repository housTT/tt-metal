# Stage Review

Review A: stages 1 (functional decoder), 2 (graph fusing), 3 (optimized decoder), batched checkpoint.
Target: Contrastive-LM/CLM-v0.1-8B (frozen Qwen3-8B encoder, last-token pooling, two MLP heads).
Autoport: `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b`, branch `hous/clm-v0.1-8b`,
live worktree at commit `fe0b69f03e`. Reviewer mode, read-only. Reviewed 2026 Oct 1.

Acronyms: PCC = Pearson correlation coefficient. HF = Hugging Face. KV = key/value. RMSNorm = root mean square
normalization. SDPA = scaled dot-product attention. RoPE = rotary position embedding. MLP = multilayer perceptron.
QKV = query/key/value projection. DRAM = device memory. L1 = core-local memory. HiFi2/HiFi4 = math fidelity modes.
BFP8 = 8-bit block floating point. BF16 = bfloat16. p50 = 50th percentile. CSV/JSON = file formats. UTC log
timestamps are quoted as written in the logs; Eastern Time (ET) is UTC minus 4 hours on this date.

Verdict: more-work-needed

The stage 1 correctness evidence is sound and re-derives cleanly (33 rows, all PCC >= 0.99972, watcher clean,
Tracy capture matches the described test). The verdict is driven by stage 3, whose optimization pass did not act on
its own profile and whose audit misstates what the profile shows, plus several documentation and provenance defects
in stages 1 to 3 that a reader would otherwise take as fact.

## Required Work

- P1: Stage 3 did not perform the optimization pass on the measured path; the audit misstates the measured geometry
  Evidence:
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/functional_decoder/tracy/layer0/prefill_perf_report.csv`,
  re-derived by this review (per-row listing in Scope Inspected):
  - At the 128-token bucket every matmul ran on 32 cores, not 64: rows 13 (QKV 128x4096x6144), 26 (wo
    128x4096x4096), 29 and 30 (w1/w3 128x4096x12288), 32 (w2 128x12288x4096) all have `Cores=32`. The wo row's
    advice text is literally "Increase grid size (currently using 32)". The "(currently using 64)" advice appears only
    on the 1024-token rows (122, 125, 126, 128) and "(currently using 80)" on row 109. Both
    `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/functional_decoder/README.md`
    ("64 cores") and
    `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/optimized_decoder/README.md`
    ("64 cores", "advises 'Increase grid size (currently using 64)' on every large matmul") are wrong for the
    128-token rows. Cause is visible in the framework:
    `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` line 880 sets
    `prefill_rows = 8`, and `find_prefill_grid` (line 3619) caps at 8x8; for 128 tokens the matmul M dimension is 4
    tiles, so the kernel used a 4x8 grid.
  - Untried `tt-perf-report` advice on the largest 128-token op. Row 13 (QKV, 265 us, 17 % of the layer, marked
    `SLOW`): "in0_block_w=1 is small, try in0_block_w=2 or above", "Output subblock 1x1 is small", "If possible place
    input 0 in L1". Rows 29, 30, 32 (w1/w3/w2, 729 us combined, 47 % of the layer, all marked `SLOW`): "If possible
    place input 0 in L1 (currently in DEV_0_DRAM_INTERLEAVED)". The stage 3 README does not mention the
    `in0_block_w=1` advice or the `SLOW` classification and records no attempt.
  - Non-matmul ops on 4 cores at 128 tokens: RMSNorm x2 (65.1 us and 65.3 us, rows 12 and 28), NlpCreateHeads
    (78.8 us, row 14), NLPConcatHeads (51.3 us, row 25). Total 260.5 us = 16.7 % of the 1559 us layer-only time.
    The stage 3 README lumps these into "remainder (about 0.45 ms)" and records no program-config or core-grid
    attempt. `/home/hous/dev/ornith-1.5-9b/tt-metal/models/common/rmsnorm.py` line 177 passes `program_config=None`
    for interleaved inputs, so the norm parallelizes only over the 4 tile rows; a sharded program-config helper
    already exists at `model_config.py` line 3880.
  - No search table exists for any dominant matmul role. The README's "Matmul geometry: Not swept" cites "a
    `models/tt_transformers` framework change affecting every model on Blackhole" as the reason. That is not an exact
    L1, divisibility, padding, or op-contract blocker. The program configs are fetched at forward time through
    `ModelArgs` getters (`/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/attention.py` lines 1075 and
    1316; `model_config.py` `mlp1_3_grid` lambda at line 882), and the autoport already injects a callable into
    `create_tt_model` (`tt/encoder.py` line 162), so a per-model override is available without changing framework
    defaults for other models. This review did not run such an override; that is the stage owner's work.
  Why this matters: the 128-token bucket is the headline serving shape for this model (README example is 38 tokens,
  `doc/probe/probe_full_encoder.json` texts are 5 to 19 tokens). At that shape 73 % of layer time is matmul on 32 of
  110 cores and a further 17 % is 4-core ops. The stage-review skill requires a precision-locked geometry sweep or an
  exact blocker for dominant rows marked `SLOW` or with `in0_block_w <= 2`; neither exists. The stage prompt
  (`03-optimized-decoder.txt`) requires "a first TTNN/API error is not enough to reject"; here there was no attempt at
  all.
  Required next step: under the selected policy (`bfp8_attn`) and the `accuracy` profile, for each dominant role at
  128 and 1024 tokens (QKV, wo, w1/w3, w2, RMSNorm, create/concat heads), measure legal candidates: core grid beyond
  4x8 at 128 tokens and beyond 8x8 at 1024 tokens (110 worker cores; K=4096 is 128 tiles, N=12288 is 384 tiles,
  N=6144 is 192 tiles, so grids such as 8x10 or 10x11 need divisor checks), `in0_block_w` for QKV from 1 upward
  through legal divisors, output subblocks >= 2, input 0 in L1 where it fits, a sharded or multi-core norm config for
  M=128. Record a per-role table (config, layer latency, PCC, kept/rejected, reason) and the resulting end-to-end
  latency, or record the exact op-contract or L1 blocker per candidate. Correct the core counts and advice quotes in
  both READMEs.

- P2: Stage 1 per-op table misattributes the 13.1 % row and the core counts
  Evidence: `doc/functional_decoder/README.md` row "wo matmul at 1024 tokens (bf16, HiFi4) | 13.1 % | FLOPs 79 %, 64
  cores". The CSV's 13.14 % entry is `MatmulDeviceOperation b={2} x 512 x 4096 x 12288`, n=2, 733.5 us and 732.7 us,
  FLOPs 79.4 %, 64 cores: the two MLP w1/w3 matmuls of the 1024-token pass. The actual wo at 1024 tokens is row 122,
  429.2 us = 3.85 %, FLOPs 90.5 %. The 128-token rows list 64 cores where the CSV says 32 (see P1).
  Why this matters: the stage 2 and 3 audits are built on this table; a reader optimizing "wo at 1024" would be
  working on a 3.85 % op.
  Required next step: regenerate the table from the CSV (op code, time, share, cores, dtype, fidelity, bound,
  advice) and correct the fused and optimized READMEs that quote it.

- P2: Stage 3 reconciliation counts test-harness ops as layer time and cites an unsourced number
  Evidence: the per-layer figures 1.609 ms and 4.692 ms (`doc/optimized_decoder/README.md`,
  `doc/optimized_full_model/perf_summary.json` `per_layer_device_ms_from_tracy`) are the sums of 24 rows per pass,
  which include two test-harness ops per pass: `TilizeDeviceOperation FP32 => FP32` and `TypecastDeviceOperation FP32
  => BF16` produced by `prepare_residual_tensor_prefill` converting the fp32 torch input (rows 10-11, 34-35, 58-59,
  82-83, 106-107; 52 us at 128 tokens, 173 us at 1024). Layer-only device time is 22 ops: 1557 to 1559 us at 128
  tokens and 4519 us at 1024. 36 x layer-only = 56.1 ms and 162.7 ms. Measured end to end (accuracy, batch 1):
  57.6 to 57.7 ms (`doc/optimized_full_model/bench_accuracy.json`, `doc/fused_decoder/eager_vs_trace_128.json`) and
  170.1 to 170.5 ms (`bench_accuracy.json` 512 and 1024-token rows, `eager_vs_trace_1024.json`). Real gap: +2.8 % at
  128 and +4.6 % at 1024, consistent sign, not "0 %". The README's "168.5 ms" for 1024 tokens appears in no committed
  JSON; it comes from `/home/hous/dev/clm-v0.1-8B/logs/bench_accuracy.log` line 899, a run at 21:57 UTC (17:57 ET)
  superseded by the committed `bench_accuracy.json` (22:17 UTC, 18:17 ET, 170.5 ms). The statement "The end-to-end
  latency equals the sum of device kernel time" is therefore not supported; "24 device ops per layer pass" is 22 plus 2
  harness ops, and the audit table lists "tilize" as a layer op.
  Why this matters: the performance-accounting requirement of the optimize skill is to name the non-device terms;
  a bound that includes harness ops hides a 3 to 5 % term (embedding op, 1 to 8 MB readback, host norm, dispatch).
  Required next step: recompute the bound from the 22 layer rows, report the gap with its sign, name the terminal
  work, and cite the artifact for every number in the table.

- P2: Stage 2 gate failed as written and the audit missed three avoidable per-layer ops
  Evidence: PLAN.md section 4 row 2 gate: "prove the traced prefill beats the untraced baseline".
  `doc/fused_decoder/eager_vs_trace_128.json` speedup_p50 0.9965; `eager_vs_trace_1024.json` 0.9986. The README
  records parity honestly but the plan gate is unchanged. Control that supports parity: within the warm 128-token
  passes of the Tracy capture (rows 36-105) the device op-to-op gap is median 0.2 us and p90 0.7 us, so eager dispatch
  never starves the device at these shapes. Note that the eager timing in
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/tests/eager_vs_trace.py` lines
  44-48 reuses pre-copied `device_inputs` while the traced path (`embed_ids`) includes the host-to-device token copy,
  which slightly favors eager. Missed audit items in the executed graph: (a) two identity typecasts per layer,
  `TypecastDeviceOperation BF16 => BF16` (rows 19-20, 2.5 us each at 128 tokens, 10.8 us and 10.4 us at 1024), from
  `attention.py` lines 1138 and 1147 casting K/V to the KV-cache dtype, which is BF16 under the accuracy policy;
  (b) two `PagedFillCacheDeviceOperation` per layer (rows 21-22, 2.5 us each at 128, 13.0 us each at 1024;
  `attention.py` lines 1200-1201) that write a KV cache this encoder never reads (SDPA prefill consumes the K/V head
  tensors directly, row 24 inputs `BFP8, BF16`), plus the cache allocation itself. Together about 0.6 % of layer time
  at 128 tokens and 1 % at 1024.
  Why this matters: the stage prompt requires that all fusing patterns be exhausted and the README claims "No
  remaining host round trips" and parity; the plan's own gate is failing and the audit is incomplete.
  Required next step: amend PLAN.md row 2 and the stage 2 README to state the accepted criterion (traced path not
  slower than eager and trace-safe) with the op-gap control cited; evaluate removing the identity casts and the dead
  fill-cache writes for the encoder (or record a minimal repro if the framework path cannot skip them) and record the
  before/after.

- P2: Stale or contradicting artifacts
  Evidence: (a) `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/doc/context_contract.json`
  (mtime 20:59 UTC, 16:59 ET, before any stage 1 run) says `current_supported_context: 1024` with a stage-7 limiting
  reason, while `doc/full_model/README.md` says 2048 is served and `layer_pcc.json` tests 2048. It does not record the
  largest context tested in prefill, and the 40960 to 2048 reduction has no byte calculation or explicit
  product-decision statement; the plan's written substitute is "upstream `clm-serve` default 2048". (b)
  `doc/optimized_full_model/replay_trace_check_accuracy.json` is a failing run (`pass: false`, "Found 176 device
  buffer(s) still alive", 22:11 UTC, 18:11 ET) committed in `fe0b69f03e` beside the passing
  `replay_trace_check_bfp8_attn.json` (22:26 UTC, 18:26 ET, 9 variants, min repeated-replay cosine 0.9999997). The
  accuracy policy is shipped as the `p150-accuracy` serve profile (`tt-model.yaml` line 62) and is the policy stages
  2 and 3 measured. The warmup code is policy-independent, so the bfp8_attn pass likely covers it, but the committed
  artifact says the opposite.
  Why this matters: a reader of the committed tree finds a failing trace-safety gate for a shipped profile and a
  context contract that disagrees with the served value.
  Required next step: rerun `tests/replay_trace_check.py --precision accuracy` with tracking enabled on the final
  code (or delete the stale file and note why); update `context_contract.json` with current 2048, largest tested
  prefill 2048 (layer and full model), and either a KV/DRAM byte calculation for longer buckets or an explicit
  statement that 2048 is a product decision matching upstream, not a device limit.

- P2: Plan-contract items of stage 1 dropped without a written substitute
  Evidence: PLAN.md row 1 requires "Token embedding and final norm PCC >= 0.995" and "batch 1 and 4". No artifact
  gives an embedding or final-norm PCC; the stage 1 README does not mention them. "Batch 4" was implemented as users
  0 to 3 run sequentially on the same input (`tests/test_functional_decoder.py` line 107), which yields PCC identical
  to 16 digits across users (0.9997860599702103) and does not exercise a batch-4 prefill. The only batched evidence is
  full-model (`doc/full_model/README.md`, batched-vs-single min cosine 0.9964 attributed to tt-metal issue 47238) and
  is not cited from the stage 1 README.
  Why this matters: the plan declares written substitutes for decode-only items; these two are prefill items that
  were silently narrowed.
  Required next step: either add the embedding and final-norm PCC rows and a batch-4 prefill row to the layer test,
  or write the substitute explicitly in the stage 1 README with the full-model artifacts that cover it.

- P2: Missing work logs and provenance for stages 1 to 3
  Evidence: PLAN.md section 4 requires `doc/<stage>/work_log.md` per stage; only `doc/probe/work_log.md` exists
  (`find doc -name work_log.md`). The Tracy run's environment (`CLM_TEST_LENGTHS=128,1024 CLM_TEST_LAYERS=0
  CLM_TEST_OUT=layer_pcc_profiled_run.json`, inferred from `layer_pcc_profiled_run.json`) and the `eager_vs_trace.py`
  invocations are recorded nowhere. The `tt-perf-report` run used no signposts ("No signposts found in the file. Using
  the entire file for analysis.", `prefill_perf_report.console.log`), so the window includes the cold first pass and
  setup ops (rows 0-9). No human-readable `prefill_perf_report.txt` exists. The primary evidence files
  `prefill_perf_report.csv`, `prefill_perf_report_stacked.csv` and `prefill_perf_report.console.log` are excluded from
  git by the repository root `.gitignore` (`*.csv` line 8, `*.log` line 7; confirmed with `git check-ignore -v`), so
  the committed README cites files that are not in the commit. `git ls-files` for `doc/functional_decoder` returns
  only the README, the two JSON files and the PNG.
  Why this matters: the stage-review skill treats paths in reports as evidence that must exist and match the run;
  a fresh checkout of `fe0b69f03e` cannot verify the stage 1 and 3 numbers.
  Required next step: add `work_log.md` for each stage with exact commands and environment; force-add the perf CSV
  and console log (or copy them to a non-ignored name) and the `.txt` table; rerun the perf report with signposts
  around warmed passes if the capture is regenerated for P1.

## Other Concerns

- `layer_pcc.json` reports cosine values above 1.0 in 7 rows (up to 1.00286 at layer 17, 2048 tokens). Cosine
  similarity cannot exceed 1; this is a float32 accumulation artifact of `torch.nn.functional.cosine_similarity` over
  8.4 M elements with large-magnitude hidden-state outliers. The PCC column is the gate and is computed separately,
  so the pass stands, but the cosine column is not trustworthy as recorded. Recompute in float64 or drop it.
- The HF reference for the layer test is the bf16 CPU model (`reference_dtype: torch.bfloat16`), not fp32. PCC of
  0.9997 against a bf16 reference is fine for the gate, but the README should say the reference precision.
- The `tt-perf-report` fidelity advice is quoted as applying to "the HiFi4 attention matmuls". In the CSV it is on wo
  (BFP8 activations x BF16 weights, rows 26 and 122). On QKV (BF16 x BF16, row 13) the advice is the weaker "HiFi2 may
  also work, it discards the lowest bit of the activations". The `bfp8_attn_hifi2` policy applies HiFi2 to QKV with
  BFP8 weights, which is the correct pairing, so the generalization is harmless but imprecise.
- The stage 3 "Expected upside: up to about 1.4x on the matmul share" is derived from the 1024-token rows (64 of
  110 cores). At 128 tokens the matmuls use 32 cores, so the upside at the headline shape is understated.
- The eager vs trace comparison covers batch 1 only; batch 4 and 8 traces were never compared with eager.
- The Tracy capture's 1024-token pass is a single cold pass (5.13 s wall in `layer_pcc_profiled_run.json`). Device
  kernel time is not affected by compile, and the four 128-token passes agree to within 0.2 %, so the numbers are
  usable, but the skill asks for warmed signposted windows.
- `doc/fused_decoder/README.md` lists "final norm moved to the host (fp32)" under fusions present. That is a host
  step added to the request path, not a fusion. Its rejected alternative ("fusing the final RMSNorm into the trace
  for all positions ... equivalent cost") has no measurement attached.
- `reference/hf_layer_reference.py` lines 75-77 and 92-100 contain dead `if False` branches in the `__main__`
  block. Harmless, but the file is cited as the verified reference convention.
- Watcher coverage exists only for the eager single-layer test. The full traced encoder path (embedding op, trace
  replay, batch 4 and 8 variants) has no watcher run: probe run 2 used `TT_METAL_WATCHER=10` but hung on the pinning
  bug before the model ran (`/home/hous/dev/clm-v0.1-8B/logs/probe_full_encoder_run2.log`). This belongs to the stage
  6/7 review but is noted here because the stage 2 and 3 READMEs point back to the stage 1 watcher run.

## Hard-Check Gaps

- No `prefill_perf_report.txt` (human-readable table); only the `--csv` console log exists.
- No signposts in the Tracy capture; the analysis window is the whole file.
- `prefill_perf_report.csv`, `_stacked.csv`, and `.console.log` are git-ignored and absent from `fe0b69f03e`.
- No `work_log.md` for stages 1, 2, 3; exact commands and environment variables for the Tracy and eager-vs-trace
  runs are unrecorded.
- Determinism evidence for stage 1 is full-model level (probe, 7 texts at the 128 bucket; replay check, 9 variants),
  not layer-level. The substitute is written in the stage 1 README and is acceptable.
- `context_contract.json` lacks the "largest context tested in prefill" field the functional-decoder skill asks for.
- The stage 3 README has no per-role search table and no `perf_summary.json` of its own (the stage 7 file is reused).

## Anomaly Ledger

- Observed anomaly: 128-token matmuls on 32 cores while both READMEs say 64 and quote "(currently using 64)".
  Evidence: `prefill_perf_report.csv` rows 13, 26, 29, 30, 32 `Cores=32`; row 26 advice "Increase grid size
  (currently using 32)"; `model_config.py` lines 880 and 3619.
  Affected path: all prefill matmuls at the 128 bucket in `models/tt_transformers`, used by every serve profile.
  Control or comparison: 1024-token rows show 64 cores (80 for the MinimalMatmul QKV), matching the README text.
  Likely subsystem: `find_prefill_grid` 8x8 cap combined with M=4 tiles at 128 tokens; README transcription from
  the 1024 rows.
  Investigation performed: full row listing and per-pass sums from the CSV; code read of the grid helpers.
  Resolution: more-work-needed (P1).

- Observed anomaly: the 13.1 % row is labeled "wo matmul at 1024 tokens".
  Evidence: CSV op code `MatmulDeviceOperation b={2} x 512 x 4096 x 12288`, n=2, 1466 us; wo at 1024 is row 122,
  429 us, 3.85 %.
  Affected path: stage 1, 2, 3 audit tables.
  Control or comparison: FLOPs 79 % and 64 cores in the README row match the w1/w3 rows, not the wo row (90 %).
  Likely subsystem: documentation.
  Investigation performed: op-code aggregation over the CSV.
  Resolution: more-work-needed (P2).

- Observed anomaly: "gap 0 %" reconciliation while 36 x layer time is below the measured end to end at 128 and
  above the committed measurement at 1024.
  Evidence: 24-row sums include harness Tilize/Typecast (52 us, 173 us); 22-row sums give 56.1 ms and 162.7 ms vs
  57.6 to 57.7 ms and 170.1 to 170.5 ms measured; "168.5 ms" traced only to `bench_accuracy.log` line 899 (superseded run).
  Affected path: stage 3 reconciliation, stage 7 `perf_summary.json` lower bound.
  Control or comparison: the four 128-token passes agree within 0.2 %, so the layer time itself is stable.
  Likely subsystem: analysis method (window boundaries) and number provenance.
  Investigation performed: per-pass segmentation of the CSV; grep of all committed JSON and logs for the figure.
  Resolution: more-work-needed (P2).

- Observed anomaly: two `Typecast BF16 => BF16` and two `PagedFillCache` ops per layer in an encoder that never
  reads the cache.
  Evidence: rows 19-22 (128 tokens) and 115-118 (1024 tokens); `attention.py` lines 1138, 1147, 1200, 1201.
  Affected path: attention prefill in every layer.
  Control or comparison: SDPA input dtypes (row 24: `BFP8, BF16`) show K/V heads are consumed directly.
  Likely subsystem: stock `tt_transformers` attention; audit omission in stages 2 and 3.
  Investigation performed: CSV row inspection and code read.
  Resolution: more-work-needed (P2, small cost, but the stage claims the audit is complete).

- Observed anomaly: RMSNorm, create-heads and concat-heads run on 4 cores at 128 tokens, 16.7 % of layer time.
  Evidence: rows 12, 14, 25, 28 `Cores=4`; 260.5 us of 1559 us.
  Affected path: every layer at the 128 bucket.
  Control or comparison: at 1024 tokens the same ops use 32 cores (rows 108, 110, 121, 124).
  Likely subsystem: interleaved-input default program configs that parallelize over tile rows.
  Investigation performed: CSV inspection; `models/common/rmsnorm.py` line 177.
  Resolution: more-work-needed (part of P1).

- Observed anomaly: traced prefill is not faster than eager (0.9965x, 0.9986x) although the plan gate requires a win.
  Evidence: `eager_vs_trace_128.json`, `eager_vs_trace_1024.json`; PLAN.md row 2.
  Affected path: stage 2 gate.
  Control or comparison: warm-pass op-to-op gaps median 0.2 us, p90 0.7 us (CSV rows 36-105): device is never idle
  waiting on host dispatch, so parity is the expected outcome.
  Likely subsystem: workload is device-bound; gate text assumes a dispatch-bound decoder.
  Investigation performed: gap statistics from the CSV; read of `eager_vs_trace.py` and the generator trace paths.
  Resolution: controlled by evidence; more-work-needed for the gate text and README citation (P2).

- Observed anomaly: "Allocating device buffers is potentially unsafe due to the existence of an active trace"
  warnings during warmup in both eager-vs-trace runs.
  Evidence: `/home/hous/dev/clm-v0.1-8B/logs/eager_vs_trace_128.log` line 528; `eager_vs_trace_1024.log` line 529,
  each emitted between the first and second "Done Capturing Prefill Trace".
  Affected path: multi-variant trace capture in `tt/encoder.py` `warmup`.
  Control or comparison: `replay_trace_check_bfp8_attn.json` with `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, 9 variants, 3 rounds alternating order, `pass: true`, no unsafe
  buffers. The warning is the second capture's own allocations while trace 1 is live, which the tracker accepts.
  Likely subsystem: trace capture; expected when more than one trace is captured.
  Investigation performed: log context read; tracker JSON read; code read of two-phase warmup.
  Resolution: controlled for `bfp8_attn`; the committed `accuracy` tracker artifact is a stale failure (P2).

- Observed anomaly: cosine values above 1.0 in `layer_pcc.json`.
  Evidence: 7 rows, maximum 1.00286.
  Affected path: evidence metric only; not the gate.
  Control or comparison: PCC column computed by `comp_pcc` is within [0.99972, 1.0].
  Likely subsystem: float32 accumulation in `cosine_similarity` over 8.4 M elements with outliers (unverified cause).
  Investigation performed: JSON scan; code read of the metric call (`test_functional_decoder.py` line 129).
  Resolution: more-work-needed (low): recompute in float64 or remove the column.

- Observed anomaly: `rotary_embedding_llama sequence tile coverage mismatch: input_Ht=4, cos_Ht=32, sin_Ht=32,
  rotary_Ht=4` warnings during 128-token compile in the full encoder.
  Evidence: `eager_vs_trace_128.log` lines 523-524; `eager_vs_trace_1024.log` lines 525-526.
  Affected path: RoPE in the traced encoder when cos/sin are sized for `max_seq_len`.
  Control or comparison: layer test uses exact-length rot mats and passes; full-model cosine at the 128 bucket mean
  0.99910 (`doc/full_model/README.md`); the op message states only tiles beyond `rotary_Ht` are zero-filled, and the
  input has exactly `rotary_Ht` tiles.
  Likely subsystem: `rotary_embedding_llama` warning for oversized cos/sin.
  Investigation performed: log read; cross-check against fidelity results.
  Resolution: controlled (benign by op semantics and by fidelity evidence); should be classified in the stage 2 README.

- Observed anomaly: `context_contract.json` predates and contradicts the stage 1 and 6 evidence.
  Evidence: mtime 20:59 UTC (16:59 ET); `current_supported_context: 1024`; `doc/full_model/README.md` serves 2048.
  Affected path: capability contract.
  Control or comparison: `model_config.py` lines 2646-2650 add 2048 for P150; `replay_trace_check_bfp8_attn.json`
  captures `2048_0_{1,4,8}_sp0`.
  Likely subsystem: documentation not updated after stage 1.
  Investigation performed: file read and timestamp comparison.
  Resolution: more-work-needed (P2).

## Scope Inspected

- Goal/skill paths:
  `/home/hous/dev/clm-v0.1-8B/PLAN.md` (sections 1 to 4);
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/stage-review/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/functional-decoder/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/graph-fusing/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/optimize/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-device-usage/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/skills/tt-enable-tracing/SKILL.md`;
  `/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/prompts/model_bringup_multigoal/01-functional-decoder.txt`,
  `02-fused-decoder.txt`, `03-optimized-decoder.txt`.
- Artifact paths (all under `/home/hous/dev/ornith-1.5-9b/tt-metal/models/autoports/contrastive_lm_clm_v0_1_8b/` unless absolute):
  `doc/functional_decoder/README.md`, `doc/functional_decoder/layer_pcc.json`,
  `doc/functional_decoder/layer_pcc_profiled_run.json`,
  `doc/functional_decoder/tracy/layer0/prefill_perf_report.csv`, `prefill_perf_report_stacked.csv`,
  `prefill_perf_report.console.log`, `reports/2026_10_01_22_01_54/ops_perf_results_2026_10_01_22_01_54.csv`;
  `doc/fused_decoder/README.md`, `eager_vs_trace_128.json`, `eager_vs_trace_1024.json`;
  `doc/optimized_decoder/README.md`; `doc/probe/README.md`, `work_log.md`, `probe_full_encoder.json`;
  `doc/context_contract.json`; `doc/full_model/README.md`; `doc/optimized_full_model/README.md`,
  `perf_summary.json`, `bench_accuracy.json`, `replay_trace_check_accuracy.json`, `replay_trace_check_bfp8_attn.json`;
  `doc/datatype_sweep/README.md`, `selected_precision_config.json`; `doc/review/REVIEW_PROMPT_TEMPLATE.md`;
  `tt-model.yaml`; `.gitignore`; `/home/hous/dev/clm-v0.1-8B/STATUS.md`;
  logs under `/home/hous/dev/clm-v0.1-8B/logs/`: `stage1_functional_decoder_v3.log`,
  `stage1_functional_decoder_watcher_v3.log`, `tracy_layer0.log`, `eager_vs_trace_128.log`, `eager_vs_trace_1024.log`,
  `probe_full_encoder.log`, `probe_full_encoder_run2.log`, `probe_full_encoder_run3.log`,
  `control_test_decoder_prefill.log`, `control_test_decoder_prefill_1x1.log`, `control_test_decoder_prefill_1x1_fix.log`,
  `bench_accuracy.log` (line 899 only); `/home/hous/dev/ornith-1.5-9b/tt-metal/generated/watcher/watcher.log`
  (mtime 22:05 UTC, 18:05 ET, 5 dumps, no exception strings).
- Code paths:
  `tests/test_functional_decoder.py`, `tests/eager_vs_trace.py`, `tests/replay_trace_check.py`, `tests/bench_encoder.py`,
  `reference/hf_layer_reference.py`, `tt/encoder.py`;
  `/home/hous/dev/ornith-1.5-9b/tt-metal/models/tt_transformers/tt/model_config.py` (lines 730-750 of `common.py`
  for `get_padded_prefill_len`; `model_config.py` lines 757-764, 875-900, 995-1035, 2561-2650, 3537-3645, 3880),
  `models/tt_transformers/tt/attention.py` (lines 119-150, 665-714, 1075-1316),
  `models/tt_transformers/tt/generator.py` (lines 583-700, 784-860), `models/common/rmsnorm.py`,
  `models/tt_transformers/tt/mlp.py`, `models/tt_transformers/tt/decoder.py`;
  `git show --stat` of `418c0e85bf`, `498325e8ed`, `fe0b69f03e`; `git show 418c0e85bf -- models/tt_transformers/tt/model_config.py`.
- Commands run (read-only): `cat`, `sed -n`, `grep`, `find`, `ls -la`, `git log/status/show/ls-files/check-ignore`
  in the worktree; four `/home/hous/dev/ornith-1.5-9b/tt-metal/python_env/bin/python` heredoc scripts over
  `layer_pcc.json` (min/mean/coverage), `prefill_perf_report.csv` (op-code shares, per-row listing, per-pass sums with
  harness separation, matmul share, 4-core share, op-to-op gap percentiles), the raw ops CSV (matmul dtype, fidelity,
  core count), and `replay_trace_check_bfp8_attn.json`. No device was opened, no `ttnn` import, no server or test run,
  no implementation file modified. Only this report file was written.

Re-derived numbers that match the READMEs: 33 PCC rows, all pass, gate 0.995; layer 0 min 0.99976 mean 0.99978;
layer 17 min 0.99999966 (reported 1.00000); layer 35 min 0.99972 mean 0.99978; global minimum 0.999724 at layer 35,
33 tokens; coverage is exactly layers {0, 17, 35} x lengths {32, 33, 127, 128, 129, 500, 1024, 2048} with users
0 to 3 at 128; padded buckets 128/1024/2048 per `get_padded_prefill_len`. Plain run 1 passed in 26.41 s; watcher
run 1 passed in 38.94 s, watcher initialized with no disabled features, 5 polls, no exceptions. Tracy run's
`layer_pcc_profiled_run.json` PCC values equal the main run's rows to 16 digits, so the capture corresponds to layer
0 at 128 tokens (users 0 to 3) and 1024 tokens. 130 ops, 11.158 ms total device time; op-code shares 17.86 %,
13.14 %, 9.45 %, 8.18 %, 7.19 %, 4.75 % match the stage 1 table's percentages (attribution errors noted above).
DRAM roofline 23.6 % (121 GB/s) matches the console log. Eager vs trace p50 and cosine values match the JSON files.

## Residual Risk

- The optimization headroom at the serving-relevant 128-token bucket is materially larger than the stage 3 README
  implies (32 of 110 cores on 73 % of layer time, 4-core ops on 17 %). Until P1 is done the "optimized decoder" label
  is not earned and later-stage latency numbers (54 ms at 128 tokens) should be read as a baseline.
- The full traced encoder path has no watcher-clean run; stage 1 watcher coverage is for the eager single layer.
- Stage 1 PCC is against a bf16 HF reference; full-model fidelity against fp32 (cosine mean 0.99909, min 0.99588) is
  the stronger evidence and lives in stage 6.
- Trace allocation safety is proven only for the `bfp8_attn` policy run; the `accuracy` profile's committed tracker
  artifact is a stale failure.
- Evidence files that the READMEs cite (perf CSV, console log) are not in version control; a checkout cannot
  reproduce this review without the live worktree.
- The eager-vs-trace parity conclusion is batch 1 only and slightly favors eager (no host-to-device copy in the eager
  loop); it is unlikely to flip, but it is not measured for batch 4 and 8.
