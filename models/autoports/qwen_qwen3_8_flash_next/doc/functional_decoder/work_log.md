# Functional-decoder work log

## 2026-08-26: target and environment

- Resolved `Qwen/Qwen3.8-Flash-Next` to checkpoint revision
  `f5d08274bafd880402bd16f5e3e6c514136ec06c` and Transformers 5.16
  `Qwen4ExpTextDecoderLayer`.
- Recorded exact target shapes: 48 layers; Gated DeltaNet by default; every
  fourth layer QSA; zero-based layer 1 adds PLE; hidden size 2560 with four
  10240-wide hyperconnection streams; 512 experts with top-10 routing; 262144
  advertised context.
- Fresh AutoDebug proved the original compiler failures were a split-brain
  environment: stale TTNN 0.65.1 native libraries from a sibling checkout and
  current 0.75 JIT sources. Added `ttenv.sh` to pin repo `python_env`,
  `PYTHONPATH`, `TT_METAL_RUNTIME_ROOT`, chip 0, and the p150 mesh descriptor.
  Full analysis: `AUTODEBUG.md`.

## 2026-08-26: implementation

- Added `tt/model_config.py` and `tt/functional_decoder.py` under the required
  repo-local autoport path.
- Implemented exact four-stream hyperconnections, target GDN and PLE state,
  paged QSA K/V and raw-index caches, partial RoPE, real 512-expert sparse MoE,
  public padding/chunking, multi-user state preparation, and trace-safe decode.
- Kept all weight conversion and host input construction at setup/test
  boundaries. Public prefill/decode runtime uses TTNN tensors only.
- Exposed prepared PLE embeddings because the checkpoint table is
  102400491520 bytes BF16. The PLE device computation remains in the decoder.

## 2026-08-26: correctness AutoFix

- Initial real checkpoint PCC: layers 0/1 passed, QSA layer 3 prefill failed at
  `0.27623364`.
- `$autofix` ran fresh-context QSA diagnosis. Isolated and retained exact
  power-of-two page quotient/remainder, flattened embedding-index row order,
  and explicit batch index-head replication repairs. Precision-only and
  other catastrophic hypotheses were refuted.
- Independent stage review then identified a real residual underfilled-context
  discrepancy: HF top-k uses `min(512, complete_blocks)`, while trace-static
  TTNN top-k must always return 512 lanes. AutoFix retained the fixed shape but
  split validity into complete-block lanes (`selected < complete_tokens`) and
  partial-tail lanes (`tail <= current_pos`). This masks filler blocks and
  includes every tail token once. `topk_multiset.log` verifies exact unique
  token multisets at 17 positions around block/page/tile/budget boundaries;
  `topk_fix_real_qsa.log` reconfirms real QSA PCC.
- Final real PCC after repair:

  | Layer | Prefill PCC | Traced decode PCC |
  | ---: | ---: | ---: |
  | 0 | 0.99871051 | 0.99997765 |
  | 1 | 0.99910986 | 0.99991739 |
  | 3 | 0.99679226 | 0.99988198 |

- Decode correctness uses capture, state restoration, trace replay, then output
  readback. It does not accept a separate eager result.
- Diagnosis and retained/refuted hypotheses: `QSA_AUTODEBUG.md` and
  `AUTOFIX.md`.

## 2026-08-26: capability and paging

- Public TTNN tests cover all layer kinds at logical lengths
  31/32/33, 63/64/65, and 127/128/129; planner coverage adds
  1, 2047/2048/2049. QSA executes a full public 2049-token non-aligned prefill.
- Page-table permutation is semantically invariant. Batch-2 QSA decode uses
  distinct page tables and positions 63/128 and checks the exact physical cache
  slots. Two-user prefill-to-batched-decode passes for every layer kind.
- Decode at batch 32 passes for every target kind. QSA uses 32 distinct page
  table rows and current positions. Evidence: `batch32_decode.log` (`3 passed`
  in 28.93 s).
- Full public 262144-token prefill passed independently for layers 0 and 1:

  ```text
  pytest -q -s '.../test_functional_decoder.py::test_full_advertised_context[0]' --long-context
  pytest -q -s '.../test_functional_decoder.py::test_full_advertised_context[1]' --long-context
  ```

  Evidence: `long_context_linear0_full.log` (`1 passed`, 56.60 s call) and
  `long_context_linear1_full.log` (`1 passed`, 62.58 s call).
- Full public QSA prefill at 262144 executes all 2048 chunks through the
  10240-wide hyperconnection, attention, sparse-MoE and output-concat path,
  while progressing the complete shuffled page/cache geometry. Evidence:
  `long_context_qsa_public_full.log` (`1 passed` in 1072.59 s).
- Public non-aligned 262143-token prefill executes the final short-chunk and
  output-slicing path for every kind: `near_max_gdn_public_final.log`,
  `near_max_ple_public_final.log`, and `near_max_qsa_public.log` (QSA `1 passed`
  in 1073.61 s). A fixed 768 MiB trace reservation initially made PLE miss
  DRAM by roughly 171 MiB; `trace_region_size=0` lets TTNN auto-size trace
  storage as the functional skill requires, after which PLE passed. The failed
  capacity probe is preserved in `near_max_linear_public.log` and
  `near_max_ple_public_retry.log`.
- `qsa_max_context_trace_decode.log` captures and replays QSA decode at current
  position 262143 with the full 262144-token cache/page/RoPE geometry.
- `trace_region_zero_validation.log` reconfirms all three trace-determinism
  cases plus the advertised-context QSA trace after the zero-region change.
- Machine-readable disclosure: `../context_contract.json`.

## 2026-08-26: fallback, trace, determinism, watcher

- `ForbidHostFallback` dynamically replaces `ttnn.from_torch`,
  `ttnn.as_tensor`, and `ttnn.to_torch` with failures around measured
  prefill/decode. Source inspection independently rejects them and `torch.` in
  public runtime entry points.
- Repeated identical trace replay is bitwise deterministic for layers 0, 1,
  and 3.
- Initial aggregate watcher discovered a real QSA tiled-index embedding fault:
  a 4096-byte NOC read into a 512-byte scratch CB. `$autotriage` produced the
  exact source/CB ledger in `AUTOTRIAGE.md`; the failure is preserved in
  `watcher_embedding_failure_20260826_1244/`.
- AutoFix converted every QSA embedding index to row-major layout on device
  before embedding. Focused real QSA watcher rerun passed at PCC
  0.99679226/0.99988198 with no fault:
  `watcher_qsa_fix_20260826_1251/pytest.log`.
- Final post-top-k watcher command, run without the profiler:

  ```text
  TT_METAL_WATCHER=10 \
  TT_METAL_LOGS_PATH=models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/watcher_final_20260826_1419 \
  pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder.py \
    --tb=short --durations=10
  ```

  Evidence: `watcher_final_20260826_1419/pytest.log` (`34 passed`, 7
  intentional long-context skips) and
  `watcher_final_20260826_1419/generated/watcher/watcher.log`. The final log
  audit found no fatal watcher exception, NOC/CB/L1/stack sanitizer message, or
  `embedding_ind_tilized` kernel.

## 2026-08-26: warmed performance and Tracy

- Normal warmed run:

  ```text
  pytest -q -s models/autoports/qwen_qwen3_8_flash_next/tests/test_functional_decoder_perf.py
  ```

  `perf_host_timing.log` records one warmed 128-token prefill and ten warmed
  trace replays per layer.
- Tracy was collected in separate prefill and decode commands for each layer.
  Decode used `TT_METAL_PROFILER_MID_RUN_DUMP=1`,
  `QWEN38_PERF_FLUSH_PROFILER=1`, `QWEN38_PERF_MODE=decode`, and one replay so
  setup/capture did not overflow the finite profiler buffers. Prefill used
  `QWEN38_PERF_MODE=prefill`.
- Each newest generated `ops_perf_results_*.csv` was copied to
  `tracy/<layer>/<mode>_ops.csv`. Reports were rendered with:

  ```text
  tt-perf-report <mode>_ops.csv \
    --start-signpost PERF_<MODE>_L<LAYER> \
    --end-signpost PERF_<MODE>_L<LAYER>_END \
    --csv <mode>_perf_report.csv --no-advice \
    > <mode>_perf_report.console.log

  tt-perf-report <mode>_ops.csv \
    --start-signpost PERF_<MODE>_L<LAYER> \
    --end-signpost PERF_<MODE>_L<LAYER>_END \
    --no-summary --no-advice \
    > <mode>_perf_report.txt
  ```

- Human-readable reports record 0 host ops. Current 128-token prefill / traced
  decode host timings in milliseconds are layer 0 `24.126221 / 5.729530`,
  layer 1 `27.602558 / 6.700191`, and layer 3
  `408.469450 / 11.338447`. Filtered device-op sums in microseconds are
  `23790 / 5396`, `27269 / 5922`, and `408094 / 11081` respectively. QSA
  provenance and reports were recollected after the final selection-mask fix.

## Finalization

- `static_checks.log` records Black formatting (`9 files would be left
  unchanged`), compileall success, valid context JSON, the single guarded
  `ttnn.embedding` site, and setup-only conversion locations.
- The first independent `$stage-review` returned `more-work-needed` for full
  public QSA context, exact underfilled top-k semantics, and device-executed
  near-max non-aligned coverage. Those findings are preserved in
  `stage_review.md` and were remediated as recorded above. Final rereview is
  preserved in `stage_review_final.md` with verdict `clean-pass` and no Required
  Work.
- Stage-owned implementation/evidence checkpoint: tt-metal branch
  `hous/qwen3.8-flash-next`, commit
  `f617ad53eb3d058cf4459860654c2eb1873d3d90`. The repo commit hook passed after
  applying its import/unused-code and generated-report whitespace formatting.
  The six original Tracy `*_ops.csv` files remain as ignored repo-local
  provenance because they exceed the repository's 500 KiB per-file commit
  policy; the signpost-filtered perf CSVs, human tables, and exact Tracy pytest
  provenance logs are committed. This work-log update is committed separately
  and its SHA is reported at handoff. No push is authorized or planned.
