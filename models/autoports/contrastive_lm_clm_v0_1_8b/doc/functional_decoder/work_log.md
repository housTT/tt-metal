# Work log: stages 1 to 3 (functional, fused, optimized decoder layer)

All times UTC, 2026 Oct 1. Host qb2-120-p11t01, chip 0 of 4 unless stated. Every device command ran through
`/home/hous/dev/clm-v0.1-8B/bin/devlock`; `source /home/hous/dev/clm-v0.1-8B/bin/ttenv.sh` provides the environment
(`TT_METAL_HOME`, `PYTHONPATH`, `ARCH_NAME=blackhole`, `HF_MODEL=Qwen/Qwen3-8B`, `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`).

- 20:56 to 21:46: first full-encoder probe hangs; root cause and workaround in `../probe/README.md` (tensor-cache
  file pinning). Stock controls: `models/tt_transformers/tests/test_decoder_prefill.py -k "128 and paged_attention"`
  passed on a 1x4 mesh (PCC 0.99998) and, after the workaround, on a 1x1 mesh (PCC 0.99998).
- 21:49: `tests/test_functional_decoder.py` v1 failed: `Seqlen must be divisible by 128` (attention.py:1043). Inputs
  were padded to 32; fixed to pad to the prefill buckets (128 / 1024 / 2048) with per-bucket rotation matrices.
- 21:53: v2 failed on a test bug (`comp_pcc` returns a float in this tree; the parser expected a string). Fixed.
- 21:59: v3 passed, 33 rows, 26.4 s: `pytest models/autoports/contrastive_lm_clm_v0_1_8b/tests/test_functional_decoder.py -x -q`
  (log `/home/hous/dev/clm-v0.1-8B/logs/stage1_functional_decoder_v3.log`). Watcher run with `TT_METAL_WATCHER=10`
  passed, 38.9 s (`stage1_functional_decoder_watcher_v3.log`); `generated/watcher/watcher.log` has no errors.
- 22:01: Tracy profile of layer 0 at 128 (users 0 to 3) and 1024 tokens:
  `CLM_TEST_LENGTHS=128,1024 CLM_TEST_LAYERS=0 CLM_TEST_OUT=layer_pcc_profiled_run.json python -m tracy -r -p -v -o doc/functional_decoder/tracy/layer0 -m pytest tests/test_functional_decoder.py -x -q`
  then `tt-perf-report <ops csv> --csv prefill_perf_report.csv`. Raw `.tracy` and `reports/` are not committed (size);
  the perf report CSVs and console log are force-added.
- 22:08 and 22:10: `tests/eager_vs_trace.py --seq-len 128` and `--seq-len 1024` (accuracy policy): eager 57.45 /
  170.21 ms, traced 57.65 / 170.45 ms. Parity; plan gate amended.
- 22:50: independent review A (`../review/review_A_stages_1_3.md`) returned more-work-needed: the stage 3 audit had
  misread the profile (matmuls on 32 cores at 128 tokens, not 64; norms and head reshapes on 4 cores) and the
  optimization pass had not acted on the leads. Docs corrected from the exact per-op rows; `tests/grid_experiment.py`
  written to test a larger prefill matmul grid via a `find_prefill_grid` override on the ModelArgs instance; results
  recorded in `../optimized_decoder/geometry_experiment.json` and README when run.
- Commits: 418c0e85bf (autoport, probe), 498325e8ed (stage 1 evidence), fe0b69f03e (two-phase warmup, docs),
  add82f9e7d (version markers, release notes).
