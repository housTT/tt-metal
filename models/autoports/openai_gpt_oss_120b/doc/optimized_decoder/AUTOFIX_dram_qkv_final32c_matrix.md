# AutoFix: cumulative DRAM-QKV precision and geometry matrix

## Starting evidence

- Source review found that every private DRAM-sharded QKV policy left
  `output_projection_geometry=None`. The retained DRAM timings therefore used
  the older output path rather than the promoted
  `32c_ibw4_pcn3_sb3` output-projection topology.
- The historical matrix measured BFP8/HiFi2 at all five legal geometries, but
  measured BFP8/LoFi and BFP4/LoFi only at 90 cores. It did not cross BFP4 with
  HiFi2. Those runs also used random activations rather than the exact
  prompt-derived production activation contract.
- At 90 cores, BFP8/LoFi was 9.081 us faster than BFP8/HiFi2. The 10-core
  BFP8/HiFi2 row was the fastest historical DRAM geometry, so a cumulative
  10-core LoFi row could plausibly beat the production path.

## Hypothesis experiments

### H1: the retained DRAM policies do not use the final output topology

- Hypothesis: missing `output_projection_geometry` makes the old comparison
  non-cumulative and can hide a whole-layer win.
- Experiment: inspect every source-defined DRAM policy and the constructor's
  output-program selection.
- Result: verified. The named 90-core policy and the candidate factory both
  omitted the production geometry.
- Fix: every DRAM policy now selects
  `_PRODUCTION_OUTPUT_PROJECTION_GEOMETRY`. A host test asserts the exact
  BFP8/BFP4 x HiFi2/LoFi x 90/45/30/15/10 cross-product and the shared final
  geometry. BF16/HiFi2 90-core remains a separate private control.
- Verification: `optimized_autofix_dram_qkv_final32c_matrix_host_checks.log.gz`.

### H2: a lower-fidelity, smaller-core DRAM row may beat production

- Hypothesis: the historical 9.081 us LoFi benefit and 10-core geometry win
  may combine with the final output topology to beat the capacity-selected
  production policy.
- Experiment: the new opt-in matrix gate compares all 20 cumulative candidates
  at configured batch 1 and 2, for both representative layer kinds. Each node
  uses exact checkpoint prompt-derived activations, the exact HF layer oracle,
  unchanged 0.95/0.99 PCC bars, warmed prefill, first/second trace equality,
  post-1000-replay equality, and same-process latency against the applicable
  automatic baseline.
- Result: verified for configured capacities 2--32. The complete 80-node
  matrix ran after operator recovery. BFP4/LoFi 15-core was the fastest
  candidate which cleared every correctness bar: versus the applicable
  automatic baseline it measured -4.724/-4.712% at batch-1 sliding/full and
  -1.458/-2.561% at batch-2 sliding/full. Reversed order reproduced strict
  wins of -4.736/-4.727/-1.368/-2.480%; all exceed 1%, so a third-order median
  was not required.
- BFP4/LoFi and BFP4/HiFi2 10-core were rejected because batch-1 sliding
  decode PCC was 0.989157931, below the unchanged 0.99 bar. BFP8 candidates
  and the remaining BFP4/HiFi2 rows failed the all-kind latency criterion.
  BFP4/LoFi 45/30-core passed but were slower than 15-core. The exact delta
  ranges across all four operating points were:

| Precision/fidelity | 90c | 45c | 30c | 15c | 10c |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFP8/HiFi2 | +6.306..+12.191% | +3.574..+9.924% | +2.591..+9.203% | +1.634..+8.548% | +0.812..+8.681% |
| BFP8/LoFi | +5.657..+11.536% | +0.366..+7.664% | -0.696..+6.839% | -1.220..+6.532% | -1.728..+6.853% |
| BFP4/HiFi2 | +4.067..+4.791% | +1.485..+2.728% | +0.509..+2.070% | -0.505..+1.424% | PCC fail / -1.386..+1.998% |
| BFP4/LoFi | +2.692..+3.586% | -2.666..-0.049% | -3.687..-0.790% | **-4.724..-1.458%** | PCC fail / -5.592..-2.210% |

- Evidence: `optimized_autofix_dram_qkv_final32c_matrix_batch1_sliding.log.gz`,
  `..._batch1_full.log.gz`, `..._batch2_sliding.log.gz`,
  `..._batch2_full.log.gz`, and
  `optimized_autofix_dram_qkv_final32c_finalist_15c_candidate_first.log.gz`.

## Hardware recovery

The first 10-core BFP8/HiFi2 smoke failed before model construction with
`Device 0 init: failed to initialize FW`. The safe wrapper captured triage and
reset the system. A post-reset 1x1 mesh smoke failed with an ERISC heartbeat
timeout. A second bounded reset/list/mesh cycle failed identically.

No TTNN/test process or `/dev/tenstorrent/*` owner remained. Stale August 26
UMD lock files were moved, not deleted, to
`/tmp/gpt_oss_120b_umd_locks.J1hIsB`; `/tmp/tt-device.lock` was preserved.
Pair-pinned 1x1 open/close smokes with `TT_VISIBLE_DEVICES=0,1` and `2,3`
both failed before model code with the same firmware/ERISC signature.

Exact retained recovery artifacts are:

- `optimized_autofix_dram_qkv_final32c_smoke_bfp8_hifi2_10c_batch1_sliding.log.gz`;
- `optimized_autofix_dram_qkv_final32c_smoke_post_reset_tt_smi.log.gz`;
- `optimized_autofix_dram_qkv_final32c_smoke_post_reset_mesh_open.log.gz`;
- `optimized_autofix_dram_qkv_final32c_second_reset.log.gz`;
- `optimized_autofix_dram_qkv_final32c_second_reset_tt_smi.log.gz`;
- `optimized_autofix_dram_qkv_final32c_second_reset_mesh_open.log.gz`;
- `optimized_autofix_dram_qkv_final32c_pair01_mesh_open.log.gz`;
- `optimized_autofix_dram_qkv_final32c_pair23_mesh_open.log.gz`.

## Final status

Operator recovery restored all four boards; the pair-pinned `0,1` open/close
smoke passed at 17:16 UTC and the full matrix then completed. Production keeps
BFP8/HiFi2 at configured capacity 1 because the required sequence-128 sliding
gate measured DRAM15 at 0.541827 ms versus the established correct BFP8 result
of 0.535201 ms, despite the prompt-derived matrix win. Capacities 2--32 promote
BFP4/LoFi DRAM-QKV `15c_ibw6_pcn10`; batch-2 integrated latency is
0.778263/0.727294 ms sliding/full versus the former non-DRAM BFP4/LoFi
0.788169--0.788481/0.744771--0.745105 ms.

The final split passed batch 1, batch 2, batch 32, the 12-test integrated suite,
and a watcher-only batch-2 run. Exact artifacts are
`optimized_autofix_dram_qkv_final32c_final_split_batch1.log.gz`,
`..._promoted_batch2.log.gz`, `..._promoted_batch32.log.gz`,
`..._final_split_integrated.log.gz`,
`..._final_split_batch2_watcher.log.gz`, and
`..._final_split_post_tt_smi.log.gz`. The earlier blocked status and the prior
claim that DRAM QKV was completely rejected are superseded by this result.

After hardware recovery, the four selectors `batch1/sliding`, `batch1/full`,
`batch2/sliding`, and `batch2/full` ran serially. This is the exact command for
the first group; only the final `-k` expression and output artifact changed for
the remaining groups:

```bash
env \
  -u TT_METAL_SIMULATOR \
  -u TT_METAL_SIMULATOR_HOME \
  -u TT_METAL_SLOW_DISPATCH_MODE \
  -u TT_METAL_DISABLE_SFPLOADMACRO \
  -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_WATCHER \
  -u TT_METAL_WATCHER_DUMP_ALL \
  TT_VISIBLE_DEVICES=0,1 \
  TT_METAL_HOME=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_RUNTIME_ROOT=/home/ttuser/dev/gpt-oss-20b/tt-metal \
  TT_METAL_CACHE=/home/ttuser/dev/gpt-oss-20b/tt-metal/.tt_metal_cache \
  TTNN_CONFIG_OVERRIDES='{"cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache","model_cache_path":"/home/ttuser/dev/gpt-oss-20b/tt-metal/.ttnn_cache"}' \
  PYTHONPATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/ttnn \
  LD_LIBRARY_PATH=/home/ttuser/dev/gpt-oss-20b/tt-metal/build/lib \
  GPT_OSS_120B_SNAPSHOT=/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a \
  GPT_OSS_120B_TENSOR_CACHE=/tmp/gpt_oss_120b_functional_decoder_tensor_cache \
  GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX=1 \
  GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_REPEATS=1000 \
  GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_ORDER=automatic,candidate \
  PYTHONUNBUFFERED=1 \
  timeout 3600 scripts/run_safe_pytest.sh --run-all \
    models/autoports/openai_gpt_oss_120b/tests/test_optimized_decoder.py::test_optimized_exact_activation_dram_qkv_cross_product \
    -q -s -k 'batch1 and sliding' 2>&1 | gzip -c > \
    models/autoports/openai_gpt_oss_120b/doc/optimized_decoder/evidence/logs/optimized_autofix_dram_qkv_final32c_matrix_batch1_sliding.log.gz
```

The finalist was rerun with
`GPT_OSS_120B_DRAM_QKV_EXACT_MATRIX_ORDER=candidate,automatic`. A result within
1% requires at least three A/B orders and the median must strictly win for both
layer kinds before promotion; all finalist deltas exceeded 1%. Production
batch 1/2/32, watcher, and integrated gates were rerun as recorded above.
