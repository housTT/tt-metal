#!/usr/bin/env bash
# Rebuild every committed artifact of the fused-decoder stage, in the only order that is correct.
#
#   bash models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/probes/regenerate_evidence.sh
#
# Order matters and is the reason this is a script rather than a list in the work log:
#
#   1. the test suites, whose logs carry the PCC records;
#   2. the model-free probes, whose logs back every rejected and chosen option;
#   3. the Tracy profiling runs, both implementations, every measured window;
#   4. perf_summary.json  <- reads (3);
#   5. WATCHER_AUDIT.md   <- reads the watcher run's log from (1);
#   6. pcc_evidence.json  <- reads (1);
#   7. the generated document blocks <- read (2), (4) and (6), so they come last;
#   8. the document gate, which re-derives all of it.
#
# A run takes about an hour on one Blackhole chip.  Live output:
#   tail -f models/autoports/qwen_qwen3_6_27b/doc/fused_decoder/logs/suite_main.log
set -x
cd /home/ttuser/dev/qwen/tt-metal
source models/autoports/qwen_qwen3_6_27b/doc/functional_decoder/ttenv.sh > /dev/null
ART=models/autoports/qwen_qwen3_6_27b/doc/fused_decoder

timeout 3600 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -v -s > $ART/logs/suite_main.log 2>&1
grep -E "^=+.*(passed|failed)" $ART/logs/suite_main.log
timeout 3600 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py -k test_full_advertised_context --long-context -v -s > $ART/logs/long_context.log 2>&1
grep -E "^=+.*(passed|failed)" $ART/logs/long_context.log
(
  export TT_METAL_LOGS_PATH=$PWD/$ART/watcher TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=0 \
         TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1
  timeout 3000 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py \
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable" \
    -v -s > $ART/logs/watcher_run.log 2>&1
)
grep -E "^=+.*(passed|failed)" $ART/logs/watcher_run.log

for p in probe_chunk_gdr probe_small_ops probe_causal_conv probe_decode_heads probe_decode_recurrence \
         probe_mlp_variants probe_output_paths probe_gated_norm_batch probe_qkv_gate_pack \
         probe_matmul_bound probe_gdn_input_folds; do
  timeout 2400 python $ART/probes/$p.py > $ART/logs/$p.log 2>&1 || echo "PROBE FAILED $p"
done

for impl in fused functional; do for kind in linear_attention full_attention; do
  for phase in prefill decode decode_batch32; do
    timeout 3000 $ART/probes/run_perf.sh $kind $phase $impl 2>&1 | grep -E "^PERF |no ops"
  done
done; done

python $ART/probes/make_perf_summary.py 2>&1 | grep -v nanobind | tail -8
rm -f $ART/watcher/generated/watcher/*.gz
gzip -9 $ART/watcher/generated/watcher/kernel_names.txt
gzip -9 $ART/watcher/generated/watcher/watcher.log
rm -rf $ART/watcher/generated/inspector
python $ART/probes/make_watcher_audit.py 2>&1 | grep -v nanobind | tail -2
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
  $ART/logs/suite_main.log $ART/logs/long_context.log $ART/logs/watcher_run.log \
  --out $ART/pcc_evidence.json 2>&1 | grep "^wrote"
python $ART/probes/make_doc_tables.py 2>&1 | grep -v nanobind | tail -1
timeout 600 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder_docs.py -v > $ART/logs/doc_gate.log 2>&1
grep -E "^=+.*(passed|failed)" $ART/logs/doc_gate.log
