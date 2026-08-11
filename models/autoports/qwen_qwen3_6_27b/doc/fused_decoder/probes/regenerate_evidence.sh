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
    -k "test_traced_decode_pcc or (test_decode_pcc and 2049) or test_bfloat8_kv_cache or test_traced_decode_batched or test_alternate_page_block_size or test_repeated_runs_stable or (test_batched_users and 32) or (test_merged_unaries_are_not_dispatched and 32) or test_conv_state_after_decode_matches_reference" \
    -v -s > $ART/logs/watcher_run.log 2>&1
)
grep -E "^=+.*(passed|failed)" $ART/logs/watcher_run.log

# Each probe log carries the build stamp too: three shipped constants and every rejection are
# certified from these logs, and a review pointed out they were the one artifact class with no tie
# to the source.  ``test_every_run_was_made_against_the_shipped_build`` checks them.
for p in $ART/probes/probe_*.py; do
  p=$(basename "$p" .py)
  timeout 2400 python $ART/probes/$p.py > $ART/logs/$p.log 2>&1 || echo "PROBE FAILED $p"
  python -m models.autoports.qwen_qwen3_6_27b.tt.build_fingerprint >> $ART/logs/$p.log
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
# Also over the repository's 500 KB file limit once the watcher run covers batch 32.
gzip -9 $ART/watcher/generated/watcher/kernel_elf_paths.txt
rm -rf $ART/watcher/generated/inspector
python $ART/probes/make_watcher_audit.py 2>&1 | grep -v nanobind | tail -2
python -m models.autoports.qwen_qwen3_6_27b.scripts.collect_evidence \
  $ART/logs/suite_main.log $ART/logs/long_context.log $ART/logs/watcher_run.log \
  --out $ART/pcc_evidence.json 2>&1 | grep "^wrote"
python $ART/probes/make_doc_tables.py 2>&1 | grep -v nanobind | tail -1
# The repository ignores *.log and *.csv, so every artifact this stage commits has to be
# force-added.  Doing it here rather than by hand is what keeps a fresh clone able to run the
# generators; ``test_every_artifact_the_gate_reads_is_tracked_by_git`` is the gate for it.
# The raw *_ops.csv are megabytes each and are committed only gzipped, so they are excluded here.
git add -f $ART/logs/*.log $ART/tracy/*/*/*_perf_report*.csv $ART/tracy/*/*/*.console.log \
           $ART/tracy/*/*/*.provenance $ART/tracy/*/*/*.gz \
           $ART/watcher/WATCHER_AUDIT.md $ART/watcher/generated/watcher/*.gz \
           $ART/tracy/rejected/*/*

# The doc gate reads its own committed log (::test_doc_gate_log_is_of_the_shipped_gates), so the
# new log is written to a scratch path and moved in only once the run is over - a run that read a
# half-written log would see no summary line and fail on itself.  When a *new* gate has just been
# added the first run legitimately fails on the previous log; run this script again and it
# converges, which is what the loop below does.
for attempt in 1 2; do
    python -m models.autoports.qwen_qwen3_6_27b.tt.build_fingerprint > /tmp/doc_gate.log 2>&1
    timeout 600 python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder_docs.py -v \
        >> /tmp/doc_gate.log 2>&1
    status=$?
    mv /tmp/doc_gate.log $ART/logs/doc_gate.log
    [ $status -eq 0 ] && break
    echo "doc gate attempt $attempt failed; retrying against the log it just wrote"
done
grep -E "^=+.*(passed|failed)" $ART/logs/doc_gate.log
git add -f $ART/logs/doc_gate.log
