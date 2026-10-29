#!/usr/bin/env bash
# tt-perf-report tables for the reduced-variant profile: PERF_PREFILL_128 and PERF_DECODE windows.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
PY=/home/hous/dev/tt-metal/python_env/bin/python
D=models/autoports/qwen_qwen3_8b/doc/optimized_full_model/profile
mkdir -p "$D"
SRC=$(ls -t generated/profiler/qwen3_stage7_reduced/reports/*/ops_perf_results*.csv | head -1)
echo "$SRC" > "$D/ops_perf_results.source.txt"
cp "$SRC" "$D/ops_perf_results.csv"
for win in "PERF_PREFILL_128:prefill_128" "PERF_DECODE:decode"; do
  sp=${win%%:*}; name=${win##*:}
  $PY -m tt_perf_report.perf_report "$D/ops_perf_results.csv" --start-signpost "$sp" --end-signpost "${sp}_END" --csv "$D/$name.csv" > "$D/$name.console.log" 2>&1
  $PY -m tt_perf_report.perf_report "$D/ops_perf_results.csv" --start-signpost "$sp" --end-signpost "${sp}_END" --no-summary > "$D/$name.txt" 2>&1
done
rm -f "$D/ops_perf_results.csv"
ls -la "$D"
