#!/usr/bin/env bash
# Stage-7 debug-gate chain on chip 2: trace allocation tracker, watcher (reduced model), reduced-variant profile.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/optimized_full_model
L=$D/logs
STATUS=$L/gate_chain_chip2.status
: > "$STATUS"
run() {
  name=$1; shift
  started=$(date -u +%H:%M:%S)
  echo "START $name $started" | tee -a "$STATUS"
  "$@" > "$L/$name.log" 2>&1
  rc=$?
  echo "exit=$rc" >> "$L/$name.log"
  echo "END $name rc=$rc $started-$(date -u +%H:%M:%S)" | tee -a "$STATUS"
}
run pytest_full_model_36_tracked timeout 3000 $S 2 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY -m pytest -q -s $M/tests/test_full_model.py -k 'test_split_sampling_trace or test_host_work_counters or test_tracked_replay_workload or test_batch_decode' --timeout 1800 -p no:cacheprovider
run positive_control timeout 600 $S 2 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY $M/doc/optimized_decoder/trace_alloc_tracking/positive_control.py
mkdir -p $D/watcher
run pytest_watcher_reduced timeout 1800 $S 2 env QWEN3_FULL_MODEL_LAYERS=2 QWEN3_EVIDENCE_SUFFIX=watcher TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$D/watcher $PY -m pytest -q -s $M/tests/test_full_model.py -k 'test_split_sampling_trace or test_host_work_counters or test_multi_row_prefill_equals_solo or test_tracked_replay_workload' -p no:cacheprovider
run profile_reduced timeout 1500 $S 2 env PYTHONUNBUFFERED=1 $PY -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_stage7_reduced $L/profile_reduced.py
echo "GATE_CHAIN_DONE $(date -u +%H:%M:%S)" | tee -a "$STATUS"
