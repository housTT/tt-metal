#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
L=models/autoports/qwen_qwen3_8b/doc/full_model/logs
PY=/home/hous/dev/tt-metal/python_env/bin/python
T=models/autoports/qwen_qwen3_8b/tests/test_full_model.py
M=models/autoports/qwen_qwen3_8b
R=bringup/references/qwen3_8b_aime24_chat.refpt
F=$L/rerun_chain_chip2.status
step() { echo "=== $1 start $(date -u +%H:%M:%S)" >> $F; }
done_step() { echo "=== $1 exit=$2 end $(date -u +%H:%M:%S)" >> $F; }
while pgrep -f "readiness_check.run_prefill_check" > /dev/null; do sleep 10; done
echo "=== prefill_check (first chain) finished $(date -u +%H:%M:%S)" >> $F
step teacher_forcing; $S 2 timeout 2400 env PYTHONUNBUFFERED=1 $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $R --mesh-device N150 > $L/run_teacher_forcing_rerun.log 2>&1; done_step teacher_forcing $?
step tracked_batch32; $S 2 timeout 3000 env PYTHONUNBUFFERED=1 QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=tracked TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 $PY -m pytest -q -s $T -k "test_batch_decode and 32" --timeout 1800 -p no:cacheprovider > $L/pytest_full_model_36_tracked_batch32_rerun.log 2>&1; done_step tracked_batch32 $?
step probe; $S 2 timeout 1200 env PYTHONUNBUFFERED=1 $PY $L/capacity_probe.py > $L/capacity_probe_chip2.log 2>&1; rc=$?; done_step probe $rc
if [ "$rc" = "0" ]; then
  step bisect; $S 2 timeout 1200 env PYTHONUNBUFFERED=1 $PY $L/capacity_bisect.py > $L/capacity_bisect_chip2.log 2>&1; done_step bisect $?
else
  echo "=== bisect skipped: probe exit $rc; device work on chip 2 stops" >> $F
fi
echo "=== chain2 done $(date -u +%H:%M:%S)" >> $F
