#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
CHIP=$1; shift
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
D=models/autoports/qwen_qwen3_8b/doc/datatype_sweep
L=$D/logs
ST=$L/natural_queue2.status
for cid in "$@"; do
  echo "START natural_${cid} chip$CHIP $(date -u +%H:%M:%S)" >> $ST
  timeout 1800 $S $CHIP env PYTHONUNBUFFERED=1 QWEN3_PRECISION_CONFIG=$D/configs/$cid.json $PY $L/natural_text_check.py $D/natural_text_${cid}_chip$CHIP.json 1500 32 > $L/natural_text_$cid.log 2>&1; rc=$?; echo "exit=$rc" >> $L/natural_text_$cid.log
  echo "END natural_${cid} rc=$rc $(date -u +%H:%M:%S)" >> $ST
done
echo "NATURAL2_DONE chip $CHIP $(date -u +%H:%M:%S)" >> $ST
