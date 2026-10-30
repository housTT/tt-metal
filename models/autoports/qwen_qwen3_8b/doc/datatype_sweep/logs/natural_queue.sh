#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
CHIP=$1; shift
ST=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/natural_queue.status
if [ "$CHIP" = "1" ]; then until grep -q EXTRA_CHIP1_DONE models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_extra.status 2>/dev/null; do sleep 10; done; fi
for cid in "$@"; do
  echo "START natural_${cid} chip$CHIP $(date -u +%H:%M:%S)" >> $ST
  timeout 1800 models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh $CHIP env PYTHONUNBUFFERED=1 QWEN3_PRECISION_CONFIG=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/configs/$cid.json /home/hous/dev/tt-metal/python_env/bin/python models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/natural_text_check.py models/autoports/qwen_qwen3_8b/doc/datatype_sweep/natural_text_${cid}_chip$CHIP.json 1500 32 > models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/natural_text_$cid.log 2>&1; rc=$?; echo "exit=$rc" >> models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/natural_text_$cid.log
  echo "END natural_${cid} rc=$rc $(date -u +%H:%M:%S)" >> $ST
done
echo "NATURAL_DONE chip $CHIP $(date -u +%H:%M:%S)" >> $ST
