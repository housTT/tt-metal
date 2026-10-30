#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/datatype_sweep
L=$D/logs
REF=bringup/references/qwen3_8b_aime24_chat.refpt
STATUS=$L/serial_tf_chip2.status
for cid in "$@"; do
  CFG=$D/configs/$cid.json
  started=$(date -u +%H:%M:%S)
  echo "START ${cid}_serial_teacher_forcing $started" | tee -a "$STATUS"
  timeout 1500 $S 2 env QWEN3_PRECISION_CONFIG=$CFG $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $REF --mesh-device N150 > $L/${cid}_serial_teacher_forcing.log 2>&1
  rc=$?; ended=$(date -u +%H:%M:%S)
  echo "exit=$rc" >> $L/${cid}_serial_teacher_forcing.log
  echo "END ${cid}_serial_teacher_forcing rc=$rc $started-$ended" | tee -a "$STATUS"
  python3 $L/write_sidecar.py $D/readiness_${cid}_serial_chip2.json run_teacher_forcing $L/${cid}_serial_teacher_forcing.log $rc $started $ended $cid $CFG 2 $REF "serialized: the only device job on the host during this run (fuser checked on chips 0, 1 and 2 before the chain)" | tee -a "$STATUS"
done
echo "SERIAL_DONE $(date -u +%H:%M:%S) $*" | tee -a "$STATUS"
