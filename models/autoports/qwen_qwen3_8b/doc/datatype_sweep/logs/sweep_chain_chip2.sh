#!/usr/bin/env bash
# Stage-8 sweep chain on chip 2: for each candidate config, run_prefill_check then run_teacher_forcing under their own timeouts.
# Usage: sweep_chain_chip2.sh <config id> [<config id> ...]
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/datatype_sweep
L=$D/logs
REF=bringup/references/qwen3_8b_aime24_chat.refpt
STATUS=$L/sweep_chain_chip2.status
run() {
  name=$1; shift
  started=$(date -u +%H:%M:%S)
  echo "START $name $started" | tee -a "$STATUS"
  "$@" > "$L/$name.log" 2>&1
  rc=$?
  ended=$(date -u +%H:%M:%S)
  echo "exit=$rc" >> "$L/$name.log"
  echo "END $name rc=$rc $started-$ended" | tee -a "$STATUS"
  LAST_RC=$rc; LAST_STARTED=$started; LAST_ENDED=$ended
}
for cid in "$@"; do
  CFG=$D/configs/$cid.json
  SIDECAR=$D/readiness_${cid}_chip2.json
  run ${cid}_prefill_check timeout 1500 $S 2 env QWEN3_PRECISION_CONFIG=$CFG $PY -m readiness_check.run_prefill_check --model-dir $M --reference $REF --mesh-device N150
  python3 $L/write_sidecar.py "$SIDECAR" run_prefill_check "$L/${cid}_prefill_check.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" "$cid" "$CFG" | tee -a "$STATUS"
  run ${cid}_teacher_forcing timeout 1500 $S 2 env QWEN3_PRECISION_CONFIG=$CFG $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $REF --mesh-device N150
  python3 $L/write_sidecar.py "$SIDECAR" run_teacher_forcing "$L/${cid}_teacher_forcing.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" "$cid" "$CFG" | tee -a "$STATUS"
done
echo "CHAIN_DONE $(date -u +%H:%M:%S) $*" | tee -a "$STATUS"
