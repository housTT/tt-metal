#!/usr/bin/env bash
# Stage-8 readiness pair on any chip: run_prefill_check then run_teacher_forcing per candidate config against a given reference.
# Usage: sweep_chain.sh <chip> <reference.refpt> <tag> <config id> [<config id> ...]
# Logs: logs/<id>_<tag>_{prefill_check,teacher_forcing}.log; sidecar readiness_<id>_<tag>_chip<chip>.json.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
CHIP=$1; REF=$2; TAG=$3; shift 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/datatype_sweep
L=$D/logs
STATUS=$L/sweep_chain_${TAG}_chip${CHIP}.status
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
  SIDECAR=$D/readiness_${cid}_${TAG}_chip${CHIP}.json
  run ${cid}_${TAG}_prefill_check timeout 1500 $S $CHIP env QWEN3_PRECISION_CONFIG=$CFG $PY -m readiness_check.run_prefill_check --model-dir $M --reference $REF --mesh-device N150
  python3 $L/write_sidecar.py "$SIDECAR" run_prefill_check "$L/${cid}_${TAG}_prefill_check.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" "$cid" "$CFG" "$CHIP" "$REF" "concurrent: other chips may run device jobs at the same time; host-wall timing is not same-regime" | tee -a "$STATUS"
  run ${cid}_${TAG}_teacher_forcing timeout 1500 $S $CHIP env QWEN3_PRECISION_CONFIG=$CFG $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $REF --mesh-device N150
  python3 $L/write_sidecar.py "$SIDECAR" run_teacher_forcing "$L/${cid}_${TAG}_teacher_forcing.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" "$cid" "$CFG" "$CHIP" "$REF" "concurrent: other chips may run device jobs at the same time; host-wall timing is not same-regime" | tee -a "$STATUS"
done
echo "CHAIN_DONE $(date -u +%H:%M:%S) chip $CHIP $TAG $*" | tee -a "$STATUS"
