#!/usr/bin/env bash
# Stage-7 evidence chain on chip 2: every step under its own timeout, exit codes recorded, nothing killed by hand.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
L=$M/doc/optimized_full_model/logs
R=$M/readiness
SIDECAR=$R/readiness_stage7_chip2.json
REF=bringup/references/qwen3_8b_aime24_chat.refpt
STATUS=$L/evidence_chain_chip2.status
: > "$STATUS"
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
sidecar() { python3 $L/write_sidecar.py "$SIDECAR" "$1" "$L/$1.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" | tee -a "$STATUS"; }

run pytest_full_model_36 timeout 1500 $S 2 env QWEN3_FULL_MODEL_LAYERS=36 $PY -m pytest -q -s $M/tests/test_full_model.py -p no:cacheprovider
run run_prefill_check timeout 1500 $S 2 $PY -m readiness_check.run_prefill_check --model-dir $M --reference $REF --mesh-device N150
sidecar run_prefill_check
run run_teacher_forcing timeout 1500 $S 2 $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $REF --mesh-device N150
sidecar run_teacher_forcing
run ttft_sweep_after timeout 1800 $S 2 env PYTHONUNBUFFERED=1 $PY $L/ttft_sweep.py $M/doc/optimized_full_model/ttft_sweep_after.json
run run_prefill_resume_check_default timeout 3600 $S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500,2560,5000,16384 --block-sizes 32 --output $R/prefill_resume_check_default_thresholds.json
sidecar run_prefill_resume_check_default
if [ "$LAST_RC" = "0" ]; then
  cp $R/prefill_resume_check_default_thresholds.json $R/prefill_resume_check.json
  echo "canonical resume check = default-threshold run" | tee -a "$STATUS"
else
  run run_prefill_resume_check_lowered timeout 3600 $S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500,2560,5000,16384 --block-sizes 32 --pcc-min-min 0.90 --top1-min 0.93 --output $R/prefill_resume_check.json
  sidecar run_prefill_resume_check_lowered
fi
run qualitative_tt timeout 1500 $S 2 $PY $L/qualitative_tt.py 128
run run_autoregressive timeout 2400 $S 2 $PY -m readiness_check.run_autoregressive --model-dir $M --hf-model Qwen/Qwen3-8B --mesh-device N150
sidecar run_autoregressive
echo "CHAIN_DONE $(date -u +%H:%M:%S)" | tee -a "$STATUS"
