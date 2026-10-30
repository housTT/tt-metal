#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/datatype_sweep
L=$D/logs
R=$M/readiness
REF=bringup/references/qwen3_8b_aime24_chat.refpt
STATUS=$L/post_selection_chip2.status
SIDECAR=$D/readiness_selected_default_chip2.json
CID=$(python3 -c "import json; print(json.load(open('$D/selected_precision_config.json'))['config_id'])")
echo "SELECTED $CID $(date -u +%H:%M:%S)" | tee -a "$STATUS"
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
sidecar() { python3 $L/write_sidecar.py "$SIDECAR" "$1" "$L/$1.log" "$LAST_RC" "$LAST_STARTED" "$LAST_ENDED" "$CID" "$D/selected_precision_config.json (default path, no QWEN3_PRECISION_CONFIG)" 2 "$2" "serialized: the only device job on the host during this run" | tee -a "$STATUS"; }

run selected_prefill_check timeout 1500 $S 2 $PY -m readiness_check.run_prefill_check --model-dir $M --reference $REF --mesh-device N150
sidecar selected_prefill_check $REF
run selected_teacher_forcing timeout 1500 $S 2 $PY -m readiness_check.run_teacher_forcing --model-dir $M --reference $REF --mesh-device N150
sidecar selected_teacher_forcing $REF
run selected_pytest_full_model_36 timeout 1800 $S 2 env QWEN3_FULL_MODEL_LAYERS=36 QWEN3_EVIDENCE_SUFFIX=stage8 $PY -m pytest -q -s $M/tests/test_full_model.py -p no:cacheprovider
run selected_propagation_36 timeout 1200 $S 2 env PYTHONUNBUFFERED=1 $PY $L/propagation_check.py 36 $D/propagation_check_selected_36.json $D/selected_precision_config.json
run selected_qualitative_tt timeout 1500 $S 2 env PYTHONUNBUFFERED=1 $PY $L/qualitative_tt.py 128
run selected_prefill_resume_check timeout 3600 $S 2 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500,2560,5000,16384 --block-sizes 32 --output $R/prefill_resume_check_stage8_default_thresholds.json
sidecar selected_prefill_resume_check "book text (readiness runner, natural text)"
run selected_batch32_same_input timeout 1800 $S 2 env PYTHONUNBUFFERED=1 $PY $L/batch32_same_input.py bringup/references/qwen3_8b_book32.refpt $D/batch32_same_input_selected_chip2.json 100
run selected_profile_reduced timeout 1500 $S 2 env PYTHONUNBUFFERED=1 $PY -m tracy -r -p -v --check-exit-code -o generated/profiler/qwen3_stage8_reduced $L/profile_reduced.py
bash $L/perf_report.sh > $L/perf_report.log 2>&1
echo "POST_DONE $(date -u +%H:%M:%S)" | tee -a "$STATUS"
