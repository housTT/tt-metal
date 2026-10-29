#!/usr/bin/env bash
# Waits for the evidence chain, then: margin probes, two resume-check controls at L=1500, the debug-gate chain.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
L=$M/doc/optimized_full_model/logs
R=$M/readiness
STATUS=$L/attribution_chain_chip2.status
until grep -q CHAIN_DONE $L/evidence_chain_chip2.status 2>/dev/null; do sleep 10; done
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
run resume_margin_probe_1500 timeout 1800 $S 2 env PYTHONUNBUFFERED=1 $PY $L/resume_margin_probe.py 1500:68 1500:32
run resume_control_stage3_kernels timeout 1500 $S 2 env QWEN3_PREFILL_KERNEL_TABLE=stage3 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500 --block-sizes 32 --output $R/prefill_resume_check_L1500_control_stage3_kernels.json
run resume_control_norm_off timeout 1500 $S 2 env QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS=0 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500 --block-sizes 32 --output $R/prefill_resume_check_L1500_control_norm_off.json
run resume_control_both_off timeout 1500 $S 2 env QWEN3_PREFILL_KERNEL_TABLE=stage3 QWEN3_PREFILL_SHARDED_NORM_MAX_ROWS=0 $PY -m readiness_check.run_prefill_resume_check --model-dir $M --mesh-device N150 --lengths 1500 --block-sizes 32 --output $R/prefill_resume_check_L1500_control_both_off.json
echo "ATTRIBUTION_DONE $(date -u +%H:%M:%S)" | tee -a "$STATUS"
bash $L/gate_chain_chip2.sh >> "$STATUS" 2>&1
echo "ALL_DONE $(date -u +%H:%M:%S)" | tee -a "$STATUS"
