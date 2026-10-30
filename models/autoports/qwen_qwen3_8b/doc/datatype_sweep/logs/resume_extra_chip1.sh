#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
ST=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_extra.status
for cid in c5_mlp_bfp4_inner c15_mlp_bfp4_inner_lm_bfp4; do
  echo "START resume_$cid chip1 $(date -u +%H:%M:%S)" >> $ST
  timeout 3600 models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh 1 env QWEN3_PRECISION_CONFIG=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/configs/$cid.json /home/hous/dev/tt-metal/python_env/bin/python -m readiness_check.run_prefill_resume_check --model-dir models/autoports/qwen_qwen3_8b --mesh-device N150 --lengths 1500,2560,5000,16384 --block-sizes 32 --output models/autoports/qwen_qwen3_8b/readiness/prefill_resume_check_stage8_${cid}_default_thresholds.json > models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_check_$cid.log 2>&1; rc=$?; echo "exit=$rc" >> models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_check_$cid.log
  echo "END resume_$cid rc=$rc $(date -u +%H:%M:%S)" >> $ST
done
echo "EXTRA_CHIP1_DONE $(date -u +%H:%M:%S)" >> $ST
