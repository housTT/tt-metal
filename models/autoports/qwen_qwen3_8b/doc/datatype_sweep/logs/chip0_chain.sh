#!/usr/bin/env bash
# Chip 0: short-prefix (32-token book prompt, 100 HF tokens) readiness pair for the KV candidate, the baseline and the MLP contenders, then the batch-32 same-input case.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
M=models/autoports/qwen_qwen3_8b
D=$M/doc/datatype_sweep
L=$D/logs
ST=$L/chip0_chain.status
bash $L/sweep_chain.sh 0 bringup/references/qwen3_8b_book32.refpt book32 c3_kv_bfp8 c0_stage7_baseline c5_mlp_bfp4_inner c6_mlp_bfp4_all
for cid in c3_kv_bfp8 c0_stage7_baseline; do
  echo "START batch32_$cid $(date -u +%H:%M:%S)" >> $ST
  timeout 1800 $S 0 env PYTHONUNBUFFERED=1 QWEN3_PRECISION_CONFIG=$D/configs/$cid.json $PY $L/batch32_same_input.py bringup/references/qwen3_8b_book32.refpt $D/batch32_same_input_${cid}_chip0.json 100 > $L/batch32_same_input_$cid.log 2>&1; rc=$?
  echo "exit=$rc" >> $L/batch32_same_input_$cid.log
  echo "END batch32_$cid rc=$rc $(date -u +%H:%M:%S)" >> $ST
done
echo "CHIP0_DONE $(date -u +%H:%M:%S)" >> $ST
