#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
ST=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/review1_device.status
echo "START propagation36_chip2 $(date -u +%H:%M:%S)" >> $ST
timeout 1200 models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh 2 env PYTHONUNBUFFERED=1 /home/hous/dev/tt-metal/python_env/bin/python models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/propagation_check.py 36 models/autoports/qwen_qwen3_8b/doc/datatype_sweep/propagation_check_selected_36.json models/autoports/qwen_qwen3_8b/doc/datatype_sweep/selected_precision_config.json > models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/selected_propagation_36.log 2>&1; rc=$?; echo "exit=$rc" >> models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/selected_propagation_36.log
echo "END propagation36_chip2 rc=$rc $(date -u +%H:%M:%S)" >> $ST
