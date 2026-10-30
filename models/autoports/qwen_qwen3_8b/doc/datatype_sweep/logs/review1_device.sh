#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
ST=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/review1_device.status
echo "START probe_1472_chip0 $(date -u +%H:%M:%S)" >> $ST
timeout 1800 models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh 0 env PYTHONUNBUFFERED=1 /home/hous/dev/tt-metal/python_env/bin/python models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_margin_probe.py 1500:1472 16384:16352 > models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_margin_probe_c9_1472.log 2>&1; rc=$?; echo "exit=$rc" >> models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_margin_probe_c9_1472.log
echo "END probe_1472_chip0 rc=$rc $(date -u +%H:%M:%S)" >> $ST
