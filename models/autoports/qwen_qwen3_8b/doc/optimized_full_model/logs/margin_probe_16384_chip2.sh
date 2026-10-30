#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
M=models/autoports/qwen_qwen3_8b
L=$M/doc/optimized_full_model/logs
S=$M/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
until grep -q FINAL_DONE $L/evidence_chain_chip2.status 2>/dev/null; do sleep 10; done
started=$(date -u +%H:%M:%S); timeout 2400 $S 2 env PYTHONUNBUFFERED=1 $PY $L/resume_margin_probe.py 16384:16352 > $L/resume_margin_probe_16384.log 2>&1; rc=$?; echo "exit=$rc" >> $L/resume_margin_probe_16384.log
echo "END resume_margin_probe_16384 rc=$rc $started-$(date -u +%H:%M:%S)" >> $L/evidence_chain_chip2.status
echo "MARGIN_16384_DONE" >> $L/evidence_chain_chip2.status
