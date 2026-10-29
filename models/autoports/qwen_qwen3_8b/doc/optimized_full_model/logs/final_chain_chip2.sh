#!/usr/bin/env bash
# Final-code evidence: waits for the first gate chain, then evidence chain, margin probe, gate chain.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
M=models/autoports/qwen_qwen3_8b
L=$M/doc/optimized_full_model/logs
S=$M/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
until grep -q ALL_DONE $L/attribution_chain_chip2.status 2>/dev/null; do sleep 10; done
mkdir -p $L/variant_sharded_norm/gate; cp $L/gate_chain_chip2.status $L/pytest_full_model_36_tracked.log $L/positive_control.log $L/pytest_watcher_reduced.log $L/profile_reduced.log $L/variant_sharded_norm/gate/ 2>/dev/null
bash $L/evidence_chain_chip2.sh > $L/evidence_chain_chip2.console.log 2>&1
started=$(date -u +%H:%M:%S); timeout 1800 $S 2 env PYTHONUNBUFFERED=1 $PY $L/resume_margin_probe.py 1500:68 1500:32 2560:2528 > $L/resume_margin_probe_final.log 2>&1; rc=$?; echo "exit=$rc" >> $L/resume_margin_probe_final.log; echo "END resume_margin_probe_final rc=$rc $started-$(date -u +%H:%M:%S)" >> $L/evidence_chain_chip2.status
bash $L/gate_chain_chip2.sh > $L/gate_chain_chip2.console.log 2>&1
echo "FINAL_DONE $(date -u +%H:%M:%S)" >> $L/evidence_chain_chip2.status
