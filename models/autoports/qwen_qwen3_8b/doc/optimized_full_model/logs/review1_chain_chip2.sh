#!/usr/bin/env bash
# Review-1 fixes: full evidence chain, margin probe and gate chain on the uniform-sharded-norm + 1D-kernel code.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
M=models/autoports/qwen_qwen3_8b
L=$M/doc/optimized_full_model/logs
S=$M/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
mkdir -p $L/round1; cp $L/evidence_chain_chip2.status $L/gate_chain_chip2.status $L/pytest_full_model_36.log $L/run_prefill_check.log $L/run_teacher_forcing.log $L/ttft_sweep_after.log $L/run_prefill_resume_check_default.log $L/run_prefill_resume_check_lowered.log $L/qualitative_tt.log $L/run_autoregressive.log $L/pytest_full_model_36_tracked.log $L/pytest_watcher_reduced.log $L/profile_reduced.log $L/resume_margin_probe_final.log $L/round1/ 2>/dev/null
cp $M/doc/optimized_full_model/ttft_sweep_after.json $M/doc/optimized_full_model/ttft_sweep_after_round1.json
cp $M/doc/full_model/perf_batch1_128_128.json $M/doc/optimized_full_model/perf_batch1_128_128_round1.json
cp $M/readiness/prefill_resume_check_default_thresholds.json $M/readiness/prefill_resume_check_default_thresholds_round1.json
bash $L/evidence_chain_chip2.sh > $L/evidence_chain_chip2.console.log 2>&1
started=$(date -u +%H:%M:%S); timeout 1800 $S 2 env PYTHONUNBUFFERED=1 $PY $L/resume_margin_probe.py 1500:68 1500:32 2560:2528 > $L/resume_margin_probe_final.log 2>&1; rc=$?; echo "exit=$rc" >> $L/resume_margin_probe_final.log; echo "END resume_margin_probe_final rc=$rc $started-$(date -u +%H:%M:%S)" >> $L/evidence_chain_chip2.status
bash $L/gate_chain_chip2.sh > $L/gate_chain_chip2.console.log 2>&1
echo "FINAL_DONE $(date -u +%H:%M:%S)" >> $L/evidence_chain_chip2.status
