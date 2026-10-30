#!/usr/bin/env bash
# Round 4: c9 robustness (book32 on chip 0, resume check on chip 1), then serialized TF of c9 and its LM-head reader variants on chip 2.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
ST=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/round4.status
bash models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/sweep_chain.sh 0 bringup/references/qwen3_8b_book32.refpt book32 c9_lm_head_bfp4 &
P0=$!
( echo "START resume_c9 chip1 $(date -u +%H:%M:%S)" >> $ST; timeout 3600 models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh 1 env QWEN3_PRECISION_CONFIG=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/configs/c9_lm_head_bfp4.json /home/hous/dev/tt-metal/python_env/bin/python -m readiness_check.run_prefill_resume_check --model-dir models/autoports/qwen_qwen3_8b --mesh-device N150 --lengths 1500,2560,5000,16384 --block-sizes 32 --output models/autoports/qwen_qwen3_8b/readiness/prefill_resume_check_stage8_c9_lm_head_bfp4_default_thresholds.json > models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_check_c9_lm_head_bfp4.log 2>&1; rc=$?; echo "exit=$rc" >> models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/resume_check_c9_lm_head_bfp4.log; echo "END resume_c9 rc=$rc $(date -u +%H:%M:%S)" >> $ST ) &
P1=$!
wait $P0 $P1
echo "ACCURACY_DONE $(date -u +%H:%M:%S)" >> $ST
until ! fuser /dev/tenstorrent/0 >/dev/null 2>&1 && ! fuser /dev/tenstorrent/1 >/dev/null 2>&1; do sleep 5; done
bash models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs/serial_tf_chip2.sh c9_lm_head_bfp4 c18_lm_head_bfp4_readers2 c19_lm_head_bfp4_readers4
echo "ROUND4_DONE $(date -u +%H:%M:%S)" >> $ST
