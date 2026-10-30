#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
PY=/home/hous/dev/tt-metal/python_env/bin/python
L=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs
ST=$L/round2_chip2.status
echo "START refs $(date -u +%H:%M:%S)" >> $ST
timeout 1800 $S 2 $PY $L/generate_reference_wrapper.py --hf-model Qwen/Qwen3-8B --prompt-source aime24 --aime24-prompt-index 1 --chat-template --gen-len 100 --top-k 100 --output bringup/references/qwen3_8b_aime24_chat_p1.refpt > $L/generate_reference_aime24_p1.log 2>&1; echo "exit=$?" >> $L/generate_reference_aime24_p1.log
timeout 1800 $S 2 $PY $L/generate_reference_wrapper.py --hf-model Qwen/Qwen3-8B --prompt-source book --prompt-len 32 --gen-len 100 --top-k 100 --output bringup/references/qwen3_8b_book32.refpt > $L/generate_reference_book32.log 2>&1; echo "exit=$?" >> $L/generate_reference_book32.log
echo "END refs $(date -u +%H:%M:%S)" >> $ST
bash $L/sweep_chain_chip2.sh c1_all_bfp8_lofi c12_all_bfp4_inner c13_all_bfp4_inner_lm_bfp4 c17_mlp_attn_bfp4_all_layers c16_all_bfp4_all_layers c15_mlp_bfp4_inner_lm_bfp4 c14_mlp_bfp4_all_lm_bfp4
echo "ROUND2_DONE $(date -u +%H:%M:%S)" >> $ST
