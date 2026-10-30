#!/usr/bin/env bash
# Round 3 queues: robustness references for the remaining contenders, one queue per chip, each waiting for that chip's current chain.
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
L=models/autoports/qwen_qwen3_8b/doc/datatype_sweep/logs
case "$1" in
  2) until grep -q ROUND2_DONE $L/round2_chip2.status 2>/dev/null; do sleep 10; done
     bash $L/sweep_chain.sh 2 bringup/references/qwen3_8b_book32.refpt book32 c15_mlp_bfp4_inner_lm_bfp4 c14_mlp_bfp4_all_lm_bfp4 ;;
  0) until grep -q CHIP0_DONE $L/chip0_chain.status 2>/dev/null; do sleep 10; done
     bash $L/sweep_chain.sh 0 bringup/references/qwen3_8b_book32.refpt book32 c13_all_bfp4_inner_lm_bfp4 c16_all_bfp4_all_layers c17_mlp_attn_bfp4_all_layers ;;
  1) until grep -q CHAIN_DONE $L/sweep_chain_aime_p1_chip1.status 2>/dev/null; do sleep 10; done
     bash $L/sweep_chain.sh 1 bringup/references/qwen3_8b_aime24_chat_p1.refpt aime_p1 c13_all_bfp4_inner_lm_bfp4 c16_all_bfp4_all_layers c17_mlp_attn_bfp4_all_layers ;;
esac
echo "QUEUE_DONE chip $1 $(date -u +%H:%M:%S)" >> $L/queue_round3.status
