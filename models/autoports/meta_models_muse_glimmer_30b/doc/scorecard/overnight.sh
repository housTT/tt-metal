#!/usr/bin/env bash
# Overnight evals on the fixed build. One server boot; jobs strictly sequential;
# every job leaves a "DONE <job> <seconds>" (or "FAIL <job>") marker and the
# orchestrator continues past failures. Kill by PID from srv.pid, never by pattern.
set -u
WT=/home/ttuser/dev/muse-glimmer/wt-toolcalling
M=models/autoports/meta_models_muse_glimmer_30b
PLUG=/home/ttuser/dev/muse-glimmer/wt-plugin-apc/src
SP=/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad
SC=$SP/scorecard
TASKS=$SP/../tasks
TTI=/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server
LM=$TTI/.workflow_venvs/.venv_evals_common/bin/lm_eval
IFB_PY=$SC/ifbench_venv/bin/python
URL=http://127.0.0.1:8000/v1/chat/completions
MODEL=meta-models/Muse-Glimmer-30B
LOG=$SC/overnight.log
exec > >(tee -a "$LOG") 2>&1

echo "waiting for sweep2 (SWEEP2_DONE marker)..."
until grep -q "SWEEP2_DONE" "$TASKS/b8cjl0et1.output" 2>/dev/null; do sleep 20; done
pkill -F $SP/srv.pid 2>/dev/null; sleep 10

export TT_METAL_HOME=$WT PYTHONPATH=$WT:$PLUG MESH_DEVICE=P300x2 ARCH_NAME=blackhole
export VLLM_TARGET_DEVICE=tt TORCHDYNAMO_DISABLE=1 VLLM_RPC_TIMEOUT=900000 VLLM_CONFIGURE_LOGGING=1
AC='{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}}'
echo "NIGHT START $(date '+%Y-%m-%d %H:%M:%S %Z')  tt-metal $(cd $WT && git rev-parse --short HEAD)"
boot () {
  nohup $WT/pc_serve_pyenv/bin/vllm serve $MODEL \
    --block-size 64 --max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 131072 --seed 9472 \
    --additional-config "$AC" --port 8000 \
    --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
    --tool-parser-plugin $WT/$M/tt/muse_glimmer_tool_parser.py \
    --reasoning-parser-plugin $WT/$M/tt/reasoning_parser.py > $SC/server_overnight.log 2>&1 &
  echo $! > $SP/srv.pid
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "READY $((i*10))s"; break; }; kill -0 $(cat $SP/srv.pid) 2>/dev/null || { echo "FAIL server"; tail -20 $SC/server_overnight.log; return 1; }; sleep 10; done
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo "FAIL server NOT_READY"; return 1; }
}
alive () { curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; }
ensure () { alive && return 0; echo "SERVER DOWN $(date +%H:%M:%S); rebooting once"; pkill -F $SP/srv.pid 2>/dev/null; sleep 20; boot; }
boot || { echo "FAIL initial boot"; exit 1; }

job () { # name cmd...
  local name=$1; shift; local t0=$(date +%s)
  echo "######## $name  $(date +%H:%M:%S) ########"
  ensure || { echo "FAIL $name server-down"; return; }
  if "$@"; then echo "DONE $name $(( $(date +%s) - t0 ))"; else echo "FAIL $name $(( $(date +%s) - t0 ))"; fi
}
lmeval () { # task seed outdir extra...
  local task=$1 seed=$2 out=$3; shift 3
  $LM --tasks $task --model local-chat-completions "$@" \
    --output_path "$out" --seed $seed --num_fewshot 0 --batch_size 1 --log_samples --show_config \
    --apply_chat_template --trust_remote_code --confirm_run_unsafe_code 2>&1 | grep -E "^\|$task|failed prompt|Error" | head -3
  ls "$out"/*/results_*.json >/dev/null 2>&1
}
AIME_ARGS=(--model_args model=$MODEL,base_url=$URL,tokenizer_backend=huggingface,max_length=131072,timeout=3600,num_concurrent=32 --include_path $SC/tasks)
GPQA_ARGS=(--model_args model=$MODEL,base_url=$URL,tokenizer_backend=huggingface,num_concurrent=16)
IFEVAL_ARGS=(--model_args model=$MODEL,base_url=$URL,tokenizer_backend=huggingface,max_length=131072,timeout=3600,num_concurrent=32)
aime () { lmeval $1 $2 $SC/${1}_seed$2 "${AIME_ARGS[@]}" --gen_kwargs "stream=false,do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=98304,until=[],seed=$2" $3; }
gpqa () { lmeval gpqa_diamond_cot_zeroshot $1 $SC/gpqa_seed$1 "${GPQA_ARGS[@]}" --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32768,seed=$1"; }
ifeval () { lmeval ifeval $1 $SC/ifeval_seed$1 "${IFEVAL_ARGS[@]}" --gen_kwargs "stream=false,do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32768,seed=$1"; }
ifbench () { # tag seed temperature [limit]
  local tag=$1 seed=$2 temp=$3 lim=${4:-0}; local d=$SC/ifbench_$tag; mkdir -p $d
  local input=data/IFBench_test.jsonl
  # the scorer keys EVERY prompt of input_data, so a limited run must score against a matching subset
  if [ "$lim" -gt 0 ]; then head -n $lim $SC/IFBench/data/IFBench_test.jsonl > $d/input_subset.jsonl; input=$d/input_subset.jsonl; fi
  python3 $SC/ifbench_gen.py $URL $d/responses.jsonl --seed $seed --temperature $temp --limit $lim && \
  (cd $SC/IFBench && $IFB_PY -m run_eval --input_data=$input --input_response_data=$d/responses.jsonl --output_dir=$d/eval 2>&1 | grep -E "Accuracy Scores|prompt-level|instruction-level")
}

# --- pre-flights (small, prove the paths before the night is committed) ---
job preflight_aime26   aime aime26 42 "--limit 2"
job preflight_ifbench  ifbench preflight 42 1.0 3
rm -rf $SC/aime26_seed42 $SC/ifbench_preflight   # drop the preflight results; real runs follow

# --- the night ---
job ifbench_seed42     ifbench seed42 42 1.0
job aime26_seed42      aime aime26 42 ""
for s in 43 44 45; do job aime25_seed$s aime aime25 $s ""; done
for s in 43 44 45; do job aime26_seed$s aime aime26 $s ""; done
for s in 43 44 45; do job gpqa_seed$s  gpqa $s; done
for s in 43 44 45; do job ifeval_seed$s ifeval $s; done
for s in 43 44; do job ifbench_seed$s ifbench seed$s $s 1.0; done
job ifbench_temp0      ifbench temp0 42 0.0

echo "server: 500s=$(grep -c ' 500 ' $SC/server_overnight.log) guard_raises=$(grep -c 'both write block' $SC/server_overnight.log)"
pkill -F $SP/srv.pid 2>/dev/null
echo "NIGHT END $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo NIGHT_DONE
