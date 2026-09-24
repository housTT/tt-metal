set -u
WT=/home/ttuser/dev/muse-glimmer/wt-toolcalling
M=models/autoports/meta_models_muse_glimmer_30b
PLUG=/home/ttuser/dev/muse-glimmer/wt-plugin-apc/src
SP=/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad
OUT=$SP/scorecard
TTI=/home/ttuser/dev/muse-glimmer/tti-release/muse-glimmer-30b/tt-inference-server
LM=$TTI/.workflow_venvs/.venv_evals_common/bin/lm_eval
PYV=$WT/pc_serve_pyenv/bin/python
URL=http://127.0.0.1:8000/v1/chat/completions
MODEL=meta-models/Muse-Glimmer-30B
pkill -F $SP/srv.pid 2>/dev/null; sleep 8
export TT_METAL_HOME=$WT PYTHONPATH=$WT:$PLUG MESH_DEVICE=P300x2 ARCH_NAME=blackhole
export VLLM_TARGET_DEVICE=tt TORCHDYNAMO_DISABLE=1 VLLM_RPC_TIMEOUT=900000 VLLM_CONFIGURE_LOGGING=1
AC='{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}}'
echo "plugin imported from: $($PYV -c 'import vllm_tt_plugin,os;print(os.path.dirname(vllm_tt_plugin.__file__))')"
echo "START $(date '+%Y-%m-%d %H:%M:%S %Z')"
nohup $WT/pc_serve_pyenv/bin/vllm serve $MODEL \
  --block-size 64 --max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 131072 --seed 9472 \
  --additional-config "$AC" --port 8000 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --tool-parser-plugin $WT/$M/tt/muse_glimmer_tool_parser.py \
  --reasoning-parser-plugin $WT/$M/tt/reasoning_parser.py > $OUT/server.log 2>&1 &
echo $! > $SP/srv.pid
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "READY $((i*10))s"; break; }; kill -0 $(cat $SP/srv.pid) 2>/dev/null || { echo DIED; tail -20 $OUT/server.log; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo NOT_READY; exit 1; }

echo "######## ifeval ($(date +%H:%M)) ########"; T0=$(date +%s)
$LM --tasks ifeval --model local-chat-completions \
  --model_args model=$MODEL,base_url=$URL,tokenizer_backend=huggingface,max_length=131072,timeout=3600,num_concurrent=32 \
  --gen_kwargs stream=false,do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32768,seed=42 \
  --output_path $OUT/ifeval --seed 42 --num_fewshot 0 --batch_size 1 --log_samples --show_config \
  --apply_chat_template --trust_remote_code --confirm_run_unsafe_code 2>&1 | grep -E "ifeval\||failed prompt"
echo "ifeval seconds: $(( $(date +%s) - T0 ))"

echo "######## gpqa_diamond_cot_zeroshot ($(date +%H:%M)) ########"; T0=$(date +%s)
$LM --tasks gpqa_diamond_cot_zeroshot --model local-chat-completions \
  --model_args model=$MODEL,base_url=$URL,tokenizer_backend=huggingface,num_concurrent=16 \
  --gen_kwargs do_sample=true,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32768,seed=42 \
  --output_path $OUT/gpqa --seed 42 --num_fewshot 0 --batch_size 1 --log_samples --show_config \
  --apply_chat_template --trust_remote_code --confirm_run_unsafe_code 2>&1 | grep -E "gpqa_diamond\||failed prompt"
echo "gpqa seconds: $(( $(date +%s) - T0 ))"

echo "######## latency sweep ($(date +%H:%M)) ########"
mkdir -p $OUT/sweep
TEMP_FLAG="--temperature 0.0"
$PYV -m vllm.entrypoints.cli.main bench serve --host 127.0.0.1 --port 8000 --model $MODEL --backend openai --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 8 --random-range-ratio 0 --num-prompts 1 --max-concurrency 1 \
  --ignore-eos --seed 1234 $TEMP_FLAG --disable-tqdm >/dev/null 2>&1 || { echo "temperature flag unsupported; using server default sampling"; TEMP_FLAG=""; }
echo "sampling flag: '${TEMP_FLAG}'"
point () { # isl osl users n
  local isl=$1 osl=$2 u=$3 n=$4 f="isl${isl}_osl${osl}_u${u}.json"
  [ -s "$OUT/sweep/$f" ] && return 0
  echo ">>> isl=$isl osl=$osl users=$u n=$n $(date +%H:%M:%S)"
  timeout 1500 $PYV -m vllm.entrypoints.cli.main bench serve --host 127.0.0.1 --port 8000 --model $MODEL \
    --backend openai --endpoint /v1/completions --dataset-name random \
    --random-input-len $isl --random-output-len $osl --random-range-ratio 0 \
    --num-prompts $n --max-concurrency $u --num-warmups 1 --ignore-eos --seed 1234 $TEMP_FLAG \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --disable-tqdm \
    --save-result --result-dir $OUT/sweep --result-filename "$f" 2>&1 | grep -E "Mean TTFT|Mean TPOT|Mean E2EL|Output token throughput|Error|error" | head -5
}
e2el () { python3 -c "import json;print(json.load(open('$1'))['mean_e2el_ms']/1000)" 2>/dev/null || echo 0; }
for isl in 128 1024 4096 16384 32768 65536 130816; do
  osl=256; [ $isl = 128 ] && osl=128
  case $isl in 128|1024|4096) mult=4;; 16384) mult=2;; *) mult=1;; esac
  for u in 1 2 4 8 16 32; do
    [ "$isl" = 65536 ] && [ "$u" = 32 ] && continue
    [ "$isl" = 130816 ] && [ "$u" -ge 16 ] && continue
    n=$(( u * mult ))
    point $isl $osl $u $n
    last=$(e2el "$OUT/sweep/isl${isl}_osl${osl}_u${u}.json")
    awk "BEGIN{exit !($last > 300)}" && { echo "  E2EL ${last}s > 300s; skipping the rest of isl=$isl"; break; }
  done
done
echo "END $(date '+%Y-%m-%d %H:%M:%S %Z')"
pkill -F $SP/srv.pid 2>/dev/null
echo SCORECARD_DONE
