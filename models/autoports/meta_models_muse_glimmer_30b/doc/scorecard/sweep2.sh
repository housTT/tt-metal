set -u
WT=/home/ttuser/dev/muse-glimmer/wt-toolcalling
M=models/autoports/meta_models_muse_glimmer_30b
PLUG=/home/ttuser/dev/muse-glimmer/wt-plugin-apc/src
SP=/tmp/claude-1000/-home-ttuser-dev-muse-glimmer/86562100-a910-4009-9d96-9c0c923aca30/scratchpad
OUT=$SP/scorecard; mkdir -p $OUT/sweep2
PYV=$WT/pc_serve_pyenv/bin/python
MODEL=meta-models/Muse-Glimmer-30B
pkill -F $SP/srv.pid 2>/dev/null; sleep 8
export TT_METAL_HOME=$WT PYTHONPATH=$WT:$PLUG MESH_DEVICE=P300x2 ARCH_NAME=blackhole
export VLLM_TARGET_DEVICE=tt TORCHDYNAMO_DISABLE=1 VLLM_RPC_TIMEOUT=900000 VLLM_CONFIGURE_LOGGING=1
AC='{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D_RING", "fabric_packet_payload_bytes": 8192, "l1_small_size": 6144, "trace_mode": "decode_only"}}'
echo "START $(date '+%Y-%m-%d %H:%M:%S %Z')"
# prefix caching OFF for the sweep: with it on, the seeded random prompts repeat across rows and warm-ups, so TTFT measured cache hits
nohup $WT/pc_serve_pyenv/bin/vllm serve $MODEL \
  --block-size 64 --max-model-len 131072 --max-num-seqs 32 --max-num-batched-tokens 131072 --seed 9472 \
  --no-enable-prefix-caching --additional-config "$AC" --port 8000 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --tool-parser-plugin $WT/$M/tt/muse_glimmer_tool_parser.py \
  --reasoning-parser-plugin $WT/$M/tt/reasoning_parser.py > $OUT/server_sweep2.log 2>&1 &
echo $! > $SP/srv.pid
for i in $(seq 1 300); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "READY $((i*10))s"; break; }; kill -0 $(cat $SP/srv.pid) 2>/dev/null || { echo DIED; tail -20 $OUT/server_sweep2.log; exit 1; }; sleep 10; done
curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo NOT_READY; exit 1; }
grep -oE "prefix caching is [a-z]+" $OUT/server_sweep2.log | tail -1
k=0
point () { # isl osl users n
  local isl=$1 osl=$2 u=$3 n=$4 f="isl${isl}_osl${osl}_u${u}.json"; k=$((k+1))
  echo ">>> isl=$isl osl=$osl users=$u n=$n seed=$((1234+k)) $(date +%H:%M:%S)"
  timeout 1500 $PYV -m vllm.entrypoints.cli.main bench serve --host 127.0.0.1 --port 8000 --model $MODEL \
    --backend openai --endpoint /v1/completions --dataset-name random \
    --random-input-len $isl --random-output-len $osl --random-range-ratio 0 \
    --num-prompts $n --max-concurrency $u --num-warmups 1 --ignore-eos --seed $((1234+k)) --temperature 0.0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 --disable-tqdm \
    --save-result --result-dir $OUT/sweep2 --result-filename "$f" 2>&1 | grep -E "Mean TTFT|Mean E2EL" | head -2
}
e2el () { python3 -c "import json;print(json.load(open('$1'))['mean_e2el_ms']/1000)" 2>/dev/null || echo 0; }
for isl in 128 1024 4096 16384 32768 65536 130816; do
  osl=256; [ $isl = 128 ] && osl=128
  case $isl in 128|1024|4096) mult=4;; 16384) mult=2;; *) mult=1;; esac
  for u in 1 2 4 8 16 32; do
    [ "$isl" = 65536 ] && [ "$u" = 32 ] && continue
    [ "$isl" = 130816 ] && [ "$u" -ge 16 ] && continue
    point $isl $osl $u $(( u * mult ))
    last=$(e2el "$OUT/sweep2/isl${isl}_osl${osl}_u${u}.json")
    awk "BEGIN{exit !($last > 300)}" && { echo "  E2EL ${last}s > 300s; skipping the rest of isl=$isl"; break; }
  done
done
echo "END $(date '+%Y-%m-%d %H:%M:%S %Z')"; pkill -F $SP/srv.pid 2>/dev/null; echo SWEEP2_DONE
