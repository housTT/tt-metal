#!/usr/bin/env bash
# usage: tracked_chain_stage10.sh <port> <status file>
# Served workload on the tracked server (TT_METAL_TRACE_ALLOC_TRACKING=1 in the engine): smoke sampling subset,
# prefix-cache stage without the shared-prefix bench, the 32 x 128 prefill burst (batched prefill, decode replays
# after it), the mixed-shape trace-reuse workload. Artifacts copied to logs/tracked_served/.
port="$1"; status="$2"
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
V=models/autoports/qwen_qwen3_8b/doc/vllm_integration/logs; D=models/autoports/qwen_qwen3_8b/doc/optimized_vllm/logs
RV=models/autoports/qwen_qwen3_8b/readiness_vllm; T=$D/tracked_served2; SERIAL=models/autoports/qwen_qwen3_8b/doc/optimized_vllm/vllm_prefill_burst_serial_benchmark.stage9.json
PY=/home/hous/dev/tt-metal/python_env/bin/python
run() { local name="$1"; shift; echo "$name start $(date -u +%H:%M:%S)" >> "$status"; "$@"; local rc=$?; echo "$name rc=$rc $(date -u +%H:%M:%S)" >> "$status"; return $rc; }
reset_apc() { curl -s -m 30 -X POST "http://localhost:$port/reset_prefix_cache" > /dev/null; }
run sampling_smoke $V/run_sampling.sh smoke $port 32 $T/sampling_smoke.log
run prefix_cache $V/run_checks.sh $port prefix_cache --no-prefix-cache-shared-bench > $T/stage_prefix_cache.log 2>&1
cp $RV/prefix_cache_check.json $T/prefix_cache_check.json
reset_apc
run prefill_burst $V/run_checks.sh $port prefill_burst --prefill-burst-serial-result $SERIAL > $T/stage_prefill_burst.log 2>&1
cp $RV/vllm_prefill_burst_benchmark.json $T/vllm_prefill_burst_benchmark.json
run mixed_shape env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 $PY $V/mixed_shape_check.py --server-url http://localhost:$port --tokenizer /home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --server-log $RV/server.log --output $T/mixed_shape_trace_reuse_tracked.json > $T/stage_mixed_shape.log 2>&1
curl -s -m 10 http://localhost:$port/health > /dev/null && echo "health ok $(date -u +%H:%M:%S)" >> "$status"
echo "chain done $(date -u +%H:%M:%S)" >> "$status"
