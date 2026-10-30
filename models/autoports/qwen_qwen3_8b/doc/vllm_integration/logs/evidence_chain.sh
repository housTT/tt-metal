#!/usr/bin/env bash
# usage: evidence_chain.sh <port> <status file>
# Served evidence chain against the live full-model server: sampling (smoke, then full), qualitative, benchmark,
# prefix_cache, prefill_burst (serial copy, then with the serial reference), mixed-shape trace reuse, degenerate check.
port="$1"; status="$2"
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
D=models/autoports/qwen_qwen3_8b/doc/vllm_integration/logs; RV=models/autoports/qwen_qwen3_8b/readiness_vllm
PY=/home/hous/dev/tt-metal/python_env/bin/python
run() { local name="$1"; shift; echo "$name start $(date -u +%H:%M:%S)" >> "$status"; "$@"; local rc=$?; echo "$name rc=$rc $(date -u +%H:%M:%S)" >> "$status"; return $rc; }
reset_apc() { curl -s -m 30 -X POST "http://localhost:$port/reset_prefix_cache" > /dev/null; }
run sampling_smoke $D/run_sampling.sh smoke $port 32 $D/sampling_smoke_full_model.log || exit 1
run sampling_full $D/run_sampling.sh full $port 32 $RV/sampling_tests.log
run qualitative $D/run_checks.sh $port qualitative > $D/stage_qualitative.log 2>&1
reset_apc
run benchmark $D/run_checks.sh $port benchmark > $D/stage_benchmark.log 2>&1
run prefix_cache $D/run_checks.sh $port prefix_cache > $D/stage_prefix_cache.log 2>&1
reset_apc
run prefill_burst_serial $D/run_checks.sh $port prefill_burst > $D/stage_prefill_burst_serial.log 2>&1
cp $RV/vllm_prefill_burst_benchmark.json $RV/vllm_prefill_burst_serial_benchmark.json
cp $RV/vllm_prefill_burst_result.json $RV/vllm_prefill_burst_serial_result.json
reset_apc
run prefill_burst $D/run_checks.sh $port prefill_burst --prefill-burst-serial-result $RV/vllm_prefill_burst_serial_benchmark.json > $D/stage_prefill_burst.log 2>&1
run mixed_shape env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 $PY $D/mixed_shape_check.py --server-url http://localhost:$port --tokenizer /home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --server-log $RV/server.log --output $RV/mixed_shape_trace_reuse.json > $D/stage_mixed_shape.log 2>&1
run degenerate env PYTHONPATH=/home/hous/dev/skills/plugins/tt-model-bringup/runtime $PY /home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_8b --missing-artifacts critical --scope vllm > $D/stage_degenerate.log 2>&1
echo "chain done $(date -u +%H:%M:%S)" >> "$status"
