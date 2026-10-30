#!/usr/bin/env bash
# usage: evidence_chain_stage10.sh <port> <status file>
# Stage-10 served evidence chain against the live full-model server (batched prefill on): sampling (smoke, then
# full), qualitative, benchmark (primary 128/128/1 and CI burst 100/100/32), prefix_cache, prefill_burst against
# the stage-9 serial reference (same workload, one row per pass), mixed-shape trace reuse, degenerate check.
port="$1"; status="$2"
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
V=models/autoports/qwen_qwen3_8b/doc/vllm_integration/logs; D=models/autoports/qwen_qwen3_8b/doc/optimized_vllm/logs
RV=models/autoports/qwen_qwen3_8b/readiness_vllm; SERIAL=models/autoports/qwen_qwen3_8b/doc/optimized_vllm/vllm_prefill_burst_serial_benchmark.stage9.json
PY=/home/hous/dev/tt-metal/python_env/bin/python
run() { local name="$1"; shift; echo "$name start $(date -u +%H:%M:%S)" >> "$status"; "$@"; local rc=$?; echo "$name rc=$rc $(date -u +%H:%M:%S)" >> "$status"; return $rc; }
reset_apc() { curl -s -m 30 -X POST "http://localhost:$port/reset_prefix_cache" > /dev/null; }
run sampling_smoke $V/run_sampling.sh smoke $port 32 $D/sampling_smoke_full_model.log
run sampling_full $V/run_sampling.sh full $port 32 $RV/sampling_tests.log
run qualitative $V/run_checks.sh $port qualitative > $D/stage_qualitative.log 2>&1
reset_apc
run benchmark $V/run_checks.sh $port benchmark > $D/stage_benchmark.log 2>&1
run prefix_cache $V/run_checks.sh $port prefix_cache > $D/stage_prefix_cache.log 2>&1
reset_apc
run prefill_burst $V/run_checks.sh $port prefill_burst --prefill-burst-serial-result $SERIAL > $D/stage_prefill_burst.log 2>&1
reset_apc
run prefill_burst_repeat $V/run_checks.sh $port prefill_burst --prefill-burst-serial-result $SERIAL > $D/stage_prefill_burst_repeat.log 2>&1
cp $RV/vllm_prefill_burst_benchmark.json $D/vllm_prefill_burst_benchmark_repeat.json
run mixed_shape env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 $PY $V/mixed_shape_check.py --server-url http://localhost:$port --tokenizer /home/hous/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --server-log $RV/server.log --output $RV/mixed_shape_trace_reuse.json > $D/stage_mixed_shape.log 2>&1
run degenerate env PYTHONPATH=/home/hous/dev/skills/plugins/tt-model-bringup/runtime $PY /home/hous/dev/skills/plugins/tt-model-bringup/runtime/readiness_check/check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_8b --missing-artifacts critical --scope vllm > $D/stage_degenerate.log 2>&1
echo "chain done $(date -u +%H:%M:%S)" >> "$status"
