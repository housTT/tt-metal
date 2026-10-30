#!/usr/bin/env bash
# usage: run_sampling.sh <full|smoke> <port> <max_num_seqs> <log path>
# Hand-run of the packaged readiness runner's sampling stage (its _find_plugin_tests_dir rejects the nested plugin layout):
# the same pytest command, targets and options as runtime/readiness_check/run_vllm_server.py.
profile="$1"; port="$2"; seqs="$3"; log="$4"
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
T=/home/hous/dev/vllm/plugins/vllm-tt-plugin/tests/tt
if [ "$profile" = "full" ]; then targets=("$T"); else
  targets=("$T/test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch" "$T/test_seeding_and_variety.py::TestSeedingAndVariety::test_top1_is_greedy" "$T/test_logprobs.py::TestLogprobs::test_chat_logprobs_all_vocab" "$T/test_host_only_params.py::TestHostOnlyParameters::test_min_p"); fi
env HF_HOME=/home/hous/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/home/hous/dev/tt-metal-wt-qwen3:/home/hous/dev/tt-metal:/home/hous/dev/skills/plugins/tt-model-bringup/runtime \
  VLLM_TT_PLUGIN_ROOT=/home/hous/dev/vllm/plugins/vllm-tt-plugin \
  timeout 7200 /home/hous/dev/tt-metal/python_env/bin/python -m pytest "${targets[@]}" -v \
  --tt-server-url=http://localhost:$port --tt-model-name=Qwen/Qwen3-8B --tt-max-num-seqs=$seqs -p no:cacheprovider > "$log" 2>&1
rc=$?; echo "sampling profile=$profile rc=$rc $(date -u)" | tee -a "$log.status"; exit $rc
