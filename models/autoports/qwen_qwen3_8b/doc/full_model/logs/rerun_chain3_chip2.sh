#!/usr/bin/env bash
cd /home/hous/dev/tt-metal-wt-qwen3 || exit 3
S=models/autoports/qwen_qwen3_8b/doc/full_model/logs/stage_env.sh
L=models/autoports/qwen_qwen3_8b/doc/full_model/logs
PY=/home/hous/dev/tt-metal/python_env/bin/python
T=models/autoports/qwen_qwen3_8b/tests/test_full_model.py
F=$L/rerun_chain_chip2.status
step() { echo "=== $1 start $(date -u +%H:%M:%S)" >> $F; }
done_step() { echo "=== $1 exit=$2 end $(date -u +%H:%M:%S)" >> $F; }
step batch_buckets; $S 2 timeout 1800 env PYTHONUNBUFFERED=1 QWEN3_FULL_MODEL_LAYERS=36 $PY -m pytest -q -s $T -k "test_batch_decode" -p no:cacheprovider > $L/pytest_full_model_36_batch_buckets.log 2>&1; done_step batch_buckets $?
step qualitative_tt; $S 2 timeout 1200 env PYTHONUNBUFFERED=1 $PY $L/qualitative_tt.py 128 > $L/qualitative_tt_rerun.log 2>&1; done_step qualitative_tt $?
echo "=== chain3 done $(date -u +%H:%M:%S)" >> $F
