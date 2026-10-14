#!/usr/bin/env bash
set -u
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh >/dev/null 2>&1
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
source python_env/bin/activate
export PYTHONPATH=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PYTHONPATH
A=models/autoports/qwen_qwen3_5_35b_a3b
E=$PWD/$A/doc/full_model/evidence1
W=/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/evidence1_work
R=/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/references
M=models.autoports.qwen_qwen3_5_35b_a3b.tests
HF=/home/hous/dev/qwen3.5-35b-a3b/weights/Qwen3.5-35B-A3B
mkdir -p $E $W
SAFE="TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN35_STRICT_CACHE=1"

manifest() { find $A/tt $A/tests -type f -not -name "*.pyc" -not -path "*__pycache__*" | sort | xargs sha256sum; git diff --stat -- ttnn | tail -1; }
step() { local name=$1; shift; echo "START $name $(date -u +%FT%TZ)"; "$@"; local rc=$?; echo "END $name rc=$rc $(date -u +%FT%TZ)"; }
menv() { [ $1 = 1 ] && echo QDEV=3; [ $1 = 2 ] && echo TT_VISIBLE_DEVICES=2,3; [ $1 = 4 ] && echo STAGE5_MESH=4; }
rmesh() { [ $1 = 1 ] && echo "--mesh-device N150"; [ $1 = 2 ] && echo "--mesh-device N300 --fabric-config FABRIC_1D"; [ $1 = 4 ] && echo "--mesh-device P300x2 --fabric-config FABRIC_1D_RING"; }
renv() { [ $1 = 2 ] && echo TT_VISIBLE_DEVICES=2,3; [ $1 != 2 ] && echo STAGE6_RUNNER=1; }

manifest > $E/source_manifest_start.txt
step cpu_tests sh -c "python -m pytest -q -p no:cacheprovider $A/tests/test_generator_warmup.py $A/tests/test_prefill_plan.py $A/tests/test_multichip_plan.py > $E/cpu_tests.log 2>&1"

for m in 4 2 1; do
  D=$E/m$m; mkdir -p $D
  step tf_aime_m$m sh -c "env $(renv $m) $SAFE python -m readiness_check.run_teacher_forcing --model-dir $A --reference $R/readiness_aime24_chat.refpt $(rmesh $m) > $D/teacher_forcing_aime24.log 2>&1"
  step tf_book_m$m sh -c "env $(renv $m) $SAFE python -m readiness_check.run_teacher_forcing --model-dir $A --reference $R/readiness_book_8193.refpt $(rmesh $m) > $D/teacher_forcing_book8193.log 2>&1"
  step prefill_m$m sh -c "env $(renv $m) $SAFE python -m readiness_check.run_prefill_check --model-dir $A --reference $R/readiness_aime24_chat.refpt $(rmesh $m) > $D/prefill_check_aime24.log 2>&1"
  step autoreg_m$m sh -c "env $(renv $m) $SAFE QWEN35_HF_OFFLOAD=0 python -m $M.hf_offload readiness_check.run_autoregressive --model-dir $A --hf-model $HF --output-dir $D/autoregressive $(rmesh $m) > $D/autoregressive.log 2>&1"
  step qualitative_m$m sh -c "env $(menv $m) $SAFE python -m $M.run_qualitative --phase tt --mesh $m --output-dir $E/qualitative > $D/qualitative.log 2>&1"
  step trace_m$m sh -c "env $(menv $m) $SAFE python -m $M.run_split_sampling_trace --mesh $m --layers '' --output $D/split_sampling_trace.json > $D/split_sampling_trace.log 2>&1"
  step evidence_m$m sh -c "env $(menv $m) $SAFE python -m $M.run_full_evidence --mesh $m $([ $m = 1 ] && echo '--batches 1,4,8') --output $D/full_evidence.json > $D/full_evidence.log 2>&1"
  step watcher_m$m sh -c "env $(menv $m) TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH=$W/watcher_m$m python -m $M.run_generator_probe --mesh $m --reference /home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/probe/hf_ref_0_3_300_8.pt --sequence device,host --output $D/watcher_probe.json > $D/watcher_probe.log 2>&1"
  step profile_m$m sh -c "cd $W && env $(menv $m) TT_METAL_RUNTIME_ROOT=/tmp/q35-functional-profiler-root TT_METAL_KERNEL_PATH=/tmp/q35-functional-profiler-root TT_METAL_CACHE=/tmp/q35-opt-profiler-cache python -m tracy -r -p -v --no-web-server --check-exit-code --op-support-count 32768 --dump-device-data-mid-run --disable-device-data-push-to-tracy -o $W/profile_m$m -m $M.run_model_profile --mesh $m --output $D/profile_run.json > $D/profile.log 2>&1"
done
step degenerate sh -c "python /home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime/readiness_check/check_degenerate_output.py --model-dir $A --missing-artifacts critical --scope autoregressive --json $E/degenerate_check.json > $E/degenerate_check.log 2>&1"

manifest > $E/source_manifest_end.txt
cmp -s $E/source_manifest_start.txt $E/source_manifest_end.txt && echo SOURCE_STABLE || echo SOURCE_CHANGED
echo CHAIN_DONE
