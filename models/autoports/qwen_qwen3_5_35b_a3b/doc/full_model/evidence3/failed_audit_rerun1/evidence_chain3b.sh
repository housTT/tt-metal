#!/usr/bin/env bash
set -u
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh >/dev/null 2>&1
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
source python_env/bin/activate
export PYTHONPATH=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PYTHONPATH
A=models/autoports/qwen_qwen3_5_35b_a3b
E=$PWD/$A/doc/full_model/evidence3/audit_rerun
M=models.autoports.qwen_qwen3_5_35b_a3b.tests
mkdir -p $E
SAFE="TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 QWEN35_STRICT_CACHE=1"

manifest() { find $A/tt $A/tests -type f -not -name "*.pyc" -not -path "*__pycache__*" | sort | xargs sha256sum; git diff --stat -- ttnn | tail -1; }
step() { local name=$1; shift; echo "START $name $(date -u +%FT%TZ)"; "$@"; local rc=$?; echo "END $name rc=$rc $(date -u +%FT%TZ)"; }
menv() { [ $1 = 1 ] && echo QDEV=3; [ $1 = 2 ] && echo TT_VISIBLE_DEVICES=2,3; [ $1 = 4 ] && echo STAGE5_MESH=4; }

manifest > $E/source_manifest_start.txt
step cpu_tests sh -c "python -m pytest -q -p no:cacheprovider $A/tests/test_generator_warmup.py $A/tests/test_generator_sampling.py $A/tests/test_prefill_plan.py $A/tests/test_multichip_plan.py > $E/cpu_tests.log 2>&1"
for m in 4 2 1; do
  D=$E/m$m; mkdir -p $D
  step micro_m$m sh -c "env $(menv $m) TT_METAL_TRACE_ALLOC_TRACKING=1 python -m $M.run_sampler_micro --mesh $m --draws 64 --output $D/sampler_micro.json > $D/sampler_micro.log 2>&1"
  step sampling_m$m sh -c "env $(menv $m) $SAFE TT_METAL_TRACE_ALLOC_TRACEBACKS=1 python -m $M.run_sampling_audit --mesh $m --output $D/sampling_audit.json > $D/sampling_audit.log 2>&1"
done

manifest > $E/source_manifest_end.txt
cmp -s $E/source_manifest_start.txt $E/source_manifest_end.txt && echo SOURCE_STABLE || echo SOURCE_CHANGED
echo CHAIN_DONE
