#!/usr/bin/env bash
set -u
source /home/hous/dev/qwen3.5-35b-a3b/scripts/environment.sh >/dev/null 2>&1
cd /home/hous/dev/qwen3.5-35b-a3b/tt-metal
source python_env/bin/activate
export PYTHONPATH=/home/hous/.claude/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.19/runtime:$PYTHONPATH
M=models.autoports.qwen_qwen3_5_35b_a3b.tests
D=/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/probe_control
REF=/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage06/probe/hf_ref_0_3_300_8.pt
step() { local name=$1; shift; echo "START $name $(date -u +%FT%TZ)"; "$@"; local rc=$?; echo "END $name rc=$rc $(date -u +%FT%TZ)"; }
menv() { [ $1 = 1 ] && echo QDEV=3; [ $1 = 2 ] && echo TT_VISIBLE_DEVICES=2,3; [ $1 = 4 ] && echo STAGE5_MESH=4; }
for m in 2 1; do
  step gen_m$m sh -c "env $(menv $m) python -m $M.run_generator_probe --mesh $m --reference $REF --sequence device,host --output $D/gen_m$m.json > $D/gen_m$m.log 2>&1"
  step model_m$m sh -c "env $(menv $m) python -m $M.run_model_probe --mesh $m --output $D/model_m$m.json > $D/model_m$m.log 2>&1"
done
echo CONTROL_DONE
