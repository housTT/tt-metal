# Environment for Qwen3.6-27B autoport bringup (v2 branch: agentic-research/hous/qwen3.6-27b-v2).
#
# Board layout on this host (from /sys/class/tenstorrent/*/tt_card_type and tt_serial):
#   two p300c boards, two Blackhole chips each.
#     board 000004613192404C : /dev/tenstorrent/0 (ARC wedged, no sysfs attrs) + /dev/tenstorrent/1
#     board 0000046131924022 : /dev/tenstorrent/2 + /dev/tenstorrent/3  (both healthy)
# This stage needs a 1x1 mesh, so it runs on device 2 - a chip of the fully intact board -
# rather than on device 1, whose partner chip is the wedged one awaiting an operator power
# cycle.
#
# Unlike the v1 branch, this checkout is built in-tree (build_Release, ENABLE_TRACY=ON) and
# ships its own venv, so TT_METAL_HOME is the repo itself: kernel sources, headers and the
# loaded _ttnn.so all come from the same tree and the same commit.  Activating
# $REPO/python_env is mandatory - a stale editable install on this host otherwise binds
# `ttnn` to a different tt-metal tree, and PYTHONPATH alone does not undo that.
export REPO=/home/ttuser/dev/qwen/tt-metal
export TT_METAL_HOME=$REPO
export TT_VISIBLE_DEVICES=2
# Exposing a single chip of a 2-chip p300 makes metal classify the cluster as CUSTOM and
# demand an explicit fabric mesh graph descriptor (tt_cluster.cpp).  The stock
# single-Blackhole-chip descriptor is the right shape for a 1x1 mesh; its p150 file name
# names the board the descriptor ships for, not the board in this host.
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

source $REPO/python_env/bin/activate
export PYTHONPATH=$REPO

# Assert ttnn resolves inside this checkout; a stale editable install elsewhere would make
# every measurement below describe code that is not under test.
python - <<'PY' || return 1 2>/dev/null || exit 1
import sys, ttnn, pathlib
repo = pathlib.Path("/home/ttuser/dev/qwen/tt-metal").resolve()
p = pathlib.Path(ttnn.__file__).resolve()
assert repo in p.parents, f"ttnn resolves OUTSIDE this checkout: {p}"
print(f"ttnn OK: {p}")
print(f"python : {sys.executable}")
PY
