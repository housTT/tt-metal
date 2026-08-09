# Environment for Qwen3.6-27B autoport bringup on this host.
# Board layout on this host (from /sys/class/tenstorrent/*/tt_card_type and tt_serial):
#   two p300c boards, two Blackhole chips each.
#     board 000004613192404C : /dev/tenstorrent/0 (ARC wedged, no sysfs attrs) + /dev/tenstorrent/1
#     board 0000046131924022 : /dev/tenstorrent/2 + /dev/tenstorrent/3  (both healthy)
# This stage needs a 1x1 mesh, so it runs on device 2 - a chip of the fully intact board -
# rather than on device 1, whose partner chip is the wedged one awaiting an operator power
# cycle.  Devices 2 and 3 are the pair the multichip stage will want.
# The built tt-metal lives in ~/.local/lib/model-bringup/tt-metal; the repo under
#   test is ~/dev/qwen/tt-metal (same commit, unbuilt).  Kernel sources must come
#   from the built tree, so never run device jobs with cwd inside a tt-metal checkout.
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export TT_VISIBLE_DEVICES=2
# Exposing a single chip of a 2-chip p300 makes metal classify the cluster as CUSTOM and
# demand an explicit fabric mesh graph descriptor (tt_cluster.cpp:281).  The stock
# single-Blackhole-chip descriptor is the right shape for a 1x1 mesh; its p150 file name
# names the board the descriptor ships for, not the board in this host.
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export REPO=/home/ttuser/dev/qwen/tt-metal
export PYTHONPATH=$REPO
