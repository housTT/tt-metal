# Environment for the *profiling* runs of the Qwen3.6-27B autoport bringup.
#
# The shared install at ~/.local/lib/model-bringup/tt-metal was configured with
# ENABLE_TRACY=OFF (build_Release/CMakeCache.txt), so any TT_METAL_DEVICE_PROFILER run there
# dies with "TT_METAL_DEVICE_PROFILER requires a Tracy-enabled build of tt-metal"
# (tt_metal/llrt/rtoptions.cpp:816).  Rather than rebuild the shared install underneath the
# rest of the pipeline, the same source tree was copied to ...-profiler and rebuilt with the
# profiler enabled (tt-metal's default).  Only the profiling runs use this env; every
# correctness run in this stage used ttenv.sh and the untouched shared install.
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal-profiler
export TT_VISIBLE_DEVICES=2
# Exposing a single chip of a 2-chip p300 makes metal classify the cluster as CUSTOM and
# demand an explicit fabric mesh graph descriptor (tt_cluster.cpp:281).  The stock
# single-Blackhole-chip descriptor is the right shape for a 1x1 mesh; its p150 file name
# names the board the descriptor ships for, not the board in this host.
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
export REPO=/home/ttuser/dev/qwen/tt-metal
# Put the profiler tree ahead of the ttnn-custom.pth entries that point at the shared install.
export PYTHONPATH=$REPO:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$TT_METAL_HOME
