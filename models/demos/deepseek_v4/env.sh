# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source this before running any deepseek_v4 bring-up script or test.
#   source models/demos/deepseek_v4/env.sh
# Environment: tt-quietbox, 4x Blackhole p300c, tt-metal built from source (0.75-dev).
# NOTE: correctness runs on real Blackhole silicon (no libttsim installed); see
# BRINGUP_PLAN.md §6.1. There are NO simulator/hardware conditionals in model code.
source /home/ttuser/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PATH=/usr/local/bin:$PATH
# ttnn is editable-installed (ttnn-custom.pth wires repo root + ttnn + tools onto sys.path).
