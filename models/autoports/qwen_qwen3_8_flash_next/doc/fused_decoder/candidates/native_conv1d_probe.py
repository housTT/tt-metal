# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact single-chip native-conv1d feasibility probe for Qwen3.8 GDN.

This adapts the Qwen3.5 TP native depthwise-conv path to this autoport's
unsharded width: 16 * 128 Q, 16 * 128 K, and 48 * 128 V channels.
"""

import math
import os

import torch

import ttnn

KEY_HEADS = 16
VALUE_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
CHANNELS = 2 * KEY_HEADS * KEY_DIM + VALUE_HEADS * VALUE_DIM
KERNEL = 4
TOKENS = 128
INPUT_LENGTH = TOKENS + KERNEL - 1


def main():
    assert CHANNELS == 10240
    torch.manual_seed(0)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        trace_region_size=0,
        l1_small_size=24576,
    )
    try:
        host_input = torch.randn(1, INPUT_LENGTH, 1, CHANNELS, dtype=torch.bfloat16)
        host_weight = torch.randn(CHANNELS, 1, KERNEL, dtype=torch.bfloat16) * 0.02
        device_input = ttnn.from_torch(
            host_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        host_weight = ttnn.from_torch(
            host_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        config_tensors_in_dram = os.environ.get("NATIVE_CONV_CONFIG_DRAM", "0") == "1"
        conv_config = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            config_tensors_in_dram=config_tensors_in_dram,
        )
        prepared_weight = ttnn.prepare_conv_weights(
            weight_tensor=host_weight,
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            batch_size=1,
            input_height=1,
            input_width=INPUT_LENGTH,
            kernel_size=(1, KERNEL),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=False,
            groups=CHANNELS,
            device=mesh,
            input_dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=compute,
        )
        slice_mode = os.environ.get("NATIVE_CONV_SLICE", "l1_full")
        if slice_mode == "l1_full":
            slice_config = ttnn.Conv2dL1FullSliceConfig
        elif slice_mode in ("dram_width_4", "dram_width_8"):
            slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceWidth,
                num_slices=int(slice_mode.rsplit("_", 1)[-1]),
            )
        else:
            raise ValueError(f"unknown NATIVE_CONV_SLICE={slice_mode!r}")
        print(f"NATIVE_CONV1D_ATTEMPT slice_mode={slice_mode} " f"config_tensors_in_dram={config_tensors_in_dram}")
        output = ttnn.conv1d(
            input_tensor=device_input,
            weight_tensor=prepared_weight,
            device=mesh,
            in_channels=CHANNELS,
            out_channels=CHANNELS,
            batch_size=1,
            input_length=INPUT_LENGTH,
            kernel_size=KERNEL,
            stride=1,
            padding=0,
            dilation=1,
            groups=CHANNELS,
            dtype=ttnn.bfloat16,
            conv_config=conv_config,
            compute_config=compute,
            slice_config=slice_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        print(
            "NATIVE_CONV1D_RESULT",
            f"channels={CHANNELS}",
            f"tokens={TOKENS}",
            f"kernel={KERNEL}",
            f"shape={tuple(output.shape)}",
            f"dtype={output.dtype}",
            f"finite={math.isfinite(float(ttnn.to_torch(output).float().mean()))}",
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
