# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import struct

KERNEL_ROOT = "models/autoports/qwen_qwen3_5_35b_a3b/tt/kernels/gdn_decode"
FP32_CBS = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18)
CB_TILES = {0: 4, 1: 4, 2: 4, 3: 4, 4: 3, 5: 1, 6: 16, 7: 4, 8: 4, 9: 4, 10: 4, 11: 1, 12: 1, 13: 4, 14: 1, 15: 4,
            16: 2, 17: 4, 18: 4}
GATE_CBS = (3, 7, 11, 13, 17, 18)


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _cbs(ttnn, cores, indices):
    descriptors = []
    for index in indices:
        dtype = ttnn.float32 if index in FP32_CBS else ttnn.bfloat16
        size = 4096 if dtype == ttnn.float32 else 2048
        descriptors.append(ttnn.CBDescriptor(
            total_size=CB_TILES[index] * size, core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)]))
    return descriptors


def _compute_config(ttnn):
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for index in FP32_CBS:
        modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                          fp32_dest_acc_en=True)
    config.unpack_to_dest_mode = modes
    return config


def _accessor(ttnn, tensor):
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def gdn_decode(q, k, v, z, decay_log, beta, state, norm_weight, *, groups, blocks=0, l2_eps=1e-6, norm_eps=1e-6,
               gates=None, z_offset=0, conv=None):
    import ttnn

    heads, head_k, head_v = state.shape[1], state.shape[-2], state.shape[-1]
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    if not blocks:
        blocks = next(b for b in (1, 2, 4) if heads * 4 // b <= grid.x * grid.y)
    per_head = 4 // blocks
    workers = heads * per_head
    cores = ttnn.num_cores_to_corerangeset(workers, grid, row_wise=True)
    coords = [(i % grid.x, i // grid.x) for i in range(workers)]
    assignment = [(i // per_head, (i % per_head) * blocks) for i in range(workers)]
    fused = blocks == 4
    if conv is not None and fused:
        raise ValueError("conv state shift needs the separate gate program (blocks < 4)")
    output = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, heads * head_v)),
                                            ttnn.bfloat16 if fused else ttnn.float32, ttnn.TILE_LAYOUT, device,
                                            ttnn.L1_MEMORY_CONFIG)
    if gates is None:
        negative, a_start, b_start = decay_log, 0, 0
    else:
        decay_log, beta, negative, a_start, b_start = gates
    tensors = (q, k, v, z, decay_log, beta, state, norm_weight, negative)
    inputs = list(dict((id(t), t) for t in tensors).values())
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[groups, int(decay_log.dtype == ttnn.float32), int(beta.dtype == ttnn.float32), blocks,
                           int(gates is not None)] + sum((_accessor(ttnn, t) for t in tensors), []),
        runtime_args=[(c, [t.buffer_address() for t in tensors[:8]] + [h, b, negative.buffer_address(), a_start + h,
                                                                       b_start + h, z_offset])
                      for (h, b), c in zip(assignment, coords)],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=[blocks] + _accessor(ttnn, state) + _accessor(ttnn, output),
        runtime_args=[(c, [state.buffer_address(), output.buffer_address(), h, b])
                      for (h, b), c in zip(assignment, coords)],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[_bits(l2_eps), _bits(head_k ** -0.5), _bits(1.0 / head_v), _bits(norm_eps), blocks,
                           int(gates is not None)],
        runtime_args=[(c, []) for c in coords], config=_compute_config(ttnn))
    ttnn.generic_op(inputs + [output], ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute], semaphores=[], cbs=_cbs(ttnn, cores, sorted(CB_TILES))))
    if fused:
        return output
    gated = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 1, heads * head_v)), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                           device, ttnn.L1_MEMORY_CONFIG)
    head_cores = ttnn.num_cores_to_corerangeset(heads, grid, row_wise=True)
    head_coords = [(i % grid.x, i // grid.x) for i in range(heads)]
    gate_reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/gate_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=head_cores,
        compile_time_args=_accessor(ttnn, output) + _accessor(ttnn, z) + _accessor(ttnn, norm_weight),
        runtime_args=[(c, [output.buffer_address(), z.buffer_address(), norm_weight.buffer_address(), h, z_offset])
                      for h, c in enumerate(head_coords)],
        config=ttnn.ReaderConfigDescriptor())
    gate_cbs = _cbs(ttnn, head_cores, GATE_CBS)
    gate_tensors = [output, z, norm_weight, gated]
    if conv is None:
        gate_writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/gate_writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=head_cores, compile_time_args=_accessor(ttnn, gated),
            runtime_args=[(c, [gated.buffer_address(), h]) for h, c in enumerate(head_coords)],
            config=ttnn.WriterConfigDescriptor())
    else:
        conv_state, current = conv
        width = conv_state.shape[-1]
        span = width * 2 // heads
        if width % heads or span % 64:
            raise ValueError(f"conv state width {width} does not split into 64-byte spans over {heads} heads")
        gate_writer = ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/gate_writer_shift.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=head_cores,
            compile_time_args=_accessor(ttnn, gated) + _accessor(ttnn, conv_state) + _accessor(ttnn, current),
            runtime_args=[(c, [gated.buffer_address(), h, conv_state.buffer_address(), current.buffer_address(),
                               h * span, span]) for h, c in enumerate(head_coords)],
            config=ttnn.WriterConfigDescriptor())
        gate_cbs.append(ttnn.CBDescriptor(
            total_size=3 * span, core_ranges=head_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=19, data_format=ttnn.bfloat16,
                                                        page_size=3 * span)]))
        gate_tensors += [conv_state, current]
    gate_compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/gate_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=head_cores, compile_time_args=[_bits(1.0 / head_v), _bits(norm_eps)],
        runtime_args=[(c, []) for c in head_coords], config=_compute_config(ttnn))
    ttnn.generic_op(gate_tensors, ttnn.ProgramDescriptor(
        kernels=[gate_reader, gate_writer, gate_compute], semaphores=[], cbs=gate_cbs))
    output.deallocate(True)
    return gated
