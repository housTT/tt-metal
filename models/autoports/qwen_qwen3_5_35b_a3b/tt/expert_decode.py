# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

KERNEL_ROOT = "models/autoports/qwen_qwen3_5_35b_a3b/tt/kernels/expert_gemv"
TILE_BYTES = {"BFLOAT16": 2048, "BFLOAT8_B": 1088, "BFLOAT4_B": 576}


def _accessor(ttnn, tensor):
    return list(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _cb(ttnn, cores, index, dtype, pages, page_size):
    return ttnn.CBDescriptor(
        total_size=pages * page_size, core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)])


def expert_swiglu_decode(x, ids, weight, *, shared=None, nper=0, kb=8, fp32_dest=True, memory_config=None):
    import ttnn

    slots = ids.shape[-1]
    kt, nt = weight.shape[-2] // 32, weight.shape[-1] // 32
    half = nt // 2
    device = x.device()
    grid = device.compute_with_storage_grid_size()
    if not nper:
        nper = next(n for n in (1, 2) if half % n == 0
                    and (slots + (shared is not None)) * half // n <= grid.x * grid.y)
    if kt % kb:
        raise ValueError("K tiles must divide by the K block")
    if shared is not None and (shared.dtype != weight.dtype or (shared.shape[-2], shared.shape[-1]) != (kt * 32, nt * 32)):
        raise ValueError("shared expert weight must match one routed expert in dtype and shape")
    per_slot = half // nper
    workers = (slots + (shared is not None)) * per_slot
    if workers > grid.x * grid.y:
        raise ValueError("expert slots do not fit the core grid")
    cores = ttnn.num_cores_to_corerangeset(workers, grid, row_wise=True)
    coords = [(i % grid.x, i // grid.x) for i in range(workers)]
    assignment = [(i // per_slot, (i % per_slot) * nper) for i in range(workers)]
    memory = memory_config or ttnn.L1_MEMORY_CONFIG
    output = ttnn.allocate_tensor_on_device(ttnn.Shape((1, slots, 32, half * 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                            device, memory)
    shared_output = output if shared is None else ttnn.allocate_tensor_on_device(
        ttnn.Shape((1, 1, 32, half * 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory)
    shared_weight = weight if shared is None else shared
    ids_bytes = slots * 2
    w_bytes = TILE_BYTES[weight.dtype.name]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[kt, nt, nper, kb, ids_bytes] + _accessor(ttnn, x) + _accessor(ttnn, ids)
        + _accessor(ttnn, weight) + _accessor(ttnn, shared_weight),
        runtime_args=[(c, [x.buffer_address(), ids.buffer_address(), weight.buffer_address(), s, j,
                           shared_weight.buffer_address(), int(s == slots)])
                      for (s, j), c in zip(assignment, coords)],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=[nper, half] + _accessor(ttnn, output) + _accessor(ttnn, shared_output),
        runtime_args=[(c, [output.buffer_address(), s, j, shared_output.buffer_address(), int(s == slots)])
                      for (s, j), c in zip(assignment, coords)],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=[kt, nper, kb], runtime_args=[(c, []) for c in coords],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False,
                                            fp32_dest_acc_en=fp32_dest))
    cbs = [_cb(ttnn, cores, 0, x.dtype, 2 * kb, TILE_BYTES[x.dtype.name]),
           _cb(ttnn, cores, 1, weight.dtype, 4 * kb * nper, w_bytes),
           _cb(ttnn, cores, 2, ttnn.uint16, 1, max(64, ids_bytes)),
           _cb(ttnn, cores, 16, ttnn.bfloat16, nper, 2048)]
    tensors = [x, ids, weight, output] if shared is None else [x, ids, weight, shared, output, shared_output]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return output if shared is None else (output, shared_output)
