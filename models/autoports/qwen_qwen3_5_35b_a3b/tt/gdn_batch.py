# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from .gdn_decode import CB_TILES, GATE_CBS, KERNEL_ROOT, _accessor, _bits, _cbs, _compute_config

BATCH_ROOT = "models/autoports/qwen_qwen3_5_35b_a3b/tt/kernels/gdn_batch"
DECODE_CBS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18)


def _cb(ttnn, cores, index, tiles, dtype):
    size = 4096 if dtype == ttnn.float32 else 2048
    return ttnn.CBDescriptor(total_size=tiles * size, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype,
                                                                         page_size=size)])


def _coords(count, grid):
    return [(i % grid.x, i // grid.x) for i in range(count)]


def _split(total, parts):
    base, extra = divmod(total, parts)
    starts, start = [], 0
    for i in range(parts):
        n = base + (1 if i < extra else 0)
        starts.append((start, n))
        start += n
    return starts


def conv_batch(mega, taps, conv_states, width):
    import ttnn

    users = len(conv_states)
    tiles = width // 32
    if width % 32 or any(t.layout != ttnn.TILE_LAYOUT for t in taps):
        raise ValueError("batched conv needs tile taps and a 32-aligned width")
    device = mega.device()
    grid = device.compute_with_storage_grid_size()
    workers = min(tiles, grid.x * grid.y)
    cores = ttnn.num_cores_to_corerangeset(workers, grid, row_wise=True)
    coords = _coords(workers, grid)
    spans = _split(tiles, workers)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape((1, users, width)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                            ttnn.L1_MEMORY_CONFIG)
    states = [s.buffer_address() for s in conv_states]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/conv_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[users] + sum((_accessor(ttnn, t) for t in (mega, *taps, conv_states[0])), []),
        runtime_args=[(c, [mega.buffer_address()] + [t.buffer_address() for t in taps] + [s, n] + states)
                      for (s, n), c in zip(spans, coords)],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/conv_writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=_accessor(ttnn, output),
        runtime_args=[(c, [output.buffer_address(), s, n]) for (s, n), c in zip(spans, coords)],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/conv_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=[], runtime_args=[(c, [n]) for (s, n), c in zip(spans, coords)],
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                            fp32_dest_acc_en=True))
    cbs = [_cb(ttnn, cores, 0, 2, ttnn.bfloat16), _cb(ttnn, cores, 1, 6, ttnn.bfloat16),
           _cb(ttnn, cores, 2, 8, ttnn.bfloat16), _cb(ttnn, cores, 3, 1, ttnn.bfloat16),
           _cb(ttnn, cores, 4, 2, ttnn.bfloat16),
           ttnn.CBDescriptor(total_size=max(users, 1) * 3 * 64, core_ranges=cores,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=6, data_format=ttnn.bfloat16,
                                                                         page_size=max(users, 1) * 3 * 64)])]
    ttnn.generic_op([mega, *taps, *conv_states, output], ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    return output


def gdn_decode_batch(qkv, mega, states, norm_weight, dt_bias, a_neg, *, q_dim, k_dim, groups, a_start, b_start,
                     z_offset, l2_eps=1e-6, norm_eps=1e-6, raw=False):
    import ttnn

    users = len(states)
    heads, head_k, head_v = states[0].shape[1], states[0].shape[-2], states[0].shape[-1]
    device = qkv.device()
    grid = device.compute_with_storage_grid_size()
    per_head = 4
    base = heads * per_head
    if base > grid.x * grid.y:
        raise ValueError(f"batched GDN decode needs {base} cores")
    group_count = max(1, min(users, (grid.x * grid.y) // base))
    user_spans = [s for s in _split(users, group_count) if s[1]]
    workers = base * len(user_spans)
    cores = ttnn.num_cores_to_corerangeset(workers, grid, row_wise=True)
    coords = _coords(workers, grid)
    assignment = [(i % base // per_head, i % base % per_head, user_spans[i // base]) for i in range(workers)]
    output = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, users, heads * head_v)), ttnn.float32,
                                            ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    addresses = [s.buffer_address() for s in states]
    shape_key = [users, max(n for _, n in user_spans)]
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[groups, q_dim // 32, k_dim // 32]
        + sum((_accessor(ttnn, t) for t in (qkv, mega, dt_bias, a_neg, states[0])), []) + shape_key,
        runtime_args=[(c, [qkv.buffer_address(), mega.buffer_address(), dt_bias.buffer_address(),
                           a_neg.buffer_address(), h, b, a_start + h, b_start + h, first, n] + addresses[first:first + n])
                      for (h, b, (first, n)), c in zip(assignment, coords)],
        config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores, compile_time_args=_accessor(ttnn, states[0]) + _accessor(ttnn, output) + shape_key,
        runtime_args=[(c, [output.buffer_address(), h, b, first, n] + addresses[first:first + n])
                      for (h, b, (first, n)), c in zip(assignment, coords)],
        config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[_bits(l2_eps), _bits(head_k ** -0.5), _bits(1.0 / head_v), _bits(norm_eps), 1, 1],
        runtime_args=[(c, [n]) for (h, b, (first, n)), c in zip(assignment, coords)],
        config=_compute_config(ttnn))
    cbs = _cbs(ttnn, cores, DECODE_CBS) + [_cb(ttnn, cores, 20, 11, ttnn.bfloat16)]
    ttnn.generic_op([qkv, mega, dt_bias, a_neg, *states, output], ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute], semaphores=[], cbs=cbs))

    gated = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, users, heads * head_v)), ttnn.bfloat16,
                                           ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    head_cores = ttnn.num_cores_to_corerangeset(heads, grid, row_wise=True)
    head_coords = _coords(heads, grid)
    gate_reader = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/gate_reader.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=head_cores,
        compile_time_args=_accessor(ttnn, output) + _accessor(ttnn, mega) + _accessor(ttnn, norm_weight),
        runtime_args=[(c, [output.buffer_address(), mega.buffer_address(), norm_weight.buffer_address(), h, z_offset])
                      for h, c in enumerate(head_coords)],
        config=ttnn.ReaderConfigDescriptor())
    gate_writer = ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_ROOT}/gate_writer.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=head_cores, compile_time_args=_accessor(ttnn, gated),
        runtime_args=[(c, [gated.buffer_address(), h]) for h, c in enumerate(head_coords)],
        config=ttnn.WriterConfigDescriptor())
    gate_compute = ttnn.KernelDescriptor(
        kernel_source=f"{BATCH_ROOT}/gate_compute.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=head_cores, compile_time_args=[_bits(1.0 / head_v), _bits(norm_eps)],
        runtime_args=[(c, []) for c in head_coords], config=_compute_config(ttnn))
    ttnn.generic_op([output, mega, norm_weight, gated], ttnn.ProgramDescriptor(
        kernels=[gate_reader, gate_writer, gate_compute], semaphores=[], cbs=_cbs(ttnn, head_cores, GATE_CBS)))
    if raw:
        return gated, output
    output.deallocate(True)
    return gated
