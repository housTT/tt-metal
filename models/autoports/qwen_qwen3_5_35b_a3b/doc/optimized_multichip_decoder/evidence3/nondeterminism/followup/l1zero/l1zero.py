KERNEL = "/home/hous/dev/qwen3.5-35b-a3b/runs/main/stage05/micro/nondeterminism/l1zero/zero.cpp"


def l1_zero(mesh, nbytes):
    import ttnn

    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    kernel = ttnn.KernelDescriptor(kernel_source=KERNEL, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                   core_ranges=cores, compile_time_args=[nbytes],
                                   runtime_args=[((x, y), []) for y in range(grid.y) for x in range(grid.x)],
                                   config=ttnn.ReaderConfigDescriptor())
    cb = ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])
    dummy = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                                           ttnn.DRAM_MEMORY_CONFIG)
    other = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh,
                                           ttnn.DRAM_MEMORY_CONFIG)
    ttnn.generic_op([dummy, other], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb]))
    dummy.deallocate(True)
    other.deallocate(True)
