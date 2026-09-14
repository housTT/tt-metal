# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "fused,row_major,modulo",
    [(False, False, None), (False, False, 1024), (True, False, None), (True, True, None)],
    ids=["update-absolute", "update-modulo", "fused-tiled", "fused-row-major"],
)
def test_paged_cache_sparse_virtual_table(device, fused, row_major, modulo):
    """Late virtual pages map into a smaller pool; inactive slots and other rows stay intact."""
    torch.manual_seed(0)
    num_users, num_heads, block_size, head_dim = 4, 2, 64, 128
    num_blocks, table_width = 8, 64
    num_caches = 2 if fused else 1
    layout = ttnn.ROW_MAJOR_LAYOUT if row_major else ttnn.TILE_LAYOUT
    padded_heads = 8 if row_major else 32
    grid_size = device.compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(i % grid_size.x, i // grid_size.x) for i in range(num_users * num_caches)]

    references = [torch.randn(num_blocks, num_heads, block_size, head_dim).bfloat16() for _ in range(num_caches)]
    caches = [ttnn.from_torch(ref, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for ref in references]
    # Every entry is an in-bounds physical ID, including the inactive/default entries.
    # The touched virtual columns are beyond the physical pool size in both modes.
    columns = [61, 62, 63] if modulo is None else [13, 14, 15]
    for iteration, physical_blocks in enumerate(([7, 2, 5], [1, 6, 3])):
        table = torch.zeros((num_users, table_width), dtype=torch.int32)
        positions = [column * block_size + offset for column, offset in zip(columns, [0, 31, 63])]
        if modulo is not None:
            positions = [position + 3 * modulo for position in positions]
        positions.append(-1)
        for user, (column, block) in enumerate(zip(columns, physical_blocks)):
            table[user, column] = block
        table_tt = ttnn.from_torch(table, dtype=ttnn.int32, device=device)
        positions_tt = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32, device=device)
        inputs = []
        for cache_index in range(num_caches):
            x = torch.randn(1, num_users, num_heads, head_dim).bfloat16()
            padded = torch.nn.functional.pad(x, (0, 0, 0, padded_heads - num_heads))
            shard_cores = cores[cache_index * num_users : (cache_index + 1) * num_users]
            shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in shard_cores])
            memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(shard_grid, [padded_heads, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
            )
            inputs.append(
                ttnn.from_torch(padded, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)
            )
            for user, block in enumerate(physical_blocks):
                references[cache_index][block, :, positions[user] % block_size, :] = x[0, user]

        if fused:
            ttnn.experimental.paged_fused_update_cache(
                caches[0], inputs[0], caches[1], inputs[1], update_idxs_tensor=positions_tt, page_table=table_tt
            )
        else:
            ttnn.experimental.paged_update_cache(
                caches[0],
                inputs[0],
                update_idxs_tensor=positions_tt,
                page_table=table_tt,
                cache_position_modulo=modulo,
            )
        # Read the full physical pool, including page zero and all untouched token rows.
        for cache_index, (cache, reference) in enumerate(zip(caches, references)):
            actual = ttnn.to_torch(cache)
            assert torch.equal(actual, reference), f"cache {cache_index}, dispatch {iteration}"


def test_paged_cache_modulo_must_fit_virtual_table(device, expect_error):
    """A large physical pool cannot make a too-short virtual table addressable."""
    cache = ttnn.from_torch(torch.zeros(32, 1, 64, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(
        torch.zeros(1, 1, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    table = ttnn.from_torch(torch.zeros(1, 8, dtype=torch.int32), dtype=ttnn.int32, device=device)
    positions = ttnn.from_torch(torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, device=device)
    with expect_error(RuntimeError, "must be <= max_num_blocks_per_seq"):
        ttnn.experimental.paged_update_cache(
            cache, x, update_idxs_tensor=positions, page_table=table, cache_position_modulo=1024
        )
