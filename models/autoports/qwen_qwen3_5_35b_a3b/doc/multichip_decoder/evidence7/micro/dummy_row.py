import torch
import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[3])
capacity = 128
table = torch.full((1, 1, capacity, 2048), float("nan")).to(torch.bfloat16)
table[..., :64, :] = torch.randn(64, 2048).to(torch.bfloat16)
tile = ttnn.from_torch(table, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
zero = ttnn.from_torch(torch.zeros(1, 1, 1, 2048), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
rm = ttnn.to_layout(tile, ttnn.ROW_MAJOR_LAYOUT)
ttnn.experimental.slice_write(zero, rm, (0, 0, capacity - 1, 0), (1, 1, capacity, 2048), (1, 1, 1, 1))
index = torch.tensor([[0, 5, capacity - 1, 63] * 8], dtype=torch.int32)
i = ttnn.from_torch(index, device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
out = ttnn.to_torch(ttnn.embedding(i, ttnn.reshape(rm, (capacity, 2048)), layout=ttnn.ROW_MAJOR_LAYOUT)).float().reshape(32, 2048)
print("DUMMY", "zero_rows", bool((out[2::4] == 0).all()), "real_exact", bool(torch.equal(out[0::4], table[0, 0, 0].float().expand(8, -1))),
      "others_untouched", bool(torch.equal(ttnn.to_torch(rm).float()[0, 0, :64], table[0, 0, :64].float())), flush=True)
ttnn.close_mesh_device(mesh)
