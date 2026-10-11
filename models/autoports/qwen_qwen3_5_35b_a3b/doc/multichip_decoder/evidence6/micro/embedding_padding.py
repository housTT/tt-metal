import torch
import ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[3])
table = torch.full((96, 2048), float("nan")).to(torch.bfloat16)
table[:64] = torch.randn(64, 2048).to(torch.bfloat16)
index = torch.tensor([[0, 5, 95, 63, 95, 1, 95, 2] * 4], dtype=torch.int32)
t = ttnn.from_torch(table, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
i = ttnn.from_torch(index, device=mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
for kind in (None, ttnn.EmbeddingsType.PADDED, ttnn.EmbeddingsType.GENERIC):
    kw = {} if kind is None else {"embeddings_type": kind}
    try:
        out = ttnn.to_torch(ttnn.embedding(i, t, padding_idx=95, layout=ttnn.ROW_MAJOR_LAYOUT, **kw)).float().reshape(32, 2048)
        pad_rows = out[[2, 4, 6]]
        real = out[[0, 1, 3]]
        expected = table[[0, 5, 63]].float()
        print("PADDING", kind, "pad_rows_zero", bool((pad_rows == 0).all()), "pad_finite", bool(torch.isfinite(pad_rows).all()),
              "real_exact", bool(torch.equal(real, expected)), flush=True)
    except Exception as error:
        print("PADDING", kind, "error", str(error).splitlines()[0][:200], flush=True)
ttnn.close_mesh_device(mesh)
