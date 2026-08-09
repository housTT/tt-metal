import torch

import ttnn

m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
W = 5120


def check(logical, padded):
    x = torch.randn(1, 1, logical, W, dtype=torch.float32)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=m, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    s = ttnn.slice(t, [0, 0, 0, 0], [1, 1, logical, W])
    same_obj = s.buffer_address() == t.buffer_address()
    p = ttnn.pad(s, [(0, 0), (0, 0), (0, padded - logical), (0, 0)], 0.0)
    alias = p.buffer_address() == s.buffer_address()
    ttnn.deallocate(s)  # <-- what _pad_seq does
    junk = ttnn.from_torch(
        torch.full((1, 1, padded, W), 7.0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=m,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    back = ttnn.to_torch(p).to(torch.float32)
    head_ok = torch.allclose(back[0, 0, :logical], x[0, 0].to(torch.bfloat16).to(torch.float32), atol=1e-2)
    print(
        f"logical={logical} padded={padded} padamt={padded-logical} slice_is_alias={same_obj} "
        f"pad_is_alias={alias} head_ok_after_dealloc={head_ok} absmax={float(back.abs().max()):.4f}",
        flush=True,
    )
    ttnn.deallocate(p)
    ttnn.deallocate(junk)


for lg, pd in ((737, 768), (743, 768), (767, 768), (736, 768), (735, 768), (646, 768)):
    check(lg, pd)
ttnn.close_mesh_device(m)
