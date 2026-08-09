import torch

import ttnn

m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
W = 64


def check(logical, padded, dtype, label):
    x = torch.arange(1, logical * W + 1, dtype=torch.float32).reshape(1, 1, logical, W)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    p = ttnn.pad(t, [(0, 0), (0, 0), (0, padded - logical), (0, 0)], 0.0)
    back = ttnn.to_torch(p).to(torch.float32)
    ok_shape = tuple(back.shape) == (1, 1, padded, W)
    head_ok = torch.equal(back[0, 0, :logical], x[0, 0]) if ok_shape else False
    tail = back[0, 0, logical:] if ok_shape else torch.tensor([float("nan")])
    tail_zero = bool((tail == 0).all()) if ok_shape else False
    tile_pad = -(-logical // 32) * 32
    print(
        f"{label} logical={logical} padded={padded} tilepad={tile_pad} "
        f"shape={tuple(back.shape)} head_ok={head_ok} tail_all_zero={tail_zero} "
        f"tail_absmax={float(tail.abs().max()) if ok_shape else 'NA'}",
        flush=True,
    )
    ttnn.deallocate(t)
    ttnn.deallocate(p)


for dt, name in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
    check(743, 768, dt, name)  # tilepad == padded  -> suspected broken
    check(646, 768, dt, name)  # tilepad  < padded
    check(1519, 1536, dt, name)
    check(1422, 1472, dt, name)
    check(17, 64, dt, name)
    check(33, 64, dt, name)  # tilepad 64 == padded

# also: slice-then-pad, which is what the layer actually does
print("--- slice(0:logical) then pad ---", flush=True)


def check_slice(total, logical, padded, label):
    x = torch.arange(1, total * W + 1, dtype=torch.float32).reshape(1, 1, total, W)
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    s = ttnn.slice(t, [0, 0, 0, 0], [1, 1, logical, W])
    p = ttnn.pad(s, [(0, 0), (0, 0), (0, padded - logical), (0, 0)], 0.0)
    back = ttnn.to_torch(p).to(torch.float32)
    head_ok = torch.equal(back[0, 0, :logical], x[0, 0, :logical])
    tail = back[0, 0, logical:]
    print(
        f"{label} total={total} logical={logical} padded={padded} shape={tuple(back.shape)} "
        f"head_ok={head_ok} tail_all_zero={bool((tail==0).all())} tail_absmax={float(tail.abs().max())}",
        flush=True,
    )


check_slice(768, 743, 768, "slicepad")
check_slice(1536, 1519, 1536, "slicepad")
check_slice(768, 646, 768, "slicepad")
ttnn.close_mesh_device(m)
