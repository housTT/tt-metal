import sys, torch, ttnn
chips = int(sys.argv[1])
ttnn.set_fabric_config(getattr(ttnn.FabricConfig, sys.argv[2]))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, chips), trace_region_size=0)
x = ttnn.from_torch(torch.ones((chips, 32, 2048), dtype=torch.bfloat16), device=mesh, dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
y = ttnn.all_reduce(x, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=1)
print("PROBE", list(mesh.get_device_ids()), [ttnn.to_torch(t).float().mean().item() for t in ttnn.get_device_tensors(y)], flush=True)
ttnn.close_mesh_device(mesh)
