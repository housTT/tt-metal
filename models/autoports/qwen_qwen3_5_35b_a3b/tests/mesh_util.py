# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os


def open_mesh(size, trace_region_size=0):
    import ttnn

    if size == 1:
        kwargs = {"physical_device_ids": [int(os.environ["QDEV"])]} if "QDEV" in os.environ else {}
        return ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=trace_region_size, **kwargs)
    if size == 2 and len(os.environ.get("TT_VISIBLE_DEVICES", "").split(",")) != 2:
        raise RuntimeError("A 1x2 mesh needs TT_VISIBLE_DEVICES naming the two chips of one card")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if size == 4 else ttnn.FabricConfig.FABRIC_1D)
    return ttnn.open_mesh_device(ttnn.MeshShape(1, size), trace_region_size=trace_region_size)


def shards(tensor):
    import ttnn

    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


class ReplicaReader:

    def __init__(self):
        self.identical = []

    def __call__(self, tensor):
        import torch

        values = shards(tensor)
        self.identical.append(all(torch.equal(values[0], value) for value in values[1:]))
        return values[0]

    def all_identical(self):
        return all(self.identical)


def decoder_class(name):
    if name == "multichip":
        from ..tt.multichip_decoder import MultichipDecoder

        return MultichipDecoder
    if name == "optimized":
        from ..tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder
    if name == "fused":
        from ..tt.fused_decoder import FusedDecoder

        return FusedDecoder
    from ..tt.functional_decoder import FunctionalDecoder

    return FunctionalDecoder
