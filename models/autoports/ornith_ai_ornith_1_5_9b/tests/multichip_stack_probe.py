# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decoder-only composition check against the optimized single-chip stack.

No embedding, final norm, logits, generator or serving path is implemented.
Each decoder output feeds the next decoder without a boundary conversion.
"""

import json

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder, fabric_router_config
from ..tt.optimized_decoder import OptimizedDecoder
from . import test_functional_decoder as H
from .test_optimization_experiments import recorded_activations


def main():
    cfg = H.hf_config()
    layers = [0, 3, 0]
    weights = {i: H.layer_state_dict(i, "real") for i in set(layers)}
    inputs = recorded_activations(0)
    results = {}
    for name, count, cls in [("baseline", 1, OptimizedDecoder), ("tp4", 4, MultichipDecoder)]:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.DISABLED if count == 1 else ttnn.FabricConfig.FABRIC_1D_RING,
            router_config=fabric_router_config() if count == 4 else ttnn.FabricRouterConfig(),
        )
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, count), trace_region_size=32 * 1024 * 1024, l1_small_size=24576)
        try:
            decoders, tables = [], []
            for index in layers:
                decoder = cls.from_state_dict(
                    weights[index], hf_config=cfg, mesh_device=mesh, layer_idx=index, max_context=1024
                )
                decoder.allocate_state(1)
                decoder.allocate_kv_cache(32)
                decoders.append(decoder)
                tables.append(
                    H.to_device(
                        mesh, torch.arange(32, dtype=torch.int32)[None], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                )

            def read(value):
                return [ttnn.to_torch(part) for part in ttnn.get_device_tensors(value)]

            def chain(value, mode, pos=None, rot=None):
                for index, (decoder, table) in enumerate(zip(decoders, tables)):
                    out = (
                        decoder.prefill_forward(value, page_table=table)
                        if mode == "prefill"
                        else decoder.decode_forward(value, current_pos=pos, rot_idxs=rot, page_table=table)
                    )
                    if index:
                        ttnn.deallocate(value)
                    value = out
                return value

            prompt = H.to_device(mesh, inputs[:, :131])
            out = chain(prompt, "prefill")
            prefill = read(out)
            ttnn.deallocate(out)
            token = H.to_device(mesh, inputs[:, 131:132])
            pos, rot = H.decode_inputs(mesh, torch.tensor([131]))
            buffers = [
                buf
                for d in decoders
                for buf in ([d.k_cache, d.v_cache] if d.is_full_attention else [d.recurrent_state, *d.conv_state])
            ]
            saved = [ttnn.clone(buf) for buf in buffers]

            def restore():
                for source, destination in zip(saved, buffers):
                    ttnn.copy(source, destination)

            eager = chain(token, "decode", pos, rot)
            first = read(eager)
            ttnn.deallocate(eager)
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced = chain(token, "decode", pos, rot)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            assert all(torch.equal(a, b) for a, b in zip(first, read(traced)))
            restore()
            decoded = []
            for step in range(8):
                for source, destination, dtype, layout in [
                    (inputs[:, 131 + step : 132 + step], token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                    (torch.tensor([131 + step]), pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    (torch.tensor([[131 + step]]), rot, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                ]:
                    ttnn.copy_host_to_device_tensor(ttnn.from_torch(source, dtype=dtype, layout=layout), destination)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                decoded.append(read(traced))
            results[name] = [prefill, decoded]
            ttnn.release_trace(mesh, trace)
            print(
                json.dumps(
                    dict(
                        name=name,
                        layers=layers,
                        logical_prefill=131,
                        trace_steps=8,
                        eager_trace_exact=True,
                        boundary_conversions=0,
                    )
                ),
                flush=True,
            )
        finally:
            ttnn.close_mesh_device(mesh)
    scores = [H.pcc(results["baseline"][0][0], part) for part in results["tp4"][0]]
    decode_scores = [
        [H.pcc(single[0], part) for part in parallel]
        for single, parallel in zip(results["baseline"][1], results["tp4"][1])
    ]
    assert min(scores + [value for step in decode_scores for value in step]) >= H.PCC_BAR
    print(json.dumps(dict(prefill_pcc=scores, decode_pcc=decode_scores, stack_contract=True)), flush=True)


if __name__ == "__main__":
    main()
