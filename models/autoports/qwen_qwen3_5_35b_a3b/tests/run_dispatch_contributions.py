# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise hot-expert native M-block loops against bounded correct calls.

This checks dispatch/contribution mechanics, not HF precision acceptance. The
large call must reproduce independent128-row calls, including the final rows,
under real weights, changing route order and nonuniform per-token scores.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import ttnn
from transformers import AutoConfig

from ..tt.optimized_decoder import OptimizedDecoder
from .layer_weights import load_layer_state_dict
from .run_layer import pcc
from .runtime_audit import device_only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-activations",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dtype",choices=("bfp4","bfp8"),default="bfp4")
    parser.add_argument("--input-dtype",choices=("bf16","bfp8"),default="bf16")
    args = parser.parse_args()
    checkpoint = Path("../weights/Qwen3.5-35B-A3B")
    cfg = AutoConfig.from_pretrained(checkpoint,local_files_only=True).text_config
    state = load_layer_state_dict(checkpoint,3)
    fixture = torch.load(args.input_activations,weights_only=True)
    assert fixture["metadata"]["layer"] == 3
    torch.manual_seed(902)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1,1))
    results = []
    try:
        model = OptimizedDecoder.from_state_dict(state,hf_config=cfg,layer_idx=3,mesh_device=mesh,
            optimization=dict(expert_backend="sparse",sparse_indexed=True,prefill_backend="dispatch",
                              dispatch_dtype=args.dtype,dispatch_input_dtype=args.input_dtype,
                              dispatch_grid=(4,4),prefill_chunk=4096,release_unused_weights=True))
        dispatcher = model.layer.feed_forward.module.experts.dispatch
        def put(x,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x.contiguous(),device=mesh,dtype=dtype,layout=layout,
                                  mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        def get(x):
            return ttnn.to_torch(x,mesh_composer=ttnn.ConcatMeshToTensor(mesh,dim=0))
        for length in (512,1024,2048,4096):
            dispatcher.policy = replace(dispatcher.policy,prefill_chunk=length)
            dispatcher.prepare_maps(mesh)
            for case in ("hot","changed_routes"):
                pool = torch.tensor([0,1,15,31,127,128,254,255])
                ids = pool[torch.arange(length)[:,None].add(torch.arange(8)[None,:])%8]
                if case=="changed_routes":
                    ids[length//2:] = (ids[length//2:]+67)%256
                scores = torch.randn(length,8).softmax(-1).bfloat16()
                source = fixture["activations"][:,:length].reshape(1,1,length,2048)
                x, values = put(source),put(scores.reshape(1,1,length,8),layout=ttnn.ROW_MAJOR_LAYOUT)
                indices = put(ids.reshape(1,1,length,8),ttnn.uint16,ttnn.ROW_MAJOR_LAYOUT)
                with device_only():
                    large = dispatcher(x,(values,indices))
                actual = get(large)
                small = []
                for offset in range(0,length,128):
                    with device_only():
                        output = dispatcher(x[:,:,offset:offset+128,:],
                            (values[:,:,offset:offset+128,:],indices[:,:,offset:offset+128,:]))
                    small.append(get(output))
                expected = torch.cat(small,dim=2)
                by_block = [pcc(expected[:,:,i:i+128],actual[:,:,i:i+128]) for i in range(0,length,128)]
                record = dict(length=length,case=case,min_block_pcc=min(by_block),
                              last_block_pcc=by_block[-1],max_abs_error=float((actual-expected).abs().max()),
                              native_input_m_tiles=length//32,runtime_audit=True)
                results.append(record)
                print(record,flush=True)
                assert min(by_block) >= 0.999
        args.output.write_text(json.dumps(dict(dtype=args.dtype,input_dtype=args.input_dtype,
            input_metadata=fixture["metadata"],reference="independent128-row native calls",
            cases=results),indent=2)+"\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
