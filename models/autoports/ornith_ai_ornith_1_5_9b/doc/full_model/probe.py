"""Reduced real-layer terminal/trace probe. Run as a module from the repo root."""

import argparse
import time

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-layers", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    mesh = open_ornith_mesh()
    try:
        model = OrnithModel(None, mesh, layer_indices=None if args.all_layers else [0, 3], max_context=2048)
        cache = model.allocate_cache(context=2048)
        table = model.page_table(cache)
        print("MODEL_READY", time.perf_counter() - start, flush=True)
        sampler = model.build_sampler()
        tokens = model.upload(
            torch.zeros((1, 1, 1, 32), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pos = model.upload(torch.tensor([131], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rot = model.upload(torch.tensor([[131]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page = model.upload(table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        logits = model.prefill_forward([list(range(131))], page_table=table, kv_cache=cache, prompt_lens=[131])
        print("PREFILL_OK", list(logits.shape), flush=True)
        sampler.sample(logits, tt_out_tok=tokens, enable_trace=False)
        print("SAMPLING_OK", ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]), flush=True)
        out = model.decode_forward(tokens, current_pos=pos, rot_idxs=rot, page_table=page, kv_cache=cache)
        print("DECODE_OK", list(out.shape), flush=True)
        sampler.sample(out, tt_out_tok=tokens, enable_trace=False)
        ttnn.synchronize_device(mesh)
        print("PROBE_OK", time.perf_counter() - start, flush=True)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
