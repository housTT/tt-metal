"""Compare semantically greedy native candidate sampling and force argmax."""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh


def main():
    mesh = open_ornith_mesh()
    report = []
    try:
        model = OrnithModel(None, mesh, layer_indices=[], max_context=2048)
        torch.manual_seed(987)
        logits = model.terminal(model.upload(torch.randn(1, 32, 4096, dtype=torch.bfloat16)))
        oracle = model.logits_to_host(logits, 32).argmax(-1)
        for force in [False, True]:
            sampler = model.build_sampler(force_argmax=force)
            tokens = model.upload(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            print("SAMPLER_START", force, flush=True)
            sampler.sample(logits, tt_out_tok=tokens, enable_trace=False)
            eager = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).flatten().long()
            sampler.sample(logits, tt_out_tok=tokens, enable_trace=True)
            sampled = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0]).flatten().long()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(32):
                sampler.sample(logits, tt_out_tok=tokens, enable_trace=True)
            ttnn.synchronize_device(mesh)
            item = dict(
                force_argmax=force,
                semantic_params=dict(k=1, p=0, temp=1),
                physical_max_top_k=32,
                eager_matches_cpu=bool(torch.equal(eager, oracle)),
                trace_matches_cpu=bool(torch.equal(sampled, oracle)),
                trace_ms=(time.perf_counter() - start) * 1000 / 32,
            )
            print(json.dumps(item), flush=True)
            report.append(item)
            assert item["eager_matches_cpu"] and item["trace_matches_cpu"]
            sampler.reset_trace()
        (Path(__file__).parent / "sampler_greedy_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print("SAMPLER_PROBE_OK", flush=True)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
