# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-topology persistent output trial adapted to caller-owned temporary lifetimes."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import (
    OrnithModel,
    SamplingCCL,
    close_ornith_mesh,
    open_ornith_mesh,
)


class PersistentCCL(SamplingCCL):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.buffers = {}
        self.enabled = False

    def line_all_gather(self, tensor, *, dim, cluster_axis=None, memory_config=None, num_links=None, buffer_key=None):
        key = (buffer_key, tuple(tensor.shape), str(tensor.dtype), dim, str(memory_config))
        kwargs = dict(
            dim=dim,
            multi_device_global_semaphore=self.ccl.get_ag_ping_pong_semaphore(),
            barrier_semaphore=self.ccl.get_barrier_semaphore(),
            num_links=1,
            topology=ttnn.Topology.Ring,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
        if not self.enabled:
            # Keep the original nonpersistent control after the model default
            # has adopted persistence; delegating to super would retest it.
            return ttnn.experimental.all_gather_async(tensor, persistent_output_buffer=None, **kwargs)
        if key not in self.buffers:
            self.buffers[key] = ttnn.experimental.all_gather_async(tensor, persistent_output_buffer=None, **kwargs)
        else:
            ttnn.experimental.all_gather_async(tensor, persistent_output_buffer=self.buffers[key], **kwargs)
        # Common sampling and embedding consumers free their outputs. Return a disposable
        # copy, retaining the preallocated CCL buffer independently of caller lifetimes.
        return ttnn.clone(self.buffers[key], memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = open_ornith_mesh()
    report = []
    try:
        model = OrnithModel(None, mesh, layer_indices=[], max_context=2048)
        model.ccl = PersistentCCL(mesh)
        torch.manual_seed(987)
        logits = model.terminal(model.upload(torch.randn(1, 32, 4096, dtype=torch.bfloat16)))
        oracle = model.logits_to_host(logits, 32).argmax(-1)
        tokens = model.upload(
            torch.zeros(1, 1, 1, 32, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ids = model.upload(torch.tensor([[123]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for mode in ("sampling", "embedding"):
            baseline = None
            for enabled in (False, True, False, True):
                model.ccl.enabled = enabled
                sampler = model.build_sampler(force_argmax=False) if mode == "sampling" else None

                def forward():
                    if sampler is not None:
                        sampler.sample(logits, tt_out_tok=tokens, enable_trace=False)
                        return tokens
                    return model.embed(ids)

                warm = forward()
                if sampler is None:
                    ttnn.deallocate(warm)
                eager = forward()
                expected = ttnn.to_torch(ttnn.get_device_tensors(eager)[0])
                if mode == "sampling":
                    assert torch.equal(expected.flatten().long(), oracle)
                if baseline is None:
                    baseline = expected
                assert torch.equal(expected, baseline)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                start = time.perf_counter()
                for _ in range(128):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed = (time.perf_counter() - start) * 1000 / 128
                for rank in ttnn.get_device_tensors(output):
                    assert torch.equal(ttnn.to_torch(rank), expected)
                item = dict(
                    mode=mode,
                    persistent=enabled,
                    trace_ms=elapsed,
                    exact_all_ranks=True,
                    temporary_ownership_copy=enabled,
                    input_shape=list(logits.shape if mode == "sampling" else ids.shape),
                    buffers=[str(k) for k in model.ccl.buffers],
                )
                print(item, flush=True)
                report.append(item)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                ttnn.release_trace(mesh, trace)
                if sampler is not None:
                    sampler.reset_trace()
                else:
                    ttnn.deallocate(eager)
                    ttnn.deallocate(output)
    finally:
        close_ornith_mesh(mesh)


if __name__ == "__main__":
    main()
