# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BF16 head fidelity comparison on recorded real hidden; decoder policy unchanged."""
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh

DOC = Path(__file__).resolve().parent
mesh = open_ornith_mesh()
report = []
try:
    model = OrnithModel(None, mesh, layer_indices=[], max_context=2048)
    source = DOC.parent / "full_model/french_head_v1/hidden.pt"
    saved = torch.load(source, weights_only=True)
    hidden = saved["hidden"][0]
    x = model.upload(hidden)
    baseline = None
    for fidelity in ["HiFi4", "HiFi2", "LoFi", "LoFi", "HiFi2", "HiFi4"]:
        model.sharded_final_norm = True
        model.head_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        y = model.terminal(x)
        scores = model.logits_to_host(y, 1)
        if baseline is None:
            baseline = scores
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = model.terminal(x)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        assert torch.equal(scores, model.logits_to_host(out, 1))
        start = time.perf_counter()
        for _ in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        report.append(
            dict(
                fidelity=fidelity,
                ms=(time.perf_counter() - start) * 1000 / 64,
                exact_baseline=bool(torch.equal(scores, baseline)),
                max_abs=float((scores - baseline).abs().max()),
                greedy_equal=bool(torch.equal(scores.argmax(-1), baseline.argmax(-1))),
            )
        )
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(y)
        ttnn.deallocate(out)
        print(report[-1], flush=True)
finally:
    close_ornith_mesh(mesh)
    (DOC / "head_fidelity_probe.json").write_text(json.dumps(report, indent=2) + "\n")
