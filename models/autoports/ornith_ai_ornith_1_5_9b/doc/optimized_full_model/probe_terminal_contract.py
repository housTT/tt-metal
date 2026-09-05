# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct pad/common-head parity; archived run sources define historical baselines."""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.model import OrnithModel, close_ornith_mesh, open_ornith_mesh
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig

DOC = Path(__file__).resolve().parent
mesh = open_ornith_mesh()
report = []
try:
    model = OrnithModel(None, mesh, layer_indices=[], max_context=2048, sharded_final_norm=True)
    hidden = torch.load(DOC.parent / "full_model/french_head_v1/hidden.pt", weights_only=True)["hidden"][0]
    x = ttnn.to_memory_config(model.upload(hidden), model.final_norm_memory)
    # Preloaded LazyWeight is a supported cache slot; the source is used only for shape metadata.
    common = LMHead1D.from_config(
        LMHead1DConfig(
            output_weights=[LazyWeight(source=w, _value=w, dtype=ttnn.bfloat16) for w in model.head_weights],
            mesh_device=mesh,
            dim=model.dim,
            program_configs=[model.head_program] * len(model.head_weights),
            compute_kernel_config=model.head_compute,
            lm_head_dtype=ttnn.bfloat16,
            input_memcfg=model.head_input_memory,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            weights_memcfgs=[w.memory_config() for w in model.head_weights],
        )
    )

    def direct():
        flat = ttnn.reshape(x, [1, 1, 1, model.dim])
        flat = ttnn.pad(flat, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0, memory_config=model.final_norm_memory)
        normalized = ttnn.rms_norm(
            flat,
            weight=model.norm_weight,
            epsilon=model.hf_config.rms_norm_eps,
            program_config=model.final_norm_program,
            memory_config=model.final_norm_memory,
        )
        sharded = ttnn.to_memory_config(normalized, model.head_input_memory)
        logits = common(sharded)
        ttnn.deallocate(sharded)
        ttnn.deallocate(normalized)
        return logits

    baseline = None
    for name, forward in [
        ("existing", lambda: model.terminal(x)),
        ("direct_common", direct),
        ("existing", lambda: model.terminal(x)),
        ("direct_common", direct),
    ]:
        y = forward()
        scores = model.logits_to_host(y, 1)
        if baseline is None:
            baseline = scores
        assert torch.equal(scores, baseline), (scores - baseline).abs().max()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        start = time.perf_counter()
        for _ in range(64):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - start) * 1000 / 64
        assert torch.equal(model.logits_to_host(out, 1), baseline)
        report.append(
            dict(candidate=name, ms=ms, exact=True, input_shape=list(x.shape), padded_shape=list(x.padded_shape))
        )
        print(report[-1], flush=True)
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(y)
        ttnn.deallocate(out)
finally:
    close_ornith_mesh(mesh)
    (DOC / "terminal_contract_probe.json").write_text(json.dumps(report, indent=2) + "\n")
