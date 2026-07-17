# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare the two common on-device samplers against Kokoro's readout, and the
selected on-device greedy argmax path. Writes sampler_comparison.json.

Both common implementations — models/common/sampling (TTTv1: TTSampling) and
models/common/modules/sampling/sampling_1d.py (TTTv2: Sampling1D) — target a
LARGE vocabulary SHARDED across the mesh and perform a cross-device vocab
all-gather (dim=3) before top-k/argmax. Kokoro's reconstruction readout is a
tiny (178) REPLICATED vocab on the (1,4) mesh, so both would (a) add an
unnecessary cross-device all-gather and (b) misinterpret the 4 replicated copies
as 4 vocab shards. Stock ``ttnn.argmax`` on the replicated logits is the
semantically-greedy, lowest-movement path — it is what tt/model.py uses.
"""
import json
import time
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).parent
V = 192  # tile-padded readout vocab (178 real + pad)
S = 128


def bench(fn, iters=50):
    ttnn.synchronize_device(bench.mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(bench.mesh)
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    out = {"selected": "ttnn.argmax (on replicated readout logits)", "candidates": {}}
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    bench.mesh = mesh
    try:
        torch.manual_seed(0)
        logits_t = torch.randn(1, 1, S, V)
        logits_t[:, :, :, 178:] = -1e4  # mask pad columns (as the model does)
        for i in range(S):
            logits_t[0, 0, i, i % 178] += 8.0  # unambiguous per-row winner (no bf16 tie)
        logits = ttnn.from_torch(
            logits_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # selected: on-device argmax over the vocab dim
        ttnn.argmax(logits, dim=-1, keepdim=False)  # warm
        ms_argmax = bench(lambda: ttnn.argmax(logits, dim=-1, keepdim=False))
        # correctness: matches host argmax
        host = logits_t[0, 0].argmax(-1)
        dev = ttnn.to_torch(
            ttnn.argmax(logits, dim=-1, keepdim=False), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
        )
        dev = dev.reshape(mesh.get_num_devices(), S)[0]
        out["candidates"]["ttnn_argmax"] = {
            "latency_ms": round(ms_argmax, 4),
            "greedy_correct_vs_host": bool(torch.equal(dev.long(), host.long())),
            "cross_device_all_gather": False,
            "verdict": "SELECTED — semantically greedy, no cross-device movement, tile-shaped.",
        }

        # candidate 1: Sampling1D (TTTv2) — designed for sharded vocab + AG dim=3
        try:
            from models.common.modules.sampling.sampling_1d import Sampling1D

            samp = Sampling1D(vocab_size=178, mesh_device=mesh, max_top_k=32)
            _ = samp  # constructed
            out["candidates"]["sampling_1d_tttv2"] = {
                "constructed": True,
                "cross_device_all_gather": True,
                "verdict": (
                    "REJECTED for greedy: expects per-device vocab shards and all-gathers over dim=3 "
                    "(cluster_shape=[1,4]); Kokoro's readout is a tiny (178) REPLICATED vocab, so the "
                    "gather would concat 4 identical copies and add movement for no benefit. Retained "
                    "as the top-k/top-p path if sampled TTS-token selection is ever needed (vocab is "
                    "tiny so it can run single-device on a (1,1) submesh)."
                ),
            }
        except Exception as e:  # noqa: BLE001
            out["candidates"]["sampling_1d_tttv2"] = {"constructed": False, "error": str(e)[:300]}

        # candidate 2: TTSampling (TTTv1) — same sharded-vocab / cluster_shape contract
        out["candidates"]["tt_sampling_tttv1"] = {
            "verdict": (
                "REJECTED for greedy: same contract as TTTv2 (requires vocab sharded across the mesh "
                "cluster axis + cross-device all-gather, needs args.cluster_shape/tt_ccl and a "
                "SamplingGenerator wrapper). Heavier than TTTv2 for our tiny replicated vocab; TTTv2 "
                "is the lighter of the two if a sampled path is needed."
            ),
        }

        (HERE / "sampler_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        print("SAMPLER_CMP_OK")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
