"""Which op in the bfloat4_b routed-expert chain issues a host write under trace capture?

`ttnn.typecast` bfloat4_b -> bfloat8_b is capturable (probe_bfp4_widen.py), so the
"Writes are not supported during trace capture" that C19 hits must come from another op in the
chain whose *only* difference under C19 is that its operands are bfloat4_b.

This probe builds a real `MultichipMoE` on the 1x4 mesh with **synthetic** expert weights (the
shapes and dtypes the C19 policy asks for, no checkpoint), calls the real
`OptimizedMoE._routed_experts` once eagerly and once inside a trace capture, and lets the traceback
name the offending line of `tt/optimized_decoder.py`.

    python probe_routed_experts_trace.py [policy-json-or-name ...]
"""

import sys
import traceback

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model import close_ornith_mesh, open_ornith_mesh
from models.autoports.ornith_ai_ornith_1_0_35b.tt.model_config import HF_MODEL_ID, OrnithDecoderConfig
from models.autoports.ornith_ai_ornith_1_0_35b.tt.multichip_decoder import MultichipMoE, local_decoder_config
from models.autoports.ornith_ai_ornith_1_0_35b.tt.precision_config import resolve_policy

TP = 4
TOKENS = 32
VALID = 1

DEFAULT_POLICIES = [
    "models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep/selected_precision_config.json",
    "models/autoports/ornith_ai_ornith_1_0_35b/doc/datatype_sweep/blocked/C19-expert-act-bfp4.config.json",
]


def build_moe(mesh, cfg_global, cfg_local, policy):
    torch.manual_seed(0)
    e, h, i = cfg_local.num_experts, cfg_local.dim, cfg_local.moe_intermediate_size

    def up(host, dtype):
        return ttnn.from_torch(
            host,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    weights = {
        "expert_gate_up": up(torch.randn(1, e, h, 2 * i) * 0.02, policy.expert_gate_up_dtype),
        "expert_down": up(torch.randn(1, e, i, h) * 0.02, policy.expert_down_dtype),
    }
    return MultichipMoE(mesh, cfg_local, weights, global_config=cfg_global, tp=TP, policy=policy)


def build_inputs(mesh, cfg_local):
    """`x`, the group mask and the router scores, at decode batch 1 (one real row of 32)."""
    torch.manual_seed(1)
    e, h = cfg_local.num_experts, cfg_local.dim

    x_host = torch.zeros(1, 1, TOKENS, h)
    x_host[:, :, :VALID, :] = torch.randn(1, 1, VALID, h) * 0.1
    x = ttnn.from_torch(
        x_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # dense routing: two local experts on for the one real row, zeros elsewhere.
    dense_host = torch.zeros(1, 1, TOKENS, e)
    dense_host[0, 0, 0, 0] = 0.6
    dense_host[0, 0, 0, 5] = 0.4
    dense = ttnn.from_torch(
        dense_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return x, dense


def run_once(moe, x, dense, mask, scores):
    return moe._routed_experts(x, dense, TOKENS, group_mask=mask, scores=scores, valid_tokens=VALID)


def shim_sparse_matmul():
    """`ttnn.sparse_matmul` that produces the requested bfloat4_b output *without* the op's own
    bfloat4_b zero-fill: run the matmul at bfloat16 and typecast down.

    This is not a candidate fix (it costs a full-width bfloat16 intermediate, which is the opposite
    of what C19 asks for). It exists to answer one question: with the sparse matmul's host-side
    zero-fill removed, is the *rest* of the bfloat4_b routed-expert chain — the two unpacking
    slices, the SwiGLU multiply, the reshape, the score multiply and the widening typecast —
    trace-capturable?
    """
    real = ttnn.sparse_matmul

    def wrapper(*args, **kwargs):
        want = kwargs.get("dtype")
        if want != ttnn.bfloat4_b:
            return real(*args, **kwargs)
        kwargs["dtype"] = ttnn.bfloat16
        wide = real(*args, **kwargs)
        narrow = ttnn.typecast(wide, ttnn.bfloat4_b, memory_config=wide.memory_config())
        ttnn.deallocate(wide)
        return narrow

    ttnn.sparse_matmul = wrapper
    return real


def probe(mesh, policy_arg, *, shim=False):
    hf = AutoConfig.from_pretrained(HF_MODEL_ID)
    cfg_global = OrnithDecoderConfig.from_hf_config(hf)
    cfg_local = local_decoder_config(cfg_global, TP)
    policy = resolve_policy(policy_arg)
    print(
        f"\n##### policy={policy.name} expert_act_dtype={policy.expert_act_dtype} "
        f"gate_up_w={policy.expert_gate_up_dtype} E_local={cfg_local.num_experts}",
        flush=True,
    )

    real_smm = shim_sparse_matmul() if shim else None
    moe = build_moe(mesh, cfg_global, cfg_local, policy)
    moe._decode_phase = True
    moe._call_tokens = TOKENS
    x, dense = build_inputs(mesh, cfg_local)
    # Both per-call quantities the decoder hoists out of the group loop, built once here so the
    # probe exercises exactly the `_routed_experts` body.
    mask = moe._active_expert_mask(dense, 1, VALID)
    scores = ttnn.permute(dense, (0, 3, 2, 1))

    # 1. eager, twice: compile then program-cache hit, the same warm state trace capture sees.
    for k in range(2):
        out = run_once(moe, x, dense, mask, scores)
        ttnn.synchronize_device(mesh)
        print(f"eager[{k}] ok: shape={list(out.shape)} dtype={out.dtype}", flush=True)
        ttnn.deallocate(out)

    # 2. traced.
    verdict = "CAPTURABLE"
    try:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            out = run_once(moe, x, dense, mask, scores)
        finally:
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.release_trace(mesh, tid)
        ttnn.deallocate(out)
    except Exception as exc:  # noqa: BLE001
        verdict = f"NOT-CAPTURABLE {type(exc).__name__}"
        traceback.print_exc()
    print(f"VERDICT policy={policy.name} shim={shim}: {verdict}", flush=True)
    if real_smm is not None:
        ttnn.sparse_matmul = real_smm
    ttnn.synchronize_device(mesh)
    return verdict


def main():
    args = sys.argv[1:] or DEFAULT_POLICIES
    mesh = open_ornith_mesh(trace_region_size=200_000_000)
    out = []
    try:
        for arg in args:
            shim = arg.startswith("shim:")
            out.append((arg, probe(mesh, arg.removeprefix("shim:"), shim=shim)))
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        print("=== SUMMARY ===", flush=True)
        for arg, verdict in out:
            print(f"{arg}: {verdict}", flush=True)
        close_ornith_mesh(mesh)
        print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
