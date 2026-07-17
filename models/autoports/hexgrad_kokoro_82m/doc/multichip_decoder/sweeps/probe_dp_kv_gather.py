import sys as _s

_s.meta_path = [m for m in _s.meta_path if "editable" not in getattr(type(m), "__module__", "")]
_s.path = [p for p in _s.path if "model-bringup" not in p]
import json
import math
import time

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.optimized_decoder import PrecisionPolicy
from models.common.modules.tt_ccl import get_tt_ccl

MODEL_ID = "hexgrad/Kokoro-82M"
cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
ac = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
p = "encoder.albert_layer_groups.0.albert_layers.0."
H = 768
nh = 12
hd = 64
NL = 12
I = 2048
pol = PrecisionPolicy()


def run(fabric, topo, label):
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 4)), trace_region_size=90000000)
    ccl = get_tt_ccl(mesh)
    R = ttnn.ReplicateTensorToMesh(mesh)

    def T(t, tr=False, dt=ttnn.bfloat16):
        t = t.t().contiguous() if tr else t.contiguous()
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=R)

    qkv_w = T(
        torch.cat(
            [sd[p + "attention.query.weight"], sd[p + "attention.key.weight"], sd[p + "attention.value.weight"]], 0
        ),
        tr=True,
        dt=ttnn.bfloat8_b,
    )
    qkv_b = T(
        torch.cat([sd[p + "attention.query.bias"], sd[p + "attention.key.bias"], sd[p + "attention.value.bias"]], 0)
    )
    dense_w = T(sd[p + "attention.dense.weight"], tr=True, dt=ttnn.bfloat8_b)
    dense_b = T(sd[p + "attention.dense.bias"])
    aln_w = T(sd[p + "attention.LayerNorm.weight"])
    aln_b = T(sd[p + "attention.LayerNorm.bias"])
    ffn_w = T(sd[p + "ffn.weight"], tr=True, dt=ttnn.bfloat8_b)
    ffn_b = T(sd[p + "ffn.bias"])
    ffo_w = T(sd[p + "ffn_output.weight"], tr=True, dt=ttnn.bfloat8_b)
    ffo_b = T(sd[p + "ffn_output.bias"])
    fln_w = T(sd[p + "full_layer_layer_norm.weight"])
    fln_b = T(sd[p + "full_layer_layer_norm.bias"])
    mm = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    nk = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    sk = nk
    cg = mesh.compute_with_storage_grid_size()
    grid = ttnn.CoreGrid(y=min(8, cg.y), x=min(10, cg.x))
    spc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=cg, exp_approx_mode=False, q_chunk_size=128, k_chunk_size=128
    )
    scale = 1.0 / math.sqrt(hd)

    def AG(x):
        return ttnn.experimental.all_gather_async(
            x,
            dim=2,
            persistent_output_buffer=None,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=2,
            topology=topo,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
        )

    def layer(h_s, Sl):  # h_s [1,1,Sl,768] seq-sharded (Sl local)
        qkv = ttnn.linear(h_s, qkv_w, bias=qkv_b, compute_kernel_config=mm, core_grid=grid, dtype=ttnn.bfloat16)
        qkv = ttnn.reshape(qkv, (1, 1, Sl, 3 * nh * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=nh, num_kv_heads=nh, transpose_k_heads=False)
        ttnn.deallocate(qkv)
        kv = ttnn.concat([k, v], dim=1)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        kv = AG(kv)  # [1,24,S,64]
        kf = kv[:, :nh, :, :]
        vf = kv[:, nh:, :, :]
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, kf, vf, attn_mask=None, is_causal=False, scale=scale, program_config=spc, compute_kernel_config=sk
        )
        ttnn.deallocate(q)
        ttnn.deallocate(kv)
        attn = ttnn.experimental.nlp_concat_heads(attn)
        attn = ttnn.reshape(attn, (1, Sl, nh * hd))
        attn = ttnn.linear(attn, dense_w, bias=dense_b, compute_kernel_config=mm, core_grid=grid, dtype=ttnn.bfloat16)
        attn = ttnn.reshape(attn, (1, 1, Sl, H))
        h_s = ttnn.layer_norm(ttnn.add(attn, h_s), weight=aln_w, bias=aln_b, epsilon=1e-12, compute_kernel_config=nk)
        ttnn.deallocate(attn)
        ff = ttnn.linear(
            h_s, ffn_w, bias=ffn_b, compute_kernel_config=mm, core_grid=grid, dtype=ttnn.bfloat16, activation="gelu"
        )
        ff = ttnn.linear(ff, ffo_w, bias=ffo_b, compute_kernel_config=mm, core_grid=grid, dtype=ttnn.bfloat16)
        h_s = ttnn.layer_norm(ttnn.add(ff, h_s), weight=fln_w, bias=fln_b, epsilon=1e-12, compute_kernel_config=nk)
        ttnn.deallocate(ff)
        return h_s

    try:
        for S in (512, 128):
            Sl = S // 4
            x = ttnn.from_torch(
                torch.randn(1, 1, S, H),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=(1, 4)),
            )

            def enc():
                h = x
                for _ in range(NL):
                    h = layer(h, Sl)
                return h

            w = enc()
            ttnn.deallocate(w)
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = enc()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            print(f"{label} DP-KV L={S} decode_traced_ms={(time.perf_counter()-t0)/50*1e3:.4f}")
            ttnn.release_trace(mesh, tid)
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


run(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring, "RING")
print("DONE")
