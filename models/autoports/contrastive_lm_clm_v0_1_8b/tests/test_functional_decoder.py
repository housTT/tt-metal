# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import (
    Mode,
    PagedAttentionConfig,
    get_padded_prefill_len,
    get_rot_transformation_mat,
)
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import get_rot_mats

HERE = os.path.dirname(os.path.abspath(__file__))
DOC_DIR = os.path.join(os.path.dirname(HERE), "doc", "functional_decoder")
LENGTHS = [32, 33, 127, 128, 129, 500, 1024, 2048]
LAYERS = [0, 17, 35]
PCC_GATE = 0.995


from models.autoports.contrastive_lm_clm_v0_1_8b.reference.hf_layer_reference import (
    capture_hidden_states,
    load_hf,
    real_tokens,
    run_layer,
)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 0}], indirect=True)
def test_functional_decoder_layers(mesh_device, reset_seeds, ensure_gc):
    os.makedirs(DOC_DIR, exist_ok=True)
    dtype = ttnn.bfloat8_b
    max_seq_len = max(LENGTHS)
    model_args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max_seq_len, use_hf_rope=False)
    model_args.n_layers = 36
    state_dict = model_args.load_state_dict()
    tokenizer, hf = load_hf()
    tokens = real_tokens(tokenizer, max_seq_len)
    position_ids = torch.arange(max_seq_len)[None]
    hs, _ = capture_hidden_states(hf, tokens)
    layer_inputs = {k: hs[k].float() for k in LAYERS}

    rot_mats_by_len = {
        p: get_rot_mats(
            head_dim=model_args.head_dim,
            device=mesh_device,
            seq_len=p,
            theta=model_args.rope_theta,
            rope_scaling=model_args.rope_scaling,
        )
        for p in sorted({get_padded_prefill_len(n) for n in LENGTHS})
    }
    transformation_mats = {
        "prefill": ttnn.as_tensor(
            get_rot_transformation_mat(model_args.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
    }
    pac = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
    page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(4, -1)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_ccl = TT_CCL(mesh_device)

    results = []
    all_pass = True
    for k in LAYERS:
        tt_layer = TransformerBlock(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            layer_num=k,
            dtype=dtype,
            transformation_mats=transformation_mats,
            args=model_args,
            paged_attention_config=pac,
        )
        for n in LENGTHS:
            x = layer_inputs[k][:n]
            ref = run_layer(hf, k, x, position_ids[:, :n]).float()
            padded = get_padded_prefill_len(n)
            x_pad = torch.zeros(1, padded, model_args.dim)
            x_pad[0, :n] = x
            for user in [0] if n != 128 else [0, 1, 2, 3]:
                tt_in = model_args.prepare_residual_tensor_prefill(x_pad)
                t0 = time.perf_counter()
                tt_out = tt_layer(
                    tt_in,
                    None,
                    rot_mats_global=rot_mats_by_len[padded],
                    rot_mats_local=None,
                    user_id=user,
                    mode=Mode.PREFILL,
                    page_table=page_table_tt,
                )
                tt_torch = ttnn.to_torch(
                    tt_out,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape
                    ),
                )
                dt = time.perf_counter() - t0
                y = tt_torch[0, 0, :n, : model_args.dim]
                passing, msg = comp_pcc(ref, y, PCC_GATE)
                pcc = float(msg.split(":")[-1].strip().split()[0]) if ":" in msg else float("nan")
                cos = torch.nn.functional.cosine_similarity(ref.flatten()[None], y.flatten()[None]).item()
                row = {
                    "layer": k,
                    "seq_len": n,
                    "padded": padded,
                    "user": user,
                    "pcc": pcc,
                    "cosine": cos,
                    "pass": bool(passing),
                    "seconds": round(dt, 4),
                }
                logger.info(f"layer {k} len {n} user {user}: {msg} cos={cos:.5f}")
                results.append(row)
                all_pass &= bool(passing)
        del tt_layer
    with open(os.path.join(DOC_DIR, "layer_pcc.json"), "w") as f:
        json.dump(
            {
                "gate": PCC_GATE,
                "lengths": LENGTHS,
                "layers": LAYERS,
                "weight_dtype": "bfloat8_b",
                "activation_dtype": "bfloat16",
                "reference_dtype": str(hf.dtype),
                "device": model_args.device_name,
                "rows": results,
            },
            f,
            indent=1,
        )
    assert all_pass, [r for r in results if not r["pass"]]
