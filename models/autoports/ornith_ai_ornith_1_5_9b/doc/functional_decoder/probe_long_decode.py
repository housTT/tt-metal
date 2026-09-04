"""AutoFix diagnostic only: exact native-context BF16 cache and isolated SDPA controls."""

import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests import test_functional_decoder as H

ROOT = Path(__file__).resolve().parent
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=0)
mesh.enable_program_cache()
rows = []
try:
    cfg = H.hf_config()
    context = cfg.max_position_embeddings
    decoder, _, _ = H.build_decoder(mesh, H.FULL_LAYER, "real", max_context=context)
    heads, dim, page = decoder.cfg.n_kv_heads, decoder.cfg.head_dim, decoder.page_block_size
    blocks = H.num_blocks_for_context(context)
    rng = torch.Generator().manual_seed(970)
    table_host = torch.randperm(blocks, generator=rng).to(torch.int32).reshape(1, blocks)
    table = H.to_device(mesh, table_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    history = []
    expected_prev = []
    for target in (decoder.k_cache, decoder.v_cache):
        values = torch.randn(1, heads, context, dim, generator=rng).to(torch.bfloat16) * 0.125
        history.append(values[:, :, : context - 1].float())
        expected_prev.append(values[0, :, -2].clone())
        logical = values.reshape(heads, blocks, page, dim).permute(1, 0, 2, 3).contiguous()
        physical = torch.empty_like(logical)
        physical[table_host[0].long()] = logical
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(physical, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), target)
        del values, logical, physical
    pos, rot = H.decode_inputs(mesh, torch.tensor([context - 1]))
    original = ttnn.transformer.paged_scaled_dot_product_attention_decode
    update_original = ttnn.experimental.paged_update_cache
    updated = []
    query = None
    cache_oracle = None
    capture_diagnostics = True
    attention_actual = None
    control = ("default", 64, None, None)

    def update(cache, tensor, **kwargs):
        if capture_diagnostics:
            updated.append(ttnn.to_torch(tensor)[0, 0, :heads].clone())
        return update_original(cache, tensor, **kwargs)

    def sdpa(q, k, v, **kwargs):
        global query, cache_oracle, attention_actual
        name, chunk, fp32, approximate = control
        kwargs["program_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            q_chunk_size=32,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
        if fp32 is not None:
            kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=approximate,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=False,
            )
        if capture_diagnostics:
            query = ttnn.to_torch(q).float()[0, 0]
            cache_read = []
            for idx, tensor in enumerate((k, v)):
                physical = ttnn.to_torch(tensor)
                assert torch.equal(physical[int(table_host[0, -1]), :, -2], expected_prev[idx])
                assert torch.equal(physical[int(table_host[0, -1]), :, -1], updated[idx])
                cache_read.append(physical[table_host[0].long()].permute(1, 0, 2, 3).reshape(heads, context, dim))
            result = []
            for head in range(heads):
                qh = query[head * 4 : (head + 1) * 4]
                logits = qh @ cache_read[0][head].float().T * dim**-0.5
                result.append(torch.softmax(logits, dim=-1) @ cache_read[1][head].float())
            cache_oracle = torch.cat(result)
            print("EXACT_CACHE_ROW_AND_PREVIOUS_ROW_PASS", flush=True)
            del cache_read
        out = original(q, k, v, **kwargs)
        attention_actual = ttnn.to_torch(out).float()[0, 0] if diagnostics_output else None
        return out

    ttnn.experimental.paged_update_cache = update
    ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa
    controls = [
        ("default", 64, None, None),
        ("explicit_default", 64, False, True),
        ("fp32_only", 64, True, True),
        ("nonapprox_only", 64, False, False),
        ("chunk256", 256, None, None),
        ("chunk512", 512, None, None),
    ]
    for scale in [0.014285416342318058, 0.5]:
        x = (torch.randn(1, 1, cfg.hidden_size, generator=torch.Generator().manual_seed(971)) * scale).to(
            torch.bfloat16
        )
        cache = H.R._new_cache(cfg)
        cache.update(history[0], history[1], H.FULL_LAYER)
        with torch.no_grad():
            golden = H.R.reference_decode(
                H.reference_layer(H.FULL_LAYER, "real"), cfg, x.float(), torch.tensor([context - 1]), cache
            )
        del cache
        x_buf = H.to_device(mesh, x)
        capture_diagnostics = True
        for control in controls:
            diagnostics_output = True
            eager = decoder.decode_forward(x_buf, current_pos=pos, rot_idxs=rot, page_table=table)
            actual = ttnn.to_torch(eager)
            record = dict(
                scale=scale,
                control=control,
                full_pcc=H.pcc(golden, actual),
                attention_pcc=H.pcc(cache_oracle, attention_actual),
                attention_relative_l2=float((cache_oracle - attention_actual).norm() / cache_oracle.norm()),
                head15_pcc=H.pcc(cache_oracle[15], attention_actual[15]),
            )
            ttnn.deallocate(eager)
            capture_diagnostics = False
            diagnostics_output = False
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = decoder.decode_forward(x_buf, current_pos=pos, rot_idxs=rot, page_table=table)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            replay = ttnn.to_torch(out)
            record.update(replay_pcc=H.pcc(golden, replay), eager_replay_equal=torch.equal(actual, replay))
            ttnn.release_trace(mesh, trace)
            ttnn.deallocate(out)
            rows.append(record)
            print(json.dumps(record), flush=True)
            (ROOT / "probe_long_decode.json").write_text(json.dumps(rows, indent=2) + "\n")
        updated.clear()
        ttnn.deallocate(x_buf)
finally:
    ttnn.close_mesh_device(mesh)
