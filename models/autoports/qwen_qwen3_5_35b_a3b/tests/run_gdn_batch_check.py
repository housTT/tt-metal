# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", type=int, choices=[1, 2, 4], required=True)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import ttnn

    from ..tt.gdn_batch import conv_batch, gdn_decode_batch
    from ..tt.generator import build_generator
    from .mesh_util import ReplicaReader, open_mesh

    mesh = open_mesh(args.mesh)
    read = ReplicaReader()
    result = dict(vars(args), output=str(args.output))
    try:
        gen = build_generator(Path(__file__).parents[1], mesh, layer_ids=(0, 3), decode_batches=(1,))
        layer = gen.model.layers[0]
        attn = layer.layer.attention
        rows = args.rows
        states = layer._states[:rows]
        g = torch.Generator().manual_seed(5)

        def put(tensor, like):
            return ttnn.from_torch(tensor, dtype=like.dtype, layout=like.layout, device=mesh,
                                   mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=like.memory_config())

        init = [(torch.randn(tuple(s.shape), generator=g) * 0.05, torch.randn(tuple(c.shape), generator=g))
                for s, c in states]

        def load_states():
            for (s, c), (hs, hc) in zip(states, init):
                ttnn.copy(put(hs, s), s)
                ttnn.copy(put(hc, c), c)

        def host(t):
            return read(t).float()

        x = ttnn.from_torch(torch.randn(1, rows, 2048, generator=g).to(torch.bfloat16), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        mc = ttnn.L1_MEMORY_CONFIG
        mega = attn.project("attention_in", x, attn.mega_weight, attn.compute_kernel_config, mc)
        width = attn.cfg.q_dim + attn.cfg.k_dim + attn.cfg.v_dim

        load_states()
        ref_conv, ref_out = [], []
        for r, (s, c) in enumerate(states):
            attn.recurrent_state, attn.fused_conv_state = s, c
            row = mega[:, r : r + 1, :]
            ref_out.append(host(attn.forward(None, _mega=row, _gated_only=True)).reshape(-1))
        ref_states = [(host(s), host(c)) for s, c in states]

        load_states()
        first_conv = [host(c) for _, c in states]
        qkv = conv_batch(mega, attn.kda_taps, [c for _, c in states], width)
        conv_host = host(qkv).reshape(rows, -1)[:, :width]
        conv_states_batch = [host(c) for _, c in states]
        load_states()
        qkv2 = conv_batch(mega, attn.kda_taps, [c for _, c in states], width)
        stride = getattr(attn, "ab_stride", attn.num_v_heads)
        gated = gdn_decode_batch(qkv2, mega, [s for s, _ in states], attn.norm_weight, attn.dt_bias, attn.a_neg,
                                 q_dim=attn.cfg.q_dim, k_dim=attn.cfg.k_dim,
                                 groups=attn.num_v_heads // attn.num_heads, a_start=width, b_start=width + stride,
                                 z_offset=(width + 2 * stride) // 32, norm_eps=attn.norm_eps, raw=True)
        gated, raw = gated
        gated_host = host(gated).reshape(rows, -1)
        heads, hv = attn.num_v_heads, 128
        o = host(raw).reshape(rows, heads, hv)
        z = host(mega).reshape(rows, -1)[:, width + 2 * stride : width + 2 * stride + heads * hv].reshape(rows, heads, hv)
        w = host(attn.norm_weight).reshape(-1)[:hv]
        torch_gated = (o * torch.rsqrt((o * o).mean(-1, keepdim=True) + attn.norm_eps) * w
                       * torch.nn.functional.silu(z)).reshape(rows, -1)
        batch_states = [(host(s), host(c)) for s, c in states]

        expected_conv = []
        hist = [torch.cat([fc.reshape(3, -1), host(mega).reshape(rows, -1)[r : r + 1, :width]]) for r, fc in enumerate(first_conv)]
        taps = [host(t).reshape(-1)[:width] for t in attn.kda_taps]
        for r in range(rows):
            acc = sum(taps[j] * hist[r][j] for j in range(4))
            expected_conv.append(torch.nn.functional.silu(acc.to(torch.bfloat16).float()))
        expected_conv = torch.stack(expected_conv)

        def cmp(a, b):
            a, b = a.reshape(-1).float(), b.reshape(-1).float()
            ok = torch.isfinite(b).all()
            pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if ok else float("nan")
            return dict(pcc=pcc, max_abs=float((a - b).abs().max()), finite=bool(ok))

        result["conv_vs_torch"] = cmp(expected_conv, conv_host)
        load_states()
        kda = []
        for r, (s_, c_) in enumerate(states):
            attn.recurrent_state, attn.fused_conv_state = s_, c_
            q_, k_, v_ = attn._convolve(ttnn.to_layout(mega[:, r : r + 1, :width], ttnn.TILE_LAYOUT))
            attn._conv_current = None
            kda.append(torch.cat([host(t).reshape(-1, t.shape[-1])[0] for t in (q_, k_, v_)]))
        kda = torch.stack(kda)
        q_dim, k_dim = attn.cfg.q_dim, attn.cfg.k_dim
        result["kda_vs_torch"] = {name: cmp(expected_conv[:, a:b], kda[:, a:b]) for name, a, b in
                                  (("q", 0, q_dim), ("k", q_dim, q_dim + k_dim), ("v", q_dim + k_dim, width))}
        result["conv_state_vs_ref"] = [cmp(ref_states[r][1], conv_states_batch[r]) for r in range(rows)]
        result["recurrent_state_vs_ref"] = [cmp(ref_states[r][0], batch_states[r][0]) for r in range(rows)]
        gated_ref = torch.stack(ref_out)
        result["gated_vs_ref"] = [cmp(gated_ref[r], gated_host[r]) for r in range(rows)]
        result["torch_gate_on_raw_vs_ref"] = [cmp(gated_ref[r], torch_gated[r]) for r in range(rows)]
        from ..tt.gdn_decode import gdn_decode

        load_states()
        attn.recurrent_state, attn.fused_conv_state = states[0]
        from ..tt import gdn_decode as gdn_module

        seen = {}
        original = gdn_module.gdn_decode

        def spy(*a, **kw):
            seen["args"] = [None if t is None else (tuple(t.shape), str(t.dtype), str(t.layout), str(t.memory_config().buffer_type)) for t in a]
            seen["kw"] = {k: (v if not hasattr(v, "shape") else ("tensor", tuple(v.shape), str(v.dtype), str(v.layout))) for k, v in kw.items() if k != "gates" and k != "conv"}
            seen["gates"] = [x if isinstance(x, int) else (tuple(x.shape), str(x.dtype), str(x.layout)) for x in kw.get("gates") or []]
            seen["conv"] = None if kw.get("conv") is None else [(tuple(x.shape), str(x.dtype), str(x.layout)) for x in kw["conv"]]
            out = original(*a, **kw)
            seen["returned"] = out
            return out

        gdn_module.gdn_decode = spy
        again = host(attn.forward(None, _mega=mega[:, 0:1, :], _gated_only=True)).reshape(-1)
        gdn_module.gdn_decode = original
        returned = seen.pop("returned")
        result["forward_gdn_decode_call"] = {k: str(v) for k, v in seen.items()}
        result["forward_kernel_return_vs_forward_flat"] = cmp(host(returned).reshape(-1), again)
        result["forward_kernel_return_padded"] = str(returned.padded_shape)
        load_states()
        again_full = host(attn.forward(None, _mega=mega[:, 0:1, :], _gated_only=True)).reshape(-1)
        load_states()
        attn.recurrent_state, attn.fused_conv_state = states[0]
        q1, k1, v1 = attn._convolve(ttnn.to_layout(mega[:, 0:1, :width], ttnn.TILE_LAYOUT))
        attn._conv_current = None
        gated1, raw1 = gdn_decode(q1, k1, v1, mega, None, None, states[0][0], attn.norm_weight,
                             groups=attn.num_v_heads // attn.num_heads, blocks=attn.optimization.gdn_decode_blocks,
                             norm_eps=attn.norm_eps, gates=(mega, attn.dt_bias, attn.a_neg, width, width + stride),
                             z_offset=(width + 2 * stride) // 32, raw=True)
        o1 = host(raw1).reshape(-1, heads * hv)[0]
        load_states()
        attn.recurrent_state, attn.fused_conv_state = states[0]
        cols = ttnn.to_layout(mega[:, 0:1, 0:width], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        q2, k2, v2 = attn._convolve(cols)
        current, attn._conv_current = attn._conv_current, None
        gated2, raw2 = gdn_decode(q2, k2, v2, mega, None, None, states[0][0], attn.norm_weight,
                                  groups=attn.num_v_heads // attn.num_heads, blocks=0, norm_eps=attn.norm_eps,
                                  gates=(mega, attn.dt_bias, attn.a_neg, width, width + stride),
                                  z_offset=(width + 2 * stride) // 32, conv=(states[0][1], current), raw=True)
        result["shift_gated_vs_forward"] = cmp(host(gated2).reshape(-1), again)
        result["shift_gated_vs_forward_kernel_return"] = cmp(host(gated2).reshape(-1), host(returned).reshape(-1))
        result["reshaped_gated2_vs_gated2"] = cmp(host(ttnn.reshape(gated2, (1, 1, gated2.shape[-1]))).reshape(-1), host(gated2).reshape(-1))
        row = mega[:, 0:1, :]
        result["mega_memory"] = str(mega.memory_config())
        result["row_memory"] = str(row.memory_config())
        result["mega_padded"] = str(mega.padded_shape)
        result["row_padded"] = str(row.padded_shape)
        load_states()
        attn.recurrent_state, attn.fused_conv_state = states[0]
        q3, k3, v3 = attn._convolve(ttnn.to_layout(row[:, :, 0:width], ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG))
        attn._conv_current = None
        gated3 = gdn_decode(q3, k3, v3, row, None, None, states[0][0], attn.norm_weight,
                            groups=attn.num_v_heads // attn.num_heads, blocks=0, norm_eps=attn.norm_eps,
                            gates=(row, attn.dt_bias, attn.a_neg, width, width + stride),
                            z_offset=(width + 2 * stride) // 32)
        result["row_slice_gated_vs_forward"] = cmp(host(gated3).reshape(-1), again)
        result["mega_vs_row_values"] = cmp(host(mega).reshape(rows, -1)[0], host(row).reshape(-1))
        result["shift_raw_vs_noshift_raw"] = cmp(host(raw2).reshape(-1, heads * hv)[0], o1)
        result["q_cols_vs_q_slice"] = cmp(host(q2).reshape(32, -1)[0], host(q1).reshape(32, -1)[0])
        result["raw_b1_vs_batch_row0"] = cmp(o1, o[0].reshape(-1))
        g1 = host(gated1).reshape(-1)
        result["b1_gated_vs_ref"] = cmp(gated_ref[0], g1)
        result["forward_again_vs_ref"] = cmp(gated_ref[0], again)
        result["forward_again_vs_b1"] = cmp(g1, again)
        result["forward_full_mega_vs_b1"] = cmp(g1, again_full)
        result["b1_gated_vs_batch"] = cmp(gated_host[0], g1)
        result["ref_flat_shape"] = list(gated_ref.shape)
        result["raw_b1_vs_batch_per_block"] = [cmp(o1.reshape(heads, 4, 32)[h, b_], o[0].reshape(heads, 4, 32)[h, b_])
                                               for h in range(2) for b_ in range(4)]
        result["raw_ratio_head0_block0"] = (o[0].reshape(heads, 4, 32)[0, 0] / o1.reshape(heads, 4, 32)[0, 0])[:8].tolist()
        groups = attn.num_v_heads // attn.num_heads
        mega_h = host(mega).reshape(rows, -1)
        bias = host(attn.dt_bias).reshape(-1)[:heads]
        neg = host(attn.a_neg).reshape(-1)[:heads]
        weight = host(attn.norm_weight).reshape(-1)[:hv]
        q_dim_, k_dim_ = attn.cfg.q_dim, attn.cfg.k_dim
        torch_rows = []
        for r in range(rows):
            conv = expected_conv[r]
            qh = conv[:q_dim_].reshape(-1, 128)
            kh = conv[q_dim_:q_dim_ + k_dim_].reshape(-1, 128)
            vh = conv[q_dim_ + k_dim_:width].reshape(heads, hv)
            outs = []
            for h in range(heads):
                q_ = qh[h // groups] / torch.sqrt((qh[h // groups] ** 2).sum() + 1e-6) * 128 ** -0.5
                k_ = kh[h // groups] / torch.sqrt((kh[h // groups] ** 2).sum() + 1e-6)
                a_ = mega_h[r, width + h]
                b_ = mega_h[r, width + stride + h]
                decay = torch.exp(neg[h] * torch.nn.functional.softplus(a_ + bias[h]))
                beta = torch.sigmoid(b_)
                S = init[r][0].reshape(heads, 128, hv)[h].clone() * decay
                delta = (vh[h] - (S * k_[:, None]).sum(0)) * beta
                S = S + k_[:, None] * delta[None, :]
                o_ = (S * q_[:, None]).sum(0)
                z_ = mega_h[r, width + 2 * stride + h * hv : width + 2 * stride + (h + 1) * hv]
                outs.append(o_ * torch.rsqrt((o_ * o_).mean() + attn.norm_eps) * weight * torch.nn.functional.silu(z_))
            torch_rows.append(torch.cat(outs))
        torch_rows = torch.stack(torch_rows)
        result["torch_vs_forward_row0"] = cmp(torch_rows[0], gated_ref[0])
        result["torch_vs_batch"] = [cmp(torch_rows[r], gated_host[r]) for r in range(rows)]
        result["torch_vs_b1_direct"] = cmp(torch_rows[0], g1)
        result["per_head_row0"] = [cmp(gated_ref[0].reshape(heads, hv)[h], gated_host[0].reshape(heads, hv)[h])
                                   for h in range(heads)]
        result["per_block_head0"] = [cmp(gated_ref[0].reshape(heads, 4, 32)[0, b_], gated_host[0].reshape(heads, 4, 32)[0, b_])
                                     for b_ in range(4)]
        result["torch_gate_on_raw_vs_kernel_gate"] = [cmp(gated_host[r], torch_gated[r]) for r in range(rows)]
        result["replicas_identical"] = read.all_identical()
        print(json.dumps(result, indent=1), flush=True)
    finally:
        args.output.write_text(json.dumps(result, indent=1))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
