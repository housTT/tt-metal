# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "doc", "probe")

TIDES_STATE = "What causes tides on Earth?"
TIDES_CANDIDATES = ["The Moon's gravitational pull.", "Photosynthesis in plants.", "Because the Earth is round."]
CUSTOMER_STATE = (
    "Customer: my invoice was charged twice and nobody answers the phone!\n\nWhich team should handle this?"
)
CUSTOMER_CANDIDATES = ["Charges, invoices, refunds", "Bugs and outages"]


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def head_forward(sd, x):
    h = torch.nn.functional.gelu(x @ sd["inp.weight"].T + sd["inp.bias"])
    h = h @ sd["hidden.0.weight"].T + sd["hidden.0.bias"]
    h = torch.nn.functional.layer_norm(h, (h.shape[-1],), sd["norms.0.weight"], sd["norms.0.bias"])
    h = torch.nn.functional.gelu(h)
    return torch.nn.functional.normalize(h @ sd["out.weight"].T + sd["out.bias"], dim=-1)


def rank(ck, e_state, e_cands):
    zs = head_forward(ck["state_head"], torch.from_numpy(e_state).float()[None])
    za = head_forward(ck["action_head"], torch.from_numpy(e_cands).float())
    logits = 100.0 * (za @ zs[0])
    return torch.softmax(logits, dim=0).tolist()


def hf_reference(texts, tok):
    from transformers import AutoModel

    torch.set_num_threads(8)
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16)
    model.eval()
    load_s = time.perf_counter() - t0
    out = []
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([tok(t, add_special_tokens=False)["input_ids"]])
            h = model(input_ids=ids).last_hidden_state[0, -1].float().numpy()
            out.append(h)
    return np.stack(out), load_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--optimizations", default="accuracy")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--trace-region", type=int, default=200_000_000)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from transformers import AutoTokenizer

    import ttnn
    from models.tt_transformers.tt.common import PagedAttentionConfig, create_tt_model
    from models.tt_transformers.tt.generator import Generator
    from models.tt_transformers.tt.model_config import DecodersPrecision

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    texts = [TIDES_STATE] + TIDES_CANDIDATES + [CUSTOMER_STATE] + CUSTOMER_CANDIDATES
    report = {
        "texts": texts,
        "max_seq_len": a.max_seq_len,
        "max_batch_size": a.max_batch_size,
        "optimizations": a.optimizations,
    }

    t0 = time.perf_counter()
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=a.trace_region, num_command_queues=1
    )
    try:
        block_size = 32
        blocks_per_seq = (a.max_seq_len + block_size - 1) // block_size
        pac = PagedAttentionConfig(block_size=block_size, max_num_blocks=max(1024, blocks_per_seq * a.max_batch_size))
        opt = DecodersPrecision.from_string(a.optimizations)
        model_args, tt_model, tt_kv_cache, state_dict = create_tt_model(
            mesh,
            instruct=False,
            max_batch_size=a.max_batch_size,
            optimizations=lambda ma: opt(ma.n_layers, ma.model_name),
            max_seq_len=a.max_seq_len,
            paged_attention_config=pac,
            dtype=ttnn.bfloat8_b,
            num_layers=a.n_layers,
        )
        report["load_s"] = round(time.perf_counter() - t0, 1)
        report["model_name"] = model_args.model_name
        report["device_name"] = model_args.device_name
        report["n_layers"] = model_args.n_layers
        report["disable_batched_prefill"] = bool(getattr(model_args, "disable_batched_prefill", False))
        report["trace_lens"] = list(model_args.trace_prefill_supported_seq_lens)
        report["max_prefill_chunk_size"] = model_args.max_prefill_chunk_size
        gen = Generator([tt_model], [model_args], mesh, tokenizer=model_args.tokenizer)
        page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(a.max_batch_size, -1)

        def embed_ids(id_lists):
            b = len(id_lists)
            lens = [len(x) for x in id_lists]
            width = max(lens)
            tokens = torch.zeros(b, width, dtype=torch.long)
            for i, ids in enumerate(id_lists):
                tokens[i, : len(ids)] = torch.tensor(ids)
            t = time.perf_counter()
            out = gen.prefill_forward_text(
                tokens,
                page_table=page_table[:b],
                kv_cache=[tt_kv_cache],
                prompt_lens=lens,
                enable_trace=True,
                return_hidden_states=True,
            )
            return out.float().numpy(), time.perf_counter() - t

        ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
        report["n_tokens"] = [len(x) for x in ids]
        single, single_t = [], []
        for x in ids:
            e, dt = embed_ids([x])
            single.append(e[0])
            single_t.append(dt)
        single = np.stack(single)
        report["single_latency_s_first"] = round(single_t[0], 3)
        report["single_latency_s"] = [round(x, 4) for x in single_t]
        rerun, rerun_t = [], []
        for x in ids:
            e, dt = embed_ids([x])
            rerun.append(e[0])
            rerun_t.append(dt)
        rerun = np.stack(rerun)
        report["single_latency_s_warm"] = [round(x, 4) for x in rerun_t]
        report["determinism_cos"] = [float(c) for c in (l2(single) * l2(rerun)).sum(-1)]
        batched, bt = embed_ids(ids[:4])
        report["batch4_latency_s_first"] = round(bt, 3)
        batched2, bt2 = embed_ids(ids[:4])
        report["batch4_latency_s_warm"] = round(bt2, 3)
        report["batch_vs_single_cos"] = [float(c) for c in (l2(batched2) * l2(single[:4])).sum(-1)]
        report["tt_raw_norms"] = [float(x) for x in np.linalg.norm(single, axis=-1)]
        np.save(os.path.join(OUT_DIR, "tt_single.npy"), single)
        np.save(os.path.join(OUT_DIR, "tt_batch4.npy"), batched2)
    finally:
        ttnn.close_mesh_device(mesh)
    report["device_closed"] = True

    ck = torch.load("/home/hous/dev/clm-v0.1-8B/checkpoints/CLM_v0.1-8B.pt", map_location="cpu", weights_only=False)
    report["tt_rank_tides"] = rank(ck, l2(single[0]), l2(single[1:4]))
    report["tt_rank_customer"] = rank(ck, l2(single[4]), l2(single[5:7]))
    if not a.skip_hf:
        ref, load_s = hf_reference(texts, tok)
        np.save(os.path.join(OUT_DIR, "hf_bf16.npy"), ref)
        report["hf_load_s"] = round(load_s, 1)
        report["hf_raw_norms"] = [float(x) for x in np.linalg.norm(ref, axis=-1)]
        report["tt_vs_hf_cos"] = [float(c) for c in (l2(single) * l2(ref)).sum(-1)]
        report["hf_rank_tides"] = rank(ck, l2(ref[0]), l2(ref[1:4]))
        report["hf_rank_customer"] = rank(ck, l2(ref[4]), l2(ref[5:7]))
    with open(os.path.join(OUT_DIR, "probe_full_encoder.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("PROBE_RESULT", json.dumps(report))


if __name__ == "__main__":
    main()
