# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reduced full-model probe: builds the generator on the (1,4) ring mesh and
exercises prefill_forward / decode_forward / generate on short prompts, checking
shapes, on-device argmax, PCC of last_hidden_state vs HF, and reconstruction
top-1 agreement HF-vs-TT. Fast (short seqs) — for debugging, not final evidence."""
import json
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

import ttnn
from models.autoports.hexgrad_kokoro_82m.tt.generator import build_generator

MODEL_ID = "hexgrad/Kokoro-82M"


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    cfg = json.load(open(hf_hub_download(MODEL_ID, "config.json")))
    vocab = cfg["vocab"]
    config = AlbertConfig(vocab_size=cfg["n_token"], **cfg["plbert"])
    sd = torch.load(hf_hub_download(MODEL_ID, "kokoro-v1_0.pth"), map_location="cpu", weights_only=True)["bert"]
    sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}
    hf = AlbertModel(config).eval()
    hf.load_state_dict(sd, strict=False)
    w_map = sd["encoder.embedding_hidden_mapping_in.weight"]
    w_word = sd["embeddings.word_embeddings.weight"]
    W = w_map.float() @ w_word.float().t()  # [H, vocab]

    def ids_of(t):
        return torch.tensor([[0] + [vocab[c] for c in t if c in vocab] + [0]], dtype=torch.long)

    text = "ðə kwˈɪk brˈaʊn fˈɑks"
    ids = ids_of(text)

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=90000000)
    try:
        gen = build_generator(model_dir="models/autoports/hexgrad_kokoro_82m", mesh_device=mesh)

        # HF reference
        with torch.no_grad():
            hf_hidden = hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
        hf_logits = (hf_hidden @ W).float()
        hf_argmax = hf_logits.argmax(-1)[0].tolist()

        # 1) prefill_forward return_all_logits -> [1, S, vocab]
        tt_logits = gen.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=True, enable_trace=False)
        print("PROBE prefill_all_logits shape", tuple(tt_logits.shape))
        tt_argmax = tt_logits.argmax(-1)[0].tolist()
        top1 = sum(1 for a, b in zip(tt_argmax, hf_argmax) if a == b) / len(hf_argmax)
        print(f"PROBE recon top1 HF-vs-TT (full visibility) = {top1:.4f}")

        # hidden PCC
        tt_hidden = gen.model.forward(ids, traced=False)
        print(f"PROBE hidden PCC vs HF = {pcc(tt_hidden, hf_hidden):.6f}")

        # 2) prefill_forward last-position -> [1,1,vocab]
        last = gen.prefill_forward(ids, prompt_lens=[ids.shape[1]], return_all_logits=False, enable_trace=False)
        print("PROBE prefill_last shape", tuple(last.shape))

        # 3) decode_forward greedy on-device argmax (traced) -> [1]
        tok = gen.decode_forward(ids, sample_on_device=True, enable_trace=True)
        print("PROBE decode greedy on-device token", int(tok[0]), "hf argmax@last", hf_argmax[-1])

        # 4) generate free-running (short)
        prompt = [0] + [vocab[c] for c in "ðə kwˈɪk" if c in vocab]
        out = gen.generate(prompt_token_ids=prompt, max_new_tokens=8, next_input=None, enable_trace=True)
        print("PROBE generate free-running preds", out, "->", gen.tokenizer.decode(out))

        # 5) teacher forcing style (next_input feeds a fixed sequence)
        gt = [vocab[c] for c in "brˈaʊn fˈɑks" if c in vocab]
        rec = []

        def ni(i, pred):
            rec.append(pred)
            return gt[i] if i < len(gt) else 0

        out2 = gen.generate(prompt_token_ids=prompt, max_new_tokens=len(gt), next_input=ni, enable_trace=True)
        print("PROBE generate TF preds", out2)
        print("PROBE counters", gen.counters)
        print("PROBE OK")
    finally:
        try:
            gen.teardown()
        except Exception as e:
            print("teardown err", e)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    sys.exit(main())
