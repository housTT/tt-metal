# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import bz2
import json
import os
import time

import torch

HF_MODEL = "Qwen/Qwen3-8B"
TALE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "tt_transformers",
        "tests",
        "tale-of-two-cities.txt.bz2",
    )
)


def load_hf(dtype=torch.bfloat16, threads=8):
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL, torch_dtype=dtype)
    model.eval()
    return tok, model


def real_tokens(tok, n):
    with bz2.open(TALE, "rt", encoding="utf-8") as f:
        text = f.read()
    ids = tok(text[: n * 12], add_special_tokens=False)["input_ids"]
    assert len(ids) >= n, (len(ids), n)
    return torch.tensor(ids[:n])


@torch.no_grad()
def capture_hidden_states(model, tokens):
    out = model(input_ids=tokens[None], output_hidden_states=True)
    return [h[0] for h in out.hidden_states], out.last_hidden_state[0]


@torch.no_grad()
def run_layer(model, k, x, position_ids):
    layer = model.layers[k]
    cos, sin = model.rotary_emb(x[None].to(model.dtype), position_ids)
    n = x.shape[0]
    mask = torch.triu(torch.full((n, n), torch.finfo(model.dtype).min, dtype=model.dtype), diagonal=1)[None, None]
    y = layer(x[None].to(model.dtype), attention_mask=mask, position_ids=position_ids, position_embeddings=(cos, sin))
    if isinstance(y, tuple):
        y = y[0]
    return y[0]


@torch.no_grad()
def final_norm(model, x):
    return model.norm(x.to(model.dtype))


if __name__ == "__main__":
    t0 = time.perf_counter()
    tok, model = load_hf()
    n = 128
    tokens = real_tokens(tok, n)
    hs, last = capture_hidden_states(model, tokens)
    pos = torch.arange(n)[None]
    report = {"load_s": round(time.perf_counter() - t0, 1), "n_layers": len(hs) - 1, "dtype": str(model.dtype)}
    for k in (0, 17, 35):
        y = run_layer(model, k, hs[k], pos)
        ref = hs[k + 1] if k + 1 < len(hs) - 1 else None
        if ref is None:
            ref = final_norm(model, y) if False else None
        if k + 1 <= len(hs) - 1:
            target = hs[k + 1]
            if k == len(hs) - 2:
                target_cmp = final_norm(model, y)
                report[f"layer{k}_matches_last_hidden_after_norm"] = float(
                    torch.nn.functional.cosine_similarity(
                        target_cmp.float().flatten()[None], last.float().flatten()[None]
                    )
                )
            else:
                report[f"layer{k}_cos_vs_next_hidden"] = float(
                    torch.nn.functional.cosine_similarity(y.float().flatten()[None], target.float().flatten()[None])
                )
                report[f"layer{k}_maxabs"] = float((y.float() - target.float()).abs().max())
    report["final_norm_identity_check"] = (
        float(
            torch.nn.functional.cosine_similarity(
                final_norm(model, hs[-1]).float().flatten()[None], last.float().flatten()[None]
            )
        )
        if False
        else None
    )
    print("HF_LAYER_REF", json.dumps(report))
