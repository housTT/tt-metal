# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build model-appropriate readiness references for Kokoro's plbert encoder.

AIME24 chat-template references are **N/A**: Kokoro is a non-autoregressive TTS
model with no HF chat template and no causal LM. The model-appropriate reference
pins the HF (AlbertModel) tied-embedding **phoneme reconstruction** top-K at each
position, over real IPA phoneme sentences. Two references, matching the two
runners' visibility:

* ``readiness_recon_prefill.refpt`` — full bidirectional visibility (the whole
  sentence encoded once); consumed by ``run_prefill_check``.
* ``readiness_recon_tf.refpt`` — growing-prefix visibility (reconstruction at the
  last position of each length-(P+i) prefix); consumed by ``run_teacher_forcing``,
  matching the generator's stateless growing-prefix decode loop.

Both are regenerated fresh from the pinned HF weights each run (no reuse).
"""
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AlbertConfig, AlbertModel

from models.common.readiness_check.schema import Reference, ReferenceEntry, save_reference

MODEL_ID = "hexgrad/Kokoro-82M"
K = 100
HERE = Path(__file__).parent

# Real IPA phoneme sentences (Kokoro's g2p phoneme alphabet), long enough for
# a stable top-K sample. First three match the decoder-stage IPA controls.
SENTENCES = [
    "ðə kwˈɪk brˈaʊn fˈɑks ʤˈʌmps ˈOvɚ ðə lˈeɪzi dˈɔɡ",
    "tˈɛnstɔɹɛnt bˈɪldz ˈAI ˈaksɛlɚˌeɪɾɚz fɔɹ ðə wˈɜːld",
    "spˈiʧ sˈɪnθəsˈɪs ɔn dˈɛdɪkˌeɪɾɪd hˈɑɹdwɛɹ ɪz fˈæst ænd ɛfˈɪʃənt",
    "ðə sˈʌn wˈʌz ʃˈaɪnɪŋ ˈOvɚ ðə mˈaʊntənz ænd ðə vˈæli bɪlˈO",
]


def _load():
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
    return vocab, hf, W


def _ids(vocab, text):
    return torch.tensor([[0] + [vocab[c] for c in text if c in vocab] + [0]], dtype=torch.long)


def _recon(hf, W, ids):
    with torch.no_grad():
        h = hf(ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    return (h @ W).float()  # [1, S, vocab]


def main():
    vocab, hf, W = _load()
    P = 1  # prompt = BOS only; reconstruct every following position
    prefill_entries, tf_entries = [], []
    for text in SENTENCES:
        ids = _ids(vocab, text)
        S = ids[0]  # [L]
        L = S.numel()
        gen_len = L - P

        # --- prefill: full-visibility reconstruction over the whole sentence
        full_logits = _recon(hf, W, ids)[0]  # [L, vocab]
        topk_full = full_logits.topk(K, dim=-1).indices.to(torch.int32)  # [L, K]
        prefill_entries.append(
            ReferenceEntry(
                prompt_text=text,
                prompt_tokens=S[:P].reshape(1, P),
                generated_tokens=S[P:].reshape(1, gen_len),
                topk_tokens=topk_full[P - 1 : P - 1 + gen_len],  # positions P-1 .. L-2
                tf_prompt_len=P,
            )
        )

        # --- teacher forcing: growing-prefix reconstruction at each last position
        tf_topk = torch.zeros((gen_len, K), dtype=torch.int32)
        for i in range(gen_len):
            prefix = S[: P + i].reshape(1, P + i)
            last_logits = _recon(hf, W, prefix)[0, P + i - 1]  # [vocab]
            tf_topk[i] = last_logits.topk(K).indices.to(torch.int32)
        tf_entries.append(
            ReferenceEntry(
                prompt_text=text,
                prompt_tokens=S[:P].reshape(1, P),
                generated_tokens=S[P:].reshape(1, gen_len),
                topk_tokens=tf_topk,
                tf_prompt_len=P,
            )
        )

    meta = {"bos_id": 0, "eos_id": 0, "pad_id": 0}
    save_reference(
        Reference(k=K, hf_model_id=MODEL_ID, entries=prefill_entries, token_ids_meta=meta),
        HERE / "readiness_recon_prefill.refpt",
    )
    save_reference(
        Reference(k=K, hf_model_id=MODEL_ID, entries=tf_entries, token_ids_meta=meta), HERE / "readiness_recon_tf.refpt"
    )
    print(f"wrote {len(SENTENCES)} entries to readiness_recon_prefill.refpt and readiness_recon_tf.refpt (K={K})")


if __name__ == "__main__":
    main()
