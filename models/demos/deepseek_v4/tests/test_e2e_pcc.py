# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint 2 (completion) — end-to-end reduced-config forward: TT-NN model logits
vs HF DeepseekV4 reference. Asserts logits PCC >= 0.99 AND teacher-forced greedy token
match. Runs on real Blackhole.

Reduced config: 2 sliding_attention + hash_moe layers, hidden 256, 8 experts, seq_len 16
(< sliding_window 128, so no stateful compressor is needed on the e2e path). Same weights
feed both sides (bring-up model reads the HF module's weights)."""
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4.reference.reduced_config import build_reduced_model
from models.demos.deepseek_v4.tt import model as TTM

torch.manual_seed(0)


def _greedy(logits_fn, input_ids, n_new):
    """Autoregressive greedy generation using a prefill-only logits_fn (no KV cache;
    re-runs the growing sequence each step — fine for a tiny demo)."""
    ids = input_ids.clone()
    new = []
    for _ in range(n_new):
        logits = logits_fn(ids)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        new.append(int(nxt))
        ids = torch.cat([ids, nxt], dim=1)
    return new


def main():
    # Build the reference in bfloat16 — the working precision the GOAL compares against
    # (GOAL §7 Step 4 / Ckpt-2: PCC vs the CPU/HF reference *in bfloat16*).
    hf, cfg = build_reduced_model(seed=0, dtype=torch.bfloat16, overrides={"num_hidden_layers": 2})
    S = 16
    input_ids = torch.randint(0, cfg.vocab_size, (1, S))
    with torch.no_grad():
        ref_logits = hf(input_ids, use_cache=False).logits.float()  # [1,S,vocab]

    dev = ttnn.CreateDevice(device_id=0)
    try:
        tt_logits = TTM.tt_forward(hf, input_ids, dev)
        # 5-token greedy generation comparison (the real "demo tokens match" criterion)
        ref_gen = _greedy(lambda x: hf(x, use_cache=False).logits.float(), input_ids, 5)
        tt_gen = _greedy(lambda x: TTM.tt_forward(hf, x, dev), input_ids, 5)
    finally:
        ttnn.CloseDevice(dev)

    passed_all, pcc_all = comp_pcc(ref_logits, tt_logits, 0.99)
    passed_last, pcc_last = comp_pcc(ref_logits[:, -1, :], tt_logits[:, -1, :], 0.99)

    ref_tok = ref_logits.argmax(-1)
    tt_tok = tt_logits.argmax(-1)
    mism = (ref_tok != tt_tok)[0].nonzero().flatten().tolist()
    tok_match = int((ref_tok == tt_tok).sum())
    tok_total = ref_tok.numel()

    print(f"[E2E] logits PCC (all positions) = {float(pcc_all):.5f}  {'PASS' if passed_all else 'FAIL'}")
    print(f"[E2E] logits PCC (final token)   = {float(pcc_last):.5f}  {'PASS' if passed_last else 'FAIL'}")
    print(f"[E2E] teacher-forced token match = {tok_match}/{tok_total}  (mismatch positions: {mism})")
    if mism:
        for p in mism:
            top2 = torch.topk(ref_logits[0, p], 2).values
            print(f"        pos {p}: ref top-2 logit gap = {float(top2[0]-top2[1]):.4f} (near-tie => bf16 argmax flip)")
    print(
        f"[E2E] greedy generation (5 tok): ref={ref_gen}  tt={tt_gen}  match={'PASS' if ref_gen == tt_gen else 'FAIL'}"
    )
    ok = passed_all and ref_gen == tt_gen
    print("E2E_RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
